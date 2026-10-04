import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_multifactor_snapshot as download
import information_runtime
import multifactor
import run_execution_supervisor as supervisor
import run_trading_terminal as terminal
import supplemental_market_data as supplemental


def test_receipt_time_and_source_age_both_apply():
    payload = {"samples": {"options": [{"received_at_ms": 2000, "value": {"observed_at_ms": 1000}}]},
               "status": {"options": {"ok": True}}}
    assert supplemental.source_view(payload, 1999)["options"]["value"] is None
    assert not supplemental.source_view(payload, 2000)["options"]["stale"]
    assert supplemental.source_view(payload, 901001)["options"]["stale"]


def test_failed_refresh_preserves_archive_but_reports_failure(tmp_path, monkeypatch):
    path = tmp_path / "supplemental.json"
    sample = {"received_at_ms": 1000, "value": {"observed_at_ms": 1000, "atm_iv_pct": 30}}
    path.write_text(json.dumps({"samples": {"options": [sample]}}))
    for name in supplemental.TTL:
        monkeypatch.setattr(supplemental, name, Mock(side_effect=ValueError("provider unavailable")))
    monkeypatch.setattr(supplemental, "now_ms", lambda: 5000)
    result = supplemental.collect(path, force=True)
    assert result["samples"]["options"] == [sample]
    view = supplemental.source_view(result, 5000)["options"]
    assert not view["ok"] and not view["stale"]
    assert view["value"]["atm_iv_pct"] == 30


def test_etf_partial_total_is_not_a_final_total():
    full = ["30 Sep 2026"] + ["0.0"] * 12 + ["(148.7)"]
    partial = ["01 Oct 2026"] + ["-"] + ["0.0"] * 11 + ["(92.9)"]
    result = supplemental.parse_etf([full, partial])
    assert result["net_flow_usd"] == -92_900_000
    assert result["complete"] is False
    assert result["latest_complete"]["net_flow_usd"] == -148_700_000
    assert result["fund_cells"][0] == "-"


def test_chain_flows_use_signed_net_and_keep_revision_status():
    result = supplemental.parse_flows([{"time": "2026-10-01T00:00:00Z", "FlowInExUSD": "100",
            "FlowOutExUSD": "140", "FlowInExUSD-status": "flash", "FlowOutExUSD-status": "flash"}])
    assert result["net_inflow_usd"] == -40
    assert result["inflow_status"] == "flash"


def test_depth_rejects_crossed_and_stale_book():
    book = {"bids": [[100, 2]], "asks": [[101, 1]], "T": 1000, "lastUpdateId": 1}
    result = supplemental.parse_depth(book, 1100)
    assert result["spread_bps"] == pytest.approx(1 / 100.5 * 10000)
    with pytest.raises(ValueError, match="timestamp"):
        supplemental.parse_depth(book, 62000)
    with pytest.raises(ValueError, match="Crossed"):
        supplemental.parse_depth({**book, "asks": [[99, 1]]}, 1100)


def test_option_skew_uses_same_expiry_and_ignores_one_sided_quotes():
    now = int(datetime(2026, 10, 2, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [{"instrument_name": f"BTC-30OCT26-{strike}-{kind}", "mark_iv": iv,
             "underlying_price": 100000, "bid_price": .01, "ask_price": .02}
            for strike, kind, iv in [(100000, "C", 30), (95000, "P", 32), (106000, "C", 29)]]
    rows.append({**rows[1], "instrument_name": "BTC-30OCT26-96000-P", "mark_iv": 90, "bid_price": 0})
    result = supplemental.parse_options(rows, now)
    assert result["approx_25delta_put_minus_call_iv_pp"] == 3
    assert len({r["expiry"] for r in result["selected_legs"]}) == 1


def test_public_forecast_fallback_without_licensed_key(monkeypatch):
    monkeypatch.delenv("TRADING_ECONOMICS_API_KEY", raising=False)
    body = '\ndays: ' + json.dumps([{"events": [{"id": 1, "currency": "USD", "name": "Non-Farm Employment Change",
            "dateline": 1790944200, "forecast": "89K", "actual": "", "previous": "162K"}]}])
    with patch.object(supplemental, "get", return_value=SimpleNamespace(text=body)) as get:
        result = supplemental.economic_consensus()
        assert result["records"][0]["Forecast"] == "89K"
        assert result["records"][0]["Actual"] is None
        get.assert_called_once_with(supplemental.FOREX_CALENDAR_URL)


def test_surprise_needs_premeeting_forecast_and_keeps_first_actual():
    row = {"CalendarId": "nfp", "Date": "2026-10-02T12:30:00Z", "Event": "Non-Farm Employment Change",
           "Forecast": "89K", "Actual": None}
    release = supplemental.utc_ms(row["Date"])
    def sample(receipt, actual, forecast="89K"):
        return {"received_at_ms": receipt, "value": {"source": "Forex Factory", "records": [
            {**row, "Actual": actual, "Forecast": forecast}]}}
    past = sample(release + 1, "105K")
    assert supplemental.economic_surprises([past], release + 100) == []
    pre = sample(release - 1, None)
    revised = sample(release + 2, "110K", "90K")
    result = supplemental.economic_surprises([pre, past, revised], release + 100)
    assert result[0]["surprise"] == 16000
    assert result[0]["actual_received_at_ms"] == release + 1
    assert supplemental.economic_surprises([pre, past], release - 1) == []


def test_factor_fetch_uses_shared_binance_cooldown():
    with patch.object(download, "BinanceTerminalClient") as client:
        client.return_value.public_get.return_value = [{"timestamp": 1}]
        assert download.binance_json("/futures/data/openInterestHist", {"symbol": "BTCUSDT"}) == [{"timestamp": 1}]
        client.return_value.public_get.assert_called_once()


def test_trial_routes_paper_and_terminal_to_same_profile(tmp_path):
    profile = ROOT / "config/multifactor_trial_20261002.json"
    candidate = multifactor.load_profile(profile)
    assert candidate["availability_mode"] == "first_seen"
    assert candidate["live_orders_allowed"] is False
    assert len(candidate["groups"]) == 6
    args = SimpleNamespace(factor_profile=profile, factor_snapshot=tmp_path / "factor.gz",
                           state_path=tmp_path / "state.json", report_path=tmp_path / "report.json",
                           trades_path=tmp_path / "trades.csv")
    command = information_runtime.paper_command(args)
    assert "--macro-snapshot" not in command
    assert command[command.index("--factor-profile") + 1] == str(profile)
    controller = terminal.TerminalController(client=Mock())
    with patch.object(terminal, "CANDIDATE_PATH", profile):
        routed = controller.supervisor_command("simulation")
        assert routed[routed.index("--factor-profile") + 1] == str(profile)
        with pytest.raises(ValueError, match="simulation-only"):
            controller.supervisor_command("live")


def test_factor_cycle_consumes_snapshots_without_blocking_on_refresh(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"execution_target": {
        "time_ms": 123, "equity": 100, "price": 100000, "signed_qty": 0}}))
    args = SimpleNamespace(mode="simulation", factor_profile=ROOT / "config/multifactor_trial_20261002.json",
                           factor_snapshot=tmp_path / "factor.gz", state_path=tmp_path / "state.json",
                           report_path=report_path, trades_path=tmp_path / "trades.csv")
    with patch.object(supervisor.strategy_supervisor, "run_checked") as run, \
         patch.object(information_runtime, "refresh_if_needed") as refresh, \
         patch.object(supervisor, "execute_report", return_value={"target_leverage": 0}) as execute, \
         patch.object(supervisor.decision_runtime, "write_judgment"):
        assert supervisor.run_cycle(args, Mock(), asof_ms=123)["signal_time_ms"] == 123
        refresh.assert_not_called()
        assert "--factor-profile" in run.call_args.args[0]
        execute.assert_called_once()
