"""Outages must neither bypass known account stops nor invent executions."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import binance_terminal_client as binance
import run_execution_supervisor as supervisor
import simulation_risk_monitor as monitor
from trading_execution import SimulationAccount

NOW = 1_791_027_000_000


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor.time, "time", lambda: NOW / 1000)
    monkeypatch.setenv("LLM_TRADE_GATE_ENABLED", "false")
    monkeypatch.setenv("SIM_TAKER_FEE", "0.00045")
    monkeypatch.setenv("SIM_SLIPPAGE_BPS", "1")
    account = SimulationAccount(tmp_path / "account.json", allow_llm=False)
    state = account.reset(1000, now_ms=NOW - 100_000)
    state.update(wallet_balance=1000, position_qty=.01, entry_price=100000,
                 observed_flat_target=True, last_signal_time_ms=NOW - 90_000,
                 position_history=[{"time_ms": NOW - 90_000, "signed_qty": .01}])
    account._save(state)
    client = Mock(spec=binance.BinanceTerminalClient)
    client.mark_price_observation.return_value = {"price": 80000, "time_ms": NOW - 1000}
    client.funding_history.return_value = []
    client.symbol_rules.side_effect = AssertionError("quantity metadata must not block a full close")
    return account, client, tmp_path / "monitor.json"


def run(setup, **kwargs):
    account, client, path = setup
    return monitor.monitor(client, NOW, account=account, status_path=path, **kwargs)


def test_hard_stop_precedes_funding_and_does_not_require_quantity_endpoint(setup):
    account, client, _ = setup
    def failing_funding(*args):
        assert account.load()["position_qty"] == 0  # exit precedes the request
        raise OSError("funding disconnected")
    client.funding_history.side_effect = failing_funding
    result = run(setup)
    assert result["fill"]["side"] == "SELL"
    assert result["monitor"]["status"] == "degraded"
    assert result["monitor"]["assessment"] == "evaluated"
    assert account.load()["risk_halt_at_utc"]
    assert account.load()["last_signal_time_ms"] == NOW - 90_000
    client.symbol_rules.assert_not_called()


def test_clock_failure_still_uses_fresh_mark_for_protective_exit(setup):
    account, client, _ = setup
    result = run(setup, clock_available=False, clock_error=OSError("time disconnected"))
    assert result["fill"]
    assert result["monitor"]["clock_source"] == "mark_endpoint"
    assert result["monitor"]["assessment_time_ms"] == NOW - 1000
    assert account.load()["position_qty"] == 0
    assert "funding_last_fetch_ms" not in account.load()
    client.funding_history.assert_not_called()


def test_late_funding_after_hard_stop_uses_settlement_inventory_and_is_idempotent(setup):
    account, client, _ = setup
    client.funding_history.return_value = [{"fundingTime": NOW - 50_000,
        "fundingRate": .001, "markPrice": 100000}]
    result = run(setup)
    assert result["fill"]
    assert result["monitor"]["status"] == "healthy"
    assert account.load()["funding_pnl"] == pytest.approx(-1)
    run(setup)
    assert account.load()["fill_count_total"] == 1
    assert account.load()["funding_pnl"] == pytest.approx(-1)


@pytest.mark.parametrize("observation", [
    {"price": 80000, "time_ms": NOW - 60001},
    {"price": 80000, "time_ms": NOW + 5001},
    {"price": float("nan"), "time_ms": NOW},
    {"price": 0, "time_ms": NOW},
    {"price": 80000, "time_ms": 0},
])
def test_invalid_or_stale_mark_cannot_change_account_or_create_fill(setup, observation):
    account, client, _ = setup
    before = account.path.read_bytes()
    client.mark_price_observation.return_value = observation
    result = run(setup)
    assert result["monitor"]["status"] == "unavailable"
    assert result["monitor"]["assessment"] == "not_evaluated"
    assert account.path.read_bytes() == before
    client.funding_history.assert_not_called()


def test_shared_cooldown_has_heartbeat_without_other_network_requests(setup):
    account, client, _ = setup
    before = account.path.read_bytes()
    client.mark_price_observation.side_effect = binance.BinanceApiError("cooldown", status=418, retry_at_ms=NOW+60000)
    assert run(setup)["monitor"]["status"] == "unavailable"
    assert account.path.read_bytes() == before
    client.funding_history.assert_not_called()
    client.symbol_rules.assert_not_called()


def test_short_inventory_hard_stop_can_fully_close_without_rules(setup):
    account, client, _ = setup
    state = account.load()
    state["position_qty"] = -.01
    state["position_history"][-1]["signed_qty"] = -.01
    account._save(state)
    client.mark_price_observation.return_value = {"price":125000, "time_ms":NOW}
    result = run(setup)
    assert result["fill"]["side"] == "BUY"
    assert account.load()["position_qty"] == 0
    client.symbol_rules.assert_not_called()


def test_corrupt_existing_account_cannot_be_reset_by_monitor(setup):
    account, client, _ = setup
    account.path.write_text("broken JSON")
    assert run(setup)["monitor"]["assessment"] == "not_evaluated"
    assert account.path.read_text() == "broken JSON"
    client.mark_price_observation.assert_not_called()


def test_mark_fallback_cannot_move_inventory_clock_backwards(setup):
    account, client, _ = setup
    state = account.load()
    state["position_history"][-1]["time_ms"] = NOW
    account._save(state)
    before = account.path.read_bytes()
    assert run(setup, clock_available=False)["monitor"]["status"] == "unavailable"
    assert account.path.read_bytes() == before


def test_failed_checks_preserve_last_success_and_record_recovery(setup):
    account, client, path = setup
    run(setup)
    client.mark_price_observation.side_effect = OSError("disconnected")
    first = run(setup)["monitor"]
    second = run(setup)["monitor"]
    assert first["last_success_at_ms"] == NOW
    assert second["consecutive_failures"] == 2
    assert monitor.status_view(path, now_ms=NOW+120001)["status"] == "stale"
    assert monitor.status_view(path, now_ms=NOW, account_epoch="different")["status"] == "not_observed"
    client.mark_price_observation.side_effect = None
    assert run(setup)["monitor"]["consecutive_failures"] == 0
    assert len(path.with_suffix(".jsonl").read_text().splitlines()) == 4


def test_slow_funding_request_cannot_fill_again_using_expired_mark(setup, monkeypatch):
    account, client, _ = setup
    def delayed(*args):
        monkeypatch.setattr(monitor.time, "time", lambda: (NOW+61000)/1000)
        return []
    client.funding_history.side_effect = delayed
    result = run(setup)
    assert result["fill"]  # already executed before funding hung
    assert result["monitor"]["status"] == "degraded"
    assert "funding_last_fetch_ms" not in account.load()


def test_fill_journal_failure_reports_the_already_persisted_protective_exit(setup):
    account, client, _ = setup
    account._record_fills = Mock(side_effect=OSError("fill journal unavailable"))
    result = run(setup)
    assert result["fill"]["side"] == "SELL"
    assert account.load()["position_qty"] == 0
    assert account.load()["fill_count_total"] == 1
    assert result["monitor"]["status"] == "degraded"
    assert result["monitor"]["assessment"] == "evaluated"
    assert "fill_archive" in result["monitor"]["errors"]
    client.funding_history.assert_not_called()


def test_funding_induced_exit_archive_failure_also_reports_actual_closure(setup):
    account, client, _ = setup
    client.mark_price_observation.return_value = {"price":100000,"time_ms":NOW}
    client.funding_history.return_value = [{"fundingTime":NOW-50000,"fundingRate":.20,"markPrice":100000}]
    original = account._record_fills
    def fail_only_fill(state, records):
        if any(records):
            raise OSError("fill journal unavailable")
        original(state, records)
    account._record_fills = fail_only_fill
    result = run(setup)
    assert account.load()["funding_pnl"] == -200
    assert account.load()["position_qty"] == 0
    assert result["fill"]["side"] == "SELL"
    assert result["monitor"]["status"] == "degraded"


def test_damaged_monitor_timestamp_does_not_break_status_endpoint(setup):
    _, _, path = setup
    run(setup)
    import json
    payload = json.loads(path.read_text())
    payload["checked_at_ms"] = "broken"
    path.write_text(json.dumps(payload))
    assert monitor.status_view(path)["status"] == "unavailable"


def test_scheduler_clock_failure_monitors_but_cannot_run_strategy(monkeypatch):
    client = Mock()
    failure = OSError("clock disconnected")
    client.server_time_ms.side_effect = failure
    client.mark_price_observation.side_effect = OSError("mark disconnected")
    observe, cycle = Mock(), Mock()
    monkeypatch.setattr(supervisor, "monitor_simulation_account", observe)
    monkeypatch.setattr(supervisor, "run_cycle", cycle)
    with pytest.raises(RuntimeError, match="No valid simulation decision clock"):
        supervisor.check_once(SimpleNamespace(mode="simulation", bar_settle_delay_seconds=3), client)
    assert observe.call_args.kwargs["clock_available"] is False
    assert observe.call_args.kwargs["clock_error"].__cause__ is failure
    cycle.assert_not_called()


def test_manual_once_also_observes_risk_before_clock_failure(monkeypatch):
    client = Mock()
    client.server_time_ms.side_effect = OSError("clock disconnected")
    client.mark_price_observation.side_effect = OSError("mark disconnected")
    observe, cycle = Mock(), Mock()
    monkeypatch.setattr(supervisor, "monitor_simulation_account", observe)
    monkeypatch.setattr(supervisor, "run_cycle", cycle)
    monkeypatch.setattr(supervisor, "BinanceTerminalClient", lambda:client)
    monkeypatch.setattr(supervisor, "parse_args", lambda:SimpleNamespace(
        mode="simulation", once=True, poll_seconds=30, bar_settle_delay_seconds=3,
        refresh_hours=12, factor_profile=None))
    with pytest.raises(RuntimeError, match="No valid simulation decision clock"):
        supervisor.main()
    observe.assert_called_once()
    assert observe.call_args.kwargs["clock_available"] is False
    cycle.assert_not_called()


def test_timestamped_mark_client_parses_public_response_without_second_clock_request():
    client = binance.BinanceTerminalClient()
    client.public_get = Mock(return_value={"symbol":"BTCUSDT", "markPrice":"80000", "time": NOW})
    assert client.mark_price_observation() == {"price":80000.0, "time_ms":NOW}
    client.public_get.assert_called_once_with("/fapi/v1/premiumIndex", {"symbol":"BTCUSDT"})
