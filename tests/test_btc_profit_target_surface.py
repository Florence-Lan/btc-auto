"""The forward signal report must deliver the selected BTC exit policy to execution."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_trade_frozen_portfolio as paper
import simulate_range_swing as sim
import timeseries_execution as hourly
from test_selected_strategy_backtest import selected_inputs


@pytest.mark.parametrize("configured", [False, True])
def test_forward_report_preserves_opt_in_profit_policy(tmp_path, monkeypatch, configured):
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "2")
    selected = selected_inputs(tmp_path)
    profile = json.loads(selected.profile.read_text())
    policy = {"mode": "net_margin_target_v1", "target_margin_return": .6,
              "margin_leverage": 10}
    if configured:
        profile["profit_exit_policy"] = policy
        selected.profile.write_text(json.dumps(profile))
    start = sim._utc_ms(selected.start_utc)
    now = start + 9 * 3_600_000 + 3000
    market, _, _ = sim.load_market_snapshot(selected.market_snapshot)
    state = {"symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(start),
             "observations": 0}
    with patch.object(sys, "argv", [
        "paper", "--tiered-drawdown", "--manifest", profile["base_manifest"],
        "--factor-profile", str(selected.profile),
        "--factor-snapshot", str(selected.factor_snapshot),
        "--event-snapshot", profile["event_snapshot"],
        "--state-path", str(tmp_path / "paper_state.json"),
        "--report-path", str(tmp_path / "paper_report.json"),
        "--trades-path", str(tmp_path / "paper_trades.csv"),
        "--asof-ms", str(now),
    ]):
        args = paper.parse_args()

    def opening(symbol, interval, timestamp, asof):
        bar = next(bar for bar in market[interval] if bar.open_time_ms == timestamp)
        return hourly.Opening(timestamp, bar.open, asof)

    with patch.object(paper, "load_or_create_state", return_value=state) as load, \
         patch.object(paper.sim, "fetch_futures_klines_range",
                      side_effect=lambda symbol, interval, fetch_start, end:
                          [bar for bar in market[interval] if bar.close_time_ms <= end]), \
         patch.object(paper.sim, "fetch_funding_history", return_value=sim.FundingHistory([], [])), \
         patch.object(hourly, "fetch_opening", side_effect=opening):
        report = paper.run_once(args)

    saved = json.loads(args.report_path.read_text())
    for shadow in (report["shadow_profile"], saved["shadow_profile"], load.call_args.args[-1]):
        if configured:
            assert shadow["profit_exit_policy"] == policy
        else:
            assert "profit_exit_policy" not in shadow
    assert report["places_orders"] is False
