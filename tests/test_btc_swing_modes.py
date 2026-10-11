"""Hourly-only BTC selection must reach forward and replay wrappers without new fills."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import active_strategy
import backtest_execution
import information_runtime
import paper_trade_frozen_portfolio as paper
import run_multifactor_shadow as shadow
import simulate_range_swing as sim
import timeseries_execution as hourly
from test_hourly_execution_timing import HOUR, cfg, inputs
from test_selected_strategy_backtest import selected_inputs


@pytest.mark.parametrize("profile, expected", [
    (None, ("trend", "timeseries_trend")),
    ({}, ("trend", "timeseries_trend")),
    ({"strategy_modes": ["trend", "timeseries_trend"]}, ("trend", "timeseries_trend")),
    ({"strategy_modes": ["timeseries_trend"]}, ("timeseries_trend",)),
])
def test_supported_modes_keep_existing_default(profile, expected):
    assert active_strategy.profile_strategy_modes(profile) == expected


@pytest.mark.parametrize("modes", [None, "timeseries_trend", [], ["trend"],
    ["range"], ["all"], ["timeseries_trend", "timeseries_trend"], ["unknown"]])
def test_invalid_selected_modes_fail_closed(modes):
    with pytest.raises(ValueError, match="Unsupported selected strategy modules"):
        active_strategy.profile_strategy_modes({"strategy_modes": modes})


@pytest.mark.parametrize("modes", [["trend", "timeseries_trend"], ["timeseries_trend"]])
def test_producer_and_independent_shadow_respect_modes(tmp_path, monkeypatch, modes):
    args = selected_inputs(tmp_path)
    profile = json.loads(args.profile.read_text())
    profile["strategy_modes"] = modes
    args.profile.write_text(json.dumps(profile))
    command = information_runtime.paper_command(SimpleNamespace(
        factor_profile=args.profile, factor_snapshot=args.factor_snapshot,
        state_path=tmp_path / "state.json", report_path=tmp_path / "report.json",
        trades_path=tmp_path / "trades.csv"))
    assert command[command.index("--strategy-modes-override") + 1] == ",".join(modes)
    assert "--tiered-drawdown" in command
    monkeypatch.setattr(sim, "repo_root", lambda: tmp_path)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(shadow.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["shadow", "--once", "--profile", str(args.profile),
                                    "--snapshot", str(args.factor_snapshot)])
    assert shadow.main() == 0
    command = run.call_args.args[0]
    assert command[command.index("--strategy-modes-override") + 1] == ",".join(modes)
    assert "selected_test_shadow_state.json" in command[command.index("--state-path") + 1]


def test_hourly_only_forward_has_same_active_target_without_tactical(tmp_path, monkeypatch):
    hours, base = inputs([100] * 6 + [107, 110])
    now = 7 * HOUR + 3000
    monkeypatch.setattr(sys, "argv", ["paper", "--tiered-drawdown"])
    args = paper.parse_args()
    settings = cfg()
    monkeypatch.setattr(paper.frozen_strategy, "load_frozen_strategy", lambda _: (
        {"freeze_id": "test", "config_sha256": "test"}, settings))
    monkeypatch.setattr(paper, "load_or_create_state", lambda *a: {
        "symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(4 * HOUR), "observations": 0})
    monkeypatch.setattr(sim, "fetch_futures_klines_range", lambda symbol, interval, *a:
                        base[:84] if interval == "5m" else hours[:7])
    monkeypatch.setattr(sim, "fetch_funding_history", lambda *a: sim.FundingHistory([], []))
    monkeypatch.setattr(hourly, "fetch_opening", lambda *a: hourly.Opening(7 * HOUR, 110, now))
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime.fromtimestamp(now / 1000, timezone.utc)
    monkeypatch.setattr(paper, "datetime", clock)
    tactical = Mock(wraps=sim.simulate)
    monkeypatch.setattr(sim, "simulate", tactical)

    def run(name, modes):
        args.strategy_modes_override = modes
        args.state_path = tmp_path / f"{name}_state.json"
        args.report_path = tmp_path / f"{name}_report.json"
        args.trades_path = tmp_path / f"{name}_trades.csv"
        return paper.run_once(args)

    original = run("combined", None)
    assert tactical.call_count == 1
    tactical.reset_mock()
    selected = run("hourly", "timeseries_trend")
    tactical.assert_not_called()
    before, after = original["execution_target"], selected["execution_target"]
    assert before["signed_qty"] > 0
    for key in ("position_id", "origin_signal_time_ms", "available_time_ms", "components"):
        assert after[key] == before[key]
    assert after["signed_qty"] == pytest.approx(before["signed_qty"])
    assert after["equity"] == pytest.approx(before["equity"])
    assert selected["config"]["strategy_modes"] == ("timeseries_trend",)
    assert selected["execution_model"] == hourly.MODEL
    assert not selected["places_orders"]
    for key in ("side_mode", "risk_per_trade", "timeseries_min_ema_spread_pct",
                "timeseries_target_vol", "timeseries_fast_ema", "timeseries_slow_ema"):
        assert selected["config"][key] == original["config"][key]


def test_selected_replay_omits_tactical_preserves_gates_and_isolated_account(tmp_path, monkeypatch):
    args = selected_inputs(tmp_path)
    profile = json.loads(args.profile.read_text())
    profile["strategy_modes"] = ["timeseries_trend"]
    args.profile.write_text(json.dumps(profile))
    tactical = Mock(side_effect=AssertionError("Tactical sleeve must stay disabled"))
    monkeypatch.setattr(sim, "simulate", tactical)
    report = backtest_execution.run_selected_strategy(args)
    tactical.assert_not_called()
    assert report["config"]["strategy_modes"] == ("timeseries_trend",)
    assert report["summary"]["fills"] > 0
    assert report["entry_gate_diagnostics"]["complete_factor_coverage_pct"] == 100
    assert report["entry_gate_diagnostics"]["public_health_coverage_pct"] == 100
    assert report["entry_gate_diagnostics"]["reason_counts"]["current_event_blocks_entries"] == 12
    assert report["execution_model"] == hourly.MODEL
    assert not report["places_orders"]
    assert not (tmp_path / "unused_execution_replay.json").exists()
