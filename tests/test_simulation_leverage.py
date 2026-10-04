"""Profile sizing overrides preserve frozen signals and share execution limits."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import active_strategy
import frozen_strategy
import paper_trade_frozen_portfolio as paper
import simulate_range_swing as sim
import timeseries_execution as hourly
from test_selected_strategy_backtest import selected_inputs


@pytest.fixture(autouse=True)
def no_extra_environment_cap(monkeypatch):
    monkeypatch.delenv("SIM_MAX_LEVERAGE", raising=False)


def test_profile_override_keeps_frozen_files_and_signal_parameters():
    path = sim.repo_root() / "config/frozen_strategy_candidate_20260917.json"
    before = path.read_bytes()
    manifest, original = frozen_strategy.load_frozen_strategy(path)
    changed = active_strategy.apply_risk_limits(original, {"risk_limits": {"portfolio_leverage_cap": 10}})
    differences = {key for key, value in asdict(original).items() if asdict(changed)[key] != value}
    assert differences == {"leverage", "timeseries_max_leverage", "portfolio_leverage_cap"}
    assert (changed.leverage, changed.timeseries_max_leverage, changed.portfolio_leverage_cap) == (10, 10, 10)
    assert active_strategy.apply_risk_limits(original, {}) is original
    assert active_strategy.apply_risk_limits(original, None) is original
    assert active_strategy.apply_risk_limits(original, {"risk_limits": {"portfolio_leverage_cap": 2}}) == original
    assert path.read_bytes() == before
    assert frozen_strategy.sha256_file(sim.repo_root() / manifest["engine_path"]) == manifest["engine_sha256"]


@pytest.mark.parametrize("cap", [True, False, "10", None, 0, -1, 10.01, float("nan"), float("inf")])
def test_profile_rejects_invalid_strategy_caps(cap):
    with pytest.raises(ValueError):
        active_strategy.profile_leverage_cap({"risk_limits": {"portfolio_leverage_cap": cap}})


def test_profile_defaults_and_fractional_caps():
    assert active_strategy.profile_leverage_cap({}) == 2
    assert active_strategy.profile_leverage_cap({}, default=3) == 3
    assert active_strategy.profile_leverage_cap({"risk_limits": {"portfolio_leverage_cap": 7.5}}) == 7.5


def test_execution_cap_priority_and_optional_environment_ceiling(monkeypatch):
    report = {"config": {"portfolio_leverage_cap": 10}}
    assert active_strategy.simulation_leverage_cap(report) == 10
    assert active_strategy.simulation_leverage_cap(report, strategy_cap=4) == 4
    assert active_strategy.simulation_leverage_cap({}) == 2
    monkeypatch.setenv("SIM_MAX_LEVERAGE", " ")
    assert active_strategy.simulation_leverage_cap(report) == 10
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "3")
    assert active_strategy.simulation_leverage_cap(report) == 3
    assert active_strategy.simulation_leverage_cap({}) == 2
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "0")
    assert active_strategy.simulation_leverage_cap(report) == 0


@pytest.mark.parametrize("environment_cap", ["nan", "inf", "-1", "10.01", "invalid"])
def test_execution_rejects_invalid_environment_caps(monkeypatch, environment_cap):
    monkeypatch.setenv("SIM_MAX_LEVERAGE", environment_cap)
    with pytest.raises(ValueError):
        active_strategy.simulation_leverage_cap(strategy_cap=10)


def test_paper_rejects_invalid_environment_before_creating_state(tmp_path, monkeypatch):
    selected = selected_inputs(tmp_path)
    profile = json.loads(selected.profile.read_text())
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "nan")
    with patch.object(sys, "argv", ["paper", "--tiered-drawdown", "--manifest", profile["base_manifest"],
         "--factor-profile", str(selected.profile), "--factor-snapshot", str(selected.factor_snapshot),
         "--event-snapshot", profile["event_snapshot"], "--state-path", str(tmp_path / "state.json"),
         "--report-path", str(tmp_path / "report.json"), "--trades-path", str(tmp_path / "trades.csv")]):
        args = paper.parse_args()
    with pytest.raises(ValueError, match="SIM_MAX_LEVERAGE"):
        paper.run_once(args)
    assert not args.state_path.exists()
    assert not args.report_path.exists()


@pytest.mark.parametrize("cap", [True, 0, -1, 10.01, float("nan"), float("inf")])
def test_report_and_explicit_strategy_caps_are_validated(cap):
    with pytest.raises(ValueError):
        active_strategy.simulation_leverage_cap({"config": {"portfolio_leverage_cap": cap}})
    with pytest.raises(ValueError):
        active_strategy.simulation_leverage_cap(strategy_cap=cap)


@pytest.mark.parametrize("cap", [2, 10])
def test_paper_pipeline_applies_profile_caps_without_changing_shadow_profile_shape(tmp_path, cap):
    selected = selected_inputs(tmp_path)
    profile = json.loads(selected.profile.read_text())
    profile["risk_limits"]["portfolio_leverage_cap"] = cap
    selected.profile.write_text(json.dumps(profile))
    market, _, _ = sim.load_market_snapshot(selected.market_snapshot)
    start = sim._utc_ms(selected.start_utc)
    now = start + 9 * 3_600_000 + 3000
    state = {"symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(start), "observations": 0}
    with patch.object(sys, "argv", ["paper", "--tiered-drawdown", "--manifest", profile["base_manifest"],
         "--factor-profile", str(selected.profile), "--factor-snapshot", str(selected.factor_snapshot),
         "--event-snapshot", profile["event_snapshot"], "--state-path", str(tmp_path / "state.json"),
         "--report-path", str(tmp_path / "report.json"), "--trades-path", str(tmp_path / "trades.csv")]):
        args = paper.parse_args()

    def fetch(symbol, interval, fetch_start, end):
        return [bar for bar in market[interval] if bar.close_time_ms <= end]

    def opening(symbol, interval, timestamp, asof):
        bar = next(bar for bar in market[interval] if bar.open_time_ms == timestamp)
        return hourly.Opening(timestamp, bar.open, asof)

    with patch.object(paper, "load_or_create_state", return_value=state), \
         patch.object(paper.sim, "fetch_futures_klines_range", side_effect=fetch), \
         patch.object(paper.sim, "fetch_funding_history", return_value=sim.FundingHistory([], [])), \
         patch.object(hourly, "fetch_opening", side_effect=opening), \
         patch.object(hourly, "build_sleeve", wraps=hourly.build_sleeve) as build, \
         patch.object(paper.sim, "simulate", wraps=paper.sim.simulate) as tactical, \
         patch.object(paper, "datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime.fromtimestamp(now / 1000, timezone.utc)
        result = paper.run_once(args)
    for used_cfg in (build.call_args.args[1], tactical.call_args.args[1]):
        assert (used_cfg.leverage, used_cfg.timeseries_max_leverage, used_cfg.portfolio_leverage_cap) == (cap, cap, cap)
    assert result["config"]["portfolio_leverage_cap"] == cap
    assert result["runtime_risk_limits"]["simulation_leverage_cap"] == cap
    assert "runtime_risk_limits" not in result["shadow_profile"]
    assert "portfolio_leverage_cap" not in result["shadow_profile"]
    assert result["config_sha256"] == json.loads(Path(profile["base_manifest"]).read_text())["config_sha256"]
    assert result["effective_config_sha256"] == frozen_strategy.canonical_config_hash(result["config"])
