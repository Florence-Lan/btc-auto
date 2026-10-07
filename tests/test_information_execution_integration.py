"""Local research overlays must preserve the upstream causal execution ledger."""
import copy
import json
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest

from test_execution_ledger import sleeve_fixture
from test_hourly_execution_timing import HOUR, cfg as hourly_config, inputs
from test_world_event_risk import observation, snapshot
import execution_ledger
import market_intelligence as intelligence
import multifactor
import paper_trade_frozen_portfolio as paper
import portfolio_risk
import simulate_range_swing as sim
import timeseries_execution as hourly
import trading_execution
import world_event_risk as world


def archived_pressure(path, timestamp):
    db = intelligence.open_db(path)
    try:
        intelligence.persist(db, {}, {
            "generated_at_ms": timestamp,
            "rule_version": intelligence.VERSION,
            "direction": {"score": -60},
            "features": {"spot_flow": {"imbalance_1h": -.1},
                         "futures_flow": {"imbalance_1h": -.1}},
        })
    finally:
        db.close()


def event_journal(timestamp):
    return snapshot(observation(first_seen_at_utc=world.iso(timestamp),
                                assessed_at_utc=world.iso(timestamp)), at=timestamp)


@pytest.mark.parametrize("overlay, scale", [("world", .25), ("intelligence", .5), ("both", .25)])
def test_filters_preserve_partial_exit_funding_cash_and_per_unit_ledger(tmp_path, overlay, scale):
    bars, cfg, sleeve = sleeve_fixture()
    saved = copy.deepcopy(sleeve)
    source_ledger = execution_ledger.attach_ledger(sleeve)["trades"][0]["_ledger"]
    database = tmp_path / "observations.sqlite3"
    archived_pressure(database, bars[0].open_time_ms)
    adjusted = [sleeve]
    if overlay in {"world", "both"}:
        adjusted, _ = world.apply_overlay(adjusted, event_journal(bars[0].open_time_ms))
    if overlay in {"intelligence", "both"}:
        adjusted, _ = intelligence.apply_shadow_overlay(adjusted, database)
    assert adjusted[0]["trades"][0]["_ledger"] == source_ledger
    assert sleeve == saved
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, adjusted, cfg, portfolio_risk.DrawdownRiskPolicy(20, 30))
    assert [p["signed_qty"] for p in result["equity_curve"]] == pytest.approx(
        [scale, .5 * scale, .5 * scale, 0])
    assert [p["equity"] for p in result["equity_curve"]] == pytest.approx(
        [100 + (p["equity"] - 100) * scale for p in saved["equity_curve"]])
    assert result["equity_curve"][1]["cash_equity"] == pytest.approx(100 + 4.845 * scale)
    assert result["equity_curve"][2]["cash_equity"] == pytest.approx(100 + 4.645 * scale)
    assert result["trades"][0]["funding_pnl"] == pytest.approx(-.2 * scale)


@pytest.mark.parametrize("cap, use_world, use_intelligence, expected", [
    (.6, True, True, .25), (.2, True, True, .2),
    (.6, False, True, .5), (.2, False, True, .2), (.6, True, False, .25),
])
def test_forward_filters_use_strictest_factor_cap_and_preserve_known_open_target(
    tmp_path, monkeypatch, cap, use_world, use_intelligence, expected,
):
    hours, base = inputs([100] * 6 + [107, 110])
    now = 7 * HOUR + 3000
    settings = replace(hourly_config(), portfolio_leverage_cap=10)
    profile_path = tmp_path / "factor_profile.json"
    factor_profile = multifactor.load_profile(
        sim.repo_root() / "config/multifactor_candidate_20260926.json")
    factor_profile.update(public_context_enabled=False, hourly_startup_enabled=False)
    profile_path.write_text(json.dumps(factor_profile), encoding="utf-8")
    factor_snapshot = tmp_path / "factors.json"
    factor_snapshot.write_text(json.dumps({"schema_version": 1, "series": {}}), encoding="utf-8")
    events = tmp_path / "events.json"
    world.save_snapshot(events, event_journal(7 * HOUR))
    database = tmp_path / "observations.sqlite3"
    archived_pressure(database, 7 * HOUR)
    monkeypatch.setattr(sys, "argv", ["paper", "--tiered-drawdown"])
    args = paper.parse_args()
    args.factor_profile, args.factor_snapshot = profile_path, factor_snapshot
    args.asof_ms = now
    monkeypatch.setattr(paper.frozen_strategy, "load_frozen_strategy", lambda _: (
        {"freeze_id": "integration", "config_sha256": "integration"}, settings))
    monkeypatch.setattr(paper, "load_or_create_state", lambda *a: {
        "symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(4 * HOUR), "observations": 0})
    monkeypatch.setattr(sim, "fetch_futures_klines_range", lambda symbol, interval, *a:
                        base[:84] if interval == "5m" else hours[:7])
    monkeypatch.setattr(sim, "fetch_funding_history", lambda *a: sim.FundingHistory([], []))
    monkeypatch.setattr(hourly, "fetch_opening", lambda *a: hourly.Opening(7 * HOUR, 110, now))
    selected_cap = [1.0]
    monkeypatch.setattr(multifactor, "decision_at", lambda *a: multifactor.Decision(
        True, selected_cap[0], 0, 0, 0, 1, {}, (), (), {}))

    def run(name):
        args.state_path = tmp_path / f"{name}_state.json"
        args.report_path = tmp_path / f"{name}_report.json"
        args.trades_path = tmp_path / f"{name}_trades.csv"
        return paper.run_once(args)

    baseline = run("baseline")
    selected_cap[0] = cap
    args.world_event_snapshot = events if use_world else None
    args.intelligence_db = database if use_intelligence else None
    filtered = run("filtered")
    target, original = filtered["execution_target"], baseline["execution_target"]
    assert original["signed_qty"] > 0
    assert target["signed_qty"] == pytest.approx(original["signed_qty"] * expected)
    assert target["position_id"] == original["position_id"]
    assert target["available_time_ms"] == original["available_time_ms"] == 7 * HOUR
    assert target["origin_signal_time_ms"] == 7 * HOUR
    assert filtered["execution_model"] == hourly.MODEL
    assert filtered["research_only"] and not filtered["places_orders"]
    assert (filtered["world_event_overlay"] is not None) == use_world
    assert (filtered["intelligence_overlay"] is not None) == use_intelligence


@pytest.mark.parametrize("option", ["world", "intelligence"])
def test_information_only_forward_reports_cannot_reach_live_client(tmp_path, monkeypatch, option):
    bars, cfg, sleeve = sleeve_fixture()
    sleeve["summary"] = {}
    monkeypatch.setattr(sys, "argv", ["paper", "--strategy-modes-override", "trend"])
    args = paper.parse_args()
    args.state_path, args.report_path, args.trades_path = [tmp_path / name for name in
                                                        ("state.json", "report.json", "trades.csv")]
    args.asof_ms = bars[-1].close_time_ms
    if option == "world":
        args.world_event_snapshot = tmp_path / "events.json"
        world.save_snapshot(args.world_event_snapshot, event_journal(0))
    else:
        args.intelligence_db = tmp_path / "observations.sqlite3"
        archived_pressure(args.intelligence_db, 0)
    monkeypatch.setattr(paper.frozen_strategy, "load_frozen_strategy", lambda _: (
        {"freeze_id": "information_only", "config_sha256": "test"}, cfg))
    monkeypatch.setattr(paper, "load_or_create_state", lambda *a: {
        "symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(0), "observations": 0})
    monkeypatch.setattr(sim, "fetch_futures_klines_range", lambda *a: bars)
    monkeypatch.setattr(sim, "fetch_funding_history", lambda *a: sim.FundingHistory([], []))
    monkeypatch.setattr(sim, "simulate", lambda *a: copy.deepcopy(sleeve))
    monkeypatch.setattr(sim, "simulate_timeseries_trend", lambda *a: {"trades": [], "summary": {}})
    report = paper.run_once(args)
    assert report["research_only"]
    client = Mock()
    with pytest.raises(ValueError, match="simulation-only"):
        trading_execution.execute_report("live", report, client)
    assert client.mock_calls == []
