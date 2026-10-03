"""Protective research targets obey the selected known-open information cutoff."""
from dataclasses import replace
from unittest.mock import patch

import pytest

import execution_portfolio
import execution_targets
from paper_trade_frozen_portfolio import annotate_open_position_fractions
import portfolio_risk
import research_hourly_exit_comparison as comparison
import research_reentry
import simulate_range_swing as sim
from test_hourly_execution_timing import inputs, cfg, HOUR
import timeseries_execution as hourly


def targets(hours, base, policy, opening=None, funding=None):
    settings = cfg()
    asof = opening.observed_at_ms if opening else hours[-1].close_time_ms
    sleeve = comparison.build_protected_sleeve(hours, settings, 4*HOUR, funding,
        policy=policy, opening=opening, asof_ms=asof)
    annotate_open_position_fractions([sleeve])
    execution_targets.prepare_sleeves([sleeve])
    combined = execution_portfolio.combine(base, [sleeve], settings,
        portfolio_risk.DrawdownRiskPolicy(), 4*HOUR,
        decision_open_prices=hourly.decision_open_prices(base, [sleeve]))
    return list(execution_targets.target_stream(base, [sleeve], combined, settings)), sleeve


def test_baseline_delegates_exactly_to_selected_hourly_constructor():
    hours, _ = inputs([100]*6+[107, 110, 112])
    opening = hourly.Opening(8*HOUR, 112, 8*HOUR+3000)
    assert comparison.build_protected_sleeve(hours[:8], cfg(), 4*HOUR,
        policy=research_reentry.ExitPolicy(), opening=opening, asof_ms=opening.observed_at_ms) == hourly.build_sleeve(
        hours[:8], cfg(), 4*HOUR, opening=opening, asof_ms=opening.observed_at_ms)


@pytest.mark.parametrize("reentry", [False, True])
def test_every_protective_prefix_matches_full_replay_through_exits_and_funding(reentry):
    values = [100]*6+[107, 110, 114, 111, 105, 90, 85, 80, 95, 115, 120, 125, 130, 135]
    hours, base = inputs(values)
    funding = sim.FundingHistory([8*HOUR+HOUR//2, 12*HOUR], [.001, -.001])
    policy = research_reentry.ExitPolicy("volatility", reentry_enabled=reentry,
                                         cooldown_bars=2, breakout_bars=2)
    with patch.object(research_reentry, "atr", return_value=[1]*len(values)):
        full, sleeve = targets(hours, base, policy, funding=funding)
        assert any(t["exit_reason"] == "volatility_exit" for t in sleeve["trades"])
        lookup = {point["time_ms"]: point for point in full}
        for index in range(6, len(values)):
            partial, _ = targets(hours[:index], base[:index*12], policy,
                hourly.Opening(index*HOUR, values[index], index*HOUR+3000), funding)
            for point in partial:
                expected = lookup[point["time_ms"]]
                assert point["signed_qty"] == pytest.approx(expected["signed_qty"])
                assert point["equity"] == pytest.approx(expected["equity"])
                assert point["position_id"] == expected["position_id"]


def test_protective_exit_uses_known_open_without_future_hourly_liquidity():
    hours, base = inputs([100]*6+[107, 110, 114, 111, 105, 90])
    policy = research_reentry.ExitPolicy("volatility")
    with patch.object(research_reentry, "atr", return_value=[1]*len(hours)):
        before, sleeve = targets(hours, base, policy)
        first = next(t for t in sleeve["trades"] if t["exit_reason"] == "volatility_exit")
        exit_ms = sim._utc_ms(first["exit_time_utc"])
        index = exit_ms//HOUR
        changed = list(hours)
        changed[index] = replace(changed[index], low=1, high=1e5, close=1e4,
                                 volume=1e20, quote_volume=1e30)
        after, _ = targets(changed, base, policy)
    cutoff = exit_ms-300000
    left = next(point for point in before if point["time_ms"] == cutoff)
    right = next(point for point in after if point["time_ms"] == cutoff)
    assert left["signed_qty"] == right["signed_qty"] == 0
    assert left["equity"] == pytest.approx(right["equity"])


def test_episode_counter_counts_closes_and_reversals_not_rebalances():
    assert comparison.closed_execution_episodes(
        [{"signed_qty": qty} for qty in (0, .01, .02, .01, 0, -.01, -.02, .01, 0)]) == 3


def test_fixed_cost_environment_restores_callers_settings(monkeypatch):
    monkeypatch.setenv("SIM_TAKER_FEE", "0.9")
    import os
    execution = {"fee_rate": .00045, "slippage_bps": 1, "max_leverage": 2, "max_notional_usdt": 0}
    with comparison.fixed_execution_environment(execution):
        assert os.environ["SIM_TAKER_FEE"] == "0.00045"
    assert os.environ["SIM_TAKER_FEE"] == "0.9"
