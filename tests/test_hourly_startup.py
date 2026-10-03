"""An activated hourly sleeve may inherit only the last closed trend."""
from dataclasses import replace

import pytest

from test_hourly_execution_timing import HOUR, cfg, inputs
import execution_portfolio
import execution_targets
from paper_trade_frozen_portfolio import annotate_open_position_fractions
import portfolio_risk
import simulate_range_swing as sim
import timeseries_execution as hourly


ACTIVATION = 9 * HOUR
TREND = [100] * 6 + [107, 110, 112, 114, 115, 116, 117]


def sleeve(hours, *, activation_ms=ACTIVATION, opening=None, settings=None, funding=None):
    return hourly.build_sleeve(
        hours, settings or cfg(), 4 * HOUR, funding, opening=opening,
        asof_ms=opening.observed_at_ms if opening else hours[-1].close_time_ms,
        activation_ms=activation_ms,
    )


def stream(hours, base, *, activation_ms=ACTIVATION, opening=None, funding=None):
    settings = cfg()
    result = sleeve(hours, activation_ms=activation_ms, opening=opening, settings=settings, funding=funding)
    annotate_open_position_fractions([result])
    execution_targets.prepare_sleeves([result])
    combined = execution_portfolio.combine(
        base, [result], settings, portfolio_risk.DrawdownRiskPolicy(), 4 * HOUR,
        include_execution_target=True,
        decision_open_prices=hourly.decision_open_prices(base, [result]),
    )
    return list(execution_targets.target_stream(base, [result], combined, settings)), result


def entries(result):
    return [(sim._utc_ms(t["entry_time_utc"]), t["side"]) for t in result["trades"]]


def test_none_activation_preserves_legacy_transition_only_replay():
    hours, _ = inputs(TREND)
    expected = hourly.build_sleeve(hours, cfg(), 4 * HOUR)
    assert sleeve(hours, activation_ms=None) == expected
    assert entries(expected) == [(7 * HOUR, "long")]


@pytest.mark.parametrize("side,values", [
    ("long", TREND),
    ("short", [100] * 6 + [93, 90, 88, 86, 85, 84, 83]),
])
def test_already_confirmed_trend_starts_at_activation_open(side, values):
    hours, base = inputs(values)
    opening = hourly.Opening(ACTIVATION, values[9], ACTIVATION + 3000)
    points, result = stream(hours[:9], base[:9 * 12], opening=opening)
    assert entries(result) == [(ACTIVATION, side)]
    target = points[-1]
    assert target["available_time_ms"] == ACTIVATION
    assert target["signed_qty"] * sim.direction(side) > 0


def test_activation_between_hours_waits_for_next_observed_open():
    hours, base = inputs(TREND)
    activation = ACTIVATION + HOUR // 2
    before = sleeve(
        hours[:9], activation_ms=activation,
        opening=hourly.Opening(ACTIVATION, TREND[9], activation + 3000),
    )
    assert before["trades"] == []
    assert all(p["signed_qty"] == 0 for p in before["equity_curve"])
    points, result = stream(
        hours[:10], base[:10 * 12], activation_ms=activation,
        opening=hourly.Opening(10 * HOUR, TREND[10], 10 * HOUR + 3000),
    )
    assert entries(result) == [(10 * HOUR, "long")]
    assert points[-1]["available_time_ms"] == 10 * HOUR
    assert points[-1]["signed_qty"] > 0


def test_startup_waits_until_the_next_open_is_observable():
    hours, _ = inputs(TREND)
    result = sleeve(hours[:9])
    assert result["trades"] == []
    assert all(p["signed_qty"] == 0 for p in result["equity_curve"])


def test_bootstrap_obeys_warmup_and_evaluation_start():
    hours, _ = inputs([100, 110, 120, 130, 140, 150, 160, 170])
    warmed = hourly.build_sleeve(hours, cfg(), 0, activation_ms=0)
    assert entries(warmed) == [(5 * HOUR, "long")]

    hours, _ = inputs(TREND)
    evaluated = hourly.build_sleeve(hours, cfg(), 10 * HOUR, activation_ms=ACTIVATION)
    assert entries(evaluated) == [(10 * HOUR, "long")]


def test_weak_trend_is_not_bootstrapped():
    hours, _ = inputs([100] * 13)
    result = sleeve(hours)
    assert result["trades"] == []
    assert all(p["signed_qty"] == 0 for p in result["equity_curve"])


def test_bootstrap_respects_allowed_side():
    hours, _ = inputs([100] * 6 + [93, 90, 88, 86, 85, 84, 83])
    result = sleeve(hours, settings=replace(cfg(), side_mode="long"))
    assert result["trades"] == []


def test_every_activated_prefix_matches_full_replay_through_reversals():
    values = [100] * 6 + [107, 110, 112, 114, 105, 90, 85, 80, 95, 115, 118]
    hours, base = inputs(values)
    full, _ = stream(hours, base)
    lookup = {p["time_ms"]: p for p in full}
    for index in range(9, len(values)):
        partial, _ = stream(
            hours[:index], base[:index * 12],
            opening=hourly.Opening(index * HOUR, values[index], index * HOUR + 3000),
        )
        for point in partial:
            expected = lookup[point["time_ms"]]
            assert point["signed_qty"] == pytest.approx(expected["signed_qty"])
            assert point["equity"] == pytest.approx(expected["equity"])
            assert point["position_id"] == expected["position_id"]


def test_funding_prefixes_match_replay_before_after_startup_and_at_reversal():
    values = [100] * 6 + [107, 110, 112, 114, 105, 90, 85, 80, 95, 115, 118]
    hours, base = inputs(values)
    events = [ACTIVATION - HOUR // 2, ACTIVATION + HOUR // 2, 11 * HOUR]
    funding = sim.FundingHistory(events, [.002, .001, -.001])
    full, result = stream(hours, base, funding=funding)
    lookup = {point["time_ms"]: point for point in full}
    assert any(abs(trade["funding_pnl"]) > 0 for trade in result["trades"])
    # No inventory existed before this generation's first executable opening.
    without_prior = sleeve(hours, funding=sim.FundingHistory(events[1:], [.001, -.001]))
    assert sleeve(hours, funding=funding)["equity_curve"] == without_prior["equity_curve"]

    for index in range(9, len(values)):
        partial, _ = stream(
            hours[:index], base[:index * 12], funding=funding,
            opening=hourly.Opening(index * HOUR, values[index], index * HOUR + 3000),
        )
        for point in partial:
            expected = lookup[point["time_ms"]]
            assert point["signed_qty"] == pytest.approx(expected["signed_qty"])
            assert point["equity"] == pytest.approx(expected["equity"])
            assert point["position_id"] == expected["position_id"]


@pytest.mark.parametrize("future_close", [40, 500])
def test_unclosed_startup_hour_cannot_change_its_open_target(future_close):
    hours, base = inputs(TREND)
    original, _ = stream(hours, base)
    changed_hours = list(hours)
    changed_hours[9] = replace(
        hours[9], high=10000, low=1, close=future_close,
        volume=1e20, quote_volume=1e30,
    )
    changed_base = list(base)
    for index in range(9 * 12, 10 * 12):
        changed_base[index] = replace(
            base[index], high=10000, low=1, close=future_close,
            volume=1e20, quote_volume=1e30,
        )
    altered, _ = stream(changed_hours, changed_base)
    decision = ACTIVATION - 300_000
    before = next(p for p in original if p["time_ms"] == decision)
    after = next(p for p in altered if p["time_ms"] == decision)
    assert before["signed_qty"] > 0
    assert before["signed_qty"] == pytest.approx(after["signed_qty"])
    assert before["equity"] == pytest.approx(after["equity"])
    assert before["position_id"] == after["position_id"]


def test_recomputation_bootstraps_once_and_keeps_later_transitions():
    hours, _ = inputs(TREND)
    first = sleeve(hours)
    assert sleeve(hours) == first
    assert entries(first) == [(ACTIVATION, "long")]
    extended, _ = inputs(TREND + [118, 119, 120])
    assert entries(sleeve(extended)) == [(ACTIVATION, "long")]

    reversals, _ = inputs(
        [100] * 6 + [107, 110, 112, 114, 105, 90, 85, 80, 95, 115, 118]
    )
    assert entries(sleeve(reversals)) == [
        (ACTIVATION, "long"), (11 * HOUR, "short"), (16 * HOUR, "long"),
    ]
