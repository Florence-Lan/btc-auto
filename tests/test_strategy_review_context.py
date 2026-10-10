"""Trend-review evidence must respect the decision cutoff and real exit rule."""
from dataclasses import replace
import sys
from unittest.mock import patch

import pytest

from test_hourly_execution_timing import HOUR, cfg, inputs
import paper_trade_frozen_portfolio as paper
import simulate_range_swing as sim
import strategy_review_context as review
import timeseries_execution as hourly


def target(side="short"):
    return {"components": [{"strategy": "timeseries_trend_6h", "side": side,
                            "signed_qty": -.25 if side == "short" else .25}]}


def test_snapshot_uses_closed_asof_bars_and_explicit_percentage_units():
    hours, _ = inputs([100] * 5 + [90, 999_999])
    asof = 6 * HOUR + 3000
    # The forming hour's close is deliberately unusable. It must never enter EMA.
    hours[-1] = replace(hours[-1], close=float("nan"), high=float("inf"))
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), asof, target())
    assert snapshot["status"] == "available"
    assert snapshot["closed_bar_count"] == 6
    assert snapshot["required_history_bars"] == 5
    assert snapshot["decision_asof_ms"] == asof
    assert snapshot["timeframe"] == "1h"
    assert snapshot["fast_ema_period_bars"] == 2
    assert snapshot["slow_ema_period_bars"] == 4
    assert snapshot["ema_fast"] == pytest.approx(93.33333333333333)
    assert snapshot["ema_slow"] == pytest.approx(96)
    assert snapshot["spread_unit"] == "percent"
    assert snapshot["ema_spread_pct"] == pytest.approx(-2.962962962962963)
    assert snapshot["entry_min_ema_spread_pct"] == pytest.approx(.5)
    assert snapshot["confirmed_side"] == "short"
    assert snapshot["last_closed_bar"] == {
        "open_time_ms": 5 * HOUR, "close_time_ms": 6 * HOUR - 1,
        "available_at_ms": 6 * HOUR - 1, "close_price": 90,
    }
    assert snapshot["data_available"] and snapshot["history_sufficient"]


def test_bar_becomes_available_at_closed_timestamp_not_its_open():
    hours, _ = inputs([100] * 5 + [90])
    before = review.timeseries_trend_snapshot(hours, cfg(), hours[-1].close_time_ms - 1)
    closed = review.timeseries_trend_snapshot(hours, cfg(), hours[-1].close_time_ms)
    assert before["closed_bar_count"] == 5
    assert before["ema_fast"] == 100
    assert closed["closed_bar_count"] == 6
    assert closed["ema_fast"] < 100


def test_existing_short_survives_neutral_threshold_even_with_opposite_ema_ordering():
    hours, _ = inputs([100] * 6 + [80, 80, 90, 96, 100])
    settings = replace(cfg(), timeseries_min_ema_spread_pct=.04)
    sleeve = hourly.build_sleeve(hours, settings, 4 * HOUR)
    quantity = sleeve["equity_curve"][-1]["signed_qty"]
    assert quantity < 0  # The frozen engine really keeps this short open.
    actual_target = {"components": [{"strategy": "timeseries_trend_6h",
                                     "signed_qty": quantity}]}
    snapshot = review.timeseries_trend_snapshot(
        hours, settings, hours[-1].close_time_ms, actual_target,
    )
    assert snapshot["ema_ordering"] == "fast_above_slow"
    assert snapshot["confirmed_side"] is None
    assert 0 < snapshot["ema_spread_pct"] < snapshot["entry_min_ema_spread_pct"]
    assert snapshot["current_target_side"] == "short"
    assert snapshot["trend_side_under_exit_rule"] == "short"
    assert snapshot["target_side_valid_under_exit_rule"] is True


def test_opposite_confirmed_threshold_invalidates_existing_target():
    hours, _ = inputs([100] * 5 + [110])
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR, target())
    assert snapshot["confirmed_side"] == "long"
    assert snapshot["current_target_side"] == "short"
    assert snapshot["trend_side_under_exit_rule"] == "long"
    assert snapshot["target_side_valid_under_exit_rule"] is False


def test_neutral_latest_bar_cannot_restore_short_after_prior_confirmed_reversal():
    hours, _ = inputs([100] * 6 + [80, 80, 90, 96, 100, 110, 110, 104, 101, 100])
    settings = replace(cfg(), timeseries_min_ema_spread_pct=.04)
    sleeve = hourly.build_sleeve(hours, settings, 4 * HOUR)
    assert sleeve["equity_curve"][-1]["signed_qty"] > 0
    snapshot = review.timeseries_trend_snapshot(
        hours, settings, hours[-1].close_time_ms, target("short"),
    )
    assert snapshot["confirmed_side"] is None
    assert snapshot["ema_ordering"] == "fast_below_slow"
    assert snapshot["last_confirmed_side"] == "long"
    assert snapshot["last_confirmed_bar"]["open_time_ms"] == 11 * HOUR
    assert snapshot["trend_side_under_exit_rule"] == "long"
    assert snapshot["target_side_valid_under_exit_rule"] is False


def test_stale_closed_history_is_not_current_evidence():
    hours, _ = inputs([100] * 5 + [90])
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 7 * HOUR + 3000, target())
    assert snapshot["status"] == "unavailable"
    assert snapshot["unavailable_reason"] == "stale_closed_history"
    assert snapshot["data_available"] and snapshot["history_sufficient"]
    assert snapshot["target_side_valid_under_exit_rule"] is None


@pytest.mark.parametrize("aggregate", [0, .25, float("nan")])
def test_corrupt_aggregate_target_cannot_validate_short_component(aggregate):
    hours, _ = inputs([100] * 5 + [90])
    intent = dict(target(), signed_qty=aggregate)
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["target_consistency_issue"] == "aggregate_target_quantity_mismatch"
    assert snapshot["target_side_valid_under_exit_rule"] is False


@pytest.mark.parametrize("tactical_qty", [.25, .5])
def test_mixed_sleeve_offset_or_net_long_does_not_invalidate_short_trend(tactical_qty):
    hours, _ = inputs([100] * 5 + [90])
    intent = target()
    intent["components"].append({
        "strategy": "trend_pullback_5m", "side": "long", "signed_qty": tactical_qty,
    })
    intent["signed_qty"] = tactical_qty - .25
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["current_target_side"] == "short"
    assert snapshot["target_consistency_issue"] is None
    assert snapshot["target_side_valid_under_exit_rule"] is True


def test_mixed_sleeve_corrupt_aggregate_still_invalidates_target():
    hours, _ = inputs([100] * 5 + [90])
    intent = target()
    intent["components"].append({
        "strategy": "trend_pullback_5m", "side": "long", "signed_qty": .5,
    })
    intent["signed_qty"] = -.25  # Actual sum is +.25.
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["target_consistency_issue"] == "aggregate_target_quantity_mismatch"
    assert snapshot["target_side_valid_under_exit_rule"] is False


def test_component_sum_validation_allows_normal_floating_point_roundoff():
    hours, _ = inputs([100] * 5 + [90])
    intent = target()
    intent["signed_qty"] = -.25 + 1e-13
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["target_consistency_issue"] is None
    assert snapshot["target_side_valid_under_exit_rule"] is True


def test_declared_component_side_must_match_quantity():
    hours, _ = inputs([100] * 5 + [90])
    intent = target()
    intent["components"][0]["side"] = "long"
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["target_consistency_issue"] == "component_declared_side_mismatch"
    assert snapshot["target_side_valid_under_exit_rule"] is False


@pytest.mark.parametrize("entry", ["not-a-timestamp", sim.iso_utc_from_ms(7 * HOUR)])
def test_invalid_or_future_component_entry_cannot_validate_current_target(entry):
    hours, _ = inputs([100] * 5 + [90])
    intent = target()
    intent["components"][0]["entry_time_utc"] = entry
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["target_consistency_issue"] in {
        "invalid_component_entry_time", "component_entry_not_available",
    }
    assert snapshot["target_side_valid_under_exit_rule"] is False


def test_future_origin_metadata_cannot_validate_current_target():
    hours, _ = inputs([100] * 5 + [90])
    intent = dict(target(), origin_signal_time_ms=7 * HOUR)
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR + 3000, intent)
    assert snapshot["target_consistency_issue"] == "target_timestamp_not_available"
    assert snapshot["target_side_valid_under_exit_rule"] is False


@pytest.mark.parametrize("bars,reason,data_available", [
    ([], "no_closed_bars", False),
    (inputs([100] * 3)[0], "insufficient_closed_history", True),
    (inputs([100] * 6)[0][1:] + inputs([100] * 6)[0][-1:],
     "closed_bar_history_gap_or_out_of_order", False),
])
def test_missing_or_invalid_history_is_explicitly_unavailable(bars, reason, data_available):
    snapshot = review.timeseries_trend_snapshot(bars, cfg(), 8 * HOUR, target())
    assert snapshot["status"] == "unavailable"
    assert snapshot["unavailable_reason"] == reason
    assert snapshot["data_available"] is data_available
    assert snapshot["ema_fast"] is None
    assert snapshot["confirmed_side"] is None
    assert snapshot["target_side_valid_under_exit_rule"] is None


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf")])
def test_invalid_closed_prices_cannot_manufacture_trend_evidence(price):
    hours, _ = inputs([100] * 6)
    hours[-1] = replace(hours[-1], close=price)
    snapshot = review.timeseries_trend_snapshot(hours, cfg(), 6 * HOUR)
    assert snapshot["status"] == "unavailable"
    assert snapshot["unavailable_reason"] == "invalid_closed_bar_price"
    assert snapshot["ema_fast"] is None


def test_opening_only_synthetic_bar_is_not_treated_as_closed_hour():
    hours, _ = inputs([100] * 6)
    opening = replace(hours[-1], open_time_ms=6 * HOUR, close_time_ms=6 * HOUR)
    snapshot = review.timeseries_trend_snapshot(hours + [opening], cfg(), 6 * HOUR + 3000)
    assert snapshot["status"] == "unavailable"
    assert snapshot["unavailable_reason"] == "invalid_closed_bar_interval"


def test_forward_report_uses_original_closed_history_and_actual_target(tmp_path):
    hours, base = inputs([100] * 6 + [107, 110])
    now = 7 * HOUR + 3000
    with patch.object(sys, "argv", ["paper", "--tiered-drawdown"]):
        args = paper.parse_args()
    args.asof_ms = now
    args.state_path = tmp_path / "state.json"
    args.report_path = tmp_path / "report.json"
    args.trades_path = tmp_path / "trades.csv"
    state = {"symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(4 * HOUR),
             "observations": 0}

    def fetch(symbol, interval, start, end):
        return base[:84] if interval == "5m" else hours[:7]

    with patch.object(paper.frozen_strategy, "load_frozen_strategy", return_value=(
            {"freeze_id": "test", "config_sha256": "test"}, cfg())), \
         patch.object(paper, "load_or_create_state", return_value=state), \
         patch.object(paper.sim, "fetch_futures_klines_range", side_effect=fetch), \
         patch.object(paper.sim, "fetch_funding_history", return_value=sim.FundingHistory([], [])), \
         patch.object(hourly, "fetch_opening", return_value=hourly.Opening(7 * HOUR, 110, now)):
        report = paper.run_once(args)
    snapshot = report["strategy_review_context"]["timeseries_trend"]
    assert report["execution_target"]["signed_qty"] > 0
    assert snapshot["current_target_side"] == "long"
    assert snapshot["trend_side_under_exit_rule"] == "long"
    assert snapshot["target_side_valid_under_exit_rule"] is True
    assert snapshot["closed_bar_count"] == 7
    assert snapshot["last_closed_bar"]["close_price"] == 107
    assert snapshot["last_closed_bar"]["available_at_ms"] == 7 * HOUR - 1
    assert snapshot["decision_asof_ms"] == report["decision_asof_ms"] == now
