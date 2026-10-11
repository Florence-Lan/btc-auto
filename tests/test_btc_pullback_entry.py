"""Meaningful long/short entry admission, causal structures and numeric rules."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import btc_pullback_entry as entry


ORIGIN = 1000 * entry.HOUR
POLICY = copy.deepcopy(entry.DEFAULT_POLICY)


def target(side="long", **changes):
    return {"target_leverage": 1 if side == "long" else -1, "position_id": "origin-id",
            "origin_signal_time_ms": ORIGIN, "origin_entry_price": 1040 if side == "long" else 960,
            **changes}


def compact(open_ms, opening, high, low, close, *, step=entry.HOUR):
    return [open_ms, opening, high, low, close, 1, 1000, open_ms + step - 1]


def hourly():
    return [compact(ORIGIN - (49 - index) * entry.HOUR, 1000, 1010, 990, 1000)
            for index in range(49)]


def five(index=0, *, side="long", kind="rejection"):
    if kind == "untouched":
        prices = (1010, 1012, 1008, 1011) if side == "long" else (990, 992, 988, 989)
    elif kind == "touch":
        prices = (1001, 1002, 999, 1000) if side == "long" else (999, 1001, 998, 1000)
    elif kind == "broken":
        prices = (999, 1000, 985, 987) if side == "long" else (1001, 1015, 1000, 1013)
    else:
        prices = (1000, 1004, 999, 1003) if side == "long" else (1000, 1001, 996, 997)
    return compact(ORIGIN + index * entry.FIVE_MINUTES, *prices, step=entry.FIVE_MINUTES)


def pending(side="long", **policy):
    return entry.build_pending(target(side), hourly(), {**POLICY, **policy}, ORIGIN + 1000)


def advance(saved, rows, *, side="long", mark=None, now=None, slip=1, policy=None, desired=None):
    if mark is None:
        mark = 1003 if side == "long" else 997
    if now is None:
        now = rows[-1][7] + 1 if rows else ORIGIN + 1000
    return entry.advance_pending(saved, desired or target(side), rows, mark, now, slip,
                                 policy or {**POLICY})


@pytest.mark.parametrize("side", ["long", "short"])
def test_frozen_corridor_improves_original_anchor_and_requires_both_touch_and_rejection(side):
    saved = pending(side)
    assert saved["status"] == "waiting_retrace"
    assert saved["atr"] == 20
    assert saved["candidate_low"] == 998 and saved["candidate_high"] == 1002
    assert saved["selected_structure"]["kind"] == "ema48"
    assert saved["numeric_priority"] == 3
    if side == "long":
        assert saved["entry_high"] <= 1040 - 2
    else:
        assert saved["entry_low"] >= 960 + 2
    untouched, decision = advance(saved, [five(side=side, kind="untouched")], side=side)
    assert untouched["status"] == "waiting_retrace" and not decision["allowed"]
    touched, decision = advance(saved, [five(side=side, kind="touch")], side=side)
    assert touched["status"] == "waiting_confirmation" and not decision["allowed"]
    ready, decision = advance(touched, [five(side=side, kind="touch"), five(1, side=side)], side=side)
    assert ready["status"] == "ready" and decision["allowed"]
    assert decision["reasons"] == [] and decision["reference_price"] == saved["reference_price"]
    assert decision["projected_fill_price"] == pytest.approx((1003 if side == "long" else 997) * (1 + (.0001 if side == "long" else -.0001)))
    assert all(key in decision for key in ["enabled", "side", "numeric_levels", "signal_origin",
        "expires_at_ms", "checked_at_ms", "candidate_low", "candidate_high", "source_key"])


@pytest.mark.parametrize("side", ["long", "short"])
def test_one_completed_bar_can_touch_then_reject_without_backdating_a_fill(side):
    saved = pending(side)
    before = copy.deepcopy(saved)
    ready, decision = advance(saved, [five(side=side)], side=side)
    assert saved == before
    assert ready["touched_at_ms"] == ready["confirmed_at_ms"] == five(side=side)[7]
    assert decision["allowed"]
    same, again = advance(ready, [five(side=side)], side=side)
    assert same == ready and again["allowed"]
    assert json.loads(json.dumps(ready)) == ready  # Persistent JSON round-trip.


@pytest.mark.parametrize("side, chasing_mark, return_to_zone", [
    ("long", 1020, 1001), ("short", 980, 999),
])
def test_ready_still_checks_slippage_and_does_not_chase_or_buy_before_rejection_held(side, chasing_mark, return_to_zone):
    ready, _ = advance(pending(side), [five(side=side)], side=side)
    stayed, decision = advance(ready, [five(side=side)], side=side, mark=chasing_mark)
    assert stayed["status"] == "ready" and not decision["allowed"]
    assert decision["reasons"] == ["btc_pullback_chase_exceeded"]
    _, decision = advance(ready, [five(side=side)], side=side, mark=return_to_zone)
    assert not decision["allowed"] and decision["reasons"] == ["btc_pullback_rejection_not_held"]
    boundary = ready["entry_high"] if side == "long" else ready["entry_low"]
    _, no_cost = advance(ready, [five(side=side)], side=side, mark=boundary, slip=0)
    _, with_cost = advance(ready, [five(side=side)], side=side, mark=boundary, slip=1)
    assert no_cost["allowed"] and not with_cost["allowed"]


@pytest.mark.parametrize("side", ["long", "short"])
def test_structure_damage_cancels_waiting_and_ready_without_same_source_resurrection(side):
    saved = pending(side)
    broken, decision = advance(saved, [five(side=side, kind="broken")], side=side)
    assert broken["status"] == "structure_invalidated" and decision["cancel_pending"]
    returned, decision = advance(broken, [five(side=side)], side=side)
    assert returned["status"] == "structure_invalidated" and not decision["allowed"]
    ready, _ = advance(saved, [five(side=side)], side=side)
    broken, _ = advance(ready, [five(side=side), five(1, side=side, kind="broken")], side=side)
    assert broken["status"] == "structure_invalidated"


def test_expiration_is_original_hour_window_and_5m_refresh_cannot_extend_it():
    saved = pending()
    ready, _ = advance(saved, [five()], desired={**target(), "signal_time_ms": ORIGIN + entry.FIVE_MINUTES})
    assert ready["expires_at_ms"] == ORIGIN + entry.HOUR
    expired, decision = advance(ready, [], now=ORIGIN + entry.HOUR)
    assert expired["status"] == "expired" and decision["cancel_pending"]
    _, decision = advance(expired, [five()], now=ORIGIN + entry.FIVE_MINUTES)
    assert not decision["allowed"]
    built = entry.build_pending(target(), hourly(), POLICY, ORIGIN + entry.HOUR)
    assert built["status"] == "expired"


@pytest.mark.parametrize("desired", [
    {**target(), "target_leverage": 0}, target("short"),
    {**target(), "position_id": "different"},
    {**target(), "origin_signal_time_ms": ORIGIN + entry.HOUR},
])
def test_flat_opposite_or_new_origin_cancels_old_pending(desired):
    changed, decision = advance(pending(), [five()], desired=desired)
    assert changed["status"] == "cancelled" and not decision["allowed"]
    assert decision["cancel_pending"] and changed["source_key"] == pending()["source_key"]


def test_future_and_forming_hourly_prices_are_not_parsed_or_used_as_pivot_confirmation():
    baseline = entry.build_pending(target(), hourly(), POLICY, ORIGIN + 1000)
    forming = compact(ORIGIN, True, float("nan"), None, "forbidden")
    future = compact(ORIGIN + entry.HOUR, None, None, None, None)
    changed = entry.build_pending(target(), hourly() + [forming, future], POLICY, ORIGIN + 1000)
    assert changed == baseline
    bars = hourly()
    bars[-1][3] = 985  # This putative pivot has no two known candles to its right.
    built = entry.build_pending(target(), bars + [forming, future], POLICY, ORIGIN + 1000)
    assert built["selected_structure"]["kind"] == "ema48"


def test_unclosed_5m_ohlc_is_ignored_and_first_bar_not_yet_closed_is_normal():
    saved = pending()
    unclosed = compact(ORIGIN, None, float("nan"), True, None, step=entry.FIVE_MINUTES)
    updated, decision = advance(saved, [unclosed], now=ORIGIN + 1000)
    assert updated == saved and decision["status"] == "waiting_retrace"
    assert not decision["allowed"] and not decision["retryable"]


@pytest.mark.parametrize("mutation", ["gap", "missing_origin", "duplicate", "bad_range", "bool_time"])
def test_bad_original_hourly_evidence_is_retryable_and_never_invents_a_structure(mutation):
    bars = hourly()
    if mutation == "gap":
        del bars[10]
    elif mutation == "missing_origin":
        bars.pop()
    elif mutation == "duplicate":
        bars.insert(10, copy.deepcopy(bars[10]))
    elif mutation == "bad_range":
        bars[10][3] = 1011
    else:
        bars[10][0] = True
    built = entry.build_pending(target(), bars, POLICY, ORIGIN + 1000)
    assert built["status"] == "data_unavailable" and built["candidate_low"] is None
    _, decision = advance(built, [], now=ORIGIN + 1000)
    assert decision["retryable"] and not decision["allowed"]


def test_five_minute_gaps_and_future_bar_only_are_retryable_without_consuming_missing_bar():
    saved = pending()
    changed, decision = advance(saved, [five(1)], now=ORIGIN + 2 * entry.FIVE_MINUTES)
    assert changed == saved and decision["status"] == "data_unavailable"
    assert decision["retryable"] and decision["reasons"] == ["btc_pullback_5m_gap"]
    recovered, decision = advance(changed, [five(), five(1)], now=ORIGIN + 2 * entry.FIVE_MINUTES)
    assert recovered["status"] == "ready" and decision["allowed"]


def test_numbers_merge_round_hundred_thousand_and_repeated_tails_without_rounding_wrong_digits():
    numbers = entry.generate_numeric_candidates(1011, 14, .1)
    by_price = {row["price"]: row for row in numbers}
    assert set(by_price) == {999, 1000, 1011, 1022}
    assert set(by_price[1000]["kinds"]) == {"round_thousand", "round_hundred", "repeated_00"}
    assert by_price[1000]["priority"] == 3
    assert by_price[1011]["kinds"] == ["repeated_11"]
    tick_three = entry.generate_numeric_candidates(1011, 14, 3)
    assert [row["price"] for row in tick_three] == [999, 1011]
    assert entry.generate_numeric_candidates(1022, 0, .1)[0]["price"] == 1022


def test_numeric_confluence_changes_selection_only_inside_nearest_structure_band():
    bars = [compact(ORIGIN - (49 - index) * entry.HOUR, 1004, 1021, 1001, 1004)
            for index in range(49)]
    bars[20][3] = 1000
    normal = entry.build_pending(target(), bars, POLICY, ORIGIN + 1000)
    assert normal["selected_structure"]["kind"] == "pivot_low"
    assert normal["selected_structure"]["price"] == 1000
    assert normal["numeric_priority"] == 3
    assert normal["selection_reason"] == "numeric_confluence_within_nearest_band"
    small_band = entry.build_pending(target(), bars, {**POLICY, "numeric_tiebreak_atr": .1}, ORIGIN + 1000)
    assert small_band["selected_structure"]["kind"] == "ema48"
    assert small_band["selected_structure"]["price"] == 1004
    assert small_band["selection_reason"] == "nearest_structure"
    assert small_band["numeric_levels"] == []


def test_digits_alone_cannot_produce_entry_when_no_legal_structural_corridor_exists():
    built = entry.build_pending(target(origin_entry_price=1003), hourly(), POLICY, ORIGIN + 1000)
    assert built["status"] == "no_structure" and built["candidate_low"] is None
    assert entry.generate_numeric_candidates(1003, 10)  # Nearby round numbers still exist.
    _, decision = advance(built, [five()], desired=target(origin_entry_price=1003))
    assert not decision["allowed"] and decision["cancel_pending"]


@pytest.mark.parametrize("changes", [
    {"origin_entry_price": True}, {"origin_entry_price": float("inf")},
    {"origin_signal_time_ms": True}, {"origin_signal_time_ms": ORIGIN + 1},
    {"origin_signal_time_ms": ORIGIN + entry.HOUR}, {"position_id": ""},
    {"position_id": False}, {"target_leverage": float("nan")},
    {"target_leverage": True},
])
def test_invalid_metadata_is_terminal_and_never_repaired_by_fresh_5m_target(changes):
    built = entry.build_pending(target(**changes), hourly(), POLICY, ORIGIN + 1000)
    assert built["status"] == "invalid_source"
    _, decision = advance(built, [five()])
    assert decision["cancel_pending"] and not decision["allowed"]


@pytest.mark.parametrize("changes", [
    {"pivot_wings": True}, {"atr_period": 0}, {"lookback_hours": 2},
    {"number_confluence_atr": float("nan")}, {"max_chase_atr": False},
    {"max_initial_entry_age_seconds": 3601}, {"round_steps": [True]},
    {"repeated_last_two_digits": 1}, {"unexpected": 1},
])
def test_policy_rejects_nonfinite_boolean_or_unapproved_extended_window(changes):
    with pytest.raises(ValueError):
        entry.validate_policy({**POLICY, **changes})


@pytest.mark.parametrize("key,bad", [
    ("candidate_high", 2000), ("entry_low", 1), ("signal_origin", ORIGIN + entry.HOUR),
    ("source_key", "different"), ("expires_at_ms", ORIGIN + 2 * entry.HOUR),
    ("atr", True), ("last_processed_open_ms", ORIGIN + 1),
    ("numeric_levels", []), ("numeric_priority", True),
])
def test_corrupt_frozen_pending_cannot_allow_trade(key, bad):
    saved = pending()
    saved[key] = bad
    updated, decision = advance(saved, [five()])
    assert updated["status"] == "invalid_source" and not decision["allowed"]


def test_bad_live_price_blocks_ready_without_mutating_frozen_confirmation():
    ready, _ = advance(pending(), [five()])
    for mark in (True, float("nan"), 0, "1003"):
        updated, decision = entry.advance_pending(ready, target(), [five()], mark,
            ORIGIN + entry.FIVE_MINUTES, 1, POLICY)
        assert updated == ready and decision["status"] == "price_unavailable"
        assert not decision["allowed"] and not decision["cancel_pending"]


def test_ready_frozen_source_stays_same_when_latest_target_price_changes():
    ready, _ = advance(pending(), [five()])
    updated, decision = advance(ready, [five()], desired={**target(), "price": 1100,
        "origin_entry_price": 1100, "signal_time_ms": ORIGIN + entry.FIVE_MINUTES})
    assert updated["reference_price"] == 1040 and updated["candidate_high"] == 1002
    assert decision["allowed"]


def test_saved_confirmation_cannot_be_used_before_that_bar_closed():
    ready, _ = advance(pending(), [five()])
    changed, decision = advance(ready, [], now=ORIGIN + 1000)
    assert changed["status"] == "invalid_source" and not decision["allowed"]
    assert decision["reasons"] == ["btc_pullback_pending_future_bar"]


def test_mapping_and_sim_candle_inputs_match_compact_cache_data():
    from simulate_range_swing import candle_from_compact
    compact_rows = hourly()
    mappings = [{"open_time_ms": row[0], "open": row[1], "high": row[2],
                 "low": row[3], "close": row[4], "close_time_ms": row[7]}
                for row in compact_rows]
    objects = [candle_from_compact(row) for row in compact_rows]
    compact_pending = entry.build_pending(target(), compact_rows, POLICY, ORIGIN + 1000)
    assert entry.build_pending(target(), mappings, POLICY, ORIGIN + 1000) == compact_pending
    assert entry.build_pending(target(), objects, POLICY, ORIGIN + 1000) == compact_pending
