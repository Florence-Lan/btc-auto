"""Fixed hourly exit behavior, causal state, restart and protective priority."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stock_exit_response as exits

HOUR = exits.HOUR_MS
INCEPTION = 200 * HOUR


def policy(variant="combined"):
    return {"exit_response_policy": {"mode": exits.MODE, "variant": variant}}


def position(side=1, *, entry=100, risk=2, level=None):
    return {"direction": side, "entry": entry, "entry_time": INCEPTION,
            "initial_stop": entry - side * risk, "stop": entry - side * risk,
            "qty": 1.0, "entry_fee": .1, "funding": .02,
            "entry_trigger": {"breakout_level": entry - side if level is None else level,
                              "direction": side, "boundary_ms": INCEPTION}}


def bars(*closes, first_close=None, as_mapping=False):
    first_close = first_close if first_close is not None else INCEPTION + HOUR
    result = []
    for i, close in enumerate(closes):
        record = {"time_ms": first_close + i * HOUR - HOUR, "close": close,
                  "open": close, "high": close + 100, "low": close - 90, "volume": 999}
        result.append(record if as_mapping else SimpleNamespace(**record))
    return result


def evaluate(pos, data, variant="combined", *, state=None, timestamp=None,
             inception=INCEPTION, trigger=None, interval=HOUR):
    if timestamp is None:
        if data:
            last = data[-1]
            timestamp = (last["time_ms"] if isinstance(last, dict) else last.time_ms) + HOUR + 1000
        else:
            timestamp = INCEPTION + HOUR + 1000
    return exits.evaluate(pos, data, policy(variant), timestamp, interval, inception,
                          entry_trigger=trigger, state=state)


@pytest.mark.parametrize("variant", exits.VARIANTS)
def test_only_fixed_declared_variants_are_valid(variant):
    exits.validate(policy(variant)["exit_response_policy"])


@pytest.mark.parametrize("bad", [None, {}, {"mode": exits.MODE},
    {"mode": "quick_exit", "variant": "combined"},
    {"mode": exits.MODE, "variant": "reverse_position"},
    {"mode": exits.MODE, "variant": "combined", "confirmation_bars": 1},
    {"mode": exits.MODE, "variant": "giveback", "arm_r": .75}])
def test_threshold_search_and_unknown_policies_are_rejected(bad):
    with pytest.raises(ValueError, match="fixed hourly_exit_response_v1"):
        exits.validate(bad)


@pytest.mark.parametrize("side", [1, -1])
def test_breakout_failure_needs_two_unique_consecutive_closes_and_equality_counts(side):
    pos = position(side)
    level = pos["entry_trigger"]["breakout_level"]
    one = evaluate(pos, bars(level), "breakout_failure")
    assert one["status"] == "confirming" and not one["triggered"]
    assert one["state"]["failure_confirmations"] == 1
    same = evaluate(pos, bars(level), "breakout_failure", state=one["state"])
    assert not same["triggered"] and same["state"] == one["state"]
    assert same["status"] == "unchanged_closed_bar"
    two = evaluate(pos, bars(level, level - side * .1), "breakout_failure", state=same["state"])
    assert two["triggered"] and two["reason"] == exits.EXIT_REASONS["breakout_failure"]
    assert two["state"]["failure_confirmations"] == 2
    assert two["exit_action"] == "close_full_position" and not two["automatic_reversal"]


@pytest.mark.parametrize("side", [1, -1])
def test_reclaimed_breakout_resets_failure_confirmation(side):
    pos = position(side)
    level = pos["entry_trigger"]["breakout_level"]
    result = evaluate(pos, bars(level, level + side * .01, level), "breakout_failure")
    assert not result["triggered"] and result["confirmation_bars"] == 1
    assert result["state"]["last_failure_closed_at_ms"] == INCEPTION + 3 * HOUR


@pytest.mark.parametrize("side", [1, -1])
def test_giveback_arms_at_exact_one_r_and_triggers_at_exact_half_r(side):
    pos = position(side)
    one_r, just_above_half_r, half_r = (100 + side * v for v in (2, 1.0001, 1))
    armed = evaluate(pos, bars(one_r), "giveback")
    assert armed["state"]["armed"] and not armed["triggered"]
    assert armed["state"]["peak_favorable_r"] == 1
    holding = evaluate(pos, bars(one_r, just_above_half_r), "giveback", state=armed["state"])
    assert not holding["triggered"]
    out = evaluate(pos, bars(one_r, just_above_half_r, half_r), "giveback", state=holding["state"])
    assert out["triggered"] and out["reason"] == exits.EXIT_REASONS["giveback"]
    assert out["favorable_r"] == out["giveback_r"] == .5


@pytest.mark.parametrize("side", [1, -1])
def test_decimal_tick_boundaries_do_not_miss_mathematical_one_r(side):
    # Binary float subtraction of 102.1 - 100 can otherwise land below 2.1.
    pos = position(side, risk=2.1)
    peak = 102.1 if side == 1 else 97.9
    close = 101.05 if side == 1 else 98.95
    result = evaluate(pos, bars(peak, close), "giveback")
    assert result["triggered"] and result["state"]["risk_distance"] == pytest.approx(2.1)


def test_unarmed_profit_is_not_protected_and_high_low_do_not_arm():
    pos = position()
    data = bars(101.99, 99, 95)
    result = evaluate(pos, data, "giveback")
    assert not result["triggered"] and not result["state"]["armed"]
    assert result["state"]["peak_favorable_r"] == pytest.approx(.995)
    assert result["state"]["last_giveback_r"] > .5
    # Enormous intrahour highs, lows and old main-strategy peaks never count.
    pos["best_close_return"] = 999
    pos["profit_exit"] = {"best_executable": 100000, "armed": True}
    again = evaluate(pos, data, "giveback")
    assert again["state"] == result["state"]


def test_risk_distance_is_immutable_initial_stop_not_tightened_protective_stop():
    pos = position()
    armed = evaluate(pos, bars(102), "giveback")
    pos["stop"] = 101.8
    result = evaluate(pos, bars(102, 101), "giveback", state=armed["state"])
    assert result["triggered"] and result["state"]["risk_distance"] == 2
    assert pos["stop"] == 101.8


@pytest.mark.parametrize("variant,closes,expected", [
    ("combined", (99, 99), "breakout_failure"),
    ("combined", (102, 101), "giveback"),
    ("giveback", (99, 99), None),
    ("breakout_failure", (102, 101), None),
])
def test_combined_is_or_while_each_single_arm_preserves_its_scope(variant, closes, expected):
    result = evaluate(position(), bars(*closes), variant)
    assert result["triggered"] is (expected is not None)
    if expected:
        assert result["reasons"] == [expected]


def test_preentry_and_preactivation_closed_bars_do_not_create_peak_or_confirmation():
    pos = position()
    # 200h closes exactly at activation/entry and is excluded; previous bar also excluded.
    history = bars(500, 500, 100, first_close=INCEPTION - HOUR)
    result = evaluate(pos, history, "giveback")
    assert not result["triggered"] and result["state"]["peak_favorable_r"] == 0
    assert result["state"]["processed_closed_bars"] == 1
    late_activation = evaluate(pos, bars(98, 98), "breakout_failure", inception=INCEPTION + HOUR)
    assert not late_activation["triggered"] and late_activation["confirmation_bars"] == 1


def test_mid_hour_entry_uses_later_close_but_no_preentry_extrema():
    pos = position()
    pos["entry_time"] = INCEPTION + HOUR // 2
    result = evaluate(pos, bars(99), "breakout_failure")
    assert result["confirmation_bars"] == 1 and not result["triggered"]
    assert result["state"]["peak_favorable_r"] == 0


def test_closes_equal_to_entry_and_activation_are_excluded():
    result = evaluate(position(), bars(98, first_close=INCEPTION), "breakout_failure")
    assert result["status"] == "waiting_for_closed_bar"
    assert result["state"]["processed_closed_bars"] == 0
    assert result["confirmation_bars"] == 0


def test_pure_function_preserves_all_inputs_and_roundtrips_persisted_state():
    pos, data, cfg = position(), bars(102), policy("giveback")
    before = deepcopy((pos, data, cfg))
    result = exits.evaluate(pos, data, cfg, INCEPTION + HOUR + 1000, HOUR, INCEPTION)
    assert (pos, data, cfg) == before
    pos["exit_response_state"] = json.loads(json.dumps(result["state"]))
    snapshot = deepcopy(pos)
    restarted = evaluate(pos, bars(102, 101), "giveback")
    assert restarted["triggered"] and pos == snapshot
    assert restarted["state"] is not pos["exit_response_state"]


def test_partial_exit_latch_survives_restart_missing_quotes_and_changed_residual_quantity():
    pos = position()
    result = evaluate(pos, bars(99, 99), "breakout_failure")
    pos["exit_response_state"] = json.loads(json.dumps(result["state"]))
    pos.update(qty=.2, pending_exit=exits.EXIT_REASONS["breakout_failure"])
    retry = evaluate(pos, [], "breakout_failure", timestamp=INCEPTION + 2 * HOUR + 30000)
    assert retry["status"] == "latched" and retry["triggered"]
    assert retry["state"] == result["state"]
    assert retry["detected_at_ms"] == result["detected_at_ms"]
    assert pos["qty"] == .2 and pos["stop"] == 98


@pytest.mark.parametrize("pending", ["protective_stop", "account_hard_stop", "liquidation_guard", "funding_budget"])
def test_original_pending_protective_exit_has_priority_even_over_a_latched_candidate(pending):
    pos = position()
    result = evaluate(pos, bars(99, 99), "breakout_failure")
    pos.update(pending_exit=pending, exit_response_state=result["state"])
    out = evaluate(pos, [], "breakout_failure", timestamp=INCEPTION + 3 * HOUR)
    assert out["status"] == "existing_exit_priority" and not out["triggered"]
    assert out["pending_exit"] == pending and out["state"] == result["state"]
    assert out["protective_exits_allowed"]


def test_existing_protective_exit_is_not_disabled_by_missing_optional_entry_metadata():
    out = evaluate({"pending_exit": "protective_stop"}, [], timestamp=INCEPTION)
    assert out["status"] == "existing_exit_priority" and not out["triggered"]


def test_latch_without_pending_exit_still_reemits_on_missing_closed_data():
    pos = position()
    result = evaluate(pos, bars(102, 101), "giveback")
    retry = evaluate(pos, [], "giveback", state=result["state"], timestamp=INCEPTION + 3 * HOUR)
    assert retry["triggered"] and retry["state"] == result["state"]


def test_delayed_history_detection_never_claims_an_earlier_execution_time():
    data = bars(99, 99, 110, 120)
    now = INCEPTION + 4 * HOUR + 1000
    out = evaluate(position(), data, "breakout_failure", timestamp=now)
    assert out["trigger_closed_at_ms"] == INCEPTION + 2 * HOUR
    assert out["detected_at_ms"] == now
    assert out["state"]["last_processed_closed_at_ms"] == out["trigger_closed_at_ms"]
    assert out["state"]["processed_closed_bars"] == 2


@pytest.mark.parametrize("reason,data,now", [
    ("unfinished_candle", bars(99), INCEPTION + HOUR - 1),
    ("stale_closed_candles", bars(99), INCEPTION + 2 * HOUR + 5001),
    ("duplicate_closed_bar", bars(99) + bars(99), INCEPTION + HOUR + 1000),
    ("out_of_order_closed_bars", list(reversed(bars(99, 99))), INCEPTION + 2 * HOUR + 1000),
    ("nonconsecutive_closed_bars", bars(99) + bars(99, first_close=INCEPTION + 3 * HOUR), INCEPTION + 3 * HOUR + 1000),
])
def test_unusable_closed_data_never_produces_a_new_trigger(reason, data, now):
    out = evaluate(position(), data, "breakout_failure", timestamp=now)
    assert out["status"] == "unavailable" and not out["triggered"]
    assert out["reason"] == reason and out["state"]["processed_closed_bars"] == 0


def test_missing_unprocessed_bar_is_not_silently_skipped_or_counted_as_consecutive():
    pos = position()
    one = evaluate(pos, bars(99), "breakout_failure")
    missing = evaluate(pos, bars(99, first_close=INCEPTION + 3 * HOUR),
                       "breakout_failure", state=one["state"])
    assert missing["reason"] == "missing_unprocessed_closed_bars" and not missing["triggered"]
    assert missing["state"] == one["state"]
    contiguous = evaluate(pos, bars(99, first_close=INCEPTION + 2 * HOUR),
                          "breakout_failure", state=one["state"])
    assert contiguous["triggered"]


def test_processed_bar_cannot_be_revised_to_retroactively_create_a_peak_or_confirmation():
    pos = position()
    one = evaluate(pos, bars(101), "giveback")
    result = evaluate(pos, bars(105, 101), "giveback", state=one["state"])
    assert not result["triggered"] and result["reason"] == "processed_closed_bar_revised"
    assert result["state"] == one["state"]


def test_regressed_history_is_unavailable_and_does_not_reset_progress():
    pos = position()
    one = evaluate(pos, bars(101, 102), "giveback")
    out = evaluate(pos, bars(101), "giveback", state=one["state"], timestamp=INCEPTION + 2 * HOUR + 1000)
    assert out["reason"] == "regressed_closed_history" and out["state"] == one["state"]


@pytest.mark.parametrize("change", [
    lambda p: p.update(entry=100.1),
    lambda p: p.update(initial_stop=97),
    lambda p: p.update(entry_time=INCEPTION + 1),
    lambda p: p["entry_trigger"].update(breakout_level=98),
])
def test_progress_cannot_be_reused_for_a_changed_position_or_original_trigger(change):
    pos = position()
    state = evaluate(pos, bars(101), "giveback")["state"]
    change(pos)
    result = evaluate(pos, bars(101, 102), "giveback", state=state)
    assert result["reason"] == "exit_response_state_identity_mismatch"
    assert result["state"] == state


def test_progress_cannot_cross_variant_or_experiment_inception():
    pos = position()
    state = evaluate(pos, bars(101), "giveback")["state"]
    for variant, inception in [("combined", INCEPTION), ("giveback", INCEPTION + 1)]:
        out = evaluate(pos, bars(101, 102), variant, inception=inception, state=state)
        assert out["reason"] == "exit_response_state_identity_mismatch"


@pytest.mark.parametrize("change,reason", [
    (lambda p: p.pop("initial_stop"), "missing_or_invalid_immutable_entry_risk"),
    (lambda p: p.update(initial_stop=101), "initial_stop_not_on_original_risk_side"),
    (lambda p: p.update(direction=True), "invalid_position_direction"),
    (lambda p: p.pop("entry_trigger"), "original_entry_trigger_required"),
    (lambda p: p["entry_trigger"].update(direction=-1), "invalid_original_entry_trigger"),
    (lambda p: p["entry_trigger"].update(boundary_ms=INCEPTION + 1), "invalid_original_entry_trigger"),
])
def test_invalid_optional_metadata_is_unavailable_and_never_changes_original_risk(change, reason):
    pos = position()
    change(pos)
    before = deepcopy(pos)
    out = evaluate(pos, bars(99, 99))
    assert out["status"] == "unavailable" and out["reason"] == reason
    assert out["protective_exits_allowed"] and pos == before


@pytest.mark.parametrize("field,value", [("close", float("nan")), ("close", float("inf")),
                                        ("close", True), ("time_ms", INCEPTION + 1)])
def test_nonfinite_or_nonhourly_bars_are_rejected(field, value):
    data = bars(100, as_mapping=True)
    data[0][field] = value
    out = evaluate(position(), data, timestamp=INCEPTION + HOUR + 1000)
    assert out["reason"] == "invalid_hourly_closed_candle" and not out["triggered"]


def test_only_closed_hourly_interval_and_nonfuture_context_are_accepted():
    assert evaluate(position(), bars(99), interval=900000)["reason"] == "hourly_interval_required"
    assert evaluate(position(), bars(99), inception=INCEPTION + 2 * HOUR)["reason"] == "invalid_or_future_inception"
    pos = position()
    pos["entry_time"] = INCEPTION + 3 * HOUR
    assert evaluate(pos, bars(99))["reason"] == "future_position_entry"


def test_future_latched_state_is_never_available_before_actual_detection():
    pos = position()
    out = evaluate(pos, bars(99, 99, 105), "breakout_failure", timestamp=INCEPTION + 3 * HOUR + 1000)
    past = evaluate(pos, [], "breakout_failure", state=out["state"], timestamp=INCEPTION + 2 * HOUR + 1000)
    assert not past["triggered"] and past["reason"] == "invalid_or_future_latched_exit"


def test_invalid_persisted_state_does_not_silently_reset_r_or_peak():
    pos = position()
    one = evaluate(pos, bars(102), "giveback")
    damaged = deepcopy(one["state"])
    damaged["peak_favorable_r"] = 0
    out = evaluate(pos, bars(102, 101), "giveback", state=damaged)
    assert out["reason"] == "invalid_exit_response_progress" and out["state"] == damaged


def test_flat_is_no_exit_and_explicit_original_trigger_matches_position_fallback():
    assert evaluate(None, [])["status"] == "flat"
    pos, data = position(), bars(99, 99)
    explicit = evaluate(pos, data, "breakout_failure", trigger=pos["entry_trigger"])
    fallback = evaluate(pos, data, "breakout_failure")
    assert explicit == fallback


def test_empty_iterator_is_a_missing_feed_not_an_exception():
    out = evaluate(position(), iter(()), timestamp=INCEPTION + HOUR + 1000)
    assert out["reason"] == "missing_closed_candles" and not out["triggered"]


def test_corrupt_latch_cannot_assert_an_unconfirmed_breakout_exit():
    pos = position()
    out = evaluate(pos, bars(99, 99), "breakout_failure")
    damaged = deepcopy(out["state"])
    damaged["failure_confirmations"] = 1
    retry = evaluate(pos, [], "breakout_failure", state=damaged, timestamp=INCEPTION + 3 * HOUR)
    assert retry["reason"] == "latched_exit_condition_not_preserved" and not retry["triggered"]
