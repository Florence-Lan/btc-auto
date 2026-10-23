"""Fixed, causal hourly exit hypotheses for an independent paper comparison.

This module neither trades nor changes a stop. The caller persists the returned
state and executes a latched FULL exit with its existing priority, depth, costs,
partial-fill retry and cooldown machinery. It must never execute at a historical
candle close or automatically reverse the position.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
import math


MODE = "hourly_exit_response_v1"
HOUR_MS = 3_600_000
VARIANTS = ("breakout_failure", "giveback", "combined")
CONFIRMATION_BARS = 2
ARM_R = 1.0
GIVEBACK_R = 0.5
EXIT_REASONS = {
    "breakout_failure": "hourly_exit_response_breakout_failure",
    "giveback": "hourly_exit_response_giveback",
    "combined": "hourly_exit_response_combined",
}


def validate(policy):
    """Reject undeclared threshold tuning, non-hourly modes and unknown arms."""
    if (not isinstance(policy, Mapping) or set(policy) != {"mode", "variant"}
            or policy.get("mode") != MODE or policy.get("variant") not in VARIANTS):
        raise ValueError("Requires the fixed hourly_exit_response_v1 mode and a declared variant")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stamp(value):
    return type(value) is int and value >= 0


def _decimal(value):
    return Decimal(str(value))


def _base(state=None):
    return {"status": "holding", "triggered": False, "reason": None, "reasons": [],
            "state": deepcopy(state), "exit_action": "close_full_position",
            "automatic_reversal": False, "protective_exits_allowed": True,
            "preserve_original_protective_stop": True,
            "confirmation_bars_required": CONFIRMATION_BARS,
            "giveback_arm_r": ARM_R, "giveback_threshold_r": GIVEBACK_R}


def _unavailable(state, reason):
    result = _base(state)
    result.update(status="unavailable", reason=reason, reasons=[reason])
    return result


def _identity(position, policy, interval_ms, inception_ms, entry_trigger):
    direction, entry = position.get("direction"), position.get("entry")
    initial_stop, entry_time = position.get("initial_stop"), position.get("entry_time")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("invalid_position_direction")
    if (not _number(entry) or entry <= 0 or not _number(initial_stop) or initial_stop <= 0
            or not _stamp(entry_time)):
        raise ValueError("missing_or_invalid_immutable_entry_risk")
    # A tightened current stop is deliberately never a fallback for initial_stop.
    distance = direction * (_decimal(entry) - _decimal(initial_stop))
    if distance <= 0:
        raise ValueError("initial_stop_not_on_original_risk_side")
    if not isinstance(entry_trigger, Mapping):
        raise ValueError("original_entry_trigger_required")
    level, boundary = entry_trigger.get("breakout_level"), entry_trigger.get("boundary_ms")
    if (not _number(level) or level <= 0 or type(entry_trigger.get("direction")) is not int
            or entry_trigger.get("direction") != direction or not _stamp(boundary)
            or boundary > entry_time):
        raise ValueError("invalid_original_entry_trigger")
    return {"mode": MODE, "variant": policy["variant"], "interval_ms": interval_ms,
            "inception_ms": inception_ms, "entry_time_ms": entry_time,
            "entry_price": entry, "initial_stop": initial_stop, "direction": direction,
            "entry_trigger": {"breakout_level": level, "direction": direction,
                              "boundary_ms": boundary}}, distance


def _new_state(identity, risk_distance):
    return {"schema_version": 1, "identity": deepcopy(identity),
            "risk_distance": float(risk_distance),
            "tracking_after_ms": max(identity["inception_ms"], identity["entry_time_ms"]),
            "last_processed_closed_at_ms": None, "last_processed_close": None,
            "processed_closed_bars": 0,
            "failure_confirmations": 0, "last_failure_closed_at_ms": None,
            "peak_favorable_distance": 0.0, "peak_favorable_r": 0.0,
            "peak_closed_at_ms": None, "last_favorable_r": None, "last_giveback_r": None,
            "armed": False, "latched": False, "trigger_reasons": [],
            "exit_reason": None, "trigger_closed_at_ms": None, "detected_at_ms": None}


def _validate_state(state, identity, risk_distance, timestamp):
    if (not isinstance(state, Mapping) or type(state.get("schema_version")) is not int
            or state.get("schema_version") != 1
            or state.get("identity") != identity):
        raise ValueError("exit_response_state_identity_mismatch")
    if (not _number(state.get("risk_distance"))
            or not math.isclose(state["risk_distance"], float(risk_distance), rel_tol=1e-14)
            or state.get("tracking_after_ms") != max(identity["inception_ms"], identity["entry_time_ms"])):
        raise ValueError("exit_response_state_risk_mismatch")
    last = state.get("last_processed_closed_at_ms")
    if last is not None and (not _stamp(last) or last % HOUR_MS or last > timestamp
                             or last <= state["tracking_after_ms"]):
        raise ValueError("invalid_or_future_exit_response_state")
    count = state.get("failure_confirmations")
    bars = state.get("processed_closed_bars")
    peak, peak_r = state.get("peak_favorable_distance"), state.get("peak_favorable_r")
    if (type(count) is not int or not 0 <= count <= CONFIRMATION_BARS
            or type(bars) is not int or bars < 0
            or not _number(peak) or peak < 0 or not _number(peak_r) or peak_r < 0
            or not math.isclose(float(_decimal(peak) / risk_distance), peak_r, rel_tol=1e-12, abs_tol=1e-12)
            or type(state.get("armed")) is not bool or type(state.get("latched")) is not bool
            or state["armed"] != (_decimal(peak) >= risk_distance)):
        raise ValueError("invalid_exit_response_progress")
    if last is None:
        if bars or count or peak or state.get("last_processed_close") is not None:
            raise ValueError("invalid_unprocessed_exit_response_state")
    elif (not _number(state.get("last_processed_close")) or state["last_processed_close"] <= 0
          or bars < 1):
        raise ValueError("invalid_exit_response_last_close")
    failure_at = state.get("last_failure_closed_at_ms")
    if count and failure_at != last:
        raise ValueError("invalid_exit_response_confirmation_state")
    if not count and failure_at is not None:
        raise ValueError("invalid_exit_response_confirmation_state")
    peak_at = state.get("peak_closed_at_ms")
    if peak > 0 and (not _stamp(peak_at) or last is None
                     or not state["tracking_after_ms"] < peak_at <= last):
        raise ValueError("invalid_exit_response_peak_time")
    if state["latched"]:
        reasons = state.get("trigger_reasons")
        allowed = {"breakout_failure", "giveback"} if identity["variant"] == "combined" else {identity["variant"]}
        trigger_at, detected = state.get("trigger_closed_at_ms"), state.get("detected_at_ms")
        if (not isinstance(reasons, list) or not reasons or any(not isinstance(r, str) for r in reasons)
                or len(set(reasons)) != len(reasons)
                or not set(reasons) <= allowed or state.get("exit_reason") not in EXIT_REASONS.values()
                or not _stamp(trigger_at) or not _stamp(detected) or last is None
                or not state["tracking_after_ms"] < trigger_at <= last <= detected <= timestamp):
            raise ValueError("invalid_or_future_latched_exit")
        expected = EXIT_REASONS["combined" if len(reasons) == 2 else reasons[0]]
        if state["exit_reason"] != expected:
            raise ValueError("latched_exit_reason_mismatch")
        if ("breakout_failure" in reasons and count != CONFIRMATION_BARS
                or "giveback" in reasons and (not state["armed"]
                    or not _number(state.get("last_giveback_r"))
                    or state["last_giveback_r"] < GIVEBACK_R)):
            raise ValueError("latched_exit_condition_not_preserved")
    elif any(state.get(key) is not None for key in ("exit_reason", "trigger_closed_at_ms", "detected_at_ms")):
        raise ValueError("invalid_unlatched_exit_state")
    elif state.get("trigger_reasons") != []:
        raise ValueError("invalid_unlatched_exit_reasons")


def _trigger_result(state, *, newly_triggered=False):
    result = _base(state)
    result.update(status="triggered" if newly_triggered else "latched", triggered=True,
                  reason=state["exit_reason"], reasons=list(state["trigger_reasons"]),
                  trigger_closed_at_ms=state["trigger_closed_at_ms"], detected_at_ms=state["detected_at_ms"],
                  closed_at_ms=state["last_processed_closed_at_ms"],
                  confirmation_bars=state["failure_confirmations"],
                  peak_favorable_r=state["peak_favorable_r"],
                  favorable_r=state["last_favorable_r"], giveback_r=state["last_giveback_r"])
    return result


def _candle(candle):
    if isinstance(candle, Mapping):
        return candle.get("time_ms"), candle.get("close")
    return getattr(candle, "time_ms", None), getattr(candle, "close", None)


def evaluate(position, candles, config, timestamp, interval_ms, inception_ms,
             entry_trigger=None, state=None):
    """Return an immutable-input evaluation and a new JSON-safe progress state.

    Only closes strictly AFTER max(entry time, experiment inception) update the
    state. A bar that opened before entry may contribute its later close: this
    rule reads no intrabar high/low and uses no pre-entry favorable extrema.
    Full history is allowed, but processed bars cannot count twice. A signal
    detected from delayed history is executable only after detected_at_ms, not
    at trigger_closed_at_ms. A latched exit survives a restart/data outage.
    """
    policy = config.get("exit_response_policy")
    validate(policy)
    if not position:
        result = _base()
        result["status"] = "flat"
        return result
    if not isinstance(position, Mapping):
        return _unavailable(state, "invalid_position")
    prior = state if state is not None else position.get("exit_response_state")
    pending_exit = position.get("pending_exit")
    # An existing original hard/protective exit always remains in charge. Own
    # latched exits may be re-emitted for remaining-depth retries below.
    if pending_exit and pending_exit not in EXIT_REASONS.values():
        result = _base(prior)
        result.update(status="existing_exit_priority", pending_exit=deepcopy(pending_exit),
                      reason="existing_exit_has_priority", reasons=["existing_exit_has_priority"])
        return result
    if interval_ms != HOUR_MS or type(interval_ms) is not int:
        return _unavailable(prior, "hourly_interval_required")
    if not _stamp(timestamp) or not _stamp(inception_ms) or inception_ms > timestamp:
        return _unavailable(prior, "invalid_or_future_inception")
    trigger = entry_trigger if entry_trigger is not None else position.get("entry_trigger")
    try:
        identity, distance = _identity(position, policy, interval_ms, inception_ms, trigger)
    except ValueError as exc:
        return _unavailable(prior, str(exc))
    if position["entry_time"] > timestamp:
        return _unavailable(prior, "future_position_entry")
    if prior is None:
        progress = _new_state(identity, distance)
    else:
        try:
            _validate_state(prior, identity, distance, timestamp)
        except ValueError as exc:
            return _unavailable(prior, str(exc))
        progress = deepcopy(prior)
    if progress["latched"]:
        return _trigger_result(progress)
    if pending_exit:
        # A persisted pending request without its original latch is not rebuilt
        # from market history; the caller must retry its existing exit request.
        result = _base(progress)
        result.update(status="existing_exit_priority", pending_exit=deepcopy(pending_exit),
                      reason="existing_exit_has_priority", reasons=["existing_exit_has_priority"])
        return result
    if not candles:
        return _unavailable(progress, "missing_closed_candles")
    bars = []
    for candle in candles:
        opened, close = _candle(candle)
        if not _stamp(opened) or opened % HOUR_MS or not _number(close) or close <= 0:
            return _unavailable(progress, "invalid_hourly_closed_candle")
        closed_at = opened + interval_ms
        if closed_at > timestamp:
            return _unavailable(progress, "unfinished_candle")
        if bars:
            gap = closed_at - bars[-1][0]
            if gap == 0:
                return _unavailable(progress, "duplicate_closed_bar")
            if gap < 0:
                return _unavailable(progress, "out_of_order_closed_bars")
            if gap != interval_ms:
                return _unavailable(progress, "nonconsecutive_closed_bars")
        bars.append((closed_at, close))
    if not bars:
        return _unavailable(progress, "missing_closed_candles")
    if timestamp - bars[-1][0] > interval_ms + 5000:
        return _unavailable(progress, "stale_closed_candles")
    last = progress["last_processed_closed_at_ms"]
    if last is not None:
        if bars[-1][0] < last:
            return _unavailable(progress, "regressed_closed_history")
        same_bar = next((close for at, close in bars if at == last), None)
        if same_bar is not None and _decimal(same_bar) != _decimal(progress["last_processed_close"]):
            return _unavailable(progress, "processed_closed_bar_revised")
    new_bars = [(at, close) for at, close in bars
                if at > progress["tracking_after_ms"] and (last is None or at > last)]
    if last is not None and new_bars and new_bars[0][0] != last + interval_ms:
        return _unavailable(progress, "missing_unprocessed_closed_bars")
    if not new_bars:
        result = _base(progress)
        result.update(status="waiting_for_closed_bar" if last is None else "unchanged_closed_bar",
                      closed_at_ms=last, confirmation_bars=progress["failure_confirmations"],
                      peak_favorable_r=progress["peak_favorable_r"])
        return result
    side = identity["direction"]
    entry, level = _decimal(identity["entry_price"]), _decimal(trigger["breakout_level"])
    for closed_at, close in new_bars:
        current = _decimal(close)
        favorable = side * (current - entry)
        peak = max(_decimal(progress["peak_favorable_distance"]), favorable)
        if peak > _decimal(progress["peak_favorable_distance"]):
            progress["peak_closed_at_ms"] = closed_at
        progress.update(peak_favorable_distance=float(peak), peak_favorable_r=float(peak / distance),
                        armed=peak >= distance, last_favorable_r=float(favorable / distance),
                        last_giveback_r=float((peak - favorable) / distance))
        failed = side * (current - level) <= 0  # Equality no longer satisfies the original strict breakout.
        if failed:
            previous_failure = progress["last_failure_closed_at_ms"]
            consecutive = previous_failure is not None and closed_at == previous_failure + interval_ms
            progress["failure_confirmations"] = min(CONFIRMATION_BARS,
                progress["failure_confirmations"] + 1 if consecutive else 1)
            progress["last_failure_closed_at_ms"] = closed_at
        else:
            progress.update(failure_confirmations=0, last_failure_closed_at_ms=None)
        progress.update(last_processed_closed_at_ms=closed_at, last_processed_close=close,
                        processed_closed_bars=progress["processed_closed_bars"] + 1)
        reasons = []
        if policy["variant"] in ("breakout_failure", "combined") and progress["failure_confirmations"] == CONFIRMATION_BARS:
            reasons.append("breakout_failure")
        if (policy["variant"] in ("giveback", "combined") and peak >= distance
                and peak - favorable >= distance / 2):
            reasons.append("giveback")
        if reasons:
            selected = "combined" if len(reasons) == 2 else reasons[0]
            progress.update(latched=True, trigger_reasons=reasons, exit_reason=EXIT_REASONS[selected],
                            trigger_closed_at_ms=closed_at, detected_at_ms=timestamp)
            return _trigger_result(progress, newly_triggered=True)
    result = _base(progress)
    result.update(status="confirming" if progress["failure_confirmations"] else "holding",
                  closed_at_ms=progress["last_processed_closed_at_ms"],
                  confirmation_bars=progress["failure_confirmations"],
                  peak_favorable_r=progress["peak_favorable_r"],
                  favorable_r=progress["last_favorable_r"], giveback_r=progress["last_giveback_r"])
    return result
