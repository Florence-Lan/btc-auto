"""Pure, causal structural pullback/rebound admission for new BTC directions.

Hourly structures and their numerical confluence are frozen at the original
signal. Completed five-minute candles establish touch and rejection; a fresh
execution quote must still pass the frozen price corridor. This module never
places orders, changes exits, or decides the underlying trend direction.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math


HOUR = 3_600_000
FIVE_MINUTES = 300_000
MODE = "structural_pullback_v1"
DEFAULT_POLICY = {
    "mode": MODE, "lookback_hours": 48, "pivot_wings": 2, "atr_period": 14,
    "zone_half_width_atr": .10, "number_confluence_atr": .15,
    "numeric_tiebreak_atr": .25, "max_retrace_atr": 2.0,
    "min_improvement_atr": .10, "max_chase_atr": .20,
    "invalidate_beyond_zone_atr": .50, "round_steps": [100, 1000],
    "repeated_last_two_digits": True, "max_initial_entry_age_seconds": 3600,
}
POLICY = DEFAULT_POLICY
TERMINAL_STATUSES = frozenset({
    "no_structure", "expired", "structure_invalidated", "cancelled", "invalid_source",
})
ACTIVE_STATUSES = frozenset({"waiting_retrace", "waiting_confirmation", "ready"})


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(name)
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name) from exc
    if (not math.isfinite(result) or (positive and result <= 0)
            or (nonnegative and result < 0)):
        raise ValueError(name)
    return result


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name)
    return value


def validate_policy(value):
    if not isinstance(value, Mapping) or value.get("mode") != MODE:
        raise ValueError("btc_pullback_policy_invalid")
    if set(value) - set(DEFAULT_POLICY):
        raise ValueError("btc_pullback_policy_unknown_key")
    result = {**DEFAULT_POLICY, **value}
    for key in ("lookback_hours", "pivot_wings", "atr_period"):
        _integer(result[key], "btc_pullback_" + key + "_invalid", minimum=1)
    if (not 1 <= result["pivot_wings"] <= 12 or not 1 <= result["atr_period"] <= 96
            or not max(48, result["atr_period"] + 1, 2 * result["pivot_wings"] + 1)
            <= result["lookback_hours"] <= 720):
        raise ValueError("btc_pullback_history_policy_invalid")
    for key in ("zone_half_width_atr", "number_confluence_atr", "numeric_tiebreak_atr",
                "max_retrace_atr", "min_improvement_atr", "max_chase_atr",
                "invalidate_beyond_zone_atr", "max_initial_entry_age_seconds"):
        result[key] = _number(result[key], "btc_pullback_" + key + "_invalid", positive=True)
    if result["max_initial_entry_age_seconds"] > 3600:
        raise ValueError("btc_pullback_original_entry_window_must_not_expand")
    if result["zone_half_width_atr"] + result["max_chase_atr"] + result["min_improvement_atr"] >= result["max_retrace_atr"]:
        raise ValueError("btc_pullback_price_corridor_invalid")
    steps = result["round_steps"]
    if (not isinstance(steps, list) or not steps
            or any(type(step) is not int or step <= 0 for step in steps)):
        raise ValueError("btc_pullback_round_steps_invalid")
    result["round_steps"] = sorted(set(steps))
    if type(result["repeated_last_two_digits"]) is not bool:
        raise ValueError("btc_pullback_repeated_digits_invalid")
    return result


def _source(target):
    if not isinstance(target, Mapping):
        raise ValueError("btc_pullback_source_invalid")
    if "target_leverage" in target:
        exposure = _number(target["target_leverage"], "btc_pullback_direction_invalid")
    elif "signed_qty" in target:
        exposure = _number(target["signed_qty"], "btc_pullback_direction_invalid")
    else:
        raise ValueError("btc_pullback_direction_missing")
    if exposure == 0:
        return None
    origin = _integer(target.get("origin_signal_time_ms"), "btc_pullback_origin_invalid", minimum=1)
    if origin % HOUR:
        raise ValueError("btc_pullback_origin_alignment_invalid")
    identifier = target.get("position_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("btc_pullback_position_id_invalid")
    side = "long" if exposure > 0 else "short"
    return {"position_id": identifier, "origin_signal_time_ms": origin, "side": side,
            "source_key": f"{identifier}:{origin}:{side}"}


def generate_numeric_candidates(reference, radius, tick_size=.1, *, round_steps=(100, 1000),
                                repeated_last_two_digits=True):
    """Return exact tradable integers: round levels and integer-part tails AA.

    A digit candidate is the integer itself, never a rounded neighboring price
    whose digits no longer match. Overlapping 00, hundred and thousand rules are
    merged. Only a bounded neighborhood is enumerated.
    """
    center = _number(reference, "btc_pullback_numeric_reference_invalid", positive=True)
    width = _number(radius, "btc_pullback_numeric_radius_invalid", nonnegative=True)
    tick = _number(tick_size, "btc_pullback_tick_size_invalid", positive=True)
    if (not isinstance(round_steps, (tuple, list)) or not round_steps
            or any(type(step) is not int or step <= 0 for step in round_steps)
            or type(repeated_last_two_digits) is not bool):
        raise ValueError("btc_pullback_numeric_rules_invalid")
    lo, hi = max(Decimal(0), Decimal(str(center)) - Decimal(str(width))), Decimal(str(center)) + Decimal(str(width))
    if hi - lo > Decimal(100_000):
        raise ValueError("btc_pullback_numeric_radius_too_large")
    quantum = Decimal(str(tick))
    values = {}

    def add(price, label, priority):
        candidate = Decimal(price)
        if not lo <= candidate <= hi or candidate <= 0 or candidate % quantum:
            return
        item = values.setdefault(candidate, {"price": float(candidate), "kinds": [], "priority": 0})
        if label not in item["kinds"]:
            item["kinds"].append(label)
        item["priority"] = max(item["priority"], priority)

    for step in sorted(set(round_steps), reverse=True):
        first = int((lo / step).to_integral_value(rounding=ROUND_CEILING))
        last = int((hi / step).to_integral_value(rounding=ROUND_FLOOR))
        for multiple in range(first, last + 1):
            label = "round_thousand" if step == 1000 else "round_hundred" if step == 100 else f"round_{step}"
            add(multiple * step, label, 3 if step == 1000 else 2)
    if repeated_last_two_digits:
        for hundred in range(int(lo // 100), int(hi // 100) + 1):
            for suffix in range(0, 100, 11):
                add(hundred * 100 + suffix, f"repeated_{suffix:02d}", 1)
    return [values[key] for key in sorted(values)]


def _field(row, key):
    return row[key] if isinstance(row, Mapping) else getattr(row, key)


def _bar_timestamps(raw, step):
    if isinstance(raw, (tuple, list)):
        if len(raw) != 8:
            raise ValueError("btc_pullback_compact_candle_invalid")
        opening, closing = raw[0], raw[7]
    else:
        try:
            opening = _field(raw, "open_time_ms")
        except (KeyError, AttributeError):
            opening = _field(raw, "time_ms")
        closing = _field(raw, "close_time_ms")
    opening = _integer(opening, "btc_pullback_candle_open_invalid")
    closing = _integer(closing, "btc_pullback_candle_close_invalid")
    if opening % step or closing != opening + step - 1:
        raise ValueError("btc_pullback_candle_alignment_invalid")
    return opening, closing


def _closed_bars(values, step, cutoff, *, earliest=0):
    if not isinstance(values, (tuple, list)):
        raise ValueError("btc_pullback_candles_unavailable")
    bars = []
    for raw in values:
        opening, closing = _bar_timestamps(raw, step)
        if closing > cutoff or opening < earliest:
            continue  # Never parse forming or post-origin OHLC.
        fields = raw[1:5] if isinstance(raw, (tuple, list)) else [_field(raw, key) for key in ("open", "high", "low", "close")]
        op, high, low, close = [_number(value, "btc_pullback_candle_price_invalid", positive=True) for value in fields]
        if low > min(op, close) or high < max(op, close) or high < low:
            raise ValueError("btc_pullback_candle_range_invalid")
        if bars and opening - bars[-1]["open_ms"] != step:
            raise ValueError("btc_pullback_hourly_gap" if step == HOUR else "btc_pullback_5m_gap")
        bars.append({"open_ms": opening, "close_ms": closing, "open": op, "high": high,
                     "low": low, "close": close})
    return bars


def _base(source, now_ms, policy):
    origin = source["origin_signal_time_ms"] if source else None
    return {"version": 1, "mode": MODE, "source": source, "source_key": source["source_key"] if source else None,
            "side": source["side"] if source else None, "signal_origin": origin,
            "expires_at_ms": origin + round(policy["max_initial_entry_age_seconds"] * 1000) if origin else None,
            "policy": policy, "created_at_ms": now_ms, "reference_price": None,
            "candidate_low": None, "candidate_high": None, "numeric_levels": [],
            "last_processed_open_ms": origin - FIVE_MINUTES if origin else None,
            "status": "waiting_retrace", "reasons": []}


def build_pending(target, hourly_candles, policy, now_ms):
    """Freeze original known hourly structures; missing evidence creates no fallback."""
    policy = validate_policy(policy)
    now = _integer(now_ms, "btc_pullback_now_invalid", minimum=1)
    try:
        source = _source(target)
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        pending = _base(None, now, policy)
        pending.update(status="invalid_source", reasons=[str(exc)])
        return pending
    pending = _base(source, now, policy)
    if source is None:
        return {**pending, "status": "cancelled", "reasons": ["btc_pullback_target_flat"]}
    origin = source["origin_signal_time_ms"]
    if origin > now:
        return {**pending, "status": "invalid_source", "reasons": ["btc_pullback_origin_future"]}
    if now >= pending["expires_at_ms"]:
        return {**pending, "status": "expired", "reasons": ["btc_pullback_origin_expired"]}
    try:
        anchor = _number(target.get("origin_entry_price"), "btc_pullback_anchor_invalid", positive=True)
        pending["reference_price"] = anchor
    except ValueError as exc:
        return {**pending, "status": "invalid_source", "reasons": [str(exc)]}
    try:
        bars = _closed_bars(hourly_candles, HOUR, min(origin - 1, now))
        if len(bars) < policy["lookback_hours"]:
            raise ValueError("btc_pullback_hourly_history_insufficient")
        if bars[-1]["close_ms"] != origin - 1:
            raise ValueError("btc_pullback_original_hourly_prefix_missing")
    except (ValueError, KeyError, AttributeError, TypeError, IndexError) as exc:
        return {**pending, "status": "data_unavailable", "reasons": [str(exc)]}
    window = bars[-policy["lookback_hours"]:]
    period = policy["atr_period"]
    atr = sum(max(bar["high"] - bar["low"], abs(bar["high"] - prior["close"]),
                  abs(bar["low"] - prior["close"]))
              for prior, bar in zip(bars[-period - 1:-1], bars[-period:])) / period
    if not math.isfinite(atr) or atr <= 0:
        return {**pending, "status": "data_unavailable", "reasons": ["btc_pullback_atr_invalid"]}
    ema = bars[0]["close"]
    for bar in bars[1:]:
        ema += (2 / 49) * (bar["close"] - ema)
    structures = [{"kind": "ema48", "price": ema, "available_at_ms": origin - 1}]
    wings = policy["pivot_wings"]
    side = source["side"]
    for index in range(wings, len(window) - wings):
        bar = window[index]
        neighbors = window[index - wings:index] + window[index + 1:index + wings + 1]
        key = "low" if side == "long" else "high"
        valid = all(bar[key] < other[key] for other in neighbors) if side == "long" else all(bar[key] > other[key] for other in neighbors)
        if valid:
            structures.append({"kind": "pivot_low" if side == "long" else "pivot_high",
                               "price": bar[key], "pivot_open_ms": bar["open_ms"],
                               "available_at_ms": window[index + wings]["close_ms"]})
    pending["atr"] = atr
    legal = []
    width = policy["zone_half_width_atr"] * atr
    chase = policy["max_chase_atr"] * atr
    improve = policy["min_improvement_atr"] * atr
    for structure in structures:
        price = structure["price"]
        distance = anchor - price if side == "long" else price - anchor
        if not 0 < distance <= policy["max_retrace_atr"] * atr:
            continue
        low, high = price - width, price + width
        entry_low, entry_high = (high, high + chase) if side == "long" else (low - chase, low)
        if low <= 0 or entry_low <= 0:
            continue
        if ((side == "long" and entry_high > anchor - improve)
                or (side == "short" and entry_low < anchor + improve)):
            continue
        numbers = generate_numeric_candidates(price, policy["number_confluence_atr"] * atr,
            round_steps=policy["round_steps"], repeated_last_two_digits=policy["repeated_last_two_digits"])
        legal.append({**structure, "distance_from_anchor": distance, "candidate_low": low,
                      "candidate_high": high, "entry_low": entry_low, "entry_high": entry_high,
                      "numeric_levels": numbers, "numeric_priority": max((item["priority"] for item in numbers), default=0)})
    if not legal:
        return {**pending, "status": "no_structure", "reasons": ["btc_pullback_no_legal_structure"]}
    closest = min(item["distance_from_anchor"] for item in legal)
    nearby = [item for item in legal if item["distance_from_anchor"] <= closest + policy["numeric_tiebreak_atr"] * atr]
    selected = min(nearby, key=lambda item: (-item["numeric_priority"], item["distance_from_anchor"], item["price"]))
    reason = "nearest_structure" if selected["distance_from_anchor"] == closest else "numeric_confluence_within_nearest_band"
    pending.update(candidate_low=selected["candidate_low"], candidate_high=selected["candidate_high"],
                   entry_low=selected["entry_low"], entry_high=selected["entry_high"],
                   numeric_levels=selected["numeric_levels"], numeric_priority=selected["numeric_priority"],
                   selected_structure={key: selected[key] for key in ("kind", "price", "available_at_ms")},
                   selection_reason=reason,
                   invalidation_price=selected["candidate_low"] - policy["invalidate_beyond_zone_atr"] * atr
                   if side == "long" else selected["candidate_high"] + policy["invalidate_beyond_zone_atr"] * atr)
    if "pivot_open_ms" in selected:
        pending["selected_structure"]["pivot_open_ms"] = selected["pivot_open_ms"]
    return pending


def _decision(pending, now, *, status=None, reasons=None, allowed=False, projected_fill=None):
    status = status or pending["status"]
    result = {"enabled": True, "allowed": allowed, "status": status,
              "reasons": list(pending.get("reasons", []) if reasons is None else reasons),
              "projected_fill_price": projected_fill, "checked_at_ms": now,
              "cancel_pending": pending["status"] in TERMINAL_STATUSES,
              "retryable": pending["status"] == "data_unavailable" or status == "data_unavailable"}
    for key in ("side", "reference_price", "candidate_low", "candidate_high", "numeric_levels",
                "signal_origin", "expires_at_ms", "source_key", "entry_low", "entry_high",
                "selected_structure", "numeric_priority", "selection_reason"):
        result[key] = pending.get(key)
    return result


def _validate_pending(pending, policy):
    if not isinstance(pending, Mapping) or pending.get("mode") != MODE or type(pending.get("version")) is not int or pending["version"] != 1:
        raise ValueError("btc_pullback_pending_invalid")
    if pending.get("status") not in ACTIVE_STATUSES | TERMINAL_STATUSES | {"data_unavailable"}:
        raise ValueError("btc_pullback_pending_status_invalid")
    if validate_policy(pending.get("policy")) != policy:
        raise ValueError("btc_pullback_policy_changed")
    if pending["status"] in TERMINAL_STATUSES:
        return
    source = pending.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("btc_pullback_pending_source_invalid")
    exposure = 1 if source.get("side") == "long" else -1 if source.get("side") == "short" else None
    expected = _source({**source, "target_leverage": exposure})
    if (source != expected or pending.get("source_key") != expected["source_key"]
            or pending.get("signal_origin") != expected["origin_signal_time_ms"]
            or pending.get("side") != expected["side"]):
        raise ValueError("btc_pullback_pending_source_invalid")
    expiry = expected["origin_signal_time_ms"] + round(policy["max_initial_entry_age_seconds"] * 1000)
    if type(pending.get("expires_at_ms")) is not int or pending["expires_at_ms"] != expiry:
        raise ValueError("btc_pullback_pending_expiry_invalid")
    if pending["status"] == "data_unavailable":
        return
    anchor = _number(pending.get("reference_price"), "btc_pullback_pending_anchor_invalid", positive=True)
    atr = _number(pending.get("atr"), "btc_pullback_pending_atr_invalid", positive=True)
    low = _number(pending.get("candidate_low"), "btc_pullback_pending_zone_invalid", positive=True)
    high = _number(pending.get("candidate_high"), "btc_pullback_pending_zone_invalid", positive=True)
    e_low = _number(pending.get("entry_low"), "btc_pullback_pending_corridor_invalid", positive=True)
    e_high = _number(pending.get("entry_high"), "btc_pullback_pending_corridor_invalid", positive=True)
    invalid = _number(pending.get("invalidation_price"), "btc_pullback_pending_invalidation_invalid")
    selected = pending.get("selected_structure")
    if not isinstance(selected, Mapping) or selected.get("kind") not in {"ema48", "pivot_low", "pivot_high"}:
        raise ValueError("btc_pullback_pending_structure_invalid")
    level = _number(selected.get("price"), "btc_pullback_pending_structure_invalid", positive=True)
    available = _integer(selected.get("available_at_ms"), "btc_pullback_pending_structure_time_invalid")
    if available >= pending["signal_origin"]:
        raise ValueError("btc_pullback_pending_structure_from_future")
    width, chase = atr * policy["zone_half_width_atr"], atr * policy["max_chase_atr"]
    long = pending["side"] == "long"
    expected_prices = (level - width, level + width, high if long else low - chase,
                       high + chase if long else low,
                       low - policy["invalidate_beyond_zone_atr"] * atr if long else high + policy["invalidate_beyond_zone_atr"] * atr)
    for actual, expected_price in zip((low, high, e_low, e_high, invalid), expected_prices):
        if not math.isclose(actual, expected_price, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("btc_pullback_pending_frozen_level_mismatch")
    if ((long and e_high > anchor - policy["min_improvement_atr"] * atr)
            or (not long and e_low < anchor + policy["min_improvement_atr"] * atr)):
        raise ValueError("btc_pullback_pending_improvement_invalid")
    distance = anchor - level if long else level - anchor
    if not 0 < distance <= policy["max_retrace_atr"] * atr:
        raise ValueError("btc_pullback_pending_retrace_invalid")
    numbers = generate_numeric_candidates(level, policy["number_confluence_atr"] * atr,
        round_steps=policy["round_steps"], repeated_last_two_digits=policy["repeated_last_two_digits"])
    if (pending.get("numeric_levels") != numbers
            or type(pending.get("numeric_priority")) is not int
            or pending["numeric_priority"] != max((item["priority"] for item in numbers), default=0)):
        raise ValueError("btc_pullback_pending_numeric_levels_invalid")
    processed = _integer(pending.get("last_processed_open_ms"), "btc_pullback_pending_bar_time_invalid")
    if processed % FIVE_MINUTES or processed < pending["signal_origin"] - FIVE_MINUTES:
        raise ValueError("btc_pullback_pending_bar_time_invalid")
    if pending["status"] != "waiting_retrace":
        touched = _integer(pending.get("touched_at_ms"), "btc_pullback_pending_touch_invalid")
        if not pending["signal_origin"] <= touched <= processed + FIVE_MINUTES - 1:
            raise ValueError("btc_pullback_pending_touch_invalid")
    if pending["status"] == "ready":
        confirmed = _integer(pending.get("confirmed_at_ms"), "btc_pullback_pending_confirmation_invalid")
        if not pending["touched_at_ms"] <= confirmed <= processed + FIVE_MINUTES - 1:
            raise ValueError("btc_pullback_pending_confirmation_invalid")


def advance_pending(pending, target, closed5m, mark_price, now_ms, slippage_bps, policy):
    """Return updated evidence and admission; terminal origins cannot resurrect.

    The caller persists the returned pending record and separately checks live
    quote freshness, factor admission, quantity limits and execution risk.
    """
    policy = validate_policy(policy)
    now = _integer(now_ms, "btc_pullback_now_invalid", minimum=1)
    slip = _number(slippage_bps, "btc_pullback_slippage_invalid", nonnegative=True)
    if slip >= 10_000:
        raise ValueError("btc_pullback_slippage_invalid")
    try:
        _validate_pending(pending, policy)
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        changed = dict(pending) if isinstance(pending, Mapping) else _base(None, now, policy)
        changed.update(status="invalid_source", reasons=[str(exc)])
        return changed, _decision(changed, now)
    updated = dict(pending)
    if pending["status"] in TERMINAL_STATUSES:
        return updated, _decision(updated, now)
    try:
        source = _source(target)
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        updated.update(status="invalid_source", reasons=[str(exc)])
        return updated, _decision(updated, now)
    if source is None or source["source_key"] != pending["source_key"]:
        updated.update(status="cancelled", reasons=["btc_pullback_target_flat" if source is None else "btc_pullback_source_changed"])
        return updated, _decision(updated, now)
    if pending["signal_origin"] > now:
        updated.update(status="invalid_source", reasons=["btc_pullback_origin_future"])
        return updated, _decision(updated, now)
    if now >= pending["expires_at_ms"]:
        updated.update(status="expired", reasons=["btc_pullback_origin_expired"])
        return updated, _decision(updated, now)
    if pending["status"] == "data_unavailable":
        return updated, _decision(updated, now)
    if pending["last_processed_open_ms"] + FIVE_MINUTES - 1 > now:
        updated.update(status="invalid_source", reasons=["btc_pullback_pending_future_bar"])
        return updated, _decision(updated, now)
    earliest = pending["last_processed_open_ms"] + FIVE_MINUTES
    latest_close = ((now + 1) // FIVE_MINUTES) * FIVE_MINUTES - 1
    try:
        bars = _closed_bars(closed5m, FIVE_MINUTES, now, earliest=earliest)
        if latest_close >= earliest:
            if not bars or bars[0]["open_ms"] != earliest or bars[-1]["close_ms"] != latest_close:
                raise ValueError("btc_pullback_5m_gap")
        elif bars:
            raise ValueError("btc_pullback_5m_from_future")
    except (ValueError, KeyError, AttributeError, TypeError, IndexError) as exc:
        return updated, _decision(updated, now, status="data_unavailable", reasons=[str(exc)])
    long = pending["side"] == "long"
    for bar in bars:
        updated["last_processed_open_ms"] = bar["open_ms"]
        if ((long and bar["close"] < pending["invalidation_price"])
                or (not long and bar["close"] > pending["invalidation_price"])):
            updated.update(status="structure_invalidated", reasons=["btc_pullback_structure_invalidated"])
            return updated, _decision(updated, now)
        touch = bar["low"] <= pending["candidate_high"] and bar["high"] >= pending["candidate_low"]
        if updated["status"] == "waiting_retrace" and touch:
            updated.update(status="waiting_confirmation", touched_at_ms=bar["close_ms"])
        rejection = ((long and bar["close"] > bar["open"] and bar["close"] > pending["candidate_high"])
                     or (not long and bar["close"] < bar["open"] and bar["close"] < pending["candidate_low"]))
        if updated["status"] == "waiting_confirmation" and rejection:
            updated.update(status="ready", confirmed_at_ms=bar["close_ms"])
    if updated["status"] != "ready":
        reason = "btc_pullback_waiting_retrace" if updated["status"] == "waiting_retrace" else "btc_pullback_waiting_confirmation"
        return updated, _decision(updated, now, reasons=[reason])
    try:
        mark = _number(mark_price, "btc_pullback_mark_unavailable", positive=True)
        projected = mark * (1 + (1 if long else -1) * slip / 10_000)
        if not math.isfinite(projected) or projected <= 0:
            raise ValueError("btc_pullback_projected_fill_invalid")
    except ValueError as exc:
        return updated, _decision(updated, now, status="price_unavailable", reasons=[str(exc)])
    if not pending["entry_low"] <= projected <= pending["entry_high"]:
        reason = "btc_pullback_chase_exceeded" if ((long and projected > pending["entry_high"]) or (not long and projected < pending["entry_low"])) else "btc_pullback_rejection_not_held"
        return updated, _decision(updated, now, reasons=[reason], projected_fill=projected)
    return updated, _decision(updated, now, reasons=[], allowed=True, projected_fill=projected)
