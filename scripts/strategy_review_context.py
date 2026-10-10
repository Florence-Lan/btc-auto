"""Read-only, closed-bar diagnostics for reviewing the frozen trend target."""
from __future__ import annotations

import math

import simulate_range_swing as sim


def _target_state(target, decision_asof_ms):
    if not isinstance(decision_asof_ms, int) or isinstance(decision_asof_ms, bool):
        return None, "invalid_decision_cutoff"
    if target is None:
        return None, None
    if not isinstance(target, dict) or not isinstance(target.get("components", []), list):
        return None, "invalid_target_components"
    sides = set()
    issue = None
    component_sum = 0.0
    for component in target.get("components", []):
        if not isinstance(component, dict):
            return None, "invalid_target_components"
        try:
            quantity = float(component.get("signed_qty", 0))
        except (TypeError, ValueError):
            return None, "invalid_component_quantity"
        if not math.isfinite(quantity):
            return None, "invalid_component_quantity"
        component_sum += quantity
        if abs(quantity) <= 1e-12:
            continue
        side = "long" if quantity > 0 else "short"
        if "side" in component and component["side"] != side:
            issue = "component_declared_side_mismatch"
        if not str(component.get("strategy", "")).startswith("timeseries_trend"):
            continue
        sides.add(side)
        if "entry_time_utc" in component:
            try:
                entry_ms = sim._utc_ms(component["entry_time_utc"])
                if entry_ms < 0 or entry_ms > decision_asof_ms:
                    issue = "component_entry_not_available"
            except (AttributeError, TypeError, ValueError, OverflowError):
                issue = "invalid_component_entry_time"
    if len(sides) != 1:
        return None, "conflicting_trend_component_sides" if sides else issue
    side = next(iter(sides))
    for key in ("origin_signal_time_ms", "available_time_ms"):
        value = target.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                  or value < 0 or value > decision_asof_ms):
            issue = "target_timestamp_not_available"
    if "signed_qty" in target:
        try:
            aggregate = float(target["signed_qty"])
            # Tactical sleeves may legitimately offset or outweigh the trend.
            # Validate actual inventory arithmetic rather than net direction.
            if (not math.isfinite(aggregate) or not math.isfinite(component_sum)
                    or not math.isclose(aggregate, component_sum,
                                        rel_tol=1e-9, abs_tol=1e-12)):
                issue = "aggregate_target_quantity_mismatch"
        except (TypeError, ValueError):
            issue = "invalid_aggregate_target_quantity"
    return side, issue


def timeseries_trend_snapshot(candles, cfg, decision_asof_ms, target=None):
    """Describe current evidence without changing signals, targets or strategy.

    The engine's entry threshold is a ratio despite its ``_pct`` config name.
    Values exposed here are percentage points. A held target stays active in
    the neutral threshold zone; a raw EMA cross alone is not the exit rule.
    Never consume an appended opening-only candle or unfinished hour's OHLC.
    """
    current_target_side, target_issue = _target_state(target, decision_asof_ms)
    result = {
        "schema_version": 1,
        "strategy": "timeseries_trend",
        "timeframe": cfg.timeseries_timeframe,
        "decision_asof_ms": decision_asof_ms,
        "status": "unavailable",
        "unavailable_reason": None,
        "data_available": False,
        "history_sufficient": False,
        "closed_bar_count": 0,
        "required_history_bars": None,
        "fast_ema_period_bars": cfg.timeseries_fast_ema,
        "slow_ema_period_bars": cfg.timeseries_slow_ema,
        "volatility_lookback_bars": cfg.timeseries_vol_lookback_bars,
        "ema_fast": None,
        "ema_slow": None,
        "ema_spread_pct": None,
        "entry_min_ema_spread_pct": None,
        "spread_unit": "percent",
        "confirmed_side": None,
        "last_confirmed_side": None,
        "last_confirmed_bar": None,
        "ema_ordering": None,
        "current_target_side": current_target_side,
        "target_consistency_issue": target_issue,
        "trend_side_under_exit_rule": None,
        "target_side_valid_under_exit_rule": None,
        "exit_rule": "opposite_confirmed_threshold_transition_at_next_known_open",
        "last_closed_bar": None,
    }
    try:
        if (not isinstance(decision_asof_ms, int) or isinstance(decision_asof_ms, bool)
                or decision_asof_ms < 0):
            raise ValueError("invalid_decision_cutoff")
        periods = (cfg.timeseries_fast_ema, cfg.timeseries_slow_ema,
                   cfg.timeseries_vol_lookback_bars)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in periods):
            raise ValueError("invalid_indicator_periods")
        if cfg.timeseries_fast_ema >= cfg.timeseries_slow_ema:
            raise ValueError("invalid_indicator_periods")
        threshold = float(cfg.timeseries_min_ema_spread_pct)
        if not math.isfinite(threshold * 100) or threshold < 0:
            raise ValueError("invalid_entry_threshold")
        step = sim.interval_to_ms(cfg.timeseries_timeframe)
        if step <= 0:
            raise ValueError("invalid_timeframe")
        result["required_history_bars"] = max(cfg.timeseries_slow_ema,
                                               cfg.timeseries_vol_lookback_bars) + 1
        result["entry_min_ema_spread_pct"] = threshold * 100
        closed = []
        for bar in candles:
            if not isinstance(bar.close_time_ms, int) or isinstance(bar.close_time_ms, bool):
                raise ValueError("invalid_bar_timestamp")
            if bar.close_time_ms <= decision_asof_ms:
                closed.append(bar)
        result["closed_bar_count"] = len(closed)
        if not closed:
            raise ValueError("no_closed_bars")
        for index, bar in enumerate(closed):
            if (not isinstance(bar.open_time_ms, int) or isinstance(bar.open_time_ms, bool)
                    or bar.open_time_ms < 0 or bar.open_time_ms % step
                    or bar.close_time_ms != bar.open_time_ms + step - 1):
                raise ValueError("invalid_closed_bar_interval")
            if index and bar.open_time_ms - closed[index - 1].open_time_ms != step:
                raise ValueError("closed_bar_history_gap_or_out_of_order")
            if not math.isfinite(bar.close) or bar.close <= 0:
                raise ValueError("invalid_closed_bar_price")
        latest = closed[-1]
        result["last_closed_bar"] = {
            "open_time_ms": latest.open_time_ms,
            "close_time_ms": latest.close_time_ms,
            # Match the frozen engine's inclusive exchange-close convention.
            "available_at_ms": latest.close_time_ms,
            "close_price": latest.close,
        }
        result["data_available"] = True
        if len(closed) < result["required_history_bars"]:
            raise ValueError("insufficient_closed_history")
        result["history_sufficient"] = True
        expected_close_ms = (decision_asof_ms + 1) // step * step - 1
        if latest.close_time_ms != expected_close_ms:
            raise ValueError("stale_closed_history")
        closes = [bar.close for bar in closed]
        fast_values = sim.ema(closes, cfg.timeseries_fast_ema)
        slow_values = sim.ema(closes, cfg.timeseries_slow_ema)
        fast, slow = fast_values[-1], slow_values[-1]
        if fast is None or slow is None or not math.isfinite(fast) or not math.isfinite(slow):
            raise ValueError("invalid_ema_values")
        spread = (fast - slow) / latest.close
        if not math.isfinite(spread * 100):
            raise ValueError("invalid_ema_spread")
        confirmed = "long" if spread >= threshold else "short" if spread <= -threshold else None
        last_confirmed_side = None
        last_confirmed_bar = None
        for bar, fast_value, slow_value in zip(closed, fast_values, slow_values):
            if fast_value is None or slow_value is None:
                continue
            historical_spread = (fast_value - slow_value) / bar.close
            if not math.isfinite(historical_spread):
                raise ValueError("invalid_ema_spread")
            historical_side = ("long" if historical_spread >= threshold else
                               "short" if historical_spread <= -threshold else None)
            if historical_side is not None:
                last_confirmed_side = historical_side
                last_confirmed_bar = {
                    "open_time_ms": bar.open_time_ms,
                    "close_time_ms": bar.close_time_ms,
                    "available_at_ms": bar.close_time_ms,
                }
        current = result["current_target_side"]
        valid = (current == last_confirmed_side and target_issue is None
                 if current is not None else False if target_issue else None)
        result.update({
            "status": "available",
            "ema_fast": fast,
            "ema_slow": slow,
            "ema_spread_pct": spread * 100,
            "confirmed_side": confirmed,
            "last_confirmed_side": last_confirmed_side,
            "last_confirmed_bar": last_confirmed_bar,
            "ema_ordering": "fast_above_slow" if fast > slow else "fast_below_slow" if fast < slow else "equal",
            "trend_side_under_exit_rule": last_confirmed_side if current else None,
            "target_side_valid_under_exit_rule": valid,
        })
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        # Missing diagnostics must be explicit, never a manufactured trend or
        # a reason for the producer to abandon an otherwise valid report.
        result["unavailable_reason"] = str(exc)
    return result
