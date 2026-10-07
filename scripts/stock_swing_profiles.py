"""Completed-bar signal dispatch for the per-stock research profiles.

The profiles were chosen by comparing a finite set of development candidates.
This module implements those causal rules; it is not independent validation or
an execution client. Existing breakout behavior remains the default.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import stock_swing_signals as baseline
import stock_entry_quality as quality
import stock_mechanism_signals as mechanisms
from stock_swing_signals import Candle, Signal


def signal_activity(candles: Sequence[Candle], index: int, config: Mapping[str, Any]) -> dict:
    """Optional closed-prefix activity gate; zero-volume bars are never interpolated."""
    lookback = config.get('signal_activity_lookback_bars', 0)
    minimum = config.get('signal_min_active_fraction', 0.5)
    if isinstance(lookback, bool) or not isinstance(lookback, int) or not 0 <= lookback <= 120:
        raise ValueError('Signal activity lookback must be an integer in [0, 120]')
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 < minimum <= 1:
        raise ValueError('Signal active fraction must be in (0, 1]')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candles):
        raise IndexError('Signal index is outside the candle history')
    if not lookback:
        return {'enabled': False, 'allowed': True}
    prefix = candles[max(0, index + 1 - lookback):index + 1]
    active = sum(bar.volume > 0 for bar in prefix)
    return {'enabled': True, 'allowed': len(prefix) == lookback and active / lookback >= minimum,
            'lookback_bars': lookback, 'observed_bars': len(prefix), 'active_bars': active,
            'active_fraction': active / lookback, 'minimum_active_fraction': minimum,
            'last_bar_time_ms': candles[index].time_ms}


def signal_at(candles, indicators, index, config=None):
    config = config or {}
    quality.validate(config)
    signal = raw_signal_at(candles, indicators, index, config)
    if signal and not quality.evaluate(candles, indicators, index, config, signal.direction)['allowed']:
        return None
    return signal


def raw_signal_at(
    candles: Sequence[Candle], indicators: Mapping[str, Sequence[float | None]],
    index: int, config: Mapping[str, Any] | None = None,
) -> Signal | None:
    """Dispatch a breakout or a fresh fast/mid EMA transition at ``index``.

    A transition needs a strict current EMA crossing, allowing equality on the
    preceding bar. The optional slow gate checks the signal close, not an EMA
    ordering or slope. All inputs used belong to the completed-bar prefix.
    """
    config = config or {}
    if not signal_activity(candles, index, config)['allowed']:
        return None
    family = config.get("signal_family", "breakout")
    if family in mechanisms.FAMILIES:
        return mechanisms.signal_at(candles, indicators, index, config)
    if family == "breakout":
        return baseline.signal_at(candles, indicators, index, config)
    if family in ("trend_pullback", "range_reversion"):
        return regime_signal_at(candles, indicators, index, config)
    if family != "ema_transition":
        raise ValueError(f"Unknown signal family: {family!r}")

    cfg = baseline._settings(config)
    require_slow = config.get("require_price_slow", True)
    if not isinstance(require_slow, bool):
        raise ValueError("require_price_slow must be a boolean")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candles):
        raise IndexError("Signal index is outside the candle history")
    minimum = max(cfg["warmup_bars"], cfg["ema_slow"], cfg["atr_period"])
    if index < 1 or index + 1 < minimum:
        return None
    try:
        values = [indicators[key][index] for key in ("ema_fast", "ema_mid", "ema_slow", "atr")]
        values.extend(indicators[key][index - 1] for key in ("ema_fast", "ema_mid"))
    except (KeyError, IndexError) as exc:
        raise ValueError("Indicator arrays do not cover the requested signal") from exc
    if any(value is None for value in values):
        return None
    fast, mid, slow, atr, previous_fast, previous_mid = [
        baseline._finite(value, "indicator") for value in values
    ]
    bar = candles[index]
    if atr <= 0 or atr / bar.close > cfg["max_atr_fraction"]:
        return None
    if fast > mid and previous_fast <= previous_mid:
        direction = 1
    elif fast < mid and previous_fast >= previous_mid:
        direction = -1
    else:
        return None
    if require_slow and direction * (bar.close - slow) <= 0:
        return None
    return Signal(direction, atr, bar.close, abs(fast - mid) / atr, bar.time_ms)


def regime_signal_at(candles, indicators, index, config):
    """Symmetric completed-bar reclaim signals, using no future prices."""
    cfg = baseline._settings(config)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candles):
        raise IndexError("Signal index is outside the candle history")
    if index < 1 or index + 1 < max(cfg['warmup_bars'], cfg['ema_slow'], cfg['atr_period'], 21):
        return None
    values = [indicators[k][index] for k in ('ema_fast', 'ema_mid', 'ema_slow', 'atr')]
    values += [indicators['ema_fast'][index - 1], indicators['ema_mid'][index - cfg['ema_slope_bars']],
               indicators['ema_slow'][index - cfg['ema_slope_bars']]]
    if any(v is None for v in values):
        return None
    fast, mid, slow, atr, previous_fast, prior_mid, prior_slow = [baseline._finite(v, 'indicator') for v in values]
    bar, previous = candles[index], candles[index - 1]
    if atr <= 0 or atr / bar.close > cfg['max_atr_fraction']:
        return None
    if config['signal_family'] == 'trend_pullback':
        if fast > mid > slow and mid > prior_mid:
            side = 1
        elif fast < mid < slow and mid < prior_mid:
            side = -1
        else:
            return None
        if (side * (previous.close - previous_fast) > 0 or side * (bar.close - fast) <= 0
                or abs(fast / mid - 1) < .001 or side * (bar.close - fast) > atr):
            return None
        return Signal(side, atr, bar.close, abs(fast - mid) / atr, bar.time_ms)
    # A flat slow EMA and a narrow fast/mid spread gate mean-reversion entries.
    if abs(slow / prior_slow - 1) > .001 or abs(fast / mid - 1) > .003:
        return None
    def band(end):
        values = [b.close for b in candles[end - 19:end + 1]]
        mean = sum(values) / 20
        deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / 20)
        return mean, mean - 2 * deviation, mean + 2 * deviation
    mean, lower, upper = band(index)
    _, previous_lower, previous_upper = band(index - 1)
    if previous.close < previous_lower and lower < bar.close < mean:
        side = 1
    elif previous.close > previous_upper and mean < bar.close < upper:
        side = -1
    else:
        return None
    return Signal(side, atr, bar.close, abs(bar.close - mean) / atr, bar.time_ms)
