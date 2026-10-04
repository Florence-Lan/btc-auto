"""Completed-bar signal dispatch for the per-stock research profiles.

The profiles were chosen by comparing a finite set of development candidates.
This module implements those causal rules; it is not independent validation or
an execution client. Existing breakout behavior remains the default.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import stock_swing_signals as baseline
from stock_swing_signals import Candle, Signal


def signal_at(
    candles: Sequence[Candle], indicators: Mapping[str, Sequence[float | None]],
    index: int, config: Mapping[str, Any] | None = None,
) -> Signal | None:
    """Dispatch a breakout or a fresh fast/mid EMA transition at ``index``.

    A transition needs a strict current EMA crossing, allowing equality on the
    preceding bar. The optional slow gate checks the signal close, not an EMA
    ordering or slope. All inputs used belong to the completed-bar prefix.
    """
    config = config or {}
    family = config.get("signal_family", "breakout")
    if family == "breakout":
        return baseline.signal_at(candles, indicators, index, config)
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
