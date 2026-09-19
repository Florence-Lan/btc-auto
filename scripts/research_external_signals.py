"""Causal research adaptations of public trading ideas; not live strategies."""
from __future__ import annotations

import simulate_range_swing as sim


def ema_signals(candles, cfg, start_ms, atr_multiple=None):
    closes = [c.close for c in candles]
    fast = sim.ema(closes, cfg.timeseries_fast_ema)
    slow = sim.ema(closes, cfg.timeseries_slow_ema)
    atr = sim.atr(candles, 14)
    confirmed = []
    for i, c in enumerate(candles):
        threshold = c.close * cfg.timeseries_min_ema_spread_pct if atr_multiple is None else (atr[i] or 0) * atr_multiple
        if fast[i] is None or slow[i] is None or (atr_multiple is not None and atr[i] is None):
            confirmed.append(None)
            continue
        difference = fast[i] - slow[i]
        confirmed.append("long" if difference >= threshold else "short" if difference <= -threshold else None)
    # Match the frozen engine's edge-triggered hysteresis, including window start.
    return [side if i > 0 and candles[i].open_time_ms >= start_ms and side != confirmed[i-1] else None for i, side in enumerate(confirmed)]


def donchian_signals(candles, start_ms, filter_name=None):
    """1h long-only 20-bar breakout / 10-bar low exit, next-open execution."""
    atr = sim.atr(candles, 14)
    signals = [None] * len(candles)
    holding = False
    for i in range(34, len(candles)):
        c = candles[i]
        if c.open_time_ms < start_ms:
            continue
        upper = max(x.high for x in candles[i-20:i])
        lower = min(x.low for x in candles[i-10:i])
        permitted = True
        if filter_name == "volume":
            permitted = c.volume > sum(x.volume for x in candles[i-20:i]) / 20
        elif filter_name == "atr":
            permitted = atr[i] > sum(atr[i-19:i+1]) / 20
        if holding and c.close < lower:
            holding = False
            signals[i] = "flat"
        elif not holding and c.close > upper and permitted:
            holding = True
            signals[i] = "long"
    return signals


def vwap_rsi_signals(candles, start_ms):
    """UTC session VWAP; RSI14 crosses 60/40; close-based exit, no intrabar stop.

    Exit at VWAP invalidation, RSI50, a 2-ATR close-based stop, or 36 bars.
    All decisions execute next open. Rules complete the external sketch and
    are our adaptation, not an exact reproduction of the author's strategy.
    """
    rsi = sim.rsi([c.close for c in candles], 14)
    atr = sim.atr(candles, 14)
    signals = [None] * len(candles)
    day = None
    volume = value = 0.0
    holding = None
    signal_close = stop_distance = 0.0
    entered = 0
    for i, c in enumerate(candles):
        current_day = c.open_time_ms // sim.MS_PER_DAY
        if current_day != day:
            day, volume, value = current_day, 0.0, 0.0
        volume += c.volume
        value += c.quote_volume
        vwap = value / volume if volume else c.close
        if c.open_time_ms < start_ms or i == 0 or rsi[i] is None or rsi[i-1] is None or atr[i] is None:
            continue
        if holding:
            side = 1 if holding == "long" else -1
            invalid = (c.close-vwap)*side < 0 or (rsi[i]-50)*side < 0
            stopped = (c.close-signal_close)*side <= -stop_distance
            if invalid or stopped or i-entered >= 36:
                signals[i], holding = "flat", None
        elif rsi[i] > 60 >= rsi[i-1] and c.close > vwap:
            signals[i] = holding = "long"
            signal_close, stop_distance, entered = c.close, 2*atr[i], i
        elif rsi[i] < 40 <= rsi[i-1] and c.close < vwap:
            signals[i] = holding = "short"
            signal_close, stop_distance, entered = c.close, 2*atr[i], i
    return signals
