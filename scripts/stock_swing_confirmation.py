"""Fixed, causal 4h trend admission built from completed hourly stock bars."""
from __future__ import annotations

import math

import stock_swing_signals as signals

HOUR = 3_600_000
FOUR_HOURS = 4 * HOUR
POLICY = {
    'mode': 'closed_4h_swing_v1', 'confirmation_timeframe': '4h',
    'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
    'minimum_ema_spread_fraction': .004, 'require_price_slow': True,
}


def validate(config):
    policy = config.get('swing_confirmation_policy')
    if policy is None:
        return
    if (policy != POLICY or config.get('signal_timeframe') != '1h'
            or any(type(policy.get(key)) is not int for key in (
                'ema_fast', 'ema_mid', 'ema_slow', 'warmup_bars'))
            or isinstance(policy.get('minimum_ema_spread_fraction'), bool)
            or policy.get('require_price_slow') is not True):
        raise ValueError('Swing confirmation requires the frozen 1h-to-4h policy')


def completed_four_hour_bars(candles, index):
    """UTC buckets; ignore incomplete edge buckets and reject every source gap."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candles):
        raise ValueError('Invalid swing confirmation index')
    prefix = candles[:index + 1]
    if any(bar.time_ms % HOUR for bar in prefix):
        raise ValueError('swing_hourly_alignment')
    if any(b.time_ms - a.time_ms != HOUR for a, b in zip(prefix, prefix[1:])):
        raise ValueError('swing_hourly_gap')
    groups = {}
    for bar in prefix:
        groups.setdefault(bar.time_ms // FOUR_HOURS * FOUR_HOURS, []).append(bar)
    result = []
    for start, group in groups.items():
        if [bar.time_ms for bar in group] != list(range(start, start + FOUR_HOURS, HOUR)):
            continue
        result.append(signals.Candle(start, group[0].open, max(bar.high for bar in group),
            min(bar.low for bar in group), group[-1].close, sum(bar.volume for bar in group)))
    return result


def evaluate(candles, index, config, direction=None):
    """A missing direction reports context; a signal must agree with that context."""
    validate(config)
    if config.get('swing_confirmation_policy') is None:
        return {'enabled': False, 'allowed': True, 'reasons': []}
    if direction is not None and (isinstance(direction, bool) or direction not in (-1, 1)):
        raise ValueError('Invalid swing confirmation direction')
    result = {'enabled': True, 'allowed': False, 'reasons': [],
              'confirmation_timeframe': '4h', 'required_bars': POLICY['warmup_bars']}
    try:
        bars = completed_four_hour_bars(candles, index)
    except ValueError as exc:
        result['reasons'] = [str(exc)]
        return result
    result.update(observed_bars=len(bars), signal_close_ms=candles[index].time_ms + HOUR,
                  confirmation_close_ms=bars[-1].time_ms + FOUR_HOURS if bars else None)
    if len(bars) < POLICY['warmup_bars']:
        result['reasons'] = ['swing_4h_warmup']
        return result
    indicators = signals.compute_indicators(bars, POLICY)
    fast, mid, slow = [indicators[key][-1] for key in ('ema_fast', 'ema_mid', 'ema_slow')]
    if any(value is None or not math.isfinite(value) for value in (fast, mid, slow)):
        result['reasons'] = ['swing_4h_indicators_unavailable']
        return result
    close = bars[-1].close
    trend = 1 if fast > mid > slow else -1 if fast < mid < slow else 0
    side = direction if direction is not None else trend
    spread = abs(fast - slow) / close
    reasons = []
    if not trend or (direction is not None and trend != direction):
        reasons.append('swing_4h_ema_ordering')
    if side and side * (close - slow) <= 0:
        reasons.append('swing_4h_price_slow')
    if spread < POLICY['minimum_ema_spread_fraction']:
        reasons.append('swing_4h_spread_too_small')
    result.update(allowed=not reasons, reasons=reasons, direction=trend,
                  ema_fast=fast, ema_mid=mid, ema_slow=slow, close=close,
                  ema_spread_fraction=spread,
                  minimum_ema_spread_fraction=POLICY['minimum_ema_spread_fraction'])
    return result
