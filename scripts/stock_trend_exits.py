"""Forward-only, two-closed-bar trend invalidation for a paper exit experiment."""
from __future__ import annotations

import stock_swing_signals as signals

MODE = 'opposite_ema_two_closed_bars'
POLICY = {'mode': MODE, 'confirmation_bars': 2, 'require_price_slow': True}


def validate(policy):
    if policy != POLICY or type(policy.get('confirmation_bars')) is not int or policy.get('require_price_slow') is not True:
        raise ValueError('Trend exit experiment requires the declared two-bar EMA policy')


def evaluate(position, candles, config, timestamp, interval_ms, activated_at_ms):
    """Use only fresh completed bars; indicator warmup does not count as confirmation."""
    validate(config['trend_exit_policy'])
    if not position:
        return {'status': 'flat', 'triggered': False}
    if not candles:
        return {'status': 'unavailable', 'triggered': False, 'reason': 'missing_closed_candles'}
    if any(b.time_ms + interval_ms > timestamp for b in candles):
        return {'status': 'unavailable', 'triggered': False, 'reason': 'unfinished_candle'}
    if any(b.time_ms - a.time_ms != interval_ms for a, b in zip(candles, candles[1:])):
        return {'status': 'unavailable', 'triggered': False, 'reason': 'nonconsecutive_candles'}
    if not 0 <= timestamp - (candles[-1].time_ms + interval_ms) <= interval_ms + 5000:
        return {'status': 'unavailable', 'triggered': False, 'reason': 'stale_closed_candles'}
    indicators = signals.compute_indicators(candles, config)
    side = position['direction']
    if side not in (-1, 1):
        raise ValueError('Invalid position direction')
    confirmations = 0
    after = max(activated_at_ms, position['entry_time'])
    for index in range(len(candles) - 1, max(-1, len(candles) - 3), -1):
        bar = candles[index]
        if bar.time_ms + interval_ms <= after:
            break
        fast, mid, slow = [indicators[k][index] for k in ('ema_fast', 'ema_mid', 'ema_slow')]
        if any(value is None for value in (fast, mid, slow)):
            break
        if side * (fast - mid) < 0 and side * (bar.close - slow) < 0:
            confirmations += 1
        else:
            break
    latest = {k: indicators[k][-1] for k in ('ema_fast', 'ema_mid', 'ema_slow')}
    return {'status': 'triggered' if confirmations == 2 else 'confirming' if confirmations else 'holding',
            'triggered': confirmations == 2, 'confirmation_bars': confirmations,
            'required_bars': 2, 'closed_at_ms': candles[-1].time_ms + interval_ms,
            'close': candles[-1].close, **latest}
