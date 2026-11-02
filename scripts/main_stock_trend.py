"""Pure, explicitly activated trend rules for the three main stock accounts.

This module performs no I/O and never changes a candle, account, or shared
experiment. Decision timestamps must come from the caller's verified clock.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import stock_entry_quality as quality
import stock_short_confirmation as short
import stock_swing_confirmation as confirmation
import stock_swing_signals as signals

HOUR = confirmation.HOUR
FOUR_HOURS = confirmation.FOUR_HOURS
POLICY_KEY = 'main_stock_trend_policy'
POLICY = {
    'mode': 'main_price_led_trend_v1',
    'confirmation_bars': 2,
    'require_close_beyond_fast_mid': True,
    'long_confirmation': 'strict_or_price_led_4h',
}


def enabled(config):
    return isinstance(config, Mapping) and POLICY_KEY in config


def validate(config):
    if not isinstance(config, Mapping):
        raise ValueError('main_stock_trend_config_invalid')
    if not enabled(config):
        return
    policy = config[POLICY_KEY]
    if (not isinstance(policy, Mapping) or dict(policy) != POLICY
            or any(type(policy.get(k)) is not type(v) for k, v in POLICY.items())):
        raise ValueError('main_stock_trend_policy_invalid')
    if not short.enabled(config):
        raise ValueError('main_stock_trend_existing_short_policy_required')
    short.validate(config)


def _stamp(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label)
    return value


def _validated_prefix(candles, index):
    if type(index) is not int or not 0 <= index < len(candles):
        raise ValueError('main_trend_index_invalid')
    prefix = list(candles[:index + 1])
    for bar in prefix:
        _stamp(bar.time_ms, 'main_trend_hour_timestamp_invalid')
        if bar.time_ms % HOUR:
            raise ValueError('main_trend_hour_alignment')
        # Revalidate even duck-typed fixtures or deserialized candle objects.
        signals.Candle(bar.time_ms, bar.open, bar.high, bar.low, bar.close, bar.volume)
    if any(b.time_ms - a.time_ms != HOUR for a, b in zip(prefix, prefix[1:])):
        raise ValueError('main_trend_hour_gap')
    return prefix


def _current_prefix(candles, timestamp):
    _stamp(timestamp, 'main_trend_decision_clock_required')
    prefix = _validated_prefix(candles, len(candles) - 1)
    if any(bar.time_ms + HOUR > timestamp for bar in prefix):
        raise ValueError('main_trend_unfinished_hour')
    if prefix[-1].time_ms + HOUR != timestamp // HOUR * HOUR:
        raise ValueError('main_trend_latest_closed_hour_missing')
    return prefix


def _ema_context(bars, config, interval):
    if len(bars) < 60:
        raise ValueError('main_trend_warmup')
    ind = signals.compute_indicators(bars, config)
    fast, previous = ind['ema_fast'][-1], ind['ema_fast'][-2]
    mid, slow, close = ind['ema_mid'][-1], ind['ema_slow'][-1], bars[-1].close
    if not all(type(v) in (int, float) and math.isfinite(v) and v > 0
               for v in (fast, previous, mid, slow, close)):
        raise ValueError('main_trend_indicators_unavailable')
    strict = 1 if fast > mid > slow and close > slow else -1 if fast < mid < slow and close < slow else 0
    price = 1 if close > fast and close > mid and fast > previous else -1 if close < fast and close < mid and fast < previous else 0
    spread = abs(fast - slow) / close
    price = price if spread >= .004 else 0
    return {'status': 'healthy', 'observed_bars': len(bars),
        'closed_at_ms': bars[-1].time_ms + interval, 'close': close,
        'ema_fast': fast, 'ema_fast_previous': previous, 'ema_mid': mid, 'ema_slow': slow,
        'ema_spread_fraction': spread, 'strict_direction': strict,
        'price_led_direction': price, 'direction': price or strict}


def evaluate_long_confirmation(candles, index, config, *, strict=None):
    """Strict OR symmetric price-led LONG context; does not create a signal."""
    validate(config)
    # Validate the supplied prefix before trusting a strict fast-path result.
    try:
        prefix = _validated_prefix(candles, index)
    except (ValueError, TypeError, AttributeError, IndexError, OverflowError):
        return {'enabled': enabled(config), 'allowed': False, 'direction': 1,
            'reasons': ['long_confirmation_data_unavailable'], 'price_led_allowed': False,
            'strict_allowed': False, 'admitted_by': None}
    strict = confirmation.evaluate(prefix, len(prefix) - 1, config, 1) if strict is None else strict
    result = {'enabled': enabled(config), 'mode': 'closed_4h_price_led_long_v1', 'direction': 1,
        'allowed': strict['allowed'], 'strict_allowed': strict['allowed'],
        'strict_reasons': list(strict['reasons']), 'price_led_allowed': False,
        'reasons': list(strict['reasons']), 'checks': {},
        'signal_close_ms': prefix[-1].time_ms + HOUR,
        'confirmation_close_ms': strict.get('confirmation_close_ms'),
        'ema_fast': strict.get('ema_fast'), 'ema_fast_previous': None,
        'ema_mid': strict.get('ema_mid'), 'ema_slow': strict.get('ema_slow'),
        'close': strict.get('close'), 'ema_spread_fraction': strict.get('ema_spread_fraction'),
        'admitted_by': 'strict' if strict['allowed'] else None}
    if not enabled(config) or strict['allowed']:
        return result
    waivable = {'swing_4h_ema_ordering', 'swing_4h_price_slow', 'swing_4h_spread_too_small'}
    if (not strict.get('enabled') or strict.get('observed_bars', 0) < 60
            or not set(strict['reasons']) <= waivable):
        return result
    try:
        bars = confirmation.completed_four_hour_bars(prefix, len(prefix) - 1)
        ctx = _ema_context(bars, confirmation.POLICY, FOUR_HOURS)
        checks = {'long_4h_close_above_fast': ctx['close'] > ctx['ema_fast'],
            'long_4h_close_above_mid': ctx['close'] > ctx['ema_mid'],
            'long_4h_fast_rising': ctx['ema_fast'] > ctx['ema_fast_previous'],
            'long_4h_spread': ctx['ema_spread_fraction'] >= .004}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError):
        return {**result, 'allowed': False, 'reasons': ['long_confirmation_data_unavailable']}
    allowed = all(checks.values())
    return {**result, **{k: ctx[k] for k in ('close', 'ema_fast', 'ema_fast_previous', 'ema_mid', 'ema_slow', 'ema_spread_fraction')},
        'allowed': allowed, 'price_led_allowed': allowed, 'checks': checks,
        'reasons': [] if allowed else [k for k, passed in checks.items() if not passed],
        'admitted_by': 'price_led_long' if allowed else None}


def signal_with_diagnostics(candles, indicators, index, config=None):
    """Retain original raw Donchian and quality rules and all short behavior."""
    config = {} if config is None else config
    validate(config)
    original = short.signal_with_diagnostics(candles, indicators, index, config)
    if not enabled(config):
        return original
    _, diagnostic = original
    raw = diagnostic.get('signal_before_confirmation')
    if not raw or raw['direction'] != 1 or diagnostic['swing_confirmation']['allowed']:
        return original
    outcome = evaluate_long_confirmation(candles, index, config, strict=diagnostic['swing_confirmation'])
    diagnostic = deepcopy(diagnostic)
    strict = diagnostic['swing_confirmation']
    diagnostic['long_confirmation'] = deepcopy(outcome)
    diagnostic['swing_confirmation'] = {**strict, 'allowed': outcome['allowed'],
        'reasons': list(outcome['reasons']), 'strict_allowed': strict['allowed'],
        'strict_reasons': list(strict['reasons']), 'strict_direction': strict.get('direction'),
        'direction': 1 if outcome['allowed'] else strict.get('direction'),
        'long_confirmation_mode': outcome.get('mode'), 'price_led_long_allowed': outcome['price_led_allowed']}
    if not outcome['allowed']:
        diagnostic['admission'].update(stage='confirmation_rejected', reasons=list(outcome['reasons']),
            confirmation_rejected_candidate=True)
        return None, diagnostic
    signal = signals.Signal(**raw)
    diagnostic['signal_after_confirmation'] = dict(raw)
    diagnostic['admission'].update(stage='admitted', reasons=[], confirmation_rejected_candidate=False)
    result = quality.evaluate(candles, indicators, index, config, 1)
    diagnostic['entry_quality'] = deepcopy(result)
    if not result['allowed']:
        diagnostic['admission'].update(stage='quality_rejected', reasons=list(result['reasons']))
        return None, diagnostic
    diagnostic['signal_after_quality'] = dict(raw)
    return signal, diagnostic


def signal_at(*args, **kwargs):
    return signal_with_diagnostics(*args, **kwargs)[0]


def trend_diagnostics(candles, config, timestamp):
    """Current completed-hour and completed-4h context, independent of a signal."""
    validate(config)
    result = {'enabled': enabled(config), 'checked_at_ms': timestamp,
        'status': 'unavailable', 'reasons': [], 'signal_close_ms': None,
        'confirmation_close_ms': None, 'hourly': None, 'four_hour': None}
    try:
        prefix = _current_prefix(candles, timestamp)
        result['signal_close_ms'] = prefix[-1].time_ms + HOUR
        result['hourly'] = _ema_context(prefix, config, HOUR)
        four = confirmation.completed_four_hour_bars(prefix, len(prefix) - 1)
        if not four or four[-1].time_ms + FOUR_HOURS != timestamp // FOUR_HOURS * FOUR_HOURS:
            raise ValueError('main_trend_latest_closed_four_hour_missing')
        result['confirmation_close_ms'] = four[-1].time_ms + FOUR_HOURS
        result['four_hour'] = _ema_context(four, confirmation.POLICY, FOUR_HOURS)
        result['status'] = 'healthy'
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
        result['reasons'] = [str(exc) if isinstance(exc, ValueError) else 'main_trend_data_unavailable']
    return result


def evaluate_position_exit(position, candles, config, timestamp, activated_at_ms):
    """Two whole post-activation hours invalidate the held direction.

    A bearish position requires fast > mid and close > both on two consecutive
    completed hours; bullish positions use the exact inverse. EMA60 is context,
    not an additional exit condition. Existing protective exits retain priority
    in the caller. No preactivation or partially preactivation hour counts.
    """
    validate(config)
    result = {'enabled': enabled(config), 'triggered': False,
        'status': 'disabled' if not enabled(config) else 'flat' if not position else 'unavailable',
        'reasons': [], 'confirmation_bars': 0, 'required_bars': 2,
        'checked_at_ms': timestamp, 'closed_at_ms': None, 'confirmations': []}
    if not enabled(config) or not position:
        return result
    try:
        prefix = _current_prefix(candles, timestamp)
        activation = _stamp(activated_at_ms, 'main_trend_activation_required')
        entry = _stamp(position['entry_time'], 'main_trend_entry_time_invalid')
        side = position['direction']
        if type(side) is not int or side not in (-1, 1):
            raise ValueError('main_trend_position_direction_invalid')
        if len(prefix) < 61:
            raise ValueError('main_trend_warmup')
        ind = signals.compute_indicators(prefix, config)
        after = max(activation, entry)
        for index in (len(prefix) - 1, len(prefix) - 2):
            bar = prefix[index]
            if bar.time_ms < after:
                break
            fast, mid, slow = (ind[k][index] for k in ('ema_fast', 'ema_mid', 'ema_slow'))
            if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (fast, mid, slow)):
                raise ValueError('main_trend_indicators_unavailable')
            checks = {'opposite_fast_mid': side * (fast - mid) < 0,
                'close_beyond_fast': side * (bar.close - fast) < 0,
                'close_beyond_mid': side * (bar.close - mid) < 0}
            row = {'open_ms': bar.time_ms, 'closed_at_ms': bar.time_ms + HOUR,
                'close': bar.close, 'ema_fast': fast, 'ema_mid': mid, 'ema_slow': slow,
                'checks': checks, 'confirmed': all(checks.values())}
            result['confirmations'].append(row)
            if not row['confirmed']:
                break
            result['confirmation_bars'] += 1
        result.update(closed_at_ms=prefix[-1].time_ms + HOUR, active_after_ms=activation,
            position_entry_ms=entry, position_direction=side)
        result['triggered'] = result['confirmation_bars'] == 2
        result['status'] = ('triggered' if result['triggered'] else
            'confirming' if result['confirmation_bars'] else 'holding')
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
        result.update(status='unavailable', triggered=False, confirmation_bars=0,
            reasons=[str(exc) if isinstance(exc, ValueError) else 'main_trend_data_unavailable'])
    return result
