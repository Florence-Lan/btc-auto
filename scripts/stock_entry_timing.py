"""Causal, opt-in breakout validity checks for new stock paper entries.

The frozen Signal remains unchanged. A pending entry carries a separate trigger
made from the ORIGINAL completed signal prefix; retries never roll its level.
This module has no exchange, order, ledger, or protective-exit side effects.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


HOUR = 3_600_000
MODE = 'breakout_still_valid_v1'
POLICY = {'mode': MODE, 'require_breakout_valid_at_fill': True}
_NO_FILL = object()


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(name)
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name) from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name)
    return result


def _time(value, name):
    if type(value) is not int or value < 0 or value % HOUR:
        raise ValueError(name)
    return value


def _get(value, key):
    return value[key] if isinstance(value, Mapping) else getattr(value, key)


def _signal(signal):
    try:
        direction = _get(signal, 'direction')
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError('entry_timing_signal_direction')
        return {'direction': direction,
                'time_ms': _time(_get(signal, 'time_ms'), 'entry_timing_signal_time'),
                'close': _number(_get(signal, 'close'), 'entry_timing_signal_close'),
                'atr': _number(_get(signal, 'atr'), 'entry_timing_signal_atr')}
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('entry_timing_signal_invalid') from exc


def enabled(config):
    return 'entry_timing_policy' in config


def validate(config):
    """Absent policy preserves historical behavior; v1 supports hourly Donchian20."""
    if not enabled(config):
        return
    policy = config['entry_timing_policy']
    if not isinstance(policy, Mapping) or dict(policy) != POLICY:
        raise ValueError('entry_timing_policy_invalid')
    if policy.get('require_breakout_valid_at_fill') is not True:
        raise ValueError('entry_timing_policy_invalid')
    if config.get('signal_family') != 'donchian20' or config.get('signal_timeframe') != '1h':
        raise ValueError('entry_timing_strategy_unsupported')
    if type(config.get('breakout_bars', 20)) is not int or config.get('breakout_bars', 20) != 20:
        raise ValueError('entry_timing_strategy_unsupported')
    # Never relax the existing 0.5 ATR chase or 1 ATR extension ceilings.
    gap = _number(config.get('entry_gap_atr', .5), 'entry_timing_gap_limit_invalid')
    extension = _number(config.get('max_breakout_extension_atr', 1.),
                        'entry_timing_extension_limit_invalid')
    if gap > .5 or extension > 1.:
        raise ValueError('entry_timing_price_limits_unsupported')


def build_trigger(closed_signal_prefix, signal, config, boundary_ms, *, observed_at_ms=None):
    """Freeze the exact prior-20 breakout level, excluding the signal candle.

    The prefix must end at the original signal candle. Passing later candles is
    rejected, rather than silently substituting a newer rolling channel. When an
    observation time is supplied, a still-forming signal candle is also rejected.
    Raises ValueError when a trustworthy trigger cannot be built.
    """
    validate(config)
    if not enabled(config):
        return None
    source = _signal(signal)
    boundary = _time(boundary_ms, 'entry_timing_boundary_invalid')
    if boundary != source['time_ms'] + HOUR:
        raise ValueError('entry_timing_boundary_mismatch')
    if observed_at_ms is not None:
        if (type(observed_at_ms) is not int or observed_at_ms < boundary):
            raise ValueError('entry_timing_unfinished_signal_bar')
    try:
        bars = list(closed_signal_prefix)
        if len(bars) < 21:
            raise ValueError('entry_timing_original_prefix_unavailable')
        times = [_time(_get(bar, 'time_ms'), 'entry_timing_hourly_alignment') for bar in bars]
        if times[-1] != source['time_ms']:
            raise ValueError('entry_timing_original_prefix_mismatch')
        if any(b - a != HOUR for a, b in zip(times, times[1:])):
            raise ValueError('entry_timing_hourly_gap')
        last_close = _number(_get(bars[-1], 'close'), 'entry_timing_candle_invalid')
        if last_close != source['close']:
            raise ValueError('entry_timing_signal_close_mismatch')
        prior = bars[-21:-1]
        levels = []
        for bar in prior:
            opening, high, low, close = [_number(_get(bar, key), 'entry_timing_candle_invalid')
                                         for key in ('open', 'high', 'low', 'close')]
            if low > min(opening, close) or high < max(opening, close) or low > high:
                raise ValueError('entry_timing_candle_invalid')
            levels.append(high if source['direction'] == 1 else low)
        level = max(levels) if source['direction'] == 1 else min(levels)
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('entry_timing_original_prefix_invalid') from exc
    extension = source['direction'] * (source['close'] - level)
    if not 0 < extension <= source['atr'] * _number(
            config.get('max_breakout_extension_atr', 1.), 'entry_timing_extension_limit_invalid'):
        raise ValueError('entry_timing_original_breakout_invalid')
    return {'version': 1, 'family': 'donchian20', 'direction': source['direction'],
            'breakout_level': level, 'breakout_bars': 20, 'signal_timeframe': '1h',
            'signal_time_ms': source['time_ms'], 'boundary_ms': boundary,
            'signal_close': source['close'], 'signal_atr': source['atr'],
            'prior_window_start_ms': times[-21], 'prior_window_end_ms': times[-2]}


def rebuild_trigger(closed_candles, pending, config, *, observed_at_ms=None):
    """Reconstruct an old pending trigger only from its original completed prefix.

    Later CLOSED candles may exist in the history, but are excluded before level
    construction. A missing original candle or preceding window raises ValueError;
    callers should cancel that pending entry and wait for a fresh hourly signal.
    """
    validate(config)
    if not enabled(config):
        return None
    try:
        signal = pending['signal']
        source = _signal(signal)
        bars = list(closed_candles)
        times = [_time(_get(bar, 'time_ms'), 'entry_timing_hourly_alignment') for bar in bars]
        if any(b - a != HOUR for a, b in zip(times, times[1:])):
            raise ValueError('entry_timing_hourly_gap')
        if source['time_ms'] not in times:
            raise ValueError('entry_timing_original_signal_bar_unavailable')
        prefix = bars[:times.index(source['time_ms']) + 1]
        return build_trigger(prefix, signal, config, pending['boundary_ms'],
                             observed_at_ms=observed_at_ms)
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('entry_timing_pending_invalid') from exc


def _trigger(trigger, source, config):
    if not isinstance(trigger, Mapping):
        raise ValueError('entry_timing_trigger_missing')
    try:
        if (type(trigger['version']) is not int or trigger['version'] != 1
                or trigger['family'] != 'donchian20' or trigger['signal_timeframe'] != '1h'
                or type(trigger['breakout_bars']) is not int or trigger['breakout_bars'] != 20
                or type(trigger['direction']) is not int or trigger['direction'] != source['direction']):
            raise ValueError('entry_timing_trigger_invalid')
        expected_times = {'signal_time_ms': source['time_ms'], 'boundary_ms': source['time_ms'] + HOUR,
                          'prior_window_start_ms': source['time_ms'] - 20 * HOUR,
                          'prior_window_end_ms': source['time_ms'] - HOUR}
        if any(_time(trigger[key], 'entry_timing_trigger_invalid') != value
               for key, value in expected_times.items()):
            raise ValueError('entry_timing_trigger_mismatch')
        if (_number(trigger['signal_close'], 'entry_timing_trigger_invalid') != source['close']
                or _number(trigger['signal_atr'], 'entry_timing_trigger_invalid') != source['atr']):
            raise ValueError('entry_timing_trigger_mismatch')
        level = _number(trigger['breakout_level'], 'entry_timing_trigger_invalid')
        extension = source['direction'] * (source['close'] - level)
        if not 0 < extension <= source['atr'] * _number(
                config.get('max_breakout_extension_atr', 1.), 'entry_timing_extension_limit_invalid'):
            raise ValueError('entry_timing_original_breakout_invalid')
        return level
    except (KeyError, TypeError) as exc:
        raise ValueError('entry_timing_trigger_invalid') from exc


def evaluate(trigger, signal, config, *, reference_price, mark_price, fill_price=_NO_FILL):
    """Check side-specific bid/ask AND mark, then optional actual VWAP/slip fill.

    Breakout loss/equality or bad provenance cancels a pending signal permanently.
    Missing prices and excessive chasing only pause entry; exits are independent.
    Supply fill_price for the final pre-booking check, never just the top quote.
    """
    result = {'enabled': enabled(config), 'allowed': True, 'cancel_pending': False, 'reasons': []}
    if not result['enabled']:
        return result
    try:
        validate(config)
        source = _signal(signal)
        level = _trigger(trigger, source, config)
    except ValueError as exc:
        return {**result, 'allowed': False, 'cancel_pending': True, 'reasons': [str(exc)]}
    result.update(breakout_level=level, direction=source['direction'],
                  signal_close=source['close'], signal_atr=source['atr'])
    prices = {'reference': reference_price, 'mark': mark_price}
    if fill_price is not _NO_FILL:
        prices['fill'] = fill_price
    reasons = []
    for label, raw_price in prices.items():
        try:
            price = _number(raw_price, 'entry_timing_' + label + '_unavailable')
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        result[label + '_price'] = price
        extension = source['direction'] * (price - level)
        if extension <= 0:
            reasons.append('breakout_' + label + '_invalidated')
            result['cancel_pending'] = True
        if label != 'mark':
            if source['direction'] * (price - source['close']) > source['atr'] * _number(
                    config.get('entry_gap_atr', .5), 'entry_timing_gap_limit_invalid'):
                reasons.append('entry_timing_' + label + '_gap_exceeded')
            if extension > source['atr'] * _number(
                    config.get('max_breakout_extension_atr', 1.), 'entry_timing_extension_limit_invalid'):
                reasons.append('entry_timing_' + label + '_extension_exceeded')
    result.update(allowed=not reasons, reasons=reasons)
    return result
