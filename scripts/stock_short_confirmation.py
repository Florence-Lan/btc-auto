"""Isolated, predeclared short-only completed-4h confirmation alternative.

No I/O or account mutation. The caller supplies the completed-hour prefix and
retains all original entry timing, event, execution, sizing and exit checks.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import stock_entry_quality as quality
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
import stock_swing_signals as signals

POLICY_KEY = 'short_confirmation_policy'
POLICY = {
    'mode': 'closed_4h_price_led_short_v1',
    'confirmation_timeframe': '4h',
    'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
    'require_close_below_fast_mid': True,
    'fast_slope_lookback_bars': 1,
    'minimum_ema_spread_fraction': .004,
}


def enabled(config):
    return POLICY_KEY in config


def validate(config):
    if not isinstance(config, Mapping):
        raise ValueError('short_confirmation_config_invalid')
    if not enabled(config):
        return
    policy = config[POLICY_KEY]
    if (not isinstance(policy, Mapping) or dict(policy) != POLICY
            or set(policy) != set(POLICY)
            or any(type(policy[key]) is not type(value) for key, value in POLICY.items())):
        raise ValueError('short_confirmation_policy_invalid')
    expected = {'signal_timeframe': '1h', 'signal_family': 'donchian20',
                'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
                'breakout_bars': 20, 'require_price_slow': True,
                'max_atr_fraction': .03, 'max_breakout_extension_atr': 1.0}
    for key, value in expected.items():
        if config.get(key) != value or isinstance(config.get(key), bool) != isinstance(value, bool):
            raise ValueError('short_confirmation_hourly_rules_changed:' + key)
        if type(value) is int and type(config.get(key)) is not int:
            raise ValueError('short_confirmation_hourly_rules_changed:' + key)
    confirmation.validate(config)
    if config.get('swing_confirmation_policy') != confirmation.POLICY:
        raise ValueError('short_confirmation_strict_policy_required')
    quality.validate(config)


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _valid_hourly_prefix(candles, index):
    for bar in candles[:index + 1]:
        if type(bar.time_ms) is not int or bar.time_ms < 0:
            return False
        if not all(_positive(getattr(bar, key)) for key in ('open', 'high', 'low', 'close')):
            return False
        if (bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close)
                or bar.low > bar.high or isinstance(bar.volume, bool)
                or not isinstance(bar.volume, (int, float))
                or not math.isfinite(bar.volume) or bar.volume < 0):
            return False
    return True


def evaluate_short_confirmation(candles, index, config, *, strict=None):
    """Return strict OR price-led SHORT context; never creates a raw signal.

    Retain the original absolute fast/slow separation as a background floor.
    The new branch intentionally does not require price below 4h EMA60 or
    fast/mid/slow bearish ordering. Its slope is EMA8[-1] < EMA8[-2].
    Only complete UTC 4h groups ending within the supplied index are used.
    """
    validate(config)
    strict = confirmation.evaluate(candles, index, config, -1) if strict is None else strict
    result = {'enabled': enabled(config), 'mode': POLICY['mode'], 'direction': -1,
        'allowed': strict['allowed'], 'strict_allowed': strict['allowed'],
        'strict_reasons': list(strict['reasons']), 'price_led_allowed': False,
        'reasons': list(strict['reasons']), 'checks': {},
        'confirmation_close_ms': strict.get('confirmation_close_ms'),
        'ema_fast': strict.get('ema_fast'), 'ema_fast_previous': None,
        'ema_mid': strict.get('ema_mid'), 'ema_slow': strict.get('ema_slow'),
        'close': strict.get('close'), 'ema_spread_fraction': strict.get('ema_spread_fraction'),
        'admitted_by': 'strict' if strict['allowed'] else None}
    if not result['enabled'] or strict['allowed']:
        return result
    waivable = {'swing_4h_ema_ordering', 'swing_4h_price_slow', 'swing_4h_spread_too_small'}
    if (not strict.get('enabled') or strict.get('observed_bars', 0) < POLICY['warmup_bars']
            or not set(strict['reasons']) <= waivable):
        return result
    try:
        if not _valid_hourly_prefix(candles, index):
            raise ValueError('invalid_hourly_prefix')
        bars = confirmation.completed_four_hour_bars(candles, index)
        if len(bars) < POLICY['warmup_bars']:
            raise ValueError('insufficient_complete_four_hour_history')
        indicators = signals.compute_indicators(bars, confirmation.POLICY)
        fast, previous = indicators['ema_fast'][-1], indicators['ema_fast'][-2]
        mid, slow, close = indicators['ema_mid'][-1], indicators['ema_slow'][-1], bars[-1].close
        if not all(_positive(value) for value in (fast, previous, mid, slow, close)):
            raise ValueError('invalid_four_hour_indicators')
        spread = abs(fast - slow) / close
        checks = {'short_4h_close_below_fast': close < fast,
                  'short_4h_close_below_mid': close < mid,
                  'short_4h_fast_falling': fast < previous,
                  'short_4h_spread': spread >= POLICY['minimum_ema_spread_fraction']}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
        return {**result, 'reasons': ['short_confirmation_data_unavailable']}
    allowed = all(checks.values())
    return {**result, 'allowed': allowed, 'price_led_allowed': allowed, 'checks': checks,
        'reasons': [] if allowed else [key for key, passed in checks.items() if not passed],
        'ema_fast': fast, 'ema_fast_previous': previous, 'ema_mid': mid, 'ema_slow': slow,
        'close': close, 'ema_spread_fraction': spread,
        'admitted_by': 'price_led_short' if allowed else None}


def signal_with_diagnostics(candles, indicators, index, config=None):
    """Same API as stock_swing_profiles; nonshort results delegate unchanged."""
    config = {} if config is None else config
    validate(config)
    original = profiles.signal_with_diagnostics(candles, indicators, index, config)
    if not enabled(config):
        return original
    signal, diagnostic = original
    raw = diagnostic.get('signal_before_confirmation')
    if not raw or raw['direction'] != -1 or diagnostic['swing_confirmation']['allowed']:
        # In particular, never resurrect a strict-context quality rejection.
        # Long decisions AND every original diagnostic field stay identical.
        return original
    outcome = evaluate_short_confirmation(candles, index, config,
                                           strict=diagnostic['swing_confirmation'])
    diagnostic = deepcopy(diagnostic)
    strict = diagnostic['swing_confirmation']
    diagnostic['short_confirmation'] = deepcopy(outcome)
    diagnostic['swing_confirmation'] = {**strict, 'allowed': outcome['allowed'],
        'reasons': list(outcome['reasons']), 'strict_allowed': strict['allowed'],
        'strict_reasons': list(strict['reasons']), 'strict_direction': strict.get('direction'),
        'direction': -1 if outcome['allowed'] else strict.get('direction'),
        'short_confirmation_mode': POLICY['mode'], 'price_led_short_allowed': outcome['price_led_allowed']}
    if not outcome['allowed']:
        diagnostic['admission'].update(stage='confirmation_rejected',
            reasons=list(outcome['reasons']), confirmation_rejected_candidate=True)
        return None, diagnostic
    signal = signals.Signal(**raw)
    diagnostic['signal_after_confirmation'] = dict(raw)
    diagnostic['admission'].update(stage='admitted', reasons=[], confirmation_rejected_candidate=False)
    outcome = quality.evaluate(candles, indicators, index, config, -1)
    diagnostic['entry_quality'] = deepcopy(outcome)
    if not outcome['allowed']:
        diagnostic['admission'].update(stage='quality_rejected', reasons=list(outcome['reasons']))
        return None, diagnostic
    diagnostic['signal_after_quality'] = dict(raw)
    return signal, diagnostic


def signal_at(candles, indicators, index, config=None):
    return signal_with_diagnostics(candles, indicators, index, config)[0]
