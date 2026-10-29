"""Predeclared completed-hour entry alternatives for a new paper epoch only.

No I/O, account mutation, execution permission, or modification of shared signal
modules. Baseline delegates unchanged; candidates alter signal admission only.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from collections.abc import Mapping

import stock_entry_quality as quality
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
import stock_swing_signals as signals

VARIANTS = ('baseline', 'turning', 'turn_pullback')
POLICY = {
    'mode': 'closed_4h_turning_entry_v1',
    'confirmation_timeframe': '4h',
    'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
    'mid_slope_lookback_bars': 3,
    'minimum_ema_spread_fraction': .004,
    'require_price_slow': True,
    'original_signal_family': 'donchian20',
    'pullback_signal_family': 'trend_pullback',
    'pullback_ema_slope_bars': 6,
    'maximum_reclaim_extension_atr': 1.0,
    'donchian_priority': True,
}


def _fixed_policy(value):
    return (isinstance(value, Mapping) and value == POLICY
            and set(value) == set(POLICY)
            and all(type(value[key]) is type(expected) for key, expected in POLICY.items()))


def validate(config):
    if not isinstance(config, Mapping):
        raise ValueError('Entry timing candidate configuration must be a mapping')
    variant = config.get('entry_timing_variant', 'baseline')
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError('Unknown entry timing variant')
    policy = config.get('entry_timing_candidate_policy')
    if policy is not None and not _fixed_policy(policy):
        raise ValueError('Entry timing candidate policy differs from the fixed declaration')
    if variant == 'baseline':
        return
    if not _fixed_policy(policy):
        raise ValueError('A fixed entry timing candidate policy is required')
    expected = {'signal_timeframe': '1h', 'signal_family': 'donchian20',
                'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
                'ema_slope_bars': 6, 'breakout_bars': 20, 'require_price_slow': True,
                'max_atr_fraction': .03, 'max_breakout_extension_atr': 1.0}
    for key, value in expected.items():
        current = config.get(key)
        if isinstance(current, bool) != isinstance(value, bool) or current != value:
            raise ValueError('Candidate requires unchanged hourly rules: ' + key)
        if type(value) is int and type(current) is not int:
            raise ValueError('Candidate requires integer hourly rule: ' + key)
    confirmation.validate(config)
    if config.get('swing_confirmation_policy') != confirmation.POLICY:
        raise ValueError('Candidate requires the original closed-4h policy')
    quality.validate(config)


def evaluate_turning_confirmation(candles, index, config, direction=None):
    """Original strict context OR a fixed fast-led turning context, all closed.

    The three-bar change is EMA24[-1] - EMA24[-4] on COMPLETED 4h bars.
    Strict admission does not acquire an additional EMA24 slope requirement.
    A missing direction describes context, never an executable raw candidate.
    """
    validate(config)
    if config.get('entry_timing_variant', 'baseline') == 'baseline':
        return confirmation.evaluate(candles, index, config, direction)
    strict = confirmation.evaluate(candles, index, config, direction)
    result = {**strict, 'mode': POLICY['mode'], 'strict_allowed': strict['allowed'],
              'strict_reasons': list(strict['reasons']), 'turning_allowed': False,
              'mid_slope_lookback_bars': 3, 'mid_now': None, 'mid_3_bars_ago': None,
              'requested_direction': direction, 'turning_checks': {}}
    # Data failures can never be waived by the alternative ordering.
    if ('ema_fast' not in strict or 'ema_mid' not in strict or 'ema_slow' not in strict
            or strict.get('observed_bars', 0) < POLICY['warmup_bars']):
        return result
    bars = confirmation.completed_four_hour_bars(candles, index)
    four = signals.compute_indicators(bars, confirmation.POLICY)
    mid_now, mid_old = four['ema_mid'][-1], four['ema_mid'][-4]
    if any(value is None or not math.isfinite(value) for value in (mid_now, mid_old)):
        if not strict['allowed']:
            result['reasons'] = ['turning_4h_mid_history_unavailable']
        return result
    fast, mid, slow, price = (strict[key] for key in ('ema_fast', 'ema_mid', 'ema_slow', 'close'))
    context_side = 1 if fast > mid and fast > slow else -1 if fast < mid and fast < slow else 0
    side = direction if direction is not None else context_side
    checks = {
        'turning_4h_fast_mid': side * (fast - mid) > 0,
        'turning_4h_fast_slow': side * (fast - slow) > 0,
        'turning_4h_price_slow': side * (price - slow) > 0,
        'turning_4h_spread': strict['ema_spread_fraction'] >= .004,
        'turning_4h_mid_slope': side * (mid_now - mid_old) > 0,
    }
    turning = bool(side and all(checks.values()))
    allowed = strict['allowed'] or turning
    result.update(allowed=allowed, reasons=[] if allowed else [key for key, ok in checks.items() if not ok],
                  turning_allowed=turning, direction=context_side,
                  mid_now=mid_now, mid_3_bars_ago=mid_old, mid_change=mid_now - mid_old,
                  turning_checks=checks,
                  admitted_by='strict' if strict['allowed'] else 'turning' if turning else None)
    return result


def signal_with_diagnostics(candles, indicators, index, config=None):
    """Same call shape as stock_swing_profiles, with detached JSON diagnostics.

    A valid original Donchian raw candidate has priority. Pullback is considered
    only when that raw candidate is absent, never as a bypass after a rejection.
    The runner must freeze/check reclaim provenance and all execution protections.
    """
    config = {} if config is None else config
    validate(config)
    variant = config.get('entry_timing_variant', 'baseline')
    if variant == 'baseline':
        return profiles.signal_with_diagnostics(candles, indicators, index, config)
    signal = profiles.raw_signal_at(candles, indicators, index, config)
    donchian = asdict(signal) if signal else None
    kind = 'donchian20' if signal else None
    reclaim = None
    if (signal is None and variant == 'turn_pullback'
            and profiles.signal_activity(candles, index, config)['allowed']):
        signal = profiles.regime_signal_at(candles, indicators, index,
                                          {**config, 'signal_family': 'trend_pullback'})
        if signal:
            kind = 'trend_pullback'
            reclaim = {'family': kind, 'direction': signal.direction,
                       'ema_fast': float(indicators['ema_fast'][index]),
                       'signal_time_ms': signal.time_ms,
                       'boundary_ms': signal.time_ms + confirmation.HOUR,
                       'signal_close': signal.close, 'signal_atr': signal.atr}
    context = evaluate_turning_confirmation(candles, index, config,
                                           signal.direction if signal else None)
    candidate = asdict(signal) if signal else None
    detail = {'variant': variant, 'signal_kind': kind, 'frozen_reclaim': reclaim,
              'donchian_raw_signal': donchian,
              'strict_4h_allowed': context['strict_allowed'],
              'turning_4h_allowed': context['turning_allowed'],
              'mid_slope_lookback_bars': 3, 'mid_now': context['mid_now'],
              'mid_3_bars_ago': context['mid_3_bars_ago'],
              'execution_permission': False}
    diagnostics = {'swing_confirmation': context,
        'signal_before_confirmation': candidate,
        'signal_after_confirmation': dict(candidate) if candidate and context['allowed'] else None,
        'signal_after_quality': None, 'entry_quality': None,
        'entry_timing_candidate': detail,
        'admission': {'stage': 'no_raw_signal' if signal is None else 'admitted',
            'reasons': ['no_raw_signal'] if signal is None else [],
            'confirmation_rejected_candidate': bool(signal and not context['allowed'])}}
    if not context['allowed']:
        if signal:
            diagnostics['admission'].update(stage='confirmation_rejected', reasons=list(context['reasons']))
        return None, diagnostics
    if signal:
        outcome = quality.evaluate(candles, indicators, index, config, signal.direction)
        diagnostics['entry_quality'] = deepcopy(outcome)
        if not outcome['allowed']:
            diagnostics['admission'].update(stage='quality_rejected', reasons=list(outcome['reasons']))
            return None, diagnostics
        diagnostics['signal_after_quality'] = dict(candidate)
    return signal, diagnostics


def signal_at(candles, indicators, index, config=None):
    return signal_with_diagnostics(candles, indicators, index, config)[0]
