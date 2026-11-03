"""Causal 1h/2h/4h alternatives for explicitly activated main-stock entries.

The original hourly signal and quality rules remain required. This module is
pure: it does not fetch data, create accounts, or change existing exit epochs.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import main_stock_trend as main
import stock_entry_quality as quality
import stock_swing_signals as signals

HOUR = main.HOUR
POLICY_KEY = 'main_stock_entry_trend_policy'
POLICY = {
    'mode': 'main_multitimeframe_entry_v1',
    'confirmation_timeframes': ['1h', '2h', '4h'],
    'combination': 'any',
    'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
    'minimum_ema_spread_fraction': .004,
    'confirmation': 'strict_or_price_led',
}
INTERVALS = {'1h': HOUR, '2h': 2 * HOUR, '4h': 4 * HOUR}


def enabled(config):
    return isinstance(config, Mapping) and POLICY_KEY in config


def validate(config):
    if not isinstance(config, Mapping):
        raise ValueError('main_entry_trend_config_invalid')
    if not enabled(config):
        return
    policy = config[POLICY_KEY]
    if (not isinstance(policy, Mapping) or dict(policy) != POLICY
            or any(type(policy.get(k)) is not type(v) for k, v in POLICY.items())):
        raise ValueError('main_entry_trend_policy_invalid')
    if not main.enabled(config):
        raise ValueError('main_entry_trend_existing_main_policy_required')
    main.validate(config)


def _completed_bars(prefix, interval):
    """Aggregate only complete UTC groups from an already validated prefix."""
    if interval == HOUR:
        return prefix
    groups = {}
    for bar in prefix:
        start = bar.time_ms // interval * interval
        groups.setdefault(start, []).append(bar)
    result = []
    for start, group in groups.items():
        if [bar.time_ms for bar in group] != list(range(start, start + interval, HOUR)):
            continue
        result.append(signals.Candle(start, group[0].open,
            max(bar.high for bar in group), min(bar.low for bar in group),
            group[-1].close, sum(bar.volume for bar in group)))
    return result


def _channel(prefix, config, timeframe, direction):
    interval = INTERVALS[timeframe]
    bars = _completed_bars(prefix, interval)
    boundary = prefix[-1].time_ms + HOUR
    result = {'timeframe': timeframe, 'allowed': False, 'status': 'unavailable',
        'reasons': [], 'admitted_by': None, 'observed_bars': len(bars),
        'required_bars': POLICY['warmup_bars'], 'closed_at_ms': bars[-1].time_ms + interval if bars else None,
        'direction': 0, 'candidate_direction': direction,
        'minimum_ema_spread_fraction': POLICY['minimum_ema_spread_fraction']}
    if not bars or result['closed_at_ms'] != boundary // interval * interval:
        return {**result, 'reasons': ['entry_trend_' + timeframe + '_latest_closed_bar_missing']}
    if len(bars) < POLICY['warmup_bars']:
        return {**result, 'status': 'warming_up', 'reasons': ['entry_trend_' + timeframe + '_warmup']}
    try:
        context = main._ema_context(bars, POLICY, interval)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError):
        return {**result, 'reasons': ['entry_trend_' + timeframe + '_indicators_unavailable']}
    result.update(context)
    side = direction if direction is not None else context['direction']
    spread = context['ema_spread_fraction'] >= POLICY['minimum_ema_spread_fraction']
    strict = bool(side and context['strict_direction'] == side and spread)
    price_led = bool(side and context['price_led_direction'] == side and spread)
    allowed = strict or price_led
    reasons = [] if allowed else ['entry_trend_' + timeframe + (
        '_spread_too_small' if not spread else '_direction_not_confirmed')]
    return {**result, 'allowed': allowed, 'reasons': reasons,
        'strict_allowed': strict, 'price_led_allowed': price_led,
        'admitted_by': 'strict' if strict else 'price_led' if price_led else None}


def _evaluate_prefix(prefix, config, direction):
    channels = {label: _channel(prefix, config, label, direction) for label in INTERVALS}
    admitted = [label for label, value in channels.items() if value['allowed']]
    return {'enabled': True, 'mode': POLICY['mode'], 'allowed': bool(admitted),
        'status': 'confirmed' if admitted else 'not_confirmed', 'direction': direction,
        'signal_close_ms': prefix[-1].time_ms + HOUR, 'checked_at_ms': prefix[-1].time_ms + HOUR,
        'confirmation_timeframe': '1h_or_2h_or_4h', 'combination': 'any',
        'channels': channels, 'admitted_by_timeframes': admitted,
        'decision_scope': 'current_context_not_entry_permission' if direction is None else 'entry_confirmation_only',
        'reasons': [] if admitted else list(dict.fromkeys(
            reason for channel in channels.values() for reason in channel['reasons']))}


def evaluate_confirmation(candles, index, config, direction=None):
    """Confirm a raw side; a None side reports context without granting entry."""
    validate(config)
    if not enabled(config):
        return {'enabled': False, 'allowed': True, 'reasons': [], 'channels': {},
                'admitted_by_timeframes': [], 'direction': direction}
    if direction is not None and (type(direction) is not int or direction not in (-1, 1)):
        raise ValueError('main_entry_trend_direction_invalid')
    try:
        prefix = main._validated_prefix(candles, index)
        return _evaluate_prefix(prefix, config, direction)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else 'main_entry_trend_data_unavailable'
        return {'enabled': True, 'mode': POLICY['mode'], 'allowed': False,
            'status': 'unavailable', 'direction': direction, 'reasons': [reason],
            'signal_close_ms': None, 'checked_at_ms': None, 'confirmation_timeframe': '1h_or_2h_or_4h',
            'combination': 'any', 'admitted_by_timeframes': [],
            'decision_scope': 'current_context_not_entry_permission' if direction is None else 'entry_confirmation_only',
            'channels': {label: {'timeframe': label, 'status': 'unavailable', 'allowed': False,
                'reasons': [reason], 'admitted_by': None} for label in INTERVALS}}


def signal_with_diagnostics(candles, indicators, index, config=None):
    """Apply OR confirmation to an original raw signal, then retain its quality gate."""
    config = {} if config is None else config
    validate(config)
    original_signal, original = main.signal_with_diagnostics(candles, indicators, index, config)
    if not enabled(config):
        return original_signal, original
    diagnostic = deepcopy(original)
    raw = diagnostic.get('signal_before_confirmation')
    direction = raw['direction'] if raw else None
    outcome = evaluate_confirmation(candles, index, config, direction)
    previous = deepcopy(diagnostic['swing_confirmation'])
    diagnostic['entry_trend_confirmation'] = deepcopy(outcome)
    diagnostic['previous_swing_confirmation'] = previous
    diagnostic['swing_confirmation'] = {**previous,
        'enabled': True, 'allowed': outcome['allowed'], 'reasons': list(outcome['reasons']),
        'confirmation_timeframe': '1h_or_2h_or_4h',
        'admitted_by_timeframes': list(outcome['admitted_by_timeframes']),
        'channels': deepcopy(outcome['channels']), 'direction': direction,
        'decision_scope': outcome['decision_scope']}
    if not raw:
        return None, diagnostic
    if not outcome['allowed']:
        diagnostic.update(signal_after_confirmation=None, signal_after_quality=None)
        diagnostic['admission'].update(stage='confirmation_rejected', reasons=list(outcome['reasons']),
                                      confirmation_rejected_candidate=True)
        return None, diagnostic
    diagnostic['signal_after_confirmation'] = dict(raw)
    # A previously rejected quality candidate must not be resurrected even if a
    # caller supplies a stateful test double or a changing external evaluator.
    if diagnostic['admission']['stage'] == 'quality_rejected':
        return None, diagnostic
    checked = quality.evaluate(candles, indicators, index, config, direction)
    diagnostic['entry_quality'] = deepcopy(checked)
    if not checked['allowed']:
        diagnostic['signal_after_quality'] = None
        diagnostic['admission'].update(stage='quality_rejected', reasons=list(checked['reasons']),
                                      confirmation_rejected_candidate=False)
        return None, diagnostic
    signal = signals.Signal(**raw)
    diagnostic['signal_after_quality'] = dict(raw)
    diagnostic['admission'].update(stage='admitted', reasons=[], confirmation_rejected_candidate=False)
    return signal, diagnostic


def signal_at(*args, **kwargs):
    return signal_with_diagnostics(*args, **kwargs)[0]


def trend_diagnostics(candles, config, timestamp):
    """Present all current complete contexts, separately from an entry permit."""
    validate(config)
    if not enabled(config):
        return main.trend_diagnostics(candles, config, timestamp)
    result = {'enabled': True, 'checked_at_ms': timestamp, 'status': 'unavailable',
        'reasons': [], 'signal_close_ms': None, 'confirmation_close_ms': None,
        'hourly': None, 'two_hour': None, 'four_hour': None, 'entry_confirmation': None}
    try:
        prefix = main._current_prefix(candles, timestamp)
        outcome = _evaluate_prefix(prefix, config, None)
        result.update(signal_close_ms=prefix[-1].time_ms + HOUR, entry_confirmation=outcome)
        for key, timeframe in (('hourly', '1h'), ('two_hour', '2h'), ('four_hour', '4h')):
            result[key] = deepcopy(outcome['channels'][timeframe])
        result['confirmation_close_ms'] = result['four_hour'].get('closed_at_ms')
        result['status'] = 'healthy' if any(
            channel['status'] == 'healthy' for channel in outcome['channels'].values()) else 'unavailable'
        if result['status'] != 'healthy':
            result['reasons'] = list(outcome['reasons'])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
        result['reasons'] = [str(exc) if isinstance(exc, ValueError) else 'main_entry_trend_data_unavailable']
    return result
