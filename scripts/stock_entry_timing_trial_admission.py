"""Frozen admission for the isolated turning/pullback timing trial."""
from dataclasses import asdict
import math

import stock_entry_timing as original
import stock_entry_timing_candidates as candidates
import stock_swing_signals as signals

POLICY = {'mode': 'trend_pullback_or_breakout_v1', 'require_frozen_boundary_at_fill': True}
_MISSING = object()


def mixed(config):
    return config.get('entry_timing_variant') == 'turn_pullback'


def original_config(config):
    return {**config, 'entry_timing_policy': original.POLICY.copy()}


def enabled(config):
    return original.enabled(config)


def validate(config):
    if not mixed(config):
        return original.validate(config)
    candidates.validate(config)
    if config.get('entry_timing_policy') != POLICY:
        raise ValueError('timing_trial_admission_policy_invalid')
    original.validate(original_config(config))


def build_trigger(bars, signal, config, boundary_ms, *, observed_at_ms=None):
    if not mixed(config):
        return original.build_trigger(bars, signal, config, boundary_ms, observed_at_ms=observed_at_ms)
    validate(config)
    source = original._signal(signal)
    if (boundary_ms != source['time_ms'] + original.HOUR or type(observed_at_ms) is not int
            or observed_at_ms < boundary_ms or not bars or bars[-1].time_ms != source['time_ms']):
        raise ValueError('timing_trial_original_closed_prefix_mismatch')
    indicators = signals.compute_indicators(bars, config)
    rebuilt, diagnostic = candidates.signal_with_diagnostics(bars, indicators, len(bars) - 1, config)
    if rebuilt is None or asdict(rebuilt) != asdict(signal):
        raise ValueError('timing_trial_original_signal_mismatch')
    kind = diagnostic['entry_timing_candidate']['signal_kind']
    if kind == 'donchian20':
        return original.build_trigger(bars, signal, original_config(config), boundary_ms,
                                      observed_at_ms=observed_at_ms)
    if kind != 'trend_pullback':
        raise ValueError('timing_trial_signal_kind_invalid')
    level = original._number(diagnostic['entry_timing_candidate']['frozen_reclaim']['ema_fast'],
                             'timing_trial_ema_invalid')
    direction = source['direction']
    structure = (min(b.low for b in bars[-2:]) - .25 * signal.atr if direction == 1
                 else max(b.high for b in bars[-2:]) + .25 * signal.atr)
    stop = signals.initial_stop(signal.close, direction, signal.atr, config)
    if (not math.isfinite(structure) or structure <= 0 or stop is None
            or direction * (structure - stop) < 0
            or not 0 < direction * (signal.close - level) <= signal.atr):
        raise ValueError('timing_trial_pullback_structure_or_extension_invalid')
    return {'version': 1, 'family': 'trend_pullback', **source, 'boundary_ms': boundary_ms,
            'reclaim_level': level, 'structural_invalidation_level': structure}


def rebuild_trigger(bars, pending, config, *, observed_at_ms=None):
    if not mixed(config):
        return original.rebuild_trigger(bars, pending, config, observed_at_ms=observed_at_ms)
    source = original._signal(pending['signal'])
    prefix = [bar for bar in bars if bar.time_ms <= source['time_ms']]
    return build_trigger(prefix, signals.Signal(**pending['signal']), config, pending['boundary_ms'],
                         observed_at_ms=observed_at_ms)


def evaluate(trigger, signal, config, *, reference_price, mark_price, fill_price=_MISSING):
    if not mixed(config) or isinstance(trigger, dict) and trigger.get('family') == 'donchian20':
        kwargs = {} if fill_price is _MISSING else {'fill_price': fill_price}
        return original.evaluate(trigger, signal, original_config(config) if mixed(config) else config,
                                 reference_price=reference_price, mark_price=mark_price, **kwargs)
    result = {'enabled': True, 'allowed': False, 'cancel_pending': False, 'reasons': []}
    try:
        validate(config)
        source = original._signal(signal)
        if (not isinstance(trigger, dict) or type(trigger.get('version')) is not int
                or trigger['version'] != 1 or trigger.get('family') != 'trend_pullback'
                or any(trigger.get(k) != value for k, value in source.items())
                or trigger.get('boundary_ms') != source['time_ms'] + original.HOUR):
            raise ValueError('timing_trial_frozen_trigger_mismatch')
        level = original._number(trigger.get('reclaim_level'), 'timing_trial_boundary_invalid')
        structure = original._number(trigger.get('structural_invalidation_level'), 'timing_trial_structure_invalid')
        if not 0 < source['direction'] * (source['close'] - level) <= source['atr']:
            raise ValueError('timing_trial_original_extension_invalid')
    except (ValueError, TypeError, KeyError) as exc:
        return {**result, 'cancel_pending': True, 'reasons': [str(exc)]}
    result.update(reclaim_level=level, direction=source['direction'], signal_close=source['close'])
    prices = {'reference': reference_price, 'mark': mark_price}
    if fill_price is not _MISSING:
        prices['fill'] = fill_price
    for label, raw in prices.items():
        try:
            price = original._number(raw, 'timing_trial_' + label + '_unavailable')
        except ValueError as exc:
            result['reasons'].append(str(exc))
            continue
        direction = source['direction']
        extension = direction * (price - level)
        if extension <= 0:
            result['reasons'].append('timing_trial_' + label + '_reclaim_invalidated')
            result['cancel_pending'] = True
        if label != 'mark':
            if direction * (price - source['close']) > source['atr'] * config['entry_gap_atr']:
                result['reasons'].append('timing_trial_' + label + '_gap_exceeded')
            if extension > source['atr']:
                result['reasons'].append('timing_trial_' + label + '_extension_exceeded')
            stop = signals.initial_stop(price, direction, source['atr'], config)
            if stop is None or direction * (structure - stop) < 0:
                result['reasons'].append('timing_trial_' + label + '_structure_outside_original_stop')
    result['allowed'] = not result['reasons']
    return result
