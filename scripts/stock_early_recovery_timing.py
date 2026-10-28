"""Frozen recovery boundary admission; original breakout admission delegates unchanged."""
from dataclasses import asdict
import math

import stock_entry_timing as original
import stock_early_recovery as recovery
from stock_swing_signals import initial_stop

MODE = 'early_recovery_boundary_v1'
POLICY = {'mode': MODE, 'require_recovery_valid_at_fill': True}
_MISSING = object()


def is_recovery(config):
    return config.get('signal_family') == 'early_recovery_1h_v1'


def enabled(config):
    return original.enabled(config)


def validate(config):
    if not is_recovery(config):
        return original.validate(config)
    recovery.validate(config)
    if config.get('entry_timing_policy') != POLICY or config.get('signal_timeframe') != '1h':
        raise ValueError('recovery_entry_timing_policy_invalid')
    if not 0 < config['entry_gap_atr'] <= .5:
        raise ValueError('recovery_entry_gap_invalid')


def build_trigger(bars, signal, config, boundary_ms, *, observed_at_ms=None):
    if not is_recovery(config):
        return original.build_trigger(bars, signal, config, boundary_ms, observed_at_ms=observed_at_ms)
    validate(config)
    source = original._signal(signal)
    if (boundary_ms != source['time_ms'] + original.HOUR or observed_at_ms is None
            or type(observed_at_ms) is not int or observed_at_ms < boundary_ms):
        raise ValueError('recovery_unfinished_or_mismatched_signal')
    if not bars or bars[-1].time_ms != source['time_ms']:
        raise ValueError('recovery_original_prefix_mismatch')
    rebuilt, diagnostic = recovery.candidate_with_diagnostics(bars, config, asof_ms=boundary_ms)
    if rebuilt is None or asdict(rebuilt) != asdict(signal):
        raise ValueError('recovery_original_signal_mismatch')
    level = diagnostic['reclaim_level']
    if not math.isfinite(level) or not 0 < source['direction'] * (source['close'] - level) <= source['atr']:
        raise ValueError('recovery_original_boundary_invalid')
    return {'version': 1, 'family': 'early_recovery_1h_v1', **source,
            'boundary_ms': boundary_ms, 'reclaim_level': level,
            'structural_invalidation_level': diagnostic['early_recovery']['structural_invalidation_level']}


def rebuild_trigger(bars, pending, config, *, observed_at_ms=None):
    if not is_recovery(config):
        return original.rebuild_trigger(bars, pending, config, observed_at_ms=observed_at_ms)
    source = original._signal(pending['signal'])
    prefix = [bar for bar in bars if bar.time_ms <= source['time_ms']]
    from stock_swing_signals import Signal
    return build_trigger(prefix, Signal(**pending['signal']), config, pending['boundary_ms'],
                         observed_at_ms=observed_at_ms)


def evaluate(trigger, signal, config, *, reference_price, mark_price, fill_price=_MISSING):
    if not is_recovery(config):
        kwargs = {} if fill_price is _MISSING else {'fill_price': fill_price}
        return original.evaluate(trigger, signal, config, reference_price=reference_price,
                                 mark_price=mark_price, **kwargs)
    result = {'enabled': True, 'allowed': False, 'cancel_pending': False, 'reasons': []}
    try:
        validate(config)
        source = original._signal(signal)
        if (not isinstance(trigger, dict) or trigger.get('version') != 1
                or trigger.get('family') != 'early_recovery_1h_v1'
                or any(trigger.get(k) != value for k, value in source.items())
                or trigger.get('boundary_ms') != source['time_ms'] + original.HOUR):
            raise ValueError('recovery_frozen_trigger_mismatch')
        level = original._number(trigger.get('reclaim_level'), 'recovery_boundary_invalid')
        structure = original._number(trigger.get('structural_invalidation_level'), 'recovery_structure_invalid')
        if not 0 < source['direction'] * (source['close'] - level) <= source['atr']:
            raise ValueError('recovery_original_boundary_invalid')
    except (ValueError, TypeError, KeyError) as exc:
        return {**result, 'cancel_pending': True, 'reasons': [str(exc)]}
    result.update(reclaim_level=level, direction=source['direction'], signal_close=source['close'])
    prices = {'reference': reference_price, 'mark': mark_price}
    if fill_price is not _MISSING:
        prices['fill'] = fill_price
    for label, raw in prices.items():
        try:
            price = original._number(raw, 'recovery_' + label + '_unavailable')
        except ValueError as exc:
            result['reasons'].append(str(exc))
            continue
        extension = source['direction'] * (price - level)
        if extension <= 0:
            result['reasons'].append('recovery_' + label + '_invalidated')
            result['cancel_pending'] = True
        if label != 'mark':
            if source['direction'] * (price - source['close']) > source['atr'] * config['entry_gap_atr']:
                result['reasons'].append('recovery_' + label + '_gap_exceeded')
            if extension > source['atr']:
                result['reasons'].append('recovery_' + label + '_extension_exceeded')
            stop = initial_stop(price, source['direction'], source['atr'], config)
            if stop is None or source['direction'] * (structure - stop) < 0:
                result['reasons'].append('recovery_' + label + '_structure_outside_original_stop')
    result['allowed'] = not result['reasons']
    return result
