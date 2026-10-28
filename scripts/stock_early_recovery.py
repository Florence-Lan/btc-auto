"""Fixed, symmetric early-recovery mechanism for an isolated paper experiment.

No clients, account state, orders or clocks are read. The last 60 consecutive
completed UTC hours seed the indicators; older prefixes and unfinished/future
price data cannot change the result. Signal.time_ms is the signal bar OPEN, as
in stock_swing_signals.Signal. Admission/fees/depth/actual-fill stops remain the
runner's responsibility. This mechanism does not assert a profitable edge.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math

from stock_swing_signals import Candle, Signal, compute_indicators, initial_stop

HOUR = 3_600_000
SIGNAL_FAMILY = 'early_recovery_1h_v1'
POLICY = {
    'mode': SIGNAL_FAMILY, 'signal_timeframe': '1h', 'history_window_bars': 60,
    'ema_fast': 8, 'ema_mid': 24, 'atr_period': 14,
    'context_lookback_bars': 6, 'minimum_context_move_atr': 1.0,
    'maximum_reclaim_extension_atr': 1.0,
    'structural_lookback_bars': 6, 'structural_buffer_atr': .25,
    'stop_atr': 2.0, 'max_stop_fraction': .05, 'max_atr_fraction': .03,
    'min_stop_fraction_source': 'symbol_profile',
}


def _number(value):
    if isinstance(value, bool):
        raise ValueError('nonfinite_market_data')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite_market_data')
    return result


def validate(config=None):
    """Validate the fixed policy inside a full runner config; return a copy.

    Only the existing per-symbol minimum stop is inherited (.5% or 2%). No
    optimization knobs are introduced. Execution-only config keys are ignored.
    """
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise ValueError('early_recovery_config_required')
    supplied = config.get('early_recovery_policy', POLICY)
    if not isinstance(supplied, Mapping) or set(supplied) != set(POLICY):
        raise ValueError('fixed_early_recovery_policy_required')
    for key, expected in POLICY.items():
        value = supplied[key]
        if isinstance(expected, (int, float)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or isinstance(expected, int) and type(value) is not int
                    or not math.isfinite(value) or value != expected):
                raise ValueError('fixed_early_recovery_policy_required:' + key)
        elif type(value) is not str or value != expected:
            raise ValueError('fixed_early_recovery_policy_required:' + key)
    if config.get('signal_family', SIGNAL_FAMILY) != SIGNAL_FAMILY:
        raise ValueError('early_recovery_signal_family_required')
    for key in ('stop_atr', 'max_stop_fraction', 'max_atr_fraction'):
        value = config.get(key, POLICY[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != POLICY[key]:
            raise ValueError('existing_risk_parameter_mismatch:' + key)
    minimum = config.get('min_stop_fraction', .02)
    if isinstance(minimum, bool) or type(minimum) not in (int, float) or minimum not in (.005, .02):
        raise ValueError('existing_symbol_minimum_stop_required')
    return dict(POLICY)


def _timestamp(value):
    if type(value) is not int or value < 0:
        raise ValueError('invalid_candle_timestamp')
    return value


def _open_time(row):
    if isinstance(row, Candle):
        return _timestamp(row.time_ms)
    if isinstance(row, Mapping):
        return _timestamp(row.get('time_ms', row.get('open_time_ms')))
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) >= 6:
        return _timestamp(row[0])
    raise ValueError('invalid_candle_row')


def _candle(row):
    stamp = _open_time(row)
    if isinstance(row, Candle):
        values = [row.open, row.high, row.low, row.close, row.volume]
    elif isinstance(row, Mapping):
        if row.get('closed') is False:
            raise ValueError('unclosed_candle')
        if 'close_time_ms' in row and _timestamp(row['close_time_ms']) != stamp + HOUR - 1:
            raise ValueError('invalid_candle_close_time')
        values = [row[key] for key in ('open', 'high', 'low', 'close')]
        values.append(row.get('volume', 0))
    else:
        if len(row) >= 7 and _timestamp(row[6]) != stamp + HOUR - 1:
            raise ValueError('invalid_candle_close_time')
        values = row[1:6]
    return Candle(stamp, *(_number(value) for value in values))


def _closed_window(rows, asof_ms):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError('invalid_candle_history')
    eligible = []
    for row in rows:
        stamp = _open_time(row)
        # Prices of these rows are deliberately never parsed/read.
        if stamp + HOUR <= asof_ms:
            eligible.append(row)
    if len(eligible) < POLICY['history_window_bars']:
        raise ValueError('insufficient_closed_history')
    bars = [_candle(row) for row in eligible[-POLICY['history_window_bars']:]]
    if any(bar.time_ms % HOUR for bar in bars):
        raise ValueError('unaligned_hourly_candles')
    if any(right.time_ms - left.time_ms != HOUR for left, right in zip(bars, bars[1:])):
        raise ValueError('nonconsecutive_hourly_candles')
    if bars[-1].time_ms + HOUR != asof_ms // HOUR * HOUR:
        raise ValueError('latest_closed_hour_missing')
    return bars


def _diagnostics(asof_ms, policy):
    details = {'policy': dict(policy), 'asof_ms': asof_ms, 'reasons': [],
               'reclaim_level': None, 'signal_ema_fast': None,
               'prior_high': None, 'prior_low': None, 'boundary_ms': None}
    return {'swing_confirmation': {'enabled': False, 'allowed': True, 'reasons': []},
            'signal_before_confirmation': None, 'signal_after_confirmation': None,
            'signal_after_quality': None, 'entry_quality': None,
            'admission': {'stage': 'no_raw_signal', 'reasons': [],
                          'confirmation_rejected_candidate': False},
            'early_recovery': details,
            'reclaim_level': None, 'signal_ema_fast': None,
            'prior_high': None, 'prior_low': None, 'boundary_ms': None}


def candidate_with_diagnostics(rows, config=None, *, asof_ms=None):
    """Return (compatible Signal or None, detached diagnostics).

    asof_ms is required explicitly or in config; no wall clock is inferred.
    Bad market data fails closed in diagnostics. Invalid fixed policy or clock
    raises ValueError before signal computation. A rejected data/shape condition
    never becomes a fill authorization.
    """
    config = {} if config is None else config
    policy = validate(config)
    asof_ms = config.get('asof_ms') if asof_ms is None else asof_ms
    if type(asof_ms) is not int or asof_ms < HOUR:
        raise ValueError('explicit_decision_asof_ms_required')
    diagnostics = _diagnostics(asof_ms, policy)
    details = diagnostics['early_recovery']

    def reject(reason):
        details['reasons'].append(reason)
        diagnostics['admission']['reasons'] = list(details['reasons'])
        return None, diagnostics

    try:
        bars = _closed_window(rows, asof_ms)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        # Never echo arbitrary provider exception text or values.
        safe = str(exc) if type(exc) is ValueError and str(exc) in {
            'invalid_candle_timestamp', 'invalid_candle_row', 'unclosed_candle',
            'invalid_candle_close_time', 'invalid_candle_history', 'insufficient_closed_history',
            'unaligned_hourly_candles', 'nonconsecutive_hourly_candles',
            'latest_closed_hour_missing', 'nonfinite_market_data'} else 'invalid_ohlcv'
        return reject(safe)
    settings = {**policy, 'ema_slow': 60, 'warmup_bars': 60,
                'min_stop_fraction': config.get('min_stop_fraction', .02)}
    indicators = compute_indicators(bars, settings)
    current, prior = bars[-1], bars[-2]
    fast, previous_fast = indicators['ema_fast'][-1], indicators['ema_fast'][-2]
    mid, previous_mid = indicators['ema_mid'][-1], indicators['ema_mid'][-2]
    atr, previous_atr = indicators['atr'][-1], indicators['atr'][-2]
    if any(value is None or not math.isfinite(value) or value <= 0
           for value in (fast, previous_fast, mid, previous_mid, atr, previous_atr)):
        return reject('positive_finite_indicators_required')
    details.update(boundary_ms=current.time_ms + HOUR, signal_bar_time_ms=current.time_ms,
                   signal_ema_fast=fast, prior_ema_fast=previous_fast,
                   signal_ema_mid=mid, prior_ema_mid=previous_mid,
                   prior_high=prior.high, prior_low=prior.low,
                   atr=atr, previous_atr=previous_atr,
                   min_stop_fraction=settings['min_stop_fraction'])
    diagnostics.update({key: details[key] for key in (
        'boundary_ms', 'signal_ema_fast', 'prior_high', 'prior_low')})
    change = prior.close - bars[-2 - policy['context_lookback_bars']].close
    details.update(context_close_change=change, context_move_atr=change / previous_atr)
    long_context = previous_fast < previous_mid and change <= -previous_atr
    short_context = previous_fast > previous_mid and change >= previous_atr
    side = (1 if long_context and prior.close <= previous_fast and current.close > fast
            and current.close > prior.high else
            -1 if short_context and prior.close >= previous_fast and current.close < fast
            and current.close < prior.low else 0)
    if not side:
        return reject('no_fresh_early_recovery')
    signal = Signal(side, atr, current.close, abs(current.close - prior.close) / atr, current.time_ms)
    raw = asdict(signal)
    diagnostics.update(signal_before_confirmation=raw, signal_after_confirmation=dict(raw))
    extension = side * (current.close - fast) / atr
    reclaim = max(fast, prior.high) if side == 1 else min(fast, prior.low)
    structure = (min(bar.low for bar in bars[-policy['structural_lookback_bars']:])
                 - policy['structural_buffer_atr'] * atr if side == 1 else
                 max(bar.high for bar in bars[-policy['structural_lookback_bars']:])
                 + policy['structural_buffer_atr'] * atr)
    stop = initial_stop(current.close, side, atr, settings)
    reasons = []
    if atr / current.close > policy['max_atr_fraction']:
        reasons.append('atr_fraction_exceeds_limit')
    if extension > policy['maximum_reclaim_extension_atr']:
        reasons.append('reclaim_extension_exceeds_one_atr')
    if stop is None:
        reasons.append('stop_distance_exceeds_limit')
    elif side * (structure - stop) < 0:
        reasons.append('structure_outside_existing_stop')
    details.update(direction=side, reclaim_level=reclaim, reclaim_extension_atr=extension,
                   structural_invalidation_level=structure, projected_initial_stop=stop,
                   stop_risk_fraction=abs(current.close - stop) / current.close if stop else None,
                   reasons=list(reasons))
    diagnostics['reclaim_level'] = reclaim
    diagnostics['entry_quality'] = {'allowed': not reasons, 'reasons': list(reasons)}
    diagnostics['admission'].update(stage='quality_rejected' if reasons else 'admitted', reasons=list(reasons))
    if reasons:
        return None, diagnostics
    diagnostics['signal_after_quality'] = dict(raw)
    return signal, diagnostics


def candidate_signal(rows, config=None, *, asof_ms=None):
    return candidate_with_diagnostics(rows, config, asof_ms=asof_ms)[0]
