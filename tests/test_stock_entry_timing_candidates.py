"""Causal and symmetric admission tests for the new, isolated entry trial."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_entry_timing_candidates as candidate
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
import stock_swing_signals as signals

HOUR = confirmation.HOUR


def config(variant='turning'):
    return {'signal_timeframe': '1h', 'signal_family': 'donchian20',
            'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
            'ema_slope_bars': 6, 'breakout_bars': 20, 'atr_period': 14,
            'max_atr_fraction': .03, 'max_breakout_extension_atr': 1.0,
            'require_price_slow': True,
            'swing_confirmation_policy': deepcopy(confirmation.POLICY),
            'entry_timing_variant': variant,
            'entry_timing_candidate_policy': deepcopy(candidate.POLICY),
            'taker_fee_rate_assumption': .000125,
            'adverse_slippage_fraction_assumption': .0002}


def turning_bars(side=1, rebound=4):
    # A declared synthetic reversal: fast turns before mid overtakes slow.
    closes = [100.] * 70 + [80.] * 40 + [100.] * rebound
    return [signals.Candle(i * HOUR, price, price + .1, price - .1, price, 1000)
            for i, value in enumerate(value for value in closes for _ in range(4))
            for price in [100 + side * (value - 100)]]


def smooth_bars(side=1, count=504):
    return [signals.Candle(i * HOUR, p, p + .005, p - .005, p, 1000)
            for i in range(count) for p in [100 + side * i * .02]]


def pullback_case(side=1):
    bars = turning_bars(side)
    # Broad preceding extrema exclude Donchian while retaining a fast reclaim.
    for i in range(len(bars) - 20, len(bars)):
        p = bars[i].close
        bars[i] = signals.Candle(i * HOUR, p, p + 2, p - 2, p, 1000)
    p = 100 + .5 * side
    bars[-1] = signals.Candle(bars[-1].time_ms, p, p + 2, p - 2, p, 1000)
    size = len(bars)
    ind = {'ema_fast': [100.] * size, 'ema_mid': [100. - side] * size,
           'ema_slow': [100. - 2 * side] * size, 'atr': [1.] * size}
    ind['ema_mid'][-7] = 100. - 2 * side
    return bars, ind


@pytest.mark.parametrize('side', [1, -1])
def test_baseline_exactly_delegates_original_signal_and_diagnostics(side):
    cfg = config('baseline')
    cfg.pop('entry_timing_candidate_policy')
    bars = smooth_bars(side)
    indicators = signals.compute_indicators(bars, cfg)
    expected = profiles.signal_with_diagnostics(bars, indicators, 499, cfg)
    assert expected[0] is not None
    assert candidate.signal_with_diagnostics(bars, indicators, 499, cfg) == expected
    assert candidate.signal_at(bars, indicators, 499, cfg) == expected[0]


@pytest.mark.parametrize('side', [1, -1])
def test_turning_admits_fast_led_context_before_mid_slow_cross(side):
    bars = turning_bars(side)
    cfg = config()
    old = confirmation.evaluate(bars, len(bars) - 1, cfg, side)
    assert not old['allowed'] and 'swing_4h_ema_ordering' in old['reasons']
    out = candidate.evaluate_turning_confirmation(bars, len(bars) - 1, cfg, side)
    assert out['allowed'] and not out['strict_allowed'] and out['turning_allowed']
    assert out['admitted_by'] == 'turning'
    assert out['mid_slope_lookback_bars'] == 3
    assert side * (out['mid_now'] - out['mid_3_bars_ago']) > 0
    assert side * (out['ema_mid'] - out['ema_slow']) < 0


@pytest.mark.parametrize('side', [1, -1])
def test_original_strict_context_is_preserved_without_new_mid_slope_gate(monkeypatch, side):
    bars = smooth_bars(side)
    cfg = config()
    original = signals.compute_indicators
    def negative_slope(rows, settings):
        out = original(rows, settings)
        if settings is confirmation.POLICY:
            out['ema_mid'][-4] = out['ema_mid'][-1] + side
        return out
    monkeypatch.setattr(signals, 'compute_indicators', negative_slope)
    out = candidate.evaluate_turning_confirmation(bars, 499, cfg, side)
    assert out['strict_allowed'] and out['allowed']
    assert not out['turning_allowed']
    assert out['admitted_by'] == 'strict'


@pytest.mark.parametrize('side', [1, -1])
def test_opposite_side_or_insufficient_fast_turn_does_not_pass(side):
    cfg = config()
    bars = turning_bars(side)
    assert not candidate.evaluate_turning_confirmation(bars, len(bars) - 1, cfg, -side)['allowed']
    early = turning_bars(side, rebound=1)
    assert not candidate.evaluate_turning_confirmation(early, len(early) - 1, cfg, side)['allowed']


@pytest.mark.parametrize('side', [1, -1])
def test_turning_requires_positive_directional_mid_change(monkeypatch, side):
    bars = turning_bars(side)
    cfg = config()
    original = signals.compute_indicators
    def flat_mid(rows, settings):
        out = original(rows, settings)
        if settings is confirmation.POLICY:
            out['ema_mid'][-4] = out['ema_mid'][-1]
        return out
    monkeypatch.setattr(signals, 'compute_indicators', flat_mid)
    out = candidate.evaluate_turning_confirmation(bars, len(bars) - 1, cfg, side)
    assert not out['allowed']
    assert 'turning_4h_mid_slope' in out['reasons']


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('rejection', ['price_slow', 'spread'])
def test_turning_retains_original_price_slow_and_spread_floor(monkeypatch, side, rejection):
    bars = turning_bars(side)
    cfg = config()
    original = confirmation.evaluate
    def rejected(*args, **kwargs):
        out = original(*args, **kwargs)
        assert not out['allowed']
        if rejection == 'price_slow':
            out['close'] = out['ema_slow']
        else:
            out['ema_spread_fraction'] = .003999
        return out
    monkeypatch.setattr(confirmation, 'evaluate', rejected)
    out = candidate.evaluate_turning_confirmation(bars, len(bars) - 1, cfg, side)
    assert not out['allowed']
    assert 'turning_4h_' + rejection in out['reasons']


@pytest.mark.parametrize('failure', ['warmup', 'gap', 'alignment'])
def test_data_failures_cannot_be_waived_by_turning(failure):
    cfg = config()
    bars = turning_bars()
    if failure == 'warmup':
        bars = bars[:239]
    elif failure == 'gap':
        del bars[80]
    else:
        b = bars[80]
        bars[80] = signals.Candle(b.time_ms + 1, b.open, b.high, b.low, b.close, b.volume)
    out = candidate.evaluate_turning_confirmation(bars, len(bars) - 1, cfg, 1)
    assert not out['allowed'] and not out['turning_allowed']
    assert out['reasons'][0].startswith('swing_')


def test_incomplete_and_future_four_hour_bars_cannot_change_context():
    cfg = config()
    prefix = turning_bars()
    expected = candidate.evaluate_turning_confirmation(prefix, len(prefix) - 1, cfg, 1)
    assert expected['allowed']
    future = [signals.Candle((len(prefix) + i) * HOUR, 10, 11, 9, 10, 1000) for i in range(8)]
    assert candidate.evaluate_turning_confirmation(prefix + future, len(prefix) - 1, cfg, 1) == expected
    incomplete = candidate.evaluate_turning_confirmation(prefix + future, len(prefix) + 2, cfg, 1)
    for key in ('confirmation_close_ms', 'ema_fast', 'ema_mid', 'ema_slow', 'mid_now', 'mid_3_bars_ago'):
        assert incomplete[key] == expected[key]


@pytest.mark.parametrize('side', [1, -1])
def test_turning_preserves_raw_donchian_without_adding_hourly_ema_ordering(side):
    cfg = config()
    bars = smooth_bars(side)
    ind = signals.compute_indicators(bars, cfg)
    # Donchian uses only the slow-price gate, not hourly fast/mid ordering.
    ind['ema_fast'][499] = ind['ema_mid'][499] - side
    raw = profiles.raw_signal_at(bars, ind, 499, cfg)
    signal, diagnostics = candidate.signal_with_diagnostics(bars, ind, 499, cfg)
    assert raw and signal == raw
    assert diagnostics['entry_timing_candidate']['signal_kind'] == 'donchian20'
    assert diagnostics['signal_before_confirmation'] == asdict(raw)


@pytest.mark.parametrize('side', [1, -1])
def test_computed_hourly_breakout_uses_turning_instead_of_failed_strict_context(side):
    bars = turning_bars(side)
    p = 100 + .2 * side
    bars[-1] = signals.Candle(bars[-1].time_ms, p, p + .1, p - .1, p, 1000)
    cfg = config()
    ind = signals.compute_indicators(bars, cfg)
    raw = profiles.raw_signal_at(bars, ind, len(bars) - 1, cfg)
    old, old_detail = profiles.signal_with_diagnostics(bars, ind, len(bars) - 1, cfg)
    new, detail = candidate.signal_with_diagnostics(bars, ind, len(bars) - 1, cfg)
    assert raw and raw.direction == side
    assert old is None and old_detail['admission']['confirmation_rejected_candidate']
    assert new == raw and detail['swing_confirmation']['admitted_by'] == 'turning'
    assert detail['signal_after_quality'] == asdict(raw)


def test_allowed_context_without_raw_is_not_counted_as_a_filtered_candidate():
    bars = turning_bars()
    cfg = config()
    signal, detail = candidate.signal_with_diagnostics(
        bars, signals.compute_indicators(bars, cfg), len(bars) - 1, cfg)
    assert signal is None and detail['swing_confirmation']['allowed']
    assert detail['signal_before_confirmation'] is None
    assert not detail['admission']['confirmation_rejected_candidate']
    assert detail['admission']['stage'] == 'no_raw_signal'


def test_original_spread_floor_is_inclusive(monkeypatch):
    bars = turning_bars()
    original = confirmation.evaluate
    def exact_floor(*args, **kwargs):
        out = original(*args, **kwargs)
        out['ema_spread_fraction'] = .004
        return out
    monkeypatch.setattr(confirmation, 'evaluate', exact_floor)
    assert candidate.evaluate_turning_confirmation(bars, len(bars) - 1, config(), 1)['allowed']


@pytest.mark.parametrize('side', [1, -1])
def test_pullback_is_extra_signal_with_original_reclaim_and_frozen_ema8(side):
    bars, ind = pullback_case(side)
    i = len(bars) - 1
    cfg = config('turn_pullback')
    assert profiles.raw_signal_at(bars, ind, i, cfg) is None
    assert candidate.signal_at(bars, ind, i, config('turning')) is None
    expected = profiles.regime_signal_at(bars, ind, i, {**cfg, 'signal_family': 'trend_pullback'})
    result, detail = candidate.signal_with_diagnostics(bars, ind, i, cfg)
    assert expected and result == expected and result.direction == side
    frozen = detail['entry_timing_candidate']['frozen_reclaim']
    assert frozen == {'family': 'trend_pullback', 'direction': side, 'ema_fast': 100.,
                      'signal_time_ms': bars[-1].time_ms, 'boundary_ms': bars[-1].time_ms + HOUR,
                      'signal_close': bars[-1].close, 'signal_atr': 1.}
    assert detail['signal_after_quality'] == asdict(result)
    ind['ema_fast'][i] = 123.
    assert frozen['ema_fast'] == 100.  # Provenance is detached, not a rolling reference.


@pytest.mark.parametrize('side', [1, -1])
def test_existing_one_atr_reclaim_extension_boundary_is_inclusive(side):
    bars, ind = pullback_case(side)
    p = 100 + side
    bars[-1] = signals.Candle(bars[-1].time_ms, p, p + 2, p - 2, p, 1000)
    signal, detail = candidate.signal_with_diagnostics(bars, ind, len(bars) - 1, config('turn_pullback'))
    assert signal and detail['entry_timing_candidate']['signal_kind'] == 'trend_pullback'


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('failure', ['prior_already_reclaimed', 'wrong_order', 'mid_not_rising', 'overextended'])
def test_pullback_keeps_existing_regime_requirements(side, failure):
    bars, ind = pullback_case(side)
    i = len(bars) - 1
    if failure == 'prior_already_reclaimed':
        p = 100 + .2 * side
        bars[-2] = signals.Candle(bars[-2].time_ms, p, p + 2, p - 2, p, 1000)
    elif failure == 'wrong_order':
        ind['ema_mid'][i] = 100 + side
    elif failure == 'mid_not_rising':
        ind['ema_mid'][i - 6] = ind['ema_mid'][i]
    else:
        p = 100 + 1.00001 * side
        bars[-1] = signals.Candle(bars[-1].time_ms, p, p + 2, p - 2, p, 1000)
    result, detail = candidate.signal_with_diagnostics(bars, ind, i, config('turn_pullback'))
    assert result is None
    assert detail['signal_before_confirmation'] is None
    assert not detail['admission']['confirmation_rejected_candidate']


def test_donchian_has_priority_and_no_fallback_after_filter_rejection(monkeypatch):
    bars = turning_bars()
    cfg = config('turn_pullback')
    ind = signals.compute_indicators(bars, cfg)
    raw = signals.Signal(-1, 1, bars[-1].close, 1, bars[-1].time_ms)
    monkeypatch.setattr(profiles, 'raw_signal_at', lambda *args: raw)
    def forbidden(*args):
        pytest.fail('Pullback must not bypass a rejected Donchian candidate')
    monkeypatch.setattr(profiles, 'regime_signal_at', forbidden)
    result, detail = candidate.signal_with_diagnostics(bars, ind, len(bars) - 1, cfg)
    assert result is None and detail['admission']['confirmation_rejected_candidate']
    assert detail['entry_timing_candidate']['signal_kind'] == 'donchian20'


def test_quality_rejection_is_preserved_after_turning_admission(monkeypatch):
    bars = smooth_bars()
    cfg = config('turn_pullback')
    ind = signals.compute_indicators(bars, cfg)
    monkeypatch.setattr(candidate.quality, 'evaluate',
                        lambda *args: {'allowed': False, 'reasons': ['quality_test_rejection']})
    result, detail = candidate.signal_with_diagnostics(bars, ind, 499, cfg)
    assert result is None
    assert detail['signal_after_confirmation'] is not None
    assert detail['signal_after_quality'] is None
    assert detail['admission']['stage'] == 'quality_rejected'


def test_signal_activity_cannot_be_bypassed_by_pullback():
    bars, ind = pullback_case()
    bars = [signals.Candle(b.time_ms, b.open, b.high, b.low, b.close, 0) for b in bars]
    cfg = {**config('turn_pullback'), 'signal_activity_lookback_bars': 12,
           'signal_min_active_fraction': .5}
    result, detail = candidate.signal_with_diagnostics(bars, ind, len(bars) - 1, cfg)
    assert result is None and detail['signal_before_confirmation'] is None


@pytest.mark.parametrize('change', [
    {'entry_timing_variant': 'other'}, {'entry_timing_variant': True},
    {'entry_timing_candidate_policy': None}, {'signal_timeframe': '15m'},
    {'ema_fast': 9}, {'ema_fast': 8.0}, {'require_price_slow': False},
    {'signal_family': 'trend_pullback'}, {'max_atr_fraction': .04},
    {'swing_confirmation_policy': None},
])
def test_candidate_configuration_fails_closed_on_undeclared_changes(change):
    with pytest.raises(ValueError):
        candidate.validate(config() | change)


@pytest.mark.parametrize('key,value', [('mid_slope_lookback_bars', 4),
                                      ('minimum_ema_spread_fraction', .003),
                                      ('donchian_priority', 1)])
def test_fixed_policy_cannot_be_tuned_or_bool_coerced(key, value):
    cfg = config()
    cfg['entry_timing_candidate_policy'][key] = value
    with pytest.raises(ValueError):
        candidate.validate(cfg)
