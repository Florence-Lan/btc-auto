"""Synthetic, causal short-confirmation cases; no inspected market replay."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_short_confirmation as candidate
import stock_entry_timing as timing
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
import stock_swing_signals as signals

HOUR = confirmation.HOUR


def config(enabled=True):
    value = {'signal_family': 'donchian20', 'signal_timeframe': '1h',
        'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
        'ema_slope_bars': 6, 'breakout_bars': 20, 'atr_period': 14,
        'max_atr_fraction': .03, 'max_breakout_extension_atr': 1.0,
        'require_price_slow': True, 'entry_gap_atr': .5,
        'swing_confirmation_policy': deepcopy(confirmation.POLICY),
        'entry_timing_policy': deepcopy(timing.POLICY)}
    if enabled:
        value[candidate.POLICY_KEY] = deepcopy(candidate.POLICY)
    return value


def price_led_short_bars():
    """Exported deterministic fixture for real-book runner integration tests.

    A synthetic rising series ends in a modest fresh Donchian low after a
    pullback. Hourly slow-price/ATR/extension pass; strict4h stays bullish.
    This fixture is not constructed from any of the six reviewed live signals.
    """
    prices = [100 + .1 * i for i in range(500)]
    prices[-4:] = [147., 142., 140., 139.9]
    bars, previous = [], prices[0]
    for i, close in enumerate(prices):
        bars.append(signals.Candle(i * HOUR, previous,
            max(previous, close) + .05, min(previous, close) - .05, close, 10000.))
        previous = close
    return bars


def smooth_bars(direction=1):
    return [signals.Candle(i * HOUR, price, price + .005, price - .005, price, 10000.)
            for i in range(500) for price in [100 + direction * .02 * i]]


def mirror(bars):
    return [signals.Candle(b.time_ms, 300 - b.open, 300 - b.low,
                           300 - b.high, 300 - b.close, b.volume) for b in bars]


def evaluate(bars, cfg=None, index=None):
    cfg = config() if cfg is None else cfg
    index = len(bars) - 1 if index is None else index
    return candidate.signal_with_diagnostics(bars, signals.compute_indicators(bars, cfg), index, cfg)


def test_price_led_short_uses_original_raw_and_does_not_require_full_bearish_ordering():
    bars, cfg = price_led_short_bars(), config()
    indicators = signals.compute_indicators(bars, cfg)
    original, old = profiles.signal_with_diagnostics(bars, indicators, 499, cfg)
    assert original is None and old['signal_before_confirmation']['direction'] == -1
    assert not old['swing_confirmation']['allowed']
    signal, detail = candidate.signal_with_diagnostics(bars, indicators, 499, cfg)
    assert signal is not None and signal.direction == -1
    assert asdict(signal) == old['signal_before_confirmation']
    outcome = detail['short_confirmation']
    assert outcome['allowed'] and outcome['price_led_allowed'] and not outcome['strict_allowed']
    assert outcome['admitted_by'] == 'price_led_short'
    assert outcome['close'] > outcome['ema_slow']
    assert outcome['ema_fast'] > outcome['ema_mid'] > outcome['ema_slow']
    assert outcome['close'] < outcome['ema_fast'] and outcome['close'] < outcome['ema_mid']
    assert outcome['ema_fast'] < outcome['ema_fast_previous']
    assert outcome['ema_spread_fraction'] >= .004
    assert detail['swing_confirmation']['allowed']
    assert detail['swing_confirmation']['strict_direction'] == 1
    assert detail['signal_after_confirmation'] == detail['signal_after_quality'] == asdict(signal)
    assert detail['admission'] == {'stage': 'admitted', 'reasons': [],
                                   'confirmation_rejected_candidate': False}


@pytest.mark.parametrize('bars', [smooth_bars(1), mirror(price_led_short_bars())])
def test_long_signal_and_every_original_diagnostic_are_unchanged(bars):
    cfg = config()
    indicators = signals.compute_indicators(bars, cfg)
    expected = profiles.signal_with_diagnostics(bars, indicators, 499, cfg)
    assert expected[1]['signal_before_confirmation']['direction'] == 1
    assert candidate.signal_with_diagnostics(bars, indicators, 499, cfg) == expected


@pytest.mark.parametrize('bars', [smooth_bars(1), smooth_bars(-1), price_led_short_bars()])
def test_without_policy_directly_delegates_baseline(bars):
    cfg = config(False)
    ind = signals.compute_indicators(bars, cfg)
    assert candidate.signal_with_diagnostics(bars, ind, 499, cfg) == profiles.signal_with_diagnostics(bars, ind, 499, cfg)
    assert candidate.signal_at(bars, ind, 499, cfg) == profiles.signal_at(bars, ind, 499, cfg)


def test_strict_short_remains_original_and_does_not_need_new_branch(monkeypatch):
    bars, cfg = smooth_bars(-1), config()
    indicators = signals.compute_indicators(bars, cfg)
    expected = profiles.signal_with_diagnostics(bars, indicators, 499, cfg)
    assert expected[0] is not None and expected[0].direction == -1
    monkeypatch.setattr(candidate, 'evaluate_short_confirmation',
        lambda *_a, **_k: pytest.fail('Strict short must not acquire a new restriction'))
    assert candidate.signal_with_diagnostics(bars, indicators, 499, cfg) == expected


def test_price_led_context_without_raw_cannot_create_signal():
    bars = price_led_short_bars()
    bars[-1] = replace(bars[-1], close=140.02)
    cfg = config()
    assert candidate.evaluate_short_confirmation(bars, 499, cfg)['allowed']
    signal, detail = evaluate(bars, cfg)
    assert signal is None and detail['signal_before_confirmation'] is None
    assert detail['admission']['stage'] == 'no_raw_signal'
    assert not detail['admission']['confirmation_rejected_candidate']


@pytest.mark.parametrize('gate', ['atr_missing', 'atr_negative', 'atr_nan', 'atr_too_large', 'extension', 'hourly_slow'])
def test_original_hourly_admission_cannot_be_waived(gate):
    bars, cfg = price_led_short_bars(), config()
    ind = signals.compute_indicators(bars, cfg)
    if gate == 'hourly_slow': ind['ema_slow'][-1] = bars[-1].close - 1
    else: ind['atr'][-1] = {'atr_missing': None, 'atr_negative': -1,
        'atr_nan': float('nan'), 'atr_too_large': bars[-1].close * .031,
        'extension': .001}[gate]
    signal, detail = candidate.signal_with_diagnostics(bars, ind, 499, cfg)
    assert signal is None and detail['signal_before_confirmation'] is None


def test_new_short_still_has_original_quality_gate(monkeypatch):
    monkeypatch.setattr(candidate.quality, 'evaluate', lambda *_a, **_k: {
        'enabled': True, 'allowed': False, 'reasons': ['fixture_quality_rejected']})
    signal, detail = evaluate(price_led_short_bars())
    assert signal is None and detail['short_confirmation']['allowed']
    assert detail['admission']['stage'] == 'quality_rejected'
    assert detail['admission']['reasons'] == ['fixture_quality_rejected']
    assert detail['signal_after_confirmation'] and detail['signal_after_quality'] is None


def test_original_strict_quality_rejection_is_not_resurrected(monkeypatch):
    monkeypatch.setattr(candidate.quality, 'evaluate', lambda *_a, **_k: {
        'enabled': True, 'allowed': False, 'reasons': ['fixture_quality_rejected']})
    bars, cfg = smooth_bars(-1), config()
    ind = signals.compute_indicators(bars, cfg)
    expected = profiles.signal_with_diagnostics(bars, ind, 499, cfg)
    assert expected[0] is None and expected[1]['admission']['stage'] == 'quality_rejected'
    assert candidate.signal_with_diagnostics(bars, ind, 499, cfg) == expected


def test_future_complete_bars_cannot_change_earlier_signal_or_diagnostics():
    bars, cfg = price_led_short_bars(), config()
    expected = evaluate(bars, cfg)
    future = [signals.Candle((500 + i) * HOUR, 300., 301., 299., 300., 10000.) for i in range(8)]
    assert evaluate(bars + future, cfg, 499) == expected


def test_three_hours_of_unfinished_four_hour_group_do_not_enter_confirmation():
    bars, cfg = price_led_short_bars(), config()
    expected = candidate.evaluate_short_confirmation(bars, 499, cfg)
    future = [signals.Candle((500 + i) * HOUR, 300., 301., 299., 300., 10000.) for i in range(3)]
    actual = candidate.evaluate_short_confirmation(bars + future, 502, cfg)
    for key in ('confirmation_close_ms', 'ema_fast', 'ema_fast_previous', 'ema_mid',
                'ema_slow', 'close', 'checks', 'allowed'):
        assert actual[key] == expected[key]


@pytest.mark.parametrize('failure', ['warmup', 'gap', 'alignment', 'nan_volume', 'negative_volume', 'bad_ohlc'])
def test_unavailable_or_malformed_history_cannot_enable_new_branch(failure):
    bars, cfg = price_led_short_bars(), config()
    if failure == 'warmup': bars = bars[-239:]
    elif failure == 'gap': del bars[80]
    elif failure == 'alignment': bars[80] = replace(bars[80], time_ms=bars[80].time_ms + 1)
    elif failure == 'nan_volume': bars[80] = SimpleNamespace(**{**vars(bars[80]), 'volume': float('nan')})
    elif failure == 'negative_volume': bars[80] = SimpleNamespace(**{**vars(bars[80]), 'volume': -1})
    else: bars[80] = SimpleNamespace(**{**vars(bars[80]), 'low': bars[80].high + 1})
    outcome = candidate.evaluate_short_confirmation(bars, len(bars) - 1, cfg)
    assert not outcome['allowed'] and not outcome['price_led_allowed']


@pytest.mark.parametrize('failed', ['fast_equal', 'mid_equal', 'slope_equal', 'spread_below', 'spread_exact'])
def test_strict_price_and_slope_boundaries_and_unchanged_spread_floor(monkeypatch, failed):
    bars, cfg = price_led_short_bars(), config()
    bars[-1] = replace(bars[-1], open=100., high=100.1, low=99.9, close=100.)
    fast, previous, mid, slow = 110., 111., 115., 90.
    if failed == 'fast_equal': fast = 100.
    elif failed == 'mid_equal': mid = 100.
    elif failed == 'slope_equal': previous = fast
    elif failed == 'spread_below': fast, previous, mid, slow = 100.2999, 101., 100.5, 99.9
    else:
        bars[-1] = replace(bars[-1], open=1250., high=1250.1, low=1249.9, close=1250.)
        fast, previous, mid, slow = 1260., 1261., 1270., 1255.
    real = signals.compute_indicators
    def indicators(rows, settings):
        result = real(rows, settings)
        if settings is confirmation.POLICY:
            result['ema_fast'][-2:] = [previous, fast]
            result['ema_mid'][-1], result['ema_slow'][-1] = mid, slow
        return result
    monkeypatch.setattr(signals, 'compute_indicators', indicators)
    result = candidate.evaluate_short_confirmation(bars, 499, cfg)
    if failed == 'spread_exact':
        assert result['ema_spread_fraction'] == .004
        assert result['allowed'] and result['checks']['short_4h_spread']
        return
    assert not result['allowed']
    key = {'fast_equal': 'short_4h_close_below_fast', 'mid_equal': 'short_4h_close_below_mid',
           'slope_equal': 'short_4h_fast_falling', 'spread_below': 'short_4h_spread'}[failed]
    assert not result['checks'][key]


def test_signal_and_original_frozen_entry_trigger_stay_compatible():
    bars, cfg = price_led_short_bars(), config()
    signal, _ = evaluate(bars, cfg)
    trigger = timing.build_trigger(bars, signal, cfg, 500 * HOUR, observed_at_ms=500 * HOUR + 1)
    assert trigger['breakout_level'] == min(b.low for b in bars[-21:-1])
    assert trigger['signal_close'] == signal.close
    assert timing.evaluate(trigger, signal, cfg, reference_price=signal.close,
                           mark_price=signal.close, fill_price=signal.close)['allowed']
    pending = {'signal': asdict(signal), 'boundary_ms': 500 * HOUR}
    future = [replace(bars[-1], time_ms=500 * HOUR, open=100., high=101., low=99., close=100.)]
    assert timing.rebuild_trigger(bars + future, pending, cfg, observed_at_ms=501 * HOUR) == trigger


def test_inputs_and_original_modules_are_not_mutated():
    bars, cfg = price_led_short_bars(), config()
    ind = signals.compute_indicators(bars, cfg)
    before = deepcopy((bars, ind, cfg))
    original = profiles.signal_with_diagnostics(bars, ind, 499, cfg)
    function = profiles.signal_with_diagnostics
    assert candidate.signal_with_diagnostics(bars, ind, 499, cfg)[0]
    assert (bars, ind, cfg) == before
    assert profiles.signal_with_diagnostics is function
    assert profiles.signal_with_diagnostics(bars, ind, 499, cfg) == original


@pytest.mark.parametrize('mutation', ['missing_policy', 'extra_policy', 'wrong_zero_type',
    'wrong_slope', 'wrong_spread', 'missing_strict', 'wrong_hourly_slow', 'wrong_family'])
def test_candidate_definition_is_frozen(mutation):
    cfg = config()
    if mutation == 'missing_policy': cfg[candidate.POLICY_KEY] = None
    elif mutation == 'extra_policy': cfg[candidate.POLICY_KEY]['unapproved'] = True
    elif mutation == 'wrong_zero_type': cfg[candidate.POLICY_KEY]['ema_fast'] = 8.0
    elif mutation == 'wrong_slope': cfg[candidate.POLICY_KEY]['fast_slope_lookback_bars'] = 2
    elif mutation == 'wrong_spread': cfg[candidate.POLICY_KEY]['minimum_ema_spread_fraction'] = .003
    elif mutation == 'missing_strict': cfg.pop('swing_confirmation_policy')
    elif mutation == 'wrong_hourly_slow': cfg['require_price_slow'] = False
    else: cfg['signal_family'] = 'trend_pullback'
    with pytest.raises(ValueError): candidate.validate(cfg)
