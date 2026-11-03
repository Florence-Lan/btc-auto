"""Independent completed-period confirmation without account or network effects."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import main_stock_entry_trend as entry
import main_stock_trend as main
import stock_entry_quality as quality
import stock_swing_signals as signals
from test_main_stock_trend import config as previous_config
from test_main_stock_trend import bars_from_prices, mirror, price_led_short_bars, smooth_bars

HOUR = entry.HOUR


def config():
    return {**previous_config(), entry.POLICY_KEY: deepcopy(entry.POLICY)}


def test_absent_policy_delegates_without_changing_legacy_decisions():
    bars, cfg = price_led_short_bars(), previous_config()
    ind = signals.compute_indicators(bars, cfg)
    assert entry.signal_with_diagnostics(bars, ind, 499, cfg) == main.signal_with_diagnostics(bars, ind, 499, cfg)
    assert entry.trend_diagnostics(bars, cfg, 500 * HOUR) == main.trend_diagnostics(bars, cfg, 500 * HOUR)


@pytest.mark.parametrize('change', ['mode', 'periods', 'combination', 'bool_period', 'float_warmup',
                                    'spread_bool', 'extra', 'missing_main'])
def test_policy_is_an_explicit_exact_main_only_overlay(change):
    cfg = config()
    if change == 'mode': cfg[entry.POLICY_KEY]['mode'] = 'other'
    elif change == 'periods': cfg[entry.POLICY_KEY]['confirmation_timeframes'] = ['2h']
    elif change == 'combination': cfg[entry.POLICY_KEY]['combination'] = 'all'
    elif change == 'bool_period': cfg[entry.POLICY_KEY]['ema_fast'] = True
    elif change == 'float_warmup': cfg[entry.POLICY_KEY]['warmup_bars'] = 60.
    elif change == 'spread_bool': cfg[entry.POLICY_KEY]['minimum_ema_spread_fraction'] = True
    elif change == 'extra': cfg[entry.POLICY_KEY]['extra'] = True
    else: del cfg[main.POLICY_KEY]
    with pytest.raises(ValueError): entry.validate(cfg)


def test_one_hour_can_admit_while_other_periods_are_still_warming_up():
    bars = bars_from_prices([100 + .1 * i for i in range(119)])
    before = deepcopy(bars)
    out = entry.evaluate_confirmation(bars, 118, config(), 1)
    assert out['allowed'] and out['admitted_by_timeframes'] == ['1h']
    assert out['channels']['1h']['admitted_by'] == 'strict'
    assert out['channels']['2h']['status'] == out['channels']['4h']['status'] == 'warming_up'
    assert out['channels']['2h']['observed_bars'] == 59
    assert bars == before


@pytest.mark.parametrize('direction', [-1, 1])
def test_actual_hourly_raw_signal_can_pass_before_four_hour_warmup(direction):
    bars = bars_from_prices([100 + .1 * i for i in range(119)])
    if direction == -1: bars = mirror(bars)
    cfg = config()
    indicators = signals.compute_indicators(bars, cfg)
    original_signal, original = main.signal_with_diagnostics(bars, indicators, 118, cfg)
    assert original_signal is None and original['signal_before_confirmation']['direction'] == direction
    assert original['admission']['stage'] == 'confirmation_rejected'
    actual, diagnostic = entry.signal_with_diagnostics(bars, indicators, 118, cfg)
    assert asdict(actual) == original['signal_before_confirmation']
    assert diagnostic['entry_trend_confirmation']['admitted_by_timeframes'] == ['1h']
    assert diagnostic['admission']['stage'] == 'admitted'
    assert diagnostic['signal_after_quality'] == asdict(actual)


def test_two_hour_uses_complete_utc_pairs_and_no_future_tail():
    bars, cfg = smooth_bars(), config()
    actual = entry.evaluate_confirmation(bars, 499, cfg, 1)
    two = actual['channels']['2h']
    assert two['observed_bars'] == 250 and two['closed_at_ms'] == 500 * HOUR
    future = [signals.Candle((500+i) * HOUR, 200, 201, 199, 200, 10000) for i in range(3)]
    assert entry.evaluate_confirmation(bars + future, 499, cfg, 1) == actual
    # The next hourly bar does not complete the next UTC two-hour pair.
    next_hour = entry.evaluate_confirmation(bars + future[:1], 500, cfg, 1)['channels']['2h']
    for key in ('observed_bars', 'closed_at_ms', 'ema_fast', 'ema_mid', 'ema_slow', 'close', 'allowed'):
        assert next_hour[key] == two[key]


def test_incomplete_leading_two_hour_bucket_is_not_a_synthetic_candle():
    bars = smooth_bars()[1:]
    out = entry.evaluate_confirmation(bars, len(bars)-1, config(), 1)
    assert out['channels']['2h']['observed_bars'] == 249
    assert out['channels']['2h']['closed_at_ms'] == 500 * HOUR


@pytest.mark.parametrize('direction', [-1, 1])
def test_strict_context_cannot_bypass_the_spread_floor(monkeypatch, direction):
    def narrow(bars, cfg, interval):
        return {'status': 'healthy', 'observed_bars': len(bars),
            'closed_at_ms': bars[-1].time_ms+interval, 'close': 100.,
            'ema_fast': 100.1, 'ema_fast_previous': 100., 'ema_mid': 100.05,
            'ema_slow': 100., 'ema_spread_fraction': .001,
            'strict_direction': direction, 'price_led_direction': 0, 'direction': direction}
    monkeypatch.setattr(main, '_ema_context', narrow)
    out = entry.evaluate_confirmation(smooth_bars(), 499, config(), direction)
    assert not out['allowed'] and out['admitted_by_timeframes'] == []
    assert all(not value['strict_allowed'] for value in out['channels'].values())
    assert all('spread_too_small' in value['reasons'][0] for value in out['channels'].values())


@pytest.mark.parametrize('direction', [-1, 1])
def test_symmetric_price_led_path_keeps_original_four_hour_option(direction):
    bars = price_led_short_bars()
    if direction == 1: bars = mirror(bars)
    out = entry.evaluate_confirmation(bars, 499, config(), direction)
    four = out['channels']['4h']
    assert out['allowed'] and '4h' in out['admitted_by_timeframes']
    assert four['admitted_by'] == 'price_led' and four['price_led_allowed']
    assert not four['strict_allowed'] and four['ema_spread_fraction'] >= .004


@pytest.mark.parametrize('fault', ['gap', 'alignment', 'nan_volume', 'invalid_ohlc', 'empty'])
def test_invalid_source_prefix_cannot_be_waived_by_other_periods(fault):
    bars = smooth_bars()
    if fault == 'gap': del bars[50]
    elif fault == 'alignment': bars[50] = replace(bars[50], time_ms=bars[50].time_ms+1)
    elif fault == 'nan_volume': bars[50] = SimpleNamespace(**{**vars(bars[50]), 'volume': float('nan')})
    elif fault == 'invalid_ohlc': bars[50] = SimpleNamespace(**{**vars(bars[50]), 'high': .1})
    else: bars = []
    out = entry.evaluate_confirmation(bars, len(bars)-1, config(), 1)
    assert not out['allowed'] and out['status'] == 'unavailable'
    assert all(not channel['allowed'] for channel in out['channels'].values())


@pytest.mark.parametrize('direction', [True, False, 0, 1.0, '1'])
def test_entry_side_requires_an_actual_long_or_short_integer(direction):
    with pytest.raises(ValueError): entry.evaluate_confirmation(smooth_bars(), 499, config(), direction)


def test_current_diagnostics_add_two_hour_context_without_an_entry_permit():
    out = entry.trend_diagnostics(smooth_bars(), config(), 500 * HOUR + 30000)
    assert out['status'] == 'healthy'
    assert out['hourly']['closed_at_ms'] == out['two_hour']['closed_at_ms'] == 500 * HOUR
    assert out['four_hour']['closed_at_ms'] == out['confirmation_close_ms'] == 500 * HOUR
    assert out['entry_confirmation']['decision_scope'] == 'current_context_not_entry_permission'
    assert out['entry_confirmation']['direction'] is None


@pytest.mark.parametrize('timestamp', [500 * HOUR - 1, 501 * HOUR, None])
def test_current_diagnostics_reject_unfinished_stale_or_absent_clock(timestamp):
    out = entry.trend_diagnostics(smooth_bars(), config(), timestamp)
    assert out['status'] == 'unavailable' and out['reasons']


def test_background_confirmation_cannot_create_an_original_raw_signal():
    bars, cfg = smooth_bars(), config()
    ind = signals.compute_indicators(bars, cfg)
    # Reuse a valid complete history but explicitly remove the raw candidate.
    ind['atr'][-1] = None
    signal, diag = entry.signal_with_diagnostics(bars, ind, 499, cfg)
    assert signal is None and diag['signal_before_confirmation'] is None
    assert diag['admission']['stage'] == 'no_raw_signal'
    assert not diag['admission']['confirmation_rejected_candidate']


def test_confirmation_rescue_rechecks_quality_and_preserves_old_forensic_result(monkeypatch):
    bars, cfg = smooth_bars(), config()
    raw = asdict(signals.Signal(1, 1., bars[-1].close, 1., bars[-1].time_ms))
    original = {'signal_before_confirmation': raw, 'signal_after_confirmation': None,
        'signal_after_quality': None, 'entry_quality': None,
        'swing_confirmation': {'enabled': True, 'allowed': False, 'reasons': ['old_4h_rejected']},
        'admission': {'stage': 'confirmation_rejected', 'reasons': ['old_4h_rejected'],
                      'confirmation_rejected_candidate': True}}
    before = deepcopy(original)
    monkeypatch.setattr(main, 'signal_with_diagnostics', lambda *_a, **_k: (None, original))
    monkeypatch.setattr(quality, 'evaluate', lambda *_a, **_k: {
        'enabled': True, 'allowed': False, 'reasons': ['quality_still_required']})
    signal, diag = entry.signal_with_diagnostics(bars, {}, 499, cfg)
    assert signal is None and diag['swing_confirmation']['allowed']
    assert diag['signal_after_confirmation'] == raw and diag['signal_after_quality'] is None
    assert diag['admission']['stage'] == 'quality_rejected'
    assert diag['previous_swing_confirmation'] == original['swing_confirmation']
    assert original == before


def test_previous_quality_rejection_is_never_retried_into_an_entry(monkeypatch):
    bars, cfg = smooth_bars(), config()
    raw = asdict(signals.Signal(1, 1., bars[-1].close, 1., bars[-1].time_ms))
    original = {'signal_before_confirmation': raw, 'signal_after_confirmation': dict(raw),
        'signal_after_quality': None, 'entry_quality': {'allowed': False},
        'swing_confirmation': {'enabled': True, 'allowed': True, 'reasons': []},
        'admission': {'stage': 'quality_rejected', 'reasons': ['original_quality'],
                      'confirmation_rejected_candidate': False}}
    monkeypatch.setattr(main, 'signal_with_diagnostics', lambda *_a, **_k: (None, original))
    monkeypatch.setattr(quality, 'evaluate', lambda *_a, **_k: pytest.fail('Do not retry a quality rejection'))
    signal, diag = entry.signal_with_diagnostics(bars, {}, 499, cfg)
    assert signal is None and diag['admission'] == original['admission']
    assert diag['swing_confirmation']['allowed']
