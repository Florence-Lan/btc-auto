"""Synthetic main-account trend decisions, with no network or account mutation."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import main_stock_trend as main
import stock_entry_quality as quality
import stock_short_confirmation as short
import stock_swing_confirmation as confirmation
import stock_swing_signals as signals

HOUR = main.HOUR


def config():
    return {'signal_family': 'donchian20', 'signal_timeframe': '1h',
        'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
        'ema_slope_bars': 6, 'breakout_bars': 20, 'atr_period': 14,
        'max_atr_fraction': .03, 'max_breakout_extension_atr': 1.0,
        'require_price_slow': True, 'entry_gap_atr': .5,
        'swing_confirmation_policy': deepcopy(confirmation.POLICY),
        short.POLICY_KEY: deepcopy(short.POLICY), main.POLICY_KEY: deepcopy(main.POLICY)}


def bars_from_prices(prices):
    bars, previous = [], prices[0]
    for index, close in enumerate(prices):
        bars.append(signals.Candle(index * HOUR, previous,
            max(previous, close) + .05, min(previous, close) - .05, close, 10000.))
        previous = close
    return bars


def price_led_short_bars():
    prices = [100 + .1 * i for i in range(500)]
    prices[-4:] = [147., 142., 140., 139.9]
    return bars_from_prices(prices)


def mirror(bars):
    return [signals.Candle(b.time_ms, 300 - b.open, 300 - b.low,
        300 - b.high, 300 - b.close, b.volume) for b in bars]


def smooth_bars(direction=1):
    return bars_from_prices([100 + direction * .02 * i for i in range(500)])


def evaluate(bars, cfg=None, index=None):
    cfg = config() if cfg is None else cfg
    index = len(bars) - 1 if index is None else index
    return main.signal_with_diagnostics(bars, signals.compute_indicators(bars, cfg), index, cfg)


def test_price_led_long_is_symmetric_without_creating_a_raw_breakout():
    bars, cfg = mirror(price_led_short_bars()), config()
    ind = signals.compute_indicators(bars, cfg)
    old_signal, old = short.signal_with_diagnostics(bars, ind, 499, cfg)
    assert old_signal is None and old['signal_before_confirmation']['direction'] == 1
    signal, diag = evaluate(bars, cfg)
    assert asdict(signal) == old['signal_before_confirmation']
    outcome = diag['long_confirmation']
    assert outcome['allowed'] and outcome['price_led_allowed'] and not outcome['strict_allowed']
    assert outcome['admitted_by'] == 'price_led_long'
    assert outcome['close'] < outcome['ema_slow']
    assert outcome['close'] > outcome['ema_fast'] and outcome['close'] > outcome['ema_mid']
    assert outcome['ema_fast'] > outcome['ema_fast_previous']
    assert outcome['ema_spread_fraction'] >= .004
    assert diag['signal_after_confirmation'] == diag['signal_after_quality'] == asdict(signal)


@pytest.mark.parametrize('bars', [price_led_short_bars(), smooth_bars(-1), smooth_bars(1)])
def test_existing_short_and_strict_long_results_are_identical(bars):
    cfg = config()
    ind = signals.compute_indicators(bars, cfg)
    assert main.signal_with_diagnostics(bars, ind, 499, cfg) == short.signal_with_diagnostics(bars, ind, 499, cfg)


def test_disabled_main_rule_delegates_original_short_module():
    cfg, bars = config(), mirror(price_led_short_bars())
    del cfg[main.POLICY_KEY]
    ind = signals.compute_indicators(bars, cfg)
    assert main.signal_with_diagnostics(bars, ind, 499, cfg) == short.signal_with_diagnostics(bars, ind, 499, cfg)
    assert main.evaluate_position_exit({'direction': -1}, bars, cfg, 500 * HOUR, 0)['status'] == 'disabled'


def test_long_context_without_original_raw_signal_cannot_create_entry():
    bars = mirror(price_led_short_bars())
    bars[-1] = replace(bars[-1], close=159.98)
    assert main.evaluate_long_confirmation(bars, 499, config())['allowed']
    signal, diag = evaluate(bars)
    assert signal is None and diag['signal_before_confirmation'] is None
    assert diag['admission']['stage'] == 'no_raw_signal'


def test_long_fallback_preserves_original_quality_gate(monkeypatch):
    monkeypatch.setattr(quality, 'evaluate', lambda *_a, **_k: {
        'enabled': True, 'allowed': False, 'reasons': ['quality_fixture_rejection']})
    signal, diag = evaluate(mirror(price_led_short_bars()))
    assert signal is None and diag['long_confirmation']['allowed']
    assert diag['signal_after_confirmation'] and diag['signal_after_quality'] is None
    assert diag['admission']['stage'] == 'quality_rejected'


@pytest.mark.parametrize('gate', ['atr_missing', 'atr_negative', 'atr_nan', 'atr_too_large', 'extension', 'hourly_slow'])
def test_long_fallback_cannot_waive_original_raw_entry_rules(gate):
    bars, cfg = mirror(price_led_short_bars()), config()
    ind = signals.compute_indicators(bars, cfg)
    if gate == 'hourly_slow':
        ind['ema_slow'][-1] = bars[-1].close + 1
    else:
        ind['atr'][-1] = {'atr_missing': None, 'atr_negative': -1,
            'atr_nan': float('nan'), 'atr_too_large': bars[-1].close * .031,
            'extension': .001}[gate]
    signal, diag = main.signal_with_diagnostics(bars, ind, 499, cfg)
    assert signal is None and diag['signal_before_confirmation'] is None


def test_appended_future_bars_cannot_change_earlier_entry():
    bars, cfg = mirror(price_led_short_bars()), config()
    expected = evaluate(bars, cfg)
    future = [signals.Candle((500+i)*HOUR, 200, 201, 199, 200, 10000) for i in range(8)]
    assert evaluate(bars + future, cfg, 499) == expected


def test_incomplete_four_hour_group_does_not_change_confirmation():
    bars, cfg = mirror(price_led_short_bars()), config()
    expected = main.evaluate_long_confirmation(bars, 499, cfg)
    extra = [signals.Candle((500+i)*HOUR, 200, 201, 199, 200, 10000) for i in range(3)]
    actual = main.evaluate_long_confirmation(bars + extra, 502, cfg)
    for key in ('confirmation_close_ms', 'ema_fast', 'ema_fast_previous', 'ema_mid',
                'ema_slow', 'close', 'checks', 'allowed'):
        assert actual[key] == expected[key]


@pytest.mark.parametrize('failure', ['warmup', 'gap', 'alignment', 'nan_volume', 'bad_ohlc'])
def test_malformed_history_cannot_enable_long_fallback(failure):
    bars = mirror(price_led_short_bars())
    if failure == 'warmup':
        bars = bars[-239:]
    elif failure == 'gap':
        del bars[50]
    elif failure == 'alignment':
        bars[50] = replace(bars[50], time_ms=bars[50].time_ms + 1)
    elif failure == 'nan_volume':
        bars[50] = SimpleNamespace(**{**vars(bars[50]), 'volume': float('nan')})
    else:
        bars[50] = SimpleNamespace(**{**vars(bars[50]), 'low': bars[50].high + 1})
    assert not main.evaluate_long_confirmation(bars, len(bars)-1, config())['allowed']


def test_current_context_updates_while_holding_and_records_both_timeframes():
    bars, cfg = mirror(price_led_short_bars()), config()
    old = deepcopy(bars)
    result = main.trend_diagnostics(bars, cfg, 500 * HOUR + 30000)
    assert result['status'] == 'healthy'
    assert result['signal_close_ms'] == result['hourly']['closed_at_ms'] == 500 * HOUR
    assert result['confirmation_close_ms'] == result['four_hour']['closed_at_ms'] == 500 * HOUR
    assert result['four_hour']['price_led_direction'] == 1
    assert result['four_hour']['strict_direction'] == -1
    assert bars == old
    next_bars = bars + [signals.Candle(500 * HOUR, 161, 162, 160, 161, 10000)]
    updated = main.trend_diagnostics(next_bars, cfg, 501 * HOUR + 30000)
    assert updated['signal_close_ms'] == 501 * HOUR
    assert updated['confirmation_close_ms'] == 500 * HOUR
    assert updated['hourly']['ema_fast'] != result['hourly']['ema_fast']


@pytest.mark.parametrize('failure', ['future', 'stale', 'gap', 'bad_clock', 'empty'])
def test_current_context_rejects_future_stale_or_incomplete_data(failure):
    bars, stamp = smooth_bars(), 500 * HOUR
    if failure == 'future': stamp -= 1
    elif failure == 'stale': stamp += HOUR
    elif failure == 'gap': del bars[-3]
    elif failure == 'bad_clock': stamp = None
    else: bars = []
    outcome = main.trend_diagnostics(bars, config(), stamp)
    assert outcome['status'] == 'unavailable' and outcome['reasons']


@pytest.mark.parametrize('side', [-1, 1])
def test_two_closed_hours_exit_without_waiting_for_slow_ema(side):
    # The reversal is above fast/mid for a short but still below slow.
    prices = [200 - .1 * i for i in range(470)]
    prior = prices[-1]
    prices += [prior - i - 1 for i in range(10)]
    prior = prices[-1]
    prices += [prior + .3 * (i + 1) for i in range(20)]
    bars = bars_from_prices(prices)
    bars = bars if side == -1 else mirror(bars)
    position = {'direction': side, 'entry_time': 490 * HOUR}
    result = main.evaluate_position_exit(position, bars, config(), 500 * HOUR, 490 * HOUR)
    assert result['triggered'] and result['confirmation_bars'] == 2
    assert result['closed_at_ms'] == 500 * HOUR
    assert all(row['confirmed'] for row in result['confirmations'])
    assert side * (bars[-1].close - result['confirmations'][0]['ema_slow']) > 0


def test_two_whole_post_activation_hours_required_no_backfill():
    bars, position = mirror(price_led_short_bars()), {'direction': -1, 'entry_time': 480 * HOUR}
    one = main.evaluate_position_exit(position, bars, config(), 500 * HOUR, 498 * HOUR + 1)
    assert not one['triggered'] and one['confirmation_bars'] == 1
    none = main.evaluate_position_exit(position, bars, config(), 500 * HOUR, 499 * HOUR + 1)
    assert not none['triggered'] and none['confirmation_bars'] == 0
    exact = main.evaluate_position_exit(position, bars, config(), 500 * HOUR, 498 * HOUR)
    assert exact['triggered']


def test_position_entry_is_also_a_whole_hour_boundary():
    bars = mirror(price_led_short_bars())
    position = {'direction': -1, 'entry_time': 498 * HOUR + 1}
    result = main.evaluate_position_exit(position, bars, config(), 500 * HOUR, 0)
    assert result['confirmation_bars'] == 1 and not result['triggered']


@pytest.mark.parametrize('failure', ['future', 'stale', 'gap', 'bad_clock', 'bad_direction', 'missing_activation'])
def test_exit_never_triggers_with_unavailable_data(failure):
    bars, timestamp, activation = mirror(price_led_short_bars()), 500 * HOUR, 0
    position = {'direction': -1, 'entry_time': 480 * HOUR}
    if failure == 'future': timestamp -= 1
    elif failure == 'stale': timestamp += HOUR
    elif failure == 'gap': del bars[-3]
    elif failure == 'bad_clock': timestamp = None
    elif failure == 'bad_direction': position['direction'] = True
    else: activation = None
    result = main.evaluate_position_exit(position, bars, config(), timestamp, activation)
    assert result['status'] == 'unavailable' and not result['triggered']


def test_exit_requires_two_consecutive_opposite_fast_mid_and_price_closes(monkeypatch):
    bars, cfg = smooth_bars(), config()
    actual = signals.compute_indicators(bars, cfg)
    actual['ema_fast'][-2:] = [bars[-2].close - 1, bars[-1].close - 1]
    actual['ema_mid'][-2:] = [bars[-2].close - 2, bars[-1].close - 2]
    # The latest close is on the wrong side of fast, so even the prior
    # completed reversal cannot accumulate across a nonqualifying bar.
    actual['ema_fast'][-1] = bars[-1].close
    monkeypatch.setattr(signals, 'compute_indicators', lambda *_a, **_k: actual)
    result = main.evaluate_position_exit({'direction': -1, 'entry_time': 0}, bars, cfg, 500 * HOUR, 0)
    assert not result['triggered'] and result['confirmation_bars'] == 0
    assert not result['confirmations'][0]['checks']['close_beyond_fast']


@pytest.mark.parametrize('change', ['mode', 'boolean_type', 'confirmation_count', 'extra_key', 'missing_short'])
def test_policy_must_match_explicit_main_policy(change):
    cfg = config()
    if change == 'mode': cfg[main.POLICY_KEY]['mode'] = 'other'
    elif change == 'boolean_type': cfg[main.POLICY_KEY]['require_close_beyond_fast_mid'] = 1
    elif change == 'confirmation_count': cfg[main.POLICY_KEY]['confirmation_bars'] = 1
    elif change == 'extra_key': cfg[main.POLICY_KEY]['extra'] = True
    else: del cfg[short.POLICY_KEY]
    with pytest.raises(ValueError):
        main.validate(cfg)
