"""Pending breakouts must remain valid at observed quotes and actual fills."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_entry_timing as timing
import stock_swing_signals as signals


CONFIG = {'signal_family': 'donchian20', 'signal_timeframe': '1h', 'breakout_bars': 20,
          'entry_gap_atr': .5, 'max_breakout_extension_atr': 1.,
          'entry_timing_policy': timing.POLICY.copy()}


def case(side=1):
    bars = [signals.Candle(i * timing.HOUR, 100, 101, 99, 100) for i in range(20)]
    close = 101.2 if side == 1 else 98.8
    bars.append(signals.Candle(20 * timing.HOUR, close, close + .1, close - .1, close))
    signal = signals.Signal(side, 1., close, .2, bars[-1].time_ms)
    boundary = 21 * timing.HOUR
    trigger = timing.build_trigger(bars, signal, CONFIG, boundary, observed_at_ms=boundary + 1000)
    return bars, signal, trigger, boundary


def prices(side=1):
    return {'reference_price': 101.3 if side == 1 else 98.7,
            'mark_price': 101.25 if side == 1 else 98.75,
            'fill_price': 101.35 if side == 1 else 98.65}


@pytest.mark.parametrize('side', [1, -1])
def test_trigger_freezes_prior_twenty_excluding_the_signal_candle(side):
    bars, signal, trigger, boundary = case(side)
    assert trigger['breakout_level'] == (101 if side == 1 else 99)
    assert trigger['breakout_level'] != (bars[-1].high if side == 1 else bars[-1].low)
    assert trigger['signal_time_ms'] == signal.time_ms
    assert trigger['boundary_ms'] == boundary
    assert trigger['prior_window_start_ms'] == 0
    assert trigger['prior_window_end_ms'] == 19 * timing.HOUR
    assert 'breakout_level' not in asdict(signal)
    result = timing.evaluate(trigger, signal, CONFIG, **prices(side))
    assert result['allowed'] and not result['cancel_pending']


def test_real_sndk_rebound_must_cancel_the_pending_short():
    # Existing October 9 00:00 signal / 00:10:23 fill: close broke 1636.60,
    # but bid=1639.27 and the VWAP/slip fill=1638.932772875 were back inside.
    bars = [signals.Candle(i * timing.HOUR, 1640, 1645, 1636.6, 1640) for i in range(20)]
    bars.append(signals.Candle(20 * timing.HOUR, 1644.87, 1644.87, 1635.63, 1636.))
    signal = signals.Signal(-1, 14.49702032016169, 1636., .613, 20 * timing.HOUR)
    trigger = timing.build_trigger(bars, signal, CONFIG, 21 * timing.HOUR)
    assert signals.entry_gap_allowed(signal, 1639.27, CONFIG)  # Old check alone admitted it.
    result = timing.evaluate(trigger, signal, CONFIG, reference_price=1639.27,
                             mark_price=1639.2, fill_price=1638.932772875)
    assert not result['allowed'] and result['cancel_pending']
    assert set(result['reasons']) == {'breakout_reference_invalidated',
                                     'breakout_mark_invalidated', 'breakout_fill_invalidated'}


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('label', ['reference', 'mark', 'fill'])
@pytest.mark.parametrize('offset', [0., -.01])
def test_equal_or_back_inside_original_level_cancels_in_both_directions(side, label, offset):
    _, signal, trigger, _ = case(side)
    values = prices(side)
    values[label + '_price'] = trigger['breakout_level'] + side * offset
    result = timing.evaluate(trigger, signal, CONFIG, **values)
    assert not result['allowed'] and result['cancel_pending']
    assert 'breakout_' + label + '_invalidated' in result['reasons']


@pytest.mark.parametrize('side', [1, -1])
def test_valid_retrace_still_outside_breakout_is_allowed_without_inventing_a_retest(side):
    _, signal, trigger, _ = case(side)
    price = trigger['breakout_level'] + side * .01
    result = timing.evaluate(trigger, signal, CONFIG, reference_price=price,
                             mark_price=price, fill_price=price)
    assert result['allowed']


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('label', ['reference', 'fill'])
def test_chase_limit_uses_actual_vwap_slip_fill_as_well_as_quote(side, label):
    _, signal, trigger, _ = case(side)
    values = prices(side)
    values[label + '_price'] = signal.close + side * .5001
    result = timing.evaluate(trigger, signal, CONFIG, **values)
    assert not result['allowed'] and not result['cancel_pending']
    assert 'entry_timing_' + label + '_gap_exceeded' in result['reasons']


@pytest.mark.parametrize('side', [1, -1])
def test_one_atr_actual_extension_can_block_despite_quote_and_gap_passing(side):
    bars, signal, _, boundary = case(side)
    # A signal .8 ATR outside the level leaves only .2 ATR extension room,
    # even though its separate chase allowance remains .5 ATR.
    close = 101.8 if side == 1 else 98.2
    bars[-1] = signals.Candle(20 * timing.HOUR, close, close + .1, close - .1, close)
    signal = signals.Signal(side, 1., close, .8, 20 * timing.HOUR)
    trigger = timing.build_trigger(bars, signal, CONFIG, boundary)
    fill = trigger['breakout_level'] + side * 1.001
    assert signals.entry_gap_allowed(signal, fill, CONFIG)
    result = timing.evaluate(trigger, signal, CONFIG, reference_price=close,
                             mark_price=close, fill_price=fill)
    assert not result['allowed'] and not result['cancel_pending']
    assert result['reasons'] == ['entry_timing_fill_extension_exceeded']


@pytest.mark.parametrize('side', [1, -1])
def test_exact_chase_and_extension_ceiling_are_included(side):
    bars, _, _, boundary = case(side)
    close = 101.5 if side == 1 else 98.5
    bars[-1] = signals.Candle(20 * timing.HOUR, close, close + .1, close - .1, close)
    signal = signals.Signal(side, 1., close, .5, 20 * timing.HOUR)
    trigger = timing.build_trigger(bars, signal, CONFIG, boundary)
    price = close + side * .5
    assert timing.evaluate(trigger, signal, CONFIG, reference_price=price,
                           mark_price=price, fill_price=price)['allowed']


@pytest.mark.parametrize('label', ['reference', 'mark', 'fill'])
@pytest.mark.parametrize('bad_price', [None, float('nan'), float('inf'), 0, -1, True, 'bad'])
def test_missing_or_invalid_prices_pause_new_entries_without_cancelling_metadata(label, bad_price):
    _, signal, trigger, _ = case()
    values = prices()
    values[label + '_price'] = bad_price
    result = timing.evaluate(trigger, signal, CONFIG, **values)
    assert not result['allowed'] and not result['cancel_pending']
    assert 'entry_timing_' + label + '_unavailable' in result['reasons']


@pytest.mark.parametrize('label', ['reference', 'mark'])
def test_missing_required_quote_or_mark_pauses(label):
    _, signal, trigger, _ = case()
    values = prices()
    values[label + '_price'] = None
    result = timing.evaluate(trigger, signal, CONFIG, **values)
    assert not result['allowed'] and not result['cancel_pending']


@pytest.mark.parametrize('patch', [
    {'version': True}, {'family': 'breakout'}, {'direction': -1}, {'direction': True},
    {'breakout_level': float('nan')}, {'breakout_level': 102}, {'breakout_bars': 21},
    {'signal_time_ms': 21 * timing.HOUR}, {'boundary_ms': 20 * timing.HOUR},
    {'prior_window_start_ms': timing.HOUR}, {'prior_window_end_ms': 18 * timing.HOUR},
    {'signal_close': 101.3}, {'signal_atr': 2.}, {'signal_timeframe': '15m'},
])
def test_bad_trigger_metadata_fails_closed_and_cancels(patch):
    _, signal, trigger, _ = case()
    result = timing.evaluate({**trigger, **patch}, signal, CONFIG, **prices())
    assert not result['allowed'] and result['cancel_pending']


@pytest.mark.parametrize('trigger', [None, {}, {'version': 1}])
def test_old_pending_without_metadata_cannot_be_silently_admitted(trigger):
    _, signal, _, _ = case()
    result = timing.evaluate(trigger, signal, CONFIG, **prices())
    assert not result['allowed'] and result['cancel_pending']


def test_rebuild_old_pending_uses_original_prefix_and_ignores_later_closed_channel():
    bars, signal, original, boundary = case()
    history = bars + [signals.Candle(21 * timing.HOUR, 1000, 1001, 999, 1000)]
    pending = {'signal': asdict(signal), 'boundary_ms': boundary}
    rebuilt = timing.rebuild_trigger(history, pending, CONFIG, observed_at_ms=22 * timing.HOUR)
    assert rebuilt == original
    assert rebuilt['breakout_level'] == 101
    # Direct construction with that later prefix is an error, not a changed level.
    with pytest.raises(ValueError, match='original_prefix_mismatch'):
        timing.build_trigger(history, signal, CONFIG, boundary)


def test_rebuild_cannot_replace_a_missing_original_signal_with_current_rolling_level():
    bars, signal, _, boundary = case()
    pending = {'signal': asdict(signal), 'boundary_ms': boundary}
    with pytest.raises(ValueError, match='original_signal_bar_unavailable'):
        timing.rebuild_trigger(bars[:-1], pending, CONFIG)
    with pytest.raises(ValueError, match='original_prefix_unavailable'):
        timing.rebuild_trigger(bars[1:], pending, CONFIG)


def test_unfinished_mismatched_or_gapped_source_cannot_create_trigger():
    bars, signal, _, boundary = case()
    with pytest.raises(ValueError, match='unfinished_signal_bar'):
        timing.build_trigger(bars, signal, CONFIG, boundary, observed_at_ms=boundary - 1)
    with pytest.raises(ValueError, match='boundary_mismatch'):
        timing.build_trigger(bars, signal, CONFIG, boundary + timing.HOUR)
    with pytest.raises(ValueError, match='hourly_gap'):
        timing.build_trigger([bars[0]] + bars, signal, CONFIG, boundary)
    wrong_signal = signals.Signal(1, 1., 101.3, .2, signal.time_ms)
    with pytest.raises(ValueError, match='signal_close_mismatch'):
        timing.build_trigger(bars, wrong_signal, CONFIG, boundary)


@pytest.mark.parametrize('patch', [
    {'entry_timing_policy': {'mode': 'typo', 'require_breakout_valid_at_fill': True}},
    {'entry_timing_policy': {'mode': timing.MODE, 'require_breakout_valid_at_fill': False}},
    {'entry_timing_policy': {'mode': timing.MODE, 'require_breakout_valid_at_fill': 1}},
    {'signal_family': 'ema_transition'}, {'signal_timeframe': '15m'}, {'breakout_bars': True},
    {'entry_gap_atr': .6}, {'max_breakout_extension_atr': 1.1}, {'entry_gap_atr': True},
])
def test_bad_policy_or_unsupported_strategy_fails_closed(patch):
    bars, signal, trigger, boundary = case()
    config = {**CONFIG, **patch}
    with pytest.raises(ValueError):
        timing.validate(config)
    with pytest.raises(ValueError):
        timing.build_trigger(bars, signal, config, boundary)
    result = timing.evaluate(trigger, signal, config, **prices())
    assert not result['allowed'] and result['cancel_pending']


def test_policy_absence_preserves_historical_behavior_without_touching_signal():
    config = {key: value for key, value in CONFIG.items() if key != 'entry_timing_policy'}
    timing.validate(config)
    assert not timing.enabled(config)
    assert timing.build_trigger([], None, config, None) is None
    assert timing.rebuild_trigger([], None, config) is None
    assert timing.evaluate(None, None, config, reference_price=None, mark_price=None) == {
        'enabled': False, 'allowed': True, 'cancel_pending': False, 'reasons': []}
