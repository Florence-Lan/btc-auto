"""Signal-stage evidence must preserve the existing causal admission decision."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_entry_quality as quality
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
import stock_swing_signals as signals

HOUR = confirmation.HOUR
CONFIG = {'signal_timeframe': '1h', 'signal_family': 'donchian20',
    'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
    'max_atr_fraction': .03, 'require_price_slow': True,
    'swing_confirmation_policy': confirmation.POLICY.copy(),
    'taker_fee_rate_assumption': .000125, 'adverse_slippage_fraction_assumption': .0002}


def bars_for(side=1, *, step=.02, wide=False, flat=False):
    bars = []
    for index in range(504):
        close = 100. if flat else 100. + side * index * step
        upper = .2 if wide and side == -1 else .0001 if step == .001 else .005
        lower = .2 if wide and side == 1 else .0001 if step == .001 else .005
        bars.append(signals.Candle(index * HOUR, close, close + upper, close - lower, close, 10000))
    return bars


def fixture_case(case, side):
    config = deepcopy(CONFIG)
    bars = bars_for(side, step=.001 if case == 'confirmation_blocked' else .02,
                    wide=case == 'quality_passed', flat=case == 'raw_absent')
    if case in ('quality_passed', 'quality_blocked'):
        config['entry_quality_policy'] = quality.POLICY.copy()
    if case == 'unconfirmed':
        config.pop('swing_confirmation_policy')
    return bars, config, signals.compute_indicators(bars, config)


def original_admission(candles, indicators, index, config):
    """The pre-change callable contract, retained as a behavioral oracle."""
    config = config or {}
    quality.validate(config)
    confirmation.validate(config)
    signal = profiles.raw_signal_at(candles, indicators, index, config)
    outcome = confirmation.evaluate(candles, index, config, signal.direction if signal else None)
    if not outcome['allowed']:
        return None, outcome
    if signal and not quality.evaluate(candles, indicators, index, config, signal.direction)['allowed']:
        return None, outcome
    return signal, outcome


@pytest.mark.parametrize('side', [1, -1])
def test_real_direction_candidate_survives_confirmation_without_changing_signal(side):
    bars, config, indicators = fixture_case('passed', side)
    raw = profiles.raw_signal_at(bars, indicators, 499, config)
    signal, evidence = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert signal == raw and signal.direction == side
    for field in ('signal_before_confirmation', 'signal_after_confirmation', 'signal_after_quality'):
        assert evidence[field] == asdict(raw)
    assert evidence['admission'] == {'stage': 'admitted', 'reasons': [],
                                     'confirmation_rejected_candidate': False}
    assert evidence['entry_quality'] == {'allowed': True, 'reasons': []}
    assert profiles.signal_at(bars, indicators, 499, config) == signal
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence


@pytest.mark.parametrize('side', [1, -1])
def test_real_hourly_breakout_is_preserved_when_four_hour_spread_blocks_it(side):
    bars, config, indicators = fixture_case('confirmation_blocked', side)
    raw = profiles.raw_signal_at(bars, indicators, 499, config)
    signal, evidence = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert raw and raw.direction == side and signal is None
    assert evidence['signal_before_confirmation'] == asdict(raw)
    assert evidence['signal_after_confirmation'] is None
    assert evidence['signal_after_quality'] is None and evidence['entry_quality'] is None
    assert evidence['admission'] == {'stage': 'confirmation_rejected',
        'reasons': ['swing_4h_spread_too_small'], 'confirmation_rejected_candidate': True}


def test_no_breakout_and_blocked_four_hour_context_is_not_a_rejected_candidate():
    bars, config, indicators = fixture_case('raw_absent', 1)
    signal, evidence = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert signal is None and not evidence['swing_confirmation']['allowed']
    assert 'swing_4h_ema_ordering' in evidence['swing_confirmation']['reasons']
    assert all(evidence[k] is None for k in ('signal_before_confirmation',
        'signal_after_confirmation', 'signal_after_quality', 'entry_quality'))
    assert evidence['admission'] == {'stage': 'no_raw_signal', 'reasons': ['no_raw_signal'],
                                     'confirmation_rejected_candidate': False}


@pytest.mark.parametrize('side', [1, -1])
def test_no_current_breakout_can_coexist_with_allowed_four_hour_context(side):
    bars, config, _ = fixture_case('passed', side)
    close = bars[498].close
    bars[499] = signals.Candle(499 * HOUR, close, close + .005, close - .005, close, 10000)
    signal, evidence = profiles.signal_with_diagnostics(bars, signals.compute_indicators(bars, config), 499, config)
    assert signal is None and evidence['swing_confirmation']['allowed']
    assert evidence['signal_before_confirmation'] is None
    assert evidence['admission']['stage'] == 'no_raw_signal'
    assert not evidence['admission']['confirmation_rejected_candidate']


@pytest.mark.parametrize('side', [1, -1])
def test_quality_rejection_keeps_the_confirmed_candidate_and_its_own_reasons(side):
    bars, config, indicators = fixture_case('quality_blocked', side)
    raw = profiles.raw_signal_at(bars, indicators, 499, config)
    signal, evidence = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert raw and signal is None and evidence['swing_confirmation']['allowed']
    assert evidence['signal_before_confirmation'] == evidence['signal_after_confirmation'] == asdict(raw)
    assert evidence['signal_after_quality'] is None
    assert evidence['admission'] == {'stage': 'quality_rejected',
        'reasons': ['quality_volatility_below_cost'], 'confirmation_rejected_candidate': False}
    assert evidence['entry_quality'] == quality.evaluate(bars, indicators, 499, config, side)


@pytest.mark.parametrize('side', [1, -1])
def test_quality_passed_returns_original_signal_with_all_three_snapshots(side):
    bars, config, indicators = fixture_case('quality_passed', side)
    signal, evidence = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert signal and signal.direction == side and evidence['entry_quality']['allowed']
    assert evidence['entry_quality']['atr_roundtrip_cost_ratio'] > 1
    assert evidence['signal_after_quality'] == asdict(signal)
    assert evidence['admission']['stage'] == 'admitted'


@pytest.mark.parametrize('case,quality_calls', [('passed', 1), ('confirmation_blocked', 0),
    ('raw_absent', 0), ('quality_passed', 1), ('quality_blocked', 1), ('unconfirmed', 1)])
def test_diagnostics_does_not_recompute_or_bypass_original_short_circuit(case, quality_calls, monkeypatch):
    bars, config, indicators = fixture_case(case, 1)
    raw = Mock(wraps=profiles.raw_signal_at)
    confirm = Mock(wraps=confirmation.evaluate)
    entry_quality = Mock(wraps=quality.evaluate)
    monkeypatch.setattr(profiles, 'raw_signal_at', raw)
    monkeypatch.setattr(confirmation, 'evaluate', confirm)
    monkeypatch.setattr(quality, 'evaluate', entry_quality)
    profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert raw.call_count == confirm.call_count == 1
    assert entry_quality.call_count == quality_calls


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('case', ['passed', 'confirmation_blocked', 'raw_absent',
                                  'quality_passed', 'quality_blocked', 'unconfirmed'])
def test_signal_and_all_diagnostics_use_only_the_requested_closed_prefix(case, side):
    bars, config, indicators = fixture_case(case, side)
    expected = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    prefix = bars[:500]
    assert profiles.signal_with_diagnostics(prefix, signals.compute_indicators(prefix, config), 499, config) == expected
    for index in range(500, len(bars)):
        bars[index] = signals.Candle(index * HOUR, 10000, 10001, 9999, 10000, 0)
    assert profiles.signal_with_diagnostics(bars, signals.compute_indicators(bars, config), 499, config) == expected


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('case', ['passed', 'confirmation_blocked', 'raw_absent',
                                  'quality_passed', 'quality_blocked', 'unconfirmed'])
def test_final_signal_and_original_confirmation_are_equivalent_to_prechange_behavior(case, side):
    bars, config, indicators = fixture_case(case, side)
    for index in (58, 59, 238, 239, 499):
        original, confirmation_evidence = original_admission(bars, indicators, index, config)
        signal, evidence = profiles.signal_with_diagnostics(bars, indicators, index, config)
        assert signal == original
        assert evidence['swing_confirmation'] == confirmation_evidence


def test_json_snapshots_do_not_mutate_inputs_or_share_mutable_stage_dicts():
    bars, config, indicators = fixture_case('quality_passed', 1)
    original = deepcopy((bars, config, indicators))
    signal, evidence = profiles.signal_with_diagnostics(bars, indicators, 499, config)
    evidence['signal_before_confirmation']['direction'] = -1
    evidence['signal_after_confirmation']['close'] = 1
    assert evidence['signal_after_quality'] == asdict(signal)
    assert (bars, config, indicators) == original
    assert profiles.signal_with_diagnostics(bars, indicators, 499, config)[0] == signal


@pytest.mark.parametrize('change', [
    {'entry_quality_policy': {**quality.POLICY, 'lookback_bars': 13}},
    {'swing_confirmation_policy': {**confirmation.POLICY, 'minimum_ema_spread_fraction': .005}},
])
def test_invalid_frozen_policy_retains_original_exception(change):
    bars, config, indicators = fixture_case('passed', 1)
    config.update(change)
    with pytest.raises(ValueError) as original:
        original_admission(bars, indicators, 499, config)
    with pytest.raises(ValueError) as current:
        profiles.signal_with_diagnostics(bars, indicators, 499, config)
    assert str(current.value) == str(original.value)
