"""Offline mechanism invariants, not a performance backtest."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_early_recovery as recovery
from stock_swing_signals import Candle, Signal

HOUR = recovery.HOUR
START = 1_790_000_000_000 // HOUR * HOUR


def config(**updates):
    return {'signal_family': recovery.SIGNAL_FAMILY,
            'early_recovery_policy': dict(recovery.POLICY),
            'stop_atr': 2, 'min_stop_fraction': .005,
            'max_stop_fraction': .05, 'max_atr_fraction': .03, **updates}


def candles(*, short=False, last=99.6):
    closes = [100.] * 53 + [99.75, 99.5, 99.25, 99., 98.75, 98.5, last]
    bars = []
    previous = closes[0]
    for index, close in enumerate(closes):
        opening = previous
        high, low = max(opening, close) + .6, min(opening, close) - .6
        if short:
            bars.append(Candle(START + index * HOUR, 200 - opening, 200 - low,
                               200 - high, 200 - close, 12))
        else:
            bars.append(Candle(START + index * HOUR, opening, high, low, close, 12))
        previous = close
    return bars


def end(bars):
    return bars[-1].time_ms + HOUR


def raw_rows(bars):
    return [[bar.time_ms, str(bar.open), str(bar.high), str(bar.low), str(bar.close),
             str(bar.volume), bar.time_ms + HOUR - 1] for bar in bars]


@pytest.mark.parametrize('short', [False, True])
def test_fixed_recovery_produces_compatible_signal_and_frozen_reclaim(short):
    bars = candles(short=short)
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    assert isinstance(signal, Signal)
    assert signal.direction == (-1 if short else 1)
    assert signal.time_ms == bars[-1].time_ms and signal.close == bars[-1].close
    assert signal.atr > 0 and signal.strength > 0
    assert diagnostic['swing_confirmation'] == {'enabled': False, 'allowed': True, 'reasons': []}
    assert diagnostic['admission']['stage'] == 'admitted'
    assert diagnostic['admission']['confirmation_rejected_candidate'] is False
    assert diagnostic['signal_before_confirmation'] == diagnostic['signal_after_confirmation'] == diagnostic['signal_after_quality'] == asdict(signal)
    assert diagnostic['entry_quality']['allowed']
    assert diagnostic['prior_high'] == bars[-2].high
    assert diagnostic['prior_low'] == bars[-2].low
    assert diagnostic['boundary_ms'] == end(bars)
    expected = (min if short else max)(diagnostic['signal_ema_fast'], bars[-2].low if short else bars[-2].high)
    assert diagnostic['reclaim_level'] == expected == diagnostic['early_recovery']['reclaim_level']
    assert signal.direction * (signal.close - diagnostic['reclaim_level']) > 0
    # This is an early repair while the fast/mid EMA ordering is still against it.
    detail = diagnostic['early_recovery']
    assert signal.direction * (detail['prior_ema_fast'] - detail['prior_ema_mid']) < 0
    assert signal.direction * (detail['structural_invalidation_level'] - detail['projected_initial_stop']) >= 0
    json.dumps(diagnostic, allow_nan=False)


def test_long_short_price_reflection_has_symmetric_atr_and_strength():
    long = recovery.candidate_signal(candles(), config(), asof_ms=end(candles()))
    short = recovery.candidate_signal(candles(short=True), config(), asof_ms=end(candles()))
    assert long.atr == pytest.approx(short.atr)
    assert long.strength == pytest.approx(short.strength)
    assert long.close + short.close == pytest.approx(200)


def test_explicit_decision_clock_and_policy_validation_are_required():
    bars = candles()
    with pytest.raises(ValueError, match='explicit_decision_asof_ms_required'):
        recovery.candidate_signal(bars, config())
    for stamp in [True, '2026-10-28', float('inf'), None, -1]:
        with pytest.raises(ValueError, match='explicit_decision_asof_ms_required'):
            recovery.candidate_signal(bars, config(asof_ms=stamp))
    assert recovery.candidate_signal(bars, config(asof_ms=end(bars))) is not None


@pytest.mark.parametrize('key,value', [('context_lookback_bars', 5), ('ema_fast', 8.0),
    ('minimum_context_move_atr', .5), ('mode', 'other'), ('structural_buffer_atr', True),
    ('maximum_reclaim_extension_atr', float('nan'))])
def test_fixed_policy_is_not_a_parameter_search_surface(key, value):
    cfg = config()
    cfg['early_recovery_policy'][key] = value
    with pytest.raises(ValueError, match='fixed_early_recovery_policy_required'):
        recovery.validate(cfg)


def test_policy_is_copied_and_original_symbol_minimum_stops_are_inherited():
    bars = candles()
    for minimum in [.005, .02]:
        cfg = config(min_stop_fraction=minimum)
        before = deepcopy(cfg)
        signal, diagnostic = recovery.candidate_with_diagnostics(bars, cfg, asof_ms=end(bars))
        assert signal is not None
        assert diagnostic['early_recovery']['min_stop_fraction'] == minimum
        assert cfg == before
        policy = recovery.validate(cfg)
        policy['ema_fast'] = 999
        assert recovery.POLICY['ema_fast'] == 8
    for values in [dict(min_stop_fraction=.01), dict(stop_atr=3), dict(max_stop_fraction=.1), dict(max_atr_fraction=.04)]:
        with pytest.raises(ValueError):
            recovery.validate(config(**values))


def test_raw_exchange_rows_and_mappings_match_shared_candle_input_without_mutation():
    bars = candles()
    expected = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    for rows in [raw_rows(bars), [asdict(bar) for bar in bars]]:
        before = deepcopy(rows)
        assert recovery.candidate_with_diagnostics(rows, config(), asof_ms=end(bars)) == expected
        assert rows == before


def test_unclosed_signal_bar_is_never_used_even_one_millisecond_before_boundary():
    bars = candles()
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars) - 1)
    assert signal is None
    assert diagnostic['admission']['reasons'] == ['insufficient_closed_history']
    assert recovery.candidate_signal(bars, config(), asof_ms=end(bars)) is not None


def test_unclosed_and_future_suffix_prices_do_not_change_the_closed_prefix_signal():
    bars = candles()
    boundary = end(bars)
    expected = recovery.candidate_with_diagnostics(bars, config(), asof_ms=boundary)
    # Prices are deliberately invalid; future price cells must not be read.
    future = [[boundary, 'secret', float('nan'), float('inf'), -1, True],
              [boundary + HOUR, 1, 2, 0, float('nan'), 0]]
    assert recovery.candidate_with_diagnostics(raw_rows(bars) + future, config(), asof_ms=boundary) == expected
    assert recovery.candidate_with_diagnostics(raw_rows(bars) + future, config(), asof_ms=boundary + 123) == recovery.candidate_with_diagnostics(bars, config(), asof_ms=boundary + 123)


def test_fixed_window_does_not_depend_on_older_history_or_external_indicators():
    bars = candles()
    older = [Candle(START - HOUR, 500, 501, 499, 500, 1)]
    assert recovery.candidate_with_diagnostics(older + bars, config(), asof_ms=end(bars)) == recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))


@pytest.mark.parametrize('kind', ['gap', 'duplicate', 'order', 'misaligned', 'wrong_close', 'explicit_unclosed'])
def test_used_hourly_window_rejects_gaps_duplicates_order_and_invalid_closure(kind):
    bars = candles()
    rows = raw_rows(bars)
    if kind == 'gap':
        del rows[20]
        rows.insert(0, [START - 2 * HOUR, 100, 101, 99, 100, 1, START - HOUR - 1])
    elif kind == 'duplicate':
        rows[20] = deepcopy(rows[19])
    elif kind == 'order':
        rows[20], rows[21] = rows[21], rows[20]
    elif kind == 'misaligned':
        rows[20][0] += 1
        rows[20][6] += 1
    elif kind == 'wrong_close':
        rows[-1][6] -= 1
    else:
        rows = [asdict(bar) for bar in bars]
        rows[-1]['closed'] = False
    signal, diagnostic = recovery.candidate_with_diagnostics(rows, config(), asof_ms=end(bars))
    assert signal is None and diagnostic['admission']['stage'] == 'no_raw_signal'
    assert diagnostic['early_recovery']['reasons']


def test_stale_closed_prefix_and_insufficient_data_cannot_become_new_signals():
    bars = candles()
    for rows in [[], bars[:59]]:
        assert recovery.candidate_signal(rows, config(), asof_ms=end(bars)) is None
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars) + HOUR)
    assert signal is None
    assert diagnostic['admission']['reasons'] == ['latest_closed_hour_missing']


@pytest.mark.parametrize('column,value', [(1, float('nan')), (2, float('inf')),
    (3, -1), (4, True), (5, float('nan')), (5, -1), (2, 1), (0, True)])
def test_nonfinite_nonpositive_inconsistent_and_nonnumeric_used_data_fail_closed(column, value):
    bars = candles()
    rows = raw_rows(bars)
    rows[-1][column] = value
    signal, diagnostic = recovery.candidate_with_diagnostics(rows, config(), asof_ms=end(bars))
    assert signal is None
    assert diagnostic['admission']['stage'] == 'no_raw_signal'
    json.dumps(diagnostic, allow_nan=False)


def test_decline_context_and_fresh_cross_are_both_required():
    bars = [Candle(START + i * HOUR, 100, 100.6, 99.4, 100, 12) for i in range(60)]
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    assert signal is None
    assert diagnostic['admission']['reasons'] == ['no_fresh_early_recovery']
    bullish = candles()
    # A price increase which remains below the preceding high is not a reclaim.
    bullish[-1] = Candle(bullish[-1].time_ms, 98.5, 99.5, 97.9, 99.3, 12)
    assert recovery.candidate_signal(bullish, config(), asof_ms=end(bullish)) is None


def test_chasing_or_structure_outside_the_original_stop_is_rejected_after_raw_candidate():
    bars = candles(last=103)
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    assert signal is None and diagnostic['signal_before_confirmation'] is not None
    assert diagnostic['admission']['stage'] == 'quality_rejected'
    assert 'reclaim_extension_exceeds_one_atr' in diagnostic['entry_quality']['reasons']
    bars = candles()
    current = bars[-1]
    bars[-1] = Candle(current.time_ms, current.open, current.high, 95, current.close, current.volume)
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    assert signal is None and diagnostic['signal_before_confirmation'] is not None
    assert 'structure_outside_existing_stop' in diagnostic['entry_quality']['reasons']
    assert diagnostic['signal_after_quality'] is None


def test_huge_finite_market_values_do_not_leak_infinite_indicators_or_raise():
    bars = [Candle(START + i * HOUR, 1e308, 1e308, 1e308, 1e308, 0) for i in range(60)]
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    assert signal is None
    json.dumps(diagnostic, allow_nan=False)


def test_existing_symbol_minimum_changes_the_projected_stop_without_changing_the_price_signal():
    bars = [Candle(bar.time_ms, 100 + (bar.open - 100) / 10,
                   100 + (bar.high - 100) / 10, 100 + (bar.low - 100) / 10,
                   100 + (bar.close - 100) / 10, bar.volume) for bar in candles()]
    results = [recovery.candidate_with_diagnostics(bars, config(min_stop_fraction=minimum),
                                                  asof_ms=end(bars)) for minimum in [.005, .02]]
    assert results[0][0] == results[1][0] and results[0][0] is not None
    for minimum, (signal, diagnostic) in zip([.005, .02], results):
        detail = diagnostic['early_recovery']
        assert detail['projected_initial_stop'] == pytest.approx(signal.close * (1 - minimum))
        assert detail['stop_risk_fraction'] == pytest.approx(minimum)


def test_high_volatility_or_excessive_original_stop_cannot_pass_the_quality_layer():
    bars = candles()
    current = bars[-1]
    bars[-1] = Candle(current.time_ms, current.open, 200, current.low, current.close, 12)
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, config(), asof_ms=end(bars))
    assert signal is None and diagnostic['signal_before_confirmation'] is not None
    assert diagnostic['early_recovery']['projected_initial_stop'] is None
    assert set(diagnostic['entry_quality']['reasons']) >= {'atr_fraction_exceeds_limit', 'stop_distance_exceeds_limit'}
