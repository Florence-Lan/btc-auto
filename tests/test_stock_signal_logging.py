"""Real stock signal stages are journaled without changing execution meaning."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import stock_entry_quality as quality
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
from test_stock_swing_confirmation import worker_fixture, hourly_bars

HOUR = confirmation.HOUR


def logged_worker(tmp_path, monkeypatch, *, side=1, case='passed', restriction=None):
    original, path, clock = worker_fixture(tmp_path, monkeypatch, side=side)
    config = deepcopy(original.config)
    if case == 'quality_blocked':
        config['entry_quality_policy'] = quality.POLICY.copy()
    if restriction == 'direction':
        config['entry_direction'] = 'short' if side == 1 else 'long'
    elif restriction == 'session':
        config['entry_session'] = 'weekday_regular_clock'
    bars = hourly_bars(side, length=500, start=100, step=.001 if case == 'confirmation_blocked' else .02)
    if case in ('confirmation_blocked', 'no_raw'):
        bars = [paper.signals.Candle(b.time_ms, 100 if case == 'no_raw' else b.close,
            (100 if case == 'no_raw' else b.close) + .0001,
            (100 if case == 'no_raw' else b.close) - .0001,
            100 if case == 'no_raw' else b.close, b.volume) for b in bars]
    price = bars[-1].close
    source_get = original.venue.get.side_effect
    def get(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '1h':
            return [[b.time_ms, b.open, b.high, b.low, b.close, b.volume, b.time_ms + HOUR - 1]
                    for b in bars] + [[600 * HOUR, 10000, 10001, 9999, 10000, 0, 601 * HOUR - 1]]
        if endpoint == 'premiumIndex':
            return {'time': clock['now'], 'markPrice': str(price), 'indexPrice': str(price),
                    'lastFundingRate': '0', 'nextFundingTime': clock['now'] + HOUR}
        if endpoint == 'depth':
            return {'T': clock['now'], 'bids': [[str(price - .0001), '1000']],
                    'asks': [[str(price + .0001), '1000']]}
        return source_get(endpoint, params)
    original.venue.get.side_effect = get
    clock['volume'] = 10000
    paper.write_json(path, paper.initial_stock('MUUSDT', 600 * HOUR - 10000, config))
    return paper.StockAccount(path, original.venue, 'MUUSDT', config), clock, bars


def journal_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def legacy_signal(bars, config):
    """The original signal_at result, before the research direction/session gate."""
    indicators = paper.signals.compute_indicators(bars, config)
    signal = profiles.raw_signal_at(bars, indicators, len(bars) - 1, config)
    context = confirmation.evaluate(bars, len(bars) - 1, config, signal.direction if signal else None)
    if not context['allowed']:
        return None
    if signal and not quality.evaluate(bars, indicators, len(bars) - 1, config, signal.direction)['allowed']:
        return None
    return signal


def assert_signal_journal_matches_state(worker, state):
    rows = journal_rows(worker.path.with_name('signals.jsonl'))
    assert len(rows) == 1
    row = rows[0]
    assert row['signal_diagnostics'] == state['signal_diagnostics']
    assert row['signal_diagnostics']['schema_version'] == 1
    assert row['signal_diagnostics']['decision_scope'] == 'signal_candidate_only'
    assert row['signal_diagnostics']['legacy_raw_signal_stage'] == 'after_confirmation_and_quality_before_research_policy'
    assert row['boundary_ms'] == 600 * HOUR
    assert json.loads(json.dumps(row, allow_nan=False)) == row
    return row, row['signal_diagnostics']


@pytest.mark.parametrize('side', [1, -1])
def test_true_hourly_candidate_before_four_hour_rejection_is_logged_separately(side, tmp_path, monkeypatch):
    worker, clock, bars = logged_worker(tmp_path, monkeypatch, side=side, case='confirmation_blocked')
    raw = profiles.raw_signal_at(bars, paper.signals.compute_indicators(bars, worker.config), 499, worker.config)
    state = worker.step()
    row, evidence = assert_signal_journal_matches_state(worker, state)
    assert raw and raw.direction == side
    assert evidence['signal_before_confirmation'] == asdict(raw)
    assert evidence['signal_after_confirmation'] is None and evidence['signal_after_quality'] is None
    assert evidence['admission'] == {'stage': 'confirmation_rejected',
        'reasons': ['swing_4h_spread_too_small'], 'confirmation_rejected_candidate': True}
    assert evidence['final_stage'] == 'confirmation_rejected'
    assert evidence['research_policy'] == {'evaluated': False, 'rejection': None}
    assert row['raw_signal'] is None and row['rejection'] is None and row['pending_signal'] is None
    assert state['position'] is None and state['fill_count_total'] == 0
    assert state['wallet_balance'] == 1000 and state['signal_status'] == 'swing_confirmation_blocked'


def test_no_raw_breakout_with_blocked_four_hour_context_does_not_log_a_rejected_candidate(tmp_path, monkeypatch):
    worker, clock, _ = logged_worker(tmp_path, monkeypatch, case='no_raw')
    state = worker.step()
    row, evidence = assert_signal_journal_matches_state(worker, state)
    assert evidence['signal_before_confirmation'] is None
    assert not evidence['swing_confirmation']['allowed']
    assert 'swing_4h_ema_ordering' in evidence['swing_confirmation']['reasons']
    assert evidence['admission'] == {'stage': 'no_raw_signal', 'reasons': ['no_raw_signal'],
                                     'confirmation_rejected_candidate': False}
    assert evidence['final_stage'] == 'no_raw_signal'
    assert row['raw_signal'] is None and row['pending_signal'] is None and row['rejection'] is None
    assert state['signal_status'] == 'swing_confirmation_blocked'  # Legacy status still describes context.
    assert state['fill_count_total'] == 0 and state['wallet_balance'] == 1000


@pytest.mark.parametrize('side', [1, -1])
def test_quality_rejection_is_not_mislabeled_as_four_hour_or_research_rejection(side, tmp_path, monkeypatch):
    worker, clock, _ = logged_worker(tmp_path, monkeypatch, side=side, case='quality_blocked')
    state = worker.step()
    row, evidence = assert_signal_journal_matches_state(worker, state)
    assert evidence['swing_confirmation']['allowed']
    assert evidence['signal_before_confirmation'] == evidence['signal_after_confirmation']
    assert evidence['signal_after_confirmation']['direction'] == side
    assert evidence['signal_after_quality'] is None
    assert evidence['admission']['stage'] == evidence['final_stage'] == 'quality_rejected'
    assert evidence['admission']['reasons'] == ['quality_volatility_below_cost']
    assert not evidence['admission']['confirmation_rejected_candidate']
    assert evidence['research_policy'] == {'evaluated': False, 'rejection': None}
    assert row['raw_signal'] is None and row['rejection'] is None
    assert state['fill_count_total'] == 0 and state['wallet_balance'] == 1000


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('restriction,reason', [('direction', 'research_direction'),
                                              ('session', 'research_session_clock')])
def test_research_rejection_retains_legacy_post_quality_raw_signal(side, restriction, reason, tmp_path, monkeypatch):
    worker, clock, bars = logged_worker(tmp_path, monkeypatch, side=side, restriction=restriction)
    original_signal = legacy_signal(bars, worker.config)
    assert original_signal and original_signal.direction == side
    state = worker.step()
    row, evidence = assert_signal_journal_matches_state(worker, state)
    assert row['raw_signal'] == evidence['signal_after_quality'] == asdict(original_signal)
    assert evidence['signal_before_confirmation'] == evidence['signal_after_confirmation'] == row['raw_signal']
    assert evidence['admission']['stage'] == 'admitted'
    assert evidence['research_policy'] == {'evaluated': True, 'rejection': reason}
    assert evidence['signal_after_research_policy'] is None
    assert evidence['final_stage'] == 'research_policy_rejected'
    assert row['rejection'] == reason and row['pending_signal'] is None
    assert state['position'] is None and state['fill_count_total'] == 0
    assert state['signal_status'] == 'direction_filtered'


@pytest.mark.parametrize('side', [1, -1])
def test_admitted_legacy_signal_keeps_current_book_fill_fees_and_single_entry(side, tmp_path, monkeypatch):
    worker, clock, bars = logged_worker(tmp_path, monkeypatch, side=side)
    original_signal = legacy_signal(bars, worker.config)
    state = worker.step()
    row, evidence = assert_signal_journal_matches_state(worker, state)
    expected_signal = asdict(original_signal)
    assert row['raw_signal'] == evidence['signal_after_quality'] == expected_signal
    assert evidence['signal_after_research_policy'] == expected_signal
    assert row['pending_signal']['signal'] == expected_signal
    assert evidence['research_policy'] == {'evaluated': True, 'rejection': None}
    assert evidence['final_stage'] == 'admitted'
    assert row['rejection'] is None and state['position']['direction'] == side
    assert state['fill_count_total'] == 1
    fill = state['fills'][0]
    expected_price = (bars[-1].close + side * .0001) * (1 + side * worker.config['adverse_slippage_fraction_assumption'])
    assert fill['reason'] == 'fresh_closed_1h_signal' and fill['time_ms'] == clock['now']
    assert fill['price'] == pytest.approx(expected_price)
    assert fill['fee'] == pytest.approx(fill['qty'] * expected_price * worker.config['taker_fee_rate_assumption'])
    assert state['wallet_balance'] == pytest.approx(1000 - fill['fee'])
    assert state['position']['entry'] == fill['price'] and state['position']['qty'] == fill['qty']
    assert state['position']['signal_time'] == 600 * HOUR
    clock['now'] += 30000
    later = worker.step()
    assert later['fill_count_total'] == 1
    assert all(later['position'][key] == value for key, value in state['position'].items())
    assert len(journal_rows(worker.path.with_name('signals.jsonl'))) == 1
    assert later['signal_diagnostics'] == evidence
