"""Independent integration audit: real execution methods, local fixtures only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
with patch('dotenv.load_dotenv', return_value=False):
    import run_parallel_simulation as paper
    import run_stock_early_recovery_shadow as trial
import stock_entry_timing as original_timing
import stock_early_recovery as recovery
import stock_early_recovery_timing as timing
from test_stock_swing_confirmation import worker_fixture
from test_stock_early_recovery import candles, config, end

HOUR = 3_600_000


@pytest.fixture(autouse=True)
def no_public_requests(monkeypatch):
    monkeypatch.setattr(paper.requests.Session, 'request', Mock(side_effect=AssertionError('No public requests in independent audit')))
    monkeypatch.setattr(paper.subprocess, 'run', Mock(side_effect=AssertionError('No public curl in independent audit')))


def baseline_pair(tmp_path, monkeypatch, *, side=1, cost=1):
    source, _, clock = worker_fixture(tmp_path / 'fixture', monkeypatch, side=side)
    cfg = deepcopy(source.config)
    cfg['entry_timing_policy'] = dict(original_timing.POLICY)
    for field in ('taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'):
        cfg[field] *= cost
    getter = source.venue.get.side_effect

    def get(endpoint, params=None):
        payload = getter(endpoint, params)
        if endpoint == 'klines' and params['interval'] == '1h':
            payload = deepcopy(payload)
            for row in payload:
                if side == 1:
                    row[3] = float(row[4]) - .2
                else:
                    row[2] = float(row[4]) + .2
        if endpoint == 'fundingRate' and clock.get('audit_funding'):
            return [clock['audit_funding']]
        if endpoint == 'premiumIndex' and 'audit_price' in clock:
            payload = {**payload, 'markPrice': str(clock['audit_price']), 'indexPrice': str(clock['audit_price'])}
        if endpoint == 'depth' and 'audit_price' in clock:
            payload = {**payload, 'bids': [[str(clock['audit_price'] - .0001), '1000']],
                       'asks': [[str(clock['audit_price'] + .0001), '1000']]}
        return payload

    source.venue.get.side_effect = get
    inception = 600 * HOUR - 10_000
    left = tmp_path / 'original' / 'state.json'
    right = tmp_path / 'isolated' / 'state.json'
    for path in (left, right):
        paper.write_json(path, paper.initial_stock('MUUSDT', inception, cfg))
    original = paper.StockAccount(left, source.venue, 'MUUSDT', cfg)
    wrapped = trial.RecoveryAccount(right, source.venue, 'MUUSDT', cfg, activated_at_ms=inception)
    # Keep the simulated receipt clock identical; no quote or exchange clock is cached.
    monkeypatch.setattr(original, 'now_ms', lambda: clock['now'])
    monkeypatch.setattr(wrapped, 'now_ms', lambda: clock['now'])
    return original, wrapped, clock


def assert_financial_and_execution_equal(left, right):
    for key in ('wallet_balance', 'equity', 'position_qty', 'realized_pnl', 'fees_paid', 'funding_pnl',
                'fill_count_total', 'fills', 'position', 'position_history', 'settled_funding',
                'pending_signal', 'swing_confirmation', 'entry_checks', 'entry_blockers',
                'signal_status', 'cooldown_until_ms', 'risk_halted', 'rule'):
        assert left[key] == right[key], key


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('cost', [1, 2])
def test_function_clone_baseline_has_identical_entry_funding_and_protective_close(tmp_path, monkeypatch, side, cost):
    original, wrapped, clock = baseline_pair(tmp_path, monkeypatch, side=side, cost=cost)
    globals_before = (paper.stock_swing_profiles, paper.stock_entry_timing, paper.journal, paper.StockAccount.step)
    first, second = original.step(), wrapped.step()
    assert first['position'] and second['position']
    assert_financial_and_execution_equal(first, second)
    assert (paper.stock_swing_profiles, paper.stock_entry_timing, paper.journal, paper.StockAccount.step) == globals_before
    entry = first['position']
    clock['audit_funding'] = {'symbol': 'MUUSDT', 'fundingTime': entry['entry_time'] + 10_000,
                             'fundingRate': '.0001', 'markPrice': str(entry['entry'])}
    clock['now'] += 30_000
    clock['audit_price'] = entry['stop'] - side * .1
    closed_left, closed_right = original.step(), wrapped.step()
    assert closed_left['position'] is None and closed_right['position'] is None
    assert closed_left['fills'][-1]['reason'] == closed_right['fills'][-1]['reason'] == 'protective_stop'
    assert closed_left['funding_pnl'] != 0
    assert_financial_and_execution_equal(closed_left, closed_right)
    for name in ('fills.jsonl', 'funding.jsonl', 'closed_trades.jsonl'):
        assert original.path.with_name(name).read_bytes() == wrapped.path.with_name(name).read_bytes()
    log = json.loads(wrapped.path.with_name('signals.jsonl').read_text().splitlines()[0])
    assert log['first_seen_at_ms'] == first['fills'][0]['time_ms']
    assert log['first_seen_at_ms'] >= log['boundary_ms']
    assert first['fills'][0]['time_ms'] >= log['first_seen_at_ms']
    assert (paper.stock_swing_profiles, paper.stock_entry_timing, paper.journal, paper.StockAccount.step) == globals_before


def recovery_config():
    return {**config(), 'signal_timeframe': '1h', 'entry_gap_atr': .5,
            'entry_timing_policy': dict(timing.POLICY)}


@pytest.mark.parametrize('short', [False, True])
def test_quality_rejection_is_not_relabelled_no_raw_candidate(short):
    bars = candles(short=short)
    # A deep structural wick is a genuine reclaim candidate, but its existing stop cannot cover it.
    index = len(bars) - 5
    bars[index] = replace(bars[index], **({'high': 104.0} if short else {'low': 96.0}))
    cfg = recovery_config()
    signal, pure = recovery.candidate_with_diagnostics(bars, cfg, asof_ms=end(bars))
    assert signal is None and pure['signal_before_confirmation'] is not None
    assert pure['admission']['stage'] == 'quality_rejected'
    actual_signal, wrapped = trial.RecoveryProfiles(end(bars) - 1).signal_with_diagnostics(bars, {}, len(bars) - 1, cfg)
    assert actual_signal is None
    for key in ('signal_before_confirmation', 'signal_after_confirmation', 'signal_after_quality', 'entry_quality', 'admission'):
        assert wrapped[key] == pure[key]


@pytest.mark.parametrize('short', [False, True])
def test_adverse_actual_fill_rechecks_frozen_structure_without_widening_stop(short):
    bars = candles(short=short)
    bars[-5] = replace(bars[-5], **({'high': 102.5} if short else {'low': 97.5}))
    cfg = recovery_config()
    signal, diagnostic = recovery.candidate_with_diagnostics(bars, cfg, asof_ms=end(bars))
    assert signal is not None
    trigger = timing.build_trigger(bars, signal, cfg, end(bars), observed_at_ms=end(bars) + 1000)
    trigger_before = deepcopy(trigger)
    entry = signal.close + signal.direction * .4 * signal.atr
    result = timing.evaluate(trigger, signal, cfg, reference_price=entry, mark_price=entry, fill_price=entry)
    assert not result['allowed']
    assert 'recovery_fill_structure_outside_original_stop' in result['reasons']
    assert not any('gap_exceeded' in reason or 'extension_exceeded' in reason for reason in result['reasons'])
    assert trigger == trigger_before


@pytest.mark.parametrize('short', [False, True])
def test_quote_boundary_equality_cancels_both_directions(short):
    bars = candles(short=short)
    cfg = recovery_config()
    signal = recovery.candidate_signal(bars, cfg, asof_ms=end(bars))
    trigger = timing.build_trigger(bars, signal, cfg, end(bars), observed_at_ms=end(bars))
    result = timing.evaluate(trigger, signal, cfg, reference_price=trigger['reclaim_level'], mark_price=signal.close)
    assert not result['allowed'] and result['cancel_pending']
    assert 'recovery_reference_invalidated' in result['reasons']


def test_recovery_pending_rebuild_uses_original_prefix_and_never_later_prices():
    bars = candles()
    cfg = recovery_config()
    signal = recovery.candidate_signal(bars, cfg, asof_ms=end(bars))
    pending = {'signal': vars(signal), 'boundary_ms': end(bars)}
    expected = timing.build_trigger(bars, signal, cfg, end(bars), observed_at_ms=end(bars))
    later = [replace(bars[-1], time_ms=end(bars), open=1., high=1., low=1., close=1.)]
    assert timing.rebuild_trigger(bars + later, pending, cfg, observed_at_ms=end(bars) + HOUR) == expected
    with pytest.raises(ValueError, match='prefix_mismatch'):
        timing.build_trigger(bars + later, signal, cfg, end(bars), observed_at_ms=end(bars) + HOUR)


@pytest.mark.parametrize('offset', [0, 1])
def test_first_forward_signal_must_close_strictly_after_inception(offset):
    bars = candles()
    cfg = recovery_config()
    _, diagnostic = trial.RecoveryProfiles(end(bars) + offset).signal_with_diagnostics(bars, {}, len(bars) - 1, cfg)
    assert diagnostic['admission']['reasons'] == ['signal_not_strictly_post_inception']


@pytest.mark.parametrize('short', [False, True])
@pytest.mark.parametrize('cost', ['normal', 'double'])
def test_candidate_stop_survives_clock_and_event_unavailability(tmp_path, monkeypatch, short, cost):
    from test_stock_early_recovery_shadow import recovery_worker
    worker, clock, _ = recovery_worker(tmp_path, monkeypatch, short=short, cost=cost)
    monkeypatch.setattr(worker, 'now_ms', lambda: clock['now'])
    entered = worker.step()
    assert entered['position']
    position = entered['position']
    getter = worker.venue.get.side_effect

    def no_clock(endpoint, params=None):
        if endpoint == 'time':
            raise RuntimeError('Fixture exchange clock unavailable')
        return getter(endpoint, params)

    worker.venue.get.side_effect = no_clock
    clock.update(now=clock['now'] + 30_000, price=position['stop'] - position['direction'] * .1,
                 signal_error=True)
    monkeypatch.setattr(paper.stock_event_risk, 'evaluate', lambda *a, **k: {
        'enabled': True, 'allowed': False, 'reasons': ['stock_event_calendar_unavailable'], 'status': 'unavailable'})
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
    assert result['fill_count_total'] == entered['fill_count_total'] + 1
    assert result['status'] == 'degraded' and 'clock' in result['errors']
    assert result['wallet_balance'] == pytest.approx(
        1000 + result['realized_pnl'] - result['fees_paid'] + result['funding_pnl'])


@pytest.mark.parametrize('short', [False, True])
def test_candidate_rechecks_new_event_blackout_before_financial_fill(tmp_path, monkeypatch, short):
    from test_stock_early_recovery_shadow import recovery_worker
    worker, _, _ = recovery_worker(tmp_path, monkeypatch, short=short)
    phases = []

    def event_gate(state, phase, reference_ms=None):
        phases.append(phase)
        allowed = phase != 'fill'
        outcome = {'enabled': True, 'allowed': allowed, 'reasons': [] if allowed else ['stock_event_blackout'],
                   'status': 'clear' if allowed else 'blackout'}
        state['stock_event_risk'] = outcome
        return outcome

    monkeypatch.setattr(worker, 'record_stock_event_risk', event_gate)
    result = worker.step()
    assert 'fill' in phases
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['fees_paid'] == 0 and result['wallet_balance'] == 1000
    assert result['signal_status'] == 'stock_event_blocked'
    assert result['entry_blockers'] == ['stock_event_blackout']
