"""Real execution-path checks with deterministic public fixtures; no network."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import run_stock_early_recovery_shadow as shadow
import stock_early_recovery as recovery
import stock_early_recovery_timing as timing
from test_stock_early_recovery import candles, config as mechanism_config, end
from test_stock_swing_confirmation import worker_fixture

HOUR = shadow.HOUR


def initialized(tmp_path):
    spec, rules, fingerprint = shadow.load_design()
    spec = {**spec, 'effective_from_ms': 600 * HOUR - 20000}
    fingerprint = {**fingerprint, 'experiment_config_sha256': shadow.digest(spec)}
    output = tmp_path / 'new_experiment'
    output.mkdir()
    manifest = shadow.prepare(output, spec, rules, fingerprint, 600 * HOUR - 10000)
    return output, spec, rules, fingerprint, manifest


def recovery_worker(tmp_path, monkeypatch, *, short=False, cost='normal'):
    original, _, clock = worker_fixture(tmp_path / 'source', monkeypatch)
    _, rules, _ = shadow.load_design()
    cfg = rules['MUUSDT']['candidate_' + cost]
    bars = [replace(bar, time_ms=(540 + i) * HOUR) for i, bar in enumerate(candles(short=short))]
    clock.update(price=bars[-1].close, signal_error=False)
    original_get = original.venue.get.side_effect
    def get(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '1h':
            if clock['signal_error']:
                raise RuntimeError('Fixture hourly outage')
            return [[b.time_ms, b.open, b.high, b.low, b.close, 10000, b.time_ms + HOUR - 1] for b in bars]
        if endpoint == 'premiumIndex':
            return {'time': clock['now'], 'markPrice': str(clock['price']),
                'indexPrice': str(clock['price']), 'lastFundingRate': '0', 'nextFundingTime': clock['now'] + HOUR}
        if endpoint == 'depth':
            return {'T': clock['now'], 'bids': [[str(clock['price'] - .0001), '1000']],
                'asks': [[str(clock['price'] + .0001), '1000']]}
        return original_get(endpoint, params)
    original.venue.get.side_effect = get
    monkeypatch.setattr(paper.stock_event_risk, 'evaluate', lambda *a, **k: {
        'enabled': True, 'allowed': True, 'reasons': [], 'status': 'clear'})
    path = tmp_path / 'candidate' / 'state.json'
    inception = 600 * HOUR - 10000
    paper.write_json(path, paper.initial_stock('MUUSDT', inception, cfg))
    worker = shadow.RecoveryAccount(path, original.venue, 'MUUSDT', cfg, activated_at_ms=inception)
    return worker, clock, bars


def test_design_changes_only_entry_mechanism_and_preserves_symbol_risk():
    spec, rules, fingerprint = shadow.load_design()
    assert len(rules) == 3 and len(fingerprint['source_sha256']) >= 40
    assert shadow.check_source_rules(spec, rules)
    permitted = {'signal_family', 'early_recovery_policy', 'entry_timing_policy',
        'candidate_id', 'research_variant', 'swing_confirmation_policy'}
    for symbol, arms in rules.items():
        for cost in shadow.COSTS:
            baseline, candidate = arms['baseline_' + cost], arms['candidate_' + cost]
            assert {k: v for k, v in baseline.items() if k not in permitted} == {
                k: v for k, v in candidate.items() if k not in permitted}
            assert candidate['min_stop_fraction'] == (.005 if symbol == 'SNDKUSDT' else .02)
        for kind in ['baseline', 'candidate']:
            for key in ['taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption']:
                assert arms[kind + '_double'][key] == 2 * arms[kind + '_normal'][key]


def test_epoch_resume_is_exact_and_corruption_never_resets(tmp_path):
    output, spec, rules, fingerprint, manifest = initialized(tmp_path)
    assert len(list(output.glob('*/*/state.json'))) == 12
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert shadow.prepare(output, spec, rules, fingerprint, None, True) == manifest
    assert all(p.read_bytes() == b for p, b in before.items())
    state_path = output / 'MUUSDT/candidate_normal/state.json'
    state = paper.read_json(state_path)
    state['wallet_balance'] += 1
    paper.write_json(state_path, state)
    with pytest.raises(ValueError, match='cash identity'):
        shadow.prepare(output, spec, rules, fingerprint, None, True)
    assert paper.read_json(state_path)['wallet_balance'] == 1001


def test_changed_source_or_missing_journal_cannot_reinitialize(tmp_path):
    output, spec, rules, fingerprint, _ = initialized(tmp_path)
    bad = deepcopy(fingerprint)
    bad['source_sha256']['scripts/stock_early_recovery.py'] = '0' * 64
    with pytest.raises(ValueError, match='changed'):
        shadow.prepare(output, spec, rules, bad, None, True)
    (output / 'MUUSDT/candidate_normal/fills.jsonl').unlink()
    with pytest.raises(shadow.LedgerIntegrityError, match='missing'):
        shadow.prepare(output, spec, rules, fingerprint, None, True)


@pytest.mark.parametrize('short', [False, True])
@pytest.mark.parametrize('cost', ['normal', 'double'])
def test_candidate_fills_real_book_and_original_protective_stop(tmp_path, monkeypatch, short, cost):
    worker, clock, _ = recovery_worker(tmp_path, monkeypatch, short=short, cost=cost)
    profiles = paper.StockAccount.step.__globals__['stock_swing_profiles']
    original_timing = paper.StockAccount.step.__globals__['stock_entry_timing']
    state = worker.step()
    assert state['position'] is not None, state['entry_blockers']
    pos = state['position']
    assert pos['direction'] == (-1 if short else 1)
    assert state['fills'][0]['time_ms'] >= clock['now']
    assert state['fills'][0]['price'] != worker.signal_bars[-1][4]
    assert state['fees_paid'] == pytest.approx(pos['qty'] * pos['entry'] * worker.config['taker_fee_rate_assumption'])
    assert abs(1000 + state['realized_pnl'] - state['fees_paid'] + state['funding_pnl'] - state['wallet_balance']) < 1e-7
    # Evidence is durably journaled at first observation.
    row = json.loads(worker.path.with_name('signals.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert row['first_seen_at_ms'] >= row['boundary_ms'] and row['first_seen_lag_ms'] >= 60000
    assert paper.StockAccount.step.__globals__['stock_swing_profiles'] is profiles
    assert paper.StockAccount.step.__globals__['stock_entry_timing'] is original_timing
    clock.update(now=clock['now'] + 300000, price=pos['stop'] - pos['direction'] * .1, signal_error=True)
    monkeypatch.setattr(paper.stock_event_risk, 'evaluate', lambda *a, **k: {
        'enabled': True, 'allowed': False, 'reasons': ['stock_event_blackout'], 'status': 'blackout'})
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
    assert 'signal' in result['errors']


def test_boundary_invalidation_cancels_retry_until_new_hour(tmp_path, monkeypatch):
    worker, clock, bars = recovery_worker(tmp_path, monkeypatch)
    _, detail = recovery.candidate_with_diagnostics(bars, worker.config, asof_ms=600 * HOUR)
    clock['price'] = detail['reclaim_level'] - .001
    state = worker.step()
    assert state['position'] is None and state['pending_signal'] is None
    assert state['last_signal_result'] == 'entry_timing_cancelled'
    clock.update(now=clock['now'] + 30000, price=bars[-1].close)
    assert worker.step()['position'] is None


@pytest.mark.parametrize('short', [False, True])
def test_actual_fill_cannot_move_stop_inside_frozen_structure(short):
    bars, cfg = candles(short=short), mechanism_config()
    cfg.update(signal_timeframe='1h', entry_gap_atr=.5, entry_timing_policy=timing.POLICY.copy())
    sig = recovery.candidate_signal(bars, cfg, asof_ms=end(bars))
    trigger = timing.build_trigger(bars, sig, cfg, end(bars), observed_at_ms=end(bars))
    side = sig.direction
    trigger['structural_invalidation_level'] = paper.signals.initial_stop(sig.close, side, sig.atr, cfg) + side * .01
    decision = timing.evaluate(trigger, sig, cfg, reference_price=sig.close, mark_price=sig.close,
        fill_price=sig.close + side * .1)
    assert not decision['allowed'] and 'recovery_fill_structure_outside_original_stop' in decision['reasons']
    assert not decision['cancel_pending']


def test_exact_inception_boundary_is_never_an_entry(tmp_path, monkeypatch):
    worker, clock, _ = recovery_worker(tmp_path, monkeypatch)
    state = paper.read_json(worker.path)
    state.update(start_ms=600 * HOUR, signal_active_after_ms=600 * HOUR)
    paper.write_json(worker.path, state)
    worker.activated_at_ms = 600 * HOUR
    assert worker.step()['position'] is None
    row = json.loads(worker.path.with_name('signals.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert row['signal_diagnostics']['admission']['reasons'] == ['signal_not_strictly_post_inception']
