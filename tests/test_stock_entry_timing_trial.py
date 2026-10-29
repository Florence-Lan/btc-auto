"""Real book/ledger paths for an isolated entry comparison, with no network."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import run_stock_entry_timing_trial as trial
import stock_entry_timing_trial_admission as admission
import stock_entry_timing_candidates as candidates
from test_stock_swing_confirmation import worker_fixture
from test_stock_macro_shadow import macro_permit

HOUR = trial.HOUR


def pullback_bars(short=False):
    prices = [100 + i * .1 for i in range(500)]
    prices[-2:] = [148.9, 149.4]
    result, previous = [], prices[0]
    for i, close in enumerate(prices):
        bar = paper.signals.Candle((100 + i) * HOUR, previous,
            max(previous, close) + .1, min(previous, close) - .1, close, 10000)
        if short:
            bar = paper.signals.Candle(bar.time_ms, 300 - bar.open, 300 - bar.low,
                                      300 - bar.high, 300 - bar.close, bar.volume)
        result.append(bar)
        previous = close
    return result


def worker(tmp_path, monkeypatch, short=False, cost='normal', variant='turn_pullback', boundary=None):
    original, _, clock = worker_fixture(tmp_path / 'source', monkeypatch)
    _, rules, _ = trial.load_design()
    cfg = rules['MUUSDT'][variant + '_' + cost]
    bars = pullback_bars(short)
    clock.update(price=bars[-1].close, hourly_error=False)
    original_get = original.venue.get.side_effect
    def get(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '1h':
            if clock['hourly_error']:
                raise RuntimeError('Fixture hourly failure')
            return [[b.time_ms, b.open, b.high, b.low, b.close, b.volume, b.time_ms + HOUR - 1] for b in bars]
        if endpoint == 'premiumIndex':
            return {'time': clock['now'], 'markPrice': str(clock['price']),
                'indexPrice': str(clock['price']), 'lastFundingRate': '0', 'nextFundingTime': clock['now'] + HOUR}
        if endpoint == 'depth':
            return {'T': clock['now'], 'bids': [[str(clock['price'] - .0001), '1000']],
                'asks': [[str(clock['price'] + .0001), '1000']]}
        return original_get(endpoint, params)
    original.venue.get.side_effect = get
    monkeypatch.setattr(trial.macro, 'evaluate', lambda cfg, symbol, at, **kwargs: macro_permit(at, boundary))
    monkeypatch.setattr(paper.stock_event_risk, 'evaluate', lambda cfg, symbol, at, **kwargs: {
        'enabled': True, 'allowed': True, 'status': 'clear', 'reasons': [], 'checked_at_ms': at,
        'valid_until_ms': at + HOUR, 'coverage_end_ms': at + 24 * HOUR, 'next_event': None})
    path = tmp_path / 'arm' / 'state.json'
    start = 600 * HOUR - 10000
    state = paper.initial_stock('MUUSDT', start, cfg)
    state['signal_active_after_ms'] = start
    paper.write_json(path, state)
    for name in ('entries.jsonl', 'fills.jsonl', 'funding.jsonl', 'closed_trades.jsonl'):
        path.with_name(name).write_text('', encoding='utf-8')
    result = trial.TimingAccount(path, original.venue, 'MUUSDT', cfg, activated_at_ms=start)
    result.macro_context_sha256 = 'a' * 64
    return result, clock, bars


def initialized(tmp_path):
    spec, rules, fingerprint = trial.load_design()
    spec = {**spec, 'effective_from_ms': 600 * HOUR - 20000}
    fingerprint = {**fingerprint, 'experiment_config_sha256': trial.digest(spec)}
    output = tmp_path / 'isolated'
    output.mkdir()
    manifest = trial.prepare(output, spec, rules, fingerprint, 600 * HOUR - 10000)
    return output, spec, rules, fingerprint, manifest


def test_design_has_18_accounts_only_declared_entry_differences_and_equal_shared_guard():
    spec, rules, _ = trial.load_design()
    assert len(trial.ARMS) == 6 and trial.check_source_rules(spec, rules)
    allowed = {'entry_timing_variant', 'entry_timing_candidate_policy', 'candidate_id',
               'research_variant', 'entry_timing_policy'}
    for symbol, arms in rules.items():
        for cost in trial.COSTS:
            baseline = arms['baseline_' + cost]
            for variant in trial.VARIANTS[1:]:
                cfg = arms[variant + '_' + cost]
                assert {k: v for k, v in cfg.items() if k not in allowed} == {
                    k: v for k, v in baseline.items() if k not in allowed}
                assert cfg[trial.macro.POLICY_KEY] == baseline[trial.macro.POLICY_KEY]
                assert cfg['min_stop_fraction'] == (.005 if symbol == 'SNDKUSDT' else .02)
        for variant in trial.VARIANTS:
            for key in ('taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'):
                assert arms[variant + '_double'][key] == 2 * arms[variant + '_normal'][key]


def test_prepare_resume_is_read_only_and_never_repairs_corrupted_cash(tmp_path):
    output, spec, rules, fingerprint, manifest = initialized(tmp_path)
    assert len(list(output.glob('*/*/state.json'))) == 18
    files = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert trial.prepare(output, spec, rules, fingerprint, None, True) == manifest
    assert all(p.read_bytes() == raw for p, raw in files.items())
    p = output / 'MUUSDT/turning_normal/state.json'
    bad = paper.read_json(p)
    bad['wallet_balance'] += 1
    paper.write_json(p, bad)
    with pytest.raises(ValueError, match='cash identity'):
        trial.prepare(output, spec, rules, fingerprint, None, True)
    assert paper.read_json(p)['wallet_balance'] == 1001


@pytest.mark.parametrize('short', [False, True])
@pytest.mark.parametrize('cost', ['normal', 'double'])
def test_fresh_pullback_real_execution_and_independent_stop(tmp_path, monkeypatch, short, cost):
    account, clock, _ = worker(tmp_path, monkeypatch, short, cost)
    original_profiles, original_timing = paper.stock_swing_profiles, paper.stock_entry_timing
    state = account.step()
    pos = state['position']
    assert pos is not None, state['entry_blockers']
    assert pos['direction'] == (-1 if short else 1)
    assert state['signal_diagnostics']['entry_timing_candidate']['signal_kind'] == 'trend_pullback'
    assert state['fills'][0]['entry_timing_variant'] == 'turn_pullback'
    assert state['fills'][0]['macro_context_sha256'] == 'a' * 64
    assert state['fees_paid'] == pytest.approx(pos['entry'] * pos['qty'] * account.config['taker_fee_rate_assumption'])
    trial.validate_ledger(state, account.config, 'MUUSDT', account.activated_at_ms)
    trial.validate_financial_journals(account.path.parent, state)
    row = json.loads(account.path.with_name('signals.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert row['first_seen_lag_ms'] >= 60000 and row['first_seen_at_ms'] >= row['boundary_ms']
    assert paper.stock_swing_profiles is original_profiles and paper.stock_entry_timing is original_timing
    clock.update(now=clock['now'] + 300000, price=pos['stop'] - pos['direction'] * .1, hourly_error=True)
    monkeypatch.setattr(trial.macro, 'evaluate', lambda cfg, symbol, at, **kw: macro_permit(at, 0))
    result = account.step()
    assert result['position'] is None and result['fills'][-1]['reason'] == 'protective_stop'
    trial.validate_financial_journals(account.path.parent, result)


def test_baseline_and_turning_do_not_invent_donchian_from_pullback(tmp_path, monkeypatch):
    for variant in ('baseline', 'turning'):
        account, _, _ = worker(tmp_path / variant, monkeypatch, variant=variant)
        result = account.step()
        assert result['position'] is None and result['fill_count_total'] == 0


def test_macro_blackout_blocks_fresh_entry_without_charging_cost(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch, boundary=0)
    result = account.step()
    assert result['position'] is None and result['wallet_balance'] == 1000 and result['fees_paid'] == 0
    assert result['entry_blockers'] == ['stock_macro_event_blackout']


def test_final_fill_rechecks_macro_after_depth(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    threshold = clock['now'] + 1000
    monkeypatch.setattr(trial.macro, 'evaluate', lambda cfg, symbol, at, **kw: macro_permit(at, threshold))
    original = paper.book_fill
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock['now'] += 2000
        return result
    monkeypatch.setattr(paper, 'book_fill', delayed)
    state = account.step()
    assert state['position'] is None and state['fees_paid'] == 0 and state['fill_count_total'] == 0


@pytest.mark.parametrize('short', [False, True])
def test_frozen_ema_reclaim_cancels_and_actual_fill_preserves_original_structure(tmp_path, monkeypatch, short):
    account, _, bars = worker(tmp_path, monkeypatch, short)
    cfg = account.config
    indicators = paper.signals.compute_indicators(bars, cfg)
    sig, detail = candidates.signal_with_diagnostics(bars, indicators, 499, cfg)
    trigger = admission.build_trigger(bars, sig, cfg, 600 * HOUR, observed_at_ms=600 * HOUR)
    assert trigger['reclaim_level'] == detail['entry_timing_candidate']['frozen_reclaim']['ema_fast']
    at_level = admission.evaluate(trigger, sig, cfg, reference_price=trigger['reclaim_level'], mark_price=sig.close)
    assert not at_level['allowed'] and at_level['cancel_pending']
    initial_stop = paper.signals.initial_stop(sig.close, sig.direction, sig.atr, cfg)
    trigger['structural_invalidation_level'] = initial_stop + sig.direction * .01
    adverse = admission.evaluate(trigger, sig, cfg, reference_price=sig.close, mark_price=sig.close,
                                 fill_price=sig.close + sig.direction * .05)
    assert not adverse['allowed']
    assert 'timing_trial_fill_structure_outside_original_stop' in adverse['reasons']
    assert not adverse['cancel_pending']


def test_exact_inception_boundary_cannot_be_backfilled(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    state = paper.read_json(account.path)
    state.update(start_ms=600 * HOUR, signal_active_after_ms=600 * HOUR)
    paper.write_json(account.path, state)
    account.activated_at_ms = 600 * HOUR
    assert account.step()['position'] is None
