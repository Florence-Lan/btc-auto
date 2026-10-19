"""The earnings gate applies at retry and fill, and never delays a stock stop."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import stock_event_risk as risk
from test_stock_swing_confirmation import worker_fixture
from test_stock_event_risk import setup_calendar, HOUR


def event_worker(tmp_path, monkeypatch, side=1, anchor=624*HOUR):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch, side)
    config, calendar_path = setup_calendar(tmp_path, anchor)
    config = {**worker.config, **config}
    worker = paper.StockAccount(path, worker.venue, 'MUUSDT', config)
    state = paper.read_json(path)
    state['rule'] = config
    paper.write_json(path, state)
    return worker, path, clock, calendar_path


@pytest.mark.parametrize('side', [1, -1])
def test_blackout_blocks_both_directions_with_auditable_event_metadata(tmp_path, monkeypatch, side):
    worker, path, _, _ = event_worker(tmp_path, monkeypatch, side)
    result = worker.step()
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['pending_signal'] is not None
    assert result['signal_status'] == 'stock_event_blocked'
    assert result['entry_blockers'] == ['stock_event_blackout']
    assert result['stock_event_risk']['event_ids'] == ['earnings']
    observation = json.loads(path.with_name('observations.jsonl').read_text().splitlines()[-1])
    assert observation['stock_event_risk']['calendar_sha256']
    assert observation['stock_event_risk']['scope'] == 'new_risk_only'


def test_pending_retry_crossing_blackout_cannot_reuse_earlier_approval_after_restart(tmp_path, monkeypatch):
    worker, path, clock, _ = event_worker(tmp_path, monkeypatch,
        anchor=624*HOUR+2*60_000)
    clock['volume'] = 0
    waiting = worker.step()
    assert waiting['pending_signal'] and waiting['stock_event_risk']['allowed']
    clock.update(now=clock['now']+300_000, volume=10000)
    restarted = paper.StockAccount(path, worker.venue, 'MUUSDT', worker.config)
    blocked = restarted.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['pending_signal'] == waiting['pending_signal']
    assert 'stock_event_blackout' in blocked['entry_blockers']


def test_final_fill_rechecks_clock_when_depth_model_crosses_window(tmp_path, monkeypatch):
    worker, _, clock, _ = event_worker(tmp_path, monkeypatch,
        anchor=624*HOUR+61_000)
    original_fill = paper.book_fill
    def slow_depth(*args, **kwargs):
        result = original_fill(*args, **kwargs)
        clock['now'] += 2000
        return result
    monkeypatch.setattr(paper, 'book_fill', slow_depth)
    blocked = worker.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['stock_event_risk']['phase'] == 'fill'
    assert blocked['stock_event_risk']['event_ids'] == ['earnings']
    assert blocked['entry_blockers'] == ['stock_event_blackout']
    assert blocked['wallet_balance'] == 1000 and blocked['fees_paid'] == 0


@pytest.mark.parametrize('fault', ['blackout', 'missing', 'expired', 'bad_policy'])
def test_calendar_fault_or_blackout_never_prevents_protective_stop(tmp_path, monkeypatch, fault):
    worker, path, clock, calendar_path = event_worker(tmp_path, monkeypatch)
    if fault == 'missing': calendar_path.unlink()
    if fault == 'expired':
        raw = json.loads(calendar_path.read_text())
        raw['valid_until_ms'] = clock['now']
        calendar_path.write_text(json.dumps(raw))
    if fault == 'bad_policy':
        worker.config[risk.POLICY_KEY] = None
    state = paper.read_json(path)
    state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
        'entry_time': clock['now']-5000, 'signal_time': clock['now']-5000,
        'initial_stop': 98, 'stop': 111, 'entry_fee': 0, 'margin': 10, 'funding': 0}
    state['position_history'] = [{'time_ms': clock['now']-5000, 'signed_qty': 1}]
    paper.write_json(path, state)
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
    assert result['fill_count_total'] == 1
    assert not result['stock_event_risk']['allowed']
    assert result['wallet_balance'] == pytest.approx(
        1000+result['realized_pnl']-result['fees_paid']+result['funding_pnl'])


def test_absent_policy_preserves_original_entry_and_does_not_load_calendar(tmp_path, monkeypatch):
    worker, _, _ = worker_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(risk, 'evaluate', lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError('disabled rule read calendar')))
    result = worker.step()
    assert result['position'] is not None and result['fill_count_total'] == 1
    assert result['stock_event_risk'] is None


def test_event_only_migration_preserves_pending_consumption_inventory_and_finances(tmp_path, monkeypatch):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch)
    original = worker.config
    event_config, _ = setup_calendar(tmp_path)
    latest = {**original, **event_config, 'candidate_id': 'event_guard_only'}
    state = paper.read_json(path)
    state.update(wallet_balance=997., realized_pnl=-2., fees_paid=1., funding_pnl=0.,
        fills=[{'sequence': 1}], fill_count_total=1, pending_signal={
            'boundary_ms': 600*HOUR, 'expires_at_ms': 601*HOUR, 'signal': {'direction': 1}},
        signal_active_after_ms=599*HOUR, last_signal_time_ms=600*HOUR,
        last_signal_result='signal', cooldown_until_ms=clock['now']+30_000,
        position={'entry': 100., 'qty': 1., 'direction': 1, 'stop': 98.},
        position_history=[{'time_ms': 600*HOUR, 'signed_qty': 1.}])
    paper.write_json(path, state)
    monkeypatch.setattr(paper, 'stock_config', lambda *_: latest)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
            'stock_rule_revision_scope': 'stock_event_risk_only'}
    manifest = {}
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    migrated = paper.read_json(path)
    for key in ('created_at_utc', 'start_ms', 'wallet_balance', 'realized_pnl', 'fees_paid',
                'funding_pnl', 'fills', 'fill_count_total', 'position', 'position_history',
                'signal_active_after_ms', 'last_signal_time_ms', 'last_signal_result',
                'pending_signal', 'cooldown_until_ms'):
        assert migrated[key] == state[key]
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    assert len(manifest['rule_revisions']) == 1
    monkeypatch.setattr(paper, 'stock_config', lambda *_: {**latest, 'risk_fraction_per_trade': .01})
    with pytest.raises(ValueError, match='cannot alter signals, exits or risk'):
        paper.synchronize_stock_rules(plan, tmp_path, manifest)
