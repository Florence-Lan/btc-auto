import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import stock_trend_exits as trend
from validate_confirmed_trend_strategy import assess
from test_stock_trend_experiment import worker_fixture


def common_worker(tmp_path, monkeypatch):
    candidate, path, clock, signal = worker_fixture(tmp_path, monkeypatch)
    state = paper.read_json(path)
    state['trend_exit_active_after_ms'] = candidate.activated_at_ms
    paper.write_json(path, state)
    worker = paper.StockAccount(path, candidate.venue, 'MUUSDT', candidate.config)
    return worker, path, clock, signal


def test_main_stock_account_runs_confirmed_exit_and_preserves_cash_identity(tmp_path, monkeypatch):
    worker, path, clock, _ = common_worker(tmp_path, monkeypatch)
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'trend_reversal_exit'
    assert result['trend_exit_status']['status'] == 'flat'
    assert result['wallet_balance'] == pytest.approx(1000 + result['realized_pnl'] - result['fees_paid'] + result['funding_pnl'])
    assert result['cooldown_until_ms'] == clock['now'] + 3 * worker.signal_interval_ms


def test_main_stop_precedes_confirmed_trend_and_missing_signal_does_not_block_exit(tmp_path, monkeypatch):
    worker, path, _, _ = common_worker(tmp_path, monkeypatch)
    state = paper.read_json(path)
    state['position']['stop'] = 101
    paper.write_json(path, state)
    responses = [(label, None, 'outage') if label == 'signal' else (label, payload, error)
                 for label, payload, error in worker.market_data()]
    result = worker.step(responses)
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'


def test_exit_migration_preserves_cash_inventory_pending_signal_and_sets_epoch(tmp_path, monkeypatch):
    worker, path, clock, _ = common_worker(tmp_path, monkeypatch)
    state = paper.read_json(path)
    state['rule'] = copy.deepcopy(worker.config)
    state['rule'].pop('trend_exit_policy')
    state['pending_signal'] = {'boundary_ms': 123, 'expires_at_ms': 456}
    paper.write_json(path, state)
    monkeypatch.setattr(paper, 'stock_config', lambda *_: worker.config)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
            'stock_rule_revision_scope': 'profit_exits_only'}
    manifest = {}
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    after = paper.read_json(path)
    for key in ('wallet_balance', 'position', 'fills', 'fees_paid', 'funding_pnl', 'pending_signal', 'start_ms'):
        assert after[key] == state[key]
    assert after['trend_exit_active_after_ms'] == clock['now']
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    assert len(manifest['rule_revisions']) == 1
    # Existing opposing bars before this migration cannot close the position.
    migrated = paper.StockAccount(path, worker.venue, 'MUUSDT', worker.config).step()
    assert migrated['position'] is not None


def test_exit_only_migration_rejects_an_entry_change(tmp_path, monkeypatch):
    worker, path, _, _ = common_worker(tmp_path, monkeypatch)
    cfg = {**worker.config, 'entry_direction': 'long'}
    monkeypatch.setattr(paper, 'stock_config', lambda *_: cfg)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='entry or risk'):
        paper.synchronize_stock_rules({'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
            'stock_rule_revision_scope': 'profit_exits_only'}, tmp_path, {})
    assert path.read_bytes() == before


def test_improvement_is_distinct_from_profitability_and_double_cost_is_required():
    control = {c: {'total_return_pct': -2, 'max_drawdown_pct': 2, 'closed_positions': 12} for c in ('cost1', 'cost2')}
    candidate = {c: {'total_return_pct': -1, 'max_drawdown_pct': 2, 'closed_positions': 12} for c in ('cost1', 'cost2')}
    result = assess(control, candidate)
    assert result['paper_exit_improvement_passed']
    assert not result['positive_at_both_costs']
    assert not result['future_profitability_proven']
    candidate['cost2']['total_return_pct'] = -3
    assert not assess(control, candidate)['paper_exit_improvement_passed']
