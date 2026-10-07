import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_parallel_simulation as runner
from test_parallel_simulation import fast_stock_fixture


def test_stock_clock_skew_does_not_relax_quote_freshness(tmp_path, monkeypatch):
    worker, path, clock, _ = fast_stock_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner.time, 'monotonic', lambda:100.)
    observed = worker.market_data(force_signal=True)
    exchange_time = clock['now']
    monkeypatch.setattr(runner, 'now_ms', lambda:exchange_time-8000)
    state = worker.step(observed)
    assert state['status'] == 'healthy'
    assert worker.now_ms() == exchange_time
    old = copy.deepcopy(observed)
    for label, payload, error in old:
        if label == 'mark':
            payload['time'] -= 61000
    with pytest.raises(ValueError, match='Stale or future'):
        worker.step(old)


def test_control_uses_its_own_candles_and_shared_execution_inputs():
    worker = Mock(symbol='SNDKUSDT', signal_timeframe='5m')
    worker.venue.get.return_value = ['control5m']
    inputs = [('book',{'bids':[['100','1']]},None),('signal',['candidate1h'],None)]
    actual = runner.comparison_responses(worker, inputs)
    assert actual[0] == inputs[0] and actual[1] == ('signal',['control5m'],None)
    assert inputs[1] == ('signal',['candidate1h'],None)


def test_latest_migration_preserves_inventory_cash_and_fills(tmp_path, monkeypatch):
    original = runner.stock_config({'stock_research_profile':'config/stock_one_r_half_atr_paper_20261006.json'}, 'SNDKUSDT')
    latest = runner.stock_config({'stock_research_profile':'config/stock_hourly_aligned_breakout_paper_20261007.json'}, 'SNDKUSDT')
    path = tmp_path/'sndk/state.json'
    state = runner.initial_stock('SNDKUSDT', 10, original)
    state.update(wallet_balance=996., realized_pnl=-3., fees_paid=2., funding_pnl=1.,
        fills=[{'sequence':1}], fill_count_total=1, pending_signal={'boundary_ms':20},
        position={'entry':100.,'qty':1.,'direction':1,'stop':98.,'initial_stop':98.,
                  'profit_exit':{'stage_done':True}}, position_history=[{'time_ms':20,'signed_qty':1.}])
    runner.write_json(path,state)
    monkeypatch.setattr(runner,'stock_config',lambda *_:latest)
    plan = {'accounts':[{'account_id':'sndk','symbol':'SNDKUSDT','state_path':str(path)}],
            'stock_rule_revision_scope':'entry_and_exit'}
    manifest = {}
    runner.synchronize_stock_rules(plan,tmp_path,manifest)
    updated = runner.read_json(path)
    for key in ('wallet_balance','realized_pnl','fees_paid','funding_pnl','fills','fill_count_total',
                'position','position_history','start_ms'):
        assert updated[key] == state[key]
    assert updated['rule'] == latest and updated['pending_signal'] is None
    runner.synchronize_stock_rules(plan,tmp_path,manifest)
    assert len(manifest['rule_revisions']) == 1


def test_switch_requires_existing_ledgers_and_does_not_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    root = tmp_path/'data/parallel_simulation/trial'
    plan = {'accounts':[{'account_id':name,'state_path':f'data/parallel_simulation/trial/{name}/state.json'}
                        for name in ('btc','mu','sndk','skhynix')]}
    with pytest.raises(RuntimeError, match='refusing to create/reset'):
        runner.require_existing_accounts(plan,root)
    assert not root.exists()
