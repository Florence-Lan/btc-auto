from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as runner
import stock_profit_exits as exits
from test_parallel_simulation import fast_stock_fixture

POLICY = {'mode': exits.MODE, 'first_target_r': 1.0, 'first_close_fraction': .5, 'trail_atr': 1.5}


def fixture(tmp_path, monkeypatch, side=1, qty=1):
    worker, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    worker.config['profit_exit_policy'] = POLICY.copy()
    worker.config['taker_fee_rate_assumption'] = .001
    state = json.loads(path.read_text())
    state['position'] = {'entry': 100, 'qty': qty, 'direction': side,
        'entry_time': clock['now'] - 5000, 'signal_time': clock['now'] - 5000,
        'initial_stop': 98 if side == 1 else 102, 'stop': 98 if side == 1 else 102,
        'entry_fee': qty * .1, 'margin': qty * 10, 'funding': qty * .02,
        'best_close_return': .5}  # Pre-activation profit must never arm the new rule.
    state.update(wallet_balance=1000 - qty * .12, fees_paid=qty*.1, funding_pnl=-qty*.02)
    state['position_history'] = [{'time_ms': state['position']['entry_time'], 'signed_qty': side * qty}]
    runner.write_json(path, state)
    clock.update(price=103 if side == 1 else 97, liquidity=100)
    original = worker.venue.get.side_effect
    def get(endpoint, params=None):
        if endpoint == 'premiumIndex':
            price = clock['price']
            return {'time': clock['now'], 'markPrice': str(price), 'indexPrice': str(price),
                    'lastFundingRate':'0', 'nextFundingTime':clock['now'] + 900000}
        if endpoint == 'depth':
            price = clock['price']
            return {'T':clock['now'], 'bids':[[str(price-.01),str(clock['liquidity'])]],
                    'asks':[[str(price+.01),str(clock['liquidity'])]]}
        return original(endpoint, params)
    worker.venue.get.side_effect = get
    return worker, path, clock, observed


@pytest.mark.parametrize('side', [1, -1])
def test_one_r_half_and_cost_protection_survive_restart(tmp_path, monkeypatch, side):
    worker, path, clock, observed = fixture(tmp_path, monkeypatch, side)
    first = worker.step()
    pos = first['position']
    assert pos['qty'] == .5
    assert pos['profit_exit']['stage_done']
    assert first['fills'][-1]['reason'] == 'one_r_partial'
    assert pos['entry_fee'] == pytest.approx(.05)
    assert pos['funding'] == pytest.approx(.01)
    be = pos['profit_exit']['break_even_quote']
    fill = be * (1-side*worker.config['adverse_slippage_fraction_assumption'])
    net = side*(fill-pos['entry']) - pos['entry_fee']/pos['qty'] - pos['funding']/pos['qty'] - fill*.001
    assert net == pytest.approx(0, abs=1e-10)
    assert side*(pos['stop']-be) >= -1e-10
    clock['now'] += 30000
    restarted = runner.StockAccount(path, worker.venue, 'MUUSDT', worker.config)
    again = restarted.step()
    assert again['position']['qty'] == .5
    assert again['fill_count_total'] == 1
    assert again['position']['profit_exit']['risk_distance'] == 2
    assert not observed.called


def test_pre_activation_peak_does_not_trigger_a_backdated_exit(tmp_path, monkeypatch):
    worker, path, clock, _ = fixture(tmp_path, monkeypatch)
    clock['price'] = 101
    result = worker.step()
    assert not result['fills']
    assert result['position']['qty'] == 1
    assert not result['position']['profit_exit']['armed']
    assert result['position']['stop'] == 98
    assert result['position']['profit_exit']['best_executable'] == pytest.approx(100.99)


def test_stage_partial_ioc_retries_only_remaining_half(tmp_path, monkeypatch):
    worker, path, clock, _ = fixture(tmp_path, monkeypatch)
    clock['liquidity'] = 2
    first = worker.step()
    assert first['position']['qty'] == .8
    assert first['position']['profit_exit']['filled_qty'] == .2
    assert not first['position']['profit_exit']['stage_done']
    clock.update(now=clock['now']+30000, liquidity=100)
    again = worker.step()
    assert again['position']['qty'] == .5
    assert again['fills'][-1]['qty'] == .3
    assert again['position']['profit_exit']['stage_done']
    rows = [json.loads(x) for x in path.with_name('closed_trades.jsonl').read_text().splitlines()]
    assert all(not x['position_closed'] for x in rows)
    assert again['wallet_balance'] == pytest.approx(1000+again['realized_pnl']-again['fees_paid']+again['funding_pnl'])


def test_protective_stop_overrides_pending_half_and_closes_residual(tmp_path, monkeypatch):
    worker, path, clock, _ = fixture(tmp_path, monkeypatch)
    clock['liquidity'] = 2
    first = worker.step()
    clock.update(now=clock['now']+30000, price=99, liquidity=100)
    again = worker.step()
    assert again['position'] is None
    assert again['fills'][-1]['reason'] == 'protective_stop'
    assert again['fills'][-1]['qty'] == .8
    assert json.loads(path.with_name('closed_trades.jsonl').read_text().splitlines()[-1])['position_closed']


def test_tiny_lot_arms_protection_without_invalid_half(tmp_path, monkeypatch):
    worker, path, clock, _ = fixture(tmp_path, monkeypatch, qty=.05)
    result = worker.step()
    assert result['position']['qty'] == .05
    assert result['fill_count_total'] == 0
    assert result['position']['profit_exit']['split_skipped'] == 'minimum_quantity_or_notional'
    assert result['position']['stop'] > 100


@pytest.mark.parametrize('side', [1, -1])
def test_trail_uses_only_current_quotes_and_closed_own_atr_never_loosens(side):
    cfg = {'profit_exit_policy': POLICY, 'adverse_slippage_fraction_assumption':.0002,
           'taker_fee_rate_assumption':.001, 'leverage':10}
    pos = {'entry':100, 'initial_stop':98 if side==1 else 102, 'stop':98 if side==1 else 102,
           'qty':1, 'direction':side, 'entry_fee':.1, 'funding':.02}
    best = 103 if side == 1 else 97
    exits.observe(pos,best,cfg,1000,.01,.01,5)
    exits.record_partial(pos,.5,.01)
    exits.protect(pos,cfg,.01,1,999)
    assert pos['stop'] == pytest.approx(101.5 if side==1 else 98.5)
    old = pos['stop']
    exits.observe(pos,102.5 if side==1 else 97.5,cfg,2000,.01,.01,5)
    exits.protect(pos,cfg,.01,3,1999)
    assert pos['stop'] == old
    assert pos['profit_exit']['last_atr_bar_ms'] == 1999


def test_source_failures_do_not_disable_scale_out_or_cost_protection(tmp_path, monkeypatch):
    worker, path, clock, _ = fixture(tmp_path, monkeypatch)
    original = worker.venue.get.side_effect
    def outage(endpoint, params=None):
        if endpoint in ('klines','fundingRate'):
            raise RuntimeError('Missing source')
        return original(endpoint,params)
    worker.venue.get.side_effect = outage
    result = worker.step()
    assert result['status'] == 'degraded'
    assert result['position']['qty'] == .5
    assert result['position']['stop'] > 100
    assert result['position']['profit_exit']['last_atr_bar_ms'] is None


def test_late_funding_after_partial_is_allocated_to_actual_residual(tmp_path, monkeypatch):
    worker, path, clock, _ = fixture(tmp_path, monkeypatch)
    result = worker.step()
    before = result['position']['funding']
    event = {'fundingTime': result['position']['entry_time'] + 1,
             'fundingRate':'.01','markPrice':'100'}
    worker.funding(result,[event],clock['now'])
    assert result['position']['funding'] == pytest.approx(before+.5)
    row = json.loads(path.with_name('funding.jsonl').read_text().splitlines()[-1])
    assert row['debit'] == 1
    assert row['open_funding_debit'] == .5
    assert row['closed_funding_debit'] == .5
    assert result['wallet_balance'] == pytest.approx(1000+result['realized_pnl']-result['fees_paid']+result['funding_pnl'])
    worker.funding(result,[event],clock['now'])
    assert result['position']['funding'] == pytest.approx(before+.5)


def test_broken_atr_does_not_block_account_hard_stop(tmp_path, monkeypatch):
    worker,path,clock,_=fixture(tmp_path,monkeypatch)
    state=json.loads(path.read_text());state['risk_halted']=True
    runner.write_json(path,state)
    monkeypatch.setattr(runner.signals,'compute_indicators',Mock(side_effect=ValueError('Bad ATR')))
    result=worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason']=='account_hard_stop'
    assert result['status']=='degraded'
    assert 'profit_atr' in result['errors']


def test_control_clone_and_exit_only_migration_preserve_pending_signal(tmp_path, monkeypatch):
    worker,path,clock,_ = fixture(tmp_path,monkeypatch)
    state=json.loads(path.read_text())
    state['rule']=copy.deepcopy(worker.config)
    state['pending_signal']={'signal': {'direction':-1}, 'boundary_ms':1,'expires_at_ms':2}
    state['last_signal_time_ms']=123
    runner.write_json(path,state)
    old_cfg=copy.deepcopy(state['rule'])
    old_cfg.pop('profit_exit_policy',None)
    state['rule']=old_cfg
    runner.write_json(path,state)
    plan={'accounts':[{'account_id':'mu','symbol':'MUUSDT','state_path':str(path)}],
          'profit_exit_comparison':{'directory':'comparison'},'stock_rule_revision_scope':'profit_exits_only'}
    root=tmp_path/'trial';root.mkdir()
    comparison=runner.prepare_profit_comparison(plan,root)
    assert json.loads((comparison[0]/'mu/state.json').read_text())==state
    monkeypatch.setattr(runner,'stock_config',lambda p,s:worker.config)
    manifest={}
    runner.synchronize_stock_rules(plan,root,manifest)
    after=json.loads(path.read_text())
    for k in ['wallet_balance','position','pending_signal','last_signal_time_ms','signal_active_after_ms']:
        assert after.get(k)==state.get(k)
    assert runner.prepare_profit_comparison(plan,root)==comparison
    runner.synchronize_stock_rules(plan,root,manifest)
    assert len(manifest['rule_revisions'])==1
    assert json.loads((comparison[0]/'mu/state.json').read_text())['rule']==old_cfg


def test_same_market_responses_produce_identical_baseline_before_trigger(tmp_path,monkeypatch):
    worker,path,clock,_=fixture(tmp_path,monkeypatch)
    clock['price']=101
    state=json.loads(path.read_text());state['rule'].pop('profit_exit_policy',None)
    control_path=tmp_path/'control/state.json';runner.write_json(control_path,state)
    control=runner.StockAccount(control_path,Mock(side_effect=AssertionError('Must reuse quotes')),'MUUSDT',state['rule'])
    responses=worker.market_data()
    candidate=worker.step(responses);baseline=control.step(responses)
    for k in ['wallet_balance','equity','fees_paid','funding_pnl','fill_count_total','position_qty']:
        assert candidate[k]==baseline[k]
    assert baseline['position']['qty']==1


@pytest.mark.parametrize('key,value',[('first_target_r',True),('trail_atr',0),('mode','unknown')])
def test_invalid_policy_is_rejected(key,value):
    policy={**POLICY,key:value}
    with pytest.raises(ValueError):exits.validate({'profit_exit_policy':policy})
