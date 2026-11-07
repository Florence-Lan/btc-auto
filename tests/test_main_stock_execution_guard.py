from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import main_stock_execution_guard as guard
import main_stock_runtime as runtime
import run_parallel_simulation as paper
import trade_profit_budget as budget
from test_main_entry_trend_runtime import worker as previous_worker
from test_main_stock_runtime import HOUR, unavailable
from test_main_stock_trend_integration import sector


def config():
    return paper.stock_config({'stock_research_profile':'config/stock_execution_guard_20261107.json'}, 'MUUSDT')


def test_nov4_observed_fill_fails_reserved_cap_without_resizing_target():
    cfg=config()
    plan=budget.stock_plan(cfg, equity=997.58614365191, entry=822.2102015352942,
        stop=846.02, step=.01, direction=-1, actual_qty=1.02)
    assert plan['allowed'] and plan['expected_net_profit_usdt'] >= 100
    before=deepcopy(plan)
    result=guard.entry_decision(cfg, plan, direction=-1, mark_price=822.40060896,
                              exit_fill={'qty':1.02,'price':823.1})
    assert not result['allowed']
    assert 'main_execution_planned_reserve_exceeded' in result['reasons']
    assert result['reserved_loss_budget_usdt']==pytest.approx(19.9517228730382)
    assert result['net_profit_goal_usdt']==100 and plan==before


def ready_plan():
    return {'allowed':True,'quantity':2.,'loss_budget_usdt':25.,'planned_loss_usdt':18.,
            'net_profit_goal_usdt':100.}


@pytest.mark.parametrize('side',[-1,1])
def test_depth_basis_stress_is_symmetric_and_slippage_not_charged_twice(side):
    cfg=config();plan=ready_plan();slip=cfg['adverse_slippage_fraction_assumption']
    zero=guard.entry_decision(cfg,plan,direction=side,mark_price=100.,
        exit_fill={'qty':2.,'price':100*(1-side*slip)})
    assert zero['allowed'] and zero['stress_loss_usdt']==pytest.approx(18.)
    raw=100-side*4.
    stressed=guard.entry_decision(cfg,plan,direction=side,mark_price=100.,
        exit_fill={'qty':2.,'price':raw*(1-side*slip)})
    assert not stressed['allowed']
    assert 'main_execution_stressed_loss_exceeds_budget' in stressed['reasons']


@pytest.mark.parametrize('filled,price',[(0.,0.),(1.,100.)])
def test_insufficient_reverse_depth_never_assumes_full_exit(filled,price):
    result=guard.entry_decision(config(),ready_plan(),direction=-1,mark_price=100.,
                              exit_fill={'qty':filled,'price':price})
    assert not result['allowed'] and 'main_execution_reverse_depth_insufficient' in result['reasons']


@pytest.mark.parametrize('side',[-1,1])
def test_net_lock_covers_fees_slippage_and_known_funding_exactly(side):
    cfg={**config(),'taker_fee_rate_assumption':.001,'trail_locked_underlying_return':.01}
    pos={'direction':side,'qty':2.,'entry':100.,'entry_fee':.2,'funding':.05}
    original=deepcopy(pos);lock=guard.lock_quote(cfg,pos)
    price=lock['quote']*(1-side*cfg['adverse_slippage_fraction_assumption'])
    assert guard.net_close(cfg,pos,{'qty':2.,'price':price})==pytest.approx(2.)
    assert side*(lock['quote']-100*(1+side*.01))>0
    assert pos==original
    with pytest.raises(ValueError,match='full_close_unavailable'):
        guard.net_close(cfg,pos,{'qty':1.,'price':price})


def worker(tmp_path,monkeypatch,activation=599*HOUR+1):
    old,clock,bars=previous_worker(tmp_path,monkeypatch)
    cfg={**old.config,guard.POLICY_KEY:deepcopy(guard.POLICY)}
    state=paper.read_json(old.path);state.update(rule=cfg,execution_guard_active_after_ms=activation)
    paper.write_json(old.path,state)
    account=runtime.MainStockAccount(old.path,old.venue,old.symbol,cfg)
    monkeypatch.setattr(account,'shared_clock_loader',unavailable)
    monkeypatch.setattr(account,'sector_loader',lambda *_a,**_k:sector(clock['sector_direction']))
    return account,clock,bars


def test_guarded_real_booking_keeps_goal_technical_stop_and_records_depth(tmp_path,monkeypatch):
    account,_,_=worker(tmp_path,monkeypatch)
    result=account.step()
    assert result['position'] is not None
    plan=result['position']['trade_profit_budget']
    assert plan['expected_net_profit_usdt']>=100
    assert plan['planned_loss_usdt']<=plan['loss_budget_usdt']*.8
    assert result['position']['stop']==plan['technical_stop_price']
    assert plan['execution_guard']['allowed']
    assert result['fills'][-1]['places_orders'] is False
    assert account.path.with_name('execution_guard_decisions.jsonl').exists()


def test_post_activation_boundary_required_and_funding_failure_keeps_old_stop(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch,activation=600*HOUR)
    result=account.step();assert result['position'] is None and not result['fills']
    assert 'main_execution_guard_awaiting_new_signal' in result['entry_blockers']


def test_clock_failure_does_not_disable_original_guarded_stop(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch)
    entered=account.step();assert entered['position']
    clock.update(price=entered['position']['stop']+1,hourly_error=True,now=clock['now']+30_000)
    get=account.venue.get.side_effect
    def failed(endpoint,params=None):
        if endpoint=='time':raise RuntimeError('offline clock failed')
        return get(endpoint,params)
    account.venue.get.side_effect=failed
    result=account.step()
    assert result['position'] is None and result['fills'][-1]['reason']=='protective_stop'


def test_full_depth_loss_guard_closes_when_top_quote_has_not_hit_stop(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch)
    entered=account.step();pos=entered['position'];assert pos and pos['direction']==-1
    get=account.venue.get.side_effect
    def expensive_depth(endpoint,params=None):
        value=get(endpoint,params)
        if endpoint=='depth':
            safe=pos['stop']-.01
            assert safe<pos['stop']
            value['bids']=[[str(safe-.01),'1000']]
            value['asks']=[[str(safe),str(pos['qty'])],
                           [str(safe*1.0049),str(pos['qty']*20)]]
        return value
    account.venue.get.side_effect=expensive_depth
    clock['now']+=30_000
    result=account.step()
    assert result['position'] is None and result['fill_count_total']==2, str(result.get('execution_guard'))
    assert result['fills'][-1]['reason']=='main_execution_loss_reserve'
    assert result['execution_guard']['net_full_close_usdt'] < -20


def test_intrabar_net_lock_arms_with_hourly_feed_unavailable(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch)
    entered=account.step();pos=entered['position'];assert pos
    # Stay below the 12% target while satisfying the original 8% activation.
    clock.update(price=pos['entry']*.915,now=clock['now']+30_000,hourly_error=True)
    armed=account.step();assert armed['position']
    detail=armed['position']['main_trend_profit_protection']
    assert detail['armed'] and detail['known_costs_included']
    assert detail['net_floor_usdt']==pytest.approx(pos['entry']*pos['qty']*account.config['trail_locked_underlying_return'])
    assert armed['position']['stop']<pos['entry']*(1-account.config['trail_locked_underlying_return'])


@pytest.mark.parametrize('mutation',[lambda s:s.update(execution_guard_active_after_ms=5000),
    lambda s:s['rule_history'][-1].update(activated_at_ms=True),
    lambda s:s['rule'].update(risk_fraction_per_trade=.5),
    lambda s:s['rule_history'][-1].update(scope='entry_trend_only')])
def test_revision_chain_rejects_hidden_risk_or_unbound_activation(mutation):
    from test_main_entry_trend_runtime import history
    import main_stock_rule_revision as revisions
    state,expected=history();previous=deepcopy(state['rule'])
    state['rule']={**previous,guard.POLICY_KEY:deepcopy(guard.POLICY)}
    state['execution_guard_active_after_ms']=6000
    state['rule_history'].append({'scope':'execution_guard_only','activated_at_ms':6000,
        'previous_rule':previous,'new_rule':deepcopy(state['rule'])})
    before=deepcopy(state)
    assert revisions.declared_main_baseline_matches(state,expected) and state==before
    mutation(state)
    assert not revisions.declared_main_baseline_matches(state,expected)


def test_migration_keeps_finances_consumed_signal_and_original_epochs(tmp_path,monkeypatch):
    cfg=config();old={k:v for k,v in cfg.items() if k!=guard.POLICY_KEY}
    state=paper.initial_stock('MUUSDT',1000,old)
    state.update(main_trend_active_after_ms=2000,entry_trend_active_after_ms=3000,
        trade_profit_budget_active_after_ms=1500,last_signal_time_ms=3500,
        signal_active_after_ms=2000,signal_timeframe='1h',pending_signal={'boundary_ms':3500})
    path=tmp_path/'mu/state.json';paper.write_json(path,state)
    monkeypatch.setattr(paper,'stock_config',lambda *_:cfg)
    monkeypatch.setattr(paper,'now_ms',lambda:4000)
    plan={'accounts':[{'account_id':'mu','symbol':'MUUSDT','state_path':str(path)}],
          'stock_rule_revision_scope':'execution_guard_only','main_stock_runtime':'price_led_short_v1'}
    paper.synchronize_stock_rules(plan,tmp_path,{})
    result=paper.read_json(path)
    allowed={'rule','rule_history','execution_guard_active_after_ms','pending_signal','last_signal_result','signal_status'}
    assert {k:v for k,v in state.items() if k not in allowed}=={k:v for k,v in result.items() if k not in allowed}
    assert result['execution_guard_active_after_ms']==4000 and result['pending_signal'] is None
    paper.synchronize_stock_rules(plan,tmp_path,{})
    assert paper.read_json(path)==result
