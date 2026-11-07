from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import main_trend_pullback_entry as p
import main_stock_runtime as runtime
import run_parallel_simulation as paper
from test_main_stock_execution_guard import worker as guarded_worker
from test_main_stock_runtime import HOUR,unavailable
from test_main_stock_trend_integration import sector

B=100*HOUR

def config():
    return paper.stock_config({'stock_research_profile':'config/stock_trend_pullback_20261107.json'},'MUUSDT')

def pending(side=1):
    return {'signal':{'direction':side,'atr':2.,'close':101. if side==1 else 99.,'strength':.5,'time_ms':B-HOUR},
        'boundary_ms':B,'expires_at_ms':B+HOUR,
        'entry_trigger':{'family':'donchian20','breakout_level':100.,'direction':side,'boundary_ms':B}}

def bar(open_ms,o,h,l,c):return [open_ms,o,h,l,c,1000.,open_ms+p.FIVE-1]

def stages(side=1):
    first=bar(B+p.FIVE,100.4,100.6,100.,100.2)
    second=bar(B+2*p.FIVE,100.2,100.85,100.1,100.8)
    if side==-1:
        first=bar(first[0],200-first[1],200-first[3],200-first[2],200-first[4])
        second=bar(second[0],200-second[1],200-second[3],200-second[2],200-second[4])
    return first,second

def ready(side=1):
    s=p.seed(pending(side),timestamp=B+30_000,active_after=B-1)
    return p.advance(s,stages(side),timestamp=B+3*p.FIVE)

@pytest.mark.parametrize('side',[-1,1])
def test_retest_and_later_completed_reversal_required(side):
    s=p.seed(pending(side),timestamp=B+30_000,active_after=B-1);before=deepcopy(s)
    first,second=stages(side)
    waiting=p.advance(s,[first],timestamp=B+2*p.FIVE)
    assert waiting['status']=='waiting_confirmation' and p.pending_entry(waiting) is None
    # The still-forming confirmation bar must not grant entry.
    forming=p.advance(waiting,[first,second],timestamp=B+3*p.FIVE-2)
    assert forming['status']=='waiting_confirmation'
    complete=p.advance(waiting,[first,second],timestamp=B+3*p.FIVE)
    assert complete['status']=='ready' and p.pending_entry(complete)
    assert s==before and complete['confirmation_close_ms']>complete['touch_close_ms']

@pytest.mark.parametrize('side',[-1,1])
def test_live_mark_quote_and_actual_fill_all_require_price_improvement(side):
    s=ready(side);cfg=config();trigger=p.pending_entry(s)['entry_trigger']
    price=100.75 if side==1 else 99.25
    decision=p.evaluate(trigger,s['signal'],cfg,reference_price=price,mark_price=price,fill_price=price)
    assert decision['allowed'] and decision['minimum_observed_improvement_atr']>=.1
    worse=s['original_close']-side*.1
    blocked=p.evaluate(trigger,s['signal'],cfg,reference_price=price,mark_price=price,fill_price=worse)
    assert not blocked['allowed'] and 'pullback_price_improvement_insufficient' in blocked['reasons']
    lost=p.evaluate(trigger,s['signal'],cfg,reference_price=s['confirmation_level'],mark_price=price)
    assert lost['cancel_pending'] and 'pullback_confirmation_lost' in lost['reasons']

def test_gap_fails_closed_and_later_revision_cannot_rewrite_the_decision_prefix():
    s=p.seed(pending(),timestamp=B+30_000,active_after=B-1);first,second=stages()
    assert p.advance(s,[second],timestamp=B+3*p.FIVE)['reasons']==['pullback_five_minute_gap']
    waiting=p.advance(s,[first],timestamp=B+2*p.FIVE)
    revised=deepcopy(first);revised[2]+=.01
    fixed=p.advance(waiting,[revised,second],timestamp=B+3*p.FIVE)
    assert fixed['status']=='ready' and fixed['revised_price_observations']==1
    assert fixed['confirmation_level']==first[2]
    assert fixed['last_price_revision']['first_seen']['high']==first[2]
    bad=pending();bad['entry_trigger']['boundary_ms']+=HOUR
    with pytest.raises(ValueError,match='provenance'):p.seed(bad,timestamp=B+30_000,active_after=B-1)
    with pytest.raises(ValueError,match='new_hourly'):p.seed(pending(),timestamp=B+30_000,active_after=B)

def test_no_touch_on_bar_that_began_before_first_observation_and_no_expiry_renewal():
    s=p.seed(pending(),timestamp=B+30_000,active_after=B-1)
    old=bar(B,100.,100.3,99.9,100.1)
    assert p.advance(s,[old],timestamp=B+p.FIVE)['status']=='waiting_retest'
    expired=p.advance(s,[],timestamp=B+3*HOUR)
    assert expired['status']=='expired' and expired['expires_at_ms']==s['expires_at_ms']
    confirmed=ready();end=confirmed['confirmation_expires_at_ms']
    assert p.advance(confirmed,[],timestamp=end)['status']=='expired'

@pytest.mark.parametrize('side',[-1,1])
def test_structure_stop_precedes_sizing_and_retains_original_risk_width(side):
    cfg=config();s=ready(side);entry=100.75 if side==1 else 99.25
    stop=p.initial_stop(s,entry,cfg)
    assert stop is not None and side*(entry-stop)>0
    assert side*(stop-s['structural_invalidation'])<=1e-12
    assert abs(entry-stop)<=max(2*s['atr'],entry*cfg['min_stop_fraction'])+1e-12
    bad=deepcopy(s);bad['structural_invalidation']=entry-side*20
    assert p.initial_stop(bad,entry,cfg) is None

def worker(tmp_path,monkeypatch):
    original,clock,bars=guarded_worker(tmp_path,monkeypatch)
    cfg={**original.config,p.POLICY_KEY:deepcopy(p.POLICY)}
    state=paper.read_json(original.path);state.update(rule=cfg,pullback_entry_active_after_ms=599*HOUR+2)
    paper.write_json(original.path,state)
    account=runtime.MainStockAccount(original.path,original.venue,original.symbol,cfg)
    monkeypatch.setattr(account,'shared_clock_loader',unavailable)
    monkeypatch.setattr(account,'sector_loader',lambda *_a,**_k:sector(clock['sector_direction']))
    get=account.venue.get.side_effect
    def five(endpoint,params=None):
        if endpoint=='klines' and params['interval']=='5m':
            end=clock['now']//p.FIVE*p.FIVE;values=[]
            for t in range(end-12*p.FIVE,end,p.FIVE):
                values.append(deepcopy(clock.get('five_overrides',{}).get(t,
                    bar(t,clock['price'],clock['price']+.0001,clock['price']-.0001,clock['price']))))
            return values
        return get(endpoint,params)
    account.venue.get.side_effect=five
    return account,clock,bars

def test_real_worker_observes_setup_without_immediate_breakout_fill(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch)
    state=account.step()
    assert state['position'] is None and state['fill_count_total']==0
    assert state['trend_pullback_setup']['status']=='waiting_retest'
    assert state['pending_signal'] is None


def test_completed_confirmation_books_better_price_with_structural_provenance_and_guards(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch);first=account.step();s=first['trend_pullback_setup']
    level,atr=s['breakout_level'],s['atr'];base=s['boundary_ms']
    touch=bar(base+p.FIVE,level+.2*atr,level+.3*atr,level+.1*atr,level+.2*atr)
    confirm=bar(base+2*p.FIVE,level+.2*atr,level+.22*atr,level-.02*atr,level+.09*atr)
    clock.update(now=base+2*p.FIVE+60_000,price=touch[4],five_overrides={touch[0]:touch})
    waiting=account.step()
    assert waiting['fill_count_total']==0 and waiting['trend_pullback_setup']['status']=='waiting_confirmation'
    # Recreate the worker to prove the observed retest survives process restart.
    restored=runtime.MainStockAccount(account.path,account.venue,account.symbol,account.config)
    monkeypatch.setattr(restored,'shared_clock_loader',unavailable)
    monkeypatch.setattr(restored,'sector_loader',lambda *_a,**_k:sector(clock['sector_direction']))
    clock.update(now=base+3*p.FIVE+60_000,price=confirm[4],five_overrides={touch[0]:touch,confirm[0]:confirm})
    result=restored.step();pos=result['position']
    assert pos is not None, result.get('entry_blockers')
    assert pos['entry']>=s['original_close']+.1*atr
    assert result['fills'][-1]['entry_method']==p.POLICY['mode']
    assert result['trend_pullback_setup']['status']=='consumed'
    assert pos['trend_pullback_entry']['structural_invalidation']>pos['entry']
    assert pos['trade_profit_budget']['expected_net_profit_usdt']>=100
    assert pos['trade_profit_budget']['planned_loss_usdt']<=20
    assert pos['trade_profit_budget']['execution_guard']['allowed']
    assert result['entry_trend_execution']['allowed']
    again=restored.step();assert again['fill_count_total']==1
    clock.update(now=clock['now']+30_000,price=pos['stop']+1,hourly_error=True)
    stopped=restored.step()
    assert stopped['position'] is None and stopped['fills'][-1]['reason']=='protective_stop'


def test_same_direction_new_hour_cannot_replace_original_level_or_extend_deadline(tmp_path,monkeypatch):
    account,clock,_=worker(tmp_path,monkeypatch);state=account.step();s=state['trend_pullback_setup'];original=deepcopy(s)
    now=s['boundary_ms']+HOUR+60_000
    # No reversal candle: observe every whole interval to stay eligible.
    account.five=[bar(t,s['original_close'],s['original_close']+.0001,s['original_close']-.0001,s['original_close'])
                  for t in range(s['first_eligible_bar_open_ms'],s['boundary_ms']+HOUR,p.FIVE)]
    newer={'signal':{**s['signal'],'time_ms':s['signal']['time_ms']+HOUR},
        'boundary_ms':s['boundary_ms']+HOUR,'expires_at_ms':s['boundary_ms']+2*HOUR,
        'entry_trigger':{'family':'donchian20','direction':s['direction'],'boundary_ms':s['boundary_ms']+HOUR,
                         'breakout_level':s['breakout_level']-1}}
    result=account.prepare_pending_entry(state,newer,{'clock':{'serverTime':now}}, {},now)
    assert result is None
    assert state['trend_pullback_setup']['id']==original['id']
    assert state['trend_pullback_setup']['expires_at_ms']==original['expires_at_ms']
    assert state['trend_pullback_setup']['breakout_level']==original['breakout_level']


def test_future_price_perturbations_do_not_change_completed_bar_decision():
    s=p.seed(pending(),timestamp=B+30_000,active_after=B-1);first,second=stages()
    normal=p.advance(s,[first],timestamp=B+2*p.FIVE)
    future=deepcopy(second);future[1:5]=[-1.,float('nan'),1e99,1e98]
    assert p.advance(s,[first,future],timestamp=B+2*p.FIVE)==normal


def test_native_exchange_numeric_string_ohlc_has_same_causal_result():
    s=p.seed(pending(),timestamp=B+30_000,active_after=B-1)
    numeric=list(stages());native=deepcopy(numeric)
    for row in native:row[1:6]=[str(x) for x in row[1:6]]
    assert p.advance(s,native,timestamp=B+3*p.FIVE)==p.advance(s,numeric,timestamp=B+3*p.FIVE)
    for bad in [True,'NaN','Infinity','0','-1',None]:
        with pytest.raises(ValueError,match='invalid_price'):p.number(bad)


def test_entry_only_migration_preserves_prior_clocks_finances_and_consumed_signals(tmp_path,monkeypatch):
    cfg=config();old={k:v for k,v in cfg.items() if k!=p.POLICY_KEY}
    state=paper.initial_stock('MUUSDT',1000,old)
    state.update(signal_timeframe='1h',main_trend_active_after_ms=1500,
        entry_trend_active_after_ms=2000,execution_guard_active_after_ms=3000,
        last_signal_time_ms=3500,signal_active_after_ms=1500,pending_signal={'boundary_ms':3500})
    path=tmp_path/'mu/state.json';paper.write_json(path,state)
    monkeypatch.setattr(paper,'stock_config',lambda *_:cfg)
    monkeypatch.setattr(paper,'now_ms',lambda:4000)
    plan={'accounts':[{'account_id':'mu','symbol':'MUUSDT','state_path':str(path)}],
        'stock_rule_revision_scope':'trend_pullback_entry_only','main_stock_runtime':'price_led_short_v1'}
    paper.synchronize_stock_rules(plan,tmp_path,{})
    result=paper.read_json(path)
    changed={'rule','rule_history','pullback_entry_active_after_ms','trend_pullback_setup','pending_signal','last_signal_result','signal_status'}
    assert {k:v for k,v in state.items() if k not in changed}=={k:v for k,v in result.items() if k not in changed}
    assert result['pullback_entry_active_after_ms']==4000 and result['trend_pullback_setup'] is None
    assert result['pending_signal'] is None and result['last_signal_time_ms']==3500
    paper.synchronize_stock_rules(plan,tmp_path,{})
    assert paper.read_json(path)==result


@pytest.mark.parametrize('change',[lambda s:s.update(pullback_entry_active_after_ms=6000),
    lambda s:s['rule_history'][-1].update(activated_at_ms=True),
    lambda s:s['rule'].update(risk_fraction_per_trade=.5),
    lambda s:s['rule_history'][-1]['new_rule'][p.POLICY_KEY].update(maximum_setup_hours=4)])
def test_full_recorded_revision_chain_rejects_hidden_policy_drift(change):
    from test_main_entry_trend_runtime import history
    import main_stock_rule_revision as revisions
    import main_stock_execution_guard as guard
    state,expected=history()
    for module,key,epoch,scope in [(guard,guard.POLICY_KEY,6000,'execution_guard_only'),
                                  (p,p.POLICY_KEY,7000,'trend_pullback_entry_only')]:
        old=deepcopy(state['rule']);state['rule']={**old,key:deepcopy(module.POLICY)}
        state['execution_guard_active_after_ms' if module is guard else 'pullback_entry_active_after_ms']=epoch
        state['rule_history'].append({'scope':scope,'activated_at_ms':epoch,'previous_rule':old,'new_rule':deepcopy(state['rule'])})
    before=deepcopy(state)
    assert revisions.declared_main_baseline_matches(state,expected) and before==state
    change(state)
    assert not revisions.declared_main_baseline_matches(state,expected)
