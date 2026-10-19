"""Isolated real execution, ledger conservation, funding and restart integrity."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json
import hashlib
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_btc_reentry_shadow as shadow
import trading_execution as execution
from test_btc_same_direction_reentry import ORIGIN,compact,hourly,point,context


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED','false')
    monkeypatch.setenv('SIM_TAKER_FEE','.00045')
    monkeypatch.setenv('SIM_SLIPPAGE_BPS','1')
    spec,profile,fingerprint=shadow.load_design()
    spec['source_report']='source.json'
    root=tmp_path/'experiment';root.mkdir()
    manifest=shadow.prepare(root,spec,profile,fingerprint,ORIGIN-1000)
    workers={arm:execution.SimulationAccount(root/arm/'state.json',cost_multiplier=
        spec['cost_multipliers'][arm.rsplit('_',1)[1]]) for arm in shadow.ARMS}
    report={'execution_target':point(), 'generated_at_utc':shadow.iso(ORIGIN),
        'config':{'portfolio_leverage_cap':10,'strategy_modes':['timeseries_trend'],
            'timeseries_fast_ema':48,'timeseries_slow_ema':240,
            'timeseries_min_ema_spread_pct':.004,'timeseries_target_vol':.11},
        'execution_entry_context':{'factor_profile_sha256':hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()},
        'shadow_profile':{field:fingerprint['source_sha256']['scripts/'+name] for field,name in {
            'factor_engine_sha256':'multifactor.py','portfolio_risk_sha256':'portfolio_risk.py',
            'public_context_sha256':'public_context.py','event_risk_sha256':'event_risk.py',
            'hourly_execution_sha256':'timeseries_execution.py','execution_portfolio_sha256':'execution_portfolio.py'}.items()},
        'execution_model':'closed_signal_known_open_v1',
        'strategy_review_context':{'timeseries_trend':context()}}
    shadow.write(tmp_path/'source.json',report)
    monkeypatch.setattr(shadow,'ROOT',tmp_path)
    monkeypatch.setattr(shadow,'freeze_context',lambda report,root:deepcopy(report))
    monkeypatch.setattr(execution.execution_entry_gate,'decision_at',lambda *a,**k:{'allowed':True,'reasons':[]})
    rules={key:Decimal(value) for key,value in {'step_size':'.001','min_qty':'.001',
        'max_qty':'120','min_notional':'5'}.items()}
    now=ORIGIN+300000
    common={'clock':{'source':'exchange','time_ms':now},'mark':{'price':1006,'time_ms':now},
        'rules':rules,'hourly':hourly(),
        'five_minute':[compact(ORIGIN,1002,1008,1001,1006,step=300000)],
        'funding_events':[],'funding_available':True,'funding_fetch_through_ms':now,
        'next_funding_time_ms':ORIGIN+8*shadow.HOUR,'observed_monotonic':time.monotonic()}
    class Collector:
        def risk_observation(self):
            return deepcopy(common)
        def complete(self,risk,states):
            common['observed_monotonic']=time.monotonic()
            return deepcopy(common)
    return root,manifest,workers,report,common,Collector(),tmp_path


def test_shared_observation_executes_only_fresh_candidate_not_old_origin_and_books_costs(setup):
    root,manifest,workers,report,common,collector,_=setup
    summaries,_=shadow.tick(root,manifest,workers,collector)
    for name in ('baseline_normal','baseline_double'):
        assert workers[name].load()['position_qty']==0
        assert workers[name].load()['wallet_balance']==1000
    for name in ('candidate_normal','candidate_double'):
        state=workers[name].load()
        assert summaries[name]['status']=='healthy',summaries[name]
        assert state['position_qty']>0 and state['fill_count_total']==1
        assert state['fees_paid']>0 and state['wallet_balance']==pytest.approx(1000-state['fees_paid'])
        assert state['fills'][0]['price']>common['mark']['price']
        assert state['btc_pullback_pending']['signal_origin']==ORIGIN
        assert state['btc_pullback_pending']['source']['position_id'].startswith('btc-reentry-')
    assert workers['candidate_double'].load()['fills'][0]['price']>workers['candidate_normal'].load()['fills'][0]['price']
    assert workers['candidate_double'].load()['fees_paid']>workers['candidate_normal'].load()['fees_paid']


def test_funding_reversal_exit_and_closed_net_conserve_wallet_without_reset(setup):
    root,manifest,workers,report,common,collector,tmp=setup
    shadow.tick(root,manifest,workers,collector)
    common['clock']['time_ms']+=30000
    common['mark'].update(price=1010,time_ms=common['clock']['time_ms'])
    common['funding_fetch_through_ms']=common['clock']['time_ms']
    common['funding_events']=[{'fundingTime':ORIGIN+301000,'fundingRate':'.0001','markPrice':'1006'}]
    shadow.tick(root,manifest,workers,collector)
    report['execution_target'].update(signed_qty=0,time_ms=ORIGIN+300000,position_id='flat')
    shadow.write(tmp/'source.json',report)
    shadow.tick(root,manifest,workers,collector)
    for name in ('candidate_normal','candidate_double'):
        state=workers[name].load()
        assert state['position_qty']==0 and state['funding_pnl']<0
        assert state['wallet_balance']==pytest.approx(1000+state['realized_pnl']-state['fees_paid']+state['funding_pnl'])
        metrics=shadow.closed_metrics(workers[name],workers[name].path.parent,1010)
        assert metrics['closed_positions']==1
        assert metrics['closed_net_pnl_usdt']==pytest.approx(state['wallet_balance']-1000)


def test_current_gate_and_missing_funding_block_new_risk_for_both_costs(setup,monkeypatch):
    root,manifest,workers,report,common,collector,_=setup
    monkeypatch.setattr(execution.execution_entry_gate,'decision_at',lambda *a,**k:{'allowed':False,'reasons':['test_event_blackout']})
    shadow.tick(root,manifest,workers,collector)
    assert all(w.load()['position_qty']==0 for w in workers.values())
    monkeypatch.setattr(execution.execution_entry_gate,'decision_at',lambda *a,**k:{'allowed':True,'reasons':[]})
    common['funding_available']=False
    common['clock']['time_ms']+=30000
    common['mark']['time_ms']=common['clock']['time_ms']
    shadow.tick(root,manifest,workers,collector)
    assert all(w.load()['position_qty']==0 for w in workers.values())


def test_hard_stop_runs_before_report_or_market_history_failure(setup):
    root,manifest,workers,report,common,collector,tmp=setup
    shadow.tick(root,manifest,workers,collector)
    common['clock']['time_ms']+=30000
    common['mark'].update(price=600,time_ms=common['clock']['time_ms'])
    (tmp/'source.json').unlink()
    summaries,_=shadow.tick(root,manifest,workers,collector)
    for name in ('candidate_normal','candidate_double'):
        state=workers[name].load()
        assert state['position_qty']==0 and state['risk_halt_at_utc']
        assert state['fills'][-1]['reason']=='account_hard_stop'
        assert summaries[name]['status']=='degraded'


def test_restart_preserves_balances_and_rejects_changed_or_missing_history(setup):
    root,manifest,workers,report,common,collector,_=setup
    shadow.tick(root,manifest,workers,collector)
    before={arm:worker.path.read_bytes() for arm,worker in workers.items()}
    restored=shadow.prepare(root,manifest['design'],manifest['profile'],manifest['fingerprint'],ORIGIN+999999,True)
    assert restored==manifest
    assert before=={arm:worker.path.read_bytes() for arm,worker in workers.items()}
    changed=deepcopy(manifest['fingerprint']);changed['extra']='changed'
    with pytest.raises(ValueError,match='incompatible'):
        shadow.prepare(root,manifest['design'],manifest['profile'],changed,ORIGIN,True)
    workers['baseline_normal'].path.with_name('event_state.json').unlink()
    with pytest.raises(FileNotFoundError):
        shadow.prepare(root,manifest['design'],manifest['profile'],manifest['fingerprint'],ORIGIN,True)


def test_require_existing_and_partial_directory_never_create_accounts(tmp_path):
    spec,profile,fingerprint=shadow.load_design()
    with pytest.raises(ValueError,match='required'):
        shadow.prepare(tmp_path,spec,profile,fingerprint,ORIGIN,True)
    (tmp_path/'orphan.json').write_text('{}')
    with pytest.raises(ValueError,match='Incomplete'):
        shadow.prepare(tmp_path,spec,profile,fingerprint,ORIGIN)
    assert not (tmp_path/'manifest.json').exists()


def test_same_fresh_original_signal_is_unchanged_in_all_arms(setup):
    root,manifest,workers,report,common,collector,tmp=setup
    # First observe the old-origin flat baseline before the new original signal.
    common['clock']['time_ms']=ORIGIN+1000
    common['mark']['time_ms']=ORIGIN+1000
    common['funding_fetch_through_ms']=ORIGIN+1000
    common['five_minute']=[]
    shadow.tick(root,manifest,workers,collector)
    report['execution_target'].update(position_id='new-original',origin_signal_time_ms=ORIGIN,
        origin_entry_price=1040,time_ms=ORIGIN+300000)
    shadow.write(tmp/'source.json',report)
    common['clock']['time_ms']=ORIGIN+300000
    common['mark']['time_ms']=ORIGIN+300000
    common['funding_fetch_through_ms']=ORIGIN+300000
    common['five_minute']=[compact(ORIGIN,1002,1008,1001,1006,step=300000)]
    shadow.tick(root,manifest,workers,collector)
    for cost in ('normal','double'):
        base=workers['baseline_'+cost].load();candidate=workers['candidate_'+cost].load()
        assert base['position_qty']>0 and base['position_qty']==candidate['position_qty']
        assert base['wallet_balance']==candidate['wallet_balance']


def test_heartbeat_protocol_and_dependency_closure(setup):
    root,manifest,workers,report,common,collector,_=setup
    accounts,clock=shadow.tick(root,manifest,workers,collector)
    status=shadow.publish_status(root,manifest,accounts,clock,30,True)
    assert status['running'] and status['places_orders'] is False and status['pid']>0
    assert status['automatic_strategy_promotion'] is False
    assert set(status['accounts'])==set(shadow.ARMS)
    assert all(arm['heartbeat_at_ms']>0 for arm in status['accounts'].values())
    # Closure traverses imported execution gates, public-context and TLS helper.
    source=shadow.source_hashes() if shadow.ROOT==Path(__file__).resolve().parents[1] else manifest['fingerprint']['source_sha256']
    assert 'scripts/trading_execution.py' in source and 'scripts/public_context.py' in source
    assert 'scripts/public_text_transport.py' in source


def test_deferred_review_blocks_pending_and_reuses_completed_provider(monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED','true')
    monkeypatch.setattr(shadow.llm_trade_gate,'build_decision_context',lambda *a,**k:{
        'admission_policy':{'historical_performance_required':True},'decision_clock':{}})
    calls=[]
    import threading
    ready=threading.Event()
    def provider(context):
        calls.append(context)
        ready.wait(2)
        return {'allow':True,'confidence':1.,'reason':'test','risk_flags':[]}
    monkeypatch.setattr(shadow.llm_trade_gate,'request_llm_decision',provider)
    review=shadow.DeferredReview()
    target={'target_leverage':1.,'position_id':'new-event'}
    kwargs={'current_qty':0.,'equity':1000.,'mark_price':1000.,'mode':'simulation'}
    _,pending=review.gate({},target,**kwargs)
    assert pending['allow'] is False and 'btc_shadow_review_pending' in pending['reason']
    ready.set()
    for future in review.futures.values():
        future.result(timeout=3)
    review.begin_tick()
    gated,approved=review.gate({},target,previous_decision=pending,**kwargs)
    assert approved['allow'] and gated['target_leverage']==1 and len(calls)==1
    review.close()


def test_deferred_rejection_expiry_schedules_genuinely_new_provider_call(monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED','true')
    monkeypatch.setenv('LLM_REJECTION_CACHE_SECONDS','60')
    monkeypatch.setattr(shadow.llm_trade_gate,'build_decision_context',lambda report,*a,**k:{
        'admission_policy':{'historical_performance_required':True},
        'decision_clock':{'time_ms':report['clock']}})
    wall=['2026-10-20T00:00:00+00:00']
    monkeypatch.setattr(shadow.llm_trade_gate,'utc_now',lambda:wall[0])
    calls=[]
    def provider(context):
        calls.append(deepcopy(context))
        return {'allow':False,'confidence':1.,'reason':'test rejection','risk_flags':[]}
    monkeypatch.setattr(shadow.llm_trade_gate,'request_llm_decision',provider)
    review=shadow.DeferredReview()
    target={'target_leverage':1.,'position_id':'same-event'}
    kwargs={'current_qty':0.,'equity':1000.,'mark_price':1000.,'mode':'simulation'}
    review.gate({'clock':100},target,**kwargs)
    for future in review.futures.values():
        future.result(timeout=3)
    review.begin_tick()
    _,first=review.gate({'clock':200},target,**kwargs)
    assert first['status']=='cached_rejected' and len(calls)==1
    wall[0]='2026-10-20T00:02:00+00:00'
    _,pending=review.gate({'clock':300},target,previous_decision=first,**kwargs)
    assert 'btc_shadow_review_pending' in pending['reason']
    for future in review.futures.values():
        future.result(timeout=3)
    review.begin_tick()
    _,second=review.gate({'clock':400},target,previous_decision=pending,**kwargs)
    assert len(calls)==2
    assert calls[1]['decision_clock']['time_ms']==300
    assert second['review_decision_clock']['time_ms']==300
    review.close()


def test_async_review_retains_original_context_hash_and_cutoff(monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED','true')
    monkeypatch.setattr(shadow.llm_trade_gate,'build_decision_context',lambda report,*a,**k:{
        'admission_policy':{'historical_performance_required':True},
        'decision_clock':{'time_ms':report['clock']}})
    monkeypatch.setattr(shadow.llm_trade_gate,'request_llm_decision',lambda context:{
        'allow':True,'confidence':1.,'reason':'approved old actual cutoff','risk_flags':[]})
    review=shadow.DeferredReview()
    target={'target_leverage':1.,'position_id':'same-event'}
    kwargs={'current_qty':0.,'equity':1000.,'mark_price':1000.,'mode':'simulation'}
    review.gate({'clock':100},target,**kwargs)
    completed=[f.result(timeout=3) for f in review.futures.values()][0][1]
    review.begin_tick()
    _,current=review.gate({'clock':900},target,**kwargs)
    assert current['status']=='cached_approved'
    assert current['review_decision_clock']=={'time_ms':100}
    assert current['review_context_sha256']==completed['review_context_sha256']
    review.close()


def test_unpublished_crossed_funding_boundary_blocks_additions(setup):
    root,manifest,workers,report,common,collector,_=setup
    for worker in workers.values():
        state=worker.load()
        state['funding_next_settlement_ms']=ORIGIN+100000
        worker._save(state)
    shadow.tick(root,manifest,workers,collector)
    for worker in workers.values():
        state=worker.load()
        assert state['position_qty']==0
        assert state['funding_pending_settlement_ms']==ORIGIN+100000


def test_source_strategy_drift_is_blocked_without_losing_risk_protection(setup):
    root,manifest,workers,report,common,collector,tmp=setup
    report['config']['timeseries_fast_ema']=47
    shadow.write(tmp/'source.json',report)
    accounts,_=shadow.tick(root,manifest,workers,collector)
    assert all(w.load()['position_qty']==0 for w in workers.values())
    assert all('configuration differs' in a['errors']['common_observation'] for a in accounts.values())


@pytest.mark.parametrize('requested',[0,-1])
@pytest.mark.parametrize('failure',['5m history unavailable','quantity metadata unavailable'])
def test_fresh_flat_or_reversal_exits_before_entry_history_or_rules_failure(setup,requested,failure):
    root,manifest,workers,report,common,collector,tmp=setup
    shadow.tick(root,manifest,workers,collector)
    assert workers['candidate_normal'].load()['position_qty']>0
    common['clock']['time_ms']+=30000
    common['mark'].update(price=1010,time_ms=common['clock']['time_ms'])
    report['execution_target'].update(signed_qty=requested,position_id='flat' if not requested else 'confirmed-short')
    shadow.write(tmp/'source.json',report)
    def unavailable(*args):
        raise RuntimeError(failure)
    collector.complete=unavailable
    accounts,_=shadow.tick(root,manifest,workers,collector)
    for name in ('candidate_normal','candidate_double'):
        state=workers[name].load()
        assert state['position_qty']==0
        assert state['fill_count_total']==2  # entry and complete close, never a reversal entry
        assert accounts[name]['source_exit']['target_qty']==0
        assert (workers[name].path.parent/'source_exits.jsonl').exists()


def test_stale_source_cannot_manufacture_trend_exit_during_history_outage(setup):
    root,manifest,workers,report,common,collector,tmp=setup
    shadow.tick(root,manifest,workers,collector)
    before=workers['candidate_normal'].load()['position_qty']
    common['clock']['time_ms']+=30000
    common['mark']['time_ms']=common['clock']['time_ms']
    report['execution_target'].update(signed_qty=0,time_ms=common['clock']['time_ms']-901000)
    shadow.write(tmp/'source.json',report)
    accounts,_=shadow.tick(root,manifest,workers,collector)
    assert workers['candidate_normal'].load()['position_qty']==before
    assert workers['candidate_double'].load()['position_qty']>0
    assert all('stale' in a['errors']['source_exit'] for a in accounts.values())


def test_review_completion_between_arms_is_released_only_at_next_tick(monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED','true')
    monkeypatch.setattr(shadow.llm_trade_gate,'build_decision_context',lambda *a,**k:{
        'admission_policy':{'historical_performance_required':True},'decision_clock':{}})
    monkeypatch.setattr(shadow.llm_trade_gate,'request_llm_decision',lambda context:{
        'allow':True,'confidence':1.,'reason':'approved','risk_flags':[]})
    review=shadow.DeferredReview()
    target={'target_leverage':1.,'position_id':'same-original-signal'}
    kwargs={'current_qty':0.,'equity':1000.,'mark_price':1000.,'mode':'simulation'}
    review.begin_tick()
    _,baseline=review.gate({},target,**kwargs)
    for future in review.futures.values():
        future.result(timeout=3)
    # Candidate executes later, after provider is already done: still pending.
    _,candidate=review.gate({},target,**kwargs)
    assert not baseline['allow'] and not candidate['allow']
    assert 'btc_shadow_review_pending' in candidate['reason']
    review.begin_tick()
    _,baseline=review.gate({},target,previous_decision=baseline,**kwargs)
    _,candidate=review.gate({},target,previous_decision=candidate,**kwargs)
    assert baseline['allow'] and candidate['allow']
    assert baseline['reviewed_at_utc']==candidate['reviewed_at_utc']
    review.close()


def test_drifted_flat_report_cannot_exit_existing_original_experiment_inventory(setup):
    root,manifest,workers,report,common,collector,tmp=setup
    shadow.tick(root,manifest,workers,collector)
    before={name:workers[name].load()['position_qty'] for name in ('candidate_normal','candidate_double')}
    assert before['candidate_normal']>0
    common['clock']['time_ms']+=30000
    common['mark']['time_ms']=common['clock']['time_ms']
    report['config']['timeseries_fast_ema']=47
    report['execution_target']['signed_qty']=0
    shadow.write(tmp/'source.json',report)
    accounts,_=shadow.tick(root,manifest,workers,collector)
    assert before=={name:workers[name].load()['position_qty'] for name in before}
    assert all('configuration differs' in a['errors']['common_observation'] for a in accounts.values())
