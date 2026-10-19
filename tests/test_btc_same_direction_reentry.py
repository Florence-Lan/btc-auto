"""Causal new origins, terminal identities and bounded same-trend attempts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import btc_same_direction_reentry as reentry
import btc_pullback_entry as pullback

ORIGIN = 1000*reentry.HOUR
PROFILE = {'btc_pullback_entry_policy':deepcopy(pullback.DEFAULT_POLICY)}


def compact(t,op,high,low,close,step=reentry.HOUR):
    return [t,op,high,low,close,1,1000,t+step-1]


def hourly(side=1,origin=ORIGIN):
    values = [compact(origin-(50-i)*reentry.HOUR,1000,1010,990,1000) for i in range(49)]
    values.append(compact(origin-reentry.HOUR,1000,1042 if side>0 else 1010,
                          990 if side>0 else 958,1040 if side>0 else 960))
    return values


def point(side=1,**changes):
    return {'signed_qty':side,'position_id':'original-trend','origin_signal_time_ms':900*reentry.HOUR,
        'origin_entry_price':1100,'time_ms':ORIGIN,'price':1040,'equity':1000,**changes}


def context(side=1,origin=ORIGIN):
    return {'status':'available','confirmed_side':'long' if side>0 else 'short',
        'target_side_valid_under_exit_rule':True,'fast_ema_period_bars':48,'slow_ema_period_bars':240,
        'entry_min_ema_spread_pct':.4,'last_closed_bar':{'close_time_ms':origin-1}}


@pytest.mark.parametrize('side',[1,-1])
def test_event_is_post_inception_causal_and_uses_new_identity(side):
    before = point(side)
    event,reason = reentry.create_event(before,hourly(side),ORIGIN+1000,ORIGIN-1000,PROFILE)
    assert reason is None
    assert event['origin_signal_time_ms']==ORIGIN
    assert event['original_source']['origin_signal_time_ms']==900*reentry.HOUR
    assert event['opportunity_id']!='original-trend'
    assert event['expires_at_ms']==ORIGIN+reentry.HOUR
    assert event['event_structure']['available_at_ms']<event['event_hour_open_ms']
    assert before==point(side)
    target = {**before,'target_leverage':side,'position_id':event['opportunity_id'],
        'origin_signal_time_ms':event['origin_signal_time_ms'],'origin_entry_price':event['origin_entry_price']}
    pending = pullback.build_pending(target,hourly(side),PROFILE['btc_pullback_entry_policy'],ORIGIN+1000)
    assert pending['status']=='waiting_retrace'
    # The hourly event alone is insufficient: future 5m touch/rejection is mandatory.
    _,decision = pullback.advance_pending(pending,target,[],1006 if side>0 else 994,
        ORIGIN+1000,1,PROFILE['btc_pullback_entry_policy'])
    assert not decision['allowed']


@pytest.mark.parametrize('start,now',[(ORIGIN,ORIGIN+1000),(ORIGIN+1,ORIGIN+1000),
    (ORIGIN-1,ORIGIN+reentry.HOUR+1000)])
def test_pre_inception_or_missed_hour_is_never_reissued(start,now):
    assert reentry.create_event(point(),hourly(),now,start,PROFILE)[0] is None


def test_forming_future_bar_is_ignored_and_gap_blocks_event():
    future = compact(ORIGIN,1000,100000,1,99999)
    original = reentry.create_event(point(),hourly(),ORIGIN+1000,ORIGIN-1000,PROFILE)
    assert reentry.create_event(point(),hourly()+[future],ORIGIN+1000,ORIGIN-1000,PROFILE)==original
    bars=[compact(ORIGIN-51*reentry.HOUR,1000,1010,990,1000),*hourly()]; bars.pop(20)
    assert reentry.create_event(point(),bars,ORIGIN+1000,ORIGIN-1000,PROFILE)[1]=='hourly_gap'


def test_no_hourly_touch_or_wrong_rejection_produces_no_event():
    bars=hourly(); bars[-1]=compact(ORIGIN-reentry.HOUR,1035,1042,1030,1040)
    assert reentry.create_event(point(),bars,ORIGIN+1000,ORIGIN-1000,PROFILE)[0] is None


def test_one_hour_is_processed_once_and_origin_never_rolls_on_retry_or_expiry():
    state=reentry.initial_state(ORIGIN-1000)
    p,state,diag=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+1000,PROFILE)
    assert diag['status']=='new_hourly_structural_event'
    identity=p['position_id']
    p2,s2,d2=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+300000,PROFILE)
    assert p2['position_id']==identity and s2['attempts']==1
    p3,s3,d3=reentry.candidate_point(point(),s2,{},hourly(),context(origin=ORIGIN+reentry.HOUR),
        ORIGIN+reentry.HOUR+1000,PROFILE)
    assert p3['origin_signal_time_ms']==point()['origin_signal_time_ms']
    assert s3['opportunity']['status']=='expired' and s3['attempts']==1


def test_missing_current_hour_context_can_retry_after_producer_catches_up():
    state=reentry.initial_state(ORIGIN-1000)
    _,state,diag=reentry.candidate_point(point(),state,{},hourly(),context(origin=ORIGIN-reentry.HOUR),
        ORIGIN+1000,PROFILE)
    assert diag['status']=='current_hour_trend_not_confirmed'
    assert state['last_processed_hour_ms'] is None
    _,state,diag=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+2000,PROFILE)
    assert diag['status']=='new_hourly_structural_event'


def test_cooldown_attempt_limit_and_transient_flat_do_not_reset_budget():
    state=reentry.initial_state(ORIGIN-1000)
    state.update(episode_key=reentry.episode_key(point()),attempts=3)
    _,state,diag=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+1000,PROFILE)
    assert diag['status']=='trend_attempt_limit'
    _,state,_=reentry.candidate_point(point(signed_qty=0),state,{},hourly(),context(),ORIGIN+2000,PROFILE)
    _,state,diag=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+3000,PROFILE)
    assert state['attempts']==3 and diag['status']=='trend_attempt_limit'
    state.update(attempts=1,last_attempt_ms=ORIGIN-1000)
    assert reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+3000,PROFILE)[2]['status']=='cooldown'


def test_event_inventory_keeps_entry_identity_and_original_source_reversal_exits():
    state=reentry.initial_state(ORIGIN-1000)
    p,state,_=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+1000,PROFILE)
    held,state,diag=reentry.candidate_point(point(),state,{'position_qty':.01},hourly(),context(),
        ORIGIN+2*reentry.HOUR,PROFILE)
    assert held['position_id']==p['position_id'] and diag['status']=='holding_event'
    reverse=point(-1,position_id='new-short',origin_signal_time_ms=ORIGIN+2*reentry.HOUR)
    p2,s2,_=reentry.candidate_point(reverse,state,{'position_qty':.01},hourly(),context(-1),
        ORIGIN+2*reentry.HOUR+1000,PROFILE)
    assert p2==reverse and s2['attempts']==0 and s2['opportunity'] is None


def test_immediate_source_close_preserves_cooldown_before_next_state_poll():
    state=reentry.initial_state(ORIGIN-1000)
    p,state,_=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+1000,PROFILE)
    closed_account={'position_qty':0,'btc_pullback_pending':{'status':'filled',
        'source':{'position_id':p['position_id']}}}
    _,state,_=reentry.candidate_point(point(signed_qty=0),state,closed_account,hourly(),context(),
        ORIGIN+400000,PROFILE)
    assert state['last_closed_ms']==ORIGIN+400000
    _,state,diag=reentry.candidate_point(point(),state,{},hourly(),context(),ORIGIN+500000,PROFILE)
    assert diag['status']=='hour_already_checked' or diag['status']=='cooldown'
    assert state['last_closed_ms']==ORIGIN+400000 and state['attempts']==1
