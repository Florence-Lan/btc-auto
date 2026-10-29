"""Causal observation prefixes and the new-long-only NQ hypothesis."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import stock_technology_guard as guard

M = 60000
EPOCH = int(datetime.fromisoformat('2026-10-29T14:00:00+00:00').timestamp()*1000)
CFG = {'technology_guard_policy':guard.POLICY}


def point(observed, value, *, previous=30000, lag=10*M):
    received = observed+lag
    return {'observed_at_ms':observed,'first_seen_at_ms':received,'available_at_ms':received,
        'received_at_ms':received,'ingested_at_ms':received+1,'source_snapshot_sha256':'a'*64,
        **{k:guard.POLICY[k] for k in ('source_id','source_config_id','provider','source_url','unit','observation_timezone','contract_review_cutoff_at_ms')},
        'instrument':{**{k:guard.POLICY[k] for k in ('symbol','contract_month','exchange','currency')},
            'automatic_roll':False,'continuous_contract':False},
        'precision':'timestamp','eligible_intraday':True,
        'market_session':{'phase':'open','is_actual_market_status':False,'asof_ms':received,
                          'valid_until_ms':EPOCH+24*60*M},
        'observed_date':datetime.fromtimestamp(observed/1000,guard.ZoneInfo('America/New_York')).date().isoformat(),
        'value':value,'provider_previous_close':previous,
        'return_basis':'same_named_instrument_provider_previousClose',
        'provider_reference_first_seen_at_ms':EPOCH-60*M,'provider_reference_available_at_ms':EPOCH-60*M}


def evaluate(rows, **kwargs):
    return guard.evaluate(rows,CFG,asof_ms=EPOCH+70*M+500,inception_ms=EPOCH,**kwargs)


def test_both_negative_refuses_only_new_long_and_preserves_input():
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    before=deepcopy(rows)
    result=evaluate(rows)
    assert result['status']=='weak' and not result['allowed'] and result['counts_as_weak_filter']
    assert result['daily_return_pct']<0 and result['return_30m_pct']<0
    assert result['protective_exits_allowed'] and rows==before
    assert evaluate([],direction=-1)['allowed']
    assert evaluate([],new_risk=False)['allowed']


@pytest.mark.parametrize('reference,current',[(29800,29900),(30100,30000),(29900,29900),(30000,30000),(30000,30100)])
def test_negative_daily_return_alone_or_flat_reference_is_not_weak(reference,current):
    result=evaluate([point(EPOCH+30*M,reference),point(EPOCH+60*M,current)])
    assert result['status']=='clear' and result['allowed'] and not result['counts_as_weak_filter']


@pytest.mark.parametrize('gap,expected',[(29,'unknown'),(30,'weak'),(35,'weak'),(36,'unknown')])
def test_reference_uses_thirty_to_thirty_five_observation_minutes(gap,expected):
    result=evaluate([point(EPOCH+(60-gap)*M,29900),point(EPOCH+60*M,29800)])
    assert result['status']==expected


def test_real_inception_and_observation_warmup_cannot_be_backfilled():
    rows=[point(EPOCH-15*M,29900,lag=0),point(EPOCH+15*M,29800,lag=0)]
    outcome=guard.evaluate(rows,CFG,asof_ms=EPOCH+20*M,inception_ms=EPOCH)
    assert outcome['status']=='unknown' and outcome['reasons']==['trial_history_warming_up']
    outcome=guard.evaluate(rows,CFG,asof_ms=EPOCH+30*M,inception_ms=EPOCH)
    assert outcome['status']=='unknown' and not outcome['counts_as_weak_filter']


def test_late_future_conflict_does_not_contaminate_past_decision():
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    future=deepcopy(rows[-1]);future.update(value=29700,ingested_at_ms=EPOCH+80*M,record_type='conflict')
    assert evaluate(rows+[future])['status']=='weak'
    outcome=guard.evaluate(rows+[future],CFG,asof_ms=EPOCH+80*M,inception_ms=EPOCH)
    assert outcome['status']=='unknown' and 'conflict' in outcome['reasons'][0]


def test_changed_observation_is_not_silently_overwritten():
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    revision=deepcopy(rows[-1]);revision.update(value=30100,ingested_at_ms=EPOCH+70*M+2)
    assert evaluate(rows+[revision])['reasons']==['session_observation_conflict']


def test_duplicate_retains_first_trial_receipt():
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    later=deepcopy(rows[-1]);later['ingested_at_ms']+=100
    result=evaluate(rows+[later])
    assert result['latest']['ingested_at_ms']==rows[-1]['ingested_at_ms']


@pytest.mark.parametrize('mutation',[
    {'eligible_intraday':False},{'value':float('nan')},{'value':True},
    {'source_config_id':'b'*64},{'provider_previous_close':None},
    {'observed_date':'2026-10-28'},{'source_snapshot_sha256':'not-a-hash'},
    {'first_seen_at_ms':EPOCH+80*M},{'record_type':'replacement'},
    {'market_session':{'phase':'closed'}},
])
def test_unusable_inputs_are_unknown_not_a_successful_weak_filter(mutation):
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    rows[-1].update(mutation)
    result=evaluate(rows)
    assert result['status']=='unknown' and not result['allowed'] and not result['counts_as_weak_filter']


def test_reference_version_change_requires_new_history():
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    rows[-1]['provider_reference_first_seen_at_ms']+=M
    assert evaluate(rows)['status']=='unknown'


def test_stale_latest_cannot_be_renewed_by_a_new_ingestion_clock():
    rows=[point(EPOCH+30*M,29900),point(EPOCH+60*M,29800)]
    outcome=guard.evaluate(rows,CFG,asof_ms=EPOCH+80*M+1,inception_ms=EPOCH)
    assert outcome['reasons']==['latest_observation_stale']


@pytest.mark.parametrize('key,value',[('daily_return_weak_below_pct',-1),('unknown_new_long_allowed',True),('lookback_minutes',25)])
def test_declared_policy_cannot_be_silently_tuned(key,value):
    config=deepcopy(CFG);config['technology_guard_policy'][key]=value
    with pytest.raises(ValueError,match='fixed_technology'):
        guard.evaluate([],config,asof_ms=EPOCH+M,inception_ms=EPOCH)
