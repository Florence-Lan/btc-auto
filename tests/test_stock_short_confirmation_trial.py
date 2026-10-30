"""Real execution and ledger invariants for the short-only confirmation trial."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_stock_short_confirmation_trial as trial
from test_stock_swing_confirmation import worker_fixture,hourly_bars
from test_stock_short_confirmation import price_led_short_bars
from test_stock_macro_shadow import macro_permit

HOUR=trial.timing_trial.HOUR


def worker(tmp_path,monkeypatch,*,variant='candidate',cost='normal',side=-1,price_led=False,epoch=None):
    original,_,clock=worker_fixture(tmp_path/'source',monkeypatch,side)
    _,rules,_=trial.load_design()
    cfg=rules['MUUSDT'][variant+'_'+cost]
    if price_led:
        bars=[replace(b,time_ms=b.time_ms+100*HOUR) for b in price_led_short_bars()]
    else:
        # Leave genuine extension/gap room for BOTH declared slippage costs.
        bars=[replace(b,high=b.close+.06,low=b.close-.06) for b in hourly_bars(1,500,step=.2,start=100)]
        if side==-1:
            bars=[trial.paper.signals.Candle(b.time_ms,300-b.open,300-b.low,300-b.high,300-b.close,b.volume) for b in bars]
    clock.update(price=bars[-1].close,hourly_error=False)
    original_get=original.venue.get.side_effect
    def get(endpoint,params=None):
        if endpoint=='klines' and params['interval']=='1h':
            if clock['hourly_error']:raise RuntimeError('Fixture hourly unavailable')
            return [[b.time_ms,b.open,b.high,b.low,b.close,b.volume,b.time_ms+HOUR-1] for b in bars]
        if endpoint=='premiumIndex':
            return {'time':clock['now'],'markPrice':str(clock['price']),'indexPrice':str(clock['price']),
                    'lastFundingRate':'0','nextFundingTime':clock['now']+HOUR}
        if endpoint=='depth':
            return {'T':clock['now'],'bids':[[str(clock['price']-.0001),'1000']],
                    'asks':[[str(clock['price']+.0001),'1000']]}
        return original_get(endpoint,params)
    original.venue.get.side_effect=get
    monkeypatch.setattr(trial.timing_trial.macro,'evaluate',lambda cfg,symbol,at,**kw:macro_permit(at))
    monkeypatch.setattr(trial.paper.stock_event_risk,'evaluate',lambda cfg,symbol,at,**kw:{
        'enabled':True,'allowed':True,'status':'clear','reasons':[], 'checked_at_ms':at,
        'valid_until_ms':at+HOUR,'coverage_end_ms':at+24*HOUR,'next_event':None})
    start=600*HOUR-10000 if epoch is None else epoch
    path=tmp_path/'account/state.json'
    state=trial.paper.initial_stock('MUUSDT',start,cfg)
    state['signal_active_after_ms']=start
    trial.paper.write_json(path,state)
    for name in ('entries.jsonl','fills.jsonl','funding.jsonl','closed_trades.jsonl'):
        path.with_name(name).write_text('',encoding='utf-8')
    w=trial.ShortAccount(path,original.venue,'MUUSDT',cfg,activated_at_ms=start,technology_arm=variant+'_'+cost)
    w.macro_context_sha256='a'*64;w.shared_context={'MUUSDT':'b'*64}
    return w,clock,bars


@pytest.mark.parametrize('cost',['normal','double'])
def test_real_price_led_short_executes_while_strict_baseline_stays_flat(tmp_path,monkeypatch,cost):
    baseline,_,_=worker(tmp_path/'base',monkeypatch,variant='baseline',cost=cost,price_led=True)
    candidate,_,_=worker(tmp_path/'candidate',monkeypatch,cost=cost,price_led=True)
    left,right=baseline.step(),candidate.step()
    assert left['position'] is None and left['fill_count_total']==0
    assert right['position']['direction']==-1 and right['fill_count_total']==1
    assert right['signal_diagnostics']['short_confirmation']['admitted_by']=='price_led_short'
    assert right['fills'][0]['short_confirmation_variant']=='candidate'
    assert right['fills'][0]['short_confirmation_context_sha256']=='b'*64
    assert right['stock_entry_timing']['allowed']
    trial.infrastructure.validate_financial_journals(candidate.path.parent,right)


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('cost',['normal','double'])
def test_original_strict_entries_and_financial_fields_are_equal(tmp_path,monkeypatch,side,cost):
    baseline,_,_=worker(tmp_path/'base',monkeypatch,variant='baseline',side=side,cost=cost)
    candidate,_,_=worker(tmp_path/'candidate',monkeypatch,side=side,cost=cost)
    left,right=baseline.step(),candidate.step()
    assert left['fill_count_total']==right['fill_count_total']==1
    for key in ('position','wallet_balance','equity','realized_pnl','fees_paid','funding_pnl','position_qty'):
        assert left[key]==right[key],key
    trial.infrastructure.validate_financial_journals(candidate.path.parent,right)


def test_new_short_still_has_to_hold_frozen_breakout_at_actual_fill(tmp_path,monkeypatch):
    w,_,bars=worker(tmp_path,monkeypatch,price_led=True)
    original=trial.paper.book_fill
    def fill(*args,**kwargs):
        result=original(*args,**kwargs)
        result['price']=min(b.low for b in bars[-21:-1])+.01
        return result
    monkeypatch.setattr(trial.paper,'book_fill',fill)
    state=w.step()
    assert state['position'] is None and state['fill_count_total']==0
    assert state['wallet_balance']==1000 and state['fees_paid']==0


def test_signal_outage_cannot_block_protective_exit_of_price_led_short(tmp_path,monkeypatch):
    w,clock,_=worker(tmp_path,monkeypatch,price_led=True)
    state=w.step();assert state['position']
    clock.update(price=state['position']['stop']+1,hourly_error=True)
    stopped=w.step()
    assert stopped['position'] is None and stopped['fill_count_total']==2
    assert stopped['fills'][-1]['reason']=='protective_stop'
    trial.infrastructure.validate_financial_journals(w.path.parent,stopped)


def test_equal_inception_boundary_cannot_be_backfilled(tmp_path,monkeypatch):
    w,_,_=worker(tmp_path,monkeypatch,price_led=True,epoch=600*HOUR)
    state=w.step()
    assert state['position'] is None and state['fill_count_total']==0
    assert state['signal_diagnostics']['admission']['reasons']==['signal_not_strictly_post_inception']


def test_extending_market_drop_cannot_revive_an_expired_hourly_short(tmp_path,monkeypatch):
    w,clock,bars=worker(tmp_path,monkeypatch,price_led=True)
    clock['price']=bars[-1].close-5
    first=w.step()
    assert first['fill_count_total']==0
    clock.update(now=601*HOUR+60000,hourly_error=True)
    second=w.step()
    assert second['pending_signal'] is None and second['fill_count_total']==0


def test_design_only_changes_short_policy_and_uses_current_main_technical_rules():
    _,rules,fp=trial.load_design()
    for arms in rules.values():
        for cost in trial.COSTS:
            cfg=deepcopy(arms['candidate_'+cost])
            assert cfg.pop('short_confirmation_policy')==trial.short.POLICY
            assert cfg==arms['baseline_'+cost]
            assert cfg['entry_timing_variant']=='baseline'
            assert 'entry_timing_candidate_policy' not in cfg and 'technology_guard_policy' not in cfg
            assert cfg['signal_family']=='donchian20' and cfg['risk_fraction_per_trade']==.0025
    assert 'scripts/stock_short_confirmation.py' in fp['source_sha256']


def test_twelve_new_epochs_and_corrupt_financial_resume_is_refused(tmp_path):
    spec,rules,fp=trial.load_design()
    root=tmp_path/'new';root.mkdir()
    epoch=spec['effective_from_ms']+1
    manifest=trial.prepare(root,spec,rules,fp,epoch)
    states=list(root.glob('*/*/state.json'));assert len(states)==12
    assert all(trial.paper.read_json(p)['start_ms']==epoch for p in states)
    assert trial.prepare(root,spec,rules,fp,None,True)==manifest
    states[0].with_name('funding.jsonl').unlink()
    with pytest.raises((ValueError,trial.infrastructure.LedgerIntegrityError)):
        trial.prepare(root,spec,rules,fp,None,True)


def test_short_trial_does_not_collect_or_apply_an_extra_nq_market_gate(tmp_path):
    journal=trial.NoAdditionalMarketJournal(tmp_path)
    assert journal.refresh(tmp_path)
    assert not journal.rows and journal.input_error is None and list(tmp_path.iterdir())==[]
