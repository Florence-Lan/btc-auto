"""Real book/financial paths, immutable feed receipts, and narrow guard scope."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_stock_technology_trial as trial
from test_stock_entry_timing_trial import worker as timing_worker
from test_stock_technology_guard import point


def technology_worker(tmp_path,monkeypatch,*,guarded=True,short=False):
    original,clock,bars=timing_worker(tmp_path,monkeypatch,short=short)
    config=deepcopy(original.config)
    if guarded:
        config['technology_guard_policy']=deepcopy(trial.guard.POLICY)
    state=trial.paper.read_json(original.path)
    state['rule']=config
    trial.paper.write_json(original.path,state)
    result=trial.TechnologyAccount(original.path,original.venue,original.symbol,config,
        activated_at_ms=original.activated_at_ms,technology_arm='guarded_normal' if guarded else 'baseline_normal')
    result.macro_context_sha256='a'*64
    result.technology_context_sha256='b'*64
    return result,clock,bars


def ready_rows(worker,clock,*,weak):
    now=clock['now']
    worker.activated_at_ms=now-120*60000
    state=trial.paper.read_json(worker.path)
    state.update(start_ms=worker.activated_at_ms,signal_active_after_ms=worker.activated_at_ms)
    trial.paper.write_json(worker.path,state)
    rows=[point(now-31*60000,150,previous=151,lag=0),point(now-60000,149 if weak else 150.5,previous=151,lag=0)]
    for row in rows:
        row.update(provider_reference_first_seen_at_ms=worker.activated_at_ms,
                   provider_reference_available_at_ms=worker.activated_at_ms)
        row['market_session']['valid_until_ms']=now+60000
    worker.technology_rows=rows
    worker.technology_input_error=None


def test_baseline_delegates_original_entry_and_financial_execution(tmp_path,monkeypatch):
    original,_,_=timing_worker(tmp_path/'original',monkeypatch)
    worker,_,_=technology_worker(tmp_path/'new',monkeypatch,guarded=False)
    left,right=original.step(),worker.step()
    for key in ('wallet_balance','equity','position_qty','fees_paid','funding_pnl','realized_pnl','position'):
        assert left[key]==right[key],key
    assert left['fill_count_total']==right['fill_count_total']==1
    trial.infrastructure.validate_financial_journals(worker.path.parent,right)


@pytest.mark.parametrize('short',[False,True])
def test_unknown_only_blocks_new_long_and_never_grants_extra_short(tmp_path,monkeypatch,short):
    worker,_,_=technology_worker(tmp_path,monkeypatch,short=short)
    state=worker.step()
    assert state['fill_count_total']==(1 if short else 0)
    assert state['wallet_balance']<1000 if short else state['wallet_balance']==1000
    trial.infrastructure.validate_financial_journals(worker.path.parent,state)


@pytest.mark.parametrize('weak',[True,False])
def test_fixed_nq_hypothesis_controls_only_otherwise_valid_entries(tmp_path,monkeypatch,weak):
    worker,clock,_=technology_worker(tmp_path,monkeypatch)
    ready_rows(worker,clock,weak=weak)
    state=worker.step()
    assert state['fill_count_total']==(0 if weak else 1)
    trial.infrastructure.validate_financial_journals(worker.path.parent,state)


def test_nq_failure_cannot_block_existing_protective_stop(tmp_path,monkeypatch):
    worker,clock,_=technology_worker(tmp_path,monkeypatch)
    ready_rows(worker,clock,weak=False)
    opened=worker.step()
    assert opened['position']
    clock['price']=opened['position']['stop']-1
    clock['hourly_error']=True
    worker.technology_rows=[]
    worker.technology_input_error='nq_unavailable'
    closed=worker.step()
    assert closed['position'] is None and closed['fill_count_total']==2
    assert closed['fills'][-1]['reason']=='protective_stop'
    trial.infrastructure.validate_financial_journals(worker.path.parent,closed)


def test_new_failure_at_fill_cannot_charge_fees_or_create_position(tmp_path,monkeypatch):
    worker,clock,_=technology_worker(tmp_path,monkeypatch)
    ready_rows(worker,clock,weak=False)
    original=trial.paper.book_fill
    def fill(*args,**kwargs):
        result=original(*args,**kwargs)
        worker.technology_input_error='nq_became_unavailable'
        return result
    monkeypatch.setattr(trial.paper,'book_fill',fill)
    state=worker.step()
    assert state['position'] is None and state['fill_count_total']==0
    assert state['wallet_balance']==1000 and state['fees_paid']==0


def test_only_guard_differs_and_old_source_experiment_is_untouched():
    _,rules,fp=trial.load_design()
    assert len(trial.ARMS)==4
    for arms in rules.values():
        for cost in trial.COSTS:
            guarded=deepcopy(arms['guarded_'+cost])
            assert guarded.pop('technology_guard_policy')==trial.guard.POLICY
            assert guarded==arms['baseline_'+cost]
    assert 'scripts/run_stock_technology_trial.py' in fp['source_sha256']


def test_independent_epoch_has_twelve_accounts_and_refuses_corrupt_resume(tmp_path):
    spec,rules,fp=trial.load_design()
    output=tmp_path/'new';output.mkdir()
    epoch=spec['effective_from_ms']+1000
    manifest=trial.prepare(output,spec,rules,fp,epoch)
    paths=list(output.glob('*/*/state.json'))
    assert len(paths)==12
    assert all(trial.paper.read_json(p)['start_ms']==epoch for p in paths)
    assert trial.prepare(output,spec,rules,fp,None,True)==manifest
    paths[0].with_name('fills.jsonl').unlink()
    with pytest.raises((ValueError,trial.infrastructure.LedgerIntegrityError)):
        trial.prepare(output,spec,rules,fp,None,True)


def test_funding_venue_never_polls_bulk_market_endpoints():
    class Forbidden:
        def get(self,*_): raise AssertionError('No dispatch allowed')
    venue=trial.FundingOnlyVenue(Forbidden())
    for endpoint in ('time','depth','klines','premiumIndex','fundingRate','exchangeInfo'):
        with pytest.raises(RuntimeError,match='bulk requests'):
            venue.get(endpoint)


def test_quote_journal_deduplicates_receipts_and_records_revisions(tmp_path,monkeypatch):
    root=tmp_path/'root';source=root/'data/cross_market_observer';source.mkdir(parents=True)
    monkeypatch.setattr(trial,'ROOT',root)
    now=trial.time.time()*1000
    row=point(int(now)-600000,29900)
    row['market_session']['valid_until_ms']=int(now)+100000
    def write(value):
        (source/'latest_observations.json').write_text(json.dumps({'generated_at_ms':int(now),'observations':{'nq_dec_2026':value}}),encoding='utf-8')
    output=tmp_path/'new';output.mkdir()
    journal=trial.QuoteJournal(output)
    write(row);journal.refresh(output)
    assert len(journal.rows)==1 and journal.input_error is None
    first=deepcopy(journal.rows[0])
    journal.refresh(output)
    assert journal.rows==[first]
    row['value']-=1;write(row);journal.refresh(output)
    assert len(journal.rows)==2 and journal.rows[-1]['record_type']=='conflict'
    assert journal.rows[0]==first
    resumed=trial.QuoteJournal(output)
    assert resumed.rows==journal.rows


def test_tail_does_not_consume_a_partially_written_receipt(tmp_path):
    p=tmp_path/'observations.jsonl'
    p.write_bytes(b'{"shared_context_sha256":"old"}\n{"shared_context_sha256":"new"}')
    assert trial.last_json_line(p)['shared_context_sha256']=='old'


def test_archived_clock_is_not_renewed_and_fresh_mark_survives_clock_failure(tmp_path,monkeypatch):
    source=tmp_path/'source';monkeypatch.setattr(trial,'SOURCE',source)
    symbol='MUUSDT';now=int(time.time()*1000)
    original_monotonic=time.monotonic()-20
    clock={'serverTime':now-20000,'_received_monotonic':original_monotonic}
    mark={'time':now-1000,'markPrice':'100','indexPrice':'100'}
    def block(directory,value):
        raw=trial.infrastructure.canonical(value);sha=trial.digest(raw)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/(sha+'.json.gz')).write_bytes(gzip.compress(raw,mtime=0))
        return sha
    entries=[{'label':label,'payload_sha256':block(source/'context_blocks',value),'error':None}
             for label,value in [('clock',clock),('mark',mark)]]
    context={'schema_version':1,'symbol':symbol,'received_at_utc':trial.paper.arithmetic.iso(now),
             'responses':entries}
    sha=block(source/'context_archive'/symbol,context)
    observation=source/symbol/'baseline_normal/observations.jsonl'
    observation.parent.mkdir(parents=True)
    observation.write_text(json.dumps({'shared_context_sha256':sha})+'\n',encoding='utf-8')
    _,responses=trial.source_batch(symbol)
    assert next(v for k,v,e in responses if k=='clock')['_received_monotonic']==original_monotonic
    checked,outcome=trial.shared.checked_snapshot(responses)
    assert outcome['source']=='unavailable'
    assert next(v for k,v,e in checked if k=='mark')==mark
    path=source/'context_blocks'/(entries[0]['payload_sha256']+'.json.gz')
    path.write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError,match='hash mismatch'):
        trial.source_batch(symbol)
