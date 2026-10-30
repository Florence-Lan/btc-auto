"""Main-only dispatch and original clock receipts, with offline real-engine fills."""
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import main_stock_runtime as runtime
import run_parallel_simulation as paper
import stock_short_confirmation as short
import stock_swing_signals as signals
import stock_swing_profiles as profiles
from test_stock_short_confirmation import config, price_led_short_bars, mirror
from test_stock_swing_confirmation import worker_fixture

HOUR = runtime.HOUR


def unavailable(*args, **kwargs):
    raise ValueError('offline shared source unavailable')


def main_worker(tmp_path, monkeypatch, *, active_after=None):
    template, path, clock = worker_fixture(tmp_path, monkeypatch)
    cfg=deepcopy(template.config)
    cfg.pop('stock_event_risk_policy',None)
    cfg['short_confirmation_policy']=deepcopy(short.POLICY)
    cfg['entry_timing_policy']=deepcopy(paper.stock_entry_timing.POLICY)
    bars=[replace(bar,time_ms=bar.time_ms+100*HOUR) for bar in price_led_short_bars()]
    clock['price']=bars[-1].close
    original_get=template.venue.get.side_effect
    def venue(endpoint,params=None):
        if endpoint=='klines' and params['interval']=='1h':
            if clock.get('hourly_error'):raise RuntimeError('Fixture hourly unavailable')
            return [[b.time_ms,b.open,b.high,b.low,b.close,b.volume,b.time_ms+HOUR-1] for b in bars]
        if endpoint=='premiumIndex':
            return {'time':clock['now'],'markPrice':str(clock['price']),'indexPrice':str(clock['price']),
                'lastFundingRate':'0','nextFundingTime':clock['now']+HOUR}
        if endpoint=='depth':
            return {'T':clock['now'],'bids':[[str(clock['price']-.0001),'1000']],
                'asks':[[str(clock['price']+.0001),'1000']]}
        return original_get(endpoint,params)
    template.venue.get.side_effect=venue
    state=paper.initial_stock('MUUSDT',600*HOUR-10000,cfg)
    state['signal_active_after_ms']=state['start_ms']
    state['short_confirmation_active_after_ms'] = 599 * HOUR if active_after is None else active_after
    paper.write_json(path, state)
    account = runtime.MainStockAccount(path, template.venue, template.symbol, cfg)
    monkeypatch.setattr(account, 'shared_clock_loader', unavailable)
    return account, clock, bars


def cached_clock(server=2_160_050_000, received=90., started=89.9):
    return {'serverTime': server, '_received_monotonic': received,
        '_public_clock_timing': {'started_monotonic': started,
            'received_monotonic': received, 'transport': 'requests'}}


@pytest.mark.parametrize('offset,active', [(1, True), (0, False), (-1, False)])
def test_activation_is_strict_and_never_changes_original_profile_globals(offset, active):
    bars = price_led_short_bars();cfg=config();boundary=bars[-1].time_ms+HOUR
    indicators=signals.compute_indicators(bars,cfg)
    original=profiles.signal_with_diagnostics
    signal, detail=runtime._SignalProfiles(boundary-offset).signal_with_diagnostics(bars,indicators,499,cfg)
    assert (signal is not None) is active
    assert detail['main_short_confirmation_runtime']['candidate_active_for_bar'] is active
    assert profiles.signal_with_diagnostics is original


def test_long_signal_is_identical_under_private_adapter():
    bars=mirror(price_led_short_bars());cfg=config();indicators=signals.compute_indicators(bars,cfg)
    expected, detail=profiles.signal_with_diagnostics(bars,indicators,499,cfg)
    actual, new=runtime._SignalProfiles(1).signal_with_diagnostics(bars,indicators,499,cfg)
    new.pop('main_short_confirmation_runtime')
    assert actual==expected and new==detail


def test_shared_clock_preload_skips_only_time_and_never_restamps_receipt(tmp_path,monkeypatch):
    account,clock,_=main_worker(tmp_path,monkeypatch)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:100.)
    payload=cached_clock(server=clock['now']-10_000)
    monkeypatch.setattr(account,'shared_clock_loader',lambda symbol:(deepcopy(payload),{'source':'shared_verified'}))
    account.venue.get.reset_mock()
    responses=account.market_data(force_rules=True,force_signal=True)
    got={label:(body,error) for label,body,error in responses}
    assert got['clock'][1] is None
    assert got['clock'][0]['_received_monotonic']==90.
    assert got['clock'][0]['_public_clock_timing']==payload['_public_clock_timing']
    assert got['clock'][0]['serverTime']==payload['serverTime']
    endpoints=[call.args[0] for call in account.venue.get.call_args_list]
    assert 'time' not in endpoints
    assert {'premiumIndex','depth','fundingRate','klines','exchangeInfo'}<=set(endpoints)
    assert account.venue.get is not runtime._ClockVenue.get


def test_expired_shared_clock_uses_normal_venue_instead_of_renewing_cache(tmp_path,monkeypatch):
    account,clock,_=main_worker(tmp_path,monkeypatch)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:106.)
    payload=cached_clock(server=clock['now']-16_000)
    monkeypatch.setattr(account,'shared_clock_loader',lambda symbol:(deepcopy(payload),{'source':'shared_verified'}))
    account.venue.get.reset_mock()
    responses=account.market_data()
    got={label:(body,error) for label,body,error in responses}
    assert got['clock'][1] is None and got['clock'][0]['serverTime']==clock['now']
    assert any(c.args[0]=='time' for c in account.venue.get.call_args_list)
    assert account._main_clock_diagnostic['source']=='venue_verified'


def test_fresh_shared_clock_can_recover_after_own_clock_failure(tmp_path,monkeypatch):
    account,clock,_=main_worker(tmp_path,monkeypatch)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:100.)
    payload=cached_clock(server=clock['now']-1000,received=99.,started=98.9)
    calls=[]
    def shared(symbol):
        calls.append(symbol)
        if len(calls)==1:raise ValueError('No source sample yet')
        return deepcopy(payload),{'source':'shared_verified'}
    monkeypatch.setattr(account,'shared_clock_loader',shared)
    get=account.venue.get.side_effect
    def venue(endpoint,params=None):
        if endpoint=='time':raise RuntimeError('Offline clock timeout')
        return get(endpoint,params)
    account.venue.get.side_effect=venue
    responses=account.market_data()
    clock_row=next(row for row in responses if row[0]=='clock')
    assert clock_row[2] is None and clock_row[1]['_received_monotonic']==99.
    assert account._main_clock_diagnostic['fallback_after_error_type']=='RuntimeError'


@pytest.mark.parametrize('change', ['missing_timing','missing_receipt','receipt_mismatch','future_receipt','old_receipt','slow_network','bool_receipt'])
def test_cached_clock_cannot_be_certified_without_original_valid_metadata(change):
    payload=cached_clock()
    if change=='missing_timing':payload.pop('_public_clock_timing')
    elif change=='missing_receipt':payload.pop('_received_monotonic')
    elif change=='receipt_mismatch':payload['_received_monotonic']=91.
    elif change=='future_receipt':payload=cached_clock(received=101.,started=100.9)
    elif change=='old_receipt':payload=cached_clock(received=84.,started=83.9)
    elif change=='slow_network':payload=cached_clock(received=99.,started=93.)
    elif change=='bool_receipt':payload['_received_monotonic']=True
    with pytest.raises(ValueError):runtime.validate_shared_clock(payload,monotonic_now=100.)


def test_real_main_price_led_short_opens_and_keeps_original_entry_gates(tmp_path,monkeypatch):
    account,_,_=main_worker(tmp_path,monkeypatch)
    result=account.step()
    assert result['position']['direction']==-1 and result['fill_count_total']==1
    assert result['short_confirmation']['admitted_by']=='price_led_short'
    assert result['stock_entry_timing']['allowed']
    assert result['rule']['risk_fraction_per_trade']==.0025
    assert paper.read_json(account.path)['main_stock_runtime']['clock']['source']=='venue_verified'


def test_activation_equal_bar_does_not_resurrect_rejected_old_signal(tmp_path,monkeypatch):
    account,_,_=main_worker(tmp_path,monkeypatch,active_after=600*HOUR)
    result=account.step()
    assert result['position'] is None and result['fill_count_total']==0
    assert not result['signal_diagnostics']['main_short_confirmation_runtime']['candidate_active_for_bar']


def test_clock_and_signal_outage_do_not_disable_existing_stop(tmp_path,monkeypatch):
    account,clock,_=main_worker(tmp_path,monkeypatch)
    entered=account.step();assert entered['position']
    clock.update(price=entered['position']['stop']+1,hourly_error=True)
    get=account.venue.get.side_effect
    def broken_clock(endpoint,params=None):
        if endpoint=='time':raise RuntimeError('Clock unavailable')
        return get(endpoint,params)
    account.venue.get.side_effect=broken_clock
    stopped=account.step()
    assert stopped['position'] is None and stopped['fills'][-1]['reason']=='protective_stop'
    assert stopped['risk_monitor']['protection_fill_available']
    assert stopped['fill_count_total']==2


def test_existing_state_epoch_last_signal_pending_and_cash_not_reset_by_constructor(tmp_path,monkeypatch):
    account,_,_=main_worker(tmp_path,monkeypatch)
    state=paper.read_json(account.path)
    state.update(last_signal_time_ms=123,pending_signal={'sentinel':'preserved'},wallet_balance=987.)
    paper.write_json(account.path,state);before=account.path.read_bytes()
    runtime.MainStockAccount(account.path,account.venue,account.symbol,account.config)
    assert account.path.read_bytes()==before


def test_activation_cannot_be_silently_changed_while_running(tmp_path,monkeypatch):
    account,_,_=main_worker(tmp_path,monkeypatch)
    state=paper.read_json(account.path);state['short_confirmation_active_after_ms']+=1;paper.write_json(account.path,state)
    with pytest.raises(ValueError,match='activation_changed'):account.step()


def test_loader_checks_real_immutable_hash_and_receipt_instead_of_file_mtime(tmp_path):
    symbol='MUUSDT';payload=cached_clock();wall=1_800_000_000_000
    def block(directory,value):
        raw=json.dumps(value).encode();identity=hashlib.sha256(raw).hexdigest()
        directory.mkdir(parents=True,exist_ok=True);(directory/(identity+'.json.gz')).write_bytes(gzip.compress(raw))
        return identity
    payload_id=block(tmp_path/'context_blocks',payload)
    from datetime import datetime,timezone
    context={'schema_version':1,'symbol':symbol,'received_at_utc':datetime.fromtimestamp(wall/1000,timezone.utc).isoformat(),
        'responses':[{'label':'clock','payload_sha256':payload_id,'error':None}]}
    identity=block(tmp_path/'context_archive'/symbol,context)
    observations=tmp_path/symbol/'baseline_normal/observations.jsonl';observations.parent.mkdir(parents=True)
    observations.write_text(json.dumps({'shared_context_sha256':identity})+'\n'+ '{"partial":',encoding='utf-8')
    result,_=runtime.load_shared_clock(symbol,source_root=tmp_path,wall_ms=wall,monotonic_now=100.)
    assert result['_received_monotonic']==90.
    with pytest.raises(ValueError):runtime.load_shared_clock(symbol,source_root=tmp_path,wall_ms=wall,monotonic_now=106.)
    (tmp_path/'context_blocks'/(payload_id+'.json.gz')).write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError,match='block_invalid'):
        runtime.load_shared_clock(symbol,source_root=tmp_path,wall_ms=wall,monotonic_now=100.)


def test_truncated_shared_gzip_falls_back_to_original_venue_clock(tmp_path,monkeypatch):
    account,clock,_=main_worker(tmp_path/'account',monkeypatch)
    raw=json.dumps(cached_clock()).encode();identity=hashlib.sha256(raw).hexdigest()
    directory=tmp_path/'broken';directory.mkdir()
    (directory/(identity+'.json.gz')).write_bytes(gzip.compress(raw)[:-8])
    def shared(symbol):
        return runtime._block(directory,identity),{'source':'shared_verified'}
    monkeypatch.setattr(account,'shared_clock_loader',shared)
    account.venue.get.reset_mock()
    result=account.market_data()
    row=next(row for row in result if row[0]=='clock')
    assert row[2] is None and row[1]['serverTime']==clock['now']
    assert any(call.args[0]=='time' for call in account.venue.get.call_args_list)
    assert account._main_clock_diagnostic['source']=='venue_verified'


def test_each_market_batch_resets_clock_diagnostics_before_any_call(tmp_path,monkeypatch):
    account,_,_=main_worker(tmp_path,monkeypatch)
    account._main_clock_diagnostic={'source':'shared_verified','old':True}
    monkeypatch.setattr(runtime,'_BASE_MARKET_DATA',lambda *args,**kwargs:[])
    assert account.market_data()==[]
    assert account._main_clock_diagnostic=={'source':'not_observed'}
