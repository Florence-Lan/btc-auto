"""Overnight cutoff and whole-position net accounting, including late funding."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stock_overnight_report import completed_positions, make_report, markdown
from run_stock_overnight_validation import request_end_exit, SymbolDispatch
import run_stock_overnight_validation as overnight
import run_stock_expectancy_shadow as single
from stock_swing_signals import Candle, DEFAULT_CONFIG
from stock_profit_candidate import FAST_CONFIG


def fill(seq,qty,fee,time,reason,gross=0):
    return {'sequence':seq,'qty':qty,'fee':fee,'time_ms':time,'reason':reason,'gross_pnl':gross}


def test_partial_exit_is_not_a_completed_trade_and_late_funding_is_counted():
    entry=fill(1,2,.2,100,'fresh_closed_1h_signal')
    partial=fill(2,1,.1,200,'protective_stop',2)
    funding=[{'fundingTime':150,'debit':.3},{'fundingTime':250,'debit':-.1}]
    rounds,current=completed_positions([entry,partial],funding)
    assert not rounds
    assert current['net_pnl']==pytest.approx(1.5)
    final=fill(3,1,.1,300,'overnight_validation_end',-1)
    rounds,current=completed_positions([entry,partial,final],funding)
    assert current is None
    assert len(rounds)==1
    assert rounds[0]['net_pnl']==pytest.approx(.4)
    # A funding record fetched after closure still belongs to that trade.
    funding.append({'fundingTime':290,'debit':.5})
    rounds,_=completed_positions([entry,partial,final],funding)
    assert rounds[0]['net_pnl']==pytest.approx(-.1)


def test_funding_boundary_excludes_entry_timestamp_and_includes_exit_timestamp():
    fills=[fill(1,1,.1,100,'fresh_closed_1h_signal'),fill(2,1,.1,200,'overnight_validation_end',1)]
    rounds,_=completed_positions(fills,[{'fundingTime':200,'debit':.3}])
    assert rounds[0]['net_pnl']==pytest.approx(.5)
    with pytest.raises(ValueError,match='Funding'):
        completed_positions(fills,[{'fundingTime':100,'debit':.3}])


def test_cutoff_schedules_exit_preserves_protective_exit_and_never_resets_money(tmp_path):
    path=tmp_path/'state.json'
    state={'wallet_balance':997.,'position':{'qty':2,'stop':90}}
    path.write_text(json.dumps(state))
    request_end_exit(path,123)
    updated=json.loads(path.read_text())
    assert updated['wallet_balance']==997
    assert updated['position']['qty']==2
    assert updated['position']['pending_exit']=='overnight_validation_end'
    assert updated['position']['trigger_observed_at_ms']==123
    updated['position']['pending_exit']='protective_stop'
    path.write_text(json.dumps(updated))
    request_end_exit(path,456)
    assert json.loads(path.read_text())['position']['pending_exit']=='protective_stop'


def test_symbol_dispatch_does_not_mix_strategies():
    class Adapter:
        def __init__(self,name): self.name=name
        def signal_at(self,*_): return self.name
    dispatch=SymbolDispatch({'MUUSDT':Adapter('pullback'),'SNDKUSDT':Adapter('volume')})
    assert dispatch.signal_at([],{},0,{'symbols':['MUUSDT']})=='pullback'
    assert dispatch.signal_at([],{},0,{'symbols':['SNDKUSDT']})=='volume'


def test_empty_night_has_no_expectancy_or_false_success(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'started_at_utc':'2026-10-05T14:00:00+00:00','end_ms':2000}))
    for cost in ('normal_cost','double_cost'):
        folder=tmp_path/'MUUSDT'/cost;folder.mkdir(parents=True)
        state={'symbol':'MUUSDT','status':'healthy','initial_balance':1000,'wallet_balance':1000,'equity':1000,
            'fees_paid':0,'funding_pnl':0,'checked_at_utc':'1970-01-01T00:00:01+00:00','position':None,'max_drawdown_pct':0}
        (folder/'state.json').write_text(json.dumps(state))
    result=make_report(tmp_path,{'experiment_id':'test','symbol_profiles':{'MUUSDT':{}},'end_utc':'1970-01-01T00:00:02+00:00'},1000)
    account=result['accounts']['MUUSDT']['normal_cost']
    assert account['completed_positions']==0
    assert account['mean_closed_net_pnl_usdt'] is None
    assert account['net_win_rate_pct'] is None
    assert account['accounting_reconciled']
    assert not account['retrospective_sample_gate_passed']
    assert '零成交不计为成功' in markdown(result)


@pytest.mark.parametrize('arm',['trend_pullback','compression_breakout','trend_volume_breakout_net60'])
def test_seeded_forward_dispatch_can_emit_real_causal_signal(tmp_path,monkeypatch,arm):
    monkeypatch.setattr(overnight,'ROOT',tmp_path)
    monkeypatch.setattr(single,'ROOT',tmp_path)
    bars=[]
    for i in range(900):
        close=100+i*.2
        bars.append(Candle(i*3_600_000,close-.08,close+.05,close-1.5,close,30 if i==899 else 10))
    seed=[[b.time_ms,b.open,b.high,b.low,b.close,b.volume,b.time_ms+3_600_000-1] for b in bars[:800]]
    cfg={**DEFAULT_CONFIG,**(FAST_CONFIG if arm!='trend_volume_breakout_net60' else {})}
    settings=single.GuardedSettings(cfg,None,999_999_999_999)
    adapter=(single.ForwardSignalAdapter(seed,tmp_path/'history.json.gz') if arm=='trend_volume_breakout_net60'
             else overnight.FastAdapter(seed,tmp_path/'history.json.gz',arm))
    result=adapter.signal_at(bars[-500:],{},499,settings)
    assert result is not None
    assert result.direction==1 and settings.side==1
    assert result.time_ms==bars[-1].time_ms
    assert (tmp_path/'history.json.gz').exists()
    assert len(list((tmp_path/'signal_observations').glob('*.json.gz')))==1
