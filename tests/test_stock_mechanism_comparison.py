import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_stock_mechanism_comparison as comparison


def test_exchange_clock_uses_monotonic_time_despite_local_clock_skew(monkeypatch):
    venue = Mock()
    venue.get.return_value = {'serverTime':100000}
    values = iter([1.,1.1,1.6])
    monkeypatch.setattr(comparison.time,'monotonic',lambda:next(values))
    monkeypatch.setattr(comparison.time,'time',lambda:92.)
    clock, info = comparison.exchange_clock(venue)
    assert clock() == 100500
    assert info['local_clock_offset_ms'] == 8000


def test_restart_preserves_cash_and_rejects_changed_rules(tmp_path, monkeypatch):
    configs = {'new':{'signal_timeframe':'1h'}, 'old':{'signal_timeframe':'5m'}}
    monkeypatch.setattr(comparison.paper, 'stock_config', lambda p,s:dict(configs[p['stock_research_profile']]))
    manifest = comparison.prepare(tmp_path,'new','old')
    path = tmp_path/'candidate/state.json'
    state = comparison.paper.read_json(path)
    state['wallet_balance'] = 997
    comparison.paper.write_json(path,state)
    assert comparison.prepare(tmp_path,'new','old') == manifest
    assert comparison.paper.read_json(path)['wallet_balance'] == 997
    configs['new']['signal_timeframe'] = '4h'
    with pytest.raises(ValueError,match='existing balances'):
        comparison.prepare(tmp_path,'new','old')
    assert comparison.paper.read_json(path)['wallet_balance'] == 997


def test_arms_share_execution_inputs_but_have_their_own_signal_series():
    candidate, control = Mock(),Mock()
    control.signal_timeframe = '5m'
    common = [('clock',{'serverTime':1},None),('book',{'asks':[['100','5']]},None),
              ('signal',['one-hour'],None),('five',['five-minute-volume'],None)]
    candidate.market_data.return_value = common
    control.venue.get.return_value = ['five-minute-signals']
    candidate.step.return_value = control.step.return_value = {'equity':1000,'status':'healthy'}
    comparison.tick({'candidate':candidate,'control':control})
    cand = {k:v for k,v,e in candidate.step.call_args.args[0]}
    base = {k:v for k,v,e in control.step.call_args.args[0]}
    assert cand['book'] == base['book'] and cand['clock'] == base['clock'] and cand['five'] == base['five']
    assert cand['signal'] == ['one-hour'] and base['signal'] == ['five-minute-signals']
    cand['book']['asks'][0][0] = '99'
    assert base['book']['asks'][0][0] == '100'
    control.venue.get.assert_called_once_with('klines',{'symbol':'SNDKUSDT','interval':'5m','limit':500})


def test_published_profile_preserves_other_stock_rules_and_risk_budget():
    paper = comparison.paper
    old = {'stock_research_profile':'config/stock_one_r_half_atr_paper_20261006.json'}
    new = {'stock_research_profile':'config/stock_hourly_aligned_breakout_paper_20261007.json'}
    for symbol in ('MUUSDT','SKHYNIXUSDT'):
        assert paper.stock_config(new,symbol) == paper.stock_config(old,symbol)
    cfg = paper.stock_config(new,'SNDKUSDT')
    assert cfg['signal_timeframe'] == '1h' and cfg['signal_family'] == 'donchian20'
    assert cfg['require_price_slow'] and cfg['ema_slow'] == 60
    assert 'profit_exit_policy' not in cfg and cfg['target_margin_return'] == .6
    assert cfg['risk_fraction_per_trade'] == .0025
    assert cfg['portfolio_gross_notional_fraction'] == 1 and cfg['leverage'] == 10
