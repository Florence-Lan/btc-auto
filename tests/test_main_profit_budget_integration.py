"""Real main-stock booking under the user-authorized net100/risk25 budget."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_parallel_simulation as paper
import main_stock_runtime as runtime
import trade_profit_budget as budget
from test_main_stock_runtime import main_worker, unavailable, HOUR


def worker(tmp_path, monkeypatch, **changes):
    original, clock, bars = main_worker(tmp_path, monkeypatch)
    cfg = {**original.config, budget.POLICY_KEY: deepcopy(budget.POLICY),
        'risk_fraction_per_trade': .025, 'portfolio_gross_notional_fraction': 10.,
        'target_margin_return': 1.2, **changes}
    state = paper.read_json(original.path)
    state.update(rule=cfg, trade_profit_budget_active_after_ms=599*HOUR)
    paper.write_json(original.path, state)
    result = runtime.MainStockAccount(original.path, original.venue, original.symbol, cfg)
    monkeypatch.setattr(result, 'shared_clock_loader', unavailable)
    return result, clock, bars


def test_main_budget_books_a_real_net100_plan_and_preserves_technical_stop(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    result = account.step()
    assert result['position'] is not None
    decision = result['entry_profit_budget']
    assert decision['actual_fill_checked'] and decision['allowed']
    assert decision['expected_net_profit_usdt'] >= 100
    assert decision['planned_loss_usdt'] <= 25
    assert decision['net_reward_risk'] >= 4
    assert result['position']['qty'] == decision['admitted_quantity']
    assert result['position']['stop'] == decision['technical_stop_price']
    assert result['position']['trade_profit_budget']['entry_rule'] == result['rule']
    assert result['wallet_balance'] == pytest.approx(1000-result['fills'][-1]['fee'])


def test_wide_stop_cannot_be_tightened_or_upsized_to_force_net100(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch,
        target_margin_return=.6, min_stop_fraction=.045)
    result = account.step()
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['signal_status'] == 'trade_profit_budget_blocked'
    assert not result['entry_profit_budget']['allowed']
    assert 'net_reward_risk_below_minimum' in result['entry_profit_budget']['reasons']
    assert result['rule']['min_stop_fraction'] == .045


def test_observed_underfill_cannot_book_a_smaller_profit_goal(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    get = account.venue.get.side_effect
    def thin(endpoint, params=None):
        result = get(endpoint, params)
        if endpoint == 'depth':
            result['bids'][0][1] = '5'
        return result
    account.venue.get.side_effect = thin
    result = account.step()
    assert result['position'] is None and result['fill_count_total'] == 0
    assert not result['entry_profit_budget']['allowed']
    assert 'actual_quantity_below_net_profit_goal' in result['entry_profit_budget']['reasons']


def test_grandfathered_position_keeps_original_rule_and_quantity(tmp_path, monkeypatch):
    original, clock, _ = main_worker(tmp_path, monkeypatch)
    before = original.step()
    old_cfg = deepcopy(original.config)
    old_position = deepcopy(before['position'])
    cfg = {**old_cfg, budget.POLICY_KEY: deepcopy(budget.POLICY),
           'risk_fraction_per_trade': .025, 'portfolio_gross_notional_fraction': 10.}
    before.update(rule=cfg, trade_profit_budget_active_after_ms=clock['now'],
                  trade_profit_budget_legacy_position_rule=old_cfg)
    paper.write_json(original.path, before)
    account = runtime.MainStockAccount(original.path, original.venue, original.symbol, cfg)
    monkeypatch.setattr(account, 'shared_clock_loader', unavailable)
    clock['now'] += 1000
    result = account.step()
    for key in ('qty','entry','entry_time','initial_stop','stop','margin'):
        assert result['position'][key] == old_position[key]
    assert 'trade_profit_budget' not in result['position']
    assert result['fill_count_total'] == 1
    assert result['main_stock_runtime']['trade_profit_budget']['grandfathered_existing_position']
    assert account.config == cfg


def test_new_budget_stop_protection_survives_clock_and_hourly_failure(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    entered = account.step()
    clock.update(price=entered['position']['stop']+1, hourly_error=True)
    get = account.venue.get.side_effect
    def broken(endpoint, params=None):
        if endpoint == 'time':
            raise RuntimeError('clock unavailable')
        return get(endpoint, params)
    account.venue.get.side_effect = broken
    result = account.step()
    assert result['position'] is None and result['fill_count_total'] == 2
    assert result['fills'][-1]['reason'] == 'protective_stop'


def test_main_profile_only_changes_authorized_budget_allocation():
    for symbol in ('MUUSDT','SNDKUSDT','SKHYNIXUSDT'):
        old = paper.stock_config({'stock_research_profile':'config/stock_main_price_led_short_20261030.json'}, symbol)
        new = paper.stock_config({'stock_research_profile':'config/stock_net100_risk25_20261031.json'}, symbol)
        assert new == {**old, budget.POLICY_KEY: budget.POLICY,
                       'risk_fraction_per_trade': .025, 'portfolio_gross_notional_fraction': 10.}


def test_budget_migration_preserves_inventory_cash_and_consumed_signal(tmp_path,monkeypatch):
    old=paper.stock_config({'stock_research_profile':'config/stock_main_price_led_short_20261030.json'},'SNDKUSDT')
    new={**old,budget.POLICY_KEY:deepcopy(budget.POLICY),
         'risk_fraction_per_trade':.025,'portfolio_gross_notional_fraction':10.}
    state=paper.initial_stock('SNDKUSDT',1000,old)
    state.update(wallet_balance=1008.,equity=1009.,signal_timeframe='1h',
        last_signal_time_ms=3000,pending_signal={'boundary_ms':3000},
        position={'entry':1171.9,'qty':.04,'direction':-1,'stop':1224.81},
        position_history=[{'time_ms':2000,'signed_qty':-.04}])
    path=tmp_path/'sndk/state.json';paper.write_json(path,state)
    monkeypatch.setattr(paper,'stock_config',lambda *_:new)
    monkeypatch.setattr(paper,'now_ms',lambda:5000)
    plan={'accounts':[{'account_id':'sndk','symbol':'SNDKUSDT','state_path':str(path)}],
          'stock_rule_revision_scope':'profit_budget_only'}
    manifest={};paper.synchronize_stock_rules(plan,tmp_path,manifest)
    after=paper.read_json(path)
    assert after['position']==state['position'] and after['wallet_balance']==state['wallet_balance']
    assert after['last_signal_time_ms']==3000 and after['start_ms']==1000
    assert after['pending_signal'] is None and after['signal_active_after_ms']==5000
    assert after['trade_profit_budget_legacy_position_rule']==old
    assert after['trade_profit_budget_active_after_ms']==5000
    paper.synchronize_stock_rules(plan,tmp_path,manifest)
    assert paper.read_json(path)==after


def test_btc_budget_activation_only_adds_metadata_and_never_revives_old_source(tmp_path,monkeypatch):
    from trading_execution import SimulationAccount
    path=tmp_path/'btc/state.json';account=SimulationAccount(path)
    account.reset(1000,now_ms=1000)
    before=account.load();before['btc_pullback_pending']={'status':'expired','expires_at_ms':2000}
    before['last_signal_time_ms']=1500;paper.write_json(path,before)
    plan={'accounts':[{'account_id':'btc','state_path':str(path)}],budget.POLICY_KEY:budget.POLICY}
    monkeypatch.setattr(paper,'now_ms',lambda:5000)
    manifest={};paper.synchronize_btc_profit_budget(plan,tmp_path,manifest)
    after=paper.read_json(path)
    for key,value in before.items():assert after[key]==value
    assert after['trade_profit_budget_active_after_ms']==5000
    assert after[budget.POLICY_KEY]==budget.POLICY
    paper.synchronize_btc_profit_budget(plan,tmp_path,manifest)
    assert paper.read_json(path)==after
