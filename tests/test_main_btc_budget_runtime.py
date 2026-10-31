"""Real main BTC entry tags, immutable old inventory and independent exits."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from main_btc_budget_runtime import BudgetSimulationAccount
import main_btc_profit_budget as planning
import trade_profit_budget as budget
from test_main_btc_profit_budget import scenario, RULES, ORIGIN


def account_and_entry():
    account = BudgetSimulationAccount(persist=False, allow_llm=False,
        policy=budget.POLICY, active_after_ms=ORIGIN-1)
    account.reset(1000, now_ms=ORIGIN-1000)
    state = account.load()
    _, target, pending, bars = scenario()
    pending['status'] = 'ready'
    state['btc_pullback_pending'] = pending
    state['btc_pullback_entry'] = {'allowed':True}
    at = ORIGIN+600000
    five = [[ORIGIN+300000,1000,1004,999,1003,1000,0,at-1]]
    account._budget_target = target
    account._budget_report = {'btc_pullback_market_data': {'hourly':bars,'five_minute':five}}
    fill = account._fill_to(state, 5., 1003., at, ORIGIN, RULES)
    return account, state, fill


def test_real_new_btc_fill_has_net100_plan_and_no_funding_mutation():
    account, state, fill = account_and_entry()
    assert fill and state[planning.POSITION_KEY]['expected_net_profit_usdt'] >= 100
    assert state[planning.POSITION_KEY]['planned_loss_usdt'] <= 25
    assert state['position_qty'] < 5.
    assert state['fill_count_total'] == 1
    assert account._fill_to(state, state['position_qty']+1,1003,ORIGIN+600001,ORIGIN,RULES) is None
    assert state['fill_count_total'] == 1


def test_original_position_without_tag_is_not_adopted_or_repriced():
    account = BudgetSimulationAccount(persist=False, allow_llm=False,
        policy=budget.POLICY, active_after_ms=ORIGIN-1)
    account.reset(1000, now_ms=ORIGIN-1000)
    state = account.load()
    from trading_execution import SimulationAccount
    SimulationAccount._fill_to(account,state,.1,1000,ORIGIN,ORIGIN,RULES)
    account._state=state
    result=account.observe(1000,ORIGIN+1000,funding_available=False)
    assert result['fill'] is None and planning.POSITION_KEY not in account.load()


def test_btc_budget_technical_stop_exits_without_funding_or_decision_clock():
    account,state,fill=account_and_entry()
    account._state=state
    stop=state[planning.POSITION_KEY]['stop']
    result=account.observe(stop-1,ORIGIN+700000,funding_available=False)
    assert result['fill']['reason']=='main_btc_budget_protective_stop'
    assert account.load()['position_qty']==0
    assert account.load()['fill_count_total']==2


def test_btc_net100_requires_complete_funding_and_then_books_once():
    account,state,fill=account_and_entry()
    account._state=state
    target=state[planning.POSITION_KEY]['technical_target']['price']
    blocked=account.observe(target,ORIGIN+700000,funding_available=False)
    assert blocked['fill'] is None
    result=account.observe(target,ORIGIN+700001,funding_available=True)
    assert result['fill']['reason']=='main_btc_net100_target'
    assert account.load()['position_qty']==0
    assert account.load()['wallet_balance']>=1100
    again=account.observe(target,ORIGIN+700002,funding_available=True)
    assert again['fill'] is None and account.load()['fill_count_total']==2


def test_btc_parent_quantity_cap_cannot_be_upsized_after_approval():
    account,state,fill=account_and_entry()
    fresh=BudgetSimulationAccount(persist=False,allow_llm=False,policy=budget.POLICY,active_after_ms=ORIGIN-1)
    fresh.reset(1000,now_ms=ORIGIN-1000)
    empty=fresh.load();empty['btc_pullback_pending']=deepcopy(state['btc_pullback_pending'])
    empty['btc_pullback_entry']={'allowed':True}
    fresh._budget_report=deepcopy(account._budget_report);fresh._budget_target=deepcopy(account._budget_target)
    result=fresh._fill_to(empty,.001,1003,ORIGIN+600000,ORIGIN,RULES)
    assert result is None and empty['position_qty']==0 and empty['fill_count_total']==0


def test_complete_reconcile_preserves_original_pullback_before_new_budget_fill(tmp_path,monkeypatch):
    from test_btc_pullback_execution import report, desired
    from test_btc_pullback_entry import five
    monkeypatch.setenv('SIM_MAX_LEVERAGE','10')
    monkeypatch.setenv('SIM_TAKER_FEE','.00045')
    monkeypatch.setenv('SIM_SLIPPAGE_BPS','1')
    bot=BudgetSimulationAccount(tmp_path/'state.json',allow_llm=False,
        policy=budget.POLICY,active_after_ms=ORIGIN-1)
    state=bot.reset(1000,now_ms=ORIGIN-1000)
    state['observed_flat_target']=True;bot._save(state)
    _,_,_,bars=scenario()
    first_report=report(now=ORIGIN+1000)
    first_report['btc_pullback_market_data']['hourly']=bars
    waiting=bot.reconcile(desired(),1000,RULES,first_report,now_ms=ORIGIN+1000)
    assert waiting['fill'] is None
    rows=[five(kind='touch'),five(1)]
    for row in rows:row[5]=1000
    now=ORIGIN+600000
    ready_report=report(rows,now)
    ready_report['btc_pullback_market_data']['hourly']=bars
    entered=bot.reconcile(desired(),1003,RULES,ready_report,now_ms=now)
    assert entered['fill'] is not None
    tag=bot.load()[planning.POSITION_KEY]
    assert tag['expected_net_profit_usdt']>=100 and tag['planned_loss_usdt']<=25
    assert entered['btc_pullback_entry']['status']=='filled'
