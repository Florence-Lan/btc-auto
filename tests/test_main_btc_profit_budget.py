"""Causal original BTC structures, real quantities and absolute episode profit."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import btc_pullback_entry as pullback
import main_btc_profit_budget as main
import trade_profit_budget as budget
from trading_execution import SimulationAccount

ORIGIN = 1000 * pullback.HOUR
RULES = {'step_size': Decimal('.001'), 'min_qty': Decimal('.001'),
         'max_qty': Decimal('120'), 'min_notional': Decimal('50')}
CONFIG = {budget.POLICY_KEY: deepcopy(budget.POLICY)}


def scenario(side=1, *, near=1100, far=1200):
    bars = [[ORIGIN-(49-i)*pullback.HOUR, 1000, 1010, 990, 1000, 1, 1000,
             ORIGIN-(48-i)*pullback.HOUR-1] for i in range(49)]
    if far is not None:
        bars[10][2] = far
    if near is not None:
        bars[20][2] = near
    target = {'target_leverage': side, 'position_id': 'original-source',
              'origin_signal_time_ms': ORIGIN, 'origin_entry_price': 1040}
    if side == -1:
        bars = [[r[0],2000-r[1],2000-r[3],2000-r[2],2000-r[4],*r[5:]] for r in bars]
        target['origin_entry_price'] = 960
    pending = pullback.build_pending(target, bars, pullback.DEFAULT_POLICY, ORIGIN+1000)
    assert pending['status'] == 'waiting_retrace'
    state = {'mode': 'simulation', 'wallet_balance': 1000., 'position_qty': 0.}
    return state, target, pending, bars


def plan(side=1, **changes):
    state,target,pending,bars = scenario(side)
    params = {'state':state,'target':target,'pending':pending,'hourly_candles':bars,
              'now_ms':ORIGIN+600000,'entry_price':1003.1 if side==1 else 996.9,
              'rules':RULES,'fee_rate':.00045,'slippage_bps':1.,'active_after_ms':ORIGIN-1}
    params.update(changes)
    return main.plan_entry(CONFIG, **params)


@pytest.mark.parametrize('side',[1,-1])
def test_original_confirmed_pivot_room_supports_real_net100_quantity_and_risk(side):
    state,target,pending,bars=scenario(side)
    before=deepcopy((state,target,pending,bars))
    result=plan(side)
    assert result['allowed']
    assert result['technical_target']['price']==(1100 if side==1 else 900)
    assert result['stop']==pending['invalidation_price']
    assert result['quantity']>0 and result['expected_net_profit_usdt']>=100
    assert result['planned_loss_usdt']<=25 and result['net_reward_risk']>=4
    assert result['notional_usdt']<=10000
    assert result['context']['known_by_ms']==ORIGIN-1
    assert all(p['available_at_ms']<ORIGIN for p in result['context']['opposite_pivots'])
    assert before==(state,target,pending,bars)


def test_no_technical_pivot_never_invents_a_target_from_desired_profit():
    state,target,pending,bars=scenario(near=None,far=None)
    result=main.technical_context(target,pending,bars,now_ms=ORIGIN+1000)
    assert not result['allowed']
    assert result['reasons']==['main_btc_budget_no_confirmed_technical_target']


def test_nearer_resistance_is_not_skipped_to_force_four_to_one_reward():
    state,target,pending,bars=scenario(near=1020)
    result=main.plan_entry(CONFIG,state=state,target=target,pending=pending,hourly_candles=bars,
        now_ms=ORIGIN+600000,entry_price=1003.1,rules=RULES,fee_rate=.00045,slippage_bps=1)
    assert result['technical_target']['price']==1020
    assert not result['allowed']
    assert result['net_reward_risk']<4


def test_future_and_unconfirmed_pivot_cannot_change_original_context():
    _,target,pending,bars=scenario()
    original=main.technical_context(target,pending,bars,now_ms=ORIGIN+1000)
    future=[ORIGIN,1000,float('nan'),0,0,0,0,ORIGIN+pullback.HOUR-1]
    extended=main.technical_context(target,pending,bars+[future],now_ms=ORIGIN+1000)
    assert extended==original
    # Last completed hour has no two completed right-wing confirmation bars.
    altered=deepcopy(bars);altered[-2][2]=1020
    context=main.technical_context(target,pending,altered,now_ms=ORIGIN+1000)['context']
    assert all(p['pivot_open_ms']!=altered[-2][0] for p in context['opposite_pivots'])


@pytest.mark.parametrize('failure',['missing_hour','missing_original_close','future_structure','expired','source_changed'])
def test_invalid_or_expired_original_evidence_blocks_new_risk(failure):
    _,target,pending,bars=scenario()
    now=ORIGIN+600000
    if failure=='missing_hour':bars.pop(15)
    elif failure=='missing_original_close':bars.pop()
    elif failure=='future_structure':pending['selected_structure']['available_at_ms']=ORIGIN
    elif failure=='expired':now=pending['expires_at_ms']
    else:target['position_id']='different'
    assert not main.technical_context(target,pending,bars,now_ms=now)['allowed']


def test_frozen_context_preserves_original_levels_and_rejects_other_source():
    state,target,pending,bars=scenario()
    first=plan()
    revised=deepcopy(bars);revised[20][2]=1030
    frozen=main.plan_entry(CONFIG,state=state,target=target,pending=pending,hourly_candles=revised,
        frozen_context=first['context'],now_ms=ORIGIN+700000,entry_price=1003.1,
        rules=RULES,fee_rate=.00045,slippage_bps=1)
    assert frozen['context']==first['context'] and frozen['technical_target']==first['technical_target']
    wrong={**target,'position_id':'other'}
    result=main.plan_entry(CONFIG,state=state,target=wrong,pending=pending,hourly_candles=bars,
        frozen_context=first['context'],now_ms=ORIGIN+700000,entry_price=1003.1,
        rules=RULES,fee_rate=.00045,slippage_bps=1)
    assert not result['allowed']


def test_final_parent_cap_and_observed_underfill_are_not_silently_upsized():
    first=plan()
    assert first['allowed']
    assert not plan(maximum_qty=first['quantity']-.001)['allowed']
    state,_,_,_=scenario()
    result=main.check_fill(CONFIG,first,state=state,entry_price=first['entry_price'],
        quantity=first['quantity']-.001,now_ms=ORIGIN+600000,rules=RULES,
        fee_rate=.00045,slippage_bps=1)
    assert not result['allowed'] and result['expected_net_profit_usdt']<100


def test_activation_old_inventory_and_optout_are_separate():
    assert not plan(active_after_ms=ORIGIN)['allowed']
    state,_,_,_=scenario();state['position_qty']=.1
    result=plan(state=state)
    assert result['status']=='existing_inventory' and not result['new_risk_allowed']
    result=main.plan_entry({},state=None,target=None,pending=None,hourly_candles=None,
        now_ms=None,entry_price=None,rules=None,fee_rate=None,slippage_bps=None)
    assert result['status']=='disabled' and result['quantity'] is None


def episode(side=1):
    account=SimulationAccount(persist=False,allow_llm=False)
    account.reset(1000,now_ms=ORIGIN-1000)
    state=account.load();mark=1003 if side==1 else 997
    actual=mark*(1+side*.0001)
    result=plan(side,state=state,entry_price=actual)
    assert result['allowed']
    at=ORIGIN+600000
    fill=account._fill_to(state,side*result['quantity'],mark,at,ORIGIN,RULES)
    state[main.POSITION_KEY]=main.position_tag(state,result,entry_sequence=fill['sequence'],
        entry_time_ms=at,entry_price=fill['price'],quantity=fill['quantity'])
    return account,state,result,fill


@pytest.mark.parametrize('side',[1,-1])
def test_only_tagged_complete_episode_can_take_net100_after_costs(side):
    _,state,result,fill=episode(side)
    before=deepcopy(state)
    mark=result['technical_target']['price']
    outcome=main.position_exit(state,mark,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=True)
    assert outcome['should_exit'] and outcome['status']=='target_reached'
    assert outcome['projected_net_pnl_usdt']>=100
    assert state==before
    unavailable=main.position_exit(state,mark,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=False)
    assert not unavailable['should_exit'] and unavailable['status']=='funding_unavailable'
    state.pop(main.POSITION_KEY)
    assert not main.position_exit(state,mark,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=True)['enabled']


def test_planned_stop_survives_missing_funding_and_incomplete_profit_ledger():
    _,state,result,_=episode()
    state['fills']=[]
    outcome=main.position_exit(state,result['stop']-.01,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=False)
    assert outcome['should_exit'] and outcome['reason']=='main_btc_budget_protective_stop'
    profit=main.position_exit(state,1200,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=True)
    assert not profit['should_exit'] and profit['status']=='ledger_unavailable'


def test_partial_exit_and_actual_funding_are_included_in_absolute_profit():
    account,state,result,_=episode()
    account._fill_to(state,state['position_qty']/2,1050,ORIGIN+650000,ORIGIN,RULES)
    account.settle_funding(state,[{'fundingTime':ORIGIN+660000,'fundingRate':'.001','markPrice':'1050'}],ORIGIN+670000)
    decision=main.position_exit(state,1250,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=True)
    qty=state['position_qty'];exit_price=1250*.9999
    expected=state['realized_pnl']-state['fees_paid']+state['funding_pnl']+qty*(exit_price-state['entry_price'])-qty*exit_price*.00045
    assert decision['projected_net_pnl_usdt']==pytest.approx(expected)
    assert decision['should_exit'] is (expected>=100)


def test_tag_rejects_unchecked_quantity_and_wrong_episode_identity():
    _,state,result,fill=episode()
    with pytest.raises(ValueError):
        main.position_tag(state,result,entry_sequence=fill['sequence'],entry_time_ms=ORIGIN+600000,
            entry_price=fill['price'],quantity=fill['quantity']+.001)
    state[main.POSITION_KEY]['account_epoch']='wrong'
    outcome=main.position_exit(state,result['stop']-1,now_ms=ORIGIN+700000,
        fee_rate=.00045,slippage_bps=1,funding_covered=False)
    assert not outcome['should_exit'] and outcome['status']=='tag_identity_invalid'
