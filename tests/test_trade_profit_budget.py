"""Fixed net100/loss25 budget math; no live data, I/O, network or optimization."""
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import trade_profit_budget as budget
import stock_swing_signals as signals


def config(**changes):
    result = {budget.POLICY_KEY: deepcopy(budget.POLICY), 'leverage': 10,
        'portfolio_gross_notional_fraction': 10, 'target_margin_return': 1.2,
        'taker_fee_rate_assumption': .000125, 'adverse_slippage_fraction_assumption': .0002,
        'min_stop_fraction': .02, 'max_stop_fraction': .05}
    result.update(changes)
    return result


def generic(cfg=None, **changes):
    values = dict(equity=1000, entry=100, stop=98, step=1, direction=1,
        technical_target=110, fee_rate=0, slippage=0)
    values.update(changes)
    return budget.quantity_plan(config() if cfg is None else cfg, **values)


def test_fixed_policy_and_absence_are_explicit_and_do_not_mutate_config():
    cfg = config();before=deepcopy(cfg)
    assert budget.validate(cfg)==budget.POLICY
    assert budget.enabled(cfg)
    generic(cfg)
    assert cfg==before
    disabled=budget.quantity_plan({},equity=None,entry=None,stop=None,step=None,
        direction=None,technical_target=None,fee_rate=None,slippage=None)
    assert disabled=={'enabled':False,'allowed':True,'status':'disabled','quantity':None,
        'reasons':[],'legacy_behavior_unchanged':True}
    assert budget.stock_plan({},equity=None,entry=None,stop=None,step=None,direction=None)==disabled


@pytest.mark.parametrize('field',list(budget.POLICY))
def test_no_policy_parameter_optimization_or_extra_keys(field):
    cfg=config();cfg[budget.POLICY_KEY][field]='other' if field=='mode' else .123
    with pytest.raises(ValueError):budget.validate(cfg)
    result=generic(cfg)
    assert not result['allowed'] and result['status']=='unavailable'


def test_boolean_policy_numbers_and_unknown_fields_rejected():
    cfg=config();cfg[budget.POLICY_KEY]['maximum_leverage']=True
    with pytest.raises(ValueError):budget.validate(cfg)
    cfg=config();cfg[budget.POLICY_KEY]['optimizer']=True
    with pytest.raises(ValueError):budget.validate(cfg)


def test_minimum_quantity_instead_of_maximum_risk_and_decimal_rounding():
    result=generic()
    assert result['allowed']
    assert result['quantity']==result['required_qty']==10
    assert result['risk_qty']==12  # Budget could hold more but do not use it.
    assert result['expected_net_profit_usdt']==100
    assert result['planned_loss_usdt']==20
    assert result['net_reward_risk']==5


def test_exact_rr4_boundary_and_discrete_quantity_can_make_trade_impossible():
    allowed=generic(technical_target=108,step=.5)
    assert allowed['allowed'] and allowed['quantity']==12.5
    assert allowed['expected_net_profit_usdt']==100 and allowed['planned_loss_usdt']==25
    rejected=generic(technical_target=108,step=1)
    assert not rejected['allowed'] and rejected['required_qty']==13 and rejected['risk_qty']==12
    assert 'required_quantity_exceeds_risk_cap' in rejected['reasons']


@pytest.mark.parametrize('equity,limit',[(900,22.5),(1000,25),(1200,25)])
def test_loss_limit_is_lower_of25_and_2point5pct_current_equity(equity,limit):
    assert generic(equity=equity)['loss_budget_usdt']==limit


def test_below_reference_balance_can_reduce_capacity_without_oversizing():
    result=generic(equity=500)
    assert not result['allowed'] and result['quantity']==10 and result['admitted_quantity']==0
    assert result['risk_qty']==6 and result['loss_budget_usdt']==12.5


@pytest.mark.parametrize('direction,stop,target',[(1,98,112),(-1,102,88)])
def test_costs_target_slippage_and_stop_slippage_are_included_once(direction,stop,target):
    fee=.001;slip=.0002
    result=generic(direction=direction,stop=stop,technical_target=target,step=.001,fee_rate=fee,slippage=slip)
    fill=target*(1-direction*slip)
    expected=direction*(fill-100)-100*fee-fill*fee
    assert result['per_unit_net_profit_usdt']==pytest.approx(expected)
    assert result['per_unit_stop_risk_usdt']==pytest.approx(signals.per_unit_stop_risk(100,stop,fee,slip))
    assert result['expected_net_profit_usdt']>=100


def test_actual_target_fill_is_not_slipped_twice():
    result=generic(technical_target=112,target_is_fill=True,fee_rate=.001,slippage=.01,step=.001)
    assert result['target_exit_fill_price']==112
    assert result['per_unit_net_profit_usdt']==pytest.approx(12-.1-.112)
    assert result['target_quote_price']==pytest.approx(112/.99)


def test_actual_entry_fee_and_known_funding_are_reflected_and_credit_not_used_to_lower_risk():
    base=generic(fee_rate=.001,step=.001)
    debit=generic(fee_rate=.001,entry_fee_per_unit=.5,funding_debit_per_unit=.3,step=.001)
    assert debit['per_unit_net_profit_usdt']==pytest.approx(base['per_unit_net_profit_usdt']-.4-.3)
    assert debit['per_unit_stop_risk_usdt']==pytest.approx(base['per_unit_stop_risk_usdt']+.4+.3)
    credit=generic(fee_rate=.001,funding_debit_per_unit=-.3,step=.001)
    assert credit['per_unit_stop_risk_usdt']==base['per_unit_stop_risk_usdt']


@pytest.mark.parametrize('limit', ['maximum_qty','liquidity_qty','depth_qty'])
def test_exchange_liquidity_and_depth_limits_reject_instead_of_undersizing(limit):
    result=generic(**{limit:9})
    assert not result['allowed'] and result['quantity']==10
    assert result['admitted_quantity']==0


def test_gross_fraction_is_actual_config_cap_below_fixed10_ceiling():
    cfg=config(portfolio_gross_notional_fraction=.5)
    result=generic(cfg)
    assert not result['allowed'] and result['quantity_caps']['gross_notional']==5
    assert 'required_quantity_exceeds_gross_notional_cap' in result['reasons']
    assert generic(config(portfolio_gross_notional_fraction=10))['allowed']
    assert generic(config(portfolio_gross_notional_fraction=11))['status']=='unavailable'
    assert generic(config(leverage=11))['status']=='unavailable'


def test_minimum_notional_or_quantity_rounds_up_without_relaxing_caps():
    result=generic(technical_target=1100,step=.1,minimum_notional=50)
    assert result['allowed'] and result['quantity']==.5
    result=generic(technical_target=1100,step=.1,minimum_qty=.25)
    assert result['allowed'] and result['quantity']==.3


@pytest.mark.parametrize('actual', [0,9,9.5])
def test_actual_partial_fill_cannot_be_marked_as_meeting_net100(actual):
    result=generic(actual_qty=actual)
    assert not result['allowed'] and result['quantity']==actual and result['admitted_quantity']==0
    assert result['actual_fill_checked']
    assert 'actual_quantity_below_net_profit_goal' in result['reasons']


def test_actual_fill_returns_observed_quantity_and_rechecks_changed_price_risk():
    result=generic(actual_qty=10)
    assert result['allowed'] and result['quantity']==10
    cfg=config(taker_fee_rate_assumption=0,adverse_slippage_fraction_assumption=0)
    result=budget.stock_plan(cfg,equity=1000,entry=101,stop=98,step=.01,direction=1,actual_qty=8.34)
    assert not result['allowed'] and result['planned_loss_usdt']==pytest.approx(25.02)


@pytest.mark.parametrize('kwargs',[
 {'equity':True},{'equity':0},{'entry':float('nan')},{'stop':float('inf')},
 {'step':0},{'direction':True},{'stop':101},{'technical_target':99},
 {'fee_rate':1},{'slippage':-1},{'funding_debit_per_unit':float('nan')},
 {'actual_qty':float('inf')},{'liquidity_qty':-1},
])
def test_bad_inputs_are_fail_closed_without_exceptions_or_nonfinite_outputs(kwargs):
    result=generic(**kwargs)
    assert not result['allowed']
    json.dumps(result,allow_nan=False)


def test_zero_observed_liquidity_is_a_valid_blocked_capacity_not_unlimited():
    result=generic(liquidity_qty=0)
    assert not result['allowed'] and result['quantity_caps']['liquidity']==0


def test_extreme_finite_decimal_arithmetic_never_leaks_infinite_json():
    result=generic(technical_target=Decimal('100.00000000000000000000000000000000000000000000000001'),step=Decimal('1e-300'))
    assert not result['allowed']
    json.dumps(result,allow_nan=False)


def test_liquidation_buffer_retains_original_assumption():
    cfg=config(maintenance_margin_fraction_assumption=.015,minimum_liquidation_distance_buffer=.01)
    assert generic(cfg)['allowed']
    result=generic(cfg,stop=91,technical_target=150)
    assert not result['allowed']
    assert 'liquidation_buffer_below_minimum' in result['reasons']


def test_actual_historical_mu_technical_levels_need_about_point96_and_fit25():
    cfg=config();before=deepcopy(cfg)
    result=budget.stock_plan(cfg,equity=999.9760654337292,entry=871.07604994,stop=894.33,step=.01,direction=-1)
    assert result['allowed'] and result['quantity']==.96
    assert result['expected_net_profit_usdt']>=100
    assert 22<result['planned_loss_usdt']<23
    assert cfg==before
    original=signals.target_exit_price(871.07604994,-1,0,cfg['taker_fee_rate_assumption'],1.2,10)
    assert result['technical_target_price']==original
    assert result['technical_stop_price']==894.33


def test_wide_original_sndk_stop_blocks_goal_without_widening_target_or_stop():
    cfg=config(target_margin_return=.6,min_stop_fraction=.005)
    result=budget.stock_plan(cfg,equity=1008.6445853208511,entry=1171.995554,stop=1224.81,step=.01,direction=-1)
    assert not result['allowed'] and result['required_qty']==1.43
    assert result['planned_loss_usdt']>75
    assert result['technical_stop_price']==1224.81 and result['original_target_margin_return']==.6
    assert result['net_reward_risk']<4


def test_low_volatility_sndk_can_fit_with_less_than10x_gross():
    cfg=config(target_margin_return=.6,min_stop_fraction=.005)
    result=budget.stock_plan(cfg,equity=1000,entry=1171.995554,stop=1177.86,step=.01,direction=-1)
    assert result['allowed'] and result['quantity']==1.43
    assert 1000<result['notional_usdt']<2000 and result['planned_loss_usdt']<25


def test_skh_fee_model_can_fit_without_changing_original_target_distance():
    cfg=config(taker_fee_rate_assumption=.001)
    result=budget.stock_plan(cfg,equity=1000,entry=1104,stop=1133.5,step=.001,direction=-1)
    assert result['allowed'] and result['quantity']==.755
    assert result['planned_loss_usdt']<25 and result['expected_net_profit_usdt']>=100


@pytest.mark.parametrize('direction,stop',[(1,98),(-1,102)])
def test_stock_target_net_margin_identity_with_actual_cost_and_funding(direction,stop):
    cfg=config(taker_fee_rate_assumption=.001,adverse_slippage_fraction_assumption=.002)
    result=budget.stock_plan(cfg,equity=1000,entry=100,stop=stop,step=.001,direction=direction,
        entry_fee_per_unit=.2,funding_debit_per_unit=.1)
    assert result['per_unit_net_profit_usdt']==pytest.approx(12)
    assert result['target_quote_price']==pytest.approx(result['target_exit_fill_price']/(1-direction*.002))


@pytest.mark.parametrize('stop',[99,94])
def test_stock_original_stop_bounds_cannot_be_relaxed_to_make_budget_fit(stop):
    result=budget.stock_plan(config(),equity=1000,entry=100,stop=stop,step=.01,direction=1)
    assert not result['allowed'] and result['status']=='unavailable'


def test_btc_style_structural_room_uses_real_target_and_preserves_losses():
    result=generic(entry=100000,stop=100500,direction=-1,technical_target=97000,
        step=.001,fee_rate=.00045,slippage=.0002)
    assert result['allowed'] and result['expected_net_profit_usdt']>=100
    assert result['planned_loss_usdt']<=25 and result['net_reward_risk']>=4
