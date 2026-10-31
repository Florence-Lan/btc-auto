"""Pure optional per-trade profit/loss quantity admission; no I/O or mutation.

Prices are existing technical stop/target levels. This module never invents
profit room, widens a stop, places an order, or guarantees future fills/funding.
Entry is already an actual/estimated adverse fill; do not charge its slippage
twice. An actual target fill must be marked target_is_fill=True.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import math

import stock_swing_signals as signals

POLICY_KEY = 'trade_profit_budget_policy'
POLICY = {
    'mode': 'net100_risk25_v1',
    'net_profit_target_usdt': 100.0,
    'maximum_planned_loss_usdt': 25.0,
    'maximum_equity_risk_fraction': 0.025,
    'minimum_net_reward_risk': 4.0,
    'maximum_leverage': 10.0,
    'maximum_gross_notional_fraction': 10.0,
}


def enabled(config):
    return isinstance(config, Mapping) and POLICY_KEY in config


def validate(config):
    if not isinstance(config, Mapping):
        raise ValueError('trade_profit_budget_config_invalid')
    if not enabled(config):
        return None
    policy = config[POLICY_KEY]
    if not isinstance(policy, Mapping) or set(policy) != set(POLICY):
        raise ValueError('trade_profit_budget_policy_invalid')
    for key, expected in POLICY.items():
        value = policy[key]
        if (isinstance(value, bool) or value != expected
                or (isinstance(expected, str) and type(value) is not str)
                or (not isinstance(expected, str) and not isinstance(value, (int, float)))):
            raise ValueError('trade_profit_budget_policy_invalid')
    return deepcopy(POLICY)


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError('trade_profit_budget_invalid_' + name)
    result = Decimal(str(value))
    if (not result.is_finite() or positive and result <= 0
            or nonnegative and result < 0):
        raise ValueError('trade_profit_budget_invalid_' + name)
    return result


def _floor(value, step):
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def _ceil(value, step):
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def _base(config):
    try:
        policy = validate(config)
    except ValueError as exc:
        return {'enabled': enabled(config), 'allowed': False, 'status': 'unavailable',
                'quantity': 0.0, 'admitted_quantity': 0.0, 'reasons': [str(exc)]}
    if policy is None:
        return {'enabled': False, 'allowed': True, 'status': 'disabled',
                'quantity': None, 'reasons': [], 'legacy_behavior_unchanged': True}
    return {'enabled': True, 'allowed': False, 'status': 'blocked',
            'quantity': 0.0, 'admitted_quantity': 0.0, 'reasons': [],
            'policy': policy, 'scope': 'new_trade_only',
            'prices_not_modified': True, 'planned_not_guaranteed': True}


def quantity_plan(config, *, equity, entry, stop, step, direction,
                  technical_target, fee_rate, slippage, target_is_fill=False,
                  entry_fee_per_unit=None, funding_debit_per_unit=0,
                  maximum_qty=None, minimum_qty=0, minimum_notional=0,
                  liquidity_qty=None, depth_qty=None, actual_qty=None,
                  maintenance_fraction=None, minimum_liquidation_buffer=0):
    """Choose the minimum tradable quantity reaching net100, or reject.

    Required quantity is CEIL to step; every risk/collateral/liquidity cap is
    FLOOR to step. Caps never silently shrink a trade below the net-profit goal.
    actual_qty rechecks an observed IOC fill and refuses underfilled net goals.
    A funding credit is included in net profit, but never reduces loss budget.
    """
    result = _base(config)
    if not result['enabled'] or result['status'] == 'unavailable':
        return result
    try:
        with localcontext() as context:
            context.prec = 50
            balance = _number(equity, 'equity', positive=True)
            price = _number(entry, 'entry', positive=True)
            protective = _number(stop, 'stop', positive=True)
            unit = _number(step, 'step', positive=True)
            target = _number(technical_target, 'technical_target', positive=True)
            fee = _number(fee_rate, 'fee_rate', nonnegative=True)
            slip = _number(slippage, 'slippage', nonnegative=True)
            funding = _number(funding_debit_per_unit, 'funding_debit_per_unit')
            paid_entry = price * fee if entry_fee_per_unit is None else _number(
                entry_fee_per_unit, 'entry_fee_per_unit', nonnegative=True)
            if type(direction) is not int or direction not in (-1, 1):
                raise ValueError('trade_profit_budget_invalid_direction')
            if type(target_is_fill) is not bool or fee >= 1 or slip >= 1:
                raise ValueError('trade_profit_budget_invalid_cost_model')
            side = Decimal(direction)
            if side * (price - protective) <= 0:
                raise ValueError('trade_profit_budget_stop_wrong_side')
            leverage = _number(config.get('leverage', POLICY['maximum_leverage']), 'leverage', positive=True)
            gross = _number(config.get('portfolio_gross_notional_fraction', POLICY['maximum_gross_notional_fraction']),
                            'gross_notional_fraction', positive=True)
            if leverage > Decimal(str(POLICY['maximum_leverage'])):
                raise ValueError('trade_profit_budget_leverage_above_policy')
            if gross > Decimal(str(POLICY['maximum_gross_notional_fraction'])):
                raise ValueError('trade_profit_budget_gross_notional_above_policy')
            target_fill = target if target_is_fill else target * (1 - side * slip)
            stop_fill = protective * (1 - side * slip)
            per_profit = side * (target_fill - price) - paid_entry - target_fill * fee - funding
            # Exact same stop-fill/fee model as per_unit_stop_risk, with actual
            # paid entry fee and any known positive funding debit included.
            per_loss = -side * (stop_fill - price) + paid_entry + stop_fill * fee + max(funding, Decimal(0))
            if per_profit <= 0:
                raise ValueError('trade_profit_budget_no_net_target_room')
            if per_loss <= 0:
                raise ValueError('trade_profit_budget_invalid_stop_risk')
            goal = Decimal(str(POLICY['net_profit_target_usdt']))
            budget = min(Decimal(str(POLICY['maximum_planned_loss_usdt'])),
                         balance * Decimal(str(POLICY['maximum_equity_risk_fraction'])))
            min_qty = _number(minimum_qty, 'minimum_qty', nonnegative=True)
            min_notional = _number(minimum_notional, 'minimum_notional', nonnegative=True)
            required = max(_ceil(goal / per_profit, unit), _ceil(min_qty, unit),
                           _ceil(min_notional / price, unit))
            caps = {
                'risk': _floor(budget / per_loss, unit),
                'gross_notional': _floor(balance * gross / price, unit),
                'leverage': _floor(balance * leverage / price, unit),
                'margin_and_entry_fee': _floor(balance / (price / leverage + paid_entry), unit),
            }
            for name, value in (('maximum_qty', maximum_qty), ('liquidity', liquidity_qty), ('depth', depth_qty)):
                if value is not None:
                    caps[name] = _floor(_number(value, name, nonnegative=True), unit)
            quantity = required if actual_qty is None else _number(actual_qty, 'actual_qty', nonnegative=True)
            reasons = []
            if quantity % unit:
                reasons.append('actual_quantity_not_step_aligned')
            for name, cap in caps.items():
                if quantity > cap:
                    reasons.append('required_quantity_exceeds_' + name + '_cap')
            profit, loss = quantity * per_profit, quantity * per_loss
            if quantity < required or profit < goal:
                reasons.append('actual_quantity_below_net_profit_goal')
            if quantity < min_qty or quantity * price < min_notional:
                reasons.append('exchange_minimum_not_met')
            reward_risk = per_profit / per_loss
            if reward_risk < Decimal(str(POLICY['minimum_net_reward_risk'])):
                reasons.append('net_reward_risk_below_minimum')
            if loss > budget:
                reasons.append('planned_loss_exceeds_budget')
            maintenance = config.get('maintenance_margin_fraction_assumption') if maintenance_fraction is None else maintenance_fraction
            liquidation = buffer = None
            if maintenance is not None:
                maintenance = _number(maintenance, 'maintenance_fraction', nonnegative=True)
                required_buffer = max(_number(minimum_liquidation_buffer, 'minimum_liquidation_buffer', nonnegative=True),
                    _number(config.get('minimum_liquidation_distance_buffer', 0), 'configured_liquidation_buffer', nonnegative=True))
                if maintenance >= 1:
                    raise ValueError('trade_profit_budget_invalid_maintenance_fraction')
                remaining_margin = price / leverage - paid_entry - funding
                liquidation = (side * price - remaining_margin) / (side - maintenance)
                buffer = side * (protective - liquidation) / price
                if buffer < required_buffer:
                    reasons.append('liquidation_buffer_below_minimum')
            result.update(
                allowed=not reasons, status='ready' if not reasons else 'blocked',
                quantity=float(quantity), admitted_quantity=float(quantity) if not reasons else 0.0,
                required_qty=float(required), risk_qty=float(caps['risk']),
                quantity_caps={key: float(value) for key, value in caps.items()},
                loss_budget_usdt=float(budget), net_profit_goal_usdt=float(goal),
                expected_net_profit_usdt=float(profit), planned_loss_usdt=float(loss),
                per_unit_net_profit_usdt=float(per_profit), per_unit_stop_risk_usdt=float(per_loss),
                net_reward_risk=float(reward_risk), entry_price=float(price),
                technical_stop_price=float(protective), technical_target_price=float(target),
                target_exit_fill_price=float(target_fill), stop_exit_fill_price=float(stop_fill),
                target_quote_price=float(target_fill / (1 - side * slip)),
                entry_fee_per_unit=float(paid_entry), funding_debit_per_unit=float(funding),
                funding_credit_does_not_reduce_loss_budget=True,
                notional_usdt=float(quantity * price), leverage=float(leverage),
                margin_usdt=float(quantity * price / leverage),
                liquidation_price=float(liquidation) if liquidation is not None else None,
                liquidation_buffer_fraction=float(buffer) if buffer is not None else None,
                actual_fill_checked=actual_qty is not None, reasons=sorted(set(reasons)),
                rounding={'required_quantity': 'CEILING', 'capacity_quantities': 'FLOOR'},
                entry_slippage_already_in_entry=True,
            )
            pending = [result]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
                elif isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('trade_profit_budget_arithmetic_overflow')
    except (ValueError, TypeError, ArithmeticError, OverflowError) as exc:
        result = _base(config)
        result.update(allowed=False, status='unavailable', quantity=0.0,
                      admitted_quantity=0.0, reasons=[str(exc) if isinstance(exc, ValueError)
                                                   else 'trade_profit_budget_invalid_arithmetic'])
    return result


def stock_plan(config, *, equity, entry, stop, step, direction, fee_rate=None,
               slippage=None, entry_fee_per_unit=None, funding_debit_per_unit=0,
               **limits):
    """Keep the original stock net-margin technical target, then size to100.

    target_exit_price returns an ACTUAL fill price. Its corresponding quote
    trigger includes exit slippage; no second target slippage deduction occurs.
    """
    result = _base(config)
    if not result['enabled'] or result['status'] == 'unavailable':
        return result
    try:
        price = _number(entry, 'entry', positive=True)
        protective = _number(stop, 'stop', positive=True)
        fee = config['taker_fee_rate_assumption'] if fee_rate is None else fee_rate
        slip = config['adverse_slippage_fraction_assumption'] if slippage is None else slippage
        target = signals.target_exit_price(float(price), direction, funding_debit_per_unit,
            fee, config['target_margin_return'], config.get('leverage', 10),
            entry_fee_per_unit=entry_fee_per_unit)
        distance = abs(protective / price - 1)
        if 'max_stop_fraction' in config and distance > _number(config['max_stop_fraction'], 'max_stop_fraction', positive=True) + Decimal('1e-12'):
            raise ValueError('trade_profit_budget_technical_stop_exceeds_original_maximum')
        if 'min_stop_fraction' in config and distance + Decimal('1e-12') < _number(config['min_stop_fraction'], 'min_stop_fraction', positive=True):
            raise ValueError('trade_profit_budget_technical_stop_below_original_minimum')
        result = quantity_plan(config, equity=equity, entry=entry, stop=stop, step=step,
            direction=direction, technical_target=target, fee_rate=fee, slippage=slip,
            target_is_fill=True, entry_fee_per_unit=entry_fee_per_unit,
            funding_debit_per_unit=funding_debit_per_unit, **limits)
        result.update(target_basis='original_target_exit_price_net_margin_return',
                      original_target_margin_return=config['target_margin_return'])
    except (ValueError, TypeError, KeyError, ArithmeticError, OverflowError) as exc:
        result.update(allowed=False, status='unavailable', quantity=0.0, admitted_quantity=0.0,
                      reasons=[str(exc) if isinstance(exc, ValueError) else 'trade_profit_budget_invalid_stock_model'])
    return result
