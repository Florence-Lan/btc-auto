"""Fixed main-paper execution protection, independent of signal generation.

Reserve loss headroom without shrinking a net100 order below its target. Depth
and mark/book stress are diagnostic bounds, never guaranteed future liquidity.
"""
from collections.abc import Mapping
import math

POLICY_KEY = 'main_stock_execution_guard_policy'
POLICY = {'mode': 'main_execution_reserve_v1', 'planned_budget_fraction': 0.8,
          'require_full_reverse_depth': True, 'stress_adverse_mark_book_basis': True,
          'profit_lock_covers_paid_costs': True}


def enabled(config):
    return isinstance(config, Mapping) and POLICY_KEY in config


def validate(config):
    if not enabled(config):
        return
    p = config[POLICY_KEY]
    if (not isinstance(p, Mapping) or set(p) != set(POLICY)
            or any(type(p[k]) is not type(v) or p[k] != v for k, v in POLICY.items())):
        raise ValueError('main_execution_guard_policy_invalid')
    import trade_profit_budget as budget
    if not budget.enabled(config):
        raise ValueError('main_execution_guard_requires_profit_budget')
    budget.validate(config)


def number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('main_execution_guard_invalid_number')
    return float(value)


def entry_decision(config, plan, *, direction, mark_price, exit_fill):
    validate(config)
    if not enabled(config):
        return {'enabled': False, 'allowed': True, 'reasons': []}
    result = {'enabled': True, 'allowed': False, 'reasons': [],
              'future_fill_guaranteed': False, 'scope': 'new_main_stock_only'}
    if not plan.get('allowed'):
        return {**result, 'status': 'base_budget_blocked', 'reasons': list(plan.get('reasons', []))}
    try:
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError('main_execution_guard_invalid_direction')
        quantity = number(plan['quantity'], positive=True)
        cap = number(plan['loss_budget_usdt'], positive=True)
        planned = number(plan['planned_loss_usdt'], positive=True)
        fee = number(config['taker_fee_rate_assumption'])
        slip = number(config['adverse_slippage_fraction_assumption'])
        if not 0 <= fee < 1 or not 0 <= slip < 1:
            raise ValueError('main_execution_guard_invalid_costs')
        mark = number(mark_price, positive=True)
        filled = number(exit_fill['qty'])
        reserve_cap = cap * POLICY['planned_budget_fraction']
        reasons = []
        if filled + max(1e-12, quantity * 1e-10) < quantity:
            reasons.append('main_execution_reverse_depth_insufficient')
            if filled <= 0:
                return {**result, 'status': 'blocked', 'reasons': reasons,
                        'requested_quantity': quantity, 'full_exit_quantity_observed': filled}
        close_price = number(exit_fill['price'], positive=True)
        # book_fill already includes adverse slippage. Remove that one modeled
        # factor before measuring raw depth/basis stress; do not charge twice.
        raw_close = close_price / (1 - direction * slip)
        adverse_basis = max(0., -direction * (raw_close - mark))
        stressed_loss = planned + quantity * adverse_basis * (1 + fee)
        number(stressed_loss, positive=True)
        if planned > reserve_cap + 1e-9:
            reasons.append('main_execution_planned_reserve_exceeded')
        if stressed_loss > cap + 1e-9:
            reasons.append('main_execution_stressed_loss_exceeds_budget')
        result.update(allowed=not reasons, status='ready' if not reasons else 'blocked', reasons=reasons,
            nominal_loss_budget_usdt=cap, reserved_loss_budget_usdt=reserve_cap,
            planned_loss_usdt=planned, stress_loss_usdt=stressed_loss,
            adverse_depth_basis_per_unit=adverse_basis, mark_price=mark,
            full_exit_quantity_observed=filled, requested_quantity=quantity,
            executable_exit_price=close_price, raw_exit_depth_price=raw_close,
            quantity_unchanged=True, net_profit_goal_usdt=plan['net_profit_goal_usdt'])
    except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        result.update(status='unavailable', reasons=[str(exc) if isinstance(exc, ValueError)
                                                   else 'main_execution_inputs_unavailable'])
    return result


def net_close(config, position, exit_fill):
    side, qty = position['direction'], number(position['qty'], positive=True)
    if type(side) is not int or side not in (-1, 1):
        raise ValueError('main_execution_guard_invalid_direction')
    if number(exit_fill['qty']) + max(1e-12, qty * 1e-10) < qty:
        raise ValueError('main_execution_full_close_unavailable')
    price = number(exit_fill['price'], positive=True)
    entry = number(position['entry'], positive=True)
    fee = number(config['taker_fee_rate_assumption'])
    if not 0 <= fee < 1:
        raise ValueError('main_execution_guard_invalid_costs')
    return side * qty * (price-entry) - number(position['entry_fee']) - number(position['funding']) - qty*price*fee


def lock_quote(config, position):
    """Quote trigger for the existing price-profit floor NET of known costs."""
    qty = number(position['qty'], positive=True)
    side = position['direction']
    if type(side) is not int or side not in (-1, 1):
        raise ValueError('main_execution_guard_invalid_direction')
    entry = number(position['entry'], positive=True)
    locked = number(config['trail_locked_underlying_return'], positive=True)
    fee = number(config['taker_fee_rate_assumption'])
    slip = number(config['adverse_slippage_fraction_assumption'])
    if not 0 <= fee < 1 or not 0 <= slip < 1:
        raise ValueError('main_execution_guard_invalid_costs')
    floor_per_unit = entry * locked
    fill_price = (side*entry + number(position['entry_fee'])/qty
                  + number(position['funding'])/qty + floor_per_unit) / (side-fee)
    quote = fill_price / (1-side*slip)
    return {'quote': number(quote, positive=True), 'net_floor_usdt': floor_per_unit*qty,
            'target_fill_price': number(fill_price, positive=True),
            'known_funding_included': True, 'future_exit_guaranteed': False}
