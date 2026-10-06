"""Forward-only 1R scale-out and cost-aware ATR protection for paper stocks."""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_DOWN

import backtest_stock_swing_120 as arithmetic
import stock_swing_signals as signals

MODE = 'one_r_half_atr'


def validate(config):
    policy = config.get('profit_exit_policy')
    if policy is None:
        return
    if policy.get('mode') != MODE:
        raise ValueError('Unknown stock profit exit policy')
    for key, expected in (('first_target_r', 1.0), ('first_close_fraction', .5), ('trail_atr', 1.5)):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
            raise ValueError('This frozen profit experiment requires ' + key + '=' + str(expected))


def enabled(config):
    return config.get('profit_exit_policy', {}).get('mode') == MODE


def rounded_qty(qty, step):
    unit = Decimal(str(step))
    return float((Decimal(str(qty)) / unit).to_integral_value(rounding=ROUND_DOWN) * unit)


def observe(position, executable, config, timestamp, step, min_qty, min_notional):
    """Observe current executable quotes only; never arm using pre-activation extrema."""
    side, entry = position['direction'], position['entry']
    progress = position.get('profit_exit')
    if progress is None:
        initial_stop = position.get('initial_stop', position['stop'])
        risk = side * (entry - initial_stop)
        if not math.isfinite(risk) or risk <= 0:
            raise ValueError('Initial stop must define positive 1R risk')
        progress = position['profit_exit'] = {
            'activated_at_ms': timestamp, 'initial_stop': initial_stop,
            'initial_qty': position['qty'], 'risk_distance': risk,
            'best_executable': executable, 'target_qty': 0.0, 'filled_qty': 0.0,
            'trigger_observed_at_ms': None, 'stage_done': False,
            'armed': False, 'last_atr_bar_ms': None,
        }
    progress['best_executable'] = (max if side == 1 else min)(progress['best_executable'], executable)
    if progress['trigger_observed_at_ms'] is None:
        estimated_fill = executable * (1 - side * config['adverse_slippage_fraction_assumption'])
        if side * (estimated_fill - entry) >= progress['risk_distance']:
            progress['trigger_observed_at_ms'] = timestamp
            target = rounded_qty(progress['initial_qty'] * .5, step)
            remainder = rounded_qty(Decimal(str(position['qty'])) - Decimal(str(target)), step)
            if (target >= min_qty and remainder >= min_qty
                    and target * estimated_fill >= min_notional
                    and remainder * estimated_fill >= min_notional):
                progress['target_qty'] = target
            else:
                # A small position cannot be divided into two valid executable lots.
                progress.update(stage_done=True, armed=True, split_skipped='minimum_quantity_or_notional')
    return progress


def requested_qty(position, step):
    progress = position['profit_exit']
    if progress['trigger_observed_at_ms'] is None or progress['stage_done']:
        return 0.0
    remaining = Decimal(str(progress['target_qty'])) - Decimal(str(progress['filled_qty']))
    return min(position['qty'], rounded_qty(remaining, step))


def record_partial(position, qty, step):
    progress = position['profit_exit']
    progress['filled_qty'] = float(Decimal(str(progress['filled_qty'])) + Decimal(str(qty)))
    progress['armed'] = True
    progress['stage_done'] = progress['target_qty'] - progress['filled_qty'] < step / 2


def protect(position, config, tick, atr=None, atr_bar_ms=None):
    """Tighten cost-aware break-even on each check; ATR uses the own closed signal bar."""
    progress = position['profit_exit']
    if not progress['armed']:
        return None
    side = position['direction']
    net_zero_fill = signals.target_exit_price(
        position['entry'], side, position['funding'] / position['qty'],
        config['taker_fee_rate_assumption'], 0, config['leverage'],
        entry_fee_per_unit=position['entry_fee'] / position['qty'])
    break_even = net_zero_fill / (1 - side * config['adverse_slippage_fraction_assumption'])
    candidate = break_even
    atr_valid = (atr is not None and math.isfinite(atr) and atr > 0 and atr_bar_ms is not None)
    if atr_valid:
        trail = progress['best_executable'] - side * config['profit_exit_policy']['trail_atr'] * atr
        candidate = (max if side == 1 else min)(candidate, trail)
        progress['last_atr_bar_ms'] = atr_bar_ms
        progress['last_atr'] = atr
    candidate = arithmetic.round_tick(candidate, tick, side == 1)
    old = position['stop']
    if side * (candidate - old) > 0:
        position['stop'] = candidate
    progress['break_even_quote'] = break_even
    progress['protected_stop'] = position['stop']
    return {'old_stop': old, 'stop': position['stop'], 'break_even_quote': break_even,
            'atr_bar_ms': atr_bar_ms if atr_valid else None}
