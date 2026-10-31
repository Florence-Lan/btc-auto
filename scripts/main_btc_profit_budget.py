"""Main-only prospective BTC absolute-profit budget and position decisions.

No network, writes, orders, producer changes or account mutation. The target is
the NEAREST confirmed opposite hourly pivot known at the original signal, not
an invented price chosen to make100 USDT possible. Existing positions without a
new-entry tag keep their original management.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math

import btc_profit_target as ledger
import btc_pullback_entry as pullback
import trade_profit_budget as budget

POLICY_KEY = budget.POLICY_KEY
POSITION_KEY = 'main_btc_profit_budget_position'
CONTEXT_MODE = 'main_btc_original_hourly_opposite_pivot_v1'


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('main_btc_budget_number_invalid')
    return float(value)


def _integer(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError('main_btc_budget_timestamp_invalid')
    return value


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _blocked(reason, **details):
    return {'enabled': True, 'allowed': False, 'status': 'blocked', 'reasons': [reason], **details}


def technical_context(target, pending, hourly_candles, *, now_ms):
    """Freeze original structural invalidation and all original opposite pivots.

    Stop comes from the already frozen pullback pending. No future hourly OHLC
    is parsed and the two right-hand confirmation candles must close before the
    original hourly signal. ``plan_entry`` picks the nearest profitable pivot
    for the actual entry; it cannot skip a nearer barrier to improve the ratio.
    """
    try:
        now = _integer(now_ms, positive=True)
        source = pullback._source(target)
        if source is None or not isinstance(pending, Mapping):
            return _blocked('main_btc_budget_original_structure_missing')
        origin = source['origin_signal_time_ms']
        if origin > now or origin % pullback.HOUR:
            return _blocked('main_btc_budget_original_structure_time_invalid')
        policy = pullback.validate_policy(pending['policy'])
        if (pending.get('source_key') != source['source_key']
                or pending.get('signal_origin') != origin
                or pending.get('side') != source['side']):
            return _blocked('main_btc_budget_source_mismatch')
        expiry = _integer(pending.get('expires_at_ms'), positive=True)
        if expiry != origin + round(policy['max_initial_entry_age_seconds'] * 1000) or now >= expiry:
            return _blocked('main_btc_budget_origin_expired')
        if pending.get('status') in pullback.TERMINAL_STATUSES:
            return _blocked('main_btc_budget_original_structure_terminal')
        stop = _number(pending.get('invalidation_price'), positive=True)
        atr = _number(pending.get('atr'), positive=True)
        structure = pending['selected_structure']
        if (not isinstance(structure, Mapping)
                or _integer(structure['available_at_ms']) >= origin):
            return _blocked('main_btc_budget_structure_from_future')
        bars = pullback._closed_bars(hourly_candles, pullback.HOUR, min(origin - 1, now))
        if len(bars) < policy['lookback_hours'] or bars[-1]['close_ms'] != origin - 1:
            return _blocked('main_btc_budget_original_prefix_missing')
        window = bars[-policy['lookback_hours']:]
        wings = policy['pivot_wings']
        side = 1 if source['side'] == 'long' else -1
        key = 'high' if side == 1 else 'low'
        pivots = []
        for index in range(wings, len(window) - wings):
            bar = window[index]
            others = window[index-wings:index] + window[index+1:index+wings+1]
            confirmed = (all(bar[key] > other[key] for other in others) if side == 1
                         else all(bar[key] < other[key] for other in others))
            if confirmed:
                pivots.append({'kind': 'pivot_high' if side == 1 else 'pivot_low',
                    'price': bar[key], 'pivot_open_ms': bar['open_ms'],
                    'available_at_ms': window[index+wings]['close_ms']})
        if not pivots:
            return _blocked('main_btc_budget_no_confirmed_technical_target')
        context = {'mode': CONTEXT_MODE, 'source_key': source['source_key'],
            'origin_signal_time_ms': origin, 'position_id': source['position_id'],
            'direction': side, 'stop': stop, 'atr': atr, 'expires_at_ms': expiry,
            'selected_entry_structure': deepcopy(structure), 'opposite_pivots': pivots,
            'original_prefix_sha256': _digest(bars), 'known_by_ms': origin - 1}
        return {'enabled': True, 'allowed': True, 'status': 'technical_context_ready',
                'reasons': [], 'context': context}
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError):
        return _blocked('main_btc_budget_original_structure_invalid')


def plan_entry(config, *, state, target, pending, hourly_candles, now_ms, entry_price,
               rules, fee_rate, slippage_bps, maximum_qty=None, actual_qty=None,
               frozen_context=None, active_after_ms=None, funding_debit_per_unit=0):
    """Return a minimum quantity only if original technical room supports it.

    Entry is the projected/actual adverse fill, not the raw mark. Re-run with
    ``actual_qty`` after all parent source/DD/LLM caps; do not restore quantity
    reduced by those guards. ``frozen_context`` should be persisted by the main
    caller once per original source, so later candle revisions do not move it.
    Reductions/reversals must close their old inventory separately.
    """
    if POLICY_KEY not in config:
        return {'enabled': False, 'allowed': True, 'status': 'disabled', 'reasons': [], 'quantity': None}
    try:
        now = _integer(now_ms, positive=True)
        if not isinstance(state, Mapping) or state.get('mode') != 'simulation':
            return _blocked('main_btc_budget_simulation_only')
        held = _number(state.get('position_qty', 0))
        if held:
            return {'enabled': True, 'allowed': True, 'status': 'existing_inventory',
                    'reasons': [], 'quantity': None, 'new_risk_allowed': False}
        source = pullback._source(target)
        if source is None:
            return {'enabled': True, 'allowed': True, 'status': 'flat_target', 'reasons': [], 'quantity': None}
        if active_after_ms is not None and source['origin_signal_time_ms'] <= _integer(active_after_ms):
            return _blocked('main_btc_budget_signal_not_after_activation')
        if frozen_context is None:
            result = technical_context(target, pending, hourly_candles, now_ms=now)
            if not result['allowed']:
                return result
            context = result['context']
        else:
            context = deepcopy(frozen_context)
        if (context.get('mode') != CONTEXT_MODE or context.get('source_key') != source['source_key']
                or context.get('direction') != (1 if source['side'] == 'long' else -1)
                or context.get('origin_signal_time_ms') != source['origin_signal_time_ms']
                or _integer(context.get('known_by_ms')) >= source['origin_signal_time_ms']
                or now >= _integer(context.get('expires_at_ms'), positive=True)):
            return _blocked('main_btc_budget_frozen_context_invalid')
        entry = _number(entry_price, positive=True)
        side = context['direction']
        stop = _number(context['stop'], positive=True)
        if side * (entry - stop) <= 0:
            return _blocked('main_btc_budget_structure_stop_invalid')
        candidates = [p for p in context['opposite_pivots']
            if _integer(p['available_at_ms']) <= context['known_by_ms']
            and side * (_number(p['price'], positive=True) - entry) > 0]
        if not candidates:
            return _blocked('main_btc_budget_no_profitable_technical_target')
        pivot = min(candidates, key=lambda p: side * (p['price'] - entry))
        wallet = _number(state['wallet_balance'], positive=True)
        cap = _number(float(rules['max_qty']), positive=True)
        if maximum_qty is not None:
            cap = min(cap, _number(maximum_qty))
        result = budget.quantity_plan(config, equity=wallet, entry=entry, stop=stop,
            step=float(rules['step_size']), direction=side, technical_target=pivot['price'],
            fee_rate=fee_rate, slippage=slippage_bps / 10000,
            funding_debit_per_unit=funding_debit_per_unit, maximum_qty=cap,
            minimum_qty=float(rules['min_qty']), minimum_notional=float(rules['min_notional']),
            actual_qty=actual_qty)
        return {**result, 'context': context, 'technical_target': deepcopy(pivot),
                'stop': stop, 'entry_price': entry, 'direction': side,
                'source_key': source['source_key'], 'planned_at_ms': now,
                'policy': deepcopy(config[POLICY_KEY])}
    except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError):
        return _blocked('main_btc_budget_plan_invalid')


build_technical_context = technical_context


def check_fill(config, plan, *, state, entry_price, quantity, now_ms, rules,
               fee_rate, slippage_bps, maximum_qty=None, funding_debit_per_unit=0):
    """Revalidate the final observed quantity/price without moving frozen levels."""
    context = plan.get('context') or {}
    target = {'target_leverage': context.get('direction', 0),
              'position_id': context.get('position_id'),
              'origin_signal_time_ms': context.get('origin_signal_time_ms')}
    return plan_entry(config, state=state, target=target, pending={}, hourly_candles=[],
        now_ms=now_ms, entry_price=entry_price, rules=rules, fee_rate=fee_rate,
        slippage_bps=slippage_bps, maximum_qty=maximum_qty, actual_qty=quantity,
        frozen_context=context, funding_debit_per_unit=funding_debit_per_unit)


def position_tag(state, plan, *, entry_sequence, entry_time_ms, entry_price, quantity):
    """Build a tag only for a verified new filled episode; never tag old inventory."""
    if (not plan.get('allowed') or plan.get('status') in ('disabled', 'existing_inventory', 'flat_target')
            or state.get('mode') != 'simulation'):
        raise ValueError('main_btc_budget_new_position_plan_required')
    sequence = _integer(entry_sequence, positive=True)
    at = _integer(entry_time_ms, positive=True)
    entry = _number(entry_price, positive=True)
    qty = _number(quantity, positive=True)
    if (sequence != state['fill_count_total'] or not state.get('fills')
            or state['fills'][-1].get('sequence') != sequence
            or not math.isclose(abs(state['position_qty']), qty, abs_tol=1e-12)
            or not math.isclose(state['entry_price'], entry, rel_tol=1e-12)
            or not math.isclose(plan['entry_price'], entry, rel_tol=1e-12)
            or not math.isclose(_number(plan['quantity'], positive=True), qty, abs_tol=1e-12)
            or state['position_qty'] * plan['direction'] <= 0
            or at < plan['context']['origin_signal_time_ms'] or at >= plan['context']['expires_at_ms']):
        raise ValueError('main_btc_budget_new_position_identity_invalid')
    latest = state['fills'][-1]
    stamp = datetime.fromisoformat(latest['time_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or round(stamp.timestamp() * 1000) != at:
        raise ValueError('main_btc_budget_new_position_time_invalid')
    return {'mode': CONTEXT_MODE, 'account_epoch': state['created_at_utc'],
        'entry_sequence': sequence, 'entry_time_ms': at, 'entry_price': entry,
        'initial_quantity': qty, 'direction': plan['direction'], 'stop': plan['stop'],
        'technical_target': deepcopy(plan['technical_target']), 'source_key': plan['source_key'],
        'origin_signal_time_ms': plan['context']['origin_signal_time_ms'],
        'target_net_pnl_usdt': 100.0, 'policy': deepcopy(plan['policy']),
        'planned_loss_usdt': plan.get('planned_loss_usdt'),
        'loss_budget_usdt': plan.get('loss_budget_usdt', 25.0),
        'expected_net_profit_usdt': plan.get('expected_net_profit_usdt'),
        'context': deepcopy(plan['context'])}


def position_exit(state, mark_price, *, now_ms, fee_rate, slippage_bps,
                  funding_covered, fills=None, journal_path=None):
    """Protect a tagged position before optional full-ledger net100 realization.

    The original caller must validate mark freshness independently. A missing
    ledger/funding blocks only the profit assertion, never a valid tagged stop.
    Parent account hard stops, source reductions and trend/reversal exits remain.
    """
    tag = state.get(POSITION_KEY)
    base = {'enabled': bool(tag), 'should_exit': False, 'checked_at_ms': now_ms}
    if not tag:
        return {**base, 'status': 'legacy_or_flat'}
    try:
        now = _integer(now_ms, positive=True)
        mark = _number(mark_price, positive=True)
        qty = _number(state.get('position_qty', 0))
        if not qty:
            return {**base, 'status': 'flat'}
        side = tag['direction']
        if (tag.get('mode') != CONTEXT_MODE or tag.get('account_epoch') != state.get('created_at_utc')
                or type(side) is not int or side not in (-1, 1) or qty * side <= 0
                or abs(qty) > _number(tag['initial_quantity'], positive=True) + 1e-12
                or now < _integer(tag['entry_time_ms'], positive=True)):
            return {**base, 'status': 'tag_identity_invalid'}
        stop = _number(tag['stop'], positive=True)
        if side * (mark - stop) <= 0:
            return {**base, 'status': 'planned_stop', 'should_exit': True,
                    'reason': 'main_btc_budget_protective_stop', 'stop': stop}
        rows, integrity = ledger.complete_fills(state, journal_path=journal_path, fills=fills)
        if not integrity['complete']:
            return {**base, 'status': 'ledger_unavailable', 'ledger': integrity}
        episodes = ledger.reconstruct_episodes(state, rows, margin_leverage=10)
        if not episodes or episodes[-1]['closed'] or episodes[-1]['entry_sequence'] != tag['entry_sequence']:
            return {**base, 'status': 'tag_episode_mismatch'}
        fee = _number(fee_rate)
        slip = _number(slippage_bps)
        if not 0 <= fee < 1 or not 0 <= slip < 10000:
            raise ValueError('main_btc_budget_cost_invalid')
        episode = episodes[-1]
        exit_price = mark * (1 - side * slip / 10000)
        net = (episode['net_realized_pnl_usdt'] + qty * (exit_price - episode['entry_price'])
               - abs(qty) * exit_price * fee)
        loss_budget = min(25., _number(tag.get('loss_budget_usdt', 25.), positive=True))
        if net <= -loss_budget:
            return {**base, 'status': 'loss_budget', 'should_exit': True,
                'reason': 'main_btc_budget_loss_limit', 'projected_net_pnl_usdt': net,
                'loss_budget_usdt': loss_budget}
        if not funding_covered:
            return {**base, 'status': 'funding_unavailable'}
        reached = net >= 100.0
        return {**base, 'status': 'target_reached' if reached else 'below_target',
            'should_exit': reached, 'reason': 'main_btc_net100_target' if reached else None,
            'projected_net_pnl_usdt': net, 'target_net_pnl_usdt': 100.0,
            'projected_exit_price': exit_price, 'ledger': integrity,
            'source_origin': {'position_id': tag['context']['position_id'],
                'origin_signal_time_ms': tag['origin_signal_time_ms'],
                'entry_sequence': tag['entry_sequence']}}
    except (ValueError, KeyError, TypeError, ArithmeticError, OSError):
        return {**base, 'status': 'ledger_or_tag_unavailable'}
