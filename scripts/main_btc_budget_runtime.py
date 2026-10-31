"""Main BTC execution adapter for newly authorized net100/risk25 positions only.

The original simulation reconcile/observe paths retain source, drawdown, LLM,
funding, timing, mark freshness and accounting controls. No live client or
producer configuration is changed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os

import main_btc_profit_budget as planning
import trade_profit_budget as budget
from trading_execution import SimulationAccount


class BudgetSimulationAccount(SimulationAccount):
    def __init__(self, *args, policy, active_after_ms, **kwargs):
        config = {budget.POLICY_KEY: deepcopy(policy)}
        budget.validate(config)
        if type(active_after_ms) is not int or active_after_ms <= 0:
            raise ValueError('main_btc_budget_activation_required')
        # This non-None marker makes the existing independent risk monitor fetch
        # funding. _profit_decision never applies this marker to legacy positions.
        kwargs['profit_exit_policy'] = {'mode': 'main_btc_net100_marker'}
        super().__init__(*args, **kwargs)
        self.budget_config = config
        self.active_after_ms = active_after_ms
        self._budget_report = None
        self._budget_target = None

    def _costs(self):
        fee = float(os.getenv('SIM_TAKER_FEE', '0.00045') or .00045) * self.cost_multiplier
        slip = float(os.getenv('SIM_SLIPPAGE_BPS', '1.0') or 1.) * self.cost_multiplier
        if not math.isfinite(fee) or not 0 <= fee < 1 or not math.isfinite(slip) or not 0 <= slip < 10000:
            raise ValueError('main_btc_budget_cost_invalid')
        return fee, slip

    def reconcile(self, target, mark_price, rules=None, report=None, **kwargs):
        state = self.load()
        copied = deepcopy(target)
        requested = float(copied.get('target_leverage') or 0)
        held = float(state.get('position_qty') or 0)
        if held and requested * held < 0:
            # First reconcile only closes the old direction. A subsequent fresh
            # invocation must separately pass the entire original admission path.
            copied['target_leverage'] = 0.
        elif not held and requested:
            origin = copied.get('origin_signal_time_ms')
            if type(origin) is int and origin > self.active_after_ms:
                policy = self.budget_config[budget.POLICY_KEY]
                ceiling = min(policy['maximum_leverage'], policy['maximum_gross_notional_fraction'])
                copied['target_leverage'] = math.copysign(ceiling, requested)
            else:
                copied['target_leverage'] = 0.
        self._budget_report = deepcopy(report) if report is not None else {}
        self._budget_target = deepcopy(target)
        try:
            result = super().reconcile(copied, mark_price, rules, report, **kwargs)
            result['main_btc_profit_budget'] = deepcopy(self.load().get('main_btc_profit_budget'))
            # Parent reports its pre-budget ceiling. The booked position is the
            # authoritative quantity after the final minimum-quantity admission.
            result['target_qty'] = float(self.load().get('position_qty') or 0)
            return result
        finally:
            self._budget_report = None
            self._budget_target = None

    def _fill_to(self, state, target_qty, mark_price, now_ms, signal_time_ms, rules):
        held = float(state.get('position_qty') or 0)
        tag = state.get(planning.POSITION_KEY)
        if held:
            # Never let a final hook cross into a new unbudgeted direction.
            if target_qty * held < 0:
                target_qty = 0.
            if tag and target_qty * held > 0 and abs(target_qty) > abs(held):
                state['main_btc_profit_budget'] = {'enabled': True, 'allowed': False,
                    'status': 'existing_budget_position_no_additions', 'reasons': ['budget_position_addition_blocked']}
                return None
            fill = super()._fill_to(state, target_qty, mark_price, now_ms, signal_time_ms, rules)
            if fill and not state.get('position_qty') and tag:
                state['last_main_btc_profit_budget_position'] = deepcopy(tag)
                state.pop(planning.POSITION_KEY, None)
                fill['main_btc_profit_budget_position'] = deepcopy(tag)
            return fill
        if not target_qty:
            return super()._fill_to(state, target_qty, mark_price, now_ms, signal_time_ms, rules)
        target = self._budget_target or {}
        report = self._budget_report or {}
        market = report.get('btc_pullback_market_data') or {}
        pending = state.get('btc_pullback_pending') or {}
        decision = state.get('btc_pullback_entry') or {}
        if not decision.get('allowed') or pending.get('status') != 'ready':
            state['main_btc_profit_budget'] = {'enabled': True, 'allowed': False,
                'status': 'waiting_for_original_pullback', 'reasons': ['original_pullback_not_ready']}
            return None
        fee, slip = self._costs()
        side = 1 if target_qty > 0 else -1
        actual_entry = mark_price * (1 + side * slip / 10000)
        source = pending.get('source_key')
        frozen = state.get('main_btc_budget_pending_context') or {}
        context = frozen.get('context') if frozen.get('source_key') == source else None
        five = market.get('five_minute') or []
        try:
            latest = five[-1]
            current_bar = ((now_ms + 1) // 300_000 - 1) * 300_000
            volume = float(latest[5]) if int(latest[0]) == current_bar else 0.
            if not math.isfinite(volume) or volume < 0:
                volume = 0.
        except (ValueError, KeyError, TypeError, IndexError):
            volume = 0.
        capacity = min(abs(target_qty), volume * .1)
        plan = planning.plan_entry(self.budget_config, state=state, target=target,
            pending=pending, hourly_candles=market.get('hourly', []), now_ms=now_ms,
            entry_price=actual_entry, rules=rules, fee_rate=fee, slippage_bps=slip,
            maximum_qty=capacity, frozen_context=context, active_after_ms=self.active_after_ms)
        if plan.get('context'):
            state['main_btc_budget_pending_context'] = {
                'source_key': source, 'context': deepcopy(plan['context'])}
        state['main_btc_profit_budget'] = deepcopy(plan)
        if not plan.get('allowed') or not plan.get('quantity'):
            return None
        checked = planning.check_fill(self.budget_config, plan, state=state,
            entry_price=actual_entry, quantity=plan['quantity'], now_ms=now_ms, rules=rules,
            fee_rate=fee, slippage_bps=slip, maximum_qty=capacity)
        state['main_btc_profit_budget'] = deepcopy(checked)
        if not checked.get('allowed'):
            return None
        fill = super()._fill_to(state, side * checked['quantity'], mark_price, now_ms, signal_time_ms, rules)
        if fill:
            state[planning.POSITION_KEY] = planning.position_tag(state, checked,
                entry_sequence=fill['sequence'], entry_time_ms=now_ms,
                entry_price=fill['price'], quantity=abs(state['position_qty']))
            fill['main_btc_profit_budget_position'] = deepcopy(state[planning.POSITION_KEY])
        return fill

    def _profit_decision(self, state, mark_price, now_ms, funding_covered, report=None, target=None):
        fee, slip = self._costs()
        result = planning.position_exit(state, mark_price, now_ms=now_ms,
            fee_rate=fee, slippage_bps=slip, funding_covered=funding_covered,
            journal_path=self.path.with_suffix('.fills.jsonl') if self.persist else None)
        result['budget_protection_enabled'] = bool(state.get(planning.POSITION_KEY))
        if result.get('reason') in {'main_btc_budget_protective_stop', 'main_btc_budget_loss_limit'}:
            # Parent funding guards apply to enabled PROFIT exits. A technical
            # stop is a risk reduction and must not be suppressed by them.
            result['enabled'] = False
            tag = state[planning.POSITION_KEY]
            result['source_origin'] = {'position_id': tag['context']['position_id'],
                'origin_signal_time_ms': tag['origin_signal_time_ms'],
                'entry_sequence': tag['entry_sequence']}
        state['profit_exit_status'] = deepcopy(result)
        return result

    @staticmethod
    def _record_profit_exit(state, decision, fill):
        SimulationAccount._record_profit_exit(state, decision, fill)
        fill['reason'] = (decision['reason'] if decision.get('reason') in {
            'main_btc_budget_protective_stop', 'main_btc_budget_loss_limit'} else 'main_btc_net100_target')
        state['last_profit_target_exit']['fill'] = deepcopy(fill)
        state['profit_exit_status'].update(reason=fill['reason'], budget_exit=True)
