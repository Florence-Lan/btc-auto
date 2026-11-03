"""Recognize explicitly recorded main-only paper rule revisions.

This never derives or updates experimental rules. Existing experiments retain
their declared baseline while the main accounts may enable the fixed short
confirmation alternative and a separately activated profit/risk budget.
The main trend overlay is a third exact, prospectively bound addition.
"""
from __future__ import annotations

from collections.abc import Mapping

import stock_short_confirmation as short


def _same_value(left, right):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same_value(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _same_policy(value):
    return (isinstance(value, Mapping) and set(value) == set(short.POLICY)
            and all(type(value[key]) is type(expected) and value[key] == expected
                    for key, expected in short.POLICY.items()))


def declared_main_baseline_matches(state, expected):
    """Accept the original rule or its exact, prospectively recorded revision.

    Only the recorded budget's fixed risk/notional allocation change is allowed;
    other risk drift, a replaced baseline or an unbound activation is rejected.
    The function performs no I/O, clock mutation or account writes.
    """
    if not isinstance(state, Mapping) or not isinstance(expected, Mapping):
        return False
    current = state.get('rule')
    if not isinstance(current, Mapping):
        return False
    if _same_value(current, expected):
        return True
    entry_key = 'main_stock_entry_trend_policy'
    if entry_key in current:
        try:
            import main_stock_entry_trend as entry
        except ImportError:
            return False
        activated = state.get('entry_trend_active_after_ms')
        old_activated = state.get('main_trend_active_after_ms')
        history = state.get('rule_history')
        if (entry_key in expected or not isinstance(getattr(entry, 'POLICY', None), Mapping)
                or not _same_value(current[entry_key], entry.POLICY)
                or type(activated) is not int or type(old_activated) is not int
                or activated <= old_activated or not isinstance(history, list)):
            return False
        previous = {key: value for key, value in current.items() if key != entry_key}
        matches = [(i, row) for i, row in enumerate(history) if isinstance(row, Mapping)
                   and ('previous_rule' in row or 'new_rule' in row)]
        if not matches:
            return False
        index, revision = matches[-1]
        if (revision.get('scope') != 'entry_trend_only'
                or type(revision.get('activated_at_ms')) is not int
                or revision['activated_at_ms'] != activated
                or not _same_value(revision.get('previous_rule'), previous)
                or not _same_value(revision.get('new_rule'), current)):
            return False
        # The overlay deliberately leaves the old signal/exit clocks intact.
        baseline = {**state, 'rule': previous, 'rule_history': history[:index]}
        return declared_main_baseline_matches(baseline, expected)
    trend_key = 'main_stock_trend_policy'
    if trend_key in current:
        # A main-only overlay must extend the complete, independently checked
        # short -> budget chain. Never authorize it by merely ignoring its key.
        try:
            import main_stock_trend as trend
        except ImportError:
            return False
        activated = state.get('main_trend_active_after_ms')
        budget_activated = state.get('trade_profit_budget_active_after_ms')
        history = state.get('rule_history')
        if (trend_key in expected or 'trade_profit_budget_policy' not in current
                or not isinstance(getattr(trend, 'POLICY', None), Mapping)
                or not _same_value(current[trend_key], trend.POLICY)
                or type(activated) is not int or type(budget_activated) is not int
                or activated <= budget_activated
                or type(state.get('signal_active_after_ms')) is not int
                or state['signal_active_after_ms'] != activated
                or not isinstance(history, list)):
            return False
        previous = {key: value for key, value in current.items() if key != trend_key}
        for index, revision in enumerate(history):
            if (not isinstance(revision, Mapping)
                    or revision.get('scope') != 'main_trend_only'
                    or type(revision.get('activated_at_ms')) is not int
                    or revision['activated_at_ms'] != activated
                    or not _same_value(revision.get('previous_rule'), previous)
                    or not _same_value(revision.get('new_rule'), current)):
                continue
            if any(isinstance(item, Mapping) and (
                    'previous_rule' in item or 'new_rule' in item)
                    for item in history[index + 1:]):
                continue
            baseline_state = {**state, 'rule': previous, 'rule_history': history[:index]}
            if declared_main_baseline_matches(baseline_state, expected):
                return True
        return False
    budget_key = 'trade_profit_budget_policy'
    if (short.POLICY_KEY in expected or budget_key in expected
            or not _same_policy(current.get(short.POLICY_KEY))):
        return False
    intermediate = {**expected, short.POLICY_KEY: current[short.POLICY_KEY]}
    has_budget = budget_key in current
    if has_budget:
        # Lazy import keeps the original baseline usable if an optional new
        # deployment is incomplete. A budget-bearing rule fails closed until
        # the exact reviewed policy module is available.
        try:
            import trade_profit_budget as budget
        except ImportError:
            return False
        if (not isinstance(getattr(budget, 'POLICY', None), Mapping)
                or not _same_value(current[budget_key], budget.POLICY)
                or not _same_value(intermediate.get('risk_fraction_per_trade'), .0025)
                or not _same_value(intermediate.get('portfolio_gross_notional_fraction'), 1.0)):
            return False
        budget_rule = {**intermediate, budget_key: budget.POLICY,
                       'risk_fraction_per_trade': .025,
                       'portfolio_gross_notional_fraction': 10.0}
        if not _same_value(current, budget_rule):
            return False
    elif not _same_value(current, intermediate):
        return False
    started = state.get('start_ms')
    activated = state.get('short_confirmation_active_after_ms')
    if (type(started) is not int or started < 0 or type(activated) is not int
            or activated <= started):
        return False
    history = state.get('rule_history')
    if not isinstance(history, list):
        return False
    short_matches = [index for index, revision in enumerate(history)
                     if isinstance(revision, Mapping)
                     and revision.get('scope') == 'short_confirmation_only'
                     and type(revision.get('activated_at_ms')) is int
                     and revision['activated_at_ms'] == activated
                     and _same_value(revision.get('previous_rule'), expected)
                     and _same_value(revision.get('new_rule'), intermediate)]
    if not has_budget:
        return bool(short_matches)
    budget_activated = state.get('trade_profit_budget_active_after_ms')
    if (type(budget_activated) is not int or budget_activated <= activated):
        return False
    for index in short_matches:
        # Historical changes before the declared baseline are expected. After
        # its short revision the only recorded rule transition may be this
        # exact, later budget addition; hidden risk changes/rollbacks break it.
        later = [revision for revision in history[index + 1:]
                 if isinstance(revision, Mapping)
                 and ('previous_rule' in revision or 'new_rule' in revision)]
        if len(later) != 1:
            continue
        revision = later[0]
        if (revision.get('scope') == 'profit_budget_only'
                and type(revision.get('activated_at_ms')) is int
                and revision['activated_at_ms'] == budget_activated
                and _same_value(revision.get('previous_rule'), intermediate)
                and _same_value(revision.get('new_rule'), current)):
            return True
    return False
