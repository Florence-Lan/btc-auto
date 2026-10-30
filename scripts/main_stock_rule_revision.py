"""Recognize one explicitly recorded main-only paper rule revision.

This never derives or updates experimental rules. Existing experiments retain
their declared baseline while the main accounts may enable the fixed short
confirmation alternative.
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

    A different risk setting, a replacement baseline, an unrecorded policy or
    an activation not bound to the current state remains a hard mismatch.
    The function performs no I/O, clock mutation or account writes.
    """
    if not isinstance(state, Mapping) or not isinstance(expected, Mapping):
        return False
    current = state.get('rule')
    if not isinstance(current, Mapping):
        return False
    if _same_value(current, expected):
        return True
    if (short.POLICY_KEY in expected or not _same_policy(current.get(short.POLICY_KEY))
            or not _same_value({key: value for key, value in current.items()
                                if key != short.POLICY_KEY}, expected)):
        return False
    started = state.get('start_ms')
    activated = state.get('short_confirmation_active_after_ms')
    if (type(started) is not int or started < 0 or type(activated) is not int
            or activated <= started):
        return False
    history = state.get('rule_history')
    if not isinstance(history, list):
        return False
    return any(isinstance(revision, Mapping)
               and revision.get('scope') == 'short_confirmation_only'
               and type(revision.get('activated_at_ms')) is int
               and revision['activated_at_ms'] == activated
               and _same_value(revision.get('previous_rule'), expected)
               and _same_value(revision.get('new_rule'), current)
               for revision in history)
