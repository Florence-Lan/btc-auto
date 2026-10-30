"""Main-only short revision compatibility never changes old experiment ledgers."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from main_stock_rule_revision import declared_main_baseline_matches
import run_stock_exit_response_shadow as shadow
import stock_short_confirmation as short


def fixture():
    expected = {'signal_family': 'donchian20', 'entry_enabled': True,
                'risk_fraction_per_trade': .0025,
                'stock_event_risk_policy': {'calendar_path': 'config/original_calendar.json'}}
    current = {**deepcopy(expected), short.POLICY_KEY: deepcopy(short.POLICY)}
    revision = {'scope': 'short_confirmation_only', 'activated_at_ms': 2000,
                'previous_rule': deepcopy(expected), 'new_rule': deepcopy(current)}
    state = {'start_ms': 1000, 'rule': current, 'short_confirmation_active_after_ms': 2000,
             'rule_history': [revision], 'mode': 'simulation', 'places_orders': False,
             'wallet_balance': 998.5, 'fees_paid': .5, 'funding_pnl': -1.,
             'fill_count_total': 3, 'position': {'direction': -1, 'qty': .1, 'stop': 102.}}
    return state, expected


def test_old_baseline_and_precise_new_revision_are_read_only():
    state, expected = fixture()
    before = deepcopy((state, expected))
    assert declared_main_baseline_matches(state, expected)
    assert (state, expected) == before
    old = {'rule': deepcopy(expected)}
    assert declared_main_baseline_matches(old, expected)
    assert old == {'rule': expected}


@pytest.mark.parametrize('mutation', [
    lambda state: state['rule'].update(risk_fraction_per_trade=.01),
    lambda state: state['rule'].update(entry_enabled=1),
    lambda state: state['rule'].update(extra_source_pin='not-approved'),
    lambda state: state['rule'].pop('signal_family'),
    lambda state: state['rule'][short.POLICY_KEY].update(ema_fast=9),
    lambda state: state['rule'][short.POLICY_KEY].update(require_close_below_fast_mid=1),
    lambda state: state['rule'][short.POLICY_KEY].update(warmup_bars=60.),
    lambda state: state['rule'][short.POLICY_KEY].update(extra=True),
    lambda state: state['rule'].update(short_confirmation_policy=None),
    lambda state: state.update(rule_history=[]),
    lambda state: state.update(rule_history={}),
    lambda state: state['rule_history'][0].update(scope='entry_timing_only'),
    lambda state: state['rule_history'][0]['previous_rule'].update(risk_fraction_per_trade=.01),
    lambda state: state['rule_history'][0]['new_rule'].update(entry_enabled=1),
    lambda state: state['rule_history'][0].update(activated_at_ms=1999),
    lambda state: state['rule_history'][0].update(activated_at_ms=2000.),
    lambda state: state.pop('short_confirmation_active_after_ms'),
    lambda state: state.update(short_confirmation_active_after_ms=True),
    lambda state: state.update(short_confirmation_active_after_ms=2000.),
    lambda state: state.update(short_confirmation_active_after_ms=1000),
    lambda state: state.update(start_ms=2000),
    lambda state: state.update(start_ms=-1),
    lambda state: state.update(start_ms=True),
    lambda state: state.pop('start_ms'),
])
def test_no_other_rule_or_unbound_activation_is_accepted(mutation):
    state, expected = fixture()
    mutation(state)
    before = deepcopy((state, expected))
    assert not declared_main_baseline_matches(state, expected)
    assert (state, expected) == before


@pytest.mark.parametrize('state,expected', [(None, {}), ({}, {}), ({'rule': None}, {}),
                                          ({'rule': {}}, None), ({'rule': []}, {})])
def test_malformed_inputs_fail_closed(state, expected):
    assert not declared_main_baseline_matches(state, expected)


def test_malformed_history_entries_do_not_hide_a_valid_bound_revision():
    state, expected = fixture()
    state['rule_history'][:0] = [None, False, 'ignored-malformed-row']
    assert declared_main_baseline_matches(state, expected)


def setup_source(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, 'ROOT', tmp_path)
    state, expected = fixture()
    accounts, rules = [], {}
    for symbol in shadow.SYMBOLS:
        source = {**deepcopy(state), 'symbol': symbol}
        path = tmp_path / (symbol + '.json')
        path.write_text(json.dumps(source), encoding='utf-8')
        accounts.append({'symbol': symbol, 'state_path': path.name})
        rules[symbol] = {'baseline_normal': deepcopy(expected)}
    plan = {'execution_mode': 'simulation', 'live_orders_allowed': False, 'accounts': accounts}
    (tmp_path / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    return {'source_plan': 'plan.json'}, rules


def test_exit_and_early_check_accept_main_revision_without_writing(tmp_path, monkeypatch):
    spec, rules = setup_source(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    rules_before = deepcopy(rules)
    assert set(shadow.check_source_rules(spec, rules)) == set(shadow.SYMBOLS)
    import run_stock_early_recovery_shadow as early
    assert early.check_source_rules is shadow.check_source_rules
    assert set(early.check_source_rules(spec, rules)) == set(shadow.SYMBOLS)
    assert rules == rules_before
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize('field,value', [('mode', 'live'), ('places_orders', True),
                                       ('symbol', 'UNRELATED')])
def test_account_identity_still_blocks(tmp_path, monkeypatch, field, value):
    spec, rules = setup_source(tmp_path, monkeypatch)
    path = tmp_path / (shadow.SYMBOLS[0] + '.json')
    state = json.loads(path.read_text(encoding='utf-8'))
    state[field] = value
    path.write_text(json.dumps(state), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='differs from the declared baseline'):
        shadow.check_source_rules(spec, rules)
    assert path.read_bytes() == before


def test_risk_drift_in_main_still_blocks_exit_restore(tmp_path, monkeypatch):
    spec, rules = setup_source(tmp_path, monkeypatch)
    path = tmp_path / (shadow.SYMBOLS[0] + '.json')
    state = json.loads(path.read_text(encoding='utf-8'))
    state['rule']['risk_fraction_per_trade'] = .01
    state['rule_history'][0]['new_rule'] = deepcopy(state['rule'])
    path.write_text(json.dumps(state), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='differs from the declared baseline'):
        shadow.check_source_rules(spec, rules)
    assert path.read_bytes() == before
