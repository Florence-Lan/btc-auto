"""Main-only short revision compatibility never changes old experiment ledgers."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from main_stock_rule_revision import declared_main_baseline_matches
import run_stock_exit_response_shadow as shadow
import stock_short_confirmation as short


def fixture():
    expected = {'signal_family': 'donchian20', 'entry_enabled': True,
                'risk_fraction_per_trade': .0025,
                'portfolio_gross_notional_fraction': 1.0,
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


@pytest.fixture
def budget_policy(monkeypatch):
    # The verifier must use the exact policy exported by the reviewed module,
    # including scalar types. This isolated stub tests that dependency contract.
    policy = {'mode': 'net100_risk25_v1', 'net_profit_target_usdt': 100.,
              'maximum_planned_loss_usdt': 25., 'maximum_equity_risk_fraction': .025,
              'minimum_net_reward_risk': 4., 'maximum_leverage': 10.,
              'maximum_gross_notional_fraction': 10.}
    monkeypatch.setitem(sys.modules, 'trade_profit_budget', SimpleNamespace(POLICY=policy))
    return policy


def with_budget(state, policy):
    previous = deepcopy(state['rule'])
    current = {**previous, 'trade_profit_budget_policy': deepcopy(policy),
               'risk_fraction_per_trade': .025, 'portfolio_gross_notional_fraction': 10.}
    state.update(rule=current, trade_profit_budget_active_after_ms=3000)
    state['rule_history'].append({'scope': 'profit_budget_only', 'activated_at_ms': 3000,
                                  'previous_rule': previous, 'new_rule': deepcopy(current)})
    return state


def test_exact_two_stage_main_revision_chain_keeps_old_baseline_and_inputs(budget_policy):
    state, expected = fixture()
    with_budget(state, budget_policy)
    original = deepcopy((state, expected))
    assert declared_main_baseline_matches(state, expected)
    assert (state, expected) == original
    assert state['rule']['risk_fraction_per_trade'] == .025
    assert expected['risk_fraction_per_trade'] == .0025


def test_actual_reviewed_budget_policy_supports_only_the_declared_allocation_chain():
    import trade_profit_budget as actual
    state, expected = fixture()
    with_budget(state, actual.POLICY)
    assert declared_main_baseline_matches(state, expected)
    assert state['rule']['trade_profit_budget_policy'] == actual.POLICY
    assert expected['portfolio_gross_notional_fraction'] == 1.0


@pytest.mark.parametrize('mutation', [
    lambda s: s['rule'].update(risk_fraction_per_trade=.02),
    lambda s: s['rule'].update(risk_fraction_per_trade=.0025),
    lambda s: s['rule'].update(portfolio_gross_notional_fraction=1.),
    lambda s: s['rule'].update(portfolio_gross_notional_fraction=10),
    lambda s: s['rule']['trade_profit_budget_policy'].update(maximum_planned_loss_usdt=24.),
    lambda s: s['rule']['trade_profit_budget_policy'].update(maximum_planned_loss_usdt=25),
    lambda s: s['rule']['trade_profit_budget_policy'].update(extra=True),
    lambda s: s['rule'].update(unreviewed_policy={}),
    lambda s: s['rule'].pop(short.POLICY_KEY),
    lambda s: s.pop('trade_profit_budget_active_after_ms'),
    lambda s: s.update(trade_profit_budget_active_after_ms=True),
    lambda s: s.update(trade_profit_budget_active_after_ms=3000.),
    lambda s: s.update(trade_profit_budget_active_after_ms=2000),
    lambda s: s['rule_history'][-1].update(scope='risk_fraction_only'),
    lambda s: s['rule_history'][-1].update(activated_at_ms=3000.),
    lambda s: s['rule_history'][-1].update(activated_at_ms=2999),
    lambda s: s['rule_history'][-1]['previous_rule'].update(risk_fraction_per_trade=.01),
    lambda s: s['rule_history'][-1]['new_rule'].update(risk_fraction_per_trade=.01),
    lambda s: s['rule_history'].pop(0),
    lambda s: s['rule_history'].reverse(),
    lambda s: s['rule_history'].append(deepcopy(s['rule_history'][-1])),
])
def test_budget_chain_rejects_any_other_rule_policy_cutoff_or_history_change(budget_policy, mutation):
    state, expected = fixture()
    with_budget(state, budget_policy)
    mutation(state)
    original = deepcopy((state, expected))
    assert not declared_main_baseline_matches(state, expected)
    assert (state, expected) == original


def test_hidden_intermediate_risk_change_is_not_ignored(budget_policy):
    state, expected = fixture()
    with_budget(state, budget_policy)
    before = deepcopy(state['rule_history'][0]['new_rule'])
    after = {**before, 'risk_fraction_per_trade': .01}
    state['rule_history'].insert(1, {'scope': 'unapproved', 'activated_at_ms': 2500,
                                   'previous_rule': before, 'new_rule': after})
    assert not declared_main_baseline_matches(state, expected)


def test_current_budget_does_not_validate_against_a_replaced_declared_baseline(budget_policy):
    state, expected = fixture()
    with_budget(state, budget_policy)
    expected['risk_fraction_per_trade'] = .025
    assert not declared_main_baseline_matches(state, expected)


def test_budget_policy_module_unavailable_fails_closed(monkeypatch, budget_policy):
    state, expected = fixture()
    with_budget(state, budget_policy)
    monkeypatch.setitem(sys.modules, 'trade_profit_budget', None)
    assert not declared_main_baseline_matches(state, expected)


def test_exit_and_early_restore_exact_budget_chain_without_rule_or_financial_writes(tmp_path, monkeypatch, budget_policy):
    spec, rules = setup_source(tmp_path, monkeypatch)
    for symbol in shadow.SYMBOLS:
        path = tmp_path / (symbol + '.json')
        state = json.loads(path.read_text(encoding='utf-8'))
        path.write_text(json.dumps(with_budget(state, budget_policy)), encoding='utf-8')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original_rules = deepcopy(rules)
    assert set(shadow.check_source_rules(spec, rules)) == set(shadow.SYMBOLS)
    import run_stock_early_recovery_shadow as early
    assert set(early.check_source_rules(spec, rules)) == set(shadow.SYMBOLS)
    assert rules == original_rules
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
