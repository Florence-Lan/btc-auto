"""A trend deployment cannot replace money, old signals or canceled accounts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import activate_main_trend_20261102 as activation
from main_stock_rule_revision import declared_main_baseline_matches
from test_main_rule_revision import fixture, with_budget
import trade_profit_budget as budget


@pytest.fixture
def trend_policy(monkeypatch):
    policy = {'mode': 'reviewed_main_trend', 'confirmation_bars': 2,
              'require_price_slow': True, 'maximum_age_seconds': 1200.}
    monkeypatch.setitem(sys.modules, 'main_stock_trend', SimpleNamespace(POLICY=policy))
    return policy


def with_trend(state, policy):
    previous = deepcopy(state['rule'])
    state.update(rule={**deepcopy(previous), activation.POLICY_KEY: deepcopy(policy)},
                 main_trend_active_after_ms=4000, signal_active_after_ms=4000)
    state['rule_history'].append({'scope': 'main_trend_only', 'activated_at_ms': 4000,
        'previous_rule': previous, 'new_rule': deepcopy(state['rule'])})
    return state


def test_precise_third_revision_keeps_frozen_baseline_and_financial_inputs(trend_policy):
    state, expected = fixture()
    with_budget(state, budget.POLICY)
    with_trend(state, trend_policy)
    original = deepcopy((state, expected))
    assert declared_main_baseline_matches(state, expected)
    assert (state, expected) == original


@pytest.mark.parametrize('mutate', [
    lambda s: s['rule'].update(risk_fraction_per_trade=.03),
    lambda s: s['rule'].update(portfolio_gross_notional_fraction=10),
    lambda s: s['rule'][activation.POLICY_KEY].update(confirmation_bars=2.),
    lambda s: s['rule'][activation.POLICY_KEY].update(require_price_slow=1),
    lambda s: s['rule'][activation.POLICY_KEY].update(extra=True),
    lambda s: s['rule'].pop('trade_profit_budget_policy'),
    lambda s: s['rule'].pop('short_confirmation_policy'),
    lambda s: s.update(main_trend_active_after_ms=3000),
    lambda s: s.update(main_trend_active_after_ms=4000.),
    lambda s: s.update(main_trend_active_after_ms=True),
    lambda s: s.pop('main_trend_active_after_ms'),
    lambda s: s.update(signal_active_after_ms=3999),
    lambda s: s.update(signal_active_after_ms=4000.),
    lambda s: s.pop('signal_active_after_ms'),
    lambda s: s['rule_history'][-1].update(scope='unreviewed'),
    lambda s: s['rule_history'][-1].update(activated_at_ms=4000.),
    lambda s: s['rule_history'][-1]['previous_rule'].update(risk_fraction_per_trade=.03),
    lambda s: s['rule_history'][-1]['new_rule'].update(risk_fraction_per_trade=.03),
    lambda s: s['rule_history'].pop(0),
    lambda s: s['rule_history'].pop(1),
    lambda s: s['rule_history'].reverse(),
    lambda s: s['rule_history'].append(deepcopy(s['rule_history'][-1])),
])
def test_third_revision_rejects_risk_policy_type_history_and_cutoff_drift(trend_policy, mutate):
    state, expected = fixture()
    with_budget(state, budget.POLICY)
    with_trend(state, trend_policy)
    mutate(state)
    original = deepcopy((state, expected))
    assert not declared_main_baseline_matches(state, expected)
    assert (state, expected) == original


def test_hidden_transition_before_trend_is_not_excused(trend_policy):
    state, expected = fixture()
    with_budget(state, budget.POLICY)
    with_trend(state, trend_policy)
    state['rule_history'].insert(2, {'scope': 'hidden', 'previous_rule': {}, 'new_rule': {}})
    assert not declared_main_baseline_matches(state, expected)


def test_optional_trend_module_missing_fails_closed_without_breaking_budget(monkeypatch, trend_policy):
    state, expected = fixture()
    with_budget(state, budget.POLICY)
    original = deepcopy(state)
    with_trend(state, trend_policy)
    monkeypatch.setitem(sys.modules, 'main_stock_trend', None)
    assert not declared_main_baseline_matches(state, expected)
    assert declared_main_baseline_matches(original, expected)


def transition():
    before = {'rule': {'risk_fraction_per_trade': .025}, 'rule_history': [],
        'start_ms': 1000, 'created_at_utc': 'original', 'wallet_balance': 1008.,
        'position': {'qty': .04, 'entry': 1171.995554, 'stop': 1224.81},
        'position_history': [{'time_ms': 1100, 'signed_qty': -.04}],
        'last_signal_time_ms': 2000, 'last_signal_result': 'signal', 'signal_status': 'holding',
        'trade_profit_budget_active_after_ms': 3000,
        'trade_profit_budget_legacy_position_rule': {'risk_fraction_per_trade': .0025},
        'pending_signal': {'boundary_ms': 2500}, 'signal_active_after_ms': 3000}
    after = deepcopy(before)
    after.update(rule={**before['rule'], activation.POLICY_KEY: {'mode': 'reviewed'}},
                 main_trend_active_after_ms=4000, signal_active_after_ms=4000, pending_signal=None)
    after['rule_history'].append({'scope': 'main_trend_only', 'activated_at_ms': 4000,
        'previous_rule': deepcopy(before['rule']), 'new_rule': deepcopy(after['rule'])})
    return before, after


def test_stopped_migration_preserves_legacy_position_and_consumed_signal():
    before, after = transition()
    original = deepcopy((before, after))
    activation.verify_stock_transition(before, after, after['rule'], 4000)
    assert (before, after) == original


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(wallet_balance=1009.),
    lambda s: s['position'].update(qty=.4),
    lambda s: s['position'].update(stop=1200.),
    lambda s: s['position'].update(entry=1180.),
    lambda s: s['position_history'].append({'time_ms': 3900, 'signed_qty': 0}),
    lambda s: s.update(last_signal_time_ms=None),
    lambda s: s.update(last_signal_result=None),
    lambda s: s.update(trade_profit_budget_active_after_ms=4000),
    lambda s: s['trade_profit_budget_legacy_position_rule'].update(risk_fraction_per_trade=.025),
    lambda s: s.update(pending_signal={'boundary_ms': 2500}),
    lambda s: s.update(signal_active_after_ms=3000),
    lambda s: s['rule_history'][-1].update(scope='profit_budget_only'),
    lambda s: s['rule_history'][-1]['previous_rule'].update(risk_fraction_per_trade=.03),
])
def test_stopped_migration_rejects_financial_legacy_signal_and_pending_changes(mutate):
    before, after = transition()
    mutate(after)
    with pytest.raises(RuntimeError):
        activation.verify_stock_transition(before, after, after['rule'], 4000)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_preservation_detects_canceled_ledger_and_old_prefix_changes(tmp_path):
    root, output = tmp_path, tmp_path/'data/research/proof'
    base = root/activation.GENERATION
    old, new = transition()
    states = []
    for name in activation.ACCOUNTS:
        path = base/name/'state.json'
        state = {**deepcopy(old), 'initial_balance': 1000., 'symbol': name,
                 'mode': 'simulation', 'places_orders': False}
        write(path, state)
        states.append(path)
        (path.parent/'fills.jsonl').write_text('{"old":true}\n', encoding='utf-8')
        (path.parent/'rule_revisions.jsonl').write_text('', encoding='utf-8')
    for relative, depth, count in activation.COMPARISONS:
        directory = root/relative
        for index in range(count):
            path = directory/str(index)/'state.json' if depth == 1 else directory/'symbol'/str(index)/'state.json'
            write(path, {'initial_balance': 1000., 'wallet_balance': 1000., 'places_orders': False})
            (path.parent/'funding.jsonl').write_text('{"debit":0}\n', encoding='utf-8')
            states.append(path)
        write(directory/'manifest.json', {'frozen': True})
        write(directory/'status.json', {'running': False, 'status': 'cancelled_by_user'})
    write(root/activation.PLAN_PATH, {'old': True})
    write(root/'config/paper_service_extensions_20261020.json', {'disabled': True})
    write(base/'manifest.json', {'current_source_sha256': {}})
    activation.capture_preservation(output, root)
    changes = {}
    for name in activation.STOCKS:
        path = base/name/'state.json'
        state = activation.read(path)
        after = deepcopy(state)
        for key in activation.STATE_CHANGES:
            if key in new:
                after[key] = deepcopy(new[key])
        write(path, after)
        changes[name] = {'new_rule': new['rule']}
    proof = {'candidate_plan': {'new': True}, 'source_sha256': {}, 'main_rule_changes': changes}
    write(root/activation.PLAN_PATH, proof['candidate_plan'])
    assert activation.verify_preservation(output, proof, 4000, root=root)['result'] == 'PASS'
    frozen_path = states[-1]
    raw = frozen_path.read_bytes()
    frozen_path.write_bytes(raw+b' ')
    with pytest.raises(RuntimeError, match='bytes were rewritten'):
        activation.verify_preservation(output, proof, 4000, root=root)
    frozen_path.write_bytes(raw)
    journal = states[-1].parent/'funding.jsonl'
    journal.write_text('{"debit":1}\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='prefix changed'):
        activation.verify_preservation(output, proof, 4000, root=root)


def test_source_closure_discovers_local_lazy_imports_and_cannot_pin_external_files(tmp_path):
    write(tmp_path/'unrelated.json', {})
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    (scripts/'main.py').write_text('def f():\n import extra\n', encoding='utf-8')
    (scripts/'extra.py').write_text('import json\n', encoding='utf-8')
    assert set(activation.source_closure(tmp_path, ['scripts/main.py'])) == {'scripts/main.py', 'scripts/extra.py'}
    with pytest.raises(ValueError):
        activation.contained(tmp_path, '../outside.json')


def test_reviewed_plan_binds_sources_and_account_rules_but_not_preparation_time():
    plan = {'prepared_at_utc': 'first', 'source_sha256': {'source': 'first'}, 'rules': {'risk': .025}}
    activation.check_reviewed(plan, {**plan, 'prepared_at_utc': 'later'})
    with pytest.raises(RuntimeError, match='Concrete reviewed plan differs'):
        activation.check_reviewed(plan, {**plan, 'source_sha256': {'source': 'later'}})
    with pytest.raises(RuntimeError):
        activation.check_reviewed(plan, {**plan, 'rules': {'risk': .03}})
