"""Direct main-paper activation preserves cash, inventory and consumed signals."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import run_stock_entry_timing_trial as trial
import stock_short_confirmation as short


def test_main_profile_changes_only_short_confirmation():
    for symbol in trial.SYMBOLS:
        old = paper.stock_config({'stock_research_profile': 'config/stock_event_guarded_20261020.json'}, symbol)
        new = paper.stock_config({'stock_research_profile': 'config/stock_main_price_led_short_20261030.json'}, symbol)
        assert new.pop(short.POLICY_KEY) == short.POLICY
        assert new == old


def test_main_short_activation_retains_every_existing_state_field(tmp_path, monkeypatch):
    old = paper.stock_config({'stock_research_profile': 'config/stock_event_guarded_20261020.json'}, 'MUUSDT')
    new = {**old, short.POLICY_KEY: deepcopy(short.POLICY)}
    state = paper.initial_stock('MUUSDT', 1000, old)
    state['signal_timeframe'] = old['signal_timeframe']
    state.update(wallet_balance=997., realized_pnl=-2., fees_paid=1., funding_pnl=0.,
        fills=[{'sequence': 1}], fill_count_total=1, pending_signal={
            'boundary_ms': 2000, 'expires_at_ms': 3000, 'signal': {'direction': -1}},
        signal_active_after_ms=1200, last_signal_time_ms=2000,
        last_signal_result='signal', cooldown_until_ms=4000,
        position={'entry': 100., 'qty': 1., 'direction': 1, 'stop': 98.},
        position_history=[{'time_ms': 2000, 'signed_qty': 1.}])
    path = tmp_path/'state.json'
    paper.write_json(path, state)
    monkeypatch.setattr(paper, 'stock_config', lambda *_: new)
    monkeypatch.setattr(paper, 'now_ms', lambda: 5000)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
        'stock_rule_revision_scope': 'short_confirmation_only',
        'main_stock_runtime': 'price_led_short_v1'}
    manifest = {}
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    migrated = paper.read_json(path)
    assert {k: v for k, v in migrated.items() if k not in (
        'rule', 'rule_history', 'short_confirmation_active_after_ms')} == {
            k: v for k, v in state.items() if k != 'rule'}
    assert migrated['short_confirmation_active_after_ms'] == 5000
    assert migrated['rule_history'][-1]['scope'] == 'short_confirmation_only'
    assert trial.declared_main_baseline_matches(migrated, old)
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    assert paper.read_json(path) == migrated
    assert len(manifest['rule_revisions']) == 1
    monkeypatch.setattr(paper, 'stock_config', lambda *_: {**new, 'risk_fraction_per_trade': .01})
    with pytest.raises(ValueError, match='cannot alter other signals'):
        paper.synchronize_stock_rules(plan, tmp_path, manifest)
    assert paper.read_json(path) == migrated


def test_experiment_accepts_only_a_recorded_exact_main_short_revision():
    expected = {'a': 1}
    new = {**expected, short.POLICY_KEY: deepcopy(short.POLICY)}
    revision = {'scope': 'short_confirmation_only', 'activated_at_ms': 2000,
                'previous_rule': expected, 'new_rule': new}
    state = {'start_ms': 1000, 'rule': new, 'rule_history': [revision],
             'short_confirmation_active_after_ms': 2000}
    assert trial.declared_main_baseline_matches(state, expected)
    for bad in ({**state, 'rule_history': []},
                {**state, 'rule': {**new, 'a': 2}},
                {**state, 'rule_history': [{**revision, 'previous_rule': {'a': 2}}]},
                {**state, 'rule_history': [{**revision, 'activated_at_ms': 999}]}):
        assert not trial.declared_main_baseline_matches(bad, expected)
