"""Real-book and rule-history coverage of the main-only entry-period overlay."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import main_stock_entry_trend as entry
import main_stock_rule_revision as revisions
import main_stock_runtime as runtime
import main_stock_trend as trend
import run_parallel_simulation as paper
import stock_swing_signals as signals
import trade_profit_budget as budget
from test_main_rule_revision import fixture as original_history, with_budget
from test_main_stock_runtime import HOUR, unavailable
from test_main_stock_trend_integration import worker as original_worker, sector


def worker(tmp_path, monkeypatch, *, activation=599 * HOUR, direction=-1):
    original, clock, bars = original_worker(tmp_path, monkeypatch,
                                           activation=598 * HOUR, direction=direction)
    cfg = {**original.config, entry.POLICY_KEY: deepcopy(entry.POLICY)}
    state = paper.read_json(original.path)
    state.update(rule=cfg, entry_trend_active_after_ms=activation)
    paper.write_json(original.path, state)
    account = runtime.MainStockAccount(original.path, original.venue, original.symbol, cfg)
    monkeypatch.setattr(account, 'shared_clock_loader', unavailable)
    monkeypatch.setattr(account, 'sector_loader', lambda *_a, **_k: sector(clock['sector_direction']))
    return account, clock, bars


@pytest.mark.parametrize('side', [-1, 1])
def test_new_policy_books_real_supported_quantity_under_original_budget(tmp_path, monkeypatch, side):
    account, clock, bars = worker(tmp_path, monkeypatch, direction=side)
    if side == 1:
        bars[:] = [replace(bar, open=300-bar.open, high=300-bar.low,
                           low=300-bar.high, close=300-bar.close) for bar in bars]
        clock['price'] = bars[-1].close
    result = account.step()
    assert result['position']['direction'] == side and result['fill_count_total'] == 1
    assert result['entry_profit_budget']['allowed']
    assert result['entry_profit_budget']['actual_fill_checked']
    assert result['entry_profit_budget']['expected_net_profit_usdt'] >= 100
    assert result['entry_profit_budget']['planned_loss_usdt'] <= 25
    assert result['entry_trend_execution']['allowed']
    assert result['entry_trend_execution']['phase'] == 'fill'
    assert result['entry_trend_confirmation']['direction'] == side
    assert result['entry_trend_confirmation']['admitted_by_timeframes']
    assert result['entry_trend_runtime']['active_after_ms'] == 599 * HOUR
    assert result['current_trend']['two_hour']['closed_at_ms'] == 600 * HOUR
    assert result['main_trend_active_after_ms'] == 598 * HOUR


@pytest.mark.parametrize('activation', [600 * HOUR, 600 * HOUR + 1])
def test_policy_activation_cannot_reopen_the_already_closed_signal(tmp_path, monkeypatch, activation):
    account, clock, _ = worker(tmp_path, monkeypatch, activation=activation)
    first = account.step()
    assert first['position'] is None and first['fill_count_total'] == 0
    assert first['signal_diagnostics']['admission']['stage'] == 'entry_trend_not_active'
    clock['now'] += 30_000
    again = account.step()
    assert again['position'] is None and again['fill_count_total'] == 0


def test_private_dispatcher_requires_new_boundary_without_mutating_profile(tmp_path, monkeypatch):
    account, _, bars = worker(tmp_path, monkeypatch)
    cfg = deepcopy(account.config)
    indicators = signals.compute_indicators(bars, cfg)
    boundary = bars[-1].time_ms + HOUR
    dispatcher = runtime._SignalProfiles(597 * HOUR, 598 * HOUR, boundary)
    signal, diagnostic = dispatcher.signal_with_diagnostics(bars, indicators, len(bars)-1, cfg)
    assert signal is None and diagnostic['admission']['reasons'] == ['entry_trend_awaiting_new_signal']
    assert account.config == cfg
    dispatcher = runtime._SignalProfiles(597 * HOUR, 598 * HOUR, boundary-1)
    signal, diagnostic = dispatcher.signal_with_diagnostics(bars, indicators, len(bars)-1, cfg)
    assert signal is not None
    assert diagnostic['entry_trend_confirmation_runtime']['candidate_active_for_bar']


def test_final_fill_admission_checks_a_newly_closed_hour_again(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    original = account.entry_fill_admission
    reached = []
    def cross_hour(*args, **kwargs):
        reached.append(True)
        clock['now'] = 601 * HOUR + 1000
        return original(*args, **kwargs)
    monkeypatch.setattr(account, 'entry_fill_admission', cross_hour)
    result = account.step()
    assert reached, 'Must reach the actual fill-admission gate.'
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['wallet_balance'] == 1000 and result['fees_paid'] == 0
    assert not result['entry_trend_execution']['allowed']
    assert result['entry_trend_execution']['phase'] == 'fill_admission'


def test_last_commit_gate_can_veto_after_real_budget_has_passed(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    original = account._entry_trend_decision
    phases = []
    def lose_data_at_commit(*args, **kwargs):
        phases.append(kwargs['phase'])
        if kwargs['phase'] == 'fill':
            account.signal_bars = []
        return original(*args, **kwargs)
    monkeypatch.setattr(account, '_entry_trend_decision', lose_data_at_commit)
    result = account.step()
    assert {'retry', 'fill_admission', 'fill'} <= set(phases)
    assert result['entry_profit_budget']['allowed']
    assert result['entry_profit_budget']['actual_fill_checked']
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['wallet_balance'] == 1000 and result['fees_paid'] == 0
    assert not result['entry_trend_execution']['allowed']


def test_original_protective_exit_does_not_wait_for_new_entry_context(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    entered = account.step()
    assert entered['position'] is not None
    clock.update(price=entered['position']['stop'] + 1, hourly_error=True, sector_direction=None)
    monkeypatch.setattr(account, '_entry_trend_decision',
        lambda *_a, **_k: pytest.fail('An exit must not depend on entry confirmation'))
    closed = account.step()
    assert closed['position'] is None and closed['fill_count_total'] == 2
    assert closed['fills'][-1]['reason'] == 'protective_stop'


def test_running_account_rejects_silent_entry_activation_replacement(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    state = paper.read_json(account.path)
    state['entry_trend_active_after_ms'] += 1
    paper.write_json(account.path, state)
    with pytest.raises(ValueError, match='activation_changed'):
        account.step()


def test_exact_migration_preserves_old_clocks_consumed_signal_inventory_and_finances(tmp_path, monkeypatch):
    old = paper.stock_config({'stock_research_profile': 'config/stock_main_trend_20261102.json'}, 'MUUSDT')
    new = {**old, entry.POLICY_KEY: deepcopy(entry.POLICY)}
    state = paper.initial_stock('MUUSDT', 1000, old)
    state.update(wallet_balance=997., equity=996., fees_paid=1., realized_pnl=-2., signal_timeframe='1h',
        main_trend_active_after_ms=4000, short_confirmation_active_after_ms=2000,
        trade_profit_budget_active_after_ms=3000, signal_active_after_ms=4000,
        last_signal_time_ms=4500, pending_signal={'boundary_ms': 4500},
        position={'entry': 100., 'qty': 1., 'direction': -1, 'stop': 110.},
        position_history=[{'time_ms': 3500, 'signed_qty': -1.}])
    path = tmp_path/'mu/state.json'
    paper.write_json(path, state)
    monkeypatch.setattr(paper, 'stock_config', lambda *_: new)
    monkeypatch.setattr(paper, 'now_ms', lambda: 5000)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
            'stock_rule_revision_scope': 'entry_trend_only', 'main_stock_runtime': 'price_led_short_v1'}
    manifest = {}
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    after = paper.read_json(path)
    assert after['entry_trend_active_after_ms'] == 5000
    assert after['pending_signal'] is None
    assert after['rule'] == new and after['rule_history'][-1]['scope'] == 'entry_trend_only'
    allowed = {'rule', 'rule_history', 'entry_trend_active_after_ms', 'pending_signal',
               'last_signal_result', 'signal_status'}
    assert {k: v for k, v in after.items() if k not in allowed} == {
        k: v for k, v in state.items() if k not in allowed}
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    assert paper.read_json(path) == after
    assert len(manifest['rule_revisions']) == 1


def test_migration_cannot_hide_a_risk_budget_change(tmp_path, monkeypatch):
    old = paper.stock_config({'stock_research_profile': 'config/stock_main_trend_20261102.json'}, 'MUUSDT')
    state = paper.initial_stock('MUUSDT', 1000, old)
    state['main_trend_active_after_ms'] = 4000
    path = tmp_path/'mu/state.json'
    paper.write_json(path, state)
    before = path.read_bytes()
    changed = {**old, entry.POLICY_KEY: deepcopy(entry.POLICY), 'risk_fraction_per_trade': .5}
    monkeypatch.setattr(paper, 'stock_config', lambda *_: changed)
    monkeypatch.setattr(paper, 'now_ms', lambda: 5000)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
            'stock_rule_revision_scope': 'entry_trend_only', 'main_stock_runtime': 'price_led_short_v1'}
    with pytest.raises(ValueError, match='cannot alter'):
        paper.synchronize_stock_rules(plan, tmp_path, {})
    assert path.read_bytes() == before


def history():
    state, expected = original_history()
    with_budget(state, budget.POLICY)
    previous = deepcopy(state['rule'])
    current = {**previous, trend.POLICY_KEY: deepcopy(trend.POLICY)}
    state.update(rule=current, main_trend_active_after_ms=4000, signal_active_after_ms=4000)
    state['rule_history'].append({'scope': 'main_trend_only', 'activated_at_ms': 4000,
        'previous_rule': previous, 'new_rule': deepcopy(current)})
    previous = deepcopy(current)
    current = {**previous, entry.POLICY_KEY: deepcopy(entry.POLICY)}
    state.update(rule=current, entry_trend_active_after_ms=5000)
    state['rule_history'].append({'scope': 'entry_trend_only', 'activated_at_ms': 5000,
        'previous_rule': previous, 'new_rule': deepcopy(current)})
    return state, expected


def test_baseline_compatibility_checks_whole_bound_chain_without_mutation():
    state, expected = history()
    before = deepcopy((state, expected))
    assert revisions.declared_main_baseline_matches(state, expected)
    assert (state, expected) == before


@pytest.mark.parametrize('mutation', [
    lambda s: s['rule'].update(risk_fraction_per_trade=.5),
    lambda s: s['rule'][entry.POLICY_KEY].update(warmup_bars=60.),
    lambda s: s.update(entry_trend_active_after_ms=True),
    lambda s: s.update(entry_trend_active_after_ms=4000),
    lambda s: s.update(main_trend_active_after_ms=3999),
    lambda s: s.update(signal_active_after_ms=5000),
    lambda s: s['rule_history'][-1].update(scope='profit_exits_only'),
    lambda s: s['rule_history'][-1].update(activated_at_ms=4999),
    lambda s: s['rule_history'][-1]['previous_rule'].update(risk_fraction_per_trade=.5),
    lambda s: s['rule_history'].append(deepcopy(s['rule_history'][-1])),
])
def test_baseline_rejects_unbound_policy_hidden_risk_or_replaced_old_clock(mutation):
    state, expected = history()
    mutation(state)
    assert not revisions.declared_main_baseline_matches(state, expected)
