"""Net targets must execute on fresh costs and never reopen a consumed signal."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import simulation_risk_monitor as monitor
import trading_execution as execution

NOW = 1_791_720_000_000
POLICY = {'mode': 'net_margin_target_v1', 'target_margin_return': .6, 'margin_leverage': 10}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for key, value in {'LLM_TRADE_GATE_ENABLED': 'false', 'SIM_MAX_LEVERAGE': '10',
                       'SIM_MAX_NOTIONAL_USDT': '0', 'SIM_TAKER_FEE': '.00045',
                       'SIM_SLIPPAGE_BPS': '1'}.items():
        monkeypatch.setenv(key, value)


def target(side=1, timestamp=NOW, origin=NOW, identifier='first'):
    return {'target_leverage': side * .2, 'signal_time_ms': timestamp,
            'origin_signal_time_ms': origin, 'position_id': identifier}


def report():
    return {'config': {'portfolio_leverage_cap': 10}, 'shadow_profile': {
        'profit_exit_policy': POLICY, 'entry_timing_policy': {
            'mode': 'origin_signal_window_v1', 'max_initial_entry_age_seconds': 3600}}}


def opened(tmp_path, side=1):
    account = execution.SimulationAccount(tmp_path / 'account.json', profit_exit_policy=POLICY)
    state = account.reset(1000, now_ms=NOW-1000)
    state['observed_flat_target'] = True
    account._save(state)
    result = account.reconcile(target(side), 100000, report=report(), now_ms=NOW)
    assert result['fill']
    return account


@pytest.mark.parametrize('side,price', [(1, 107000), (-1, 93000)])
def test_monitor_closes_at_net_target_without_llm_or_metadata(tmp_path, side, price, monkeypatch):
    account = opened(tmp_path, side)
    monkeypatch.setattr(monitor.time, 'time', lambda: (NOW+1000)/1000)
    client = Mock()
    client.mark_price_observation.return_value = {'price': price, 'time_ms': NOW+1000,
                                                'next_funding_time_ms': NOW+100000}
    client.funding_history.return_value = []
    client.symbol_rules.side_effect = AssertionError('Full reduction needs no sizing metadata')
    with patch.object(execution, 'apply_llm_trade_gate', side_effect=AssertionError('No exit review')):
        result = monitor.monitor(client, NOW+1000, account=account, status_path=tmp_path/'risk.json')
    assert account.load()['position_qty'] == 0
    assert result['fill']['reason'] == 'net_margin_target'
    assert result['monitor']['profit_target_fill']
    assert not result['monitor']['hard_stop_fill']
    assert account.load()['last_profit_target_exit']['projected_net_roi_pct'] >= 60
    assert account.load()['wallet_balance'] - 1000 >= account.load()['last_profit_target_exit']['target_net_pnl_usdt']
    client.symbol_rules.assert_not_called()


def test_gross_sixty_percent_is_insufficient_after_costs(tmp_path):
    account = opened(tmp_path)
    entry = account.load()['entry_price']
    result = account.observe(entry*1.06, NOW+1000)
    assert result['fill'] is None
    assert result['profit_exit_status']['projected_net_roi_pct'] < 60
    assert account.load()['position_qty'] > 0


def test_reconcile_profit_close_blocks_refreshed_same_origin_but_allows_new_signal(tmp_path):
    account = opened(tmp_path)
    with patch.object(execution, 'apply_llm_trade_gate', side_effect=AssertionError('No exit review')):
        result = account.reconcile(target(timestamp=NOW+1000), 107000, report=report(), now_ms=NOW+1000)
    assert result['fill']['reason'] == 'net_margin_target'
    assert result['profit_exit_status']['status'] == 'closed_target'
    before = account.load()['fill_count_total']
    blocked = account.reconcile(target(timestamp=NOW+300000), 107000, report=report(), now_ms=NOW+300000)
    assert blocked['fill'] is None
    assert blocked['entry_guard']['status'] == 'profit_target_signal_consumed'
    # A flat report cannot erase the completed signal lock.
    flat = target(timestamp=NOW+600000); flat['target_leverage'] = 0
    account.reconcile(flat, 107000, report=report(), now_ms=NOW+600000)
    blocked = account.reconcile(target(timestamp=NOW+700000), 107000, report=report(), now_ms=NOW+700000)
    assert blocked['fill'] is None
    assert account.load()['fill_count_total'] == before
    new = target(timestamp=NOW+900000, origin=NOW+900000, identifier='second')
    assert account.reconcile(new, 107000, report=report(), now_ms=NOW+900000)['fill']


def test_funding_outage_defers_target_while_trend_close_still_executes(tmp_path):
    account = opened(tmp_path)
    result = account.observe(107000, NOW+1000, funding_available=False)
    assert result['fill'] is None
    assert result['profit_exit_status']['status'] == 'funding_unavailable'
    flat = target(timestamp=NOW+2000); flat['target_leverage'] = 0
    result = account.reconcile(flat, 107000, report=report(), funding_available=False, now_ms=NOW+2000)
    assert result['fill']
    assert result['fill'].get('reason') != 'net_margin_target'


def test_hard_stop_precedes_funding_with_policy_enabled(tmp_path, monkeypatch):
    account = opened(tmp_path)
    monkeypatch.setattr(monitor.time, 'time', lambda: (NOW+1000)/1000)
    client = Mock()
    client.mark_price_observation.return_value = {'price': 10000, 'time_ms': NOW+1000}
    def outage(*args):
        assert account.load()['position_qty'] == 0
        raise OSError('Funding offline')
    client.funding_history.side_effect = outage
    result = monitor.monitor(client, NOW+1000, account=account, status_path=tmp_path/'risk.json')
    assert result['fill']['reason'] == 'account_hard_stop'
    assert result['monitor']['hard_stop_fill']
    assert not result['monitor']['profit_target_fill']


def test_crossed_funding_boundary_waits_for_published_settlement(tmp_path, monkeypatch):
    account = opened(tmp_path)
    elapsed = {'ms': 1000}
    monkeypatch.setattr(monitor.time, 'time', lambda: (NOW+elapsed['ms'])/1000)
    client = Mock()
    client.mark_price_observation.return_value = {'price': 107000, 'time_ms': NOW+1000,
                                                'next_funding_time_ms': NOW+2000}
    def delayed(*args):
        elapsed['ms'] = 3000
        return []
    client.funding_history.side_effect = delayed
    result = monitor.monitor(client, NOW+1000, account=account, status_path=tmp_path/'risk.json')
    assert result['fill'] is None
    assert account.load()['funding_pending_settlement_ms'] == NOW+2000
    assert result['monitor']['status'] == 'degraded'
    assert account.load()['profit_exit_status']['status'] == 'funding_unavailable'
    client.funding_history.side_effect = None
    client.funding_history.return_value = [{'fundingTime': NOW+2000, 'fundingRate': .001,
                                           'markPrice': 107000}]
    client.mark_price_observation.return_value = {'price': 107000, 'time_ms': NOW+3000,
                                                'next_funding_time_ms': NOW+100000}
    recovered = monitor.monitor(client, NOW+3000, account=account, status_path=tmp_path/'risk.json')
    assert recovered['fill']['reason'] == 'net_margin_target'
    assert account.load()['funding_pnl'] < 0


def test_target_archive_failure_reports_persisted_closure(tmp_path, monkeypatch):
    account = opened(tmp_path)
    monkeypatch.setattr(monitor.time, 'time', lambda: (NOW+1000)/1000)
    client = Mock()
    client.mark_price_observation.return_value = {'price': 107000, 'time_ms': NOW+1000,
                                                'next_funding_time_ms': NOW+100000}
    client.funding_history.return_value = []
    original = account._record_fills
    def fail_on_fill(state, rows):
        if any(rows):
            raise OSError('Archive offline')
        original(state, rows)
    account._record_fills = fail_on_fill
    result = monitor.monitor(client, NOW+1000, account=account, status_path=tmp_path/'risk.json')
    assert result['fill']['reason'] == 'net_margin_target'
    assert result['monitor']['profit_target_fill']
    assert 'fill_archive' in result['monitor']['errors']
    assert account.load()['position_qty'] == 0


def test_current_inventory_policy_backfill_preserves_all_cash_and_history(tmp_path):
    account = opened(tmp_path, -1)
    state = account.load()
    state['fills'][0].pop('position_id'); state['fills'][0].pop('origin_signal_time_ms')
    account._save(state)
    # An old fill journal has the same original record, without newly invented metadata.
    account.path.with_suffix('.fills.jsonl').write_text(json.dumps({**state['fills'][0],
        'account_epoch': state['created_at_utc']})+'\n')
    account.path.with_name('last_report.json').write_text(json.dumps({'execution_target': {
        'signed_qty': -.002, 'position_id': 'legacy-hourly', 'origin_signal_time_ms': NOW-1000}}))
    before = {k: state[k] for k in ['wallet_balance','fill_count_total','fees_paid','funding_pnl',
                                   'position_history','fills','entry_price','position_qty']}
    result = account.observe(100000, NOW+1000)
    assert result['profit_exit_status']['ledger']['complete']
    assert result['profit_exit_status']['source_origin']['position_id'] == 'legacy-hourly'
    for k,v in before.items():
        assert account.load()[k] == v


def test_invalid_optional_policy_never_prevents_hard_risk_exit(tmp_path, monkeypatch):
    account = opened(tmp_path)
    account = execution.SimulationAccount(account.path, profit_exit_policy={**POLICY, 'margin_leverage': 0})
    monkeypatch.setattr(monitor.time, 'time', lambda: (NOW+1000)/1000)
    client = Mock()
    client.mark_price_observation.return_value = {'price': 10000, 'time_ms': NOW+1000}
    client.funding_history.return_value = []
    result = monitor.monitor(client, NOW+1000, account=account, status_path=tmp_path/'risk.json')
    assert result['fill']['reason'] == 'account_hard_stop'
    assert account.load()['position_qty'] == 0
    assert account.load()['profit_exit_status']['status'] == 'invalid_policy'


def test_source_can_be_completed_after_first_legacy_observation(tmp_path):
    account = opened(tmp_path)
    state = account.load()
    state['fills'][0].pop('position_id'); state['fills'][0].pop('origin_signal_time_ms')
    account._save(state)
    account.path.with_suffix('.fills.jsonl').write_text(json.dumps({**state['fills'][0],
        'account_epoch': state['created_at_utc']})+'\n')
    account.observe(100000, NOW+1000)
    assert not account.load()['profit_target_origin']['position_id']
    result = account.reconcile(target(timestamp=NOW+2000), 107000, report=report(), now_ms=NOW+2000)
    assert result['fill']['reason'] == 'net_margin_target'
    assert account.load()['profit_target_consumed_signals'][-1]['position_id'] == 'first'
    assert account.reconcile(target(timestamp=NOW+300000), 107000, report=report(), now_ms=NOW+300000)['fill'] is None


def test_delayed_funding_target_archive_failure_recovers_actual_timestamp(tmp_path, monkeypatch):
    account = opened(tmp_path)
    elapsed = {'ms': 1000}
    monkeypatch.setattr(monitor.time, 'time', lambda: (NOW+elapsed['ms'])/1000)
    client = Mock()
    client.mark_price_observation.return_value = {'price': 107000, 'time_ms': NOW+1000,
                                                'next_funding_time_ms': NOW+100000}
    def delayed(*args):
        elapsed['ms'] = 3000
        return []
    client.funding_history.side_effect = delayed
    original = account._record_fills
    def fail_on_fill(state, rows):
        if any(rows):
            raise OSError('Archive offline')
        original(state, rows)
    account._record_fills = fail_on_fill
    result = monitor.monitor(client, NOW+1000, account=account, status_path=tmp_path/'risk.json')
    assert result['fill']['reason'] == 'net_margin_target'
    assert result['monitor']['profit_target_fill']
    assert 'fill_archive' in result['monitor']['errors']


def test_final_ledger_read_cannot_fill_using_expired_mark(tmp_path, monkeypatch):
    account = opened(tmp_path)
    elapsed = {'seconds': 0}
    monkeypatch.setattr(execution.time, 'monotonic', lambda: elapsed['seconds'])
    original = execution.btc_profit_target.evaluate_profit_target
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        elapsed['seconds'] = 61
        return result
    monkeypatch.setattr(execution.btc_profit_target, 'evaluate_profit_target', delayed)
    before = account.path.read_bytes()
    with pytest.raises((RuntimeError, ValueError), match='stale'):
        account.observe(107000, NOW+1000, funding_fetch_through_ms=NOW+1000,
                        next_funding_time_ms=NOW+100000, mark_time_ms=NOW+1000)
    assert account.path.read_bytes() == before
