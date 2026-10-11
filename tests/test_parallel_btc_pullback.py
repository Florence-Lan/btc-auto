"""Pending BTC entries retry on quotes; completed inventory keeps bar cadence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import market_data_runtime as market
import btc_pullback_entry as pullback
import run_parallel_simulation as runner


HOUR = 3_600_000
FIVE = 300_000
ORIGIN = 100 * HOUR
POLICY = {'mode': 'structural_pullback_v1'}


class History:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.missing = set()

    def public_get(self, path, params):
        self.calls.append((path, dict(params)))
        assert path == '/fapi/v1/klines'  # Funding belongs to execution's own cutoff.
        if self.failure:
            raise self.failure
        step = HOUR if params['interval'] == '1h' else FIVE
        rows = []
        first = (params['startTime'] + step - 1) // step * step
        for timestamp in range(first, params['endTime'] + 1, step):
            if (params['interval'], timestamp) in self.missing:
                continue
            if timestamp + step - 1 > params['endTime']:
                # A forming bar's OHLC/volume must never enter the report.
                rows.append([timestamp, '100', 'bad high', 'bad low', 'bad close',
                             'bad volume', 'bad close time', 'bad quote volume'])
            else:
                rows.append([timestamp, '100', '102', '98', '101', '10',
                             timestamp + step - 1, '1000', 1, '0', '0', '0'])
        return rows[:params['limit']]


def test_pullback_input_is_original_hourly_prefix_and_only_closed_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    client = History()
    cutoff = ORIGIN + FIVE + 10_000
    data = runner.btc_pullback_market_data(client, {'origin_signal_time_ms': ORIGIN}, cutoff)
    assert data['complete']
    assert len(data['hourly']) == 49
    assert data['hourly'][0][0] == ORIGIN - 49 * HOUR
    assert data['hourly'][-1][7] == ORIGIN - 1
    assert len(data['five_minute']) == 1
    assert data['five_minute'][0][0] == ORIGIN
    assert all(row[7] <= cutoff for key in ['hourly', 'five_minute'] for row in data[key])
    # The cache can carry forming opening ticks, but the injected report cannot.
    assert all(row[0] < ORIGIN for row in data['hourly'])
    assert all(row[0] < ORIGIN + FIVE for row in data['five_minute'])
    client.calls.clear()
    again = runner.btc_pullback_market_data(client, {'origin_signal_time_ms': ORIGIN}, cutoff + 1000)
    assert again['hourly'] == data['hourly'] and again['five_minute'] == data['five_minute']
    assert not client.calls


def test_first_signal_minutes_need_no_unfinished_confirmation_candle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    data = runner.btc_pullback_market_data(History(), {'origin_signal_time_ms': ORIGIN}, ORIGIN + 1000)
    assert data['complete'] and data['five_minute'] == []


@pytest.mark.parametrize('policy,expected_hours', [
    ({'mode': 'structural_pullback_v1', 'lookback_hours': 72}, 72),
    ({'mode': 'structural_pullback_v1', 'lookback_hours': 97, 'atr_period': 96}, 97),
])
def test_valid_longer_policy_loads_the_complete_original_indicator_prefix(
    tmp_path, monkeypatch, policy, expected_hours,
):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    client = History()
    point = {'signed_qty': .002, 'position_id': 'old-hourly',
             'origin_signal_time_ms': ORIGIN, 'origin_entry_price': 100000}
    # A later decision still anchors its larger prefix to the original source.
    cutoff = ORIGIN + 2 * FIVE + 1000
    data = runner.btc_pullback_market_data(client, point, cutoff, policy)
    assert data['complete'] and len(data['hourly']) == expected_hours
    assert data['hourly'][0][0] == ORIGIN - expected_hours * HOUR
    assert data['hourly'][-1][7] == ORIGIN - 1
    assert len(data['five_minute']) == 2
    pending = pullback.build_pending(point, data['hourly'], policy, cutoff)
    # This synthetic market has no favorable structure, but history is usable.
    assert pending['status'] == 'no_structure'
    assert 'btc_pullback_hourly_history_insufficient' not in pending['reasons']
    assert pending['atr'] == 4
    client.calls.clear()
    runner.btc_pullback_market_data(client, point, cutoff + 30_000, policy)
    assert not client.calls


def test_missing_closed_history_is_explicit_and_recovers_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    client = History()
    client.missing.add(('1h', ORIGIN - 2 * HOUR))
    point = {'origin_signal_time_ms': ORIGIN}
    missing = runner.btc_pullback_market_data(client, point, ORIGIN + FIVE + 1000)
    assert not missing['complete']
    assert missing['reason'] == 'btc_pullback_data_unavailable'
    assert 'incomplete' in missing['missing_reason']
    client.missing.clear()
    assert runner.btc_pullback_market_data(client, point, ORIGIN + FIVE + 1000)['complete']
    client.failure = ConnectionError('Offline')
    assert not runner.btc_pullback_market_data(client, point, ORIGIN + 2 * FIVE + 1000)['complete']


@pytest.mark.parametrize('origin', [None, True, ORIGIN + 1, ORIGIN + HOUR])
def test_bad_source_never_requests_or_uses_future_hourly_history(tmp_path, monkeypatch, origin):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    client = History()
    result = runner.btc_pullback_market_data(client, {'origin_signal_time_ms': origin}, ORIGIN + 1000)
    assert not result['complete'] and not client.calls


@pytest.fixture
def running(tmp_path, monkeypatch):
    clock = {'time_ms': ORIGIN + FIVE + 1000, 'source': 'exchange'}
    point = {'time_ms': ORIGIN, 'available_time_ms': ORIGIN + FIVE - 1,
             'signed_qty': .002, 'equity': 1000, 'price': 100000,
             'origin_signal_time_ms': ORIGIN, 'origin_entry_price': 100000,
             'position_id': 'hourly-first'}
    report = {'execution_target': point}
    profile = {'candidate_id': 'fixture', 'btc_pullback_entry_policy': POLICY,
               'entry_timing_policy': {'mode': 'origin_signal_window_v1',
                                       'max_initial_entry_age_seconds': 3600}}
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report))
    profile_path = tmp_path / 'profile.json'
    profile_path.write_text(json.dumps(profile))
    state = {'last_signal_time_ms': ORIGIN, 'last_mark_price': 100000,
             'entry_price': 0, 'wallet_balance': 1000, 'position_qty': 0,
             'fill_count_total': 0, 'fees_paid': 0, 'funding_pnl': 0,
             'max_drawdown_pct': 0, 'btc_pullback_pending': {'status': 'waiting_retrace'},
             'btc_pullback_entry': {'enabled': True, 'allowed': False,
                                    'status': 'waiting_retrace', 'reasons': []}}
    execution = Mock()
    execution.load.side_effect = lambda: state
    execution.snapshot.return_value = {'account': {'margin_balance': 1000, 'realized_return_pct': 0}}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SimulationAccount', Mock(return_value=execution))
    monkeypatch.setattr(runner.decision_runtime, 'resolve_clock', Mock(side_effect=lambda client: dict(clock)))
    monkeypatch.setattr(runner.simulation_risk_monitor, 'monitor', Mock(return_value={
        'monitor': {'status': 'healthy', 'errors': {}}}))
    monkeypatch.setattr(runner.execution_entry_gate, 'decision_at', lambda report, timestamp, side: {
        'allowed': True, 'status': 'allowed', 'checked_at_ms': timestamp, 'reasons': []})
    monkeypatch.setattr(runner, 'now_ms', lambda: clock['time_ms'])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 0)
    executor = Mock(return_value={'mode': 'simulation'})
    monkeypatch.setattr(runner, 'execute_report', executor)
    loader = Mock(return_value={'hourly': [], 'five_minute': [], 'asof_ms': clock['time_ms'],
                              'complete': True, 'diagnostics': {}})
    monkeypatch.setattr(runner, 'btc_pullback_market_data', loader)
    account = {'account_id': 'btc', 'state_path': 'btc/state.json',
               'strategy_path': 'profile.json', 'strategy_report_path': 'report.json'}
    step = lambda: runner.btc_step(account, tmp_path, {'start_ms': ORIGIN - HOUR}, Mock())
    return step, state, report, report_path, profile, profile_path, clock, executor, loader


@pytest.mark.parametrize('status', ['waiting_retrace', 'waiting_confirmation', 'ready', 'data_unavailable'])
def test_pending_direction_retries_same_five_minute_report(running, status):
    step, state, _, _, _, _, clock, executor, loader = running
    state['btc_pullback_pending']['status'] = status
    state['btc_pullback_entry']['status'] = status
    step()
    clock['time_ms'] += 30_000
    step()
    assert executor.call_count == 2 and loader.call_count == 2
    sent = executor.call_args.args[1]
    assert sent['shadow_profile']['btc_pullback_entry_policy'] == POLICY
    assert sent['btc_pullback_market_data']['complete']
    assert loader.call_args.args[3] == POLICY


@pytest.mark.parametrize('policy', ['absent', None])
def test_unconfigured_profile_never_loads_or_retries_extra_history(running, policy):
    step, _, _, _, profile, profile_path, _, executor, loader = running
    if policy == 'absent':
        profile.pop('btc_pullback_entry_policy')
    else:
        profile['btc_pullback_entry_policy'] = None
    profile_path.write_text(json.dumps(profile))
    step()
    executor.assert_not_called()
    loader.assert_not_called()


def test_disabled_selected_profile_overrides_a_stale_producer_policy(running):
    step, _, report, report_path, profile, profile_path, _, executor, loader = running
    profile.pop('btc_pullback_entry_policy')
    profile_path.write_text(json.dumps(profile))
    report['shadow_profile'] = {'btc_pullback_entry_policy': POLICY}
    report['execution_target']['time_ms'] += FIVE
    report_path.write_text(json.dumps(report))
    step()
    assert executor.call_count == 1
    assert 'btc_pullback_entry_policy' not in executor.call_args.args[1]['shadow_profile']
    loader.assert_not_called()


@pytest.mark.parametrize('status', ['cancelled', 'expired', 'no_structure', 'structure_invalidated', 'invalid_source'])
def test_cancelled_source_does_not_restart_from_refresh_status(running, status):
    step, state, _, _, _, _, _, executor, loader = running
    state['btc_pullback_pending']['status'] = status
    state['btc_pullback_entry']['status'] = 'data_unavailable'
    step()
    executor.assert_not_called()
    loader.assert_not_called()


def test_existing_same_direction_position_stays_on_five_minute_cadence(running):
    step, state, report, report_path, _, _, _, executor, loader = running
    state['position_qty'] = .002
    state['entry_price'] = 99500
    view = step()
    executor.assert_not_called()
    loader.assert_not_called()
    assert view['entry_price'] == 99500
    report['execution_target']['time_ms'] += FIVE
    report_path.write_text(json.dumps(report))
    step()
    assert executor.call_count == 1
    loader.assert_not_called()


@pytest.mark.parametrize('signed_qty', [0, -.002])
def test_changed_source_reaches_executor_for_cancellation_or_reversal(running, signed_qty):
    step, _, report, report_path, _, _, _, executor, loader = running
    report['execution_target']['signed_qty'] = signed_qty
    report_path.write_text(json.dumps(report))
    step()
    assert executor.call_count == 1
    if signed_qty == 0:
        loader.assert_not_called()


def test_missing_entry_data_degrades_without_suppressing_execution(running):
    step, state, _, _, _, _, _, executor, loader = running
    before = deepcopy(state)
    loader.return_value = {'hourly': [], 'five_minute': [], 'asof_ms': ORIGIN,
                           'complete': False, 'reason': 'btc_pullback_data_unavailable',
                           'missing_reason': 'No original completed hour'}
    view = step()
    assert executor.call_count == 1
    assert not executor.call_args.args[1]['btc_pullback_market_data']['complete']
    assert view['status'] == 'degraded'
    assert view['signal_status'] == 'entry_data_unavailable'
    assert 'btc_pullback_data_unavailable' in view['entry_blockers']
    assert state == before  # Only the execution owner may mutate the account.


def test_expired_wait_reaches_executor_without_new_history_request(running):
    step, state, report, report_path, _, _, clock, executor, loader = running
    report['execution_target']['time_ms'] = ORIGIN + HOUR - FIVE
    report['execution_target']['available_time_ms'] = ORIGIN + HOUR - 1
    state['last_signal_time_ms'] = report['execution_target']['time_ms']
    report_path.write_text(json.dumps(report))
    clock['time_ms'] = ORIGIN + HOUR
    step()
    assert executor.call_count == 1
    loader.assert_not_called()


def test_a_new_source_can_replace_terminal_wait_even_at_same_report_time(running):
    step, state, _, _, _, _, _, executor, loader = running
    state['btc_pullback_pending'].update(status='cancelled', source_key=f'old:{ORIGIN}:long')
    state['btc_pullback_entry']['status'] = 'cancelled'
    step()
    assert executor.call_count == 1 and loader.call_count == 1


def test_waiting_status_and_cash_average_are_published_from_execution_ledger(running):
    step, state, _, _, _, _, _, _, _ = running
    state['btc_pullback_entry'].update(candidate_low=99000, candidate_high=99200)
    view = step()
    assert view['btc_pullback_entry'] == state['btc_pullback_entry']
    assert view['status'] == 'healthy'
