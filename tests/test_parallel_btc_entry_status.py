from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as runner


@pytest.mark.parametrize('gate_status,expected', [('rejected', 'llm_trade_gate_rejected'),
    ('cached_rejected', 'llm_trade_gate_rejected'), ('error_blocked', 'llm_trade_gate_unavailable')])
def test_model_rejection_is_visible_for_current_entry(btc_status_fixture, gate_status, expected):
    step, _, state, report, path, _, _, reconcile = btc_status_fixture
    report['execution_target'].update(signed_qty=-1, position_id='current')
    path.write_text(json.dumps(report), encoding='utf-8')
    state['llm_trade_gate'] = {'enabled': True, 'status': gate_status,
        'requested_target_leverage': -.2, 'decision_key': 'current:short',
        'allow': False, 'reason': 'Current market conflict'}
    result = step()
    assert result['entry_blockers'] == [expected]
    assert result['signal_status'] == ('entry_data_unavailable' if gate_status == 'error_blocked' else 'entry_blocked')
    assert result['llm_trade_gate'] == state['llm_trade_gate']
    reconcile.assert_not_called()


@pytest.mark.parametrize('point,held,requested', [
    ({'signed_qty': 0, 'position_id': 'current'}, -.2, -.2),
    ({'signed_qty': -1, 'position_id': 'next'}, 0, -.2),
    ({'signed_qty': -1, 'position_id': 'current'}, -3, -.2),
    ({'signed_qty': -1, 'position_id': 'current'}, -2, -.2),
    ({'signed_qty': 1, 'position_id': 'current'}, 0, -.2),
])
def test_old_model_rejection_does_not_label_exits_holds_or_different_signals_blocked(point, held, requested):
    state = {'position_qty': held, 'last_mark_price': 100,
        'llm_trade_gate': {'enabled': True, 'status': 'rejected',
            'requested_target_leverage': requested, 'decision_key': 'current:short'}}
    assert runner.btc_llm_entry_blocker(point, state, 1000) is None


@pytest.fixture
def btc_status_fixture(tmp_path, monkeypatch):
    timestamp = 1_800_000
    elapsed = {'seconds': 0.0}
    account = {'account_id': 'btc', 'state_path': 'btc/state.json',
               'strategy_path': 'profile.json', 'strategy_report_path': 'report.json',
               'entry_qualification': {'approved_for_forward_simulation': True}}
    event_path = tmp_path / 'events.json'
    event_path.write_text(json.dumps({'schema_version': 1, 'events': []}), encoding='utf-8')
    report = {'generated_at_utc': '1970-01-01T00:30:00+00:00',
              'execution_target': {'time_ms': timestamp - 300_000,
                  'available_time_ms': timestamp - 1, 'signed_qty': 0},
              'event_overlay': {}, 'execution_entry_context': {'event_snapshot': str(event_path)}}
    (tmp_path / 'profile.json').write_text('{"candidate_id": "btc_fixture"}', encoding='utf-8')
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    state = {'last_signal_time_ms': report['execution_target']['time_ms'],
             'last_mark_price': 100, 'wallet_balance': 1000, 'position_qty': 0,
             'fill_count_total': 0, 'fees_paid': 0, 'funding_pnl': 0, 'max_drawdown_pct': 0,
             'execution_entry_gate': {'allowed': True, 'checked_at_ms': timestamp - 600_000}}
    execution = Mock()
    execution.load.side_effect = lambda: state
    execution.snapshot.return_value = {'account': {'margin_balance': 1000, 'realized_return_pct': 0}}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SimulationAccount', Mock(return_value=execution))
    monkeypatch.setattr(runner.decision_runtime, 'resolve_clock', Mock(return_value={
        'time_ms': timestamp, 'source': 'exchange'}))
    monkeypatch.setattr(runner.simulation_risk_monitor, 'monitor', Mock(return_value={
        'monitor': {'status': 'healthy', 'errors': {}}}))
    monkeypatch.setattr(runner, 'now_ms', lambda: timestamp)
    monkeypatch.setattr(runner.time, 'monotonic', lambda: elapsed['seconds'])
    reconcile = Mock(return_value={'mode': 'simulation'})
    monkeypatch.setattr(runner, 'execute_report', reconcile)
    step = lambda: runner.btc_step(account, tmp_path, {'start_ms': 1000}, Mock())
    return step, event_path, state, report, report_path, elapsed, account, reconcile


def test_current_snapshot_outage_recovery_and_recurrence_without_new_candle(btc_status_fixture):
    step, event_path, state, _, _, _, _, reconcile = btc_status_fixture
    original = event_path.read_text(encoding='utf-8')
    event_path.unlink()
    blocked = step()
    assert blocked['status'] == 'degraded'
    assert blocked['signal_status'] == 'entry_data_unavailable'
    assert blocked['entry_blockers'] == ['entry_context_unavailable:FileNotFoundError']
    assert blocked['execution_entry_gate']['allowed'] is False
    assert blocked['execution_entry_gate']['checked_at_ms'] == 1_800_000
    assert set(blocked['execution_entry_gate']['by_side']) == {'long', 'short'}
    # These checks must not rewrite execution/account history to refresh a card.
    assert state['execution_entry_gate']['allowed'] is True
    event_path.write_text(original, encoding='utf-8')
    recovered = step()
    assert recovered['status'] == 'healthy'
    assert recovered['signal_status'] == 'no_signal'
    assert recovered['entry_blockers'] == []
    assert recovered['execution_entry_gate']['allowed'] is True
    event_path.write_text('{broken', encoding='utf-8')
    again = step()
    assert again['status'] == 'degraded'
    assert again['entry_blockers'] == ['entry_context_unavailable:JSONDecodeError']
    reconcile.assert_not_called()


@pytest.mark.parametrize('quantity,allowed', [(0, True), (1, False), (-1, True)])
def test_flat_target_checks_both_sides_and_nonflat_target_checks_intended_side(monkeypatch, quantity, allowed):
    def decision(report, timestamp, side):
        passed = side == 'short'
        return {'allowed': passed, 'checked_at_ms': timestamp,
                'reasons': [] if passed else ['current_factor:direction_conflict']}
    evaluate = Mock(side_effect=decision)
    monkeypatch.setattr(runner.execution_entry_gate, 'decision_at', evaluate)
    view = runner.btc_entry_view({}, {'time_ms': 1000, 'signed_qty': quantity}, 2000, 'exchange')
    assert view['execution_entry_gate']['allowed'] is allowed
    assert view['entry_blockers'] == ([] if allowed else ['current_factor:direction_conflict'])
    expected = {'long', 'short'} if not quantity else {'long' if quantity > 0 else 'short'}
    assert set(view['execution_entry_gate']['by_side']) == expected
    assert evaluate.call_count == len(expected)


def test_missing_factors_are_visible_even_without_entry_signal(btc_status_fixture, monkeypatch):
    step, _, _, _, _, _, _, _ = btc_status_fixture
    monkeypatch.setattr(runner.execution_entry_gate, 'decision_at', lambda report, timestamp, side: {
        'allowed': False, 'checked_at_ms': timestamp,
        'reasons': ['current_factor:missing_or_stale_factors'],
        'factor': {'coverage': 0.85, 'missing_groups': ['global_risk']}})
    result = step()
    assert result['status'] == 'degraded'
    assert result['signal_status'] == 'entry_data_unavailable'
    assert result['entry_blockers'] == ['current_factor:missing_or_stale_factors']
    assert result['execution_entry_gate']['by_side']['short']['factor']['missing_groups'] == ['global_risk']


def test_entry_diagnostics_never_prevent_exit_reconciliation(btc_status_fixture):
    step, event_path, state, report, _, _, _, reconcile = btc_status_fixture
    state['last_signal_time_ms'] = 1000
    state['position_qty'] = 1
    event_path.unlink()
    result = step()
    reconcile.assert_called_once()
    assert reconcile.call_args.args[0] == 'simulation'
    assert reconcile.call_args.args[1]['execution_target']['signed_qty'] == 0
    assert result['status'] == 'degraded'


def test_entry_gate_accounts_for_elapsed_time_crossing_event_boundary(btc_status_fixture, monkeypatch):
    step, event_path, _, _, _, elapsed, _, _ = btc_status_fixture
    event_path.write_text(json.dumps({'schema_version': 1, 'events': [{
        'event_id': 'boundary', 'published_at_utc': '1970-01-01T00:29:00Z',
        'starts_at_utc': '1970-01-01T00:30:04Z', 'ends_at_utc': '1970-01-01T00:31:00Z',
        'severity': 1, 'block_entries': True}]}), encoding='utf-8')
    def delayed_monitor(*args, **kwargs):
        elapsed['seconds'] = 5
        return {'monitor': {'status': 'healthy', 'errors': {}}}
    monkeypatch.setattr(runner.simulation_risk_monitor, 'monitor', delayed_monitor)
    result = step()
    assert result['execution_entry_gate']['checked_at_ms'] == 1_805_000
    assert result['entry_blockers'] == ['current_event_blocks_entries']
    assert result['signal_status'] == 'entry_blocked'
    assert result['status'] == 'healthy'
    assert result['entry_source_status']['signal_age_ms'] == 305_000


def test_source_freshness_uses_current_clock_and_retains_report_timestamp(btc_status_fixture):
    step, _, state, report, report_path, _, _, _ = btc_status_fixture
    report['execution_target']['time_ms'] = 1
    state['last_signal_time_ms'] = 1
    report_path.write_text(json.dumps(report), encoding='utf-8')
    result = step()
    source = result['entry_source_status']
    assert result['status'] == 'degraded'
    assert source['clock_source'] == 'exchange'
    assert source['signal_stale'] is True
    assert source['signal_age_ms'] == 1_799_999
    assert source['signal_available_time_ms'] == 1_799_999
    assert source['report_generated_at_utc'] == report['generated_at_utc']


def test_qualification_failure_preserves_existing_signal_label(btc_status_fixture):
    step, _, _, _, _, _, account, _ = btc_status_fixture
    account['entry_qualification']['approved_for_forward_simulation'] = False
    result = step()
    assert result['signal_status'] == 'strategy_not_qualified'
    assert result['entry_blockers'] == ['strategy_not_qualified']


def test_factor_source_dates_refresh_from_loaded_snapshot_without_changing_risk_decision(tmp_path):
    factors = runner.execution_entry_gate.multifactor
    day = factors.DAY
    timestamp = 20 * day
    profile = {'candidate_id': 'source_diagnostic', 'live_orders_allowed': False,
        'groups': {'btc_momentum': 1}, 'minimum_group_coverage': 1,
        'minimum_risk_multiplier': .35, 'block_alignment_below': -.35,
        'block_stress_at': .9, 'availability_mode': 'first_seen'}
    profile_path = tmp_path / 'factor_profile.json'
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    factor_path = tmp_path / 'factors.json'
    payload = {'schema_version': 1, 'series': {'btc_close': [
        [stamp, stamp, 100, stamp] for stamp in [13 * day, 19 * day, timestamp]],
        'oil': [[12 * day, 12 * day, 100, 12 * day]]},
        'metadata': {'source_status': {'oil': {'ok': True, 'status': 'ok',
            'latest_observed_at_ms': 12 * day, 'next_retry_at_ms': timestamp + 1000}}}}
    factor_path.write_text(json.dumps(payload), encoding='utf-8')
    report = {'multifactor_overlay': {}, 'execution_entry_context': {
        'factor_profile': str(profile_path), 'factor_profile_sha256': factors.profile_hash(profile),
        'factor_snapshot': str(factor_path)}}
    point = {'time_ms': timestamp, 'signed_qty': 0}
    view = runner.btc_entry_view(report, point, timestamp, 'exchange')
    oil = view['entry_source_status']['factors']['oil']
    assert oil == {'ok': True, 'status': 'stale', 'fetch_status': 'ok', 'latest_observed_at_ms': 12 * day,
        'next_retry_at_ms': timestamp + 1000, 'age_ms': 8 * day,
        'max_age_ms': 7 * day, 'stale': True, 'applied': False, 'ignored': False}
    assert view['execution_entry_gate']['by_side']['long']['factor_source_status']['oil'] == oil
    # Oil is diagnostic only for this BTC-momentum-only profile.
    assert view['execution_entry_gate']['allowed'] is True
    assert view['entry_blockers'] == []
    assert payload['metadata']['source_status']['oil']['status'] == 'ok'
    payload['metadata']['source_status']['oil']['latest_observed_at_ms'] = 14 * day
    payload['series']['oil'].append([14 * day, 14 * day, 100, 14 * day])
    factor_path.write_text(json.dumps(payload), encoding='utf-8')
    refreshed = runner.btc_entry_view(report, point, timestamp, 'exchange')
    assert refreshed['entry_source_status']['factors']['oil']['status'] == 'ok'
    assert refreshed['entry_source_status']['factors']['oil']['age_ms'] == 6 * day
    assert refreshed['entry_source_status']['factors']['oil']['stale'] is False
    assert refreshed['execution_entry_gate']['allowed'] is True


@pytest.mark.parametrize('metadata', [{}, {'source_status': []}, {'source_status': {'oil': 'bad'}}])
def test_missing_or_malformed_source_metadata_is_diagnostic_only(metadata):
    snapshot = Mock(metadata=metadata, series={})
    result = runner.execution_entry_gate.factor_source_status(snapshot, 1_800_000)
    assert result['oil']['status'] == 'missing'
    assert result['oil']['age_ms'] is None


@pytest.mark.parametrize('metadata', [{}, {'source_status': {'oil': {
    'ok': False, 'status': 'stale', 'latest_observed_at_ms': 12 * 86_400_000}}}])
def test_usable_oil_series_overrides_missing_or_outdated_collector_dates(metadata):
    factors = runner.execution_entry_gate.multifactor
    day = factors.DAY
    timestamp = 20 * day
    snapshot = factors.Snapshot({'schema_version': 1, 'metadata': metadata,
        'series': {'oil': [[17 * day, 20 * day, 100, 20 * day]]}}, 'first_seen')
    assert snapshot.window('oil', timestamp, runner.execution_entry_gate.OIL_MAX_AGE_MS) == 100
    oil = runner.execution_entry_gate.factor_source_status(snapshot, timestamp)['oil']
    assert oil['latest_observed_at_ms'] == 17 * day
    assert oil['age_ms'] == 3 * day
    assert oil['status'] == 'ok'
    assert oil['stale'] is False


def test_future_first_seen_oil_row_cannot_refresh_current_source_date():
    factors = runner.execution_entry_gate.multifactor
    day = factors.DAY
    timestamp = 20 * day
    snapshot = factors.Snapshot({'schema_version': 1,
        'metadata': {'source_status': {'oil': {'ok': True, 'status': 'ok',
            'latest_observed_at_ms': 19 * day}}},
        'series': {'oil': [[12 * day, 12 * day, 100, 12 * day],
            [19 * day, 19 * day, 105, timestamp + 1]]}}, 'first_seen')
    assert snapshot.window('oil', timestamp, runner.execution_entry_gate.OIL_MAX_AGE_MS) is None
    oil = runner.execution_entry_gate.factor_source_status(snapshot, timestamp)['oil']
    assert oil['latest_observed_at_ms'] == 12 * day
    assert oil['age_ms'] == 8 * day
    assert oil['status'] == 'stale'
    after = runner.execution_entry_gate.factor_source_status(snapshot, timestamp + 1)['oil']
    assert after['latest_observed_at_ms'] == 19 * day
    assert after['status'] == 'ok'


def test_truncated_factor_gzip_blocks_increases_and_preserves_reductions(tmp_path):
    gate = runner.execution_entry_gate
    profile = {'candidate_id': 'truncated_source', 'live_orders_allowed': False,
        'groups': {'btc_momentum': 1}, 'minimum_group_coverage': 1,
        'minimum_risk_multiplier': .35, 'block_alignment_below': -.35,
        'block_stress_at': .9, 'availability_mode': 'first_seen'}
    profile_path = tmp_path / 'profile.json'
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    factor_path = tmp_path / 'truncated.json.gz'
    factor_path.write_bytes(b'\x1f\x8b\x08\x00')
    report = {'execution_entry_context': {'factor_profile': str(profile_path),
        'factor_profile_sha256': gate.multifactor.profile_hash(profile),
        'factor_snapshot': str(factor_path)}}
    result = gate.decision_at(report, 1_800_000, 'long')
    assert result['allowed'] is False
    assert result['reasons'] == ['entry_context_unavailable:EOFError']
    assert gate.constrain_quantity(0, 1, result) == 0
    assert gate.constrain_quantity(.5, 1, result) == .5
    assert gate.constrain_quantity(0, 0, result) == 0
    assert gate.constrain_quantity(2, 1, result) == 1
    assert gate.constrain_quantity(-1, 1, result) == 0


@pytest.fixture
def oil_policy_gate_fixture(tmp_path):
    gate = runner.execution_entry_gate
    day = gate.multifactor.DAY
    timestamp = 20 * day
    profile = {'candidate_id': 'optional_oil_current_gate', 'live_orders_allowed': False,
        'groups': {'global_risk': 1}, 'minimum_group_coverage': 1,
        'minimum_risk_multiplier': .35, 'block_alignment_below': -.35,
        'block_stress_at': .9, 'availability_mode': 'first_seen',
        'optional_features': ['oil'], 'optional_features_effective_at_ms': timestamp}
    profile_path = tmp_path / 'optional_profile.json'
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    payload = {'schema_version': 1, 'series': {
        name: [[12 * day, 12 * day, 100, 12 * day], [19 * day, 19 * day, 100, 19 * day]]
        for name in ['sp500', 'nasdaq']}}
    payload['series'].update(vix=[[19 * day, 19 * day, 15, 19 * day]],
        oil=[[4 * day, 4 * day, 100, 4 * day], [12 * day, 12 * day, 100, 12 * day]])
    factor_path = tmp_path / 'optional_factors.json'
    factor_path.write_text(json.dumps(payload), encoding='utf-8')
    report = {'execution_entry_context': {'factor_profile': str(profile_path),
        'factor_profile_sha256': gate.multifactor.profile_hash(profile),
        'factor_snapshot': str(factor_path)}}
    return timestamp, profile, profile_path, payload, factor_path, report


def test_actual_entry_gate_optional_stale_oil_is_ignored_only_after_policy_activation(oil_policy_gate_fixture):
    timestamp, _, _, _, _, report = oil_policy_gate_fixture
    point = {'time_ms': timestamp - 300_000, 'signed_qty': 0}
    before = runner.btc_entry_view(report, point, timestamp - 1, 'exchange')
    assert before['execution_entry_gate']['allowed'] is False
    assert before['entry_blockers'] == ['current_factor:missing_or_stale_factors']
    assert before['entry_data_unavailable'] is True
    assert before['entry_source_status']['factors']['oil']['ignored'] is False
    after = runner.btc_entry_view(report, point, timestamp, 'exchange')
    assert after['execution_entry_gate']['allowed'] is True
    assert after['entry_blockers'] == []
    assert after['entry_data_unavailable'] is False
    oil = after['entry_source_status']['factors']['oil']
    assert oil['status'] == 'stale'
    assert oil['ignored'] is True
    assert oil['applied'] is False
    assert oil['latest_observed_at_ms'] == 12 * runner.execution_entry_gate.multifactor.DAY
    for decision in after['execution_entry_gate']['by_side'].values():
        assert decision['factor']['ignored_features'] == ('oil',)
        assert decision['factor']['features']['oil_change'] is None


def test_optional_oil_does_not_hide_missing_required_global_risk_input(oil_policy_gate_fixture):
    timestamp, _, _, payload, factor_path, report = oil_policy_gate_fixture
    payload['series'].pop('vix')
    factor_path.write_text(json.dumps(payload), encoding='utf-8')
    view = runner.btc_entry_view(report, {'time_ms': timestamp, 'signed_qty': 0}, timestamp, 'exchange')
    assert view['execution_entry_gate']['allowed'] is False
    assert view['entry_data_unavailable'] is True
    assert view['entry_blockers'] == ['current_factor:missing_or_stale_factors']
    assert view['entry_source_status']['factors']['oil']['ignored'] is True
    gate = runner.execution_entry_gate
    assert gate.constrain_quantity(0, 1, view['execution_entry_gate']) == 0
    assert gate.constrain_quantity(.5, 1, view['execution_entry_gate']) == .5
    assert gate.constrain_quantity(2, 1, view['execution_entry_gate']) == 1


def test_optional_oil_fresh_observation_still_participates(oil_policy_gate_fixture):
    timestamp, _, _, payload, factor_path, report = oil_policy_gate_fixture
    day = runner.execution_entry_gate.multifactor.DAY
    payload['series']['oil'] = [[12 * day, 12 * day, 100, 12 * day],
                                [19 * day, 19 * day, 100, 19 * day]]
    factor_path.write_text(json.dumps(payload), encoding='utf-8')
    view = runner.btc_entry_view(report, {'time_ms': timestamp, 'signed_qty': 0}, timestamp, 'exchange')
    assert view['execution_entry_gate']['allowed'] is True
    oil = view['entry_source_status']['factors']['oil']
    assert oil['status'] == 'ok'
    assert oil['ignored'] is False
    assert oil['applied'] is True


def test_optional_oil_diagnostics_do_not_degrade_running_btc(btc_status_fixture, oil_policy_gate_fixture):
    step, _, state, _, report_path, _, _, reconcile = btc_status_fixture
    timestamp, _, _, _, _, report = oil_policy_gate_fixture
    # Use the policy fixture's exchange epoch and keep the same reconciled
    # target while rechecking its current optional-oil source diagnostics.
    fixture_clock = runner.decision_runtime.resolve_clock.return_value
    fixture_clock['time_ms'] = timestamp
    report['execution_target'] = {'time_ms': timestamp - 300_000,
        'available_time_ms': timestamp - 1, 'signed_qty': 0}
    state['last_signal_time_ms'] = report['execution_target']['time_ms']
    report_path.write_text(json.dumps(report), encoding='utf-8')
    result = step()
    assert result['status'] == 'healthy'
    assert result['signal_status'] == 'no_signal'
    assert result['entry_blockers'] == []
    assert result['entry_source_status']['factors']['oil']['status'] == 'stale'
    assert result['entry_source_status']['factors']['oil']['ignored'] is True
    reconcile.assert_not_called()
