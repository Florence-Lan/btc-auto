"""Diagnostics expose actual vintage dates without relaxing the factor engine."""
from copy import deepcopy
from dataclasses import asdict
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_multifactor_snapshot as download
import execution_entry_gate as gate
import multifactor as mf
from factor_data_freshness import DAY, HOUR, describe_factors


def profile():
    return {'candidate_id': 'diagnostics_only', 'live_orders_allowed': False,
        'groups': {'btc_momentum': 1}, 'minimum_group_coverage': 1,
        'minimum_risk_multiplier': .35, 'block_alignment_below': -.35,
        'block_stress_at': .9, 'availability_mode': 'first_seen'}


def row(observed, value=100, seen=None):
    return [observed, observed, value, observed if seen is None else seen]


def test_successful_download_is_separate_from_old_observations_and_future_vintages():
    now = 100 * DAY
    payload = {'schema_version': 1, 'series': {
        'oil': [row(now - 8 * DAY), row(now - DAY, seen=now + 1)],
        'broad_dollar': [row(now - 13 * DAY), row(now - 20 * DAY)],
        'fed_assets': [row(now - 14 * DAY)],
    }, 'metadata': {'source_status': {
        'oil': {'status': 'healthy', 'ok': True, 'fetch_ok': True,
                'last_attempt_ms': now - 1000, 'last_fetch_success_ms': now - 1000,
                'latest_observed_at_ms': now - DAY},
    }}}
    original = deepcopy(payload)
    snapshot = mf.Snapshot(payload)
    result = describe_factors(snapshot, now, profile())
    oil = result['factors']['oil']
    assert oil['latest_observed_at_ms'] == now - 8 * DAY
    assert oil['status'] == 'stale' and oil['age_ms'] == 8 * DAY
    assert oil['fetch_ok'] is True and oil['last_fetch_success_ms'] == now - 1000
    assert oil['max_age_ms'] == 7 * DAY
    assert oil['latest_observed_at_utc'].endswith('+00:00')
    assert result['factors']['broad_dollar']['status'] == 'fresh'
    assert result['factors']['broad_dollar']['age_ms'] == 13 * DAY
    assets = result['factors']['fed_assets']
    assert assets['slow_frequency'] and assets['cadence'] == 'weekly'
    assert assets['status'] == 'fresh' and assets['values_available'] is False
    assert any('历史不足' in x for x in assets['notes'])
    assert result['display_only'] is True and payload == original
    assert describe_factors(snapshot, now + 1)['factors']['oil']['latest_observed_at_ms'] == now - DAY


@pytest.mark.parametrize('name,limit', [
    ('btc_close', 3 * HOUR), ('global_long_short', 3 * HOUR),
    ('funding', 12 * HOUR), ('fed_target', 7 * DAY),
    ('fed_assets', 16 * DAY), ('ust_10y', 7 * DAY),
    ('eurusd', 16 * DAY), ('oil', 7 * DAY), ('gold', 7 * DAY),
])
def test_display_inclusive_boundaries_match_the_existing_snapshot_policy(name, limit):
    now = 100 * DAY
    snapshot = mf.Snapshot({'schema_version': 1, 'series': {name: [row(now - limit)]}})
    current = describe_factors(snapshot, now)['factors'][name]
    assert current['max_age_ms'] == limit and current['status'] == 'fresh'
    assert snapshot.window(name, now, limit) == 100
    expired = describe_factors(snapshot, now + 1)['factors'][name]
    assert expired['status'] == 'stale' and snapshot.window(name, now + 1, limit) is None


def test_optional_input_and_unused_series_do_not_invent_health_or_thresholds():
    now = 100 * DAY
    config = profile()
    config.update(optional_features=['oil'], optional_features_effective_at_ms=now)
    snapshot = mf.Snapshot({'schema_version': 1, 'series': {'fed_effective': [row(now - 20 * DAY)]},
        'metadata': {'source_status': {'oil': {'status': 'stale', 'last_attempt_ms': now,
                                            'last_success_ms': now - DAY}}}})
    data = describe_factors(snapshot, now, config)['factors']
    assert data['oil']['optional'] and not data['oil']['required_for_group']
    assert data['oil']['status'] == 'missing' and data['oil']['fetch_ok'] is None
    assert data['oil']['last_fetch_success_ms'] == now - DAY
    assert data['oil']['fetch_success_time_kind'] == 'legacy_usable_collection_record'
    assert data['fed_effective']['status'] == 'not_used'
    assert data['fed_effective']['max_age_ms'] is None
    assert not data['fed_effective']['required_for_group']
    assert data['gold']['optional']
    assert not describe_factors(snapshot, now - 1, config)['factors']['oil']['optional']


def test_future_collection_metadata_never_becomes_a_past_success():
    now = 100 * DAY
    snapshot = mf.Snapshot({'schema_version': 1, 'series': {}, 'metadata': {'source_status': {
        'oil': {'ok': True, 'fetch_ok': True, 'status': 'healthy', 'last_attempt_ms': now + 1,
                'last_fetch_success_ms': now + 1, 'last_success_ms': now + 1}}}})
    oil = describe_factors(snapshot, now)['factors']['oil']
    assert oil['fetch_ok'] is None
    assert oil['last_fetch_success_ms'] is None and oil['last_fetch_attempt_ms'] is None
    assert oil['status'] == 'missing'


def test_added_gate_diagnostics_preserve_every_existing_factor_decision(tmp_path):
    now = 100 * DAY
    payload = {'schema_version': 1, 'series': {'btc_close': [
        row(now - 7 * DAY), row(now - DAY), row(now)], 'oil': [row(now - 8 * DAY)]}}
    config = profile()
    profile_path, factor_path = tmp_path / 'profile.json', tmp_path / 'factors.json'
    profile_path.write_text(json.dumps(config), encoding='utf-8')
    factor_path.write_text(json.dumps(payload), encoding='utf-8')
    report = {'multifactor_overlay': {}, 'execution_entry_context': {
        'factor_profile': str(profile_path), 'factor_profile_sha256': mf.profile_hash(config),
        'factor_snapshot': str(factor_path)}}
    baseline = asdict(mf.decision_at(mf.Snapshot(payload), now, 'long', config))
    result = gate.decision_at(report, now, 'long')
    assert result['allowed'] is True and result['reasons'] == []
    assert result['factor'] == baseline
    assert result['factor_data_freshness']['factors']['oil']['status'] == 'stale'
    assert result['factor_data_freshness']['factors']['oil']['group_enabled'] is False


def test_unexpected_diagnostic_data_does_not_raise_into_the_trading_gate():
    snapshot = Mock(metadata={}, series={}, mode='first_seen')
    snapshot.window.side_effect = ValueError('invalid display metadata')
    result = describe_factors(snapshot, 100 * DAY)
    assert result['status'] == 'diagnostics_unavailable' and result['display_only']
    assert result['error_type'] == 'ValueError' and result['factors'] == {}


def test_transport_success_is_retained_when_oil_is_stale_and_later_fetch_fails(tmp_path, monkeypatch):
    clock = [100 * DAY]
    monkeypatch.setattr(download, 'now_ms', lambda: clock[0])
    old = row(clock[0] - 9 * DAY)
    fetch = Mock(return_value=[old.copy()])
    monkeypatch.setattr(download, 'fetch_fred', fetch)
    path = tmp_path / 'factors.json.gz'
    collected = download.collect(path, 0, clock[0], sources=['oil'])
    state = collected['metadata']['source_status']['oil']
    assert state['fetch_ok'] is True and state['last_fetch_success_ms'] == clock[0]
    assert not state['ok'] and state['status'] == 'stale'
    assert state['last_success_ms'] is None  # Existing usable-data semantics unchanged.
    assert state['next_retry_at_ms'] == clock[0] + 60_000
    first_success = clock[0]
    clock[0] += 60_000
    fetch.side_effect = requests.exceptions.ConnectionError('offline')
    failed = download.collect(path, 0, clock[0], sources=['oil'])
    state = failed['metadata']['source_status']['oil']
    assert state['fetch_ok'] is False and state['last_fetch_success_ms'] == first_success
    assert state['last_attempt_ms'] == clock[0] and state['last_success_ms'] is None
    display = describe_factors(mf.Snapshot(failed), clock[0])['factors']['oil']
    assert display['fetch_ok'] is False and display['status'] == 'stale'
    assert display['latest_observed_at_ms'] == old[0]
