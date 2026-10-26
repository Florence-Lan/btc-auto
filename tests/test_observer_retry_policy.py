"""Retry scheduling is opt-in, bounded, causal and cannot bypass public denial."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cross_market_observations as co
from public_transport import InvalidPublicResponse, PublicTransportError

NOW = int(datetime(2026, 10, 26, 14, tzinfo=timezone.utc).timestamp() * 1000)
POLICY = {'initial_delay_seconds': 60, 'maximum_delay_seconds': 900, 'backoff_multiplier': 2}


@pytest.fixture(autouse=True)
def isolated_deadlines():
    co._DEADLINES.clear()
    yield
    co._DEADLINES.clear()


def source(*, poll=21600):
    return {'id': 'official_daily', 'parser': 'dated_csv', 'provider': 'fixture',
            'url': 'https://example.org/daily.csv', 'timezone': 'UTC',
            'date_column': 'Date', 'date_format': '%Y-%m-%d', 'poll_seconds': poll,
            'timeout_seconds': .5, 'instrument': {'kind': 'official_daily_yield', 'tenor': '10Y'},
            'fields': {'us_10y_pct': {'unit': 'percent', 'column': '10Y', 'max_age_days': 4}}}


def config(*, policy=True, poll=21600):
    value = {'schema_version': 1, 'mode': 'OBSERVATION_ONLY',
             'collection_timeout_seconds': 1, 'sources': [source(poll=poll)]}
    if policy:
        value['retry_policy'] = copy.deepcopy(POLICY)
    return value


def fail(exc):
    def fetch(*_):
        raise exc
    return fetch


def collect(conf, *, previous=None, now=NOW, exc=None):
    return co.collect_observations(conf, previous=previous, now_ms=now,
        fetcher=fail(exc) if exc is not None else lambda *_: 'Date,10Y\n2026-10-25,4.2\n')


def state(snapshot):
    return snapshot['sources']['official_daily']


def test_policy_is_opt_in_legacy_daily_failure_still_waits_six_hours():
    old = collect(config(policy=False), exc=co.requests.Timeout('secret'))
    assert state(old)['next_due_at_ms'] == NOW + 21_600_000
    assert 'consecutive_failures' not in state(old)
    assert 'retryable_transport' not in state(old)['error']
    assert 'secret' not in json.dumps(old)


def test_transient_failures_backoff_to_cap_then_success_resets_and_restores_daily_poll():
    conf, previous, now = config(), None, NOW
    for count, delay in enumerate((60, 120, 240, 480, 900, 900, 900), 1):
        result = collect(conf, previous=previous, now=now, exc=co.requests.ReadTimeout('private'))
        row = state(result)
        assert row['consecutive_failures'] == row['consecutive_retryable_failures'] == count
        assert row['next_due_at_ms'] == now + delay * 1000
        assert row['retry_delay_ms'] == delay * 1000
        before_due = co.collect_observations(conf, previous=result, now_ms=row['next_due_at_ms'] - 1,
            fetcher=lambda *_: pytest.fail('No request before scheduled backoff.'))
        assert state(before_due)['consecutive_failures'] == count
        previous, now = result, row['next_due_at_ms']
    success = collect(conf, previous=previous, now=now)
    assert state(success)['consecutive_failures'] == state(success)['consecutive_retryable_failures'] == 0
    assert state(success)['next_due_at_ms'] == now + 21_600_000
    failed_again = collect(conf, previous=success, now=state(success)['next_due_at_ms'], exc=TimeoutError())
    assert state(failed_again)['retry_delay_ms'] == 60_000


@pytest.mark.parametrize('exc,expected', [
    (co.requests.ReadTimeout(), True), (co.requests.ConnectTimeout(), True),
    (co.requests.ConnectionError(), True), (socket.gaierror(), True),
    (co.requests.exceptions.ChunkedEncodingError(), True),
    (co.requests.exceptions.SSLError(), False), (co.requests.exceptions.ProxyError(), False),
    (InvalidPublicResponse(reason_code='response_not_json'), False),
    (co.ObservationError('instrument_identity_mismatch:symbol'), False),
    (ValueError('invalid schema'), False), (json.JSONDecodeError('invalid', 'x', 0), False),
    (co.PublicHttpError(500), False), (co.PublicHttpError(404), False)])
def test_only_recoverable_transport_categories_get_short_schedule(exc, expected):
    result = collect(config(), exc=exc)
    row = state(result)
    assert row['consecutive_failures'] == 1
    assert row['consecutive_retryable_failures'] == int(expected)
    assert row['retry_delay_ms'] == (60_000 if expected else 21_600_000)
    assert row['error']['retryable_transport'] is expected


def test_aggregate_validation_or_tls_failure_cannot_masquerade_as_connection_retry():
    for category in ('invalid_response', 'tls', 'proxy'):
        exc = PublicTransportError('public_failure', [
            {'transport': 'requests', 'error_type': 'Timeout', 'category': 'timeout'},
            {'transport': 'curl', 'error_type': 'Failure', 'category': category}])
        assert state(collect(config(), exc=exc))['retry_delay_ms'] == 21_600_000
    tls = co.requests.exceptions.SSLError()
    tls.failures = [{'category': 'timeout'}]
    assert state(collect(config(), exc=tls))['retry_delay_ms'] == 21_600_000


@pytest.mark.parametrize('status', [403, 429])
@pytest.mark.parametrize('retry_after', [None, 30, 7200, 172800])
def test_http_denial_respects_poll_hour_minimum_and_unshortened_retry_after(status, retry_after):
    result = collect(config(poll=60), exc=co.PublicHttpError(status, retry_after))
    wait = max(3600, retry_after or 0)
    assert state(result)['next_due_at_ms'] == NOW + wait * 1000
    assert state(result)['consecutive_retryable_failures'] == 0
    co.collect_observations(config(poll=60), previous=result, now_ms=NOW + wait * 1000 - 1,
        fetcher=lambda *_: pytest.fail('403/429 cooldown cannot be bypassed.'))
    daily = collect(config(), exc=co.PublicHttpError(status, retry_after))
    assert state(daily)['retry_delay_ms'] == max(21600, wait) * 1000


def test_failure_sequence_is_per_source_and_nonretryable_error_breaks_transport_streak():
    conf = config()
    first = collect(conf, exc=co.requests.Timeout())
    second = collect(conf, previous=first, now=NOW + 60_000, exc=co.ObservationError('csv_date_header_missing'))
    assert state(second)['consecutive_failures'] == 2 and state(second)['consecutive_retryable_failures'] == 0
    third = collect(conf, previous=second, now=state(second)['next_due_at_ms'], exc=co.requests.Timeout())
    assert state(third)['consecutive_failures'] == 3 and state(third)['retry_delay_ms'] == 60_000
    both = config()
    other = source(); other.update(id='other', url='https://example.org/other.csv')
    other['fields'] = {'other_yield': {'unit': 'percent', 'column': '10Y', 'max_age_days': 4}}
    both['sources'].append(other)
    result = co.collect_observations(both, now_ms=NOW, fetcher=lambda s, _: (
        fail(co.requests.Timeout())() if s['id'] == 'official_daily' else 'Date,10Y\n2026-10-25,4.2\n'))
    assert state(result)['consecutive_failures'] == 1
    assert result['sources']['other']['consecutive_failures'] == 0


def test_cache_and_failure_do_not_renew_observation_or_first_receipt():
    conf = config(poll=60)
    first = collect(conf)
    failed = collect(conf, previous=first, now=NOW + 60_000, exc=co.requests.Timeout())
    cached = co.collect_observations(conf, previous=failed, now_ms=NOW + 61_000,
        fetcher=lambda *_: pytest.fail('Cache should suppress request.'))
    for snap in (failed, cached):
        row = snap['observations']['us_10y_pct']
        assert row['value'] == 4.2 and row['observed_date'] == '2026-10-25'
        assert row['first_seen_at_ms'] == row['available_at_ms'] == row['received_at_ms'] == NOW
        assert not row['eligible_intraday']


def test_clock_rollback_never_shortens_long_denial_or_transport_schedule():
    conf = config(poll=60)
    first = collect(conf, exc=co.PublicHttpError(429, 172800))
    for now in (NOW - 1, NOW, NOW + 1, NOW + 100_000):
        first = co.collect_observations(conf, previous=first, now_ms=now,
            fetcher=lambda *_: pytest.fail('Rollback must not cause new GET.'))
        assert state(first)['next_due_at_ms'] == NOW + 172_800_000
        assert state(first)['consecutive_failures'] == 1


def test_total_collection_deadline_is_counted_and_late_result_does_not_mutate_snapshot(monkeypatch):
    conf = config()
    conf['collection_timeout_seconds'] = .03
    conf['sources'][0]['timeout_seconds'] = .02
    release, done, calls = threading.Event(), threading.Event(), []
    def delayed(source, fetcher, now_fn, retry_policy=None):
        calls.append(source['id'])
        try:
            assert release.wait(2)
            return {'us_10y_pct': {'value': 9.0}}, NOW, 100, None
        finally:
            done.set()
    monkeypatch.setattr(co, '_source_fetch', delayed)
    try:
        result = co.collect_observations(conf, now_ms=NOW, fetcher=lambda *_: None)
        assert state(result)['status'] == 'deadline_exceeded'
        assert state(result)['consecutive_failures'] == state(result)['consecutive_retryable_failures'] == 1
        assert state(result)['next_due_at_ms'] == NOW + 60_000
        saved = copy.deepcopy(result)
    finally:
        release.set()
        assert done.wait(2)
    assert result == saved and result['observations']['us_10y_pct']['value'] is None
    assert calls == ['official_daily']


def test_retry_after_supports_http_dates_and_malformed_header_keeps_denial(monkeypatch):
    assert co._retry_after_seconds('Wed, 28 Oct 2026 14:00:00 GMT', NOW) == 172800
    assert co._retry_after_seconds('7200', NOW) == 7200
    for value in ('not-a-date', '1' * 10000, '-10', True, None):
        assert co._retry_after_seconds(value, NOW) is None
    assert co._retry_after_seconds('Mon, 26 Oct 2026 13:00:00 GMT', NOW) == 0
    calls = []
    class Response:
        status_code = 429
        headers = {'Retry-After': 'Wed, 28 Oct 2026 14:00:00 GMT'}
        def close(self):
            pass
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(co.requests, 'get', get)
    monkeypatch.setattr(co, '_clock_ms', lambda: NOW)
    result = co.collect_observations(config(poll=60), now_ms=NOW)
    assert state(result)['next_due_at_ms'] == NOW + 172_800_000
    assert len(calls) == 1 and calls[0][1]['allow_redirects'] is False


@pytest.mark.parametrize('policy', [None, True, [], {}, {'unexpected': 60},
    {**POLICY, 'unknown': 1}, {**POLICY, 'initial_delay_seconds': True},
    {**POLICY, 'initial_delay_seconds': '60'}, {**POLICY, 'initial_delay_seconds': 29},
    {**POLICY, 'maximum_delay_seconds': 30}, {**POLICY, 'maximum_delay_seconds': 3601},
    {**POLICY, 'backoff_multiplier': 1}, {**POLICY, 'backoff_multiplier': 2.0},
    {**POLICY, 'backoff_multiplier': float('nan')}, {**POLICY, 'maximum_delay_seconds': float('inf')}])
def test_invalid_policy_is_rejected_before_any_request(policy):
    conf = config(); conf['retry_policy'] = policy
    with pytest.raises(ValueError):
        co.collect_observations(conf, now_ms=NOW, fetcher=lambda *_: pytest.fail('Invalid policy must not fetch.'))


def test_policy_cannot_hide_at_source_level_or_change_observation_identity():
    conf = config(); conf['sources'][0]['retry_policy'] = POLICY
    with pytest.raises(ValueError):
        co.validate_config(conf)
    without = collect(config(policy=False, poll=60))
    changed = collect(config(poll=60), previous=without, now=NOW + 60_000)
    assert changed['observations']['us_10y_pct']['source_config_id'] == without['observations']['us_10y_pct']['source_config_id']
    assert changed['observations']['us_10y_pct']['available_at_ms'] == NOW
