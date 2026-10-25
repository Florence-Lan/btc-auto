"""Bounded public fallback and causal, credential-free clock diagnostics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
from public_transport import PublicTransportError


STAMP = 1_790_000_000_000


def venue(monkeypatch, primary, fallback=None, delay=.2):
    clock = [100.]
    monkeypatch.setattr(paper.time, 'monotonic', lambda: clock[0])
    response = Mock(status_code=200)
    response.json.return_value = primary
    monkeypatch.setattr(paper.requests, 'get', Mock(return_value=response))
    monkeypatch.setattr(paper.shutil, 'which', lambda name: 'curl.exe')
    def read(*args, **kwargs):
        clock[0] += delay
        return subprocess.CompletedProcess([], 0,
            json.dumps(fallback if fallback is not None else {'serverTime': STAMP}) + '\n200', '')
    curl = Mock(side_effect=read)
    monkeypatch.setattr(paper.subprocess, 'run', curl)
    result = paper.PublicAster()
    result.deadline = SimpleNamespace(run=lambda key, callback, **kwargs: callback())
    return result, response, curl, clock


@pytest.mark.parametrize('bad_code', [None, [], {}, 'not-an-api-code', float('nan'), float('inf')])
def test_unparseable_code_uses_single_fallback_without_changing_clock_receipt(monkeypatch, bad_code):
    bot, response, curl, clock = venue(monkeypatch, {'code': bad_code, 'serverTime': STAMP})
    payload = bot.get('time')
    stamp, received, elapsed = paper.verified_public_clock(payload, 100., clock[0])
    assert stamp == STAMP and received == pytest.approx(100.2) and elapsed == pytest.approx(.2)
    response.close.assert_called_once()
    curl.assert_called_once()
    failure, success = payload['_public_clock_attempts']
    assert failure['ok'] is False and failure['reason_code'] == 'api_code_invalid'
    assert failure['elapsed_scope'] == 'network' and failure['elapsed_ms'] == 0
    assert success['ok'] is True and success['transport'] == 'curl'
    assert success['network_elapsed_ms'] == pytest.approx(200)
    assert success['elapsed_scope'] == 'caller_wait'


@pytest.mark.parametrize('endpoint', ['time', 'depth'])
def test_two_bad_responses_fail_closed_and_render_only_safe_reasons(monkeypatch, endpoint):
    secret = 'private-proxy-user:private-proxy-password'
    body = {'code': {'url': 'https://' + secret + '@proxy.invalid/?token=private-token'},
            'serverTime': STAMP}
    bot, response, curl, _ = venue(monkeypatch, body, body)
    with pytest.raises(PublicTransportError) as caught:
        bot.get(endpoint)
    assert len(caught.value.failures) == 2
    assert all(f['reason_code'] == 'api_code_invalid' for f in caught.value.failures)
    assert all(f['elapsed_scope'] == 'network' for f in caught.value.failures)
    rendered = str(caught.value) + json.dumps(caught.value.failures)
    assert 'private-' not in rendered and 'proxy.invalid' not in rendered
    assert 'api_code_invalid' in rendered
    response.close.assert_called_once()
    curl.assert_called_once()


@pytest.mark.parametrize('value', [-1003, '-1003'])
def test_valid_api_error_keeps_original_no_retry_behavior(monkeypatch, value):
    bot, response, curl, _ = venue(monkeypatch, {'code': value, 'msg': 'remote text'})
    with pytest.raises(RuntimeError, match='HTTP 200'):
        bot.get('time')
    curl.assert_not_called()
    response.close.assert_called_once()


@pytest.mark.parametrize('status', [418, 429])
def test_rate_limit_still_stops_fallback_and_sets_cooldown(monkeypatch, status):
    bot, response, curl, _ = venue(monkeypatch, {'serverTime': STAMP})
    response.status_code = status
    monkeypatch.setattr(paper, 'now_ms', lambda: STAMP)
    with pytest.raises(RuntimeError, match='HTTP'):
        bot.get('time')
    assert bot.cooldown == STAMP + 120_000
    curl.assert_not_called()


@pytest.mark.parametrize('bad_stamp', [None, True, 0, -1, 1.5, '1000'])
def test_invalid_timestamp_classified_and_fresh_fallback_preserved(monkeypatch, bad_stamp):
    bot, _, curl, _ = venue(monkeypatch, {'serverTime': bad_stamp})
    result = bot.get('time')
    assert result['serverTime'] == STAMP
    assert result['_public_clock_attempts'][0]['reason_code'] == 'clock_timestamp_invalid'
    curl.assert_called_once()


def test_slow_success_keeps_actual_network_duration_and_uses_fresh_fallback(monkeypatch):
    bot, response, _, clock = venue(monkeypatch, {'serverTime': STAMP})
    def slow(*args, **kwargs):
        clock[0] += 5.0001
        return response
    monkeypatch.setattr(paper.requests, 'get', slow)
    result = bot.get('time')
    failure = result['_public_clock_attempts'][0]
    assert failure['reason_code'] == 'clock_roundtrip_exceeded'
    assert failure['elapsed_scope'] == 'network'
    assert failure['elapsed_ms'] > 5000
    assert paper.verified_public_clock(result, 100., clock[0])[2] == pytest.approx(.2)


def test_budget_timeout_reports_caller_wait_not_a_fabricated_network_duration(monkeypatch):
    bot, _, curl, clock = venue(monkeypatch, {'serverTime': STAMP})
    def unavailable(*args, **kwargs):
        clock[0] += .01
        raise requests.Timeout('private request URL with proxy credentials')
    bot.deadline = SimpleNamespace(run=unavailable)
    result = bot.get('time')
    failure = result['_public_clock_attempts'][0]
    assert failure['category'] == 'timeout' and failure['elapsed_scope'] == 'caller_wait'
    assert failure['elapsed_ms'] == pytest.approx(10)
    assert 'private' not in json.dumps(result)
    paper.requests.get.assert_not_called()
    curl.assert_called_once()


@pytest.mark.parametrize('bad_code', [None, {'secret': 'private-proxy-password'}])
def test_curl_preferred_bad_code_falls_back_once_to_requests(monkeypatch, bad_code):
    bot, response, curl, clock = venue(monkeypatch, {'serverTime': STAMP},
        {'code': bad_code, 'serverTime': STAMP})
    bot._curl_preferred_until = 200.
    result = bot.get('time')
    failed, success = result['_public_clock_attempts']
    assert failed['transport'] == 'curl' and failed['reason_code'] == 'api_code_invalid'
    assert failed['elapsed_scope'] == 'network' and failed['elapsed_ms'] == pytest.approx(200)
    assert success['transport'] == 'requests' and success['ok'] is True
    assert result['_public_clock_timing']['transport'] == 'requests'
    assert paper.verified_public_clock(result, 100., clock[0])[1] == pytest.approx(100.2)
    assert 'private-' not in json.dumps(result)
    paper.requests.get.assert_called_once()
    response.close.assert_called_once()
    curl.assert_called_once()


def test_curl_preferred_two_bad_responses_fail_closed_in_actual_attempt_order(monkeypatch):
    body = {'code': {'secret': 'private-token'}, 'serverTime': STAMP}
    bot, response, curl, _ = venue(monkeypatch, body, body)
    bot._curl_preferred_until = 200.
    with pytest.raises(PublicTransportError) as caught:
        bot.get('time')
    assert [f['transport'] for f in caught.value.failures] == ['curl', 'requests']
    assert all(f['reason_code'] == 'api_code_invalid' for f in caught.value.failures)
    assert all(f['elapsed_scope'] == 'network' for f in caught.value.failures)
    assert 'private-token' not in str(caught.value) + json.dumps(caught.value.failures)
    paper.requests.get.assert_called_once()
    response.close.assert_called_once()
    curl.assert_called_once()


def test_market_data_persists_both_safe_failed_attempts_without_requesting_a_new_clock(tmp_path, monkeypatch):
    failures = [{'transport': transport, 'error_type': 'InvalidPublicResponse',
        'category': 'invalid_response', 'reason_code': reason,
        'elapsed_ms': elapsed, 'elapsed_scope': 'network'}
        for transport, reason, elapsed in [('requests', 'clock_roundtrip_exceeded', 6100),
                                         ('curl', 'api_code_invalid', 2300)]]
    error = PublicTransportError('Public verified fallback transport failed', failures)
    cfg = {'signal_timeframe': '1h'}
    path = tmp_path / 'state.json'
    paper.write_json(path, paper.initial_stock('MUUSDT', 1, cfg))
    def get(endpoint, params=None):
        if endpoint == 'time':
            raise error
        return []
    backend = SimpleNamespace(get=Mock(side_effect=get))
    worker = paper.StockAccount(path, backend, 'MUUSDT', cfg)
    clock_record = next(row for row in worker.market_data() if row[0] == 'clock')
    assert clock_record[1] is None
    assert 'clock_roundtrip_exceeded' in clock_record[2] and 'api_code_invalid' in clock_record[2]
    assert '6100' in clock_record[2] and '2300' in clock_record[2]
    assert sum(call.args[0] == 'time' for call in backend.get.call_args_list) == 1


def test_joined_read_returns_private_diagnostics_without_rejuvenating_source(monkeypatch):
    bot, _, curl, clock = venue(monkeypatch, {'serverTime': STAMP})
    original = {'serverTime': STAMP, '_public_clock_timing': {
        'started_monotonic': 90., 'received_monotonic': 90.2, 'transport': 'requests'}}
    before = copy.deepcopy(original)
    bot.deadline = SimpleNamespace(run=lambda *a, **kw: (200, original))
    first, second = bot.get('time'), bot.get('time')
    first['_public_clock_attempts'][0]['transport'] = 'changed'
    first['_public_clock_timing']['received_monotonic'] = 100.
    assert second['_public_clock_attempts'][0]['transport'] == 'requests'
    assert original == before and second['_public_clock_timing']['received_monotonic'] == 90.2
    assert paper.verified_public_clock(second, 100., clock[0])[1] == 90.2
    clock[0] = 106.
    with pytest.raises(paper.PublicClockValidationError) as caught:
        paper.verified_public_clock(second, 106., clock[0])
    assert caught.value.reason_code == 'clock_receipt_stale'
    curl.assert_not_called()


@pytest.mark.parametrize('payload,start,received,reason', [
    ([], 100., 100.1, 'clock_payload_invalid'),
    ({'serverTime': None}, 100., 100.1, 'clock_timestamp_invalid'),
    ({'serverTime': STAMP, '_public_clock_timing': None}, 100., 100.1, 'clock_timing_invalid'),
    ({'serverTime': STAMP}, 101., 100., 'clock_timing_invalid'),
    ({'serverTime': STAMP}, 100., 105.001, 'clock_roundtrip_exceeded'),
])
def test_reason_codes_leave_legacy_valueerror_contract_intact(payload, start, received, reason):
    with pytest.raises(ValueError) as caught:
        paper.verified_public_clock(payload, start, received)
    assert isinstance(caught.value, paper.PublicClockValidationError)
    assert caught.value.reason_code == reason
