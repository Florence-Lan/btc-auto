"""Transport fallbacks must not discard fresh clocks or rejuvenate old ones."""
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


def sample(start=100., received=100.2):
    return {'serverTime': 1_790_000_000_000, '_public_clock_timing': {
        'started_monotonic': start, 'received_monotonic': received, 'transport': 'requests'}}


def test_failed_primary_then_fresh_fallback_uses_only_successful_attempt(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(paper.time, 'monotonic', lambda: clock[0])
    def failed(*args, **kwargs):
        clock[0] += 17
        raise requests.Timeout('first stack timed out')
    def fallback(*args, **kwargs):
        clock[0] += .2
        return subprocess.CompletedProcess([], 0, json.dumps({'serverTime': 1_790_000_000_000})+'\n200', '')
    monkeypatch.setattr(paper.requests, 'get', failed)
    monkeypatch.setattr(paper.shutil, 'which', lambda name: 'curl.exe')
    monkeypatch.setattr(paper.subprocess, 'run', fallback)
    venue = paper.PublicAster()
    venue.deadline = SimpleNamespace(run=lambda key, callback, **kwargs: callback())
    payload = venue.get('time')
    timestamp, receipt, roundtrip = paper.verified_public_clock(payload, 100., clock[0])
    assert timestamp == 1_790_000_000_000
    assert receipt == pytest.approx(117.2)
    assert roundtrip == pytest.approx(.2)
    assert venue.last_public_transport['transport'] == 'curl'
    assert payload['_public_clock_timing']['started_monotonic'] == 117.


def test_slow_successful_primary_is_rejected_and_fast_fallback_can_recover(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(paper.time, 'monotonic', lambda: clock[0])
    response = Mock(status_code=200)
    response.json.return_value = {'serverTime': 1}
    def slow(*args, **kwargs):
        clock[0] += 5.1
        return response
    monkeypatch.setattr(paper.requests, 'get', slow)
    monkeypatch.setattr(paper.shutil, 'which', lambda name: 'curl.exe')
    def fast(*args, **kwargs):
        clock[0] += .1
        return subprocess.CompletedProcess([], 0, '{"serverTime":1790000000000}\n200', '')
    monkeypatch.setattr(paper.subprocess, 'run', fast)
    venue = paper.PublicAster()
    venue.deadline = SimpleNamespace(run=lambda key, callback, **kwargs: callback())
    payload = venue.get('time')
    assert payload['serverTime'] == 1_790_000_000_000
    assert payload['_public_clock_timing']['transport'] == 'curl'
    response.close.assert_called_once()


def test_delayed_consumption_does_not_refresh_observation_receipt():
    payload = sample()
    before = copy.deepcopy(payload)
    assert paper.verified_public_clock(payload, 105., 110.)[1] == 100.2
    with pytest.raises(ValueError, match='Stale public exchange clock receipt'):
        paper.verified_public_clock(payload, 116., 116.)
    assert payload == before


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(serverTime=True),
    lambda p: p.update(serverTime=0),
    lambda p: p.update(serverTime=1.5),
    lambda p: p.update(_public_clock_timing=None),
    lambda p: p['_public_clock_timing'].update(started_monotonic=float('nan')),
    lambda p: p['_public_clock_timing'].update(received_monotonic=float('inf')),
    lambda p: p['_public_clock_timing'].update(started_monotonic=True),
    lambda p: p['_public_clock_timing'].update(started_monotonic=101.),
    lambda p: p['_public_clock_timing'].update(received_monotonic=111.),
    lambda p: p['_public_clock_timing'].update(started_monotonic=90.),
    lambda p: p['_public_clock_timing'].update(transport='unverified'),
])
def test_invalid_future_or_slow_timing_is_rejected(mutate):
    payload = sample()
    mutate(payload)
    with pytest.raises(ValueError):
        paper.verified_public_clock(payload, 100., 110.)


def test_legacy_public_venue_retains_outer_call_five_second_limit():
    assert paper.verified_public_clock({'serverTime': 123}, 10., 15.) == (123, 15., 5.)
    with pytest.raises(ValueError, match='five seconds'):
        paper.verified_public_clock({'serverTime': 123}, 10., 15.001)


def test_stock_market_data_preserves_actual_success_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(paper.time, 'monotonic', lambda: 110.)
    cfg = {'signal_timeframe': '1h'}
    path = tmp_path/'state.json'
    paper.write_json(path, paper.initial_stock('MUUSDT', 1, cfg))
    observed = sample()
    venue = SimpleNamespace(get=lambda endpoint, params: observed if endpoint == 'time' else [])
    worker = paper.StockAccount(path, venue, 'MUUSDT', cfg)
    result = {label: (payload, error) for label, payload, error in worker.market_data()}
    clock, error = result['clock']
    assert error is None
    assert clock['_received_monotonic'] == 100.2
    assert '_received_monotonic' not in observed
