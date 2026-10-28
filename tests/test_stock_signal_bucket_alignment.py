"""Signal cache freshness follows the exchange snapshot, not batch completion.

Exercise real market_data/step methods against a fake venue and in-memory
ledgers. No transport client is constructed and no active account is read.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper

HOUR = 3_600_000
FIVE = 300_000
BOUNDARY = 1_793_160_000_000  # 2026-10-28 04:00 UTC, aligned to every tested interval.


class MemoryPath:
    def __init__(self, store, name='state.json'):
        self.store, self.name = store, name

    def read_text(self, **kwargs):
        return json.dumps(self.store[self.name])

    def with_name(self, name):
        return MemoryPath(self.store, name)


def worker_fixture(monkeypatch, timeframe='1h'):
    cfg = paper.stock_config({'stock_research_profile':
                              'config/stock_event_guarded_20261020.json'}, 'MUUSDT')
    cfg.pop('stock_event_risk_policy', None)
    if timeframe != '1h':
        cfg.pop('swing_confirmation_policy', None)
        cfg.pop('entry_timing_policy', None)
    cfg['signal_timeframe'] = timeframe
    clock = {'now': BOUNDARY - 2_000, 'stale_labels': set(), 'failed_labels': set(),
             'invalid_clock': False}
    monkeypatch.setattr(paper, 'now_ms', lambda: clock['now'])
    monkeypatch.setattr(paper.time, 'monotonic', lambda: clock['now'] / 1000)
    store = {'state.json': paper.initial_stock('MUUSDT', BOUNDARY - 8 * HOUR, cfg)}
    monkeypatch.setattr(paper, 'write_json',
                        lambda path, value: store.__setitem__(path.name, copy.deepcopy(value)))
    monkeypatch.setattr(paper, 'journal',
                        lambda path, value: store.setdefault(path.name, []).append(copy.deepcopy(value)))
    calls = []
    rules = {'symbol': 'MUUSDT', 'status': 'TRADING', 'filters': [
        {'filterType': 'LOT_SIZE', 'stepSize': '0.01', 'minQty': '0.01', 'maxQty': '100'},
        {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
        {'filterType': 'MIN_NOTIONAL', 'notional': '5'}]}

    class FakeVenue:
        def get(self, endpoint, params=None):
            calls.append((endpoint, copy.deepcopy(params)))
            now = clock['now']
            label = ('signal' if params['interval'] == timeframe else 'five'
                     ) if endpoint == 'klines' else {'time': 'clock'}.get(endpoint, endpoint)
            if label in clock['failed_labels']:
                raise RuntimeError('fixture source unavailable')
            if endpoint == 'time':
                return {'serverTime': True if clock['invalid_clock'] else now}
            if endpoint == 'premiumIndex':
                return {'time': now, 'markPrice': '100', 'indexPrice': '100',
                        'lastFundingRate': '0', 'nextFundingTime': now + HOUR}
            if endpoint == 'depth':
                return {'T': now, 'bids': [['99.99', '1000']], 'asks': [['100.01', '1000']]}
            if endpoint == 'exchangeInfo':
                return {'symbols': [rules]}
            if endpoint == 'fundingRate':
                return []
            if endpoint == 'klines':
                span = paper.SIGNAL_INTERVALS[params['interval']]
                end = now // span * span
                # A successful provider response can still lag publication of
                # the latest closed bar. It must not certify a current cache.
                if label in clock['stale_labels']:
                    end -= span
                stop = 0 if label in clock['stale_labels'] else -1
                return [[end - i * span, 100, 101, 99, 100, 1000,
                         end - (i - 1) * span - 1]
                        for i in range(500 if label == 'signal' else 12, stop, -1)]
            raise AssertionError(endpoint)

    worker = paper.StockAccount(MemoryPath(store), FakeVenue(), 'MUUSDT', cfg)
    return worker, clock, calls, store


def requested_signal(calls, timeframe):
    return any(endpoint == 'klines' and params['interval'] == timeframe
               for endpoint, params in calls)


@pytest.mark.parametrize('timeframe', ['15m', '1h', '4h'])
@pytest.mark.parametrize('force_signal', [False, True])
def test_cross_signal_boundary_retries_next_poll(monkeypatch, timeframe, force_signal):
    worker, clock, calls, _ = worker_fixture(monkeypatch, timeframe)
    span = paper.SIGNAL_INTERVALS[timeframe]
    responses = worker.market_data(force_rules=True, force_signal=force_signal)
    assert requested_signal(calls, timeframe)
    clock['now'] = BOUNDARY + 1_000
    result = worker.step(market_data=responses)
    assert result['risk_monitor']['status'] == 'healthy'
    assert worker.signal_bars[-1][0] + span == BOUNDARY - span
    assert worker.candle_bucket == (BOUNDARY - 2_000) // FIVE
    calls.clear()
    clock['now'] = BOUNDARY + 31_000
    worker.step(market_data=worker.market_data(force_rules=True))
    assert any(endpoint == 'exchangeInfo' for endpoint, _ in calls)
    assert requested_signal(calls, timeframe)
    assert worker.signal_bars[-1][0] + span == BOUNDARY
    assert worker.candle_bucket == BOUNDARY // FIVE


def test_cross_five_minute_boundary_does_not_certify_future_bucket(monkeypatch):
    worker, clock, calls, _ = worker_fixture(monkeypatch)
    clock['now'] = BOUNDARY + FIVE - 2_000
    responses = worker.market_data()
    clock['now'] = BOUNDARY + FIVE + 1_000
    worker.step(market_data=responses)
    assert worker.signal_bars[-1][0] + HOUR == BOUNDARY
    assert worker.five[-1][0] + FIVE == BOUNDARY
    assert worker.candle_bucket == BOUNDARY // FIVE
    calls.clear()
    clock['now'] = BOUNDARY + FIVE + 31_000
    worker.step()
    assert requested_signal(calls, '1h')
    assert worker.five[-1][0] + FIVE == BOUNDARY + FIVE
    assert worker.candle_bucket == (BOUNDARY + FIVE) // FIVE


@pytest.mark.parametrize('timeframe', ['15m', '1h', '4h'])
@pytest.mark.parametrize('stale_label', ['signal', 'five'])
def test_http_success_with_stale_completed_snapshot_retries(monkeypatch, timeframe, stale_label):
    worker, clock, calls, _ = worker_fixture(monkeypatch, timeframe)
    clock.update(now=BOUNDARY + 1_000, stale_labels={stale_label})
    result = worker.step()
    assert 'signal' not in result['errors'] and 'five' not in result['errors']
    assert worker.candle_bucket is None
    calls.clear()
    clock.update(now=BOUNDARY + 31_000, stale_labels=set())
    worker.step()
    assert requested_signal(calls, timeframe)
    assert worker.candle_bucket == BOUNDARY // FIVE


@pytest.mark.parametrize('failure', ['clock', 'invalid_clock', 'signal', 'five', 'missing_five'])
def test_failed_forced_refresh_invalidates_previously_certified_bucket(monkeypatch, failure):
    worker, clock, calls, _ = worker_fixture(monkeypatch)
    clock['now'] = BOUNDARY + 1_000
    worker.step()
    assert worker.candle_bucket == BOUNDARY // FIVE
    clock.update(now=BOUNDARY + 11_000,
                 invalid_clock=failure == 'invalid_clock',
                 failed_labels={failure} if failure in {'clock', 'signal', 'five'} else set())
    responses = worker.market_data(force_signal=True)
    if failure == 'missing_five':
        responses = [row for row in responses if row[0] != 'five']
    worker.step(market_data=responses)
    assert worker.candle_bucket is None
    calls.clear()
    clock.update(now=BOUNDARY + 31_000, invalid_clock=False, failed_labels=set())
    worker.step()
    assert requested_signal(calls, '1h')
    assert worker.candle_bucket == BOUNDARY // FIVE


def test_healthy_same_bucket_keeps_cache_and_never_admits_forming_bar(monkeypatch):
    worker, clock, calls, store = worker_fixture(monkeypatch)
    clock['now'] = BOUNDARY + 1_000
    worker.step()
    calls.clear()
    clock['now'] = BOUNDARY + 31_000
    worker.step(market_data=worker.market_data(force_rules=True))
    assert not requested_signal(calls, '1h')
    assert worker.signal_bars[-1][0] + HOUR == BOUNDARY
    assert all(int(row[6]) < clock['now'] for row in worker.signal_bars)
    assert store['state.json']['fill_count_total'] == 0


@pytest.mark.parametrize('failure', ['clock', 'signal', 'five', 'missing_five'])
def test_cache_failure_does_not_suppress_fresh_mark_protective_stop(monkeypatch, failure):
    worker, clock, _, store = worker_fixture(monkeypatch)
    clock.update(now=BOUNDARY + 1_000,
                 failed_labels={failure} if failure != 'missing_five' else set())
    state = store['state.json']
    state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
        'entry_time': BOUNDARY - HOUR, 'signal_time': BOUNDARY - 2 * HOUR,
        'initial_stop': 98, 'stop': 101, 'entry_fee': .0125, 'margin': 10, 'funding': 0}
    state['position_history'] = [{'time_ms': BOUNDARY - HOUR, 'signed_qty': 1}]
    responses = worker.market_data()
    if failure == 'missing_five':
        responses = [row for row in responses if row[0] != 'five']
    result = worker.step(market_data=responses)
    assert worker.candle_bucket is None
    assert result['risk_monitor']['status'] == ('degraded' if failure == 'clock' else 'healthy')
    assert result['risk_monitor']['protection_fill_available']
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
