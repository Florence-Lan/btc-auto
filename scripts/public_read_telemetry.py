"""Optional best-effort PUBLIC-read counters; no requests or execution policy.

Only fixed endpoint/transport/event identifiers and bounded integer metadata are
stored. A missed event is never claimed to be counted. Lock contention, corrupt
state, clock rollback and every ordinary filesystem failure return False.
Counters cover the current and 59 preceding UTC minute buckets, not a server
quota or a complete account of traffic through a network address.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

try:
    from process_lock import exclusive_process_lock
except ModuleNotFoundError:
    from scripts.process_lock import exclusive_process_lock


MAX_BYTES = 256 * 1024
MAX_NUMBER = 2_147_483_647
MAX_TIME_MS = 253_402_300_799_999
MINUTE_MS = 60_000
ENDPOINTS = (
    '/fapi/v1/ping', '/fapi/v1/time', '/fapi/v1/exchangeInfo',
    '/fapi/v1/premiumIndex', '/fapi/v1/klines', '/fapi/v1/fundingRate',
    '/fapi/v1/depth', '/fapi/v1/ticker/price', '/fapi/v1/ticker/bookTicker',
    '/futures/data/globalLongShortAccountRatio',
    '/futures/data/topLongShortPositionRatio', '/futures/data/takerlongshortRatio',
    '/futures/data/openInterestHist',
)
TRANSPORTS = ('requests', 'curl', 'none')
EVENTS = ('network_start', 'response', 'transport_error', 'cooldown_blocked')
STATUS_KEYS = frozenset(('1xx', '2xx', '3xx', '4xx', '5xx', '418', '429'))
DIMENSIONS = {'endpoints': list(ENDPOINTS), 'transports': list(TRANSPORTS), 'events': list(EVENTS)}
TOP_KEYS = frozenset(('schema_version', 'window_minutes', 'window_basis', 'observed_only',
    'complete', 'counts_scope', 'dimensions', 'metric_key_format', 'first_recorded_at_ms',
    'last_recorded_at_ms', 'buckets', 'window_totals', 'window_status_counts',
    'latest_event', 'latest_server_weight'))


def _time(value):
    if type(value) is not int or not 0 <= value <= MAX_TIME_MS:
        raise ValueError('invalid_time')
    return value


def _number(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= MAX_NUMBER:
        raise ValueError('invalid_number')
    return value


def _event(endpoint, transport, event, at_ms, http_status, weight):
    if (type(endpoint) is not str or endpoint not in ENDPOINTS
            or type(transport) is not str or transport not in TRANSPORTS
            or type(event) is not str or event not in EVENTS
            or (transport == 'none' and event != 'cooldown_blocked')):
        raise ValueError('invalid_event')
    _time(at_ms)
    if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
        raise ValueError('invalid_status')
    if type(weight) is str:
        if not re.fullmatch(r'[0-9]{1,10}', weight):
            raise ValueError('invalid_weight')
        weight = int(weight)
    if weight is not None:
        _number(weight)
        if event != 'response':
            raise ValueError('weight_requires_response')
    return {'endpoint': endpoint, 'transport': transport, 'event': event,
            'at_ms': at_ms, 'http_status': http_status, 'used_weight_1m': weight}


def _metric(row):
    return ':'.join(str(values.index(row[key])) for key, values in (
        ('endpoint', ENDPOINTS), ('transport', TRANSPORTS), ('event', EVENTS)))


def _counts(raw, *, statuses=False):
    if type(raw) is not dict or len(raw) > (7 if statuses else len(ENDPOINTS)*len(TRANSPORTS)*len(EVENTS)):
        raise ValueError('invalid_counts')
    for key, value in raw.items():
        _number(value, positive=True)
        if statuses:
            if key not in STATUS_KEYS:
                raise ValueError('invalid_status_counter')
        else:
            if type(key) is not str or not re.fullmatch(r'[0-9]{1,2}:[0-9]:[0-9]', key):
                raise ValueError('invalid_metric')
            a, b, c = map(int, key.split(':'))
            if key != f'{a}:{b}:{c}' or not (a < len(ENDPOINTS) and b < len(TRANSPORTS) and c < len(EVENTS)):
                raise ValueError('invalid_metric')
            if b == 2 and c != 3:
                raise ValueError('invalid_metric')
    return raw


def _totals(buckets, field):
    result = {}
    for bucket in buckets:
        for key, count in bucket[field].items():
            result[key] = _number(result.get(key, 0)+count, positive=True)
    return result


def _new(at_ms):
    return {'schema_version': 1, 'window_minutes': 60,
            'window_basis': 'current_and_59_preceding_UTC_minute_buckets',
            'observed_only': True, 'complete': False,
            'counts_scope': 'best_effort_local_events_not_server_quota_or_total_network_traffic',
            'dimensions': DIMENSIONS, 'metric_key_format': 'endpoint_index:transport_index:event_index',
            'first_recorded_at_ms': at_ms, 'last_recorded_at_ms': at_ms,
            'buckets': [], 'window_totals': {}, 'window_status_counts': {},
            'latest_event': None, 'latest_server_weight': None}


def _validate(payload):
    if type(payload) is not dict or set(payload) != TOP_KEYS:
        raise ValueError('invalid_schema')
    template = _new(0)
    for key in ('schema_version', 'window_minutes', 'window_basis', 'observed_only', 'complete',
                'counts_scope', 'dimensions', 'metric_key_format'):
        if type(payload[key]) is not type(template[key]) or payload[key] != template[key]:
            raise ValueError('invalid_schema')
    first, last = _time(payload['first_recorded_at_ms']), _time(payload['last_recorded_at_ms'])
    if first > last:
        raise ValueError('invalid_clock_order')
    buckets = payload['buckets']
    if type(buckets) is not list or not 1 <= len(buckets) <= 60:
        raise ValueError('invalid_buckets')
    previous = -1
    floor = last//MINUTE_MS*MINUTE_MS
    for row in buckets:
        if type(row) is not dict or set(row) != {'minute_ms', 'counts', 'status_counts'}:
            raise ValueError('invalid_bucket')
        minute = _time(row['minute_ms'])
        if minute % MINUTE_MS or not max(0, floor-59*MINUTE_MS) <= minute <= floor or minute <= previous:
            raise ValueError('invalid_bucket_time')
        previous = minute
        if not _counts(row['counts']):
            raise ValueError('empty_bucket')
        _counts(row['status_counts'], statuses=True)
        responses = sum(count for key,count in row['counts'].items() if key.endswith(':1'))
        statuses = row['status_counts']
        if (sum(count for key,count in statuses.items() if key.endswith('xx')) > responses
                or statuses.get('418',0)+statuses.get('429',0) > statuses.get('4xx',0)):
            raise ValueError('inconsistent_response_counters')
    _counts(payload['window_totals'])
    _counts(payload['window_status_counts'], statuses=True)
    if (payload['window_totals'] != _totals(buckets, 'counts')
            or payload['window_status_counts'] != _totals(buckets, 'status_counts')):
        raise ValueError('inconsistent_totals')
    latest = payload['latest_event']
    if type(latest) is not dict or set(latest) != {'endpoint','transport','event','at_ms','http_status','used_weight_1m'}:
        raise ValueError('invalid_latest')
    if _event(latest['endpoint'], latest['transport'], latest['event'], latest['at_ms'],
              latest['http_status'], latest['used_weight_1m']) != latest or latest['at_ms'] != last:
        raise ValueError('invalid_latest')
    if buckets[-1]['minute_ms'] != floor or not buckets[-1]['counts'].get(_metric(latest)):
        raise ValueError('latest_event_not_counted')
    weight = payload['latest_server_weight']
    if weight is not None:
        if type(weight) is not dict or set(weight) != {'at_ms', 'endpoint', 'transport', 'used_weight_1m'}:
            raise ValueError('invalid_latest_weight')
        _event(weight['endpoint'], weight['transport'], 'response', weight['at_ms'], None, weight['used_weight_1m'])
        _number(weight['used_weight_1m'])
        if not max(first, 0, floor-59*MINUTE_MS) <= weight['at_ms'] <= last:
            raise ValueError('invalid_latest_weight')


def record_event(path: Path, *, endpoint: str, transport: str, event: str, at_ms: int,
                 http_status=None, used_weight_1m=None) -> bool:
    """Attempt one nonblocking, cross-process-safe aggregate update.

    Returns True only after atomic persistence. False is deliberately silent
    and never authorizes a retry, request, cache hit, fill or clock renewal.
    """
    temporary = None
    try:
        row = _event(endpoint, transport, event, at_ms, http_status, used_weight_1m)
        path = Path(path)
        with exclusive_process_lock(path.with_name(path.name+'.lock')):
            if path.exists():
                with path.open('rb') as handle:
                    raw = handle.read(MAX_BYTES+1)
                if not raw or len(raw) > MAX_BYTES:
                    return False
                payload = json.loads(raw.decode('utf-8'))
                _validate(payload)
                if at_ms < payload['last_recorded_at_ms']:
                    return False
            else:
                payload = _new(at_ms)
            minute = at_ms//MINUTE_MS*MINUTE_MS
            earliest = max(0, minute-59*MINUTE_MS)
            buckets = [b for b in payload['buckets'] if b['minute_ms'] >= earliest]
            if not buckets or buckets[-1]['minute_ms'] != minute:
                buckets.append({'minute_ms': minute, 'counts': {}, 'status_counts': {}})
            counter = buckets[-1]['counts']
            key = _metric(row)
            counter[key] = _number(counter.get(key, 0)+1, positive=True)
            if event == 'response' and http_status is not None:
                statuses = buckets[-1]['status_counts']
                for status_key in (f'{http_status//100}xx',) + ((str(http_status),) if http_status in {418,429} else ()):
                    statuses[status_key] = _number(statuses.get(status_key, 0)+1, positive=True)
            payload.update(last_recorded_at_ms=at_ms, buckets=buckets, latest_event=row,
                           window_totals=_totals(buckets, 'counts'),
                           window_status_counts=_totals(buckets, 'status_counts'))
            weight = payload['latest_server_weight']
            if weight is not None and weight['at_ms'] < earliest:
                payload['latest_server_weight'] = None
            if row['used_weight_1m'] is not None:
                payload['latest_server_weight'] = {key: row[key] for key in ('at_ms','endpoint','transport','used_weight_1m')}
            _validate(payload)
            raw = (json.dumps(payload, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8')
            if len(raw) > MAX_BYTES:
                return False
            with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name+'.',
                                             suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
            os.replace(temporary, path)
            return True
    except Exception:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
