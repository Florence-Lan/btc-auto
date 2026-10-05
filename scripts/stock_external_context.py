"""Historical public hourly market context, with explicit reconstructed availability."""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import requests

from stock_swing_signals import _ema

HOUR = 3_600_000
DELAY = 20 * 60_000
SYMBOLS = {'nq': 'NQ=F', 'es': 'ES=F', 'sox': '^SOX', 'vix': '^VIX',
           'dollar': 'DX-Y.NYB', 'yield10': '^TNX'}
MAX_AGE = {'nq': 90 * 60_000, 'es': 90 * 60_000, 'sox': 18 * HOUR,
           'vix': 24 * HOUR, 'dollar': 24 * HOUR, 'yield10': 24 * HOUR}
OVERLAYS = ('price_only', 'nq_confirmation', 'nq_sector_risk_confirmation')


def snapshot_from_raw(directory, cutoff_ms, output):
    if output.exists():
        raise FileExistsError('Preserve the archived external data')
    manifest = json.loads((directory / 'fetch_manifest.json').read_text())
    series, coverage = {}, {}
    for source in manifest['sources']:
        name = source['name']
        path = directory / 'raw' / (name + '.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        result = json.loads(path.read_text()).get('chart', {}).get('result')
        if source['status'] != 200 or not result:
            coverage[name] = {'usable_rows': 0, 'error': 'source_unavailable'}
            series[name] = []
            continue
        chart = result[0]
        if chart.get('meta', {}).get('dataGranularity') != '1h':
            raise ValueError('Hourly data required: ' + name)
        quotes = chart['indicators']['quote'][0]
        rows, dropped = [], Counter()
        for timestamp, close in zip(chart.get('timestamp', []), quotes.get('close', [])):
            if close is None:
                dropped['missing_close'] += 1
                continue
            if not math.isfinite(close) or close <= 0:
                raise ValueError('Invalid external close: ' + name)
            start = int(timestamp) * 1000
            available = start + HOUR + DELAY
            if available > cutoff_ms:
                dropped['not_available_at_cutoff'] += 1
                continue
            rows.append([start, available, float(close)])
        times = [r[1] for r in rows]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('Duplicate or misordered provider timestamps: ' + name)
        series[name] = rows
        coverage[name] = {'usable_rows': len(rows), 'first_available_ms': times[0] if times else None,
            'last_available_ms': times[-1] if times else None, 'dropped': dict(dropped),
            'provider_meta': {k: chart['meta'].get(k) for k in ('symbol', 'instrumentType', 'exchangeName', 'exchangeTimezoneName')}}
    if set(series) != set(SYMBOLS):
        raise ValueError('Every declared source must have an explicit coverage record')
    payload = {'metadata': {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'provider': 'Yahoo Finance chart endpoint', 'symbols': SYMBOLS,
        'estimated_availability': 'Provider hourly opening timestamp +1hour +20minutes assumed delay; incomplete hourly/daily bars cannot enter earlier decisions.',
        'actual_historical_first_seen_available': False,
        'research_only': True, 'raw_fetch_manifest': manifest,
        'limitations': ['Downloaded latest historical vintage, not actual archived first-seen historical quotes.',
            'NQ/ES provider front-contract series: individual contract roll and adjustment history is not reconstructed.',
            'Missing values remain missing; no interpolation, weekend quotes, spot NASDAQ or ETF substitutions are invented.',
            'Cash-market indices can remain closed while futures trade; freshness bounds are separate for each instrument.',
            'Yahoo historical bars can be revised, delayed or irregular; +20min is an explicit research assumption, not proof of original publication time.']},
        'series': series, 'coverage': coverage}
    with gzip.open(output, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))
    return payload


class MarketContext:
    def __init__(self, payload):
        self.series = {}
        for name, rows in payload['series'].items():
            if name not in MAX_AGE:
                raise ValueError('Unknown declared context source')
            if any(r[1] != r[0] + HOUR + DELAY for r in rows):
                raise ValueError('External hourly close must honor the declared delay')
            times = [r[1] for r in rows]
            if any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError('Context available timestamps must increase')
            values = [r[2] for r in rows]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('Context values must be finite and positive')
            self.series[name] = {'rows': rows, 'times': times, 'values': values,
                'ema20': _ema(values, 20), 'ema50': _ema(values, 50)}

    def at(self, name, timestamp):
        source = self.series.get(name)
        if not source:
            return None, name + '_missing'
        index = bisect.bisect_right(source['times'], timestamp) - 1
        if index < 55:
            return None, name + '_insufficient_history'
        asof = source['times'][index]
        if timestamp - asof > MAX_AGE[name]:
            return None, name + '_stale'
        values = source['values']
        fast, slow, prior_slow = source['ema20'][index], source['ema50'][index], source['ema50'][index - 6]
        if fast is None or slow is None or prior_slow is None:
            return None, name + '_insufficient_history'
        change6 = values[index] / values[index - 6] - 1
        direction = (1 if fast > slow and slow > prior_slow and change6 > 0 else
                     -1 if fast < slow and slow < prior_slow and change6 < 0 else 0)
        return {'available_ms': asof, 'provider_open_ms': source['rows'][index][0],
                'age_ms': timestamp - asof, 'close': values[index], 'ema20': fast,
                'ema50': slow, 'return6_observations': change6,
                'return3_observations': values[index] / values[index - 3] - 1,
                'direction': direction}, None

    def decision(self, timestamp, direction, overlay):
        if overlay not in OVERLAYS or isinstance(direction, bool) or direction not in (-1, 1):
            raise ValueError('Invalid external overlay or trade direction')
        required = () if overlay == OVERLAYS[0] else ('nq',) if overlay == OVERLAYS[1] else tuple(SYMBOLS)
        details, reasons = {}, []
        for name in required:
            point, error = self.at(name, timestamp)
            if error:
                reasons.append(error)
            else:
                details[name] = point
        for name in ('nq', 'es', 'sox'):
            if name in details and details[name]['direction'] != direction:
                reasons.append(name + '_trend_disagrees')
        vix = details.get('vix')
        if vix and (vix['close'] > 25 or vix['return3_observations'] > .1):
            reasons.append('vix_risk_spike')
        dollar, yields = details.get('dollar'), details.get('yield10')
        if direction == 1 and dollar and yields and dollar['return6_observations'] > .002 and yields['return6_observations'] > .005:
            reasons.append('dollar_and_yield_rise_veto_long')
        return {'allowed': not reasons, 'reasons': reasons, 'factors': details,
                'decision_at_ms': timestamp, 'direction': direction, 'overlay': overlay}


def fetch_raw(directory, start_ms, end_ms):
    if directory.exists():
        raise FileExistsError('Use an unused raw archive directory')
    raw = directory / 'raw'
    raw.mkdir(parents=True)
    def fetch(item):
        name, symbol = item
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + requests.utils.quote(symbol, safe='')
        response = requests.get(url, params={'period1': start_ms // 1000, 'period2': end_ms // 1000,
            'interval': '1h'}, headers={'User-Agent': 'btc-auto-public-research/1.0'}, timeout=25)
        content = response.content
        (raw / (name + '.json')).write_bytes(content)
        return {'name': name, 'symbol': symbol, 'url': url, 'status': response.status_code,
                'sha256': hashlib.sha256(content).hexdigest(),
                'fetched_at_utc': datetime.now(timezone.utc).isoformat()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        sources = list(pool.map(fetch, SYMBOLS.items()))
    (directory / 'fetch_manifest.json').write_text(json.dumps({'start_seconds': start_ms // 1000,
        'end_seconds': end_ms // 1000, 'sources': sources}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--start-utc', default='2026-02-01T00:00:00+00:00')
    parser.add_argument('--end-utc', default='2026-10-04T12:00:00+00:00')
    parser.add_argument('--from-existing-raw', action='store_true')
    args = parser.parse_args()
    start = int(datetime.fromisoformat(args.start_utc.replace('Z', '+00:00')).timestamp() * 1000)
    end = int(datetime.fromisoformat(args.end_utc.replace('Z', '+00:00')).timestamp() * 1000)
    if not args.from_existing_raw:
        fetch_raw(args.archive_dir, start, end)
    payload = snapshot_from_raw(args.archive_dir, end, args.archive_dir / 'market_context.json.gz')
    print(json.dumps(payload['coverage'], ensure_ascii=False, indent=2))
