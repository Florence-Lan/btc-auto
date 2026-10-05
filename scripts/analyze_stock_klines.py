#!/usr/bin/env python3
"""Read-only candle structure review; no strategy selection or execution."""
import argparse
import bisect
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from stock_swing_signals import Candle, compute_indicators

STEP = 300_000
DAY = 86_400_000
PERIODS = {'5m': STEP, '15m': 3 * STEP, '4h': 48 * STEP, '1d': DAY}


def iso(timestamp):
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()


def aggregate(rows, interval, end):
    """Only complete contiguous groups; no interpolation or partial higher bars."""
    groups = defaultdict(list)
    for row in rows:
        groups[int(row[0]) // interval * interval].append(row)
    result = []
    for time, group in sorted(groups.items()):
        if time + interval > end or [int(r[0]) for r in group] != list(range(time, time + interval, STEP)):
            continue
        result.append([time, float(group[0][1]), max(float(r[2]) for r in group),
            min(float(r[3]) for r in group), float(group[-1][4]), sum(float(r[5]) for r in group)])
    return result


def metrics(rows, end, days):
    first = end - days * DAY
    times = [int(r[0]) for r in rows]
    index = bisect.bisect_left(times, first)
    window = rows[index:]
    anchor = float(rows[index - 1][4]) if index else float(rows[0][1])
    changes = [float(r[4]) / float(r[1]) - 1 for r in window]
    last = float(window[-1][4])
    peak = float(window[0][1])
    drawdown = 0
    for row in window:
        peak = max(peak, float(row[4]))
        drawdown = max(drawdown, 1 - float(row[4]) / peak)
    return {'start_utc': iso(int(window[0][0])), 'end_utc_exclusive': iso(end),
        'bars': len(window), 'trade_close_return_pct': (last / anchor - 1) * 100,
        'high': max(float(r[2]) for r in window), 'low': min(float(r[3]) for r in window),
        'zero_volume_5m_pct': sum(float(r[5]) == 0 for r in window) / len(window) * 100,
        'zero_volume_and_flat_5m_pct': sum(float(r[5]) == 0 and float(r[2]) == float(r[3]) for r in window) / len(window) * 100,
        'median_5m_high_low_pct': statistics.median((float(r[2]) / float(r[3]) - 1) * 100 for r in window),
        'p90_5m_high_low_pct': sorted((float(r[2]) / float(r[3]) - 1) * 100 for r in window)[int((len(window) - 1) * .9)],
        'close_sampled_drawdown_pct': drawdown * 100,
        'positive_body_pct': sum(v > 0 for v in changes) / len(changes) * 100}


def run(source_path, tail_path, output, chart_data):
    source = json.loads(gzip.decompress(source_path.read_bytes()))
    tail = json.loads(tail_path.read_text())
    assert tail['parent_snapshot_sha256'] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    end = tail['end_ms_exclusive']
    result = {'source_snapshot': str(source_path), 'source_snapshot_sha256': tail['parent_snapshot_sha256'],
        'tail_sha256': hashlib.sha256(tail_path.read_bytes()).hexdigest(), 'end_utc_exclusive': iso(end),
        'places_orders': False, 'symbols': {}, 'limitations': [
            'Aster perpetual traded candles, not underlying stock venue prices or executable bid/ask quotes.',
            'Five-minute zero-volume candles preserved, including stale carried prices. No interpolation.',
            'Daily and higher bars aggregate UTC-aligned complete groups; labels display Asia/Shanghai.',
            'Trade-path returns are descriptive price changes, not strategy PnL or forecasts.',
            'Historical data already studied; this review does not establish a 60% win rate.']}
    chart = {'end': end, 'symbols': {}, 'periods': {
        '1d': {'label': '日线 · 全部历史', 'ema': [20, 50, 200]},
        '4h': {'label': '4小时 · 近90日', 'ema': [20, 50, 200]},
        '15m': {'label': '15分钟 · 近7日', 'ema': [8, 24, 60]},
        '5m': {'label': '5分钟 · 近2日', 'ema': [8, 24, 60]}}}
    for sym in ['MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT']:
        old = {int(r[0]): r for r in source['symbols'][sym]['trade_5m']}
        additions = tail['symbols'][sym]['trade_5m']
        revised = [int(r[0]) for r in additions if int(r[0]) in old and old[int(r[0])][:7] != r[:7]]
        old.update({int(r[0]): r for r in additions})
        rows = [old[t] for t in sorted(old)]
        assert [int(r[0]) for r in rows] == list(range(int(rows[0][0]), end, STEP)), sym
        item = {'start_utc': iso(int(rows[0][0])), 'end_utc_exclusive': iso(end), 'total_5m_bars': len(rows),
            'overlap_revised_bars': len(revised), 'last_closed_trade_price': float(rows[-1][4]),
            'last_closed_trade_bar_volume': float(rows[-1][5]), 'current_mark': tail['symbols'][sym]['mark_current'],
            'windows': {str(days): metrics(rows, end, days) for days in [1, 7, 30, 90]}, 'timeframes': {}}
        chart_item = {'name': {'MUUSDT': '美光 · MU', 'SNDKUSDT': '闪迪 · SNDK', 'SKHYNIXUSDT': '海力士 · SKHYNIX'}[sym],
                      'data': {}, 'fills': []}
        for period, interval in PERIODS.items():
            bars = aggregate(rows, interval, end)
            ema = chart['periods'][period]['ema']
            cfg = {'ema_fast': ema[0], 'ema_mid': ema[1], 'ema_slow': ema[2]}
            indicators = compute_indicators([Candle(int(r[0]), *r[1:6]) for r in bars], cfg)
            atr = indicators['atr'][-1]
            latest = {k: v[-1] for k, v in indicators.items()}
            recent = [r for r in bars if r[0] >= end - 7 * DAY]
            item['timeframes'][period] = {'complete_bars': len(bars), 'last_bar_open_utc': iso(bars[-1][0]),
                'last_close': bars[-1][4], 'ema_periods': ema, 'indicators': latest,
                'atr_fraction_pct': atr / bars[-1][4] * 100 if atr else None,
                'recent7d_zero_volume_pct': sum(r[5] == 0 for r in recent) / len(recent) * 100 if recent else None}
            days = {'5m': 2, '15m': 7, '4h': 90, '1d': 1000}[period]
            chart_item['data'][period] = [[int(r[0]), *[round(v, 5) for v in r[1:6]],
                *[round(indicators[k][i], 5) if indicators[k][i] is not None else None
                  for k in ['ema_fast', 'ema_mid', 'ema_slow']]]
                for i, r in enumerate(bars) if r[0] >= end - days * DAY]
        name = {'MUUSDT': 'mu', 'SNDKUSDT': 'sndk', 'SKHYNIXUSDT': 'skhynix'}[sym]
        fills = Path(f'data/parallel_simulation/btc_memory_stocks_1000_each_10x_20261005/{name}/fills.jsonl')
        if fills.exists():
            chart_item['fills'] = [{'t': f['time_ms'], 'p': f['price'], 'side': f['side']}
                                  for f in map(json.loads, fills.read_text().splitlines())]
        result['symbols'][sym] = item
        chart['symbols'][sym] = chart_item
        print(sym, json.dumps({'windows': item['windows'], 'timeframes': item['timeframes']}, ensure_ascii=False), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    chart_data.write_text(json.dumps(chart, ensure_ascii=False, separators=(',', ':')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz'))
    parser.add_argument('--tail', type=Path, default=Path('data/research/stock_kline_review_20261005/public_tail.json'))
    parser.add_argument('--output', type=Path, default=Path('data/research/stock_kline_review_20261005/analysis.json'))
    parser.add_argument('--chart-data', type=Path, default=Path('data/research/stock_kline_review_20261005/chart_data.json'))
    args = parser.parse_args()
    run(args.snapshot, args.tail, args.output, args.chart_data)
