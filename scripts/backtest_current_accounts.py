"""Read-only current-account historical diagnostics using public price proxies.

Stocks reuse StockAccount including its current profit exits. Historical quotes
and visible depth are unavailable: zero-spread candle opens and preceding volume
are explicit proxies, never observed fills. BTC's full multifactor result remains
unavailable without first-seen archives; its price-only control is separate.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import requests

import active_strategy
import backtest_execution as btc_engine
import frozen_strategy
import run_parallel_simulation as paper
import simulate_range_swing as sim
import timeseries_execution
from paper_trade_frozen_portfolio import annotate_open_position_fractions

ROOT = Path(__file__).resolve().parents[1]
STEP = 300_000
DAY = 86_400_000


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def public_get(session, base, endpoint, params=None):
    for attempt in range(3):
        try:
            response = session.get(base + endpoint, params=params, timeout=25)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and int(result.get('code', 0)) < 0:
                raise ValueError('Public endpoint error: ' + str(result))
            return result
        except requests.HTTPError as exc:
            if exc.response.status_code not in (500, 502, 503, 504) or attempt == 2:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        time.sleep(attempt + 1)


def candles(session, base, endpoint, symbol, interval, start, end, step):
    rows = {}
    cursor = start
    while cursor < end:
        batch = public_get(session, base, endpoint, {
            'pair' if endpoint == '/indexPriceKlines' else 'symbol': symbol,
            'interval': interval, 'startTime': cursor, 'endTime': end - 1, 'limit': 1500})
        if not isinstance(batch, list) or not batch:
            break
        latest = max(int(row[0]) for row in batch)
        if latest < cursor:
            raise ValueError('Pagination stalled')
        rows.update({int(row[0]): row for row in batch if start <= int(row[0]) and int(row[6]) < end})
        cursor = latest + step
        time.sleep(.3)
    missing = sorted(set(range(start, end, step)) - rows.keys())
    # Refetch source omissions once, without interpolation.
    for first in missing[:20]:
        batch = public_get(session, base, endpoint, {
            'pair' if endpoint == '/indexPriceKlines' else 'symbol': symbol,
            'interval': interval, 'startTime': first, 'endTime': first + step - 1, 'limit': 1})
        rows.update({int(row[0]): row for row in batch if int(row[0]) == first and int(row[6]) < end})
    missing = sorted(set(range(start, end, step)) - rows.keys())
    print(symbol, endpoint, interval, 'bars', len(rows), 'missing', len(missing), flush=True)
    return [rows[t] for t in sorted(rows)], missing


def funding(session, base, symbol, start, end):
    cursor, events = start, {}
    while cursor < end:
        batch = public_get(session, base, '/fundingRate', {
            'symbol': symbol, 'startTime': cursor, 'endTime': end - 1, 'limit': 1000})
        if not isinstance(batch, list):
            raise ValueError('Invalid funding response')
        for event in batch:
            events[json.dumps(event, sort_keys=True)] = event
        if len(batch) < 1000:
            break
        latest = max(int(event['fundingTime']) for event in batch)
        if latest <= cursor:
            raise ValueError('Funding pagination stalled')
        cursor = latest
    return sorted(events.values(), key=lambda e: int(e['fundingTime']))


def download(account, start, end, out, config):
    symbol = account['symbol']
    destination = out / 'inputs' / (symbol + '.json.gz')
    if destination.exists():
        value = json.loads(gzip.decompress(destination.read_bytes()))
        if value['start_ms'] == start and value['end_ms'] == end:
            return value
        raise ValueError('Existing input has different boundaries')
    base = ('https://fapi.binance.com/fapi/v1' if account['account_id'] == 'btc'
            else 'https://fapi.asterdex.com/fapi/v3')
    warmup = start - (45 if account['account_id'] == 'btc' else 7) * DAY
    session = requests.Session()
    exchange = public_get(session, base, '/exchangeInfo')
    rules = next(row for row in exchange['symbols'] if row['symbol'] == symbol)
    value = {'symbol': symbol, 'base': base, 'start_ms': start, 'end_ms': end,
             'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'rules': rules, 'gaps': {}}
    series = [('trade', '/klines', '5m', STEP)]
    if account['account_id'] == 'btc':
        series.append(('hourly', '/klines', '1h', 3_600_000))
    else:
        series.extend([('mark', '/markPriceKlines', '5m', STEP),
                       ('index', '/indexPriceKlines', '5m', STEP)])
        if config['signal_timeframe'] != '5m':
            series.append(('signal', '/klines', config['signal_timeframe'], paper.SIGNAL_INTERVALS[config['signal_timeframe']]))
    for label, endpoint, interval, step in series:
        value[label], value['gaps'][label] = candles(session, base, endpoint, symbol, interval, warmup, end, step)
    value['funding'] = funding(session, base, symbol, warmup, end)
    if not value['funding']:
        raise ValueError('No historical funding returned for ' + symbol)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':')).encode(), mtime=0))
    return value


class MemoryPath:
    def __init__(self, name, contents):
        self.name, self.contents = name, contents

    def read_text(self):
        return self.contents[self.name]

    def with_name(self, name):
        return MemoryPath(name, self.contents)


class HistoricalVenue:
    def __init__(self, base, cache_path=None):
        self.base, self.session = base, requests.Session()
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path and cache_path.exists() else {}

    def get(self, endpoint, params):
        # Used only when an actual funding event lacks its settlement mark.
        if endpoint != 'markPriceKlines' or params.get('interval') != '1m':
            raise ValueError('Unexpected historical venue request')
        key = json.dumps(params, sort_keys=True)
        if key not in self.cache:
            self.cache[key] = public_get(self.session, self.base, '/' + endpoint, params)
            if self.cache_path:
                save(self.cache_path, self.cache)
        return self.cache[key]


def stock_replay(source, cfg, start, end, cost, out):
    if any(source['gaps'].values()):
        return {'status': 'unavailable_incomplete_price_history', 'gaps': source['gaps']}
    cfg = deepcopy(cfg)
    cfg['taker_fee_rate_assumption'] *= cost
    cfg['adverse_slippage_fraction_assumption'] *= cost
    contents = {'state.json': json.dumps(paper.initial_stock(source['symbol'], start, cfg))}
    logs = {'closed_trades.jsonl': [], 'fills.jsonl': [], 'funding.jsonl': [], 'signals.jsonl': []}
    def memory_write(path, value):
        path.contents[path.name] = json.dumps(value)
    def memory_journal(path, row):
        if path.name in logs:
            logs[path.name].append(deepcopy(row))
    original_write, original_journal, original_clock = paper.write_json, paper.journal, paper.now_ms
    original_indicators = paper.signals.compute_indicators
    indicator_cache = {}
    def cached_indicators(bars, config):
        # Sources are frozen within a replay. Preserve the production 500-bar
        # prefix exactly; only repeated calculations of that prefix are cached.
        key = (bars[0].time_ms, bars[-1].time_ms, len(bars), json.dumps(config, sort_keys=True))
        if key not in indicator_cache:
            if len(indicator_cache) >= 8:
                indicator_cache.pop(next(iter(indicator_cache)))
            indicator_cache[key] = original_indicators(bars, config)
        return indicator_cache[key]
    clock = [start]
    worker = paper.StockAccount(MemoryPath('state.json', contents),
        HistoricalVenue(source['base'], out.parent / 'funding_mark_cache' / (source['symbol'] + '.json')), source['symbol'], cfg)
    trade = source['trade']
    signal = source.get('signal', trade)
    signal_times = [int(row[0]) for row in signal]
    trade_times = [int(row[0]) for row in trade]
    marks = {int(row[0]): row for row in source['mark']}
    indexes = {int(row[0]): row for row in source['index']}
    events = source['funding']
    event_times = [int(row['fundingTime']) for row in events]
    equity_path, status_counts = [], {}
    signal_interval = worker.signal_interval_ms
    paper.signals.compute_indicators = cached_indicators
    paper.write_json, paper.journal = memory_write, memory_journal
    paper.now_ms = lambda: clock[0]
    try:
        for i, bar in enumerate(trade):
            timestamp = int(bar[0])
            if not start <= timestamp < end:
                continue
            clock[0] = timestamp
            signal_end = bisect_right(signal_times, timestamp - signal_interval)
            preceding = trade[max(0, i - 500):i]
            closed_signal = signal[max(0, signal_end - 500):signal_end]
            assert preceding and int(preceding[-1][6]) < timestamp
            assert closed_signal and int(closed_signal[-1][6]) < timestamp
            price, mark, index = float(bar[1]), float(marks[timestamp][1]), float(indexes[timestamp][1])
            published = bisect_right(event_times, timestamp)
            recent = events[max(0, published - 100):published]
            last_rate = float(recent[-1]['fundingRate']) if recent else 0.0
            # Scheduled settlement boundary is not a future funding rate input.
            next_funding = event_times[published] if published < len(events) else end + DAY
            volume = float(preceding[-1][5])
            responses = [('clock', {'serverTime': timestamp}, None),
                ('rules', {'symbols': [source['rules']]}, None),
                ('mark', {'time': timestamp, 'markPrice': str(mark), 'indexPrice': str(index),
                          'lastFundingRate': str(last_rate), 'nextFundingTime': next_funding}, None),
                ('book', {'T': timestamp, 'lastUpdateId': timestamp,
                          'bids': [[str(price), str(volume)]], 'asks': [[str(price), str(volume)]]}, None),
                ('signal', closed_signal, None), ('five', preceding, None), ('funding', recent, None)]
            state = worker.step(responses)
            status_counts[state['signal_status']] = status_counts.get(state['signal_status'], 0) + 1
            equity_path.append({'time_ms': timestamp, 'equity': state['equity'], 'quantity': state['position_qty']})
            if len(equity_path) % 1500 == 0:
                print(source['symbol'], 'cost', cost, 'replayed', len(equity_path), 'equity', state['equity'], flush=True)
    finally:
        paper.write_json, paper.journal, paper.now_ms = original_write, original_journal, original_clock
        paper.signals.compute_indicators = original_indicators
    assert len(equity_path) == (end - start) // STEP
    state = json.loads(contents['state.json'])
    pos = state['position']
    terminal_mark = float(marks[end - STEP][4])
    unrealized = pos['direction'] * pos['qty'] * (terminal_mark - pos['entry']) if pos else 0
    equity = state['wallet_balance'] + unrealized
    terminal_trade = float(trade[bisect_left(trade_times, end) - 1][4])
    close_cost = 0
    if pos:
        close_fill = terminal_trade * (1 - pos['direction'] * cfg['adverse_slippage_fraction_assumption'])
        liquidation_equity = state['wallet_balance'] + pos['direction'] * pos['qty'] * (close_fill - pos['entry']) - pos['qty'] * close_fill * cfg['taker_fee_rate_assumption']
        close_cost = equity - liquidation_equity
    equity_path.append({'time_ms': end - 1, 'equity': equity, 'quantity': state['position_qty']})
    peak, drawdown = 1000., 0.
    for point in equity_path:
        peak = max(peak, point['equity'])
        drawdown = max(drawdown, (1 - point['equity'] / peak) * 100)
    trades = logs['closed_trades.jsonl']
    grouped = {}
    for row in trades:
        grouped.setdefault(row['entry_time_ms'], {'net_pnl': 0., 'closed': False})
        grouped[row['entry_time_ms']]['net_pnl'] += row['net_pnl']
        grouped[row['entry_time_ms']]['closed'] |= row['position_closed']
    full_trades = [v['net_pnl'] for v in grouped.values() if v['closed']]
    result = {'status': 'price_proxy_diagnostic', 'symbol': source['symbol'], 'cost_multiplier': cost,
        'summary': {'initial_equity': 1000, 'final_equity': equity, 'net_pnl': equity - 1000,
            'total_return_pct': (equity / 1000 - 1) * 100,
            'estimated_liquidated_return_pct': ((equity - close_cost) / 1000 - 1) * 100,
            'max_drawdown_pct': max(drawdown, state['max_drawdown_pct']),
            'fees': state['fees_paid'], 'funding_pnl': state['funding_pnl'], 'unrealized_pnl': unrealized,
            'fills': state['fill_count_total'], 'closed_positions': len(full_trades),
            'closed_net_win_rate_pct': sum(p > 0 for p in full_trades) / len(full_trades) * 100 if full_trades else None,
            'open_quantity': state['position_qty'], 'observations': len(equity_path) - 1},
        'signal_status_counts': status_counts, 'config': cfg, 'equity_curve': equity_path,
        'closed_trades': trades, 'funding_settlements': logs['funding.jsonl'], 'final_state': state}
    assert abs(state['wallet_balance'] - (1000 + state['realized_pnl'] - state['fees_paid'] + state['funding_pnl'])) < 1e-7
    save(out / f"{source['symbol']}_cost{cost}.json", result)
    return {key: value for key, value in result.items() if key not in ('equity_curve', 'closed_trades', 'funding_settlements', 'final_state')}


def btc_price_control(source, profile, start, end, out):
    if any(source['gaps'].values()):
        return {'status': 'unavailable_incomplete_price_history', 'gaps': source['gaps']}
    manifest, cfg = frozen_strategy.load_frozen_strategy(ROOT / profile['base_manifest'])
    cfg = replace(active_strategy.apply_risk_limits(cfg, profile), max_drawdown_stop_pct=0)
    base = [sim.candle_from_kline(row) for row in source['trade']]
    hourly = [sim.candle_from_kline(row) for row in source['hourly']]
    rates = sim.FundingHistory(times=[int(e['fundingTime']) for e in source['funding']],
                              rates=[float(e['fundingRate']) for e in source['funding']])
    asof = base[-1].close_time_ms
    opening = timeseries_execution.opening_from_base(base, hourly[-1].close_time_ms + 1, asof)
    sleeves = [sim.simulate(base, replace(cfg, strategy_modes=('trend',)), start, None, rates),
               timeseries_execution.build_sleeve(hourly, replace(cfg, strategy_modes=('timeseries_trend',)),
                   start, rates, opening=opening, asof_ms=asof, activation_ms=None)]
    annotate_open_position_fractions(sleeves)
    btc_engine.execution_targets.prepare_sleeves(sleeves)
    results = {}
    for cost in (1, 2):
        report = btc_engine.replay(base, sleeves, cfg, start, source['funding'], initial=1000, cost_multiplier=cost)
        report.update(status='price_only_control_NOT_current_multifactor_backtest', profile=profile,
                      start_utc=paper.arithmetic.iso(start), end_utc_exclusive=paper.arithmetic.iso(end))
        save(out / f'BTCUSDT_price_only_cost{cost}.json', report)
        results[f'cost{cost}'] = report['summary']
        print('BTC PRICE ONLY', cost, json.dumps(report['summary']), flush=True)
    return {'status': 'price_only_control_NOT_current_multifactor_backtest', 'runs': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end-utc', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    end = paper.arithmetic.parse_time(args.end_utc)
    start = end - 30 * DAY
    if end % 3_600_000:
        raise ValueError('Use a completed hour boundary')
    out = args.output_dir.resolve()
    if any((out / name).exists() for name in ('results.json', 'BTCUSDT_price_only_cost1.json')):
        raise FileExistsError('Refusing to overwrite completed replay')
    plan = json.loads((ROOT / 'config/parallel_simulation_plan_20261005.json').read_text(encoding='utf-8'))
    configs = {a['account_id']: paper.stock_config(plan, a['symbol']) for a in plan['accounts'] if a['account_id'] != 'btc'}
    profile = json.loads((ROOT / plan['accounts'][0]['strategy_path']).read_text(encoding='utf-8'))
    paths = ['config/parallel_simulation_plan_20261005.json', plan['stock_research_profile'],
             plan['accounts'][0]['strategy_path'], profile['base_manifest'],
             json.loads((ROOT / plan['stock_research_profile']).read_text(encoding='utf-8'))['base_config']]
    for path in paths:
        save(out / 'inputs' / Path(path).name, json.loads((ROOT / path).read_text(encoding='utf-8')))
    declaration = {'start_utc': paper.arithmetic.iso(start), 'end_utc_exclusive': paper.arithmetic.iso(end),
        'days': 30, 'initial_equity_each': 1000, 'max_leverage': 10, 'plan': plan, 'places_orders': False,
        'btc_current_strategy_result': 'unavailable_missing_first_seen_factor_and_news_archives',
        'limitations': ['Stock quotes use zero-spread trade opens, with depth proxied by previous completed 5m volume; historical order books are unavailable.',
            'Stock checks occur every 5m rather than every 30 seconds; intrabar touches are not reconstructed.',
            'Funding admission uses last published settlement rate, not the unarchived contemporaneous indicative rate.',
            'Current contract filters and configured fees apply historically; historical changes are unverified.',
            'Current rules are applied before their activation; these are retrospective diagnostics, not forward returns.',
            'BTC price-only control excludes six-factor, news and source-health gates and is not the current strategy return.'],
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'scripts').glob('*.py')}}
    save(out / 'declaration.json', declaration)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {a['account_id']: pool.submit(download, a, start, end, out, configs.get(a['account_id'])) for a in plan['accounts']}
        sources, errors = {}, {}
        for key, future in futures.items():
            try:
                sources[key] = future.result()
            except Exception as exc:
                errors[key] = type(exc).__name__ + ': ' + str(exc)
                print(key, errors[key], flush=True)
    save(out / 'download_status.json', {'completed': list(sources), 'errors': errors})
    if args.download_only:
        return
    results = {'declaration': declaration, 'accounts': {}, 'download_errors': errors}
    for account in plan['accounts']:
        key = account['account_id']
        if key not in sources:
            results['accounts'][key] = {'status': 'unavailable', 'reason': errors[key]}
        elif key == 'btc':
            results['accounts'][key] = {'current_strategy_status': declaration['btc_current_strategy_result'],
                                       'price_only_control': btc_price_control(sources[key], profile, start, end, out)}
        else:
            results['accounts'][key] = {f'cost{cost}': stock_replay(sources[key], configs[key], start, end, cost, out) for cost in (1, 2)}
        save(out / 'results.json', results)
        print('DONE', key, flush=True)
    results['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'inputs').iterdir()}
    save(out / 'results.json', results)


if __name__ == '__main__':
    main()
