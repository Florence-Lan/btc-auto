"""Isolated, public-data SNDK paper comparison with different signal timeframes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

from process_lock import exclusive_process_lock
import run_parallel_simulation as paper


def exchange_clock(venue):
    before = time.monotonic()
    payload = venue.get('time', {})
    after = time.monotonic()
    timestamp = payload['serverTime']
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise ValueError('Invalid public exchange clock')
    if after-before > 5:
        raise ValueError('Public clock roundtrip exceeded five seconds')
    return (lambda: timestamp+int((time.monotonic()-after)*1000)), {
        'source':'public_exchange_time_with_monotonic_elapsed',
        'observed_at_ms':timestamp, 'roundtrip_ms':int((after-before)*1000),
        'local_clock_offset_ms':timestamp-int(time.time()*1000)}


def prepare(root, candidate_profile, control_profile):
    plans = {'candidate':{'stock_research_profile':candidate_profile},
             'control':{'stock_research_profile':control_profile}}
    rules = {arm:paper.stock_config(plan, 'SNDKUSDT') for arm,plan in plans.items()}
    path = root/'manifest.json'
    if path.exists():
        manifest = paper.read_json(path)
        if manifest['rules'] != rules:
            raise ValueError('Use a new directory for changed rules; existing balances are preserved')
        for arm in rules:
            state = paper.read_json(root/arm/'state.json')
            if state['mode'] != 'simulation' or state['places_orders'] is not False or state['rule'] != rules[arm]:
                raise ValueError('Existing paper ledger changed; refusing to reset')
        return manifest
    if any((root/arm/'state.json').exists() for arm in rules):
        raise ValueError('Incomplete comparison; refusing to overwrite accounts')
    start = paper.now_ms()
    manifest = {'started_at_ms':start, 'started_at_utc':paper.arithmetic.iso(start),
        'rules':rules, 'places_orders':False, 'forward_validated':False,
        'original_accounts_modified':False, 'initial_balance_each':1000,
        'comparison_basis':'Fresh flat ledgers with a common forward start, identical public book/mark/funding/5m liquidity responses, arm-specific completed signal candles.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [Path(__file__), *(paper.ROOT/'scripts'/name for name in
               ('run_parallel_simulation.py','stock_mechanism_signals.py','stock_swing_profiles.py',
                'stock_swing_signals.py','stock_profit_exits.py'))]}}
    for arm,cfg in rules.items():
        paper.write_json(root/arm/'state.json',paper.initial_stock('SNDKUSDT', start, cfg))
    paper.write_json(path, manifest)
    return manifest


def tick(workers):
    candidate, control = workers['candidate'],workers['control']
    # Shared execution inputs, separate signal series because the periods differ.
    common = candidate.market_data(force_rules=True, force_signal=True)
    try:
        control_signal = ('signal', control.venue.get('klines', {
            'symbol':'SNDKUSDT', 'interval':control.signal_timeframe, 'limit':500}), None)
    except Exception as exc:
        control_signal = ('signal', None, type(exc).__name__+': '+str(exc)[:160])
    result = {}
    for arm,worker in workers.items():
        responses = deepcopy(common)
        if arm == 'control':
            responses = [r for r in responses if r[0] != 'signal']+[control_signal]
        try:
            state = worker.step(responses)
        except Exception as exc:
            state = paper.read_json(worker.path)
            state.update(status='degraded', errors={'worker':type(exc).__name__+': '+str(exc)[:160]})
            paper.journal(worker.path.with_name('errors.jsonl'), state['errors'])
        result[arm] = {k:state.get(k) for k in ('equity','wallet_balance','return_pct','position_qty',
            'fill_count_total','fees_paid','funding_pnl','max_drawdown_pct','status','errors',
            'signal_status','signal_timeframe','signal_family','entry_blockers','updated_at_utc')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-profile', required=True)
    parser.add_argument('--control-profile', default='config/stock_one_r_half_atr_paper_20261006.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError('Polling must be at least five seconds')
    root = args.output_dir.resolve()
    allowed = (paper.ROOT/'data/paper_trading').resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError('Use a separate child directory under data/paper_trading')
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig, lambda *_:stop.set())
    with exclusive_process_lock(root/'runner.lock'):
        venue = paper.PublicAster()
        paper.now_ms, clock_info = exchange_clock(venue)
        manifest = prepare(root, args.candidate_profile, args.control_profile)
        workers = {arm:paper.StockAccount(root/arm/'state.json', venue, 'SNDKUSDT', cfg)
                   for arm,cfg in manifest['rules'].items()}
        while not stop.is_set():
            try:
                paper.now_ms, clock_info = exchange_clock(venue)
                accounts = tick(workers)
            except Exception as exc:
                accounts = {arm:{'status':'degraded', 'errors':{'exchange_clock':
                    type(exc).__name__+': '+str(exc)[:160]}} for arm in workers}
            status = {'mode':'SIMULATION', 'places_orders':False, 'forward_validated':False,
                'original_accounts_modified':False, 'started_at_utc':manifest['started_at_utc'],
                'updated_at_utc':paper.arithmetic.iso(paper.now_ms()), 'accounts':accounts,
                'poll_seconds':args.poll_seconds, 'running':not args.once and not stop.is_set(),
                'clock':clock_info}
            paper.write_json(root/'status.json', status)
            print(json.dumps(status), flush=True)
            if args.once:
                break
            stop.wait(args.poll_seconds)
        status['running'] = False
        paper.write_json(root/'status.json', status)


if __name__ == '__main__':
    main()
