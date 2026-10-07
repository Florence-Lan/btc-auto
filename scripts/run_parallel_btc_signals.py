"""Collect latest six-factor inputs and publish isolated BTC paper signals.

This producer never executes targets. The four-account runner alone writes the
BTC execution ledger; old signal and execution histories remain independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from binance_terminal_client import BinanceTerminalClient
import decision_runtime
import information_runtime
from process_lock import exclusive_process_lock
import run_parallel_simulation as parallel
from trading_execution import write_json


def signal_args(plan):
    account = next(a for a in plan['accounts'] if a['account_id'] == 'btc')
    report = parallel.ROOT / account['strategy_report_path']
    return SimpleNamespace(
        factor_profile=parallel.ROOT / account['strategy_path'],
        factor_snapshot=parallel.ROOT / 'data/snapshots/multifactor_latest.json.gz',
        state_path=report.with_name('state.json'), report_path=report,
        trades_path=report.with_name('trades.csv'))


def evaluate(args, client):
    clock = decision_runtime.resolve_clock(client)
    args.signal_asof_ms = clock['time_ms']
    # Use the original forward engine and gates, including first-seen inputs.
    completed = subprocess.run(information_runtime.paper_command(args),
        cwd=parallel.ROOT, timeout=240)
    if completed.returncode:
        raise RuntimeError(f'BTC signal evaluation exited {completed.returncode}')
    report = json.loads(args.report_path.read_text(encoding='utf-8'))
    point = report.get('execution_target') or report.get('summary', {}).get('last_equity_point') or {}
    timestamp = int(point.get('time_ms') or 0)
    if timestamp < clock['time_ms'] - 900_000:
        raise ValueError('BTC signal report is older than fifteen minutes')
    return timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError('Poll interval must be at least five seconds')
    plan = json.loads(parallel.PLAN.read_text(encoding='utf-8'))
    if plan['execution_mode'] != 'simulation' or plan['live_orders_allowed'] is not False:
        raise ValueError('Only order-disabled simulation plans are accepted')
    config = signal_args(plan)
    root = config.report_path.parent
    with exclusive_process_lock(root / 'producer.lock'):
        profile = json.loads(config.factor_profile.read_text(encoding='utf-8'))
        information_runtime.start_worker(config.factor_snapshot,
            parallel.ROOT / profile['event_snapshot'],
            parallel.ROOT / 'data/snapshots/supplemental_market_latest.json')
        client = BinanceTerminalClient()
        last_bar = None
        status = {'pid': os.getpid(), 'running': True, 'places_orders': False,
            'plan_id': plan['plan_id'], 'profile_path': str(config.factor_profile),
            'profile_sha256': hashlib.sha256(config.factor_profile.read_bytes()).hexdigest(),
            'report_path': str(config.report_path), 'evaluations': 0}
        while True:
            try:
                clock = decision_runtime.resolve_clock(client)
                # Wait three seconds after the five-minute close.
                bar = (clock['time_ms'] - 3000) // 300_000
                if bar != last_bar:
                    status.update(status='evaluating', updated_at_utc=parallel.arithmetic.iso(clock['time_ms']))
                    write_json(root / 'status.json', status)
                    timestamp = evaluate(config, client)
                    last_bar = bar
                    status.update(signal_time_ms=timestamp, evaluations=status['evaluations'] + 1)
                status.update(status='healthy', error=None)
            except Exception as exc:
                status.update(status='degraded', error=decision_runtime._error(exc))
                print(status['error'], flush=True)
            status.update(updated_at_utc=parallel.arithmetic.iso(parallel.now_ms()))
            if args.once:
                status['running'] = False
            write_json(root / 'status.json', status)
            if args.once:
                return 0 if status['status'] == 'healthy' else 1
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
