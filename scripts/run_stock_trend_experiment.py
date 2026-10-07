#!/usr/bin/env python3
"""Six independent paper ledgers comparing trend exits, without stopping original accounts."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from process_lock import exclusive_process_lock
import hashlib
import json
import os
from pathlib import Path
import signal
import threading

import run_parallel_simulation as paper
import stock_trend_exits as trend

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'config/stock_trend_reversal_experiment_20261006.json'
STOP = threading.Event()


class TrendStockAccount(paper.StockAccount):
    def __init__(self, *args, activated_at_ms, **kwargs):
        super().__init__(*args, **kwargs)
        self.activated_at_ms = activated_at_ms
        trend.validate(self.config['trend_exit_policy'])

    def additional_exit_reason(self, state, pos, data, errors, timestamp):
        outcome = {'status': 'unavailable', 'triggered': False}
        try:
            if data.get('clock') and not errors.get('signal') and self.signal_bars:
                if any(int(r[6]) != int(r[0]) + self.signal_interval_ms - 1 for r in self.signal_bars):
                    raise ValueError('Invalid candle closing time')
                bars = [paper.arithmetic.candle(r) for r in self.signal_bars]
                outcome = trend.evaluate(pos, bars, self.config, timestamp,
                                         self.signal_interval_ms, self.activated_at_ms)
            if pos.get('pending_exit') == 'trend_reversal_exit':
                outcome['status'] = 'exit_pending'
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            # Missing trend observations do not become an additional entry admission gate.
            outcome = {'status': 'unavailable', 'triggered': False, 'reason': str(exc)[:180]}
        state['trend_exit_status'] = {**outcome, 'checked_at_ms': paper.now_ms()}
        if outcome['triggered'] and not pos.get('pending_exit'):
            paper.journal(self.path.with_name('trend_decisions.jsonl'), {
                'observed_at_ms': paper.now_ms(), 'position_direction': pos['direction'], **outcome})
            return 'trend_reversal_exit'
        return None

    def step(self, market_data=None):
        state = super().step(market_data)
        if not state.get('position'):
            state['trend_exit_status'] = {'status': 'flat', 'triggered': False, 'checked_at_ms': paper.now_ms()}
            paper.write_json(self.path, state)
        elif state.get('trend_exit_status', {}).get('status') in (None, 'flat'):
            state['trend_exit_status'] = {'status': 'waiting_for_closed_bar', 'triggered': False, 'checked_at_ms': paper.now_ms()}
            paper.write_json(self.path, state)
        return state


def prepare(profile, root):
    """Copy source snapshots once; restarting never recreates existing accounts."""
    if profile['mode'] != 'simulation' or profile['places_orders'] is not False:
        raise ValueError('Only simulated accounts are allowed')
    trend.validate(profile['trend_exit_policy'])
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = paper.read_json(manifest_path)
        if manifest['profile'] != profile:
            raise ValueError('Use a new experiment ID for different declared rules')
        for account_id in profile['stock_account_ids']:
            for arm in ('candidate', 'control'):
                state = paper.read_json(root / account_id / arm / 'state.json')
                if not state or state['rule'] != manifest['rules'][account_id][arm]:
                    raise RuntimeError('Existing experiment ledger missing or changed; preserving it')
        return manifest
    if any(root.glob('*/candidate/state.json')) or any(root.glob('*/control/state.json')):
        raise RuntimeError('Incomplete experiment exists; refusing to reset its balances')
    plan = paper.read_json(ROOT / profile['source_plan'])
    selected = {a['account_id']: a for a in plan['accounts'] if a['account_id'] in profile['stock_account_ids']}
    if set(selected) != set(profile['stock_account_ids']):
        raise ValueError('Missing source stock accounts')
    snapshots = {id: paper.read_json(ROOT / account['state_path']) for id, account in selected.items()}
    for id, state in snapshots.items():
        if state['mode'] != 'simulation' or state['places_orders'] is not False or state['symbol'] != selected[id]['symbol']:
            raise ValueError('Source must be the matching paper ledger')
    activated = paper.now_ms()
    manifest = {'experiment_id': profile['experiment_id'], 'activated_at_ms': activated,
                'started_at_utc': paper.arithmetic.iso(activated), 'profile': profile,
                'places_orders': False, 'baseline': {}, 'rules': {}, 'symbols': {},
                'comparison_basis': 'Changes from identical activation cash and inventory; same public responses per symbol',
                'original_accounts_modified': False, 'source_hashes_at_activation': {
                    name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
                    for name in ('run_stock_trend_experiment.py', 'stock_trend_exits.py', 'run_parallel_simulation.py',
                                 'stock_swing_signals.py', 'stock_swing_profiles.py', 'stock_profit_exits.py')}}
    for id, source in snapshots.items():
        paper.write_json(root / 'activation_snapshots' / id / 'state.json', source)
        manifest['baseline'][id] = {k: source.get(k) for k in ('equity', 'wallet_balance', 'realized_pnl',
            'fees_paid', 'funding_pnl', 'fill_count_total', 'position_qty')}
        manifest['symbols'][id] = source['symbol']
        manifest['rules'][id] = {}
        for arm in ('control', 'candidate'):
            state = copy.deepcopy(source)
            cfg = state['rule']
            cfg['entry_enabled'] = True
            cfg['entry_qualification'] = {**cfg.get('entry_qualification', {}),
                'approved_for_forward_simulation': True, 'forward_validated': False,
                'reason': '用户授权正常模拟实验，历史盈利筛查不阻止开仓。'}
            if arm == 'candidate':
                cfg['trend_exit_policy'] = copy.deepcopy(profile['trend_exit_policy'])
            else:
                cfg.pop('trend_exit_policy', None)
            # Both arms start accepting only newly closed entry signals; no old target is replayed.
            state.update(signal_active_after_ms=activated, last_signal_time_ms=None,
                pending_signal=None, last_signal_result=None, signal_status='waiting_for_first_forward_candle')
            manifest['rules'][id][arm] = copy.deepcopy(cfg)
            paper.write_json(root / id / arm / 'state.json', state)
    paper.write_json(manifest_path, manifest)
    return manifest


def view(state, baseline):
    keys = ('symbol', 'status', 'checked_at_utc', 'equity', 'wallet_balance', 'realized_pnl',
            'fees_paid', 'funding_pnl', 'position_qty', 'fill_count_total', 'errors',
            'signal_status', 'entry_blockers', 'trend_exit_status', 'position', 'last_mark_price')
    result = {k: state.get(k) for k in keys}
    result['experiment_fill_count'] = state['fill_count_total'] - baseline['fill_count_total']
    result['recent_fills'] = [fill for fill in state['fills'] if fill['sequence'] > baseline['fill_count_total']][-10:]
    return result


def tick(workers, baseline):
    """Read once per stock; either arm's failure cannot stop the other arm."""
    candidate, control = workers['candidate'], workers['control']
    responses = candidate.market_data(force_rules=control.rules is None,
        force_signal=control.candle_bucket != paper.now_ms() // paper.FIVE_MINUTES)
    result = {}
    for arm, worker in workers.items():
        try:
            result[arm] = view(worker.step(responses), baseline)
        except Exception as exc:
            state = paper.read_json(worker.path)
            result[arm] = {**view(state, baseline), 'status': 'degraded',
                'checked_at_utc': paper.arithmetic.iso(paper.now_ms()),
                'errors': {'worker': type(exc).__name__ + ': ' + str(exc)[:180]}}
            paper.journal(worker.path.with_name('errors.jsonl'), result[arm])
    return result


def run(once=False, poll_seconds=30):
    profile = paper.read_json(PROFILE)
    root = ROOT / 'data/paper_trading' / profile['experiment_id']
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(root / 'runner.lock'):
        _run_locked(profile, root, once, poll_seconds)


def _run_locked(profile, root, once, poll_seconds):
    manifest = prepare(profile, root)
    venue = paper.PublicAster()
    workers = {}
    for id in profile['stock_account_ids']:
        workers[id] = {}
        for arm in ('candidate', 'control'):
            path = root / id / arm / 'state.json'
            cfg = manifest['rules'][id][arm]
            cls = TrendStockAccount if arm == 'candidate' else paper.StockAccount
            extra = {'activated_at_ms': manifest['activated_at_ms']} if arm == 'candidate' else {}
            workers[id][arm] = cls(path, venue, manifest['symbols'][id], cfg, **extra)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    accounts = {}
    def publish():
        paper.write_json(root / 'status.json', {'experiment_id': profile['experiment_id'],
            'pid': os.getpid(), 'mode': 'SIMULATION', 'places_orders': False, 'running': not STOP.is_set(),
            'started_at_utc': manifest['started_at_utc'], 'updated_at_utc': paper.arithmetic.iso(paper.now_ms()),
            'poll_seconds': poll_seconds, 'baseline': manifest['baseline'], 'accounts': accounts,
            'forward_validated': False, 'original_accounts_modified': False})
    publish()
    def observe(id):
        try:
            return id, tick(workers[id], manifest['baseline'][id])
        except Exception as exc:
            error = {'worker': type(exc).__name__ + ': ' + str(exc)[:180]}
            result = {}
            for arm, worker in workers[id].items():
                result[arm] = {**view(paper.read_json(worker.path), manifest['baseline'][id]),
                    'status': 'degraded', 'checked_at_utc': paper.arithmetic.iso(paper.now_ms()), 'errors': error}
            paper.journal(root / id / 'errors.jsonl', {'time_ms': paper.now_ms(), 'errors': error})
            return id, result
    with ThreadPoolExecutor(max_workers=3) as pool:
        while not STOP.is_set():
            for id, result in pool.map(observe, workers):
                accounts[id] = result
                publish()
            if once:
                break
            STOP.wait(poll_seconds)
    STOP.set()
    publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError('Poll interval must be at least five seconds')
    run(args.once, args.poll_seconds)
