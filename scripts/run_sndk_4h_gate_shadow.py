"""Prospective isolated SNDK closed-4h admission on/off paper experiment."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import Future
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from process_lock import exclusive_process_lock
import run_parallel_simulation as paper


DEFAULT_CONFIG = paper.ROOT / 'config/sndk_4h_gate_shadow_20261012.json'
ARMS = ('baseline_normal', 'candidate_normal', 'baseline_double', 'candidate_double')
HOUR = 3_600_000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def source_hashes():
    """Freeze the local execution dependency closure, excluding environment/secrets."""
    pending = [Path(__file__).resolve()]
    result = {}
    while pending:
        path = pending.pop()
        key = path.relative_to(paper.ROOT).as_posix()
        if key in result:
            continue
        raw = path.read_bytes()
        result[key] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                local = paper.ROOT / 'scripts' / (module.split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(result.items()))


def load_design(config_path=DEFAULT_CONFIG):
    path = Path(config_path).resolve()
    spec = paper.read_json(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'SIMULATION'
            or spec.get('places_orders') is not False or spec.get('symbol') != 'SNDKUSDT'
            or spec.get('initial_balance_usdt_each') != 1000
            or spec.get('candidate_only_removed_rule') != 'swing_confirmation_policy'
            or spec.get('cost_multipliers') != {'normal': 1, 'double': 2}
            or spec.get('historical_warmup_counts_as_forward_performance') is not False
            or spec.get('automatic_strategy_promotion') is not False):
        raise ValueError('Invalid frozen simulation experiment design')
    profile_path = (paper.ROOT / spec['baseline_profile']).resolve()
    profile = paper.read_json(profile_path)
    base_path = (paper.ROOT / profile['base_config']).resolve()
    baseline = paper.stock_config({'stock_research_profile': spec['baseline_profile']}, 'SNDKUSDT')
    if (baseline.get('swing_confirmation_policy') != paper.swing_confirmation.POLICY
            or baseline.get('signal_timeframe') != '1h'
            or baseline.get('signal_family') != 'donchian20'
            or baseline.get('entry_direction') != 'both'
            or baseline.get('entry_signal_validity_minutes') != 60
            or baseline.get('entry_timing_policy', {}).get('require_breakout_valid_at_fill') is not True
            or baseline.get('risk_fraction_per_trade') != .0025):
        raise ValueError('Baseline must preserve the authorized hourly SNDK execution rules')
    rules = {}
    for arm in ARMS:
        cfg = deepcopy(baseline)
        if arm.startswith('candidate_'):
            cfg.pop('swing_confirmation_policy')
        factor = spec['cost_multipliers'][arm.rsplit('_', 1)[1]]
        for key in ('taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'):
            cfg[key] *= factor
        rules[arm] = cfg
    fingerprint = {
        'experiment_config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'profile_sha256': {p.relative_to(paper.ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (profile_path, base_path)},
        'source_sha256': source_hashes(),
        'rules_sha256': {arm: digest(cfg) for arm, cfg in rules.items()},
    }
    return spec, rules, fingerprint


def prepare(root, spec, rules, fingerprint, inception_ms, require_existing=False):
    """No silent reset, rule migration, or historical decision reactivation."""
    path = root / 'manifest.json'
    states = {arm: root / arm / 'state.json' for arm in ARMS}
    if path.exists():
        manifest = paper.read_json(path)
        if (manifest.get('fingerprint') != fingerprint or manifest.get('rules') != rules
                or manifest.get('experiment_id') != spec['experiment_id']
                or manifest.get('mode') != 'SIMULATION' or manifest.get('places_orders') is not False):
            raise ValueError('Frozen experiment changed; refusing incompatible restart or balance reset')
        for arm, state_path in states.items():
            if not state_path.is_file():
                raise ValueError('Incomplete existing experiment; refusing to reset a missing ledger')
            state = paper.read_json(state_path)
            if not isinstance(state, dict):
                raise ValueError('Existing paper ledger invalid; refusing balance reset')
            if (state.get('mode') != 'simulation' or state.get('places_orders') is not False
                    or state.get('rule') != rules[arm] or state.get('symbol') != 'SNDKUSDT'
                    or state.get('initial_balance') != 1000
                    or state.get('start_ms') != manifest['started_at_ms']
                    or state.get('signal_active_after_ms') != manifest['started_at_ms']):
                raise ValueError('Existing paper ledger incompatible; refusing balance reset')
            performance_path = state_path.with_name('performance.json')
            if not performance_path.is_file():
                raise ValueError('Missing existing performance history; refusing experiment reset')
            performance = paper.read_json(performance_path)
            if not isinstance(performance, dict):
                raise ValueError('Existing performance history invalid; refusing experiment reset')
            if performance.get('start_ms') != manifest['started_at_ms']:
                raise ValueError('Existing performance history incompatible; refusing experiment reset')
        return manifest
    if require_existing:
        raise ValueError('Existing experiment required; refusing fresh ledger creation')
    if any(p.name != 'runner.lock' for p in root.iterdir()):
        raise ValueError('Incomplete experiment directory; refusing to overwrite prior artifacts')
    if isinstance(inception_ms, bool) or not isinstance(inception_ms, int) or inception_ms <= 0:
        raise ValueError('A verified exchange UTC inception is required')
    manifest = {
        'schema_version': 1, 'experiment_id': spec['experiment_id'], 'mode': 'SIMULATION',
        'places_orders': False, 'original_accounts_modified': False, 'forward_validated': False,
        'started_at_ms': inception_ms, 'started_at_utc': paper.arithmetic.iso(inception_ms),
        'initial_balance_each': 1000, 'arms': list(ARMS), 'rules': rules,
        'fingerprint': fingerprint, 'design': spec,
        'sole_strategy_difference': 'candidate removes swing_confirmation_policy',
        'comparison_basis': 'Identical shared public clock/book/mark/funding/5m liquidity and completed 1h signal snapshots; independently sized risk at normal and doubled costs.',
        'historical_warmup_counts_as_forward_performance': False,
        'automatic_strategy_promotion': False,
    }
    for arm, state_path in states.items():
        state = paper.initial_stock('SNDKUSDT', inception_ms, rules[arm])
        state['signal_active_after_ms'] = inception_ms
        state['experiment_arm'] = arm
        paper.write_json(state_path, state)
        paper.write_json(state_path.with_name('performance.json'), {
            'start_ms': inception_ms, 'peak_equity': 1000., 'max_sampled_drawdown_pct': 0.,
            'equity_samples': 0, 'last_sample_utc': None})
    paper.write_json(path, manifest)
    return manifest


def exchange_clock(venue):
    before = time.monotonic()
    payload = venue.get('time', {})
    after = time.monotonic()
    timestamp, received, roundtrip = paper.verified_public_clock(payload, before, after)
    clock = lambda: timestamp + int((time.monotonic() - received) * 1000)
    return clock, {'source': 'public_exchange_time_with_monotonic_elapsed',
                   'observed_at_ms': timestamp, 'roundtrip_ms': int(roundtrip * 1000),
                   'local_clock_offset_ms': timestamp - int(time.time() * 1000)}


def start_experiment(root, spec, rules, fingerprint, venue, require_existing=False,
                     prepare_experiment=None):
    """Resume verified ledgers through clock outages; never invent inception time."""
    prepare_experiment = prepare_experiment or prepare
    # Validate financial history before handling any recoverable public-data
    # failure. Missing/corrupt/incompatible ledgers remain fatal.
    manifest = (prepare_experiment(root, spec, rules, fingerprint, None, True)
                if require_existing or (root / 'manifest.json').exists() else None)
    try:
        clock, clock_info = exchange_clock(venue)
    except (paper.requests.RequestException, OSError, paper.subprocess.SubprocessError,
            ValueError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            message = str(exc)
            prefix = 'Aster public time HTTP '
            if not (message == 'Aster public cooldown active'
                    or message.startswith(prefix) and message[len(prefix):].isdigit()):
                raise
        if manifest is None:
            # New experiments still require a verified exchange inception.
            raise
        # Raw network messages may include credentials embedded in proxy URLs.
        clock_info = {'source': 'unavailable', 'error': type(exc).__name__,
                      'diagnostics': paper.safe_transport_error(exc)}
    else:
        paper.now_ms = clock
    if manifest is None:
        manifest = prepare_experiment(root, spec, rules, fingerprint, paper.now_ms())
    # With no exchange clock, the normal tick blocks new risk, independently
    # checks fresh marks for protection, and retries shared time next poll.
    return manifest, clock_info


class SharedPublicVenue:
    """Share funding mark fallbacks too; cache errors and immutable copies per tick."""
    def __init__(self, venue):
        self.venue = venue
        self.lock = threading.Lock()
        self.cache = {}

    def begin_tick(self):
        with self.lock:
            self.cache = {}

    def get(self, endpoint, params=None):
        if endpoint not in paper.PublicAster.allowed:
            raise ValueError('Only public market endpoints allowed')
        key = (endpoint, json.dumps(params or {}, sort_keys=True))
        with self.lock:
            owner = key not in self.cache
            future = self.cache.setdefault(key, Future())
        if owner:
            try:
                future.set_result(self.venue.get(endpoint, params))
            except Exception as exc:
                future.set_exception(exc)
        return deepcopy(future.result())


def checked_snapshot(responses):
    """Validate shared causal candles, and let the paper engine manage fresh exits."""
    data = {label: (payload, error) for label, payload, error in responses}
    payload, error = data.get('clock', (None, 'No common exchange clock'))
    clock_info = {'source': 'unavailable'}
    timestamp = None
    if not error:
        try:
            timestamp = payload['serverTime']
            received = float(payload['_received_monotonic'])
            elapsed = time.monotonic() - received
            if (isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0
                    or not math.isfinite(received) or not 0 <= elapsed <= 15):
                raise ValueError('Invalid or stale common exchange clock')
            paper.now_ms = lambda: timestamp + int((time.monotonic() - received) * 1000)
            clock_info = {'source': 'public_exchange_time_with_monotonic_elapsed',
                          'observed_at_ms': timestamp, 'observation_age_ms': int(elapsed * 1000)}
        except (ValueError, KeyError, TypeError) as exc:
            data['clock'] = (None, str(exc))
            timestamp = None
    else:
        data['clock'] = (None, error or 'No common exchange clock')
    if timestamp is not None:
        for label, interval in (('signal', HOUR), ('five', paper.FIVE_MINUTES)):
            rows, candle_error = data.get(label, (None, 'Missing completed candles'))
            if candle_error:
                continue
            try:
                if not isinstance(rows, list) or not rows:
                    raise ValueError('Empty candle response')
                closed = []
                for row in rows:
                    if int(row[0]) % interval or int(row[6]) != int(row[0]) + interval - 1:
                        raise ValueError('Invalid candle alignment/closing time')
                    if int(row[6]) < timestamp:
                        candle = paper.arithmetic.candle(row)
                        if (not all(math.isfinite(v) for v in (candle.open, candle.high, candle.low,
                                                               candle.close, candle.volume))
                                or candle.low <= 0 or candle.volume < 0
                                or not candle.low <= min(candle.open, candle.close)
                                <= max(candle.open, candle.close) <= candle.high):
                            raise ValueError('Invalid completed candle prices/volume')
                        closed.append(row)
                expected = timestamp // interval * interval - interval
                if not closed or int(closed[-1][0]) != expected:
                    raise ValueError('Latest completed candle unavailable')
                if any(int(b[0]) - int(a[0]) != interval for a, b in zip(closed, closed[1:])):
                    raise ValueError('Nonconsecutive completed candles')
                data[label] = (closed, None)
            except (ValueError, TypeError, IndexError, KeyError) as exc:
                data[label] = (None, str(exc))
    return [(label, payload, error) for label, (payload, error) in data.items()], clock_info


def read_rows(path):
    if not path.exists():
        return []
    # Malformed journals are not silently excluded from financial metrics.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def closed_metrics(root, state):
    """Group partial exit legs by original position; incorporate late funding."""
    performance_path = root / 'performance.json'
    performance = (paper.read_json(performance_path) if performance_path.exists() else {
        'start_ms': state.get('start_ms'), 'peak_equity': 1000.,
        'max_sampled_drawdown_pct': 0., 'equity_samples': 0, 'last_sample_utc': None})
    # The engine samples risk before fills. Also retain after-fill fee/slippage
    # drawdown without changing its execution/risk state or the frozen rules.
    peak = max(performance['peak_equity'], state.get('peak_equity', 1000.), state.get('equity', 1000.))
    sampled_dd = max(0., 1 - state.get('equity', 1000.) / peak) * 100 if peak > 0 else 100.
    performance.update(peak_equity=peak, max_sampled_drawdown_pct=max(
        performance['max_sampled_drawdown_pct'], state.get('max_drawdown_pct', 0.), sampled_dd))
    if state.get('updated_at_utc') and state['updated_at_utc'] != performance['last_sample_utc']:
        performance['equity_samples'] += 1
        performance['last_sample_utc'] = state['updated_at_utc']
    paper.write_json(performance_path, performance)
    groups = {}
    for row in read_rows(root / 'closed_trades.jsonl'):
        key = row['entry_time_ms']
        group = groups.setdefault(key, {'entry_time_ms': key, 'closed_at_ms': None,
            'gross_pnl': 0., 'fees': 0., 'funding_debit': 0., 'net_pnl': 0., 'exit_legs': 0})
        group['gross_pnl'] += row['gross_pnl']
        group['fees'] += row['allocated_entry_fee'] + row['fee']
        group['funding_debit'] += row['allocated_funding_debit']
        group['net_pnl'] += row['net_pnl']
        group['exit_legs'] += 1
        if row['position_closed']:
            group['closed_at_ms'] = row['time_ms']
    unallocated = 0.
    for event in read_rows(root / 'funding.jsonl'):
        debit = event.get('closed_funding_debit', 0.)
        if not debit:
            continue
        candidates = [g for g in groups.values() if g['entry_time_ms'] < int(event['fundingTime'])
                      and (g['closed_at_ms'] is None or int(event['fundingTime']) <= g['closed_at_ms'])]
        if len(candidates) != 1:
            unallocated += debit
            continue
        candidates[0]['funding_debit'] += debit
        candidates[0]['net_pnl'] -= debit
    closed = [g for g in groups.values() if g['closed_at_ms'] is not None]
    nets = [g['net_pnl'] for g in closed]
    wins = [v for v in nets if v > 0]
    losses = [v for v in nets if v < 0]
    net = sum(nets)
    return {'closed_positions': len(closed), 'net_wins': len(wins), 'net_losses': len(losses),
        'net_flat': len(nets) - len(wins) - len(losses),
        'net_win_rate_pct': len(wins) / len(nets) * 100 if nets else None,
        'closed_net_pnl_usdt': net,
        'closed_gross_pnl_usdt': sum(g['gross_pnl'] for g in closed),
        'closed_fees_usdt': sum(g['fees'] for g in closed),
        'closed_funding_pnl_usdt': -sum(g['funding_debit'] for g in closed),
        'best_closed_net_pnl_usdt': max(wins) if wins else None,
        'closed_net_excluding_best_winner_usdt': net - max(wins) if wins else net,
        'net_profit_factor': sum(wins) / -sum(losses) if losses else None,
        'max_sampled_drawdown_pct': performance['max_sampled_drawdown_pct'],
        'equity_samples': performance['equity_samples'],
        'fees_paid_all_usdt': state.get('fees_paid', 0.),
        'funding_pnl_all_usdt': state.get('funding_pnl', 0.),
        'open_or_partial_positions_excluded': sum(g['closed_at_ms'] is None for g in groups.values()),
        'unallocated_late_closed_funding_debit_usdt': unallocated,
        'grouped_positions': list(groups.values())}


def tick(workers, shared_venue):
    shared_venue.begin_tick()
    try:
        common = workers['baseline_normal'].market_data(force_rules=True, force_signal=True)
    except Exception as exc:
        common = [(label, None, type(exc).__name__ + ': ' + str(exc)[:160])
                  for label in ('clock', 'mark', 'book', 'funding', 'five', 'rules', 'signal')]
    common, clock_info = checked_snapshot(common)
    result = {}
    for arm, worker in workers.items():
        try:
            state = worker.step(deepcopy(common))
        except Exception as exc:
            state = paper.read_json(worker.path)
            state.update(status='degraded', errors={'worker': type(exc).__name__ + ': ' + str(exc)[:160]})
            paper.journal(worker.path.with_name('errors.jsonl'), {'time_utc': paper.arithmetic.iso(paper.now_ms()),
                          **state['errors']})
        summary = {k: state.get(k) for k in ('equity', 'wallet_balance', 'return_pct', 'position_qty',
            'fill_count_total', 'fees_paid', 'funding_pnl', 'max_drawdown_pct', 'status', 'errors',
            'signal_status', 'signal_timeframe', 'signal_family', 'entry_blockers', 'entry_checks',
            'updated_at_utc', 'checked_at_utc', 'swing_confirmation', 'stock_entry_timing',
            'last_signal_time_ms', 'signal_active_after_ms')}
        try:
            summary['metrics'] = closed_metrics(worker.path.parent, state)
        except (ValueError, KeyError, TypeError) as exc:
            summary['status'] = 'degraded'
            summary.setdefault('errors', {})['metrics'] = str(exc)[:160]
        heartbeat_ms = int(time.time() * 1000)
        summary.update(heartbeat_at_ms=heartbeat_ms, heartbeat_at_utc=paper.arithmetic.iso(heartbeat_ms))
        result[arm] = summary
    return result, clock_info


def publish_status(root, manifest, accounts, clock_info, poll_seconds, running):
    heartbeat_ms = int(time.time() * 1000)
    comparisons = {}
    for cost in ('normal', 'double'):
        base, candidate = accounts['baseline_' + cost], accounts['candidate_' + cost]
        if base.get('metrics') and candidate.get('metrics'):
            comparisons[cost] = {
                'candidate_minus_baseline_closed_net_usdt': candidate['metrics']['closed_net_pnl_usdt']
                    - base['metrics']['closed_net_pnl_usdt'],
                'candidate_minus_baseline_equity_usdt': (candidate.get('equity') or 1000)
                    - (base.get('equity') or 1000),
                'candidate_minus_baseline_drawdown_percentage_points': candidate['metrics']['max_sampled_drawdown_pct']
                    - base['metrics']['max_sampled_drawdown_pct']}
    status = {
        'mode': 'SIMULATION', 'places_orders': False, 'pid': os.getpid(), 'running': running,
        'status': 'degraded' if any(a.get('status') != 'healthy' for a in accounts.values()) else 'healthy',
        'forward_validated': False, 'automatic_strategy_promotion': False,
        'original_accounts_modified': False, 'experiment_id': manifest['experiment_id'],
        'started_at_utc': manifest['started_at_utc'], 'started_at_ms': manifest['started_at_ms'],
        'updated_at_utc': paper.arithmetic.iso(paper.now_ms()),
        'heartbeat_at_ms': heartbeat_ms, 'heartbeat_at_utc': paper.arithmetic.iso(heartbeat_ms),
        'poll_seconds': poll_seconds, 'clock': clock_info, 'accounts': accounts, 'comparisons': comparisons,
        'review': {'elapsed_days': max(0., paper.now_ms() - manifest['started_at_ms']) / 86_400_000,
                   'minimum_elapsed_days': manifest['design']['review_minimum_elapsed_days'],
                   'minimum_closed_positions_each': manifest['design']['review_minimum_closed_positions_each'],
                   'interpretation': 'Prospective sampling; no profitability claim or automatic switch.'}}
    paper.write_json(root / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--require-existing', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError('Polling must be at least five seconds')
    spec, rules, fingerprint = load_design(args.config)
    root = (args.output_dir or paper.ROOT / spec['output_dir']).resolve()
    allowed = (paper.ROOT / 'data/paper_trading').resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError('Use an isolated child directory under data/paper_trading')
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with exclusive_process_lock(root / 'runner.lock'):
        venue = SharedPublicVenue(paper.PublicAster())
        manifest, clock_info = start_experiment(
            root, spec, rules, fingerprint, venue, args.require_existing)
        workers = {arm: paper.StockAccount(root / arm / 'state.json', venue, 'SNDKUSDT', rules[arm])
                   for arm in ARMS}
        accounts = {arm: {'status': 'starting', 'errors': {}} for arm in ARMS}
        publish_status(root, manifest, accounts, clock_info, args.poll_seconds, not args.once)
        try:
            while not stop.is_set():
                accounts, clock_info = tick(workers, venue)
                status = publish_status(root, manifest, accounts, clock_info, args.poll_seconds,
                                        not args.once and not stop.is_set())
                print(json.dumps({k: status[k] for k in ('experiment_id', 'pid', 'status',
                                    'updated_at_utc', 'running', 'comparisons')}), flush=True)
                if args.once:
                    break
                stop.wait(args.poll_seconds)
        finally:
            publish_status(root, manifest, accounts, clock_info, args.poll_seconds, False)


if __name__ == '__main__':
    main()
