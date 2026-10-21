"""Isolated prospective stock macro-entry gate comparison; public GETs only."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import gzip
import json
import os
from pathlib import Path
import signal
import threading
import time

from process_lock import exclusive_process_lock
import run_parallel_simulation as paper
import run_sndk_4h_gate_shadow as shared
import stock_macro_event_risk as macro

ROOT = paper.ROOT
DEFAULT_CONFIG = ROOT / 'config/stock_macro_shadow_20261021.json'
SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')
ARMS = shared.ARMS


def source_hashes():
    pending, result = [Path(__file__).resolve()], {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(ROOT).as_posix()
        if relative in result:
            continue
        raw = path.read_bytes()
        result[relative] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            modules = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in modules:
                local = ROOT / 'scripts' / (name.split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(result.items()))


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = paper.read_json(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'SIMULATION'
            or spec.get('places_orders') is not False or spec.get('symbols') != list(SYMBOLS)
            or spec.get('initial_balance_usdt_each') != 1000
            or spec.get('cost_multipliers') != {'normal': 1, 'double': 2}
            or spec.get('historical_warmup_counts_as_forward_performance') is not False
            or spec.get('automatic_strategy_promotion') is not False
            or spec.get('review_minimum_elapsed_days') != 14
            or spec.get('review_minimum_closed_positions_each') != 10
            or spec.get('candidate_only_added_rule') != macro.POLICY_KEY):
        raise ValueError('Invalid isolated stock macro experiment design')
    output = (ROOT / spec['output_dir']).resolve()
    output.relative_to((ROOT / 'data/paper_trading').resolve())
    if output == (ROOT / 'data/paper_trading').resolve():
        raise ValueError('An isolated stock macro output is required')
    snapshot = (output / 'current_context/public_context.json').relative_to(ROOT).as_posix()
    rules = {}
    for symbol in SYMBOLS:
        baseline = paper.stock_config({'stock_research_profile': spec['baseline_profile']}, symbol)
        if (baseline.get('signal_timeframe') != '1h' or baseline.get('entry_direction') != 'both'
                or baseline.get('swing_confirmation_policy') != paper.swing_confirmation.POLICY
                or baseline.get('risk_fraction_per_trade') != .0025
                or not paper.stock_event_risk.enabled(baseline)
                or macro.enabled(baseline)):
            raise ValueError('Baseline must retain original stock swing, risk and earnings rules')
        rules[symbol] = {}
        for arm in ARMS:
            config = deepcopy(baseline)
            if arm.startswith('candidate_'):
                config[macro.POLICY_KEY] = {**macro.POLICY, 'snapshot_path': snapshot,
                    'effective_from_ms': spec['effective_from_ms']}
                macro.validate(config)
            factor = spec['cost_multipliers'][arm.rsplit('_', 1)[1]]
            for key in ('taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'):
                config[key] *= factor
            rules[symbol][arm] = config
    profile_path = ROOT / spec['baseline_profile']
    base_path = ROOT / paper.read_json(profile_path)['base_config']
    fingerprint = {'experiment_config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'profile_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (profile_path, base_path)}, 'source_sha256': source_hashes(),
        'rules_sha256': {symbol: {arm: shared.digest(config) for arm, config in arms.items()}
                         for symbol, arms in rules.items()}}
    return spec, rules, fingerprint


def prepare(output, spec, rules, fingerprint, inception_ms, require_existing=False):
    path = output / 'manifest.json'
    if path.exists():
        manifest = paper.read_json(path)
        if (manifest.get('fingerprint') != fingerprint or manifest.get('rules') != rules
                or manifest.get('design') != spec or manifest.get('mode') != 'SIMULATION'
                or manifest.get('places_orders') is not False):
            raise ValueError('Frozen stock macro experiment changed; no silent reset allowed')
        for symbol in SYMBOLS:
            for arm in ARMS:
                ledger_path = output / symbol / arm / 'state.json'
                performance_path = output / symbol / arm / 'performance.json'
                if not ledger_path.is_file() or not performance_path.is_file():
                    raise ValueError('Existing stock macro history incomplete; refusing reset')
                state = paper.read_json(ledger_path)
                performance = paper.read_json(performance_path)
                if not isinstance(state, dict) or not isinstance(performance, dict):
                    raise ValueError('Existing stock macro history invalid; refusing reset')
                if (state.get('rule') != rules[symbol][arm] or state.get('symbol') != symbol
                        or state.get('mode') != 'simulation' or state.get('places_orders') is not False
                        or state.get('initial_balance') != 1000
                        or state.get('start_ms') != manifest['started_at_ms']
                        or state.get('signal_active_after_ms') != manifest['started_at_ms']
                        or performance.get('start_ms') != manifest['started_at_ms']):
                    raise ValueError('Existing stock macro ledger incompatible; refusing reset')
        return manifest
    if require_existing:
        raise ValueError('Existing stock macro ledgers required')
    if any(p.name != 'runner.lock' for p in output.iterdir()):
        raise ValueError('Incomplete stock macro directory; refusing overwrite')
    if type(inception_ms) is not int or inception_ms <= 0 or inception_ms < spec['effective_from_ms']:
        raise ValueError('Verified post-activation exchange inception required')
    manifest = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'experiment_id': spec['experiment_id'], 'started_at_ms': inception_ms,
        'started_at_utc': paper.arithmetic.iso(inception_ms), 'design': spec,
        'rules': rules, 'fingerprint': fingerprint, 'original_accounts_modified': False,
        'automatic_strategy_promotion': False, 'historical_warmup_counts_as_forward_performance': False,
        'forward_validated': False, 'sole_strategy_difference': macro.POLICY_KEY,
        'comparison_basis': 'Per-symbol shared market observations and one immutable public-calendar snapshot per tick; independently funded and sized normal/double cost arms.'}
    for symbol in SYMBOLS:
        for arm in ARMS:
            state = paper.initial_stock(symbol, inception_ms, rules[symbol][arm])
            state.update(signal_active_after_ms=inception_ms, experiment_arm=arm,
                         experiment_id=spec['experiment_id'])
            paper.write_json(output / symbol / arm / 'state.json', state)
            paper.write_json(output / symbol / arm / 'performance.json', {
                'start_ms': inception_ms, 'peak_equity': 1000., 'max_sampled_drawdown_pct': 0.,
                'equity_samples': 0, 'last_sample_utc': None})
    paper.write_json(path, manifest)
    return manifest


def _entry_permit_deadlines(decision, source):
    """Bound one checked permit without re-reading either calendar repeatedly.

    Earnings coverage expires at its deadline. Public source freshness retains
    the collector's inclusive one-hour endpoint. Both event starts are closed.
    """
    def stamp(key, payload=decision):
        value = payload.get(key)
        if type(value) is not int or value < 0:
            raise ValueError('stock_macro_combined_permit_metadata_invalid:' + source)
        return value

    if type(decision.get('enabled')) is not bool or type(decision.get('allowed')) is not bool:
        raise ValueError('stock_macro_combined_permit_metadata_invalid:' + source)
    if not decision['enabled']:
        return []
    checked = stamp('checked_at_ms')
    if decision.get('status') == 'not_active_yet':
        return [(stamp('effective_from_ms'), False, source + '_activation_recheck_required')]
    if decision.get('status') != 'clear' or 'next_event' not in decision:
        raise ValueError('stock_macro_combined_permit_metadata_invalid:' + source)
    deadlines = [(stamp('valid_until_ms'), source == 'stock_macro_event',
                  source + ('_source_expired_during_evaluation' if source == 'stock_macro_event'
                            else '_calendar_expired')),
                 (stamp('coverage_end_ms'), source == 'stock_macro_event',
                  source + '_calendar_coverage_expired')]
    upcoming = decision['next_event']
    if upcoming is not None:
        if not isinstance(upcoming, dict):
            raise ValueError('stock_macro_combined_permit_metadata_invalid:' + source)
        start = stamp('starts_at_ms' if source == 'stock_macro_event' else 'blackout_start_ms', upcoming)
        if start <= checked:
            raise ValueError('stock_macro_combined_permit_metadata_invalid:' + source)
        deadlines.append((start, False, source + '_blackout'))
    return deadlines


class MacroAccount(paper.StockAccount):
    def record_stock_event_risk(self, state, phase, reference_ms=None):
        # The baseline follows its original company-calendar-only path exactly.
        if not macro.enabled(self.config):
            state['stock_macro_event_risk'] = None
            return super().record_stock_event_risk(state, phase, reference_ms)
        outcome = macro.evaluate(self.config, self.symbol,
            self.now_ms() if reference_ms is None else reference_ms,
            root=ROOT, reference_clock=self.now_ms)
        outcome['phase'] = phase
        state['stock_macro_event_risk'] = outcome
        earnings = super().record_stock_event_risk(state, phase, reference_ms)
        final_ms = self.now_ms()
        reasons = list(dict.fromkeys(earnings['reasons'] + outcome['reasons']))
        allowed = earnings['allowed'] and outcome['allowed']
        if outcome.get('enabled') is not True:
            reasons.append('stock_macro_combined_permit_metadata_invalid:stock_macro_event')
            allowed = False
        deadlines = []
        if allowed:
            try:
                if type(final_ms) is not int or final_ms < 0:
                    raise ValueError('stock_macro_combined_reference_clock_invalid')
                for source, decision in (('stock_event', earnings), ('stock_macro_event', outcome)):
                    deadlines.extend(_entry_permit_deadlines(decision, source))
                    if decision['enabled'] and final_ms < decision['checked_at_ms']:
                        raise ValueError('stock_macro_combined_reference_clock_invalid')
                reasons.extend(reason for deadline, inclusive, reason in deadlines
                               if final_ms > deadline or (final_ms == deadline and not inclusive))
            except (ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
                reasons.append(reason if reason.startswith('stock_macro_combined_')
                               else 'stock_macro_combined_permit_metadata_invalid')
            allowed = not reasons
        blackout_reasons = {'stock_event_blackout', 'stock_macro_event_blackout',
                            'stock_event_post_event_bars_unavailable'}
        status = ('clear' if allowed else 'blackout'
                  if reasons and all(reason in blackout_reasons for reason in reasons) else 'unavailable')
        combined = {**earnings, 'enabled': True,
            'allowed': bool(allowed), 'status': status, 'reasons': list(dict.fromkeys(reasons)),
            'checked_at_ms': final_ms, 'macro': outcome,
            'scope': 'new_risk_only', 'protective_exits_allowed': True,
            'permit_deadlines': [{'at_ms': deadline, 'inclusive': inclusive, 'reason': reason}
                                 for deadline, inclusive, reason in deadlines]}
        state['stock_event_risk'] = combined
        return combined


def freeze_context(output):
    path = output / 'current_context/public_context.json'
    try:
        raw = (ROOT / 'data/snapshots/public_context_latest.json').read_bytes()
        source = json.loads(raw)
        # Preserve the actual calendars, events and receipt clocks. Unrelated
        # news/quotes and obsolete coverage polls are not needed for this tick.
        payload = {'schema_version': source.get('schema_version'),
            'generated_at_utc': source.get('generated_at_utc'),
            'calendars': source.get('calendars'),
            'events': [row for row in source.get('events', []) if row.get('category') == 'scheduled_macro'],
            'coverage_checks': source.get('coverage_checks', [])[-1:],
            'original_source_sha256': hashlib.sha256(raw).hexdigest()}
        raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (OSError, ValueError, UnicodeError, TypeError, AttributeError):
        # Replace the prior tick: a failed read cannot silently reuse old permission.
        raw = b'{}'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_bytes(raw)
    temporary.replace(path)
    digest = hashlib.sha256(raw).hexdigest()
    archive = output / 'context_archive' / (digest + '.json.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        # Content-addressed evidence makes every logged admission reproducible.
        with archive.open('xb') as handle:
            handle.write(gzip.compress(raw, mtime=0))
    return digest


def publish(output, manifest, accounts, clocks, snapshot_hash, running, poll_seconds):
    now = int(time.time() * 1000)
    comparisons = {}
    for symbol, arms in accounts.items():
        comparisons[symbol] = {}
        for cost in ('normal', 'double'):
            baseline, candidate = (arms.get(prefix + '_' + cost, {}) for prefix in ('baseline', 'candidate'))
            if baseline.get('metrics') and candidate.get('metrics'):
                comparisons[symbol][cost] = {
                    'candidate_minus_baseline_equity_usdt': candidate['equity'] - baseline['equity'],
                    'candidate_minus_baseline_closed_net_usdt': candidate['metrics']['closed_net_pnl_usdt'] - baseline['metrics']['closed_net_pnl_usdt']}
    flat = {symbol + '_' + arm: value for symbol, arms in accounts.items() for arm, value in arms.items()}
    status = {'mode': 'SIMULATION', 'places_orders': False, 'pid': os.getpid(), 'running': running,
        'status': 'healthy' if len(flat) == 12 and all(a.get('status') == 'healthy' for a in flat.values()) else 'degraded',
        'experiment_id': manifest['experiment_id'], 'started_at_ms': manifest['started_at_ms'],
        'started_at_utc': manifest['started_at_utc'], 'heartbeat_at_ms': now,
        'heartbeat_at_utc': paper.arithmetic.iso(now), 'updated_at_utc': paper.arithmetic.iso(now),
        'poll_seconds': poll_seconds, 'accounts': flat, 'by_symbol': accounts, 'clock': clocks,
        'public_context_sha256': snapshot_hash, 'comparisons': comparisons,
        'original_accounts_modified': False, 'automatic_strategy_promotion': False,
        'forward_validated': False, 'review': {'elapsed_days': max(0, now - manifest['started_at_ms']) / 86400000,
            'minimum_elapsed_days': 14, 'minimum_closed_positions_each': 10,
            'interpretation': 'First review scale per symbol/arm; no automatic promotion or profitability claim.'}}
    paper.write_json(output / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--require-existing', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError('Polling must be at least five seconds')
    spec, rules, fingerprint = load_design(args.config)
    output = (args.output_dir or ROOT / spec['output_dir']).resolve()
    if output != (ROOT / spec['output_dir']).resolve():
        raise ValueError('Stock macro output must match the frozen design')
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with exclusive_process_lock(output / 'runner.lock'):
        venue = shared.SharedPublicVenue(paper.PublicAster())
        paper.now_ms, clock = shared.exchange_clock(venue)
        manifest = prepare(output, spec, rules, fingerprint, paper.now_ms(), args.require_existing)
        workers = {symbol: {arm: MacroAccount(output / symbol / arm / 'state.json', venue, symbol, rules[symbol][arm])
                            for arm in ARMS} for symbol in SYMBOLS}
        accounts = {symbol: {arm: {'status': 'starting', 'errors': {}} for arm in ARMS} for symbol in SYMBOLS}
        clocks, context_hash = {'initial': clock}, None
        publish(output, manifest, accounts, clocks, context_hash, not args.once, args.poll_seconds)
        try:
            while not stop.is_set():
                context_hash = freeze_context(output)
                for symbol in SYMBOLS:
                    accounts[symbol], clocks[symbol] = shared.tick(workers[symbol], venue)
                    for arm, worker in workers[symbol].items():
                        state = paper.read_json(worker.path)
                        accounts[symbol][arm]['stock_macro_event_risk'] = state.get('stock_macro_event_risk')
                        accounts[symbol][arm]['stock_event_risk'] = state.get('stock_event_risk')
                status = publish(output, manifest, accounts, clocks, context_hash,
                                 not args.once and not stop.is_set(), args.poll_seconds)
                print(json.dumps({key: status[key] for key in ('experiment_id', 'pid', 'status', 'heartbeat_at_utc')}), flush=True)
                if args.once:
                    break
                stop.wait(args.poll_seconds)
        finally:
            publish(output, manifest, accounts, clocks, context_hash, False, args.poll_seconds)


if __name__ == '__main__':
    main()
