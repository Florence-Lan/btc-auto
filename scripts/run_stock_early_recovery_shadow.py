"""Independent prospective early-recovery entry trial. Public GETs; never orders."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import signal
import threading
import time
import json
from types import FunctionType

from process_lock import exclusive_process_lock
import run_parallel_simulation as paper
import run_sndk_4h_gate_shadow as shared
import run_stock_exit_response_shadow as infrastructure
import stock_early_recovery as recovery
import stock_early_recovery_timing as timing
import stock_early_recovery_metrics as metrics

ROOT = paper.ROOT
DEFAULT_CONFIG = ROOT / 'config/stock_early_recovery_shadow_20261028.json'
SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')
ARMS = ('baseline_normal', 'candidate_normal', 'baseline_double', 'candidate_double')
COSTS = ('normal', 'double')
HOUR = 3_600_000
digest = infrastructure.digest
contained = infrastructure.contained
validate_fingerprint = infrastructure.validate_fingerprint
validate_ledger = infrastructure.validate_ledger
validate_financial_journals = infrastructure.validate_financial_journals
LedgerIntegrityError = infrastructure.LedgerIntegrityError
check_source_rules = infrastructure.check_source_rules
freeze_calendar = infrastructure.freeze_calendar
archive_context = infrastructure.archive_context
compact_observations = infrastructure.compact_observations
estimated_close = infrastructure.estimated_close


class RecoveryProfiles:
    def __init__(self, activated_at_ms):
        self.activated_at_ms = activated_at_ms

    def signal_with_diagnostics(self, bars, indicators, index, config):
        if bars[index].time_ms + HOUR <= self.activated_at_ms:
            return None, {'swing_confirmation': {'enabled': False, 'allowed': True, 'reasons': []},
                'signal_before_confirmation': None, 'signal_after_confirmation': None,
                'signal_after_quality': None, 'entry_quality': None,
                'admission': {'stage': 'no_raw_signal', 'reasons': ['signal_not_strictly_post_inception'],
                              'confirmation_rejected_candidate': False}}
        if not timing.is_recovery(config):
            return paper.stock_swing_profiles.signal_with_diagnostics(bars, indicators, index, config)
        prefix = bars[:index + 1]
        sig, detail = recovery.candidate_with_diagnostics(prefix, config,
            asof_ms=prefix[-1].time_ms + HOUR)
        # Preserve raw candidates rejected by structure/extension checks. A
        # quality rejection must never be reported as absence of opportunity.
        detail['swing_confirmation'].update(confirmation_timeframe='4h',
                                           alternative_entry='early_recovery_1h_v1')
        return sig, detail


def isolated_call(method, self, *args):
    """Keep the proven execution code, with dependencies private to THIS call.

    Copy globals at call time so the shared-context journal wrapper and current
    verified clock remain effective. No imported module or production class is
    monkeypatched. Only the signal/timing dependencies differ for candidate arms.
    """
    dependencies = {**method.__globals__, 'stock_swing_profiles': RecoveryProfiles(self.activated_at_ms),
                    'stock_entry_timing': timing}
    original_journal = dependencies['journal']
    def journal(path, row):
        if Path(path).name == 'signals.jsonl':
            seen = self.now_ms()
            row = {**row, 'first_seen_at_ms': seen,
                   'first_seen_lag_ms': seen - row['boundary_ms'],
                   'entry_validity_minutes': 60}
        original_journal(path, row)
    dependencies['journal'] = journal
    cloned = FunctionType(method.__code__, dependencies, method.__name__,
                          method.__defaults__, method.__closure__)
    return cloned(self, *args)


class RecoveryAccount(paper.StockAccount):
    def __init__(self, path, venue, symbol, config, *, activated_at_ms):
        self.activated_at_ms = activated_at_ms
        isolated_call(paper.StockAccount.__init__, self, path, venue, symbol, config)

    def step(self, market_data=None):
        return isolated_call(paper.StockAccount.step, self, market_data)


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = paper.read_json(path)
    expected = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'symbols': list(SYMBOLS), 'variants': ['baseline', 'candidate'],
        'cost_multipliers': {'normal': 1, 'double': 2}, 'initial_balance_usdt_each': 1000,
        'automatic_strategy_promotion': False, 'historical_warmup_counts_as_forward_performance': False,
        'review_minimum_elapsed_days': 14, 'review_minimum_closed_positions_each': 10,
        'candidate_only_added_rule': 'early_recovery_1h_v1', 'poll_seconds': 30}
    if any(spec.get(k) != v or type(spec.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('Invalid fixed early-recovery experiment design')
    if type(spec.get('effective_from_ms')) is not int or spec['effective_from_ms'] <= 0:
        raise ValueError('Explicit prospective declaration time required')
    output = contained(spec['output_dir'], 'data/paper_trading')
    calendar = (output / 'calendar_current.json').relative_to(ROOT).as_posix()
    rules = {}
    for symbol in SYMBOLS:
        baseline = paper.stock_config({'stock_research_profile': spec['baseline_profile']}, symbol)
        if (baseline.get('signal_family') != 'donchian20' or baseline.get('signal_timeframe') != '1h'
                or baseline.get('entry_direction') != 'both'
                or baseline.get('swing_confirmation_policy') != paper.swing_confirmation.POLICY
                or baseline.get('risk_fraction_per_trade') != .0025
                or baseline.get('cooldown_signal_bars') != 3
                or baseline.get('entry_quality_policy') is not None
                or not paper.stock_event_risk.enabled(baseline)):
            raise ValueError('Baseline must retain current stock entry, earnings and risk rules')
        baseline['stock_event_risk_policy'] = {**baseline['stock_event_risk_policy'], 'calendar_path': calendar}
        rules[symbol] = {}
        for arm in ARMS:
            config = deepcopy(baseline)
            if arm.startswith('candidate_'):
                config.update(signal_family='early_recovery_1h_v1',
                    early_recovery_policy=deepcopy(recovery.POLICY),
                    entry_timing_policy=deepcopy(timing.POLICY),
                    candidate_id=symbol.lower() + '_early_recovery_20261028',
                    research_variant='independent_early_recovery_prospective_trial')
                config.pop('swing_confirmation_policy', None)
                recovery.validate(config)
                timing.validate(config)
            multiplier = spec['cost_multipliers'][arm.rsplit('_', 1)[1]]
            for key in ('taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'):
                config[key] *= multiplier
            rules[symbol][arm] = config
    profile = ROOT / spec['baseline_profile']
    base = ROOT / paper.read_json(profile)['base_config']
    fingerprint = {'experiment_config_sha256': digest(path.read_bytes()),
        'profile_sha256': {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in (profile, base)},
        'source_sha256': source_hashes(),
        'rules_sha256': {s: {a: digest(c) for a, c in arms.items()} for s, arms in rules.items()}}
    return spec, rules, fingerprint


def source_hashes():
    pending, result = [Path(__file__).resolve()], {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(ROOT).as_posix()
        if relative in result:
            continue
        raw = path.read_bytes()
        result[relative] = digest(raw)
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT / 'scripts' / (name.split('.')[0] + '.py')
                if candidate.is_file():
                    pending.append(candidate)
    return dict(sorted(result.items()))


def prepare(output, spec, rules, fingerprint, inception_ms, require_existing=False):
    validate_fingerprint(fingerprint, rules)
    path = output / 'manifest.json'
    if path.exists():
        manifest = paper.read_json(path)
        if (manifest.get('fingerprint') != fingerprint or manifest.get('rules') != rules
                or manifest.get('design') != spec or manifest.get('mode') != 'SIMULATION'
                or manifest.get('places_orders') is not False):
            raise ValueError('Frozen early-recovery experiment changed; new generation required')
        for symbol in SYMBOLS:
            for arm in ARMS:
                directory = output / symbol / arm
                if not (directory / 'state.json').is_file() or not (directory / 'performance.json').is_file():
                    raise ValueError('Existing early-recovery history incomplete; refusing reset')
                validate_ledger(paper.read_json(directory / 'state.json'), rules[symbol][arm], symbol,
                                manifest['started_at_ms'])
                validate_financial_journals(directory, paper.read_json(directory / 'state.json'))
                if paper.read_json(directory / 'performance.json').get('start_ms') != manifest['started_at_ms']:
                    raise ValueError('Existing performance epoch incompatible')
        return manifest
    if require_existing:
        raise ValueError('Existing experiment required; explicit initialization is necessary')
    if any(p.name != 'runner.lock' for p in output.iterdir()):
        raise ValueError('Incomplete early-recovery directory; refusing overwrite')
    if type(inception_ms) is not int or inception_ms < spec['effective_from_ms']:
        raise ValueError('Verified post-declaration exchange inception required')
    manifest = {'schema_version': 1, 'experiment_id': spec['experiment_id'], 'mode': 'SIMULATION',
        'places_orders': False, 'started_at_ms': inception_ms, 'started_at_utc': paper.arithmetic.iso(inception_ms),
        'design': spec, 'rules': rules, 'fingerprint': fingerprint,
        'original_accounts_modified': False, 'automatic_strategy_promotion': False,
        'historical_warmup_counts_as_forward_performance': False, 'forward_validated': False,
        'comparison_basis': 'Independent flat 1000-USDT accounts; shared public observations and calendar per symbol batch. Cost and exit paths may diverge.'}
    for symbol in SYMBOLS:
        for arm in ARMS:
            directory = output / symbol / arm
            state = paper.initial_stock(symbol, inception_ms, rules[symbol][arm])
            state.update(signal_active_after_ms=inception_ms, experiment_arm=arm,
                         experiment_id=spec['experiment_id'])
            paper.write_json(directory / 'state.json', state)
            paper.write_json(directory / 'performance.json', {'start_ms': inception_ms, 'peak_equity': 1000.,
                'max_sampled_drawdown_pct': 0., 'equity_samples': 0, 'last_sample_utc': None})
            for journal in ('entries.jsonl', 'fills.jsonl', 'funding.jsonl', 'closed_trades.jsonl'):
                (directory / journal).write_text('', encoding='utf-8')
    paper.write_json(path, manifest)
    return manifest


def tick_symbol(output, symbol, workers, venue, calendar_hash, context):
    venue.begin_tick()
    try:
        force = any(w.candle_bucket != w.now_ms() // paper.FIVE_MINUTES for w in workers.values())
        responses = workers['baseline_normal'].market_data(
            force_rules=any(w.rules is None for w in workers.values()), force_signal=force)
    except Exception as exc:
        responses = [(label, None, type(exc).__name__) for label in
                     ('clock', 'mark', 'book', 'funding', 'five', 'rules', 'signal')]
    responses, clock = shared.checked_snapshot(responses)
    context[symbol] = archive_context(output, symbol, responses, calendar_hash,
        workers['baseline_normal'].rules, workers['baseline_normal'].signal_bars)
    states, result = {}, {}
    for arm, worker in workers.items():
        financial_integrity = True
        try:
            quarantine = worker.path.with_name('quarantine.json')
            if quarantine.exists():
                raise LedgerIntegrityError('Arm quarantined; inspect financial journals before resuming')
            before = paper.read_json(worker.path)
            validate_ledger(before, worker.config, symbol, worker.activated_at_ms)
            validate_financial_journals(worker.path.parent, before)
            state = worker.step(deepcopy(responses))
            validate_ledger(state, worker.config, symbol, worker.activated_at_ms)
            validate_financial_journals(worker.path.parent, state)
        except Exception as exc:
            state = paper.read_json(worker.path)
            financial_integrity = not worker.path.with_name('quarantine.json').exists()
            if isinstance(exc, LedgerIntegrityError):
                financial_integrity = False
                paper.write_json(worker.path.with_name('quarantine.json'), {
                    'observed_at_ms': int(time.time() * 1000), 'error_type': type(exc).__name__,
                    'reason': 'Immutable or financial integrity check failed; explicit repair required'})
            try:
                validate_ledger(state, worker.config, symbol, worker.activated_at_ms)
                validate_financial_journals(worker.path.parent, state)
            except (ValueError, TypeError, KeyError, OSError):
                financial_integrity = False
                paper.write_json(worker.path.with_name('quarantine.json'), {
                    'observed_at_ms': int(time.time() * 1000), 'error_type': type(exc).__name__,
                    'reason': 'Financial state/journal inconsistency; no further arm steps permitted'})
            state = {**state, 'status': 'degraded', 'errors': {'worker': type(exc).__name__ + ': ' + str(exc)[:160]}}
            paper.journal(worker.path.with_name('errors.jsonl'), {
                'time_utc': paper.arithmetic.iso(int(time.time() * 1000)), **state['errors']})
        states[arm] = state
        summary = {k: state.get(k) for k in ('equity', 'wallet_balance', 'return_pct', 'position_qty',
            'fill_count_total', 'fees_paid', 'funding_pnl', 'status', 'errors', 'signal_status',
            'entry_blockers', 'swing_confirmation', 'exit_response_status', 'position', 'checked_at_utc')}
        summary['financial_integrity'] = financial_integrity
        summary['start_ms'] = state['start_ms']
        summary['initial_balance'] = state['initial_balance']
        summary['estimated_close'] = (estimated_close(state, responses, worker.config, worker.now_ms(), worker.rules)
                                      if financial_integrity else {'status': 'unavailable', 'reason': 'financial_integrity'})
        try:
            summary['metrics'] = shared.closed_metrics(worker.path.parent, state) if financial_integrity else None
        except (ValueError, KeyError, TypeError) as exc:
            summary['status'] = 'degraded'
            summary['metrics'] = None
            summary['errors'] = {**summary.get('errors', {}), 'metrics': type(exc).__name__}
        summary['heartbeat_at_ms'] = int(time.time() * 1000)
        result[arm] = summary
    now = int(time.time() * 1000)
    comparisons = {cost: metrics.compare_independent_arms(
        result['baseline_' + cost], result['candidate_' + cost],
        asof_ms=now, inception_ms=workers['baseline_normal'].activated_at_ms) for cost in COSTS}
    return result, comparisons, clock


def publish(output, manifest, accounts, comparisons, clocks, *, running, cycles, calendar_hash):
    now = int(time.time() * 1000)
    flat = {symbol + '_' + arm: value for symbol, arms in accounts.items() for arm, value in arms.items()}
    status = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'experiment_id': manifest['experiment_id'], 'pid': os.getpid(), 'running': running,
        'status': 'healthy' if len(flat) == 12 and all(v.get('status') == 'healthy' and
            v.get('financial_integrity') is True for v in flat.values()) else 'degraded',
        'started_at_ms': manifest['started_at_ms'], 'started_at_utc': manifest['started_at_utc'],
        'heartbeat_at_ms': now, 'heartbeat_at_utc': paper.arithmetic.iso(now), 'poll_seconds': 30,
        'cycles': cycles, 'by_symbol': accounts, 'accounts': flat, 'comparisons': comparisons,
        'clock': clocks, 'calendar_sha256': calendar_hash, 'fingerprint': manifest['fingerprint'],
        'original_accounts_modified': False, 'automatic_strategy_promotion': False, 'forward_validated': False,
        'independent_initial_capital_usdt': 12000, 'original_account_capital_included': False,
        'paired_comparison': False,
        'review': metrics.review_readiness(accounts, asof_ms=now, inception_ms=manifest['started_at_ms'])}
    paper.write_json(output / 'status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--require-existing', action='store_true')
    args = parser.parse_args(argv)
    spec, rules, fingerprint = load_design(args.config)
    if args.check:
        references = check_source_rules(spec, rules)
        print(json.dumps({'validation': 'configuration_fingerprint_and_original_rule_match',
            'mode': 'SIMULATION', 'places_orders': False, 'network_requests_performed': False,
            'output_initialized': False, 'ledger_count': 12, 'source_file_count': len(fingerprint['source_sha256']),
            'fingerprint': fingerprint, 'original_state_paths': references}, ensure_ascii=False))
        return
    output = contained(spec['output_dir'], 'data/paper_trading')
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with exclusive_process_lock(output / 'runner.lock'):
        venue = shared.SharedPublicVenue(paper.PublicAster())
        manifest, clock = shared.start_experiment(output, spec, rules, fingerprint, venue,
            args.require_existing, prepare_experiment=prepare)
        workers = {symbol: {arm: RecoveryAccount(output / symbol / arm / 'state.json', venue, symbol,
            rules[symbol][arm], activated_at_ms=manifest['started_at_ms']) for arm in ARMS} for symbol in SYMBOLS}
        accounts, comparisons, clocks, context = {}, {}, {'initial': clock}, {}
        cycles, calendar_hash = 0, None
        try:
            with compact_observations(output, context):
                while not stop.is_set():
                    calendar_hash = freeze_calendar(output, spec)
                    for symbol in SYMBOLS:
                        accounts[symbol], comparisons[symbol], clocks[symbol] = tick_symbol(
                            output, symbol, workers[symbol], venue, calendar_hash, context)
                    cycles += 1
                    status = publish(output, manifest, accounts, comparisons, clocks,
                        running=not args.once and not stop.is_set(), cycles=cycles, calendar_hash=calendar_hash)
                    print(json.dumps({k: status[k] for k in ('experiment_id', 'pid', 'status', 'heartbeat_at_utc', 'cycles')}), flush=True)
                    if args.once:
                        break
                    stop.wait(30)
        finally:
            publish(output, manifest, accounts, comparisons, clocks, running=False,
                    cycles=cycles, calendar_hash=calendar_hash)


if __name__ == '__main__':
    main()
