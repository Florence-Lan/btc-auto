"""Prospective turning/continuation entry comparison; public GETs, never orders."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import threading
import time
from types import FunctionType

from process_lock import exclusive_process_lock
import run_parallel_simulation as paper
import run_sndk_4h_gate_shadow as shared
import run_stock_exit_response_shadow as infrastructure
import run_stock_macro_shadow as macro_runner
import stock_macro_event_risk as macro
import stock_entry_timing_candidates as candidates
import stock_entry_timing_trial_admission as timing
import stock_early_recovery_metrics as metrics

ROOT = paper.ROOT
DEFAULT_CONFIG = ROOT / 'config/stock_entry_timing_trial_20261029.json'
SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')
VARIANTS = ('baseline', 'turning', 'turn_pullback')
COSTS = ('normal', 'double')
ARMS = tuple(v + '_' + c for c in COSTS for v in VARIANTS)
HOUR = 3_600_000
digest = infrastructure.digest
contained = infrastructure.contained
validate_fingerprint = infrastructure.validate_fingerprint
validate_ledger = infrastructure.validate_ledger
validate_financial_journals = infrastructure.validate_financial_journals
LedgerIntegrityError = infrastructure.LedgerIntegrityError
freeze_calendar = infrastructure.freeze_calendar
archive_context = infrastructure.archive_context
compact_observations = infrastructure.compact_observations
estimated_close = infrastructure.estimated_close


class TimingProfiles:
    def __init__(self, activated_at_ms):
        self.activated_at_ms = activated_at_ms

    def signal_with_diagnostics(self, bars, indicators, index, config):
        if bars[index].time_ms + HOUR <= self.activated_at_ms:
            return None, {'swing_confirmation': {'enabled': False, 'allowed': True, 'reasons': []},
                'signal_before_confirmation': None, 'signal_after_confirmation': None,
                'signal_after_quality': None, 'entry_quality': None,
                'admission': {'stage': 'no_raw_signal', 'reasons': ['signal_not_strictly_post_inception'],
                              'confirmation_rejected_candidate': False}}
        return candidates.signal_with_diagnostics(bars, indicators, index, config)


def isolated_call(method, self, *args):
    """Private dependencies; preserve original execution and current journal wrapper."""
    dependencies = {**method.__globals__, 'stock_swing_profiles': TimingProfiles(self.activated_at_ms),
                    'stock_entry_timing': timing}
    original_journal = dependencies['journal']
    def journal(path, row):
        if Path(path).name == 'signals.jsonl':
            seen = self.now_ms()
            row.update(first_seen_at_ms=seen, first_seen_lag_ms=seen - row['boundary_ms'],
                       entry_validity_minutes=60, entry_timing_variant=self.config['entry_timing_variant'],
                       macro_context_sha256=self.macro_context_sha256)
        elif Path(path).name == 'fills.jsonl':
            # Original fill() passes the SAME record stored in state['fills'];
            # adding provenance in place keeps journal/state exact equality.
            state = args[0]
            detail = (state.get('signal_diagnostics') or {}).get('entry_timing_candidate') or {}
            row.update(entry_timing_variant=self.config['entry_timing_variant'],
                       signal_kind=detail.get('signal_kind', 'donchian20'),
                       macro_context_sha256=self.macro_context_sha256)
        original_journal(path, row)
    dependencies['journal'] = journal
    cloned = FunctionType(method.__code__, dependencies, method.__name__,
                          method.__defaults__, method.__closure__)
    return cloned(self, *args)


class TimingAccount(macro_runner.MacroAccount):
    def __init__(self, path, venue, symbol, config, *, activated_at_ms):
        self.activated_at_ms = activated_at_ms
        self.macro_context_sha256 = None
        isolated_call(paper.StockAccount.__init__, self, path, venue, symbol, config)

    def step(self, market_data=None):
        return isolated_call(paper.StockAccount.step, self, market_data)

    def fill(self, state, result, timestamp, direction, reason, trigger_ms=None):
        return isolated_call(paper.StockAccount.fill, self, state, result, timestamp, direction, reason, trigger_ms)


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = paper.read_json(path)
    expected = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'symbols': list(SYMBOLS), 'variants': list(VARIANTS),
        'cost_multipliers': {'normal': 1, 'double': 2}, 'initial_balance_usdt_each': 1000,
        'automatic_strategy_promotion': False, 'historical_warmup_counts_as_forward_performance': False,
        'review_minimum_elapsed_days': 14, 'review_minimum_closed_positions_each': 10,
        'candidate_only_changed_group': 'entry_confirmation_and_fresh_pullback',
        'shared_scheduled_macro_guard': True, 'poll_seconds': 30}
    if any(spec.get(k) != v or type(spec.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('Invalid fixed entry-timing prospective design')
    if type(spec.get('effective_from_ms')) is not int or spec['effective_from_ms'] <= 0:
        raise ValueError('Explicit prospective declaration time required')
    output = contained(spec['output_dir'], 'data/paper_trading')
    calendar = (output / 'calendar_current.json').relative_to(ROOT).as_posix()
    context = (output / 'current_context/public_context.json').relative_to(ROOT).as_posix()
    rules = {}
    for symbol in SYMBOLS:
        baseline = paper.stock_config({'stock_research_profile': spec['baseline_profile']}, symbol)
        if (baseline.get('signal_family') != 'donchian20' or baseline.get('signal_timeframe') != '1h'
                or baseline.get('entry_direction') != 'both'
                or baseline.get('swing_confirmation_policy') != paper.swing_confirmation.POLICY
                or baseline.get('risk_fraction_per_trade') != .0025
                or baseline.get('cooldown_signal_bars') != 3
                or baseline.get('entry_quality_policy') is not None
                or not paper.stock_event_risk.enabled(baseline) or macro.enabled(baseline)):
            raise ValueError('Current stock technical baseline and unchanged risk are required')
        baseline['stock_event_risk_policy'] = {**baseline['stock_event_risk_policy'], 'calendar_path': calendar}
        baseline[macro.POLICY_KEY] = {**macro.POLICY, 'snapshot_path': context,
                                     'effective_from_ms': spec['effective_from_ms']}
        rules[symbol] = {}
        for arm in ARMS:
            variant, cost = arm.rsplit('_', 1)
            config = deepcopy(baseline)
            config['entry_timing_variant'] = variant
            if variant != 'baseline':
                config['entry_timing_candidate_policy'] = deepcopy(candidates.POLICY)
                config['candidate_id'] = symbol.lower() + '_' + variant + '_20261029'
                config['research_variant'] = variant + '_prospective_entry_timing_trial'
            if variant == 'turn_pullback':
                config['entry_timing_policy'] = deepcopy(timing.POLICY)
            candidates.validate(config)
            timing.validate(config)
            macro.validate(config)
            multiplier = spec['cost_multipliers'][cost]
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


from main_stock_rule_revision import declared_main_baseline_matches


def check_source_rules(spec, rules):
    # The new baseline shares a declared macro ENTRY guard with both candidates.
    # Strip only that guard/label and restore the main calendar path for equality.
    plan = paper.read_json(ROOT / spec['source_plan'])
    if plan.get('execution_mode') != 'simulation' or plan.get('live_orders_allowed') is not False:
        raise ValueError('Original main plan must remain simulation only')
    accounts = {a['symbol']: a for a in plan['accounts'] if a['symbol'] in SYMBOLS}
    if set(accounts) != set(SYMBOLS):
        raise ValueError('All main stock accounts are required')
    for symbol, account in accounts.items():
        state = paper.read_json(ROOT / account['state_path'])
        expected = deepcopy(rules[symbol]['baseline_normal'])
        expected.pop(macro.POLICY_KEY)
        expected.pop('entry_timing_variant')
        expected['stock_event_risk_policy']['calendar_path'] = state['rule']['stock_event_risk_policy']['calendar_path']
        if not declared_main_baseline_matches(state, expected) or state.get('places_orders') is not False:
            raise ValueError('Main rules differ from declared technical baseline: ' + symbol)
    return {s: accounts[s]['state_path'] for s in SYMBOLS}


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
            raise ValueError('Frozen entry-timing experiment changed; new generation required')
        for symbol in SYMBOLS:
            for arm in ARMS:
                directory = output / symbol / arm
                if not (directory / 'state.json').is_file() or not (directory / 'performance.json').is_file():
                    raise ValueError('Existing entry-timing history incomplete; refusing reset')
                validate_ledger(paper.read_json(directory / 'state.json'), rules[symbol][arm], symbol,
                                manifest['started_at_ms'])
                validate_financial_journals(directory, paper.read_json(directory / 'state.json'))
                if paper.read_json(directory / 'performance.json').get('start_ms') != manifest['started_at_ms']:
                    raise ValueError('Existing performance epoch incompatible')
        return manifest
    if require_existing:
        raise ValueError('Existing experiment required; explicit initialization is necessary')
    if any(p.name != 'runner.lock' for p in output.iterdir()):
        raise ValueError('Incomplete entry-timing directory; refusing overwrite')
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
            'entry_blockers', 'swing_confirmation', 'stock_macro_event_risk', 'stock_event_risk', 'signal_diagnostics', 'position', 'checked_at_utc')}
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
    pairs = {'turning_vs_baseline': ('baseline', 'turning'),
             'turn_pullback_vs_baseline': ('baseline', 'turn_pullback'),
             'pullback_addition': ('turning', 'turn_pullback')}
    comparisons = {cost: {name: metrics.compare_independent_arms(
        result[left + '_' + cost], result[right + '_' + cost],
        asof_ms=now, inception_ms=workers['baseline_normal'].activated_at_ms)
        for name, (left, right) in pairs.items()} for cost in COSTS}
    return result, comparisons, clock


def publish(output, manifest, accounts, comparisons, clocks, *, running, cycles, calendar_hash, macro_hash=None):
    now = int(time.time() * 1000)
    flat = {symbol + '_' + arm: value for symbol, arms in accounts.items() for arm, value in arms.items()}
    reviews = {}
    for variant in VARIANTS[1:]:
        mapped = {symbol: {prefix + '_' + cost: arms.get((variant if prefix == 'candidate' else 'baseline') + '_' + cost)
                           for cost in COSTS for prefix in ('baseline', 'candidate')}
                  for symbol, arms in accounts.items()}
        reviews[variant] = metrics.review_readiness(mapped, asof_ms=now, inception_ms=manifest['started_at_ms'])
    status = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'experiment_id': manifest['experiment_id'], 'pid': os.getpid(), 'running': running,
        'status': 'healthy' if len(flat) == 18 and all(v.get('status') == 'healthy' and
            v.get('financial_integrity') is True for v in flat.values()) else 'degraded',
        'started_at_ms': manifest['started_at_ms'], 'started_at_utc': manifest['started_at_utc'],
        'heartbeat_at_ms': now, 'heartbeat_at_utc': paper.arithmetic.iso(now), 'poll_seconds': 30,
        'cycles': cycles, 'by_symbol': accounts, 'accounts': flat, 'comparisons': comparisons,
        'clock': clocks, 'calendar_sha256': calendar_hash, 'public_context_sha256': macro_hash,
        'fingerprint': manifest['fingerprint'], 'original_accounts_modified': False,
        'automatic_strategy_promotion': False, 'forward_validated': False,
        'independent_initial_capital_usdt': 18000, 'original_account_capital_included': False,
        'paired_comparison': False, 'review': {'by_variant': reviews,
            'ready_for_first_review': len(flat) == 18 and all(v['ready_for_first_review'] for v in reviews.values()),
            'readiness_is_pass': False, 'minimum_elapsed_days': 14,
            'minimum_closed_positions_each': 10, 'expected_account_count': 18}}
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
            'output_initialized': False, 'ledger_count': 18, 'source_file_count': len(fingerprint['source_sha256']),
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
        workers = {symbol: {arm: TimingAccount(output / symbol / arm / 'state.json', venue, symbol,
            rules[symbol][arm], activated_at_ms=manifest['started_at_ms']) for arm in ARMS} for symbol in SYMBOLS}
        accounts, comparisons, clocks, context = {}, {}, {'initial': clock}, {}
        cycles, calendar_hash, macro_hash = 0, None, None
        try:
            with compact_observations(output, context):
                while not stop.is_set():
                    calendar_hash = freeze_calendar(output, spec)
                    macro_hash = macro_runner.freeze_context(output)
                    for arms in workers.values():
                        for worker in arms.values():
                            worker.macro_context_sha256 = macro_hash
                    for symbol in SYMBOLS:
                        accounts[symbol], comparisons[symbol], clocks[symbol] = tick_symbol(
                            output, symbol, workers[symbol], venue, calendar_hash, context)
                    cycles += 1
                    status = publish(output, manifest, accounts, comparisons, clocks,
                        running=not args.once and not stop.is_set(), cycles=cycles, calendar_hash=calendar_hash, macro_hash=macro_hash)
                    print(json.dumps({k: status[k] for k in ('experiment_id', 'pid', 'status', 'heartbeat_at_utc', 'cycles')}), flush=True)
                    if args.once:
                        break
                    stop.wait(30)
        finally:
            publish(output, manifest, accounts, comparisons, clocks, running=False,
                    cycles=cycles, calendar_hash=calendar_hash, macro_hash=macro_hash)


if __name__ == '__main__':
    main()
