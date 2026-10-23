"""Independent hourly exit-response paper experiment; public GETs, no order API."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

from process_lock import exclusive_process_lock
import run_parallel_simulation as paper
import run_sndk_4h_gate_shadow as shared
import stock_exit_response as exits
import stock_exit_response_metrics as paired

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/stock_exit_response_shadow_20261023.json'
SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')
VARIANTS = ('baseline', 'breakout_failure', 'giveback', 'combined')
COSTS = ('normal', 'double')
ARMS = tuple(v + '_' + c for c in COSTS for v in VARIANTS)
HOUR = 3_600_000


class LedgerIntegrityError(ValueError):
    """Do not continue an arm after a partial financial write or corrupted state."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


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


def contained(path, parent):
    resolved = (ROOT / path).resolve()
    resolved.relative_to((ROOT / parent).resolve())
    if resolved == (ROOT / parent).resolve():
        raise ValueError('An isolated experiment subdirectory is required')
    return resolved


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = paper.read_json(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'SIMULATION'
            or spec.get('places_orders') is not False or spec.get('symbols') != list(SYMBOLS)
            or spec.get('variants') != list(VARIANTS)
            or spec.get('cost_multipliers') != {'normal': 1, 'double': 2}
            or spec.get('initial_balance_usdt_each') != 1000
            or spec.get('automatic_strategy_promotion') is not False
            or spec.get('historical_warmup_counts_as_forward_performance') is not False
            or spec.get('candidate_only_added_rule') != 'exit_response_policy'
            or spec.get('review_minimum_elapsed_days') != 14
            or spec.get('review_minimum_closed_positions_each') != 10
            or spec.get('review_minimum_completed_early_exit_pairs') != 10
            or type(spec.get('effective_from_ms')) is not int or spec['effective_from_ms'] <= 0
            or spec.get('poll_seconds') != 30):
        raise ValueError('Invalid fixed hourly exit-response design')
    output = contained(spec['output_dir'], 'data/paper_trading')
    calendar_path = (output / 'calendar_current.json').relative_to(ROOT).as_posix()
    rules = {}
    for symbol in SYMBOLS:
        baseline = paper.stock_config({'stock_research_profile': spec['baseline_profile']}, symbol)
        if (baseline.get('signal_timeframe') != '1h' or baseline.get('signal_family') != 'donchian20'
                or baseline.get('entry_direction') != 'both'
                or baseline.get('swing_confirmation_policy') != paper.swing_confirmation.POLICY
                or baseline.get('risk_fraction_per_trade') != .0025
                or baseline.get('cooldown_signal_bars') != 3
                or baseline.get('trend_exit_policy') is not None
                or baseline.get('profit_exit_policy') is not None
                or baseline.get('exit_response_policy') is not None
                or not paper.stock_event_risk.enabled(baseline)):
            raise ValueError('Baseline must be the current hourly stock strategy')
        # One immutable calendar copy per batch, with its real provenance preserved.
        baseline['stock_event_risk_policy'] = {
            **baseline['stock_event_risk_policy'], 'calendar_path': calendar_path}
        rules[symbol] = {}
        for arm in ARMS:
            variant, cost = arm.rsplit('_', 1)
            config = deepcopy(baseline)
            if variant != 'baseline':
                config['exit_response_policy'] = {'mode': 'hourly_exit_response_v1', 'variant': variant}
                exits.validate(config['exit_response_policy'])
            for key in ('taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'):
                config[key] *= spec['cost_multipliers'][cost]
            rules[symbol][arm] = config
    profile = ROOT / spec['baseline_profile']
    base = ROOT / paper.read_json(profile)['base_config']
    fingerprint = {'experiment_config_sha256': digest(path.read_bytes()),
        'profile_sha256': {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in (profile, base)},
        'source_sha256': source_hashes(),
        'rules_sha256': {s: {a: digest(c) for a, c in arms.items()} for s, arms in rules.items()}}
    return spec, rules, fingerprint


def check_source_rules(spec, rules):
    """Read original rules only; never move, initialize or migrate their ledgers."""
    plan = paper.read_json(ROOT / spec['source_plan'])
    if plan.get('execution_mode') != 'simulation' or plan.get('live_orders_allowed') is not False:
        raise ValueError('Original plan must remain simulation only')
    accounts = {a['symbol']: a for a in plan['accounts'] if a['symbol'] in SYMBOLS}
    if set(accounts) != set(SYMBOLS):
        raise ValueError('All original stock accounts must exist')
    for symbol, account in accounts.items():
        state = paper.read_json(ROOT / account['state_path'])
        expected = deepcopy(rules[symbol]['baseline_normal'])
        expected['stock_event_risk_policy']['calendar_path'] = state['rule']['stock_event_risk_policy']['calendar_path']
        if (state.get('mode') != 'simulation' or state.get('places_orders') is not False
                or state.get('symbol') != symbol or state.get('rule') != expected):
            raise ValueError('Original account rule differs from the declared baseline: ' + symbol)
    return {s: accounts[s]['state_path'] for s in SYMBOLS}


def validate_ledger(state, rule, symbol, started):
    if (not isinstance(state, dict) or state.get('rule') != rule or state.get('symbol') != symbol
            or state.get('mode') != 'simulation' or state.get('places_orders') is not False
            or state.get('initial_balance') != 1000 or state.get('start_ms') != started
            or state.get('signal_active_after_ms') != started):
        raise ValueError('Existing exit-response ledger incompatible; no reset allowed')
    numbers = [state.get(k) for k in ('wallet_balance', 'realized_pnl', 'fees_paid', 'funding_pnl', 'equity')]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in numbers):
        raise ValueError('Existing financial values invalid')
    if abs(1000 + numbers[1] - numbers[2] + numbers[3] - numbers[0]) > 1e-7:
        raise ValueError('Existing ledger cash identity failed')
    if (type(state.get('fill_count_total')) is not int or state['fill_count_total'] < 0
            or len(state.get('fills', [])) != min(state['fill_count_total'], 100)):
        raise ValueError('Existing fill count invalid')


def validate_financial_journals(directory, state):
    """A crash between append and state commit must stop, never replay a debit/fill."""
    try:
        names = ('fills.jsonl', 'funding.jsonl', 'closed_trades.jsonl', 'entries.jsonl')
        if any(not (directory / name).is_file() for name in names):
            raise ValueError('Financial journal missing')
        fills = shared.read_rows(directory / 'fills.jsonl')
        funding = shared.read_rows(directory / 'funding.jsonl')
        closed = shared.read_rows(directory / 'closed_trades.jsonl')
        shared.read_rows(directory / 'entries.jsonl')
        if len(fills) != state['fill_count_total'] or fills[-100:] != state.get('fills', []):
            raise ValueError('Fill journal/state mismatch')
        if [r.get('sequence') for r in fills] != list(range(1, len(fills) + 1)):
            raise ValueError('Duplicate or missing fill sequence')
        if len({(r.get('symbol'), r.get('fundingTime')) for r in funding}) != len(funding):
            raise ValueError('Duplicate funding settlement journal')
        if len({r.get('sequence') for r in closed}) != len(closed):
            raise ValueError('Duplicate closing fill journal')
        checks = ((sum(r['fee'] for r in fills), state['fees_paid']),
                  (-sum(r['debit'] for r in funding), state['funding_pnl']),
                  (sum(r['gross_pnl'] for r in closed), state['realized_pnl']),
                  (sum(r['qty'] * (1 if r['side'] == 'BUY' else -1) for r in fills), state.get('position_qty', 0)))
        if any(not math.isfinite(a) or abs(a - b) > 1e-7 for a, b in checks):
            raise ValueError('Financial journal reconciliation failed')
        pos = state.get('position')
        if pos and abs(pos['qty'] * pos['direction'] - state.get('position_qty', 0)) > 1e-8:
            raise ValueError('Position inventory inconsistent')
        if not pos and abs(state.get('position_qty', 0)) > 1e-8:
            raise ValueError('Missing position inventory')
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise LedgerIntegrityError(str(exc)) from exc


def validate_fingerprint(fingerprint, rules):
    if not isinstance(fingerprint, dict) or set(fingerprint) != {
            'experiment_config_sha256', 'profile_sha256', 'source_sha256', 'rules_sha256'}:
        raise ValueError('Complete experiment fingerprint is required')
    groups = [fingerprint.get('profile_sha256'), fingerprint.get('source_sha256')]
    hashes = [fingerprint.get('experiment_config_sha256')]
    for group in groups:
        if not isinstance(group, dict) or not group:
            raise ValueError('Complete source/profile fingerprint is required')
        hashes.extend(group.values())
    if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in hashes):
        raise ValueError('Invalid experiment fingerprint')
    if fingerprint.get('rules_sha256') != {s: {a: digest(c) for a, c in arms.items()} for s, arms in rules.items()}:
        raise ValueError('Rule fingerprint mismatch')


def prepare(output, spec, rules, fingerprint, inception_ms, require_existing=False):
    validate_fingerprint(fingerprint, rules)
    path = output / 'manifest.json'
    if path.exists():
        manifest = paper.read_json(path)
        if (manifest.get('fingerprint') != fingerprint or manifest.get('rules') != rules
                or manifest.get('design') != spec or manifest.get('mode') != 'SIMULATION'
                or manifest.get('places_orders') is not False):
            raise ValueError('Frozen exit-response experiment changed; new generation required')
        for symbol in SYMBOLS:
            for arm in ARMS:
                directory = output / symbol / arm
                if not (directory / 'state.json').is_file() or not (directory / 'performance.json').is_file():
                    raise ValueError('Existing exit-response history incomplete; refusing reset')
                validate_ledger(paper.read_json(directory / 'state.json'), rules[symbol][arm], symbol,
                                manifest['started_at_ms'])
                validate_financial_journals(directory, paper.read_json(directory / 'state.json'))
                if paper.read_json(directory / 'performance.json').get('start_ms') != manifest['started_at_ms']:
                    raise ValueError('Existing performance epoch incompatible')
        return manifest
    if require_existing:
        raise ValueError('Existing experiment required; explicit initialization is necessary')
    if any(p.name != 'runner.lock' for p in output.iterdir()):
        raise ValueError('Incomplete exit-response directory; refusing overwrite')
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


class ExitResponseAccount(paper.StockAccount):
    """Only the optional exit hook changes; entries and hard guards use the old engine."""
    def __init__(self, *args, activated_at_ms, **kwargs):
        super().__init__(*args, **kwargs)
        self.activated_at_ms = activated_at_ms
        self._entry_context = None
        if self.config.get('exit_response_policy'):
            exits.validate(self.config['exit_response_policy'])

    def fill(self, state, result, timestamp, direction, reason, trigger_ms=None):
        if reason == 'fresh_closed_1h_signal':
            pending = state.get('pending_signal') or {}
            self._entry_context = {'entry_time_ms': timestamp,
                'signal_boundary_ms': pending.get('boundary_ms'), 'direction': direction,
                'entry_price': result['price'], 'initial_qty': result['qty'],
                'entry_trigger': deepcopy(pending.get('entry_trigger'))}
        return super().fill(state, result, timestamp, direction, reason, trigger_ms)

    def entry_metadata(self, state):
        pos = state.get('position')
        if not pos:
            return
        if 'entry_trigger' not in pos:
            context = self._entry_context
            if not context or context.get('entry_time_ms') != pos['entry_time']:
                # Exact original signal only; restart must never roll the breakout boundary.
                candidates = [r for r in shared.read_rows(self.path.with_name('signals.jsonl'))
                              if r.get('boundary_ms') == pos['signal_time']]
                context = {'entry_trigger': (candidates[-1].get('pending_signal') or {}).get('entry_trigger')} if candidates else {}
            trigger = context.get('entry_trigger')
            if (isinstance(trigger, dict) and trigger.get('direction') == pos['direction']
                    and trigger.get('boundary_ms') == pos['signal_time']):
                pos['entry_trigger'] = deepcopy(trigger)
        rows = shared.read_rows(self.path.with_name('entries.jsonl'))
        existing = [r for r in rows if r.get('entry_time_ms') == pos['entry_time']]
        if len(existing) > 1:
            raise LedgerIntegrityError('Duplicate immutable entry metadata')
        if existing:
            immutable = {'signal_boundary_ms': pos['signal_time'], 'direction': pos['direction'],
                         'entry_price': pos['entry'], 'initial_stop': pos['initial_stop'],
                         'entry_trigger': pos.get('entry_trigger')}
            if any(existing[0].get(k) != v for k, v in immutable.items()):
                raise LedgerIntegrityError('Immutable entry risk/trigger metadata changed')
        else:
            original_fills = [f for f in shared.read_rows(self.path.with_name('fills.jsonl')) if f.get('time_ms') == pos['entry_time']
                              and f.get('reason') == 'fresh_closed_1h_signal']
            initial_qty = ((self._entry_context or {}).get('initial_qty')
                           or (original_fills[0]['qty'] if len(original_fills) == 1 else None))
            paper.journal(self.path.with_name('entries.jsonl'), {'entry_time_ms': pos['entry_time'],
                'signal_boundary_ms': pos['signal_time'], 'direction': pos['direction'],
                'entry_price': pos['entry'], 'initial_stop': pos['initial_stop'],
                'initial_qty': initial_qty,
                'entry_trigger': deepcopy(pos.get('entry_trigger'))})

    def additional_exit_reason(self, state, position, data, errors, timestamp):
        if self.config.get('exit_response_policy') is None:
            state['exit_response_status'] = {'status': 'baseline', 'triggered': False}
            return None
        self.entry_metadata(state)
        try:
            bars = ([paper.arithmetic.candle(r) for r in self.signal_bars]
                    if data.get('clock') and not errors.get('clock') and not errors.get('signal') else [])
            outcome = exits.evaluate(position, bars, self.config, timestamp, self.signal_interval_ms,
                self.activated_at_ms, entry_trigger=position.get('entry_trigger'),
                state=position.get('exit_response_state'))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            outcome = {'status': 'unavailable', 'triggered': False, 'reason': type(exc).__name__}
        if outcome.get('state') is not None:
            position['exit_response_state'] = outcome['state']
        diagnostic = {k: v for k, v in outcome.items() if k != 'state'}
        state['exit_response_status'] = diagnostic
        if outcome.get('triggered') and not position.get('pending_exit'):
            paper.journal(self.path.with_name('exit_decisions.jsonl'), {
                'observed_at_ms': timestamp, 'entry_time_ms': position['entry_time'], **diagnostic})
            return outcome['reason']
        return None

    def step(self, market_data=None):
        self._entry_context = None
        before = paper.read_json(self.path)
        self.entry_metadata(before)
        # Persist recovered immutable context BEFORE an immediate stop can close it.
        if before.get('position'):
            paper.write_json(self.path, before)
        state = super().step(market_data)
        self.entry_metadata(state)
        if not state.get('position'):
            state['exit_response_status'] = {'status': 'flat', 'triggered': False}
        elif self.config.get('exit_response_policy') is None:
            state['exit_response_status'] = {'status': 'baseline', 'triggered': False}
        elif (state.get('exit_response_status') or {}).get('status') == 'unavailable':
            state['status'] = 'degraded'
            state['errors'] = {**state.get('errors', {}),
                              'exit_response': state['exit_response_status'].get('reason', 'unavailable')}
        paper.write_json(self.path, state)
        return state


def freeze_calendar(output, spec):
    original = paper.stock_config({'stock_research_profile': spec['baseline_profile']}, SYMBOLS[0])
    source = ROOT / original['stock_event_risk_policy']['calendar_path']
    try:
        raw = source.read_bytes()
        json.loads(raw)
    except (OSError, ValueError):
        raw = b'{}'
    target = output / 'calendar_current.json'
    target.write_bytes(raw)
    archive = output / 'calendar_archive' / (digest(raw) + '.json.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_bytes(gzip.compress(raw, mtime=0))
    return digest(raw)


@contextmanager
def compact_observations(output, context):
    """Deduplicate raw books/candles across arms, only within this new process."""
    original = paper.journal
    output = output.resolve()
    def append(path, row):
        path = Path(path)
        if path.name == 'observations.jsonl' and path.resolve().is_relative_to(output):
            row = {k: row.get(k) for k in ('time_utc', 'symbol', 'equity', 'position_qty',
                'signal_status', 'entry_blockers', 'last_signal_time_ms', 'errors')}
            row['shared_context_sha256'] = context.get(row['symbol'])
        original(path, row)
    paper.journal = append
    try:
        yield
    finally:
        paper.journal = original


def archive_context(output, symbol, responses, calendar_hash, cached_rules=None, cached_signal=None):
    blocks = output / 'context_blocks'
    blocks.mkdir(parents=True, exist_ok=True)
    def block(value):
        if value is None:
            return None
        raw = canonical(value)
        sha = digest(raw)
        target = blocks / (sha + '.json.gz')
        if not target.exists():
            target.write_bytes(gzip.compress(raw, mtime=0))
        return sha
    payload = {'schema_version': 1, 'symbol': symbol,
        'received_at_utc': paper.arithmetic.iso(int(time.time() * 1000)),
        'calendar_sha256': calendar_hash,
        'responses': [{'label': label, 'payload_sha256': block(value), 'error': error}
                      for label, value, error in responses],
        'cached_contract_rules_sha256': block(cached_rules),
        'cached_signal_candles_sha256': block(cached_signal)}
    raw = canonical(payload)
    sha = digest(raw)
    directory = output / 'context_archive' / symbol
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (sha + '.json.gz')).write_bytes(gzip.compress(raw, mtime=0))
    return sha


def estimated_close(state, responses, config, asof_ms, cached_contract=None):
    """Diagnostic full close at current visible depth, never an actual paper fill."""
    result = {'status': 'unavailable', 'equity_after_estimated_close_usdt': None,
              'note': 'Current visible depth and assumed costs; not an execution guarantee.'}
    if state.get('errors', {}).get('funding'):
        return {**result, 'reason': 'funding_unavailable'}
    pos = state.get('position')
    if not pos:
        return {**result, 'status': 'available', 'equity_after_estimated_close_usdt': state['wallet_balance'],
                'estimated_exit_fee_usdt': 0., 'fillable_qty': 0.}
    data = {label: payload for label, payload, error in responses if not error}
    try:
        book = data['book']
        paper.fresh(book.get('T') or book.get('E'), asof_ms)
        # The frozen contract filters are the actual step used by the ledger.
        rules = data.get('rules')
        if not rules and not cached_contract:
            return {**result, 'reason': 'contract_rules_not_in_current_context'}
        contract = (next(r for r in rules['symbols'] if r['symbol'] == state['symbol'])
                    if rules else cached_contract)
        filters = {r['filterType']: r for r in contract['filters']}
        lot = filters.get('MARKET_LOT_SIZE') or filters['LOT_SIZE']
        fill = paper.book_fill(book, -pos['direction'] * pos['qty'], float(lot['stepSize']),
                               config['adverse_slippage_fraction_assumption'])
        if fill['qty'] + 1e-10 < pos['qty']:
            return {**result, 'reason': 'insufficient_visible_full_close_depth', 'fillable_qty': fill['qty']}
        fee = fill['qty'] * fill['price'] * config['taker_fee_rate_assumption']
        value = state['wallet_balance'] + pos['direction'] * fill['qty'] * (fill['price'] - pos['entry']) - fee
        return {**result, 'status': 'available', 'equity_after_estimated_close_usdt': value,
                'estimated_exit_fee_usdt': fee, 'fillable_qty': fill['qty'], 'exit_price': fill['price']}
    except (KeyError, TypeError, ValueError, StopIteration):
        return {**result, 'reason': 'current_close_inputs_unavailable'}


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
    comparisons = {}
    now = workers['baseline_normal'].now_ms()
    for cost in COSTS:
        comparisons[cost] = {}
        baseline = result['baseline_' + cost]
        for variant in VARIANTS[1:]:
            arm = variant + '_' + cost
            candidate = result[arm]
            valid = candidate.get('financial_integrity') and baseline.get('financial_integrity')
            try:
                if not valid:
                    raise ValueError('financial_integrity_unavailable')
                diagnostic = paired.update_diagnostics(output, symbol, cost, variant,
                    states['baseline_' + cost], states[arm], workers['baseline_normal'].signal_bars, now)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                diagnostic = {'status': 'unavailable', 'source_issues': [type(exc).__name__], 'summary': {}}
            comparisons[cost][variant] = {'comparison_valid': bool(valid),
                'equity_delta_usdt': candidate['equity'] - baseline['equity'] if valid else None,
                'estimated_close_delta_usdt': (candidate['estimated_close']['equity_after_estimated_close_usdt']
                    - baseline['estimated_close']['equity_after_estimated_close_usdt']
                    if valid and candidate['estimated_close']['status'] == baseline['estimated_close']['status'] == 'available' else None),
                'closed_net_delta_usdt': (candidate['metrics']['closed_net_pnl_usdt'] - baseline['metrics']['closed_net_pnl_usdt']
                    if candidate.get('metrics') and baseline.get('metrics') else None), 'paired': diagnostic}
    return result, comparisons, clock


def publish(output, manifest, accounts, comparisons, clocks, *, running, cycles, calendar_hash):
    now = int(time.time() * 1000)
    flat = {symbol + '_' + arm: value for symbol, arms in accounts.items() for arm, value in arms.items()}
    elapsed = max(0, now - manifest['started_at_ms']) / 86_400_000
    sufficient = len(flat) == 24 and all(v.get('financial_integrity') is True and
        (v.get('metrics') or {}).get('closed_positions', 0) >= 10 for v in flat.values())
    pairs_sufficient = len(comparisons) == 3 and all(
        ((comparisons.get(symbol, {}).get(cost, {}).get(variant, {}).get('paired') or {}).get('summary') or {}).get('completed_followups', 0) >= 10
        for symbol in SYMBOLS for cost in COSTS for variant in VARIANTS[1:])
    status = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'experiment_id': manifest['experiment_id'], 'pid': os.getpid(), 'running': running,
        'status': 'healthy' if len(flat) == 24 and all(v.get('status') == 'healthy' for v in flat.values()) else 'degraded',
        'started_at_ms': manifest['started_at_ms'], 'started_at_utc': manifest['started_at_utc'],
        'heartbeat_at_ms': now, 'heartbeat_at_utc': paper.arithmetic.iso(now), 'poll_seconds': 30,
        'cycles': cycles, 'by_symbol': accounts, 'accounts': flat, 'comparisons': comparisons,
        'clock': clocks, 'calendar_sha256': calendar_hash, 'fingerprint': manifest['fingerprint'],
        'original_accounts_modified': False, 'automatic_strategy_promotion': False, 'forward_validated': False,
        'independent_initial_capital_usdt': 24000, 'original_account_capital_included': False,
        'review': {'elapsed_days': elapsed, 'minimum_elapsed_days': 14, 'minimum_closed_positions_each': 10,
            'minimum_completed_early_exit_pairs': 10,
            'closed_trade_sample_sufficient': sufficient, 'completed_pair_sample_sufficient': pairs_sufficient,
            'ready_for_first_review': elapsed >= 14 and sufficient and pairs_sufficient,
            'readiness_is_pass': False, 'interpretation': 'Review readiness is not evidence of improvement. Require cost-matched net returns, drawdown, continuation and churn analysis; never auto-promote.'}}
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
            'output_initialized': False, 'ledger_count': 24, 'source_file_count': len(fingerprint['source_sha256']),
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
        workers = {symbol: {arm: ExitResponseAccount(output / symbol / arm / 'state.json', venue, symbol,
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
