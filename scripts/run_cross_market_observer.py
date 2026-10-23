"""Independent forward admission observations; never creates orders or account ledgers."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
from process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/cross_market_shadow_20261023.json'
ACCOUNTS = ('btc', 'mu', 'sndk', 'skhynix')
ARMS = ('baseline', 'event_only', 'market_only', 'sector_budget', 'combined')
_COLLECT_LOCK = threading.Lock()
_COLLECT_INFLIGHT = None
SELECTED = ('symbol', 'rule', 'position', 'position_qty', 'entry_price', 'initial_balance',
    'equity', 'wallet_balance', 'realized_pnl', 'fees_paid', 'funding_pnl', 'fill_count_total',
    'last_mark_price', 'updated_at_utc', 'checked_at_utc', 'created_at_utc', 'start_ms',
    'pending_signal', 'btc_pullback_pending', 'entry_checks', 'entry_blockers', 'signal_status',
    'last_signal_time_ms', 'risk_monitor', 'risk_halted', 'account_risk', 'signal_active_after_ms',
    'entry_qualification', 'funding_status', 'last_signal_result', 'btc_entry_timing')


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(canonical(payload) + b'\n')
    temporary.replace(path)


def contained(value, prefix=None):
    path = (ROOT / value).resolve()
    path.relative_to((ROOT / prefix).resolve() if prefix else ROOT.resolve())
    return path


def account_specs(spec):
    accounts = spec['accounts']
    if isinstance(accounts, list):
        accounts = {row['account_id']: row for row in accounts}
    if set(accounts) != set(ACCOUNTS):
        raise ValueError('Exactly the original four account references are required')
    return accounts


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = read(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'OBSERVATION_ONLY'
            or spec.get('places_orders') is not False or spec.get('no_pnl_simulated') is not True
            or type(spec.get('poll_seconds')) is not int or spec['poll_seconds'] < 5
            or type(spec.get('collect_seconds')) is not int or spec['collect_seconds'] < 30):
        raise ValueError('Invalid order-disabled observation-only design')
    for row in account_specs(spec).values():
        contained(row['state_path'], 'data/parallel_simulation')
        if row.get('status_path'):
            contained(row['status_path'], 'data')
    for key in ('btc_report_path', 'observation_config', 'international_calendar'):
        contained(spec[key])
    if spec.get('status_path'):
        contained(spec['status_path'], 'data')
    output = contained(spec.get('output_dir', 'data/cross_market_observer'), 'data')
    if output == (ROOT / 'data').resolve() or output.is_relative_to((ROOT / 'data/parallel_simulation').resolve()) or output.is_relative_to((ROOT / 'data/paper_trading').resolve()):
        raise ValueError('Observer evidence must be separate from every financial ledger')
    return spec


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
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                local = ROOT / 'scripts' / (name.split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(result.items()))


def fingerprint(spec, config_path):
    # The calendar is maintained data, never a restart-blocking byte pin.
    return {'configuration_sha256': digest(Path(config_path).read_bytes()),
        'observation_configuration_sha256': digest(contained(spec['observation_config']).read_bytes()),
        'source_sha256': source_hashes(), 'calendar_byte_pinning': False}


def check_design(config_path=DEFAULT_CONFIG):
    """Validate references and definition bytes without output initialization or GETs."""
    from cross_market_observations import load_config
    from international_event_calendar import load_calendar, evaluate_international_event_calendar
    import cross_market_risk as risk

    spec = load_design(config_path)
    load_config(contained(spec['observation_config']))
    calendar = load_calendar(contained(spec['international_calendar']))
    # Shape is checked at its recorded verification clock, separately from its
    # current trading availability. An old calendar cannot regain freshness.
    verified = calendar.get('verified_at_ms')
    if type(verified) is not int or verified <= 0:
        raise ValueError('Calendar verification clock missing or invalid')
    outcome = evaluate_international_event_calendar(calendar, asof_ms=verified,
        action='new_risk', policy=spec.get('event_policy'))
    if outcome.get('status') not in ('clear', 'blackout'):
        raise ValueError('Calendar structure rejected: ' + ','.join(outcome.get('reasons', [])))
    # Empty observations intentionally fail availability but still validate
    # the fixed research-policy schema without consulting a provider.
    risk.evaluate_cross_market_risk({}, spec.get('market_policy', {}), asof_ms=verified)
    references = set()
    for row in account_specs(spec).values():
        references.add(row['state_path'])
        if row.get('status_path'):
            references.add(row['status_path'])
    if spec.get('status_path'):
        references.add(spec['status_path'])
    references.add(spec['btc_report_path'])
    checks = {}
    for relative in sorted(references):
        payload, raw_hash = stable_json(contained(relative))
        if not isinstance(payload, dict) or not payload:
            raise ValueError('Observer reference must be a nonempty JSON object: ' + relative)
        if payload.get('places_orders') is True:
            raise ValueError('Observer reference unexpectedly permits orders: ' + relative)
        checks[relative] = raw_hash
    frozen = fingerprint(spec, config_path)
    return {'mode': 'OBSERVATION_ONLY', 'places_orders': False, 'no_pnl_simulated': True,
        'validation': 'configuration_and_references_only', 'network_requests_performed': False,
        'output_initialized': False, 'source_file_count': len(frozen['source_sha256']),
        'fingerprint': frozen, 'reference_sha256': checks,
        'calendar_shape_checked_at_ms': verified,
        'calendar_current_freshness_not_renewed': True}


def prepare(output, spec, frozen, now_ms, require_existing=False):
    output = Path(output)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if (manifest.get('design') != spec or manifest.get('fingerprint') != frozen
                or manifest.get('mode') != 'OBSERVATION_ONLY' or manifest.get('places_orders') is not False
                or manifest.get('no_pnl_simulated') is not True):
            raise ValueError('Frozen observer definition changed; no silent reset allowed')
        tracker = read(output / 'tracker.json')
        if tracker.get('started_at_ms') != manifest['started_at_ms']:
            raise ValueError('Observer tracking epoch differs; no silent reset allowed')
        return manifest
    if require_existing:
        raise ValueError('Existing observation manifest/tracker required')
    if any(p.name != 'runner.lock' for p in output.iterdir()):
        raise ValueError('Partial observer output cannot be overwritten')
    if type(now_ms) is not int or now_ms <= 0:
        raise ValueError('Valid observation inception is required')
    manifest = {'schema_version': 1, 'experiment_id': spec.get('experiment_id', 'cross_market_shadow_20261023'),
        'mode': 'OBSERVATION_ONLY', 'places_orders': False, 'no_pnl_simulated': True,
        'started_at_ms': now_ms, 'started_at_utc': iso(now_ms), 'clock_source': 'host_utc_observation_only',
        'design': spec, 'fingerprint': frozen, 'arms': list(ARMS), 'original_accounts_modified': False,
        'automatic_strategy_promotion': False, 'baseline_is_reported_not_fill_permit': True,
        'comparison_basis': 'Concurrent observations on original account signals; no counterfactual portfolio or returns.',
        'old_origins_excluded': True, 'calendar_is_maintainable_data': True}
    write(output / 'tracker.json', {'started_at_ms': now_ms, 'journal_sequence': 0,
        'journal_offset': 0, 'last_record_start': 0, 'last_record_sha256': None,
        'cycles': 0, 'opportunities': {}, 'counted': {}, 'market_history': {}, 'last_collect_at_ms': None})
    write(manifest_path, manifest)
    return manifest


def stable_json(path):
    path = Path(path)
    for _ in range(3):
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return json.loads(raw.decode('utf-8-sig')), digest(raw)
    raise ValueError('Source changed repeatedly while reading')


def inspect_inputs(spec, *, asof_ms):
    states, metadata, errors = {}, {}, {}
    shared_status = {}
    if spec.get('status_path'):
        try:
            shared_status, status_hash = stable_json(contained(spec['status_path']))
            metadata['shared_status_sha256'] = status_hash
        except (OSError, ValueError, TypeError) as exc:
            errors['shared_status'] = type(exc).__name__
    status_cache = {}
    for account_id, row in account_specs(spec).items():
        try:
            state, raw_hash = stable_json(contained(row['state_path']))
            selected = {key: deepcopy(state[key]) for key in SELECTED if key in state}
            reported = shared_status.get('accounts', {}).get(account_id, {})
            if row.get('status_path'):
                path = contained(row['status_path'])
                key = str(path)
                if key not in status_cache:
                    status_cache[key] = stable_json(path)
                payload, reported_hash = status_cache[key]
                reported = payload.get('accounts', {}).get(account_id, payload)
                metadata[account_id + '_status_sha256'] = reported_hash
            # Runtime status can be fresh while a flat financial state is unchanged.
            selected['_reported'] = {key: deepcopy(reported[key]) for key in SELECTED + ('status', 'errors') if key in reported}
            selected['_financial_fields'] = {key: deepcopy(state.get(key)) for key in
                ('wallet_balance', 'realized_pnl', 'fees_paid', 'funding_pnl', 'fill_count_total', 'position_qty', 'initial_balance')}
            selected['_source_sha256'] = raw_hash
            selected['_consistency_errors'] = []
            for key in ('wallet_balance', 'position_qty', 'fill_count_total', 'last_signal_time_ms'):
                if key in state and key in reported:
                    left, right = state[key], reported[key]
                    same = (abs(left - right) <= 1e-8 if type(left) in (int, float) and type(right) in (int, float)
                            else left == right)
                    if not same:
                        selected['_consistency_errors'].append('state_status_disagree:' + key)
            states[account_id] = selected
            metadata[account_id + '_state_sha256'] = raw_hash
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors[account_id] = type(exc).__name__
            states[account_id] = {'_input_error': type(exc).__name__, '_reported': {}}
    return states, digest({'states': states, 'source_metadata': metadata}), {'source_metadata': metadata,
        'errors': errors, 'read_at_ms': asof_ms, 'financial_fields_are_read_only': True}


def selected_report(spec):
    report, raw_hash = stable_json(contained(spec['btc_report_path']))
    return {key: deepcopy(report.get(key)) for key in
        ('execution_target', 'generated_at_utc', 'decision_asof_ms', 'mode', 'places_orders')}, raw_hash


def timestamp_ms(value):
    try:
        if type(value) is int:
            return value if value >= 0 else None
        if not isinstance(value, str):
            return None
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        stamp = int(parsed.timestamp() * 1000)
        return stamp if stamp >= 0 else None
    except (ValueError, AttributeError, TypeError, OverflowError):
        return None


def reported_baseline(account_id, state, report, *, asof_ms, inception_ms, max_age_ms=120000):
    reported = state.get('_reported') or {}
    view = {**state, **reported}
    pending = state.get('btc_pullback_pending') if account_id == 'btc' else state.get('pending_signal')
    pending_data = pending if isinstance(pending, dict) else {}
    if account_id == 'btc':
        point = report.get('execution_target') or {}
        origin = point.get('origin_signal_time_ms')
        amount = point.get('signed_qty', point.get('target_leverage', 0)) or 0
        direction = (1 if amount > 0 else -1 if amount < 0 else 0) if isinstance(amount, (int, float)) and math.isfinite(amount) else 0
    else:
        signal_row = pending_data.get('signal') or {}
        origin = pending_data.get('boundary_ms')
        direction = signal_row.get('direction', 0)
        if direction not in (-1, 1):
            direction = 0
    valid_origin = type(origin) is int and 0 < origin <= asof_ms
    opportunity_id = f'{account_id}:{origin}:{direction}' if valid_origin and direction else None
    future = type(origin) is int and origin > asof_ms
    eligible_epoch = bool(opportunity_id and origin > inception_ms)
    reasons = []
    reasons.extend('baseline_' + reason for reason in state.get('_consistency_errors', []))
    checked = timestamp_ms(view.get('checked_at_utc'))
    if checked is None:
        checked = timestamp_ms((view.get('risk_monitor') or {}).get('checked_at_ms'))
    if checked is None or not 0 <= asof_ms - checked <= max_age_ms:
        reasons.append('baseline_source_stale_or_unavailable')
    if state.get('_input_error'):
        reasons.append('baseline_state_unavailable')
    if not pending:
        reasons.append('baseline_no_pending_entry')
    elif not isinstance(pending, dict):
        reasons.append('baseline_pending_invalid')
    if pending:
        expiry = pending_data.get('expires_at_ms')
        if type(expiry) is not int or not valid_origin or not origin < expiry or asof_ms >= expiry:
            reasons.append('baseline_pending_expired_or_invalid')
    if future or not valid_origin or not direction:
        reasons.append('baseline_origin_unavailable_or_future')
    if view.get('status') != 'healthy':
        reasons.append('baseline_reported_unhealthy_or_unknown')
    blockers = view.get('entry_blockers')
    if not isinstance(blockers, list):
        reasons.append('baseline_original_blockers_unknown')
    else:
        reasons.extend('original:' + str(reason) for reason in blockers)
    status = str(view.get('signal_status') or '')
    if any(word in status for word in ('blocked', 'unavailable', 'expired', 'invalid', 'halt', 'cooldown',
            'insufficient', 'stop_distance', 'not_qualified', 'no_signal')):
        reasons.append('baseline_original_signal_not_admitted')
    if account_id == 'btc':
        pending_state = pending_data.get('status')
        if pending_state != 'ready':
            reasons.append('baseline_pullback_not_ready')
        pending_origin = pending_data.get('signal_origin')
        source = pending_data.get('source') or {}
        if (pending_origin != origin or source.get('origin_signal_time_ms') != origin
                or source.get('position_id') != point.get('position_id')
                or pending_data.get('side') != ('long' if direction == 1 else 'short')):
            reasons.append('baseline_btc_pending_report_identity_mismatch')
        generated = timestamp_ms(report.get('generated_at_utc'))
        if generated is None or not 0 <= asof_ms - generated <= max_age_ms:
            reasons.append('baseline_btc_report_stale_or_unavailable')
    else:
        if view.get('last_signal_time_ms') != origin:
            reasons.append('baseline_stock_pending_report_identity_mismatch')
        signal_time = (pending_data.get('signal') or {}).get('time_ms')
        if type(signal_time) is not int or not valid_origin or signal_time >= origin:
            reasons.append('baseline_stock_signal_time_invalid')
        passed = (view.get('entry_checks') or {}).get('passed')
        if not isinstance(passed, dict) or not passed or not all(value is True for value in passed.values()):
            reasons.append('baseline_original_entry_checks_not_all_reported_passed')
        checked_entry = (view.get('entry_checks') or {}).get('checked_at_ms')
        if type(checked_entry) is not int or not 0 <= asof_ms - checked_entry <= max_age_ms:
            reasons.append('baseline_original_entry_checks_stale_or_unavailable')
    quantity = state.get('position_qty')
    if type(quantity) not in (int, float) or not math.isfinite(quantity):
        reasons.append('baseline_inventory_unknown')
    elif abs(quantity) > 1e-12:
        reasons.append('baseline_existing_inventory_not_new_entry')
    return {'allowed': not reasons, 'status': 'reported_eligible' if not reasons else 'not_reported_eligible',
        'reasons': list(dict.fromkeys(reasons)), 'direction': direction, 'origin_ms': origin,
        'opportunity_id': opportunity_id, 'post_inception': eligible_epoch,
        'pre_inception': bool(opportunity_id and not eligible_epoch),
        'baseline_is_reported_not_fill_permit': True, 'actual_fill_permission': None,
        'reported_signal_status': status, 'source_checked_at_ms': checked}


def _combined(*gates):
    reasons = list(dict.fromkeys(reason for gate in gates for reason in gate.get('reasons', [])))
    allowed = all(gate.get('allowed') is True for gate in gates)
    return {'allowed': allowed, 'status': 'observed_allowed' if allowed else 'observed_blocked', 'reasons': reasons}


def blocked_classification(arm, decision, event, market, sector):
    if decision['allowed']:
        return 'allowed'
    active = {'event_only': [event], 'market_only': [market], 'sector_budget': [sector],
        'combined': [event, market, sector]}.get(arm, [])
    if any(gate.get('status') in ('unavailable', 'warming_up', 'warmup') for gate in active):
        return 'unavailable_or_warmup'
    if event in active and event.get('status') == 'blackout':
        return 'event_blackout'
    if market in active and market.get('status') == 'shock':
        return 'market_shock'
    if sector in active and sector.get('allowed') is False and sector.get('status') == 'blocked':
        return 'sector_budget'
    return 'unknown_blocked'


def evaluate_cycle(manifest, spec, states, btc_report, snapshot, calendar, *, asof_ms, tracker):
    import cross_market_risk as risk
    from international_event_calendar import evaluate_international_event_calendar

    tracker = deepcopy(tracker)
    tracker['market_history'] = risk.update_forward_history(snapshot, tracker.get('market_history', {}),
        asof_ms=asof_ms, inception_ms=manifest['started_at_ms'])
    exposure = risk.portfolio_exposure(states, asof_ms=asof_ms, policy=spec.get('portfolio_policy'))
    event = evaluate_international_event_calendar(calendar, asof_ms=asof_ms,
        action='new_risk', policy=spec.get('event_policy'))
    decisions, counted_updates = [], []
    for account_id in ACCOUNTS:
        baseline = reported_baseline(account_id, states.get(account_id, {}), btc_report,
            asof_ms=asof_ms, inception_ms=manifest['started_at_ms'],
            max_age_ms=spec.get('source_max_age_ms', 120000))
        market = risk.evaluate_cross_market_risk(snapshot, spec.get('market_policy', {}), asof_ms=asof_ms,
            direction=baseline['direction'], history=tracker['market_history'])
        state = states.get(account_id, {})
        # This is an explicitly labelled original-budget assumption, not a sized order.
        equity = (state.get('_reported') or {}).get('equity', state.get('equity', state.get('wallet_balance', 0)))
        fraction = (state.get('rule') or {}).get('risk_fraction_per_trade', .0025)
        proposed = float(equity or 0) * float(fraction) if account_id != 'btc' else 0
        sector = risk.evaluate_sector_budget(exposure, account_id=account_id, direction=baseline['direction'],
            proposed_risk_usdt=proposed, policy=spec.get('portfolio_policy'))
        gates = {'baseline': _combined(baseline), 'event_only': _combined(baseline, event),
            'market_only': _combined(baseline, market), 'sector_budget': _combined(baseline, sector),
            'combined': _combined(baseline, event, market, sector)}
        opportunity = baseline['opportunity_id']
        if opportunity and baseline['post_inception']:
            tracker.setdefault('opportunities', {}).setdefault(opportunity, {'account_id': account_id,
                'origin_ms': baseline['origin_ms'], 'direction': baseline['direction'], 'first_seen_at_ms': asof_ms})
        for arm, decision in gates.items():
            key = f'{opportunity}:{arm}' if opportunity else None
            effective = bool(baseline['post_inception'] and baseline['allowed'])
            counted = bool(effective and key not in tracker.setdefault('counted', {}))
            row = {'account_id': account_id, 'arm': arm, 'checked_at_ms': asof_ms,
                'opportunity_id': opportunity, 'origin_ms': baseline['origin_ms'], 'direction': baseline['direction'],
                'post_inception': baseline['post_inception'], 'pre_inception': baseline['pre_inception'],
                'baseline_is_reported_not_fill_permit': True, 'no_pnl_simulated': True,
                'eligible_for_effective_filter_comparison': effective, 'counted_once': counted,
                'candidate_effectively_filtered': bool(effective and arm != 'baseline' and not decision['allowed']),
                'blocked_classification': blocked_classification(arm, decision, event, market, sector),
                **decision}
            decisions.append(row)
            if counted:
                update = {'key': key, 'account_id': account_id, 'arm': arm, 'allowed': decision['allowed'],
                    'counted_at_ms': asof_ms, 'opportunity_id': opportunity, 'reasons': deepcopy(decision['reasons']),
                    'origin_ms': baseline['origin_ms'], 'direction': baseline['direction'],
                    'blocked_classification': row['blocked_classification']}
                tracker['counted'][key] = update
                counted_updates.append(update)
    tracker['cycles'] = int(tracker.get('cycles', 0)) + 1
    return decisions, tracker, {'event': event, 'portfolio_exposure': exposure,
        'counted_updates': counted_updates, 'observation_only': True,
        'sector_risk_assumption': 'Current original positions plus one original per-account risk budget; no counterfactual holdings.'}


def archive(output, payload):
    raw = payload if isinstance(payload, bytes) else canonical(payload)
    key = digest(raw)
    path = Path(output) / 'evidence' / (key + '.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as handle:
            handle.write(gzip.compress(raw, mtime=0))
    return key


def recover_tracker(output, manifest):
    tracker = read(Path(output) / 'tracker.json')
    if tracker.get('started_at_ms') != manifest['started_at_ms']:
        raise ValueError('Observer tracker epoch mismatch')
    path = Path(output) / 'decisions.jsonl'
    saved_sequence = tracker.get('journal_sequence', 0)
    checkpoint = tracker.get('journal_offset')
    sequence = saved_sequence if checkpoint is not None else 0
    offset = checkpoint if checkpoint is not None else 0
    last_start = tracker.get('last_record_start', 0)
    last_hash = tracker.get('last_record_sha256')
    if type(offset) is not int or offset < 0 or type(last_start) is not int or not 0 <= last_start <= offset:
        raise ValueError('Observer history checkpoint invalid')
    size = path.stat().st_size if path.exists() else 0
    if size < offset or not path.exists() and saved_sequence:
        raise ValueError('Observer decision history was shortened')
    if path.exists():
        with path.open('rb') as handle:
            if checkpoint is not None and saved_sequence:
                if not isinstance(last_hash, str) or last_start == offset:
                    raise ValueError('Observer history checkpoint proof missing')
                handle.seek(last_start)
                if digest(handle.read(offset - last_start)) != last_hash:
                    raise ValueError('Observer last committed history record changed')
            elif checkpoint is not None and (offset or last_hash is not None):
                raise ValueError('Observer empty history checkpoint invalid')
            handle.seek(offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    raise ValueError('Observer decision history contains an incomplete record')
                row = json.loads(line)
                if row['sequence'] != sequence + 1 or row['started_at_ms'] != manifest['started_at_ms']:
                    raise ValueError('Observer decision history sequence/epoch invalid')
                sequence = row['sequence']
                if sequence > saved_sequence:
                    for update in row.get('counted_updates', []):
                        tracker.setdefault('counted', {})[update['key']] = update
                        if update.get('opportunity_id') and update.get('origin_ms'):
                            tracker.setdefault('opportunities', {}).setdefault(update['opportunity_id'], {
                                'account_id': update['account_id'], 'origin_ms': update['origin_ms'],
                                'direction': update['direction'], 'first_seen_at_ms': update['counted_at_ms']})
                last_start, last_hash, offset = start, digest(line), handle.tell()
    if sequence < saved_sequence:
        raise ValueError('Observer decision history was shortened')
    tracker['journal_sequence'] = sequence
    tracker.update(journal_offset=offset, last_record_start=last_start, last_record_sha256=last_hash)
    tracker['cycles'] = max(tracker.get('cycles', 0), sequence)
    return tracker


def bounded_collect(collector, config, *, previous=None, now_ms=None, timeout_seconds=35):
    """A stuck optional collector cannot stop heartbeats or grow unbounded workers.

    Timed-out results are discarded. The daemon may finish later but has no
    writer capability; only this runner publishes accepted result bytes.
    """
    global _COLLECT_INFLIGHT
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 45:
        raise ValueError('Observer collection deadline must be positive and at most 45 seconds')
    with _COLLECT_LOCK:
        if _COLLECT_INFLIGHT is not None:
            if _COLLECT_INFLIGHT['thread'].is_alive():
                raise TimeoutError('Prior optional collection is still outstanding')
            _COLLECT_INFLIGHT = None
        results = queue.Queue(maxsize=1)
        def work():
            try:
                value = collector(deepcopy(config), previous=deepcopy(previous), now_ms=now_ms)
                results.put((True, value))
            except BaseException as exc:
                results.put((False, exc))
        thread = threading.Thread(target=work, name='cross-market-observer-collection', daemon=True)
        _COLLECT_INFLIGHT = {'thread': thread, 'results': results}
        thread.start()
    try:
        success, result = results.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError('Optional collection deadline exceeded') from exc
    finally:
        with _COLLECT_LOCK:
            if _COLLECT_INFLIGHT is not None and _COLLECT_INFLIGHT['results'] is results and not thread.is_alive():
                _COLLECT_INFLIGHT = None
    if not success:
        raise result
    if not isinstance(result, dict):
        raise ValueError('Collector snapshot must be an object')
    return result


def publish(output, manifest, spec, tracker, *, running, asof_ms, status='starting', **details):
    counts = {arm: {'reported_eligible_opportunities': 0, 'observed_allowed': 0, 'observed_blocked': 0,
        'observed_blocked_unavailable_or_warmup': 0, 'observed_blocked_event_blackout': 0,
        'observed_blocked_market_shock': 0, 'observed_blocked_sector_budget': 0,
        'observed_blocked_unknown': 0} for arm in ARMS}
    for row in tracker.get('counted', {}).values():
        counts[row['arm']]['reported_eligible_opportunities'] += 1
        counts[row['arm']]['observed_allowed' if row['allowed'] else 'observed_blocked'] += 1
        if not row['allowed']:
            category = row.get('blocked_classification', 'unknown_blocked')
            label = 'unknown' if category == 'unknown_blocked' else category
            counts[row['arm']]['observed_blocked_' + label] += 1
    payload = {'schema_version': 1, 'mode': 'OBSERVATION_ONLY', 'places_orders': False,
        'no_pnl_simulated': True, 'running': running, 'pid': os.getpid(), 'status': status,
        'experiment_id': manifest['experiment_id'], 'started_at_ms': manifest['started_at_ms'],
        'started_at_utc': manifest['started_at_utc'], 'updated_at_utc': iso(asof_ms),
        'heartbeat_at_ms': asof_ms, 'heartbeat_at_utc': iso(asof_ms),
        'poll_seconds': spec['poll_seconds'], 'collect_seconds': spec['collect_seconds'],
        'cycles': tracker.get('cycles', 0), 'post_inception_opportunity_count': len(tracker.get('opportunities', {})),
        'counts': counts, 'baseline_is_reported_not_fill_permit': True,
        'original_accounts_modified': False, 'automatic_strategy_promotion': False,
        'pnl_comparison_available': False, 'cost_sensitivity_evaluated': False,
        'review_interpretation': 'Admission observations only. Missing candidate data blocks its observed new-risk gate; no return, fill or promotion claim.',
        'counting_rule': 'One first reported-eligible observation per post-inception opportunity and arm; later recovery never revises that first observation.',
        'calendar_coverage_note': 'International supplement only; no claim of complete Fed, BEA or company-event coverage. Original account gates are retained as reported.',
        **details}
    write(Path(output) / 'status.json', payload)
    return payload


def run_once(output, spec, manifest, *, asof_ms=None, collector=None):
    from cross_market_observations import collect_observations, load_config
    from international_event_calendar import load_calendar

    output = Path(output)
    now = int(time.time() * 1000) if asof_ms is None else asof_ms
    tracker = recover_tracker(output, manifest)
    publish(output, manifest, spec, tracker, running=True, asof_ms=now, status='collecting')
    calendar, raw_calendar, calendar_refresh = maintained_calendar(output, spec, asof_ms=asof_ms)
    snapshot_path = output / 'latest_observations.json'
    previous = read(snapshot_path) if snapshot_path.exists() else None
    due = tracker.get('last_collect_at_ms') is None or now - tracker['last_collect_at_ms'] >= spec['collect_seconds'] * 1000
    errors = {}
    if due:
        try:
            collect = collector or collect_observations
            snapshot = bounded_collect(collect, load_config(contained(spec['observation_config'])),
                previous=previous, now_ms=None if asof_ms is None else now,
                timeout_seconds=spec.get('collector_deadline_seconds', 35))
        except Exception as exc:
            snapshot = deepcopy(previous) if isinstance(previous, dict) else {'schema_version': 1, 'observations': {}, 'sources': {}}
            snapshot.update(observer_collection_error=type(exc).__name__, observer_collection_failed_at_ms=now)
            errors['collection'] = type(exc).__name__
        write(snapshot_path, snapshot)
        tracker['last_collect_at_ms'] = now
    else:
        snapshot = previous or {'schema_version': 1, 'observations': {}, 'sources': {}}
    # Take a fresh observation cutoff after slow reads; no host time is used as an exchange fill clock.
    now = int(time.time() * 1000) if asof_ms is None else asof_ms
    states, inputs_hash, inputs_diagnostic = inspect_inputs(spec, asof_ms=now)
    try:
        report, report_hash = selected_report(spec)
    except (OSError, ValueError, TypeError) as exc:
        report, report_hash = {}, None
        errors['btc_report'] = type(exc).__name__
    if calendar_refresh.get('success') is False:
        errors['calendar_refresh'] = calendar_refresh.get('error_type', 'VerificationUnavailable')
    context_hashes = {'selected_original_states': archive(output, states),
        'observations': archive(output, snapshot), 'international_calendar_raw': archive(output, raw_calendar),
        'selected_btc_report': archive(output, report), 'original_btc_report_sha256': report_hash,
        'inputs_digest': inputs_hash}
    decisions, tracker, diagnostics = evaluate_cycle(manifest, spec, states, report, snapshot, calendar,
        asof_ms=now, tracker=tracker)
    sequence = tracker['journal_sequence'] + 1
    record = {'sequence': sequence, 'started_at_ms': manifest['started_at_ms'], 'observed_at_ms': now,
        'mode': 'OBSERVATION_ONLY', 'places_orders': False, 'no_pnl_simulated': True,
        'context_sha256': context_hashes, 'decisions': decisions, 'counted_updates': diagnostics['counted_updates']}
    journal_path = output / 'decisions.jsonl'
    record_bytes = canonical(record) + b'\n'
    expected_offset = tracker['journal_offset']
    if (journal_path.stat().st_size if journal_path.exists() else 0) != expected_offset:
        raise ValueError('Observer history changed after checkpoint verification')
    with journal_path.open('ab') as handle:
        handle.write(record_bytes)
        handle.flush()
        os.fsync(handle.fileno())
        tracker.update(journal_offset=handle.tell(), last_record_start=expected_offset,
            last_record_sha256=digest(record_bytes))
    tracker['journal_sequence'] = sequence
    write(output / 'tracker.json', tracker)
    write(output / 'selected_states_latest.json', states)
    write(output / 'calendar_current.json', calendar)
    return publish(output, manifest, spec, tracker, running=True, asof_ms=now,
        status='degraded' if errors or inputs_diagnostic['errors'] else 'observing', decisions=decisions,
        context_sha256=context_hashes, source_diagnostics=inputs_diagnostic, errors=errors,
        event=diagnostics['event'], portfolio_exposure=diagnostics['portfolio_exposure'],
        market_history=tracker.get('market_history', {}), calendar_refresh=calendar_refresh,
        sector_risk_assumption=diagnostics['sector_risk_assumption'])


def maintained_calendar(output, spec, *, asof_ms=None, refresher=None):
    """Renew only through official semantic verification; never heartbeat clocks."""
    from refresh_international_event_calendar import refresh_calendar, load_verified_calendar

    output = Path(output)
    template = contained(spec['international_calendar'])
    now = int(time.time() * 1000) if asof_ms is None else asof_ms
    status = {}
    if spec.get('calendar_refresh_enabled') is True:
        attempt = output / 'calendar_refresh_status.json'
        try:
            status = read(attempt)
            next_attempt = status.get('next_attempt_after_ms')
            due = type(next_attempt) is not int or next_attempt <= now or status.get('last_attempt_at_ms', now + 1) > now
        except (OSError, ValueError, TypeError):
            due = True
        if due:
            try:
                status = (refresher or refresh_calendar)(template, output, asof_ms=asof_ms)
            except Exception as exc:
                # A local failure must not refresh the old calendar's clocks.
                status = {'success': False, 'error_type': type(exc).__name__,
                    'last_attempt_at_ms': now, 'next_attempt_after_ms': now + 3_600_000}
                write(attempt, status)
    try:
        validated = load_verified_calendar(template, output)
        raw = (output / 'calendar_verified.json').read_bytes()
        if len(raw) > 1024 * 1024 or json.loads(raw.decode('utf-8-sig')) != validated:
            raise ValueError('Verified calendar changed during capture')
        return validated, raw, status
    except (OSError, ValueError, TypeError, KeyError):
        try:
            raw = template.read_bytes()
            if len(raw) > 1024 * 1024:
                raise ValueError('Calendar size limit')
            calendar = json.loads(raw.decode('utf-8-sig'))
            return calendar, raw, status
        except (OSError, ValueError, TypeError):
            return {}, b'{}', {**status, 'success': False, 'error_type': 'CalendarUnavailable'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--require-existing', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--check', action='store_true', help='Read-only definition/reference validation; no network or output initialization')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_design(args.config), ensure_ascii=False, indent=2))
        return 0
    spec = load_design(args.config)
    expected = contained(spec.get('output_dir', 'data/cross_market_observer'))
    output = (args.output_dir or expected).resolve()
    if output != expected:
        raise ValueError('Observer output must equal its frozen design')
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with exclusive_process_lock(output / 'runner.lock'):
        manifest = prepare(output, spec, fingerprint(spec, args.config), int(time.time() * 1000), args.require_existing)
        try:
            while not stop.is_set():
                try:
                    status = run_once(output, spec, manifest)
                except Exception as exc:
                    tracker = recover_tracker(output, manifest)
                    status = publish(output, manifest, spec, tracker, running=True,
                        asof_ms=int(time.time() * 1000), status='degraded', errors={'cycle': type(exc).__name__})
                print(json.dumps({key: status[key] for key in ('experiment_id', 'status', 'heartbeat_at_utc')}, ensure_ascii=False), flush=True)
                if args.once:
                    break
                stop.wait(spec['poll_seconds'])
        finally:
            prior = read(output / 'status.json') if (output / 'status.json').exists() else {}
            prior.update(running=False, heartbeat_at_ms=int(time.time() * 1000))
            prior['heartbeat_at_utc'] = prior['updated_at_utc'] = iso(prior['heartbeat_at_ms'])
            write(output / 'status.json', prior)


if __name__ == '__main__':
    main()
