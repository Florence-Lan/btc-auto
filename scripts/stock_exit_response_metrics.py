"""Read-only, causal follow-up diagnostics for independent exit experiments.

These diagnostics never change a trading ledger or feed a trading decision.
Matching is by signal boundary and direction, separately for each symbol/cost.
Cash-window fees/funding are explanatory and must not be added to closed net PnL.
Only completed hourly *closes* measure continuation; intrabar extrema are unused.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re

HOUR = 3_600_000
POLICY_REASONS = frozenset({
    'hourly_exit_response_breakout_failure',
    'hourly_exit_response_giveback',
    'hourly_exit_response_combined',
})


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _integer(value):
    value = _number(value)
    return int(value) if value is not None and value >= 0 and value.is_integer() else None


def _component(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Invalid diagnostics path component')
    return value


def _read_rows(path, label, issues):
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except OSError as exc:
        issues.append({'source': label, 'error_type': type(exc).__name__})
        return [], False
    rows, seen, valid = [], set(), True
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Object required')
            key = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
        except (ValueError, TypeError, OverflowError):
            issues.append({'source': label, 'line': line_number, 'error_type': 'invalid_json_row'})
            valid = False
            continue
        if key in seen:
            issues.append({'source': label, 'line': line_number, 'error_type': 'duplicate_row'})
            valid = False
            continue
        seen.add(key)
        rows.append(row)
    return rows, valid


def _entry(row, asof):
    timestamp, boundary = _integer(row.get('entry_time_ms')), _integer(row.get('signal_boundary_ms'))
    direction = row.get('direction')
    price, stop, qty = [_number(row.get(k)) for k in ('entry_price', 'initial_stop', 'initial_qty')]
    if (timestamp is None or boundary is None or boundary % HOUR
            or boundary > timestamp or timestamp > asof
            or isinstance(direction, bool) or direction not in (-1, 1)
            or any(v is None or v <= 0 for v in (price, stop, qty))
            or direction * (price - stop) <= 0):
        return None
    return {'entry_time_ms': timestamp, 'signal_boundary_ms': boundary, 'direction': direction,
            'entry_price': price, 'initial_stop': stop, 'initial_qty': qty,
            'entry_trigger': row.get('entry_trigger')}


def _close(entry, rows, asof):
    legs = sorted((row for row in rows
                   if _integer(row.get('entry_time_ms')) == entry['entry_time_ms']
                   and _integer(row.get('time_ms')) is not None
                   and entry['entry_time_ms'] <= int(row['time_ms']) <= asof),
                  key=lambda row: (int(row['time_ms']), _integer(row.get('sequence')) or 0))
    quantities = [_number(row.get('qty')) for row in legs]
    valid_qty = all(q is not None and q > 0 for q in quantities)
    qty = sum(quantities) if valid_qty else None
    tolerance = max(1e-9, entry['initial_qty'] * 1e-8)
    complete = bool(legs and valid_qty and legs[-1].get('position_closed') is True
                    and abs(qty - entry['initial_qty']) <= tolerance)
    prices = [_number(row.get('price')) for row in legs]
    net = [_number(row.get('net_pnl')) for row in legs]
    return {'complete': complete,
            'time_ms': int(legs[-1]['time_ms']) if complete else None,
            'vwap': sum(q * p for q, p in zip(quantities, prices)) / qty
                    if complete and all(p is not None and p > 0 for p in prices) else None,
            'net_pnl': sum(net) if complete and all(n is not None for n in net) else None,
            'exit_reasons': list(dict.fromkeys(str(row.get('reason', 'unknown')) for row in legs)),
            'closed_qty': qty,
            'partial_realized_net_pnl': sum(net) if net and all(n is not None for n in net) else None}


def _holds(state, entry):
    pos = state.get('position')
    return (isinstance(pos, dict)
            and _integer(pos.get('entry_time', pos.get('entry_time_ms'))) == entry['entry_time_ms']
            and (_number(pos.get('qty')) or 0) > 0)


def _with_late_funding(closure, entry, funding, known):
    """Allocate later-published closed-inventory funding to its unique position.

    Standard paper accounts hold one position at a time. The open-inventory
    portion is already in closed legs; only closed_funding_debit is added here.
    """
    closure['closed_journal_net_pnl'] = closure['net_pnl']
    closure['late_closed_funding_debit'] = None
    closure['net_pnl_basis'] = 'closed legs net less known late closed-inventory funding'
    if not closure['complete']:
        return closure
    late = []
    for row in funding:
        timestamp = _integer(row.get('fundingTime'))
        if timestamp is not None and entry['entry_time_ms'] < timestamp <= closure['time_ms']:
            late.append(_number(row.get('closed_funding_debit', 0)))
    if not known or any(value is None for value in late):
        closure['net_pnl'] = None
        return closure
    closure['late_closed_funding_debit'] = sum(late)
    closure['net_pnl'] = (closure['net_pnl'] - sum(late)
                          if closure['net_pnl'] is not None else None)
    return closure


def _candles(rows, asof, issues):
    result = {}
    conflicts = set()
    for row in rows:
        try:
            start, last = _integer(row[0]), _integer(row[6])
            price = _number(row[4])
            if (start is None or last is None or start % HOUR or last != start + HOUR - 1
                    or price is None or price <= 0):
                raise ValueError('Invalid hourly close')
            boundary = last + 1
            if boundary > asof:
                continue
            if boundary in result and result[boundary] != price:
                conflicts.add(boundary)
            result[boundary] = price
        except (IndexError, TypeError, ValueError):
            issues.append({'source': 'closed_candles', 'error_type': 'invalid_hourly_close'})
    for boundary in conflicts:
        result.pop(boundary, None)
        issues.append({'source': 'closed_candles', 'time_ms': boundary, 'error_type': 'conflicting_hourly_close'})
    return result


def _follow_up(previous, start, end, complete, baseline_entry, exit_price, candles, asof, issues):
    # Preserve the first observed close across rolling feeds and subsequent revisions.
    observed = {}
    for row in previous.get('observed_hourly_closes', []):
        timestamp, price = _integer(row.get('close_ms')), _number(row.get('close'))
        if timestamp is not None and start < timestamp <= end and price is not None and price > 0:
            observed[timestamp] = row
    for timestamp, price in candles.items():
        if not start < timestamp <= end:
            continue
        if timestamp in observed:
            if observed[timestamp]['close'] != price:
                issues.append({'source': 'closed_candles', 'time_ms': timestamp,
                               'error_type': 'revised_close_first_observation_retained'})
            continue
        observed[timestamp] = {'close_ms': timestamp, 'close': price, 'first_seen_at_ms': asof}
    rows = [observed[k] for k in sorted(observed)]
    side, entry = baseline_entry['direction'], baseline_entry['entry_price']
    risk = abs(entry - baseline_entry['initial_stop'])
    original_r = [side * (row['close'] - entry) / risk for row in rows]
    from_exit = [side * (row['close'] - exit_price) / risk for row in rows] if exit_price is not None else []
    trigger = baseline_entry.get('entry_trigger')
    level = _number(trigger.get('breakout_level')) if isinstance(trigger, dict) else None
    breakout_sides = [side * (row['close'] - level) > 0 for row in rows] if level is not None else []
    # Merely staying beyond the boundary after a giveback exit is continuation,
    # not evidence of a new crossing. Only observed close-to-close crossings count.
    resumed = any(not a and b for a, b in zip(breakout_sides, breakout_sides[1:])) if len(breakout_sides) >= 2 else None
    first_expected = (start // HOUR + 1) * HOUR
    expected = max(0, (end // HOUR * HOUR - first_expected) // HOUR + 1)
    missing = max(0, expected - len(rows))
    return {'start_ms': start, 'end_ms': end, 'status': 'complete' if complete else 'censored',
            'closed_hour_count': len(rows), 'expected_closed_hour_count': expected,
            'missing_hour_count': missing, 'coverage_complete': missing == 0,
            'last_close_ms': rows[-1]['close_ms'] if rows else None,
            'max_favorable_r_from_exit': max(0.0, max(from_exit)) if from_exit else None,
            'max_adverse_r_from_exit': max(0.0, -min(from_exit)) if from_exit else None,
            'max_original_entry_r': max(original_r) if original_r else None,
            'reentered_breakout_side': resumed,
            'breakout_side_observed': any(breakout_sides) if breakout_sides else None,
            'reentry_basis': 'observed hourly close on invalid side followed by close on original breakout side',
            'observed_hourly_closes': rows,
            'price_basis': 'completed_hourly_close; R=baseline_initial_entry_stop_distance'}


def _window_activity(entries, closes, fills, funding, valid, start, end, asof, candidate_state):
    extras = sorted((entry for entry in entries if start < entry['entry_time_ms'] <= end),
                    key=lambda entry: entry['entry_time_ms'])
    fully_closed, open_positions, partial = [], [], []
    for entry in extras:
        closure = _with_late_funding(_close(entry, closes, min(end, asof)),
                                    entry, funding, valid['funding'])
        if closure['complete']:
            fully_closed.append(closure)
        else:
            open_positions.append({'entry_time_ms': entry['entry_time_ms'], 'direction': entry['direction'],
                                   'initial_qty': entry['initial_qty'], 'closed_qty': closure['closed_qty']})
            if closure['partial_realized_net_pnl'] is not None:
                partial.append(closure['partial_realized_net_pnl'])
    fill_rows = [row for row in fills if _integer(row.get('time_ms')) is not None
                 and start < int(row['time_ms']) <= end]
    fee_values = [_number(row.get('fee')) for row in fill_rows]
    funding_rows = [row for row in funding if _integer(row.get('fundingTime')) is not None
                    and start < int(row['fundingTime']) <= end]
    funding_values = [_number(row.get('debit')) for row in funding_rows]
    nets = [row['net_pnl'] for row in fully_closed]
    notional = sum(entry['initial_qty'] * entry['entry_price'] for entry in extras)
    live_position = candidate_state.get('position')
    live_entry = (_integer(live_position.get('entry_time', live_position.get('entry_time_ms')))
                  if isinstance(live_position, dict) else None)
    untracked_open = (live_entry is not None and start < live_entry <= end
                      and (_number(live_position.get('qty')) or 0) > 0
                      and live_entry not in {entry['entry_time_ms'] for entry in extras})
    fees_known = (valid['fills'] and all(x is not None and x >= 0 for x in fee_values)
                  and all(any(_integer(row.get('time_ms')) == entry['entry_time_ms']
                              for row in fill_rows) for entry in extras))
    funding_known = valid['funding'] and all(x is not None for x in funding_values)
    net_known = valid['closed_trades'] and all(x is not None for x in nets)
    incomplete = bool(open_positions) or untracked_open or not all(valid.values()) or not (fees_known and funding_known and net_known)
    return {'extra_entries': len(extras), 'extra_entry_times_ms': [e['entry_time_ms'] for e in extras],
            'extra_entry_directions': [e['direction'] for e in extras],
            'extra_entry_notional': notional,
            'fees_paid': sum(fee_values) if fees_known else None,
            'funding_pnl': -sum(funding_values) if funding_known else None,
            'completed_extra_positions': len(fully_closed),
            'closed_net_pnl': sum(nets) if net_known else None,
            'partial_realized_net_pnl': sum(partial) if partial and valid['closed_trades'] else None,
            'open_positions': open_positions,
            'untracked_open_position': untracked_open,
            'incomplete': incomplete,
            'status': 'incomplete' if incomplete else 'complete',
            'accounting_basis': ('closed_net_pnl covers complete extra positions only; fees/funding are separate '
                                 'cash-window diagnostics, not additive to that net; funding uses settlement time; '
                                 'known later-published closed-inventory funding is allocated once to trade net')}


def update_diagnostics(output: Path, symbol: str, cost: str, variant: str,
                       baseline_state: dict, candidate_state: dict,
                       closed_candles: list, asof_ms: int) -> dict:
    """Update one persistent, non-trading tracker; no unmatched PnL improvement.

    The journals must be append-only. Missing sources and ambiguous cohorts are
    reported rather than inferred from a flat state. All monetary values are USDT.
    A baseline trade already fully closed after the candidate remains matchable.
    """
    symbol, cost, variant = map(_component, (symbol, cost, variant))
    if variant == 'baseline':
        raise ValueError('Candidate variant must differ from baseline')
    asof = _integer(asof_ms)
    if asof is None:
        raise ValueError('Invalid diagnostics asof')
    output = Path(output)
    target = output / 'paired' / symbol / f'{variant}_{cost}.json'
    issues, previous = [], {}
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding='utf-8'))
            if (not isinstance(previous, dict) or previous.get('schema_version') != 1 or previous.get('symbol') != symbol
                    or previous.get('cost') != cost or previous.get('variant') != variant):
                previous = {}
                issues.append({'source': 'tracker', 'error_type': 'tracker_identity_mismatch'})
        except (OSError, ValueError, TypeError):
            issues.append({'source': 'tracker', 'error_type': 'invalid_tracker'})
    if (_integer(previous.get('asof_ms')) or 0) > asof:
        return {'schema_version': 1, 'symbol': symbol, 'cost': cost, 'variant': variant,
                'asof_ms': asof, 'status': 'out_of_order', 'source_issues': [
                    {'source': 'tracker', 'error_type': 'asof_precedes_saved_tracker'}],
                'matched_pairs': [], 'unpaired_entries': [], 'summary': {}}
    data, valid = {}, {}
    for arm in ('baseline', variant):
        data[arm], valid[arm] = {}, {}
        for name in ('entries', 'fills', 'closed_trades', 'funding'):
            data[arm][name], valid[arm][name] = _read_rows(
                output / symbol / f'{arm}_{cost}' / f'{name}.jsonl', f'{arm}.{name}', issues)
    candles = _candles(closed_candles, asof, issues)
    entries, by_key, unpaired = {}, {}, []
    for arm in ('baseline', variant):
        entries[arm], by_key[arm] = [], {}
        for row in data[arm]['entries']:
            entry = _entry(row, asof)
            if entry is None:
                if (_integer(row.get('entry_time_ms')) or 0) > asof:
                    continue
                issues.append({'source': f'{arm}.entries', 'error_type': 'invalid_entry'})
                valid[arm]['entries'] = False
                continue
            entries[arm].append(entry)
            key = (entry['signal_boundary_ms'], entry['direction'])
            by_key[arm].setdefault(key, []).append(entry)
    old_pairs = {row['pair_id']: row for row in previous.get('matched_pairs', [])
                 if isinstance(row, dict) and 'pair_id' in row}
    pairs = []
    for key in sorted(set(by_key['baseline']) | set(by_key[variant])):
        baseline_entries, candidates = by_key['baseline'].get(key, []), by_key[variant].get(key, [])
        if len(baseline_entries) != 1 or len(candidates) != 1:
            reason = 'ambiguous_signal_cohort' if len(baseline_entries) > 1 or len(candidates) > 1 else 'no_matching_signal_entry'
            for arm, rows in (('baseline', baseline_entries), (variant, candidates)):
                unpaired.extend({'arm': arm, 'entry_time_ms': row['entry_time_ms'],
                                 'signal_boundary_ms': key[0], 'direction': key[1], 'reason': reason} for row in rows)
            continue
        baseline, candidate = baseline_entries[0], candidates[0]
        pair_id = f'{key[0]}:{key[1]}'
        critical_valid = all(valid[arm][name] for arm in ('baseline', variant)
                             for name in ('entries', 'closed_trades'))
        if not critical_valid and pair_id in old_pairs:
            retained = copy.deepcopy(old_pairs[pair_id])
            retained.update(status='source_unavailable', metrics_fresh=False)
            pairs.append(retained)
            continue
        baseline_close = _with_late_funding(_close(baseline, data['baseline']['closed_trades'], asof),
                                            baseline, data['baseline']['funding'], valid['baseline']['funding'])
        candidate_close = _with_late_funding(_close(candidate, data[variant]['closed_trades'], asof),
                                             candidate, data[variant]['funding'], valid[variant]['funding'])
        pair = {'pair_id': pair_id, 'signal_boundary_ms': key[0], 'direction': key[1],
                'baseline_entry': baseline, 'candidate_entry': candidate,
                'entry_differences': {'time_ms': candidate['entry_time_ms'] - baseline['entry_time_ms'],
                                      'qty': candidate['initial_qty'] - baseline['initial_qty'],
                                      'price': candidate['entry_price'] - baseline['entry_price']},
                'baseline_exit': baseline_close if baseline_close['complete'] else None,
                'candidate_exit': candidate_close if candidate_close['complete'] else None,
                'candidate_early_exit': None, 'follow_up': None, 'candidate_window_activity': None,
                'status': 'candidate_still_open' if not candidate_close['complete'] else 'not_earlier_than_baseline'}
        if not critical_valid:
            pair['status'] = 'source_unavailable'
        if candidate_close['complete'] and critical_valid:
            early_time = candidate_close['time_ms']
            later_baseline = baseline_close['complete'] and baseline_close['time_ms'] > early_time
            still_holding = not baseline_close['complete'] and _holds(baseline_state, baseline)
            if baseline['entry_time_ms'] <= early_time and (later_baseline or still_holding):
                pair['candidate_early_exit'] = candidate_close
                pair['policy_early_exit'] = any(reason in POLICY_REASONS for reason in candidate_close['exit_reasons'])
                end = baseline_close['time_ms'] if later_baseline else asof
                old = old_pairs.get(pair_id, {})
                if (old.get('baseline_entry') != baseline or old.get('candidate_entry') != candidate):
                    old = {}
                pair['follow_up'] = _follow_up(old.get('follow_up') or {}, early_time, end,
                                             bool(later_baseline), baseline, candidate_close['vwap'],
                                             candles, asof, issues)
                window_valid = {k: valid[variant][k] for k in ('entries', 'fills', 'closed_trades', 'funding')}
                activity = _window_activity(entries[variant], data[variant]['closed_trades'],
                                            data[variant]['fills'], data[variant]['funding'],
                                            window_valid, early_time, end, asof, candidate_state)
                directions = [candidate['direction']] + activity['extra_entry_directions']
                activity['direction_switches'] = sum(a != b for a, b in zip(directions, directions[1:]))
                pair['candidate_window_activity'] = activity
                pair['status'] = 'complete' if later_baseline else 'censored'
            elif not baseline_close['complete']:
                pair['status'] = 'baseline_holding_unknown'
        pairs.append(pair)
    current_ids = {row['pair_id'] for row in pairs}
    for pair_id, old in old_pairs.items():
        if pair_id not in current_ids:
            # A vanished append-only cohort is a source failure, not zero improvement.
            retained = copy.deepcopy(old)
            retained['status'] = 'source_unavailable'
            retained['metrics_fresh'] = False
            pairs.append(retained)
            issues.append({'source': 'tracker', 'pair_id': pair_id, 'error_type': 'previous_cohort_missing'})
    result = {'schema_version': 1, 'symbol': symbol, 'cost': cost, 'variant': variant,
              'asof_ms': asof, 'status': 'degraded' if issues else 'ok', 'source_issues': issues,
              'matched_pairs': pairs, 'unpaired_entries': unpaired,
              'summary': {'matched_count': sum(p['status'] != 'source_unavailable' for p in pairs),
                          'early_exit_count': sum(p.get('candidate_early_exit') is not None and p['status'] != 'source_unavailable' for p in pairs),
                          'completed_followups': sum(p['status'] == 'complete' for p in pairs),
                          'censored_followups': sum(p['status'] == 'censored' for p in pairs)},
              'limitations': ['No unmatched PnL improvement is computed.',
                              'Close-based MFE/MAE omit intrabar extrema and unobserved hours.',
                              'Censored baseline windows and open extra positions remain incomplete.',
                              'Cash-window fees/funding must not be added to completed-trade net PnL.']}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, target)
    return result
