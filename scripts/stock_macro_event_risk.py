"""Opt-in, entry-only stock admission using the existing public macro calendar.

The collector owns the fixed 30-minute-before / 90-minute-after windows. This
adapter validates their provenance and reuses event_risk's causal decision,
including its inclusive end boundary and supersession semantics. It does not
fetch data, book orders, close positions, or renew cached source receipt clocks.
Read the snapshot again for retries and immediately before each entry booking.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import event_risk
import public_context


MODE = 'public_scheduled_macro_blackout_v1'
POLICY_KEY = 'stock_macro_event_risk_policy'
POLICY = {'mode': MODE, 'pre_event_minutes': 30, 'post_event_minutes': 90,
          'calendar_failure_action': 'block_new_risk'}
REQUIRED_SOURCES = ('fomc_calendar', 'economic_calendar')
SOURCE_MAX_AGE_MS = int(public_context.SOURCE_MAX_AGE.total_seconds() * 1000)
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024


def enabled(config):
    return POLICY_KEY in config


def _time(value, reason):
    if type(value) is not int or value < 0:
        raise ValueError(reason)
    return value


def _text(value, reason):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(reason)
    return value


def _stamp(value, reason='stock_macro_event_time_invalid'):
    """Reject naive timestamps instead of inheriting the machine's timezone."""
    _text(value, reason)
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(reason)
        return _time(int(parsed.astimezone(timezone.utc).timestamp() * 1000), reason)
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        raise ValueError(reason) from exc


def _url(value):
    _text(value, 'stock_macro_event_source_invalid')
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('stock_macro_event_source_invalid')


def validate(config):
    """An absent policy opts out; an explicit policy is the fixed v1 experiment."""
    if not enabled(config):
        return
    policy = config[POLICY_KEY]
    if not isinstance(policy, Mapping) or set(policy) != set(POLICY) | {
            'snapshot_path', 'effective_from_ms'}:
        raise ValueError('stock_macro_event_policy_invalid')
    for key, expected in POLICY.items():
        if type(policy[key]) is not type(expected) or policy[key] != expected:
            raise ValueError('stock_macro_event_policy_invalid')
    _text(policy['snapshot_path'], 'stock_macro_event_snapshot_path_invalid')
    _time(policy['effective_from_ms'], 'stock_macro_event_effective_time_invalid')


def _coverage(payload, reference_ms):
    """Use the last causally observed check, never a newer success or fetch flag."""
    checks = payload.get('coverage_checks')
    if not isinstance(checks, list) or not checks:
        raise ValueError('stock_macro_event_coverage_unavailable')
    selected = None
    previous = -1
    generated = _stamp(payload.get('generated_at_utc'), 'stock_macro_event_snapshot_time_invalid')
    for row in checks:
        if not isinstance(row, Mapping):
            raise ValueError('stock_macro_event_coverage_invalid')
        stamp = _stamp(row.get('available_at_utc'), 'stock_macro_event_coverage_time_invalid')
        if not previous <= stamp <= generated:
            raise ValueError('stock_macro_event_coverage_time_invalid')
        previous = stamp
        if stamp <= reference_ms:
            selected = row, stamp
    if selected is None:
        raise ValueError('stock_macro_event_coverage_not_observed')
    row, checked = selected
    if reference_ms - checked > SOURCE_MAX_AGE_MS:
        raise ValueError('stock_macro_event_coverage_stale')
    states = row.get('sources')
    if not isinstance(states, Mapping):
        raise ValueError('stock_macro_event_coverage_invalid')
    sources = {}
    for name in REQUIRED_SOURCES:
        state = states.get(name)
        if (not isinstance(state, Mapping) or type(state.get('ok')) is not bool
                or type(state.get('fetch_ok')) is not bool):
            raise ValueError('stock_macro_event_source_status_invalid:' + name)
        success = _stamp(state.get('last_success_at_utc'),
                         'stock_macro_event_source_clock_invalid:' + name)
        if success > checked:
            raise ValueError('stock_macro_event_source_clock_invalid:' + name)
        if not state['ok']:
            raise ValueError('stock_macro_event_source_unavailable:' + name)
        if reference_ms - success > SOURCE_MAX_AGE_MS:
            raise ValueError('stock_macro_event_source_stale:' + name)
        sources[name] = {'ok': True, 'fetch_ok': state['fetch_ok'],
                         'last_success_at_ms': success, 'age_ms': reference_ms - success,
                         'valid_until_ms': success + SOURCE_MAX_AGE_MS,
                         'status': 'live' if state['fetch_ok'] else 'cached'}
    return generated, checked, sources


def _events(payload, reference_ms, generated, sources):
    """Require real calendar records and their unchanged collector event mirrors."""
    calendars = payload.get('calendars')
    raw_events = payload.get('events')
    if not isinstance(calendars, list) or not isinstance(raw_events, list):
        raise ValueError('stock_macro_event_snapshot_invalid')
    calendar_rows, revisions = [], {}
    for row in calendars:
        if not isinstance(row, Mapping):
            raise ValueError('stock_macro_event_calendar_invalid')
        if row.get('source') not in REQUIRED_SOURCES:
            continue
        for field in ('calendar_id', 'headline', 'revision_id'):
            _text(row.get(field), 'stock_macro_event_calendar_invalid')
        _url(row.get('source_url'))
        seen = _stamp(row.get('first_seen_at_utc'))
        _stamp(row.get('scheduled_at_utc'))
        if seen > generated:
            raise ValueError('stock_macro_event_calendar_time_invalid')
        superseded = row.get('superseded_at_utc')
        if superseded is not None and not seen <= _stamp(superseded) <= generated:
            raise ValueError('stock_macro_event_calendar_time_invalid')
        revision = row['revision_id']
        if revision in revisions:
            raise ValueError('stock_macro_event_duplicate_revision')
        revisions[revision] = row
        calendar_rows.append(row)

    # No independent reimplementation of time windows or first-seen conversion.
    expected = {row['event_id']: row for row in public_context.make_events([], calendar_rows)}
    actual = {}
    for row in raw_events:
        if not isinstance(row, Mapping):
            raise ValueError('stock_macro_event_record_invalid')
        if row.get('category') != 'scheduled_macro':
            continue
        identity = _text(row.get('event_id'), 'stock_macro_event_record_invalid')
        if identity in actual:
            raise ValueError('stock_macro_event_duplicate_id')
        if type(row.get('block_entries')) is not bool:
            raise ValueError('stock_macro_event_block_flag_invalid')
        for key in ('published_at_utc', 'available_at_utc', 'starts_at_utc', 'ends_at_utc'):
            _stamp(row.get(key))
        if row.get('superseded_at_utc') is not None:
            _stamp(row['superseded_at_utc'])
        mirror = expected.get(identity)
        if mirror is None or any(row.get(key) != value for key, value in mirror.items()):
            raise ValueError('stock_macro_event_calendar_event_mismatch')
        if type(row.get('severity')) not in (float, int):
            raise ValueError('stock_macro_event_record_invalid')
        actual[identity] = row
    if actual.keys() != expected.keys():
        raise ValueError('stock_macro_event_calendar_event_missing')

    # A green fetch flag or empty event list cannot stand in for calendar coverage.
    # A calendar reviewed after a source's receipt is unavailable to that receipt.
    horizons = {}
    for name, state in sources.items():
        covered = []
        for row in calendar_rows:
            if row['source'] != name:
                continue
            seen = _stamp(row['first_seen_at_utc'])
            retired = (row.get('superseded_at_utc') is not None
                       and _stamp(row['superseded_at_utc']) <= reference_ms)
            if seen <= reference_ms and not retired and seen > state['last_success_at_ms']:
                raise ValueError('stock_macro_event_calendar_after_source_receipt:' + name)
            if seen > state['last_success_at_ms'] or retired:
                continue
            event = expected.get('calendar:' + row['revision_id'])
            if event is not None and _stamp(event['ends_at_utc']) >= reference_ms:
                covered.append(_stamp(event['ends_at_utc']))
        if not covered:
            raise ValueError('stock_macro_event_calendar_not_covered:' + name)
        horizons[name] = max(covered)

    events = event_risk.events_from_payload({'schema_version': 1, 'events': list(actual.values())})
    # Only disclose records known to the selected source receipt and causal clock.
    eligible = []
    details = {}
    for event in events:
        calendar = revisions[event.event_id.removeprefix('calendar:')]
        receipt = sources[calendar['source']]['last_success_at_ms']
        if event.available_at_ms <= min(reference_ms, receipt):
            eligible.append(event)
            details[event.event_id] = {
                'event_id': event.event_id, 'headline': event.headline,
                'starts_at_ms': event.starts_at_ms, 'ends_at_ms': event.ends_at_ms,
                'scheduled_at_ms': _stamp(calendar['scheduled_at_utc']),
                'source': calendar['source'], 'source_url': calendar['source_url'],
                'first_seen_at_ms': _stamp(calendar['first_seen_at_utc']),
                'available_at_ms': event.available_at_ms,
                'superseded_at_ms': event.superseded_at_ms,
                'time_basis': calendar.get('time_basis', 'source')}
    return tuple(eligible), details, horizons


def evaluate(config, symbol, reference_ms, *, root=None, reference_clock=None):
    """Return fail-closed admission data, without interrupting protective exits.

    Archived snapshots may contain later revisions/checks. Only checks observed
    by reference_ms are used; no future check rescues an unavailable old source.
    effective_from_ms keeps the opt-in experiment out of earlier baseline trades.
    """
    result = {'enabled': enabled(config), 'allowed': True, 'reasons': [],
              'status': 'disabled', 'event_ids': [], 'checked_at_ms': reference_ms,
              'scope': 'new_risk_only', 'protective_exits_allowed': True}
    if not result['enabled']:
        return result
    try:
        validate(config)
        reference_ms = _time(reference_ms, 'stock_macro_event_reference_time_invalid')
        policy = config[POLICY_KEY]
        result.update(effective_from_ms=policy['effective_from_ms'], snapshot_path=policy['snapshot_path'])
        if reference_ms < policy['effective_from_ms']:
            return {**result, 'status': 'not_active_yet'}
        path = Path(policy['snapshot_path'])
        if not path.is_absolute():
            path = (Path(root) if root is not None else Path(__file__).resolve().parents[1]) / path
        content = path.read_bytes()
        if len(content) > MAX_SNAPSHOT_BYTES:
            raise ValueError('stock_macro_event_snapshot_too_large')
        result['snapshot_sha256'] = hashlib.sha256(content).hexdigest()
        payload = json.loads(content.decode('utf-8'))
        if reference_clock is not None:
            refreshed = _time(reference_clock(), 'stock_macro_event_reference_time_invalid')
            if refreshed < reference_ms:
                raise ValueError('stock_macro_event_reference_clock_reversed')
            reference_ms = refreshed
            result['checked_at_ms'] = refreshed
        if (not isinstance(payload, Mapping) or type(payload.get('schema_version')) is not int
                or payload['schema_version'] != 1):
            raise ValueError('stock_macro_event_snapshot_invalid')
        generated, checked, sources = _coverage(payload, reference_ms)
        events, details, horizons = _events(payload, reference_ms, generated, sources)
        decision = event_risk.event_decision_at(events, reference_ms)
        active = [details[identity] for identity in decision.active_event_ids]
        upcoming = [details[event.event_id] for event in events
                    if event.starts_at_ms > reference_ms
                    and (event.superseded_at_ms is None or reference_ms < event.superseded_at_ms)]
        result.update(allowed=decision.allowed, status='clear' if decision.allowed else 'blackout',
                      reasons=[] if decision.allowed else ['stock_macro_event_blackout'],
                      event_ids=list(decision.active_event_ids), blocked_events=active,
                      known_event_count=len(events), source_status=sources,
                      coverage_at_ms=checked, coverage_age_ms=reference_ms - checked,
                      coverage_end_ms=min(horizons.values()), source_coverage_end_ms=horizons,
                      valid_until_ms=min(state['valid_until_ms'] for state in sources.values()),
                      source_max_age_ms=SOURCE_MAX_AGE_MS,
                      next_event=min(upcoming, key=lambda event: event['starts_at_ms']) if upcoming else None)
        return result
    except OSError:
        reason = 'stock_macro_event_snapshot_unavailable'
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
        reason = str(exc) if str(exc).startswith('stock_macro_event_') else 'stock_macro_event_snapshot_invalid'
    return {**result, 'allowed': False, 'status': 'unavailable', 'reasons': [reason]}
