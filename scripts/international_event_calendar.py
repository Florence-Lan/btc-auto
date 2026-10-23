"""Offline admission and provenance diagnostics for an opt-in calendar candidate.

Nothing here fetches sources, renews verification clocks, changes active profiles,
or places orders. The pinned calendar is usable only after its real first-seen
time. Fed/BEA and company earnings gates remain separate; this is not a complete
world-event feed. Re-read the file and re-evaluate immediately before new risk.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import json
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
MAX_CALENDAR_BYTES = 1024 * 1024
DEFAULT_POLICY = {
    'pre_event_minutes': 30,
    'post_event_minutes': 90,
    'max_verification_age_hours': 24,
}
PROTECTIVE_ACTIONS = frozenset({'reduce', 'close', 'protective_exit'})


def _time(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('international_calendar_time_invalid:' + label)
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('international_calendar_text_invalid:' + label)
    return value


def _urls(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError('international_calendar_sources_missing:' + label)
    for value in values:
        parsed = urlsplit(_text(value, label))
        if (parsed.scheme != 'https' or not parsed.hostname
                or parsed.username or parsed.password):
            raise ValueError('international_calendar_source_invalid:' + label)


def _published(value, label):
    # None means the official page did not display a publication date. It is
    # never replaced by our fetch date or interpreted as the actual release.
    if value is not None:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError('international_calendar_publication_date_invalid:' + label)


def _local_time(value, zone, expected_ms, label):
    local = datetime.fromisoformat(_text(value, label))
    if local.tzinfo is None or local.utcoffset() is None:
        raise ValueError('international_calendar_local_time_invalid:' + label)
    try:
        zoned = local.astimezone(ZoneInfo(zone))
    except ZoneInfoNotFoundError as exc:
        raise ValueError('international_calendar_timezone_unavailable:' + zone) from exc
    if (int(local.timestamp() * 1000) != expected_ms
            or zoned.utcoffset() != local.utcoffset()
            or zoned.replace(tzinfo=None) != local.replace(tzinfo=None)):
        raise ValueError('international_calendar_local_time_invalid:' + label)
    return local


def _clock(row, label, asof_ms):
    first = _time(row.get('first_seen_at_ms'), label + '.first_seen')
    available = _time(row.get('available_at_ms'), label + '.available')
    verified = _time(row.get('verified_at_ms'), label + '.verified')
    if not first <= available <= verified:
        raise ValueError('international_calendar_clock_order_invalid:' + label)
    if verified > asof_ms:
        raise ValueError('international_calendar_not_yet_observed:' + label)
    return first, available, verified


def _policy(policy):
    values = dict(DEFAULT_POLICY)
    if policy is not None:
        if not isinstance(policy, Mapping) or set(policy) - set(values):
            raise ValueError('international_calendar_policy_invalid')
        values.update(policy)
    for name, value in values.items():
        maximum = 168 if name == 'max_verification_age_hours' else 24 * 60
        minimum = 1 if name == 'max_verification_age_hours' else 0
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError('international_calendar_policy_invalid')
    return values


def load_calendar(path):
    """Read pinned bytes only; failure cannot create a fresh or empty calendar."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_CALENDAR_BYTES:
            raise ValueError('international_calendar_too_large')
        payload = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(payload, Mapping):
            raise ValueError('international_calendar_payload_invalid')
        return payload
    except OSError as exc:
        raise ValueError('international_calendar_unavailable') from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('international_calendar_payload_invalid') from exc


def _source_diagnostics(calendar, asof_ms, policy):
    sources = calendar.get('source_verifications')
    required = calendar.get('required_sources')
    if (not isinstance(sources, Mapping) or not isinstance(required, list)
            or not required or len(required) != len(set(required))):
        raise ValueError('international_calendar_sources_missing')
    result, failures = {}, []
    max_age = policy['max_verification_age_hours'] * HOUR_MS
    for name in required:
        _text(name, 'required_source')
        row = sources.get(name)
        if not isinstance(row, Mapping):
            raise ValueError('international_calendar_source_missing:' + name)
        _urls(row.get('source_urls'), name)
        _published(row.get('source_published_date'), name)
        _, _, verified = _clock(row, name, asof_ms)
        success = _time(row.get('last_success_at_ms'), name + '.last_success')
        checked = _time(row.get('checked_at_ms'), name + '.checked')
        if success > verified or not verified <= checked <= asof_ms:
            raise ValueError('international_calendar_source_clock_invalid:' + name)
        if type(row.get('ok')) is not bool or type(row.get('fetch_ok')) is not bool:
            raise ValueError('international_calendar_source_status_invalid:' + name)
        valid_until = _time(row.get('valid_until_ms'), name + '.valid_until')
        horizon = _time(row.get('coverage_end_ms'), name + '.coverage_end')
        if valid_until < verified or horizon < verified:
            raise ValueError('international_calendar_source_horizon_invalid:' + name)
        expires = min(valid_until, verified + max_age, success + max_age)
        result[name] = {
            'ok': row['ok'], 'fetch_ok': row['fetch_ok'],
            'fetch_status': 'live' if row['fetch_ok'] else 'cached',
            'checked_at_ms': checked, 'last_success_at_ms': success,
            'verified_at_ms': verified, 'verification_age_ms': asof_ms - verified,
            'content_last_updated_date': row.get('content_last_updated_date'),
            'source_published_date': row.get('source_published_date'),
            'valid_until_ms': expires, 'coverage_end_ms': horizon,
            'source_urls': list(row['source_urls']),
            'note': 'Fetch success and a scheduled event are not an actual release observation.',
        }
        if not row['ok']:
            failures.append('international_calendar_source_unavailable:' + name)
        if asof_ms > expires:
            failures.append('international_calendar_source_stale:' + name)
        if asof_ms >= horizon:
            failures.append('international_calendar_source_coverage_expired:' + name)
    return result, failures


def _event(row, asof_ms, policy, source_status):
    if not isinstance(row, Mapping):
        raise ValueError('international_calendar_event_invalid')
    identity = _text(row.get('event_id'), 'event_id')
    _text(row.get('headline'), identity + '.headline')
    _text(row.get('timezone'), identity + '.timezone')
    _text(row.get('uncertainty'), identity + '.uncertainty')
    _urls(row.get('source_urls'), identity)
    _published(row.get('source_published_date'), identity)
    _, available, verified = _clock(row, identity, asof_ms)
    sources = row.get('source_ids')
    if not isinstance(sources, list) or not sources or any(name not in source_status for name in sources):
        raise ValueError('international_calendar_event_source_missing:' + identity)
    if any(verified > source_status[name]['verified_at_ms'] for name in sources):
        raise ValueError('international_calendar_event_after_source_review:' + identity)
    if type(row.get('block_new_risk')) is not bool:
        raise ValueError('international_calendar_event_block_flag_invalid:' + identity)
    actual = row.get('actual_at_ms')
    if actual is not None:
        _time(actual, identity + '.actual')
        observed = _time(row.get('actual_observed_at_ms'), identity + '.actual_observed')
        if actual > observed or observed > verified or observed < available:
            raise ValueError('international_calendar_actual_not_observed:' + identity)
    elif row.get('actual_observed_at_ms') is not None:
        raise ValueError('international_calendar_actual_invalid:' + identity)
    scheduled = row.get('scheduled_at_ms')
    basis = row.get('time_basis')
    if basis == 'official_exact_time':
        start = end = _time(scheduled, identity + '.scheduled')
        # Validate the explicitly stored local offset against the UTC anchor.
        _local_time(row.get('scheduled_at_local'), row['timezone'], start, identity)
    elif basis == 'conservative_full_decision_day':
        if scheduled is not None:
            raise ValueError('international_calendar_unknown_time_fabricated:' + identity)
        start = _time(row.get('bounded_start_ms'), identity + '.bounded_start')
        end = _time(row.get('bounded_end_ms'), identity + '.bounded_end')
        begin_local = _local_time(row.get('bounded_start_local'), row['timezone'], start, identity)
        end_local = _local_time(row.get('bounded_end_local'), row['timezone'], end, identity)
        if (end - start != 24 * HOUR_MS or begin_local.hour or begin_local.minute
                or begin_local.second or end_local.hour or end_local.minute or end_local.second):
            raise ValueError('international_calendar_event_bounds_invalid:' + identity)
        _text(row.get('window_assumption'), identity + '.window_assumption')
    else:
        raise ValueError('international_calendar_time_basis_invalid:' + identity)
    blackout_start = start - policy['pre_event_minutes'] * MINUTE_MS
    blackout_end = end + policy['post_event_minutes'] * MINUTE_MS
    phase = ('upcoming' if asof_ms < blackout_start else
             'blackout' if asof_ms <= blackout_end else 'window_elapsed')
    return {
        'event_id': identity, 'headline': row['headline'], 'scheduled_at_ms': scheduled,
        'scheduled_at_local': row.get('scheduled_at_local'), 'timezone': row['timezone'],
        'time_basis': basis, 'actual_at_ms': actual,
        'actual_observed_at_ms': row.get('actual_observed_at_ms'),
        'release_status': 'actual_observed' if actual is not None else 'actual_not_observed',
        'phase': phase, 'starts_at_ms': blackout_start, 'ends_at_ms': blackout_end,
        'bounded_start_ms': start if scheduled is None else None,
        'bounded_end_ms': end if scheduled is None else None,
        'window_assumption': row.get('window_assumption'), 'uncertainty': row['uncertainty'],
        'first_seen_at_ms': row['first_seen_at_ms'], 'available_at_ms': available,
        'verified_at_ms': verified, 'source_urls': list(row['source_urls']),
        'source_published_date': row.get('source_published_date'),
        'blocks_new_risk': row['block_new_risk'] and phase == 'blackout',
    }


def evaluate_international_event_calendar(calendar, *, asof_ms, action='new_risk', policy=None):
    """Evaluate only this independently opted-in calendar; never block exits.

    Clock/coverage failures fail closed for new risk. A later fetch timestamp
    cannot rescue an old verification or make this calendar available in the
    past. Window ends are inclusive, matching the existing macro gate.
    """
    result = {
        'enabled': True, 'allowed': False, 'status': 'unavailable', 'reasons': [],
        'checked_at_ms': asof_ms, 'scope': 'new_risk_only',
        'protective_exits_allowed': True, 'event_ids': [], 'blocked_events': [],
        'next_event': None, 'source_status': {}, 'diagnostics': [],
    }
    if action in PROTECTIVE_ACTIONS:
        return {**result, 'allowed': True, 'status': 'protective_action',
                'diagnostics': ['Calendar admission does not restrict reductions or protective exits.']}
    try:
        asof_ms = _time(asof_ms, 'asof')
        if action != 'new_risk':
            raise ValueError('international_calendar_action_invalid')
        settings = _policy(policy)
        if (not isinstance(calendar, Mapping) or type(calendar.get('schema_version')) is not int
                or calendar['schema_version'] != 1):
            raise ValueError('international_calendar_payload_invalid')
        result['calendar_id'] = _text(calendar.get('calendar_id'), 'calendar_id')
        _, _, verified = _clock(calendar, 'calendar', asof_ms)
        start = _time(calendar.get('coverage_start_ms'), 'coverage_start')
        end = _time(calendar.get('coverage_end_ms'), 'coverage_end')
        valid_until = _time(calendar.get('valid_until_ms'), 'valid_until')
        if not start <= verified < end or valid_until < verified:
            raise ValueError('international_calendar_coverage_invalid')
        result['coverage'] = {'start_ms': start, 'end_ms': end,
                              'scope': calendar.get('coverage_scope'), 'verified_at_ms': verified}
        if asof_ms < start or asof_ms >= end:
            raise ValueError('international_calendar_outside_coverage')
        if asof_ms > min(valid_until, verified + settings['max_verification_age_hours'] * HOUR_MS):
            raise ValueError('international_calendar_stale')
        sources, source_failures = _source_diagnostics(calendar, asof_ms, settings)
        result['source_status'] = sources
        result['valid_until_ms'] = min(valid_until, *(s['valid_until_ms'] for s in sources.values()))
        if source_failures:
            return {**result, 'reasons': source_failures}
        rows = calendar.get('events')
        if not isinstance(rows, list) or not rows or len(rows) > 256:
            raise ValueError('international_calendar_events_missing')
        events = [_event(row, asof_ms, settings, sources) for row in rows]
        if len({row['event_id'] for row in events}) != len(events):
            raise ValueError('international_calendar_duplicate_event')
        for raw, event in zip(rows, events):
            horizon = min(end, *(sources[name]['coverage_end_ms'] for name in raw['source_ids']))
            if event['ends_at_ms'] >= horizon:
                raise ValueError('international_calendar_event_outside_coverage:' + event['event_id'])
        active = [row for row in events if row['blocks_new_risk']]
        upcoming = [row for row in events if row['phase'] == 'upcoming']
        result.update(
            allowed=not active, status='blackout' if active else 'clear',
            reasons=['international_calendar_blackout'] if active else [],
            event_ids=[row['event_id'] for row in active], blocked_events=active,
            next_event=min(upcoming, key=lambda row: row['starts_at_ms']) if upcoming else None,
            known_event_count=len(events), event_diagnostics=events, policy=settings,
            diagnostics=[
                'Scheduled times are not actual release observations; elapsed windows do not prove publication.',
                'This pinned candidate covers named JP/ECB events only; retain existing Fed/BEA and earnings gates.',
                'Source checked/fetch success clocks do not replace source verification or observation dates.',
            ],
        )
        return result
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        reason = str(exc) if str(exc).startswith('international_calendar_') else 'international_calendar_payload_invalid'
        return {**result, 'allowed': False, 'status': 'unavailable', 'reasons': [reason]}
