"""Bounded, offline reference-session diagnostics; never quote freshness evidence.

``session_status(session, asof_ms)`` accepts one schema-version-1 configuration:
session_id, schedule_kind, timezone, basis, source_urls, source_published_date,
first_seen_at_ms, verified_at_ms, valid_from_ms, valid_until_ms,
reviewed_local_dates, holiday_exclusions. Dates are New York calendar dates.

The two supported *standard* schedules are deliberately narrow. An operator
must review the dates and exclude every local date affected by a holiday,
including its preceding evening when relevant. Exclusions mean unknown, not
closed. No generic holiday inference, network fetch, system clock, mutable
state, actual exchange-status assertion, or quote-age exemption lives here.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DAY_MS = 86_400_000
MAX_VALIDITY_MS = 14 * DAY_MS
MAX_REVIEWED_DATES = 16
MAX_TIMESTAMP_MS = 253_402_300_799_999
TIMEZONE = 'America/New_York'
SCHEDULES = {
    'cme_nymex_23x5': {
        'basis': 'scheduled_exchange_reference',
        'open_minute': 18 * 60, 'close_minute': 17 * 60,
        'limitation': 'Declared product standard schedule only; holidays and '
                      'unscheduled or product-specific halts are not live-verified.',
    },
    'oanda_fx_reference': {
        'basis': 'indicative_broker_reference',
        'open_minute': 17 * 60 + 5, 'close_minute': 16 * 60 + 59,
        'limitation': 'OANDA non-TRY/non-NZD reference only; OTC FX has no '
                      'single central exchange calendar. Other venues can differ.',
    },
}


def _timestamp(value):
    if type(value) is not int or not 0 <= value <= MAX_TIMESTAMP_MS:
        raise ValueError('session_config_invalid')
    return value


def _date(value):
    if type(value) is not str or len(value) != 10:
        raise ValueError('session_config_invalid')
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('session_config_invalid')
    return parsed


def _dates(values, *, allow_empty=False):
    if (type(values) is not list or len(values) > MAX_REVIEWED_DATES
            or (not values and not allow_empty)):
        raise ValueError('session_config_invalid')
    result = {_date(value) for value in values}
    if len(result) != len(values):
        raise ValueError('session_config_invalid')
    return result


def _source_urls(values):
    if type(values) is not list or not 1 <= len(values) <= 8:
        raise ValueError('session_config_invalid')
    for value in values:
        if type(value) is not str or len(value) > 2048:
            raise ValueError('session_config_invalid')
        parsed = urlsplit(value)
        if (parsed.scheme != 'https' or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.fragment or any(ord(c) < 33 for c in value)):
            raise ValueError('session_config_invalid')
    return list(values)


def _validate(session):
    if (not isinstance(session, Mapping)
            or type(session.get('schema_version')) is not int
            or session['schema_version'] != 1):
        raise ValueError('session_config_invalid')
    identity = session.get('session_id')
    if type(identity) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', identity):
        raise ValueError('session_config_invalid')
    kind = session.get('schedule_kind')
    if type(kind) is not str or kind not in SCHEDULES:
        raise ValueError('session_config_invalid')
    schedule = SCHEDULES[kind]
    if session.get('timezone') != TIMEZONE or session.get('basis') != schedule['basis']:
        raise ValueError('session_config_invalid')
    sources = _source_urls(session.get('source_urls'))
    published = session.get('source_published_date')
    if 'source_published_date' not in session:
        raise ValueError('session_config_invalid')
    if published is not None:
        _date(published)
    first_seen = _timestamp(session.get('first_seen_at_ms'))
    verified = _timestamp(session.get('verified_at_ms'))
    start = _timestamp(session.get('valid_from_ms'))
    end = _timestamp(session.get('valid_until_ms'))
    if first_seen > verified or not 0 < end - start <= MAX_VALIDITY_MS:
        raise ValueError('session_config_invalid')
    reviewed = _dates(session.get('reviewed_local_dates'))
    excluded = _dates(session.get('holiday_exclusions'), allow_empty=True)
    zone = ZoneInfo(TIMEZONE)
    return identity, kind, schedule, sources, published, first_seen, verified, start, end, reviewed, excluded, zone


def _phase(local, schedule):
    weekday = local.weekday()
    # Seconds/milliseconds retain their exact half-open boundary meaning.
    minute = (local.hour * 60 + local.minute
              + local.second / 60 + local.microsecond / 60_000_000)
    opening, closing = schedule['open_minute'], schedule['close_minute']
    if weekday == 5 or (weekday == 6 and minute < opening):
        return 'closed'
    if weekday == 4 and minute >= closing:
        return 'closed'
    if weekday < 4 and closing <= minute < opening:
        return 'maintenance'
    return 'open'


def _next_open(local, schedule, phase):
    if phase == 'open':
        return None
    local_date = local.date()
    if phase == 'closed' and local.weekday() in (4, 5):
        local_date += timedelta(days=6 - local.weekday())
    hour, minute = divmod(schedule['open_minute'], 60)
    # The supported opening times occur outside DST folds/gaps. Construct the
    # future local date first, so a weekend spanning US DST gets its new offset.
    return datetime.combine(local_date, time(hour, minute), tzinfo=local.tzinfo)


def session_status(session: Mapping, asof_ms: int) -> dict:
    """Describe a verified bounded reference schedule at an explicit instant.

    All uncertainty fails to ``unknown``. ``next_open_at_ms`` is supplied only
    for a closed/maintenance phase when every intervening local date is reviewed
    and the opening is strictly before valid_until_ms. No result makes a stale
    observation fresh or confirms that an exchange/OTC venue is actually open.
    """
    result = {
        'phase': 'unknown', 'next_open_at_ms': None,
        'next_open_reason_code': None, 'reason_code': 'session_config_invalid',
        'session_id': None, 'schedule_kind': None, 'basis': None,
        'source_urls': [], 'source_published_date': None,
        'timezone': TIMEZONE, 'asof_ms': None, 'local_at': None,
        'first_seen_at_ms': None, 'verified_at_ms': None,
        'valid_from_ms': None, 'valid_until_ms': None,
        'is_actual_market_status': False, 'freshness_override': False,
        'limitation': 'Reference schedule unavailable; observed data freshness remains required.',
    }
    try:
        _timestamp(asof_ms)
    except (ValueError, TypeError, OverflowError):
        result['reason_code'] = 'session_asof_invalid'
        return result
    result['asof_ms'] = asof_ms
    try:
        (identity, kind, schedule, sources, published, first_seen, verified,
         start, end, reviewed, excluded, zone) = _validate(session)
        local = datetime.fromtimestamp(asof_ms / 1000, timezone.utc).astimezone(zone)
    except ZoneInfoNotFoundError:
        result['reason_code'] = 'session_timezone_unavailable'
        return result
    except (ValueError, TypeError, OverflowError, OSError):
        return result
    result.update(
        session_id=identity, schedule_kind=kind, basis=schedule['basis'],
        source_urls=sources, source_published_date=published,
        local_at=local.isoformat(), first_seen_at_ms=first_seen,
        verified_at_ms=verified, valid_from_ms=start, valid_until_ms=end,
        limitation=schedule['limitation'],
    )
    if asof_ms < first_seen or asof_ms < verified:
        result['reason_code'] = 'session_not_yet_verified'
    elif not start <= asof_ms < end:
        result['reason_code'] = 'session_outside_validity'
    elif local.date() in excluded:
        result['reason_code'] = 'session_holiday_excluded'
    elif local.date() not in reviewed:
        result['reason_code'] = 'session_local_date_unreviewed'
    else:
        phase = _phase(local, schedule)
        result.update(phase=phase, reason_code='session_reference_schedule')
        try:
            opening = _next_open(local, schedule, phase)
        except (ValueError, OverflowError):
            result['next_open_reason_code'] = 'session_next_open_outside_validity'
            return result
        if opening is None:
            return result
        opening_ms = int(opening.timestamp() * 1000)
        if not start <= opening_ms < end:
            result['next_open_reason_code'] = 'session_next_open_outside_validity'
            return result
        day = local.date()
        while day <= opening.date():
            if day in excluded:
                result['next_open_reason_code'] = 'session_next_open_holiday_excluded'
                return result
            if day not in reviewed:
                result['next_open_reason_code'] = 'session_next_open_date_unreviewed'
                return result
            day += timedelta(days=1)
        result['next_open_at_ms'] = opening_ms
        result['next_open_reason_code'] = 'session_reference_schedule'
    return result
