"""Reference-only, bounded session phases, provenance, holidays and DST."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import market_sessions as sessions


def ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def config(kind='cme_nymex_23x5'):
    return {
        'schema_version': 1, 'session_id': 'wti' if kind == 'cme_nymex_23x5' else 'usd_jpy',
        'schedule_kind': kind, 'timezone': 'America/New_York',
        'basis': 'scheduled_exchange_reference' if kind == 'cme_nymex_23x5' else 'indicative_broker_reference',
        'source_urls': ['https://www.cmegroup.com/trading-hours.html'] if kind == 'cme_nymex_23x5' else ['https://www.oanda.com/us-en/trading/hours-of-operation/'],
        'source_published_date': None,
        'first_seen_at_ms': ms('2026-10-24T00:00:00+00:00'),
        'verified_at_ms': ms('2026-10-24T00:00:00+00:00'),
        'valid_from_ms': ms('2026-10-24T00:00:00-04:00'),
        'valid_until_ms': ms('2026-11-02T00:00:00-05:00'),
        'reviewed_local_dates': [f'2026-10-{day}' for day in range(24, 32)] + ['2026-11-01'],
        'holiday_exclusions': [],
    }


@pytest.mark.parametrize('at,phase,next_open', [
    ('2026-10-25T21:59:59.999+00:00', 'closed', '2026-10-25T22:00:00+00:00'),
    ('2026-10-25T22:00:00+00:00', 'open', None),
    ('2026-10-26T20:59:59.999+00:00', 'open', None),
    ('2026-10-26T21:00:00+00:00', 'maintenance', '2026-10-26T22:00:00+00:00'),
    ('2026-10-26T21:59:59.999+00:00', 'maintenance', '2026-10-26T22:00:00+00:00'),
    ('2026-10-26T22:00:00+00:00', 'open', None),
    ('2026-10-30T20:59:59.999+00:00', 'open', None),
    ('2026-10-30T21:00:00+00:00', 'closed', '2026-11-01T23:00:00+00:00'),
    ('2026-10-31T12:00:00+00:00', 'closed', '2026-11-01T23:00:00+00:00'),
    ('2026-11-01T22:59:59.999+00:00', 'closed', '2026-11-01T23:00:00+00:00'),
    ('2026-11-01T23:00:00+00:00', 'open', None),
])
def test_cme_half_open_phases_and_new_york_dst(at, phase, next_open):
    result = sessions.session_status(config(), ms(at))
    assert result['phase'] == phase
    assert result['next_open_at_ms'] == (ms(next_open) if next_open else None)
    assert result['basis'] == 'scheduled_exchange_reference'
    assert not result['is_actual_market_status'] and not result['freshness_override']


@pytest.mark.parametrize('at,phase,next_open', [
    ('2026-10-25T21:04:59.999+00:00', 'closed', '2026-10-25T21:05:00+00:00'),
    ('2026-10-25T21:05:00+00:00', 'open', None),
    ('2026-10-26T20:58:59.999+00:00', 'open', None),
    ('2026-10-26T20:59:00+00:00', 'maintenance', '2026-10-26T21:05:00+00:00'),
    ('2026-10-26T21:04:59.999+00:00', 'maintenance', '2026-10-26T21:05:00+00:00'),
    ('2026-10-26T21:05:00+00:00', 'open', None),
    ('2026-10-30T20:59:00+00:00', 'closed', '2026-11-01T22:05:00+00:00'),
    ('2026-11-01T22:04:59.999+00:00', 'closed', '2026-11-01T22:05:00+00:00'),
    ('2026-11-01T22:05:00+00:00', 'open', None),
])
def test_oanda_is_indicative_and_keeps_six_minute_daily_break(at, phase, next_open):
    result = sessions.session_status(config('oanda_fx_reference'), ms(at))
    assert result['phase'] == phase
    assert result['next_open_at_ms'] == (ms(next_open) if next_open else None)
    assert result['basis'] == 'indicative_broker_reference'
    assert 'non-TRY/non-NZD' in result['limitation'] and 'OTC' in result['limitation']


def test_european_dst_does_not_shift_new_york_open_a_week_early():
    london = ZoneInfo('Europe/London')
    before = datetime.fromisoformat('2026-10-24T22:00:00+00:00').astimezone(london)
    after = datetime.fromisoformat('2026-10-25T22:00:00+00:00').astimezone(london)
    assert before.utcoffset().total_seconds() == 3600
    assert after.utcoffset().total_seconds() == 0
    result = sessions.session_status(config(), int(after.timestamp() * 1000))
    assert result['phase'] == 'open' and result['local_at'].endswith('-04:00')


@pytest.mark.parametrize('at', ['2026-11-01T05:30:00+00:00', '2026-11-01T06:30:00+00:00'])
def test_both_new_york_fallback_folds_point_to_same_sunday_open(at):
    result = sessions.session_status(config(), ms(at))
    assert result['phase'] == 'closed'
    assert result['next_open_at_ms'] == ms('2026-11-01T23:00:00+00:00')


def test_expiry_is_exclusive_and_never_claims_weekend_closed_after_expiry():
    raw = config()
    raw['valid_until_ms'] = ms('2026-10-31T00:00:00+00:00')
    result = sessions.session_status(raw, raw['valid_until_ms'])
    assert result['phase'] == 'unknown' and result['next_open_at_ms'] is None
    assert result['reason_code'] == 'session_outside_validity'


def test_unreviewed_weekend_date_is_unknown_not_inferred_closed():
    raw = config()
    raw['reviewed_local_dates'].remove('2026-10-31')
    result = sessions.session_status(raw, ms('2026-10-31T12:00:00+00:00'))
    assert result['phase'] == 'unknown'
    assert result['reason_code'] == 'session_local_date_unreviewed'


def test_known_holiday_exclusion_is_unknown_even_when_standard_schedule_would_open():
    raw = config()
    raw['holiday_exclusions'] = ['2026-10-26']
    result = sessions.session_status(raw, ms('2026-10-26T12:00:00+00:00'))
    assert result['phase'] == 'unknown' and result['next_open_at_ms'] is None
    assert result['reason_code'] == 'session_holiday_excluded'


@pytest.mark.parametrize('change,reason', [
    ('end', 'session_next_open_outside_validity'),
    ('equal_end', 'session_next_open_outside_validity'),
    ('middle_missing', 'session_next_open_date_unreviewed'),
    ('opening_missing', 'session_next_open_date_unreviewed'),
    ('middle_holiday', 'session_next_open_holiday_excluded'),
    ('opening_holiday', 'session_next_open_holiday_excluded'),
])
def test_confirmed_friday_closed_does_not_invent_unverified_next_open(change, reason):
    raw = config()
    if change in {'end', 'equal_end'}:
        raw['valid_until_ms'] = ms('2026-11-01T23:00:00+00:00') - (1 if change == 'end' else 0)
    elif change.endswith('missing'):
        raw['reviewed_local_dates'].remove('2026-10-31' if change.startswith('middle') else '2026-11-01')
    else:
        raw['holiday_exclusions'] = ['2026-10-31' if change.startswith('middle') else '2026-11-01']
    result = sessions.session_status(raw, ms('2026-10-30T22:00:00+00:00'))
    assert result['phase'] == 'closed' and result['next_open_at_ms'] is None
    assert result['next_open_reason_code'] == reason


def test_future_verification_cannot_backfill_earlier_session():
    raw = config()
    at = ms('2026-10-25T12:00:00+00:00')
    raw['verified_at_ms'] = at + 1
    result = sessions.session_status(raw, at)
    assert result['phase'] == 'unknown'
    assert result['reason_code'] == 'session_not_yet_verified'


@pytest.mark.parametrize('at', [None, True, 1.5, -1, float('nan'), float('inf'), '0', 253402300800000])
def test_bad_asof_is_explicit_unknown(at):
    result = sessions.session_status(config(), at)
    assert result['phase'] == 'unknown' and result['reason_code'] == 'session_asof_invalid'


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('schema_version', 2), ('session_id', 'invalid id'),
    ('schedule_kind', 'all_futures'), ('schedule_kind', {}), ('timezone', 'Europe/London'),
    ('basis', 'actual_market_status'), ('source_urls', []),
    ('source_urls', ['http://example.com']), ('source_urls', ['https://user:secret@example.com']),
    ('source_urls', ['https://example.com/path#secret']), ('source_urls', ['https://example.com/\n']),
    ('source_published_date', '2026-1-01'), ('verified_at_ms', True),
    ('verified_at_ms', float('nan')), ('first_seen_at_ms', 253402300800000),
    ('reviewed_local_dates', []), ('reviewed_local_dates', ['2026-10-25', '2026-10-25']),
    ('reviewed_local_dates', ['2026-02-30']), ('holiday_exclusions', None),
])
def test_invalid_schema_fails_unknown(field, value):
    raw = config()
    raw[field] = value
    result = sessions.session_status(raw, ms('2026-10-25T22:00:00+00:00'))
    assert result['phase'] == 'unknown' and result['next_open_at_ms'] is None
    assert result['reason_code'] == 'session_config_invalid'


@pytest.mark.parametrize('raw', [None, {}, [], 'calendar'])
def test_missing_calendar_is_unknown(raw):
    assert sessions.session_status(raw, ms('2026-10-25T12:00:00+00:00'))['phase'] == 'unknown'


@pytest.mark.parametrize('change', ['long', 'reverse', 'clock_order', 'too_many_dates'])
def test_accidental_unbounded_or_inconsistent_configuration_is_rejected(change):
    raw = config()
    if change == 'long':
        raw['valid_until_ms'] = raw['valid_from_ms'] + sessions.MAX_VALIDITY_MS + 1
    elif change == 'reverse':
        raw['valid_until_ms'] = raw['valid_from_ms']
    elif change == 'clock_order':
        raw['first_seen_at_ms'] = raw['verified_at_ms'] + 1
    else:
        raw['reviewed_local_dates'] = [f'2026-10-{i:02}' for i in range(1, 18)]
    assert sessions.session_status(raw, ms('2026-10-25T12:00:00+00:00'))['phase'] == 'unknown'


def test_timezone_database_missing_fails_unknown(monkeypatch):
    def missing(_):
        raise ZoneInfoNotFoundError('missing')
    monkeypatch.setattr(sessions, 'ZoneInfo', missing)
    result = sessions.session_status(config(), ms('2026-10-25T22:00:00+00:00'))
    assert result['phase'] == 'unknown' and result['reason_code'] == 'session_timezone_unavailable'


def test_pure_repeatable_result_does_not_mutate_config_or_observation_clock():
    raw = config()
    snapshot = deepcopy(raw)
    at = ms('2026-10-25T21:00:00+00:00')
    result = sessions.session_status(raw, at)
    assert raw == snapshot
    assert sessions.session_status(raw, at) == result
    assert result['verified_at_ms'] == raw['verified_at_ms']
    assert result['asof_ms'] == at and not result['freshness_override']


def test_opening_beyond_representable_dates_is_not_invented():
    raw = config()
    at = ms('9999-12-31T22:00:00+00:00')
    raw.update(first_seen_at_ms=at - sessions.DAY_MS,
               verified_at_ms=at - sessions.DAY_MS,
               valid_from_ms=at - sessions.DAY_MS,
               valid_until_ms=sessions.MAX_TIMESTAMP_MS,
               reviewed_local_dates=['9999-12-31'])
    result = sessions.session_status(raw, at)
    # Some Windows C runtimes cannot convert this extreme timestamp at all.
    # Both that uncertainty and an unrepresentable future opening remain safe.
    assert result['phase'] in {'closed', 'unknown'}
    assert result['next_open_at_ms'] is None
