"""Independent calendar provenance, uncertainty, DST and admission boundaries."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import international_event_calendar as calendar


MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR
ROOT = Path(__file__).resolve().parents[1]


def ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


NOW = ms('2026-10-22T16:00:00+00:00')
URL = 'https://www.e-stat.go.jp/release-calendar/statistics/00200573'


def iso(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def payload(*, asof=NOW, anchor=NOW + 2 * HOUR):
    clocks = {'first_seen_at_ms': NOW - HOUR, 'available_at_ms': NOW - HOUR,
              'verified_at_ms': asof}
    source = {**clocks, 'source_urls': [URL], 'source_published_date': None,
              'content_last_updated_date': '2026-01-23', 'ok': True, 'fetch_ok': True,
              'last_success_at_ms': asof, 'checked_at_ms': asof,
              'valid_until_ms': asof + 2 * DAY, 'coverage_end_ms': NOW + 14 * DAY}
    event = {**clocks, 'event_id': 'jp-cpi', 'headline': 'Scheduled CPI',
             'source_ids': ['official'], 'source_urls': [URL], 'source_published_date': None,
             'scheduled_at_ms': anchor, 'scheduled_at_local': iso(anchor),
             'timezone': 'UTC', 'time_basis': 'official_exact_time',
             'actual_at_ms': None, 'actual_observed_at_ms': None, 'block_new_risk': True,
             'uncertainty': 'Schedule only; actual release has not been observed.'}
    return {'schema_version': 1, 'calendar_id': 'independent-test', **clocks,
            'coverage_start_ms': NOW - HOUR, 'coverage_end_ms': NOW + 14 * DAY,
            'valid_until_ms': asof + 2 * DAY, 'coverage_scope': 'named events only',
            'required_sources': ['official'], 'source_verifications': {'official': source},
            'events': [event]}


def evaluate(raw=None, at=NOW, **kwargs):
    return calendar.evaluate_international_event_calendar(
        payload() if raw is None else raw, asof_ms=at, **kwargs)


@pytest.mark.parametrize('offset,allowed', [(-30 * MINUTE - 1, True),
    (-30 * MINUTE, False), (0, False), (90 * MINUTE, False), (90 * MINUTE + 1, True)])
def test_default_window_has_exact_inclusive_boundaries(offset, allowed):
    anchor = NOW + 2 * HOUR
    at = anchor + offset
    result = evaluate(payload(asof=at, anchor=anchor), at)
    assert result['allowed'] is allowed
    assert result['scope'] == 'new_risk_only' and result['protective_exits_allowed']
    if not allowed:
        event = result['blocked_events'][0]
        assert event['scheduled_at_ms'] == anchor
        assert event['starts_at_ms'] == anchor - 30 * MINUTE
        assert event['ends_at_ms'] == anchor + 90 * MINUTE
        assert result['reasons'] == ['international_calendar_blackout']


@pytest.mark.parametrize('action', ['reduce', 'close', 'protective_exit'])
def test_missing_or_invalid_calendar_never_blocks_protective_actions(action):
    result = calendar.evaluate_international_event_calendar(None, asof_ms=-1, action=action)
    assert result['allowed'] and result['status'] == 'protective_action'
    assert result['event_ids'] == [] and result['protective_exits_allowed']


@pytest.mark.parametrize('raw', [None, {}, [], {'schema_version': True},
                               {'schema_version': 2}, {'schema_version': 1}])
def test_missing_malformed_calendar_fails_closed_for_new_risk(raw):
    result = calendar.evaluate_international_event_calendar(raw, asof_ms=NOW)
    assert not result['allowed'] and result['status'] == 'unavailable'
    assert result['protective_exits_allowed']


def test_future_review_cannot_be_applied_to_an_earlier_decision():
    result = evaluate(payload(asof=NOW + 1))
    assert result['reasons'] == ['international_calendar_not_yet_observed:calendar']
    assert not result['allowed'] and result['event_ids'] == []


def test_green_fetch_or_new_checked_clock_cannot_renew_an_old_source_review():
    raw = payload()
    source = raw['source_verifications']['official']
    source.update(first_seen_at_ms=NOW - 26 * HOUR, available_at_ms=NOW - 26 * HOUR,
                  verified_at_ms=NOW - 25 * HOUR, last_success_at_ms=NOW - 25 * HOUR,
                  checked_at_ms=NOW, fetch_ok=True)
    result = evaluate(raw)
    assert result['reasons'] == ['international_calendar_source_stale:official']
    assert not result['allowed']
    diagnostics = result['source_status']['official']
    assert diagnostics['fetch_ok'] and diagnostics['verification_age_ms'] == 25 * HOUR
    assert diagnostics['content_last_updated_date'] == '2026-01-23'


def test_cached_fresh_verified_source_is_distinct_from_failed_fetch():
    raw = payload()
    raw['source_verifications']['official']['fetch_ok'] = False
    result = evaluate(raw)
    assert result['allowed']
    assert result['source_status']['official']['fetch_status'] == 'cached'
    raw['source_verifications']['official']['ok'] = False
    failed = evaluate(raw)
    assert failed['reasons'] == ['international_calendar_source_unavailable:official']


def test_source_receipt_age_is_not_hidden_by_new_verification():
    raw = payload()
    raw['source_verifications']['official']['last_success_at_ms'] = NOW - 25 * HOUR
    result = evaluate(raw)
    assert not result['allowed']
    assert result['reasons'] == ['international_calendar_source_stale:official']


def test_expiry_and_coverage_boundaries_are_separate():
    raw = payload(anchor=NOW + 4 * DAY)
    assert evaluate(raw, NOW + DAY)['allowed']
    assert evaluate(raw, NOW + DAY + 1)['reasons'] == ['international_calendar_stale']
    raw['coverage_end_ms'] = NOW
    raw['verified_at_ms'] = NOW - 1
    result = evaluate(raw)
    assert result['reasons'] == ['international_calendar_outside_coverage']


def test_event_window_cannot_claim_coverage_beyond_its_reviewed_source():
    raw = payload()
    raw['source_verifications']['official']['coverage_end_ms'] = NOW + 2 * HOUR
    result = evaluate(raw)
    assert not result['allowed']
    assert result['reasons'] == ['international_calendar_event_outside_coverage:jp-cpi']


@pytest.mark.parametrize('mutation', ['empty_events', 'duplicate', 'missing_source',
    'bad_source_flag', 'bool_as_time', 'naive_local', 'wrong_offset', 'future_event_review',
    'future_actual', 'actual_without_observation', 'unknown_action'])
def test_invalid_provenance_never_becomes_clear(mutation):
    raw = payload()
    event = raw['events'][0]
    kwargs = {}
    if mutation == 'empty_events': raw['events'] = []
    if mutation == 'duplicate': raw['events'].append(deepcopy(event))
    if mutation == 'missing_source': raw['source_verifications'] = {}
    if mutation == 'bad_source_flag': raw['source_verifications']['official']['ok'] = 'true'
    if mutation == 'bool_as_time': event['scheduled_at_ms'] = True
    if mutation == 'naive_local': event['scheduled_at_local'] = '2026-10-22T18:00:00'
    if mutation == 'wrong_offset': event.update(timezone='Asia/Tokyo')
    if mutation == 'future_event_review': event['verified_at_ms'] = NOW + 1
    if mutation == 'future_actual': event.update(actual_at_ms=NOW + 1, actual_observed_at_ms=NOW + 1)
    if mutation == 'actual_without_observation': event['actual_at_ms'] = NOW
    if mutation == 'unknown_action': kwargs['action'] = 'flip_and_increase'
    result = evaluate(raw, **kwargs)
    assert not result['allowed'] and result['status'] == 'unavailable'


def test_elapsed_schedule_does_not_claim_actual_publication():
    at = NOW + 4 * HOUR
    result = evaluate(payload(asof=at), at)
    assert result['allowed']
    event = result['event_diagnostics'][0]
    assert event['phase'] == 'window_elapsed'
    assert event['actual_at_ms'] is None and event['release_status'] == 'actual_not_observed'


def test_actual_release_must_have_a_causal_observation_and_review():
    at = NOW + 3 * HOUR
    raw = payload(asof=at)
    raw['events'][0].update(actual_at_ms=NOW + 2 * HOUR,
                            actual_observed_at_ms=NOW + 2 * HOUR + 1)
    result = evaluate(raw, at)
    assert result['event_diagnostics'][0]['release_status'] == 'actual_observed'
    raw['events'][0]['actual_observed_at_ms'] = at + 1
    assert not evaluate(raw, at)['allowed']


def test_boj_unknown_time_blocks_entire_decision_day_without_fake_anchor():
    begin = ms('2026-10-30T00:00:00+09:00')
    end = ms('2026-10-31T00:00:00+09:00')
    for at in [begin - 30 * MINUTE, begin, begin + 12 * HOUR, end + 90 * MINUTE]:
        raw = payload(asof=at)
        raw['events'][0].update(
            scheduled_at_ms=None, scheduled_at_local=None, timezone='Asia/Tokyo',
            time_basis='conservative_full_decision_day', bounded_start_ms=begin,
            bounded_end_ms=end, bounded_start_local='2026-10-30T00:00:00+09:00',
            bounded_end_local='2026-10-31T00:00:00+09:00',
            window_assumption='Conservative candidate full-day bound; not an official release time.')
        result = evaluate(raw, at)
        assert not result['allowed'] and result['status'] == 'blackout'
        assert result['blocked_events'][0]['scheduled_at_ms'] is None
        assert result['blocked_events'][0]['bounded_end_ms'] == end
    raw['events'][0]['scheduled_at_ms'] = begin + 12 * HOUR
    assert evaluate(raw, at)['reasons'] == ['international_calendar_unknown_time_fabricated:jp-cpi']


def test_ecb_dst_is_cet_on_october_29_not_summer_time():
    anchor = ms('2026-10-29T14:15:00+01:00')
    raw = payload(asof=anchor, anchor=anchor)
    raw['events'][0].update(scheduled_at_local='2026-10-29T14:15:00+01:00',
                            timezone='Europe/Berlin')
    assert evaluate(raw, anchor)['status'] == 'blackout'
    # A self-consistent UTC anchor with the wrong seasonal local offset is rejected.
    raw['events'][0].update(scheduled_at_ms=ms('2026-10-29T14:15:00+02:00'),
                            scheduled_at_local='2026-10-29T14:15:00+02:00')
    assert not evaluate(raw, anchor)['allowed']
    assert evaluate(raw, anchor)['status'] == 'unavailable'


def test_pinned_official_calendar_keeps_new_first_seen_and_release_uncertainty():
    path = ROOT / 'config/international_event_calendar_20261023.json'
    raw = calendar.load_calendar(path)
    result = evaluate(raw, raw['verified_at_ms'])
    assert result['allowed'] and result['known_event_count'] == 6
    before = evaluate(raw, raw['first_seen_at_ms'] - 1)
    assert not before['allowed']
    events = {event['event_id']: event for event in raw['events']}
    assert events['jp_national_cpi_202609_release_20261023']['scheduled_at_ms'] == ms('2026-10-23T07:30:00+08:00')
    assert events['jp_tokyo_cpi_202610_preliminary_20261030']['scheduled_at_ms'] == ms('2026-10-30T07:30:00+08:00')
    assert events['ecb_policy_decision_20261029']['scheduled_at_ms'] == ms('2026-10-29T21:15:00+08:00')
    assert events['ecb_press_conference_20261029']['scheduled_at_ms'] == ms('2026-10-29T21:45:00+08:00')
    assert all(event['actual_at_ms'] is None for event in raw['events'])
    company = raw['company_reviews']['SKHYNIXUSDT']
    assert company['source_published_date'] == '2026-10-21'
    assert company['first_release_time_ms'] is None and company['call_end_time_ms'] is None
    assert company['first_release_before_ms'] == company['call_scheduled_at_ms']
    assert company['existing_company_calendar_coverage_extended'] is False
    assert raw['first_seen_at_ms'] > ms('2026-10-22T16:30:00+00:00')


def test_loader_rejects_missing_corrupt_or_oversized_bytes(tmp_path):
    path = tmp_path / 'calendar.json'
    with pytest.raises(ValueError, match='international_calendar_unavailable'):
        calendar.load_calendar(path)
    for value in ['{', 'null', '[]']:
        path.write_text(value, encoding='utf-8')
        with pytest.raises(ValueError, match='international_calendar_payload_invalid'):
            calendar.load_calendar(path)
    path.write_bytes(b' ' * (calendar.MAX_CALENDAR_BYTES + 1))
    with pytest.raises(ValueError, match='international_calendar_too_large'):
        calendar.load_calendar(path)
    path.write_text(json.dumps(payload()), encoding='utf-8-sig')
    assert calendar.load_calendar(path)['calendar_id'] == 'independent-test'


@pytest.mark.parametrize('policy', [{'pre_event_minutes': True}, {'post_event_minutes': -1},
    {'max_verification_age_hours': 0}, {'unknown': 1}])
def test_invalid_candidate_policy_fails_closed(policy):
    assert evaluate(policy=policy)['reasons'] == ['international_calendar_policy_invalid']
