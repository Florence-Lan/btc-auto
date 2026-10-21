"""Macro admission preserves collector causality, source expiry and exit scope."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import event_risk
import public_context as pc
import stock_macro_event_risk as risk

NOW = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
MINUTE = 60_000
HOUR = 60 * MINUTE


def ms(value):
    return int(value.timestamp() * 1000)


def iso(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def snapshot(at=ms(NOW), anchor=ms(NOW) + 2 * HOUR):
    first = NOW - timedelta(days=1)
    calendars = [pc.calendar_item('fomc', 'FOMC policy decision',
        datetime.fromtimestamp(anchor / 1000, timezone.utc), 'fomc_calendar', pc.FED_CALENDAR, first),
        pc.calendar_item('gdp', 'Gross Domestic Product Advance Release', NOW + timedelta(days=9),
                         'economic_calendar', pc.NYFED_CALENDAR, first),
        pc.calendar_item('pce', 'Personal Income and the PCE Deflator', NOW + timedelta(days=9),
                         'economic_calendar', pc.NYFED_CALENDAR, first),
        pc.calendar_item('fomc-next', 'FOMC policy decision', NOW + timedelta(days=30),
                         'fomc_calendar', pc.FED_CALENDAR, first)]
    calendars = pc.merge_calendar([], calendars, first)
    return {'schema_version': 1, 'generated_at_utc': iso(at), 'calendars': calendars,
        'events': pc.make_events([], calendars), 'coverage_checks': [{
            'available_at_utc': iso(at), 'sources': {name: {
                'ok': True, 'fetch_ok': True, 'status': 'live',
                'last_success_at_utc': iso(at - 1)} for name in risk.REQUIRED_SOURCES}}]}


def setup_snapshot(tmp_path, raw=None, *, anchor=ms(NOW) + 2 * HOUR, at=ms(NOW)):
    path = tmp_path / 'public_context.json'
    path.write_text(json.dumps(snapshot(at, anchor) if raw is None else raw), encoding='utf-8')
    cfg = {risk.POLICY_KEY: {**risk.POLICY, 'snapshot_path': str(path),
                            'effective_from_ms': ms(NOW) - 12 * HOUR}}
    return cfg, path


def write(path, raw):
    path.write_text(json.dumps(raw), encoding='utf-8')


def evaluate(cfg, at=ms(NOW), **kwargs):
    return risk.evaluate(cfg, 'MUUSDT', at, **kwargs)


def test_disabled_and_pre_effective_do_not_read_later_snapshot(tmp_path):
    assert evaluate({})['status'] == 'disabled'
    cfg, path = setup_snapshot(tmp_path)
    path.unlink()
    prior = evaluate(cfg, cfg[risk.POLICY_KEY]['effective_from_ms'] - 1)
    assert prior['allowed'] and prior['status'] == 'not_active_yet'
    assert prior['event_ids'] == [] and 'snapshot_sha256' not in prior
    active = evaluate(cfg)
    assert active['reasons'] == ['stock_macro_event_snapshot_unavailable']
    assert active['protective_exits_allowed'] and active['scope'] == 'new_risk_only'


@pytest.mark.parametrize('patch', [None, {}, {**risk.POLICY, 'snapshot_path': 'x'},
    {**risk.POLICY, 'snapshot_path': '', 'effective_from_ms': 0},
    {**risk.POLICY, 'snapshot_path': 'x', 'effective_from_ms': True},
    {**risk.POLICY, 'snapshot_path': 'x', 'effective_from_ms': 0, 'pre_event_minutes': 29},
    {**risk.POLICY, 'snapshot_path': 'x', 'effective_from_ms': 0, 'post_event_minutes': 90.0}])
def test_invalid_opt_in_policy_fails_closed_without_interrupting_exits(patch):
    cfg = {risk.POLICY_KEY: patch}
    with pytest.raises(ValueError):
        risk.validate(cfg)
    result = evaluate(cfg)
    assert result['enabled'] and not result['allowed']
    assert result['protective_exits_allowed']


@pytest.mark.parametrize('offset,allowed', [(-30 * MINUTE - 1, True),
    (-30 * MINUTE, False), (0, False), (90 * MINUTE, False), (90 * MINUTE + 1, True)])
def test_exact_existing_window_including_shared_engine_end_boundary(tmp_path, offset, allowed):
    anchor = ms(NOW) + 2 * HOUR
    at = anchor + offset
    raw = snapshot(at, anchor)
    cfg, _ = setup_snapshot(tmp_path, raw)
    outcome = evaluate(cfg, at)
    expected = event_risk.event_decision_at(event_risk.events_from_payload(raw), at)
    assert outcome['allowed'] is allowed is expected.allowed
    assert outcome['event_ids'] == list(expected.active_event_ids)
    assert outcome['status'] == ('clear' if allowed else 'blackout')
    if not allowed:
        assert outcome['reasons'] == ['stock_macro_event_blackout']
        assert outcome['blocked_events'][0]['scheduled_at_ms'] == anchor


def test_gdp_and_pce_both_block_without_direction_or_double_window(tmp_path):
    at = ms(NOW + timedelta(days=9))
    cfg, _ = setup_snapshot(tmp_path, at=at)
    outcome = evaluate(cfg, at)
    assert not outcome['allowed'] and len(outcome['event_ids']) == 2
    assert {row['headline'] for row in outcome['blocked_events']} == {
        'Gross Domestic Product Advance Release', 'Personal Income and the PCE Deflator'}
    assert all(row['starts_at_ms'] == at - 30 * MINUTE for row in outcome['blocked_events'])
    assert all(row['ends_at_ms'] == at + 90 * MINUTE for row in outcome['blocked_events'])
    assert outcome['protective_exits_allowed']


def test_cached_source_uses_original_receipt_and_ignores_unrelated_news_failure(tmp_path):
    raw = snapshot()
    states = raw['coverage_checks'][0]['sources']
    for state in states.values():
        state.update(fetch_ok=False, status='cached', last_success_at_utc=iso(ms(NOW) - 59 * MINUTE))
    states['un_news'] = {'ok': False}
    cfg, _ = setup_snapshot(tmp_path, raw)
    outcome = evaluate(cfg)
    assert outcome['allowed']
    assert outcome['source_status']['fomc_calendar']['status'] == 'cached'
    assert outcome['source_status']['fomc_calendar']['age_ms'] == 59 * MINUTE
    assert outcome['valid_until_ms'] == ms(NOW) + MINUTE
    expired = evaluate(cfg, ms(NOW) + MINUTE + 1)
    assert expired['reasons'] == ['stock_macro_event_source_stale:fomc_calendar']


@pytest.mark.parametrize('mutation,reason', [
    ('stale_content', 'stock_macro_event_source_stale:fomc_calendar'),
    ('missing_clock', 'stock_macro_event_source_clock_invalid:fomc_calendar'),
    ('future_clock', 'stock_macro_event_source_clock_invalid:fomc_calendar'),
    ('naive_clock', 'stock_macro_event_source_clock_invalid:fomc_calendar'),
    ('bad_ok', 'stock_macro_event_source_status_invalid:fomc_calendar'),
    ('unavailable', 'stock_macro_event_source_unavailable:fomc_calendar'),
    ('missing_source', 'stock_macro_event_source_status_invalid:fomc_calendar'),
    ('empty_content', 'stock_macro_event_calendar_not_covered:fomc_calendar')])
def test_green_fetch_and_new_generation_cannot_replace_fresh_valid_content(tmp_path, mutation, reason):
    raw = snapshot()
    states = raw['coverage_checks'][0]['sources']
    state = states['fomc_calendar']
    if mutation == 'stale_content': state['last_success_at_utc'] = iso(ms(NOW) - HOUR - 1)
    if mutation == 'missing_clock': state.pop('last_success_at_utc')
    if mutation == 'future_clock': state['last_success_at_utc'] = iso(ms(NOW) + 1)
    if mutation == 'naive_clock': state['last_success_at_utc'] = '2026-10-20T14:00:00'
    if mutation == 'bad_ok': state['ok'] = 'true'
    if mutation == 'unavailable': state['ok'] = False
    if mutation == 'missing_source': states.pop('fomc_calendar')
    if mutation == 'empty_content': raw.update(calendars=[], events=[])
    cfg, _ = setup_snapshot(tmp_path, raw)
    outcome = evaluate(cfg)
    assert not outcome['allowed'] and outcome['reasons'] == [reason]


def test_historical_check_is_not_rescued_by_future_healthy_check(tmp_path):
    raw = snapshot()
    future = deepcopy(raw['coverage_checks'][0])
    future['available_at_utc'] = iso(ms(NOW) + MINUTE)
    for state in future['sources'].values():
        state['last_success_at_utc'] = iso(ms(NOW) + MINUTE)
    raw['coverage_checks'][0]['sources']['economic_calendar']['ok'] = False
    raw['coverage_checks'].append(future)
    raw['generated_at_utc'] = future['available_at_utc']
    cfg, _ = setup_snapshot(tmp_path, raw)
    outcome = evaluate(cfg)
    assert outcome['reasons'] == ['stock_macro_event_source_unavailable:economic_calendar']
    assert outcome['event_ids'] == [] and 'next_event' not in outcome
    assert evaluate(cfg, ms(NOW) + MINUTE)['allowed']


@pytest.mark.parametrize('mutation', ['json', 'null', 'schema_bool', 'naive_event',
    'wrong_window', 'missing_event', 'orphan_event', 'duplicate_event', 'bool_block',
    'duplicate_revision', 'coverage_unordered', 'coverage_future', 'coverage_stale'])
def test_malformed_incomplete_or_unobserved_snapshots_fail_closed(tmp_path, mutation):
    raw = snapshot()
    if mutation == 'schema_bool': raw['schema_version'] = True
    if mutation == 'naive_event': raw['events'][0]['starts_at_utc'] = '2026-10-20T15:30:00'
    if mutation == 'wrong_window': raw['events'][0]['ends_at_utc'] = iso(ms(NOW) + 9 * HOUR)
    if mutation == 'missing_event': raw['events'].pop(0)
    if mutation == 'orphan_event': raw['calendars'].pop(0)
    if mutation == 'duplicate_event': raw['events'].append(deepcopy(raw['events'][0]))
    if mutation == 'bool_block': raw['events'][0]['block_entries'] = 'false'
    if mutation == 'duplicate_revision': raw['calendars'].append(deepcopy(raw['calendars'][0]))
    if mutation == 'coverage_unordered':
        row = deepcopy(raw['coverage_checks'][0])
        row['available_at_utc'] = iso(ms(NOW) - 1)
        raw['coverage_checks'].append(row)
    if mutation == 'coverage_future':
        raw['coverage_checks'][0]['available_at_utc'] = iso(ms(NOW) + 1)
        raw['generated_at_utc'] = iso(ms(NOW) + 1)
    if mutation == 'coverage_stale':
        raw['coverage_checks'][0]['available_at_utc'] = iso(ms(NOW) - HOUR - 1)
    cfg, path = setup_snapshot(tmp_path, raw)
    if mutation == 'json': path.write_text('{', encoding='utf-8')
    if mutation == 'null': path.write_text('null', encoding='utf-8')
    outcome = evaluate(cfg)
    assert not outcome['allowed'] and outcome['status'] == 'unavailable'
    assert outcome['event_ids'] == [] and outcome['protective_exits_allowed']


def test_revision_supersession_and_first_seen_match_shared_engine(tmp_path):
    anchor = ms(NOW) + 15 * MINUTE
    changed_at = ms(NOW) + MINUTE
    raw = snapshot(changed_at, anchor)
    old = raw['calendars'][0]
    old['superseded_at_utc'] = iso(changed_at)
    replacement = {**old, 'first_seen_at_utc': iso(changed_at), 'revision_id': 'new-revision',
                   'scheduled_at_utc': iso(anchor + 2 * HOUR)}
    replacement.pop('superseded_at_utc')
    raw['calendars'].append(replacement)
    raw['events'] = pc.make_events([], raw['calendars'])
    # Both causal coverage versions are archived in this later snapshot.
    previous = deepcopy(raw['coverage_checks'][0])
    previous['available_at_utc'] = iso(ms(NOW))
    for state in previous['sources'].values():
        state['last_success_at_utc'] = iso(ms(NOW))
    raw['coverage_checks'].insert(0, previous)
    for state in raw['coverage_checks'][-1]['sources'].values():
        state['last_success_at_utc'] = iso(changed_at)
    cfg, _ = setup_snapshot(tmp_path, raw)
    prior = evaluate(cfg)
    assert not prior['allowed'] and prior['event_ids'] == ['calendar:' + old['revision_id']]
    assert prior['known_event_count'] == 4  # New revision never leaks backward.
    after = evaluate(cfg, changed_at)
    assert after['allowed'] and after['event_ids'] == []
    assert after['next_event']['event_id'] == 'calendar:new-revision'


def test_final_recheck_reads_current_bytes_and_clock_after_disk_load(tmp_path):
    at = ms(NOW)
    anchor = at + 30 * MINUTE + 1000
    cfg, path = setup_snapshot(tmp_path, anchor=anchor)
    assert evaluate(cfg)['allowed']
    changed = snapshot(at, at + 30 * MINUTE)
    write(path, changed)
    assert evaluate(cfg)['reasons'] == ['stock_macro_event_blackout']
    write(path, snapshot(at, anchor))
    outcome = evaluate(cfg, reference_clock=lambda: at + 2000)
    assert outcome['checked_at_ms'] == at + 2000
    assert outcome['reasons'] == ['stock_macro_event_blackout']
    assert evaluate(cfg, reference_clock=lambda: at - 1)['reasons'] == [
        'stock_macro_event_reference_clock_reversed']


def test_relative_experiment_snapshot_path_and_source_horizon(tmp_path):
    cfg, path = setup_snapshot(tmp_path)
    cfg[risk.POLICY_KEY]['snapshot_path'] = path.name
    result = evaluate(cfg, root=tmp_path)
    assert result['allowed'] and result['source_max_age_ms'] == HOUR
    assert result['coverage_end_ms'] == ms(NOW + timedelta(days=9)) + 90 * MINUTE
    assert set(result['source_status']) == set(risk.REQUIRED_SOURCES)


def test_future_revision_is_invisible_until_observed_but_inconsistent_receipt_blocks(tmp_path):
    at = ms(NOW)
    raw = snapshot(at + MINUTE)
    raw['calendars'][0]['first_seen_at_utc'] = iso(at + MINUTE)
    raw['events'] = pc.make_events([], raw['calendars'])
    earlier = deepcopy(raw['coverage_checks'][0])
    earlier['available_at_utc'] = iso(at)
    for state in earlier['sources'].values():
        state['last_success_at_utc'] = iso(at)
    raw['coverage_checks'].insert(0, earlier)
    cfg, _ = setup_snapshot(tmp_path, raw)
    prior = evaluate(cfg, at)
    assert prior['allowed'] and prior['known_event_count'] == 3
    assert prior['next_event']['source'] == 'economic_calendar'
    outcome = evaluate(cfg, at + MINUTE)
    assert outcome['reasons'] == ['stock_macro_event_calendar_after_source_receipt:fomc_calendar']


def test_content_expiry_uses_existing_inclusive_one_hour_limit(tmp_path):
    at = ms(NOW)
    raw = snapshot()
    for state in raw['coverage_checks'][0]['sources'].values():
        state.update(fetch_ok=False, last_success_at_utc=iso(at - HOUR))
    cfg, _ = setup_snapshot(tmp_path, raw)
    assert evaluate(cfg, at)['allowed']
    assert evaluate(cfg, at + 1)['reasons'] == ['stock_macro_event_source_stale:fomc_calendar']
