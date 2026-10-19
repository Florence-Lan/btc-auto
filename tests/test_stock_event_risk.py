"""Causal earnings admission boundaries, data failures and public provenance."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_event_risk as risk

HOUR = risk.HOUR


def calendar_data(anchor=624 * HOUR, symbol='MUUSDT'):
    return {'version': 1, 'calendar_id': 'fixture', 'first_seen_at_ms': 500 * HOUR,
        'available_at_ms': 500 * HOUR, 'verified_at_ms': 500 * HOUR,
        'valid_until_ms': 800 * HOUR, 'coverage_start_ms': 500 * HOUR,
        'coverage_end_ms': 800 * HOUR,
        'symbol_reviews': {symbol: {'status': 'official_event_verified',
            'verified_at_ms': 500 * HOUR, 'source_urls': ['https://example.test/official']}},
        'events': [{'event_id': 'earnings', 'symbols': [symbol], 'anchor_time_ms': anchor,
            'anchor_kind': risk.ANCHOR, 'first_release_time_ms': None,
            'first_release_time_status': 'unknown', 'call_end_time_ms': None,
            'call_end_time_status': 'unknown', 'first_seen_at_ms': 500 * HOUR,
            'available_at_ms': 500 * HOUR, 'verified_at_ms': 500 * HOUR,
            'source_url': 'https://example.test/official', 'source_title': 'Official call notice',
            'source_published_date': '2026-10-14'}]}


def setup_calendar(tmp_path, anchor=624 * HOUR, symbol='MUUSDT'):
    path = tmp_path / 'events.json'
    path.write_text(json.dumps(calendar_data(anchor, symbol)), encoding='utf-8')
    cfg = {risk.POLICY_KEY: {**risk.POLICY, 'calendar_path': str(path),
                            'effective_from_ms': 500 * HOUR}}
    return cfg, path


def bar(start, **patch):
    return {'time_ms': start, **patch}


def test_disabled_is_compatible_and_pre_activation_never_reads_today_calendar(tmp_path):
    assert risk.evaluate({}, 'MUUSDT', 1)['allowed']
    assert not risk.evaluate({}, 'MUUSDT', 1)['enabled']
    cfg, path = setup_calendar(tmp_path)
    path.unlink()
    historical = risk.evaluate(cfg, 'MUUSDT', 500 * HOUR - 1)
    assert historical['allowed'] and historical['status'] == 'not_active_yet'
    assert 'calendar_sha256' not in historical and historical['event_ids'] == []
    active = risk.evaluate(cfg, 'MUUSDT', 500 * HOUR)
    assert not active['allowed'] and active['reasons'] == ['stock_event_calendar_unavailable']


@pytest.mark.parametrize('patch', [None, {}, {**risk.POLICY, 'calendar_path': 'x'},
    {**risk.POLICY, 'calendar_path': '', 'effective_from_ms': 0},
    {**risk.POLICY, 'calendar_path': 'x', 'effective_from_ms': True},
    {**risk.POLICY, 'calendar_path': 'x', 'effective_from_ms': 0, 'post_event_hours': 3},
    {**risk.POLICY, 'calendar_path': 'x', 'effective_from_ms': 0, 'pre_event_hours': 24.0}])
def test_bad_explicit_policy_fails_closed_without_raising_to_exit_caller(patch):
    cfg = {risk.POLICY_KEY: patch}
    with pytest.raises(ValueError):
        risk.validate(cfg)
    outcome = risk.evaluate(cfg, 'MUUSDT', 600 * HOUR)
    assert outcome['enabled'] and not outcome['allowed']
    assert outcome['protective_exits_allowed'] and outcome['scope'] == 'new_risk_only'


@pytest.mark.parametrize('at,allowed', [(600*HOUR-1, True), (600*HOUR, False),
    (624*HOUR, False), (626*HOUR-1, False)])
def test_inclusive_24h_blackout_boundary(tmp_path, at, allowed):
    cfg, _ = setup_calendar(tmp_path)
    result = risk.evaluate(cfg, 'MUUSDT', at)
    assert result['allowed'] is allowed
    if not allowed:
        assert result['reasons'] == ['stock_event_blackout']
        assert result['event_ids'] == ['earnings']
        assert result['blocked_events'][0]['first_release_time_status'] == 'unknown'


def test_two_actual_complete_post_event_hours_and_no_partial_hour_substitution(tmp_path):
    anchor = 624 * HOUR + HOUR // 2
    cfg, _ = setup_calendar(tmp_path, anchor)
    rows = [bar(i * HOUR) for i in range(623, 627)]
    # 624:00--625:00 straddles the call and is ineligible.
    at_two_hours = risk.evaluate(cfg, 'MUUSDT', anchor + 2 * HOUR, closed_hourly_bars=rows)
    assert not at_two_hours['allowed']
    assert at_two_hours['blocked_events'][0]['earliest_resume_at_ms'] == 627 * HOUR
    assert risk.evaluate(cfg, 'MUUSDT', 627 * HOUR, closed_hourly_bars=rows)['allowed']
    for source in (None, rows[:-1], [bar(625*HOUR), bar(627*HOUR)],
                   [bar(625*HOUR), bar(625*HOUR), bar(626*HOUR)],
                   [[625*HOUR, 1, 1, 1, 1, 1, 626*HOUR]]):
        result = risk.evaluate(cfg, 'MUUSDT', 627 * HOUR, closed_hourly_bars=source)
        assert not result['allowed']
        assert result['reasons'] == ['stock_event_post_event_bars_unavailable']


def test_expiry_coverage_and_symbol_are_explainable_and_do_not_fake_no_event(tmp_path):
    cfg, path = setup_calendar(tmp_path)
    no_symbol = risk.evaluate(cfg, 'SNDKUSDT', 600*HOUR)
    assert no_symbol['reasons'] == ['stock_event_symbol_not_covered']
    source = json.loads(path.read_text())
    source['symbol_reviews']['SNDKUSDT'] = {'status': 'no_announced_event_found',
        'verified_at_ms': 500*HOUR, 'source_urls': ['https://example.test/ir']}
    path.write_text(json.dumps(source))
    clear = risk.evaluate(cfg, 'SNDKUSDT', 600*HOUR)
    assert clear['allowed'] and clear['known_event_count'] == 0
    assert clear['calendar_remaining_ms'] == 200*HOUR
    expired = risk.evaluate(cfg, 'SNDKUSDT', 800*HOUR)
    assert not expired['allowed'] and expired['reasons'] == ['stock_event_calendar_expired']
    assert expired['valid_until_ms'] == 800*HOUR
    assert expired['calendar_remaining_ms'] == 0


@pytest.mark.parametrize('corruption', ['json', 'null', 'duplicate', 'missing_event',
    'bad_anchor', 'future', 'bool_time', 'invented_release', 'missing_symbol_review'])
def test_invalid_or_future_snapshot_blocks_without_leaking_future_events(tmp_path, corruption):
    cfg, path = setup_calendar(tmp_path)
    raw = json.loads(path.read_text())
    if corruption == 'json':
        path.write_text('{')
    elif corruption == 'null':
        path.write_text('null')
    else:
        if corruption == 'duplicate': raw['events'] *= 2
        if corruption == 'missing_event': raw['events'] = []
        if corruption == 'bad_anchor': raw['events'][0]['anchor_kind'] = 'estimated_first_release'
        if corruption == 'future': raw.update(verified_at_ms=601*HOUR)
        if corruption == 'bool_time': raw['events'][0]['anchor_time_ms'] = True
        if corruption == 'invented_release': raw['events'][0]['first_release_time_ms'] = 624*HOUR
        if corruption == 'missing_symbol_review': del raw['symbol_reviews']['MUUSDT']
        path.write_text(json.dumps(raw))
    result = risk.evaluate(cfg, 'MUUSDT', 600*HOUR)
    assert not result['allowed'] and result['status'] == 'unavailable'
    assert result['event_ids'] == []
    if corruption == 'future':
        assert result['reasons'] == ['stock_event_calendar_not_yet_available']


def test_recheck_reads_new_bytes_and_can_block_previously_approved_pending(tmp_path):
    cfg, path = setup_calendar(tmp_path, 625*HOUR)
    before = risk.evaluate(cfg, 'MUUSDT', 600*HOUR)
    assert before['allowed']
    changed = json.loads(path.read_text())
    changed['events'][0]['anchor_time_ms'] = 624*HOUR
    path.write_text(json.dumps(changed))
    after = risk.evaluate(cfg, 'MUUSDT', 600*HOUR)
    assert not after['allowed']
    assert after['calendar_sha256'] != before['calendar_sha256']


def test_shipped_calendar_timezone_mapping_and_full_lifecycle_horizon():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root/'config/stock_event_calendar_20261020.json').read_text(encoding='utf-8'))
    events = {event['event_id']: event for event in raw['events']}
    assert events['skhynix_2026q3_call_20261028']['anchor_time_ms'] == 1793145600000
    assert events['sandisk_fy2027q1_call_20261029']['anchor_time_ms'] == 1793305800000
    assert raw['valid_until_ms'] == raw['coverage_end_ms'] == 1793462400000
    cfg = {risk.POLICY_KEY: {**risk.POLICY, 'calendar_path': 'config/stock_event_calendar_20261020.json',
                            'effective_from_ms': raw['available_at_ms']}}
    for symbol in raw['symbol_reviews']:
        assert risk.evaluate(cfg, symbol, raw['available_at_ms'], root=root)['allowed']


def test_official_verifier_rejects_different_notice_instead_of_retiming_event():
    raw = {'events': [{'event_id': 'skhynix_2026q3_call_20261028',
        'anchor_time_ms': 1793145600000, 'source_url': 'https://example.test/notice',
        'source_published_date': '2026-10-14'}]}
    before = deepcopy(raw)
    evidence = risk.verify_official_notices(raw, fetch=lambda _: (
        b'<p>October 28, 2026 9:00 a.m. (Seoul Time)</p>'))
    assert evidence[0]['known_call_start_confirmed']
    assert raw == before and evidence[0]['source_published_date'] == '2026-10-14'
    with pytest.raises(ValueError, match='not_confirmed'):
        risk.verify_official_notices(raw, fetch=lambda _: b'<p>October 29, 2026</p>')


def test_refresh_tracks_partial_source_failure_without_extending_or_rewriting_causal_times(
        tmp_path, monkeypatch, capsys):
    _, path = setup_calendar(tmp_path)
    before = json.loads(path.read_text())
    outcome = [{'event_id': 'earnings', 'known_call_start_confirmed': False,
                'error': 'HTTPError: 403 from official source'}]
    monkeypatch.setattr(risk, 'verify_official_notices', lambda *_args, **_kwargs: outcome)
    monkeypatch.setattr(risk.time, 'time', lambda: 600*HOUR/1000)
    assert risk.main(['--calendar', str(path), '--refresh-official']) == 1
    after = json.loads(path.read_text())
    for key in ('first_seen_at_ms', 'available_at_ms', 'verified_at_ms',
                'coverage_start_ms', 'coverage_end_ms', 'valid_until_ms', 'events'):
        assert after[key] == before[key]
    assert after['fetch_attempt_at_ms'] == 600*HOUR
    assert 'fetch_success_at_ms' not in after
    assert after['source_fetch_verification'] == outcome
    assert list(path.with_name('events_history').glob('*.json'))
    assert 'all_known_notices_confirmed' in capsys.readouterr().out


def test_slow_snapshot_read_cannot_use_approval_from_before_window(tmp_path, monkeypatch):
    cfg, path = setup_calendar(tmp_path, 624*HOUR+2000)
    clock = {'now': 600*HOUR+1000}
    original = Path.read_bytes
    def slow_read(self):
        payload = original(self)
        if self == path:
            clock['now'] += 2000
        return payload
    monkeypatch.setattr(Path, 'read_bytes', slow_read)
    decision = risk.evaluate(cfg, 'MUUSDT', 600*HOUR+1000,
                             reference_clock=lambda: clock['now'])
    assert decision['checked_at_ms'] == 600*HOUR+3000
    assert decision['reasons'] == ['stock_event_blackout']
