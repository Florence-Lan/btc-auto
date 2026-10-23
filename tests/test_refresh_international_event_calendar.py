"""Official semantics, bounded GETs and immutable first-seen/coverage renewal."""
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refresh_international_event_calendar as refresh
from international_event_calendar import evaluate_international_event_calendar
from public_request_deadline import PublicRequestDeadline


NOW = int(datetime.fromisoformat('2026-10-23T02:00:00+00:00').timestamp() * 1000)
HOUR = 3_600_000
HTML = {
    'cpi_schedule': '<h1>Schedule of Release</h1> Last Update : 23 January 2026 <table><tr><td>September</td><td>October 23</td><td>October</td><td>October 30</td></tr></table>',
    'cpi_estat': '<div>2026年9月分 公表日 2026-10-23 08:30</div><div>東京都区部（中旬速報値） 2025年度平均 公表日 2026-03-31 08:30 2026年10月分 公表日 2026-10-30 08:30</div>',
    'boj': '<div>2026年10月29日（木）、30日（金）</div><table><tr><td>30日（金）</td><td>未定</td><td>○ ● 政策委員会・金融政策決定会合の決定内容</td></tr></table><p>2026年10月30日（金）15:30～（ライブ配信予定）</p>',
    'ecb_calendar': '<p>29/10/2026 Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 2), followed by press conference</p><p>29/10/2026 Press conference following the Governing Council meeting of the ECB in Frankfurt</p>',
    'ecb_decision': '<p>The ECB’s monetary policy decisions are published in a press release at 14:15 CET on the day of the Governing Council monetary policy meeting.</p>',
    'ecb_conference': '<h1>Press conference</h1><h2>14:45 CET - Berlin, Germany</h2>',
}


def setup(tmp_path):
    raw = refresh.load_calendar(refresh.DEFAULT_TEMPLATE)
    path = tmp_path / 'template.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    return path, tmp_path / 'observer', raw


def fetcher(patch=None, code=200):
    pages = {**HTML, **(patch or {})}
    by_url = {url: name for name, url in refresh.PAGES.items()}
    return lambda url: {'status_code': code, 'body': pages[by_url[url]].encode('utf-8')}


def test_success_requires_all_six_pages_and_preserves_original_clocks_and_horizon(tmp_path):
    path, output, template = setup(tmp_path)
    before = path.read_bytes()
    result = refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=fetcher())
    assert result['success'] and result['status'] == 'verified'
    assert result['next_attempt_after_ms'] == NOW + 6 * HOUR
    current = refresh.load_verified_calendar(path, output)
    for key in ('first_seen_at_ms', 'available_at_ms', 'coverage_start_ms', 'coverage_end_ms'):
        assert current[key] == template[key]
    assert current['verified_at_ms'] == NOW
    assert current['valid_until_ms'] == NOW + 24 * HOUR
    assert current['company_reviews'] == template['company_reviews']
    assert all(row['actual_at_ms'] is None for row in current['events'])
    assert [row['first_seen_at_ms'] for row in current['events']] == [row['first_seen_at_ms'] for row in template['events']]
    assert path.read_bytes() == before
    assert evaluate_international_event_calendar(current, asof_ms=NOW)['allowed']
    assert len(result['pages']) == 6
    for name, detail in result['pages'].items():
        body = HTML[name].encode('utf-8')
        expected = hashlib.sha256(body).hexdigest()
        assert detail['raw_sha256'] == expected
        assert gzip.decompress((output / 'evidence/international_calendar_sources' / (expected + '.html.gz')).read_bytes()) == body


@pytest.mark.parametrize('name,old,new', [
    ('cpi_estat', '2026-10-23 08:30', '2026-10-23 08:31'),
    ('cpi_estat', '2026-10-30 08:30', '2026-10-31 08:30'),
    ('boj', '未定', '12:00'),
    ('boj', '15:30', '16:30'),
    ('ecb_calendar', '29/10/2026', '30/10/2026'),
    ('ecb_decision', '14:15 CET', '14:15 CEST'),
    ('ecb_conference', '14:45 CET', '15:45 CET'),
])
def test_http_200_with_changed_or_missing_semantics_does_not_renew(tmp_path, name, old, new):
    path, output, _ = setup(tmp_path)
    result = refresh.refresh_calendar(path, output, asof_ms=NOW,
        fetcher=fetcher({name: HTML[name].replace(old, new)}))
    assert not result['success'] and result['error_type'] == 'VerificationFailure'
    assert result['next_attempt_after_ms'] == NOW + HOUR
    assert not (output / 'calendar_verified.json').exists()


def test_failure_retains_old_verified_bytes_and_does_not_reset_their_expiry(tmp_path):
    path, output, _ = setup(tmp_path)
    refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=fetcher())
    before = (output / 'calendar_verified.json').read_bytes()
    result = refresh.refresh_calendar(path, output, asof_ms=NOW + 7 * HOUR, fetcher=fetcher(code=403))
    assert not result['success'] and result['retained_calendar_compatible']
    assert result['retained_verified_at_ms'] == NOW
    assert result['retained_valid_until_ms'] == NOW + 24 * HOUR
    assert (output / 'calendar_verified.json').read_bytes() == before
    retained = refresh.load_verified_calendar(path, output)
    assert not evaluate_international_event_calendar(retained, asof_ms=NOW + 24 * HOUR + 1)['allowed']


def test_changed_template_bytes_and_inception_do_not_inherit_old_runtime_calendar(tmp_path):
    path, output, _ = setup(tmp_path)
    refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=fetcher())
    template = json.loads(path.read_text(encoding='utf-8'))
    template['first_seen_at_ms'] += 1
    path.write_text(json.dumps(template), encoding='utf-8')
    with pytest.raises(refresh.VerificationFailure, match='runtime_calendar_template_mismatch'):
        refresh.load_verified_calendar(path, output)
    result = refresh.refresh_calendar(path, output, asof_ms=NOW + HOUR, fetcher=fetcher(code=403))
    assert not result['success'] and result['retained_calendar_compatible'] is False


def test_coverage_cannot_be_extended_by_a_successful_http_response(tmp_path):
    path, output, template = setup(tmp_path)
    calls = []
    def no_fetch(url):
        calls.append(url)
        return fetcher()(url)
    result = refresh.refresh_calendar(path, output, asof_ms=template['coverage_end_ms'], fetcher=no_fetch)
    assert not result['success'] and result['failure_reason'] == 'template_coverage_expired'
    assert calls == []


def test_transport_exceptions_do_not_publish_proxy_credentials(tmp_path):
    path, output, _ = setup(tmp_path)
    def failed(_url):
        raise requests.ConnectionError('https://private-user:private-password@proxy.invalid/path')
    result = refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=failed)
    serialized = json.dumps(result)
    assert not result['success']
    assert 'ConnectionError' in serialized
    assert 'private-user' not in serialized and 'private-password' not in serialized


def test_slow_get_batch_is_bounded_without_waiting_for_abandoned_sockets(tmp_path, monkeypatch):
    path, output, _ = setup(tmp_path)
    monkeypatch.setattr(refresh, 'TOTAL_BUDGET_SECONDS', 0.06)
    monkeypatch.setattr(refresh, '_DEADLINE', PublicRequestDeadline(timeout_seconds=0.04, max_pending=6))
    def slow(url):
        time.sleep(0.3)
        return fetcher()(url)
    started = time.monotonic()
    result = refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=slow)
    assert time.monotonic() - started < 0.25
    assert not result['success'] and not (output / 'calendar_verified.json').exists()


def test_future_template_or_unexpected_event_definition_is_not_auto_verified(tmp_path):
    path, output, template = setup(tmp_path)
    template['events'][0]['scheduled_at_local'] = '2026-10-24T08:30:00+09:00'
    path.write_text(json.dumps(template), encoding='utf-8')
    result = refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=fetcher())
    assert not result['success'] and result['failure_reason'].startswith('fixed_event_time_changed:')
    template['events'][0]['scheduled_at_local'] = '2026-10-23T08:30:00+09:00'
    template['first_seen_at_ms'] = NOW + 1
    path.write_text(json.dumps(template), encoding='utf-8')
    result = refresh.refresh_calendar(path, output, asof_ms=NOW, fetcher=fetcher())
    assert not result['success'] and result['failure_reason'] == 'template_not_yet_available'


@pytest.mark.parametrize('mutation', ['future_verification', 'disabled_event', 'actual_added'])
def test_refresh_does_not_overwrite_future_clocks_or_silently_change_schedule_scope(tmp_path, mutation):
    path, output, template = setup(tmp_path)
    if mutation == 'future_verification': template['verified_at_ms'] = NOW + 1
    if mutation == 'disabled_event': template['events'][0]['block_new_risk'] = False
    if mutation == 'actual_added': template['events'][0]['actual_at_ms'] = NOW - 1
    path.write_text(json.dumps(template), encoding='utf-8')
    calls = []
    result = refresh.refresh_calendar(path, output, asof_ms=NOW,
        fetcher=lambda url: calls.append(url) or fetcher()(url))
    assert not result['success'] and calls == []
