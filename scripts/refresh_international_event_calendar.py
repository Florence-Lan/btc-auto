"""Low-frequency semantic re-verification of the fixed October calendar candidate.

Public GETs only. No 403 bypass, announcement values, new event discovery, active
calendar changes, or coverage extension. Successful evidence is written only to
the independent observer output. Callers must use load_verified_calendar rather
than accepting an artifact belonging to different template bytes or inception.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import queue
import re
import threading
import time
import unicodedata

import requests

from international_event_calendar import load_calendar, evaluate_international_event_calendar
from public_request_deadline import PublicRequestDeadline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / 'config/international_event_calendar_20261023.json'
DEFAULT_OUTPUT = ROOT / 'data/cross_market_observer'
MAX_BODY_BYTES = 2 * 1024 * 1024
TOTAL_BUDGET_SECONDS = 24
_DEADLINE = PublicRequestDeadline(timeout_seconds=23, max_pending=6)
PAGES = {
    'cpi_schedule': 'https://www.stat.go.jp/english/data/cpi/1582.htm',
    'cpi_estat': 'https://www.e-stat.go.jp/release-calendar/statistics/00200573',
    'boj': 'https://www.boj.or.jp/about/calendar/index.htm',
    'ecb_calendar': 'https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html',
    'ecb_decision': 'https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html',
    'ecb_conference': 'https://www.ecb.europa.eu/press/press_conference/html/index.en.html',
}
SOURCE_PAGES = {'japan_cpi': ('cpi_schedule', 'cpi_estat'), 'boj_calendar': ('boj',),
                'ecb_calendar': ('ecb_calendar', 'ecb_decision', 'ecb_conference')}


def _ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


EXPECTED = {
    'jp_national_cpi_202609_release_20261023': ('2026-10-23T08:30:00+09:00', 'Asia/Tokyo'),
    'jp_tokyo_cpi_202610_preliminary_20261030': ('2026-10-30T08:30:00+09:00', 'Asia/Tokyo'),
    'boj_governor_conference_20261030': ('2026-10-30T15:30:00+09:00', 'Asia/Tokyo'),
    'ecb_policy_decision_20261029': ('2026-10-29T14:15:00+01:00', 'Europe/Berlin'),
    'ecb_press_conference_20261029': ('2026-10-29T14:45:00+01:00', 'Europe/Berlin'),
}
BOJ_UNKNOWN = 'boj_policy_decision_20261030_unknown_time'


class VerificationFailure(ValueError):
    pass


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _text(body):
    parser = _Text()
    parser.feed(body.decode('utf-8-sig'))
    return ' '.join(unicodedata.normalize('NFKC', ' '.join(parser.parts)).split())


def _fetch_page(url):
    # No redirect or access-control workaround. The outer shared daemon deadline
    # bounds slow DNS/trickling bodies and prevents duplicate outstanding reads.
    with requests.get(url, timeout=(5, 8), stream=True, allow_redirects=False,
                      headers={'User-Agent': 'btc-auto-paper-calendar-verifier/1.0'}) as response:
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise VerificationFailure('source_body_too_large')
            chunks.append(chunk)
        return {'status_code': response.status_code, 'body': b''.join(chunks)}


def _fetch_pages(fetcher, fixed_asof=None):
    results, messages = {}, queue.Queue()
    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS

    def worker(name, url):
        try:
            row = _DEADLINE.run(url, lambda: fetcher(url))
            if (not isinstance(row, dict) or type(row.get('status_code')) is not int
                    or not isinstance(row.get('body'), bytes) or len(row['body']) > MAX_BODY_BYTES):
                raise VerificationFailure('source_response_invalid')
            row = {**row, 'received_at_ms': fixed_asof if fixed_asof is not None else int(time.time() * 1000)}
        except Exception as exc:
            # Never serialize exception messages: request/proxy errors can embed credentials.
            row = {'error_type': type(exc).__name__, 'status_code': None}
        messages.put((name, row))

    for name, url in PAGES.items():
        threading.Thread(target=worker, args=(name, url), daemon=True,
                         name='international-calendar-read').start()
    while len(results) < len(PAGES):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            name, row = messages.get(timeout=remaining)
            results[name] = row
        except queue.Empty:
            break
    for name in PAGES:
        results.setdefault(name, {'status_code': None, 'error_type': 'TotalDeadlineExceeded'})
    return results


def _require(text, pattern, reason):
    if not re.search(pattern, text, re.IGNORECASE):
        raise VerificationFailure(reason)


def _verify_pages(pages):
    texts = {}
    for name, row in pages.items():
        if row.get('status_code') != 200:
            raise VerificationFailure('official_page_unavailable:' + name)
        texts[name] = _text(row['body'])
    _require(texts['cpi_schedule'], r'Last Update\s*:\s*23 January 2026', 'cpi_schedule_vintage_changed')
    _require(texts['cpi_schedule'], r'September\s+October\s+23\s+October\s+October\s+30', 'cpi_dates_changed')
    _require(texts['cpi_estat'], r'2026年9月分\s+公表日\s+2026-10-23\s+08:30', 'national_cpi_time_changed')
    _require(texts['cpi_estat'], r'東京都区部.{0,160}2026年10月分\s+公表日\s+2026-10-30\s+08:30', 'tokyo_cpi_time_changed')
    _require(texts['boj'], r'2026年10月29日\s*\(木\)\s*、\s*30日\s*\(金\)', 'boj_meeting_dates_changed')
    _require(texts['boj'], r'30日\s*\(金\)\s*未定.{0,100}政策委員会・金融政策決定会合の決定内容', 'boj_decision_time_changed')
    _require(texts['boj'], r'2026年10月30日\s*\(金\)\s*15:30', 'boj_conference_time_changed')
    _require(texts['ecb_calendar'], r'29/10/2026\s+Governing Council of the ECB:\s*monetary policy meeting in Frankfurt\s*\(Day 2\),?\s*followed by press conference', 'ecb_meeting_date_changed')
    _require(texts['ecb_calendar'], r'29/10/2026\s+Press conference following the Governing Council', 'ecb_conference_date_changed')
    _require(texts['ecb_decision'], r'monetary policy decisions are published in a press release at 14:15 CET', 'ecb_decision_convention_changed')
    _require(texts['ecb_conference'], r'Press conference.{0,150}14:45 CET', 'ecb_conference_convention_changed')


def _verify_template(template, at):
    if (template.get('candidate_only') is not True or template.get('places_orders') is not False
            or set(template.get('required_sources', [])) != set(SOURCE_PAGES)):
        raise VerificationFailure('unexpected_calendar_scope')
    events = {event['event_id']: event for event in template.get('events', [])}
    if len(template.get('events', [])) != 6 or set(events) != set(EXPECTED) | {BOJ_UNKNOWN}:
        raise VerificationFailure('fixed_six_events_changed')
    for row in [template, *template.get('source_verifications', {}).values(), *events.values()]:
        clocks = [row.get(key) for key in ('first_seen_at_ms', 'available_at_ms', 'verified_at_ms')]
        if (any(type(stamp) is not int or stamp < 0 for stamp in clocks)
                or not clocks[0] <= clocks[1] <= clocks[2] <= at):
            raise VerificationFailure('template_not_yet_available')
    for row in events.values():
        if (row.get('block_new_risk') is not True or row.get('actual_at_ms') is not None
                or row.get('actual_observed_at_ms') is not None):
            raise VerificationFailure('fixed_schedule_scope_changed')
    for identity, (local, zone) in EXPECTED.items():
        row = events[identity]
        if (row.get('scheduled_at_local') != local or row.get('scheduled_at_ms') != _ms(local)
                or row.get('timezone') != zone or row.get('time_basis') != 'official_exact_time'):
            raise VerificationFailure('fixed_event_time_changed:' + identity)
    row = events[BOJ_UNKNOWN]
    if (row.get('scheduled_at_ms') is not None or row.get('time_basis') != 'conservative_full_decision_day'
            or row.get('bounded_start_ms') != _ms('2026-10-30T00:00:00+09:00')
            or row.get('bounded_end_ms') != _ms('2026-10-31T00:00:00+09:00')):
        raise VerificationFailure('boj_uncertainty_bounds_changed')
    for key in ('first_seen_at_ms', 'available_at_ms'):
        if type(template.get(key)) is not int or template[key] > at:
            raise VerificationFailure('template_not_yet_available')
    if type(template.get('coverage_end_ms')) is not int or at >= template['coverage_end_ms']:
        raise VerificationFailure('template_coverage_expired')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_verified_calendar(template_path=DEFAULT_TEMPLATE, output_dir=DEFAULT_OUTPUT):
    """Reject old runtime bytes when a template or its inception has changed."""
    raw = Path(template_path).read_bytes()
    template = load_calendar(template_path)
    current = load_calendar(Path(output_dir) / 'calendar_verified.json')
    if (current.get('template_sha256') != _sha(raw)
            or any(current.get(key) != template.get(key) for key in
                   ('calendar_id', 'first_seen_at_ms', 'available_at_ms', 'coverage_start_ms', 'coverage_end_ms'))):
        raise VerificationFailure('runtime_calendar_template_mismatch')
    return current


def refresh_calendar(template_path=DEFAULT_TEMPLATE, output_dir=DEFAULT_OUTPUT, *, asof_ms=None, fetcher=None):
    """Return safe attempt status. Failure never rewrites the old verified file.

    Production callers omit asof_ms so verification is stamped after the bounded
    GET batch. The override exists for offline fixtures, not historical backfill.
    last_attempt_at_ms / next_attempt_after_ms support six-hour success and
    one-hour failure scheduling without retrying every observer poll.
    """
    output = Path(output_dir)
    start = asof_ms if asof_ms is not None else int(time.time() * 1000)
    status = {'schema_version': 1, 'status': 'failed', 'success': False,
              'last_attempt_started_at_ms': start, 'last_attempt_at_ms': start,
              'coverage_extended': False, 'first_seen_backfilled': False, 'pages': {}}
    template = None
    try:
        if type(start) is not int or start < 0:
            raise VerificationFailure('invalid_reference_time')
        raw = Path(template_path).read_bytes()
        template = load_calendar(template_path)
        _verify_template(template, start)
        pages = _fetch_pages(fetcher or _fetch_page, fixed_asof=asof_ms)
        checked = asof_ms if asof_ms is not None else int(time.time() * 1000)
        if checked < start:
            raise VerificationFailure('verification_clock_reversed')
        status['last_attempt_at_ms'] = checked
        archive = output / 'evidence/international_calendar_sources'
        for name, response in pages.items():
            detail = {'url': PAGES[name], 'http_status': response.get('status_code'),
                      'error_type': response.get('error_type'), 'received_at_ms': response.get('received_at_ms')}
            body = response.get('body')
            if body is not None:
                content_hash = _sha(body)
                archive.mkdir(parents=True, exist_ok=True)
                path = archive / (content_hash + '.html.gz')
                if not path.exists():
                    with path.open('xb') as handle:
                        handle.write(gzip.compress(body, mtime=0))
                detail['raw_sha256'] = content_hash
            status['pages'][name] = detail
        _verify_pages(pages)
        current = deepcopy(template)
        current.update(verified_at_ms=checked, valid_until_ms=min(checked + 86_400_000, template['coverage_end_ms']),
                       template_sha256=_sha(raw), automatic_verification='fixed_official_semantics_v1')
        for source, names in SOURCE_PAGES.items():
            current['source_verifications'][source].update(
                verified_at_ms=checked, checked_at_ms=checked,
                last_success_at_ms=min(pages[name]['received_at_ms'] for name in names),
                valid_until_ms=current['valid_until_ms'], ok=True, fetch_ok=True,
                verification_method='fixed_official_semantics_v1')
        for event in current['events']:
            event['verified_at_ms'] = checked
        # Company review clocks are deliberately unchanged: these six GETs did
        # not re-read the company invitation or discover its first release.
        current.setdefault('verification_history', []).append({
            'verified_at_ms': checked, 'method': 'fixed_official_semantics_v1',
            'source_content_sha256': {name: row['raw_sha256'] for name, row in status['pages'].items()},
            'first_seen_and_coverage_preserved': True})
        outcome = evaluate_international_event_calendar(current, asof_ms=checked)
        if outcome['status'] == 'unavailable':
            raise VerificationFailure('renewed_calendar_invalid')
        _write(output / 'calendar_verified.json', current)
        status.update(status='verified', success=True, verified_at_ms=checked,
                      valid_until_ms=current['valid_until_ms'], calendar_id=current['calendar_id'],
                      template_sha256=current['template_sha256'])
    except Exception as exc:
        status['error_type'] = type(exc).__name__
        status['failure_reason'] = str(exc) if isinstance(exc, VerificationFailure) else 'verification_unavailable'
        try:
            retained = load_verified_calendar(template_path, output)
            status.update(retained_calendar_compatible=True,
                          retained_verified_at_ms=retained['verified_at_ms'],
                          retained_valid_until_ms=retained['valid_until_ms'])
        except (OSError, ValueError, TypeError, KeyError):
            status['retained_calendar_compatible'] = False
    status['next_attempt_after_ms'] = status['last_attempt_at_ms'] + (6 if status['success'] else 1) * 3_600_000
    _write(output / 'calendar_refresh_status.json', status)
    return status
