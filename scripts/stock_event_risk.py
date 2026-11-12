"""Causal, opt-in earnings calendar admission for new stock paper risk.

No network, order, ledger, or exit side effects. Read the calendar again for each
retry and immediately before booking an entry; a cached initial signal approval
is never an entry permit. A missing/expired calendar blocks new risk only.
An explicit, dated MU simulation waiver can skip an unverified source review;
it never relabels the source as verified or suppresses a known event window.
An explicitly removed MU source needs no periodic simulation waiver. Its empty
event coverage is not a claim that no events exist; other known events still gate.

The v1 anchor is the OFFICIAL SCHEDULED CALL START, not an invented first release
or call end. Unknown release/end times stay explicit. Calendar review must cover
an earlier first release; verified changes create a new causal snapshot.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import time


HOUR = 3_600_000
MODE = 'official_earnings_blackout_v1'
POLICY = {'mode': MODE, 'pre_event_hours': 24, 'post_event_hours': 2,
          'minimum_complete_hour_bars': 2, 'calendar_failure_action': 'block_new_risk'}
POLICY_KEY = 'stock_event_risk_policy'
ANCHOR = 'official_earnings_call_start'
MAX_SNAPSHOT_AGE_MS = 14 * 24 * HOUR


def enabled(config):
    return POLICY_KEY in config


def _time(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name)
    return value


def validate(config):
    """Only the fixed v1 experiment is accepted; an absent policy opts out."""
    if not enabled(config):
        return
    policy = config[POLICY_KEY]
    if not isinstance(policy, Mapping) or set(policy) != set(POLICY) | {
            'calendar_path', 'effective_from_ms'}:
        raise ValueError('stock_event_policy_invalid')
    for key, expected in POLICY.items():
        if type(policy[key]) is not type(expected) or policy[key] != expected:
            raise ValueError('stock_event_policy_invalid')
    if not isinstance(policy['calendar_path'], str) or not policy['calendar_path'].strip():
        raise ValueError('stock_event_calendar_path_invalid')
    _time(policy['effective_from_ms'], 'stock_event_effective_time_invalid')


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name)
    return value


def _removed_source(review, symbol, reference_ms, *, simulation):
    """A causal, MU-only removal; never grant the same exception to live orders."""
    if not isinstance(review, Mapping) or review.get('status') != 'source_removed':
        return {}
    error = 'stock_event_source_removal_invalid'
    if (not simulation or symbol != 'MUUSDT'
            or set(review) != {'status', 'verified_at_ms', 'source_urls', 'source_removal'}
            or review['source_urls'] != []):
        raise ValueError(error)
    removal = review['source_removal']
    if (not isinstance(removal, Mapping) or set(removal) != {
            'mode', 'first_seen_at_ms', 'available_at_ms', 'effective_from_ms',
            'user_authorization'} or removal['mode'] != 'remove_unavailable_source_for_simulation'):
        raise ValueError(error)
    first, available, effective = (_time(removal[k], error) for k in (
        'first_seen_at_ms', 'available_at_ms', 'effective_from_ms'))
    if not _time(review['verified_at_ms'], error) <= first <= available <= effective:
        raise ValueError(error)
    if reference_ms < effective:
        raise ValueError('stock_event_source_removal_not_yet_active')
    authorization = _nonempty(removal['user_authorization'], error)
    return {'source_removed': True, 'data_quality': 'source_removed',
            'source_removal_effective_from_ms': effective,
            'source_removal_reason': authorization,
            'review_due_reason': 'Unavailable MU source removed by user; no periodic '
                                 'waiver required. Other known event windows remain enforced.'}


def _calendar(raw, reference_ms, symbol, *, allow_simulation_waiver=False):
    if not isinstance(raw, Mapping) or type(raw.get('version')) is not int or raw['version'] != 1:
        raise ValueError('stock_event_calendar_invalid')
    calendar_id = _nonempty(raw.get('calendar_id'), 'stock_event_calendar_invalid')
    times = {key: _time(raw.get(key), 'stock_event_calendar_time_invalid') for key in (
        'first_seen_at_ms', 'available_at_ms', 'verified_at_ms', 'valid_until_ms',
        'coverage_start_ms', 'coverage_end_ms')}
    if not (times['first_seen_at_ms'] <= times['available_at_ms'] <= times['verified_at_ms']
            < times['valid_until_ms'] <= times['coverage_end_ms']
            and times['coverage_start_ms'] <= times['available_at_ms']
            and times['coverage_start_ms'] < times['coverage_end_ms']
            and times['valid_until_ms'] - times['verified_at_ms'] <= MAX_SNAPSHOT_AGE_MS):
        raise ValueError('stock_event_calendar_time_invalid')
    # Do not disclose or apply events from a snapshot first observed in the future.
    if reference_ms < times['available_at_ms'] or reference_ms < times['verified_at_ms']:
        raise ValueError('stock_event_calendar_not_yet_available')
    coverage = raw.get('symbol_reviews')
    candidate = coverage.get(symbol) if isinstance(coverage, Mapping) else None
    removed = _removed_source(candidate, symbol, reference_ms, simulation=allow_simulation_waiver)
    if not removed:
        if reference_ms >= times['valid_until_ms']:
            raise ValueError('stock_event_calendar_expired')
        if not times['coverage_start_ms'] <= reference_ms < times['coverage_end_ms']:
            raise ValueError('stock_event_calendar_outside_coverage')
    if not isinstance(coverage, Mapping) or symbol not in coverage:
        raise ValueError('stock_event_symbol_not_covered')
    review = coverage[symbol]
    if (not isinstance(review, Mapping)
            or review.get('status') not in ('official_event_verified', 'no_announced_event_found',
                                             'latest_review_unverified', 'source_removed')
            or _time(review.get('verified_at_ms'), 'stock_event_symbol_review_invalid') > times['verified_at_ms']
            or not isinstance(review.get('source_urls'), list)
            or (not review['source_urls'] and not removed)
            or any(not isinstance(url, str) or not url.startswith('https://') for url in review['source_urls'])):
        raise ValueError('stock_event_symbol_review_invalid')
    if reference_ms < review['verified_at_ms']:
        raise ValueError('stock_event_symbol_review_not_yet_available')
    waiver_metadata = {}
    if review['status'] == 'latest_review_unverified':
        waiver = review.get('simulation_unverified_review_waiver')
        if not (allow_simulation_waiver and symbol == 'MUUSDT' and isinstance(waiver, Mapping)):
            raise ValueError('stock_event_symbol_review_invalid')
        expected = {'mode', 'first_seen_at_ms', 'available_at_ms', 'effective_from_ms',
                    'valid_until_ms', 'user_authorization'}
        if (set(waiver) != expected
                or waiver.get('mode') != 'ignore_unverified_review_for_simulation'):
            raise ValueError('stock_event_review_waiver_invalid')
        clocks = {key: _time(waiver[key], 'stock_event_review_waiver_invalid') for key in (
            'first_seen_at_ms', 'available_at_ms', 'effective_from_ms', 'valid_until_ms')}
        if not (times['verified_at_ms'] <= clocks['first_seen_at_ms'] <= clocks['available_at_ms']
                <= clocks['effective_from_ms'] < clocks['valid_until_ms'] <= times['valid_until_ms']):
            raise ValueError('stock_event_review_waiver_invalid')
        authorization = _nonempty(waiver['user_authorization'], 'stock_event_review_waiver_invalid')
        if not clocks['effective_from_ms'] <= reference_ms < clocks['valid_until_ms']:
            raise ValueError('stock_event_symbol_review_invalid')
        waiver_metadata = {'review_waiver_applied': True, 'data_quality': 'unverified',
                          'review_waiver_reason': authorization,
                          'review_waiver_effective_from_ms': clocks['effective_from_ms'],
                          'review_waiver_valid_until_ms': clocks['valid_until_ms'],
                          'review_due_reason': 'MU source remains unverified; the active user-authorized '
                                               'simulation waiver skips only that review blocker. '
                                               'Known event windows remain enforced.'}
    rows = raw.get('events')
    if not isinstance(rows, list):
        raise ValueError('stock_event_calendar_invalid')
    events, ids = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('stock_event_record_invalid')
        event_id = _nonempty(row.get('event_id'), 'stock_event_record_invalid')
        if event_id in ids:
            raise ValueError('stock_event_duplicate_id')
        ids.add(event_id)
        symbols = row.get('symbols')
        if (not isinstance(symbols, list) or not symbols
                or any(not isinstance(item, str) or item not in coverage for item in symbols)):
            raise ValueError('stock_event_record_invalid')
        stamp = _time(row.get('anchor_time_ms'), 'stock_event_anchor_invalid')
        first_seen = _time(row.get('first_seen_at_ms'), 'stock_event_record_time_invalid')
        available = _time(row.get('available_at_ms'), 'stock_event_record_time_invalid')
        verified = _time(row.get('verified_at_ms'), 'stock_event_record_time_invalid')
        if not (times['first_seen_at_ms'] <= first_seen <= available <= verified <= times['verified_at_ms']
                and times['coverage_start_ms'] <= stamp < times['coverage_end_ms']):
            raise ValueError('stock_event_record_time_invalid')
        if (row.get('anchor_kind') != ANCHOR or row.get('first_release_time_ms') is not None
                or row.get('first_release_time_status') != 'unknown'
                or row.get('call_end_time_ms') is not None
                or row.get('call_end_time_status') != 'unknown'
                or not isinstance(row.get('source_url'), str)
                or not row['source_url'].startswith('https://')):
            raise ValueError('stock_event_record_invalid')
        _nonempty(row.get('source_title'), 'stock_event_record_invalid')
        _nonempty(row.get('source_published_date'), 'stock_event_record_invalid')
        if symbol in symbols and available <= reference_ms and verified <= reference_ms:
            events.append(dict(row))
    if review['status'] == 'official_event_verified' and not events:
        raise ValueError('stock_event_symbol_event_missing')
    if removed and events:
        # Removal of one unavailable list cannot discard known events from
        # other sources, or refresh their calendar coverage after expiry.
        if reference_ms >= times['valid_until_ms']:
            raise ValueError('stock_event_calendar_expired')
        if not times['coverage_start_ms'] <= reference_ms < times['coverage_end_ms']:
            raise ValueError('stock_event_calendar_outside_coverage')
    return {'calendar_id': calendar_id, **times, 'symbol_review_status': review['status'],
            **waiver_metadata, **removed, 'calendar_required': not removed or bool(events)}, events


def _post_event_bars(rows, anchor, reference_ms):
    """Require two actual, consecutive, fully closed UTC hours after the anchor.

    For a 04:30 start, 04:00--05:00 is excluded. The earliest eligible pair is
    05:00--06:00 and 06:00--07:00. Future or malformed candles never qualify.
    """
    if rows is None:
        return False, []
    starts = []
    try:
        for row in rows:
            if isinstance(row, Mapping):
                start = row['time_ms']
                close_time = start + HOUR - 1
            elif hasattr(row, 'time_ms'):
                start = row.time_ms
                close_time = start + HOUR - 1
            else:
                start, close_time = row[0], row[6]
            _time(start, 'stock_event_hourly_data_invalid')
            _time(close_time, 'stock_event_hourly_data_invalid')
            if start % HOUR or close_time != start + HOUR - 1:
                return False, []
            if start >= anchor and start + HOUR <= reference_ms:
                starts.append(start)
        if len(starts) < 2 or any(b <= a for a, b in zip(starts, starts[1:])):
            return False, starts[-2:]
        pair = starts[-2:]
        return pair[1] - pair[0] == HOUR, pair
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return False, []


def evaluate(config, symbol, reference_ms, *, closed_hourly_bars=None, root=None,
             reference_clock=None):
    """Return entry-only admission, with causal provenance and source freshness.

    effective_from_ms is mandatory and prevents historical tests from inheriting
    a calendar learned today. Failures remain data in the result, so independent
    stop/reduction handling can continue even with an invalid policy/file.
    """
    result = {'enabled': enabled(config), 'allowed': True, 'reasons': [], 'status': 'disabled',
              'event_ids': [], 'checked_at_ms': reference_ms,
              'scope': 'new_risk_only', 'protective_exits_allowed': True}
    if not result['enabled']:
        return result
    try:
        validate(config)
        reference_ms = _time(reference_ms, 'stock_event_reference_time_invalid')
        policy = config[POLICY_KEY]
        result.update(effective_from_ms=policy['effective_from_ms'],
                      calendar_path=policy['calendar_path'])
        if reference_ms < policy['effective_from_ms']:
            return {**result, 'status': 'not_active_yet'}
        path = Path(policy['calendar_path'])
        if not path.is_absolute():
            path = (Path(root) if root is not None else Path(__file__).resolve().parents[1]) / path
        payload = path.read_bytes()
        if len(payload) > 1_000_000:
            raise ValueError('stock_event_calendar_too_large')
        result['calendar_sha256'] = hashlib.sha256(payload).hexdigest()
        raw = json.loads(payload.decode('utf-8'))
        if reference_clock is not None:
            # Disk/decoding can itself cross a blackout or expiry boundary.
            reference_ms = _time(reference_clock(), 'stock_event_reference_time_invalid')
            result['checked_at_ms'] = reference_ms
        # Keep an expired coverage deadline visible even when admission fails.
        # Never expose metadata from a snapshot unavailable at this causal time.
        if (isinstance(raw, Mapping) and type(raw.get('available_at_ms')) is int
                and raw['available_at_ms'] <= reference_ms
                and type(raw.get('valid_until_ms')) is int
                and type(raw.get('coverage_end_ms')) is int):
            result.update(valid_until_ms=raw['valid_until_ms'], coverage_end_ms=raw['coverage_end_ms'],
                          calendar_remaining_ms=raw['valid_until_ms'] - reference_ms,
                          review_due_reason=raw.get('review_due_reason'),
                          fetch_attempt_at_ms=raw.get('fetch_attempt_at_ms'),
                          source_fetch_verification=raw.get('source_fetch_verification'))
        metadata, events = _calendar(raw, reference_ms, symbol,
                                    allow_simulation_waiver=config.get('live_orders_allowed') is False)
        result.update(metadata, calendar_age_ms=reference_ms - metadata['verified_at_ms'],
                      calendar_remaining_ms=metadata['valid_until_ms'] - reference_ms,
                      fetch_success_at_ms=raw.get('fetch_success_at_ms'),
                      known_event_count=len(events), status='clear')
        blocked, reasons = [], []
        upcoming = []
        for event in events:
            anchor = event['anchor_time_ms']
            start = anchor - 24 * HOUR
            earliest = max(anchor + 2 * HOUR, ((anchor + HOUR - 1) // HOUR + 2) * HOUR)
            detail = {'event_id': event['event_id'], 'anchor_time_ms': anchor,
                      'anchor_kind': event['anchor_kind'], 'blackout_start_ms': start,
                      'earliest_resume_at_ms': earliest, 'first_release_time_status': 'unknown',
                      'call_end_time_status': 'unknown', 'source_url': event['source_url'],
                      'source_published_date': event['source_published_date'],
                      'first_seen_at_ms': event['first_seen_at_ms'],
                      'available_at_ms': event['available_at_ms'],
                      'verified_at_ms': event['verified_at_ms']}
            if anchor >= reference_ms:
                upcoming.append(detail)
            if reference_ms < start:
                continue
            if reference_ms < earliest:
                blocked.append(detail)
                reasons.append('stock_event_blackout')
            else:
                bars_ok, pair = _post_event_bars(closed_hourly_bars, anchor, reference_ms)
                if not bars_ok:
                    blocked.append({**detail, 'observed_post_event_hour_starts_ms': pair})
                    reasons.append('stock_event_post_event_bars_unavailable')
        status = ('blackout' if reasons else 'source_removed' if result.get('source_removed') else 'unverified_review_ignored'
                  if result.get('review_waiver_applied') else 'clear')
        result.update(allowed=not reasons, status=status,
                      reasons=list(dict.fromkeys(reasons)), blocked_events=blocked,
                      event_ids=[event['event_id'] for event in blocked],
                      next_event=min(upcoming, key=lambda item: item['anchor_time_ms']) if upcoming else None,
                      release_time_caveat='call_start_anchor_first_release_and_call_end_unknown')
        return result
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
        reason = str(exc) if str(exc).startswith('stock_event_') else 'stock_event_calendar_invalid'
    except OSError:
        reason = 'stock_event_calendar_unavailable'
    return {**result, 'allowed': False, 'status': 'unavailable', 'reasons': [reason]}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def verify_official_notices(calendar, *, fetch=None, strict=True):
    """Re-read pinned evidence without inventing a new publication/first release.

    This verifies known call notices, not the completeness of future company
    announcements. It never changes an event time or extends coverage by itself.
    """
    if fetch is None:
        import requests

        def fetch(url):
            response = requests.get(url, timeout=(5, 15), headers={
                'User-Agent': 'btc-auto-paper-calendar-verifier/1.0 public-source-reader'})
            response.raise_for_status()
            return response.content

    expected = {
        'skhynix_2026q3_call_20261028': (1793145600000,
            r'October\s+28,?\s+2026\s+9:00\s*a\.?m\.?\s*\(Seoul\s+Time\)'),
        'sandisk_fy2027q1_call_20261029': (1793305800000,
            r'October\s+29,?\s+2026,?\s+at\s+1:30\s*p\.?m\.?\s+Pacific\s+Time'),
    }
    outcomes = []
    for event in calendar['events']:
        event_id = event['event_id']
        if event_id not in expected or event['anchor_time_ms'] != expected[event_id][0]:
            raise ValueError('stock_event_source_verifier_unsupported_event')
        source = event['source_url']
        outcome = {'event_id': event_id, 'source_url': source,
                   'source_published_date': event['source_published_date'],
                   'anchor_time_ms': event['anchor_time_ms']}
        try:
            payload = fetch(source)
            if payload.startswith(b'%PDF'):
                from pypdf import PdfReader
                content = ' '.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(payload)).pages)
            else:
                parser = _Text()
                parser.feed(payload.decode('utf-8'))
                content = ' '.join(parser.parts)
            text = re.sub(r'\s+', ' ', content.replace('\xa0', ' ')).strip()
            if not re.search(expected[event_id][1], text, re.IGNORECASE):
                raise ValueError('stock_event_official_anchor_not_confirmed:' + event_id)
            outcome.update(source_sha256=hashlib.sha256(payload).hexdigest(),
                           known_call_start_confirmed=True)
        except Exception as exc:
            if strict:
                raise
            # Public URLs contain no credentials; preserve a bounded diagnostic.
            outcome.update(known_call_start_confirmed=False,
                           error=type(exc).__name__ + ': ' + str(exc)[:200])
        outcomes.append(outcome)
    return outcomes


def main(argv=None):
    """Independent local status / repeatable public-source verification command."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calendar', default='config/stock_event_calendar_20261020.json')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-official', action='store_true', help='Read official known-call notices; do not write')
    action.add_argument('--refresh-official', action='store_true',
        help='Verify known-call notices, archive prior bytes, update fetch metadata; never extend coverage')
    parser.add_argument('--at-ms', type=int, help='Read-only causal evaluation time; refresh always uses current time')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    path = Path(args.calendar)
    if not path.is_absolute():
        path = root / path
    now = int(time.time() * 1000)
    output = {'mode': 'simulation', 'places_orders': False, 'checked_at_ms': now,
              'calendar_path': str(path)}
    try:
        old = path.read_bytes()
        calendar = json.loads(old.decode('utf-8'))
        if args.verify_official or args.refresh_official:
            verified = verify_official_notices(calendar, strict=False)
            output['official_notice_verification'] = verified
            output['verification_scope'] = 'known_call_notices_only_not_complete_new_announcement_discovery'
            output['all_known_notices_confirmed'] = all(
                item['known_call_start_confirmed'] for item in verified)
            if args.refresh_official:
                if args.at_ms is not None:
                    raise ValueError('stock_event_refresh_cannot_backdate')
                # The original immutable publication and first observation never move.
                # Re-reading an old notice does not prove no new announcement exists.
                calendar['fetch_attempt_at_ms'] = now
                if output['all_known_notices_confirmed']:
                    calendar['fetch_success_at_ms'] = now
                calendar['source_fetch_verification'] = verified
                archive = path.parent / (path.stem + '_history')
                archive.mkdir(parents=True, exist_ok=True)
                prior = archive / (hashlib.sha256(old).hexdigest() + '.json')
                if not prior.exists():
                    prior.write_bytes(old)
                content = json.dumps(calendar, ensure_ascii=False, indent=2) + '\n'
                temp = path.with_suffix(path.suffix + '.tmp')
                temp.write_text(content, encoding='utf-8')
                temp.replace(path)
                output['refreshed'] = True
        reference = now if args.at_ms is None else args.at_ms
        policy = {**POLICY, 'calendar_path': str(path),
                  'effective_from_ms': calendar.get('available_at_ms', now)}
        cfg = {POLICY_KEY: policy}
        output['symbols'] = {symbol: evaluate(cfg, symbol, reference, root=root)
                             for symbol in calendar.get('symbol_reviews', {})}
        output['coverage_end_ms'] = calendar.get('coverage_end_ms')
        output['coverage_end_utc'] = datetime.fromtimestamp(
            calendar['coverage_end_ms'] / 1000, timezone.utc).isoformat()
        output['review_due_reason'] = calendar.get('review_due_reason')
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1 if (output.get('all_known_notices_confirmed') is False or not output['symbols']
            or any(item['status'] == 'unavailable' for item in output['symbols'].values())) else 0
    except Exception as exc:
        output.update(error=type(exc).__name__ + ': ' + str(exc)[:200], refreshed=False)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
