"""Local, causal NQ sector context for the main stock accounts; no network I/O.

Each account owns an append-only journal. The observation collector's original
times, identities and calendar coverage are retained; only actual ingestion by
this account receives the caller's verified decision timestamp.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import zlib

import stock_technology_guard as guard

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/cross_market_observer/latest_observations.json'
MAX_SOURCE_BYTES = 2_000_000
MAX_RESTORE_BYTES = 3_000_000
HISTORY_MS = 2 * 60 * 60_000
POLICY = deepcopy(guard.POLICY)
_CONFIG = {'technology_guard_policy': POLICY}
_ADDED_KEYS = {'ingested_at_ms', 'source_snapshot_sha256', 'record_type',
               'conflict_reason', 'conflict_sha256'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_bounded(path, limit):
    with Path(path).open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('sector_source_oversized')
    return raw


def _content(row):
    return {k: row.get(k) for k in ('value', 'provider_previous_close',
        'provider_reference_first_seen_at_ms', 'provider_reference_available_at_ms',
        'instrument', 'provider', 'source_id', 'source_config_id', 'observed_date')}


def _identity(row):
    return (row['provider'], row['source_id'], row['source_config_id'],
            _canonical(row['instrument']), row['observed_date'], row['observed_at_ms'])


def _unknown(timestamp, reason):
    return {'enabled': True, 'status': 'unknown', 'direction': None,
        'reasons': [reason], 'checked_at_ms': timestamp,
        'daily_return_pct': None, 'return_30m_pct': None,
        'latest': None, 'reference': None, 'valid_until_ms': None,
        'scope': 'sector_context', 'proxy': 'named_NQ_futures_not_full_sector_breadth',
        'protective_exits_allowed': True}


def evaluate(rows, *, timestamp, inception_ms):
    """Pure direction from causal journal rows and the frozen NQ metadata rules."""
    if (type(inception_ms) is not int or inception_ms <= 0
            or type(timestamp) is not int or timestamp < inception_ms):
        return _unknown(timestamp, 'sector_verified_clock_required')
    # A coverage review creates new, prospectively covered observations; it
    # never renews old rows. Validate old rows at their actual ingestion, then
    # omit expired coverage only from this evaluation's working list. Conflict
    # evidence remains effective for its provider date even after its original
    # coverage ends, so expiry cannot clear a known inconsistent observation.
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return _unknown(timestamp, 'immutable_journal_rows_required')
    visible = []
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError('journal_row_object_required')
            ingested = row.get('ingested_at_ms')
            if type(ingested) is int and ingested > timestamp:
                continue
            point = guard._point(row, asof_ms=ingested, policy=POLICY)
            if point['observed_at_ms'] >= inception_ms:
                visible.append((row, point))
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return _unknown(timestamp, str(exc) if isinstance(exc, ValueError) else 'sector_history_invalid')
    if visible:
        current_date = max(point['observed_date'] for _, point in visible)
        current = [(row, point) for row, point in visible if point['observed_date'] == current_date]
        if any(point['record_type'] == 'conflict' for _, point in current):
            return _unknown(timestamp, 'session_observation_conflict')
        rows = [row for row, point in current if timestamp < point['session_reference_valid_until_ms']]
    else:
        rows = []
    outcome = guard.evaluate(rows, _CONFIG, asof_ms=timestamp,
                             inception_ms=inception_ms, direction=1)
    result = _unknown(timestamp, 'sector_history_unavailable')
    result.update({k: deepcopy(outcome[k]) for k in ('reasons', 'daily_return_pct',
        'return_30m_pct', 'latest', 'reference', 'valid_until_ms')})
    daily, intraday = outcome['daily_return_pct'], outcome['return_30m_pct']
    if (outcome.get('status') in ('weak', 'clear')
            and all(type(v) in (int, float) and math.isfinite(v) for v in (daily, intraday))):
        direction = 1 if daily > 0 and intraday > 0 else -1 if daily < 0 and intraday < 0 else 0
        result.update(direction=direction, status={1: 'bullish', -1: 'bearish', 0: 'neutral'}[direction],
            reasons=[] if direction else ['sector_daily_and_30m_not_directionally_aligned'])
    return result


class SectorContext:
    """Persist new quotes only; bounded recovery retains all session conflicts.

    The small conflict journal is separate so a busy quote journal's bounded
    tail cannot forget an earlier conflict in the same provider date. A corrupt
    or oversized conflict index fails closed instead of silently resetting it.
    """

    def __init__(self, journal_path, inception_ms, source_path=None):
        if type(inception_ms) is not int or inception_ms <= 0:
            raise ValueError('sector_inception_required')
        self.path = Path(journal_path)
        self.inception_ms = inception_ms
        self.source_path = Path(source_path) if source_path is not None else SOURCE
        self.conflicts_path = self.path.with_name(self.path.stem + '_conflicts.jsonl')
        self.archive_path = self.path.parent / (self.path.stem + '_source_archive')
        self.rows = []
        self.conflicts = []
        self._loaded = False
        self._load_error = None

    def _read_rows(self, path, *, tail):
        if not path.exists():
            return []
        with path.open('rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if not tail and size > MAX_RESTORE_BYTES:
                raise ValueError('sector_conflict_index_oversized')
            offset = max(0, size - MAX_RESTORE_BYTES) if tail else 0
            handle.seek(offset)
            raw = handle.read(MAX_RESTORE_BYTES + 1)
        if raw and not raw.endswith(b'\n'):
            raise ValueError('sector_journal_incomplete_tail')
        lines = raw.splitlines()
        if offset:
            # Starting in a complete line also conservatively drops that first
            # line; two hours of history are far smaller than this bounded tail.
            lines = lines[1:]
        rows = []
        for line in lines:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('sector_journal_row_invalid')
            self._verify_row(row)
            rows.append(row)
        return rows

    def _verify_row(self, row):
        ingested = row.get('ingested_at_ms')
        guard._point(row, asof_ms=ingested, policy=POLICY)
        if row['observed_at_ms'] < self.inception_ms or ingested < self.inception_ms:
            raise ValueError('sector_journal_preinception_quote')
        sha = row['source_snapshot_sha256']
        # Check every restored row against its immutable original source. A
        # hash alone cannot authenticate edited quote fields in a JSONL row.
        archive = self.archive_path / (sha + '.json.gz')
        if archive.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError('sector_source_archive_oversized')
        with gzip.open(archive, 'rb') as handle:
            raw = handle.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES or _hash(raw) != sha:
            raise ValueError('sector_source_archive_hash_mismatch')
        source = json.loads(raw)
        self._validate_source_time(source, ingested)
        original = source['observations']['nq_dec_2026']
        if _canonical({k: v for k, v in row.items() if k not in _ADDED_KEYS}) != _canonical(original):
            raise ValueError('sector_journal_source_mismatch')
        if row.get('record_type', 'observation') not in ('observation', 'conflict'):
            raise ValueError('sector_journal_record_type_invalid')

    @staticmethod
    def _validate_source_time(source, timestamp):
        if not isinstance(source, Mapping):
            raise ValueError('sector_observer_snapshot_object_required')
        observations = source.get('observations')
        if not isinstance(observations, Mapping):
            raise ValueError('sector_observer_observations_object_required')
        row = observations.get('nq_dec_2026')
        if not isinstance(row, Mapping):
            raise ValueError('sector_observer_quote_object_required')
        generated = source.get('generated_at_ms')
        if type(generated) is not int or not 0 <= timestamp - generated <= 20 * 60_000:
            raise ValueError('sector_observer_snapshot_stale_or_future')
        if any(type(row.get(k)) is not int or row[k] > generated for k in
               ('observed_at_ms', 'first_seen_at_ms', 'available_at_ms', 'received_at_ms')):
            raise ValueError('sector_snapshot_before_original_receipt')
        session = row.get('market_session')
        if not isinstance(session, Mapping) or type(session.get('asof_ms')) is not int:
            raise ValueError('sector_snapshot_session_reference_required')
        if session['asof_ms'] > generated:
            raise ValueError('sector_snapshot_before_session_reference')

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            self.rows = self._read_rows(self.path, tail=True)
            self.conflicts = self._read_rows(self.conflicts_path, tail=False)
            # A legacy/corrupt side index cannot silently forget a marker that
            # is still visible in the primary immutable journal.
            index = {r.get('conflict_sha256') for r in self.conflicts}
            if any(r.get('record_type') == 'conflict' and r.get('conflict_sha256') not in index
                   for r in self.rows):
                raise ValueError('sector_conflict_index_incomplete')
        except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, OverflowError) as exc:
            self._load_error = str(exc) if isinstance(exc, ValueError) else 'sector_journal_recovery_failed'

    def _append(self, path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab') as handle:
            handle.write(_canonical(row) + b'\n')

    def _archive(self, sha, raw):
        self.archive_path.mkdir(parents=True, exist_ok=True)
        target = self.archive_path / (sha + '.json.gz')
        if not target.exists():
            # Exclusive creation keeps an existing archive immutable.
            try:
                with target.open('xb') as handle:
                    handle.write(gzip.compress(raw, mtime=0))
            except FileExistsError:
                pass
        # Verify even an already-existing archive before referring to it.
        if target.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError('sector_source_archive_oversized')
        with gzip.open(target, 'rb') as handle:
            existing = handle.read(MAX_SOURCE_BYTES + 1)
        if len(existing) > MAX_SOURCE_BYTES or _hash(existing) != sha:
            raise ValueError('sector_source_archive_hash_mismatch')

    def decision(self, timestamp):
        if type(timestamp) is not int or timestamp < self.inception_ms:
            return _unknown(timestamp, 'sector_verified_clock_required')
        self._load()
        if self._load_error:
            return _unknown(timestamp, self._load_error)
        try:
            raw = _read_bounded(self.source_path, MAX_SOURCE_BYTES)
            source = json.loads(raw)
            self._validate_source_time(source, timestamp)
            row = deepcopy(source['observations']['nq_dec_2026'])
            if row.get('observed_at_ms', 0) < self.inception_ms:
                raise ValueError('sector_post_inception_quote_required')
            sha = _hash(raw)
            row.update(ingested_at_ms=timestamp, source_snapshot_sha256=sha)
            guard._point(row, asof_ms=timestamp, policy=POLICY)
            if timestamp - row['observed_at_ms'] > POLICY['maximum_quote_age_minutes'] * 60_000:
                raise ValueError('sector_latest_observation_stale')
            if any(r['ingested_at_ms'] > timestamp for r in self.rows + self.conflicts):
                raise ValueError('sector_decision_clock_regressed')
            dates = [r['observed_date'] for r in self.rows + self.conflicts]
            if dates and row['observed_date'] < max(dates):
                raise ValueError('sector_observation_date_regressed')
            self.rows = [r for r in self.rows if r['observed_date'] == row['observed_date']
                         and r['observed_at_ms'] >= row['observed_at_ms'] - HISTORY_MS]
            current_conflicts = [r for r in self.conflicts if r['observed_date'] == row['observed_date']]
            key = _identity(row)
            previous = next((r for r in self.rows if _identity(r) == key), None)
            if previous is not None and _content(previous) != _content(row):
                row.update(record_type='conflict', conflict_reason='same_observation_changed')
                row['conflict_sha256'] = _hash(_canonical({'identity': list(key[:3]) + [key[3].decode('utf-8')] + list(key[4:]),
                                                          'content': _content(row)}))
                if not any(r.get('conflict_sha256') == row['conflict_sha256'] for r in self.conflicts):
                    self._archive(sha, raw)
                    # Persist the safety index first; an interrupted primary
                    # append then remains safely unknown after recovery.
                    self._append(self.conflicts_path, row)
                    self.conflicts.append(row)
                    current_conflicts.append(row)
                    self._append(self.path, row)
            elif previous is None:
                self._archive(sha, raw)
                self._append(self.path, row)
                self.rows.append(row)
            result = evaluate(self.rows + current_conflicts,
                              timestamp=timestamp, inception_ms=self.inception_ms)
            result['inception_ms'] = self.inception_ms
            result['source_snapshot_sha256'] = sha
            return result
        except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, OverflowError) as exc:
            return _unknown(timestamp, str(exc) if isinstance(exc, ValueError) else 'sector_source_unavailable_or_invalid')
