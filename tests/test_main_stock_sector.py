"""Local quote journal recovery and causal sector context; synthetic metadata."""
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import main_stock_sector as sector

M = 60000
EPOCH = int(datetime.fromisoformat('2026-11-02T14:00:00+00:00').timestamp() * 1000)


def point(observed, value, previous=30000, *, lag=10*M):
    received = observed + lag
    policy = sector.POLICY
    return {'observed_at_ms': observed, 'first_seen_at_ms': received,
        'available_at_ms': received, 'received_at_ms': received,
        **{k: policy[k] for k in ('source_id', 'source_config_id', 'provider',
            'source_url', 'unit', 'observation_timezone', 'contract_review_cutoff_at_ms')},
        'instrument': {**{k: policy[k] for k in ('symbol', 'contract_month', 'exchange', 'currency')},
            'automatic_roll': False, 'continuous_contract': False},
        'precision': 'timestamp', 'eligible_intraday': True,
        'market_session': {'phase': 'open', 'is_actual_market_status': False,
            'asof_ms': received, 'valid_until_ms': EPOCH + 24 * 60 * M},
        'observed_date': datetime.fromtimestamp(observed / 1000,
            sector.guard.ZoneInfo('America/New_York')).date().isoformat(),
        'value': value, 'provider_previous_close': previous,
        'return_basis': 'same_named_instrument_provider_previousClose',
        'provider_reference_first_seen_at_ms': EPOCH - 60 * M,
        'provider_reference_available_at_ms': EPOCH - 60 * M}


def write_source(path, row, *, generated=None):
    generated = row['received_at_ms'] if generated is None else generated
    source = {'generated_at_ms': generated, 'observations': {'nq_dec_2026': deepcopy(row)}}
    raw = json.dumps(source, sort_keys=True).encode()
    path.write_bytes(raw)
    return source, raw


def context(tmp_path, *, inception=EPOCH):
    source = tmp_path / 'latest.json'
    return sector.SectorContext(tmp_path / 'main_sector_quotes.jsonl', inception, source), source


def ingest_pair(ctx, source, first=30100, latest=30200, *, gap=30, previous=30000):
    old = point(EPOCH + (60-gap) * M, first, previous)
    write_source(source, old)
    ctx.decision(old['received_at_ms'] + 1)
    new = point(EPOCH + 60 * M, latest, previous)
    write_source(source, new)
    return ctx.decision(new['received_at_ms'] + 1)


@pytest.mark.parametrize('first,latest,status,direction', [
    (30100, 30200, 'bullish', 1), (29900, 29800, 'bearish', -1),
    (29800, 29900, 'neutral', 0), (30200, 30100, 'neutral', 0),
    (30000, 30000, 'neutral', 0), (30100, 30000, 'neutral', 0)])
def test_direction_requires_daily_and_real_thirty_minute_alignment(tmp_path, first, latest, status, direction):
    ctx, source = context(tmp_path)
    result = ingest_pair(ctx, source, first, latest)
    assert result['status'] == status and result['direction'] == direction
    assert result['reference']['value'] == first and result['latest']['value'] == latest
    assert result['valid_until_ms'] == EPOCH + 80 * M
    assert result['protective_exits_allowed']


@pytest.mark.parametrize('gap,known', [(29, False), (30, True), (35, True), (36, False)])
def test_only_thirty_to_thirty_five_minute_actual_references_are_accepted(tmp_path, gap, known):
    ctx, source = context(tmp_path)
    result = ingest_pair(ctx, source, gap=gap)
    assert (result['direction'] == 1) == known
    if not known:
        assert result['status'] == 'unknown' and result['direction'] is None


def test_activation_prevents_importing_old_quotes_or_manufacturing_warmup(tmp_path):
    ctx, source = context(tmp_path)
    row = point(EPOCH - M, 30100)
    write_source(source, row)
    result = ctx.decision(row['received_at_ms'])
    assert result['reasons'] == ['sector_post_inception_quote_required']
    assert not ctx.path.exists()
    row = point(EPOCH + 5 * M, 30100)
    write_source(source, row)
    assert ctx.decision(row['received_at_ms'])['reasons'] == ['trial_history_warming_up']
    later = ctx.decision(EPOCH + 31 * M)
    assert later['direction'] is None


def test_duplicate_never_renews_original_quote_or_ingestion_time(tmp_path):
    ctx, source = context(tmp_path)
    result = ingest_pair(ctx, source)
    original = ctx.path.read_bytes()
    ingested = result['latest']['ingested_at_ms']
    # A new collector wrapper and session asof do not create a new quote.
    row = point(EPOCH + 60 * M, 30200)
    row['received_at_ms'] += M
    row['market_session']['asof_ms'] += M
    write_source(source, row, generated=EPOCH + 71 * M)
    duplicate = ctx.decision(EPOCH + 71 * M + 1)
    assert ctx.path.read_bytes() == original
    assert duplicate['latest']['ingested_at_ms'] == ingested
    assert duplicate['latest']['received_at_ms'] == EPOCH + 70 * M
    assert duplicate['valid_until_ms'] == EPOCH + 80 * M
    stale = ctx.decision(EPOCH + 80 * M + 1)
    assert stale['status'] == 'unknown' and stale['direction'] is None


def test_missing_source_does_not_turn_cached_history_into_current_context(tmp_path):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    source.unlink()
    result = ctx.decision(EPOCH + 71 * M)
    assert result['status'] == 'unknown' and result['direction'] is None


@pytest.mark.parametrize('mutation', ['generated_future', 'original_future', 'session_future',
    'source_stale', 'identity', 'closed', 'expired_coverage', 'ineligible', 'nan'])
def test_bad_original_metadata_or_clock_never_grants_a_direction(tmp_path, mutation):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    row = point(EPOCH + 61 * M, 30300)
    stamp, generated = EPOCH + 71 * M, EPOCH + 71 * M
    if mutation == 'generated_future': generated += 1
    elif mutation == 'original_future': row['received_at_ms'] = stamp + 1
    elif mutation == 'session_future': row['market_session']['asof_ms'] = stamp + 1
    elif mutation == 'source_stale': generated -= 21 * M
    elif mutation == 'identity': row['source_config_id'] = 'b' * 64
    elif mutation == 'closed': row['market_session']['phase'] = 'closed'
    elif mutation == 'expired_coverage': row['market_session']['valid_until_ms'] = stamp
    elif mutation == 'ineligible': row['eligible_intraday'] = False
    else: row['value'] = float('nan')
    write_source(source, row, generated=generated)
    result = ctx.decision(stamp)
    assert result['status'] == 'unknown' and result['direction'] is None


def test_session_asof_equal_to_generated_is_valid_even_with_later_ingestion(tmp_path):
    ctx, source = context(tmp_path)
    result = ingest_pair(ctx, source)
    assert result['direction'] == 1
    # Actual current collector snapshots use exactly this equality. Source
    # generation may be a few milliseconds after quote receipt.
    row = point(EPOCH + 61 * M, 30300)
    generated = row['received_at_ms'] + 5
    row['market_session']['asof_ms'] = generated
    write_source(source, row, generated=generated)
    assert ctx.decision(generated + 1)['direction'] == 1


def test_guard_unknown_cannot_be_promoted_from_residual_numeric_returns(monkeypatch):
    def unknown(*_a, **_k):
        return {'status': 'unknown', 'reasons': ['metadata_invalid'],
            'daily_return_pct': 1, 'return_30m_pct': 1,
            'latest': None, 'reference': None, 'valid_until_ms': None}
    monkeypatch.setattr(sector.guard, 'evaluate', unknown)
    result = sector.evaluate([], timestamp=EPOCH, inception_ms=EPOCH)
    assert result['status'] == 'unknown' and result['direction'] is None
    assert result['reasons'] == ['metadata_invalid']


def test_conflicting_same_observation_is_immutable_and_persists_after_recovery(tmp_path):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    original = ctx.path.read_bytes()
    changed = point(EPOCH + 60 * M, 30300)
    write_source(source, changed)
    outcome = ctx.decision(EPOCH + 70 * M + 2)
    assert outcome['status'] == 'unknown' and outcome['reasons'] == ['session_observation_conflict']
    assert ctx.path.read_bytes().startswith(original)
    assert ctx.conflicts_path.exists()
    recovered = sector.SectorContext(ctx.path, EPOCH, source)
    assert recovered.decision(EPOCH + 70 * M + 3)['reasons'] == ['session_observation_conflict']
    # Even when the early conflicting quote leaves the two-hour working
    # history, its marker remains effective for the same provider date.
    changed = point(EPOCH + 200 * M, 30400)
    write_source(source, changed)
    assert recovered.decision(EPOCH + 210 * M + 1)['reasons'] == ['session_observation_conflict']


def test_valid_recovery_preserves_receipt_and_source_archives(tmp_path):
    ctx, source = context(tmp_path)
    result = ingest_pair(ctx, source)
    before = ctx.path.read_bytes()
    recovered = sector.SectorContext(ctx.path, EPOCH, source)
    restored = recovered.decision(EPOCH + 70 * M + 2)
    assert restored['direction'] == 1 and ctx.path.read_bytes() == before
    assert restored['latest'] == result['latest']
    for p in ctx.archive_path.glob('*.json.gz'):
        raw = gzip.decompress(p.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == p.name.split('.')[0]


@pytest.mark.parametrize('damage', ['journal_field', 'archive_hash', 'invalid_gzip', 'missing_archive', 'incomplete_tail', 'bad_conflict_index'])
def test_corrupt_recovery_cannot_silently_reset_history(tmp_path, damage):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    if damage == 'journal_field':
        rows = [json.loads(line) for line in ctx.path.read_text().splitlines()]
        rows[0]['value'] += 1
        ctx.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif damage == 'archive_hash':
        p = next(ctx.archive_path.glob('*.json.gz'))
        p.write_bytes(gzip.compress(b'{"changed":true}'))
    elif damage == 'invalid_gzip':
        next(ctx.archive_path.glob('*.json.gz')).write_bytes(b'\x1f\x8b\x08\x00' + b'\x00'*6 + b'bad-stream')
    elif damage == 'missing_archive':
        next(ctx.archive_path.glob('*.json.gz')).unlink()
    elif damage == 'incomplete_tail':
        with ctx.path.open('ab') as handle: handle.write(b'{"unfinished":')
    else:
        ctx.conflicts_path.write_bytes(b'not-json\n')
    before = ctx.path.read_bytes()
    restored = sector.SectorContext(ctx.path, EPOCH, source)
    result = restored.decision(EPOCH + 71 * M)
    assert result['status'] == 'unknown' and result['direction'] is None
    assert ctx.path.read_bytes() == before


def test_oversized_source_and_conflict_index_fail_closed(tmp_path, monkeypatch):
    ctx, source = context(tmp_path)
    source.write_bytes(b'x' * (sector.MAX_SOURCE_BYTES + 1))
    assert ctx.decision(EPOCH + M)['reasons'] == ['sector_source_oversized']
    other = sector.SectorContext(tmp_path / 'other.jsonl', EPOCH, source)
    monkeypatch.setattr(sector, 'MAX_RESTORE_BYTES', 30)
    other.conflicts_path.write_bytes(b'x' * 31)
    assert other.decision(EPOCH + M)['reasons'] == ['sector_conflict_index_oversized']


def test_reference_version_change_requires_new_real_history(tmp_path):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    row = point(EPOCH + 61 * M, 30300)
    row['provider_reference_first_seen_at_ms'] += M
    row['provider_reference_available_at_ms'] += M
    write_source(source, row)
    result = ctx.decision(EPOCH + 71 * M + 1)
    assert result['status'] == 'unknown' and result['direction'] is None


def test_verified_clock_regression_and_invalid_clock_never_produce_direction(tmp_path):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    assert ctx.decision(None)['reasons'] == ['sector_verified_clock_required']
    # Source itself was generated later, so this also independently fails the
    # original-receipt check before any new ingestion can occur.
    assert ctx.decision(EPOCH + 69 * M)['status'] == 'unknown'


def test_conflict_before_bounded_main_tail_cannot_be_forgotten_on_restart(tmp_path, monkeypatch):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    write_source(source, point(EPOCH + 60 * M, 30300))
    assert ctx.decision(EPOCH + 70 * M + 2)['reasons'] == ['session_observation_conflict']
    for minute in range(80, 201, 10):
        write_source(source, point(EPOCH + minute * M, 30300 + minute))
        ctx.decision(EPOCH + (minute + 10) * M + 1)
    assert ctx.path.stat().st_size > 6000
    assert ctx.conflicts_path.stat().st_size < 6000
    monkeypatch.setattr(sector, 'MAX_RESTORE_BYTES', 6000)
    recovered = sector.SectorContext(ctx.path, EPOCH, source)
    result = recovered.decision(EPOCH + 210 * M + 2)
    assert result['reasons'] == ['session_observation_conflict']
    assert result['direction'] is None
    assert all(r['observed_at_ms'] > EPOCH + 60 * M for r in recovered.rows)


def test_new_provider_date_does_not_inherit_previous_date_conflict(tmp_path):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    write_source(source, point(EPOCH + 60 * M, 30300))
    ctx.decision(EPOCH + 70 * M + 2)
    first = point(EPOCH + 24 * 60 * M + 30 * M, 30100)
    first['market_session']['valid_until_ms'] += 24 * 60 * M
    write_source(source, first)
    assert ctx.decision(first['received_at_ms'] + 1)['direction'] is None
    latest = point(EPOCH + 24 * 60 * M + 60 * M, 30200)
    latest['market_session']['valid_until_ms'] += 24 * 60 * M
    write_source(source, latest)
    result = ctx.decision(latest['received_at_ms'] + 1)
    assert result['status'] == 'bullish' and result['direction'] == 1


@pytest.mark.parametrize('payload', [None, [], 1, 'invalid', True])
def test_nonobject_json_source_fails_closed_without_affecting_journal(tmp_path, payload):
    ctx, source = context(tmp_path)
    source.write_text(json.dumps(payload))
    outcome = ctx.decision(EPOCH + 10 * M)
    assert outcome['status'] == 'unknown' and outcome['direction'] is None
    assert outcome['reasons'] == ['sector_observer_snapshot_object_required']
    assert not ctx.path.exists()


@pytest.mark.parametrize('level', ['observations', 'quote', 'session'])
@pytest.mark.parametrize('payload', [None, [], 1, 'invalid', True])
def test_nested_nonobject_source_metadata_fails_closed(tmp_path, level, payload):
    ctx, source = context(tmp_path)
    row = point(EPOCH + 5 * M, 30100)
    value, _ = write_source(source, row)
    if level == 'observations': value['observations'] = payload
    elif level == 'quote': value['observations']['nq_dec_2026'] = payload
    else: value['observations']['nq_dec_2026']['market_session'] = payload
    source.write_text(json.dumps(value))
    outcome = ctx.decision(row['received_at_ms'] + 1)
    assert outcome['status'] == 'unknown' and outcome['direction'] is None
    assert not ctx.path.exists()


def test_invalid_deflate_body_is_caught_during_archive_recovery(tmp_path):
    ctx, source = context(tmp_path)
    ingest_pair(ctx, source)
    archive = next(ctx.archive_path.glob('*.json.gz'))
    compressed = archive.read_bytes()
    damaged = compressed[:10] + b'\xff' * 10 + compressed[-8:]
    with pytest.raises(zlib.error):
        gzip.decompress(damaged)
    archive.write_bytes(damaged)
    recovered = sector.SectorContext(ctx.path, EPOCH, source)
    outcome = recovered.decision(EPOCH + 71 * M)
    assert outcome['status'] == 'unknown' and outcome['direction'] is None


def test_expired_old_coverage_does_not_poison_prospectively_covered_reference(tmp_path):
    ctx, source = context(tmp_path)
    old = point(EPOCH + 10 * M, 30050)
    old['market_session']['valid_until_ms'] = EPOCH + 40 * M
    write_source(source, old)
    ctx.decision(old['received_at_ms'] + 1)
    old_journal = ctx.path.read_bytes()
    reference = point(EPOCH + 50 * M, 30100)
    write_source(source, reference)
    intermediate = ctx.decision(reference['received_at_ms'] + 1)
    assert intermediate['direction'] is None  # No renewed old reference.
    latest = point(EPOCH + 80 * M, 30200)
    write_source(source, latest)
    outcome = ctx.decision(latest['received_at_ms'] + 1)
    assert outcome['status'] == 'bullish' and outcome['direction'] == 1
    assert outcome['reference']['observed_at_ms'] == EPOCH + 50 * M
    assert ctx.path.read_bytes().startswith(old_journal)
    assert json.loads(ctx.path.read_text().splitlines()[0])['market_session']['valid_until_ms'] == EPOCH + 40 * M
    recovered = sector.SectorContext(ctx.path, EPOCH, source)
    assert recovered.decision(latest['received_at_ms'] + 2)['direction'] == 1


def test_expired_conflict_still_blocks_new_coverage_in_same_provider_date(tmp_path):
    ctx, source = context(tmp_path)
    old = point(EPOCH + 10 * M, 30050)
    old['market_session']['valid_until_ms'] = EPOCH + 40 * M
    write_source(source, old)
    ctx.decision(old['received_at_ms'] + 1)
    old['value'] = 30060
    write_source(source, old)
    assert ctx.decision(old['received_at_ms'] + 2)['reasons'] == ['session_observation_conflict']
    reference = point(EPOCH + 50 * M, 30100)
    write_source(source, reference)
    ctx.decision(reference['received_at_ms'] + 1)
    latest = point(EPOCH + 80 * M, 30200)
    write_source(source, latest)
    outcome = ctx.decision(latest['received_at_ms'] + 1)
    assert outcome['reasons'] == ['session_observation_conflict']
    assert outcome['status'] == 'unknown' and outcome['direction'] is None
    recovered = sector.SectorContext(ctx.path, EPOCH, source)
    assert recovered.decision(latest['received_at_ms'] + 2)['reasons'] == ['session_observation_conflict']
