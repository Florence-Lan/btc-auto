"""Causal, read-only and restart invariants for the separate admission observer."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_cross_market_observer as observer
import cross_market_observations as observations
import cross_market_risk as risk
import international_event_calendar as events

NOW = 1_800_000_000_000
EPOCH = NOW - 60_000


def stock_state(origin=NOW - 30_000):
    return {'symbol': 'MUUSDT', 'wallet_balance': 1000., 'equity': 1000., 'initial_balance': 1000.,
        'realized_pnl': 0., 'fees_paid': 0., 'funding_pnl': 0., 'fill_count_total': 0,
        'position_qty': 0., 'position': None, 'last_mark_price': 100.,
        'rule': {'risk_fraction_per_trade': .0025}, 'checked_at_utc': observer.iso(NOW),
        'pending_signal': {'boundary_ms': origin, 'expires_at_ms': NOW + 3_600_000,
            'signal': {'direction': 1, 'time_ms': origin - 3_600_000}},
        'last_signal_time_ms': origin, 'signal_status': 'observed_entry_checks_passed',
        'entry_checks': {'passed': {'funding': True, 'spread': True}, 'checked_at_ms': NOW}, 'entry_blockers': [],
        '_reported': {'status': 'healthy', 'checked_at_utc': observer.iso(NOW),
            'signal_status': 'observed_entry_checks_passed', 'entry_blockers': []}}


def btc_state(origin=NOW - 30_000):
    return {'position_qty': 0., 'wallet_balance': 1000., 'checked_at_utc': observer.iso(NOW),
        'btc_pullback_pending': {'status': 'ready', 'signal_origin': origin, 'expires_at_ms': NOW + 30_000,
            'side': 'long', 'source': {'origin_signal_time_ms': origin, 'position_id': 'p1'}},
        '_reported': {'status': 'healthy', 'checked_at_utc': observer.iso(NOW),
            'signal_status': 'signal_observed', 'entry_blockers': []}}


def btc_report(origin=NOW - 30_000):
    return {'generated_at_utc': observer.iso(NOW), 'execution_target': {
        'origin_signal_time_ms': origin, 'signed_qty': .01, 'position_id': 'p1'}}


def tracker():
    return {'started_at_ms': EPOCH, 'journal_sequence': 0, 'cycles': 0,
        'opportunities': {}, 'counted': {}, 'market_history': {}}


@pytest.fixture
def gates(monkeypatch):
    def gate(**extra):
        return {'allowed': True, 'status': 'clear', 'reasons': [], **extra}
    monkeypatch.setattr(risk, 'update_forward_history', lambda snapshot, history, **kw:
        {**history, 'observed_cutoff': kw['asof_ms'], 'inception': kw['inception_ms']})
    monkeypatch.setattr(risk, 'portfolio_exposure', lambda *args, **kw: {'status': 'available'})
    monkeypatch.setattr(risk, 'evaluate_cross_market_risk', lambda *args, **kw: gate())
    monkeypatch.setattr(risk, 'evaluate_sector_budget', lambda *args, **kw: gate())
    monkeypatch.setattr(events, 'evaluate_international_event_calendar', lambda *args, **kw: gate())
    return gate


def evaluate(states, *, prior=None, now=NOW):
    origin = (states.get('btc', {}).get('btc_pullback_pending') or {}).get('signal_origin', NOW - 30_000)
    return observer.evaluate_cycle({'started_at_ms': EPOCH}, {}, states, btc_report(origin), {}, {},
        asof_ms=now, tracker=prior or tracker())


@pytest.mark.parametrize('value', [True, False, -1, 1.5, '2026-10-23T09:00:00', '1960-01-01T00:00:00Z', None])
def test_timestamp_never_uses_host_timezone_or_boolean(value):
    assert observer.timestamp_ms(value) is None


def test_timestamp_preserves_explicit_offset():
    assert observer.timestamp_ms('2026-10-23T09:00:00+08:00') == observer.timestamp_ms('2026-10-23T01:00:00Z')


def test_pre_inception_origins_are_visible_but_never_counted(gates):
    states = {account: stock_state(EPOCH - 1) for account in ('mu', 'sndk', 'skhynix')}
    states['btc'] = btc_state(EPOCH - 1)
    rows, updated, _ = evaluate(states)
    assert not updated['counted'] and not updated['opportunities']
    assert all(not row['counted_once'] and not row['post_inception'] for row in rows)
    assert all(row['baseline_is_reported_not_fill_permit'] for row in rows)


def test_a_pending_signal_is_not_a_fill_permit(gates):
    state = stock_state()
    state['entry_checks'] = {}
    baseline = observer.reported_baseline('mu', state, {}, asof_ms=NOW, inception_ms=EPOCH)
    assert not baseline['allowed'] and baseline['actual_fill_permission'] is None
    assert 'baseline_original_entry_checks_not_all_reported_passed' in baseline['reasons']
    state = stock_state()
    state['_reported']['entry_blockers'] = ['swing_4h_ema_ordering']
    rows, updated, _ = evaluate({'mu': state})
    assert not updated['counted']
    assert not any(row['candidate_effectively_filtered'] for row in rows)


@pytest.mark.parametrize('change,reason', [
    ({'expires_at_ms': NOW}, 'baseline_pending_expired_or_invalid'),
    ({'expires_at_ms': None}, 'baseline_pending_expired_or_invalid'),
    ({'boundary_ms': NOW + 1}, 'baseline_origin_unavailable_or_future'),
])
def test_pending_lifetime_and_future_origin_fail_closed(change, reason):
    state = stock_state()
    state['pending_signal'].update(change)
    result = observer.reported_baseline('mu', state, {}, asof_ms=NOW, inception_ms=EPOCH)
    assert not result['allowed'] and reason in result['reasons']


def test_stale_and_inconsistent_sources_are_never_reported_eligible():
    state = stock_state()
    state['_reported']['checked_at_utc'] = observer.iso(NOW - 120_001)
    state['_consistency_errors'] = ['state_status_disagree:position_qty']
    result = observer.reported_baseline('mu', state, {}, asof_ms=NOW, inception_ms=EPOCH)
    assert not result['allowed']
    assert 'baseline_source_stale_or_unavailable' in result['reasons']
    assert 'baseline_state_status_disagree:position_qty' in result['reasons']


def test_reported_quote_checks_do_not_override_later_original_execution_rejections():
    state = stock_state()
    state['_reported']['signal_status'] = 'insufficient_observed_liquidity_or_minimum_quantity'
    result = observer.reported_baseline('mu', state, {}, asof_ms=NOW, inception_ms=EPOCH)
    assert not result['allowed'] and 'baseline_original_signal_not_admitted' in result['reasons']
    state = stock_state()
    state['entry_checks']['checked_at_ms'] = NOW - 120_001
    result = observer.reported_baseline('mu', state, {}, asof_ms=NOW, inception_ms=EPOCH)
    assert not result['allowed'] and 'baseline_original_entry_checks_stale_or_unavailable' in result['reasons']


def test_btc_pending_must_match_original_origin_position_and_fresh_report():
    state, report = btc_state(), btc_report()
    assert observer.reported_baseline('btc', state, report, asof_ms=NOW, inception_ms=EPOCH)['allowed']
    report['execution_target']['position_id'] = 'different'
    report['generated_at_utc'] = observer.iso(NOW - 120_001)
    result = observer.reported_baseline('btc', state, report, asof_ms=NOW, inception_ms=EPOCH)
    assert not result['allowed']
    assert 'baseline_btc_pending_report_identity_mismatch' in result['reasons']
    assert 'baseline_btc_report_stale_or_unavailable' in result['reasons']


def test_once_per_opportunity_arm_no_retroactive_reclassification(gates, monkeypatch):
    state = stock_state()
    monkeypatch.setattr(risk, 'evaluate_cross_market_risk', lambda *args, **kw:
        gates(allowed=False, status='warming_up', reasons=['nq:forward_history_warming_up']))
    rows, first, diagnostics = evaluate({'mu': state})
    assert len(first['counted']) == 5
    assert len(diagnostics['counted_updates']) == 5
    first_market = next(row for row in first['counted'].values() if row['arm'] == 'market_only')
    assert first_market['blocked_classification'] == 'unavailable_or_warmup'
    assert first_market['reasons'] == ['nq:forward_history_warming_up']
    monkeypatch.setattr(risk, 'evaluate_cross_market_risk', lambda *args, **kw: gates())
    rows, second, diagnostics = evaluate({'mu': state}, prior=first, now=NOW + 1000)
    assert len(second['counted']) == 5 and not diagnostics['counted_updates']
    assert second['counted'] == first['counted']
    assert next(row for row in rows if row['account_id'] == 'mu' and row['arm'] == 'market_only')['allowed']
    assert second['market_history']['inception'] == EPOCH


@pytest.mark.parametrize('arm,event,market,sector,expected', [
    ('event_only', {'status': 'blackout'}, {}, {}, 'event_blackout'),
    ('market_only', {}, {'status': 'shock'}, {}, 'market_shock'),
    ('sector_budget', {}, {}, {'status': 'blocked', 'allowed': False}, 'sector_budget'),
    ('combined', {'status': 'blackout'}, {'status': 'unavailable'}, {}, 'unavailable_or_warmup'),
])
def test_missing_data_is_not_counted_as_a_market_shock(arm, event, market, sector, expected):
    assert observer.blocked_classification(arm, {'allowed': False}, event, market, sector) == expected


def test_manifest_refuses_definition_drift_and_missing_history(tmp_path):
    spec, frozen = {'mode': 'OBSERVATION_ONLY'}, {'source_sha256': {'new.py': 'a'}}
    manifest = observer.prepare(tmp_path, spec, frozen, EPOCH)
    assert observer.prepare(tmp_path, spec, frozen, NOW, True) == manifest
    with pytest.raises(ValueError, match='definition changed'):
        observer.prepare(tmp_path, spec, {'source_sha256': {'new.py': 'b'}}, NOW, True)
    (tmp_path / 'tracker.json').unlink()
    with pytest.raises(FileNotFoundError):
        observer.prepare(tmp_path, spec, frozen, NOW, True)


def test_require_existing_never_initializes_a_new_generation(tmp_path):
    with pytest.raises(ValueError, match='Existing observation'):
        observer.prepare(tmp_path, {}, {}, EPOCH, True)
    assert not (tmp_path / 'manifest.json').exists()


def test_restart_recovers_counted_journal_after_tracker_write_interruption(tmp_path):
    manifest = observer.prepare(tmp_path, {}, {}, EPOCH)
    update = {'key': 'mu:1:1:combined', 'arm': 'combined', 'allowed': False,
        'blocked_classification': 'market_shock', 'reasons': ['energy_and_technology_shock']}
    (tmp_path / 'decisions.jsonl').write_text(json.dumps({'sequence': 1, 'started_at_ms': EPOCH,
        'counted_updates': [update]}) + '\n', encoding='utf-8')
    recovered = observer.recover_tracker(tmp_path, manifest)
    assert recovered['journal_sequence'] == 1 and recovered['counted'][update['key']] == update


def test_blocked_collector_has_a_deadline_and_one_inflight_worker(monkeypatch):
    monkeypatch.setattr(observer, '_COLLECT_INFLIGHT', None)
    release = threading.Event()
    calls = []
    def stuck(*args, **kwargs):
        calls.append(1)
        release.wait(2)
        return {}
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match='deadline'):
            observer.bounded_collect(stuck, {}, timeout_seconds=.03)
        with pytest.raises(TimeoutError, match='still outstanding'):
            observer.bounded_collect(stuck, {}, timeout_seconds=.03)
        assert time.monotonic() - started < .5 and calls == [1]
    finally:
        release.set()
        observer._COLLECT_INFLIGHT['thread'].join(timeout=1)


@pytest.fixture
def project(tmp_path, monkeypatch, gates):
    monkeypatch.setattr(observer, 'ROOT', tmp_path)
    output = tmp_path / 'data/cross_market_observer'
    output.mkdir(parents=True)
    base = tmp_path / 'data/parallel_simulation/original'
    accounts, status = {}, {'accounts': {}}
    originals = []
    for account in observer.ACCOUNTS:
        path = base / account / 'state.json'
        path.parent.mkdir(parents=True)
        state = btc_state() if account == 'btc' else stock_state()
        reported = state.pop('_reported')
        if account != 'btc':
            reported['last_signal_time_ms'] = state['last_signal_time_ms']
        observer.write(path, state)
        originals.append(path)
        accounts[account] = {'state_path': path.relative_to(tmp_path).as_posix()}
        status['accounts'][account] = reported
    observer.write(base / 'status.json', status)
    observer.write(base / 'report.json', btc_report())
    calendar = tmp_path / 'config/calendar.json'
    observer.write(calendar, {'test_calendar_bytes': 1})
    observer.write(tmp_path / 'config/observations.json', {})
    spec = {'accounts': accounts, 'status_path': (base / 'status.json').relative_to(tmp_path).as_posix(),
        'btc_report_path': (base / 'report.json').relative_to(tmp_path).as_posix(),
        'international_calendar': 'config/calendar.json', 'observation_config': 'config/observations.json',
        'poll_seconds': 30, 'collect_seconds': 300}
    manifest = observer.prepare(output, spec, {}, EPOCH)
    monkeypatch.setattr(observations, 'load_config', lambda path: {})
    return tmp_path, output, spec, manifest, originals, calendar


def test_one_real_cycle_is_read_only_and_archives_the_exact_context(project):
    _, output, spec, manifest, originals, calendar = project
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
    snapshot = {'observations': {}, 'sources': {}}
    def collect(*args, **kwargs):
        heartbeat = observer.read(output / 'status.json')
        assert heartbeat['status'] == 'collecting' and heartbeat['running']
        assert kwargs['now_ms'] == NOW
        return snapshot
    status = observer.run_once(output, spec, manifest, asof_ms=NOW, collector=collect)
    assert status['mode'] == 'OBSERVATION_ONLY' and status['no_pnl_simulated']
    assert status['places_orders'] is False and status['pnl_comparison_available'] is False
    assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in originals} == before
    assert status['counts']['baseline']['reported_eligible_opportunities'] == 4
    raw = gzip.decompress((output / 'evidence' / (status['context_sha256']['international_calendar_raw'] + '.json.gz')).read_bytes())
    assert raw == calendar.read_bytes()
    assert (output / 'selected_states_latest.json').exists()
    again = observer.run_once(output, spec, manifest, asof_ms=NOW + 1000, collector=lambda *a, **k: pytest.fail('not due'))
    assert again['counts'] == status['counts'] and again['cycles'] == 2


def test_collection_failure_preserves_actual_prior_receipt_times(project):
    _, output, spec, manifest, _, _ = project
    previous = {'observations': {'brent': {'value': 100., 'first_seen_at_ms': NOW - 50_000,
        'available_at_ms': NOW - 49_000}}, 'sources': {}}
    observer.write(output / 'latest_observations.json', previous)
    def fail(*args, **kwargs):
        raise TimeoutError('provider down')
    status = observer.run_once(output, spec, manifest, asof_ms=NOW, collector=fail)
    after = observer.read(output / 'latest_observations.json')
    assert after['observations'] == previous['observations']
    assert after['observer_collection_error'] == 'TimeoutError'
    assert status['status'] == 'degraded'


def test_real_collection_receipts_are_not_backdated_to_request_start(project, monkeypatch):
    _, output, spec, manifest, _, _ = project
    monkeypatch.setattr(observer.time, 'time', lambda: NOW / 1000)
    seen = []
    def collect(*args, **kwargs):
        seen.append(kwargs['now_ms'])
        return {'observations': {}, 'sources': {}}
    observer.run_once(output, spec, manifest, collector=collect)
    assert seen == [None]


def test_read_only_check_validates_references_without_network_or_writes(project, monkeypatch):
    root, output, spec, _, originals, calendar = project
    spec = {**spec, 'schema_version': 1, 'mode': 'OBSERVATION_ONLY', 'places_orders': False,
        'no_pnl_simulated': True}
    config = root / 'config/observer.json'
    observer.write(config, spec)
    observer.write(calendar, {'verified_at_ms': NOW})
    monkeypatch.setattr(observer, 'source_hashes', lambda: {'scripts/observer.py': 'frozen'})
    monkeypatch.setattr(observations, 'collect_observations', lambda *a, **k: pytest.fail('check must not fetch'))
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = observer.check_design(config)
    after = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before == after
    assert not result['output_initialized'] and not result['network_requests_performed']
    assert len(result['reference_sha256']) == len(originals) + 2
    assert result['source_file_count'] == 1
    assert result['calendar_current_freshness_not_renewed']


def test_check_cli_exits_before_initializing_output(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['observer', '--check', '--config', 'some-config.json'])
    monkeypatch.setattr(observer, 'check_design', lambda path: {'read_only': True})
    monkeypatch.setattr(observer, 'prepare', lambda *a, **k: pytest.fail('check must not initialize'))
    monkeypatch.setattr(observer, 'run_once', lambda *a, **k: pytest.fail('check must not run a cycle'))
    assert observer.main() == 0
    assert json.loads(capsys.readouterr().out) == {'read_only': True}


def checkpoint_history(output, manifest, count=400):
    lines = [observer.canonical({'sequence': n, 'started_at_ms': EPOCH,
        'counted_updates': [], 'padding': 'x' * 8192}) + b'\n' for n in range(1, count + 1)]
    (output / 'decisions.jsonl').write_bytes(b''.join(lines))
    state = observer.read(output / 'tracker.json')
    state.update(journal_sequence=count, journal_offset=sum(map(len, lines)),
        last_record_start=sum(map(len, lines[:-1])), last_record_sha256=observer.digest(lines[-1]), cycles=count)
    observer.write(output / 'tracker.json', state)
    return lines


def test_incremental_checkpoint_does_not_rescan_multi_megabyte_journal(tmp_path, monkeypatch):
    manifest = observer.prepare(tmp_path, {}, {}, EPOCH)
    lines = checkpoint_history(tmp_path, manifest)
    journal = tmp_path / 'decisions.jsonl'
    actual_open = Path.open
    received = []
    class Meter:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def read(self, *args):
            data = self.handle.read(*args)
            received.append(len(data))
            return data
        def readline(self, *args):
            data = self.handle.readline(*args)
            received.append(len(data))
            return data
    def metered(path, *args, **kwargs):
        handle = actual_open(path, *args, **kwargs)
        return Meter(handle) if path == journal and args and args[0] == 'rb' else handle
    monkeypatch.setattr(Path, 'open', metered)
    for _ in range(3):
        recovered = observer.recover_tracker(tmp_path, manifest)
        assert recovered['journal_sequence'] == len(lines)
    # Each cycle authenticates only the last committed record, then seeks EOF.
    assert sum(received) == 3 * len(lines[-1])
    assert sum(received) < journal.stat().st_size // 100


def test_checkpoint_recovers_only_uncommitted_tail_without_double_counting(tmp_path):
    manifest = observer.prepare(tmp_path, {}, {}, EPOCH)
    lines = checkpoint_history(tmp_path, manifest, 3)
    update = {'key': 'mu:1799999970000:1:combined', 'account_id': 'mu', 'arm': 'combined',
        'allowed': False, 'counted_at_ms': NOW, 'opportunity_id': 'mu:1799999970000:1',
        'origin_ms': NOW - 30_000, 'direction': 1, 'blocked_classification': 'market_shock',
        'reasons': ['energy_and_technology_shock']}
    appended = observer.canonical({'sequence': 4, 'started_at_ms': EPOCH, 'counted_updates': [update]}) + b'\n'
    with (tmp_path / 'decisions.jsonl').open('ab') as handle:
        handle.write(appended)
    first = observer.recover_tracker(tmp_path, manifest)
    assert first['journal_sequence'] == 4 and first['journal_offset'] == sum(map(len, lines)) + len(appended)
    assert first['last_record_sha256'] == observer.digest(appended)
    assert len(first['counted']) == len(first['opportunities']) == 1
    observer.write(tmp_path / 'tracker.json', first)
    second = observer.recover_tracker(tmp_path, manifest)
    assert second['counted'] == first['counted'] and second['opportunities'] == first['opportunities']


@pytest.mark.parametrize('damage', ['truncate', 'committed_tail_change', 'partial_append'])
def test_checkpoint_rejects_shortened_changed_or_partial_history(tmp_path, damage):
    manifest = observer.prepare(tmp_path, {}, {}, EPOCH)
    lines = checkpoint_history(tmp_path, manifest, 2)
    path = tmp_path / 'decisions.jsonl'
    if damage == 'truncate':
        path.write_bytes(lines[0])
    elif damage == 'committed_tail_change':
        path.write_bytes(lines[0] + lines[1].replace(b'xxxx', b'yyyy', 1))
    else:
        with path.open('ab') as handle:
            handle.write(b'{"partial":true')
    with pytest.raises(ValueError):
        observer.recover_tracker(tmp_path, manifest)
