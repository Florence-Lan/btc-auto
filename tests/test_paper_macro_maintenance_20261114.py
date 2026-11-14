"""Finite source reviews must not restore MU, old signals, or expired contracts."""
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import market_sessions as sessions
import stock_event_risk as events
import cross_market_observations as observations

ROOT = Path(__file__).resolve().parents[1]
HOUR = 3_600_000


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp()*1000)


def test_current_company_review_keeps_mu_removed_and_original_event_provenance():
    path = 'config/stock_event_calendar_20261020.json'
    current = json.loads((ROOT/path).read_text(encoding='utf-8'))
    record = current['verification_history'][-1]
    previous = json.loads((ROOT/record['previous_calendar_archive']).read_text(encoding='utf-8'))
    assert record['review_id'] == 'official_company_calendar_partial_20261114'
    assert current['symbol_reviews']['MUUSDT'] == previous['symbol_reviews']['MUUSDT']
    assert current['events'] == previous['events']
    assert current['first_seen_at_ms'] == previous['first_seen_at_ms']
    assert current['available_at_ms'] == previous['available_at_ms']
    assert current['verified_at_ms'] > previous['verified_at_ms']
    assert current['coverage_end_ms'] == current['valid_until_ms'] == stamp('2026-11-17T00:00:00+08:00')
    cfg = {events.POLICY_KEY: {**events.POLICY, 'calendar_path': path,
                             'effective_from_ms': current['available_at_ms']},
           'live_orders_allowed': False}
    checked = current['verified_at_ms']
    bars = [{'time_ms': i*HOUR} for i in range(checked//HOUR-2, checked//HOUR)]
    for symbol in ('SNDKUSDT', 'SKHYNIXUSDT'):
        assert events.evaluate(cfg, symbol, checked-1, root=ROOT)['reasons'] == ['stock_event_calendar_not_yet_available']
        assert events.evaluate(cfg, symbol, checked, root=ROOT, closed_hourly_bars=bars)['allowed']
        assert events.evaluate(cfg, symbol, current['valid_until_ms'], root=ROOT)['reasons'] == ['stock_event_calendar_expired']
    removed = events.evaluate(cfg, 'MUUSDT', current['valid_until_ms']+HOUR, root=ROOT)
    assert removed['allowed'] and removed['status'] == 'source_removed'


def test_current_session_review_has_real_boundaries_and_preserves_contract_review_cutoffs():
    config = observations.load_config(ROOT/'config/cross_market_observation_20261023.json')
    reviewed = [s for s in config['sources'] if 'session_calendar' in s]
    assert len(reviewed) == 4
    for source in reviewed:
        session = source['session_calendar']
        assert session['valid_from_ms'] == session['verified_at_ms'] > session['first_seen_at_ms']
        assert session['valid_until_ms'] == stamp('2026-11-22T00:00:00-05:00')
        assert session['reviewed_local_dates'] == [f'2026-11-{i}' for i in range(14,22)]
        assert sessions.session_status(session, session['verified_at_ms']-1)['phase'] == 'unknown'
        assert sessions.session_status(session, stamp('2026-11-14T12:00:00-05:00'))['phase'] == 'closed'
        assert sessions.session_status(session, session['valid_until_ms'])['phase'] == 'unknown'
        sunday = '2026-11-15T18:00:00-05:00' if source['id'] != 'yahoo_usdjpy' else '2026-11-15T17:05:00-05:00'
        assert sessions.session_status(session, stamp(sunday))['phase'] == 'open'
    wti = next(s for s in reviewed if s['id'] == 'yahoo_wti_dec_2026')
    assert wti['expected_symbol'] == 'CLZ26.NYM'
    assert wti['contract_review_cutoff_utc'] == '2026-11-15T00:00:00Z'
