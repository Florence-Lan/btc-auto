"""Removing an unavailable source is distinct from renewing an unverified review."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_event_risk as risk
import run_parallel_simulation as paper
from test_stock_event_risk import HOUR, bar, setup_calendar
from test_stock_event_risk_integration import event_worker


def removed_review():
    return {'status': 'source_removed', 'verified_at_ms': 500*HOUR, 'source_urls': [],
            'source_removal': {'mode': 'remove_unavailable_source_for_simulation',
                'first_seen_at_ms': 540*HOUR, 'available_at_ms': 550*HOUR,
                'effective_from_ms': 560*HOUR, 'user_authorization': 'Remove unavailable source.'}}


def fixture(tmp_path, *, symbol='MUUSDT', known=False):
    cfg, path = setup_calendar(tmp_path, anchor=624*HOUR, symbol=symbol)
    cfg['live_orders_allowed'] = False
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['symbol_reviews'][symbol] = removed_review()
    if not known:
        raw['events'] = []
    path.write_text(json.dumps(raw), encoding='utf-8')
    return cfg, path, raw


@pytest.mark.parametrize('at', [560*HOUR, 600*HOUR, 800*HOUR, 10000*HOUR])
def test_removed_empty_source_does_not_need_a_new_waiver_after_other_coverage_expires(tmp_path, at):
    cfg, _, _ = fixture(tmp_path)
    outcome = risk.evaluate(cfg, 'MUUSDT', at)
    assert outcome['allowed'] and outcome['status'] == 'source_removed'
    assert outcome['source_removed'] and outcome['data_quality'] == 'source_removed'
    assert outcome['calendar_required'] is False and outcome['protective_exits_allowed']
    assert not outcome.get('review_waiver_applied', False)


def test_source_removal_is_not_retroactive(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    before = risk.evaluate(cfg, 'MUUSDT', 560*HOUR-1)
    assert not before['allowed'] and before['reasons'] == ['stock_event_source_removal_not_yet_active']


@pytest.mark.parametrize('context', ['missing', True, None, 0, 'false'])
def test_source_removal_never_allows_live_or_unspecified_context(tmp_path, context):
    cfg, _, _ = fixture(tmp_path)
    if context == 'missing': cfg.pop('live_orders_allowed')
    else: cfg['live_orders_allowed'] = context
    assert not risk.evaluate(cfg, 'MUUSDT', 600*HOUR)['allowed']


@pytest.mark.parametrize('symbol', ['SNDKUSDT', 'SKHYNIXUSDT'])
def test_another_symbol_cannot_use_mu_removal(tmp_path, symbol):
    cfg, _, _ = fixture(tmp_path, symbol=symbol)
    assert not risk.evaluate(cfg, symbol, 600*HOUR)['allowed']


@pytest.mark.parametrize('patch', [
    {'source_urls': ['https://example.test/active']},
    {'simulation_unverified_review_waiver': {}},
    {'source_removal': None},
    {'source_removal': {'mode': 'ignore_all_events'}},
])
def test_malformed_or_mixed_source_state_is_not_a_permit(tmp_path, patch):
    cfg, path, raw = fixture(tmp_path)
    raw['symbol_reviews']['MUUSDT'].update(patch)
    path.write_text(json.dumps(raw), encoding='utf-8')
    assert not risk.evaluate(cfg, 'MUUSDT', 600*HOUR)['allowed']


@pytest.mark.parametrize('patch', [
    {'first_seen_at_ms': True}, {'first_seen_at_ms': 501*HOUR, 'available_at_ms': 500*HOUR},
    {'first_seen_at_ms': 499*HOUR}, {'available_at_ms': 561*HOUR},
    {'user_authorization': ''}, {'valid_until_ms': 800*HOUR},
])
def test_removal_requires_valid_authorized_clocks_without_timed_waiver(tmp_path, patch):
    cfg, path, raw = fixture(tmp_path)
    raw['symbol_reviews']['MUUSDT']['source_removal'].update(patch)
    path.write_text(json.dumps(raw), encoding='utf-8')
    assert not risk.evaluate(cfg, 'MUUSDT', 600*HOUR)['allowed']


def test_known_mu_event_from_another_source_still_blocks_and_requires_post_event_bars(tmp_path):
    cfg, _, _ = fixture(tmp_path, known=True)
    blocked = risk.evaluate(cfg, 'MUUSDT', 600*HOUR)
    assert not blocked['allowed'] and blocked['status'] == 'blackout'
    assert blocked['reasons'] == ['stock_event_blackout']
    assert blocked['calendar_required'] and blocked['source_removed']
    missing = risk.evaluate(cfg, 'MUUSDT', 626*HOUR)
    assert missing['reasons'] == ['stock_event_post_event_bars_unavailable']
    recovered = risk.evaluate(cfg, 'MUUSDT', 626*HOUR,
                             closed_hourly_bars=[bar(624*HOUR), bar(625*HOUR)])
    assert recovered['allowed'] and recovered['known_event_count'] == 1
    expired = risk.evaluate(cfg, 'MUUSDT', 800*HOUR,
                           closed_hourly_bars=[bar(624*HOUR), bar(625*HOUR)])
    assert not expired['allowed'] and expired['reasons'] == ['stock_event_calendar_expired']


@pytest.mark.parametrize('fault', ['missing_file', 'corrupt_json', 'missing_events', 'corrupt_event', 'future_snapshot'])
def test_source_removal_does_not_turn_corrupt_or_unavailable_calendar_into_a_permit(tmp_path, fault):
    cfg, path, raw = fixture(tmp_path)
    if fault == 'missing_events': raw.pop('events')
    elif fault == 'corrupt_event': raw['events'] = [{}]
    elif fault == 'future_snapshot': raw['verified_at_ms'] = 601*HOUR
    path.write_text(json.dumps(raw), encoding='utf-8')
    if fault == 'missing_file': path.unlink()
    elif fault == 'corrupt_json': path.write_text('{', encoding='utf-8')
    assert not risk.evaluate(cfg, 'MUUSDT', 600*HOUR)['allowed']


def test_runtime_rereads_new_known_event_after_a_source_removed_initial_check(tmp_path, monkeypatch):
    worker, state_path, clock, path = event_worker(tmp_path, monkeypatch, anchor=700*HOUR)
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['symbol_reviews']['MUUSDT'] = removed_review()
    known = deepcopy(raw['events'])
    raw['events'] = []
    path.write_text(json.dumps(raw), encoding='utf-8')
    clock['volume'] = 0
    waiting = worker.step()
    assert waiting['stock_event_risk']['allowed'] and waiting['pending_signal']
    known[0]['anchor_time_ms'] = clock['now'] + 2*HOUR
    raw['events'] = known
    path.write_text(json.dumps(raw), encoding='utf-8')
    clock.update(now=clock['now']+300_000, volume=10000)
    blocked = paper.StockAccount(state_path, worker.venue, 'MUUSDT', worker.config).step()
    assert not blocked['stock_event_risk']['allowed']
    assert blocked['stock_event_risk']['reasons'] == ['stock_event_blackout']
    assert blocked['position'] is None and blocked['fill_count_total'] == 0


def test_current_calendar_removes_mu_source_without_changing_other_reviews_or_event_clocks():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root/'config/stock_event_calendar_20261020.json').read_text(encoding='utf-8'))
    record = raw['verification_history'][-1]
    previous = json.loads((root/record['previous_calendar_archive']).read_text(encoding='utf-8'))
    for key in ('events', 'first_seen_at_ms', 'available_at_ms', 'verified_at_ms',
                'coverage_start_ms', 'coverage_end_ms', 'valid_until_ms'):
        assert raw[key] == previous[key]
    for symbol in ('SNDKUSDT', 'SKHYNIXUSDT'):
        assert raw['symbol_reviews'][symbol] == previous['symbol_reviews'][symbol]
    review = raw['symbol_reviews']['MUUSDT']
    assert review['source_urls'] == [] and 'simulation_unverified_review_waiver' not in review
    assert review['verified_at_ms'] == previous['symbol_reviews']['MUUSDT']['verified_at_ms']
    cfg = {risk.POLICY_KEY: {**risk.POLICY, 'calendar_path': 'config/stock_event_calendar_20261020.json',
                           'effective_from_ms': raw['available_at_ms']}, 'live_orders_allowed': False}
    after_expiry = raw['valid_until_ms'] + HOUR
    assert risk.evaluate(cfg, 'MUUSDT', after_expiry, root=root)['allowed']
    for symbol in ('SNDKUSDT', 'SKHYNIXUSDT'):
        assert risk.evaluate(cfg, symbol, after_expiry, root=root)['reasons'] == ['stock_event_calendar_expired']
