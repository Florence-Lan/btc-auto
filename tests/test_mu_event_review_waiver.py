"""Causal MU paper-only exception without claiming official verification."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_event_risk as risk
import run_parallel_simulation as paper
from test_stock_event_risk import HOUR, bar, calendar_data, setup_calendar
from test_stock_event_risk_integration import event_worker


MODE = 'ignore_unverified_review_for_simulation'
AUTHORIZATION = 'User requested skipping unavailable MU event review for paper trading.'


def waiver(**patch):
    return {'mode': MODE, 'first_seen_at_ms': 540 * HOUR,
            'available_at_ms': 550 * HOUR, 'effective_from_ms': 560 * HOUR,
            'valid_until_ms': 790 * HOUR, 'user_authorization': AUTHORIZATION,
            **patch}


def write(path, raw):
    path.write_text(json.dumps(raw), encoding='utf-8')


def waived_calendar(tmp_path, *, symbol='MUUSDT', anchor=700 * HOUR,
                    known_event=False):
    config, path = setup_calendar(tmp_path, anchor, symbol)
    config['live_orders_allowed'] = False
    raw = json.loads(path.read_text())
    review = raw['symbol_reviews'][symbol]
    review['status'] = 'latest_review_unverified'
    review['simulation_unverified_review_waiver'] = waiver()
    if not known_event:
        raw['events'] = []
    write(path, raw)
    return config, path, raw


def test_active_paper_waiver_retains_unknown_quality_and_original_provenance(tmp_path):
    config, path, raw = waived_calendar(tmp_path)
    before = deepcopy(raw)
    outcome = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert outcome['allowed'] and outcome['status'] == 'unverified_review_ignored'
    assert outcome['data_quality'] == 'unverified'
    assert outcome['review_waiver_applied'] is True
    assert outcome['review_waiver_reason'] == AUTHORIZATION
    assert outcome['symbol_review_status'] == 'latest_review_unverified'
    assert outcome['reasons'] == [] and outcome['event_ids'] == []
    assert outcome['known_event_count'] == 0
    assert outcome['protective_exits_allowed'] and outcome['scope'] == 'new_risk_only'
    for field in ('first_seen_at_ms', 'available_at_ms', 'verified_at_ms',
                  'valid_until_ms', 'coverage_start_ms', 'coverage_end_ms'):
        assert outcome[field] == before[field]
    assert json.loads(path.read_text()) == before


def test_renewed_mu_paper_exception_keeps_calendar_and_official_review_unchanged():
    root = Path(__file__).resolve().parents[1]
    path = 'config/stock_event_calendar_20261020_history/8c10aa680cec50b52768d0bc3ba19b7a0d310cf27fd35075fb8b5d5d8efdadcb.json'
    raw = json.loads((root/path).read_text(encoding='utf-8'))
    renewal = raw['verification_history'][-1]
    assert renewal['review_id'] == 'mu_simulation_unverified_waiver_renewal_20261111'
    previous = json.loads((root/renewal['previous_calendar_archive']).read_text(encoding='utf-8'))
    for field in ('first_seen_at_ms', 'available_at_ms', 'verified_at_ms',
                  'valid_until_ms', 'coverage_start_ms', 'coverage_end_ms', 'events'):
        assert raw[field] == previous[field]
    for symbol in ('SNDKUSDT', 'SKHYNIXUSDT'):
        assert raw['symbol_reviews'][symbol] == previous['symbol_reviews'][symbol]
    review = raw['symbol_reviews']['MUUSDT']
    old_review = previous['symbol_reviews']['MUUSDT']
    key = 'simulation_unverified_review_waiver'
    assert {k:v for k,v in review.items() if k != key} == {k:v for k,v in old_review.items() if k != key}
    current = review[key]
    assert current == renewal['new_waiver']
    assert old_review[key] == renewal['previous_waiver']
    assert current['user_authorization'] == old_review[key]['user_authorization']
    start = current['effective_from_ms']
    assert start == current['first_seen_at_ms'] == current['available_at_ms']
    assert start > old_review[key]['valid_until_ms']
    assert current['valid_until_ms'] == raw['valid_until_ms']
    cfg = {risk.POLICY_KEY: {**risk.POLICY, 'calendar_path': path,
                            'effective_from_ms': raw['available_at_ms']},
           'live_orders_allowed': False}
    assert not risk.evaluate(cfg, 'MUUSDT', start-1, root=root)['allowed']
    admitted = risk.evaluate(cfg, 'MUUSDT', start, root=root)
    assert admitted['allowed'] and admitted['status'] == 'unverified_review_ignored'
    assert admitted['data_quality'] == 'unverified' and admitted['protective_exits_allowed']
    assert not risk.evaluate({**cfg, 'live_orders_allowed': True}, 'MUUSDT', start, root=root)['allowed']
    expired = risk.evaluate(cfg, 'MUUSDT', current['valid_until_ms'], root=root)
    assert not expired['allowed'] and expired['reasons'] == ['stock_event_calendar_expired']


@pytest.mark.parametrize('context', ['absent', True, None, 0, '', []])
def test_only_literal_order_disabled_context_may_apply_waiver(tmp_path, context):
    config, _, _ = waived_calendar(tmp_path)
    if context == 'absent':
        config.pop('live_orders_allowed')
    else:
        config['live_orders_allowed'] = context
    outcome = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert not outcome['allowed'] and outcome['status'] == 'unavailable'
    assert not outcome.get('review_waiver_applied', False)


@pytest.mark.parametrize('symbol', ['SNDKUSDT', 'SKHYNIXUSDT'])
def test_waiver_is_specific_to_mu(tmp_path, symbol):
    config, _, _ = waived_calendar(tmp_path, symbol=symbol)
    outcome = risk.evaluate(config, symbol, 600 * HOUR)
    assert not outcome['allowed'] and outcome['status'] == 'unavailable'
    assert not outcome.get('review_waiver_applied', False)


@pytest.mark.parametrize('at', [550 * HOUR, 560 * HOUR - 1,
                               790 * HOUR, 790 * HOUR + 1])
def test_waiver_never_applies_before_activation_or_at_expiry(tmp_path, at):
    config, _, _ = waived_calendar(tmp_path)
    outcome = risk.evaluate(config, 'MUUSDT', at)
    assert not outcome['allowed'] and outcome['status'] == 'unavailable'
    assert not outcome.get('review_waiver_applied', False)


def test_waiver_activation_boundary_is_inclusive(tmp_path):
    config, _, _ = waived_calendar(tmp_path)
    assert risk.evaluate(config, 'MUUSDT', 560 * HOUR)['allowed']


@pytest.mark.parametrize('patch', [
    {'mode': 'ignore_all_calendar_failures'},
    {'first_seen_at_ms': True},
    {'first_seen_at_ms': 551 * HOUR},
    {'available_at_ms': 561 * HOUR},
    {'effective_from_ms': 790 * HOUR},
    {'valid_until_ms': 801 * HOUR},
    {'user_authorization': ''},
    {'user_authorization': True},
])
def test_malformed_or_unbounded_waiver_is_not_an_entry_permit(tmp_path, patch):
    config, path, raw = waived_calendar(tmp_path)
    raw['symbol_reviews']['MUUSDT']['simulation_unverified_review_waiver'] = waiver(**patch)
    write(path, raw)
    outcome = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert not outcome['allowed'] and outcome['status'] == 'unavailable'


@pytest.mark.parametrize('fault', ['absent_waiver', 'null_waiver', 'missing_review',
    'missing_calendar', 'corrupt_calendar', 'future_calendar', 'expired_calendar',
    'invalid_review_clock', 'invalid_source_urls', 'different_review_status'])
def test_exception_cannot_mask_other_calendar_or_review_failures(tmp_path, fault):
    config, path, raw = waived_calendar(tmp_path)
    review = raw['symbol_reviews']['MUUSDT']
    if fault == 'absent_waiver': review.pop('simulation_unverified_review_waiver')
    elif fault == 'null_waiver': review['simulation_unverified_review_waiver'] = None
    elif fault == 'missing_review': raw['symbol_reviews'] = {}
    elif fault == 'future_calendar': raw['verified_at_ms'] = 601 * HOUR
    elif fault == 'expired_calendar': raw['valid_until_ms'] = 600 * HOUR
    elif fault == 'invalid_review_clock': review['verified_at_ms'] = True
    elif fault == 'invalid_source_urls': review['source_urls'] = []
    elif fault == 'different_review_status': review['status'] = 'provider_temporarily_unavailable'
    write(path, raw)
    if fault == 'missing_calendar': path.unlink()
    elif fault == 'corrupt_calendar': path.write_text('{', encoding='utf-8')
    outcome = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert not outcome['allowed'] and outcome['status'] == 'unavailable'


def test_waiver_keeps_known_event_blackout_and_post_event_hour_requirements(tmp_path):
    config, _, _ = waived_calendar(tmp_path, anchor=624 * HOUR, known_event=True)
    before = risk.evaluate(config, 'MUUSDT', 600 * HOUR - 1)
    assert before['allowed'] and before['status'] == 'unverified_review_ignored'
    blocked = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert not blocked['allowed'] and blocked['status'] == 'blackout'
    assert blocked['review_waiver_applied'] and blocked['data_quality'] == 'unverified'
    assert blocked['reasons'] == ['stock_event_blackout']
    assert blocked['event_ids'] == ['earnings']
    missing_bars = risk.evaluate(config, 'MUUSDT', 626 * HOUR)
    assert not missing_bars['allowed']
    assert missing_bars['reasons'] == ['stock_event_post_event_bars_unavailable']
    after = risk.evaluate(config, 'MUUSDT', 626 * HOUR,
                          closed_hourly_bars=[bar(624 * HOUR), bar(625 * HOUR)])
    assert after['allowed'] and after['status'] == 'unverified_review_ignored'


def test_next_verified_source_returns_to_normal_review_without_using_waiver(tmp_path):
    config, path, raw = waived_calendar(tmp_path)
    assert risk.evaluate(config, 'MUUSDT', 600 * HOUR)['review_waiver_applied']
    raw['symbol_reviews']['MUUSDT']['status'] = 'no_announced_event_found'
    write(path, raw)
    restored = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert restored['allowed'] and restored['status'] == 'clear'
    assert not restored.get('review_waiver_applied', False)


def test_each_evaluation_rereads_waiver_removal_and_new_known_event(tmp_path):
    config, path, raw = waived_calendar(tmp_path)
    admitted = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert admitted['allowed']
    review = raw['symbol_reviews']['MUUSDT']
    review.pop('simulation_unverified_review_waiver')
    write(path, raw)
    removed = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert not removed['allowed']
    assert removed['calendar_sha256'] != admitted['calendar_sha256']
    review['simulation_unverified_review_waiver'] = waiver()
    raw['events'] = calendar_data(624 * HOUR)['events']
    write(path, raw)
    announced = risk.evaluate(config, 'MUUSDT', 600 * HOUR)
    assert not announced['allowed'] and announced['reasons'] == ['stock_event_blackout']


@pytest.mark.parametrize('state_patch,allowed', [
    ({}, True), ({'mode': 'live'}, False), ({'mode': None}, False),
    ({'places_orders': True}, False), ({'places_orders': 0}, False),
    ({'places_orders': None}, False),
])
def test_paper_context_is_private_and_requires_explicit_simulation_state(
        tmp_path, monkeypatch, state_patch, allowed):
    worker, path, _, calendar_path = event_worker(tmp_path, monkeypatch,
                                                  anchor=700 * HOUR)
    assert 'live_orders_allowed' not in worker.config
    raw = json.loads(calendar_path.read_text())
    raw['events'] = []
    raw['symbol_reviews']['MUUSDT'].update(status='latest_review_unverified',
        simulation_unverified_review_waiver=waiver())
    write(calendar_path, raw)
    state = paper.read_json(path)
    state.update(state_patch)
    before_config, before_rule = deepcopy(worker.config), deepcopy(state['rule'])
    monkeypatch.setattr(worker, 'now_ms', lambda: 600 * HOUR)
    outcome = worker.record_stock_event_risk(state, 'observation', 600 * HOUR)
    assert outcome['allowed'] is allowed
    assert worker.config == before_config and state['rule'] == before_rule
    assert 'live_orders_allowed' not in worker.config
    assert outcome['phase'] == 'observation'


@pytest.mark.parametrize('side', [1, -1])
def test_paper_retry_cannot_reuse_waiver_removed_after_initial_approval(
        tmp_path, monkeypatch, side):
    worker, path, clock, calendar_path = event_worker(tmp_path, monkeypatch, side,
                                                     anchor=700 * HOUR)
    worker.config['live_orders_allowed'] = False
    raw = json.loads(calendar_path.read_text())
    raw['events'] = []
    raw['symbol_reviews']['MUUSDT'].update(status='latest_review_unverified',
        simulation_unverified_review_waiver=waiver())
    write(calendar_path, raw)
    clock['volume'] = 0
    waiting = worker.step()
    assert waiting['pending_signal'] and waiting['stock_event_risk']['allowed']
    raw['symbol_reviews']['MUUSDT'].pop('simulation_unverified_review_waiver')
    write(calendar_path, raw)
    clock.update(now=clock['now'] + 300_000, volume=10000)
    restarted = paper.StockAccount(path, worker.venue, 'MUUSDT', worker.config)
    blocked = restarted.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['pending_signal'] == waiting['pending_signal']
    assert not blocked['stock_event_risk']['allowed']


def test_final_fill_rechecks_waiver_expiry_after_depth_evaluation(tmp_path, monkeypatch):
    worker, _, clock, calendar_path = event_worker(tmp_path, monkeypatch,
                                                   anchor=700 * HOUR)
    worker.config['live_orders_allowed'] = False
    raw = json.loads(calendar_path.read_text())
    raw['events'] = []
    raw['symbol_reviews']['MUUSDT'].update(status='latest_review_unverified',
        simulation_unverified_review_waiver=waiver(valid_until_ms=clock['now'] + 1000))
    write(calendar_path, raw)
    original_fill = paper.book_fill
    def slow_depth(*args, **kwargs):
        result = original_fill(*args, **kwargs)
        clock['now'] += 2000
        return result
    monkeypatch.setattr(paper, 'book_fill', slow_depth)
    blocked = worker.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['stock_event_risk']['phase'] == 'fill'
    assert not blocked['stock_event_risk']['allowed']
    assert blocked['wallet_balance'] == 1000 and blocked['fees_paid'] == 0
