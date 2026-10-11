"""Execution regressions for the opt-in hourly entry-timing revision."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import stock_entry_timing as timing
from test_stock_swing_confirmation import worker_fixture
from test_parallel_btc_entry_status import btc_status_fixture


def timed_stock(tmp_path, monkeypatch, side):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch, side)
    cfg = {**worker.config, 'entry_timing_policy': timing.POLICY.copy()}
    worker = paper.StockAccount(path, worker.venue, 'MUUSDT', cfg)
    state = paper.read_json(path)
    state['rule'] = cfg
    paper.write_json(path, state)
    original = worker.venue.get.side_effect
    def get(endpoint, params=None):
        result = original(endpoint, params)
        if 'entry_price' in clock and endpoint == 'premiumIndex':
            result.update(markPrice=str(clock['entry_price']), indexPrice=str(clock['entry_price']))
        if 'entry_price' in clock and endpoint == 'depth':
            result.update(bids=[[str(clock['entry_price']-.0001), '1000']],
                          asks=[[str(clock['entry_price']+.0001), '1000']])
        return result
    worker.venue.get.side_effect = get
    clock['volume'] = 0
    return worker, path, clock


@pytest.mark.parametrize('side', [1, -1])
def test_failed_breakout_cancels_pending_and_cannot_retry_same_hour(tmp_path, monkeypatch, side):
    worker, path, clock = timed_stock(tmp_path, monkeypatch, side)
    waiting = worker.step()
    trigger = waiting['pending_signal']['entry_trigger']
    assert waiting['fill_count_total'] == 0
    clock.update(now=clock['now']+300_000, volume=1000,
                 entry_price=trigger['breakout_level']-side*.01)
    cancelled = worker.step()
    assert cancelled['pending_signal'] is None
    assert cancelled['position'] is None and cancelled['fill_count_total'] == 0
    assert cancelled['signal_status'] == 'entry_signal_invalidated'
    assert 'breakout_reference_invalidated' in cancelled['entry_blockers']
    # Returning outside the old level cannot resurrect this already failed signal.
    clock.update(now=clock['now']+30_000, entry_price=trigger['signal_close'])
    restarted = paper.StockAccount(path, worker.venue, 'MUUSDT', worker.config)
    again = restarted.step()
    assert again['pending_signal'] is None and again['fill_count_total'] == 0
    assert again['signal_status'] == 'entry_signal_invalidated'
    assert again['last_signal_time_ms'] == waiting['last_signal_time_ms']


@pytest.mark.parametrize('side', [1, -1])
def test_valid_breakout_retry_survives_restart_and_records_final_fill_check(tmp_path, monkeypatch, side):
    worker, path, clock = timed_stock(tmp_path, monkeypatch, side)
    waiting = worker.step()
    original_trigger = copy.deepcopy(waiting['pending_signal']['entry_trigger'])
    # This tiny-ATR fixture needs room for the configured 2 bps adverse fill.
    clock.update(now=clock['now']+300_000, volume=1000,
                 entry_price=original_trigger['breakout_level']+side*original_trigger['signal_atr']*.04)
    restarted = paper.StockAccount(path, worker.venue, 'MUUSDT', worker.config)
    entered = restarted.step()
    assert entered['position']['direction'] == side
    assert entered['fill_count_total'] == 1 and entered['pending_signal'] is None
    assert entered['stock_entry_timing']['allowed']
    assert entered['stock_entry_timing']['phase'] == 'fill'
    assert entered['stock_entry_timing']['breakout_level'] == original_trigger['breakout_level']
    assert side*(entered['position']['entry']-original_trigger['breakout_level']) > 0


def test_final_depth_fill_cannot_exceed_extension_limit(tmp_path, monkeypatch):
    worker, path, clock = timed_stock(tmp_path, monkeypatch, 1)
    waiting = worker.step()
    trigger = waiting['pending_signal']['entry_trigger']
    clock.update(now=clock['now']+300_000, volume=1000)
    original = paper.book_fill
    def deep_fill(*args, **kwargs):
        result = original(*args, **kwargs)
        if result['qty']:
            result['price'] = trigger['breakout_level']+trigger['signal_atr']*1.1
        return result
    monkeypatch.setattr(paper, 'book_fill', deep_fill)
    blocked = worker.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert 'entry_timing_fill_extension_exceeded' in blocked['entry_blockers']
    assert blocked['pending_signal'] is not None


def test_entry_only_migration_preserves_consumed_signal_inventory_and_cash(tmp_path, monkeypatch):
    original = {'signal_timeframe': '1h', 'candidate_id': 'original', 'risk': .0025}
    latest = {**original, 'candidate_id': 'timed', 'entry_timing_policy': timing.POLICY.copy()}
    path = tmp_path/'mu/state.json'
    state = paper.initial_stock('MUUSDT', 10, original)
    state.update(wallet_balance=997., realized_pnl=-2., fees_paid=1., funding_pnl=0.,
        fills=[{'sequence': 1}], fill_count_total=1, pending_signal={'boundary_ms': 20},
        last_signal_time_ms=20, cooldown_until_ms=30,
        position={'entry': 100., 'qty': 1., 'direction': 1, 'stop': 98., 'initial_stop': 98.},
        position_history=[{'time_ms': 20, 'signed_qty': 1.}])
    paper.write_json(path, state)
    monkeypatch.setattr(paper, 'stock_config', lambda *_: latest)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}],
            'stock_rule_revision_scope': 'entry_timing_only'}
    manifest = {}
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    migrated = paper.read_json(path)
    for key in ('created_at_utc', 'start_ms', 'wallet_balance', 'realized_pnl', 'fees_paid',
                'funding_pnl', 'fills', 'fill_count_total', 'position', 'position_history',
                'last_signal_time_ms', 'cooldown_until_ms'):
        assert migrated[key] == state[key]
    assert migrated['pending_signal'] is None
    paper.synchronize_stock_rules(plan, tmp_path, manifest)
    assert len(manifest['rule_revisions']) == 1
    altered = {**latest, 'risk': .01}
    monkeypatch.setattr(paper, 'stock_config', lambda *_: altered)
    with pytest.raises(ValueError, match='cannot alter signals, exits or risk'):
        paper.synchronize_stock_rules(plan, tmp_path, manifest)


def test_runner_expiry_visible_even_when_five_minute_report_is_fresh(btc_status_fixture):
    step, _, _, report, report_path, _, account, reconcile = btc_status_fixture
    profile_path = report_path.with_name('profile.json')
    profile_path.write_text(json.dumps({'candidate_id': 'btc_fixture', 'entry_timing_policy': {
        'mode': 'origin_signal_window_v1', 'max_initial_entry_age_seconds': 3600}}), encoding='utf-8')
    report['execution_target'].update(signed_qty=-1, origin_signal_time_ms=1, position_id='old')
    # The fixture clock is 30m: use a shorter declared window for this boundary check.
    profile = json.loads(profile_path.read_text())
    profile['entry_timing_policy']['max_initial_entry_age_seconds'] = 900
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    report_path.write_text(json.dumps(report), encoding='utf-8')
    result = step()
    assert result['btc_entry_timing']['status'] == 'expired'
    assert 'btc_initial_entry_origin_expired' in result['entry_blockers']
    assert result['signal_status'] == 'entry_blocked'
    reconcile.assert_not_called()


def test_explicit_null_policy_does_not_silently_disable_stock_protection():
    with pytest.raises(ValueError, match='entry_timing_policy_invalid'):
        timing.validate({'entry_timing_policy': None, 'signal_family': 'donchian20',
                         'signal_timeframe': '1h'})
