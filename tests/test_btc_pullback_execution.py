"""A pending improved entry cannot change exits, fabricate fills or reopen itself."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import trading_execution as execution
import btc_pullback_entry as entry
from test_btc_pullback_entry import ORIGIN, hourly, five, target as source_target


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for key, value in {'LLM_TRADE_GATE_ENABLED': 'false', 'SIM_MAX_LEVERAGE': '10',
                       'SIM_MAX_NOTIONAL_USDT': '0', 'SIM_TAKER_FEE': '.00045',
                       'SIM_SLIPPAGE_BPS': '1'}.items():
        monkeypatch.setenv(key, value)


def desired(side='long', **changes):
    return {**source_target(side), 'target_leverage': .2 if side == 'long' else -.2,
            'signal_time_ms': ORIGIN, **changes}


def report(rows=(), now=ORIGIN+1000):
    return {'config': {'portfolio_leverage_cap': 10}, 'shadow_profile': {
        'entry_timing_policy': {'mode': 'origin_signal_window_v1',
                                'max_initial_entry_age_seconds': 3600},
        'btc_pullback_entry_policy': copy.deepcopy(entry.DEFAULT_POLICY)},
        'btc_pullback_market_data': {'complete': True, 'asof_ms': now,
                                    'hourly': hourly(), 'five_minute': list(rows)}}


def account(tmp_path, allow_llm=False):
    result = execution.SimulationAccount(tmp_path/'state.json', allow_llm=allow_llm)
    state = result.reset(1000, now_ms=ORIGIN-1000)
    state['observed_flat_target'] = True
    result._save(state)
    return result


@pytest.mark.parametrize('side', ['long', 'short'])
def test_touch_waits_for_confirmation_then_real_fill_improves_anchor(tmp_path, side):
    bot = account(tmp_path)
    first = bot.reconcile(desired(side), 1000, report=report(), now_ms=ORIGIN+1000)
    assert first['fill'] is None and first['btc_pullback_entry']['status'] == 'waiting_retrace'
    now = ORIGIN+entry.FIVE_MINUTES
    touch = bot.reconcile(desired(side), 1000,
        report=report([five(side=side, kind='touch')], now), now_ms=now)
    assert touch['fill'] is None and touch['btc_pullback_entry']['status'] == 'waiting_confirmation'
    ready_at = ORIGIN+2*entry.FIVE_MINUTES
    entered = bot.reconcile(desired(side), 1003 if side == 'long' else 997,
        report=report([five(side=side, kind='touch'), five(1, side=side)], ready_at), now_ms=ready_at)
    fill = entered['fill']
    assert fill and entered['btc_pullback_entry']['status'] == 'filled'
    assert fill['price'] == pytest.approx(fill['btc_pullback_entry']['projected_fill_price'])
    assert fill['price'] < desired(side)['origin_entry_price'] if side == 'long' else fill['price'] > desired(side)['origin_entry_price']
    assert fill['btc_pullback_entry']['entry_low'] <= fill['price'] <= fill['btc_pullback_entry']['entry_high']
    assert bot.load()['btc_pullback_pending']['source_key'] in bot.load()['btc_pullback_consumed_sources']


@pytest.mark.parametrize('side', ['long', 'short'])
def test_same_report_can_retry_real_price_without_chasing_and_survives_restart(tmp_path, side):
    bot = account(tmp_path)
    now = ORIGIN+entry.FIVE_MINUTES
    rejected = bot.reconcile(desired(side), 1020 if side == 'long' else 980,
        report=report([five(side=side)], now), now_ms=now)
    assert rejected['fill'] is None and rejected['btc_pullback_entry']['status'] == 'ready'
    frozen = copy.deepcopy(bot.load()['btc_pullback_pending'])
    restarted = execution.SimulationAccount(bot.path, allow_llm=False)
    at = now+30_000
    entered = restarted.reconcile(desired(side), 1003 if side == 'long' else 997,
        report=report([five(side=side)], at), now_ms=at)
    assert entered['fill'] and restarted.load()['btc_pullback_pending']['candidate_low'] == frozen['candidate_low']
    assert restarted.load()['btc_pullback_pending']['expires_at_ms'] == frozen['expires_at_ms']


def test_price_wait_skips_expensive_review_but_closes_reversal_inventory(tmp_path):
    bot = account(tmp_path, allow_llm=True)
    old = report(); old['shadow_profile'].pop('btc_pullback_entry_policy')
    with patch.object(execution, 'apply_llm_trade_gate', lambda r,t,**kw:(t, {'enabled': False})):
        assert bot.reconcile(desired('short'), 1000, report=old, now_ms=ORIGIN)['fill']
    with patch.object(execution, 'apply_llm_trade_gate', side_effect=AssertionError('Wait should precede review')):
        result = bot.reconcile(desired('long'), 1000, report=report(), now_ms=ORIGIN+1000)
    assert result['fill'] and bot.load()['position_qty'] == 0
    assert result['llm_trade_gate']['status'] == 'waiting_for_pullback'


def test_flat_cancels_pending_permanently_but_new_hour_source_is_eligible(tmp_path):
    bot = account(tmp_path)
    bot.reconcile(desired(), 1000, report=report(), now_ms=ORIGIN+1000)
    bot.reconcile(desired(target_leverage=0), 1000, report=report(), now_ms=ORIGIN+2000)
    returned = bot.reconcile(desired(), 1003,
        report=report([five()], ORIGIN+entry.FIVE_MINUTES), now_ms=ORIGIN+entry.FIVE_MINUTES)
    assert returned['fill'] is None and returned['btc_pullback_entry']['status'] == 'cancelled'
    next_origin = ORIGIN+entry.HOUR
    fresh = desired(origin_signal_time_ms=next_origin, position_id='new', signal_time_ms=next_origin)
    updated = report(now=next_origin+1000)
    for row in updated['btc_pullback_market_data']['hourly']:
        row[0] += entry.HOUR; row[7] += entry.HOUR
    following = bot.reconcile(fresh, 1000, report=updated, now_ms=next_origin+1000)
    assert following['btc_pullback_entry']['status'] == 'waiting_retrace'


def test_filled_source_cannot_reopen_after_flat_report(tmp_path):
    bot = account(tmp_path)
    now = ORIGIN+entry.FIVE_MINUTES
    good = report([five()], now)
    assert bot.reconcile(desired(), 1003, report=good, now_ms=now)['fill']
    assert bot.reconcile(desired(target_leverage=0), 1003, report=good, now_ms=now+1000)['fill']
    reopened = bot.reconcile(desired(), 1003, report=good, now_ms=now+2000)
    assert reopened['fill'] is None and reopened['btc_pullback_entry']['status'] == 'cancelled'
    assert 'btc_pullback_source_consumed' in reopened['btc_pullback_entry']['reasons']


def test_missing_market_can_recover_without_blocking_exits_or_existing_management(tmp_path):
    bot = account(tmp_path)
    missing = report(); missing['btc_pullback_market_data'] = {'complete': False}
    assert bot.reconcile(desired(), 1003, report=missing, now_ms=ORIGIN+1000)['fill'] is None
    now = ORIGIN+entry.FIVE_MINUTES
    assert bot.reconcile(desired(), 1003, report=report([five()], now), now_ms=now)['fill']
    add = bot.reconcile(desired(target_leverage=.3), 1003, report=missing, now_ms=now+1000)
    assert add['fill'] and add['btc_pullback_entry']['status'] == 'existing_direction'
    close = bot.reconcile(desired(target_leverage=0), 1003, report=missing, now_ms=now+2000)
    assert close['fill'] and bot.load()['position_qty'] == 0


def test_invalid_policy_does_not_stop_account_hard_exit(tmp_path):
    bot = account(tmp_path)
    now = ORIGIN+entry.FIVE_MINUTES
    assert bot.reconcile(desired(), 1003, report=report([five()], now), now_ms=now)['fill']
    state = bot.load(); state['max_drawdown_pct'] = 16; bot._save(state)
    invalid = report(); invalid['shadow_profile']['btc_pullback_entry_policy'] = {'mode': 'bad'}
    result = bot.reconcile(desired('short'), 1003, report=invalid, now_ms=now+1000)
    assert result['fill'] and result['fill']['reason'] == 'account_hard_stop'
    assert bot.load()['position_qty'] == 0


def test_ledger_work_cannot_use_stale_mark_for_new_entry(tmp_path, monkeypatch):
    bot = account(tmp_path)
    now = ORIGIN+entry.FIVE_MINUTES
    elapsed = {'seconds': 0}
    monkeypatch.setattr(execution.time, 'monotonic', lambda: elapsed['seconds'])
    original = entry.advance_pending
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs); elapsed['seconds'] += 61
        return result
    monkeypatch.setattr(entry, 'advance_pending', delayed)
    with pytest.raises(ValueError, match='stale'):
        bot.reconcile(desired(), 1003, report=report([five()], now), now_ms=now,
                      mark_time_ms=now, execution_clock=lambda: now)
    assert bot.load()['fill_count_total'] == 0


def test_final_validation_crossing_five_minute_close_defers_entry(tmp_path, monkeypatch):
    bot = account(tmp_path)
    now = ORIGIN+2*entry.FIVE_MINUTES-2
    elapsed = {'seconds': 0, 'calls': 0}
    monkeypatch.setattr(execution.time, 'monotonic', lambda: elapsed['seconds'])
    original = entry.advance_pending
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs); elapsed['calls'] += 1
        if elapsed['calls'] == 3: elapsed['seconds'] = .005
        return result
    monkeypatch.setattr(entry, 'advance_pending', delayed)
    result = bot.reconcile(desired(), 1003, report=report([five()], now), now_ms=now,
        mark_time_ms=now, execution_clock=lambda: now,
        funding_fetch_through_ms=now, next_funding_time_ms=now+100_000)
    assert result['fill'] is None and result['btc_pullback_entry']['status'] == 'data_unavailable'
    assert 'btc_pullback_5m_gap' in result['btc_pullback_entry']['reasons']


def test_review_can_expire_original_window_without_creating_a_new_entry(tmp_path, monkeypatch):
    bot = account(tmp_path, allow_llm=True)
    now = ORIGIN+entry.FIVE_MINUTES
    elapsed = {'seconds': 0}
    monkeypatch.setattr(execution.time, 'monotonic', lambda: elapsed['seconds'])
    def late_review(report, target, **kwargs):
        elapsed['seconds'] = 3301
        return target, {'enabled': True, 'status': 'approved'}
    monkeypatch.setattr(execution, 'apply_llm_trade_gate', late_review)
    result = bot.reconcile(desired(), 1003, report=report([five()], now), now_ms=now,
                          execution_clock=lambda: now)
    assert result['fill'] is None and result['btc_pullback_entry']['status'] == 'expired'
    assert bot.load()['position_qty'] == 0
