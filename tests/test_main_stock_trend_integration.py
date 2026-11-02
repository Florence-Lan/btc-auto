"""Offline real-book coverage of the activated main-account trend repair."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import main_stock_runtime as runtime
import main_stock_trend as trend
import run_parallel_simulation as paper
import stock_swing_signals as signals
import trade_profit_budget as budget
from test_main_profit_budget_integration import worker as budget_worker
from test_main_stock_runtime import HOUR, unavailable


def sector(direction=-1):
    return {'status': {1: 'bullish', -1: 'bearish', 0: 'neutral', None: 'unknown'}[direction],
            'direction': direction, 'reasons': ['fixture_unavailable'] if direction is None else []}


def worker(tmp_path, monkeypatch, *, activation=599 * HOUR, direction=-1):
    original, clock, bars = budget_worker(tmp_path, monkeypatch)
    cfg = {**original.config, trend.POLICY_KEY: deepcopy(trend.POLICY)}
    state = paper.read_json(original.path)
    state.update(rule=cfg, main_trend_active_after_ms=activation)
    paper.write_json(original.path, state)
    result = runtime.MainStockAccount(original.path, original.venue, original.symbol, cfg)
    monkeypatch.setattr(result, 'shared_clock_loader', unavailable)
    clock['sector_direction'] = direction
    monkeypatch.setattr(result, 'sector_loader', lambda *_a, **_k: sector(clock['sector_direction']))
    return result, clock, bars


def holding_worker(tmp_path, monkeypatch, *, activation=598 * HOUR, direction=None):
    account, clock, bars = worker(tmp_path, monkeypatch, activation=activation, direction=direction)
    # A smooth completed-hour recovery opposes the seeded legacy short.  The
    # live mark is unchanged from entry, so neither price stop nor target wins.
    bars[:] = [signals.Candle((100 + i) * HOUR, 100 + i * .02,
                 100 + i * .02 + .005, 100 + i * .02 - .005,
                 100 + i * .02, 10000.) for i in range(500)]
    clock['price'] = bars[-1].close
    state = paper.read_json(account.path)
    legacy = {key: deepcopy(value) for key, value in account.config.items()
              if key not in (trend.POLICY_KEY, budget.POLICY_KEY)}
    legacy.update(risk_fraction_per_trade=.0025, portfolio_gross_notional_fraction=1.)
    position = {'entry': clock['price'], 'qty': 1., 'direction': -1,
        'entry_time': 597 * HOUR + 10_000, 'signal_time': 597 * HOUR,
        'stop': 130., 'initial_stop': 130., 'entry_fee': 0., 'margin': 11.,
        'funding': 0., 'atr': 1., 'best_close_return': 0.,
        'last_trail_bar_ms': 598 * HOUR - 1}
    state.update(position=position, position_qty=-1.,
        trade_profit_budget_legacy_position_rule=legacy,
        position_history=[{'time_ms': position['entry_time'], 'signed_qty': -1.}],
        last_signal_time_ms=597 * HOUR, next_funding_ms=clock['now'] + HOUR)
    paper.write_json(account.path, state)
    return account, clock, bars, deepcopy(position)


@pytest.mark.parametrize('direction', [1, None])
def test_sector_opposition_or_unknown_cannot_book_new_short(tmp_path, monkeypatch, direction):
    account, _, _ = worker(tmp_path, monkeypatch, direction=direction)
    result = account.step()
    assert result['position'] is None
    assert result['fill_count_total'] == 0 and result['fees_paid'] == 0
    assert result['wallet_balance'] == 1000
    assert result['entry_blockers']


def test_sector_must_be_rechecked_before_final_booking(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    original = account.entry_fill_admission
    observed = []
    def flip_at_fill(*args, **kwargs):
        observed.append(True)
        clock['sector_direction'] = 1
        return original(*args, **kwargs)
    monkeypatch.setattr(account, 'entry_fill_admission', flip_at_fill)
    result = account.step()
    assert observed, 'The real engine must reach final admission before this check can pass.'
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['wallet_balance'] == 1000 and result['fees_paid'] == 0


def test_sector_can_veto_at_the_last_commit_gate_after_budget_passed(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    phases = []
    def last_moment_change(timestamp, *, phase='observation'):
        phases.append(phase)
        return sector(1 if phase == 'fill' else -1)
    monkeypatch.setattr(account, 'sector_loader', last_moment_change)
    result = account.step()
    assert {'retry', 'fill_admission', 'fill'} <= set(phases)
    assert result['entry_profit_budget']['actual_fill_checked']
    assert result['entry_profit_budget']['allowed']
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['wallet_balance'] == 1000 and result['fees_paid'] == 0


def test_same_direction_still_requires_real_profit_budget_and_entry_checks(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    result = account.step()
    assert result['position']['direction'] == -1 and result['fill_count_total'] == 1
    assert result['entry_profit_budget']['actual_fill_checked']
    assert result['entry_profit_budget']['expected_net_profit_usdt'] >= 100
    assert result['entry_profit_budget']['planned_loss_usdt'] <= 25
    assert result['stock_entry_timing']['allowed']


def test_price_led_long_can_book_through_the_real_engine_with_all_original_gates(tmp_path, monkeypatch):
    account, clock, bars = worker(tmp_path, monkeypatch, direction=1)
    bars[:] = [replace(bar, open=300 - bar.open, high=300 - bar.low,
                       low=300 - bar.high, close=300 - bar.close) for bar in bars]
    clock['price'] = bars[-1].close
    result = account.step()
    assert result['position']['direction'] == 1 and result['fill_count_total'] == 1
    assert result['long_confirmation']['admitted_by'] == 'price_led_long'
    assert result['entry_profit_budget']['actual_fill_checked']
    assert result['entry_profit_budget']['expected_net_profit_usdt'] >= 100
    assert result['entry_profit_budget']['planned_loss_usdt'] <= 25
    assert result['stock_entry_timing']['allowed']


def test_legacy_short_uses_new_technical_exit_without_a_sector_quote(tmp_path, monkeypatch):
    account, _, _, _ = holding_worker(tmp_path, monkeypatch)
    result = account.step()
    assert result['position'] is None
    assert result['fill_count_total'] == 1
    assert 'trend' in result['fills'][-1]['reason']
    assert result['rule'][budget.POLICY_KEY] == budget.POLICY
    assert result['rule']['risk_fraction_per_trade'] == .025
    assert account.config == result['rule']


def test_activation_requires_two_new_closed_hours_and_preserves_legacy_inventory(tmp_path, monkeypatch):
    account, clock, bars, old_position = holding_worker(tmp_path, monkeypatch, activation=599 * HOUR)
    first = account.step()
    assert first['fill_count_total'] == 0 and first['position'] is not None
    for key in ('entry', 'qty', 'entry_time', 'margin', 'initial_stop', 'stop'):
        assert first['position'][key] == old_position[key]
    assert 'trade_profit_budget' not in first['position']
    clock['now'] = 601 * HOUR + 60_000
    bars.append(replace(bars[-1], time_ms=600 * HOUR,
                        open=110., high=110.005, low=109.995, close=110.))
    clock['price'] = 110.
    closed = account.step()
    assert closed['position'] is None and closed['fill_count_total'] == 1
    assert 'trend' in closed['fills'][-1]['reason']


def test_partial_trend_exit_remains_pending_until_inventory_is_flat(tmp_path, monkeypatch):
    account, clock, bars, _ = holding_worker(tmp_path, monkeypatch)
    get = account.venue.get.side_effect
    # The original execution model consumes at most 10% of visible depth.
    depth = {'quantity': '4'}
    def thin(endpoint, params=None):
        value = get(endpoint, params)
        if endpoint == 'depth':
            value['asks'][0][1] = depth['quantity']
        return value
    account.venue.get.side_effect = thin
    partial = account.step()
    assert partial['position']['qty'] == pytest.approx(.6)
    pending = partial['position']['pending_exit']
    assert 'trend' in pending and partial['fill_count_total'] == 1
    # Once observed, the exit must survive a missing hourly feed and sector.
    depth['quantity'] = '1000'
    clock.update(now=clock['now'] + 30_000, hourly_error=True, sector_direction=None)
    closed = account.step()
    assert closed['position'] is None and closed['fill_count_total'] == 2
    assert closed['fills'][-1]['reason'] == pending


@pytest.mark.parametrize('sector_failure', ['unknown', 'exception'])
def test_sector_clock_and_hourly_failure_cannot_disable_original_stop(tmp_path, monkeypatch, sector_failure):
    account, clock, _ = worker(tmp_path, monkeypatch)
    entered = account.step()
    assert entered['position'] is not None
    clock.update(price=entered['position']['stop'] + 1, hourly_error=True, sector_direction=None)
    if sector_failure == 'exception':
        def failed_sector(*_a, **_k):
            raise OSError('Offline observer file unavailable')
        monkeypatch.setattr(account, 'sector_loader', failed_sector)
    get = account.venue.get.side_effect
    def broken(endpoint, params=None):
        if endpoint == 'time':
            raise RuntimeError('Offline clock unavailable')
        return get(endpoint, params)
    account.venue.get.side_effect = broken
    stopped = account.step()
    assert stopped['position'] is None and stopped['fill_count_total'] == 2
    assert stopped['fills'][-1]['reason'] == 'protective_stop'


def test_malformed_sector_result_cannot_interrupt_protective_stop_commit(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch)
    entered = account.step()
    assert entered['position'] is not None
    clock['price'] = entered['position']['stop'] + 1
    monkeypatch.setattr(account, 'sector_loader', lambda *_a, **_k: None)
    stopped = account.step()
    assert stopped['position'] is None and stopped['fill_count_total'] == 2
    assert stopped['fills'][-1]['reason'] == 'protective_stop'
    assert paper.read_json(account.path)['fill_count_total'] == 2


def test_equal_activation_boundary_cannot_reuse_the_old_raw_short(tmp_path, monkeypatch):
    account, clock, _ = worker(tmp_path, monkeypatch, activation=600 * HOUR)
    first = account.step()
    assert first['position'] is None and first['fill_count_total'] == 0
    clock['now'] += 30_000
    retried = account.step()
    assert retried['position'] is None and retried['fill_count_total'] == 0


def test_running_account_rejects_silent_trend_activation_replacement(tmp_path, monkeypatch):
    account, _, _ = worker(tmp_path, monkeypatch)
    state = paper.read_json(account.path)
    state['main_trend_active_after_ms'] += 1
    paper.write_json(account.path, state)
    with pytest.raises(ValueError, match='activation_changed'):
        account.step()


def test_sector_recovery_cannot_extend_expired_pending_signal(tmp_path, monkeypatch):
    account, clock, bars = worker(tmp_path, monkeypatch, direction=None)
    first = account.step()
    pending = first.get('pending_signal')
    clock.update(now=601 * HOUR + 60_000, sector_direction=-1)
    # The provider continues returning the old closed-hour prefix.  A newly
    # available sector quote cannot make the old breakout fresh again.
    second = account.step()
    assert second['position'] is None and second['fill_count_total'] == 0
    if pending is not None:
        assert second.get('pending_signal') is None


def test_holding_trend_diagnostic_recovers_after_hourly_feed_outage(tmp_path, monkeypatch):
    account, clock, _, _ = holding_worker(tmp_path, monkeypatch, activation=600 * HOUR + 1)
    clock['hourly_error'] = True
    unavailable_state = account.step()
    assert unavailable_state['position'] is not None
    assert unavailable_state['current_trend']['status'] != 'healthy'
    clock.update(now=clock['now'] + 30_000, hourly_error=False)
    recovered = account.step()
    assert recovered['position'] is not None and recovered['fill_count_total'] == 0
    assert recovered['current_trend']['status'] == 'healthy'
    assert recovered['current_trend']['signal_close_ms'] == 600 * HOUR
    assert recovered['current_trend']['confirmation_close_ms'] == 600 * HOUR
    assert recovered['current_trend']['hourly']['direction'] == 1


def test_fresh_intrabar_gain_protects_legacy_short_without_retroactive_fill(tmp_path, monkeypatch):
    account, clock, _, position = holding_worker(tmp_path, monkeypatch, activation=600 * HOUR + 1)
    # This MU fixture retains MU's original 8% activation / 3% lock.  SNDK's
    # thresholds differ; the overlay must respect whichever old rule it holds.
    clock['price'] = position['entry'] * .91
    protected = account.step()
    assert protected['position'] is not None and protected['fill_count_total'] == 0
    assert protected['main_sector_status']['status'] == 'unknown'
    assert protected['position']['stop'] <= position['entry'] * .97
    assert protected['position']['stop'] < position['stop']
    assert protected['position']['qty'] == position['qty']
    # A later current quote crosses the newly observed stop.  The engine books
    # this later market price, never the earlier unobserved profit excursion.
    clock.update(now=clock['now'] + 30_000, price=position['entry'], hourly_error=True)
    exited = account.step()
    assert exited['position'] is None and exited['fill_count_total'] == 1
    assert exited['fills'][-1]['reason'] == 'protective_stop'
    assert exited['fills'][-1]['price'] > protected['position']['stop']


def test_thin_book_cannot_arm_intrabar_profit_from_an_unfillable_full_quantity(tmp_path, monkeypatch):
    account, clock, _, position = holding_worker(tmp_path, monkeypatch, activation=600 * HOUR + 1)
    clock['price'] = position['entry'] * .91
    get = account.venue.get.side_effect
    def thin(endpoint, params=None):
        value = get(endpoint, params)
        if endpoint == 'depth':
            value['asks'][0][1] = '1'
        return value
    account.venue.get.side_effect = thin
    result = account.step()
    assert result['position'] is not None and result['fill_count_total'] == 0
    assert result['position']['stop'] == position['stop']
    assert not (result['position'].get('main_trend_profit_protection') or {}).get('armed')
