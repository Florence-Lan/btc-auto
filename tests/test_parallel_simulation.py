from __future__ import annotations
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as runner


def test_visible_depth_partial_and_adverse_prices():
    book = {'T': 123, 'asks': [['100', '1'], ['100.1', '2']], 'bids': [['99.9', '3']]}
    buy = runner.book_fill(book, 1, 0.01)
    assert buy['qty'] == 0.3
    assert buy['partial']
    assert buy['price'] > 100
    sell = runner.book_fill(book, -1, 0.01)
    assert sell['qty'] == 0.3
    assert sell['price'] < 99.9


def test_never_fabricate_liquidity_beyond_visible_band():
    book = {'T': 123, 'asks': [['100', '0.09'], ['110', '100']], 'bids': [['99', '1']]}
    assert runner.book_fill(book, 1, 0.01)['qty'] == 0


def test_stale_or_future_observation_blocks():
    for timestamp in (1, 100_001):
        with pytest.raises(ValueError):
            runner.fresh(timestamp, 90_000)


def test_funding_idempotent_uses_inventory_at_settlement(tmp_path):
    venue = Mock()
    path = tmp_path / 'state.json'
    state = runner.initial_stock('MUUSDT', 1000, {})
    state['position_history'] = [
        {'time_ms': 1100, 'signed_qty': 2}, {'time_ms': 2100, 'signed_qty': 0}]
    events = [{'fundingTime': 2000, 'fundingRate': '0.01', 'markPrice': '100'},
              {'fundingTime': 2500, 'fundingRate': '0.01', 'markPrice': '100'}]
    account = runner.StockAccount(path, venue, 'MUUSDT', {})
    account.funding(state, events, 3000)
    account.funding(state, events, 3000)
    assert state['wallet_balance'] == 998
    assert state['funding_pnl'] == -2
    assert len(state['settled_funding']) == 2
    venue.get.assert_not_called()


def test_same_timestamp_entry_receives_no_prior_funding(tmp_path):
    state = runner.initial_stock('MUUSDT', 1000, {})
    state['position_history'] = [{'time_ms': 2000, 'signed_qty': 2}]
    account = runner.StockAccount(tmp_path / 'state.json', Mock(), 'MUUSDT', {})
    account.funding(state, [{'fundingTime': 2000, 'fundingRate': '0.01', 'markPrice': '100'}], 3000)
    assert state['funding_pnl'] == 0


def test_bootstrap_common_epoch_and_preserves_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    plan = {'plan_id': 'trial', 'accounts': [
        {'account_id': 'btc', 'state_path': 'data/parallel_simulation/trial/btc/state.json'}]}
    for name in ('run_parallel_simulation.py', 'stock_swing_signals.py', 'stock_swing_profiles.py', 'trading_execution.py'):
        p = tmp_path / 'scripts' / name
        p.parent.mkdir(exist_ok=True)
        p.write_text('frozen')
    root, manifest = runner.bootstrap(plan)
    p = tmp_path / plan['accounts'][0]['state_path']
    state = json.loads(p.read_text())
    assert state['created_at_utc'] == manifest['start_utc']
    assert state['wallet_balance'] == 1000
    state['wallet_balance'] = 998
    runner.write_json(p, state)
    _, again = runner.bootstrap(plan)
    assert manifest == again
    assert json.loads(p.read_text())['wallet_balance'] == 998
    p.unlink()
    with pytest.raises(RuntimeError):
        runner.bootstrap(plan)


def stock_fixture(tmp_path, monkeypatch, opened=False):
    stamp = 10 * runner.FOUR_HOURS + 100_000
    monkeypatch.setattr(runner, 'now_ms', lambda: stamp)
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'config/stock_swing_120_candidate_20261004.json').read_text())
    path = tmp_path / 'state.json'
    state = runner.initial_stock('MUUSDT', stamp - 10_000, cfg)
    if opened:
        state['position'] = {'entry': 100, 'qty': 1, 'direction': 1, 'entry_time': stamp - 5000,
            'signal_time': stamp - 5000, 'stop': 95, 'entry_fee': 0, 'margin': 10, 'funding': 0}
        state['position_history'] = [{'time_ms': stamp - 5000, 'signed_qty': 1}]
    runner.write_json(path, state)
    rules = {'symbol': 'MUUSDT', 'status': 'TRADING', 'filters': [
        {'filterType':'LOT_SIZE', 'stepSize':'0.01', 'minQty':'0.01', 'maxQty':'100'},
        {'filterType':'PRICE_FILTER', 'tickSize':'0.01'},
        {'filterType':'MIN_NOTIONAL', 'notional':'5'}]}
    def get(endpoint, params=None):
        if endpoint == 'time': return {'serverTime': stamp}
        if endpoint == 'premiumIndex': return {'time': stamp, 'markPrice':'94', 'indexPrice':'94',
            'lastFundingRate':'0', 'nextFundingTime':stamp+1_000_000}
        if endpoint == 'depth': return {'T':stamp, 'bids':[['94','100']], 'asks':[['94.1','100']]}
        if endpoint == 'exchangeInfo': return {'symbols':[rules]}
        raise RuntimeError('Source unavailable')
    venue = Mock()
    venue.get.side_effect = get
    return runner.StockAccount(path, venue, 'MUUSDT', cfg), path


def test_protective_exit_continues_when_signals_and_funding_unavailable(tmp_path, monkeypatch):
    account, path = stock_fixture(tmp_path, monkeypatch, opened=True)
    result = account.step()
    assert result['status'] == 'degraded'
    assert result['position'] is None
    assert result['fill_count_total'] == 1
    assert result['fills'][0]['reason'] == 'protective_stop'
    assert result['wallet_balance'] == pytest.approx(1000 + result['realized_pnl'] - result['fees_paid'])
    assert result['fills'][0]['observed_trigger_to_fill_ms'] == 0


def test_missing_sources_block_new_entries_and_preserve_capital(tmp_path, monkeypatch):
    account, path = stock_fixture(tmp_path, monkeypatch)
    result = account.step()
    assert result['status'] == 'degraded'
    assert result['position'] is None
    assert result['wallet_balance'] == 1000
    assert not result['fills']


def test_source_before_common_start_is_only_warmup(tmp_path, monkeypatch):
    account, path = stock_fixture(tmp_path, monkeypatch)
    account.candle_bucket = runner.now_ms() // runner.FIVE_MINUTES
    account.rules = {'status':'TRADING', 'filters': [
        {'filterType':'LOT_SIZE', 'stepSize':'0.01', 'minQty':'0.01', 'maxQty':'100'},
        {'filterType':'PRICE_FILTER', 'tickSize':'0.01'}]}
    account.rules_at_ms = runner.now_ms()
    boundary = runner.now_ms() // runner.FOUR_HOURS * runner.FOUR_HOURS
    account.signal_bars = [[boundary-runner.FOUR_HOURS,100,101,99,100,1000,boundary-1]]
    account.five = [[boundary-runner.FIVE_MINUTES,100,101,99,100,1000,boundary-1]]
    original = account.venue.get.side_effect
    account.venue.get.side_effect = lambda e,p=None: [] if e == 'fundingRate' else original(e,p)
    signal_method = Mock(side_effect=AssertionError('Pre-start signal must not be evaluated'))
    monkeypatch.setattr(runner.stock_swing_profiles, 'signal_at', signal_method)
    result = account.step()
    assert result['wallet_balance'] == 1000
    signal_method.assert_not_called()


def fast_stock_fixture(tmp_path, monkeypatch):
    interval = runner.SIGNAL_INTERVALS['15m']
    boundary = 200 * interval
    clock = {'now': boundary + 60_000, 'volume': 0, 'funding_error': False, 'stale_signal': False}
    monkeypatch.setattr(runner, 'now_ms', lambda: clock['now'])
    # Advance both clock domains together in deterministic execution fixtures.
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['now'] / 1000)
    plan = json.loads(runner.PLAN.read_text())
    cfg = runner.stock_config(plan, 'MUUSDT')
    cfg = {**cfg, 'entry_enabled': True}  # Execution fixtures also exercise unqualified research signals.
    path = tmp_path / 'state.json'
    runner.write_json(path, runner.initial_stock('MUUSDT', boundary - 10_000, cfg))
    rules = {'symbol': 'MUUSDT', 'status': 'TRADING', 'filters': [
        {'filterType': 'LOT_SIZE', 'stepSize': '0.01', 'minQty': '0.01', 'maxQty': '100'},
        {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
        {'filterType': 'MIN_NOTIONAL', 'notional': '5'}]}
    def get(endpoint, params=None):
        now = clock['now']
        if endpoint == 'time': return {'serverTime': now}
        if endpoint == 'premiumIndex': return {'time': now, 'markPrice': '100', 'indexPrice': '100',
            'lastFundingRate': '0', 'nextFundingTime': now + interval}
        if endpoint == 'depth': return {'T': now, 'bids': [['99.99', '100']], 'asks': [['100.01', '100']]}
        if endpoint == 'exchangeInfo': return {'symbols': [rules]}
        if endpoint == 'fundingRate':
            if clock['funding_error']: raise RuntimeError('Funding source unavailable')
            return []
        if endpoint == 'klines':
            span = runner.SIGNAL_INTERVALS[params['interval']]
            end = now // span * span
            if params['interval'] == '15m' and clock['stale_signal']: end -= span
            volume = clock['volume'] if params['interval'] == '5m' else 1000
            # Include the open candle; it must never reach the signal function.
            return [[end - i * span, 100, 101, 99, 100, volume, end - (i-1) * span - 1]
                    for i in range(150, 0 if params['interval'] == '15m' and clock['stale_signal'] else -1, -1)]
        raise AssertionError(endpoint)
    venue = Mock()
    venue.get.side_effect = get
    observed = Mock(return_value=runner.signals.Signal(1, 1, 100, 1, boundary - interval))
    monkeypatch.setattr(runner.stock_swing_profiles, 'signal_at', observed)
    return runner.StockAccount(path, venue, 'MUUSDT', cfg), path, clock, observed


def test_15m_closed_signal_retries_after_five_minutes_and_survives_restart(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    first = account.step()
    assert first['pending_signal'] and first['fill_count_total'] == 0
    assert observed.call_args[0][0][-1].time_ms == 199 * 900_000
    assert all(call.args[1]['interval'] in ('15m', '5m')
               for call in account.venue.get.call_args_list if call.args[0] == 'klines')
    clock['now'] += 9 * 60_000
    clock['volume'] = 1000
    # A process restart preserves the eligible signal instead of consuming it twice.
    restarted = runner.StockAccount(path, account.venue, 'MUUSDT', account.config)
    result = restarted.step()
    assert result['position'] and result['fill_count_total'] == 1
    assert result['fills'][0]['reason'] == 'fresh_closed_15m_signal'
    assert result['pending_signal'] is None
    assert result['wallet_balance'] == pytest.approx(1000 - result['fees_paid'])
    clock['now'] += 30_000
    again = restarted.step()
    assert again['fill_count_total'] == 1
    assert observed.call_count == 1


def test_expired_signal_cannot_enter_using_stale_candles(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    first = account.step()
    clock['now'] = first['pending_signal']['expires_at_ms']
    clock['volume'] = 1000
    clock['stale_signal'] = True
    result = account.step()
    assert result['pending_signal'] is None
    assert result['position'] is None
    assert result['fill_count_total'] == 0
    assert observed.call_count == 1


def test_revised_zero_volume_rechecked_in_same_bucket_without_restart(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    first = account.step()
    assert first['entry_blockers'] == ['prior_5m_volume']
    original_bucket = account.candle_bucket
    clock.update(now=clock['now'] + 30_000, volume=1000)
    result = account.step()
    assert account.candle_bucket == original_bucket
    assert result['fill_count_total'] == 1
    assert result['entry_checks']['prior_5m_volume'] == 1000
    assert result['entry_blockers'] == []
    assert observed.call_count == 1
    account.step()
    assert json.loads(path.read_text())['fill_count_total'] == 1


def test_zero_volume_remains_blocked_and_open_candle_is_not_liquidity(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    original = account.venue.get.side_effect
    def open_volume(endpoint, params=None):
        result = original(endpoint, params)
        if endpoint == 'klines' and params['interval'] == '5m':
            result[-1][5] = 1000  # Only the still-open candle has volume.
        return result
    account.venue.get.side_effect = open_volume
    for _ in range(2):
        result = account.step()
        assert result['entry_blockers'] == ['prior_5m_volume']
        assert result['fill_count_total'] == 0
        clock['now'] += 30_000


@pytest.mark.parametrize('direction', [1, -1])
def test_wide_spread_blocks_both_entries_and_pending_signal_can_retry(tmp_path, monkeypatch, direction):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    account.config['max_entry_spread_fraction'] = .001
    clock['volume'] = 1000
    observed.return_value = runner.signals.Signal(direction, 1, 100, 1, 199 * 900_000)
    original = account.venue.get.side_effect
    wide = {'enabled': True}
    def quotes(endpoint, params=None):
        if endpoint == 'depth' and wide['enabled']:
            return {'T': clock['now'], 'bids': [['99.8', '100']], 'asks': [['100.2', '100']]}
        return original(endpoint, params)
    account.venue.get.side_effect = quotes
    blocked = account.step()
    assert blocked['entry_blockers'] == ['bid_ask_spread']
    assert blocked['entry_checks']['spread_fraction'] == pytest.approx(.004)
    assert blocked['fill_count_total'] == 0
    assert blocked['wallet_balance'] == 1000
    assert blocked['pending_signal'] is not None
    wide['enabled'] = False
    clock['now'] += 30_000
    entered = account.step()
    assert entered['fill_count_total'] == 1
    assert entered['position']['direction'] == direction


def test_entry_spread_guard_never_prevents_protective_exit(tmp_path, monkeypatch):
    account, path = stock_fixture(tmp_path, monkeypatch, opened=True)
    account.config['max_entry_spread_fraction'] = .0001
    # Existing fixture bid 94 / ask 94.1 is wider than this entry limit.
    result = account.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'


@pytest.mark.parametrize('failure', ['outage', 'empty'])
def test_failed_liquidity_refresh_never_uses_cached_positive_volume(tmp_path, monkeypatch, failure):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    clock.update(volume=1000, funding_error=True)
    first = account.step()
    assert first['pending_signal'] and first['fill_count_total'] == 0
    clock.update(now=clock['now'] + 30_000, funding_error=False)
    original = account.venue.get.side_effect
    def failed_refresh(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '5m':
            if failure == 'empty': return []
            raise RuntimeError('Volume source unavailable')
        return original(endpoint, params)
    account.venue.get.side_effect = failed_refresh
    result = account.step()
    assert result['status'] == 'degraded'
    assert 'five' in result['entry_blockers']
    assert result['pending_signal'] and result['fill_count_total'] == 0
    assert result['wallet_balance'] == 1000


@pytest.mark.parametrize('symbol', ['MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT'])
def test_active_stock_profiles_allow_both_directions(symbol):
    cfg = runner.stock_config(json.loads(runner.PLAN.read_text()), symbol)
    assert cfg['entry_direction'] == 'both'
    for direction in (-1, 1):
        assert runner.entry_policy.rejection(cfg, symbol, 1791189900000, direction) is None


def test_bidirectional_short_signal_enters(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    observed.return_value = runner.signals.Signal(-1, 1, 100, 1, clock['now'] - 960_000)
    clock['volume'] = 1000
    result = account.step()
    assert result['fill_count_total'] == 1
    assert result['position']['direction'] == -1
    assert result['fills'][0]['side'] == 'SELL'


def test_unqualified_strategy_observes_signal_without_opening(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    account.config = {**account.config, 'entry_enabled': False}
    clock['volume'] = 1000
    result = account.step()
    assert result['pending_signal']
    assert result['signal_status'] == 'strategy_not_qualified'
    assert result['entry_blockers'] == ['strategy_not_qualified']
    assert result['fill_count_total'] == 0


def test_unqualified_strategy_still_executes_protective_exit(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    clock['volume'] = 1000
    state = account.step()
    state['position']['stop'] = 101
    runner.write_json(path, state)
    account.config = {**account.config, 'entry_enabled': False}
    clock['now'] += 30_000
    result = account.step()
    assert result['position'] is None
    assert result['fill_count_total'] == 2
    assert result['fills'][-1]['reason'] == 'protective_stop'


def test_new_bar_without_signal_cancels_prior_pending_entry(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    first = account.step()
    clock['now'] = first['pending_signal']['expires_at_ms'] + 1000
    clock['volume'] = 1000
    observed.return_value = None
    result = account.step()
    assert result['signal_status'] == 'no_signal'
    assert result['pending_signal'] is None
    assert result['fill_count_total'] == 0
    assert observed.call_count == 2
    clock['now'] += 30_000
    assert account.step()['signal_status'] == 'no_signal'
    assert observed.call_count == 2


def test_signal_expiring_during_entry_calculation_never_fills(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    first = account.step()
    clock.update(now=first['pending_signal']['expires_at_ms'] - 1000, volume=1000)
    original = runner.book_fill
    def delayed_fill(*args, **kwargs):
        result = original(*args, **kwargs)
        clock['now'] += 1000
        return result
    monkeypatch.setattr(runner, 'book_fill', delayed_fill)
    result = account.step()
    assert result['fill_count_total'] == 0
    assert result['position'] is None
    assert result['pending_signal'] is None
    assert result['signal_status'] == 'signal_expired'


def test_funding_outage_does_not_consume_15m_signal(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    clock.update(volume=1000, funding_error=True)
    first = account.step()
    assert first['status'] == 'degraded'
    assert first['pending_signal'] and first['fill_count_total'] == 0
    clock.update(now=clock['now'] + 30_000, funding_error=False)
    result = account.step()
    assert result['fill_count_total'] == 1
    assert observed.call_count == 1


def test_15m_revision_never_trades_pre_activation_candle(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    state = json.loads(path.read_text())
    state['signal_active_after_ms'] = clock['now'] - 10_000
    runner.write_json(path, state)
    clock['volume'] = 1000
    result = account.step()
    assert result['signal_status'] == 'waiting_for_first_forward_candle'
    assert result['fill_count_total'] == 0
    observed.assert_not_called()


def test_cooldown_tracks_three_15m_bars_after_exit(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    clock['volume'] = 1000
    state = account.step()
    state['position']['stop'] = 101  # Current observed bid triggers the protective exit.
    runner.write_json(path, state)
    clock['now'] += 30_000
    result = account.step()
    assert result['position'] is None
    assert result['cooldown_until_ms'] == clock['now'] + 45 * 60_000
    assert result['fill_count_total'] == 2
    assert result['wallet_balance'] == pytest.approx(1000 + result['realized_pnl'] - result['fees_paid'])


def test_rule_migration_preserves_ledger_and_records_revision_once(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    state = json.loads(path.read_text())
    state.update(wallet_balance=1005, equity=1005, funding_pnl=5, observations=300)
    state['rule']['signal_timeframe'] = '4h'
    runner.write_json(path, state)
    cfg = account.config
    monkeypatch.setattr(runner, 'stock_config', lambda plan, symbol: cfg)
    plan = {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': str(path)}]}
    manifest = {'start_ms': state['start_ms']}
    runner.synchronize_stock_rules(plan, tmp_path, manifest)
    after = json.loads(path.read_text())
    for key in ('wallet_balance', 'equity', 'funding_pnl', 'observations', 'start_ms', 'created_at_utc', 'fills'):
        assert after[key] == state[key]
    assert after['rule']['signal_timeframe'] == '15m'
    assert after['signal_active_after_ms'] == clock['now']
    backup = tmp_path / 'rule_revisions' / str(clock['now']) / 'mu' / 'previous_state.json'
    assert json.loads(backup.read_text()) == state
    runner.synchronize_stock_rules(plan, tmp_path, manifest)
    assert len(json.loads(path.read_text())['rule_history']) == 1
    assert len(manifest['rule_revisions']) == 1
