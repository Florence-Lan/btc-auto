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
    account.four = [[boundary-runner.FOUR_HOURS,100,101,99,100,1000,boundary-1]]
    account.five = [[boundary-runner.FIVE_MINUTES,100,101,99,100,1000,boundary-1]]
    original = account.venue.get.side_effect
    account.venue.get.side_effect = lambda e,p=None: [] if e == 'fundingRate' else original(e,p)
    signal_method = Mock(side_effect=AssertionError('Pre-start signal must not be evaluated'))
    monkeypatch.setattr(runner.stock_swing_profiles, 'signal_at', signal_method)
    result = account.step()
    assert result['wallet_balance'] == 1000
    signal_method.assert_not_called()
