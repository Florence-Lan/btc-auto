"""Regressions for stopped-paper-account execution and worker reliability."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as runner
from public_request_deadline import PublicRequestDeadline


@pytest.mark.parametrize('side,level', [(1, 'asks'), (-1, 'bids')])
def test_rounded_ioc_price_excludes_cancelled_deeper_quantity(side, level):
    book = {'T': 123, level: [['100', '10'], [str(100 + side * .4), '5']]}
    fill = runner.book_fill(book, side * 2, 1, slip=.0002)
    assert fill['qty'] == 1
    assert fill['price'] == pytest.approx(100 * (1 + side * .0002))
    assert fill['partial']


@pytest.mark.parametrize('side', [1, -1])
def test_multiple_partial_closes_allocate_costs_once_and_reconcile_cash(tmp_path, side):
    config = {'adverse_slippage_fraction_assumption': .0002,
              'taker_fee_rate_assumption': .001, 'cooldown_signal_bars': 3,
              'signal_timeframe': '15m'}
    account = runner.StockAccount(tmp_path / 'state.json', Mock(), 'MUUSDT', config)
    account.now_ms = Mock(side_effect=[2000, 3000, 3000])
    state = runner.initial_stock('MUUSDT', 1000, config)
    position = {'entry': 100, 'qty': 2, 'direction': side, 'entry_time': 1100,
                'signal_time': 1000, 'entry_fee': .2, 'funding': .06, 'margin': 20}
    state.update(position=position, wallet_balance=999.74, fees_paid=.2, funding_pnl=-.06)
    level = 'bids' if side == 1 else 'asks'
    position = account.close(state, position, {'T': 2000, level: [['102', '3']]},
                             2, 'protective_stop', 1900, .01, .01)
    assert position['qty'] == pytest.approx(1.7)
    position = account.close(state, position, {'T': 3000, level: [['101', '100']]},
                             position['qty'], 'protective_stop', 1900, .01, .01)
    assert position is None
    rows = [json.loads(row) for row in (tmp_path / 'closed_trades.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['position_closed'] for row in rows] == [False, True]
    assert sum(row['allocated_entry_fee'] for row in rows) == pytest.approx(.2)
    assert sum(row['allocated_funding_debit'] for row in rows) == pytest.approx(.06)
    assert sum(row['net_pnl'] for row in rows) == pytest.approx(state['wallet_balance'] - 1000)
    assert state['wallet_balance'] == pytest.approx(
        1000 + state['realized_pnl'] - state['fees_paid'] + state['funding_pnl'])
    assert state['position_history'][-1]['signed_qty'] == 0


def curl_response(payload, status=200):
    return subprocess.CompletedProcess([], 0, json.dumps(payload) + f'\n{status}', '')


def test_aster_fallback_discovers_windows_curl_and_preserves_public_tls(monkeypatch):
    monkeypatch.setattr(runner.requests, 'get', Mock(side_effect=requests.ConnectionError('reset')))
    executable = r'C:\Windows\System32\curl.exe'
    monkeypatch.setattr(runner.shutil, 'which', lambda name: executable if name == 'curl.exe' else None)
    transport = Mock(return_value=curl_response({'serverTime': 1000}))
    monkeypatch.setattr(runner.subprocess, 'run', transport)
    assert runner.PublicAster().get('time') == {'serverTime': 1000}
    command = transport.call_args.args[0]
    assert command[0] == executable
    assert command[1] == '-q'
    assert command[command.index('--proto') + 1] == '=https'
    assert command[command.index('--url') + 1] == 'https://fapi.asterdex.com/fapi/v3/time'
    assert '--insecure' not in command and '-k' not in command
    assert transport.call_args.kwargs['timeout'] == 18


def test_aster_missing_curl_reports_transport_failure_without_file_not_found(monkeypatch):
    monkeypatch.setattr(runner.requests, 'get', Mock(side_effect=requests.Timeout('timeout')))
    monkeypatch.setattr(runner.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='fallback curl transport unavailable'):
        runner.PublicAster().get('time')


def test_aster_hung_requests_use_bounded_fallback_and_no_duplicate_pending_thread(monkeypatch):
    release, closed = threading.Event(), threading.Event()
    response = Mock(status_code=200)
    response.json.return_value = {'markPrice': 'expired'}
    response.close.side_effect = closed.set
    def request(*args, **kwargs):
        release.wait(2)
        return response
    primary = Mock(side_effect=request)
    monkeypatch.setattr(runner.requests, 'get', primary)
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'curl.exe')
    fallback = Mock(return_value=curl_response({'markPrice': 'fresh'}))
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    venue = runner.PublicAster()
    venue.deadline = PublicRequestDeadline(timeout_seconds=.02)
    try:
        for _ in range(2):
            assert venue.get('premiumIndex', {'symbol': 'MUUSDT'}) == {'markPrice': 'fresh'}
        assert primary.call_count == 1
        assert fallback.call_count == 2
    finally:
        release.set()
        assert closed.wait(1)


def test_parallel_accounts_share_one_inflight_exchange_clock(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    response = Mock(status_code=200)
    response.json.return_value = {'serverTime': 1000}
    def request(*args, **kwargs):
        entered.set()
        assert release.wait(1)
        return response
    primary = Mock(side_effect=request)
    monkeypatch.setattr(runner.requests, 'get', primary)
    fallback = Mock(side_effect=AssertionError('Concurrent public clock needs no fallback'))
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    venue = runner.PublicAster()
    results = {}
    first = threading.Thread(target=lambda: results.update(first=venue.get('time')), name='first-caller')
    first.start()
    assert entered.wait(1)
    with venue.deadline._lock:
        pending = venue.deadline._pending[('time', None, None)]
    original_wait = pending.done.wait
    def wait(timeout):
        # Release the source only after the second caller has joined this read.
        if threading.current_thread().name == 'second-caller':
            release.set()
        return original_wait(timeout)
    monkeypatch.setattr(pending.done, 'wait', wait)
    second = threading.Thread(target=lambda: results.update(second=venue.get('time')), name='second-caller')
    second.start()
    first.join(1)
    second.join(1)
    assert not first.is_alive() and not second.is_alive()
    assert results['first'] == results['second'] == {'serverTime': 1000}
    # Both normal polls share the read; no redundant transport or response close.
    assert primary.call_count == 1
    response.close.assert_called_once()
    fallback.assert_not_called()


def test_aster_non_json_rate_limit_still_enters_cooldown(monkeypatch):
    response = Mock(status_code=429)
    response.json.side_effect = ValueError('HTML rate-limit response')
    primary = Mock(return_value=response)
    monkeypatch.setattr(runner.requests, 'get', primary)
    venue = runner.PublicAster()
    with pytest.raises(ValueError, match='HTML rate-limit'):
        venue.get('time')
    with pytest.raises(RuntimeError, match='cooldown active'):
        venue.get('time')
    primary.assert_called_once()
    response.close.assert_called_once()


@pytest.mark.parametrize('transport', ['requests', 'curl'])
def test_aster_rate_limit_blocks_followup_without_another_transport(monkeypatch, transport):
    response = Mock(status_code=429)
    response.json.return_value = {'code': -1003}
    primary = Mock(return_value=response)
    if transport == 'curl':
        primary.side_effect = requests.ConnectionError('reset')
    monkeypatch.setattr(runner.requests, 'get', primary)
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'curl.exe')
    fallback = Mock(return_value=curl_response({'code': -1003}, 429))
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    venue = runner.PublicAster()
    with pytest.raises(RuntimeError, match='HTTP 429'):
        venue.get('time')
    with pytest.raises(RuntimeError, match='cooldown active'):
        venue.get('time')
    assert primary.call_count == 1
    assert fallback.call_count == (transport == 'curl')


def test_stalled_worker_is_degraded_without_refreshing_its_market_heartbeat():
    accounts = {'mu': {'status': 'healthy', 'checked_at_utc': 'old', 'errors': {}},
                'sndk': {'status': 'healthy', 'checked_at_utc': 'current', 'errors': {}}}
    original = copy.deepcopy(accounts)
    views = runner.heartbeat_views(accounts, {'mu': 0, 'sndk': 95}, 100, 30)
    assert views['mu']['status'] == 'degraded'
    assert views['mu']['checked_at_utc'] == 'old'
    assert views['mu']['heartbeat_stale']
    assert 'worker_heartbeat' in views['mu']['errors']
    assert views['sndk']['status'] == 'healthy'
    assert not views['sndk']['heartbeat_stale']
    assert accounts == original


def runner_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'STOP', threading.Event())
    monkeypatch.setattr(runner.signal, 'signal', Mock())
    account = {'account_id': 'btc', 'symbol': 'BTCUSDT', 'state_path': 'btc/state.json'}
    plan = {'plan_id': 'test', 'accounts': [account]}
    monkeypatch.setattr(runner, 'bootstrap', lambda plan: (tmp_path, {'start_utc': 'start'}))
    monkeypatch.setattr(runner, 'prepare_profit_comparison', lambda *args: None)
    monkeypatch.setattr(runner, 'synchronize_stock_rules', Mock())
    monkeypatch.setattr(runner, 'BinanceTerminalClient', Mock())
    return plan, SimpleNamespace(once=True, poll_seconds=30)


def test_status_file_sharing_failure_does_not_terminate_account_worker(tmp_path, monkeypatch):
    plan, args = runner_fixture(tmp_path, monkeypatch)
    original_write = runner.write_json
    calls = []
    def write(path, payload):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError('status temporarily opened by another app')
        original_write(path, payload)
    monkeypatch.setattr(runner, 'write_json', write)
    step = Mock(return_value={'symbol': 'BTCUSDT', 'status': 'healthy', 'equity': 1000})
    monkeypatch.setattr(runner, 'btc_step', step)
    runner._run_locked(plan, tmp_path, args)
    step.assert_called_once()
    status = json.loads((tmp_path / 'status.json').read_text(encoding='utf-8'))
    assert status['accounts']['btc']['status'] == 'healthy'
    assert not status['running']
    assert (tmp_path / 'runner_errors.jsonl').exists()


def test_coordinator_marks_a_hung_account_stale_before_its_worker_returns(tmp_path, monkeypatch):
    plan, args = runner_fixture(tmp_path, monkeypatch)
    args.once = False
    clock = {'monotonic': 0.0}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['monotonic'])
    release = threading.Event()
    class FastStop(threading.Event):
        def wait(self, timeout=None):
            return super().wait(.005 if timeout == 5 else timeout)
    stop = FastStop()
    monkeypatch.setattr(runner, 'STOP', stop)
    def blocked_worker(*args):
        clock['monotonic'] = 100.0
        assert release.wait(1)
        return {'symbol': 'BTCUSDT', 'status': 'healthy'}
    monkeypatch.setattr(runner, 'btc_step', blocked_worker)
    published = []
    def publish(path, payload):
        published.append(payload)
        if payload['accounts']['btc']['heartbeat_stale']:
            stop.set()
            release.set()
        return True
    monkeypatch.setattr(runner, 'publish_status', publish)
    runner._run_locked(plan, tmp_path, args)
    stalled = [row for row in published if row['accounts']['btc']['heartbeat_stale']]
    assert stalled and stalled[0]['running']
    assert stalled[0]['accounts']['btc']['status'] == 'degraded'
    assert stalled[0]['accounts']['btc']['checked_at_utc'] is None
    assert not published[-1]['running']


def test_fatal_worker_exception_publishes_stopped_state(tmp_path, monkeypatch):
    plan, args = runner_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'btc_step', Mock(side_effect=SystemExit('fatal test worker')))
    with pytest.raises(SystemExit, match='fatal test worker'):
        runner._run_locked(plan, tmp_path, args)
    status = json.loads((tmp_path / 'status.json').read_text(encoding='utf-8'))
    assert not status['running']
    assert status['accounts']['btc']['status'] == 'degraded'
    assert 'SystemExit' in status['accounts']['btc']['errors']['worker']
    assert (tmp_path / 'runner_errors.jsonl').exists()


@pytest.mark.parametrize('journal_unavailable', [False, True])
def test_corrupt_ledger_during_exception_reporting_preserves_file_and_worker_status(
        tmp_path, monkeypatch, journal_unavailable):
    plan, args = runner_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'btc/state.json'
    path.parent.mkdir()
    path.write_text('{corrupt ledger', encoding='utf-8')
    step = Mock(side_effect=requests.Timeout('public read deadline'))
    monkeypatch.setattr(runner, 'btc_step', step)
    if journal_unavailable:
        monkeypatch.setattr(runner, 'journal', Mock(side_effect=PermissionError('journal unavailable')))
    runner._run_locked(plan, tmp_path, args)
    status = json.loads((tmp_path / 'status.json').read_text(encoding='utf-8'))
    view = status['accounts']['btc']
    assert view['status'] == 'degraded'
    assert 'Timeout' in view['errors']['worker']
    assert 'JSONDecodeError' in view['errors']['ledger']
    assert path.read_text(encoding='utf-8') == '{corrupt ledger'


def test_stock_ledger_read_explicitly_uses_utf8(tmp_path):
    class LedgerPath:
        def read_text(self, *, encoding):
            assert encoding == 'utf-8'
            return json.dumps({'start_ms': 1000}, ensure_ascii=False)
    venue = Mock(side_effect=AssertionError('unexpected network'))
    venue.get.return_value = {}
    account = runner.StockAccount(LedgerPath(), venue, 'MUUSDT', {})
    account.market_data()
