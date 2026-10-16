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
    with pytest.raises(runner.PublicTransportError, match='fallback curl transport unavailable'):
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
    with pytest.raises(RuntimeError, match='HTTP 429'):
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


def protective_stock(tmp_path, monkeypatch, *, opened=True, stop=95):
    current = {'now': 10 * runner.FOUR_HOURS + 100_000}
    monkeypatch.setattr(runner, 'now_ms', lambda: current['now'])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: current['now'] / 1000)
    config = json.loads((runner.ROOT / 'config/stock_swing_120_candidate_20261004.json').read_text(encoding='utf-8'))
    path = tmp_path / 'state.json'
    state = runner.initial_stock('MUUSDT', current['now'] - 10_000, config)
    if opened:
        state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
            'entry_time': current['now'] - 5_000, 'signal_time': current['now'] - 5_000,
            'stop': stop, 'initial_stop': stop, 'entry_fee': 0, 'margin': 10, 'funding': 0}
        state['position_history'] = [{'time_ms': current['now'] - 5_000, 'signed_qty': 1}]
    runner.write_json(path, state)
    account = runner.StockAccount(path, Mock(side_effect=AssertionError('offline fixture')), 'MUUSDT', config)
    def snapshot():
        stamp = current['now']
        return [('clock', {'serverTime': stamp}, None),
            ('mark', {'time': stamp, 'markPrice': '94', 'indexPrice': '94',
                      'lastFundingRate': '0', 'nextFundingTime': stamp + 1_000_000}, None),
            ('book', {'T': stamp, 'bids': [['94', '100']], 'asks': [['94.1', '100']]}, None),
            ('rules', {'symbols': [{'symbol': 'MUUSDT', 'status': 'TRADING', 'filters': [
                {'filterType': 'LOT_SIZE', 'stepSize': '.01', 'minQty': '.01', 'maxQty': '100'},
                {'filterType': 'PRICE_FILTER', 'tickSize': '.01'}]}]}, None),
            ('funding', None, 'Funding unavailable'), ('signal', None, 'Signal unavailable'),
            ('five', None, 'Liquidity candle unavailable')]
    return account, path, current, snapshot


@pytest.mark.parametrize('clock_failure', ['missing', 'expired_receipt', 'bad_server_time', 'positive_stale', 'positive_future'])
def test_stock_bad_clock_keeps_fresh_mark_and_depth_protection(tmp_path, monkeypatch, clock_failure):
    account, _, _, snapshot = protective_stock(tmp_path, monkeypatch)
    responses = snapshot()
    if clock_failure == 'missing':
        responses[0] = ('clock', None, 'Clock unavailable')
    elif clock_failure == 'expired_receipt':
        responses[0][1]['_received_monotonic'] = runner.time.monotonic() - 16
    elif clock_failure == 'bad_server_time':
        responses[0][1]['serverTime'] = 'invalid'
    else:
        responses[0][1].update(serverTime=runner.now_ms() + (-120_000 if clock_failure == 'positive_stale' else 120_000),
                               _received_monotonic=runner.time.monotonic())
    result = account.step(responses)
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
    assert result['status'] == 'degraded'
    assert result['risk_monitor']['assessment'] == 'evaluated'
    assert result['risk_monitor']['clock_source'] == 'mark_endpoint'
    assert result['risk_monitor']['consecutive_failures'] == 0
    assert 'clock' in result['errors']
    assert account.exchange_clock_anchor is None


@pytest.mark.parametrize('bad_source', ['index', 'signal', 'five', 'rules'])
def test_bad_entry_inputs_cannot_prevent_existing_stock_stop(tmp_path, monkeypatch, bad_source):
    account, _, _, snapshot = protective_stock(tmp_path, monkeypatch)
    responses = snapshot()
    # Establish last known quantity metadata, then fail a non-protective input.
    account.rules = responses[3][1]['symbols'][0]
    if bad_source == 'index':
        responses[1][1]['indexPrice'] = 'invalid'
    else:
        responses = [(label, {'symbols': []} if label == 'rules' else [[1]], None)
                     if label == bad_source else (label, payload, error)
                     for label, payload, error in responses]
    result = account.step(responses)
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
    assert bad_source in result['errors']
    assert result['risk_monitor']['assessment'] == 'evaluated'


def test_stock_failed_marks_preserve_last_success_and_recover_without_stale_fill(tmp_path, monkeypatch):
    account, _, current, snapshot = protective_stock(tmp_path, monkeypatch, stop=90)
    good = account.step(snapshot())
    last_success = good['risk_monitor']['last_success_at_ms']
    original_position = copy.deepcopy(good['position'])
    current['now'] += 30_000
    stale = snapshot()
    stale[1][1].update(time=current['now'] - 60_001, markPrice='1')
    with pytest.raises(ValueError, match='Stale or future'):
        account.step(stale)
    failed = runner.read_json(account.path)
    assert failed['position'] == original_position
    assert failed['fill_count_total'] == 0
    assert failed['last_mark_price'] == 94
    assert failed['wallet_balance'] == good['wallet_balance']
    assert failed['risk_monitor']['status'] == 'unavailable'
    assert failed['risk_monitor']['assessment'] == 'not_evaluated'
    assert failed['risk_monitor']['last_success_at_ms'] == last_success
    assert failed['risk_monitor']['mark_age_ms'] == 60_001
    assert failed['risk_monitor']['consecutive_failures'] == 1
    missing = [(label, None, 'Unavailable') if label == 'mark' else (label, payload, error)
               for label, payload, error in snapshot()]
    assert account.step(missing)['risk_monitor']['consecutive_failures'] == 2
    current['now'] += 30_000
    recovered = account.step(snapshot())
    assert recovered['risk_monitor']['assessment'] == 'evaluated'
    assert recovered['risk_monitor']['last_success_at_ms'] == current['now']
    assert recovered['risk_monitor']['last_success_at_ms'] > last_success
    assert recovered['risk_monitor']['consecutive_failures'] == 0
    assert recovered['fill_count_total'] == 0


def test_stock_without_fresh_depth_records_stop_pending_without_inventing_fill(tmp_path, monkeypatch):
    account, _, _, snapshot = protective_stock(tmp_path, monkeypatch)
    responses = [(label, None, 'Unavailable') if label == 'book' else (label, payload, error)
                 for label, payload, error in snapshot()]
    result = account.step(responses)
    assert result['position']['pending_exit'] == 'protective_stop'
    assert result['fill_count_total'] == 0
    assert result['risk_monitor']['assessment'] == 'evaluated'
    assert not result['risk_monitor']['protection_fill_available']


def test_aster_transport_diagnostics_keep_codes_and_redact_exception_messages(monkeypatch):
    monkeypatch.setattr(runner.requests, 'get', Mock(side_effect=requests.ConnectionError(
        'https://user:secret@proxy.invalid?api_key=topsecret')))
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'curl.exe')
    fallback = Mock(return_value=subprocess.CompletedProcess([], 56, '', 'secret proxy response'))
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    with pytest.raises(runner.PublicTransportError) as error:
        runner.PublicAster().get('premiumIndex', {'symbol': 'MUUSDT'})
    assert error.value.failures[-1]['curl_exit_code'] == 56
    assert error.value.failures[-1]['category'] == 'receive'
    import traceback
    trace = ''.join(traceback.format_exception(error.value))
    assert 'topsecret' not in trace and 'secret proxy' not in trace and 'user:secret' not in trace
    environment = fallback.call_args.kwargs['env']
    assert not any('API_KEY' in key.upper() or 'API_SECRET' in key.upper() for key in environment)


@pytest.mark.parametrize('failure', ['chunked', 'decoding', 'bad_json', 'null'])
def test_aster_broken_response_uses_bounded_verified_fallback(monkeypatch, failure):
    response = Mock(status_code=200)
    response.json.side_effect = {'chunked': requests.exceptions.ChunkedEncodingError('private response'),
                                'decoding': requests.exceptions.ContentDecodingError('private response'),
                                'bad_json': ValueError('private response'), 'null': None}[failure]
    if failure == 'null':
        response.json.return_value = None
    primary = Mock(return_value=response)
    monkeypatch.setattr(runner.requests, 'get', primary)
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'curl.exe')
    fallback = Mock(return_value=curl_response({'serverTime': 1000}))
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    assert runner.PublicAster().get('time') == {'serverTime': 1000}
    primary.assert_called_once()
    fallback.assert_called_once()
    response.close.assert_called_once()


def test_aster_late_rate_limit_suppresses_unsafe_primary_exception_trace(monkeypatch):
    venue = runner.PublicAster()
    def fail(*args, **kwargs):
        venue.cooldown = runner.now_ms() + 120_000
        raise requests.ConnectionError('https://user:secret@proxy.invalid?api_key=topsecret')
    monkeypatch.setattr(venue.deadline, 'run', fail)
    fallback = Mock(side_effect=AssertionError('Rate limiting must not switch transports'))
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    with pytest.raises(RuntimeError, match='cooldown active') as error:
        venue.get('time')
    import traceback
    trace = ''.join(traceback.format_exception(error.value))
    assert 'topsecret' not in trace and 'user:secret' not in trace
    fallback.assert_not_called()


@pytest.mark.parametrize('resolution', ['mark_fallback', 'all_sources_failed_initially', 'mark_fallback_after_host_clock_jump'])
def test_btc_clock_outage_still_observes_risk_and_never_runs_strategy(tmp_path, monkeypatch, resolution):
    timestamp = 1_791_027_000_000
    timing = {'wall': timestamp / 1000, 'monotonic': 1000.0}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'now_ms', lambda: int(timing['wall'] * 1000))
    monkeypatch.setattr(runner.time, 'time', lambda: timing['wall'])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: timing['monotonic'])
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED', 'false')
    monkeypatch.setenv('SIM_TAKER_FEE', '0.00045')
    monkeypatch.setenv('SIM_SLIPPAGE_BPS', '1')
    path = tmp_path / 'btc/state.json'
    account = runner.SimulationAccount(path, allow_llm=False)
    state = account.reset(1000, now_ms=timestamp - 100_000)
    state.update(wallet_balance=1000, position_qty=.01, entry_price=100000,
                 observed_flat_target=True, last_signal_time_ms=timestamp - 90_000,
                 position_history=[{'time_ms': timestamp - 90_000, 'signed_qty': .01}])
    account._save(state)
    (tmp_path / 'profile.json').write_text('{"candidate_id":"offline"}', encoding='utf-8')
    client = Mock()
    if resolution == 'mark_fallback_after_host_clock_jump':
        client.server_time_ms.return_value = timestamp
        runner.decision_runtime.resolve_clock(client)
        timing.update(wall=timing['wall'] - 3600, monotonic=timing['monotonic'] + 1)
    client.server_time_ms.side_effect = OSError('Clock disconnected')
    mark = {'price': 80000, 'time_ms': timestamp + (1000 if resolution == 'mark_fallback_after_host_clock_jump' else -1000)}
    client.mark_price_observation.side_effect = ([mark, AssertionError('Do not refetch a validated mark')]
        if resolution != 'all_sources_failed_initially' else [OSError('First mark failed'), mark])
    client.symbol_rules.side_effect = AssertionError('No metadata needed for BTC full protective close')
    execute = Mock(side_effect=AssertionError('Fallback must not run strategy entries'))
    monkeypatch.setattr(runner, 'execute_report', execute)
    result = runner.btc_step({'state_path': 'btc/state.json', 'strategy_path': 'profile.json'},
                             tmp_path, {'start_ms': timestamp - 100_000}, client)
    assert account.load()['position_qty'] == 0
    assert account.load()['fill_count_total'] == 1
    assert result['status'] == 'degraded'
    assert result['risk_monitor']['assessment'] == 'evaluated'
    assert result['entry_blockers'] == ['exchange_clock_unavailable']
    assert client.mark_price_observation.call_count == (2 if resolution == 'all_sources_failed_initially' else 1)
    client.funding_history.assert_not_called()
    execute.assert_not_called()


def test_aster_successful_curl_is_temporarily_preferred_then_requests_is_retried(monkeypatch):
    monotonic = {'now': 1000.0}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: monotonic['now'])
    response = Mock(status_code=200)
    response.json.return_value = {'serverTime': 2000}
    primary = Mock(side_effect=[requests.Timeout('bounded source deadline'), response])
    fallback = Mock(return_value=curl_response({'serverTime': 1000}))
    monkeypatch.setattr(runner.requests, 'get', primary)
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'curl.exe')
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    venue = runner.PublicAster()
    assert venue.get('time') == {'serverTime': 1000}
    monotonic['now'] += 1
    assert venue.get('time') == {'serverTime': 1000}
    assert primary.call_count == 1 and fallback.call_count == 2
    monotonic['now'] += 301
    assert venue.get('time') == {'serverTime': 2000}
    assert primary.call_count == 2
    assert venue.last_public_transport['transport'] == 'requests'


def test_aster_preferred_curl_failure_can_recover_via_requests(monkeypatch):
    primary = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {'serverTime': 2000}
    primary.side_effect = [requests.Timeout('first source deadline'), response, response]
    fallback = Mock(side_effect=[curl_response({'serverTime': 1000}), subprocess.CompletedProcess([], 56, '', '')])
    monkeypatch.setattr(runner.requests, 'get', primary)
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'curl.exe')
    monkeypatch.setattr(runner.subprocess, 'run', fallback)
    venue = runner.PublicAster()
    assert venue.get('time') == {'serverTime': 1000}
    assert venue.get('time') == {'serverTime': 2000}
    assert venue._curl_preferred_until == 0
    assert venue.get('time') == {'serverTime': 2000}
    assert primary.call_count == 3 and fallback.call_count == 2
