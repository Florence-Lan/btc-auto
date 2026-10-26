"""Actual client telemetry/guard wiring with no network, .env or production I/O."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
# The client normally loads .env at import. These public-only tests explicitly
# suppress that read, including when run by themselves in the real workspace.
with patch('dotenv.load_dotenv', return_value=False):
    import binance_terminal_client as bc
import public_read_telemetry as telemetry


def response(payload, *, status=200, weight='17', extra_headers=None):
    result=requests.Response()
    result.status_code=status
    result._content=json.dumps(payload).encode('utf-8')
    result.encoding='utf-8'
    if weight is not None:
        result.headers['X-MBX-USED-WEIGHT-1M']=weight
    result.headers.update(extra_headers or {})
    return result


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def count(path, event, transport=None):
    data=read(path)
    return sum(value for key,value in data['window_totals'].items()
               if telemetry.EVENTS[int(key.split(':')[2])]==event
               and (transport is None or telemetry.TRANSPORTS[int(key.split(':')[1])]==transport))


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    monkeypatch.delenv('BINANCE_PUBLIC_DIAGNOSTICS', raising=False)
    monkeypatch.delenv('BINANCE_API_KEY', raising=False)
    monkeypatch.delenv('BINANCE_API_SECRET', raising=False)
    monkeypatch.setenv('BINANCE_FUTURES_BASE_URL',bc.MAINNET_FUTURES_URL)
    monkeypatch.setattr(bc,'COOLDOWN_PATH',tmp_path/'cooldown.json')
    monkeypatch.setattr(bc,'RULES_CACHE_PATH',tmp_path/'rules.json')
    monkeypatch.setattr(bc,'PUBLIC_READ_TELEMETRY_PATH',tmp_path/'diagnostics'/'reads.json')
    # Every test starts with a deny-by-default network/process boundary.
    monkeypatch.setattr(requests.Session,'get',Mock(side_effect=AssertionError('Unexpected network')))
    monkeypatch.setattr(bc.subprocess,'run',Mock(side_effect=AssertionError('Unexpected subprocess')))
    client=bc.BinanceTerminalClient()
    yield client
    client.session.close()


def enable(client):
    client._public_diagnostics=True


def ban(client):
    until=int(time.time()*1000)+300_000
    bc.COOLDOWN_PATH.write_text(json.dumps({'base_url':client.base_url,
        'retry_at_ms':until,'reason':'test cooldown'}),encoding='utf-8')
    return until


def install_curl(monkeypatch,tmp_path,payload,*,status=200,weight='24'):
    """Exercise the real curl adapter while mocking only executable dispatch."""
    monkeypatch.setattr(bc.shutil,'which',lambda _: 'mock-curl')
    original_directory=bc.tempfile.TemporaryDirectory
    monkeypatch.setattr(bc.tempfile,'TemporaryDirectory',
        lambda **kwargs: original_directory(dir=tmp_path,**kwargs))
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        header_path=Path(command[command.index('--dump-header')+1])
        header_path.write_text(f'HTTP/1.1 {status} mocked\r\nX-MBX-USED-WEIGHT-1M: {weight}\r\n\r\n',
                               encoding='iso-8859-1')
        return SimpleNamespace(returncode=0,stdout=json.dumps(payload)+'\n'+str(status),stderr='')
    monkeypatch.setattr(bc.subprocess,'run',run)
    return calls


def test_default_off_never_creates_diagnostics_or_calls_recorder(isolated_client,monkeypatch):
    client=isolated_client
    recorder=Mock(side_effect=AssertionError('Disabled recorder was called'))
    monkeypatch.setattr(bc,'record_event',recorder)
    client.session.get=Mock(return_value=response({'serverTime':123456}))
    assert not client._public_diagnostics
    assert client.server_time_ms()==123456
    recorder.assert_not_called()
    assert not bc.PUBLIC_READ_TELEMETRY_PATH.parent.exists()


def test_opt_in_environment_records_actual_request_and_numeric_header(isolated_client,monkeypatch):
    monkeypatch.setenv('BINANCE_PUBLIC_DIAGNOSTICS','1')
    client=bc.BinanceTerminalClient()
    try:
        client.session.get=Mock(return_value=response({'serverTime':123456},weight='00041'))
        assert client._public_diagnostics and client.server_time_ms()==123456
        path=bc.PUBLIC_READ_TELEMETRY_PATH
        assert count(path,'network_start','requests')==1
        assert count(path,'response','requests')==1
        data=read(path)
        assert data['latest_server_weight']['used_weight_1m']==41
        assert data['window_status_counts']=={'2xx':1}
        assert data['complete'] is False and data['observed_only'] is True
    finally:
        client.session.close()


def test_existing_cooldown_records_local_block_without_network(isolated_client):
    client=isolated_client;enable(client);ban(client)
    request=Mock(side_effect=AssertionError('Cooldown must stop network'))
    client.session.get=request
    with pytest.raises(bc.BinanceApiError,match='paused until'):
        client.public_get('/fapi/v1/time')
    request.assert_not_called()
    path=bc.PUBLIC_READ_TELEMETRY_PATH
    assert count(path,'cooldown_blocked','none')==1
    assert count(path,'network_start')==count(path,'response')==0
    assert read(path)['window_status_counts']=={}


def test_requests_failure_then_actual_mock_curl_counts_two_attempts(isolated_client,monkeypatch,tmp_path):
    client=isolated_client;enable(client)
    secret='DO_NOT_RECORD_PROXY_SECRET'
    client.session.get=Mock(side_effect=requests.ConnectionError('https://user:'+secret+'@proxy.invalid'))
    calls=install_curl(monkeypatch,tmp_path,{'serverTime':789012})
    assert client.server_time_ms()==789012 and len(calls)==1
    path=bc.PUBLIC_READ_TELEMETRY_PATH
    assert count(path,'network_start')==2
    assert count(path,'network_start','requests')==count(path,'network_start','curl')==1
    assert count(path,'transport_error','requests')==1
    assert count(path,'response','curl')==1
    assert read(path)['latest_server_weight']['used_weight_1m']==24
    assert secret not in path.read_text()


def test_curl_preference_records_only_the_transport_actually_dispatched(isolated_client,monkeypatch,tmp_path):
    client=isolated_client;enable(client)
    client._curl_preferred_until=time.monotonic()+60
    calls=install_curl(monkeypatch,tmp_path,{'serverTime':555})
    client.session.get=Mock(side_effect=AssertionError('Curl success must not start requests'))
    assert client.server_time_ms()==555 and len(calls)==1
    assert count(bc.PUBLIC_READ_TELEMETRY_PATH,'network_start','curl')==1
    assert count(bc.PUBLIC_READ_TELEMETRY_PATH,'network_start','requests')==0


def test_curl_failure_is_counted_without_copying_stderr(isolated_client,monkeypatch,tmp_path):
    client=isolated_client;enable(client)
    client._curl_preferred_until=time.monotonic()+60
    monkeypatch.setattr(bc.shutil,'which',lambda _: 'mock-curl')
    monkeypatch.setattr(bc.subprocess,'run',lambda *_,**kw:SimpleNamespace(
        returncode=28,stdout='',stderr='DO_NOT_RECORD_CURL_SECRET'))
    client.session.get=Mock(return_value=response({'serverTime':999}))
    assert client.server_time_ms()==999
    path=bc.PUBLIC_READ_TELEMETRY_PATH
    assert count(path,'network_start')==2 and count(path,'transport_error','curl')==1
    assert 'DO_NOT_RECORD_CURL_SECRET' not in path.read_text()


def test_queued_real_worker_rechecks_new_shared_cooldown_before_get(isolated_client,monkeypatch):
    client=isolated_client;enable(client)
    entered,release=(threading.Event() for _ in range(2))
    actual_guard=client._guard_public_read
    calls=[]
    def guard(endpoint):
        calls.append(endpoint)
        if len(calls)==2:
            entered.set()
            assert release.wait(2)
        return actual_guard(endpoint)
    monkeypatch.setattr(client,'_guard_public_read',guard)
    request=Mock(side_effect=AssertionError('Late worker must observe shared cooldown'))
    client.session.get=request
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(client.public_get,'/fapi/v1/time')
        try:
            assert entered.wait(1)
            ban(client)
        finally:
            release.set()
        with pytest.raises(bc.BinanceApiError,match='paused until'):
            future.result(timeout=2)
    request.assert_not_called()
    path=bc.PUBLIC_READ_TELEMETRY_PATH
    assert count(path,'cooldown_blocked')==1 and count(path,'network_start')==0


def test_new_cooldown_between_primary_failure_and_fallback_prevents_curl(isolated_client):
    client=isolated_client;enable(client)
    def fail(*_,**kwargs):
        ban(client)
        raise requests.ConnectionError('mock transport ended during a shared cooldown update')
    client.session.get=fail
    client._curl_public_get=Mock(side_effect=AssertionError('No fallback during cooldown'))
    with pytest.raises(bc.BinanceApiError):
        client.public_get('/fapi/v1/time')
    client._curl_public_get.assert_not_called()
    path=bc.PUBLIC_READ_TELEMETRY_PATH
    assert count(path,'network_start')==1 and count(path,'cooldown_blocked')==1


@pytest.mark.parametrize('failure',['false','exception','corrupt_file'])
def test_failed_telemetry_never_changes_payload_or_creates_a_clock_cache(isolated_client,monkeypatch,failure):
    client=isolated_client;enable(client)
    if failure=='false':
        monkeypatch.setattr(bc,'record_event',lambda *_,**kw:False)
    elif failure=='exception':
        def broken(*_,**kw):raise OSError('DO_NOT_LEAK_DIAGNOSTICS_IO')
        monkeypatch.setattr(bc,'record_event',broken)
    else:
        bc.PUBLIC_READ_TELEMETRY_PATH.parent.mkdir()
        bc.PUBLIC_READ_TELEMETRY_PATH.write_text('{corrupt')
    client.session.get=Mock(side_effect=[response({'serverTime':100000}),response({'serverTime':200000})])
    assert client.server_time_ms()==100000 and client.server_time_ms()==200000
    assert client.session.get.call_count==2
    if failure=='corrupt_file':
        assert bc.PUBLIC_READ_TELEMETRY_PATH.read_text()=='{corrupt'


def test_mark_source_clock_and_next_funding_are_returned_unchanged(isolated_client):
    client=isolated_client;enable(client)
    payload={'symbol':'BTCUSDT','markPrice':'81234.5','time':1793020000123,'nextFundingTime':1793030400000}
    client.session.get=Mock(return_value=response(payload))
    assert client.mark_price_observation()=={'price':81234.5,'time_ms':1793020000123,
                                             'next_funding_time_ms':1793030400000}
    assert count(bc.PUBLIC_READ_TELEMETRY_PATH,'network_start')==1


def test_telemetry_never_stores_params_symbol_raw_body_or_other_headers(isolated_client):
    client=isolated_client;enable(client)
    payload={'serverTime':123456,'raw':'DO_NOT_COPY_BODY_SECRET'}
    client.session.get=Mock(return_value=response(payload,weight='23',extra_headers={
        'Set-Cookie':'DO_NOT_COPY_COOKIE','X-Private':'DO_NOT_COPY_HEADER'}))
    params={'symbol':'DO_NOT_COPY_SYMBOL','interval':'DO_NOT_COPY_INTERVAL','note':'DO_NOT_COPY_PARAM'}
    assert client.public_get('/fapi/v1/time',params)==payload
    raw=bc.PUBLIC_READ_TELEMETRY_PATH.read_text()
    assert 'DO_NOT_COPY' not in raw and 'https://' not in raw
    assert 'params' not in raw and read(bc.PUBLIC_READ_TELEMETRY_PATH)['latest_server_weight']['used_weight_1m']==23


@pytest.mark.parametrize('private_key',['signature','api_key','api_secret','authorization'])
def test_private_query_fields_are_rejected_before_telemetry_or_network(isolated_client,private_key):
    client=isolated_client;enable(client)
    client.session.get=Mock(side_effect=AssertionError('Private field must fail before dispatch'))
    with pytest.raises(ValueError,match='Credentials'):
        client.public_get('/fapi/v1/time',{private_key:'DO_NOT_COPY_PRIVATE'})
    client.session.get.assert_not_called()
    assert not bc.PUBLIC_READ_TELEMETRY_PATH.exists()


@pytest.mark.parametrize('status',[418,429])
def test_actual_http_limit_persists_shared_pause_and_does_not_retry(isolated_client,status):
    client=isolated_client;enable(client)
    until=int(time.time()*1000)+300_000
    client.session.get=Mock(return_value=response({'code':-1003,
        'msg':f'DO_NOT_COPY_LIMIT_BODY banned until {until}'},status=status,weight='77'))
    client._curl_public_get=Mock(side_effect=AssertionError('HTTP limit must not fallback'))
    with pytest.raises(bc.BinanceApiError) as error:
        client.public_get('/fapi/v1/time')
    assert error.value.status==status and read(bc.COOLDOWN_PATH)['retry_at_ms']>=until
    assert client.session.get.call_count==1
    client._curl_public_get.assert_not_called()
    path=bc.PUBLIC_READ_TELEMETRY_PATH
    assert count(path,'network_start')==count(path,'response')==1
    assert read(path)['window_status_counts'][str(status)]==1
    assert 'DO_NOT_COPY_LIMIT_BODY' not in path.read_text()
    with pytest.raises(bc.BinanceApiError):client.public_get('/fapi/v1/time')
    assert client.session.get.call_count==1 and count(path,'cooldown_blocked')==1


@pytest.mark.parametrize('persist_failure',['lock_busy','replace'])
def test_known_http_ban_survives_failure_to_publish_first_cooldown(isolated_client,monkeypatch,persist_failure):
    client=isolated_client;enable(client)
    until=int(time.time()*1000)+600_000
    client.session.get=Mock(return_value=response({'code':-1003,'msg':f'banned until {until}'},status=429))
    if persist_failure=='lock_busy':
        @contextmanager
        def unavailable(_,**kwargs):
            raise bc.BinanceApiError('Offline lock unavailable',status=418,
                                     retry_at_ms=int(time.time()*1000)+60_000)
            yield  # pragma: no cover - keep the same context manager contract
        monkeypatch.setattr(bc,'_cooldown_update_lock',unavailable)
    else:
        original=Path.replace
        def fail_cooldown(self,target):
            if Path(target)==bc.COOLDOWN_PATH:
                raise PermissionError('Offline persistence failure')
            return original(self,target)
        monkeypatch.setattr(Path,'replace',fail_cooldown)
    with pytest.raises((bc.BinanceApiError,OSError)):
        client.public_get('/fapi/v1/time')
    assert client.session.get.call_count==1
    with pytest.raises(bc.BinanceApiError) as error:
        client.public_get('/fapi/v1/time')
    assert client.session.get.call_count==1, 'A known server ban must still block dispatch after persistence failed.'
    assert error.value.retry_at_ms>=until


def test_unpublished_ban_blocks_another_client_in_same_process_and_expires(isolated_client,monkeypatch):
    client=isolated_client
    until=(int(time.time())+600)*1000
    original=Path.replace
    def fail_cooldown(self,target):
        if Path(target)==bc.COOLDOWN_PATH:
            raise PermissionError('Offline initial persistence failure')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',fail_cooldown)
    with pytest.raises(OSError):
        client.record_cooldown(f'banned until {until}')
    assert not bc.COOLDOWN_PATH.exists()
    other=bc.BinanceTerminalClient()
    try:
        other.session.get=Mock(side_effect=AssertionError('Known in-process ban must apply to another client'))
        with pytest.raises(bc.BinanceApiError) as error:
            other.public_get('/fapi/v1/time')
        assert error.value.retry_at_ms>=until
        other.session.get.assert_not_called()
        monkeypatch.setattr(bc.time,'time',lambda:(until+6000)/1000)
        expected=until+6000
        other.session.get=Mock(return_value=response({'serverTime':expected}))
        assert other.server_time_ms()==expected
        other.session.get.assert_called_once()
    finally:
        other.session.close()
