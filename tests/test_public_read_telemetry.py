"""Safe optional observations, corrupt-state refusal and process contention."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import public_read_telemetry as telemetry
from process_lock import exclusive_process_lock

NOW = 1_793_020_000_000
ENDPOINT = '/fapi/v1/time'


def record(path, **overrides):
    args = {'endpoint': ENDPOINT, 'transport': 'requests', 'event': 'network_start', 'at_ms': NOW}
    args.update(overrides)
    return telemetry.record_event(path, **args)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def count(payload, event):
    index = str(telemetry.EVENTS.index(event))
    return sum(v for k,v in payload['window_totals'].items() if k.split(':')[2] == index)


def test_network_response_local_block_and_server_weight_remain_distinct(tmp_path):
    path = tmp_path/'nested'/'counts.json'
    assert record(path)
    assert record(path, event='response', http_status=200, used_weight_1m='00042')
    assert record(path, transport='none', event='cooldown_blocked', http_status=418)
    data = load(path)
    assert count(data,'network_start') == 1 and count(data,'response') == 1
    assert count(data,'cooldown_blocked') == 1
    assert data['window_status_counts'] == {'2xx': 1}
    assert data['latest_server_weight']['used_weight_1m'] == 42
    assert data['latest_event']['used_weight_1m'] is None
    assert data['observed_only'] and data['complete'] is False
    assert 'not_server_quota' in data['counts_scope']
    assert path.stat().st_size <= telemetry.MAX_BYTES


def test_actual_http_limits_have_response_counters_separate_from_guard(tmp_path):
    path = tmp_path/'counts.json'
    for status in (418,429,500):
        assert record(path, event='response', http_status=status)
    assert record(path, event='transport_error', transport='curl')
    assert load(path)['window_status_counts'] == {'4xx':2,'418':1,'429':1,'5xx':1}


def test_rolls_sixty_minute_buckets_without_resetting_inception(tmp_path):
    path = tmp_path/'counts.json'
    start = NOW//60_000*60_000
    for i in range(61):
        assert record(path, at_ms=start+i*60_000)
    data = load(path)
    assert len(data['buckets']) == 60 and count(data,'network_start') == 60
    assert data['first_recorded_at_ms'] == start
    assert data['buckets'][0]['minute_ms'] == start+60_000
    assert record(path, at_ms=start+180*60_000)
    assert len(load(path)['buckets']) == 1 and count(load(path),'network_start') == 1


def test_old_server_weight_expires_instead_of_looking_current(tmp_path):
    path = tmp_path/'counts.json'
    assert record(path,event='response',used_weight_1m=99)
    assert record(path,at_ms=NOW+61*60_000)
    assert load(path)['latest_server_weight'] is None


def test_clock_rollback_refuses_write_preserving_exact_bytes(tmp_path):
    path=tmp_path/'counts.json'
    assert record(path)
    raw=path.read_bytes()
    assert record(path,at_ms=NOW-1) is False
    assert path.read_bytes()==raw
    assert record(path) and count(load(path),'network_start')==2


class Hostile:
    def __str__(self):
        raise AssertionError('arbitrary input must not be stringified')
    def __repr__(self):
        raise AssertionError('arbitrary input must not be represented')


@pytest.mark.parametrize('field,value', [
    ('endpoint','/fapi/v1/order'),('endpoint','/fapi/v1/time?token=SECRET'),
    ('endpoint','https://user:SECRET@evil.invalid'),('endpoint',{}),('endpoint',Hostile()),
    ('transport','proxy:SECRET'),('transport',Hostile()),('transport','none'),
    ('event','SECRET'),('event',Hostile()),('event',{}),
    ('at_ms',True),('at_ms',-1),('at_ms',float('nan')),('at_ms','123'),
    ('http_status',True),('http_status',99),('http_status',600),('http_status',Hostile()),
    ('used_weight_1m',-1),('used_weight_1m',True),('used_weight_1m',float('nan')),
    ('used_weight_1m','token=SECRET'),('used_weight_1m',' 12'),
    ('used_weight_1m','99999999999'),('used_weight_1m',Hostile()),
])
def test_unknown_or_malicious_arguments_return_false_without_writes(tmp_path, field, value):
    path=tmp_path/'counts.json'
    assert record(path,**{field:value}) is False
    assert not path.exists()


@pytest.mark.parametrize('raw', [b'',b'{',b'[]',b'{"secret":"DO_NOT_COPY"}',b'\xff',b'x'*(telemetry.MAX_BYTES+1)],
                         ids=['empty','json','array','hidden','encoding','oversized'])
def test_corrupt_or_oversized_file_is_not_reset_or_replaced(tmp_path,raw):
    path=tmp_path/'counts.json';path.write_bytes(raw)
    assert record(path) is False and path.read_bytes()==raw


@pytest.mark.parametrize('change', ['totals','unknown_key','future_bucket','duplicate_bucket','string_counter',
                                  'bool_total','unbacked_status','flags','latest','dimensions'])
def test_semantically_corrupt_schema_is_not_silently_accepted(tmp_path,change):
    path=tmp_path/'counts.json';assert record(path)
    data=load(path)
    if change=='totals':data['window_totals']={}
    elif change=='unknown_key':data['secret']='DO_NOT_COPY'
    elif change=='future_bucket':data['buckets'][0]['minute_ms']+=60_000
    elif change=='duplicate_bucket':data['buckets'].append(deepcopy(data['buckets'][0]))
    elif change=='string_counter':data['buckets'][0]['counts'][next(iter(data['buckets'][0]['counts']))]='1'
    elif change=='bool_total':data['window_totals'][next(iter(data['window_totals']))]=True
    elif change=='unbacked_status':
        data['buckets'][0]['status_counts']={'2xx':1}
        data['window_status_counts']={'2xx':1}
    elif change=='flags':data['complete']=True
    elif change=='latest':data['latest_event']['endpoint']='/fapi/v1/order'
    else:data['dimensions']['endpoints'].append('SECRET')
    raw=json.dumps(data).encode();path.write_bytes(raw)
    assert record(path) is False and path.read_bytes()==raw


def test_lock_busy_is_nonblocking_and_preserves_existing_file(tmp_path):
    path=tmp_path/'counts.json';assert record(path);raw=path.read_bytes()
    with exclusive_process_lock(path.with_name(path.name+'.lock')):
        assert record(path) is False
    assert path.read_bytes()==raw
    assert record(path)


@pytest.mark.parametrize('stage',['mkdir','replace','tempfile'])
def test_first_directory_and_atomic_write_errors_never_escape(tmp_path,monkeypatch,stage):
    path=tmp_path/'new'/'counts.json'
    def fail(*args,**kwargs):raise OSError('DO_NOT_LEAK_FILESYSTEM_SECRET')
    if stage=='mkdir':monkeypatch.setattr(Path,'mkdir',fail)
    elif stage=='replace':monkeypatch.setattr(telemetry.os,'replace',fail)
    else:monkeypatch.setattr(telemetry.tempfile,'NamedTemporaryFile',fail)
    assert record(path) is False and not path.exists()


def test_failed_replace_preserves_previous_valid_counts(tmp_path,monkeypatch):
    path=tmp_path/'counts.json';assert record(path);raw=path.read_bytes()
    def fail(*args):raise PermissionError('private details')
    monkeypatch.setattr(telemetry.os,'replace',fail)
    assert record(path) is False and path.read_bytes()==raw


def test_dense_sixty_minute_window_remains_bounded_and_exact(tmp_path):
    path=tmp_path/'counts.json';assert record(path)
    data=load(path);minute=NOW//60_000*60_000
    counts={f'{a}:{b}:{c}':1 for a in range(len(telemetry.ENDPOINTS))
            for b in range(3) for c in range(4) if b!=2 or c==3}
    data['first_recorded_at_ms']=minute-59*60_000
    data['buckets']=[{'minute_ms':minute-i*60_000,'counts':dict(counts),'status_counts':{}}
                     for i in reversed(range(60))]
    data['window_totals']={key:60 for key in counts}
    path.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
    assert record(path)
    assert path.stat().st_size<=telemetry.MAX_BYTES
    assert count(load(path),'network_start')==len(telemetry.ENDPOINTS)*2*60+1


def test_cross_process_success_counts_are_exact_despite_dropped_busy_events(tmp_path):
    path=tmp_path/'counts.json'
    program=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
             'from public_read_telemetry import record_event;'
             'ok=sum(record_event(Path(sys.argv[2]),endpoint="/fapi/v1/time",transport="requests",'
             'event="network_start",at_ms=1793020000000) for _ in range(30));print(ok)')
    processes=[subprocess.Popen([sys.executable,'-c',program,str(ROOT/'scripts'),str(path)],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    accepted=0
    for process in processes:
        out,err=process.communicate(timeout=15)
        assert process.returncode==0,err
        accepted+=int(out.strip())
    assert accepted>0 and count(load(path),'network_start')==accepted
    assert len(path.read_bytes())<=telemetry.MAX_BYTES
