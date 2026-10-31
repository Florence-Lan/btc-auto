"""Paper continuity tests use fake processes; no market loops or orders start."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_paper_services as paper


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


@pytest.fixture
def layout(tmp_path):
    accounts = []
    for account_id in ('btc', 'mu', 'sndk', 'skhynix'):
        state = f'data/parallel_simulation/original/{account_id}/state.json'
        save(tmp_path / state, {'wallet_balance': 1000, 'position_qty': .1})
        accounts.append({'account_id': account_id, 'state_path': state,
                         'strategy_path': 'config/profile.json'})
    accounts[0]['strategy_report_path'] = 'data/parallel_simulation/original/btc_signals/report.json'
    plan = {'execution_mode': 'simulation', 'live_orders_allowed': False,
            'runner_script': 'scripts/run_parallel_simulation.py',
            'btc_signal_producer_script': 'scripts/run_parallel_btc_signals.py',
            'plan_id': 'original', 'status_path': 'data/parallel_simulation/original/status.json',
            'accounts': accounts}
    save(tmp_path / 'config/parallel_simulation_plan_20261005.json', plan)
    save(tmp_path / 'data/parallel_simulation/original/manifest.json', {'start_utc': 'original'})
    shadow = tmp_path / 'data/paper_trading/sndk_4h_gate_shadow_20261012'
    save(shadow / 'manifest.json', {'start_utc': 'original'})
    for arm in paper.ARMS:
        save(shadow / arm / 'state.json', {'wallet_balance': 1000})
    for script in paper.ALLOWED:
        (tmp_path / 'scripts').mkdir(exist_ok=True)
        (tmp_path / 'scripts' / script).write_text('# dummy', encoding='utf-8')
    return tmp_path, shadow, plan


def services(layout):
    root, shadow, _ = layout
    return paper.build_services(root, shadow, python=str(root / '.venv/Scripts/python.exe'))


def test_guards_preserve_existing_ledgers_and_require_simulation(layout):
    root, shadow, plan = layout
    original = (root / plan['accounts'][2]['state_path']).read_bytes()
    result = services(layout)
    assert [s.name for s in result] == ['main', 'btc_signals', 'sndk_shadow']
    assert '--require-existing' in result[0].command and '--require-existing' in result[2].command
    assert (root / plan['accounts'][2]['state_path']).read_bytes() == original
    plan['live_orders_allowed'] = True
    save(root / 'config/parallel_simulation_plan_20261005.json', plan)
    with pytest.raises(ValueError, match='simulation plan'):
        services(layout)
    plan['live_orders_allowed'] = False
    save(root / 'config/parallel_simulation_plan_20261005.json', plan)
    (shadow / paper.ARMS[0] / 'state.json').unlink()
    with pytest.raises(ValueError, match='all four ledgers'):
        services(layout)
    assert not (shadow / paper.ARMS[0] / 'state.json').exists()


def test_guards_reject_account_paths_outside_generation(layout):
    root, _, plan = layout
    plan['accounts'][0]['state_path'] = '../escape/state.json'
    save(root / 'config/parallel_simulation_plan_20261005.json', plan)
    with pytest.raises(ValueError):
        services(layout)


def test_disabled_comparison_never_restarts_and_primary_services_remain_managed(layout):
    root, shadow, plan = layout
    save(root/'config/paper_service_extensions_20261020.json', {
        'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'sndk_shadow': {'enabled': False},
        'btc_reentry_shadow': {'enabled': False,
            'config_path': 'config/btc_reentry_compat_shadow_20261020.json',
            'output_dir': 'data/paper_trading/btc_reentry_compat_shadow_20261020'},
        'stock_macro_shadow': {'enabled': False,
            'config_path': 'config/stock_macro_shadow_20261021.json',
            'output_dir': 'data/paper_trading/stock_macro_shadow_20261021'}})
    old_shadow = (str(root/'.venv/Scripts/python.exe'), '-u',
        str(root/'scripts/run_sndk_4h_gate_shadow.py'), '--require-existing',
        '--poll-seconds', '30', '--output-dir', str(shadow))
    assert paper.canonical_command(old_shadow, root) is None
    definitions = services(layout)
    assert [service.name for service in definitions] == ['main', 'btc_signals']
    before = {a['account_id']: (root/a['state_path']).read_bytes() for a in plan['accounts']}
    backend = FakeProcesses()
    control = paper.Supervisor(definitions, backend, root/'data/control_test', root, lambda: 1000)
    for _ in range(3):
        control.tick()
    assert set(control.managed) == {'main', 'btc_signals'}
    assert {service.name for service in backend.started} == {'main', 'btc_signals'}
    assert len(backend.started) == 2 and not backend.stopped
    assert before == {a['account_id']: (root/a['state_path']).read_bytes() for a in plan['accounts']}


def test_disabled_shadow_cannot_bypass_primary_ledger_or_script_guards(layout):
    root, _, plan = layout
    save(root/'config/paper_service_extensions_20261020.json', {
        'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'sndk_shadow': {'enabled': False}})
    primary = root/plan['accounts'][0]['state_path']
    raw = primary.read_bytes()
    primary.unlink()
    with pytest.raises(ValueError, match='Original account ledger missing'):
        services(layout)
    assert not primary.exists()
    primary.write_bytes(raw)
    (root/'scripts/run_parallel_btc_signals.py').unlink()
    with pytest.raises(ValueError, match='required paper script'):
        services(layout)


def identity(service, pid=10, parent=1, created=100, command=None):
    argv = tuple(command or service.command)
    return paper.ProcessIdentity(pid, parent, created, ' '.join(argv), argv)


class FakeProcesses:
    def __init__(self, inventory=None):
        self.rows = inventory or {}
        self.started, self.bound, self.stopped = [], [], []
        self.inventory_error = False
        self.cleanup_error = False

    def inventory(self):
        if self.inventory_error:
            raise RuntimeError('inventory failed')
        return dict(self.rows)

    def bind(self, process):
        self.bound.append(process)

    def start(self, service, log):
        self.started.append(service)
        process = identity(service, pid=20 + len(self.started), created=200)
        self.rows[process.pid] = process
        return process

    def terminate_tree(self, process, inventory):
        if self.cleanup_error:
            raise RuntimeError('cleanup incomplete')
        self.stopped.append(process)
        if process.same_process(self.rows.get(process.pid)):
            self.rows.pop(process.pid)

    def close(self):
        pass



def supervisor(layout, backend, clock):
    root, _, _ = layout
    return paper.Supervisor(services(layout)[:1], backend, root / 'data/paper_services', root, lambda: clock[0])


def healthy(service, pid, now, age=0):
    save(service.status_path, {'pid': pid, 'places_orders': False, 'updated_at_utc': paper.iso(now),
        'accounts': {name: {'checked_at_utc': paper.iso(now - age), 'heartbeat_age_seconds': age,
                            'status': 'degraded', 'errors': {'provider': 'temporarily unavailable'}}
                     for name in service.account_ids}})


def test_adopts_windows_venv_launcher_and_worker_without_duplicate(layout):
    service = services(layout)[0]
    launcher = identity(service, 10, 1, 100)
    worker = identity(service, 11, 10, 101)
    backend = FakeProcesses({10: launcher, 11: worker})
    clock = [1000]
    control = supervisor(layout, backend, clock)
    control.tick()
    assert not backend.started and {p.pid for p in backend.bound} == {10, 11}
    assert control.managed['main'].adoptions == 1
    healthy(service, 11, 1400)
    clock[0] = 1400
    control.tick()
    assert control.managed['main'].status == 'healthy'
    assert not backend.stopped  # Provider errors with fresh heartbeats are not hangs.


def test_same_script_independent_duplicates_or_other_flags_fail_closed(layout):
    service = services(layout)[0]
    backend = FakeProcesses({10: identity(service, 10), 12: identity(service, 12)})
    control = supervisor(layout, backend, [1000])
    control.tick()
    assert control.managed['main'].status == 'conflict' and not backend.started
    backend.rows = {10: identity(service, command=(*service.command, '--once'))}
    control.tick()
    assert control.managed['main'].status == 'conflict' and not backend.started


def test_inventory_failure_never_launches_a_clone(layout):
    backend = FakeProcesses()
    backend.inventory_error = True
    control = supervisor(layout, backend, [1000])
    with pytest.raises(RuntimeError, match='inventory failed'):
        control.tick()
    assert not backend.started


def test_stalled_account_restart_waits_for_cleanup_and_bounded_backoff(layout):
    service = services(layout)[0]
    process = identity(service)
    backend = FakeProcesses({10: process})
    clock = [1000]
    control = supervisor(layout, backend, clock)
    control.tick()
    healthy(service, 10, 1400, age=301)
    clock[0] = 1400
    backend.cleanup_error = True
    control.tick()
    assert control.managed['main'].status == 'cleanup_blocked' and not backend.started
    backend.cleanup_error = False
    control.tick()
    state = control.managed['main']
    assert state.identity is None and state.next_retry == 1405 and len(backend.stopped) == 1
    clock[0] = 1404
    control.tick()
    assert not backend.started
    clock[0] = 1405
    control.tick()
    assert len(backend.started) == 1
    assert [paper.restart_delay(n) for n in (1, 2, 3, 20, 1000)] == [5, 10, 20, 300, 300]


def test_reused_pid_is_never_adopted_as_original_worker(layout):
    service = services(layout)[0]
    original = identity(service)
    backend = FakeProcesses({10: original})
    clock = [1000]
    control = supervisor(layout, backend, clock)
    control.tick()
    replacement = identity(service, created=999, command=('python.exe', 'unrelated.py'))
    backend.rows[10] = replacement
    clock[0] = 1010
    control.tick()
    assert backend.stopped == [original]
    assert backend.rows[10] == replacement
    assert control.managed['main'].identity is None and not backend.started


def test_descendant_cleanup_includes_evaluator_and_excludes_old_reused_parent_children(layout):
    service = services(layout)[0]
    root = identity(service, 10, 1, 100)
    child = identity(service, 11, 10, 101)
    evaluator = identity(service, 12, 11, 102, command=('python.exe', 'paper_trade_frozen_portfolio.py'))
    old = identity(service, 13, 10, 99)
    other = identity(service, 14, 1, 103)
    assert {p.pid for p in paper.descendant_identities(root, {p.pid: p for p in (root, child, evaluator, old, other)})} == {11, 12}


def test_producer_evaluation_and_shadow_arm_heartbeat_thresholds(layout):
    producer = services(layout)[1]
    process = identity(producer)
    save(producer.status_path, {'pid': 10, 'places_orders': False, 'updated_at_utc': paper.iso(1000)})
    assert paper.heartbeat_health(producer, process, 1250, 1000)[0] == 'healthy'
    assert paper.heartbeat_health(producer, process, 1601, 1000)[0] == 'global_heartbeat_stale'
    save(producer.status_path, {'pid': 10, 'places_orders': False, 'updated_at_utc': paper.iso(1400),
                               'status': 'degraded', 'error': 'TimeoutExpired: evaluation timeout'})
    assert paper.heartbeat_health(producer, process, 1400, 100)[0] == 'evaluator_timeout'
    save(producer.status_path, {'pid': 10, 'places_orders': False, 'updated_at_utc': paper.iso(1400),
                               'status': 'degraded', 'error': 'requests.Timeout: provider unavailable'})
    assert paper.heartbeat_health(producer, process, 1400, 1000)[0] == 'healthy'
    shadow = services(layout)[2]
    save(shadow.status_path, {'pid': 10, 'places_orders': False, 'heartbeat_at_ms': 1400_000,
        'accounts': {arm: {'heartbeat_at_ms': (1099 if arm == 'candidate_double' else 1400) * 1000}
                     for arm in paper.ARMS}})
    assert paper.heartbeat_health(shadow, identity(shadow), 1400, 400)[0] == 'account_heartbeat_stale:candidate_double'


@pytest.mark.skipif(os.name != 'nt', reason='Windows command parser')
def test_windows_command_parser_preserves_quoted_paths():
    argv = paper.split_windows_command('"C:\\My Python\\python.exe" -u "D:\\My Project\\scripts\\run_parallel_simulation.py" --require-existing')
    assert argv == ('C:\\My Python\\python.exe', '-u', 'D:\\My Project\\scripts\\run_parallel_simulation.py', '--require-existing')


@pytest.mark.skipif(os.name != 'nt', reason='Windows native process jobs')
def test_native_jobs_adopt_existing_dummy_workers_and_kill_future_descendants():
    # Dummy sleepers only: verify the real Windows venv redirector/job behavior
    # without touching running paper services or placing any market request.
    script = (
        "import subprocess, sys, time; "
        "first=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(first.pid, flush=True); sys.stdin.readline(); "
        "second=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(second.pid, flush=True); time.sleep(120)"
    )
    parent = subprocess.Popen([sys.executable, '-u', '-c', script], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    backend = paper.WindowsProcesses()
    root = None
    members = []
    try:
        existing_child = int(parent.stdout.readline())
        inventory = backend.inventory()
        root = inventory[parent.pid]
        members = paper.descendant_identities(root, inventory)
        assert existing_child in {member.pid for member in members}
        backend.bind(root)
        for member in members:
            backend.bind(member)
        parent.stdin.write('\n')
        parent.stdin.flush()
        future_child = int(parent.stdout.readline())
        time.sleep(.1)
        inventory = backend.inventory()
        descendants = paper.descendant_identities(root, inventory)
        assert future_child in {member.pid for member in descendants}
        backend.terminate_tree(root, inventory)
        parent.wait(timeout=5)
        remaining = backend.inventory()
        assert not ({parent.pid, existing_child, future_child} & remaining.keys())
        assert not backend.jobs
    finally:
        if root is not None:
            backend.terminate_tree(root, backend.inventory())
        backend.close()
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
