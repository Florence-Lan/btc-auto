#!/usr/bin/env python3
"""Keep existing, order-disabled paper services alive on Windows.

Only the fixed paper scripts below can be adopted/launched. Account
ledgers are never initialized or edited here. Status in data/paper_services
records process identities, heartbeat ages and bounded restart delays.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from process_lock import exclusive_process_lock
import paper_service_extensions as extensions

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/parallel_simulation_plan_20261005.json'
SHADOW = ROOT / 'data/paper_trading/sndk_4h_gate_shadow_20261012'
ARMS = ('baseline_normal', 'candidate_normal', 'baseline_double', 'candidate_double')
ALLOWED = ('run_parallel_simulation.py', 'run_parallel_btc_signals.py',
           'run_sndk_4h_gate_shadow.py', 'run_btc_reentry_shadow.py')
STOP = threading.Event()


def iso(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp,
                                  timezone.utc).isoformat()


def contained(root, value, prefix=None):
    path = (root / value).resolve()
    path.relative_to((root / prefix).resolve() if prefix else root.resolve())
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


@dataclass(frozen=True)
class Service:
    name: str
    command: tuple[str, ...]
    status_path: Path
    account_ids: tuple[str, ...] = ()
    global_stale_seconds: float = 600
    account_stale_seconds: float = 300
    startup_grace_seconds: float = 300


def build_services(root=ROOT, shadow=SHADOW, python=sys.executable):
    root = root.resolve()
    extra = extensions.load(root)
    plan = read_json(root / 'config/parallel_simulation_plan_20261005.json')
    if plan.get('execution_mode') != 'simulation' or plan.get('live_orders_allowed') is not False:
        raise ValueError('Only a simulation plan with live_orders_allowed=false is accepted')
    if plan.get('runner_script') != 'scripts/run_parallel_simulation.py':
        raise ValueError('Unexpected paper runner')
    if plan.get('btc_signal_producer_script') != 'scripts/run_parallel_btc_signals.py':
        raise ValueError('Unexpected paper producer')
    accounts = plan['accounts']
    if len(accounts) != 4 or {a['account_id'] for a in accounts} != {'btc', 'mu', 'sndk', 'skhynix'}:
        raise ValueError('The original four accounts are required')
    generation = contained(root, 'data/parallel_simulation/' + plan['plan_id'], 'data/parallel_simulation')
    if generation.parent != root / 'data/parallel_simulation':
        raise ValueError('Invalid account generation path')
    manifest = generation / 'manifest.json'
    if not manifest.is_file():
        raise ValueError('Existing main simulation manifest is required')
    for account in accounts:
        path = contained(root, account['state_path'], 'data/parallel_simulation')
        if path != generation / account['account_id'] / 'state.json' or not path.is_file():
            raise ValueError('Original account ledger missing or outside its generation')
        contained(root, account['strategy_path'], 'config')
    main_status = contained(root, plan['status_path'], 'data/parallel_simulation')
    if main_status != generation / 'status.json':
        raise ValueError('Status path must belong to the existing account generation')
    btc = next(account for account in accounts if account['account_id'] == 'btc')
    report = contained(root, btc['strategy_report_path'], 'data/parallel_simulation')
    report.relative_to(generation)
    shadow = contained(root, shadow, 'data/paper_trading')
    if not (shadow / 'manifest.json').is_file() or any(
            not (shadow / arm / 'state.json').is_file() for arm in ARMS):
        raise ValueError('Existing shadow manifest and all four ledgers are required; bootstrap separately')
    commands = [
        (python, '-u', str(root / 'scripts' / ALLOWED[0]), '--require-existing', '--poll-seconds', '30'),
        (python, '-u', str(root / 'scripts' / ALLOWED[1]), '--poll-seconds', '30'),
        (python, '-u', str(extra.get('sndk_script', root / 'scripts' / ALLOWED[2])), '--require-existing', '--poll-seconds', '30',
         '--output-dir', str(shadow)),
    ]
    if any(not Path(command[2]).is_file() for command in commands):
        raise ValueError('A required paper script is unavailable')
    services = [Service('main', commands[0], main_status, tuple(a['account_id'] for a in accounts)),
            Service('btc_signals', commands[1], report.with_name('status.json'), startup_grace_seconds=600),
            Service('sndk_shadow', commands[2], shadow / 'status.json', ARMS)]
    if 'reentry_output' in extra:
        extensions.require_reentry_ledgers(root, extra['reentry_config'], extra['reentry_output'])
        script = root / 'scripts' / ALLOWED[3]
        if not script.is_file():
            raise ValueError('Required BTC reentry script unavailable')
        command = (python, '-u', str(script), '--require-existing', '--poll-seconds', '30',
                   '--output-dir', str(extra['reentry_output']), '--config', str(extra['reentry_config']))
        services.append(Service('btc_reentry_shadow', command, extra['reentry_output'] / 'status.json',
                                ARMS, startup_grace_seconds=600))
    return services


def split_windows_command(command_line):
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argc = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(command_line, ctypes.byref(argc))
    if not pointer:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return tuple(pointer[i] for i in range(argc.value))
    finally:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(pointer)


def canonical_command(argv, root=ROOT, shadow=SHADOW):
    """Only equivalent defaults/path spellings are normalized; unknown flags fail."""
    argv = list(argv)
    executable = str((root / argv.pop(0)).resolve()).casefold() if argv else ''
    if argv and argv[0] == '-u':
        argv.pop(0)
    if not argv:
        return None
    script = contained(root, argv.pop(0))
    extra = extensions.load(root, verify_frozen=False)
    allowed_paths = {(root / 'scripts' / name).resolve() for name in ALLOWED}
    if 'sndk_script' in extra:
        allowed_paths.add(extra['sndk_script'].resolve())
    if script not in allowed_paths:
        return None
    script_name = script.name.casefold()
    flags = {}
    while argv:
        flag = argv.pop(0)
        if flag == '--require-existing' and flag not in flags:
            flags[flag] = True
        elif flag in ('--poll-seconds', '--output-dir', '--config') and flag not in flags and argv:
            flags[flag] = argv.pop(0)
        else:
            return None
    if flags.get('--poll-seconds', '30') != '30':
        return None
    if script_name == ALLOWED[1]:
        if set(flags) - {'--poll-seconds'}:
            return None
    elif not flags.get('--require-existing'):
        return None
    if script_name not in (ALLOWED[2], ALLOWED[3]) and '--output-dir' in flags:
        return None
    if script_name != ALLOWED[3] and '--config' in flags:
        return None
    if script_name == ALLOWED[3]:
        if 'reentry_output' not in extra:
            return None
        output = contained(root, flags.get('--output-dir', extra['reentry_output']), 'data/paper_trading')
        config = contained(root, flags.get('--config', extra['reentry_config']), 'config')
        if output != extra['reentry_output'] or config != extra['reentry_config']:
            return None
        return executable, str(script).casefold(), str(output).casefold(), str(config).casefold()
    output = str(contained(root, flags.get('--output-dir', root / 'data/paper_trading/sndk_4h_gate_shadow_20261012'), 'data/paper_trading')).casefold() \
        if script_name == ALLOWED[2] else None
    return executable, str(script).casefold(), output


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    creation_time: int
    command_line: str
    argv: tuple[str, ...]

    def same_process(self, other):
        return other is not None and (self.pid, self.creation_time, self.command_line) == \
            (other.pid, other.creation_time, other.command_line)


def descendant_identities(root_process, inventory):
    """Creation dates exclude children of an older process that reused the PID."""
    pending = [root_process]
    descendants = []
    seen = {root_process.pid}
    while pending:
        parent = pending.pop()
        for child in inventory.values():
            if child.pid not in seen and child.parent_pid == parent.pid and child.creation_time >= parent.creation_time:
                descendants.append(child)
                pending.append(child)
                seen.add(child.pid)
    return descendants


class WindowsProcesses:
    """CIM supplies cmdlines; native handles prevent terminating a reused PID."""
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('The paper process supervisor requires Windows')
        from ctypes import wintypes
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        self.k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.k.OpenProcess.restype = wintypes.HANDLE
        self.k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.k.CreateJobObjectW.restype = wintypes.HANDLE
        self.k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.jobs = {}
        self.children = {}

    def inventory(self):
        # Only Python paper workers and their descendants (including curl) are
        # relevant. Do not read other processes' command lines, which can contain
        # unrelated application credentials.
        expression = r"""
        $ErrorActionPreference = 'Stop'
        $all = @(Get-CimInstance Win32_Process)
        $selected = @($all | Where-Object { $_.Name -match '^python(w|[0-9.]*)?\.exe$' })
        $ids = @{}; foreach ($p in $selected) { $ids[[int]$p.ProcessId] = $true }
        do {
          $changed = $false
          foreach ($p in $all) {
            if ($ids.ContainsKey([int]$p.ParentProcessId) -and -not $ids.ContainsKey([int]$p.ProcessId)) {
              $selected += $p; $ids[[int]$p.ProcessId] = $true; $changed = $true
            }
          }
        } while ($changed)
        @($selected | ForEach-Object {
          [pscustomobject]@{ pid=[int]$_.ProcessId; parent_pid=[int]$_.ParentProcessId;
            creation_time=$_.CreationDate.ToUniversalTime().ToFileTimeUtc().ToString(); command_line=$_.CommandLine }
        }) | ConvertTo-Json -Compress
        """
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', expression],
                                capture_output=True, text=True, timeout=25,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError('Windows process inventory unavailable')
        rows = json.loads(result.stdout or '[]')
        if isinstance(rows, dict):
            rows = [rows]
        return {int(row['pid']): ProcessIdentity(int(row['pid']), int(row['parent_pid']),
                    int(row['creation_time']), row['command_line'], split_windows_command(row['command_line']))
                for row in rows if row.get('command_line')}

    def _open_verified(self, identity, rights):
        from ctypes import wintypes
        handle = self.k.OpenProcess(rights | 0x1000, False, identity.pid)
        if not handle:
            raise OSError('Unable to open the recorded process')
        times = [wintypes.FILETIME() for _ in range(4)]
        if not self.k.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            self.k.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        created = times[0].dwLowDateTime | (times[0].dwHighDateTime << 32)
        # CIM creation times have microsecond precision; FILETIME is 100 ns.
        if created // 10 != identity.creation_time // 10:
            self.k.CloseHandle(handle)
            raise OSError('Process PID has been reused')
        return handle

    def bind(self, identity):
        """A job kills future descendants even if the supervisor crashes."""
        from ctypes import wintypes
        class Basic(ctypes.Structure):
            _fields_ = [('per_process', ctypes.c_int64), ('per_job', ctypes.c_int64),
                ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t), ('maximum', ctypes.c_size_t),
                ('active', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('read', 'write', 'other', 'read_bytes', 'write_bytes', 'other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if identity.pid in self.jobs:
            return
        job = self.k.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        process = None
        try:
            if not self.k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            process = self._open_verified(identity, 0x0100 | 0x0001)
            if not self.k.AssignProcessToJobObject(job, process):
                raise ctypes.WinError(ctypes.get_last_error())
            self.jobs[identity.pid] = job
        except BaseException:
            self.k.CloseHandle(job)
            raise
        finally:
            if process:
                self.k.CloseHandle(process)

    def start(self, service, log_path):
        with log_path.open('ab', buffering=0) as stream:
            process = subprocess.Popen(service.command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
        self.children[process.pid] = process
        inventory = self.inventory()
        identity = inventory.get(process.pid)
        if identity is None:
            process.wait(timeout=5)
            raise RuntimeError('Paper process exited before its identity could be recorded')
        try:
            self.bind(identity)
        except BaseException:
            self.terminate_tree(identity, inventory)
            raise
        return identity

    def terminate_tree(self, identity, inventory):
        current = inventory.get(identity.pid)
        if current is not None and not identity.same_process(current):
            # The job still refers to the original process group, regardless of
            # PID reuse. Never traverse or terminate the unrelated replacement.
            job = self.jobs.pop(identity.pid, None)
            if job:
                self.k.CloseHandle(job)
            self.children.pop(identity.pid, None)
            return
        descendants = descendant_identities(identity, inventory)
        # Stop the parent first so it cannot launch more subprocesses. The job
        # catches any descendant created after the inventory snapshot.
        targets = ([identity] if current is not None else []) + list(reversed(descendants))
        handles = []
        try:
            for target in targets:
                try:
                    handle = self._open_verified(target, 0x0001 | 0x00100000)
                except OSError:
                    if self.inventory().get(target.pid) is None:
                        continue
                    raise
                handles.append(handle)
                if not self.k.TerminateProcess(handle, 1) and self.k.WaitForSingleObject(handle, 0) != 0:
                    # Killing a launcher can race a child exiting naturally.
                    # Windows returns ACCESS_DENIED for an already terminated
                    # process; its verified handle is signaled in that case.
                    raise ctypes.WinError(ctypes.get_last_error())
            for target in [identity] + descendants:
                job = self.jobs.pop(target.pid, None)
                if job:
                    self.k.CloseHandle(job)
            for handle in handles:
                if self.k.WaitForSingleObject(handle, 5000) != 0:
                    raise RuntimeError('Paper process tree cleanup did not complete')
        finally:
            for handle in handles:
                self.k.CloseHandle(handle)
        child = self.children.pop(identity.pid, None)
        if child is not None:
            child.wait(timeout=5)

    def close(self):
        for job in self.jobs.values():
            self.k.CloseHandle(job)
        self.jobs.clear()


def timestamp(payload, *keys):
    for key in keys:
        value = payload.get(key)
        if value is not None:
            try:
                if key.endswith('_ms'):
                    return float(value) / 1000
                return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError, AttributeError):
                pass
    return None


def heartbeat_health(service, identity, now, monitored_for, worker_pids=()):
    ages = {}
    try:
        payload = read_json(service.status_path)
    except (OSError, ValueError, UnicodeError):
        payload = {}
    updated = timestamp(payload, 'heartbeat_at_ms', 'heartbeat_at_utc', 'updated_at_utc')
    ages['global'] = None if updated is None else max(0, now - updated)
    grace = monitored_for < service.startup_grace_seconds
    accepted_pids = {identity.pid, *worker_pids}
    if payload and payload.get('places_orders') is not False:
        return 'unsafe_status', ages
    if (service.name == 'btc_signals' and payload.get('pid') in accepted_pids
            and payload.get('status') == 'degraded'
            and 'TimeoutExpired' in str(payload.get('error') or '')):
        # subprocess.run kills a venv redirector at its 240-second deadline;
        # its evaluator can survive. Restart the owned job to clean that child
        # before another evaluation is attempted.
        return 'evaluator_timeout', ages
    if not grace and payload.get('pid') not in accepted_pids:
        return 'status_pid_mismatch', ages
    if not grace and (ages['global'] is None or ages['global'] >= service.global_stale_seconds):
        return 'global_heartbeat_stale', ages
    for account_id in service.account_ids:
        account = payload.get('accounts', {}).get(account_id, {})
        heartbeat = timestamp(account, 'heartbeat_at_ms', 'heartbeat_at_utc', 'checked_at_utc')
        age = account.get('heartbeat_age_seconds') if 'heartbeat_age_seconds' in account else \
            (None if heartbeat is None else max(0, now - heartbeat))
        ages[account_id] = age
        if not grace and (age is None or float(age) >= service.account_stale_seconds):
            return 'account_heartbeat_stale:' + account_id, ages
    if grace and (payload.get('pid') not in accepted_pids or ages['global'] is None):
        return 'starting', ages
    # Degraded market inputs are valid heartbeats. Restarting a live worker for
    # a provider outage would discard its network retry backoff without fixing it.
    return 'healthy', ages


def restart_delay(failures):
    return min(300, 5 * 2 ** min(max(0, failures - 1), 6))


@dataclass
class Managed:
    identity: ProcessIdentity | None = None
    worker_identities: tuple[ProcessIdentity, ...] = ()
    observed_since: float = 0
    failures: int = 0
    next_retry: float = 0
    starts: int = 0
    adoptions: int = 0
    status: str = 'waiting'
    error: str | None = None
    heartbeat_ages: dict = field(default_factory=dict)


class Supervisor:
    def __init__(self, services, processes, directory, root=ROOT, clock=time.time):
        self.services, self.processes, self.directory = services, processes, directory
        self.root, self.clock = root, clock
        self.managed = {service.name: Managed() for service in services}
        self.boot = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + str(os.getpid())
        directory.mkdir(parents=True, exist_ok=True)

    def event(self, service, event, **details):
        path = self.directory / 'events.jsonl'
        if path.exists() and path.stat().st_size > 5_000_000:
            path.replace(path.with_suffix('.previous.jsonl'))
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time_utc': iso(self.clock()), 'service': service,
                                     'event': event, **details}, ensure_ascii=False) + '\n')

    def failure(self, service, state, error, now):
        state.failures += 1
        state.next_retry = now + restart_delay(state.failures)
        state.status, state.error = 'retry_wait', error
        self.event(service.name, 'failure', error=error, next_retry_utc=iso(state.next_retry))

    def cleanup(self, state, inventory):
        self.processes.terminate_tree(state.identity, inventory)
        # Remembered identities cover venv workers even if their launcher has
        # already exited or its PID was reused. A fresh snapshot is essential:
        # the root cleanup may already have killed these members.
        for member in state.worker_identities:
            current_inventory = self.processes.inventory()
            self.processes.terminate_tree(member, current_inventory)

    def matching(self, service, inventory):
        expected = canonical_command(service.command, self.root)
        matches, conflicts = [], []
        for identity in inventory.values():
            try:
                canonical = canonical_command(identity.argv, self.root)
                if canonical == expected:
                    matches.append(identity)
                # The same paper script with other flags must not race an
                # existing account writer, even if it does not match for adoption.
                elif any((candidate := contained(self.root, arg)).name.casefold() == Path(expected[1]).name.casefold()
                         and (candidate.parent == (self.root / 'scripts').resolve()
                              or candidate == extensions.load(self.root, verify_frozen=False).get('sndk_script'))
                         for arg in identity.argv[1:3] if not arg.startswith('-')):
                    conflicts.append(identity)
            except (ValueError, OSError):
                continue
        # Windows venv python.exe redirects to a child with exactly the same
        # argv. A launcher + worker chain is one service, not duplicate writers.
        matching_pids = {identity.pid for identity in matches}
        roots = [identity for identity in matches if identity.parent_pid not in matching_pids]
        return roots, conflicts, matches

    def tick(self):
        now = self.clock()
        inventory = self.processes.inventory()  # Fail closed: never spawn when inventory fails.
        for service in self.services:
            state = self.managed[service.name]
            if state.identity is not None:
                current = inventory.get(state.identity.pid)
                if not state.identity.same_process(current) or any(
                        not member.same_process(inventory.get(member.pid)) for member in state.worker_identities):
                    # A terminated producer may leave a paper evaluator alive.
                    # Cleanup must finish before replacement is allowed.
                    try:
                        self.cleanup(state, inventory)
                    except Exception as exc:
                        state.status, state.error = 'cleanup_blocked', type(exc).__name__ + ': ' + str(exc)[:180]
                        continue
                    state.identity = None
                    state.worker_identities = ()
                    self.failure(service, state, 'process_exited', now)
                    continue
                health, ages = heartbeat_health(service, state.identity, now, now - state.observed_since,
                                               tuple(member.pid for member in state.worker_identities))
                state.heartbeat_ages = ages
                if health not in ('healthy', 'starting'):
                    try:
                        self.cleanup(state, inventory)
                    except Exception as exc:
                        state.status, state.error = 'cleanup_blocked', type(exc).__name__ + ': ' + str(exc)[:180]
                        continue
                    state.identity = None
                    state.worker_identities = ()
                    self.failure(service, state, health, now)
                else:
                    state.status, state.error = health, None
                    if health == 'healthy' and now - state.observed_since >= 900:
                        state.failures = 0
                continue
            if now < state.next_retry:
                continue
            matches, conflicts, members = self.matching(service, inventory)
            if conflicts or len(matches) > 1:
                state.status, state.error = 'conflict', 'Other process for this paper script; refusing a duplicate'
                continue
            try:
                if matches:
                    identity = matches[0]
                    try:
                        self.processes.bind(identity)
                        for member in members:
                            if member.pid != identity.pid:
                                self.processes.bind(member)
                    except Exception:
                        # Partial adoption must not leave a job handle orphaned
                        # or create a second untracked service.
                        self.processes.terminate_tree(identity, self.processes.inventory())
                        raise
                    state.adoptions += 1
                    event = 'adopted'
                else:
                    logfile = self.directory / 'logs' / (service.name + '_' + self.boot + '_' + str(state.starts + 1) + '.log')
                    logfile.parent.mkdir(parents=True, exist_ok=True)
                    identity = self.processes.start(service, logfile)
                    # A newly launched venv worker inherits the launcher's job.
                    fresh_inventory = self.processes.inventory()
                    roots, conflicts, members = self.matching(service, fresh_inventory)
                    if conflicts or len(roots) != 1 or roots[0].pid != identity.pid:
                        self.processes.terminate_tree(identity, fresh_inventory)
                        raise RuntimeError('Paper process group could not be uniquely verified')
                    state.starts += 1
                    event = 'started'
                state.identity, state.observed_since = identity, now
                state.worker_identities = tuple(member for member in members if member.pid != identity.pid)
                state.status, state.error = 'starting', None
                inventory[identity.pid] = identity
                self.event(service.name, event, pid=identity.pid, creation_time=str(identity.creation_time))
            except Exception as exc:
                self.failure(service, state, type(exc).__name__ + ': ' + str(exc)[:180], now)
        self.publish(True)

    def publish(self, running, error=None):
        now = self.clock()
        write_json(self.directory / 'status.json', {'mode': 'SIMULATION', 'places_orders': False,
            'pid': os.getpid(), 'running': running, 'updated_at_utc': iso(now), 'error': error,
            'startup_coverage': 'current_user_logon; not before_login',
            'services': {name: {**asdict(state),
                'next_retry_utc': iso(state.next_retry) if state.next_retry else None}
                for name, state in self.managed.items()}})

    def shutdown(self):
        try:
            inventory = self.processes.inventory()
            for service in self.services:
                state = self.managed[service.name]
                if state.identity is not None:
                    try:
                        self.cleanup(state, inventory)
                    except Exception as exc:
                        self.event(service.name, 'cleanup_failed', error=type(exc).__name__ + ': ' + str(exc)[:180])
        finally:
            self.processes.close()
            self.publish(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shadow-output-dir', type=Path, default=SHADOW)
    parser.add_argument('--check', action='store_true', help='Validate configuration/ledgers without starting processes')
    args = parser.parse_args()
    try:
        services = build_services(shadow=args.shadow_output_dir)
    except (OSError, ValueError, KeyError) as exc:
        print('Paper service configuration rejected: ' + str(exc), flush=True)
        return 78
    if args.check:
        print(json.dumps({service.name: {'command': service.command, 'status_path': str(service.status_path)}
                          for service in services}, ensure_ascii=False, indent=2))
        return 0
    directory = ROOT / 'data/paper_services'
    lock = exclusive_process_lock(directory / 'supervisor.lock')
    try:
        lock.__enter__()
    except OSError as exc:
        print('Paper supervisor lock unavailable: ' + str(exc)[:180], flush=True)
        return 73
    try:
        processes = WindowsProcesses()
        supervisor = Supervisor(services, processes, directory)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: STOP.set())
        try:
            while not STOP.is_set():
                try:
                    # Safety guards are re-read before every process action.
                    current = build_services(shadow=args.shadow_output_dir)
                    if current != services:
                        raise ValueError('Service paths changed; restart the supervisor after review')
                except (OSError, ValueError, KeyError) as exc:
                    supervisor.event('supervisor', 'configuration_rejected', error=str(exc)[:180])
                    return 78
                try:
                    supervisor.tick()
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)[:180]
                    supervisor.event('supervisor', 'monitor_error', error=error)
                    supervisor.publish(True, error)
                STOP.wait(10)
        finally:
            supervisor.shutdown()
    finally:
        lock.__exit__(None, None, None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
