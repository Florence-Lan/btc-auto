"""Review and activate the main-stock trend overlay without resetting accounts.

No action runs on import. plan/check/capture-control only write research evidence.
pause/apply/resume are separate, explicit maintenance actions. Canceled comparison
tasks, their manifests and all 98 financial accounts remain stopped and unchanged.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GENERATION = 'data/parallel_simulation/btc_memory_stocks_latest_1000_each_20261007'
PLAN_PATH = 'config/parallel_simulation_plan_20261005.json'
PROFILE = 'config/stock_main_trend_20261102.json'
OUTPUT = 'data/research/main_trend_activation_20261102'
REVISION = 'main_stock_trend_20261102'
POLICY_KEY = 'main_stock_trend_policy'
ACCOUNTS = ('btc', 'mu', 'sndk', 'skhynix')
STOCKS = ACCOUNTS[1:]
SUPERVISOR = 'btc-auto-paper-services-1BB40889'
BINDINGS = {
    SUPERVISOR: 'data/paper_services/start_paper_services.ps1',
    'btc-auto-exit-response-1BB40889': 'data/paper_trading/stock_exit_response_shadow_20261023/start_exit_response.ps1',
    'btc-auto-early-recovery-1BB40889': 'data/paper_trading/stock_early_recovery_shadow_20261028/start_early_recovery.ps1',
    'btc-auto-entry-timing-1BB40889': 'data/paper_trading/stock_entry_timing_trial_20261029/start_entry_timing.ps1',
    'btc-auto-technology-1BB40889': 'data/paper_trading/stock_technology_trial_20261030/start_technology.ps1',
    'btc-auto-short-confirmation-1BB40889': 'scripts/run_stock_short_confirmation_trial.py',
}
COMPARISONS = (
    ('data/paper_trading/sndk_4h_gate_shadow_20261012', 1, 4),
    ('data/paper_trading/btc_reentry_compat_shadow_20261020', 1, 4),
    ('data/paper_trading/stock_macro_shadow_20261021', 2, 12),
    ('data/paper_trading/stock_exit_response_shadow_20261023', 2, 24),
    ('data/paper_trading/stock_early_recovery_shadow_20261028', 2, 12),
    ('data/paper_trading/stock_entry_timing_trial_20261029', 2, 18),
    ('data/paper_trading/stock_technology_trial_20261030', 2, 12),
    ('data/paper_trading/stock_short_confirmation_trial_20261030', 2, 12),
)
CHANGED_SOURCES = {
    'scripts/run_parallel_simulation.py', 'scripts/main_stock_runtime.py',
    'scripts/main_stock_rule_revision.py', 'scripts/main_stock_trend.py',
    'scripts/main_stock_sector.py', 'scripts/stock_technology_guard.py',
}
STATE_CHANGES = {
    'rule', 'rule_history', 'main_trend_active_after_ms', 'signal_active_after_ms',
    'pending_signal', 'main_trend_status',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def equal(left, right):
    """Preserve scalar types as well as values in audited financial/config data."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def digest(path, limit=None):
    result, count = hashlib.sha256(), 0
    with Path(path).open('rb') as handle:
        while limit is None or count < limit:
            chunk = handle.read(1_048_576 if limit is None else min(1_048_576, limit-count))
            if not chunk:
                break
            count += len(chunk)
            result.update(chunk)
    require(limit is None or count == limit, 'Captured history was truncated: '+str(path))
    return result.hexdigest()


def contained(root, relative, prefix=None):
    path = (root / relative).resolve()
    path.relative_to((root / prefix).resolve() if prefix else root.resolve())
    return path


def atomic_write(path, payload, *, new=False):
    path = Path(path)
    require(not new or not path.exists(), 'Refusing to overwrite evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.main_trend.tmp')
    require(not temporary.exists(), 'Unexpected maintenance temporary file')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def source_closure(root, paths):
    pending, result = [root/name for name in paths], {}
    while pending:
        path = pending.pop().resolve()
        relative = path.relative_to(root).as_posix()
        if relative in result:
            continue
        raw = path.read_bytes()
        result[relative] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            names = ([item.name for item in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = root/'scripts'/(name.split('.')[0]+'.py')
                if candidate.is_file():
                    pending.append(candidate)
    return dict(sorted(result.items()))


def build_plan(root=ROOT, profile=PROFILE):
    import run_parallel_simulation as paper
    import main_stock_trend as trend
    root = Path(root).resolve()
    active = read(root/PLAN_PATH)
    require(active.get('execution_mode') == 'simulation' and active.get('live_orders_allowed') is False,
            'Only order-disabled simulation can be maintained')
    require(active.get('plan_id') == Path(GENERATION).name
            and active.get('main_stock_runtime') == 'price_led_short_v1', 'Unexpected main generation/runtime')
    require(len(active['accounts']) == 4 and {a['account_id'] for a in active['accounts']} == set(ACCOUNTS),
            'The original four accounts are required')
    profile_path = contained(root, profile, 'config')
    candidate = {**deepcopy(active), 'stock_research_profile': profile_path.relative_to(root).as_posix(),
        'stock_rule_revision_scope': 'main_trend_only',
        'stock_rule_revision_reason': 'User authorized prospective main-stock trend repair; hourly direction, sector context and existing-position invalidation overlay; preserve all budgets and financial history.'}
    changes = {}
    for account in active['accounts']:
        name = account['account_id']
        path = contained(root, account['state_path'])
        require(path == root/GENERATION/name/'state.json' and path.is_file(), 'Unexpected main ledger path')
        state = read(path)
        require(state.get('initial_balance') == 1000 and state.get('places_orders', False) is False,
                'Existing initial capital/order-disabled account required')
        if name == 'btc':
            continue
        old = state['rule']
        require(POLICY_KEY not in old and 'main_trend_active_after_ms' not in state,
                'Trend already active or partial migration detected')
        with patch.object(paper, 'ROOT', root):
            new = paper.stock_config(candidate, account['symbol'])
        require(equal(new, {**old, POLICY_KEY: trend.POLICY}), 'Trend profile alters an unrelated rule: '+name)
        changes[name] = {'path': account['state_path'], 'previous_rule': old, 'new_rule': new,
                         'created_at_utc': state['created_at_utc'], 'start_ms': state['start_ms']}
    base = root/GENERATION
    manifest = read(base/'manifest.json')
    sources = source_closure(root, ('scripts/run_parallel_simulation.py', 'scripts/run_parallel_btc_signals.py',
                                   'scripts/run_paper_services.py', 'scripts/main_stock_rule_revision.py',
                                   'scripts/main_stock_trend.py', 'scripts/main_stock_sector.py'))
    old_sources = manifest['current_source_sha256']
    for name in old_sources:
        sources.setdefault(name, digest(root/name))
    drift = {name: {'before': expected, 'after': sources[name]} for name, expected in old_sources.items()
             if expected != sources[name]}
    require(set(drift) <= CHANGED_SOURCES, 'Unexpected preexisting source drift: '+str(sorted(set(drift)-CHANGED_SOURCES)))
    bindings = read(root/'config/paper_service_extensions_20261020.json')
    for key in ('sndk_shadow', 'btc_reentry_shadow', 'stock_macro_shadow'):
        require(bindings.get(key, {}).get('enabled') is False, 'Canceled comparison would restart: '+key)
    return {'schema_version': 1, 'revision_id': REVISION, 'prepared_at_utc': utc(),
        'candidate_plan': candidate, 'main_rule_changes': changes,
        'active_plan_sha256': digest(root/PLAN_PATH), 'profile_sha256': digest(profile_path),
        'base_config_sha256': digest(contained(root, read(profile_path)['base_config'], 'config')),
        'manifest_sha256': digest(base/'manifest.json'), 'bindings_sha256': digest(root/'config/paper_service_extensions_20261020.json'),
        'source_sha256': dict(sorted(sources.items())), 'source_changes': drift,
        'helper_sha256': digest(Path(__file__)), 'financial_writes_planned': False,
        'account_count': 102, 'main_stock_accounts_changed': 3, 'comparisons_restarted': False}


def check_reviewed(reviewed, actual):
    ignored = {'prepared_at_utc'}
    require(equal({k:v for k,v in reviewed.items() if k not in ignored},
                  {k:v for k,v in actual.items() if k not in ignored}),
            'Concrete reviewed plan differs; regenerate and review before maintenance')


def powershell(command):
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        "$ErrorActionPreference='Stop';[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();"+command],
        capture_output=True, text=True, encoding='utf-8', timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0, 'Maintenance task operation failed: '+result.stderr[-1200:])
    return result.stdout


def tasks(root=ROOT):
    names = ','.join("'"+name+"'" for name in BINDINGS)
    raw = powershell("@(Get-ScheduledTask -TaskName "+names+") | ForEach-Object { [pscustomobject]@{name=$_.TaskName;state=[string]$_.State;enabled=$_.Settings.Enabled;actions=@($_.Actions | ForEach-Object {[pscustomobject]@{execute=$_.Execute;arguments=$_.Arguments;working_directory=$_.WorkingDirectory}})} } | ConvertTo-Json -Depth 5 -Compress")
    values = json.loads(raw)
    result = {row['name']: row for row in (values if isinstance(values, list) else [values])}
    require(set(result) == set(BINDINGS), 'Missing/ambiguous recovery task')
    archived = read(root/'data/research/cancel_comparisons_20261101/final_verification.json')['live_checks']['task_states']
    for name, launcher in BINDINGS.items():
        actions = result[name]['actions']
        require(len(actions) == 1 and str(root/launcher).casefold() in actions[0]['arguments'].casefold()
                and Path(actions[0]['working_directory']).resolve() == root,
                'Unexpected task ownership: '+name)
        require(equal(actions, archived[name]['actions']), 'Task action differs from cancellation verification: '+name)
        if name != SUPERVISOR:
            require(result[name]['enabled'] is False and result[name]['state'] != 'Running',
                    'Canceled comparison task is enabled/running: '+name)
    return result


def matching(inventory, path):
    return [item for item in inventory.values()
            if any(arg.casefold() == str(path).casefold() for arg in item.argv[1:])]


def assert_stopped(root=ROOT):
    from run_paper_services import WindowsProcesses
    current = tasks(root)
    require(current[SUPERVISOR]['enabled'] is False and current[SUPERVISOR]['state'] != 'Running',
            'Main supervisor must be explicitly disabled/stopped first')
    scripts = ['run_paper_services.py', 'run_parallel_simulation.py', 'run_parallel_btc_signals.py',
        'run_btc_reentry_shadow.py', 'run_stock_macro_shadow.py', 'run_stock_exit_response_shadow.py',
        'run_stock_early_recovery_shadow.py', 'run_stock_entry_timing_trial.py',
        'run_stock_technology_trial.py', 'run_stock_short_confirmation_trial.py', 'run_sndk_4h_gate_shadow.py']
    backend = WindowsProcesses()
    try:
        inventory = backend.inventory()
        require(not any(matching(inventory, root/'scripts'/name) for name in scripts), 'A protected writer remains')
        frozen = root/read(root/'config/paper_service_extensions_20261020.json')['sndk_frozen_runtime']['runtime_root']
        require(not matching(inventory, frozen/'scripts/run_sndk_4h_gate_shadow.py'), 'Frozen comparison remains running')
    finally:
        backend.close()
    return current


def capture_control(output, root=ROOT):
    from run_paper_services import WindowsProcesses, descendant_identities
    backend = WindowsProcesses()
    try:
        current, inventory = tasks(root), backend.inventory()
        found = matching(inventory, root/'scripts/run_paper_services.py')
        ids = {row.pid for row in found}
        tops = [row for row in found if row.parent_pid not in ids]
        require(len(tops) == 1, 'Missing/ambiguous owned supervisor process')
        parent = tops[0]
        atomic_write(output/'control_before.json', {'captured_at_utc': utc(), 'tasks': current,
            'root': asdict(parent), 'members': [asdict(item) for item in [parent, *descendant_identities(parent, inventory)]],
            'observer_processes': [asdict(row) for row in matching(inventory, root/'scripts/run_cross_market_observer.py')]}, new=True)
    finally:
        backend.close()


def pause(output, root=ROOT):
    from run_paper_services import WindowsProcesses, ProcessIdentity
    before, current = read(output/'control_before.json'), tasks(root)
    require(equal(current, before['tasks']), 'Task state/definition changed; recapture before stopping')
    backend = WindowsProcesses()
    try:
        inventory = backend.inventory()
        raw = before['root']
        identity = ProcessIdentity(**{**raw, 'argv': tuple(raw['argv'])})
        require(identity.same_process(inventory.get(identity.pid)), 'Supervisor identity changed')
        powershell("Disable-ScheduledTask -TaskName '"+SUPERVISOR+"' | Out-Null\nStop-ScheduledTask -TaskName '"+SUPERVISOR+"'")
        for raw in before['members']:
            identity = ProcessIdentity(**{**raw, 'argv': tuple(raw['argv'])})
            inventory = backend.inventory()
            if identity.same_process(inventory.get(identity.pid)):
                backend.terminate_tree(identity, inventory)
        atomic_write(output/'paused.json', {'paused_at_utc': utc(), 'tasks': assert_stopped(root)}, new=True)
    finally:
        backend.close()


def capture_preservation(output, root=ROOT):
    """Snapshot all 102 states and hash financial journals at exact byte prefixes."""
    state_paths = [root/GENERATION/name/'state.json' for name in ACCOUNTS]
    metadata = {root/PLAN_PATH, root/GENERATION/'manifest.json', root/'config/paper_service_extensions_20261020.json'}
    for relative, depth, count in COMPARISONS:
        base = root/relative
        found = sorted(base.glob('*/state.json' if depth == 1 else '*/*/state.json'))
        require(len(found) == count, 'Unexpected canceled account count: '+relative)
        state_paths.extend(found)
        metadata.update(base/name for name in ('manifest.json', 'status.json', 'cancellation_20261101.json') if (base/name).is_file())
    result = {'captured_at_utc': utc(), 'states': {}, 'files': {}, 'initial_balance_total': 0}
    histories = set()
    for path in state_paths:
        raw, relative = path.read_bytes(), path.relative_to(root).as_posix()
        state = json.loads(raw)
        require(state.get('initial_balance') == 1000 and state.get('places_orders', False) is False, 'Wrong account mode/capital')
        result['initial_balance_total'] += state['initial_balance']
        result['states'][relative] = state
        backup = output/'stopped_before'/relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(raw)
        for child in path.parent.iterdir():
            if (child.is_file() and child.suffix.lower() in ('.jsonl', '.csv', '.json')
                    and any(token in child.name.lower() for token in ('fills', 'funding', 'trades', 'balance', 'performance', 'rule_revisions'))):
                histories.add(child)
    require(len(result['states']) == 102, 'Exactly 102 preserved accounts are required')
    for path in sorted(metadata | histories):
        size = path.stat().st_size
        relative = path.relative_to(root).as_posix()
        result['files'][relative] = {'size': size, 'sha256': digest(path, size),
                                    'history': path in histories}
        if path in metadata:
            backup = output/'stopped_before'/relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(path.read_bytes())
    atomic_write(output/'preservation.json', result, new=True)
    return result


def verify_stock_transition(before, after, expected, activated):
    require(equal({k:v for k,v in before.items() if k not in STATE_CHANGES},
                  {k:v for k,v in after.items() if k not in STATE_CHANGES}),
            'Stock migration changed financial/inventory/epoch/consumed-signal data')
    require(equal(after['rule'], expected) and after['main_trend_active_after_ms'] == activated
            and after['signal_active_after_ms'] == activated and after['pending_signal'] is None,
            'Wrong forward activation or pending cancellation')
    require(equal(after['rule_history'][:-1], before.get('rule_history', [])), 'Historical rule chain was altered')
    revision = after['rule_history'][-1]
    require(revision['scope'] == 'main_trend_only' and type(revision['activated_at_ms']) is int
            and revision['activated_at_ms'] == activated and equal(revision['previous_rule'], before['rule'])
            and equal(revision['new_rule'], expected), 'Unbound main trend revision')


def verify_preservation(output, proof, activated, *, live=False, root=ROOT):
    before = read(output/'preservation.json')
    mutable = {GENERATION+'/'+name+'/state.json': name for name in STOCKS}
    for relative, old in before['states'].items():
        current = read(root/relative)
        is_main = relative.startswith(GENERATION+'/')
        if relative in mutable:
            name = mutable[relative]
            if not live:
                verify_stock_transition(old, current, proof['main_rule_changes'][name]['new_rule'], activated)
            else:
                require(equal(current['rule'], proof['main_rule_changes'][name]['new_rule'])
                        and current['main_trend_active_after_ms'] == activated
                        and current['signal_active_after_ms'] == activated, 'Live main trend activation changed')
                for key in ('created_at_utc', 'start_ms', 'initial_balance', 'symbol', 'mode', 'places_orders'):
                    require(equal(current.get(key), old.get(key)), 'Original account identity changed')
        elif live and is_main:
            for key in ('created_at_utc', 'initial_balance', 'symbol', 'mode', 'trade_profit_budget_policy',
                        'trade_profit_budget_active_after_ms'):
                require(equal(current.get(key), old.get(key)), 'BTC policy/identity changed')
        else:
            require(equal(current, old), 'Unrelated financial account was changed: '+relative)
            backup = output/'stopped_before'/relative
            require(digest(root/relative) == digest(backup), 'Unrelated account bytes were rewritten')
    for relative, row in before['files'].items():
        path = root/relative
        if relative in (PLAN_PATH, GENERATION+'/manifest.json'):
            continue
        if row['history']:
            require(path.stat().st_size >= row['size'] and digest(path, row['size']) == row['sha256'],
                    'Financial journal prefix changed: '+relative)
            may_append = relative in {GENERATION+'/'+name+'/rule_revisions.jsonl' for name in STOCKS}
            if not (live and relative.startswith(GENERATION+'/')) and not may_append:
                require(path.stat().st_size == row['size'], 'Stopped/canceled journal was appended: '+relative)
        else:
            require(digest(path) == row['sha256'], 'Canceled metadata/binding changed: '+relative)
    require(equal(read(root/PLAN_PATH), proof['candidate_plan']), 'Active plan differs from reviewed plan')
    manifest = read(root/GENERATION/'manifest.json')
    require(equal(manifest['current_source_sha256'], proof['source_sha256']), 'Runtime source pins differ')
    for name, expected in proof['source_sha256'].items():
        require(digest(root/name) == expected, 'Pinned runtime source changed: '+name)
    return {'result': 'PASS', 'verified_at_utc': utc(), 'accounts': 102, 'canceled_accounts_unchanged': 98,
            'main_initial_balance_total': sum(read(root/GENERATION/name/'state.json')['initial_balance'] for name in ACCOUNTS),
            'financial_prefixes_unchanged': True, 'live_financial_changes_allowed': live}


def apply(output, reviewed, root=ROOT):
    import run_parallel_simulation as paper
    from process_lock import exclusive_process_lock
    import run_paper_services as services
    require(not (output/'activation.json').exists() and not (output/'activation_failed.json').exists(),
            'Prior activation/failure evidence exists; review before another attempt')
    assert_stopped(root)
    with ExitStack() as stack:
        for relative in ('data/paper_services/supervisor.lock', GENERATION+'/runner.lock',
                         GENERATION+'/btc_signals_entry_timing_20261011/producer.lock'):
            stack.enter_context(exclusive_process_lock(root/relative))
        assert_stopped(root)
        actual = build_plan(root, reviewed['candidate_plan']['stock_research_profile'])
        check_reviewed(reviewed, actual)
        before = capture_preservation(output, root)
        activated = int(time.time()*1000)
        require(all(activated > state.get('trade_profit_budget_active_after_ms', 0)
                    for path, state in before['states'].items() if path.startswith(GENERATION+'/')),
                'Activation clock precedes existing budget')
        try:
            manifest = read(root/GENERATION/'manifest.json')
            old_manifest = deepcopy(manifest)
            with patch.object(paper, 'ROOT', root), patch.object(paper, 'now_ms', lambda: activated):
                paper.synchronize_stock_rules(actual['candidate_plan'], root/GENERATION, manifest)
            manifest = read(root/GENERATION/'manifest.json')
            manifest.setdefault('source_compatibility_revisions', []).append({
                'revision_id': REVISION, 'applied_at_utc': utc(),
                'previous_current_source_sha256': deepcopy(old_manifest['current_source_sha256']),
                'replacement_current_source_sha256': actual['source_sha256'],
                'all_original_accounts_and_history_preserved': True,
                'canceled_comparison_manifests_unchanged': True,
                'evidence': (output/'activation.json').relative_to(root).as_posix()})
            manifest['current_source_sha256'] = actual['source_sha256']
            # The generation's original source_hashes, start, capital and historical
            # stock_rule reference remain intact; the revision records the new rule.
            allowed_manifest = {'rule_revisions', 'current_source_sha256', 'source_compatibility_revisions'}
            require(equal({k:v for k,v in old_manifest.items() if k not in allowed_manifest},
                          {k:v for k,v in manifest.items() if k not in allowed_manifest}), 'Unexpected manifest mutation')
            require(equal(manifest['rule_revisions'][:-3], old_manifest.get('rule_revisions', [])), 'Old manifest revisions changed')
            atomic_write(root/GENERATION/'manifest.json', manifest)
            atomic_write(root/PLAN_PATH, actual['candidate_plan'])
            require([service.name for service in services.build_services(root=root)] == ['main', 'btc_signals'],
                    'Resume configuration would restart a comparison')
            result = verify_preservation(output, actual, activated, root=root)
            record = {**actual, 'success': True, 'activated_at_ms': activated,
                      'activated_at_utc': datetime.fromtimestamp(activated/1000, timezone.utc).isoformat(),
                      'preservation_check': result, 'completed_at_utc': utc()}
            atomic_write(output/'activation.json', record, new=True)
            return record
        except BaseException as exc:
            atomic_write(output/'activation_failed.json', {'failed_at_utc': utc(), 'error': str(exc),
                'main_rule_may_be_partially_applied': True, 'financial_history_never_rewound': True,
                'resume_forbidden_until_reviewed': True}, new=True)
            raise


def resume(output, root=ROOT):
    import run_paper_services as services
    record = read(output/'activation.json')
    require(record.get('success') is True and not (output/'activation_failed.json').exists(), 'Successful reviewed activation required')
    assert_stopped(root)
    verify_preservation(output, record, record['activated_at_ms'], root=root)
    before = read(output/'control_before.json')
    current = tasks(root)
    require(equal(current[SUPERVISOR]['actions'], before['tasks'][SUPERVISOR]['actions']), 'Supervisor action changed')
    require(before['tasks'][SUPERVISOR]['enabled'] is True and before['tasks'][SUPERVISOR]['state'] == 'Running',
            'Original supervisor was not enabled/running')
    require([service.name for service in services.build_services(root=root)] == ['main', 'btc_signals'], 'Unexpected resume service')
    powershell("Enable-ScheduledTask -TaskName '"+SUPERVISOR+"' | Out-Null\nStart-ScheduledTask -TaskName '"+SUPERVISOR+"'")
    atomic_write(output/'resumed.json', {'resumed_at_utc': utc(), 'tasks': tasks(root), 'services': ['main', 'btc_signals']}, new=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'check', 'capture-control', 'pause', 'apply', 'verify', 'resume'))
    parser.add_argument('--profile', default=PROFILE)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--plan-file', default='plan.json', help='New evidence filename inside --output')
    parser.add_argument('--reviewed-plan', type=Path)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    output = contained(ROOT, args.output, 'data/research')
    output.mkdir(parents=True, exist_ok=True)
    if args.action in ('plan', 'check'):
        result = build_plan(ROOT, args.profile)
        if args.action == 'plan':
            require(Path(args.plan_file).name == args.plan_file and args.plan_file.endswith('.json'),
                    '--plan-file must be a JSON filename within the evidence directory')
            atomic_write(output/args.plan_file, result, new=True)
        else:
            require(args.reviewed_plan is not None, '--reviewed-plan is required')
            check_reviewed(read(args.reviewed_plan), result)
        print(json.dumps({'action': args.action, 'passed': True, 'output': str(output),
                          'source_changes': result['source_changes'], 'production_writes': False}))
    elif args.action == 'capture-control':
        capture_control(output)
    elif args.action == 'pause':
        require(args.reviewed_plan is not None, '--reviewed-plan is required before pausing')
        check_reviewed(read(args.reviewed_plan), build_plan(ROOT, args.profile))
        pause(output)
    elif args.action == 'apply':
        require(args.reviewed_plan is not None, '--reviewed-plan is required')
        record = apply(output, read(args.reviewed_plan))
        print(json.dumps({'success': True, 'activated_at_utc': record['activated_at_utc'],
                          'preservation_check': record['preservation_check']}))
    elif args.action == 'resume':
        resume(output)
    else:
        record = read(output/'activation.json')
        result = verify_preservation(output, record, record['activated_at_ms'], live=args.live)
        atomic_write(output/('verified_live.json' if args.live else 'verified_stopped.json'), result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
