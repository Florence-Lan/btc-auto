"""No real services: preserve old experiment identity and reject duplicate writers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_paper_services import layout, save, identity, FakeProcesses
import run_paper_services as paper
import paper_service_extensions as extensions


def write_extensions(root, **extra):
    save(root / extensions.CONFIG, {'schema_version': 1, 'mode': 'SIMULATION',
                                   'places_orders': False, **extra})


def enable_reentry(root):
    save(root / extensions.REENTRY_CONFIG,
         {'mode': 'SIMULATION', 'places_orders': False, 'automatic_strategy_promotion': False})
    write_extensions(root, btc_reentry_shadow={'enabled': True,
        'config_path': extensions.REENTRY_CONFIG, 'output_dir': extensions.REENTRY_OUTPUT})
    output = root / extensions.REENTRY_OUTPUT
    save(output / 'manifest.json', {'started_at_ms': 123})
    for arm in extensions.ARMS:
        save(output / arm / 'state.json', {'mode': 'simulation'})
    return output


def freeze_sndk(layout):
    if os.name != 'nt':
        pytest.skip('Windows junction binding')
    root, shadow, _ = layout
    frozen = root / 'data/runtime/frozen_test'
    contents = {'scripts/run_sndk_4h_gate_shadow.py': '# original source\n',
                'config/profile.json': '{}', 'config/sndk_4h_gate_shadow_20261012.json': '{}'}
    files = {}
    for relative, body in contents.items():
        path = frozen / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding='utf-8')
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = {'source_sha256': {'scripts/run_sndk_4h_gate_shadow.py': files['scripts/run_sndk_4h_gate_shadow.py']},
                   'profile_sha256': {'config/profile.json': files['config/profile.json']},
                   'experiment_config_sha256': files['config/sndk_4h_gate_shadow_20261012.json']}
    save(shadow / 'manifest.json', {'fingerprint': fingerprint})
    record = {'schema_version': 1, 'service': 'sndk_shadow',
              'runtime_root': frozen.relative_to(root).as_posix(),
              'experiment_root': 'data/paper_trading/sndk_4h_gate_shadow_20261012',
              'fingerprint': fingerprint, 'files': files}
    save(frozen / 'runtime_manifest.json', record)
    # Creation only, with literal quoted paths; all targets are within tmp_path.
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = ('New-Item -ItemType Junction -Path ' + quote(frozen/'data')
               + ' -Target ' + quote(root/'data') + ' -ErrorAction Stop | Out-Null')
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                   check=True, capture_output=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    write_extensions(root, sndk_frozen_runtime={'runtime_root': record['runtime_root']})
    return frozen


def test_new_experiment_requires_all_existing_ledgers_and_fixed_paths(layout):
    root, shadow, _ = layout
    output = enable_reentry(root)
    services = paper.build_services(root, shadow)
    assert services[-1].name == 'btc_reentry_shadow'
    assert services[-1].account_ids == extensions.ARMS
    assert '--require-existing' in services[-1].command
    assert paper.canonical_command(services[-1].command, root) is not None
    assert paper.canonical_command((*services[-1].command, '--once'), root) is None
    changed = list(services[-1].command)
    changed[changed.index('--output-dir')+1] = str(root/'data/paper_trading/other')
    assert paper.canonical_command(changed, root) is None
    (output / extensions.ARMS[0] / 'state.json').unlink()
    with pytest.raises(ValueError, match='all four ledgers'):
        paper.build_services(root, shadow)
    assert not (output / extensions.ARMS[0] / 'state.json').exists()


def test_extension_cannot_enable_live_or_outside_workspace(layout):
    root, _, _ = layout
    write_extensions(root, btc_reentry_shadow={'enabled': True,
        'config_path': '../other.json', 'output_dir': extensions.REENTRY_OUTPUT})
    with pytest.raises(ValueError, match='binding'):
        extensions.load(root)
    write_extensions(root, sndk_frozen_runtime={'runtime_root': '../outside'})
    with pytest.raises(ValueError):
        extensions.load(root)
    raw = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': True}
    save(root / extensions.CONFIG, raw)
    with pytest.raises(ValueError, match='extensions'):
        extensions.load(root)


def test_frozen_runtime_preserves_old_fingerprint_and_rejects_source_edits(layout):
    root, shadow, _ = layout
    frozen = freeze_sndk(layout)
    original = (shadow / 'manifest.json').read_bytes()
    service = paper.build_services(root, shadow)[2]
    assert Path(service.command[2]) == frozen/'scripts/run_sndk_4h_gate_shadow.py'
    assert service.status_path == shadow/'status.json'
    assert (shadow/'manifest.json').read_bytes() == original
    (frozen/'scripts/run_sndk_4h_gate_shadow.py').write_text('# changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source changed'):
        paper.build_services(root, shadow)


def test_old_sndk_writer_is_a_conflict_when_frozen_runtime_selected(layout):
    root, shadow, _ = layout
    frozen = freeze_sndk(layout)
    service = paper.build_services(root, shadow)[2]
    old_command = list(service.command)
    old_command[2] = str(root/'scripts/run_sndk_4h_gate_shadow.py')
    old_process = identity(service, command=old_command)
    backend = FakeProcesses({old_process.pid: old_process})
    control = paper.Supervisor([service], backend, root/'data/control_test', root, lambda: 1000)
    control.tick()
    assert control.managed['sndk_shadow'].status == 'conflict'
    assert not backend.started and not backend.bound and not backend.stopped
    mixed_case = list(old_command)
    mixed_case[2] = str(root/'Scripts/RUN_SNDK_4H_GATE_SHADOW.PY')
    mixed_case.append('--once')  # Unknown flags still conflict with the same writer.
    other = identity(service, command=mixed_case)
    backend.rows = {other.pid: other}
    control.tick()
    assert control.managed['sndk_shadow'].status == 'conflict'
    assert not backend.started
