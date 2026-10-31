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


def test_corrected_btc_binding_preserves_old_generation_and_refuses_other_paths(layout):
    root, shadow, _ = layout
    old = enable_reentry(root)
    original = (old / 'manifest.json').read_bytes()
    new_config = 'config/btc_reentry_compat_shadow_20261020.json'
    new_output = extensions.REENTRY_BINDINGS[new_config]
    save(root / new_config, {'mode': 'SIMULATION', 'places_orders': False, 'automatic_strategy_promotion': False})
    save(root / new_output / 'manifest.json', {'started_at_ms': 456})
    for arm in extensions.ARMS:
        save(root / new_output / arm / 'state.json', {'mode': 'simulation'})
    write_extensions(root, btc_reentry_shadow={'enabled': True, 'config_path': new_config, 'output_dir': new_output})
    service = paper.build_services(root, shadow)[-1]
    assert service.status_path == root / new_output / 'status.json'
    assert paper.canonical_command(service.command, root) is not None
    assert (old / 'manifest.json').read_bytes() == original
    write_extensions(root, btc_reentry_shadow={'enabled': True, 'config_path': new_config, 'output_dir': extensions.REENTRY_OUTPUT})
    with pytest.raises(ValueError, match='binding'):
        extensions.load(root)


def test_stock_macro_service_requires_all_twelve_ledgers_and_exact_command(layout):
    root, shadow, _ = layout
    save(root / extensions.MACRO_CONFIG, {'mode': 'SIMULATION', 'places_orders': False,
        'automatic_strategy_promotion': False, 'symbols': list(extensions.MACRO_SYMBOLS)})
    output = root / extensions.MACRO_OUTPUT
    save(output / 'manifest.json', {'started_at_ms': 456})
    for symbol in extensions.MACRO_SYMBOLS:
        for arm in extensions.ARMS:
            save(output / symbol / arm / 'state.json', {'mode': 'simulation'})
    write_extensions(root, stock_macro_shadow={'enabled': True,
        'config_path': extensions.MACRO_CONFIG, 'output_dir': extensions.MACRO_OUTPUT})
    service = paper.build_services(root, shadow)[-1]
    assert service.name == 'stock_macro_shadow' and len(service.account_ids) == 12
    assert paper.canonical_command(service.command, root) is not None
    assert paper.canonical_command((*service.command, '--once'), root) is None
    (output / extensions.MACRO_SYMBOLS[0] / extensions.ARMS[0] / 'state.json').unlink()
    with pytest.raises(ValueError, match='all twelve'):
        paper.build_services(root, shadow)


def test_pinned_review_environment_rejects_binary_drift_without_editing_global_settings(layout):
    root, _, _ = layout
    assert extensions.review_environment(root) == {}
    executable = root / 'review.exe'
    executable.write_bytes(b'first')
    config = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'provider': 'codex', 'model': 'gpt-6.1-sol', 'reasoning_effort': 'medium',
        'codex_bin': str(executable), 'codex_sha256': hashlib.sha256(b'first').hexdigest(),
        'cli_version': 'fixture'}
    save(root / extensions.REVIEW_CONFIG, config)
    expected = extensions.review_environment(root)
    assert expected['LLM_TRADE_GATE_ENABLED'] == 'true'
    assert expected['LLM_CODEX_REASONING_EFFORT'] == 'medium'
    executable.write_bytes(b'changed')
    with pytest.raises(ValueError, match='missing or changed'):
        extensions.review_environment(root)


def test_explicit_disabled_comparisons_need_no_shadow_ledgers_scripts_or_frozen_runtime(layout):
    root, shadow, plan = layout
    main_before = {a['account_id']: (root/a['state_path']).read_bytes() for a in plan['accounts']}
    write_extensions(root, sndk_shadow={'enabled': False},
        sndk_frozen_runtime={'runtime_root': 'data/runtime/absent_historical_runtime'},
        btc_reentry_shadow={'enabled': False, 'config_path': extensions.REENTRY_CONFIG,
                           'output_dir': extensions.REENTRY_OUTPUT},
        stock_macro_shadow={'enabled': False, 'config_path': extensions.MACRO_CONFIG,
                            'output_dir': extensions.MACRO_OUTPUT})
    for path in [shadow/'manifest.json', *(shadow/arm/'state.json' for arm in paper.ARMS),
                 *(root/'scripts'/name for name in paper.ALLOWED[2:])]:
        path.unlink()
    assert extensions.load(root) == {'sndk_enabled': False}
    for _ in range(2):
        services = paper.build_services(root, shadow)
        assert [s.name for s in services] == ['main', 'btc_signals']
        assert all(paper.canonical_command(service.command, root) is not None for service in services)
    assert not (shadow/'manifest.json').exists()
    assert not (root/'data/runtime/absent_historical_runtime').exists()
    assert main_before == {a['account_id']: (root/a['state_path']).read_bytes() for a in plan['accounts']}


@pytest.mark.parametrize('switch', [None, False, {}, {'enabled': None}, {'enabled': 0},
                                   {'enabled': 'false'}, {'enabled': False, 'runtime_root': 'unused'}])
def test_sndk_switch_requires_an_explicit_typed_boolean(layout, switch):
    root, _, _ = layout
    write_extensions(root, sndk_shadow=switch)
    with pytest.raises(ValueError, match='SNDK shadow service switch'):
        extensions.load(root)


def test_explicit_reenable_restores_sndk_original_ledger_guard(layout):
    root, shadow, _ = layout
    write_extensions(root, sndk_shadow={'enabled': True})
    assert [s.name for s in paper.build_services(root, shadow)] == ['main', 'btc_signals', 'sndk_shadow']
    (shadow/paper.ARMS[0]/'state.json').unlink()
    with pytest.raises(ValueError, match='all four ledgers'):
        paper.build_services(root, shadow)
    write_extensions(root, sndk_shadow={'enabled': False})
    assert [s.name for s in paper.build_services(root, shadow)] == ['main', 'btc_signals']


def test_disabled_sndk_does_not_validate_an_inactive_frozen_source_binding(layout):
    root, shadow, _ = layout
    # Disabled historical bindings are not used for any command. If re-enabled,
    # the original containment/fingerprint validation is restored immediately.
    write_extensions(root, sndk_shadow={'enabled': False},
                     sndk_frozen_runtime={'runtime_root': '../outside'})
    assert [s.name for s in paper.build_services(root, shadow)] == ['main', 'btc_signals']
    write_extensions(root, sndk_shadow={'enabled': True},
                     sndk_frozen_runtime={'runtime_root': '../outside'})
    with pytest.raises(ValueError):
        extensions.load(root)
