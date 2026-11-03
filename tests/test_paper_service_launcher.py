"""Run the actual generated wrapper against a harmless native stderr producer."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell 5 wrapper')


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def wrapper(tmp_path, codes, *, working_directory=None):
    installer = (ROOT / 'scripts/install_paper_services_task.ps1').read_text(encoding='utf-8')
    generated = installer.split('$launcher = @"\n', 1)[1].split('\n"@', 1)[0]
    generated = generated.replace('`$', '$')
    counter, sleeps = tmp_path / 'attempts.json', tmp_path / 'sleeps.jsonl'
    worker = tmp_path / 'native_worker.py'
    worker.write_text(
        'import json, pathlib, sys\n'
        f'p = pathlib.Path({str(counter)!r})\n'
        'attempt = json.loads(p.read_text()) if p.exists() else 0\n'
        'p.write_text(json.dumps(attempt + 1))\n'
        'sys.stderr.write("expected native stderr\\n")\n'
        f'sys.exit({codes!r}[attempt])\n', encoding='utf-8')
    for name, value in {
        'rootLiteral': working_directory or tmp_path,
        'pythonLiteral': sys.executable,
        'scriptLiteral': worker,
        'directoryLiteral': tmp_path,
    }.items():
        generated = generated.replace('$' + name, literal(value))
    # Keep the real retry decisions and delay calculation; replace only waiting.
    stub = (
        'function Start-Sleep { param([double]$Seconds)\n'
        'if ($ErrorActionPreference -ne "Stop") { throw "Error preference not restored" }\n'
        f'Add-Content -LiteralPath {literal(sleeps)} -Value $Seconds\n'
        '}\n')
    path = tmp_path / 'wrapper.ps1'
    path.write_text(stub + generated, encoding='utf-8-sig')
    return path, counter, sleeps


def run(path):
    return subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
         '-File', str(path)], capture_output=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW)


@pytest.mark.parametrize('code', [0, 73, 78])
def test_native_stderr_preserves_terminal_exit_code(tmp_path, code):
    path, counter, sleeps = wrapper(tmp_path, [code])
    result = run(path)
    assert result.returncode == 0, result.stderr
    assert json.loads(counter.read_text()) == 1
    assert not sleeps.exists()
    logs = list(tmp_path.glob('supervisor_*.log'))
    assert len(logs) == 1
    assert 'expected native stderr' in logs[0].read_text(encoding='utf-16')


def test_native_stderr_and_failures_reach_existing_retry_backoff(tmp_path):
    path, counter, sleeps = wrapper(tmp_path, [1, 2, 0])
    result = run(path)
    assert result.returncode == 0, result.stderr
    assert json.loads(counter.read_text()) == 3
    assert sleeps.read_text().splitlines() == ['10', '20']
    assert 'expected native stderr' in ''.join(
        path.read_text(encoding='utf-16') for path in tmp_path.glob('supervisor_*.log'))


def test_wrapper_keeps_non_native_setup_errors_terminating(tmp_path):
    path, counter, sleeps = wrapper(tmp_path, [0], working_directory=tmp_path / 'missing')
    result = run(path)
    assert result.returncode != 0
    assert not counter.exists()
    assert not sleeps.exists()
