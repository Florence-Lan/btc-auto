"""Exercise the real PowerShell pipe with non-ASCII process identities."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_paper_services as paper

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell pipe')


@pytest.mark.parametrize('legacy_codepage', [936, 437])
def test_inventory_preserves_unicode_identity_from_legacy_console(monkeypatch, legacy_codepage):
    command = '"C:\\模拟盘\\python.exe" -u "D:\\策略\\run_parallel_simulation.py"'
    real_run = subprocess.run

    def fixture_run(args, **kwargs):
        args = list(args)
        # Inject CIM results, leaving the production selection, JSON generation,
        # pipe decoding and command-line splitting intact. No worker is launched.
        prefix = (
            f'[Console]::OutputEncoding=[System.Text.Encoding]::GetEncoding({legacy_codepage});'
            'function Get-CimInstance { '
            "[pscustomobject]@{Name='python.exe';ProcessId=123;ParentProcessId=1;"
            "CreationDate=[datetime]'2026-11-06T12:00:00Z';CommandLine='"
            + command + "'} };"
        )
        args[-1] = prefix + args[-1]
        return real_run(args, **kwargs)

    monkeypatch.setattr(paper.subprocess, 'run', fixture_run)
    backend = paper.WindowsProcesses.__new__(paper.WindowsProcesses)
    rows = backend.inventory()
    assert set(rows) == {123}
    assert rows[123].command_line == command
    assert rows[123].argv == ('C:\\模拟盘\\python.exe', '-u', 'D:\\策略\\run_parallel_simulation.py')


def test_inventory_failure_remains_an_error_with_unicode_stderr(monkeypatch):
    real_run = subprocess.run

    def failing_run(args, **kwargs):
        args = list(args)
        args[-1] = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();"
                    "[Console]::Error.WriteLine('进程清单不可用');exit 1")
        return real_run(args, **kwargs)

    monkeypatch.setattr(paper.subprocess, 'run', failing_run)
    backend = paper.WindowsProcesses.__new__(paper.WindowsProcesses)
    with pytest.raises(RuntimeError, match='inventory unavailable'):
        backend.inventory()
