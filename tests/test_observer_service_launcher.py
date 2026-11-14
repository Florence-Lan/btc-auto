"""Exercise the generated observer wrapper with real native stderr and codes."""
import json
import os

import pytest

from test_paper_service_launcher import wrapper, run

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell 5 wrapper')


@pytest.mark.parametrize('code', [0, 73, 78])
def test_observer_native_stderr_keeps_normal_terminal_exit_codes(tmp_path, code):
    path, counter, sleeps = wrapper(tmp_path, [code],
        installer_name='install_cross_market_observer_task.ps1')
    result = run(path)
    assert result.returncode == 0, result.stderr
    assert json.loads(counter.read_text()) == 1
    assert not sleeps.exists()
    logs = list(tmp_path.glob('observer_*.log'))
    assert len(logs) == 1 and 'expected native stderr' in logs[0].read_text(encoding='utf-16')


def test_observer_native_failures_reach_retry_and_restore_error_preference(tmp_path):
    path, counter, sleeps = wrapper(tmp_path, [1, 2, 0],
        installer_name='install_cross_market_observer_task.ps1')
    result = run(path)
    assert result.returncode == 0, result.stderr
    assert json.loads(counter.read_text()) == 3
    assert sleeps.read_text().splitlines() == ['10', '20']


def test_observer_setup_error_does_not_enter_native_retry_loop(tmp_path):
    path, counter, sleeps = wrapper(tmp_path, [0], working_directory=tmp_path/'missing',
        installer_name='install_cross_market_observer_task.ps1')
    result = run(path)
    assert result.returncode != 0
    assert not counter.exists() and not sleeps.exists()
