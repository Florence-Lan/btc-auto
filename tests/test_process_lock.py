import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from process_lock import exclusive_process_lock


def test_duplicate_writer_is_rejected_and_clean_exit_releases_lock(tmp_path):
    path = tmp_path / "runner.lock"
    with exclusive_process_lock(path):
        with pytest.raises(OSError):
            with exclusive_process_lock(path):
                pytest.fail("Duplicate runner acquired the lock")
    with exclusive_process_lock(path):
        pass


def test_error_does_not_leave_runner_locked(tmp_path):
    path = tmp_path / "runner.lock"
    with pytest.raises(RuntimeError):
        with exclusive_process_lock(path):
            raise RuntimeError("runner failure")
    with exclusive_process_lock(path):
        pass


def test_independent_process_is_blocked_and_crash_releases_lock(tmp_path):
    path = tmp_path / "runner.lock"
    script = (
        "from process_lock import exclusive_process_lock; import os, sys; "
        "lock = exclusive_process_lock(sys.argv[1]); lock.__enter__(); "
        "print('locked', flush=True); os._exit(0)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "scripts")}
    with exclusive_process_lock(path):
        blocked = subprocess.run([sys.executable, "-c", script, str(path)],
                                 env=env, capture_output=True, text=True, timeout=10)
        assert blocked.returncode != 0 and "locked" not in blocked.stdout
    crashed = subprocess.run([sys.executable, "-c", script, str(path)],
                             env=env, capture_output=True, text=True, timeout=10)
    assert crashed.returncode == 0 and crashed.stdout.strip() == "locked"
    with exclusive_process_lock(path):
        pass
