import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_trading_terminal as terminal


@pytest.mark.skipif(os.name == "nt", reason="POSIX process reaping")
def test_stop_waits_for_own_child_before_checking_liveness():
    controller = terminal.TerminalController(client=Mock())
    controller.runtime = Mock(return_value={"pid": 12345, "running": True})
    child = Mock(pid=12345)
    controller.process = child
    # The exited child remains visible until wait() reaps it.
    with patch.object(terminal.os, "kill") as kill, \
         patch.object(terminal, "process_alive", return_value=False) as alive, \
         patch.object(terminal, "write_json") as write:
        result = controller.stop()
    kill.assert_called_once_with(12345, signal.SIGTERM)
    child.wait.assert_called_once_with(timeout=5)
    alive.assert_called_once_with(12345)
    assert controller.process is None
    assert result["pid"] is None and not result["running"]
    write.assert_called_once()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process reaping")
def test_failed_stop_does_not_report_success():
    controller = terminal.TerminalController(client=Mock())
    controller.runtime = Mock(return_value={"pid": 12345, "running": True})
    controller.process = Mock(pid=12345)
    controller.process.wait.side_effect = subprocess.TimeoutExpired("supervisor", 5)
    with patch.object(terminal.os, "kill"), patch.object(terminal, "write_json") as write:
        with pytest.raises(RuntimeError, match="仍存活"):
            controller.stop()
    write.assert_not_called()
    assert controller.process is not None
