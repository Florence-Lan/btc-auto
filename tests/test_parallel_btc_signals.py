import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_btc_signals as producer
import run_parallel_simulation as parallel


def test_signal_paths_are_isolated_from_old_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, 'ROOT', tmp_path)
    args = producer.signal_args({'accounts': [{'account_id': 'btc',
        'strategy_path': 'config/latest.json',
        'strategy_report_path': 'data/parallel_simulation/new/btc_signals/report.json'}]})
    assert args.state_path == tmp_path / 'data/parallel_simulation/new/btc_signals/state.json'
    assert args.report_path.parent == args.state_path.parent
    assert args.trades_path.parent == args.state_path.parent


@pytest.mark.parametrize('age,returncode,expected', [(300_000, 0, None),
    (900_001, 0, 'older than fifteen'), (300_000, 1, 'exited 1')])
def test_producer_rejects_stale_or_failed_evaluations(tmp_path, monkeypatch, age, returncode, expected):
    now = 2_000_000
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'execution_target': {'time_ms': now-age}}))
    args = SimpleNamespace(report_path=path)
    monkeypatch.setattr(producer.decision_runtime, 'resolve_clock', lambda _: {'time_ms': now})
    monkeypatch.setattr(producer.information_runtime, 'paper_command', lambda _: ['paper-engine'])
    process = Mock(return_value=SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(producer.subprocess, 'run', process)
    if expected:
        with pytest.raises((RuntimeError, ValueError), match=expected):
            producer.evaluate(args, Mock())
    else:
        assert producer.evaluate(args, Mock()) == now-age
        assert args.signal_asof_ms == now
    process.assert_called_once_with(['paper-engine'], cwd=parallel.ROOT, timeout=240)
