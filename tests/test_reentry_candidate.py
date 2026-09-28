import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from test_strategy_engine import ROOT
import reentry_candidate
import run_multifactor_shadow as runner
import trading_execution


def test_candidate_rejects_live_permission_and_code_drift(tmp_path):
    original = json.loads((ROOT / "config/reentry_selected_20260928.json").read_text())
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({**original, "live_orders_allowed": True}))
    with pytest.raises(ValueError, match="research-only"):
        reentry_candidate.load_candidate(path, ROOT / original["base_manifest"])
    original["input_hashes"]["scripts/research_reentry.py"] = "wrong"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="changed"):
        reentry_candidate.load_candidate(path, ROOT / original["base_manifest"])
    original = json.loads((ROOT / "config/reentry_selected_20260928.json").read_text())
    original["policy"]["cooldown_bars"] = 1
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="parameters"):
        reentry_candidate.load_candidate(path, ROOT / original["base_manifest"])


def test_reentry_research_report_never_reaches_live_client():
    client = Mock()
    with pytest.raises(ValueError, match="simulation-only"):
        trading_execution.execute_report("live", {"research_only": True, "research_candidate": "btc_reentry_selected_20260928"}, client)
    assert client.mock_calls == []


def test_failed_collection_skips_paper_run_and_persists_health(tmp_path):
    profile = {"candidate_id": "test_candidate", "base_manifest": "unused", "event_snapshot": "unused"}
    with patch.object(runner.sim, "repo_root", return_value=tmp_path), \
         patch.object(runner.multifactor, "load_profile", return_value=profile), \
         patch.object(runner.sys, "argv", ["runner", "--once"]), \
         patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
        assert runner.main() == 1
    assert run.call_count == 1
    status = json.loads((tmp_path / "data/paper_trading/test_candidate_status.json").read_text())
    assert status["status"] == "degraded"
    assert status["stage"] == "factor_collection"
    assert status["places_orders"] is False
