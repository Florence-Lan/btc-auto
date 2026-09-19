import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_execution_supervisor as supervisor
import trading_execution


def test_candidate_live_cycle_stops_before_refresh_or_execution():
    with patch.object(supervisor.strategy_supervisor, "refresh_macro_if_needed") as refresh:
        with pytest.raises(ValueError, match="simulation-only"):
            supervisor.run_cycle(SimpleNamespace(mode="live"), Mock())
        refresh.assert_not_called()


def test_candidate_report_cannot_reach_live_client():
    client = Mock()
    with pytest.raises(ValueError, match="simulation-only"):
        trading_execution.execute_report(
            "live", {"freeze_id": "btc_trend_filter_research_20260917"}, client
        )
    assert client.mock_calls == []
