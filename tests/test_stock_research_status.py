"""The terminal exposes historical research without accessing an execution account."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stock_research_status as research


@pytest.fixture
def artifacts(tmp_path):
    selection = json.loads((ROOT / "config/stock_research_dashboard.json").read_text())
    result = json.loads((ROOT / selection["results_path"]).read_text())
    for relative in ["config/stock_research_dashboard.json", *selection.values()]:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return tmp_path, selection, result


def save(root, selection, result):
    (root / selection["results_path"]).write_text(json.dumps(result))


def test_results_and_costs_are_read_from_each_independent_ledger(artifacts):
    root, selection, result = artifacts
    status = research.research_status(root)
    assert status["status"] == "available"
    assert status["places_orders"] is False
    assert status["execution_status"] == "research_only"
    assert status["forward_status"] == "prepared_not_started"
    assert status["forward_start_utc"] is None
    assert status["forward_validated"] is False
    assert status["assumptions"]["target_margin_return_pct"] == 120
    for stock in status["stocks"]:
        for cost in (1, 2):
            actual = stock["scenarios"]["liquidity"][str(cost)]["full"]
            source = result["runs"][stock["symbol"]][f"full_prior5m_volume10pct_cost{cost}"]
            assert actual["estimated_close_return_pct"] == source["estimated_close_return_pct"]
            assert actual["closed_return_pct"] == pytest.approx(source["net_closed_pnl"] / source["initial_equity"] * 100)
            assert actual["estimated_open_close_net_pnl"] == source["estimated_open_close_net_pnl"]
            assert stock["scenarios"]["liquidity"][str(cost)]["recent60d"] is None
    assert status["stocks"][1]["scenarios"]["liquidity"]["1"]["full"]["estimated_close_return_pct"] < 0


def test_missing_artifacts_do_not_become_zero_return(tmp_path):
    status = research.research_status(tmp_path)
    assert status["status"] == "unavailable"
    assert status["stocks"] == []
    assert status["places_orders"] is False


def test_partial_data_remains_explicitly_missing(artifacts):
    root, selection, result = artifacts
    del result["runs"]["SNDKUSDT"]["full_prior5m_volume10pct_cost2"]
    del result["base_config"]["taker_fee_rate_assumption"]
    save(root, selection, result)
    status = research.research_status(root)
    assert status["status"] == "available"
    assert status["stocks"][1]["scenarios"]["liquidity"]["2"]["full"] is None
    assert status["assumptions"]["taker_fee_pct"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.5"])
def test_invalid_returns_make_results_unavailable(artifacts, value):
    root, selection, result = artifacts
    result["runs"]["MUUSDT"]["full_prior5m_volume10pct_cost1"]["estimated_close_return_pct"] = value
    save(root, selection, result)
    assert research.research_status(root)["status"] == "unavailable"


def test_only_research_artifacts_are_accepted(artifacts):
    root, selection, result = artifacts
    result["places_orders"] = True
    save(root, selection, result)
    assert research.research_status(root)["status"] == "unavailable"
    selection["results_path"] = "config/active_simulation_candidate.json"
    (root / "config/stock_research_dashboard.json").write_text(json.dumps(selection))
    assert research.research_status(root)["status"] == "unavailable"


def test_missing_forward_record_is_not_reported_as_started(artifacts):
    root, selection, _ = artifacts
    (root / selection["forward_plan_path"]).unlink()
    status = research.research_status(root)
    assert status["forward_status"] == "not_observed"
    assert status["forward_start_utc"] is None


def test_latest_mechanism_failures_do_not_promote_a_positive_historical_return(artifacts):
    root, _, _ = artifacts
    status = research.research_status(root)
    assert status["mechanism_reviewed_at_utc"]
    assert status["report_url"] == "/docs/stock_mechanism_review_20261005.md"
    for stock in status["stocks"]:
        assert stock["mechanism_review"]["trial_count"] == 8
        assert stock["mechanism_review"]["passed_count"] == 0
    assert status["stocks"][2]["scenarios"]["liquidity"]["2"]["full"]["estimated_close_return_pct"] > 0
    assert status["forward_validated"] is False and status["places_orders"] is False


def test_optional_mechanism_file_missing_keeps_original_metrics_available(artifacts):
    root, selection, _ = artifacts
    (root / selection["mechanism_path"]).unlink()
    status = research.research_status(root)
    assert status["status"] == "available"
    assert status["mechanism_reviewed_at_utc"] is None
    assert all("mechanism_review" not in stock for stock in status["stocks"])


def test_nonresearch_mechanism_summary_is_not_published(artifacts):
    root, selection, _ = artifacts
    path = root / selection["mechanism_path"]
    payload = json.loads(path.read_text())
    payload["places_orders"] = True
    path.write_text(json.dumps(payload))
    status = research.research_status(root)
    assert status["status"] == "available"
    assert all("mechanism_review" not in stock for stock in status["stocks"])


def test_research_route_does_not_query_btc_controller(monkeypatch):
    import run_trading_terminal as terminal
    payload = {"status": "available", "stocks": [], "places_orders": False}
    monkeypatch.setattr(terminal.stock_research_status, "research_status", Mock(return_value=payload))
    controller = Mock()
    monkeypatch.setattr(terminal, "CONTROLLER", controller)
    handler = object.__new__(terminal.TerminalHandler)
    handler.path = "/api/terminal/research?v=123"
    handler.send_json = Mock()
    handler.do_GET()
    handler.send_json.assert_called_once_with(payload)
    controller.status.assert_not_called()
    assert not controller.mock_calls
