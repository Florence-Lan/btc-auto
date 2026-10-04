"""A data-path failure cannot erase a verified decision or invent a fill."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decision_runtime as decision
import run_execution_supervisor as supervisor
import trading_execution as execution

NOW = 1_791_060_305_000
BAR = NOW // 300_000 * 300_000 - 300_000


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(decision.time, "time", lambda: NOW / 1000)
    monkeypatch.setenv("LLM_TRADE_GATE_ENABLED", "false")
    path = tmp_path / "judgment.json"
    writer = decision.write_judgment
    monkeypatch.setattr(decision, "STATUS_PATH", path)
    monkeypatch.setattr(decision, "write_judgment", lambda payload: writer(payload, path=path))
    account_path = tmp_path / "account.json"
    monkeypatch.setattr(supervisor, "SIMULATION_STATE_PATH", account_path)
    account = execution.SimulationAccount(account_path, allow_llm=False)
    account.reset(1000, now_ms=NOW-1_000_000)
    report_path = tmp_path / "report.json"
    report = {"execution_target": {"time_ms": BAR, "equity": 100, "price": 100000,
              "signed_qty": .0002, "position_id": "new-signal", "origin_signal_time_ms": BAR},
              "decision_asof_ms": NOW, "generated_at_utc": execution.utc_now(),
              "paper_inception_utc": "current-generation",
              "market_data": {"complete": True}}
    report_path.write_text(json.dumps(report))
    args = SimpleNamespace(mode="simulation", factor_profile=None, report_path=report_path,
                           state_path=tmp_path / "state.json", macro_snapshot=tmp_path / "macro.gz",
                           refresh_hours=12, poll_seconds=30, trades_path=tmp_path / "trades.csv",
                           bar_settle_delay_seconds=3)
    args.state_path.write_text(json.dumps({"created_at_utc": "current-generation"}))
    client = Mock()
    client.simulation_clock = {"time_ms": NOW, "source": "exchange"}
    client.server_time_ms.return_value = NOW
    client.mark_price_observation.return_value = {"price":100000, "time_ms":NOW-1000,
                                                 "next_funding_time_ms":NOW+300000}
    client.symbol_rules.return_value = execution.DEFAULT_SIMULATION_RULES
    client.funding_history.return_value = []
    return args, client, account, path


def test_time_endpoint_failure_can_still_run_due_judgment(setup, monkeypatch):
    args, client, _, _ = setup
    client.server_time_ms.side_effect = OSError("time path unavailable")
    observer = Mock()
    cycle = Mock(return_value={"signal_time_ms":BAR})
    monkeypatch.setattr(supervisor, "monitor_simulation_account", observer)
    monkeypatch.setattr(supervisor, "run_cycle", cycle)
    monkeypatch.setattr(supervisor, "last_processed_signal_ms", lambda mode: BAR-300_000)
    assert supervisor.check_once(args, client) == NOW-1000
    assert client.simulation_clock["source"] == "mark_endpoint"
    assert observer.call_args.kwargs["clock_source"] == "mark_endpoint"
    cycle.assert_called_once()


def test_fallback_clock_does_not_skip_bar_settlement_delay(setup, monkeypatch):
    args, client, _, path = setup
    client.server_time_ms.side_effect = OSError("time path unavailable")
    boundary = NOW // 300_000 * 300_000
    client.mark_price_observation.return_value = {"price":100000,"time_ms":boundary+1000}
    monkeypatch.setattr(decision.time, "time", lambda: (boundary+1000)/1000)
    monkeypatch.setattr(supervisor, "monitor_simulation_account", Mock())
    cycle = Mock()
    monkeypatch.setattr(supervisor, "run_cycle", cycle)
    assert supervisor.check_once(args, client) == boundary+1000
    cycle.assert_not_called()
    assert json.loads(path.read_text())["decision_status"] == "waiting_for_bar"


def test_execution_outage_preserves_judgment_and_account_cursor(setup, monkeypatch):
    args, client, account, path = setup
    before = account.path.read_bytes()
    monkeypatch.setattr(supervisor, "execute_report", Mock(side_effect=OSError("price disconnected")))
    refresh = Mock(side_effect=AssertionError("Current report should be reused"))
    monkeypatch.setattr(supervisor.strategy_supervisor, "run_shadow_once", refresh)
    result = supervisor.run_cycle(args, client, asof_ms=NOW, required_bar=BAR)
    status = json.loads(path.read_text())
    assert status["decision_status"] == "evaluated"
    assert status["execution_status"] == "deferred"
    assert status["target_leverage"] == .2
    assert result["signal_time_ms"] == BAR
    assert account.path.read_bytes() == before


def test_retry_reuses_report_and_then_executes_with_current_checks(setup, monkeypatch):
    args, client, _, path = setup
    execute = Mock(side_effect=[OSError("temporary outage"),{"target_leverage":.2}])
    monkeypatch.setattr(supervisor, "execute_report", execute)
    refresh = Mock(side_effect=AssertionError("Do not recompute the same bar"))
    monkeypatch.setattr(supervisor.strategy_supervisor, "run_shadow_once", refresh)
    supervisor.run_cycle(args, client, asof_ms=NOW, required_bar=BAR)
    supervisor.run_cycle(args, client, asof_ms=NOW+1000, required_bar=BAR)
    assert execute.call_count == 2
    assert json.loads(path.read_text())["execution_status"] == "completed"


@pytest.mark.parametrize("change", ["generation", "bar", "unfinished_cutoff"])
def test_retry_cannot_reuse_another_generation_or_bar(setup, monkeypatch, change):
    args, client, _, _ = setup
    report = json.loads(args.report_path.read_text())
    if change == "generation":
        report["paper_inception_utc"] = "previous-generation"
    elif change == "bar":
        report["execution_target"]["time_ms"] = BAR + 300000
    else:
        report["decision_asof_ms"] = BAR + 299998
    args.report_path.write_text(json.dumps(report))
    refresh = Mock(side_effect=RuntimeError("Regeneration required"))
    monkeypatch.setattr(supervisor.strategy_supervisor, "run_shadow_once", refresh)
    monkeypatch.setattr(supervisor.strategy_supervisor, "refresh_macro_if_needed", Mock())
    with pytest.raises(RuntimeError, match="Regeneration required"):
        supervisor.run_cycle(args, client, asof_ms=NOW, required_bar=BAR)
    refresh.assert_called_once()


def test_future_report_cannot_execute(setup, monkeypatch):
    args, client, _, _ = setup
    report = json.loads(args.report_path.read_text())
    report["execution_target"]["available_time_ms"] = NOW+1
    args.report_path.write_text(json.dumps(report))
    execute = Mock()
    monkeypatch.setattr(supervisor, "execute_report", execute)
    with pytest.raises(RuntimeError, match="ahead"):
        supervisor.run_cycle(args, client, asof_ms=NOW, required_bar=BAR)
    execute.assert_not_called()


def test_data_unavailable_is_not_a_flat_profit_judgment(setup):
    args, _, _, path = setup
    supervisor.record_unavailable(args, OSError("closed bar missing"))
    status = json.loads(path.read_text())
    assert status["decision_status"] == "unavailable"
    assert "target_leverage" not in status
    assert status["execution_status"] == "deferred"


def test_simulation_execute_uses_fresh_mark_when_time_endpoint_fails(setup, monkeypatch):
    args, client, account, _ = setup
    account.reconcile({"target_leverage":0.0,"position_id":"flat","signal_time_ms":BAR-300000},
                      100000, now_ms=NOW-5000)
    client.server_time_ms.side_effect = OSError("time path unavailable")
    monkeypatch.setattr(execution, "SimulationAccount", lambda:account)
    result = execution.execute_report("simulation", json.loads(args.report_path.read_text()), client)
    assert result["fill"]["side"] == "BUY"
    assert account.load()["position_qty"] > 0
    client.market_order.assert_not_called()


@pytest.mark.parametrize("target_leverage", [0.0,.4])
def test_price_expired_during_llm_cannot_fill_even_a_reduction(setup, monkeypatch, target_leverage):
    _, _, account, _ = setup
    account.reconcile({"target_leverage":.2,"signal_time_ms":BAR},100000,now_ms=NOW-1000)
    before = account.path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        account.reconcile({"target_leverage":target_leverage,"signal_time_ms":BAR+1},100000,
            now_ms=NOW,mark_time_ms=NOW,execution_clock=lambda:NOW+60001)
    assert account.path.read_bytes() == before


def test_unfresh_execution_mark_never_changes_account(setup, monkeypatch):
    args, client, account, _ = setup
    client.mark_price_observation.return_value = {"price":100000,"time_ms":NOW-60001}
    monkeypatch.setattr(execution, "SimulationAccount", lambda:account)
    before = account.path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        execution.execute_report("simulation",json.loads(args.report_path.read_text()),client)
    assert account.path.read_bytes() == before
