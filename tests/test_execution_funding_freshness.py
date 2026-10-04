"""Cash coverage and a final price check must survive execution-side delays."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decision_runtime as decision
import trading_execution as execution


NOW = 1_791_060_305_000


def target(leverage=.2):
    return {"target_leverage": leverage, "signal_time_ms": NOW - 300000,
            "signal_price": 100000, "position_id": "fresh-position"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    timer = {"monotonic": 1000.0}
    monkeypatch.setattr(execution.time, "monotonic", lambda: timer["monotonic"])
    monkeypatch.setenv("LLM_TRADE_GATE_ENABLED", "false")
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "2")
    monkeypatch.setenv("SIM_MAX_NOTIONAL_USDT", "0")
    monkeypatch.setenv("SIM_TAKER_FEE", "0.00045")
    monkeypatch.setenv("SIM_SLIPPAGE_BPS", "1")
    account = execution.SimulationAccount(tmp_path / "account.json", allow_llm=False)
    account.reset(1000, now_ms=NOW - 1000000)
    account.reconcile(target(0), 100000, now_ms=NOW - 1000)
    return account, timer


def test_known_settlement_free_gap_preserves_precise_fetch_watermark(setup):
    account, _ = setup
    result = account.reconcile(target(), 100000, now_ms=NOW + 500,
        funding_available=True, funding_fetch_through_ms=NOW,
        next_funding_time_ms=NOW + 1000)
    assert result["fill"]["side"] == "BUY"
    assert account.load()["funding_last_fetch_ms"] == NOW
    assert account.load()["funding_status"] == "ok"


@pytest.mark.parametrize("offset", [0, 1])
def test_uncovered_funding_boundary_blocks_additions(setup, offset):
    account, _ = setup
    result = account.reconcile(target(), 100000, now_ms=NOW + 1000 + offset,
        funding_available=True, funding_fetch_through_ms=NOW,
        next_funding_time_ms=NOW + 1000)
    assert result["fill"] is None
    assert account.load()["position_qty"] == 0
    assert account.load()["funding_last_fetch_ms"] == NOW
    assert account.load()["funding_status"] == "unavailable_new_risk_blocked"


def test_uncovered_funding_boundary_does_not_prevent_fresh_price_exit(setup):
    account, _ = setup
    account.reconcile(target(), 100000, now_ms=NOW)
    result = account.reconcile(target(0), 100000, now_ms=NOW + 1001,
        mark_time_ms=NOW + 1001, funding_available=True,
        funding_fetch_through_ms=NOW, next_funding_time_ms=NOW + 1000)
    assert result["fill"]["side"] == "SELL"
    assert account.load()["position_qty"] == 0
    assert account.load()["funding_last_fetch_ms"] == NOW


@pytest.mark.parametrize("offset,allowed", [(0, True), (1, False)])
def test_unknown_next_settlement_only_covers_the_successful_cutoff(setup, offset, allowed):
    account, _ = setup
    result = account.reconcile(target(), 100000, now_ms=NOW + offset,
        funding_available=True, funding_fetch_through_ms=NOW)
    assert bool(result["fill"]) is allowed


def test_llm_delay_crossing_a_funding_boundary_cannot_add_exposure(setup):
    account, _ = setup
    result = account.reconcile(target(), 100000, now_ms=NOW,
        funding_available=True, funding_fetch_through_ms=NOW,
        next_funding_time_ms=NOW + 1000, mark_time_ms=NOW,
        execution_clock=lambda: NOW + 1000)
    assert result["fill"] is None
    assert account.load()["funding_status"] == "unavailable_new_risk_blocked"


def test_final_clock_cannot_move_backwards_from_the_initial_verified_clock(setup):
    account, _ = setup
    account.reconcile(target(), 100000, now_ms=NOW)
    result = account.reconcile(target(0), 100000, now_ms=NOW + 1000,
                               execution_clock=lambda: NOW - 1)
    assert result["fill"]["side"] == "SELL"
    assert account.load()["position_history"][-1]["time_ms"] == NOW + 1000


def test_slow_current_gate_cannot_create_a_fill_at_an_expired_price(setup, monkeypatch):
    account, timer = setup
    before = account.path.read_bytes()
    def delayed_gate(report, timestamp, side):
        timer["monotonic"] += 60.01
        return {"allowed": True, "status": "allowed", "reasons": []}
    monkeypatch.setattr(execution.execution_entry_gate, "decision_at", delayed_gate)
    with pytest.raises(ValueError, match="stale"):
        account.reconcile(target(), 100000, now_ms=NOW, mark_time_ms=NOW)
    assert account.path.read_bytes() == before


def test_slow_current_gate_cannot_cross_an_uncovered_funding_boundary(setup, monkeypatch):
    account, timer = setup
    def delayed_gate(report, timestamp, side):
        timer["monotonic"] += 1.0
        return {"allowed": True, "status": "allowed", "reasons": []}
    monkeypatch.setattr(execution.execution_entry_gate, "decision_at", delayed_gate)
    result = account.reconcile(target(), 100000, now_ms=NOW, mark_time_ms=NOW,
        funding_available=True, funding_fetch_through_ms=NOW, next_funding_time_ms=NOW + 1000)
    assert result["fill"] is None
    assert account.load()["funding_status"] == "unavailable_new_risk_blocked"


@pytest.mark.parametrize("next_time", [None, 0, -1, NOW - 1, NOW, True, "bad"])
def test_invalid_next_funding_metadata_is_unknown(next_time):
    assert execution._next_funding_boundary({"time_ms": NOW,
        "next_funding_time_ms": next_time}) is None


def execute_setup(account, monkeypatch, timestamps, observations):
    source = SimpleNamespace(server_time_ms=Mock(),
        symbol_rules=Mock(return_value=execution.DEFAULT_SIMULATION_RULES),
        mark_price_observation=Mock(side_effect=observations),
        funding_history=Mock(return_value=[]), market_order=Mock())
    times = iter(timestamps)
    def clock(client, **kwargs):
        return {"time_ms": next(times), "source": "exchange", "degraded": False}
    monkeypatch.setattr(decision, "resolve_clock", clock)
    monkeypatch.setattr(execution, "SimulationAccount", lambda: account)
    report = {"execution_target": {"time_ms": NOW - 300000, "equity": 100,
               "price": 100000, "signed_qty": .0002, "position_id": "fresh-position"}}
    return source, report


def mark(timestamp, next_time=None):
    result = {"price": 100000, "time_ms": timestamp}
    if next_time is not None:
        result["next_funding_time_ms"] = next_time
    return result


def funding(timestamp):
    return {"fundingTime": timestamp, "fundingRate": "0.0001", "markPrice": "100000"}


def test_production_flow_does_not_refetch_during_known_settlement_free_gap(setup, monkeypatch):
    account, _ = setup
    source, report = execute_setup(account, monkeypatch,
        [NOW, NOW, NOW + 20, NOW + 25],
        [mark(NOW, NOW + 1000), mark(NOW + 20, NOW + 1000)])
    result = execution.execute_report("simulation", report, source)
    assert result["fill"]["side"] == "BUY"
    assert source.funding_history.call_count == 1
    assert source.funding_history.call_args.args[1] == NOW
    assert account.load()["funding_last_fetch_ms"] == NOW
    source.market_order.assert_not_called()


def test_production_flow_refetches_after_crossing_real_settlement_boundary(setup, monkeypatch):
    account, _ = setup
    source, report = execute_setup(account, monkeypatch,
        [NOW, NOW, NOW + 20, NOW + 25, NOW + 25],
        [mark(NOW, NOW + 10), mark(NOW + 20, NOW + 1010), mark(NOW + 25, NOW + 1010)])
    source.funding_history.side_effect = [[], [funding(NOW + 11)]]
    result = execution.execute_report("simulation", report, source)
    assert result["fill"]["side"] == "BUY"
    assert [call.args[1] for call in source.funding_history.call_args_list] == [NOW, NOW + 20]
    assert account.load()["funding_last_fetch_ms"] == NOW + 20


def test_failed_boundary_refresh_keeps_previous_successful_cutoff_and_blocks_additions(setup, monkeypatch):
    account, _ = setup
    source, report = execute_setup(account, monkeypatch,
        [NOW, NOW, NOW + 20, NOW + 25, NOW + 25],
        [mark(NOW, NOW + 10), mark(NOW + 20, NOW + 1010), mark(NOW + 25, NOW + 1010)])
    source.funding_history.side_effect = [[], OSError("funding unavailable")]
    result = execution.execute_report("simulation", report, source)
    assert result["fill"] is None
    assert account.load()["funding_last_fetch_ms"] == NOW
    assert account.load()["funding_status"] == "unavailable_new_risk_blocked"


def test_simulation_target_age_uses_verified_clock_instead_of_local_wall_time(setup, monkeypatch):
    account, _ = setup
    monkeypatch.setattr(decision.time, "time", lambda: (NOW + 3 * 3600000) / 1000)
    source, report = execute_setup(account, monkeypatch,
        [NOW, NOW, NOW, NOW], [mark(NOW), mark(NOW)])
    assert execution.execute_report("simulation", report, source)["fill"]["side"] == "BUY"


def test_old_mark_clock_cannot_freeze_elapsed_llm_time_across_a_settlement(setup, monkeypatch):
    account, timer = setup
    account.allow_llm = True
    def slow_llm(report, requested, **kwargs):
        timer["monotonic"] += 20
        return requested, {"status": "allowed"}
    monkeypatch.setattr(execution, "apply_llm_trade_gate", slow_llm)
    result = account.reconcile(target(), 100000, now_ms=NOW, mark_time_ms=NOW,
        funding_fetch_through_ms=NOW, next_funding_time_ms=NOW + 10000,
        execution_clock=lambda: NOW)
    assert result["fill"] is None
    state = account.load()
    assert state["equity_curve"][-1]["time_ms"] == NOW + 20000
    assert state["funding_pending_settlement_ms"] == NOW + 10000


def test_old_mark_clock_cannot_hide_price_expiry_during_llm(setup, monkeypatch):
    account, timer = setup
    account.allow_llm = True
    before = account.path.read_bytes()
    def slow_llm(report, requested, **kwargs):
        timer["monotonic"] += 61
        return requested, {"status": "allowed"}
    monkeypatch.setattr(execution, "apply_llm_trade_gate", slow_llm)
    with pytest.raises(ValueError, match="stale"):
        account.reconcile(target(), 100000, now_ms=NOW, mark_time_ms=NOW,
                          execution_clock=lambda: NOW)
    assert account.path.read_bytes() == before


def test_late_publication_blocks_new_risk_until_the_actual_settlement_arrives(setup, monkeypatch):
    account, _ = setup
    source, report = execute_setup(account, monkeypatch,
        [NOW, NOW, NOW + 20, NOW + 25, NOW + 25],
        [mark(NOW, NOW + 10), mark(NOW + 20, NOW + 1010), mark(NOW + 25, NOW + 1010)])
    first = execution.execute_report("simulation", report, source)
    assert first["fill"] is None
    assert account.load()["funding_pending_settlement_ms"] == NOW + 10
    assert account.load()["funding_last_fetch_ms"] == NOW + 20
    # The mark endpoint has advanced to the following funding period. The
    # outstanding settlement from the preceding period must still be required.
    source, report = execute_setup(account, monkeypatch,
        [NOW + 30] * 4, [mark(NOW + 30, NOW + 1010)] * 2)
    source.funding_history.return_value = []
    second = execution.execute_report("simulation", report, source)
    assert second["fill"] is None
    assert account.load()["funding_pending_settlement_ms"] == NOW + 10
    source, report = execute_setup(account, monkeypatch,
        [NOW + 40] * 4, [mark(NOW + 40, NOW + 1010)] * 2)
    source.funding_history.return_value = [funding(NOW + 11)]
    third = execution.execute_report("simulation", report, source)
    assert third["fill"]["side"] == "BUY"
    assert account.load()["funding_pending_settlement_ms"] is None
    assert len(account.load()["funding_settlements"]) == 1


def test_late_publication_does_not_prevent_an_existing_position_exit(setup, monkeypatch):
    account, _ = setup
    account.reconcile(target(), 100000, now_ms=NOW)
    source, report = execute_setup(account, monkeypatch,
        [NOW, NOW, NOW + 20, NOW + 25, NOW + 25],
        [mark(NOW, NOW + 10), mark(NOW + 20, NOW + 1010), mark(NOW + 25, NOW + 1010)])
    report["execution_target"]["signed_qty"] = 0
    result = execution.execute_report("simulation", report, source)
    assert result["fill"]["side"] == "SELL"
    assert account.load()["position_qty"] == 0
    assert account.load()["funding_pending_settlement_ms"] == NOW + 10


@pytest.mark.parametrize("field", ["time_ms", "available_time_ms"])
def test_a_future_strategy_target_is_rejected_even_if_it_is_fresh(field):
    point = {"time_ms": NOW - 1, "equity": 100, "price": 100000,
             "signed_qty": .0002, "available_time_ms": NOW}
    point[field] = NOW + 1
    with pytest.raises(RuntimeError, match="future"):
        execution.target_from_report({"execution_target": point}, now_ms=NOW)


def test_pending_settlement_query_overlap_survives_an_extended_outage(setup):
    account, _ = setup
    state = account.load()
    state["funding_pending_settlement_ms"] = NOW
    state["funding_last_fetch_ms"] = NOW + 3 * 86400000
    account._save(state)
    source = SimpleNamespace(funding_history=Mock(return_value=[]))
    execution.simulation_funding(source, account, NOW + 3 * 86400000)
    assert source.funding_history.call_args.args[0] == NOW
