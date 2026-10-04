"""The authorized simulation ceiling reaches execution without widening live limits."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import active_strategy
import decision_runtime
import public_context
import run_execution_supervisor as supervisor
import simulate_range_swing as sim
import trading_execution as execution
from binance_terminal_client import BinanceTerminalClient


NOW = 1_791_060_305_000
PRICE = 100_000


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.delenv("SIM_MAX_LEVERAGE", raising=False)
    for name, value in {"SIM_MAX_NOTIONAL_USDT": "0", "SIM_TAKER_FEE": "0",
                        "SIM_SLIPPAGE_BPS": "0", "LLM_TRADE_GATE_ENABLED": "false"}.items():
        monkeypatch.setenv(name, value)


def account():
    result = execution.SimulationAccount(Path("unused_leverage_test.json"), persist=False, allow_llm=False)
    state = result.reset(1000, now_ms=NOW - 1000)
    state["observed_flat_target"] = True
    result._save(state)
    return result


def report(leverage=8, cap=10):
    return {"config": {"portfolio_leverage_cap": cap},
            "execution_target": {"time_ms": NOW, "equity": 100, "price": PRICE,
                                 "signed_qty": leverage * 100 / PRICE, "position_id": "signal"}}


def target(leverage, timestamp=NOW):
    return {"target_leverage": leverage, "signal_time_ms": timestamp, "position_id": "signal"}


@pytest.mark.parametrize("requested, expected", [(8, 8), (-8, -8), (12, 10), (-12, -10)])
def test_simulation_report_ceiling_and_account_quantity(requested, expected):
    payload = report(requested)
    parsed = execution.target_from_report(payload, now_ms=NOW,
        leverage_cap=active_strategy.simulation_leverage_cap(payload))
    assert parsed["target_leverage"] == expected
    result = account().reconcile(target(requested), PRICE, report=payload, now_ms=NOW)
    assert result["target_qty"] == pytest.approx(expected * .01)
    assert result["target_leverage"] == expected
    assert result["desired_target_leverage"] == requested
    assert result["simulation_leverage_cap"] == 10
    assert result["account"]["simulation_leverage_cap"] == 10


@pytest.mark.parametrize("sign", [1, -1])
def test_legacy_report_and_default_parser_keep_two_times(sign):
    payload = report(sign * 8)
    payload.pop("config")
    assert execution.target_from_report(payload, now_ms=NOW)["target_leverage"] == sign * 2
    result = account().reconcile(target(sign * 8), PRICE, report=payload, now_ms=NOW)
    assert result["target_qty"] == sign * .02
    assert result["simulation_leverage_cap"] == 2


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("environment_cap, notional_cap, expected", [
    ("3", "0", 3), ("10", "1500", 1.5), ("0", "0", 0)])
def test_lower_environment_and_notional_ceilings_remain_effective(
        sign, environment_cap, notional_cap, expected, monkeypatch):
    monkeypatch.setenv("SIM_MAX_LEVERAGE", environment_cap)
    monkeypatch.setenv("SIM_MAX_NOTIONAL_USDT", notional_cap)
    result = account().reconcile(target(sign * 8), PRICE, report=report(sign * 8), now_ms=NOW)
    assert result["target_qty"] == pytest.approx(sign * expected * .01)
    assert result["target_leverage"] == pytest.approx(sign * expected)


def test_snapshot_reserves_margin_at_saved_ceiling_and_handles_zero(monkeypatch):
    a = account()
    a.reconcile(target(8), PRICE, report=report(), now_ms=NOW)
    assert a.snapshot(PRICE)["account"]["available_balance"] == pytest.approx(200)
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "0")
    assert a.snapshot(PRICE)["account"]["available_balance"] == 0
    a.reconcile(target(0, NOW + 1), PRICE, report=report(0), now_ms=NOW + 1)
    assert a.load()["simulation_leverage_cap"] == 0
    assert a.snapshot(PRICE)["account"]["available_balance"] == pytest.approx(1000)
    # The persisted zero bound must remain safe even after the environment changes.
    monkeypatch.delenv("SIM_MAX_LEVERAGE")
    assert a.snapshot(PRICE)["account"]["available_balance"] == pytest.approx(1000)


def blocked_report(tmp_path, reason):
    payload = report()
    path = tmp_path / "public.json"
    events = []
    if reason == "event":
        events = [{"event_id": "release", "published_at_utc": sim.iso_utc_from_ms(NOW - 1000),
                   "starts_at_utc": sim.iso_utc_from_ms(NOW),
                   "ends_at_utc": sim.iso_utc_from_ms(NOW + 5000),
                   "severity": 1, "block_entries": True}]
    path.write_text(json.dumps({"schema_version": 1, "events": events,
        "coverage_checks": [{"available_at_utc": sim.iso_utc_from_ms(NOW - 1000),
            "sources": {name: {"ok": reason != "source"} for name in public_context.REQUIRED}}]}))
    payload["execution_entry_context"] = {"event_snapshot": str(path)}
    return payload


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("reason", ["event", "source"])
def test_higher_ceiling_does_not_bypass_current_entry_gates(tmp_path, sign, reason):
    payload = blocked_report(tmp_path, reason)
    a = account()
    result = a.reconcile(target(sign * 8), PRICE, report=payload, now_ms=NOW)
    assert result["fill"] is None
    assert result["target_qty"] == 0
    assert result["execution_entry_gate"]["allowed"] is False
    a.reconcile(target(sign * 4, NOW + 1), PRICE, report=report(), now_ms=NOW + 1)
    held = a.load()["position_qty"]
    assert a.reconcile(target(sign * 8, NOW + 2), PRICE,
                       report=payload, now_ms=NOW + 2)["fill"] is None
    assert a.load()["position_qty"] == held
    reduced = a.reconcile(target(sign * 2, NOW + 3), PRICE, report=payload, now_ms=NOW + 3)
    assert reduced["target_qty"] == sign * .02
    reversed_target = a.reconcile(target(-sign * 8, NOW + 4), PRICE,
                                  report=payload, now_ms=NOW + 4)
    assert reversed_target["fill"] is not None
    assert a.load()["position_qty"] == 0


@pytest.mark.parametrize("sign", [1, -1])
def test_missing_funding_blocks_additions_but_allows_reduction_and_exit(sign):
    a = account()
    assert a.reconcile(target(sign * 8), PRICE, report=report(), now_ms=NOW,
                       funding_available=False)["fill"] is None
    a.reconcile(target(sign * 4, NOW + 1), PRICE, report=report(), now_ms=NOW + 1)
    assert a.reconcile(target(sign * 8, NOW + 2), PRICE, report=report(), now_ms=NOW + 2,
                       funding_available=False)["fill"] is None
    assert a.reconcile(target(sign * 2, NOW + 3), PRICE, report=report(), now_ms=NOW + 3,
                       funding_available=False)["target_qty"] == sign * .02
    a.reconcile(target(0, NOW + 4), PRICE, report=report(), now_ms=NOW + 4, funding_available=False)
    assert a.load()["position_qty"] == 0


@pytest.mark.parametrize("sign", [1, -1])
def test_eight_fifteen_percent_drawdown_controls_are_unchanged(sign):
    a = account()
    state = a.load()
    state["wallet_balance"] = 900
    a._save(state)
    reduced = a.reconcile(target(sign * 8), PRICE, report=report(), now_ms=NOW)
    assert reduced["account_risk"]["soft_limit_pct"] == 8
    assert reduced["account_risk"]["hard_limit_pct"] == 15
    assert reduced["account_risk"]["status"] == "throttled"
    assert 0 < abs(reduced["target_qty"]) < .072
    state = a.load()
    state["wallet_balance"] = 850
    a._save(state)
    stopped = a.reconcile(target(sign * 8, NOW + 1), PRICE, report=report(), now_ms=NOW + 1)
    assert stopped["account_risk"]["status"] == "halted"
    assert a.load()["position_qty"] == 0
    state = a.load()
    state["wallet_balance"] = 1000
    a._save(state)
    assert a.reconcile(target(sign * 8, NOW + 2), PRICE,
                       report=report(), now_ms=NOW + 2)["fill"] is None


def test_simulation_execute_and_supervisor_judgment_use_report_ceiling(tmp_path, monkeypatch):
    a = account()
    payload = report(8)
    bar = NOW - 300000
    payload["execution_target"].update(time_ms=bar, available_time_ms=bar + 299999)
    payload.update(paper_inception_utc="generation", generated_at_utc="2026-10-03T22:05:05+00:00",
                   decision_asof_ms=NOW)
    client = Mock()
    client.server_time_ms.return_value = NOW
    client.mark_price_observation.return_value = {"price": PRICE, "time_ms": NOW,
                                                 "next_funding_time_ms": NOW + 300000}
    client.symbol_rules.return_value = execution.DEFAULT_SIMULATION_RULES
    client.funding_history.return_value = []
    monkeypatch.setattr(execution, "SimulationAccount", lambda: a)
    parsed = []
    original = execution.target_from_report
    def capture_target(*args, **kwargs):
        parsed.append(kwargs["leverage_cap"])
        return original(*args, **kwargs)
    monkeypatch.setattr(execution, "target_from_report", capture_target)
    result = execution.execute_report("simulation", payload, client)
    assert parsed == [10, 10]
    assert result["target_qty"] == .08
    client.market_order.assert_not_called()
    report_path, state_path = tmp_path / "report.json", tmp_path / "paper.json"
    report_path.write_text(json.dumps(payload))
    state_path.write_text(json.dumps({"created_at_utc": "generation"}))
    captured = []
    monkeypatch.setattr(decision_runtime, "write_judgment", lambda judgment: captured.append(dict(judgment)))
    monkeypatch.setattr(supervisor, "execute_report", lambda *args: result)
    args = SimpleNamespace(mode="simulation", report_path=report_path, state_path=state_path, factor_profile=None)
    supervisor.run_cycle(args, client, asof_ms=NOW, required_bar=bar)
    assert captured[0]["target_leverage"] == 8
    assert captured[-1]["effective_target_leverage"] == 8


def test_live_parser_ignores_simulation_cap_and_live_ready_rejects_ten(monkeypatch):
    monkeypatch.setenv("SIM_MAX_LEVERAGE", "10")
    captured = []
    executor = Mock()
    executor.reconcile.side_effect = lambda value, *args: captured.append(value) or {}
    monkeypatch.setattr(execution, "LiveExecutor", lambda client: executor)
    live_report = report(8)
    now = int(execution.datetime.now(execution.timezone.utc).timestamp() * 1000)
    live_report["execution_target"]["time_ms"] = now
    execution.execute_report("live", live_report, Mock())
    assert captured[0]["target_leverage"] == 2
    client = object.__new__(BinanceTerminalClient)
    client.base_url = "https://fapi.binance.com"
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("LIVE_MAX_NOTIONAL_USDT", "10000")
    monkeypatch.setenv("LIVE_LEVERAGE", "10")
    with pytest.raises(RuntimeError, match="between 1 and 2"):
        client.validate_live_ready()
