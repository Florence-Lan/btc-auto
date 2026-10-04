"""The selected simulation limit must agree with the displayed execution target."""
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_trading_terminal as terminal


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("environment, expected", [(None, 10), ("3", 3)])
def test_simulation_status_uses_selected_cap(monkeypatch, side, environment, expected):
    if environment is None:
        monkeypatch.delenv("SIM_MAX_LEVERAGE", raising=False)
    else:
        monkeypatch.setenv("SIM_MAX_LEVERAGE", environment)
    monkeypatch.setenv("SIM_MAX_NOTIONAL_USDT", "0")
    candidate = {"candidate_id": "test_10x", "risk_limits": {"portfolio_leverage_cap": 10}}
    report = {"execution_target": {"equity": 1000, "price": 100000, "signed_qty": side * .08}}
    values = {terminal.CANDIDATE_PATH: candidate, terminal.REPORT_PATH: report}
    monkeypatch.setattr(terminal, "read_json", lambda path, default=None: values.get(path, default))
    account = {"wallet_balance": 1000, "margin_balance": 1000, "available_balance": 1000}
    snapshot = {"state": {"initial_balance": 1000}, "account": account, "positions": [],
                "open_orders": [], "recent_trades": [], "equity_curve": [], "max_drawdown_pct": 0}
    monkeypatch.setattr(terminal, "SimulationAccount", lambda: Mock(snapshot=Mock(return_value=snapshot)))
    monkeypatch.setattr(terminal.simulation_risk_monitor, "status_view", lambda **kwargs: {})
    monkeypatch.setattr(terminal.decision_runtime, "status_view", lambda **kwargs: {})
    monkeypatch.setattr(terminal, "tail_lines", lambda *args: [])
    client = Mock(configured=False, live_trading_enabled=False)
    client.mark_price.return_value = 100000
    client.cooldown_until_ms.return_value = None
    controller = terminal.TerminalController(client)
    controller.mode = lambda: "simulation"
    controller.runtime = lambda: {"running": True}
    controller.emergency = lambda: {}
    controller._macro_status = lambda *args: ({}, {})
    status = controller.status()
    assert status["risk"]["portfolio_leverage_cap"] == 10
    assert status["execution"]["leverage"] == expected
    assert status["strategy"]["target_leverage"] == side * min(8, expected)
    assert status["strategy"]["target_notional"] == side * min(8, expected) * 1000
    assert status["execution"]["places_orders"] is False
