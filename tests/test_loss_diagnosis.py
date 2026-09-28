from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_strategy_losses as diagnosis


def test_loss_attribution_reconciles_without_double_counting_slippage():
    trades = [{"net_pnl": -4, "fees": 1, "funding_pnl": -.5, "slippage_cost": .25}]
    result = diagnosis.accounting(trades)
    assert result["price_pnl_before_friction"] == -2.25
    assert result["price_pnl_before_friction"] - result["slippage_cost"] - result["fees"] + result["funding_pnl"] == -4


def test_constant_risk_counterfactual_preserves_unmodified_trades():
    trade = {"initial_qty": 2, "pnl": 10, "fees": 1, "net_pnl": 9, "funding_pnl": 0, "slippage_cost": .2,
             "entry_price": 100, "side": "long", "_open_qty_fraction": .65}
    adjusted = diagnosis.constant_scale([{"trades": [trade]}], .675)[0]["trades"][0]
    assert adjusted["net_pnl"] == pytest.approx(6.075)
    assert adjusted["_open_qty_fraction"] == .65
    assert adjusted["entry_price"] == 100 and adjusted["side"] == "long"
    assert trade["net_pnl"] == 9


def test_inventory_audit_detects_partial_exit_before_final_exit():
    result = diagnosis.inventory_audit([{
        "trades": [{"strategy": "trend", "side": "long", "entry_time_utc": "1970-01-01T00:00:00+00:00",
                    "exit_time_utc": "1970-01-01T00:10:00+00:00", "initial_qty": 1}],
        "equity_curve": [{"time_ms": 0, "signed_qty": 1}, {"time_ms": 300000, "signed_qty": .65},
                         {"time_ms": 600000, "signed_qty": 0}],
    }])
    assert len(result["partial_exit_trades"]) == 1
    assert result["partial_exit_trades"][0]["minimum_remaining_fraction"] == .65
