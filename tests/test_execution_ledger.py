from dataclasses import asdict

import pytest

from test_strategy_engine import candle, config
import execution_ledger
import portfolio_risk
import diagnose_strategy_losses as diagnosis


def sleeve_fixture():
    bars = [candle(0, 100, 100, 100, 100), candle(1, 100, 110, 100, 110),
            candle(2, 110, 110, 90, 90), candle(3, 90, 90, 90, 90)]
    cfg = config(initial_equity=100, taker_fee=.001, maker_fee=.001, portfolio_leverage_cap=10)
    trade = dict(side="long", entry_time_utc=bars[0].open_time_utc,
                 exit_time_utc=bars[3].open_time_utc, entry_price=100, avg_exit_price=100,
                 initial_qty=1, pnl=-.2, fees=.2, net_pnl=-.4, return_on_equity_pct=-.4,
                 bars_held=3, exit_reason="stop", signal_reason="trend_test", liquidation_price=0,
                 funding_pnl=-.2, slippage_cost=0, strategy="trend_pullback_5m")
    # Entry -0.1 fee; half exit +5 -0.055 fee; funding -0.2;
    # final half exit -5 -0.045 fee. Curve includes estimated remaining exit fees.
    curve = [dict(time_ms=b.open_time_ms, price=b.close, signed_qty=q, equity=e)
             for b, q, e in zip(bars, [1, .5, .5, 0], [99.8, 109.79, 99.6, 99.6])]
    return bars, cfg, dict(trades=[trade], config=asdict(cfg), equity_curve=curve)


def test_partial_realization_funding_fees_reconcile_without_double_counting():
    bars, cfg, sleeve = sleeve_fixture()
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, [sleeve], cfg, portfolio_risk.DrawdownRiskPolicy(20, 30))
    curve = result["equity_curve"]
    assert [p["signed_qty"] for p in curve] == [1, .5, .5, 0]
    assert [p["equity"] for p in curve] == pytest.approx([99.8, 109.79, 99.6, 99.6])
    assert curve[1]["cash_equity"] == pytest.approx(104.845)
    assert curve[2]["cash_equity"] == pytest.approx(104.645)
    assert sum(t["net_pnl"] for t in result["trades"]) == pytest.approx(-.4)
    assert result["risk_diagnostics"]["legacy_endpoint_trades"] == 0


def test_overlay_scales_ledger_once_and_leaves_source_unchanged():
    bars, cfg, sleeve = sleeve_fixture()
    adjusted = diagnosis.constant_scale([sleeve], .5)
    assert "_ledger" not in sleeve["trades"][0]
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, adjusted, cfg, portfolio_risk.DrawdownRiskPolicy(20, 30))
    assert result["equity_curve"][1]["signed_qty"] == .25
    assert result["equity_curve"][1]["equity"] == pytest.approx(104.895)
    assert result["summary"]["final_equity"] == pytest.approx(99.8)


def test_hourly_end_settles_at_close_without_using_future_close_at_open():
    bars = [candle(i, 100, 100, 100, 100) for i in range(12)]
    cfg = config(initial_equity=100, taker_fee=0, maker_fee=0, timeseries_timeframe="1h")
    _, _, source = sleeve_fixture()
    trade = {**source["trades"][0], "entry_time_utc": bars[0].open_time_utc,
             "exit_time_utc": bars[0].open_time_utc, "strategy": "timeseries_trend_6h",
             "exit_reason": "end", "net_pnl": 10, "pnl": 10, "fees": 0}
    sleeve = {"trades": [trade], "config": asdict(cfg),
              "equity_curve": [dict(time_ms=0, price=110, signed_qty=1, equity=110)]}
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(bars, [sleeve], cfg, portfolio_risk.DrawdownRiskPolicy())
    assert all(p["equity"] == 100 for p in result["equity_curve"][:-1])
    assert result["equity_curve"][-1]["equity"] == 110
    assert result["equity_curve"][-2]["signed_qty"] == 1
    assert result["equity_curve"][-1]["signed_qty"] == 0


def test_partial_exit_releases_capacity_before_next_entry():
    bars, cfg, sleeve = sleeve_fixture()
    from dataclasses import replace
    cfg = replace(cfg, portfolio_leverage_cap=1.5)
    second = {**sleeve["trades"][0], "entry_time_utc": bars[2].open_time_utc,
              "exit_time_utc": bars[3].open_time_utc, "entry_price": 110, "initial_qty": 1,
              "net_pnl": 0, "pnl": 0, "fees": 0, "strategy": "other"}
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, [sleeve, {"trades": [second]}], cfg, portfolio_risk.DrawdownRiskPolicy(20, 30))
    # At the next open: equity 109.79, gross held 55 -> capacity 109.685.
    t = next(t for t in result["trades"] if t["strategy"] == "other")
    assert t["initial_qty"] == pytest.approx((109.79 * 1.5 - 55) / 110)


def test_simultaneous_entries_book_first_fee_before_second_capacity_check():
    from dataclasses import replace
    bars, cfg, sleeve = sleeve_fixture()
    cfg = replace(cfg, portfolio_leverage_cap=1.5, taker_fee=.1)
    raw = sleeve["trades"][0]
    first = {**raw, "_ledger": [
        {"time_ms": 0, "remaining_fraction": 1, "cash_per_unit": -10},
        {"time_ms": bars[-1].close_time_ms, "remaining_fraction": 0, "cash_per_unit": -.4}]}
    second = {**raw, "strategy": "second"}
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, [{"trades": [first]}, {"trades": [second]}], cfg, portfolio_risk.DrawdownRiskPolicy(40, 60))
    trade = next(t for t in result["trades"] if t["strategy"] == "second")
    assert trade["initial_qty"] == pytest.approx(.2)
