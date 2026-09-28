import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_multifactor_snapshot as download
import frozen_strategy
import multifactor as mf
import portfolio_risk
import simulate_range_swing as sim
import trading_execution


def profile(*groups):
    p = mf.load_profile(ROOT / "config/multifactor_candidate_20260926.json")
    if groups:
        p["groups"] = {name: 1 for name in groups}
    return p


def snapshot(values, first_seen=None):
    return mf.Snapshot({"schema_version": 1, "series": {
        name: [[i * mf.DAY, i * mf.DAY, value, first_seen or i * mf.DAY]
               for i, value in enumerate(data)] for name, data in values.items()
    }})


def test_first_seen_does_not_backfill_forward_decisions():
    payload = {"schema_version": 1, "series": {"x": [[0, mf.DAY, 100, 10 * mf.DAY]]}}
    forward = mf.Snapshot(payload)
    reconstructed = mf.Snapshot(payload, "reconstructed")
    assert forward.window("x", 5 * mf.DAY, 20 * mf.DAY) is None
    assert reconstructed.window("x", 5 * mf.DAY, 20 * mf.DAY) == 100
    assert forward.window("x", 10 * mf.DAY, 20 * mf.DAY) == 100


def test_future_published_observation_is_not_used():
    p = {"schema_version": 1, "series": {"x": [[0, 0, 100, 0], [1, 100, 1000, 2]]}}
    assert mf.Snapshot(p).window("x", 50, 100) == 100


def test_late_backfill_does_not_replace_newest_observation():
    p = {"schema_version": 1, "series": {"x": [[10, 10, 100, 10], [1, 1, 50, 20]]}}
    assert mf.Snapshot(p).window("x", 20, 100) == 100


@pytest.mark.parametrize("row", [[0, 0, float("nan"), 0], [10, 1, 3, 10], [10, 10, 3, 1]])
def test_invalid_observation_rejected(row):
    with pytest.raises(ValueError):
        mf.Snapshot({"schema_version": 1, "series": {"x": [row]}})


def test_macro_bearishness_penalizes_long_not_short():
    data = {name: [1 + i * .05 for i in range(20)]
            for name in ("ust_2y", "ust_10y", "ust_real_10y")}
    s = snapshot(data)
    long = mf.decision_at(s, 19 * mf.DAY, "long", profile("treasury"))
    short = mf.decision_at(s, 19 * mf.DAY, "short", profile("treasury"))
    assert not long.allowed
    assert short.allowed and short.risk_multiplier <= 1
    assert short.side_alignment == -long.side_alignment


def test_negative_real_yield_uses_percentage_points():
    s = snapshot({name: [-2 + i * .01 for i in range(20)]
                  for name in ("ust_2y", "ust_10y", "ust_real_10y")})
    d = mf.decision_at(s, 19 * mf.DAY, "long", profile("treasury"))
    assert d.coverage == 1
    assert d.features["ust_real_10y_change"] == pytest.approx(.07)


def test_strong_dollar_quote_conventions_align():
    values = {name: [100 - i for i in range(20)] for name in ("eurusd", "gbpusd")}
    values.update({name: [100 + i for i in range(20)]
                   for name in ("usdjpy", "usdcny", "usdchf", "broad_dollar")})
    d = mf.decision_at(snapshot(values), 19 * mf.DAY, "long", profile("fx"))
    assert d.contributions["fx"] < 0


def test_high_vix_blocks_both_sides():
    values = {name: [100] * 20 for name in ("sp500", "nasdaq", "oil")}
    values["vix"] = [50] * 20
    for side in ("long", "short"):
        d = mf.decision_at(snapshot(values), 19 * mf.DAY, side, profile("global_risk"))
        assert not d.allowed
        assert "market_stress" in d.reasons
        assert d.risk_multiplier == 0


def test_ablated_group_removes_its_stress_effect_too():
    values = {name: [100] * 20 for name in ("btc_close", "sp500", "nasdaq", "oil")}
    values["vix"] = [50] * 20
    d = mf.decision_at(snapshot(values), 19 * mf.DAY, "long", profile("btc_momentum"))
    assert d.allowed and d.stress == 0


def test_long_short_ratio_is_not_a_direct_buy_signal():
    values = {"btc_close": [100] * 20, "global_long_short": [3] * 20,
              "top_position_long_short": [3] * 20, "taker_buy_sell": [1] * 20,
              "open_interest": [100] * 20, "funding": [.0005] * 20}
    d = mf.decision_at(snapshot(values), 19 * mf.DAY, "long", profile("positioning"))
    assert d.contributions["positioning"] < 0


def test_stale_or_missing_required_group_blocks_entry():
    s = snapshot({"btc_close": [100] * 20})
    d = mf.decision_at(s, 19 * mf.DAY, "long", profile())
    assert not d.allowed and "fed" in d.missing_groups
    stale = mf.decision_at(s, 20 * mf.DAY, "short", profile("btc_momentum"))
    assert not stale.allowed and stale.coverage == 0


def test_overlay_preserves_direction_and_open_quantity_fraction():
    s = snapshot({"btc_close": [100] * 20})
    raw = {"entry_time_utc": sim.iso_utc_from_ms(19 * mf.DAY), "side": "long",
           "initial_qty": 2, "net_pnl": 10, "fees": 2, "_open_qty_fraction": .4}
    original = copy.deepcopy(raw)
    output, diagnostics = mf.apply_overlay([{"trades": [raw]}], s, profile("btc_momentum"))
    trade = output[0]["trades"][0]
    assert raw == original
    assert trade["side"] == "long" and trade["_open_qty_fraction"] == .4
    assert 0 < trade["initial_qty"] < raw["initial_qty"]
    assert trade["net_pnl"] / raw["net_pnl"] == trade["fees"] / raw["fees"]
    assert diagnostics["group_coverage_pct"]["btc_momentum"] == 100


def test_archive_never_rewrites_first_seen_values():
    old = [[1, 2, 100, 3]]
    result = download.merge_rows(old, [[1, 2, 200, 4], [2, 3, 150, 4]])
    assert result == [[1, 2, 100, 3], [2, 3, 150, 4]]


def test_fred_missing_values_and_release_lag():
    rows = download.fred_rows("observation_date,DGS2\n2026-01-01,4.2\n2026-01-02,.\n", "DGS2", 2, 2_000_000_000_000)
    assert len(rows) == 1 and rows[0][1] - rows[0][0] == 2 * mf.DAY


def test_derivative_pagination_retains_oldest_and_newest_completed_hours():
    end = 1000 * mf.HOUR

    def endpoint(path, params):
        # Model Binance returning the LAST limit rows in the requested window.
        rows = [{"timestamp": i * mf.HOUR, "longShortRatio": "1.2"}
                for i in range(1001) if params["startTime"] <= i * mf.HOUR <= params["endTime"]]
        return rows[-params["limit"]:]

    with patch.object(download, "binance_json", side_effect=endpoint):
        rows = download.fetch_derivatives("global_long_short", 0, end)
    assert rows[0][0] == end - 29 * mf.DAY
    assert rows[-1][0] == end - 2 * mf.HOUR
    assert len({r[0] for r in rows}) == len(rows)


def test_research_report_cannot_reach_live_client():
    client = Mock()
    with pytest.raises(ValueError, match="simulation-only"):
        trading_execution.execute_report("live", {"research_only": True}, client)
    assert not client.mock_calls


def test_drawdown_halt_remains_latched_after_recovery():
    _, cfg = frozen_strategy.load_frozen_strategy(ROOT / "config/frozen_strategy_candidate_20260917.json")
    cfg = replace(cfg, initial_equity=100)
    bars = [sim.Candle(i * 300000, sim.iso_utc_from_ms(i * 300000), p, p, p, p, 10, 1000,
                       (i + 1) * 300000 - 1) for i, p in enumerate([100, 80, 100, 100, 100])]

    def trade(entry, qty):
        return {"side": "long", "entry_time_utc": bars[entry].open_time_utc,
                "exit_time_utc": bars[4].open_time_utc, "entry_price": 100,
                "avg_exit_price": 100, "initial_qty": qty, "pnl": 0, "fees": 0, "net_pnl": 0,
                "bars_held": 4 - entry, "exit_reason": "test", "signal_reason": "test"}

    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, [{"trades": [trade(0, 1), trade(3, .1)]}], cfg, portfolio_risk.DrawdownRiskPolicy())
    assert result["risk_diagnostics"]["blocked_entries"] == 1
    assert result["equity_curve"][-1]["drawdown_risk_multiplier"] == 0
    assert len(result["trades"]) == 1
