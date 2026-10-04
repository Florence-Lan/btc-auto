"""Economic/timing integration checks for the isolated public-data replay."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backtest_stock_swing_120 as replay
from stock_swing_signals import Candle, Signal, per_unit_stop_risk, DEFAULT_CONFIG


HOUR = replay.HOUR
ENTRY = 4 * HOUR


def config(symbols=None):
    return {
        **DEFAULT_CONFIG,
        "symbols": symbols or ["MUUSDT"],
        "initial_equity_usdt": 10000,
        "leverage": 10,
        "target_margin_return": 1.2,
        "taker_fee_rate_assumption": 0.001,
        "adverse_slippage_fraction_assumption": 0.0002,
        "risk_fraction_per_trade": 0.005,
        "portfolio_stop_risk_fraction": 0.01,
        "portfolio_gross_notional_fraction": 1.0,
        "max_positions": 2,
        "cooldown_4h_bars": 3,
        "max_holding_calendar_days": 15,
        "account_hard_drawdown_fraction": 0.15,
        "account_soft_drawdown_fraction": 0.08,
        "entry_max_adverse_funding_rate": 0.0003,
        "max_mark_index_basis_fraction": 0.005,
        "max_contract_mark_basis_fraction": 0.005,
        "maintenance_margin_fraction_assumption": 0.015,
        "minimum_liquidation_distance_buffer": 0.01,
        "trail_activation_underlying_return": 0.08,
        "trail_locked_underlying_return": 0.03,
        "trail_atr": 2.0,
        "max_funding_debit_fraction_initial_margin": 0.1,
    }


def bar(timestamp, opening=100, high=101, low=99, close=100):
    return Candle(timestamp, opening, high, low, close, 10)


def funding(timestamp=0, rate=0, price=100):
    return {"fundingTime": timestamp, "fundingRate": str(rate), "markPrice": str(price)}


def source(bars, direction=1, marks=None, funding_events=None, updates=None):
    events = funding_events or [funding()]
    by_hour = {}
    for event in events:
        by_hour.setdefault(event["fundingTime"] // HOUR * HOUR, []).append(event)
    mark_bars = marks or [bar(item.time_ms) for item in bars]
    return {
        "trade": {item.time_ms: item for item in bars},
        "mark": {item.time_ms: item for item in mark_bars},
        "index": {item.time_ms: bar(item.time_ms) for item in bars},
        "signals": {ENTRY: Signal(direction, 1, 100, 0.5, 0)},
        "closed_updates": updates or {},
        "funding": events,
        "funding_times": [event["fundingTime"] for event in events],
        "funding_by_hour": by_hour,
        "tick": 0.01, "step": 0.01, "min_qty": 0.01,
        "max_qty": 1000000, "min_notional": 5,
    }


def run(monkeypatch, prepared, cfg=None, end=None):
    settings = cfg or config(list(prepared))
    monkeypatch.setattr(replay, "prepare", lambda snapshot, settings: copy.deepcopy(prepared))
    last = max(timestamp for item in prepared.values() for timestamp in item["trade"])
    return replay.simulate({}, settings, ENTRY, last + HOUR if end is None else end)


@pytest.mark.parametrize("direction", [1, -1])
def test_same_hour_stop_and_target_resolves_to_stop(monkeypatch, direction):
    extremes = bar(ENTRY, high=114, low=97) if direction == 1 else bar(ENTRY, high=103, low=86)
    result = run(monkeypatch, {"MUUSDT": source([extremes], direction)})
    assert result["summary"]["closed_trades"] == 1
    assert result["trades"][0]["exit_reason"] == "stop"
    assert result["trades"][0]["net_pnl"] < 0
    assert result["summary"]["target_trades_net_at_least_120pct_margin"] == 0


@pytest.mark.parametrize("direction", [1, -1])
def test_target_net_profit_includes_actual_fills_both_fees_and_funding(monkeypatch, direction):
    exit_hour = ENTRY + HOUR
    target_bar = bar(exit_hour, high=115, low=99) if direction == 1 else bar(exit_hour, high=101, low=85)
    prepared = source([bar(ENTRY), target_bar], direction,
                      funding_events=[funding(), funding(exit_hour, direction * 0.0002)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    expected_net = direction * trade["qty"] * (trade["exit_fill"] - trade["entry_fill"])
    expected_net -= trade["entry_fee"] + trade["exit_fee"] + trade["funding_debit"]
    assert trade["exit_reason"] == "target"
    assert trade["entry_fee"] > 0 and trade["exit_fee"] > 0 and trade["funding_debit"] > 0
    assert trade["net_pnl"] == pytest.approx(expected_net)
    assert trade["net_return_initial_margin_pct"] >= 120 - 1e-7
    assert result["summary"]["final_mark_equity"] == pytest.approx(10000 + expected_net)


def test_funding_debit_raises_target_so_a_gross_twelve_percent_move_is_insufficient(monkeypatch):
    exit_hour = ENTRY + HOUR
    prepared = source([bar(ENTRY), bar(exit_hour, high=112.25, low=99, close=110)],
                      funding_events=[funding(), funding(exit_hour, 0.005)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    assert result["summary"]["closed_trades"] == 0
    assert len(result["summary"]["open_positions"]) == 1


def test_missing_settlement_mark_uses_recorded_hour_open_proxy(monkeypatch):
    exit_hour = ENTRY + HOUR
    event = funding(exit_hour, 0.0002)
    del event["markPrice"]
    prepared = source([bar(ENTRY), bar(exit_hour, high=115)],
                      funding_events=[funding(), event])
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    assert trade["funding_debit"] == pytest.approx(trade["qty"] * 100 * 0.0002)
    assert trade["funding_mark_proxy_events"] == 1
    assert trade["net_return_initial_margin_pct"] >= 120 - 1e-7


def test_future_inhour_funding_credit_cannot_lower_target_before_price_path(monkeypatch):
    exit_hour = ENTRY + HOUR
    # A 0.5% credit would move TP below 112.00 if granted prematurely. The
    # uncredited TP is above 112.20; the intrahour high therefore cannot exit.
    prepared = source([bar(ENTRY), bar(exit_hour, high=112, low=99, close=110)],
                      funding_events=[funding(), funding(exit_hour + HOUR // 2, -0.005)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    assert result["summary"]["closed_trades"] == 0
    assert len(result["summary"]["open_positions"]) == 1


def test_inhour_credit_is_not_collected_by_a_position_exited_that_hour(monkeypatch):
    exit_hour = ENTRY + HOUR
    prepared = source([bar(ENTRY), bar(exit_hour, high=115, low=99)],
                      funding_events=[funding(), funding(exit_hour + HOUR // 2, -0.005)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    assert result["trades"][0]["exit_reason"] == "target"
    assert result["trades"][0]["funding_debit"] == 0


def test_existing_position_books_exact_open_funding_before_gap_target(monkeypatch):
    exit_hour = ENTRY + HOUR
    # This open clears the old TP, but not TP after the already-held position's
    # funding debit at this boundary. Exiting without funding is optimistic.
    prepared = source([bar(ENTRY), bar(exit_hour, opening=113, high=113, low=112, close=113)],
                      funding_events=[funding(), funding(exit_hour, 0.01)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    assert result["summary"]["closed_trades"] == 0
    assert len(result["summary"]["open_positions"]) == 1


def test_fresh_entry_after_boundary_does_not_collect_previous_funding_credit(monkeypatch):
    prepared = source([bar(ENTRY, high=112, low=99, close=110)],
                      funding_events=[funding(), funding(ENTRY, -0.005)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    assert result["summary"]["closed_trades"] == 0
    assert len(result["summary"]["open_positions"]) == 1


def test_end_of_sample_reports_open_position_without_inventing_a_close(monkeypatch):
    last = bar(ENTRY, high=106, low=99, close=105)
    prepared = source([last], marks=[last])
    result = run(monkeypatch, {"MUUSDT": prepared})
    summary = result["summary"]
    assert result["trades"] == []
    assert summary["closed_trades"] == 0 and summary["target_trades_net_at_least_120pct_margin"] == 0
    assert len(summary["open_positions"]) == 1
    assert summary["final_mark_equity"] > summary["initial_equity"]
    # Marked equity does not subtract an exit cost for an order never executed.
    assert summary["open_positions"][0]["estimated_close_net_pnl"] < summary["final_mark_equity"] - 10000


def test_liquidation_stress_exhausts_margin_without_double_charging_prior_fees_or_funding(monkeypatch):
    hours = [ENTRY, ENTRY + HOUR, ENTRY + 2 * HOUR]
    prepared = source([bar(timestamp) for timestamp in hours],
                      marks=[bar(hours[0]), bar(hours[1]), bar(hours[2], low=89)],
                      funding_events=[funding(), funding(hours[1], 0.0002)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    assert trade["exit_reason"] == "liquidation_stress"
    assert trade["entry_fee"] > 0 and trade["funding_debit"] > 0
    assert trade["net_pnl"] == pytest.approx(-trade["initial_margin"])
    assert trade["net_return_initial_margin_pct"] == pytest.approx(-100)
    assert result["summary"]["final_mark_equity"] == pytest.approx(10000 - trade["initial_margin"])


def test_correlation_and_position_caps_apply_across_all_three_stocks(monkeypatch):
    symbols = ["MUUSDT", "SNDKUSDT", "SKHYNIXUSDT"]
    prepared = {symbol: source([bar(ENTRY), bar(ENTRY + HOUR, high=115)]) for symbol in symbols}
    result = run(monkeypatch, prepared, config(symbols))
    assert len(result["trades"]) == 2
    assert result["summary"]["skipped_signals"]["position_cap"] == 1
    planned_loss = sum(trade["qty"] * per_unit_stop_risk(trade["entry_fill"], trade["initial_stop"], 0.001, 0.0002)
                       for trade in result["trades"])
    assert planned_loss <= 10000 * 0.01
    for trade in result["trades"]:
        assert trade["initial_margin"] == pytest.approx(trade["qty"] * trade["entry_fill"] / 10)


def test_ten_times_isolated_setting_does_not_override_account_notional_cap(monkeypatch):
    symbols = ["MUUSDT", "SNDKUSDT", "SKHYNIXUSDT"]
    settings = config(symbols)
    settings.update({"taker_fee_rate_assumption": 0, "adverse_slippage_fraction_assumption": 0,
                     "risk_fraction_per_trade": 0.5, "portfolio_stop_risk_fraction": 1.0})
    prepared = {symbol: source([bar(ENTRY), bar(ENTRY + HOUR, high=115)]) for symbol in symbols}
    result = run(monkeypatch, prepared, settings)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["qty"] * trade["entry_fill"] == 10000
    assert trade["initial_margin"] == 1000
    assert 120 <= trade["net_return_initial_margin_pct"] <= 120.1 + 1e-9
    assert result["summary"]["account_return_pct"] == pytest.approx(trade["net_return_initial_margin_pct"] / 10)


def test_trailing_stop_from_completed_bar_only_applies_at_next_open(monkeypatch):
    times = [ENTRY + offset * HOUR for offset in range(5)]
    bars = [bar(timestamp) for timestamp in times[:3]]
    bars += [bar(times[3], high=109, low=100, close=109), bar(times[4], opening=106, high=106, low=106, close=106)]
    marks = [bar(timestamp) for timestamp in times[:3]]
    marks += [bar(times[3], high=109, low=100, close=109), bar(times[4], opening=106, high=106, low=106, close=106)]
    prepared = source(bars, marks=marks, updates={times[4]: (109, 1)})
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    assert trade["exit_reason"] == "stop_gap"
    assert trade["exit_hour_utc"] == replay.iso(times[4])
    assert trade["hours_held"] == 4


def snapshot_from_hourly(rows):
    funding_events = [funding(time) for time in range(0, rows[-1][0] + 1, 8 * HOUR)]
    return {"symbols": {"MUUSDT": {
        "trade_1h": rows, "mark_1h": rows, "index_1h": rows,
        "funding": funding_events,
        "rules_current": {"filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "1000000"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ]},
    }}}


def test_real_preparation_enters_after_full_200_bar_warmup_and_future_rows_do_not_change_past():
    rows = []
    for index in range(840):
        price = 100 + index * 0.01
        rows.append([index * HOUR, price, price + 0.005, price - 0.005, price, 10])
    settings = config()
    end = 801 * HOUR
    full = replay.simulate(snapshot_from_hourly(rows), settings, 0, end)
    prefix = replay.simulate(snapshot_from_hourly(rows[:801]), settings, 0, end)
    assert full == prefix
    assert full["summary"]["closed_trades"] == 0
    assert full["summary"]["open_positions"][0]["entry_utc"] == replay.iso(800 * HOUR)
    assert all(point["positions"] == 0 for point in full["equity_path"][:-1])
    assert full["equity_path"][-1]["positions"] == 1
