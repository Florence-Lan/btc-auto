"""Execution-resolution regressions for the isolated stock replay.

Uniform five-minute paths decide stops and liquidation stress. Signals and
trailing changes still become available only after a complete four-hour bar.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backtest_stock_swing_120 as replay
from stock_swing_signals import Candle, Signal, DEFAULT_CONFIG


HOUR = replay.HOUR
STEP = 5 * 60_000
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


def source(bars, direction=1, marks=None, funding_events=None, updates=None,
           entry=ENTRY, step=STEP):
    events = funding_events or [funding(max(0, entry - HOUR))]
    by_step = {}
    for event in events:
        by_step.setdefault(event["fundingTime"] // step * step, []).append(event)
    return {
        "trade": {item.time_ms: item for item in bars},
        "mark": {item.time_ms: item for item in (marks or bars)},
        # Entries are at four-hour boundaries; indexes remain hourly.
        "index": {item.time_ms: bar(item.time_ms) for item in bars if item.time_ms % HOUR == 0},
        "signals": {entry: Signal(direction, 1, 100, 0.5, entry - 4 * HOUR)},
        "closed_updates": updates or {},
        "funding": events,
        "funding_times": [event["fundingTime"] for event in events],
        "funding_by_hour": by_step,
        "tick": 0.01, "step": 0.01, "min_qty": 0.01,
        "max_qty": 1000000, "min_notional": 5,
    }


def run(monkeypatch, prepared, cfg=None, start=ENTRY, step=STEP):
    monkeypatch.setattr(replay, "prepare", lambda snapshot, settings: copy.deepcopy(prepared))
    last = max(timestamp for item in prepared.values() for timestamp in item["trade"])
    return replay.simulate({"execution_step_ms": step}, cfg or config(list(prepared)), start, last + step)


def candle_row(item):
    return [item.time_ms, item.open, item.high, item.low, item.close, item.volume]


def real_snapshot(hourly, execution=None, funding_events=None):
    if execution is None:
        execution = [
            [row[0] + offset, *row[1:]]
            for row in hourly for offset in range(0, HOUR, STEP)
        ]
    events = funding_events or [funding(t) for t in range(0, hourly[-1][0] + 1, 8 * HOUR)]
    return {
        "execution_step_ms": STEP,
        "symbols": {"MUUSDT": {
            "trade_1h": hourly, "mark_1h": hourly, "index_1h": hourly,
            "trade_5m": execution, "mark_5m": copy.deepcopy(execution),
            "funding": events,
            "rules_current": {"filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "1000000"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ]},
        }},
    }


def first_closed_bar_signal(monkeypatch):
    monkeypatch.setattr(
        replay, "signal_at",
        lambda candles, indicators, index, settings: Signal(1, 1, 100, 0.5, 0) if index == 0 else None,
    )


def test_uniform_five_minute_path_stops_before_later_same_hour_liquidation(monkeypatch):
    entry = replay.parse_time("2026-06-09T12:00:00Z")
    stop_time = replay.parse_time("2026-06-09T16:05:00Z")
    liquidation_time = replay.parse_time("2026-06-09T16:35:00Z")
    times = list(range(entry, liquidation_time + STEP, STEP))
    trades = [bar(t, low=97 if t == stop_time else 99) for t in times]
    marks = [bar(t, low=89 if t == liquidation_time else 99) for t in times]
    prepared = source(trades, marks=marks, entry=entry)
    result = run(monkeypatch, {"MUUSDT": prepared}, start=entry)
    trade = result["trades"][0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_utc"] == replay.iso(stop_time)
    assert trade["hours_held"] == pytest.approx(4 + 5 / 60)
    assert result["summary"]["liquidation_stress_count"] == 0
    assert result["summary"]["execution_step_ms"] == STEP
    assert -100 < trade["net_return_initial_margin_pct"] < 0

    # The exact same underlying path aggregated to hours loses temporal order:
    # the 16:00 candle contains both extremes, so conservative liquidation-first
    # processing fabricates a liquidation after an earlier executable stop.
    def hourly(items):
        groups = {}
        for item in items:
            groups.setdefault(item.time_ms // HOUR * HOUR, []).append(item)
        return [Candle(t, group[0].open, max(x.high for x in group),
                       min(x.low for x in group), group[-1].close, 10)
                for t, group in groups.items()]

    coarse = source(hourly(trades), marks=hourly(marks), entry=entry, step=HOUR)
    coarse_result = run(monkeypatch, {"MUUSDT": coarse}, start=entry, step=HOUR)
    assert coarse_result["trades"][0]["exit_reason"] == "liquidation_stress"


def test_real_prepare_buckets_funding_into_five_minutes_with_hourly_indexes():
    hourly = [candle_row(bar(t)) for t in range(0, 8 * HOUR, HOUR)]
    event_time = ENTRY + 7 * 60_000
    snapshot = real_snapshot(hourly, funding_events=[funding(), funding(event_time, 0.0002)])
    prepared = replay.prepare(snapshot, config())["MUUSDT"]
    assert set(prepared["funding_by_hour"]) == {0, ENTRY + STEP}
    assert prepared["funding_by_hour"][ENTRY + STEP][0]["fundingTime"] == event_time
    assert len(prepared["trade"]) == 8 * 12
    assert len(prepared["index"]) == 8
    assert ENTRY + STEP not in prepared["index"]


def test_funding_after_exit_in_a_later_five_minute_bucket_is_not_charged(monkeypatch):
    hourly = [candle_row(bar(t)) for t in range(0, 8 * HOUR, HOUR)]
    snapshot = real_snapshot(hourly, funding_events=[funding(), funding(ENTRY + 17 * 60_000, 0.01)])
    target_row = next(row for row in snapshot["symbols"]["MUUSDT"]["trade_5m"] if row[0] == ENTRY + STEP)
    target_row[2] = 115
    first_closed_bar_signal(monkeypatch)
    result = replay.simulate(snapshot, config(), ENTRY, ENTRY + 4 * STEP)
    trade = result["trades"][0]
    assert trade["exit_reason"] == "target"
    assert trade["exit_utc"] == replay.iso(ENTRY + STEP)
    assert trade["funding_debit"] == 0
    assert trade["net_return_initial_margin_pct"] >= 120 - 1e-7


def test_credit_is_deferred_until_end_of_its_five_minute_bucket_not_end_of_hour(monkeypatch):
    prepared = source(
        [bar(ENTRY, high=112, close=110), bar(ENTRY + STEP, high=112, close=110)],
        funding_events=[funding(), funding(ENTRY + 2 * 60_000, -0.005)],
    )
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    # First-bar 112 is below the uncredited target. The surviving position then
    # receives funding, making the same 112 high sufficient in the next bucket.
    assert trade["exit_utc"] == replay.iso(ENTRY + STEP)
    assert trade["exit_reason"] == "target"
    assert trade["funding_debit"] == pytest.approx(-trade["qty"] * 100 * 0.005)
    assert trade["net_return_initial_margin_pct"] >= 120 - 1e-7


def test_credit_during_target_bucket_is_withheld_after_ambiguous_five_minute_exit(monkeypatch):
    prepared = source(
        [bar(ENTRY), bar(ENTRY + STEP, high=115)],
        funding_events=[funding(), funding(ENTRY + STEP + 2 * 60_000, -0.005)],
    )
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    assert trade["exit_reason"] == "target"
    assert trade["exit_utc"] == replay.iso(ENTRY + STEP)
    assert trade["funding_debit"] == 0
    assert trade["uncredited_ambiguous_funding_events"] == 1


@pytest.mark.parametrize("direction", [1, -1])
def test_five_minute_target_uses_actual_fills_both_fees_and_funding(monkeypatch, direction):
    next_time = ENTRY + STEP
    target = bar(next_time, high=115) if direction == 1 else bar(next_time, high=101, low=85)
    prepared = source([bar(ENTRY), target], direction=direction,
                      funding_events=[funding(), funding(next_time, direction * 0.0002)])
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    expected = direction * trade["qty"] * (trade["exit_fill"] - trade["entry_fill"])
    expected -= trade["entry_fee"] + trade["exit_fee"] + trade["funding_debit"]
    assert trade["exit_reason"] == "target"
    assert trade["exit_utc"] == replay.iso(next_time)
    assert trade["entry_fee"] > 0 and trade["exit_fee"] > 0 and trade["funding_debit"] > 0
    assert trade["net_pnl"] == pytest.approx(expected)
    assert trade["net_return_initial_margin_pct"] >= 120 - 1e-7
    assert result["summary"]["final_mark_equity"] == pytest.approx(10000 + expected)
    assert [row["time_utc"] for row in result["equity_path"]] == [
        replay.iso(ENTRY + STEP), replay.iso(ENTRY + 2 * STEP),
    ]


def test_trailing_update_waits_for_true_four_hour_close(monkeypatch):
    close_time = 8 * HOUR
    bars = [bar(t) for t in range(ENTRY, close_time + STEP, STEP)]
    by_time = {item.time_ms: index for index, item in enumerate(bars)}
    bars[by_time[7 * HOUR]] = bar(7 * HOUR, high=109, low=100, close=109)
    bars[by_time[7 * HOUR + STEP]] = bar(7 * HOUR + STEP, opening=106, high=106, low=106, close=106)
    bars[by_time[close_time - STEP]] = bar(close_time - STEP, high=109, low=100, close=109)
    bars[by_time[close_time]] = bar(close_time, opening=106, high=106, low=106, close=106)
    prepared = source(bars, updates={close_time: (109, 1)})
    result = run(monkeypatch, {"MUUSDT": prepared})
    trade = result["trades"][0]
    assert trade["exit_reason"] == "stop_gap"
    assert trade["exit_utc"] == replay.iso(close_time)
    assert trade["hours_held"] == 4


def test_real_five_minute_execution_accepts_indexes_only_at_hour_boundaries(monkeypatch):
    hourly = [candle_row(bar(t)) for t in range(0, 8 * HOUR, HOUR)]
    snapshot = real_snapshot(hourly)
    first_closed_bar_signal(monkeypatch)
    result = replay.simulate(snapshot, config(), ENTRY, ENTRY + HOUR)
    assert len(result["equity_path"]) == 12
    assert result["summary"]["closed_trades"] == 0
    assert result["summary"]["open_positions"][0]["entry_utc"] == replay.iso(ENTRY)


def test_future_five_minute_extremes_cannot_change_completed_four_hour_entry():
    hourly = []
    for index in range(840):
        price = 100 + index * 0.01
        hourly.append([index * HOUR, price, price + 0.005, price - 0.005, price, 10])
    end = 801 * HOUR
    full_snapshot = real_snapshot(hourly)
    future = full_snapshot["symbols"]["MUUSDT"]
    for key in ("trade_5m", "mark_5m"):
        for row in future[key]:
            if row[0] >= end:
                row[1:5] = [200, 400, 1, 300]
    prefix_snapshot = real_snapshot(hourly[:801])
    full = replay.simulate(full_snapshot, config(), 0, end)
    prefix = replay.simulate(prefix_snapshot, config(), 0, end)
    assert full == prefix
    assert full["summary"]["open_positions"][0]["entry_utc"] == replay.iso(800 * HOUR)
    assert all(point["positions"] == 0 for point in full["equity_path"][:-12])
    assert all(point["positions"] == 1 for point in full["equity_path"][-12:])


def test_prepare_computes_different_ema_profiles_for_each_stock(monkeypatch):
    import stock_swing_profiles

    hourly = [candle_row(bar(t * HOUR, opening=t + 1, high=t + 2, low=t + 1, close=t + 1))
              for t in range(24)]
    snapshot = real_snapshot(hourly)
    snapshot["symbols"]["SNDKUSDT"] = copy.deepcopy(snapshot["symbols"]["MUUSDT"])
    settings = config(["MUUSDT", "SNDKUSDT"])
    settings["symbol_profiles"] = {
        "MUUSDT": {"ema_fast": 2, "ema_mid": 3, "ema_slow": 4, "warmup_bars": 4},
        "SNDKUSDT": {"ema_fast": 3, "ema_mid": 4, "ema_slow": 5, "warmup_bars": 5},
    }
    observed = {}

    def capture(candles, indicators, index, profile):
        if index == 5:
            observed[profile["ema_fast"]] = indicators["ema_fast"][index]
        return None

    monkeypatch.setattr(stock_swing_profiles, "signal_at", capture)
    prepared = replay.prepare(snapshot, settings)
    # Four-hour closes are 4,8,12,16,20,24. SMA-seeded 2/3-period EMAs end
    # at 22/20 respectively; accidentally using the shared defaults fails here.
    assert observed == {2: pytest.approx(22), 3: pytest.approx(20)}
    assert prepared["MUUSDT"]["profile_config"]["ema_fast"] == 2
    assert prepared["SNDKUSDT"]["profile_config"]["ema_fast"] == 3


@pytest.mark.parametrize("missing", ["trade_5m", "mark_5m"])
def test_missing_five_minute_row_is_rejected_instead_of_using_mixed_resolution(missing):
    hourly = [candle_row(bar(t)) for t in range(0, 8 * HOUR, HOUR)]
    snapshot = real_snapshot(hourly)
    del snapshot["symbols"]["MUUSDT"][missing][13]
    with pytest.raises(ValueError, match="Incomplete trade/mark/index alignment|Missing execution candles"):
        replay.prepare(snapshot, config())
