from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stock_swing_signals as swing


def trend_candles(direction=1, length=240):
    bars = []
    for index in range(length):
        close = 150 + direction * index * 0.1
        bars.append(swing.Candle(index * 14_400_000, close, close + 0.05, close - 0.05, close))
    return bars


@pytest.mark.parametrize("direction", [1, -1])
def test_completed_signal_is_unchanged_when_future_prices_change(direction):
    bars = trend_candles(direction)
    index = 210
    prefix = bars[:index + 1]
    prefix_ind = swing.compute_indicators(prefix)
    expected = swing.signal_at(prefix, prefix_ind, index)
    assert expected is not None and expected.direction == direction
    future = bars[:index + 1] + [
        swing.Candle(bar.time_ms, 500, 510, 490, 500) for bar in bars[index + 1:]
    ]
    full_ind = swing.compute_indicators(future)
    assert swing.signal_at(future, full_ind, index) == expected
    for key in prefix_ind:
        assert full_ind[key][:index + 1] == prefix_ind[key]


@pytest.mark.parametrize("direction", [1, -1])
def test_breakout_uses_prior_bars_and_excludes_current_extreme(direction):
    bars = trend_candles(direction, 211)
    last = bars[-1]
    high, low = (last.close + 3, last.low) if direction == 1 else (last.high, last.close - 3)
    bars[-1] = swing.Candle(last.time_ms, last.open, high, low, last.close)
    signal = swing.signal_at(bars, swing.compute_indicators(bars), len(bars) - 1)
    assert signal is not None and signal.direction == direction
    # A close exactly at the prior breakout boundary is not a breakout.
    boundary = max(bar.high for bar in bars[-21:-1]) if direction == 1 else min(bar.low for bar in bars[-21:-1])
    bars[-1] = swing.Candle(last.time_ms, boundary, max(high, boundary), min(low, boundary), boundary)
    assert swing.signal_at(bars, swing.compute_indicators(bars), len(bars) - 1) is None


def test_warmup_and_strict_trend_ordering_block_entry():
    bars = trend_candles(length=211)
    indicators = swing.compute_indicators(bars)
    assert swing.signal_at(bars, indicators, 198) is None
    assert swing.signal_at(bars, indicators, 199) is not None
    indicators["ema_fast"][-1] = indicators["ema_mid"][-1]
    assert swing.signal_at(bars, indicators, 210) is None


def test_indicator_seed_and_wilder_recursion_have_known_values():
    bars = [swing.Candle(i, value, value + 1, value - 1, value) for i, value in enumerate([10, 11, 13, 12, 14])]
    values = swing.compute_indicators(bars, {"ema_fast": 2, "ema_mid": 3, "ema_slow": 4, "atr_period": 2})
    assert values["ema_fast"][0] is None
    assert values["ema_fast"][1] == 10.5
    assert values["ema_fast"][2] == pytest.approx(10.5 + (2 / 3) * 2.5)
    assert values["atr"][:2] == [None, 2.0]
    assert values["atr"][2] == 2.5
    assert values["atr"][3] == 2.25


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("funding", [-0.35, 0.0, 0.7])
def test_exit_target_solves_net_margin_profit_after_both_fees_and_funding(direction, funding):
    entry, fee, qty, leverage = 123.45, 0.0005, 7.3, 10
    exit_price = swing.target_exit_price(entry, direction, funding, fee)
    pnl = qty * direction * (exit_price - entry) - fee * qty * (entry + exit_price) - funding * qty
    initial_margin = qty * entry / leverage
    assert pnl / initial_margin == pytest.approx(1.2)
    gross_target = entry * (1 + direction * 0.12)
    if funding >= 0:
        assert direction * (exit_price - gross_target) > 0


def test_funding_debit_moves_target_further_and_credit_moves_it_closer():
    for direction in (1, -1):
        baseline = swing.target_exit_price(100, direction, 0, 0.0005)
        debit = swing.target_exit_price(100, direction, 0.3, 0.0005)
        credit = swing.target_exit_price(100, direction, -0.3, 0.0005)
        assert direction * (debit - baseline) > 0
        assert direction * (credit - baseline) < 0


@pytest.mark.parametrize("direction", [1, -1])
def test_next_open_gap_filter_is_directional_and_inclusive(direction):
    signal = swing.Signal(direction, 2, 100, 0.5, 0)
    assert swing.entry_gap_allowed(signal, 100 + direction)
    assert not swing.entry_gap_allowed(signal, 100 + direction * 1.001)
    assert swing.entry_gap_allowed(signal, 100 - direction * 10)


@pytest.mark.parametrize("direction", [1, -1])
def test_stop_floor_cap_and_actual_entry_price(direction):
    assert swing.initial_stop(100, direction, 0.3) == 100 * (1 - direction * 0.02)
    assert swing.initial_stop(100, direction, 2.5) == 100 * (1 - direction * 0.05)
    assert swing.initial_stop(100, direction, 2.5001) is None
    assert swing.initial_stop(200, direction, 2.5) == 200 * (1 - direction * 0.025)


@pytest.mark.parametrize("stop", [96, 104])
def test_quantity_floor_respects_total_planned_loss_including_exit_slippage(stop):
    equity, entry, step = 1000, 100, 0.03
    quantity = swing.position_size(equity, entry, stop, step)
    side = 1 if stop < entry else -1
    stop_fill = stop * (1 - side * 0.0002)
    actual_per_unit_loss = side * (entry - stop_fill) + 0.0005 * (entry + stop_fill)
    assert actual_per_unit_loss == pytest.approx(swing.per_unit_stop_risk(entry, stop))
    assert quantity * actual_per_unit_loss <= equity * 0.005
    assert (quantity + step) * actual_per_unit_loss > equity * 0.005
    assert quantity / step == pytest.approx(round(quantity / step))
    assert swing.position_size(0, entry, stop, step) == 0
    assert swing.position_size(1, entry, stop, 1) == 0


@pytest.mark.parametrize("bad_args", [
    {"equity": -1}, {"entry": 0}, {"stop": 100}, {"step_size": 0},
    {"risk_fraction": 0}, {"fee_rate": math.nan}, {"slippage": -0.1},
])
def test_position_size_rejects_invalid_or_unsafe_data(bad_args):
    args = {"equity": 1000, "entry": 100, "stop": 96, "step_size": 0.01, **bad_args}
    with pytest.raises(ValueError):
        swing.position_size(**args)


def test_invalid_direction_and_unreachable_short_target_are_rejected():
    with pytest.raises(ValueError):
        swing.target_exit_price(100, 0, 0, 0.0005)
    with pytest.raises(ValueError):
        swing.target_exit_price(100, -1, 100, 0.0005)


def test_malformed_ohlc_or_duplicate_timestamps_are_rejected():
    with pytest.raises(ValueError):
        swing.Candle(0, 100, 99, 98, 100)
    bar = swing.Candle(0, 100, 101, 99, 100)
    with pytest.raises(ValueError):
        swing.compute_indicators([bar, bar])
