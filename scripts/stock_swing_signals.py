"""Causal signals and risk arithmetic for an isolated stock-perpetual study.

The caller must supply completed, consecutive signal candles. No exchange client,
orders, account state, or active BTC strategy is imported by this module. Prices
used by the risk helpers are actual entry fills; slippage on that fill must not be
charged a second time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Mapping, Sequence


DEFAULT_CONFIG = {
    "ema_fast": 20,
    "ema_mid": 50,
    "ema_slow": 200,
    "ema_slope_bars": 6,
    "breakout_bars": 20,
    "atr_period": 14,
    "warmup_bars": 200,
    "max_breakout_extension_atr": 1.0,
    "max_atr_fraction": 0.03,
    "entry_gap_atr": 0.5,
    "stop_atr": 2.0,
    "min_stop_fraction": 0.02,
    "max_stop_fraction": 0.05,
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _direction(direction: int) -> int:
    if isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError("direction must be 1 (long) or -1 (short)")
    return direction


@dataclass(frozen=True)
class Candle:
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.time_ms, bool) or not isinstance(self.time_ms, int) or self.time_ms < 0:
            raise ValueError("time_ms must be a nonnegative integer")
        prices = [_positive(getattr(self, key), key) for key in ("open", "high", "low", "close")]
        opening, high, low, close = prices
        if low > min(opening, close) or high < max(opening, close) or low > high:
            raise ValueError("Candle OHLC is inconsistent")
        if _finite(self.volume, "volume") < 0:
            raise ValueError("volume must be nonnegative")


@dataclass(frozen=True)
class Signal:
    direction: int
    atr: float
    close: float
    strength: float
    time_ms: int

    def __post_init__(self) -> None:
        _direction(self.direction)
        _positive(self.atr, "signal ATR")
        _positive(self.close, "signal close")
        _finite(self.strength, "signal strength")


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    for key in ("ema_fast", "ema_mid", "ema_slow", "ema_slope_bars", "breakout_bars", "atr_period", "warmup_bars"):
        value = _positive(cfg[key], key)
        if not value.is_integer():
            raise ValueError(f"{key} must be an integer")
        cfg[key] = int(value)
    if not cfg["ema_fast"] < cfg["ema_mid"] < cfg["ema_slow"]:
        raise ValueError("EMA periods must satisfy fast < mid < slow")
    for key in ("max_breakout_extension_atr", "max_atr_fraction", "entry_gap_atr", "stop_atr", "min_stop_fraction", "max_stop_fraction"):
        cfg[key] = _positive(cfg[key], key)
    if not 0 < cfg["min_stop_fraction"] <= cfg["max_stop_fraction"] < 1:
        raise ValueError("Stop fractions must satisfy 0 < min <= max < 1")
    return cfg


def _ema(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    previous = sum(values[:period]) / period
    result[period - 1] = previous
    alpha = 2.0 / (period + 1)
    for index in range(period, len(values)):
        previous += alpha * (values[index] - previous)
        result[index] = previous
    return result


def compute_indicators(
    candles: Sequence[Candle], config: Mapping[str, Any] | None = None,
) -> dict[str, list[float | None]]:
    """SMA-seeded EMAs and Wilder ATR, using only each bar's prefix."""
    cfg = _settings(config)
    if any(previous.time_ms >= current.time_ms for previous, current in zip(candles, candles[1:])):
        raise ValueError("Candles must have strictly increasing timestamps")
    closes = [candle.close for candle in candles]
    true_ranges = []
    for index, candle in enumerate(candles):
        previous_close = candles[index - 1].close if index else candle.close
        true_ranges.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)))
    atr_values: list[float | None] = [None] * len(candles)
    period = cfg["atr_period"]
    if len(candles) >= period:
        previous_atr = sum(true_ranges[:period]) / period
        atr_values[period - 1] = previous_atr
        for index in range(period, len(candles)):
            previous_atr = (previous_atr * (period - 1) + true_ranges[index]) / period
            atr_values[index] = previous_atr
    return {
        "ema_fast": _ema(closes, cfg["ema_fast"]),
        "ema_mid": _ema(closes, cfg["ema_mid"]),
        "ema_slow": _ema(closes, cfg["ema_slow"]),
        "atr": atr_values,
    }


def signal_at(
    candles: Sequence[Candle], indicators: Mapping[str, Sequence[float | None]],
    index: int, config: Mapping[str, Any] | None = None,
) -> Signal | None:
    """Return a completed-bar breakout; the breakout window excludes this bar.

    Strength is the breakout extension divided by ATR, for deterministic ranking
    of simultaneous signals. It is not a forecast probability.
    """
    cfg = _settings(config)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candles):
        raise IndexError("Signal index is outside the candle history")
    minimum = max(cfg["warmup_bars"], cfg["ema_slow"], cfg["ema_mid"] + cfg["ema_slope_bars"], cfg["atr_period"], cfg["breakout_bars"] + 1)
    if index + 1 < minimum:
        return None
    try:
        values = [indicators[key][index] for key in ("ema_fast", "ema_mid", "ema_slow", "atr")]
        values.append(indicators["ema_mid"][index - cfg["ema_slope_bars"]])
    except (KeyError, IndexError) as exc:
        raise ValueError("Indicator arrays do not cover the requested signal") from exc
    if any(value is None for value in values):
        return None
    fast, mid, slow, atr_value, previous_mid = [_finite(value, "indicator") for value in values]
    candle = candles[index]
    if atr_value <= 0 or atr_value / candle.close > cfg["max_atr_fraction"]:
        return None
    prior = candles[index - cfg["breakout_bars"]:index]
    if fast > mid > slow and mid > previous_mid:
        direction, boundary = 1, max(bar.high for bar in prior)
    elif fast < mid < slow and mid < previous_mid:
        direction, boundary = -1, min(bar.low for bar in prior)
    else:
        return None
    extension = direction * (candle.close - boundary)
    if extension <= 0 or extension > cfg["max_breakout_extension_atr"] * atr_value:
        return None
    return Signal(direction, atr_value, candle.close, extension / atr_value, candle.time_ms)


def entry_gap_allowed(
    signal: Signal, next_open: float, config: Mapping[str, Any] | None = None,
) -> bool:
    """Reject adverse gaps using only the observed next opening price."""
    cfg = _settings(config)
    opening = _positive(next_open, "next open")
    return signal.direction * (opening - signal.close) <= cfg["entry_gap_atr"] * signal.atr


def initial_stop(
    entry: float, direction: int, atr: float,
    config: Mapping[str, Any] | None = None,
) -> float | None:
    """Freeze entry risk; return None when a volatility stop exceeds the cap."""
    cfg = _settings(config)
    price = _positive(entry, "entry")
    side = _direction(direction)
    volatility = _positive(atr, "ATR")
    distance_fraction = max(cfg["stop_atr"] * volatility / price, cfg["min_stop_fraction"])
    if distance_fraction > cfg["max_stop_fraction"]:
        return None
    return price * (1 - side * distance_fraction)


def target_exit_price(
    entry: float, direction: int, funding_debit_per_unit: float,
    fee_rate: float, target_margin_return: float = 1.2, leverage: float = 10,
    *, entry_fee_per_unit: float | None = None,
) -> float:
    """Actual exit fill giving the requested net return on initial margin.

    Funding debits are positive and credits negative. Entry and exit fees are
    charged on their respective actual notional values. Slippage is represented
    by actual entry/exit fills and is not another fee in this equation.
    An explicit entry fee preserves the actual paid cost across later fee revisions.
    """
    price = _positive(entry, "entry")
    side = _direction(direction)
    funding = _finite(funding_debit_per_unit, "funding debit per unit")
    fee = _finite(fee_rate, "fee rate")
    target = _finite(target_margin_return, "target margin return")
    multiple = _positive(leverage, "leverage")
    if not 0 <= fee < 1 or target < 0:
        raise ValueError("Fee must be in [0, 1) and target return must be nonnegative")
    if entry_fee_per_unit is None:
        exit_price = (price * (side + target / multiple + fee) + funding) / (side - fee)
    else:
        paid_entry_fee = _finite(entry_fee_per_unit, 'entry fee per unit')
        if paid_entry_fee < 0:
            raise ValueError('Entry fee cannot be negative')
        exit_price = (price * (side + target / multiple) + paid_entry_fee + funding) / (side - fee)
    if exit_price <= 0 or not math.isfinite(exit_price):
        raise ValueError("Requested net target has no positive exit price")
    return exit_price


def per_unit_stop_risk(
    entry: float, stop: float, fee_rate: float = 0.0005, slippage: float = 0.0002,
) -> float:
    """Planned loss including both fees and adverse slippage on the stop fill.

    Entry is already an actual fill. Funding incurred later and price gaps can
    exceed this planned loss; the caller must enforce separate portfolio and
    funding controls.
    """
    price, stop_price = _positive(entry, "entry"), _positive(stop, "stop")
    fee, slip = _finite(fee_rate, "fee rate"), _finite(slippage, "slippage")
    if not 0 <= fee < 1 or not 0 <= slip < 1 or stop_price == price:
        raise ValueError("Invalid fee, slippage, or zero-distance stop")
    side = 1 if stop_price < price else -1
    stop_fill = stop_price * (1 - side * slip)
    return abs(price - stop_fill) + fee * (price + stop_fill)


def position_size(
    equity: float, entry: float, stop: float, step_size: float,
    risk_fraction: float = 0.005, fee_rate: float = 0.0005,
    slippage: float = 0.0002,
) -> float:
    """Floor quantity to the venue step without exceeding the planned risk.

    Returns zero when no complete quantity step fits. Exchange minimum notional,
    remaining correlated risk, isolated margin, and liquidity are caller checks.
    """
    balance = _finite(equity, "equity")
    step = _positive(step_size, "step size")
    fraction = _finite(risk_fraction, "risk fraction")
    if balance < 0 or not 0 < fraction <= 1:
        raise ValueError("Equity must be nonnegative and risk fraction in (0, 1]")
    risk = per_unit_stop_risk(entry, stop, fee_rate, slippage)
    step_decimal = Decimal(str(step))
    quantity = Decimal(str(balance)) * Decimal(str(fraction)) / Decimal(str(risk))
    floored = (quantity / step_decimal).to_integral_value(rounding=ROUND_DOWN) * step_decimal
    return float(floored)
