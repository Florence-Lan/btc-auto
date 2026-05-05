#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open_time_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    close_time_ms: int


@dataclass
class PendingEntry:
    side: str
    signal_index: int
    created_index: int
    expires_index: int
    target_price: float
    stop_price: float
    tp1: float
    tp2: float
    tp3: float
    signal_score: float
    risk_multiplier: float
    signal_reason: str


@dataclass
class Position:
    side: str
    entry_index: int
    entry_time_utc: str
    entry_price: float
    qty: float
    initial_qty: float
    stop_price: float
    tp1: float
    tp2: float
    tp3: float
    liquidation_price: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    best_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    exit_notional_sum: float = 0.0
    exited_qty: float = 0.0
    exit_notes: Optional[str] = None


@dataclass
class Trade:
    side: str
    entry_time_utc: str
    exit_time_utc: str
    entry_price: float
    avg_exit_price: float
    initial_qty: float
    pnl: float
    fees: float
    net_pnl: float
    return_on_equity_pct: float
    bars_held: int
    exit_reason: str
    signal_reason: str
    liquidation_price: float


@dataclass(frozen=True)
class StrategyConfig:
    initial_equity: float
    leverage: float
    risk_per_trade: float
    taker_fee: float
    maker_fee: float
    bb_period: int
    bb_std: float
    rsi_period: int
    atr_period: int
    adx_period: int
    ema_fast: int
    ema_slow: int
    max_adx: float
    max_bandwidth: float
    max_ema_spread: float
    side_mode: str
    drift_lookback_bars: int
    countertrend_drift_limit_pct: float
    long_rsi: float
    short_rsi: float
    reclaim_buffer_pct: float
    min_band_excursion_pct: float
    entry_pullback_atr: float
    entry_edge_ratio: float
    min_pullback_pct: float
    max_wait_bars: int
    stop_atr: float
    min_stop_pct: float
    min_reward_risk: float
    tp1_close_ratio: float
    tp2_close_ratio: float
    break_even_buffer_pct: float
    trail_atr: float
    max_hold_bars: int
    cooldown_bars: int
    max_drawdown_stop_pct: float
    high_adx_drift_threshold: float
    min_high_adx_drift_pct: float
    min_signal_score: float
    adaptive_risk_enabled: bool
    min_risk_multiplier: float
    max_risk_multiplier: float
    maintenance_margin_pct: float
    liquidation_fee_pct: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    depth_impact_bps: float
    depth_impact_exponent: float
    min_depth_quote: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    factors = {
        "m": 60_000,
        "h": 60 * 60_000,
        "d": 24 * 60 * 60_000,
        "w": 7 * 24 * 60 * 60_000,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported interval: {interval}")
    return value * factors[unit]


def normalize_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.strip().upper()
    if ":" in symbol:
        symbol = symbol.split(":", 1)[0]
    return symbol.replace("/", "").replace("-", "").replace("_", "")


def iso_utc_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def candle_from_kline(entry: Sequence[Any]) -> Candle:
    return Candle(
        open_time_ms=int(entry[0]),
        open_time_utc=iso_utc_from_ms(int(entry[0])),
        open=float(entry[1]),
        high=float(entry[2]),
        low=float(entry[3]),
        close=float(entry[4]),
        volume=float(entry[5]),
        quote_volume=float(entry[7]),
        close_time_ms=int(entry[6]),
    )


def load_candles_from_snapshot(path: Path) -> List[Candle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candles: List[Candle] = []
    for entry in payload.get("klines", []):
        open_dt = datetime.fromisoformat(entry["open_time_utc"])
        close_dt = datetime.fromisoformat(entry["close_time_utc"])
        candles.append(
            Candle(
                open_time_ms=int(open_dt.timestamp() * 1000),
                open_time_utc=entry["open_time_utc"],
                open=float(entry["open"]),
                high=float(entry["high"]),
                low=float(entry["low"]),
                close=float(entry["close"]),
                volume=float(entry["volume_base"]),
                quote_volume=float(entry.get("volume_quote", float(entry["volume_base"]) * float(entry["close"]))),
                close_time_ms=int(close_dt.timestamp() * 1000),
            )
        )
    return sorted(candles, key=lambda candle: candle.open_time_ms)


def fetch_futures_klines(symbol: str, interval: str, days: float) -> List[Candle]:
    interval_ms = interval_to_ms(interval)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - int(days * 24 * 60 * 60 * 1000)
    end_ms = now_ms
    candles: List[Candle] = []
    seen: set[int] = set()

    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-auto-simulator/1.0"})
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            }
            last_error: Optional[Exception] = None
            for attempt in range(4):
                try:
                    response = session.get(
                        f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/klines",
                        params=params,
                        timeout=45,
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"failed to fetch Binance klines after retries: {last_error}") from last_error
            batch = response.json()
            if not batch:
                break
            for raw in batch:
                candle = candle_from_kline(raw)
                if candle.open_time_ms not in seen:
                    candles.append(candle)
                    seen.add(candle.open_time_ms)
            next_cursor = int(batch[-1][0]) + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1500:
                break
            time.sleep(0.08)

    candles = sorted(candles, key=lambda candle: candle.open_time_ms)
    return [candle for candle in candles if candle.close_time_ms <= end_ms]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return result
    seed = mean(values[:period])
    result[period - 1] = seed
    alpha = 2 / (period + 1)
    prev = seed
    for index in range(period, len(values)):
        prev = values[index] * alpha + prev * (1 - alpha)
        result[index] = prev
    return result


def bollinger(values: Sequence[float], period: int, std_mult: float) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    mid: List[Optional[float]] = [None] * len(values)
    upper: List[Optional[float]] = [None] * len(values)
    lower: List[Optional[float]] = [None] * len(values)
    bandwidth: List[Optional[float]] = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        avg = mean(window)
        dev = stddev(window)
        mid[index] = avg
        upper[index] = avg + std_mult * dev
        lower[index] = avg - std_mult * dev
        bandwidth[index] = ((upper[index] - lower[index]) / avg) if avg else None
    return mid, upper, lower, bandwidth


def rsi(values: Sequence[float], period: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return result

    gains = []
    losses = []
    for index in range(1, period + 1):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    result[period] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    for index in range(period + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        result[index] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    return result


def atr(candles: Sequence[Candle], period: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(candles)
    trs: List[float] = []
    for index, candle in enumerate(candles):
        if index == 0:
            tr = candle.high - candle.low
        else:
            prev_close = candles[index - 1].close
            tr = max(
                candle.high - candle.low,
                abs(candle.high - prev_close),
                abs(candle.low - prev_close),
            )
        trs.append(tr)
        if index == period - 1:
            result[index] = mean(trs)
        elif index >= period:
            prev = result[index - 1]
            result[index] = ((prev or tr) * (period - 1) + tr) / period
    return result


def adx(candles: Sequence[Candle], period: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(candles)
    if len(candles) <= period * 2:
        return result

    tr_values = [0.0] * len(candles)
    plus_dm = [0.0] * len(candles)
    minus_dm = [0.0] * len(candles)
    for index in range(1, len(candles)):
        current = candles[index]
        previous = candles[index - 1]
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm[index] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[index] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr_values[index] = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )

    smoothed_tr = sum(tr_values[1 : period + 1])
    smoothed_plus = sum(plus_dm[1 : period + 1])
    smoothed_minus = sum(minus_dm[1 : period + 1])
    dx_values: List[Optional[float]] = [None] * len(candles)

    for index in range(period, len(candles)):
        if index > period:
            smoothed_tr = smoothed_tr - smoothed_tr / period + tr_values[index]
            smoothed_plus = smoothed_plus - smoothed_plus / period + plus_dm[index]
            smoothed_minus = smoothed_minus - smoothed_minus / period + minus_dm[index]
        if smoothed_tr <= 0:
            continue
        plus_di = 100 * smoothed_plus / smoothed_tr
        minus_di = 100 * smoothed_minus / smoothed_tr
        denom = plus_di + minus_di
        dx_values[index] = 0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom

    first_adx_index = period * 2
    initial_dx = [value for value in dx_values[period:first_adx_index] if value is not None]
    if len(initial_dx) < period:
        return result
    prev_adx = mean(initial_dx)
    result[first_adx_index - 1] = prev_adx
    for index in range(first_adx_index, len(candles)):
        dx = dx_values[index]
        if dx is None:
            continue
        prev_adx = (prev_adx * (period - 1) + dx) / period
        result[index] = prev_adx
    return result


def indicators(candles: Sequence[Candle], cfg: StrategyConfig) -> Dict[str, List[Optional[float]]]:
    closes = [candle.close for candle in candles]
    mid, upper, lower, bandwidth = bollinger(closes, cfg.bb_period, cfg.bb_std)
    return {
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "rsi": rsi(closes, cfg.rsi_period),
        "atr": atr(candles, cfg.atr_period),
        "adx": adx(candles, cfg.adx_period),
        "ema_fast": ema(closes, cfg.ema_fast),
        "ema_slow": ema(closes, cfg.ema_slow),
    }


def is_ready(values: Iterable[Optional[float]]) -> bool:
    return all(value is not None for value in values)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def side_aligned_drift(side: str, drift_value: Optional[float]) -> float:
    if drift_value is None:
        return 0.0
    return drift_value if side == "long" else -drift_value


def signal_quality_score(
    side: str,
    rsi_value: float,
    adx_value: float,
    bandwidth: float,
    ema_spread: float,
    drift_value: Optional[float],
    cfg: StrategyConfig,
) -> float:
    aligned_drift = side_aligned_drift(side, drift_value)
    if adx_value >= cfg.high_adx_drift_threshold and aligned_drift < cfg.min_high_adx_drift_pct:
        return 0.0

    if side == "long":
        rsi_depth = clamp((cfg.long_rsi - rsi_value) / 10, 0.0, 1.0)
    else:
        rsi_depth = clamp((rsi_value - cfg.short_rsi) / 10, 0.0, 1.0)

    drift_strength = clamp(aligned_drift / 0.012, -0.35, 1.0)
    bandwidth_quality = clamp(bandwidth / 0.018, 0.0, 1.0)
    ema_quality = clamp(ema_spread / 0.006, 0.0, 1.0)
    adx_penalty = clamp((adx_value - 24) / 12, 0.0, 1.0)

    score = (
        0.42
        + 0.20 * rsi_depth
        + 0.24 * max(drift_strength, 0.0)
        + 0.10 * bandwidth_quality
        + 0.08 * ema_quality
        - 0.08 * adx_penalty
    )
    if aligned_drift < 0:
        score += 0.20 * drift_strength
    return clamp(score, 0.0, 1.0)


def risk_multiplier_from_score(score: float, cfg: StrategyConfig) -> float:
    if not cfg.adaptive_risk_enabled:
        return 1.0
    if cfg.min_signal_score >= 1:
        return cfg.min_risk_multiplier
    normalized = clamp((score - cfg.min_signal_score) / (1 - cfg.min_signal_score), 0.0, 1.0)
    return cfg.min_risk_multiplier + (cfg.max_risk_multiplier - cfg.min_risk_multiplier) * normalized


def signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
) -> Optional[Tuple[str, str, float]]:
    candle = candles[index]
    mid = ind["bb_mid"][index]
    upper = ind["bb_upper"][index]
    lower = ind["bb_lower"][index]
    bandwidth = ind["bb_bandwidth"][index]
    rsi_value = ind["rsi"][index]
    adx_value = ind["adx"][index]
    ema_fast_value = ind["ema_fast"][index]
    ema_slow_value = ind["ema_slow"][index]
    drift_value: Optional[float] = None

    if not is_ready([mid, upper, lower, bandwidth, rsi_value, adx_value, ema_fast_value, ema_slow_value]):
        return None

    if cfg.drift_lookback_bars > 0 and index - cfg.drift_lookback_bars >= 0:
        previous_slow = ind["ema_slow"][index - cfg.drift_lookback_bars]
        if previous_slow is not None and candle.close:
            drift_value = (ema_slow_value - previous_slow) / candle.close

    ema_spread = abs((ema_fast_value - ema_slow_value) / candle.close) if candle.close else 0.0
    in_range = (
        adx_value <= cfg.max_adx
        and bandwidth <= cfg.max_bandwidth
        and ema_spread <= cfg.max_ema_spread
    )
    if not in_range:
        return None

    reclaim_buffer = candle.close * cfg.reclaim_buffer_pct
    min_excursion = candle.close * cfg.min_band_excursion_pct
    lower_reclaimed = candle.low <= lower - min_excursion and candle.close >= lower + reclaim_buffer
    upper_reclaimed = candle.high >= upper + min_excursion and candle.close <= upper - reclaim_buffer

    def side_allowed(side: str) -> bool:
        if cfg.side_mode == "long" and side != "long":
            return False
        if cfg.side_mode == "short" and side != "short":
            return False
        if cfg.side_mode == "auto" and drift_value is not None:
            if side == "short" and drift_value > cfg.countertrend_drift_limit_pct:
                return False
            if side == "long" and drift_value < -cfg.countertrend_drift_limit_pct:
                return False
        return True

    drift_text = f" drift={drift_value:.4f}" if drift_value is not None else ""

    if lower_reclaimed and candle.close < mid and rsi_value <= cfg.long_rsi and side_allowed("long"):
        score = signal_quality_score("long", rsi_value, adx_value, bandwidth, ema_spread, drift_value, cfg)
        if score < cfg.min_signal_score:
            return None
        reason = (
            f"lower_band_reclaim rsi={rsi_value:.1f} adx={adx_value:.1f} "
            f"bandwidth={bandwidth:.4f} ema_spread={ema_spread:.4f}{drift_text} score={score:.3f}"
        )
        return "long", reason, score

    if upper_reclaimed and candle.close > mid and rsi_value >= cfg.short_rsi and side_allowed("short"):
        score = signal_quality_score("short", rsi_value, adx_value, bandwidth, ema_spread, drift_value, cfg)
        if score < cfg.min_signal_score:
            return None
        reason = (
            f"upper_band_reclaim rsi={rsi_value:.1f} adx={adx_value:.1f} "
            f"bandwidth={bandwidth:.4f} ema_spread={ema_spread:.4f}{drift_text} score={score:.3f}"
        )
        return "short", reason, score

    return None


def build_pending_entry(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    signal_index: int,
    created_index: int,
    side: str,
    reason: str,
    signal_score: float,
    cfg: StrategyConfig,
) -> Optional[PendingEntry]:
    signal_candle = candles[signal_index]
    atr_value = ind["atr"][signal_index]
    mid = ind["bb_mid"][signal_index]
    upper = ind["bb_upper"][signal_index]
    lower = ind["bb_lower"][signal_index]
    if not is_ready([atr_value, mid, upper, lower]):
        return None

    min_pullback = signal_candle.close * cfg.min_pullback_pct
    pullback = max(atr_value * cfg.entry_pullback_atr, min_pullback)

    if side == "long":
        target = max(lower, signal_candle.close - pullback)
        if target > lower + (upper - lower) * cfg.entry_edge_ratio:
            return None
        stop = target - max(atr_value * cfg.stop_atr, target * cfg.min_stop_pct)
        risk = target - stop
        tp1 = max(mid, target + risk * 0.9)
        tp2 = max(upper - atr_value * 0.15, target + risk * 1.4)
        tp3 = max(target + risk * 2.4, tp2 + atr_value * 0.5)
        reward_risk = (tp2 - target) / risk if risk > 0 else 0.0
    else:
        target = min(upper, signal_candle.close + pullback)
        if target < upper - (upper - lower) * cfg.entry_edge_ratio:
            return None
        stop = target + max(atr_value * cfg.stop_atr, target * cfg.min_stop_pct)
        risk = stop - target
        tp1 = min(mid, target - risk * 0.9)
        tp2 = min(lower + atr_value * 0.15, target - risk * 1.4)
        tp3 = min(target - risk * 2.4, tp2 - atr_value * 0.5)
        reward_risk = (target - tp2) / risk if risk > 0 else 0.0

    if risk <= 0 or reward_risk < cfg.min_reward_risk:
        return None

    return PendingEntry(
        side=side,
        signal_index=signal_index,
        created_index=created_index,
        expires_index=created_index + cfg.max_wait_bars,
        target_price=target,
        stop_price=stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        signal_score=signal_score,
        risk_multiplier=risk_multiplier_from_score(signal_score, cfg),
        signal_reason=reason,
    )


def direction(side: str) -> int:
    return 1 if side == "long" else -1


def adverse_price(price: float, side: str, is_entry: bool, bps: float) -> float:
    if bps <= 0:
        return price
    adjustment = bps / 10_000
    if side == "long":
        return price * (1 + adjustment) if is_entry else price * (1 - adjustment)
    return price * (1 - adjustment) if is_entry else price * (1 + adjustment)


def depth_impact_bps(candle: Candle, notional: float, cfg: StrategyConfig) -> float:
    if cfg.depth_impact_bps <= 0 or notional <= 0:
        return 0.0
    available_quote = max(candle.quote_volume, cfg.min_depth_quote)
    participation = max(notional / available_quote, 0.0)
    return cfg.depth_impact_bps * (participation ** cfg.depth_impact_exponent)


def execution_price(
    raw_price: float,
    side: str,
    is_entry: bool,
    qty: float,
    candle: Candle,
    cfg: StrategyConfig,
    extra_bps: float = 0.0,
) -> float:
    base_bps = cfg.entry_slippage_bps if is_entry else cfg.exit_slippage_bps
    notional = abs(raw_price * qty)
    total_bps = base_bps + depth_impact_bps(candle, notional, cfg) + extra_bps
    return adverse_price(raw_price, side, is_entry, total_bps)


def liquidation_price(entry_price: float, side: str, leverage: float, cfg: StrategyConfig) -> float:
    margin_rate = 1 / leverage
    maintenance = cfg.maintenance_margin_pct + cfg.liquidation_fee_pct
    if side == "long":
        denominator = max(1 - maintenance, 1e-9)
        return entry_price * (1 - margin_rate) / denominator
    denominator = 1 + maintenance
    return entry_price * (1 + margin_rate) / denominator


def exit_position_part(
    position: Position,
    candle: Candle,
    exit_price: float,
    qty: float,
    fee_rate: float,
    reason: str,
) -> Tuple[float, float]:
    qty = min(qty, position.qty)
    side_dir = direction(position.side)
    gross = (exit_price - position.entry_price) * qty * side_dir
    fee = abs(exit_price * qty) * fee_rate
    position.qty -= qty
    position.realized_pnl += gross
    position.fees_paid += fee
    position.exit_notional_sum += exit_price * qty
    position.exited_qty += qty
    position.exit_notes = reason
    return gross, fee


def update_trailing_stop(position: Position, candle: Candle, atr_value: Optional[float], cfg: StrategyConfig) -> None:
    if atr_value is None or not position.tp1_hit:
        return
    if position.side == "long":
        position.best_price = max(position.best_price, candle.high)
        trailed = position.best_price - atr_value * cfg.trail_atr
        breakeven = position.entry_price * (1 + cfg.break_even_buffer_pct)
        position.stop_price = max(position.stop_price, trailed, breakeven)
    else:
        position.best_price = min(position.best_price, candle.low)
        trailed = position.best_price + atr_value * cfg.trail_atr
        breakeven = position.entry_price * (1 - cfg.break_even_buffer_pct)
        position.stop_price = min(position.stop_price, trailed, breakeven)


def close_trade_record(
    position: Position,
    candle: Candle,
    equity_before_trade: float,
    exit_reason: str,
    signal_reason: str,
) -> Trade:
    net = position.realized_pnl - position.fees_paid
    avg_exit_price = (
        position.exit_notional_sum / position.exited_qty
        if position.exited_qty
        else position.entry_price
    )
    return Trade(
        side=position.side,
        entry_time_utc=position.entry_time_utc,
        exit_time_utc=candle.open_time_utc,
        entry_price=position.entry_price,
        avg_exit_price=avg_exit_price,
        initial_qty=position.initial_qty,
        pnl=position.realized_pnl,
        fees=position.fees_paid,
        net_pnl=net,
        return_on_equity_pct=(net / equity_before_trade * 100) if equity_before_trade else 0.0,
        bars_held=0,
        exit_reason=exit_reason,
        signal_reason=signal_reason,
        liquidation_price=position.liquidation_price,
    )


def simulate(candles: Sequence[Candle], cfg: StrategyConfig) -> Dict[str, Any]:
    ind = indicators(candles, cfg)
    equity = cfg.initial_equity
    peak_equity = equity
    max_drawdown = 0.0
    equity_curve: List[Dict[str, float]] = []
    trades: List[Trade] = []
    pending: Optional[PendingEntry] = None
    position: Optional[Position] = None
    position_equity_base = equity
    position_signal_reason = ""
    cooldown_until = 0
    skipped_low_equity = 0

    warmup = max(cfg.bb_period, cfg.rsi_period + 1, cfg.atr_period, cfg.adx_period * 2, cfg.ema_slow) + 2

    for index in range(warmup, len(candles)):
        candle = candles[index]
        atr_value = ind["atr"][index]

        if position is not None:
            exit_reason = ""
            if position.side == "long":
                liquidation_hit = candle.low <= position.liquidation_price
                stop_hit = candle.low <= position.stop_price
                tp1_hit = (not position.tp1_hit) and candle.high >= position.tp1
                tp2_hit = (not position.tp2_hit) and candle.high >= position.tp2
                tp3_hit = candle.high >= position.tp3
                max_hold_hit = index - position.entry_index >= cfg.max_hold_bars

                if liquidation_hit:
                    fill_price = execution_price(position.liquidation_price, position.side, False, position.qty, candle, cfg)
                    exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "liquidation")
                    exit_reason = "liquidation"
                elif stop_hit:
                    fill_price = execution_price(position.stop_price, position.side, False, position.qty, candle, cfg)
                    exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "stop")
                    exit_reason = "stop"
                else:
                    if tp1_hit:
                        exit_qty = position.initial_qty * cfg.tp1_close_ratio
                        fill_price = execution_price(position.tp1, position.side, False, exit_qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, exit_qty, cfg.taker_fee, "tp1")
                        position.tp1_hit = True
                        position.stop_price = max(position.stop_price, position.entry_price * (1 + cfg.break_even_buffer_pct))
                    if tp2_hit and position.qty > 0:
                        exit_qty = position.initial_qty * cfg.tp2_close_ratio
                        fill_price = execution_price(position.tp2, position.side, False, exit_qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, exit_qty, cfg.taker_fee, "tp2")
                        position.tp2_hit = True
                    if tp3_hit and position.qty > 0:
                        fill_price = execution_price(position.tp3, position.side, False, position.qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "tp3")
                        exit_reason = "tp3"
                    elif max_hold_hit and position.qty > 0:
                        fill_price = execution_price(candle.close, position.side, False, position.qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "max_hold")
                        exit_reason = "max_hold"
            else:
                liquidation_hit = candle.high >= position.liquidation_price
                stop_hit = candle.high >= position.stop_price
                tp1_hit = (not position.tp1_hit) and candle.low <= position.tp1
                tp2_hit = (not position.tp2_hit) and candle.low <= position.tp2
                tp3_hit = candle.low <= position.tp3
                max_hold_hit = index - position.entry_index >= cfg.max_hold_bars

                if liquidation_hit:
                    fill_price = execution_price(position.liquidation_price, position.side, False, position.qty, candle, cfg)
                    exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "liquidation")
                    exit_reason = "liquidation"
                elif stop_hit:
                    fill_price = execution_price(position.stop_price, position.side, False, position.qty, candle, cfg)
                    exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "stop")
                    exit_reason = "stop"
                else:
                    if tp1_hit:
                        exit_qty = position.initial_qty * cfg.tp1_close_ratio
                        fill_price = execution_price(position.tp1, position.side, False, exit_qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, exit_qty, cfg.taker_fee, "tp1")
                        position.tp1_hit = True
                        position.stop_price = min(position.stop_price, position.entry_price * (1 - cfg.break_even_buffer_pct))
                    if tp2_hit and position.qty > 0:
                        exit_qty = position.initial_qty * cfg.tp2_close_ratio
                        fill_price = execution_price(position.tp2, position.side, False, exit_qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, exit_qty, cfg.taker_fee, "tp2")
                        position.tp2_hit = True
                    if tp3_hit and position.qty > 0:
                        fill_price = execution_price(position.tp3, position.side, False, position.qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "tp3")
                        exit_reason = "tp3"
                    elif max_hold_hit and position.qty > 0:
                        fill_price = execution_price(candle.close, position.side, False, position.qty, candle, cfg)
                        exit_position_part(position, candle, fill_price, position.qty, cfg.taker_fee, "max_hold")
                        exit_reason = "max_hold"

            if position is not None and position.qty > 0 and not exit_reason:
                update_trailing_stop(position, candle, atr_value, cfg)

            if position is not None and position.qty <= 1e-12:
                trade = close_trade_record(position, candle, position_equity_base, exit_reason or position.exit_notes or "exit", position_signal_reason)
                trade.bars_held = index - position.entry_index
                equity += trade.net_pnl
                trades.append(trade)
                position = None
                pending = None
                cooldown_until = index + cfg.cooldown_bars

        if position is None:
            trading_halted = cfg.max_drawdown_stop_pct > 0 and max_drawdown * 100 >= cfg.max_drawdown_stop_pct
            if trading_halted:
                pending = None

            if not trading_halted and pending is not None and index > pending.expires_index:
                pending = None

            if not trading_halted and pending is not None and index >= pending.created_index:
                can_fill = candle.low <= pending.target_price <= candle.high
                if can_fill:
                    risk_capital = max(equity, 0.0) * cfg.risk_per_trade * pending.risk_multiplier
                    raw_risk_per_unit = abs(pending.target_price - pending.stop_price)
                    rough_qty_by_risk = risk_capital / raw_risk_per_unit if raw_risk_per_unit > 0 else 0.0
                    rough_qty_by_leverage = (max(equity, 0.0) * cfg.leverage) / pending.target_price
                    rough_qty = min(rough_qty_by_risk, rough_qty_by_leverage)
                    entry_price = execution_price(pending.target_price, pending.side, True, rough_qty, candle, cfg)
                    risk_per_unit = abs(entry_price - pending.stop_price)
                    qty_by_risk = risk_capital / risk_per_unit if risk_per_unit > 0 else 0.0
                    qty_by_leverage = (max(equity, 0.0) * cfg.leverage) / entry_price
                    qty = min(qty_by_risk, qty_by_leverage)
                    entry_price = execution_price(pending.target_price, pending.side, True, qty, candle, cfg)
                    if qty <= 0:
                        skipped_low_equity += 1
                        pending = None
                    else:
                        entry_fee = abs(entry_price * qty) * cfg.maker_fee
                        position_equity_base = equity
                        position_signal_reason = pending.signal_reason
                        liq_price = liquidation_price(entry_price, pending.side, cfg.leverage, cfg)
                        position = Position(
                            side=pending.side,
                            entry_index=index,
                            entry_time_utc=candle.open_time_utc,
                            entry_price=entry_price,
                            qty=qty,
                            initial_qty=qty,
                            stop_price=pending.stop_price,
                            tp1=pending.tp1,
                            tp2=pending.tp2,
                            tp3=pending.tp3,
                            liquidation_price=liq_price,
                            best_price=candle.high if pending.side == "long" else candle.low,
                            fees_paid=entry_fee,
                        )
                        pending = None

            if not trading_halted and position is None and pending is None and index >= cooldown_until:
                signal_index = index - 1
                signal = signal_for_index(candles, ind, signal_index, cfg)
                if signal is not None:
                    side, reason, signal_score = signal
                    pending = build_pending_entry(candles, ind, signal_index, index, side, reason, signal_score, cfg)

        marked_equity = equity
        if position is not None:
            side_dir = direction(position.side)
            unrealized = (candle.close - position.entry_price) * position.qty * side_dir
            close_fee_estimate = abs(candle.close * position.qty) * cfg.taker_fee
            marked_equity = equity + position.realized_pnl + unrealized - position.fees_paid - close_fee_estimate
        peak_equity = max(peak_equity, marked_equity)
        drawdown = (peak_equity - marked_equity) / peak_equity if peak_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        equity_curve.append(
            {
                "time_ms": candle.open_time_ms,
                "equity": marked_equity,
                "drawdown_pct": drawdown * 100,
            }
        )

    if position is not None:
        last = candles[-1]
        fill_price = execution_price(last.close, position.side, False, position.qty, last, cfg)
        exit_position_part(position, last, fill_price, position.qty, cfg.taker_fee, "end")
        trade = close_trade_record(position, last, position_equity_base, "end", position_signal_reason)
        trade.bars_held = len(candles) - 1 - position.entry_index
        equity += trade.net_pnl
        trades.append(trade)

    summary = summarize_results(candles, trades, equity_curve, cfg, equity, max_drawdown)
    return {
        "summary": summary,
        "trades": [asdict(trade) for trade in trades],
        "equity_curve": equity_curve,
        "config": asdict(cfg),
    }


def summarize_results(
    candles: Sequence[Candle],
    trades: Sequence[Trade],
    equity_curve: Sequence[Dict[str, float]],
    cfg: StrategyConfig,
    final_equity: float,
    max_drawdown: float,
) -> Dict[str, Any]:
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    liquidation_count = sum(1 for trade in trades if trade.exit_reason == "liquidation")
    max_consecutive_losses = 0
    current_losses = 0
    for trade in trades:
        if trade.net_pnl < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0

    return {
        "symbol_window": {
            "start_utc": candles[0].open_time_utc if candles else None,
            "end_utc": candles[-1].open_time_utc if candles else None,
            "candles": len(candles),
        },
        "initial_equity": cfg.initial_equity,
        "final_equity": final_equity,
        "equity_multiple": final_equity / cfg.initial_equity if cfg.initial_equity else 0.0,
        "total_return_pct": (final_equity / cfg.initial_equity - 1) * 100 if cfg.initial_equity else 0.0,
        "max_drawdown_pct": max_drawdown * 100,
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "average_win": mean(wins) if wins else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
        "best_trade": max((trade.net_pnl for trade in trades), default=0.0),
        "worst_trade": min((trade.net_pnl for trade in trades), default=0.0),
        "max_consecutive_losses": max_consecutive_losses,
        "liquidations": liquidation_count,
        "average_bars_held": mean([trade.bars_held for trade in trades]) if trades else 0.0,
        "total_fees": sum(trade.fees for trade in trades),
        "target_10x_gap_multiple": 10 / (final_equity / cfg.initial_equity) if final_equity > 0 and cfg.initial_equity else None,
        "last_equity_point": equity_curve[-1] if equity_curve else None,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_trades_csv(path: Path, trades: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trades:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)


def print_summary(summary: Dict[str, Any]) -> None:
    window = summary["symbol_window"]
    print("Range swing simulation")
    print(f"Window: {window['start_utc']} -> {window['end_utc']} ({window['candles']} candles)")
    print(f"Initial equity: {summary['initial_equity']:.2f}")
    print(f"Final equity: {summary['final_equity']:.2f}")
    print(f"Equity multiple: {summary['equity_multiple']:.3f}x")
    print(f"Total return: {summary['total_return_pct']:.2f}%")
    print(f"Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"Trades: {summary['trades']}")
    print(f"Win rate: {summary['win_rate_pct']:.2f}%")
    profit_factor = summary["profit_factor"]
    print(f"Profit factor: {profit_factor:.3f}" if profit_factor is not None else "Profit factor: n/a")
    print(f"Best / worst trade: {summary['best_trade']:.2f} / {summary['worst_trade']:.2f}")
    print(f"Max consecutive losses: {summary['max_consecutive_losses']}")
    print(f"Liquidations: {summary['liquidations']}")
    print(f"Total fees: {summary['total_fees']:.2f}")
    gap = summary["target_10x_gap_multiple"]
    if gap is not None:
        print(f"10x target still needs: {gap:.2f}x from this result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a BTCUSDT futures range-swing strategy without placing real orders.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--snapshot", type=Path, help="Use a saved snapshot JSON instead of fetching Binance klines.")
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--leverage", type=float, default=30.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.18)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--taker-fee", type=float, default=0.00045)
    parser.add_argument("--bb-period", type=int, default=36)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--adx-period", type=int, default=14)
    parser.add_argument("--ema-fast", type=int, default=48)
    parser.add_argument("--ema-slow", type=int, default=144)
    parser.add_argument("--max-adx", type=float, default=29.0)
    parser.add_argument("--max-bandwidth", type=float, default=0.040)
    parser.add_argument("--max-ema-spread", type=float, default=0.018)
    parser.add_argument("--side-mode", choices=["auto", "both", "long", "short"], default="auto")
    parser.add_argument("--drift-lookback-bars", type=int, default=144)
    parser.add_argument("--countertrend-drift-limit-pct", type=float, default=0.0015)
    parser.add_argument("--long-rsi", type=float, default=39.0)
    parser.add_argument("--short-rsi", type=float, default=67.0)
    parser.add_argument("--reclaim-buffer-pct", type=float, default=0.00012)
    parser.add_argument("--min-band-excursion-pct", type=float, default=0.0)
    parser.add_argument("--entry-pullback-atr", type=float, default=0.85)
    parser.add_argument("--entry-edge-ratio", type=float, default=0.18)
    parser.add_argument("--min-pullback-pct", type=float, default=0.0008)
    parser.add_argument("--max-wait-bars", type=int, default=2)
    parser.add_argument("--stop-atr", type=float, default=1.15)
    parser.add_argument("--min-stop-pct", type=float, default=0.0055)
    parser.add_argument("--min-reward-risk", type=float, default=1.30)
    parser.add_argument("--tp1-close-ratio", type=float, default=0.35)
    parser.add_argument("--tp2-close-ratio", type=float, default=0.30)
    parser.add_argument("--break-even-buffer-pct", type=float, default=0.00035)
    parser.add_argument("--trail-atr", type=float, default=1.2)
    parser.add_argument("--max-hold-bars", type=int, default=42)
    parser.add_argument("--cooldown-bars", type=int, default=4)
    parser.add_argument("--max-drawdown-stop-pct", type=float, default=50.0)
    parser.add_argument("--high-adx-drift-threshold", type=float, default=26.0)
    parser.add_argument("--min-high-adx-drift-pct", type=float, default=0.0012)
    parser.add_argument("--min-signal-score", type=float, default=0.50)
    parser.add_argument("--adaptive-risk-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-risk-multiplier", type=float, default=0.95)
    parser.add_argument("--max-risk-multiplier", type=float, default=1.10)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.004)
    parser.add_argument("--liquidation-fee-pct", type=float, default=0.001)
    parser.add_argument("--entry-slippage-bps", type=float, default=0.5)
    parser.add_argument("--exit-slippage-bps", type=float, default=1.5)
    parser.add_argument("--depth-impact-bps", type=float, default=8.0)
    parser.add_argument("--depth-impact-exponent", type=float, default=0.5)
    parser.add_argument("--min-depth-quote", type=float, default=2_000_000.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--trades-csv", type=Path)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StrategyConfig:
    if args.initial_equity <= 0:
        raise ValueError("--initial-equity must be > 0")
    if args.leverage <= 0:
        raise ValueError("--leverage must be > 0")
    if not 0 < args.risk_per_trade < 1:
        raise ValueError("--risk-per-trade must be between 0 and 1")
    if args.drift_lookback_bars < 0:
        raise ValueError("--drift-lookback-bars must be >= 0")
    if not 0 <= args.entry_edge_ratio <= 0.5:
        raise ValueError("--entry-edge-ratio must be between 0 and 0.5")
    if args.max_drawdown_stop_pct < 0:
        raise ValueError("--max-drawdown-stop-pct must be >= 0")
    if args.high_adx_drift_threshold < 0 or args.min_high_adx_drift_pct < 0:
        raise ValueError("--high-adx-drift-threshold and --min-high-adx-drift-pct must be >= 0")
    if not 0 <= args.min_signal_score <= 1:
        raise ValueError("--min-signal-score must be between 0 and 1")
    if args.min_risk_multiplier <= 0 or args.max_risk_multiplier <= 0:
        raise ValueError("--min-risk-multiplier and --max-risk-multiplier must be > 0")
    if args.min_risk_multiplier > args.max_risk_multiplier:
        raise ValueError("--min-risk-multiplier must be <= --max-risk-multiplier")
    if args.maintenance_margin_pct < 0 or args.liquidation_fee_pct < 0:
        raise ValueError("--maintenance-margin-pct and --liquidation-fee-pct must be >= 0")
    if args.entry_slippage_bps < 0 or args.exit_slippage_bps < 0:
        raise ValueError("--entry-slippage-bps and --exit-slippage-bps must be >= 0")
    if args.depth_impact_bps < 0 or args.depth_impact_exponent < 0 or args.min_depth_quote <= 0:
        raise ValueError("--depth-impact-bps and --depth-impact-exponent must be >= 0; --min-depth-quote must be > 0")
    return StrategyConfig(
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
        taker_fee=args.taker_fee,
        maker_fee=args.maker_fee,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        rsi_period=args.rsi_period,
        atr_period=args.atr_period,
        adx_period=args.adx_period,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        max_adx=args.max_adx,
        max_bandwidth=args.max_bandwidth,
        max_ema_spread=args.max_ema_spread,
        side_mode=args.side_mode,
        drift_lookback_bars=args.drift_lookback_bars,
        countertrend_drift_limit_pct=args.countertrend_drift_limit_pct,
        long_rsi=args.long_rsi,
        short_rsi=args.short_rsi,
        reclaim_buffer_pct=args.reclaim_buffer_pct,
        min_band_excursion_pct=args.min_band_excursion_pct,
        entry_pullback_atr=args.entry_pullback_atr,
        entry_edge_ratio=args.entry_edge_ratio,
        min_pullback_pct=args.min_pullback_pct,
        max_wait_bars=args.max_wait_bars,
        stop_atr=args.stop_atr,
        min_stop_pct=args.min_stop_pct,
        min_reward_risk=args.min_reward_risk,
        tp1_close_ratio=args.tp1_close_ratio,
        tp2_close_ratio=args.tp2_close_ratio,
        break_even_buffer_pct=args.break_even_buffer_pct,
        trail_atr=args.trail_atr,
        max_hold_bars=args.max_hold_bars,
        cooldown_bars=args.cooldown_bars,
        max_drawdown_stop_pct=args.max_drawdown_stop_pct,
        high_adx_drift_threshold=args.high_adx_drift_threshold,
        min_high_adx_drift_pct=args.min_high_adx_drift_pct,
        min_signal_score=args.min_signal_score,
        adaptive_risk_enabled=args.adaptive_risk_enabled,
        min_risk_multiplier=args.min_risk_multiplier,
        max_risk_multiplier=args.max_risk_multiplier,
        maintenance_margin_pct=args.maintenance_margin_pct,
        liquidation_fee_pct=args.liquidation_fee_pct,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        depth_impact_bps=args.depth_impact_bps,
        depth_impact_exponent=args.depth_impact_exponent,
        min_depth_quote=args.min_depth_quote,
    )


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)

    if args.snapshot:
        candles = load_candles_from_snapshot(args.snapshot)
        source = str(args.snapshot)
    else:
        symbol = normalize_symbol(args.symbol)
        candles = fetch_futures_klines(symbol, args.interval, args.days)
        source = f"binance_futures:{symbol}:{args.interval}:{args.days}d"

    if len(candles) < max(cfg.ema_slow, cfg.bb_period, cfg.adx_period * 2) + 10:
        raise ValueError(f"Not enough candles for the configured indicators: {len(candles)}")

    result = simulate(candles, cfg)
    result["source"] = source
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    print_summary(result["summary"])

    if args.output_json:
        save_json(args.output_json, result)
        print(f"Saved JSON: {args.output_json}")
    if args.trades_csv:
        save_trades_csv(args.trades_csv, result["trades"])
        print(f"Saved trades CSV: {args.trades_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
