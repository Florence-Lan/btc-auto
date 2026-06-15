#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
MINUTES_PER_DAY = 24 * 60
MS_PER_DAY = 24 * 60 * 60 * 1000


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
class MarketContext:
    periods: Dict[str, Dict[str, Dict[str, List[float]]]]
    funding_times: List[int]
    funding_rates: List[float]


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
    confirm_timeframes: Tuple[str, ...]
    confirm_drift_lookback_bars: int
    confirm_countertrend_drift_limit_pct: float
    confirm_countertrend_ema_spread_pct: float
    confirm_countertrend_min_adx: float
    confirm_min_space_pct: float
    trend_stretch_filter_timeframes: Tuple[str, ...]
    max_trend_stretch_ema_spread: float
    min_trend_stretch_adx: float
    adaptive_risk_enabled: bool
    min_risk_multiplier: float
    max_risk_multiplier: float
    market_context_enabled: bool
    market_context_periods: Tuple[str, ...]
    min_market_context_score: float
    market_context_score_weight: float
    maintenance_margin_pct: float
    liquidation_fee_pct: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    depth_impact_bps: float
    depth_impact_exponent: float
    min_depth_quote: float
    strategy_modes: Tuple[str, ...] = ("range", "trend")
    trend_confirm_timeframes: Tuple[str, ...] = ("15m", "1h", "4h")
    trend_min_signal_score: float = 0.75
    trend_min_adx: float = 30.0
    trend_min_ema_spread: float = 0.0030
    trend_min_drift_pct: float = 0.0015
    trend_pullback_lookback_bars: int = 8
    trend_pullback_atr: float = 0.80
    trend_entry_pullback_atr: float = 0.15
    trend_stop_atr: float = 1.35
    trend_min_stop_pct: float = 0.0035
    trend_tp1_rr: float = 1.0
    trend_tp2_rr: float = 2.0
    trend_tp3_rr: float = 3.2
    trend_min_reward_risk: float = 1.60
    trend_short_rsi_min: float = 22.0
    trend_short_rsi_max: float = 58.0
    trend_long_rsi_min: float = 42.0
    trend_long_rsi_max: float = 78.0
    event_min_signal_score: float = 0.95
    event_risk_multiplier: float = 0.20
    event_core_score_penalty: float = 0.10
    event_regime_max_adx: float = 32.0
    event_regime_max_aligned_ema_spread: float = 0.006
    event_entry_pullback_atr: float = 0.12
    event_stop_atr: float = 1.20
    event_stop_buffer_atr: float = 0.20
    event_min_stop_pct: float = 0.0030
    event_tp1_rr: float = 1.0
    event_tp2_rr: float = 1.8
    event_tp3_rr: float = 2.8
    event_min_reward_risk: float = 1.35
    exhaustion_min_adx: float = 32.0
    exhaustion_min_ema_spread: float = 0.012
    exhaustion_volume_ratio: float = 1.15
    fake_breakout_lookback_bars: int = 48
    fake_breakout_buffer_atr: float = 0.12
    squeeze_lookback_bars: int = 48
    squeeze_max_bandwidth: float = 0.012
    squeeze_volume_ratio: float = 1.25
    shock_atr_multiple: float = 3.0
    shock_volume_ratio: float = 1.8
    sweep_lookback_bars: int = 36
    sweep_wick_ratio: float = 0.55
    sweep_volume_ratio: float = 1.20
    wide_failure_lookback_bars: int = 72
    wide_failure_min_range_atr: float = 4.0


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


def minimum_history_days(base_interval: str, cfg: StrategyConfig, buffer_days: float = 1.0) -> float:
    intervals = tuple(dict.fromkeys((base_interval, *cfg.confirm_timeframes, *cfg.trend_confirm_timeframes)))
    required_days = 0.0
    indicator_bars = max(cfg.bb_period, cfg.rsi_period + 1, cfg.atr_period, cfg.adx_period * 2, cfg.ema_slow)
    for interval in intervals:
        bars = indicator_bars
        if interval in cfg.confirm_timeframes or interval in cfg.trend_confirm_timeframes:
            bars += cfg.confirm_drift_lookback_bars + 2
        interval_minutes = interval_to_ms(interval) / 60_000
        required_days = max(required_days, bars * interval_minutes / MINUTES_PER_DAY)
    return required_days + buffer_days


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


def fetch_json_with_retries(session: requests.Session, path: str, params: Dict[str, Any]) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            response = session.get(
                f"{BINANCE_FUTURES_BASE_URL}{path}",
                params=params,
                timeout=45,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch Binance endpoint {path}: {last_error}") from last_error


def fetch_futures_data_history(
    session: requests.Session,
    path: str,
    base_params: Dict[str, Any],
    period: str,
    days: float,
    timestamp_field: str = "timestamp",
) -> List[Dict[str, Any]]:
    interval_ms = interval_to_ms(period)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - int(min(days, 29.0) * MS_PER_DAY)
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    cursor = start_ms
    while cursor < now_ms:
        params = {
            **base_params,
            "period": period,
            "startTime": cursor,
            "endTime": now_ms,
            "limit": 500,
        }
        batch = fetch_json_with_retries(session, path, params)
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            timestamp = int(item[timestamp_field])
            if timestamp not in seen:
                rows.append(item)
                seen.add(timestamp)
        next_cursor = int(batch[-1][timestamp_field]) + interval_ms
        if next_cursor <= cursor or len(batch) < 500:
            break
        cursor = next_cursor
        time.sleep(0.04)
    return sorted(rows, key=lambda item: int(item[timestamp_field]))


def series_from_rows(rows: Sequence[Dict[str, Any]], value_field: str, timestamp_field: str = "timestamp") -> Dict[str, List[float]]:
    timestamps: List[float] = []
    values: List[float] = []
    for row in rows:
        value = row.get(value_field)
        timestamp = row.get(timestamp_field)
        if value is None or timestamp is None:
            continue
        try:
            timestamps.append(float(timestamp))
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return {"timestamps": timestamps, "values": values}


def fetch_premium_index_klines(session: requests.Session, symbol: str, period: str, days: float) -> List[Dict[str, Any]]:
    interval_ms = interval_to_ms(period)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - int(min(days, 29.0) * MS_PER_DAY)
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    cursor = start_ms
    while cursor < now_ms:
        params = {
            "symbol": symbol,
            "interval": period,
            "startTime": cursor,
            "endTime": now_ms,
            "limit": 1500,
        }
        batch = fetch_json_with_retries(session, "/fapi/v1/premiumIndexKlines", params)
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            timestamp = int(item[0])
            if timestamp not in seen:
                rows.append({"timestamp": timestamp, "premium": item[4]})
                seen.add(timestamp)
        next_cursor = int(batch[-1][0]) + interval_ms
        if next_cursor <= cursor or len(batch) < 1500:
            break
        cursor = next_cursor
        time.sleep(0.04)
    return sorted(rows, key=lambda item: int(item["timestamp"]))


def fetch_market_context(symbol: str, days: float, periods: Sequence[str]) -> MarketContext:
    periods = tuple(dict.fromkeys(periods))
    period_payload: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-auto-market-context/1.0"})
        for period in periods:
            metrics: Dict[str, Dict[str, List[float]]] = {}
            metrics["open_interest_value"] = series_from_rows(
                fetch_futures_data_history(
                    session,
                    "/futures/data/openInterestHist",
                    {"symbol": symbol},
                    period,
                    days,
                ),
                "sumOpenInterestValue",
            )
            metrics["taker_buy_sell_ratio"] = series_from_rows(
                fetch_futures_data_history(
                    session,
                    "/futures/data/takerlongshortRatio",
                    {"symbol": symbol},
                    period,
                    days,
                ),
                "buySellRatio",
            )
            metrics["top_position_ratio"] = series_from_rows(
                fetch_futures_data_history(
                    session,
                    "/futures/data/topLongShortPositionRatio",
                    {"symbol": symbol},
                    period,
                    days,
                ),
                "longShortRatio",
            )
            metrics["top_account_ratio"] = series_from_rows(
                fetch_futures_data_history(
                    session,
                    "/futures/data/topLongShortAccountRatio",
                    {"symbol": symbol},
                    period,
                    days,
                ),
                "longShortRatio",
            )
            metrics["global_account_ratio"] = series_from_rows(
                fetch_futures_data_history(
                    session,
                    "/futures/data/globalLongShortAccountRatio",
                    {"symbol": symbol},
                    period,
                    days,
                ),
                "longShortRatio",
            )
            metrics["premium_index"] = series_from_rows(
                fetch_premium_index_klines(session, symbol, period, days),
                "premium",
            )
            period_payload[period] = metrics

        funding_rows = fetch_json_with_retries(
            session,
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1000},
        )
    funding_times: List[int] = []
    funding_rates: List[float] = []
    if isinstance(funding_rows, list):
        cutoff = int(datetime.now(timezone.utc).timestamp() * 1000) - int(min(days, 29.0) * MS_PER_DAY)
        for row in funding_rows:
            try:
                timestamp = int(row["fundingTime"])
                if timestamp >= cutoff:
                    funding_times.append(timestamp)
                    funding_rates.append(float(row["fundingRate"]))
            except (KeyError, TypeError, ValueError):
                continue
    return MarketContext(periods=period_payload, funding_times=funding_times, funding_rates=funding_rates)


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


def aggregate_candles(candles: Sequence[Candle], interval: str) -> List[Candle]:
    if not candles:
        return []
    target_ms = interval_to_ms(interval)
    if len(candles) >= 2:
        base_ms = candles[1].open_time_ms - candles[0].open_time_ms
    else:
        base_ms = target_ms
    expected_count = max(int(target_ms / base_ms), 1) if base_ms > 0 and target_ms % base_ms == 0 else 1

    grouped: Dict[int, List[Candle]] = {}
    for candle in candles:
        bucket_start = (candle.open_time_ms // target_ms) * target_ms
        grouped.setdefault(bucket_start, []).append(candle)

    aggregated: List[Candle] = []
    for bucket_start in sorted(grouped):
        bucket = sorted(grouped[bucket_start], key=lambda item: item.open_time_ms)
        if len(bucket) < expected_count or bucket[0].open_time_ms != bucket_start:
            continue
        open_time_ms = bucket_start
        close_time_ms = bucket_start + target_ms - 1
        aggregated.append(
            Candle(
                open_time_ms=open_time_ms,
                open_time_utc=iso_utc_from_ms(open_time_ms),
                open=bucket[0].open,
                high=max(item.high for item in bucket),
                low=min(item.low for item in bucket),
                close=bucket[-1].close,
                volume=sum(item.volume for item in bucket),
                quote_volume=sum(item.quote_volume for item in bucket),
                close_time_ms=close_time_ms,
            )
        )
    return aggregated


def build_higher_timeframe_context(
    candles: Sequence[Candle],
    cfg: StrategyConfig,
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Dict[str, Any]] = {}
    for timeframe in dict.fromkeys((*cfg.confirm_timeframes, *cfg.trend_confirm_timeframes)):
        higher_candles = aggregate_candles(candles, timeframe)
        if not higher_candles:
            continue
        context[timeframe] = {
            "candles": higher_candles,
            "close_times": [candle.close_time_ms for candle in higher_candles],
            "indicators": indicators(higher_candles, cfg),
        }
    return context


def higher_timeframe_allows(
    side: str,
    signal_candle: Candle,
    higher_context: Dict[str, Dict[str, Any]],
    cfg: StrategyConfig,
) -> Tuple[bool, str]:
    if not cfg.confirm_timeframes:
        return True, ""

    notes: List[str] = []
    for timeframe in cfg.confirm_timeframes:
        context = higher_context.get(timeframe)
        if context is None:
            return False, f"{timeframe}_missing"

        close_times: Sequence[int] = context["close_times"]
        higher_index = bisect_right(close_times, signal_candle.close_time_ms) - 1
        if higher_index < 0:
            return False, f"{timeframe}_not_closed"

        higher_candles: Sequence[Candle] = context["candles"]
        higher_ind: Dict[str, List[Optional[float]]] = context["indicators"]
        candle = higher_candles[higher_index]
        ema_fast_value = higher_ind["ema_fast"][higher_index]
        ema_slow_value = higher_ind["ema_slow"][higher_index]
        adx_value = higher_ind["adx"][higher_index]
        upper = higher_ind["bb_upper"][higher_index]
        lower = higher_ind["bb_lower"][higher_index]
        if not is_ready([ema_fast_value, ema_slow_value, adx_value, upper, lower]):
            return False, f"{timeframe}_indicators_not_ready"

        drift_value = 0.0
        drift_index = higher_index - cfg.confirm_drift_lookback_bars
        if cfg.confirm_drift_lookback_bars > 0 and drift_index >= 0:
            previous_slow = higher_ind["ema_slow"][drift_index]
            if previous_slow is None:
                return False, f"{timeframe}_drift_not_ready"
            drift_value = (ema_slow_value - previous_slow) / candle.close if candle.close else 0.0

        ema_spread = (ema_fast_value - ema_slow_value) / candle.close if candle.close else 0.0
        if side == "long":
            countertrend = (
                ema_spread <= -cfg.confirm_countertrend_ema_spread_pct
                and drift_value <= -cfg.confirm_countertrend_drift_limit_pct
                and adx_value >= cfg.confirm_countertrend_min_adx
            )
            space = (upper - signal_candle.close) / signal_candle.close if signal_candle.close else 0.0
        else:
            countertrend = (
                ema_spread >= cfg.confirm_countertrend_ema_spread_pct
                and drift_value >= cfg.confirm_countertrend_drift_limit_pct
                and adx_value >= cfg.confirm_countertrend_min_adx
            )
            space = (signal_candle.close - lower) / signal_candle.close if signal_candle.close else 0.0

        if countertrend:
            return False, (
                f"{timeframe}_countertrend ema_spread={ema_spread:.4f} "
                f"drift={drift_value:.4f} adx={adx_value:.1f}"
            )
        if (
            timeframe in cfg.trend_stretch_filter_timeframes
            and abs(ema_spread) >= cfg.max_trend_stretch_ema_spread
            and adx_value >= cfg.min_trend_stretch_adx
        ):
            return False, (
                f"{timeframe}_trend_stretch ema_spread={ema_spread:.4f} "
                f"adx={adx_value:.1f}"
            )
        if space < cfg.confirm_min_space_pct:
            return False, f"{timeframe}_space={space:.4f}"

        notes.append(
            f"{timeframe}:ema_spread={ema_spread:.4f} drift={drift_value:.4f} "
            f"adx={adx_value:.1f} space={space:.4f}"
        )

    return True, " confirm=" + ";".join(notes)


def is_ready(values: Iterable[Optional[float]]) -> bool:
    return all(value is not None for value in values)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def side_aligned_drift(side: str, drift_value: Optional[float]) -> float:
    if drift_value is None:
        return 0.0
    return drift_value if side == "long" else -drift_value


def series_value_at(series: Dict[str, List[float]], timestamp_ms: int) -> Optional[float]:
    timestamps = series.get("timestamps", [])
    values = series.get("values", [])
    index = bisect_right(timestamps, float(timestamp_ms)) - 1
    if index < 0 or index >= len(values):
        return None
    return values[index]


def series_change_pct_at(series: Dict[str, List[float]], timestamp_ms: int, lookback: int = 6) -> Optional[float]:
    timestamps = series.get("timestamps", [])
    values = series.get("values", [])
    index = bisect_right(timestamps, float(timestamp_ms)) - 1
    previous_index = index - lookback
    if previous_index < 0 or index >= len(values):
        return None
    previous = values[previous_index]
    if previous == 0:
        return None
    return (values[index] / previous - 1) * 100


def funding_rate_at(market_context: MarketContext, timestamp_ms: int) -> Optional[float]:
    index = bisect_right(market_context.funding_times, timestamp_ms) - 1
    if index < 0 or index >= len(market_context.funding_rates):
        return None
    return market_context.funding_rates[index]


def aligned_ratio_delta(side: str, ratio: Optional[float]) -> float:
    if ratio is None:
        return 0.0
    return ratio - 1 if side == "long" else 1 - ratio


def market_context_for_signal(
    side: str,
    candle: Candle,
    market_context: Optional[MarketContext],
    cfg: StrategyConfig,
) -> Tuple[bool, float, float, str]:
    if not cfg.market_context_enabled or market_context is None:
        return True, 0.0, 1.0, ""

    observations = 0
    score = 0.50
    notes: List[str] = []

    for period in cfg.market_context_periods:
        metrics = market_context.periods.get(period)
        if not metrics:
            continue
        taker = series_value_at(metrics.get("taker_buy_sell_ratio", {}), candle.close_time_ms)
        top_position = series_value_at(metrics.get("top_position_ratio", {}), candle.close_time_ms)
        top_account = series_value_at(metrics.get("top_account_ratio", {}), candle.close_time_ms)
        global_account = series_value_at(metrics.get("global_account_ratio", {}), candle.close_time_ms)
        oi_change = series_change_pct_at(metrics.get("open_interest_value", {}), candle.close_time_ms, 6)
        premium = series_value_at(metrics.get("premium_index", {}), candle.close_time_ms)

        period_score = 0.0
        period_observations = 0
        for ratio, scale, weight in (
            (taker, 0.35, 0.24),
            (top_position, 0.20, 0.20),
            (top_account, 0.35, 0.12),
            (global_account, 0.35, 0.08),
        ):
            if ratio is not None:
                period_score += weight * clamp(aligned_ratio_delta(side, ratio) / scale, -1.0, 1.0)
                period_observations += 1

        if oi_change is not None:
            flow_delta = aligned_ratio_delta(side, taker)
            if oi_change > 0.15:
                period_score += 0.12 if flow_delta > 0 else -0.08
            elif oi_change < -0.15 and flow_delta < 0:
                period_score -= 0.06
            period_observations += 1

        if premium is not None:
            aligned_premium = -premium if side == "long" else premium
            period_score += 0.08 * clamp(aligned_premium / 0.0015, -1.0, 1.0)
            period_observations += 1

        if period_observations:
            score += period_score / max(len(cfg.market_context_periods), 1)
            observations += period_observations
            note_parts = []
            if taker is not None:
                note_parts.append(f"taker={taker:.3f}")
            if oi_change is not None:
                note_parts.append(f"oi6={oi_change:.2f}%")
            if top_position is not None:
                note_parts.append(f"topPos={top_position:.3f}")
            if premium is not None:
                note_parts.append(f"premium={premium:.5f}")
            if note_parts:
                notes.append(f"{period}:" + ",".join(note_parts))

    funding = funding_rate_at(market_context, candle.close_time_ms)
    if funding is not None:
        aligned_funding = -funding if side == "long" else funding
        score += 0.08 * clamp(aligned_funding / 0.00025, -1.0, 1.0)
        observations += 1
        notes.append(f"funding={funding:.5f}")

    if observations == 0:
        return True, 0.0, 1.0, " market=missing"

    score = clamp(score, 0.0, 1.0)
    allowed = score >= cfg.min_market_context_score
    score_delta = (score - 0.50) * cfg.market_context_score_weight
    risk_multiplier = clamp(0.75 + score * 0.55, 0.70, 1.20)
    return allowed, score_delta, risk_multiplier, f" market_score={score:.3f} " + ";".join(notes)


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


def strategy_enabled(name: str, cfg: StrategyConfig) -> bool:
    return "all" in cfg.strategy_modes or name in cfg.strategy_modes


def side_allowed_by_mode(side: str, cfg: StrategyConfig, drift_value: Optional[float] = None) -> bool:
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


def optional_mean(values: Sequence[Optional[float]], end_index: int, lookback: int) -> Optional[float]:
    start = max(0, end_index - lookback + 1)
    window = [value for value in values[start : end_index + 1] if value is not None]
    return mean(window) if window else None


def prior_range(candles: Sequence[Candle], index: int, lookback: int) -> Optional[Tuple[float, float]]:
    start = max(0, index - lookback)
    window = candles[start:index]
    if not window:
        return None
    return max(item.high for item in window), min(item.low for item in window)


def candle_range(candle: Candle) -> float:
    return max(candle.high - candle.low, 0.0)


def candle_body(candle: Candle) -> float:
    return abs(candle.close - candle.open)


def upper_wick_ratio(candle: Candle) -> float:
    span = candle_range(candle)
    return 0.0 if span <= 0 else (candle.high - max(candle.open, candle.close)) / span


def lower_wick_ratio(candle: Candle) -> float:
    span = candle_range(candle)
    return 0.0 if span <= 0 else (min(candle.open, candle.close) - candle.low) / span


def close_position_ratio(candle: Candle) -> float:
    span = candle_range(candle)
    return 0.5 if span <= 0 else (candle.close - candle.low) / span


def volume_ratio(candles: Sequence[Candle], index: int, lookback: int = 48) -> float:
    start = max(0, index - lookback)
    window = [item.quote_volume for item in candles[start:index] if item.quote_volume > 0]
    if not window:
        return 1.0
    return candles[index].quote_volume / mean(window)


def base_event_score(
    *,
    strength: float,
    rejection: float,
    volume: float,
    context: float,
) -> float:
    return clamp(
        0.46
        + 0.22 * clamp(strength, 0.0, 1.0)
        + 0.16 * clamp(rejection, 0.0, 1.0)
        + 0.10 * clamp(volume, 0.0, 1.0)
        + 0.06 * clamp(context, 0.0, 1.0),
        0.0,
        1.0,
    )


def apply_market_score(
    side: str,
    candle: Candle,
    base_score: float,
    market_context: Optional[MarketContext],
    cfg: StrategyConfig,
) -> Optional[Tuple[float, str]]:
    market_allowed, market_delta, _market_risk, market_text = market_context_for_signal(
        side,
        candle,
        market_context,
        cfg,
    )
    if not market_allowed:
        return None
    if market_text == " market=missing":
        return None
    return clamp(base_score + market_delta, 0.0, 1.0), market_text


def local_ema_spread(ind: Dict[str, List[Optional[float]]], candles: Sequence[Candle], index: int) -> Optional[float]:
    ema_fast_value = ind["ema_fast"][index]
    ema_slow_value = ind["ema_slow"][index]
    close = candles[index].close
    if not is_ready([ema_fast_value, ema_slow_value]) or not close:
        return None
    return (ema_fast_value - ema_slow_value) / close


def reversal_regime_allows(side: str, candles: Sequence[Candle], ind: Dict[str, List[Optional[float]]], index: int, cfg: StrategyConfig) -> bool:
    adx_value = ind["adx"][index]
    ema_spread = local_ema_spread(ind, candles, index)
    if not is_ready([adx_value, ema_spread]):
        return False
    if side == "short" and ema_spread > cfg.event_regime_max_aligned_ema_spread:
        return False
    if side == "long" and ema_spread < -cfg.event_regime_max_aligned_ema_spread:
        return False
    if adx_value < cfg.event_regime_max_adx:
        return True
    if side == "short":
        return ema_spread <= cfg.event_regime_max_aligned_ema_spread
    return ema_spread >= -cfg.event_regime_max_aligned_ema_spread


def event_priority(reason: str) -> int:
    if reason.startswith("exhaustion_"):
        return 60
    if reason.startswith("fake_breakout_"):
        return 50
    if reason.startswith("sweep_"):
        return 40
    if reason.startswith("wide_failure_"):
        return 30
    if reason.startswith("squeeze_"):
        return 20
    if reason.startswith("shock_"):
        return 10
    return 0


def is_reversal_event(reason: str) -> bool:
    return reason.startswith(
        (
            "exhaustion_",
            "fake_breakout_",
            "sweep_",
            "wide_failure_",
        )
    )


def adjusted_candidate_score(candidate: Tuple[str, str, float], cfg: StrategyConfig) -> float:
    reason = candidate[1]
    if reason.startswith(
        (
            "exhaustion_",
            "fake_breakout_",
            "squeeze_",
            "shock_",
            "sweep_",
            "wide_failure_",
        )
    ):
        return candidate[2] - cfg.event_core_score_penalty
    return candidate[2]


def range_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    higher_context: Optional[Dict[str, Dict[str, Any]]] = None,
    market_context: Optional[MarketContext] = None,
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

    drift_text = f" drift={drift_value:.4f}" if drift_value is not None else ""

    if lower_reclaimed and candle.close < mid and rsi_value <= cfg.long_rsi and side_allowed_by_mode("long", cfg, drift_value):
        base_score = signal_quality_score("long", rsi_value, adx_value, bandwidth, ema_spread, drift_value, cfg)
        market_allowed, market_delta, _market_risk, market_text = market_context_for_signal(
            "long",
            candle,
            market_context,
            cfg,
        )
        if not market_allowed:
            return None
        score = clamp(base_score + market_delta, 0.0, 1.0)
        if score < cfg.min_signal_score:
            return None
        confirm_allowed, confirm_text = higher_timeframe_allows("long", candle, higher_context or {}, cfg)
        if not confirm_allowed:
            return None
        reason = (
            f"range_lower_band_reclaim rsi={rsi_value:.1f} adx={adx_value:.1f} "
            f"bandwidth={bandwidth:.4f} ema_spread={ema_spread:.4f}{drift_text} "
            f"base_score={base_score:.3f} score={score:.3f}{market_text}{confirm_text}"
        )
        return "long", reason, score

    if upper_reclaimed and candle.close > mid and rsi_value >= cfg.short_rsi and side_allowed_by_mode("short", cfg, drift_value):
        base_score = signal_quality_score("short", rsi_value, adx_value, bandwidth, ema_spread, drift_value, cfg)
        market_allowed, market_delta, _market_risk, market_text = market_context_for_signal(
            "short",
            candle,
            market_context,
            cfg,
        )
        if not market_allowed:
            return None
        score = clamp(base_score + market_delta, 0.0, 1.0)
        if score < cfg.min_signal_score:
            return None
        confirm_allowed, confirm_text = higher_timeframe_allows("short", candle, higher_context or {}, cfg)
        if not confirm_allowed:
            return None
        reason = (
            f"range_upper_band_reclaim rsi={rsi_value:.1f} adx={adx_value:.1f} "
            f"bandwidth={bandwidth:.4f} ema_spread={ema_spread:.4f}{drift_text} "
            f"base_score={base_score:.3f} score={score:.3f}{market_text}{confirm_text}"
        )
        return "short", reason, score

    return None


def higher_timeframe_trend_allows(
    side: str,
    signal_candle: Candle,
    higher_context: Dict[str, Dict[str, Any]],
    cfg: StrategyConfig,
) -> Tuple[bool, str]:
    if not cfg.trend_confirm_timeframes:
        return True, ""

    notes: List[str] = []
    for timeframe in cfg.trend_confirm_timeframes:
        context = higher_context.get(timeframe)
        if context is None:
            return False, f"{timeframe}_missing"

        close_times: Sequence[int] = context["close_times"]
        higher_index = bisect_right(close_times, signal_candle.close_time_ms) - 1
        if higher_index < 0:
            return False, f"{timeframe}_not_closed"

        higher_candles: Sequence[Candle] = context["candles"]
        higher_ind: Dict[str, List[Optional[float]]] = context["indicators"]
        candle = higher_candles[higher_index]
        ema_fast_value = higher_ind["ema_fast"][higher_index]
        ema_slow_value = higher_ind["ema_slow"][higher_index]
        adx_value = higher_ind["adx"][higher_index]
        if not is_ready([ema_fast_value, ema_slow_value, adx_value]):
            return False, f"{timeframe}_indicators_not_ready"

        drift_value = 0.0
        drift_index = higher_index - cfg.confirm_drift_lookback_bars
        if cfg.confirm_drift_lookback_bars > 0 and drift_index >= 0:
            previous_slow = higher_ind["ema_slow"][drift_index]
            if previous_slow is None:
                return False, f"{timeframe}_drift_not_ready"
            drift_value = (ema_slow_value - previous_slow) / candle.close if candle.close else 0.0

        ema_spread = (ema_fast_value - ema_slow_value) / candle.close if candle.close else 0.0
        aligned_spread = ema_spread if side == "long" else -ema_spread
        aligned_drift = drift_value if side == "long" else -drift_value
        if aligned_spread < cfg.trend_min_ema_spread:
            return False, f"{timeframe}_spread={ema_spread:.4f}"
        if aligned_drift < cfg.trend_min_drift_pct:
            return False, f"{timeframe}_drift={drift_value:.4f}"
        if adx_value < cfg.trend_min_adx:
            return False, f"{timeframe}_adx={adx_value:.1f}"

        notes.append(
            f"{timeframe}:ema_spread={ema_spread:.4f} drift={drift_value:.4f} adx={adx_value:.1f}"
        )

    return True, " trend_confirm=" + ";".join(notes)


def trend_signal_score(
    side: str,
    rsi_value: float,
    adx_value: float,
    ema_spread: float,
    drift_value: float,
    pullback_distance_atr: float,
    cfg: StrategyConfig,
) -> float:
    aligned_spread = ema_spread if side == "long" else -ema_spread
    aligned_drift = drift_value if side == "long" else -drift_value
    trend_strength = clamp((adx_value - cfg.trend_min_adx) / 22, 0.0, 1.0)
    spread_strength = clamp(aligned_spread / 0.010, 0.0, 1.0)
    drift_strength = clamp(aligned_drift / 0.010, 0.0, 1.0)
    pullback_quality = clamp(1.0 - abs(pullback_distance_atr - 0.35) / 1.25, 0.0, 1.0)
    if side == "short":
        rsi_quality = clamp((cfg.trend_short_rsi_max - rsi_value) / 22, 0.0, 1.0)
    else:
        rsi_quality = clamp((rsi_value - cfg.trend_long_rsi_min) / 22, 0.0, 1.0)
    return clamp(
        0.38
        + 0.18 * trend_strength
        + 0.18 * spread_strength
        + 0.14 * drift_strength
        + 0.08 * pullback_quality
        + 0.04 * rsi_quality,
        0.0,
        1.0,
    )


def trend_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    higher_context: Optional[Dict[str, Dict[str, Any]]] = None,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    candle = candles[index]
    atr_value = ind["atr"][index]
    rsi_value = ind["rsi"][index]
    adx_value = ind["adx"][index]
    ema_fast_value = ind["ema_fast"][index]
    ema_slow_value = ind["ema_slow"][index]
    if not is_ready([atr_value, rsi_value, adx_value, ema_fast_value, ema_slow_value]):
        return None

    drift_value = 0.0
    if cfg.drift_lookback_bars > 0 and index - cfg.drift_lookback_bars >= 0:
        previous_slow = ind["ema_slow"][index - cfg.drift_lookback_bars]
        if previous_slow is None:
            return None
        drift_value = (ema_slow_value - previous_slow) / candle.close if candle.close else 0.0

    ema_spread = (ema_fast_value - ema_slow_value) / candle.close if candle.close else 0.0
    if adx_value < cfg.trend_min_adx:
        return None

    lookback_start = max(0, index - cfg.trend_pullback_lookback_bars + 1)
    recent = candles[lookback_start : index + 1]
    candidates: List[Tuple[str, str, float]] = []

    short_pullback = max(item.high for item in recent) >= ema_fast_value - atr_value * cfg.trend_pullback_atr
    short_trigger = candle.close < ema_fast_value and candle.close < candle.open
    short_rsi_ok = cfg.trend_short_rsi_min <= rsi_value <= cfg.trend_short_rsi_max
    short_aligned = -ema_spread >= cfg.trend_min_ema_spread and -drift_value >= cfg.trend_min_drift_pct
    if short_aligned and short_pullback and short_trigger and short_rsi_ok and side_allowed_by_mode("short", cfg, None):
        pullback_distance_atr = abs(max(item.high for item in recent) - ema_fast_value) / atr_value if atr_value else 0.0
        base_score = trend_signal_score("short", rsi_value, adx_value, ema_spread, drift_value, pullback_distance_atr, cfg)
        market_allowed, market_delta, _market_risk, market_text = market_context_for_signal("short", candle, market_context, cfg)
        if market_allowed:
            score = clamp(base_score + market_delta, 0.0, 1.0)
            if score >= cfg.trend_min_signal_score:
                confirm_allowed, confirm_text = higher_timeframe_trend_allows("short", candle, higher_context or {}, cfg)
                if confirm_allowed:
                    candidates.append(
                        (
                            "short",
                            (
                                f"trend_short_pullback rsi={rsi_value:.1f} adx={adx_value:.1f} "
                                f"ema_spread={ema_spread:.4f} drift={drift_value:.4f} "
                                f"pullback_atr={pullback_distance_atr:.2f} base_score={base_score:.3f} "
                                f"score={score:.3f}{market_text}{confirm_text}"
                            ),
                            score,
                        )
                    )

    long_pullback = min(item.low for item in recent) <= ema_fast_value + atr_value * cfg.trend_pullback_atr
    long_trigger = candle.close > ema_fast_value and candle.close > candle.open
    long_rsi_ok = cfg.trend_long_rsi_min <= rsi_value <= cfg.trend_long_rsi_max
    long_aligned = ema_spread >= cfg.trend_min_ema_spread and drift_value >= cfg.trend_min_drift_pct
    if long_aligned and long_pullback and long_trigger and long_rsi_ok and side_allowed_by_mode("long", cfg, None):
        pullback_distance_atr = abs(ema_fast_value - min(item.low for item in recent)) / atr_value if atr_value else 0.0
        base_score = trend_signal_score("long", rsi_value, adx_value, ema_spread, drift_value, pullback_distance_atr, cfg)
        market_allowed, market_delta, _market_risk, market_text = market_context_for_signal("long", candle, market_context, cfg)
        if market_allowed:
            score = clamp(base_score + market_delta, 0.0, 1.0)
            if score >= cfg.trend_min_signal_score:
                confirm_allowed, confirm_text = higher_timeframe_trend_allows("long", candle, higher_context or {}, cfg)
                if confirm_allowed:
                    candidates.append(
                        (
                            "long",
                            (
                                f"trend_long_pullback rsi={rsi_value:.1f} adx={adx_value:.1f} "
                                f"ema_spread={ema_spread:.4f} drift={drift_value:.4f} "
                                f"pullback_atr={pullback_distance_atr:.2f} base_score={base_score:.3f} "
                                f"score={score:.3f}{market_text}{confirm_text}"
                            ),
                            score,
                        )
                    )

    return max(candidates, key=lambda item: item[2]) if candidates else None


def exhaustion_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    if index < 1:
        return None
    setup_index = index - 1
    setup = candles[setup_index]
    candle = candles[index]
    upper = ind["bb_upper"][setup_index]
    lower = ind["bb_lower"][setup_index]
    rsi_value = ind["rsi"][setup_index]
    adx_value = ind["adx"][setup_index]
    atr_value = ind["atr"][setup_index]
    ema_fast_value = ind["ema_fast"][setup_index]
    ema_slow_value = ind["ema_slow"][setup_index]
    if not is_ready([upper, lower, rsi_value, adx_value, atr_value, ema_fast_value, ema_slow_value]):
        return None

    ema_spread = (ema_fast_value - ema_slow_value) / candle.close if candle.close else 0.0
    if adx_value < cfg.exhaustion_min_adx or abs(ema_spread) < cfg.exhaustion_min_ema_spread:
        return None

    vol_ratio = volume_ratio(candles, index)
    if vol_ratio < cfg.exhaustion_volume_ratio:
        return None

    candidates: List[Tuple[str, str, float]] = []
    if (
        ema_spread > 0
        and rsi_value >= 74
        and setup.high >= upper
        and candle.close < setup.close
        and candle.close < setup.open
        and candle.close < candle.open
        and close_position_ratio(setup) <= 0.45
        and upper_wick_ratio(setup) >= 0.30
        and reversal_regime_allows("short", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("short", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((adx_value - cfg.exhaustion_min_adx) / 22, 0.0, 1.0),
            rejection=upper_wick_ratio(setup),
            volume=clamp((vol_ratio - 1.0) / 1.5, 0.0, 1.0),
            context=clamp((rsi_value - 70) / 15, 0.0, 1.0),
        )
        scored = apply_market_score("short", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(
                    (
                        "short",
                        (
                            f"exhaustion_top_reversal rsi={rsi_value:.1f} adx={adx_value:.1f} "
                            f"ema_spread={ema_spread:.4f} upper_wick={upper_wick_ratio(setup):.2f} "
                            f"vol_ratio={vol_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}"
                        ),
                        score,
                    )
                )

    if (
        ema_spread < 0
        and rsi_value <= 26
        and setup.low <= lower
        and candle.close > setup.close
        and candle.close > setup.open
        and candle.close > candle.open
        and close_position_ratio(setup) >= 0.55
        and lower_wick_ratio(setup) >= 0.30
        and reversal_regime_allows("long", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("long", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((adx_value - cfg.exhaustion_min_adx) / 22, 0.0, 1.0),
            rejection=lower_wick_ratio(setup),
            volume=clamp((vol_ratio - 1.0) / 1.5, 0.0, 1.0),
            context=clamp((30 - rsi_value) / 15, 0.0, 1.0),
        )
        scored = apply_market_score("long", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(
                    (
                        "long",
                        (
                            f"exhaustion_bottom_reversal rsi={rsi_value:.1f} adx={adx_value:.1f} "
                            f"ema_spread={ema_spread:.4f} lower_wick={lower_wick_ratio(setup):.2f} "
                            f"vol_ratio={vol_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}"
                        ),
                        score,
                    )
                )
    return max(candidates, key=lambda item: item[2]) if candidates else None


def fake_breakout_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    if index < 1:
        return None
    setup_index = index - 1
    setup = candles[setup_index]
    candle = candles[index]
    atr_value = ind["atr"][setup_index]
    rsi_value = ind["rsi"][setup_index]
    if not is_ready([atr_value, rsi_value]):
        return None
    bounds = prior_range(candles, setup_index, cfg.fake_breakout_lookback_bars)
    if bounds is None:
        return None
    prior_high, prior_low = bounds
    buffer = atr_value * cfg.fake_breakout_buffer_atr
    vol_ratio = volume_ratio(candles, index)
    candidates: List[Tuple[str, str, float]] = []

    if (
        setup.high > prior_high + buffer
        and setup.close < prior_high
        and candle.close < setup.close
        and candle.close < prior_high
        and candle.close < candle.open
        and reversal_regime_allows("short", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("short", cfg, None)
    ):
        rejection = clamp((setup.high - candle.close) / max(atr_value, 1e-9), 0.0, 1.0)
        base_score = base_event_score(
            strength=clamp((setup.high - prior_high) / max(atr_value, 1e-9), 0.0, 1.0),
            rejection=max(rejection, upper_wick_ratio(setup)),
            volume=clamp((vol_ratio - 1.0) / 1.4, 0.0, 1.0),
            context=clamp((rsi_value - 55) / 25, 0.0, 1.0),
        )
        scored = apply_market_score("short", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("short", f"fake_breakout_bull_trap prior_high={prior_high:.2f} rsi={rsi_value:.1f} vol_ratio={vol_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    if (
        setup.low < prior_low - buffer
        and setup.close > prior_low
        and candle.close > setup.close
        and candle.close > prior_low
        and candle.close > candle.open
        and reversal_regime_allows("long", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("long", cfg, None)
    ):
        rejection = clamp((candle.close - setup.low) / max(atr_value, 1e-9), 0.0, 1.0)
        base_score = base_event_score(
            strength=clamp((prior_low - setup.low) / max(atr_value, 1e-9), 0.0, 1.0),
            rejection=max(rejection, lower_wick_ratio(setup)),
            volume=clamp((vol_ratio - 1.0) / 1.4, 0.0, 1.0),
            context=clamp((45 - rsi_value) / 25, 0.0, 1.0),
        )
        scored = apply_market_score("long", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("long", f"fake_breakout_bear_trap prior_low={prior_low:.2f} rsi={rsi_value:.1f} vol_ratio={vol_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    return max(candidates, key=lambda item: item[2]) if candidates else None


def squeeze_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    candle = candles[index]
    upper = ind["bb_upper"][index]
    lower = ind["bb_lower"][index]
    bandwidth = ind["bb_bandwidth"][index]
    atr_value = ind["atr"][index]
    ema_fast_value = ind["ema_fast"][index]
    ema_slow_value = ind["ema_slow"][index]
    if not is_ready([upper, lower, bandwidth, atr_value, ema_fast_value, ema_slow_value]):
        return None
    avg_bandwidth = optional_mean(ind["bb_bandwidth"], index - 1, cfg.squeeze_lookback_bars)
    if avg_bandwidth is None or avg_bandwidth > cfg.squeeze_max_bandwidth:
        return None
    vol_ratio = volume_ratio(candles, index)
    if vol_ratio < cfg.squeeze_volume_ratio:
        return None
    bounds = prior_range(candles, index, cfg.squeeze_lookback_bars)
    if bounds is None:
        return None
    prior_high, prior_low = bounds
    ema_spread = (ema_fast_value - ema_slow_value) / candle.close if candle.close else 0.0
    previous = candles[index - 1] if index > 0 else candle
    previous_upper = ind["bb_upper"][index - 1] if index > 0 else None
    previous_lower = ind["bb_lower"][index - 1] if index > 0 else None
    if not is_ready([previous_upper, previous_lower]):
        return None
    previous_inside_band = previous_lower <= previous.close <= previous_upper
    body_ratio = candle_body(candle) / max(candle_range(candle), 1e-9)
    candidates: List[Tuple[str, str, float]] = []

    if (
        previous_inside_band
        and candle.close > upper
        and candle.close > prior_high
        and ema_spread >= cfg.trend_min_ema_spread * 0.5
        and close_position_ratio(candle) >= 0.70
        and body_ratio >= 0.45
        and side_allowed_by_mode("long", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((candle.close - prior_high) / max(atr_value, 1e-9), 0.0, 1.0),
            rejection=close_position_ratio(candle),
            volume=clamp((vol_ratio - 1.0) / 1.8, 0.0, 1.0),
            context=clamp((cfg.squeeze_max_bandwidth - avg_bandwidth) / cfg.squeeze_max_bandwidth, 0.0, 1.0),
        )
        scored = apply_market_score("long", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("long", f"squeeze_breakout_long avg_bandwidth={avg_bandwidth:.4f} vol_ratio={vol_ratio:.2f} ema_spread={ema_spread:.4f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    if (
        previous_inside_band
        and candle.close < lower
        and candle.close < prior_low
        and ema_spread <= -cfg.trend_min_ema_spread * 0.5
        and close_position_ratio(candle) <= 0.30
        and body_ratio >= 0.45
        and side_allowed_by_mode("short", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((prior_low - candle.close) / max(atr_value, 1e-9), 0.0, 1.0),
            rejection=1.0 - close_position_ratio(candle),
            volume=clamp((vol_ratio - 1.0) / 1.8, 0.0, 1.0),
            context=clamp((cfg.squeeze_max_bandwidth - avg_bandwidth) / cfg.squeeze_max_bandwidth, 0.0, 1.0),
        )
        scored = apply_market_score("short", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("short", f"squeeze_breakout_short avg_bandwidth={avg_bandwidth:.4f} vol_ratio={vol_ratio:.2f} ema_spread={ema_spread:.4f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    return max(candidates, key=lambda item: item[2]) if candidates else None


def shock_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    if index < 1:
        return None
    setup_index = index - 1
    setup = candles[setup_index]
    candle = candles[index]
    atr_value = ind["atr"][setup_index]
    ema_fast_value = ind["ema_fast"][setup_index]
    if not is_ready([atr_value, ema_fast_value]):
        return None
    span_atr = candle_range(setup) / max(atr_value, 1e-9)
    vol_ratio = volume_ratio(candles, setup_index)
    if span_atr < cfg.shock_atr_multiple or vol_ratio < cfg.shock_volume_ratio:
        return None

    candidates: List[Tuple[str, str, float]] = []
    body_ratio = candle_body(setup) / max(candle_range(setup), 1e-9)
    if (
        setup.close > setup.open
        and setup.close > ema_fast_value
        and close_position_ratio(setup) >= 0.75
        and body_ratio >= 0.55
        and candle.close > setup.close
        and candle.close > candle.open
        and side_allowed_by_mode("long", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((span_atr - cfg.shock_atr_multiple) / 2.5, 0.0, 1.0),
            rejection=close_position_ratio(setup),
            volume=clamp((vol_ratio - cfg.shock_volume_ratio) / 2.5, 0.0, 1.0),
            context=body_ratio,
        )
        scored = apply_market_score("long", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score + 0.04:
                candidates.append(("long", f"shock_momentum_long span_atr={span_atr:.2f} vol_ratio={vol_ratio:.2f} body={body_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    if (
        setup.close < setup.open
        and setup.close < ema_fast_value
        and close_position_ratio(setup) <= 0.25
        and body_ratio >= 0.55
        and candle.close < setup.close
        and candle.close < candle.open
        and side_allowed_by_mode("short", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((span_atr - cfg.shock_atr_multiple) / 2.5, 0.0, 1.0),
            rejection=1.0 - close_position_ratio(setup),
            volume=clamp((vol_ratio - cfg.shock_volume_ratio) / 2.5, 0.0, 1.0),
            context=body_ratio,
        )
        scored = apply_market_score("short", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score + 0.04:
                candidates.append(("short", f"shock_momentum_short span_atr={span_atr:.2f} vol_ratio={vol_ratio:.2f} body={body_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    return max(candidates, key=lambda item: item[2]) if candidates else None


def sweep_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    if index < 1:
        return None
    setup_index = index - 1
    setup = candles[setup_index]
    candle = candles[index]
    atr_value = ind["atr"][setup_index]
    rsi_value = ind["rsi"][setup_index]
    if not is_ready([atr_value, rsi_value]):
        return None
    bounds = prior_range(candles, setup_index, cfg.sweep_lookback_bars)
    if bounds is None:
        return None
    prior_high, prior_low = bounds
    vol_ratio = volume_ratio(candles, setup_index)
    if vol_ratio < cfg.sweep_volume_ratio:
        return None
    candidates: List[Tuple[str, str, float]] = []

    if (
        setup.high > prior_high
        and setup.close < prior_high
        and upper_wick_ratio(setup) >= cfg.sweep_wick_ratio
        and candle.close < setup.close
        and candle.close < prior_high
        and candle.close < candle.open
        and reversal_regime_allows("short", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("short", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((setup.high - prior_high) / max(atr_value, 1e-9), 0.0, 1.0),
            rejection=upper_wick_ratio(setup),
            volume=clamp((vol_ratio - 1.0) / 1.6, 0.0, 1.0),
            context=clamp((rsi_value - 50) / 30, 0.0, 1.0),
        )
        scored = apply_market_score("short", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("short", f"sweep_upper_liquidity prior_high={prior_high:.2f} wick={upper_wick_ratio(setup):.2f} vol_ratio={vol_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    if (
        setup.low < prior_low
        and setup.close > prior_low
        and lower_wick_ratio(setup) >= cfg.sweep_wick_ratio
        and candle.close > setup.close
        and candle.close > prior_low
        and candle.close > candle.open
        and reversal_regime_allows("long", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("long", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp((prior_low - setup.low) / max(atr_value, 1e-9), 0.0, 1.0),
            rejection=lower_wick_ratio(setup),
            volume=clamp((vol_ratio - 1.0) / 1.6, 0.0, 1.0),
            context=clamp((50 - rsi_value) / 30, 0.0, 1.0),
        )
        scored = apply_market_score("long", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("long", f"sweep_lower_liquidity prior_low={prior_low:.2f} wick={lower_wick_ratio(setup):.2f} vol_ratio={vol_ratio:.2f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    return max(candidates, key=lambda item: item[2]) if candidates else None


def wide_failure_signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    if index < 1:
        return None
    setup_index = index - 1
    setup = candles[setup_index]
    candle = candles[index]
    atr_value = ind["atr"][setup_index]
    bandwidth = ind["bb_bandwidth"][setup_index]
    if not is_ready([atr_value, bandwidth]):
        return None
    bounds = prior_range(candles, setup_index, cfg.wide_failure_lookback_bars)
    if bounds is None:
        return None
    prior_high, prior_low = bounds
    range_width_atr = (prior_high - prior_low) / max(atr_value, 1e-9)
    if range_width_atr < cfg.wide_failure_min_range_atr and bandwidth < cfg.max_bandwidth * 0.70:
        return None

    candidates: List[Tuple[str, str, float]] = []
    if (
        setup.high > prior_high
        and setup.close < prior_high
        and candle.close < setup.close
        and candle.close < prior_high
        and close_position_ratio(setup) <= 0.45
        and reversal_regime_allows("short", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("short", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp(range_width_atr / 8, 0.0, 1.0),
            rejection=clamp((setup.high - candle.close) / max(atr_value, 1e-9), 0.0, 1.0),
            volume=clamp((volume_ratio(candles, setup_index) - 1.0) / 1.5, 0.0, 1.0),
            context=clamp(bandwidth / cfg.max_bandwidth, 0.0, 1.0),
        )
        scored = apply_market_score("short", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("short", f"wide_failure_upper range_atr={range_width_atr:.2f} bandwidth={bandwidth:.4f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    if (
        setup.low < prior_low
        and setup.close > prior_low
        and candle.close > setup.close
        and candle.close > prior_low
        and close_position_ratio(setup) >= 0.55
        and reversal_regime_allows("long", candles, ind, setup_index, cfg)
        and side_allowed_by_mode("long", cfg, None)
    ):
        base_score = base_event_score(
            strength=clamp(range_width_atr / 8, 0.0, 1.0),
            rejection=clamp((candle.close - setup.low) / max(atr_value, 1e-9), 0.0, 1.0),
            volume=clamp((volume_ratio(candles, setup_index) - 1.0) / 1.5, 0.0, 1.0),
            context=clamp(bandwidth / cfg.max_bandwidth, 0.0, 1.0),
        )
        scored = apply_market_score("long", candle, base_score, market_context, cfg)
        if scored is not None:
            score, market_text = scored
            if score >= cfg.event_min_signal_score:
                candidates.append(("long", f"wide_failure_lower range_atr={range_width_atr:.2f} bandwidth={bandwidth:.4f} base_score={base_score:.3f} score={score:.3f}{market_text}", score))

    return max(candidates, key=lambda item: item[2]) if candidates else None


def signal_for_index(
    candles: Sequence[Candle],
    ind: Dict[str, List[Optional[float]]],
    index: int,
    cfg: StrategyConfig,
    higher_context: Optional[Dict[str, Dict[str, Any]]] = None,
    market_context: Optional[MarketContext] = None,
) -> Optional[Tuple[str, str, float]]:
    candidates: List[Tuple[str, str, float]] = []
    if strategy_enabled("trend", cfg):
        trend_signal = trend_signal_for_index(candles, ind, index, cfg, higher_context, market_context)
        if trend_signal is not None:
            candidates.append(trend_signal)
    if strategy_enabled("exhaustion", cfg):
        exhaustion_signal = exhaustion_signal_for_index(candles, ind, index, cfg, market_context)
        if exhaustion_signal is not None:
            candidates.append(exhaustion_signal)
    if strategy_enabled("fake_breakout", cfg):
        fake_breakout_signal = fake_breakout_signal_for_index(candles, ind, index, cfg, market_context)
        if fake_breakout_signal is not None:
            candidates.append(fake_breakout_signal)
    if strategy_enabled("squeeze", cfg):
        squeeze_signal = squeeze_signal_for_index(candles, ind, index, cfg, market_context)
        if squeeze_signal is not None:
            candidates.append(squeeze_signal)
    if strategy_enabled("shock", cfg):
        shock_signal = shock_signal_for_index(candles, ind, index, cfg, market_context)
        if shock_signal is not None:
            candidates.append(shock_signal)
    if strategy_enabled("sweep", cfg):
        sweep_signal = sweep_signal_for_index(candles, ind, index, cfg, market_context)
        if sweep_signal is not None:
            candidates.append(sweep_signal)
    if strategy_enabled("wide_failure", cfg):
        wide_failure_signal = wide_failure_signal_for_index(candles, ind, index, cfg, market_context)
        if wide_failure_signal is not None:
            candidates.append(wide_failure_signal)
    if strategy_enabled("range", cfg):
        range_signal = range_signal_for_index(candles, ind, index, cfg, higher_context, market_context)
        if range_signal is not None:
            candidates.append(range_signal)
    if not candidates:
        return None

    reversal_events = [item for item in candidates if is_reversal_event(item[1])]
    if len(reversal_events) > 1:
        best_reversal = max(reversal_events, key=lambda item: (event_priority(item[1]), item[2]))
        candidates = [item for item in candidates if not is_reversal_event(item[1])]
        candidates.append(best_reversal)

    return max(candidates, key=lambda item: (adjusted_candidate_score(item, cfg), event_priority(item[1])))


def build_trend_pending_entry(
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
    if atr_value is None:
        return None

    pullback = atr_value * cfg.trend_entry_pullback_atr
    if side == "long":
        target = signal_candle.close - pullback
        stop = target - max(atr_value * cfg.trend_stop_atr, target * cfg.trend_min_stop_pct)
        risk = target - stop
        tp1 = target + risk * cfg.trend_tp1_rr
        tp2 = target + risk * cfg.trend_tp2_rr
        tp3 = target + risk * cfg.trend_tp3_rr
        reward_risk = (tp2 - target) / risk if risk > 0 else 0.0
    else:
        target = signal_candle.close + pullback
        stop = target + max(atr_value * cfg.trend_stop_atr, target * cfg.trend_min_stop_pct)
        risk = stop - target
        tp1 = target - risk * cfg.trend_tp1_rr
        tp2 = target - risk * cfg.trend_tp2_rr
        tp3 = target - risk * cfg.trend_tp3_rr
        reward_risk = (target - tp2) / risk if risk > 0 else 0.0

    if risk <= 0 or reward_risk < cfg.trend_min_reward_risk:
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


def build_event_pending_entry(
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
    if atr_value is None:
        return None

    pullback = atr_value * cfg.event_entry_pullback_atr
    min_stop = max(atr_value * cfg.event_stop_atr, signal_candle.close * cfg.event_min_stop_pct)
    stop_buffer = atr_value * cfg.event_stop_buffer_atr
    reversal_prefixes = (
        "exhaustion_",
        "fake_breakout_",
        "sweep_",
        "wide_failure_",
    )
    use_extreme_stop = reason.startswith(reversal_prefixes)

    if side == "long":
        target = signal_candle.close - pullback
        stop = target - min_stop
        if use_extreme_stop:
            stop = min(stop, signal_candle.low - stop_buffer)
        risk = target - stop
        tp1 = target + risk * cfg.event_tp1_rr
        tp2 = target + risk * cfg.event_tp2_rr
        tp3 = target + risk * cfg.event_tp3_rr
        reward_risk = (tp2 - target) / risk if risk > 0 else 0.0
    else:
        target = signal_candle.close + pullback
        stop = target + min_stop
        if use_extreme_stop:
            stop = max(stop, signal_candle.high + stop_buffer)
        risk = stop - target
        tp1 = target - risk * cfg.event_tp1_rr
        tp2 = target - risk * cfg.event_tp2_rr
        tp3 = target - risk * cfg.event_tp3_rr
        reward_risk = (target - tp2) / risk if risk > 0 else 0.0

    if risk <= 0 or reward_risk < cfg.event_min_reward_risk:
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
        risk_multiplier=risk_multiplier_from_score(signal_score, cfg) * cfg.event_risk_multiplier,
        signal_reason=reason,
    )


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
    if reason.startswith("trend_"):
        return build_trend_pending_entry(candles, ind, signal_index, created_index, side, reason, signal_score, cfg)
    if reason.startswith(
        (
            "exhaustion_",
            "fake_breakout_",
            "squeeze_",
            "shock_",
            "sweep_",
            "wide_failure_",
        )
    ):
        return build_event_pending_entry(candles, ind, signal_index, created_index, side, reason, signal_score, cfg)

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


def simulate(
    candles: Sequence[Candle],
    cfg: StrategyConfig,
    evaluation_start_ms: Optional[int] = None,
    market_context: Optional[MarketContext] = None,
) -> Dict[str, Any]:
    ind = indicators(candles, cfg)
    higher_context = build_higher_timeframe_context(candles, cfg)
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

    start_index = warmup
    if evaluation_start_ms is not None:
        start_index = max(warmup, next((i for i, candle in enumerate(candles) if candle.open_time_ms >= evaluation_start_ms), len(candles)))

    for index in range(start_index, len(candles)):
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
                signal = signal_for_index(candles, ind, signal_index, cfg, higher_context, market_context)
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

    summary_candles = candles[start_index:] if start_index < len(candles) else candles[-1:]
    summary = summarize_results(summary_candles, trades, equity_curve, cfg, equity, max_drawdown)
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
    print("Multi-strategy futures simulation")
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


def parse_timeframes(raw_value: str) -> Tuple[str, ...]:
    if not raw_value.strip():
        return ()
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a BTCUSDT futures range-swing strategy without placing real orders.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=float, default=60.0)
    parser.add_argument("--snapshot", type=Path, help="Use a saved snapshot JSON instead of fetching Binance klines.")
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.03)
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
    parser.add_argument("--long-rsi", type=float, default=35.0)
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
    parser.add_argument("--trail-atr", type=float, default=2.0)
    parser.add_argument("--max-hold-bars", type=int, default=36)
    parser.add_argument("--cooldown-bars", type=int, default=4)
    parser.add_argument("--max-drawdown-stop-pct", type=float, default=10.0)
    parser.add_argument("--high-adx-drift-threshold", type=float, default=26.0)
    parser.add_argument("--min-high-adx-drift-pct", type=float, default=0.0012)
    parser.add_argument("--min-signal-score", type=float, default=0.50)
    parser.add_argument("--confirm-timeframes", default="15m,30m,1h,4h", help="Comma-separated higher timeframes used to filter 5m entries. Empty disables the filter.")
    parser.add_argument("--confirm-drift-lookback-bars", type=int, default=16)
    parser.add_argument("--confirm-countertrend-drift-limit-pct", type=float, default=0.0012)
    parser.add_argument("--confirm-countertrend-ema-spread-pct", type=float, default=0.0005)
    parser.add_argument("--confirm-countertrend-min-adx", type=float, default=18.0)
    parser.add_argument("--confirm-min-space-pct", type=float, default=0.0015)
    parser.add_argument("--trend-stretch-filter-timeframes", default="4h")
    parser.add_argument("--max-trend-stretch-ema-spread", type=float, default=0.030)
    parser.add_argument("--min-trend-stretch-adx", type=float, default=18.0)
    parser.add_argument("--adaptive-risk-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-risk-multiplier", type=float, default=0.80)
    parser.add_argument("--max-risk-multiplier", type=float, default=1.20)
    parser.add_argument("--market-context-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--market-context-periods", default="5m,15m,1h,4h")
    parser.add_argument("--min-market-context-score", type=float, default=0.0)
    parser.add_argument("--market-context-score-weight", type=float, default=0.10)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.004)
    parser.add_argument("--liquidation-fee-pct", type=float, default=0.001)
    parser.add_argument("--entry-slippage-bps", type=float, default=0.5)
    parser.add_argument("--exit-slippage-bps", type=float, default=1.5)
    parser.add_argument("--depth-impact-bps", type=float, default=8.0)
    parser.add_argument("--depth-impact-exponent", type=float, default=0.5)
    parser.add_argument("--min-depth-quote", type=float, default=2_000_000.0)
    parser.add_argument(
        "--strategy-modes",
        default="trend",
        help="Comma-separated strategy modules: range,trend,exhaustion,fake_breakout,squeeze,shock,sweep,wide_failure,all.",
    )
    parser.add_argument("--trend-confirm-timeframes", default="15m,1h,4h")
    parser.add_argument("--trend-min-signal-score", type=float, default=0.82)
    parser.add_argument("--trend-min-adx", type=float, default=34.0)
    parser.add_argument("--trend-min-ema-spread", type=float, default=0.0030)
    parser.add_argument("--trend-min-drift-pct", type=float, default=0.0015)
    parser.add_argument("--trend-pullback-lookback-bars", type=int, default=8)
    parser.add_argument("--trend-pullback-atr", type=float, default=0.80)
    parser.add_argument("--trend-entry-pullback-atr", type=float, default=0.15)
    parser.add_argument("--trend-stop-atr", type=float, default=1.35)
    parser.add_argument("--trend-min-stop-pct", type=float, default=0.0035)
    parser.add_argument("--trend-tp1-rr", type=float, default=1.0)
    parser.add_argument("--trend-tp2-rr", type=float, default=2.0)
    parser.add_argument("--trend-tp3-rr", type=float, default=3.2)
    parser.add_argument("--trend-min-reward-risk", type=float, default=1.60)
    parser.add_argument("--trend-short-rsi-min", type=float, default=22.0)
    parser.add_argument("--trend-short-rsi-max", type=float, default=58.0)
    parser.add_argument("--trend-long-rsi-min", type=float, default=42.0)
    parser.add_argument("--trend-long-rsi-max", type=float, default=78.0)
    parser.add_argument("--event-min-signal-score", type=float, default=0.95)
    parser.add_argument("--event-risk-multiplier", type=float, default=0.20)
    parser.add_argument("--event-core-score-penalty", type=float, default=0.10)
    parser.add_argument("--event-regime-max-adx", type=float, default=32.0)
    parser.add_argument("--event-regime-max-aligned-ema-spread", type=float, default=0.006)
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
    confirm_timeframes = parse_timeframes(args.confirm_timeframes)
    for timeframe in confirm_timeframes:
        interval_to_ms(timeframe)
    market_context_periods = parse_timeframes(args.market_context_periods)
    for period in market_context_periods:
        interval_to_ms(period)
    trend_stretch_filter_timeframes = parse_timeframes(args.trend_stretch_filter_timeframes)
    for timeframe in trend_stretch_filter_timeframes:
        interval_to_ms(timeframe)
    strategy_modes = tuple(item.lower() for item in parse_timeframes(args.strategy_modes))
    valid_strategy_modes = {
        "range",
        "trend",
        "exhaustion",
        "fake_breakout",
        "squeeze",
        "shock",
        "sweep",
        "wide_failure",
        "all",
    }
    if not strategy_modes or any(item not in valid_strategy_modes for item in strategy_modes):
        raise ValueError("--strategy-modes must contain one or more of: range,trend,all")
    trend_confirm_timeframes = parse_timeframes(args.trend_confirm_timeframes)
    for timeframe in trend_confirm_timeframes:
        interval_to_ms(timeframe)
    if args.confirm_drift_lookback_bars < 0:
        raise ValueError("--confirm-drift-lookback-bars must be >= 0")
    if args.confirm_countertrend_drift_limit_pct < 0 or args.confirm_countertrend_ema_spread_pct < 0:
        raise ValueError("--confirm-countertrend-* values must be >= 0")
    if args.confirm_countertrend_min_adx < 0 or args.confirm_min_space_pct < 0:
        raise ValueError("--confirm-countertrend-min-adx and --confirm-min-space-pct must be >= 0")
    if args.max_trend_stretch_ema_spread < 0 or args.min_trend_stretch_adx < 0:
        raise ValueError("--max-trend-stretch-ema-spread and --min-trend-stretch-adx must be >= 0")
    if args.min_risk_multiplier <= 0 or args.max_risk_multiplier <= 0:
        raise ValueError("--min-risk-multiplier and --max-risk-multiplier must be > 0")
    if args.min_risk_multiplier > args.max_risk_multiplier:
        raise ValueError("--min-risk-multiplier must be <= --max-risk-multiplier")
    if not 0 <= args.min_market_context_score <= 1:
        raise ValueError("--min-market-context-score must be between 0 and 1")
    if args.market_context_score_weight < 0:
        raise ValueError("--market-context-score-weight must be >= 0")
    if args.maintenance_margin_pct < 0 or args.liquidation_fee_pct < 0:
        raise ValueError("--maintenance-margin-pct and --liquidation-fee-pct must be >= 0")
    if args.entry_slippage_bps < 0 or args.exit_slippage_bps < 0:
        raise ValueError("--entry-slippage-bps and --exit-slippage-bps must be >= 0")
    if args.depth_impact_bps < 0 or args.depth_impact_exponent < 0 or args.min_depth_quote <= 0:
        raise ValueError("--depth-impact-bps and --depth-impact-exponent must be >= 0; --min-depth-quote must be > 0")
    if not 0 <= args.trend_min_signal_score <= 1:
        raise ValueError("--trend-min-signal-score must be between 0 and 1")
    if args.trend_pullback_lookback_bars < 1:
        raise ValueError("--trend-pullback-lookback-bars must be >= 1")
    if (
        args.trend_min_adx < 0
        or args.trend_min_ema_spread < 0
        or args.trend_min_drift_pct < 0
        or args.trend_pullback_atr < 0
        or args.trend_entry_pullback_atr < 0
        or args.trend_stop_atr <= 0
        or args.trend_min_stop_pct <= 0
        or args.trend_min_reward_risk <= 0
    ):
        raise ValueError("--trend-* thresholds must be non-negative; stop/reward values must be > 0")
    if not (0 <= args.trend_short_rsi_min <= args.trend_short_rsi_max <= 100):
        raise ValueError("--trend-short-rsi-min/max must be ordered within 0..100")
    if not (0 <= args.trend_long_rsi_min <= args.trend_long_rsi_max <= 100):
        raise ValueError("--trend-long-rsi-min/max must be ordered within 0..100")
    if not 0 <= args.event_min_signal_score <= 1:
        raise ValueError("--event-min-signal-score must be between 0 and 1")
    if args.event_risk_multiplier <= 0 or args.event_core_score_penalty < 0:
        raise ValueError("--event-risk-multiplier must be > 0 and --event-core-score-penalty must be >= 0")
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
        confirm_timeframes=confirm_timeframes,
        confirm_drift_lookback_bars=args.confirm_drift_lookback_bars,
        confirm_countertrend_drift_limit_pct=args.confirm_countertrend_drift_limit_pct,
        confirm_countertrend_ema_spread_pct=args.confirm_countertrend_ema_spread_pct,
        confirm_countertrend_min_adx=args.confirm_countertrend_min_adx,
        confirm_min_space_pct=args.confirm_min_space_pct,
        trend_stretch_filter_timeframes=trend_stretch_filter_timeframes,
        max_trend_stretch_ema_spread=args.max_trend_stretch_ema_spread,
        min_trend_stretch_adx=args.min_trend_stretch_adx,
        adaptive_risk_enabled=args.adaptive_risk_enabled,
        min_risk_multiplier=args.min_risk_multiplier,
        max_risk_multiplier=args.max_risk_multiplier,
        market_context_enabled=args.market_context_enabled,
        market_context_periods=market_context_periods,
        min_market_context_score=args.min_market_context_score,
        market_context_score_weight=args.market_context_score_weight,
        maintenance_margin_pct=args.maintenance_margin_pct,
        liquidation_fee_pct=args.liquidation_fee_pct,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        depth_impact_bps=args.depth_impact_bps,
        depth_impact_exponent=args.depth_impact_exponent,
        min_depth_quote=args.min_depth_quote,
        strategy_modes=strategy_modes,
        trend_confirm_timeframes=trend_confirm_timeframes,
        trend_min_signal_score=args.trend_min_signal_score,
        trend_min_adx=args.trend_min_adx,
        trend_min_ema_spread=args.trend_min_ema_spread,
        trend_min_drift_pct=args.trend_min_drift_pct,
        trend_pullback_lookback_bars=args.trend_pullback_lookback_bars,
        trend_pullback_atr=args.trend_pullback_atr,
        trend_entry_pullback_atr=args.trend_entry_pullback_atr,
        trend_stop_atr=args.trend_stop_atr,
        trend_min_stop_pct=args.trend_min_stop_pct,
        trend_tp1_rr=args.trend_tp1_rr,
        trend_tp2_rr=args.trend_tp2_rr,
        trend_tp3_rr=args.trend_tp3_rr,
        trend_min_reward_risk=args.trend_min_reward_risk,
        trend_short_rsi_min=args.trend_short_rsi_min,
        trend_short_rsi_max=args.trend_short_rsi_max,
        trend_long_rsi_min=args.trend_long_rsi_min,
        trend_long_rsi_max=args.trend_long_rsi_max,
        event_min_signal_score=args.event_min_signal_score,
        event_risk_multiplier=args.event_risk_multiplier,
        event_core_score_penalty=args.event_core_score_penalty,
        event_regime_max_adx=args.event_regime_max_adx,
        event_regime_max_aligned_ema_spread=args.event_regime_max_aligned_ema_spread,
    )


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)

    if args.snapshot:
        candles = load_candles_from_snapshot(args.snapshot)
        source = str(args.snapshot)
        evaluation_start_ms = None
        market_context = None
    else:
        symbol = normalize_symbol(args.symbol)
        required_days = minimum_history_days(args.interval, cfg)
        fetch_days = args.days + required_days if cfg.confirm_timeframes else args.days
        candles = fetch_futures_klines(symbol, args.interval, fetch_days)
        evaluation_start_ms = candles[-1].open_time_ms - int(args.days * MS_PER_DAY) if candles else None
        source = f"binance_futures:{symbol}:{args.interval}:{args.days}d(+{required_days:.1f}d_warmup)"
        market_context = (
            fetch_market_context(symbol, min(args.days, 30.0), cfg.market_context_periods)
            if cfg.market_context_enabled
            else None
        )

    if len(candles) < max(cfg.ema_slow, cfg.bb_period, cfg.adx_period * 2) + 10:
        raise ValueError(f"Not enough candles for the configured indicators: {len(candles)}")

    result = simulate(candles, cfg, evaluation_start_ms, market_context)
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
