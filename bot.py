import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import ccxt
from dotenv import load_dotenv


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def normalize_symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol


@dataclass
class Config:
    api_key: str
    api_secret: str
    symbol: str
    leverage: int
    margin_mode: str
    hedge_mode: bool
    timeframe: str
    klines_limit: int
    fast_ema: int
    slow_ema: int
    fast_ma: int
    slow_ma: int
    ma_period: int
    trend_spread_threshold: float
    long_trend_spread_threshold: float
    short_trend_spread_threshold: float
    confirm_timeframe: str
    confirm_klines_limit: int
    retest_tolerance: float
    long_retest_tolerance: float
    short_retest_tolerance: float
    regime_trend_strength: float
    regime_range_strength: float
    regime_bandwidth_threshold: float
    range_bb_period: int
    range_bb_std: float
    range_rsi_period: int
    range_rsi_long: float
    range_rsi_short: float
    range_band_buffer: float
    range_flow_filter: float
    trade_lookback: int
    big_trade_usdt: float
    depth_levels: int
    flow_score_threshold: float
    long_flow_threshold: float
    short_flow_threshold: float
    sentiment_filter_enabled: bool
    sentiment_filter_mode: str
    sentiment_period: str
    sentiment_lookback: int
    sentiment_min_sources: int
    sentiment_long_threshold: float
    sentiment_short_threshold: float
    sentiment_oi_scale_pct: float
    sentiment_funding_scale: float
    sentiment_long_short_source: str
    sentiment_lsr_contrarian: bool
    sentiment_funding_contrarian: bool
    sentiment_open_interest_weight: float
    sentiment_long_short_weight: float
    sentiment_taker_ratio_weight: float
    sentiment_funding_weight: float
    orderbook_filter_enabled: bool
    orderbook_filter_range_only: bool
    orderbook_long_threshold: float
    orderbook_short_threshold: float
    vol_guard_enabled: bool
    vol_guard_atr_period: int
    vol_guard_atr_pct_soft: float
    vol_guard_atr_pct_hard: float
    vol_guard_soft_multiplier: float
    vol_guard_hard_multiplier: float
    trade_weight: float
    stop_loss_pct: float
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float
    tp1_close_ratio: float
    tp2_close_ratio: float
    max_hold_bars: int
    min_profit_usdt: float
    estimated_fee_rate: float
    profit_space_lookback: int
    profit_space_buffer: float
    runner_mode: bool
    trailing_stop_loose_pct: float
    trailing_stop_pct: float
    trailing_stop_tight_pct: float
    trend_reversal_exit_enabled: bool
    trend_reversal_min_profit_pct: float
    range_stop_loss_pct: float
    range_take_profit_pct: float
    range_max_hold_bars: int
    entry_mode: str
    entry_confirm_bars: int
    entry_confirm_extra_mixed: int
    entry_confirm_extra_range: int
    entry_cooldown_bars: int
    entry_atr_period: int
    entry_atr_multiplier: float
    entry_atr_min_pullback_pct: float
    entry_atr_max_pullback_pct: float
    entry_pullback_long_trend_pct: float
    entry_pullback_short_trend_pct: float
    entry_pullback_long_range_pct: float
    entry_pullback_short_range_pct: float
    entry_max_wait_bars: int
    entry_refresh_on_same_signal: bool
    entry_limit_tif: str
    entry_limit_offset_pct: float
    margin_usdt: float
    regime_size_trend_multiplier: float
    regime_size_mixed_multiplier: float
    regime_size_range_multiplier: float
    auto_position_sizing_enabled: bool
    auto_size_min_multiplier: float
    auto_size_max_multiplier: float
    auto_size_sensitivity: float
    auto_size_flow_ref: float
    auto_size_trend_ref: float
    auto_size_space_ref: float
    auto_size_stop_adjust_enabled: bool
    auto_size_stop_adjust_min: float
    auto_size_stop_adjust_max: float
    poll_seconds: int
    dry_run: bool
    testnet: bool

    @staticmethod
    def from_env() -> "Config":
        symbol = normalize_symbol(os.getenv("SYMBOL", "BTC/USDT:USDT"))
        return Config(
            api_key=os.getenv("BINANCE_API_KEY", "").strip(),
            api_secret=os.getenv("BINANCE_API_SECRET", "").strip(),
            symbol=symbol,
            leverage=env_int("LEVERAGE", 10),
            margin_mode=os.getenv("MARGIN_MODE", "isolated").lower(),
            hedge_mode=env_bool("HEDGE_MODE", False),
            timeframe=os.getenv("TIMEFRAME", "5m"),
            klines_limit=env_int("KLINES_LIMIT", 200),
            fast_ema=env_int("FAST_EMA", 7),
            slow_ema=env_int("SLOW_EMA", 25),
            fast_ma=env_int("FAST_MA", 7),
            slow_ma=env_int("SLOW_MA", 25),
            ma_period=env_int("MA_PERIOD", 99),
            trend_spread_threshold=env_float("TREND_SPREAD_THRESHOLD", 0.0008),
            long_trend_spread_threshold=env_float(
                "LONG_TREND_SPREAD_THRESHOLD",
                env_float("TREND_SPREAD_THRESHOLD", 0.0008),
            ),
            short_trend_spread_threshold=env_float(
                "SHORT_TREND_SPREAD_THRESHOLD",
                env_float("TREND_SPREAD_THRESHOLD", 0.0008),
            ),
            confirm_timeframe=os.getenv("CONFIRM_TIMEFRAME", "15m"),
            confirm_klines_limit=env_int("CONFIRM_KLINES_LIMIT", 300),
            retest_tolerance=env_float("RETEST_TOLERANCE", 0.002),
            long_retest_tolerance=env_float(
                "LONG_RETEST_TOLERANCE", env_float("RETEST_TOLERANCE", 0.002)
            ),
            short_retest_tolerance=env_float(
                "SHORT_RETEST_TOLERANCE", env_float("RETEST_TOLERANCE", 0.002)
            ),
            regime_trend_strength=env_float("REGIME_TREND_STRENGTH", 0.0012),
            regime_range_strength=env_float("REGIME_RANGE_STRENGTH", 0.0007),
            regime_bandwidth_threshold=env_float("REGIME_BANDWIDTH_THRESHOLD", 0.015),
            range_bb_period=env_int("RANGE_BB_PERIOD", 20),
            range_bb_std=env_float("RANGE_BB_STD", 2.0),
            range_rsi_period=env_int("RANGE_RSI_PERIOD", 14),
            range_rsi_long=env_float("RANGE_RSI_LONG", 35.0),
            range_rsi_short=env_float("RANGE_RSI_SHORT", 65.0),
            range_band_buffer=env_float("RANGE_BAND_BUFFER", 0.0015),
            range_flow_filter=env_float("RANGE_FLOW_FILTER", 0.08),
            trade_lookback=env_int("TRADE_LOOKBACK", 500),
            big_trade_usdt=env_float("BIG_TRADE_USDT", 100000.0),
            depth_levels=env_int("DEPTH_LEVELS", 20),
            flow_score_threshold=env_float("FLOW_SCORE_THRESHOLD", 0.12),
            long_flow_threshold=env_float(
                "LONG_FLOW_THRESHOLD", env_float("FLOW_SCORE_THRESHOLD", 0.12)
            ),
            short_flow_threshold=env_float(
                "SHORT_FLOW_THRESHOLD", env_float("FLOW_SCORE_THRESHOLD", 0.12)
            ),
            sentiment_filter_enabled=env_bool("SENTIMENT_FILTER_ENABLED", False),
            sentiment_filter_mode=os.getenv("SENTIMENT_FILTER_MODE", "trend_only").strip().lower(),
            sentiment_period=os.getenv("SENTIMENT_PERIOD", "5m").strip(),
            sentiment_lookback=env_int("SENTIMENT_LOOKBACK", 24),
            sentiment_min_sources=env_int("SENTIMENT_MIN_SOURCES", 2),
            sentiment_long_threshold=env_float("SENTIMENT_LONG_THRESHOLD", 0.08),
            sentiment_short_threshold=env_float("SENTIMENT_SHORT_THRESHOLD", 0.08),
            sentiment_oi_scale_pct=env_float("SENTIMENT_OI_SCALE_PCT", 0.010),
            sentiment_funding_scale=env_float("SENTIMENT_FUNDING_SCALE", 0.0005),
            sentiment_long_short_source=os.getenv("SENTIMENT_LONG_SHORT_SOURCE", "global").strip().lower(),
            sentiment_lsr_contrarian=env_bool("SENTIMENT_LSR_CONTRARIAN", True),
            sentiment_funding_contrarian=env_bool("SENTIMENT_FUNDING_CONTRARIAN", True),
            sentiment_open_interest_weight=env_float("SENTIMENT_OI_WEIGHT", 0.35),
            sentiment_long_short_weight=env_float("SENTIMENT_LONG_SHORT_WEIGHT", 0.30),
            sentiment_taker_ratio_weight=env_float("SENTIMENT_TAKER_RATIO_WEIGHT", 0.25),
            sentiment_funding_weight=env_float("SENTIMENT_FUNDING_WEIGHT", 0.10),
            orderbook_filter_enabled=env_bool("ORDERBOOK_FILTER_ENABLED", True),
            orderbook_filter_range_only=env_bool("ORDERBOOK_FILTER_RANGE_ONLY", True),
            orderbook_long_threshold=env_float("ORDERBOOK_LONG_THRESHOLD", 0.02),
            orderbook_short_threshold=env_float("ORDERBOOK_SHORT_THRESHOLD", 0.02),
            vol_guard_enabled=env_bool("VOL_GUARD_ENABLED", False),
            vol_guard_atr_period=env_int("VOL_GUARD_ATR_PERIOD", 16),
            vol_guard_atr_pct_soft=env_float("VOL_GUARD_ATR_PCT_SOFT", 0.0030),
            vol_guard_atr_pct_hard=env_float("VOL_GUARD_ATR_PCT_HARD", 0.0050),
            vol_guard_soft_multiplier=env_float("VOL_GUARD_SOFT_MULTIPLIER", 0.6),
            vol_guard_hard_multiplier=env_float("VOL_GUARD_HARD_MULTIPLIER", 0.35),
            trade_weight=env_float("TRADE_WEIGHT", 0.6),
            stop_loss_pct=env_float("STOP_LOSS_PCT", 0.006),
            tp1_pct=env_float("TP1_PCT", 0.004),
            tp2_pct=env_float("TP2_PCT", 0.008),
            tp3_pct=env_float("TP3_PCT", 0.012),
            tp1_close_ratio=env_float("TP1_CLOSE_RATIO", 0.2),
            tp2_close_ratio=env_float("TP2_CLOSE_RATIO", 0.3),
            max_hold_bars=env_int("MAX_HOLD_BARS", 0),
            min_profit_usdt=env_float("MIN_PROFIT_USDT", 10.0),
            estimated_fee_rate=env_float("ESTIMATED_FEE_RATE", 0.0004),
            profit_space_lookback=env_int("PROFIT_SPACE_LOOKBACK", 96),
            profit_space_buffer=env_float("PROFIT_SPACE_BUFFER", 1.1),
            runner_mode=env_bool("RUNNER_MODE", True),
            trailing_stop_loose_pct=env_float(
                "TRAILING_STOP_LOOSE_PCT", env_float("TRAILING_STOP_PCT", 0.01) * 1.35
            ),
            trailing_stop_pct=env_float("TRAILING_STOP_PCT", 0.01),
            trailing_stop_tight_pct=env_float(
                "TRAILING_STOP_TIGHT_PCT", env_float("TRAILING_STOP_PCT", 0.01) * 0.65
            ),
            trend_reversal_exit_enabled=env_bool("TREND_REVERSAL_EXIT_ENABLED", True),
            trend_reversal_min_profit_pct=env_float("TREND_REVERSAL_MIN_PROFIT_PCT", 0.006),
            range_stop_loss_pct=env_float("RANGE_STOP_LOSS_PCT", 0.0045),
            range_take_profit_pct=env_float("RANGE_TAKE_PROFIT_PCT", 0.007),
            range_max_hold_bars=env_int("RANGE_MAX_HOLD_BARS", 24),
            entry_mode=os.getenv("ENTRY_MODE", "market").strip().lower(),
            entry_confirm_bars=env_int("ENTRY_CONFIRM_BARS", 2),
            entry_confirm_extra_mixed=env_int("ENTRY_CONFIRM_EXTRA_MIXED", 1),
            entry_confirm_extra_range=env_int("ENTRY_CONFIRM_EXTRA_RANGE", 2),
            entry_cooldown_bars=env_int("ENTRY_COOLDOWN_BARS", 2),
            entry_atr_period=env_int("ENTRY_ATR_PERIOD", 14),
            entry_atr_multiplier=env_float("ENTRY_ATR_MULTIPLIER", 0.7),
            entry_atr_min_pullback_pct=env_float("ENTRY_ATR_MIN_PULLBACK_PCT", 0.0006),
            entry_atr_max_pullback_pct=env_float("ENTRY_ATR_MAX_PULLBACK_PCT", 0.0045),
            entry_pullback_long_trend_pct=env_float(
                "ENTRY_PULLBACK_LONG_TREND_PCT", 0.0008
            ),
            entry_pullback_short_trend_pct=env_float(
                "ENTRY_PULLBACK_SHORT_TREND_PCT", 0.0008
            ),
            entry_pullback_long_range_pct=env_float(
                "ENTRY_PULLBACK_LONG_RANGE_PCT", 0.0012
            ),
            entry_pullback_short_range_pct=env_float(
                "ENTRY_PULLBACK_SHORT_RANGE_PCT", 0.0012
            ),
            entry_max_wait_bars=env_int("ENTRY_MAX_WAIT_BARS", 3),
            entry_refresh_on_same_signal=env_bool(
                "ENTRY_REFRESH_ON_SAME_SIGNAL", False
            ),
            entry_limit_tif=os.getenv("ENTRY_LIMIT_TIF", "IOC").strip().upper(),
            entry_limit_offset_pct=env_float("ENTRY_LIMIT_OFFSET_PCT", 0.0002),
            margin_usdt=env_float("MARGIN_USDT", 30.0),
            regime_size_trend_multiplier=env_float("REGIME_SIZE_TREND_MULTIPLIER", 1.0),
            regime_size_mixed_multiplier=env_float("REGIME_SIZE_MIXED_MULTIPLIER", 0.7),
            regime_size_range_multiplier=env_float("REGIME_SIZE_RANGE_MULTIPLIER", 0.55),
            auto_position_sizing_enabled=env_bool("AUTO_POSITION_SIZING_ENABLED", False),
            auto_size_min_multiplier=env_float("AUTO_SIZE_MIN_MULTIPLIER", 0.6),
            auto_size_max_multiplier=env_float("AUTO_SIZE_MAX_MULTIPLIER", 1.4),
            auto_size_sensitivity=env_float("AUTO_SIZE_SENSITIVITY", 0.85),
            auto_size_flow_ref=env_float("AUTO_SIZE_FLOW_REF", 0.2),
            auto_size_trend_ref=env_float("AUTO_SIZE_TREND_REF", 0.0018),
            auto_size_space_ref=env_float("AUTO_SIZE_SPACE_REF", 1.3),
            auto_size_stop_adjust_enabled=env_bool("AUTO_SIZE_STOP_ADJUST_ENABLED", True),
            auto_size_stop_adjust_min=env_float("AUTO_SIZE_STOP_ADJUST_MIN", 0.85),
            auto_size_stop_adjust_max=env_float("AUTO_SIZE_STOP_ADJUST_MAX", 1.2),
            poll_seconds=env_int("POLL_SECONDS", 20),
            dry_run=env_bool("DRY_RUN", True),
            testnet=env_bool("BINANCE_TESTNET", False),
        )


@dataclass
class PositionMemory:
    side: str = "NONE"
    entry_mode: str = "NONE"
    entry_price: float = 0.0
    tp1_done: bool = False
    tp2_done: bool = False
    bars_held: int = 0
    peak_price: float = 0.0
    trough_price: float = 0.0

    def reset(self) -> None:
        self.side = "NONE"
        self.entry_mode = "NONE"
        self.entry_price = 0.0
        self.tp1_done = False
        self.tp2_done = False
        self.bars_held = 0
        self.peak_price = 0.0
        self.trough_price = 0.0


@dataclass
class PendingEntryMemory:
    signal: str = "NONE"
    mode: str = "NONE"
    regime: str = "MIXED"
    target_price: float = 0.0
    bars_waited: int = 0

    def active(self) -> bool:
        return self.signal in {"LONG", "SHORT"} and self.target_price > 0

    def clear(self) -> None:
        self.signal = "NONE"
        self.mode = "NONE"
        self.regime = "MIXED"
        self.target_price = 0.0
        self.bars_waited = 0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def create_exchange(cfg: Config) -> ccxt.binance:
    exchange = ccxt.binance(
        {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
            "timeout": 15000,
        }
    )
    if cfg.testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def ema_series(values: List[float], period: int) -> List[float]:
    if period <= 1:
        return values[:]
    if len(values) < period:
        return [0.0] * len(values)

    out = [0.0] * len(values)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    multiplier = 2 / (period + 1)
    prev = sma
    for i in range(period, len(values)):
        prev = (values[i] - prev) * multiplier + prev
        out[i] = prev
    return out


def sma_series(values: List[float], period: int) -> List[float]:
    if period <= 1:
        return values[:]
    if len(values) < period:
        return [0.0] * len(values)

    out = [0.0] * len(values)
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def stddev_series(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) < period:
        return [0.0] * len(values)

    out = [0.0] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / period
        out[i] = var ** 0.5
    return out


def rsi_series(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) <= period:
        return [50.0] * len(values)

    out = [50.0] * len(values)
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains[i] = max(diff, 0.0)
        losses[i] = max(-diff, 0.0)

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100 - 100 / (1 + rs)

    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - 100 / (1 + rs)
    return out


def atr_from_ohlcv(closed_ohlcv: List[List[float]], period: int) -> float:
    if period <= 1 or len(closed_ohlcv) < 2:
        return 0.0
    if len(closed_ohlcv) < period:
        return 0.0

    tr_values: List[float] = []
    for i in range(len(closed_ohlcv) - period, len(closed_ohlcv)):
        cur = closed_ohlcv[i]
        prev_close = float(closed_ohlcv[i - 1][4]) if i > 0 else float(cur[4])
        high = float(cur[2])
        low = float(cur[3])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    if not tr_values:
        return 0.0
    return sum(tr_values) / len(tr_values)


def trend_signal(closes: List[float], cfg: Config) -> Tuple[int, Dict[str, float]]:
    ema_fast = ema_series(closes, cfg.fast_ema)
    ema_slow = ema_series(closes, cfg.slow_ema)
    ma_fast = sma_series(closes, cfg.fast_ma)
    ma_slow = sma_series(closes, cfg.slow_ma)
    ma_long = sma_series(closes, cfg.ma_period)
    idx = len(closes) - 1
    required = max(cfg.fast_ema, cfg.slow_ema, cfg.fast_ma, cfg.slow_ma, cfg.ma_period)
    if idx < required:
        return 0, {
            "ema_spread": 0.0,
            "ema_slope": 0.0,
            "ma_spread": 0.0,
            "ma_fast_slope": 0.0,
            "ma_long_gap": 0.0,
            "ma_long_slope": 0.0,
        }

    price = closes[idx]
    ema_fast_now = ema_fast[idx]
    ema_fast_prev = ema_fast[idx - 1] if idx > 0 else ema_fast_now
    ema_slow_now = ema_slow[idx]
    ma_fast_now = ma_fast[idx]
    ma_fast_prev = ma_fast[idx - 1] if idx > 0 else ma_fast_now
    ma_slow_now = ma_slow[idx]
    ma_long_now = ma_long[idx]
    ma_long_prev = ma_long[idx - 1] if idx > 0 else ma_long_now

    ema_spread = (
        (ema_fast_now - ema_slow_now) / ema_slow_now if ema_slow_now else 0.0
    )
    ema_slope = (
        (ema_fast_now - ema_fast_prev) / ema_fast_prev if ema_fast_prev else 0.0
    )
    ma_spread = (ma_fast_now - ma_slow_now) / ma_slow_now if ma_slow_now else 0.0
    ma_fast_slope = (
        (ma_fast_now - ma_fast_prev) / ma_fast_prev if ma_fast_prev else 0.0
    )
    ma_long_gap = (price - ma_long_now) / ma_long_now if ma_long_now else 0.0
    ma_long_slope = (
        (ma_long_now - ma_long_prev) / ma_long_prev if ma_long_prev else 0.0
    )

    if (
        ema_spread > cfg.long_trend_spread_threshold
        and ema_slope > 0
        and ma_spread > cfg.long_trend_spread_threshold
        and ma_fast_slope > 0
        and price > ema_fast_now
        and price > ma_long_now
        and ma_long_slope > 0
    ):
        return 1, {
            "ema_spread": ema_spread,
            "ema_slope": ema_slope,
            "ma_spread": ma_spread,
            "ma_fast_slope": ma_fast_slope,
            "ma_long_gap": ma_long_gap,
            "ma_long_slope": ma_long_slope,
        }
    if (
        ema_spread < -cfg.short_trend_spread_threshold
        and ema_slope < 0
        and ma_spread < -cfg.short_trend_spread_threshold
        and ma_fast_slope < 0
        and price < ema_fast_now
        and price < ma_long_now
        and ma_long_slope < 0
    ):
        return -1, {
            "ema_spread": ema_spread,
            "ema_slope": ema_slope,
            "ma_spread": ma_spread,
            "ma_fast_slope": ma_fast_slope,
            "ma_long_gap": ma_long_gap,
            "ma_long_slope": ma_long_slope,
        }
    return 0, {
        "ema_spread": ema_spread,
        "ema_slope": ema_slope,
        "ma_spread": ma_spread,
        "ma_fast_slope": ma_fast_slope,
        "ma_long_gap": ma_long_gap,
        "ma_long_slope": ma_long_slope,
    }


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ratio_to_imbalance(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    return (ratio - 1.0) / (ratio + 1.0)


def pick_latest(items) -> Dict[str, float]:
    if isinstance(items, list) and items:
        row = items[-1]
        if isinstance(row, dict):
            return row
    return {}


def market_sentiment_metrics(
    exchange: ccxt.binance, market_id: str, cfg: Config
) -> Dict[str, float]:
    period = cfg.sentiment_period
    limit = max(2, min(cfg.sentiment_lookback, 500))
    score_parts: List[Tuple[float, float]] = []
    oi_score = 0.0
    lsr_score = 0.0
    taker_ratio_score = 0.0
    funding_score = 0.0
    oi_change = 0.0
    long_short_ratio = 1.0
    taker_buy_sell_ratio = 1.0
    funding_rate = 0.0
    source_count = 0

    try:
        oi_hist = exchange.fapiDataGetOpenInterestHist(
            {"symbol": market_id, "period": period, "limit": limit}
        )
        if isinstance(oi_hist, list) and len(oi_hist) >= 2:
            first = oi_hist[0]
            last = oi_hist[-1]
            first_oi = safe_float(first.get("sumOpenInterest"))
            last_oi = safe_float(last.get("sumOpenInterest"))
            if first_oi > 0:
                oi_change = (last_oi - first_oi) / first_oi
                scale = max(cfg.sentiment_oi_scale_pct, 1e-6)
                oi_score = clamp(oi_change / scale, -1.0, 1.0)
                score_parts.append((oi_score, cfg.sentiment_open_interest_weight))
                source_count += 1
    except Exception as exc:
        logging.debug("Sentiment OI 拉取失败: %s", exc)

    try:
        ratio_params = {"symbol": market_id, "period": period, "limit": limit}
        if cfg.sentiment_long_short_source == "top_account":
            ratio_data = exchange.fapiDataGetTopLongShortAccountRatio(ratio_params)
        elif cfg.sentiment_long_short_source == "top_position":
            ratio_data = exchange.fapiDataGetTopLongShortPositionRatio(ratio_params)
        else:
            ratio_data = exchange.fapiDataGetGlobalLongShortAccountRatio(ratio_params)
        latest = pick_latest(ratio_data)
        long_short_ratio = safe_float(latest.get("longShortRatio"), 1.0)
        imbalance = ratio_to_imbalance(long_short_ratio)
        lsr_score = -imbalance if cfg.sentiment_lsr_contrarian else imbalance
        lsr_score = clamp(lsr_score, -1.0, 1.0)
        score_parts.append((lsr_score, cfg.sentiment_long_short_weight))
        source_count += 1
    except Exception as exc:
        logging.debug("Sentiment 多空账户比拉取失败: %s", exc)

    try:
        taker_data = exchange.fapiDataGetTakerlongshortRatio(
            {"symbol": market_id, "period": period, "limit": limit}
        )
        latest = pick_latest(taker_data)
        taker_buy_sell_ratio = safe_float(latest.get("buySellRatio"), 1.0)
        taker_ratio_score = clamp(ratio_to_imbalance(taker_buy_sell_ratio), -1.0, 1.0)
        score_parts.append((taker_ratio_score, cfg.sentiment_taker_ratio_weight))
        source_count += 1
    except Exception as exc:
        logging.debug("Sentiment 主动买卖比拉取失败: %s", exc)

    try:
        fr_data = exchange.fapiPublicGetFundingRate({"symbol": market_id, "limit": 1})
        latest = pick_latest(fr_data)
        funding_rate = safe_float(latest.get("fundingRate"), 0.0)
        fr_scale = max(cfg.sentiment_funding_scale, 1e-8)
        funding_score = clamp(funding_rate / fr_scale, -1.0, 1.0)
        if cfg.sentiment_funding_contrarian:
            funding_score = -funding_score
        score_parts.append((funding_score, cfg.sentiment_funding_weight))
        source_count += 1
    except Exception as exc:
        logging.debug("Sentiment 资金费率拉取失败: %s", exc)

    weight_sum = sum(max(0.0, w) for _, w in score_parts)
    sentiment_score = (
        sum(score * max(0.0, weight) for score, weight in score_parts) / weight_sum
        if weight_sum > 0
        else 0.0
    )

    return {
        "sentiment_score": sentiment_score,
        "sentiment_source_count": float(source_count),
        "oi_change": oi_change,
        "oi_score": oi_score,
        "long_short_ratio": long_short_ratio,
        "lsr_score": lsr_score,
        "taker_buy_sell_ratio": taker_buy_sell_ratio,
        "taker_ratio_score": taker_ratio_score,
        "funding_rate": funding_rate,
        "funding_score": funding_score,
    }


def order_flow_metrics(
    exchange: ccxt.binance, market_id: str, cfg: Config
) -> Dict[str, float]:
    trades = exchange.fapiPublicGetTrades(
        {"symbol": market_id, "limit": min(cfg.trade_lookback, 1000)}
    )
    large_buy_notional = 0.0
    large_sell_notional = 0.0
    large_count = 0

    for t in trades:
        price = float(t["price"])
        qty = float(t["qty"])
        notional = price * qty
        if notional < cfg.big_trade_usdt:
            continue
        large_count += 1
        # isBuyerMaker=true 表示卖方是主动成交（主动卖出）。
        if to_bool(t.get("isBuyerMaker", False)):
            large_sell_notional += notional
        else:
            large_buy_notional += notional

    total_large = large_buy_notional + large_sell_notional
    trade_imbalance = (
        (large_buy_notional - large_sell_notional) / total_large if total_large else 0.0
    )

    depth = exchange.fapiPublicGetDepth(
        {"symbol": market_id, "limit": min(cfg.depth_levels, 100)}
    )
    bid_value = 0.0
    ask_value = 0.0
    for px, qty in depth.get("bids", [])[: cfg.depth_levels]:
        bid_value += float(px) * float(qty)
    for px, qty in depth.get("asks", [])[: cfg.depth_levels]:
        ask_value += float(px) * float(qty)

    depth_total = bid_value + ask_value
    depth_imbalance = (bid_value - ask_value) / depth_total if depth_total else 0.0

    trade_weight = min(max(cfg.trade_weight, 0.0), 1.0)
    flow_score = trade_weight * trade_imbalance + (1 - trade_weight) * depth_imbalance
    return {
        "trade_imbalance": trade_imbalance,
        "depth_imbalance": depth_imbalance,
        "flow_score": flow_score,
        "large_buy_notional": large_buy_notional,
        "large_sell_notional": large_sell_notional,
        "large_count": float(large_count),
    }


def flow_signal(metrics: Dict[str, float], cfg: Config) -> int:
    score = metrics["flow_score"]
    if score > cfg.long_flow_threshold:
        return 1
    if score < -cfg.short_flow_threshold:
        return -1
    return 0


def confirm_trend_signal(
    exchange: ccxt.binance, symbol: str, cfg: Config
) -> Tuple[int, Dict[str, float]]:
    ohlcv = exchange.fetch_ohlcv(
        symbol,
        timeframe=cfg.confirm_timeframe,
        limit=cfg.confirm_klines_limit,
    )
    required = max(cfg.slow_ema, cfg.slow_ma, cfg.ma_period, cfg.fast_ema, cfg.fast_ma)
    if len(ohlcv) < required + 2:
        return 0, {
            "ema_spread": 0.0,
            "ema_slope": 0.0,
            "ma_spread": 0.0,
            "ma_fast_slope": 0.0,
            "ma_long_gap": 0.0,
            "ma_long_slope": 0.0,
        }
    closed = ohlcv[:-1]
    closes = [float(c[4]) for c in closed]
    return trend_signal(closes, cfg)


def retest_status(closes: List[float], cfg: Config) -> Dict[str, float]:
    idx = len(closes) - 1
    need = max(cfg.fast_ema, cfg.fast_ma)
    if idx < need:
        return {"long_ok": 0.0, "short_ok": 0.0, "gap": 0.0}

    ema_fast = ema_series(closes, cfg.fast_ema)
    ma_fast = sma_series(closes, cfg.fast_ma)
    anchor = (ema_fast[idx] + ma_fast[idx]) / 2
    price = closes[idx]
    gap = (price - anchor) / anchor if anchor else 0.0
    long_ok = price >= anchor and abs(gap) <= cfg.long_retest_tolerance
    short_ok = price <= anchor and abs(gap) <= cfg.short_retest_tolerance
    return {
        "long_ok": 1.0 if long_ok else 0.0,
        "short_ok": 1.0 if short_ok else 0.0,
        "gap": gap,
    }


def regime_status(closes: List[float], cfg: Config) -> Dict[str, float]:
    idx = len(closes) - 1
    need = max(cfg.slow_ema, cfg.range_bb_period)
    if idx < need:
        return {"regime": "MIXED", "trend_strength": 0.0, "bandwidth": 0.0}

    ema_fast = ema_series(closes, cfg.fast_ema)
    ema_slow = ema_series(closes, cfg.slow_ema)
    bb_mid = sma_series(closes, cfg.range_bb_period)
    bb_std = stddev_series(closes, cfg.range_bb_period)
    price = closes[idx]
    mid = bb_mid[idx]
    std = bb_std[idx]
    upper = mid + cfg.range_bb_std * std
    lower = mid - cfg.range_bb_std * std
    trend_strength = abs(ema_fast[idx] - ema_slow[idx]) / price if price > 0 else 0.0
    bandwidth = (upper - lower) / price if price > 0 else 0.0

    regime = "MIXED"
    if (
        trend_strength >= cfg.regime_trend_strength
        and bandwidth >= cfg.regime_bandwidth_threshold
    ):
        regime = "TREND"
    elif trend_strength <= cfg.regime_range_strength:
        regime = "RANGE"

    return {
        "regime": regime,
        "trend_strength": trend_strength,
        "bandwidth": bandwidth,
    }


def range_signal(closes: List[float], flow_score: float, cfg: Config) -> Tuple[int, Dict[str, float]]:
    idx = len(closes) - 1
    need = max(cfg.range_bb_period, cfg.range_rsi_period + 1)
    if idx < need:
        return 0, {"rsi": 50.0, "band_pos": 0.0}

    bb_mid = sma_series(closes, cfg.range_bb_period)
    bb_std = stddev_series(closes, cfg.range_bb_period)
    rsi = rsi_series(closes, cfg.range_rsi_period)
    price = closes[idx]
    mid = bb_mid[idx]
    std = bb_std[idx]
    upper = mid + cfg.range_bb_std * std
    lower = mid - cfg.range_bb_std * std

    if upper <= lower or price <= 0:
        return 0, {"rsi": rsi[idx], "band_pos": 0.5}

    band_pos = (price - lower) / (upper - lower)
    long_touch = price <= lower * (1 + cfg.range_band_buffer)
    short_touch = price >= upper * (1 - cfg.range_band_buffer)
    long_ok_flow = flow_score >= -cfg.range_flow_filter
    short_ok_flow = flow_score <= cfg.range_flow_filter

    if long_touch and rsi[idx] <= cfg.range_rsi_long and long_ok_flow:
        return 1, {"rsi": rsi[idx], "band_pos": band_pos}
    if short_touch and rsi[idx] >= cfg.range_rsi_short and short_ok_flow:
        return -1, {"rsi": rsi[idx], "band_pos": band_pos}
    return 0, {"rsi": rsi[idx], "band_pos": band_pos}


def combined_signal(trend: int, flow: int, confirm_trend: int, retest: Dict[str, float]) -> str:
    long_retest = retest["long_ok"] > 0.5
    short_retest = retest["short_ok"] > 0.5
    if trend == 1 and flow == 1 and confirm_trend == 1 and long_retest:
        return "LONG"
    if trend == -1 and flow == -1 and confirm_trend == -1 and short_retest:
        return "SHORT"
    return "HOLD"


def combined_signal_by_regime(
    regime: str,
    trend: int,
    flow: int,
    confirm_trend: int,
    retest: Dict[str, float],
    range_sig: int,
) -> Tuple[str, str]:
    long_retest = retest["long_ok"] > 0.5
    short_retest = retest["short_ok"] > 0.5
    if regime == "TREND":
        if trend == 1 and flow == 1 and confirm_trend == 1 and long_retest:
            return "LONG", "TREND"
        if trend == -1 and flow == -1 and confirm_trend == -1 and short_retest:
            return "SHORT", "TREND"
        return "HOLD", "NONE"
    if regime == "RANGE":
        if range_sig == 1:
            return "LONG", "RANGE"
        if range_sig == -1:
            return "SHORT", "RANGE"
        return "HOLD", "NONE"

    # MIXED：优先趋势信号，其次震荡信号。
    if trend == 1 and flow == 1 and confirm_trend == 1 and long_retest:
        return "LONG", "TREND"
    if trend == -1 and flow == -1 and confirm_trend == -1 and short_retest:
        return "SHORT", "TREND"
    if range_sig == 1:
        return "LONG", "RANGE"
    if range_sig == -1:
        return "SHORT", "RANGE"
    return "HOLD", "NONE"


def profit_target_profile(cfg: Config) -> Dict[str, float]:
    notional = cfg.margin_usdt * cfg.leverage
    if notional <= 0:
        return {
            "required_move_pct": 0.0,
            "tp1_pct": cfg.tp1_pct,
            "tp2_pct": cfg.tp2_pct,
            "tp3_pct": cfg.tp3_pct,
        }

    roundtrip_fee = notional * max(cfg.estimated_fee_rate, 0.0) * 2
    required_move_pct = max(0.0, (cfg.min_profit_usdt + roundtrip_fee) / notional)
    tp1_pct = max(cfg.tp1_pct, required_move_pct * 0.6)
    tp2_pct = max(cfg.tp2_pct, required_move_pct)
    tp3_pct = max(cfg.tp3_pct, required_move_pct * 1.4)
    return {
        "required_move_pct": required_move_pct,
        "tp1_pct": tp1_pct,
        "tp2_pct": tp2_pct,
        "tp3_pct": tp3_pct,
    }


def recent_price_space_pct(closed_ohlcv: List[List[float]], lookback: int) -> float:
    if lookback <= 1 or len(closed_ohlcv) < lookback:
        return 0.0
    window = closed_ohlcv[-lookback:]
    highest = max(float(c[2]) for c in window)
    lowest = min(float(c[3]) for c in window)
    last_close = float(window[-1][4])
    if last_close <= 0:
        return 0.0
    return max(0.0, (highest - lowest) / last_close)


def fetch_position_state(exchange: ccxt.binance, market_id: str) -> Tuple[float, float]:
    data = exchange.fapiPrivateV2GetPositionRisk({"symbol": market_id})
    if isinstance(data, list):
        net = 0.0
        entry_price = 0.0
        chosen_abs = 0.0
        for row in data:
            if row.get("symbol") != market_id:
                continue
            amt = float(row.get("positionAmt", 0.0))
            side = str(row.get("positionSide", "BOTH")).upper()
            if side == "SHORT":
                signed = -abs(amt)
            else:
                signed = amt
            net += signed

            if abs(signed) > chosen_abs:
                chosen_abs = abs(signed)
                entry_price = float(row.get("entryPrice", 0.0))
        return net, entry_price
    return float(data.get("positionAmt", 0.0)), float(data.get("entryPrice", 0.0))


def sync_position_memory(
    memory: PositionMemory, position_amt: float, entry_price: float
) -> None:
    if abs(position_amt) < 1e-12:
        memory.reset()
        return

    side = "LONG" if position_amt > 0 else "SHORT"
    mode_hint = memory.entry_mode if memory.entry_mode in {"TREND", "RANGE"} else "TREND"
    entry_changed = (
        memory.entry_price <= 0
        or entry_price <= 0
        or abs(memory.entry_price - entry_price) / max(entry_price, 1.0) > 1e-5
    )
    if memory.side != side or entry_changed:
        memory.side = side
        memory.entry_mode = mode_hint
        memory.entry_price = entry_price
        memory.tp1_done = False
        memory.tp2_done = False
        memory.bars_held = 0
        base_price = entry_price if entry_price > 0 else 0.0
        memory.peak_price = base_price
        memory.trough_price = base_price
    else:
        memory.bars_held += 1


def fetch_last_price(exchange: ccxt.binance, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker.get("last") or ticker.get("mark") or ticker.get("close"))


def calc_order_amount(
    exchange: ccxt.binance,
    symbol: str,
    raw_amount: float,
    min_amount: float,
) -> float:
    precise = float(exchange.amount_to_precision(symbol, raw_amount))
    if precise < min_amount:
        precise = float(exchange.amount_to_precision(symbol, min_amount))
    return precise


def place_market_order(
    exchange: ccxt.binance,
    symbol: str,
    side: str,
    amount: float,
    reduce_only: bool,
    dry_run: bool,
) -> None:
    if amount <= 0:
        return
    params = {"reduceOnly": reduce_only}
    if dry_run:
        logging.info(
            "[DRY_RUN] create_order side=%s amount=%s reduceOnly=%s",
            side,
            amount,
            reduce_only,
        )
        return
    exchange.create_order(symbol, "market", side, amount, None, params)
    logging.info(
        "ORDER sent: side=%s amount=%s reduceOnly=%s", side, amount, reduce_only
    )


def place_limit_order(
    exchange: ccxt.binance,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    reduce_only: bool,
    tif: str,
    dry_run: bool,
) -> None:
    if amount <= 0 or price <= 0:
        return
    params = {"reduceOnly": reduce_only, "timeInForce": tif}
    if dry_run:
        logging.info(
            "[DRY_RUN] create_order type=limit side=%s amount=%s price=%s reduceOnly=%s tif=%s",
            side,
            amount,
            price,
            reduce_only,
            tif,
        )
        return
    exchange.create_order(symbol, "limit", side, amount, price, params)
    logging.info(
        "ORDER sent: type=limit side=%s amount=%s price=%.2f reduceOnly=%s tif=%s",
        side,
        amount,
        price,
        reduce_only,
        tif,
    )


def aggressive_limit_price(last_price: float, signal: str, offset_pct: float) -> float:
    offset = max(offset_pct, 0.0)
    if signal == "LONG":
        return last_price * (1 + offset)
    if signal == "SHORT":
        return last_price * max(0.0, 1 - offset)
    return last_price


def required_confirm_bars(regime: str, cfg: Config) -> int:
    base = max(cfg.entry_confirm_bars, 1)
    if regime == "RANGE":
        return base + max(cfg.entry_confirm_extra_range, 0)
    if regime == "MIXED":
        return base + max(cfg.entry_confirm_extra_mixed, 0)
    return base


def regime_position_scale(regime: str, cfg: Config) -> float:
    if regime == "RANGE":
        return cfg.regime_size_range_multiplier
    if regime == "MIXED":
        return cfg.regime_size_mixed_multiplier
    return cfg.regime_size_trend_multiplier


def orderbook_entry_allowed(
    signal: str,
    signal_mode: str,
    depth_imbalance: float,
    cfg: Config,
) -> bool:
    if signal not in {"LONG", "SHORT"}:
        return True
    if not cfg.orderbook_filter_enabled:
        return True
    if cfg.orderbook_filter_range_only and signal_mode != "RANGE":
        return True
    if signal == "LONG":
        return depth_imbalance >= cfg.orderbook_long_threshold
    return depth_imbalance <= -cfg.orderbook_short_threshold


def sentiment_entry_allowed(
    signal: str,
    signal_mode: str,
    sentiment_score: float,
    sentiment_source_count: float,
    cfg: Config,
) -> bool:
    if signal not in {"LONG", "SHORT"}:
        return True
    if not cfg.sentiment_filter_enabled:
        return True
    mode = cfg.sentiment_filter_mode
    if mode == "trend_only" and signal_mode != "TREND":
        return True
    if mode == "range_only" and signal_mode != "RANGE":
        return True
    if mode not in {"all", "trend_only", "range_only"}:
        return True
    if sentiment_source_count < max(cfg.sentiment_min_sources, 1):
        # 数据不足时 fail-open，避免网络抖动导致错失全部信号。
        return True
    if signal == "LONG":
        return sentiment_score >= cfg.sentiment_long_threshold
    return sentiment_score <= -cfg.sentiment_short_threshold


def volatility_position_scale(last_price: float, atr_value: float, cfg: Config) -> Tuple[float, str]:
    if not cfg.vol_guard_enabled or last_price <= 0:
        return 1.0, "NONE"
    atr_pct = (atr_value / last_price) if atr_value > 0 else 0.0
    if atr_pct >= cfg.vol_guard_atr_pct_hard:
        return cfg.vol_guard_hard_multiplier, "HARD"
    if atr_pct >= cfg.vol_guard_atr_pct_soft:
        return cfg.vol_guard_soft_multiplier, "SOFT"
    return 1.0, "NONE"


def auto_position_scale(
    signal: str,
    signal_mode: str,
    regime: str,
    trend_strength: float,
    flow_score: float,
    space_pct: float,
    required_space_pct: float,
    cfg: Config,
) -> Tuple[float, Dict[str, float]]:
    meta = {
        "flow_strength": 1.0,
        "regime_strength": 1.0,
        "space_strength": 1.0,
        "space_ratio": 1.0,
        "stop_adjust": 1.0,
        "composite": 1.0,
        "raw_scale": 1.0,
        "effective_stop_pct": cfg.stop_loss_pct,
    }
    if not cfg.auto_position_sizing_enabled:
        return 1.0, meta

    mode = signal_mode if signal_mode in {"TREND", "RANGE", "MIXED"} else regime
    mode = mode if mode in {"TREND", "RANGE", "MIXED"} else "MIXED"
    if signal not in {"LONG", "SHORT"}:
        mode = "MIXED"

    flow_ref = max(cfg.auto_size_flow_ref, 1e-9)
    trend_ref = max(cfg.auto_size_trend_ref, 1e-9)
    space_ref = max(cfg.auto_size_space_ref, 1e-9)

    flow_strength = clamp(abs(flow_score) / flow_ref, 0.0, 2.0)
    trend_strength_score = clamp(trend_strength / trend_ref, 0.0, 2.0)
    range_strength_score = clamp(cfg.regime_range_strength / max(trend_strength, 1e-9), 0.0, 2.0)
    if mode == "TREND":
        regime_strength = trend_strength_score
    elif mode == "RANGE":
        regime_strength = range_strength_score
    else:
        regime_strength = (trend_strength_score + range_strength_score) * 0.5

    space_ratio = 1.0
    if required_space_pct > 0:
        space_ratio = space_pct / required_space_pct
    space_strength = clamp(space_ratio / space_ref, 0.0, 2.0)

    # 固定权重：订单流强度 > 行情匹配度 > 可用价格空间。
    composite = 0.45 * flow_strength + 0.30 * regime_strength + 0.25 * space_strength
    raw_scale = 1.0 + cfg.auto_size_sensitivity * (composite - 1.0)

    effective_stop_pct = cfg.range_stop_loss_pct if mode == "RANGE" else cfg.stop_loss_pct
    stop_adjust = 1.0
    if cfg.auto_size_stop_adjust_enabled and effective_stop_pct > 0:
        stop_adjust = clamp(
            cfg.stop_loss_pct / effective_stop_pct,
            cfg.auto_size_stop_adjust_min,
            cfg.auto_size_stop_adjust_max,
        )

    scaled = raw_scale * stop_adjust
    final_scale = clamp(scaled, cfg.auto_size_min_multiplier, cfg.auto_size_max_multiplier)
    meta.update(
        {
            "flow_strength": flow_strength,
            "regime_strength": regime_strength,
            "space_strength": space_strength,
            "space_ratio": space_ratio,
            "stop_adjust": stop_adjust,
            "composite": composite,
            "raw_scale": raw_scale,
            "effective_stop_pct": effective_stop_pct,
        }
    )
    return final_scale, meta


def close_opposite_by_signal(
    exchange: ccxt.binance,
    symbol: str,
    signal: str,
    position_amt: float,
    dry_run: bool,
) -> bool:
    if signal == "LONG" and position_amt < 0:
        place_market_order(
            exchange,
            symbol,
            "buy",
            abs(position_amt),
            reduce_only=True,
            dry_run=dry_run,
        )
        return True
    if signal == "SHORT" and position_amt > 0:
        place_market_order(
            exchange,
            symbol,
            "sell",
            abs(position_amt),
            reduce_only=True,
            dry_run=dry_run,
        )
        return True
    return False


def entry_pullback_pct(signal: str, signal_mode: str, cfg: Config) -> float:
    mode = signal_mode if signal_mode in {"TREND", "RANGE"} else "TREND"
    if signal == "LONG":
        if mode == "RANGE":
            return max(cfg.entry_pullback_long_range_pct, 0.0)
        return max(cfg.entry_pullback_long_trend_pct, 0.0)
    if signal == "SHORT":
        if mode == "RANGE":
            return max(cfg.entry_pullback_short_range_pct, 0.0)
        return max(cfg.entry_pullback_short_trend_pct, 0.0)
    return 0.0


def entry_pullback_with_atr_pct(
    last_price: float, signal: str, signal_mode: str, cfg: Config, atr_value: float
) -> float:
    base = entry_pullback_pct(signal, signal_mode, cfg)
    if last_price <= 0:
        return base
    atr_pct = (atr_value / last_price) if atr_value > 0 else 0.0
    atr_component = atr_pct * max(cfg.entry_atr_multiplier, 0.0)
    dynamic = max(base, atr_component, cfg.entry_atr_min_pullback_pct)
    return min(dynamic, cfg.entry_atr_max_pullback_pct)


def entry_target_price(
    last_price: float, signal: str, signal_mode: str, cfg: Config, atr_value: float
) -> float:
    pullback = entry_pullback_with_atr_pct(last_price, signal, signal_mode, cfg, atr_value)
    if signal == "LONG":
        return last_price * max(0.0, 1 - pullback)
    if signal == "SHORT":
        return last_price * (1 + pullback)
    return last_price


def close_reduce_only(
    exchange: ccxt.binance,
    symbol: str,
    position_amt: float,
    close_amount: float,
    min_amount: float,
    dry_run: bool,
) -> float:
    abs_pos = abs(position_amt)
    if abs_pos <= 0:
        return 0.0

    target = min(abs_pos, close_amount)
    if target <= 0:
        return 0.0

    precise = float(exchange.amount_to_precision(symbol, target))
    if precise <= 0:
        precise = float(exchange.amount_to_precision(symbol, abs_pos))
    if precise < min_amount:
        precise = float(exchange.amount_to_precision(symbol, abs_pos))
    precise = min(abs_pos, precise)
    if precise <= 0:
        return 0.0

    close_side = "sell" if position_amt > 0 else "buy"
    place_market_order(
        exchange=exchange,
        symbol=symbol,
        side=close_side,
        amount=precise,
        reduce_only=True,
        dry_run=dry_run,
    )
    return precise


def manage_open_position(
    exchange: ccxt.binance,
    symbol: str,
    position_amt: float,
    entry_price: float,
    last_price: float,
    current_candle: List[float],
    previous_candle: List[float],
    flow: int,
    memory: PositionMemory,
    cfg: Config,
    min_amount: float,
    dry_run: bool,
) -> bool:
    if abs(position_amt) < 1e-12:
        return False

    side = "LONG" if position_amt > 0 else "SHORT"
    entry_mode = memory.entry_mode if memory.entry_mode in {"TREND", "RANGE"} else "TREND"
    targets = profit_target_profile(cfg)
    qty = abs(position_amt)
    if entry_price <= 0:
        entry_price = memory.entry_price if memory.entry_price > 0 else last_price
    if side == "LONG":
        if memory.peak_price <= 0:
            memory.peak_price = entry_price
        memory.peak_price = max(memory.peak_price, last_price)
    else:
        if memory.trough_price <= 0:
            memory.trough_price = entry_price
        memory.trough_price = min(memory.trough_price, last_price)

    mode_max_hold_bars = cfg.range_max_hold_bars if entry_mode == "RANGE" else cfg.max_hold_bars
    if mode_max_hold_bars > 0 and memory.bars_held >= mode_max_hold_bars:
        closed = close_reduce_only(
            exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
        )
        if closed > 0:
            logging.info(
                "触发时间止盈/止损：mode=%s 持仓K线=%s",
                entry_mode,
                memory.bars_held,
            )
            return True

    if entry_mode == "RANGE":
        if side == "LONG":
            stop_price = entry_price * (1 - cfg.range_stop_loss_pct)
            tp_price = entry_price * (1 + cfg.range_take_profit_pct)
            if last_price <= stop_price:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info(
                        "触发震荡多头止损：price=%.2f stop=%.2f",
                        last_price,
                        stop_price,
                    )
                    return True
            if last_price >= tp_price:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info(
                        "触发震荡多头止盈：price=%.2f tp=%.2f",
                        last_price,
                        tp_price,
                    )
                    return True
        else:
            stop_price = entry_price * (1 + cfg.range_stop_loss_pct)
            tp_price = entry_price * (1 - cfg.range_take_profit_pct)
            if last_price >= stop_price:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info(
                        "触发震荡空头止损：price=%.2f stop=%.2f",
                        last_price,
                        stop_price,
                    )
                    return True
            if last_price <= tp_price:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info(
                        "触发震荡空头止盈：price=%.2f tp=%.2f",
                        last_price,
                        tp_price,
                    )
                    return True
        return False

    if side == "LONG":
        stop_price = entry_price * (1 - cfg.stop_loss_pct)
        if memory.tp1_done:
            stop_price = max(stop_price, entry_price)
        tp1 = entry_price * (1 + targets["tp1_pct"])
        tp2 = entry_price * (1 + targets["tp2_pct"])

        if last_price <= stop_price:
            closed = close_reduce_only(
                exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
            )
            if closed > 0:
                logging.info("触发多头止损/保本：price=%.2f stop=%.2f", last_price, stop_price)
                return True

        if not memory.tp2_done and last_price >= tp2:
            ratio = cfg.tp2_close_ratio if memory.tp1_done else (
                cfg.tp1_close_ratio + cfg.tp2_close_ratio
            )
            closed = close_reduce_only(
                exchange,
                symbol,
                position_amt,
                qty * min(max(ratio, 0.0), 1.0),
                min_amount,
                dry_run=dry_run,
            )
            if closed > 0:
                memory.tp1_done = True
                memory.tp2_done = True
                logging.info("触发多头TP2分批止盈：close_qty=%.6f", closed)
                return True

        if not memory.tp1_done and last_price >= tp1:
            closed = close_reduce_only(
                exchange,
                symbol,
                position_amt,
                qty * min(max(cfg.tp1_close_ratio, 0.0), 1.0),
                min_amount,
                dry_run=dry_run,
            )
            if closed > 0:
                memory.tp1_done = True
                logging.info("触发多头TP1分批止盈并启用保本止损：close_qty=%.6f", closed)
                return True

        if cfg.runner_mode and memory.tp1_done and qty > 0:
            runup_pct = (
                (memory.peak_price - entry_price) / entry_price if entry_price > 0 else 0.0
            )
            trail_pct = 0.0
            if runup_pct >= targets["tp3_pct"]:
                trail_pct = cfg.trailing_stop_tight_pct
            elif runup_pct >= targets["tp2_pct"]:
                trail_pct = cfg.trailing_stop_pct
            elif runup_pct >= targets["tp1_pct"]:
                trail_pct = cfg.trailing_stop_loose_pct
            if trail_pct > 0:
                trail_stop = memory.peak_price * (1 - trail_pct)
                if last_price <= trail_stop:
                    closed = close_reduce_only(
                        exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                    )
                    if closed > 0:
                        logging.info(
                            "触发多头分段移动止盈：price=%.2f trail_stop=%.2f peak=%.2f trail=%.3f%%",
                            last_price,
                            trail_stop,
                            memory.peak_price,
                            trail_pct * 100,
                        )
                        return True
        elif not cfg.runner_mode:
            tp3 = entry_price * (1 + targets["tp3_pct"])
            if last_price >= tp3:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info("触发多头TP3：price=%.2f tp3=%.2f", last_price, tp3)
                    return True

        if (
            cfg.trend_reversal_exit_enabled
            and entry_price > 0
            and qty > 0
            and flow < 0
            and previous_candle
            and current_candle
        ):
            profit_pct = (last_price - entry_price) / entry_price
            cur_open = float(current_candle[1])
            cur_close = float(current_candle[4])
            prev_low = float(previous_candle[3])
            bearish_reversal = cur_close < cur_open and cur_close < prev_low
            if profit_pct >= cfg.trend_reversal_min_profit_pct and bearish_reversal:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info(
                        "触发多头反转K线退出：profit=%.3f%% close=%.2f prev_low=%.2f",
                        profit_pct * 100,
                        cur_close,
                        prev_low,
                    )
                    return True

    else:
        stop_price = entry_price * (1 + cfg.stop_loss_pct)
        if memory.tp1_done:
            stop_price = min(stop_price, entry_price)
        tp1 = entry_price * (1 - targets["tp1_pct"])
        tp2 = entry_price * (1 - targets["tp2_pct"])

        if last_price >= stop_price:
            closed = close_reduce_only(
                exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
            )
            if closed > 0:
                logging.info("触发空头止损/保本：price=%.2f stop=%.2f", last_price, stop_price)
                return True

        if not memory.tp2_done and last_price <= tp2:
            ratio = cfg.tp2_close_ratio if memory.tp1_done else (
                cfg.tp1_close_ratio + cfg.tp2_close_ratio
            )
            closed = close_reduce_only(
                exchange,
                symbol,
                position_amt,
                qty * min(max(ratio, 0.0), 1.0),
                min_amount,
                dry_run=dry_run,
            )
            if closed > 0:
                memory.tp1_done = True
                memory.tp2_done = True
                logging.info("触发空头TP2分批止盈：close_qty=%.6f", closed)
                return True

        if not memory.tp1_done and last_price <= tp1:
            closed = close_reduce_only(
                exchange,
                symbol,
                position_amt,
                qty * min(max(cfg.tp1_close_ratio, 0.0), 1.0),
                min_amount,
                dry_run=dry_run,
            )
            if closed > 0:
                memory.tp1_done = True
                logging.info("触发空头TP1分批止盈并启用保本止损：close_qty=%.6f", closed)
                return True

        if cfg.runner_mode and memory.tp1_done and qty > 0:
            runup_pct = (
                (entry_price - memory.trough_price) / entry_price if entry_price > 0 else 0.0
            )
            trail_pct = 0.0
            if runup_pct >= targets["tp3_pct"]:
                trail_pct = cfg.trailing_stop_tight_pct
            elif runup_pct >= targets["tp2_pct"]:
                trail_pct = cfg.trailing_stop_pct
            elif runup_pct >= targets["tp1_pct"]:
                trail_pct = cfg.trailing_stop_loose_pct
            if trail_pct > 0:
                trail_stop = memory.trough_price * (1 + trail_pct)
                if last_price >= trail_stop:
                    closed = close_reduce_only(
                        exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                    )
                    if closed > 0:
                        logging.info(
                            "触发空头分段移动止盈：price=%.2f trail_stop=%.2f trough=%.2f trail=%.3f%%",
                            last_price,
                            trail_stop,
                            memory.trough_price,
                            trail_pct * 100,
                        )
                        return True
        elif not cfg.runner_mode:
            tp3 = entry_price * (1 - targets["tp3_pct"])
            if last_price <= tp3:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info("触发空头TP3：price=%.2f tp3=%.2f", last_price, tp3)
                    return True

        if (
            cfg.trend_reversal_exit_enabled
            and entry_price > 0
            and qty > 0
            and flow > 0
            and previous_candle
            and current_candle
        ):
            profit_pct = (entry_price - last_price) / entry_price
            cur_open = float(current_candle[1])
            cur_close = float(current_candle[4])
            prev_high = float(previous_candle[2])
            bullish_reversal = cur_close > cur_open and cur_close > prev_high
            if profit_pct >= cfg.trend_reversal_min_profit_pct and bullish_reversal:
                closed = close_reduce_only(
                    exchange, symbol, position_amt, qty, min_amount, dry_run=dry_run
                )
                if closed > 0:
                    logging.info(
                        "触发空头反转K线退出：profit=%.3f%% close=%.2f prev_high=%.2f",
                        profit_pct * 100,
                        cur_close,
                        prev_high,
                    )
                    return True

    return False


def apply_signal(
    exchange: ccxt.binance,
    symbol: str,
    signal: str,
    position_amt: float,
    open_amount: float,
    open_limit_price: float,
    open_limit_tif: str,
    dry_run: bool,
) -> bool:
    if signal == "HOLD":
        logging.info("Signal HOLD, no order.")
        return False

    if signal == "LONG":
        if position_amt > 0:
            logging.info("Already LONG, skip.")
            return False
        if position_amt < 0:
            close_amt = abs(position_amt)
            place_market_order(
                exchange, symbol, "buy", close_amt, reduce_only=True, dry_run=dry_run
            )
        place_limit_order(
            exchange,
            symbol,
            "buy",
            open_amount,
            open_limit_price,
            reduce_only=False,
            tif=open_limit_tif,
            dry_run=dry_run,
        )
        return True

    if signal == "SHORT":
        if position_amt < 0:
            logging.info("Already SHORT, skip.")
            return False
        if position_amt > 0:
            close_amt = abs(position_amt)
            place_market_order(
                exchange, symbol, "sell", close_amt, reduce_only=True, dry_run=dry_run
            )
        place_limit_order(
            exchange,
            symbol,
            "sell",
            open_amount,
            open_limit_price,
            reduce_only=False,
            tif=open_limit_tif,
            dry_run=dry_run,
        )
        return True
    return False


def validate_config(cfg: Config) -> None:
    if cfg.fast_ema >= cfg.slow_ema:
        raise ValueError("FAST_EMA must be smaller than SLOW_EMA")
    if cfg.fast_ma >= cfg.slow_ma:
        raise ValueError("FAST_MA must be smaller than SLOW_MA")
    if cfg.slow_ma >= cfg.ma_period:
        raise ValueError("SLOW_MA must be smaller than MA_PERIOD")
    if cfg.fast_ema <= 1 or cfg.slow_ema <= 1:
        raise ValueError("FAST_EMA and SLOW_EMA must be > 1")
    if cfg.fast_ma <= 1 or cfg.slow_ma <= 1:
        raise ValueError("FAST_MA and SLOW_MA must be > 1")
    if cfg.ma_period <= 1:
        raise ValueError("MA_PERIOD must be > 1")
    if cfg.long_flow_threshold <= 0 or cfg.short_flow_threshold <= 0:
        raise ValueError("LONG_FLOW_THRESHOLD and SHORT_FLOW_THRESHOLD must be > 0")
    if cfg.sentiment_filter_mode not in {"all", "trend_only", "range_only"}:
        raise ValueError("SENTIMENT_FILTER_MODE must be one of: all / trend_only / range_only")
    if cfg.sentiment_period not in {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}:
        raise ValueError("SENTIMENT_PERIOD must be one of Binance futures data periods")
    if cfg.sentiment_lookback < 2 or cfg.sentiment_lookback > 500:
        raise ValueError("SENTIMENT_LOOKBACK must be in [2, 500]")
    if cfg.sentiment_min_sources < 1 or cfg.sentiment_min_sources > 4:
        raise ValueError("SENTIMENT_MIN_SOURCES must be in [1, 4]")
    if cfg.sentiment_long_threshold < -1 or cfg.sentiment_long_threshold > 1:
        raise ValueError("SENTIMENT_LONG_THRESHOLD must be in [-1, 1]")
    if cfg.sentiment_short_threshold < -1 or cfg.sentiment_short_threshold > 1:
        raise ValueError("SENTIMENT_SHORT_THRESHOLD must be in [-1, 1]")
    if cfg.sentiment_oi_scale_pct <= 0 or cfg.sentiment_oi_scale_pct >= 1:
        raise ValueError("SENTIMENT_OI_SCALE_PCT must be in (0, 1)")
    if cfg.sentiment_funding_scale <= 0 or cfg.sentiment_funding_scale >= 0.05:
        raise ValueError("SENTIMENT_FUNDING_SCALE must be in (0, 0.05)")
    if cfg.sentiment_long_short_source not in {"global", "top_account", "top_position"}:
        raise ValueError("SENTIMENT_LONG_SHORT_SOURCE must be: global / top_account / top_position")
    for w_name, w in {
        "SENTIMENT_OI_WEIGHT": cfg.sentiment_open_interest_weight,
        "SENTIMENT_LONG_SHORT_WEIGHT": cfg.sentiment_long_short_weight,
        "SENTIMENT_TAKER_RATIO_WEIGHT": cfg.sentiment_taker_ratio_weight,
        "SENTIMENT_FUNDING_WEIGHT": cfg.sentiment_funding_weight,
    }.items():
        if w < 0:
            raise ValueError(f"{w_name} must be >= 0")
    if (
        cfg.sentiment_open_interest_weight
        + cfg.sentiment_long_short_weight
        + cfg.sentiment_taker_ratio_weight
        + cfg.sentiment_funding_weight
    ) <= 0:
        raise ValueError("At least one sentiment weight must be > 0")
    if cfg.orderbook_long_threshold < 0 or cfg.orderbook_long_threshold >= 1:
        raise ValueError("ORDERBOOK_LONG_THRESHOLD must be in [0, 1)")
    if cfg.orderbook_short_threshold < 0 or cfg.orderbook_short_threshold >= 1:
        raise ValueError("ORDERBOOK_SHORT_THRESHOLD must be in [0, 1)")
    if cfg.vol_guard_atr_period < 2:
        raise ValueError("VOL_GUARD_ATR_PERIOD must be >= 2")
    if cfg.vol_guard_atr_pct_soft <= 0 or cfg.vol_guard_atr_pct_soft >= 0.2:
        raise ValueError("VOL_GUARD_ATR_PCT_SOFT must be in (0, 0.2)")
    if cfg.vol_guard_atr_pct_hard <= 0 or cfg.vol_guard_atr_pct_hard >= 0.3:
        raise ValueError("VOL_GUARD_ATR_PCT_HARD must be in (0, 0.3)")
    if cfg.vol_guard_atr_pct_hard < cfg.vol_guard_atr_pct_soft:
        raise ValueError("VOL_GUARD_ATR_PCT_HARD should be >= VOL_GUARD_ATR_PCT_SOFT")
    if cfg.vol_guard_soft_multiplier <= 0 or cfg.vol_guard_soft_multiplier > 1:
        raise ValueError("VOL_GUARD_SOFT_MULTIPLIER must be in (0, 1]")
    if cfg.vol_guard_hard_multiplier <= 0 or cfg.vol_guard_hard_multiplier > 1:
        raise ValueError("VOL_GUARD_HARD_MULTIPLIER must be in (0, 1]")
    if cfg.vol_guard_hard_multiplier > cfg.vol_guard_soft_multiplier:
        raise ValueError("VOL_GUARD_HARD_MULTIPLIER should be <= VOL_GUARD_SOFT_MULTIPLIER")
    if cfg.long_trend_spread_threshold <= 0 or cfg.short_trend_spread_threshold <= 0:
        raise ValueError("LONG_TREND_SPREAD_THRESHOLD and SHORT_TREND_SPREAD_THRESHOLD must be > 0")
    if cfg.long_trend_spread_threshold < cfg.short_trend_spread_threshold:
        raise ValueError("LONG_TREND_SPREAD_THRESHOLD should be >= SHORT_TREND_SPREAD_THRESHOLD")
    if cfg.long_retest_tolerance < 0 or cfg.short_retest_tolerance < 0:
        raise ValueError("LONG_RETEST_TOLERANCE and SHORT_RETEST_TOLERANCE must be >= 0")
    if cfg.long_retest_tolerance > cfg.short_retest_tolerance:
        raise ValueError("LONG_RETEST_TOLERANCE should be <= SHORT_RETEST_TOLERANCE")
    if cfg.regime_range_strength <= 0 or cfg.regime_trend_strength <= 0:
        raise ValueError("REGIME_RANGE_STRENGTH and REGIME_TREND_STRENGTH must be > 0")
    if cfg.regime_range_strength >= cfg.regime_trend_strength:
        raise ValueError("REGIME_RANGE_STRENGTH should be < REGIME_TREND_STRENGTH")
    if cfg.regime_bandwidth_threshold <= 0:
        raise ValueError("REGIME_BANDWIDTH_THRESHOLD must be > 0")
    if cfg.range_bb_period < 10:
        raise ValueError("RANGE_BB_PERIOD should be >= 10")
    if cfg.range_bb_std <= 0:
        raise ValueError("RANGE_BB_STD must be > 0")
    if cfg.range_rsi_period < 5:
        raise ValueError("RANGE_RSI_PERIOD should be >= 5")
    if not (0 < cfg.range_rsi_long < cfg.range_rsi_short < 100):
        raise ValueError("RANGE_RSI_LONG < RANGE_RSI_SHORT and both in (0,100)")
    if cfg.range_band_buffer < 0:
        raise ValueError("RANGE_BAND_BUFFER must be >= 0")
    if cfg.range_flow_filter < 0:
        raise ValueError("RANGE_FLOW_FILTER must be >= 0")
    if cfg.confirm_klines_limit < 50:
        raise ValueError("CONFIRM_KLINES_LIMIT should be >= 50")
    if cfg.stop_loss_pct <= 0:
        raise ValueError("STOP_LOSS_PCT must be > 0")
    if not (0 < cfg.tp1_pct < cfg.tp2_pct):
        raise ValueError("TP1_PCT < TP2_PCT must hold and all > 0")
    if cfg.tp3_pct <= 0:
        raise ValueError("TP3_PCT must be > 0")
    if not cfg.runner_mode and cfg.tp2_pct >= cfg.tp3_pct:
        raise ValueError("When RUNNER_MODE=false, TP2_PCT must be < TP3_PCT")
    if not (0 < cfg.tp1_close_ratio <= 1):
        raise ValueError("TP1_CLOSE_RATIO must be in (0, 1]")
    if not (0 < cfg.tp2_close_ratio <= 1):
        raise ValueError("TP2_CLOSE_RATIO must be in (0, 1]")
    if cfg.tp1_close_ratio + cfg.tp2_close_ratio > 1:
        raise ValueError("TP1_CLOSE_RATIO + TP2_CLOSE_RATIO must be <= 1")
    if cfg.max_hold_bars < 0:
        raise ValueError("MAX_HOLD_BARS must be >= 0")
    if cfg.min_profit_usdt < 0:
        raise ValueError("MIN_PROFIT_USDT must be >= 0")
    if cfg.estimated_fee_rate < 0:
        raise ValueError("ESTIMATED_FEE_RATE must be >= 0")
    if cfg.profit_space_lookback < 10:
        raise ValueError("PROFIT_SPACE_LOOKBACK should be >= 10")
    if cfg.profit_space_buffer <= 0:
        raise ValueError("PROFIT_SPACE_BUFFER must be > 0")
    if cfg.trailing_stop_loose_pct <= 0 or cfg.trailing_stop_loose_pct >= 1:
        raise ValueError("TRAILING_STOP_LOOSE_PCT must be in (0, 1)")
    if cfg.trailing_stop_pct <= 0 or cfg.trailing_stop_pct >= 1:
        raise ValueError("TRAILING_STOP_PCT must be in (0, 1)")
    if cfg.trailing_stop_tight_pct <= 0 or cfg.trailing_stop_tight_pct >= 1:
        raise ValueError("TRAILING_STOP_TIGHT_PCT must be in (0, 1)")
    if not (cfg.trailing_stop_loose_pct >= cfg.trailing_stop_pct >= cfg.trailing_stop_tight_pct):
        raise ValueError(
            "Need TRAILING_STOP_LOOSE_PCT >= TRAILING_STOP_PCT >= TRAILING_STOP_TIGHT_PCT"
        )
    if cfg.trend_reversal_min_profit_pct < 0 or cfg.trend_reversal_min_profit_pct >= 0.2:
        raise ValueError("TREND_REVERSAL_MIN_PROFIT_PCT must be in [0, 0.2)")
    if cfg.range_stop_loss_pct <= 0 or cfg.range_stop_loss_pct >= 1:
        raise ValueError("RANGE_STOP_LOSS_PCT must be in (0, 1)")
    if cfg.range_take_profit_pct <= 0 or cfg.range_take_profit_pct >= 1:
        raise ValueError("RANGE_TAKE_PROFIT_PCT must be in (0, 1)")
    if cfg.range_max_hold_bars < 0:
        raise ValueError("RANGE_MAX_HOLD_BARS must be >= 0")
    if cfg.entry_mode not in {"market", "pullback_limit"}:
        raise ValueError("ENTRY_MODE must be 'market' or 'pullback_limit'")
    if cfg.entry_confirm_bars < 1:
        raise ValueError("ENTRY_CONFIRM_BARS must be >= 1")
    if cfg.entry_confirm_extra_mixed < 0:
        raise ValueError("ENTRY_CONFIRM_EXTRA_MIXED must be >= 0")
    if cfg.entry_confirm_extra_range < 0:
        raise ValueError("ENTRY_CONFIRM_EXTRA_RANGE must be >= 0")
    if cfg.entry_cooldown_bars < 0:
        raise ValueError("ENTRY_COOLDOWN_BARS must be >= 0")
    if cfg.entry_atr_period < 2:
        raise ValueError("ENTRY_ATR_PERIOD must be >= 2")
    if cfg.entry_atr_multiplier < 0:
        raise ValueError("ENTRY_ATR_MULTIPLIER must be >= 0")
    if cfg.entry_atr_min_pullback_pct < 0 or cfg.entry_atr_min_pullback_pct >= 0.1:
        raise ValueError("ENTRY_ATR_MIN_PULLBACK_PCT must be in [0, 0.1)")
    if cfg.entry_atr_max_pullback_pct <= 0 or cfg.entry_atr_max_pullback_pct >= 0.2:
        raise ValueError("ENTRY_ATR_MAX_PULLBACK_PCT must be in (0, 0.2)")
    if cfg.entry_atr_min_pullback_pct > cfg.entry_atr_max_pullback_pct:
        raise ValueError("ENTRY_ATR_MIN_PULLBACK_PCT should be <= ENTRY_ATR_MAX_PULLBACK_PCT")
    if cfg.entry_pullback_long_trend_pct < 0 or cfg.entry_pullback_long_trend_pct >= 0.1:
        raise ValueError("ENTRY_PULLBACK_LONG_TREND_PCT must be in [0, 0.1)")
    if cfg.entry_pullback_short_trend_pct < 0 or cfg.entry_pullback_short_trend_pct >= 0.1:
        raise ValueError("ENTRY_PULLBACK_SHORT_TREND_PCT must be in [0, 0.1)")
    if cfg.entry_pullback_long_range_pct < 0 or cfg.entry_pullback_long_range_pct >= 0.1:
        raise ValueError("ENTRY_PULLBACK_LONG_RANGE_PCT must be in [0, 0.1)")
    if cfg.entry_pullback_short_range_pct < 0 or cfg.entry_pullback_short_range_pct >= 0.1:
        raise ValueError("ENTRY_PULLBACK_SHORT_RANGE_PCT must be in [0, 0.1)")
    if cfg.entry_max_wait_bars < 0:
        raise ValueError("ENTRY_MAX_WAIT_BARS must be >= 0")
    if cfg.entry_limit_tif not in {"GTC", "IOC", "FOK"}:
        raise ValueError("ENTRY_LIMIT_TIF must be one of: GTC / IOC / FOK")
    if cfg.entry_limit_offset_pct < 0 or cfg.entry_limit_offset_pct >= 0.05:
        raise ValueError("ENTRY_LIMIT_OFFSET_PCT must be in [0, 0.05)")
    if cfg.regime_size_trend_multiplier <= 0 or cfg.regime_size_trend_multiplier > 1:
        raise ValueError("REGIME_SIZE_TREND_MULTIPLIER must be in (0, 1]")
    if cfg.regime_size_mixed_multiplier <= 0 or cfg.regime_size_mixed_multiplier > 1:
        raise ValueError("REGIME_SIZE_MIXED_MULTIPLIER must be in (0, 1]")
    if cfg.regime_size_range_multiplier <= 0 or cfg.regime_size_range_multiplier > 1:
        raise ValueError("REGIME_SIZE_RANGE_MULTIPLIER must be in (0, 1]")
    if cfg.regime_size_mixed_multiplier > cfg.regime_size_trend_multiplier:
        raise ValueError(
            "REGIME_SIZE_MIXED_MULTIPLIER should be <= REGIME_SIZE_TREND_MULTIPLIER"
        )
    if cfg.regime_size_range_multiplier > cfg.regime_size_mixed_multiplier:
        raise ValueError(
            "REGIME_SIZE_RANGE_MULTIPLIER should be <= REGIME_SIZE_MIXED_MULTIPLIER"
        )
    if cfg.auto_size_min_multiplier <= 0:
        raise ValueError("AUTO_SIZE_MIN_MULTIPLIER must be > 0")
    if cfg.auto_size_max_multiplier <= 0:
        raise ValueError("AUTO_SIZE_MAX_MULTIPLIER must be > 0")
    if cfg.auto_size_min_multiplier > cfg.auto_size_max_multiplier:
        raise ValueError("AUTO_SIZE_MIN_MULTIPLIER must be <= AUTO_SIZE_MAX_MULTIPLIER")
    if cfg.auto_size_sensitivity < 0 or cfg.auto_size_sensitivity > 2:
        raise ValueError("AUTO_SIZE_SENSITIVITY must be in [0, 2]")
    if cfg.auto_size_flow_ref <= 0 or cfg.auto_size_flow_ref > 5:
        raise ValueError("AUTO_SIZE_FLOW_REF must be in (0, 5]")
    if cfg.auto_size_trend_ref <= 0 or cfg.auto_size_trend_ref >= 0.1:
        raise ValueError("AUTO_SIZE_TREND_REF must be in (0, 0.1)")
    if cfg.auto_size_space_ref <= 0 or cfg.auto_size_space_ref > 10:
        raise ValueError("AUTO_SIZE_SPACE_REF must be in (0, 10]")
    if cfg.auto_size_stop_adjust_min <= 0:
        raise ValueError("AUTO_SIZE_STOP_ADJUST_MIN must be > 0")
    if cfg.auto_size_stop_adjust_max <= 0:
        raise ValueError("AUTO_SIZE_STOP_ADJUST_MAX must be > 0")
    if cfg.auto_size_stop_adjust_min > cfg.auto_size_stop_adjust_max:
        raise ValueError("AUTO_SIZE_STOP_ADJUST_MIN must be <= AUTO_SIZE_STOP_ADJUST_MAX")
    if cfg.leverage <= 0:
        raise ValueError("LEVERAGE must be > 0")
    if cfg.margin_usdt <= 0:
        raise ValueError("MARGIN_USDT must be > 0")
    if cfg.poll_seconds < 5:
        raise ValueError("POLL_SECONDS should be >= 5")
    if cfg.margin_mode not in {"isolated", "cross"}:
        raise ValueError("MARGIN_MODE must be 'isolated' or 'cross'")
    if not cfg.api_key or not cfg.api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required")


def main() -> None:
    load_dotenv()
    setup_logging()
    cfg = Config.from_env()
    validate_config(cfg)

    exchange = create_exchange(cfg)
    markets = exchange.load_markets()
    if cfg.symbol not in markets:
        raise ValueError(f"Symbol not found: {cfg.symbol}")

    market = exchange.market(cfg.symbol)
    market_symbol = market["symbol"]
    market_id = market["id"]
    min_amount = float((market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.001)

    logging.info(
        (
            "Start bot | symbol=%s leverage=%sx mode=%s hedge=%s timeframe=%s "
            "confirm_tf=%s dry_run=%s min_profit=%.2fU runner=%s "
            "trail(L/M/T)=%.2f%%/%.2f%%/%.2f%% rev_exit=%s@%.2f%% "
            "entry=%s confirm=%s wait=%s pullback(TL/TS/RL/RS)=%.3f%%/%.3f%%/%.3f%%/%.3f%% "
            "confirm_extra(MIXED/RANGE)=%s/%s cooldown=%s "
            "entry_atr(period/mul/min/max)=%s/%.2f/%.3f%%/%.3f%% "
            "open_limit(tif/offset)=%s/%.3f%% "
            "ob_filter=%s(range_only=%s L/S=%.3f/%.3f) "
            "sent_filter=%s(mode=%s period=%s src=%s min_src=%s thr=%.3f/%.3f w[oi/ls/tk/fr]=%.2f/%.2f/%.2f/%.2f) "
            "regime_size(T/M/R)=%.2f/%.2f/%.2f "
            "trend_thr(L/S)=%.4f%%/%.4f%% retest(L/S)=%.3f%%/%.3f%% "
            "regime(trend/range/bw)=%.4f%%/%.4f%%/%.2f%% "
            "auto_size=%s(min/max=%.2f/%.2f sens=%.2f ref[f/t/s]=%.3f/%.4f/%.2f stop_adj=%s@%.2f~%.2f)"
        ),
        market_symbol,
        cfg.leverage,
        cfg.margin_mode,
        cfg.hedge_mode,
        cfg.timeframe,
        cfg.confirm_timeframe,
        cfg.dry_run,
        cfg.min_profit_usdt,
        cfg.runner_mode,
        cfg.trailing_stop_loose_pct * 100,
        cfg.trailing_stop_pct * 100,
        cfg.trailing_stop_tight_pct * 100,
        cfg.trend_reversal_exit_enabled,
        cfg.trend_reversal_min_profit_pct * 100,
        cfg.entry_mode,
        cfg.entry_confirm_bars,
        cfg.entry_max_wait_bars,
        cfg.entry_pullback_long_trend_pct * 100,
        cfg.entry_pullback_short_trend_pct * 100,
        cfg.entry_pullback_long_range_pct * 100,
        cfg.entry_pullback_short_range_pct * 100,
        cfg.entry_confirm_extra_mixed,
        cfg.entry_confirm_extra_range,
        cfg.entry_cooldown_bars,
        cfg.entry_atr_period,
        cfg.entry_atr_multiplier,
        cfg.entry_atr_min_pullback_pct * 100,
        cfg.entry_atr_max_pullback_pct * 100,
        cfg.entry_limit_tif,
        cfg.entry_limit_offset_pct * 100,
        cfg.orderbook_filter_enabled,
        cfg.orderbook_filter_range_only,
        cfg.orderbook_long_threshold,
        cfg.orderbook_short_threshold,
        cfg.sentiment_filter_enabled,
        cfg.sentiment_filter_mode,
        cfg.sentiment_period,
        cfg.sentiment_long_short_source,
        cfg.sentiment_min_sources,
        cfg.sentiment_long_threshold,
        cfg.sentiment_short_threshold,
        cfg.sentiment_open_interest_weight,
        cfg.sentiment_long_short_weight,
        cfg.sentiment_taker_ratio_weight,
        cfg.sentiment_funding_weight,
        cfg.regime_size_trend_multiplier,
        cfg.regime_size_mixed_multiplier,
        cfg.regime_size_range_multiplier,
        cfg.long_trend_spread_threshold * 100,
        cfg.short_trend_spread_threshold * 100,
        cfg.long_retest_tolerance * 100,
        cfg.short_retest_tolerance * 100,
        cfg.regime_trend_strength * 100,
        cfg.regime_range_strength * 100,
        cfg.regime_bandwidth_threshold * 100,
        cfg.auto_position_sizing_enabled,
        cfg.auto_size_min_multiplier,
        cfg.auto_size_max_multiplier,
        cfg.auto_size_sensitivity,
        cfg.auto_size_flow_ref,
        cfg.auto_size_trend_ref,
        cfg.auto_size_space_ref,
        cfg.auto_size_stop_adjust_enabled,
        cfg.auto_size_stop_adjust_min,
        cfg.auto_size_stop_adjust_max,
    )

    if not cfg.dry_run:
        exchange.set_position_mode(cfg.hedge_mode, market_symbol)
        exchange.set_margin_mode(cfg.margin_mode, market_symbol)
        exchange.set_leverage(cfg.leverage, market_symbol)
        logging.info("Position mode, leverage and margin mode configured on Binance.")
    else:
        logging.info("DRY_RUN enabled: no real orders will be sent.")

    last_candle_ts = None
    position_memory = PositionMemory()
    pending_entry = PendingEntryMemory()
    last_entry_signal = "HOLD"
    last_entry_mode = "NONE"
    signal_streak = 0
    cooldown_bars_remaining = 0
    cooldown_skipped_signals = 0
    orderbook_skipped_signals = 0
    sentiment_skipped_signals = 0
    last_had_position = False

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                market_symbol,
                timeframe=cfg.timeframe,
                limit=cfg.klines_limit,
            )
            trend_required = max(
                cfg.slow_ema,
                cfg.slow_ma,
                cfg.ma_period,
                cfg.fast_ema,
                cfg.fast_ma,
                cfg.entry_atr_period,
                cfg.vol_guard_atr_period,
            )
            if len(ohlcv) < trend_required + 2:
                logging.warning("Not enough K-lines yet, wait next cycle.")
                time.sleep(cfg.poll_seconds)
                continue

            # 仅使用已收盘 K 线（丢弃当前尚未走完的 K 线）。
            closed = ohlcv[:-1]
            candle_ts = int(closed[-1][0])
            if candle_ts == last_candle_ts:
                time.sleep(cfg.poll_seconds)
                continue
            last_candle_ts = candle_ts

            closes = [float(c[4]) for c in closed]
            trend, trend_meta = trend_signal(closes, cfg)
            metrics = order_flow_metrics(exchange, market_id, cfg)
            sentiment = market_sentiment_metrics(exchange, market_id, cfg)
            flow = flow_signal(metrics, cfg)
            confirm_trend, _ = confirm_trend_signal(exchange, market_symbol, cfg)
            retest = retest_status(closes, cfg)
            regime = regime_status(closes, cfg)
            range_sig, range_meta = range_signal(closes, metrics["flow_score"], cfg)
            signal, signal_mode = combined_signal_by_regime(
                regime=regime["regime"],
                trend=trend,
                flow=flow,
                confirm_trend=confirm_trend,
                retest=retest,
                range_sig=range_sig,
            )
            raw_signal = signal
            targets = profit_target_profile(cfg)
            required_space_pct = targets["required_move_pct"] * cfg.profit_space_buffer
            space_pct = recent_price_space_pct(closed, cfg.profit_space_lookback)
            space_ok = required_space_pct <= 0 or space_pct >= required_space_pct
            if signal in {"LONG", "SHORT"} and not space_ok:
                signal = "HOLD"
                signal_mode = "NONE"
            atr_value = atr_from_ohlcv(closed, cfg.entry_atr_period)
            vol_atr_value = atr_from_ohlcv(closed, cfg.vol_guard_atr_period)

            if signal in {"LONG", "SHORT"}:
                if signal == last_entry_signal and signal_mode == last_entry_mode:
                    signal_streak += 1
                else:
                    signal_streak = 1
                    last_entry_signal = signal
                    last_entry_mode = signal_mode
            else:
                signal_streak = 0
                last_entry_signal = "HOLD"
                last_entry_mode = "NONE"

            last_price = fetch_last_price(exchange, market_symbol)
            if cfg.dry_run:
                position_amt = 0.0
                entry_price = 0.0
            else:
                position_amt, entry_price = fetch_position_state(exchange, market_id)
            sync_position_memory(position_memory, position_amt, entry_price)
            had_position = abs(position_amt) > 1e-12
            if last_had_position and not had_position and cfg.entry_cooldown_bars > 0:
                cooldown_bars_remaining = max(cooldown_bars_remaining, cfg.entry_cooldown_bars)
            cooldown_active = False
            if not had_position and cooldown_bars_remaining > 0:
                cooldown_active = True
                cooldown_bars_remaining -= 1
                if pending_entry.active():
                    pending_entry.clear()
            if had_position and pending_entry.active():
                pending_entry.clear()
            regime_confirm_bars = required_confirm_bars(regime["regime"], cfg)
            entry_signal = (
                signal if signal_streak >= regime_confirm_bars else "HOLD"
            )
            if entry_signal in {"LONG", "SHORT"} and cooldown_active:
                cooldown_skipped_signals += 1
                entry_signal = "HOLD"
            entry_signal_mode = signal_mode if entry_signal in {"LONG", "SHORT"} else "NONE"
            orderbook_blocked = False
            sentiment_blocked = False
            if entry_signal in {"LONG", "SHORT"} and not orderbook_entry_allowed(
                entry_signal,
                entry_signal_mode,
                metrics["depth_imbalance"],
                cfg,
            ):
                orderbook_skipped_signals += 1
                entry_signal = "HOLD"
                entry_signal_mode = "NONE"
                orderbook_blocked = True
            if entry_signal in {"LONG", "SHORT"} and not sentiment_entry_allowed(
                signal=entry_signal,
                signal_mode=entry_signal_mode,
                sentiment_score=sentiment["sentiment_score"],
                sentiment_source_count=sentiment["sentiment_source_count"],
                cfg=cfg,
            ):
                sentiment_skipped_signals += 1
                entry_signal = "HOLD"
                entry_signal_mode = "NONE"
                sentiment_blocked = True

            size_regime = regime["regime"]
            if not had_position and pending_entry.active() and pending_entry.regime in {
                "TREND",
                "RANGE",
                "MIXED",
            }:
                size_regime = pending_entry.regime
            size_signal = entry_signal
            size_signal_mode = entry_signal_mode
            if (
                size_signal not in {"LONG", "SHORT"}
                and not had_position
                and pending_entry.active()
            ):
                size_signal = pending_entry.signal
                if pending_entry.mode in {"TREND", "RANGE", "MIXED"}:
                    size_signal_mode = pending_entry.mode
                elif size_regime in {"TREND", "RANGE", "MIXED"}:
                    size_signal_mode = size_regime
            vol_scale, vol_level = volatility_position_scale(last_price, vol_atr_value, cfg)
            regime_scale = regime_position_scale(size_regime, cfg)
            auto_scale, auto_meta = auto_position_scale(
                signal=size_signal,
                signal_mode=size_signal_mode,
                regime=size_regime,
                trend_strength=regime["trend_strength"],
                flow_score=metrics["flow_score"],
                space_pct=space_pct,
                required_space_pct=required_space_pct,
                cfg=cfg,
            )
            target_notional = (
                cfg.margin_usdt * cfg.leverage * vol_scale * regime_scale * auto_scale
            )
            if target_notional > 0 and last_price > 0:
                raw_amount = target_notional / last_price
                open_amount = calc_order_amount(
                    exchange, market_symbol, raw_amount, min_amount
                )
            else:
                open_amount = 0.0
            last_had_position = had_position

            logging.info(
                (
                    "price=%.2f trend=%s(ema_spread=%.4f%% ema_slope=%.4f%% "
                    "ma_spread=%.4f%% ma7_slope=%.4f%% ma99_gap=%.4f%% ma99_slope=%.4f%%) "
                    "confirm=%s retest_gap=%.4f%% flow=%s(score=%.3f trade_imb=%.3f depth_imb=%.3f large=%s) "
                    "ob_filter=%s(blocked=%s skip=%s) "
                    "sent=%.3f(src=%s blocked=%s skip=%s oi=%.3f%% lsr=%.3f tk=%.3f fr=%.4f%%) "
                    "regime=%s(str=%.4f%% bw=%.2f%% rsi=%.1f band_pos=%.2f range_sig=%s mode=%s) "
                    "space=%.2f%%/req=%.2f%% tp=%.2f%%/%.2f%%/%.2f%% trail=%.2f%% raw=%s signal=%s "
                    "confirm=%s/%s cd=%s(skip=%s) atr_entry=%.3f%% atr_vol=%.3f%% "
                    "vol_scale=%.2f(%s) size_regime=%s reg_scale=%.2f "
                    "auto_scale=%.2f(sig=%s mode=%s flow/reg/space=%.2f/%.2f/%.2f ratio=%.2f stop=%.2f) "
                    "position=%.6f entry=%.2f bars_held=%s open_qty=%.6f pending=%s@%.2f(wait=%s)"
                ),
                last_price,
                trend,
                trend_meta["ema_spread"] * 100,
                trend_meta["ema_slope"] * 100,
                trend_meta["ma_spread"] * 100,
                trend_meta["ma_fast_slope"] * 100,
                trend_meta["ma_long_gap"] * 100,
                trend_meta["ma_long_slope"] * 100,
                confirm_trend,
                retest["gap"] * 100,
                flow,
                metrics["flow_score"],
                metrics["trade_imbalance"],
                metrics["depth_imbalance"],
                int(metrics["large_count"]),
                cfg.orderbook_filter_enabled,
                orderbook_blocked,
                orderbook_skipped_signals,
                sentiment["sentiment_score"],
                int(sentiment["sentiment_source_count"]),
                sentiment_blocked,
                sentiment_skipped_signals,
                sentiment["oi_change"] * 100,
                sentiment["long_short_ratio"],
                sentiment["taker_buy_sell_ratio"],
                sentiment["funding_rate"] * 100,
                regime["regime"],
                regime["trend_strength"] * 100,
                regime["bandwidth"] * 100,
                range_meta["rsi"],
                range_meta["band_pos"],
                range_sig,
                signal_mode,
                space_pct * 100,
                required_space_pct * 100,
                targets["tp1_pct"] * 100,
                targets["tp2_pct"] * 100,
                targets["tp3_pct"] * 100,
                cfg.trailing_stop_pct * 100,
                raw_signal,
                entry_signal,
                signal_streak,
                regime_confirm_bars,
                cooldown_bars_remaining,
                cooldown_skipped_signals,
                (atr_value / last_price * 100) if last_price > 0 else 0.0,
                (vol_atr_value / last_price * 100) if last_price > 0 else 0.0,
                vol_scale,
                vol_level,
                size_regime,
                regime_scale,
                auto_scale,
                size_signal,
                size_signal_mode,
                auto_meta["flow_strength"],
                auto_meta["regime_strength"],
                auto_meta["space_strength"],
                auto_meta["space_ratio"],
                auto_meta["stop_adjust"],
                position_amt,
                entry_price,
                position_memory.bars_held,
                open_amount,
                pending_entry.signal,
                pending_entry.target_price,
                pending_entry.bars_waited,
            )

            risk_action_done = manage_open_position(
                exchange=exchange,
                symbol=market_symbol,
                position_amt=position_amt,
                entry_price=entry_price,
                last_price=last_price,
                current_candle=closed[-1],
                previous_candle=closed[-2] if len(closed) >= 2 else [],
                flow=flow,
                memory=position_memory,
                cfg=cfg,
                min_amount=min_amount,
                dry_run=cfg.dry_run,
            )
            if risk_action_done:
                continue

            if cfg.entry_mode == "pullback_limit":
                if abs(position_amt) < 1e-12 and pending_entry.active():
                    triggered = (
                        pending_entry.signal == "LONG" and last_price <= pending_entry.target_price
                    ) or (
                        pending_entry.signal == "SHORT" and last_price >= pending_entry.target_price
                    )
                    if triggered:
                        if not sentiment_entry_allowed(
                            signal=pending_entry.signal,
                            signal_mode=pending_entry.mode,
                            sentiment_score=sentiment["sentiment_score"],
                            sentiment_source_count=sentiment["sentiment_source_count"],
                            cfg=cfg,
                        ):
                            sentiment_skipped_signals += 1
                            logging.info(
                                "挂单触发但被情绪过滤拒绝：signal=%s mode=%s sent=%.3f src=%s",
                                pending_entry.signal,
                                pending_entry.mode,
                                sentiment["sentiment_score"],
                                int(sentiment["sentiment_source_count"]),
                            )
                            pending_entry.clear()
                            continue
                        if not orderbook_entry_allowed(
                            pending_entry.signal,
                            pending_entry.mode,
                            metrics["depth_imbalance"],
                            cfg,
                        ):
                            orderbook_skipped_signals += 1
                            logging.info(
                                "挂单触发但被盘口过滤拒绝：signal=%s mode=%s depth_imb=%.3f",
                                pending_entry.signal,
                                pending_entry.mode,
                                metrics["depth_imbalance"],
                            )
                            pending_entry.clear()
                            continue
                        memory_mode = (
                            pending_entry.mode
                            if pending_entry.mode in {"TREND", "RANGE"}
                            else "TREND"
                        )
                        opened = apply_signal(
                            exchange=exchange,
                            symbol=market_symbol,
                            signal=pending_entry.signal,
                            position_amt=position_amt,
                            open_amount=open_amount,
                            open_limit_price=pending_entry.target_price,
                            open_limit_tif=cfg.entry_limit_tif,
                            dry_run=cfg.dry_run,
                        )
                        pending_entry.clear()
                        if opened:
                            position_memory.entry_mode = memory_mode
                        continue
                    pending_entry.bars_waited += 1
                    if (
                        cfg.entry_max_wait_bars > 0
                        and pending_entry.bars_waited >= cfg.entry_max_wait_bars
                    ):
                        logging.info(
                            "挂单超时取消：signal=%s target=%.2f wait=%s",
                            pending_entry.signal,
                            pending_entry.target_price,
                            pending_entry.bars_waited,
                        )
                        pending_entry.clear()

                if entry_signal in {"LONG", "SHORT"}:
                    if entry_signal == "LONG" and position_amt > 0:
                        logging.info("Already LONG, skip.")
                        continue
                    if entry_signal == "SHORT" and position_amt < 0:
                        logging.info("Already SHORT, skip.")
                        continue

                    has_opposite = (entry_signal == "LONG" and position_amt < 0) or (
                        entry_signal == "SHORT" and position_amt > 0
                    )
                    if has_opposite:
                        close_opposite_by_signal(
                            exchange=exchange,
                            symbol=market_symbol,
                            signal=entry_signal,
                            position_amt=position_amt,
                            dry_run=cfg.dry_run,
                        )
                        if cfg.entry_cooldown_bars > 0:
                            cooldown_bars_remaining = max(
                                cooldown_bars_remaining, cfg.entry_cooldown_bars
                            )
                            pending_entry.clear()
                            logging.info(
                                "反手后进入冷静期：bars=%s",
                                cooldown_bars_remaining,
                            )
                            continue

                    target_price = entry_target_price(
                        last_price, entry_signal, entry_signal_mode, cfg, atr_value
                    )
                    mode = (
                        entry_signal_mode if entry_signal_mode in {"TREND", "RANGE"} else "TREND"
                    )
                    should_replace = (
                        not pending_entry.active()
                        or pending_entry.signal != entry_signal
                        or cfg.entry_refresh_on_same_signal
                    )
                    if should_replace:
                        pending_entry.signal = entry_signal
                        pending_entry.mode = mode
                        pending_entry.regime = regime["regime"]
                        pending_entry.target_price = target_price
                        pending_entry.bars_waited = 0
                        logging.info(
                            "设置挂单等待入场：signal=%s mode=%s target=%.2f pullback=%.3f%% atr=%.3f%% max_wait=%s",
                            entry_signal,
                            mode,
                            target_price,
                            entry_pullback_with_atr_pct(
                                last_price, entry_signal, mode, cfg, atr_value
                            )
                            * 100,
                            (atr_value / last_price * 100) if last_price > 0 else 0.0,
                            cfg.entry_max_wait_bars,
                        )
            else:
                has_opposite = (entry_signal == "LONG" and position_amt < 0) or (
                    entry_signal == "SHORT" and position_amt > 0
                )
                if has_opposite:
                    close_opposite_by_signal(
                        exchange=exchange,
                        symbol=market_symbol,
                        signal=entry_signal,
                        position_amt=position_amt,
                        dry_run=cfg.dry_run,
                    )
                    if cfg.entry_cooldown_bars > 0:
                        cooldown_bars_remaining = max(
                            cooldown_bars_remaining, cfg.entry_cooldown_bars
                        )
                        pending_entry.clear()
                        logging.info(
                            "反手后进入冷静期：bars=%s",
                            cooldown_bars_remaining,
                        )
                        continue
                opened = apply_signal(
                    exchange=exchange,
                    symbol=market_symbol,
                    signal=entry_signal,
                    position_amt=position_amt,
                    open_amount=open_amount,
                    open_limit_price=aggressive_limit_price(
                        last_price, entry_signal, cfg.entry_limit_offset_pct
                    ),
                    open_limit_tif=cfg.entry_limit_tif,
                    dry_run=cfg.dry_run,
                )
                if opened and entry_signal in {"LONG", "SHORT"}:
                    memory_mode = (
                        entry_signal_mode if entry_signal_mode in {"TREND", "RANGE"} else "TREND"
                    )
                    position_memory.entry_mode = memory_mode
                    pending_entry.clear()
        except KeyboardInterrupt:
            logging.info("Bot stopped by user.")
            break
        except Exception as exc:
            logging.exception("Cycle failed: %s", exc)
            time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
