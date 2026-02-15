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
    trend_spread_threshold: float
    trade_lookback: int
    big_trade_usdt: float
    depth_levels: int
    flow_score_threshold: float
    trade_weight: float
    margin_usdt: float
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
            fast_ema=env_int("FAST_EMA", 20),
            slow_ema=env_int("SLOW_EMA", 55),
            trend_spread_threshold=env_float("TREND_SPREAD_THRESHOLD", 0.0008),
            trade_lookback=env_int("TRADE_LOOKBACK", 500),
            big_trade_usdt=env_float("BIG_TRADE_USDT", 100000.0),
            depth_levels=env_int("DEPTH_LEVELS", 20),
            flow_score_threshold=env_float("FLOW_SCORE_THRESHOLD", 0.12),
            trade_weight=env_float("TRADE_WEIGHT", 0.6),
            margin_usdt=env_float("MARGIN_USDT", 30.0),
            poll_seconds=env_int("POLL_SECONDS", 20),
            dry_run=env_bool("DRY_RUN", True),
            testnet=env_bool("BINANCE_TESTNET", False),
        )


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


def trend_signal(closes: List[float], cfg: Config) -> Tuple[int, Dict[str, float]]:
    fast = ema_series(closes, cfg.fast_ema)
    slow = ema_series(closes, cfg.slow_ema)
    idx = len(closes) - 1
    if idx < max(cfg.fast_ema, cfg.slow_ema):
        return 0, {"spread": 0.0, "slope": 0.0}

    price = closes[idx]
    fast_now = fast[idx]
    fast_prev = fast[idx - 1] if idx > 0 else fast_now
    slow_now = slow[idx]
    spread = (fast_now - slow_now) / slow_now if slow_now else 0.0
    slope = (fast_now - fast_prev) / fast_prev if fast_prev else 0.0

    if spread > cfg.trend_spread_threshold and slope > 0 and price > fast_now:
        return 1, {"spread": spread, "slope": slope}
    if spread < -cfg.trend_spread_threshold and slope < 0 and price < fast_now:
        return -1, {"spread": spread, "slope": slope}
    return 0, {"spread": spread, "slope": slope}


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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
    if score > cfg.flow_score_threshold:
        return 1
    if score < -cfg.flow_score_threshold:
        return -1
    return 0


def combined_signal(trend: int, flow: int) -> str:
    if trend == 1 and flow == 1:
        return "LONG"
    if trend == -1 and flow == -1:
        return "SHORT"
    return "HOLD"


def fetch_position_amt(exchange: ccxt.binance, market_id: str) -> float:
    data = exchange.fapiPrivateV2GetPositionRisk({"symbol": market_id})
    if isinstance(data, list):
        net = 0.0
        for row in data:
            if row.get("symbol") != market_id:
                continue
            amt = float(row.get("positionAmt", 0.0))
            side = str(row.get("positionSide", "BOTH")).upper()
            if side == "SHORT":
                net -= abs(amt)
            else:
                net += amt
        return net
    return float(data.get("positionAmt", 0.0))


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


def apply_signal(
    exchange: ccxt.binance,
    symbol: str,
    signal: str,
    position_amt: float,
    open_amount: float,
    dry_run: bool,
) -> None:
    if signal == "HOLD":
        logging.info("Signal HOLD, no order.")
        return

    if signal == "LONG":
        if position_amt > 0:
            logging.info("Already LONG, skip.")
            return
        if position_amt < 0:
            close_amt = abs(position_amt)
            place_market_order(
                exchange, symbol, "buy", close_amt, reduce_only=True, dry_run=dry_run
            )
        place_market_order(
            exchange, symbol, "buy", open_amount, reduce_only=False, dry_run=dry_run
        )
        return

    if signal == "SHORT":
        if position_amt < 0:
            logging.info("Already SHORT, skip.")
            return
        if position_amt > 0:
            close_amt = abs(position_amt)
            place_market_order(
                exchange, symbol, "sell", close_amt, reduce_only=True, dry_run=dry_run
            )
        place_market_order(
            exchange, symbol, "sell", open_amount, reduce_only=False, dry_run=dry_run
        )


def validate_config(cfg: Config) -> None:
    if cfg.fast_ema >= cfg.slow_ema:
        raise ValueError("FAST_EMA must be smaller than SLOW_EMA")
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
        "Start bot | symbol=%s leverage=%sx mode=%s hedge=%s timeframe=%s dry_run=%s",
        market_symbol,
        cfg.leverage,
        cfg.margin_mode,
        cfg.hedge_mode,
        cfg.timeframe,
        cfg.dry_run,
    )

    if not cfg.dry_run:
        exchange.set_position_mode(cfg.hedge_mode, market_symbol)
        exchange.set_margin_mode(cfg.margin_mode, market_symbol)
        exchange.set_leverage(cfg.leverage, market_symbol)
        logging.info("Position mode, leverage and margin mode configured on Binance.")
    else:
        logging.info("DRY_RUN enabled: no real orders will be sent.")

    last_candle_ts = None

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                market_symbol,
                timeframe=cfg.timeframe,
                limit=cfg.klines_limit,
            )
            if len(ohlcv) < cfg.slow_ema + 2:
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
            flow = flow_signal(metrics, cfg)
            signal = combined_signal(trend, flow)

            last_price = fetch_last_price(exchange, market_symbol)
            target_notional = cfg.margin_usdt * cfg.leverage
            raw_amount = target_notional / last_price
            open_amount = calc_order_amount(
                exchange, market_symbol, raw_amount, min_amount
            )

            position_amt = 0.0 if cfg.dry_run else fetch_position_amt(exchange, market_id)

            logging.info(
                (
                    "price=%.2f trend=%s(spread=%.4f%% slope=%.4f%%) "
                    "flow=%s(score=%.3f trade_imb=%.3f depth_imb=%.3f large=%s) "
                    "signal=%s position=%.6f open_qty=%.6f"
                ),
                last_price,
                trend,
                trend_meta["spread"] * 100,
                trend_meta["slope"] * 100,
                flow,
                metrics["flow_score"],
                metrics["trade_imbalance"],
                metrics["depth_imbalance"],
                int(metrics["large_count"]),
                signal,
                position_amt,
                open_amount,
            )

            apply_signal(
                exchange=exchange,
                symbol=market_symbol,
                signal=signal,
                position_amt=position_amt,
                open_amount=open_amount,
                dry_run=cfg.dry_run,
            )
        except KeyboardInterrupt:
            logging.info("Bot stopped by user.")
            break
        except Exception as exc:
            logging.exception("Cycle failed: %s", exc)
            time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
