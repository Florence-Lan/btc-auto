#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class EndpointSet:
    market_type: str
    base_url: str
    ticker_path: str
    kline_path: str
    trade_path: str
    depth_path: str
    premium_path: Optional[str] = None
    open_interest_path: Optional[str] = None


ENDPOINTS = {
    "futures": EndpointSet(
        market_type="futures",
        base_url="https://fapi.binance.com",
        ticker_path="/fapi/v1/ticker/24hr",
        kline_path="/fapi/v1/klines",
        trade_path="/fapi/v1/trades",
        depth_path="/fapi/v1/depth",
        premium_path="/fapi/v1/premiumIndex",
        open_interest_path="/fapi/v1/openInterest",
    ),
    "spot": EndpointSet(
        market_type="spot",
        base_url="https://api.binance.com",
        ticker_path="/api/v3/ticker/24hr",
        kline_path="/api/v3/klines",
        trade_path="/api/v3/trades",
        depth_path="/api/v3/depth",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env(root: Path) -> None:
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def normalize_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if ":" in symbol:
        symbol = symbol.split(":", 1)[0]
    return symbol.replace("/", "").replace("-", "").replace("_", "")


def iso_utc_from_ms(timestamp_ms: Any) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).isoformat()


def fetch_json(
    session: requests.Session, endpoints: EndpointSet, path: str, params: Dict[str, Any]
) -> Any:
    response = session.get(
        f"{endpoints.base_url}{path}",
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def normalize_kline(entry: List[Any], fetched_at_ms: int) -> Dict[str, Any]:
    return {
        "open_time_utc": iso_utc_from_ms(entry[0]),
        "close_time_utc": iso_utc_from_ms(entry[6]),
        "open": float(entry[1]),
        "high": float(entry[2]),
        "low": float(entry[3]),
        "close": float(entry[4]),
        "volume_base": float(entry[5]),
        "volume_quote": float(entry[7]),
        "trades": int(entry[8]),
        "taker_buy_base": float(entry[9]),
        "taker_buy_quote": float(entry[10]),
        "is_closed": int(entry[6]) < fetched_at_ms,
    }


def normalize_trade(entry: Dict[str, Any]) -> Dict[str, Any]:
    price = float(entry["price"])
    qty = float(entry["qty"])
    return {
        "id": entry.get("id"),
        "time_utc": iso_utc_from_ms(entry["time"]),
        "price": price,
        "qty_base": qty,
        "notional_quote": round(price * qty, 8),
        "side": "sell_aggressor" if entry.get("isBuyerMaker") else "buy_aggressor",
    }


def normalize_depth_levels(levels: List[List[Any]]) -> List[Dict[str, float]]:
    normalized: List[Dict[str, float]] = []
    for price_raw, qty_raw in levels:
        price = float(price_raw)
        qty = float(qty_raw)
        normalized.append(
            {
                "price": price,
                "qty_base": qty,
                "notional_quote": round(price * qty, 8),
            }
        )
    return normalized


def summarize_depth(
    bids: List[Dict[str, float]], asks: List[Dict[str, float]]
) -> Dict[str, Optional[float]]:
    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    bid_notional = sum(level["notional_quote"] for level in bids)
    ask_notional = sum(level["notional_quote"] for level in asks)
    total = bid_notional + ask_notional
    imbalance = ((bid_notional - ask_notional) / total) if total else 0.0
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": (best_ask - best_bid) if best_bid is not None and best_ask is not None else None,
        "bid_notional_quote": round(bid_notional, 8),
        "ask_notional_quote": round(ask_notional, 8),
        "imbalance": round(imbalance, 8),
    }


def latest_closed_kline(klines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for kline in reversed(klines):
        if kline.get("is_closed"):
            return kline
    return None


def build_snapshot(
    session: requests.Session,
    endpoints: EndpointSet,
    symbol: str,
    interval: str,
    kline_limit: int,
    trade_limit: int,
    depth_limit: int,
) -> Dict[str, Any]:
    ticker = fetch_json(
        session,
        endpoints,
        endpoints.ticker_path,
        {"symbol": symbol},
    )
    klines_raw = fetch_json(
        session,
        endpoints,
        endpoints.kline_path,
        {"symbol": symbol, "interval": interval, "limit": kline_limit},
    )
    trades_raw = fetch_json(
        session,
        endpoints,
        endpoints.trade_path,
        {"symbol": symbol, "limit": trade_limit},
    )
    depth_raw = fetch_json(
        session,
        endpoints,
        endpoints.depth_path,
        {"symbol": symbol, "limit": depth_limit},
    )

    premium = None
    if endpoints.premium_path:
        premium = fetch_json(
            session,
            endpoints,
            endpoints.premium_path,
            {"symbol": symbol},
        )

    open_interest = None
    if endpoints.open_interest_path:
        open_interest = fetch_json(
            session,
            endpoints,
            endpoints.open_interest_path,
            {"symbol": symbol},
        )

    fetched_at = datetime.now(timezone.utc)
    fetched_at_ms = int(fetched_at.timestamp() * 1000)

    klines = [normalize_kline(entry, fetched_at_ms) for entry in klines_raw]
    trades = [normalize_trade(entry) for entry in trades_raw]
    bids = normalize_depth_levels(depth_raw.get("bids", []))
    asks = normalize_depth_levels(depth_raw.get("asks", []))

    snapshot = {
        "fetched_at_utc": fetched_at.isoformat(),
        "exchange": "binance",
        "market_type": endpoints.market_type,
        "symbol": symbol,
        "interval": interval,
        "limits": {
            "kline_limit": kline_limit,
            "trade_limit": trade_limit,
            "depth_limit": depth_limit,
        },
        "ticker_24h": {
            "last_price": float(ticker["lastPrice"]),
            "open_price": float(ticker["openPrice"]),
            "high_price": float(ticker["highPrice"]),
            "low_price": float(ticker["lowPrice"]),
            "price_change": float(ticker["priceChange"]),
            "price_change_percent": float(ticker["priceChangePercent"]),
            "weighted_avg_price": float(ticker["weightedAvgPrice"]),
            "volume_base": float(ticker["volume"]),
            "volume_quote": float(ticker["quoteVolume"]),
            "trade_count": int(ticker["count"]),
            "open_time_utc": iso_utc_from_ms(ticker["openTime"]),
            "close_time_utc": iso_utc_from_ms(ticker["closeTime"]),
        },
        "futures_mark_data": None,
        "open_interest": None,
        "latest_closed_kline": latest_closed_kline(klines),
        "latest_kline": klines[-1] if klines else None,
        "klines": klines,
        "recent_trades": trades,
        "order_book": {
            "last_update_id": depth_raw.get("lastUpdateId"),
            "bids": bids,
            "asks": asks,
            "summary": summarize_depth(bids, asks),
        },
    }

    if premium is not None:
        snapshot["futures_mark_data"] = {
            "mark_price": float(premium["markPrice"]),
            "index_price": float(premium["indexPrice"]),
            "last_funding_rate": float(premium["lastFundingRate"]),
            "next_funding_time_utc": iso_utc_from_ms(premium["nextFundingTime"]),
            "exchange_time_utc": iso_utc_from_ms(premium["time"]),
        }

    if open_interest is not None:
        snapshot["open_interest"] = {
            "value_base": float(open_interest["openInterest"]),
            "time_utc": iso_utc_from_ms(open_interest["time"]),
        }

    return snapshot


def resolve_output_dir(root: Path, output_dir: str) -> Path:
    target = Path(output_dir)
    if not target.is_absolute():
        target = root / target
    target.mkdir(parents=True, exist_ok=True)
    return target


def parse_args() -> argparse.Namespace:
    default_symbol = os.getenv("SYMBOL", "BTCUSDT")
    default_interval = os.getenv("TIMEFRAME", "5m")
    default_kline_limit = int(os.getenv("KLINES_LIMIT", "200"))

    parser = argparse.ArgumentParser(
        description="Fetch Binance market data and save a reusable JSON snapshot.",
    )
    parser.add_argument(
        "--market-type",
        choices=sorted(ENDPOINTS.keys()),
        default="futures",
        help="Binance market type to query. Defaults to futures.",
    )
    parser.add_argument(
        "--symbol",
        default=default_symbol,
        help="Trading symbol, for example BTCUSDT or BTC/USDT:USDT.",
    )
    parser.add_argument(
        "--interval",
        default=default_interval,
        help="Kline interval, for example 1m, 5m, 15m, 1h.",
    )
    parser.add_argument(
        "--kline-limit",
        type=positive_int,
        default=default_kline_limit,
        help="Number of klines to fetch.",
    )
    parser.add_argument(
        "--trade-limit",
        type=positive_int,
        default=20,
        help="Number of recent trades to fetch.",
    )
    parser.add_argument(
        "--depth-limit",
        type=positive_int,
        default=20,
        help="Number of order book levels to fetch on each side.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/binance_snapshots",
        help="Output directory for JSON snapshots.",
    )
    return parser.parse_args()


def main() -> int:
    root = repo_root()
    load_project_env(root)

    args = parse_args()

    symbol = normalize_symbol(args.symbol)
    interval = args.interval
    kline_limit = args.kline_limit

    output_dir = resolve_output_dir(root, args.output_dir)
    endpoints = ENDPOINTS[args.market_type]

    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-auto-market-fetcher/1.0"})
        snapshot = build_snapshot(
            session=session,
            endpoints=endpoints,
            symbol=symbol,
            interval=interval,
            kline_limit=kline_limit,
            trade_limit=args.trade_limit,
            depth_limit=args.depth_limit,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamped_path = output_dir / f"{symbol.lower()}_{args.market_type}_{timestamp}.json"
    latest_path = output_dir / f"{symbol.lower()}_{args.market_type}_latest.json"
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)

    timestamped_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")

    print(f"Saved snapshot: {timestamped_path}")
    print(f"Updated latest snapshot: {latest_path}")
    print(f"Last price: {snapshot['ticker_24h']['last_price']}")
    print(f"24h change: {snapshot['ticker_24h']['price_change_percent']}%")
    print(f"Best bid/ask: {snapshot['order_book']['summary']['best_bid']} / {snapshot['order_book']['summary']['best_ask']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
