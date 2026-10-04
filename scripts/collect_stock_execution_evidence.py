#!/usr/bin/env python3
"""Archive public books/trades and historical fill-bar evidence; no account access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://fapi.asterdex.com/fapi/v3"
SYMBOLS = ("MUUSDT", "SNDKUSDT", "SKHYNIXUSDT")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def millis(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def book_metrics(book: dict) -> dict:
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    if not bids or not asks or bids[0][0] >= asks[0][0]:
        return {"valid_two_sided_book": False}
    mid = (bids[0][0] + asks[0][0]) / 2
    return {"valid_two_sided_book": True, "mid_price": mid,
            "spread_bps": (asks[0][0] - bids[0][0]) / mid * 10000,
            "bid_notional_within20bps": sum(p * q for p, q in bids if p >= mid * .998),
            "ask_notional_within20bps": sum(p * q for p, q in asks if p <= mid * 1.002),
            "warning": "Visible depth snapshot only; no reserved liquidity or demonstrated fills."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=Path("data/research/stock_swing_liquidity_20261004"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve evidence in a new directory")
    args.output_dir.mkdir(parents=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir()
    session = requests.Session()
    session.headers["User-Agent"] = "stock-public-execution-evidence/1.0"
    manifest, errors = [], []

    def fetch(endpoint: str, params: dict, key: str):
        began = utc()
        try:
            response = session.get(BASE + endpoint, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and int(payload.get("code", 0)) < 0:
                raise ValueError(f"API error {payload.get('code')}")
            content = response.content
            (raw_dir / (key + ".json")).write_bytes(content)
            manifest.append({"key": key, "endpoint": endpoint, "params": params,
                             "request_start_utc": began, "request_end_utc": utc(),
                             "sha256": hashlib.sha256(content).hexdigest()})
            return payload
        except (requests.RequestException, ValueError) as error:
            errors.append({"key": key, "request_start_utc": began, "error": str(error)})
            return None

    observed, historical = {}, {}
    server = fetch("/time", {}, "server_time")
    for symbol in SYMBOLS:
        book = fetch("/depth", {"symbol": symbol, "limit": 100}, symbol + "_depth")
        trades = fetch("/trades", {"symbol": symbol, "limit": 1000}, symbol + "_recent_trades")
        candles = fetch("/klines", {"symbol": symbol, "interval": "5m", "limit": 12}, symbol + "_recent_5m")
        observed[symbol] = {"book": book_metrics(book) if book else None,
                            "recent_trade_count": len(trades) if isinstance(trades, list) else None,
                            "most_recent_trade_ms": max((int(t["time"]) for t in trades), default=None) if isinstance(trades, list) else None,
                            "recent_candles_received": len(candles) if isinstance(candles, list) else None}
        path = args.ledger_dir / f"{symbol}_full_prior5m_volume10pct_cost1_trades.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        details = []
        for index, row in enumerate(rows):
            for kind in ("entry", "exit"):
                t = millis(row[kind + "_utc"])
                aggregate = fetch("/aggTrades", {"symbol": symbol, "startTime": t,
                                                  "endTime": t + 300000 - 1, "limit": 1000},
                                  f"{symbol}_{index}_{kind}")
                item = {"ledger_index": index, "kind": kind, "bar_start_utc": row[kind + "_utc"],
                        "model_fill": float(row[kind + "_fill"]), "model_qty": float(row["qty"])}
                if isinstance(aggregate, list):
                    within = [a for a in aggregate if t <= int(a["T"]) < t + 300000]
                    item.update({"aggregates_returned": len(aggregate), "within_bar_count": len(within),
                                 "bar_quantity_observed": sum(float(a["q"]) for a in within),
                                 "possibly_truncated": len(aggregate) >= 1000,
                                 "first_trade_delay_ms": min((int(a["T"]) - t for a in within), default=None),
                                 "observed_min_price": min((float(a["p"]) for a in within), default=None),
                                 "observed_max_price": max((float(a["p"]) for a in within), default=None)})
                else:
                    item["unavailable"] = True
                details.append(item)
                time.sleep(.15)
        historical[symbol] = details
        print(symbol, "archived", len(details), "fill-bar queries", flush=True)
    output = {"collected_at_utc": utc(), "places_orders": False, "server_time": server,
              "observations": observed, "historical_fill_bar_queries": historical,
              "requests": manifest, "errors": errors,
              "limitations": ["One public observation batch, not a forward paper service or a fill demonstration.",
                              "Empty historical response may mean expired/unavailable history; cannot alone prove no trade.",
                              "Trade history does not reconstruct historical resting depth or hypothetical order impact.",
                              "No live account endpoints, API keys, or orders used."]}
    (args.output_dir / "evidence.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
