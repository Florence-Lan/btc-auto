#!/usr/bin/env python3
"""Freeze public equity perpetual data without account credentials or orders."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


HOUR = 3_600_000
VENUES = {
    "aster": ("https://fapi.asterdex.com", "/fapi/v3"),
    "binance": ("https://fapi.binance.com", "/fapi/v1"),
}


def millis(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Explicit UTC offset is required")
    return int(parsed.timestamp() * 1000)


def utc(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def get(session: requests.Session, base: str, prefix: str, endpoint: str, params: dict | None = None):
    response = session.get(base + prefix + endpoint, params=params, timeout=40)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and int(payload.get("code", 0)) < 0:
        raise RuntimeError(f"Public endpoint {endpoint}: {payload}")
    return payload


def fetch_candles(session, base, prefix, endpoint, symbol, start, end):
    collected = {}
    cursor = start
    while cursor < end:
        params = {"interval": "1h", "startTime": cursor, "endTime": end - 1, "limit": 1500}
        params["pair" if endpoint == "/indexPriceKlines" else "symbol"] = symbol
        batch = get(session, base, prefix, endpoint, params)
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected candle response for {symbol} {endpoint}")
        if not batch:
            break
        times = [int(row[0]) for row in batch]
        if times != sorted(times) or max(times) < cursor:
            raise RuntimeError(f"Pagination failed for {symbol} {endpoint}")
        for row in batch:
            # Exclude the partial listing-hour candle and unfinished bars.
            if int(row[0]) >= start and int(row[6]) < end:
                collected[int(row[0])] = row
        cursor = max(times) + HOUR
        if len(batch) < 1500:
            break
    rows = [collected[t] for t in sorted(collected)]
    if not rows:
        raise RuntimeError(f"No complete data for {symbol} {endpoint}")
    gaps = sum(int(right[0]) - int(left[0]) != HOUR for left, right in zip(rows, rows[1:]))
    if gaps:
        raise RuntimeError(f"{symbol} {endpoint}: {gaps} missing-hour gaps")
    return rows


def fetch_funding(session, base, prefix, symbol, start, end):
    events = {}
    cursor = start
    while cursor < end:
        batch = get(session, base, prefix, "/fundingRate", {
            "symbol": symbol, "startTime": cursor, "endTime": end - 1, "limit": 1000})
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected funding response for {symbol}")
        if not batch:
            break
        times = [int(event["fundingTime"]) for event in batch]
        if times != sorted(times) or max(times) < cursor:
            raise RuntimeError(f"Funding pagination failed for {symbol}")
        for event in batch:
            if start <= int(event["fundingTime"]) < end:
                # Preserve corporate-action/special funding alongside regular.
                key = (int(event["fundingTime"]), event.get("rateType", "Regular"),
                       str(event["fundingRate"]), str(event.get("markPrice", "")))
                events[key] = event
        if len(batch) < 1000:
            break
        # Inclusive overlap preserves a second event at the last timestamp if
        # a page boundary cuts through a regular/special funding pair.
        next_cursor = max(times)
        if next_cursor <= cursor:
            raise RuntimeError(f"Funding pagination stalled for {symbol}")
        cursor = next_cursor
    return [events[key] for key in sorted(events)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", choices=VENUES, default="aster")
    parser.add_argument("--end-utc", default="2026-10-04T12:00:00Z")
    parser.add_argument("--symbols", default="MUUSDT,SNDKUSDT,SKHYNIXUSDT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {args.output}")
    base, prefix = VENUES[args.venue]
    end = millis(args.end_utc)
    session = requests.Session()
    session.headers["User-Agent"] = "memory-stock-swing-public-research/1.0"
    exchange = get(session, base, prefix, "/exchangeInfo")
    server_time = get(session, base, prefix, "/time")
    if end > int(server_time["serverTime"]):
        raise ValueError("Snapshot cutoff cannot be in the future")
    rules = {row["symbol"]: row for row in exchange["symbols"]}
    funding_info = get(session, base, prefix, "/fundingInfo")
    result = {
        "venue": args.venue, "base_url": base, "api_prefix": prefix,
        "end_ms_exclusive": end, "end_utc_exclusive": utc(end),
        "retrieved_server_time": server_time, "funding_info_current": funding_info,
        "symbols": {},
    }
    for symbol in args.symbols.split(","):
        rule = rules[symbol]
        if rule["status"] != "TRADING":
            raise ValueError(f"{symbol} is not trading")
        start = ((int(rule["onboardDate"]) + HOUR - 1) // HOUR) * HOUR
        series = {"rules_current": rule, "start_ms": start, "start_utc": utc(start)}
        for label, endpoint in (("trade_1h", "/klines"), ("mark_1h", "/markPriceKlines"), ("index_1h", "/indexPriceKlines")):
            series[label] = fetch_candles(session, base, prefix, endpoint, symbol, start, end)
            print(f"{symbol} {label}: {len(series[label])} complete candles", flush=True)
        series["funding"] = fetch_funding(session, base, prefix, symbol, start, end)
        if not series["funding"]:
            raise RuntimeError(f"Missing funding history for {symbol}")
        result["symbols"][symbol] = series
        print(f"{symbol} funding: {len(series['funding'])} events", flush=True)
    encoded = json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(encoded, mtime=0))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"snapshot": str(args.output), "sha256": digest, "cutoff": utc(end)}))


if __name__ == "__main__":
    main()
