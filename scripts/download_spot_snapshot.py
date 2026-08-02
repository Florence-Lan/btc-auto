#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import download_market_snapshot as market_snapshot
import simulate_range_swing as sim


BINANCE_SPOT_BASE_URL = "https://api.binance.com"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json_with_retries(
    session: requests.Session,
    path: str,
    params: dict[str, Any],
) -> Any:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(
                f"{BINANCE_SPOT_BASE_URL}{path}",
                params=params,
                timeout=45,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch Binance spot endpoint {path}: {last_error}") from last_error


def fetch_spot_klines_range(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[sim.Candle]:
    interval_ms = sim.interval_to_ms(interval)
    candles: list[sim.Candle] = []
    seen: set[int] = set()
    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-auto-spot-snapshot/1.0"})
        cursor = start_ms
        while cursor < end_ms:
            batch = fetch_json_with_retries(
                session,
                "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            if not isinstance(batch, list) or not batch:
                break
            for raw in batch:
                candle = sim.candle_from_kline(raw)
                if candle.open_time_ms not in seen and candle.close_time_ms <= end_ms:
                    candles.append(candle)
                    seen.add(candle.open_time_ms)
            next_cursor = int(batch[-1][0]) + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
            time.sleep(0.05)
    return sorted(candles, key=lambda candle: candle.open_time_ms)


def save_spot_snapshot(
    path: Path,
    symbol: str,
    interval: str,
    candles: list[sim.Candle],
    start_ms: int,
    end_ms: int,
    gap_stats: dict[str, float | int] | None = None,
) -> None:
    payload = {
        "version": 1,
        "metadata": {
            "symbol": symbol,
            "interval": interval,
            "start_utc": sim.iso_utc_from_ms(start_ms),
            "end_utc": sim.iso_utc_from_ms(end_ms),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "binance_spot_public_api",
            "gap_stats": gap_stats or {},
        },
        "candles": [sim.candle_to_compact(candle) for candle in candles],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "wt", encoding="utf-8")
    else:
        handle = path.open("w", encoding="utf-8")
    with handle:
        json.dump(payload, handle, separators=(",", ":"))


def load_spot_snapshot(path: Path) -> tuple[list[sim.Candle], dict[str, Any]]:
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        payload = json.load(handle)
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported spot snapshot version: {payload.get('version')}")
    candles = [sim.candle_from_compact(row) for row in payload.get("candles", [])]
    return candles, dict(payload.get("metadata", {}))


def candle_gap_stats(interval: str, candles: list[sim.Candle]) -> dict[str, float | int]:
    step = sim.interval_to_ms(interval)
    missing = 0
    gap_count = 0
    largest_gap = 0
    for left, right in zip(candles, candles[1:]):
        distance = right.open_time_ms - left.open_time_ms
        if distance != step:
            gap_count += 1
            missing_here = max(0, distance // step - 1)
            missing += missing_here
            largest_gap = max(largest_gap, missing_here)
    expected = len(candles) + missing
    coverage_pct = len(candles) / expected * 100 if expected else 0.0
    return {
        "gap_count": gap_count,
        "missing_candles": missing,
        "largest_gap_candles": largest_gap,
        "coverage_pct": coverage_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Binance spot candles into an immutable snapshot.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start-utc", default="2020-01-01T00:00:00Z")
    parser.add_argument("--end-utc", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument(
        "--output",
        type=Path,
        default=sim.repo_root() / "data/snapshots/btcusdt_spot_1h_2020_present.json.gz",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = sim.normalize_symbol(args.symbol)
    interval_ms = sim.interval_to_ms(args.interval)
    start_ms = market_snapshot.parse_utc(args.start_utc)
    end_ms = market_snapshot.parse_utc(args.end_utc)
    if end_ms <= start_ms:
        raise ValueError("--end-utc must be after --start-utc")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {args.output}")
    candles = fetch_spot_klines_range(symbol, args.interval, start_ms, end_ms)
    if not candles:
        raise RuntimeError("No Binance spot candles returned")
    gaps = candle_gap_stats(args.interval, candles)
    if float(gaps["coverage_pct"]) < 99.5:
        raise RuntimeError(f"Spot snapshot coverage is too low: {gaps['coverage_pct']:.3f}%")
    if candles[0].open_time_ms > start_ms + interval_ms:
        raise RuntimeError("Spot snapshot starts materially after the requested window")
    save_spot_snapshot(args.output, symbol, args.interval, candles, start_ms, end_ms, gaps)
    print(f"{args.interval}: {len(candles)} spot candles")
    print(
        f"Coverage: {gaps['coverage_pct']:.4f}% "
        f"({gaps['gap_count']} gaps, {gaps['missing_candles']} missing candles)"
    )
    print(f"Snapshot: {args.output}")
    print(f"SHA256: {file_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
