#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import simulate_range_swing as sim


def parse_utc(value: str) -> int:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contiguous(interval: str, candles: list[sim.Candle]) -> None:
    step = sim.interval_to_ms(interval)
    gaps = [
        (left.open_time_utc, right.open_time_utc)
        for left, right in zip(candles, candles[1:])
        if right.open_time_ms - left.open_time_ms != step
    ]
    if gaps:
        first = gaps[0]
        raise RuntimeError(
            f"{interval} snapshot has {len(gaps)} gap(s); first gap: {first[0]} -> {first[1]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Binance futures candles and funding into an immutable snapshot.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-utc", default="2020-01-01T00:00:00Z")
    parser.add_argument("--end-utc", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--intervals", default="5m,1h,6h")
    parser.add_argument(
        "--output",
        type=Path,
        default=sim.repo_root() / "data/snapshots/btcusdt_2020_present.json.gz",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = sim.normalize_symbol(args.symbol)
    start_ms = parse_utc(args.start_utc)
    end_ms = parse_utc(args.end_utc)
    if end_ms <= start_ms:
        raise ValueError("--end-utc must be after --start-utc")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {args.output}")

    interval_names = sim.parse_timeframes(args.intervals)
    if not interval_names:
        raise ValueError("--intervals must not be empty")
    intervals: dict[str, list[sim.Candle]] = {}
    for interval in interval_names:
        sim.interval_to_ms(interval)
        candles = sim.fetch_futures_klines_range(symbol, interval, start_ms, end_ms)
        if not candles:
            raise RuntimeError(f"No candles returned for {interval}")
        validate_contiguous(interval, candles)
        intervals[interval] = candles
        print(f"{interval}: {len(candles)} candles")

    funding = sim.fetch_funding_history(symbol, start_ms, end_ms)
    if not funding.times:
        raise RuntimeError("No funding history returned")
    sim.save_market_snapshot(args.output, symbol, intervals, funding, start_ms, end_ms)
    print(f"Funding events: {len(funding.times)}")
    print(f"Snapshot: {args.output}")
    print(f"SHA256: {file_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
