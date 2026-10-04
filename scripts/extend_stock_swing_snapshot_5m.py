#!/usr/bin/env python3
"""Append uniform five-minute public execution data to a frozen stock snapshot."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from download_stock_swing_snapshot import get


STEP = 300_000


def repair_gaps(session, base, prefix, endpoint, symbol, start, end, collected):
    """Refetch missing intervals; never synthesize or interpolate prices."""
    missing = sorted(set(range(start, end, STEP)) - collected.keys())
    repaired = 0
    while missing:
        first = missing[0]
        count = 1
        while count < min(len(missing), 1500) and missing[count] == first + count * STEP:
            count += 1
        last = first + count * STEP
        batch = fetch_with_retry(session, base, prefix, endpoint, {
            "symbol": symbol, "interval": "5m", "startTime": first,
            "endTime": last - 1, "limit": 1500})
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected gap response for {symbol} {endpoint}")
        returned = {int(row[0]): row for row in batch
                    if first <= int(row[0]) < last and int(row[6]) < end}
        if any(time not in returned for time in missing[:count]):
            raise ValueError(f"Public source cannot repair {symbol} {endpoint} gap at {first}")
        collected.update(returned)
        repaired += count
        missing = missing[count:]
        time.sleep(0.3)
    return repaired


def fetch_with_retry(session, base, prefix, endpoint, params):
    for attempt in range(3):
        try:
            return get(session, base, prefix, endpoint, params)
        except requests.HTTPError as error:
            if error.response.status_code not in (500, 502, 503, 504) or attempt == 2:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        time.sleep(1 + attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-with-gaps", action="store_true",
                        help="Preserve unavailable bars explicitly; the archive is not replay-ready")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    snapshot = json.loads(gzip.decompress(args.source.read_bytes()))
    cache = args.output.parent / "five_minute_cache"
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "memory-stock-swing-public-research/1.0"
    end = snapshot["end_ms_exclusive"]
    coverage_gaps = {}
    for symbol, source in snapshot["symbols"].items():
        start = source["start_ms"]
        for label, endpoint in (("trade_5m", "/klines"), ("mark_5m", "/markPriceKlines")):
            cached = cache / f"{symbol}_{label}.json"
            collected = {int(row[0]): row for row in json.loads(cached.read_text())} if cached.exists() else {}
            cursor = max(collected) + STEP if collected else start
            pages = 0
            while cursor < end:
                batch = fetch_with_retry(session, snapshot["base_url"], snapshot["api_prefix"], endpoint, {
                    "symbol": symbol, "interval": "5m", "startTime": cursor,
                    "endTime": end - 1, "limit": 1500})
                if not isinstance(batch, list) or not batch:
                    raise RuntimeError(f"Missing {symbol} {label} at {cursor}")
                latest = max(int(row[0]) for row in batch)
                if latest < cursor:
                    raise RuntimeError("Pagination stalled")
                for row in batch:
                    if start <= int(row[0]) and int(row[6]) < end:
                        collected[int(row[0])] = row
                cursor = latest + STEP
                pages += 1
                if pages % 4 == 0 or cursor >= end:
                    temporary = cached.with_suffix(".partial")
                    temporary.write_text(json.dumps([collected[t] for t in sorted(collected)], separators=(",", ":")))
                    temporary.replace(cached)
                if pages % 12 == 0:
                    print(symbol, label, "rows", len(collected), flush=True)
                # Bound public weight; no retries for 418/429 or rate-limit bypass.
                time.sleep(0.3)
            try:
                repaired = repair_gaps(session, snapshot["base_url"], snapshot["api_prefix"],
                                       endpoint, symbol, start, end, collected)
            except ValueError:
                if not args.archive_with_gaps:
                    raise
                repaired = 0
            rows = [collected[t] for t in sorted(collected)]
            if not rows or int(rows[0][0]) != start or int(rows[-1][0]) != end - STEP:
                raise ValueError(f"Incomplete endpoints for {symbol} {label}")
            missing = sorted(set(range(start, end, STEP)) - collected.keys())
            if missing:
                if not args.archive_with_gaps:
                    raise ValueError(f"Missing five-minute bars for {symbol} {label}")
                coverage_gaps[f"{symbol}:{label}"] = missing
                print(symbol, label, "UNAVAILABLE bars", len(missing), flush=True)
            if repaired:
                temporary = cached.with_suffix(".partial")
                temporary.write_text(json.dumps(rows, separators=(",", ":")))
                temporary.replace(cached)
                print(symbol, label, "refetched missing bars", repaired, flush=True)
            source[label] = rows
            print(symbol, label, "DONE", len(rows), flush=True)
    snapshot["parent_snapshot_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    snapshot["execution_step_ms"] = STEP
    snapshot["execution_coverage_complete"] = not coverage_gaps
    snapshot["execution_gaps"] = coverage_gaps
    snapshot["five_minute_retrieved_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_bytes(gzip.compress(json.dumps(snapshot, separators=(",", ":"), sort_keys=True).encode(), mtime=0))
    print(json.dumps({"snapshot": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
