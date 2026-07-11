#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import macro_regime
import simulate_range_swing as sim


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SYMBOLS = {
    "sp500": "ES=F",
    "nasdaq": "NQ=F",
    "vix": "^VIX",
    "dollar": "DX-Y.NYB",
    "gold": "GC=F",
    "silver": "SI=F",
}
FEAR_GREED_URL = "https://api.alternative.me/fng/"


def parse_utc(value: str) -> int:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def fetch_daily_series(symbol: str, start_ms: int, end_ms: int) -> list[list[float | int]]:
    response = requests.get(
        YAHOO_CHART_URL.format(symbol=requests.utils.quote(symbol, safe="")),
        params={
            "period1": start_ms // 1000,
            "period2": end_ms // 1000 + 1,
            "interval": "1d",
            "events": "history",
        },
        headers={"User-Agent": "btc-auto-macro-snapshot/1.0"},
        timeout=45,
    )
    response.raise_for_status()
    result = response.json().get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"No chart result returned for {symbol}")
    chart = result[0]
    timestamps = chart.get("timestamp", [])
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", [])
    rows: list[list[float | int]] = []
    for timestamp, close in zip(timestamps, closes):
        if close is None:
            continue
        observed_ms = int(timestamp) * 1000
        # A daily close is deliberately made available 24 hours later to prevent lookahead.
        available_ms = observed_ms + macro_regime.MS_PER_DAY
        if available_ms <= end_ms + macro_regime.MS_PER_DAY:
            rows.append([available_ms, float(close)])
    if len(rows) < 30:
        raise RuntimeError(f"Insufficient daily history returned for {symbol}: {len(rows)}")
    return rows


def fetch_fear_greed(start_ms: int, end_ms: int) -> list[list[float | int]]:
    response = requests.get(
        FEAR_GREED_URL,
        params={"limit": 0, "format": "json"},
        headers={"User-Agent": "btc-auto-macro-snapshot/1.0"},
        timeout=45,
    )
    response.raise_for_status()
    rows: list[list[float | int]] = []
    for item in response.json().get("data", []):
        try:
            observed_ms = int(item["timestamp"]) * 1000
            value = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        available_ms = observed_ms + macro_regime.MS_PER_DAY
        if start_ms <= observed_ms <= end_ms and 0 <= value <= 100:
            rows.append([available_ms, value])
    rows.sort(key=lambda row: int(row[0]))
    if len(rows) < 30:
        raise RuntimeError(f"Insufficient Fear & Greed history returned: {len(rows)}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze point-in-time macro market data for BTC research.")
    parser.add_argument("--start-utc", default="2019-10-01T00:00:00Z")
    parser.add_argument("--end-utc", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument(
        "--output",
        type=Path,
        default=sim.repo_root() / "data/snapshots/macro_20191001_present.json.gz",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_ms = parse_utc(args.start_utc)
    end_ms = parse_utc(args.end_utc)
    if end_ms <= start_ms:
        raise ValueError("--end-utc must be after --start-utc")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {args.output}")
    series: dict[str, list[list[float | int]]] = {}
    for name, symbol in SYMBOLS.items():
        series[name] = fetch_daily_series(symbol, start_ms, end_ms)
        print(f"{name} ({symbol}): {len(series[name])} observations")
    series["fear_greed"] = fetch_fear_greed(start_ms, end_ms)
    print(f"fear_greed: {len(series['fear_greed'])} observations")
    payload: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "start_utc": datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat(),
            "provider": "Yahoo Finance chart endpoint",
            "symbols": SYMBOLS,
            "sentiment_provider": "Alternative.me Fear & Greed Index API",
            "frequency": "1d",
            "availability_lag_hours": 24,
            "lookahead_policy": "daily close becomes usable 24 hours after provider timestamp",
        },
        "series": series,
    }
    macro_regime.save_macro_snapshot(args.output, payload)
    print(f"Snapshot: {args.output}")
    print(f"SHA256: {sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
