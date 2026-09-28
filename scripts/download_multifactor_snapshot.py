"""Archive public data without keys; retain first-seen observations across refreshes."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

import download_macro_snapshot as legacy
import multifactor as mf

# Conservative estimated release delays for reconstructed research ONLY.
# Actual forward availability is max(estimate, first_seen). FRED latest is not ALFRED vintage data.
FRED = {
    "fed_effective": ("DFF", 2),
    "fed_target": ("DFEDTARU", 2), "fed_assets": ("WALCL", 2),
    "ust_2y": ("DGS2", 2), "ust_10y": ("DGS10", 2), "ust_real_10y": ("DFII10", 2),
    "broad_dollar": ("DTWEXBGS", 10),
    "eurusd": ("DEXUSEU", 10), "gbpusd": ("DEXUSUK", 10),
    "usdjpy": ("DEXJPUS", 10), "usdcny": ("DEXCHUS", 10), "usdchf": ("DEXSZUS", 10),
    "sp500": ("SP500", 2), "nasdaq": ("NASDAQCOM", 2),
    "vix": ("VIXCLS", 2), "oil": ("DCOILWTICO", 3),
}
DERIVATIVES = {
    "global_long_short": ("globalLongShortAccountRatio", "longShortRatio"),
    "top_position_long_short": ("topLongShortPositionRatio", "longShortRatio"),
    "taker_buy_sell": ("takerlongshortRatio", "buySellRatio"),
    "open_interest": ("openInterestHist", "sumOpenInterest"),
}
BINANCE = "https://fapi.binance.com"


def fred_rows(text, series_id, lag_days, end_ms):
    rows = []
    for raw in csv.DictReader(io.StringIO(text)):
        date = raw.get("observation_date") or raw.get("DATE")
        if not date:
            raise ValueError("FRED response is not an observation CSV")
        try:
            value = float(raw[series_id])
        except (ValueError, KeyError):
            continue
        observed = legacy.parse_utc(date)
        available = observed + lag_days * mf.DAY
        if math.isfinite(value) and available <= end_ms:
            rows.append([observed, available, value, end_ms])
    if not rows:
        raise ValueError(f"No usable FRED rows: {series_id}")
    return rows


def fetch_fred(name, start_ms, end_ms):
    series_id, lag = FRED[name]
    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={
        "id": series_id,
        "cosd": datetime.fromtimestamp(start_ms / 1000, timezone.utc).date().isoformat(),
        "coed": datetime.fromtimestamp(end_ms / 1000, timezone.utc).date().isoformat(),
    }, timeout=25)
    response.raise_for_status()
    return fred_rows(response.text, series_id, lag, end_ms)


def binance_json(path, params):
    response = requests.get(BINANCE + path, params=params, timeout=20)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, list):
        raise ValueError("Unexpected Binance response")
    return result


def fetch_derivatives(name, start_ms, end_ms):
    path, field = DERIVATIVES[name]
    # The exchange only retains roughly 30 days; no invented older observations.
    cursor = max(start_ms, end_ms - 29 * mf.DAY)
    output = {}
    while cursor < end_ms:
        window_end = min(cursor + 499 * mf.HOUR, end_ms)
        batch = binance_json("/futures/data/" + path, {
            "symbol": "BTCUSDT", "period": "1h", "limit": 500,
            "startTime": cursor, "endTime": window_end,
        })
        for item in batch:
            observed = int(item["timestamp"])
            available = observed + mf.HOUR + 5 * 60_000
            value = float(item[field])
            if cursor <= observed <= window_end and available <= end_ms and math.isfinite(value) and value > 0:
                output[observed] = [observed, available, value, end_ms]
        # Endpoint returns the LAST limit records and can round startTime backwards.
        # Bounded windows avoid silently losing the first part of a 30-day request.
        cursor = window_end + 1
    if not output:
        raise ValueError(f"No usable Binance observations: {name}")
    return sorted(output.values())


def fetch_btc(start_ms, end_ms):
    cursor, output = start_ms, []
    while cursor < end_ms:
        batch = binance_json("/fapi/v1/klines", {
            "symbol": "BTCUSDT", "interval": "1h", "limit": 1000,
            "startTime": cursor, "endTime": end_ms,
        })
        if not batch:
            break
        for row in batch:
            observed, available = int(row[6]) + 1, int(row[6]) + 1 + 60_000
            if available <= end_ms:
                output.append([observed, available, float(row[4]), end_ms])
        next_cursor = int(batch[-1][6]) + 1
        if next_cursor <= cursor:
            raise ValueError("Candle pagination did not advance")
        cursor = next_cursor
    if not output:
        raise ValueError("No closed BTC candles")
    return output


def fetch_funding(start_ms, end_ms):
    cursor, output = start_ms, []
    while cursor < end_ms:
        batch = binance_json("/fapi/v1/fundingRate", {
            "symbol": "BTCUSDT", "limit": 1000, "startTime": cursor, "endTime": end_ms,
        })
        if not batch:
            break
        for row in batch:
            observed = int(row["fundingTime"])
            if observed + 60_000 <= end_ms:
                output.append([observed, observed + 60_000, float(row["fundingRate"]), end_ms])
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise ValueError("Funding pagination did not advance")
        cursor = next_cursor
    if not output:
        raise ValueError("No funding observations")
    return output


def fetch_gold(start_ms, end_ms):
    return [[int(t) - mf.DAY, int(t), value, end_ms]
            for t, value in legacy.fetch_daily_series("GC=F", start_ms, end_ms) if t <= end_ms]


def merge_rows(previous, incoming):
    # Never rewrite an observation already used by a forward decision.
    merged = {int(row[0]): row for row in previous}
    for row in incoming:
        merged.setdefault(int(row[0]), row)
    return sorted(merged.values())


def collect(output: Path, start_ms: int, end_ms: int):
    previous = {"series": {}}
    if output.exists():
        with gzip.open(output, "rt", encoding="utf-8") as handle:
            previous = json.load(handle)
        mf.Snapshot(previous)  # Validate before preserving an existing archive.
    jobs = {name: (fetch_fred, name) for name in FRED}
    jobs.update({name: (fetch_derivatives, name) for name in DERIVATIVES})
    jobs.update({"btc_close": (fetch_btc, None), "funding": (fetch_funding, None), "gold": (fetch_gold, None)})

    def run(item):
        name, (fn, arg) = item
        try:
            rows = fn(arg, start_ms, end_ms) if arg else fn(start_ms, end_ms)
            if not rows:
                raise ValueError("Empty provider response")
            # First-seen is actual receipt time, not the time the batch started.
            received = int(datetime.now(timezone.utc).timestamp() * 1000)
            for row in rows:
                row[3] = max(end_ms, received)
            return name, rows, None
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            return name, [], f"{type(exc).__name__}: {str(exc)[:220]}"

    series = dict(previous.get("series", {}))
    errors = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, rows, error in pool.map(run, jobs.items()):
            if error:
                errors[name] = error
            series[name] = merge_rows(series.get(name, []), rows)
            print(f"{name}: {'ERROR ' + error if error else str(len(rows)) + ' observations'}", flush=True)
    payload = {
        "schema_version": 1, "series": series,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "errors": errors, "fred_series": FRED,
            "sources": ["FRED CSV (latest vintage)", "Binance USD-M public REST", "Yahoo GC=F"],
            "availability": "forward uses max(estimated availability, first_seen); existing rows immutable",
            "research_warning": "Reconstructed FRED history can contain revisions; not point-in-time OOS.",
            "derivatives_history_limit_days": 30,
            "international_events": "Separate timestamped event file required; proxies do not constitute a news feed.",
        },
    }
    mf.Snapshot(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/snapshots/multifactor_latest.json.gz"))
    parser.add_argument("--history-days", type=int, default=180)
    args = parser.parse_args()
    if args.history_days < 60:
        parser.error("--history-days must be at least 60")
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    payload = collect(args.output, end - args.history_days * mf.DAY, end)
    return 1 if payload["metadata"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
