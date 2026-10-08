"""Archive public data without keys; retain first-seen observations across refreshes."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

import download_macro_snapshot as legacy
import multifactor as mf
from binance_terminal_client import BinanceApiError, BinanceTerminalClient
from factor_data_freshness import OIL_MAX_AGE_MS

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
REFRESH_SECONDS = 3600
RETRY_BASE_SECONDS = 60
RETRY_MAX_SECONDS = 900
STALE_RETRY_SECONDS = 60


def now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def retry_delay_ms(failures):
    # One attempt per source per worker cycle; no synchronous retry loop.
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * 2 ** min(max(failures - 1, 0), 4)) * 1000


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
    # Share the terminal's persisted 418/429 cooldown instead of bypassing it.
    result = BinanceTerminalClient().public_get(path, params)
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


def oil_freshness_error(rows, timestamp):
    # Share the entry gate's age limit; HTTP success alone is not usable data.
    available = [int(row[0]) for row in rows
                 if max(int(row[1]), int(row[3])) <= timestamp]
    latest = max(available, default=None)
    if latest is None:
        return "Stale oil source: no currently available observations"
    age = timestamp - latest
    if age <= OIL_MAX_AGE_MS:
        return None
    observed = datetime.fromtimestamp(latest / 1000, timezone.utc).isoformat()
    return (f"Stale oil source: latest_observed_at_utc={observed} "
            f"age_days={age / mf.DAY:.6f} max_age_days={OIL_MAX_AGE_MS / mf.DAY:g}")


def collect(output: Path, start_ms: int, end_ms: int, *, only_due=False, force=False, sources=None):
    previous = {"series": {}}
    if output.exists():
        with gzip.open(output, "rt", encoding="utf-8") as handle:
            previous = json.load(handle)
        mf.Snapshot(previous)  # Validate before preserving an existing archive.
    jobs = {name: (fetch_fred, name) for name in FRED}
    jobs.update({name: (fetch_derivatives, name) for name in DERIVATIVES})
    jobs.update({"btc_close": (fetch_btc, None), "funding": (fetch_funding, None), "gold": (fetch_gold, None)})
    if sources is not None:
        sources = set(sources)
        unknown = sources - jobs.keys()
        if unknown:
            raise ValueError(f"Unknown factor sources: {sorted(unknown)}")
        jobs = {name: job for name, job in jobs.items() if name in sources}

    metadata = previous.get("metadata", {})
    errors = dict(metadata.get("errors", {}))
    source_status = {name: dict(state) for name, state in metadata.get("source_status", {}).items()}
    # Adopt old archives once, without interpreting a new partial-refresh mtime
    # as successful collection for every source.
    archive_time = int(output.stat().st_mtime * 1000) if output.exists() else 0
    for name in jobs:
        if name not in source_status:
            error = errors.get(name)
            interval = RETRY_BASE_SECONDS if error else REFRESH_SECONDS
            source_status[name] = {
                "ok": bool(previous.get("series", {}).get(name)) and not error,
                "error": error, "consecutive_failures": 1 if error else 0,
                "next_retry_at_ms": archive_time + interval * 1000
                if previous.get("series", {}).get(name) or error else 0,
            }
        # Older archives considered an HTTP 200 healthy even when the oil data
        # was already too old. Adopt the freshness policy without waiting for
        # their next hourly download or rewriting any archived observations.
        if name == "oil" and source_status[name].get("ok") is True:
            stale = oil_freshness_error(previous.get("series", {}).get(name, []), end_ms)
            if stale:
                source_status[name].update(ok=False, status="stale", error=stale,
                                           next_retry_at_ms=0)
                errors[name] = stale
    if only_due and not force:
        timestamp = now_ms()
        jobs = {name: job for name, job in jobs.items()
                if timestamp >= int(source_status[name].get("next_retry_at_ms") or 0)}
    if not jobs:
        return previous

    binance_sources = set(DERIVATIVES) | {"btc_close", "funding"}
    binance_lock = threading.Lock()

    def run(item):
        name, (fn, arg) = item
        try:
            # Preserve overlap for corrections/late publication; avoid re-downloading
            # six months of BTC candles and funding on every hourly refresh.
            since = start_ms
            if previous.get("series", {}).get(name):
                overlap = 90 * mf.DAY if name == "gold" else (14 * mf.DAY if name in FRED else 2 * mf.DAY)
                since = max(start_ms, max(int(r[0]) for r in previous["series"][name]) - overlap)
            if name in binance_sources:
                # Serialize this batch so a 418/429 on one source persists the
                # shared pause before another source attempts a network call.
                with binance_lock:
                    rows = fn(arg, since, end_ms) if arg else fn(since, end_ms)
            else:
                rows = fn(arg, since, end_ms) if arg else fn(since, end_ms)
            if not rows:
                raise ValueError("Empty provider response")
            # First-seen is actual receipt time, not the time the batch started.
            received = now_ms()
            for row in rows:
                row[3] = max(end_ms, received)
            # A malformed provider response must fail only its own source.
            mf.Snapshot({"schema_version": 1, "series": {name: rows}})
            return name, rows, None, received, 0
        except Exception as exc:
            return name, [], f"{type(exc).__name__}: {str(exc)[:220]}", now_ms(), int(
                getattr(exc, "retry_at_ms", None) or 0)

    series = dict(previous.get("series", {}))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, rows, error, received, provider_retry in pool.map(run, jobs.items()):
            state = source_status[name]
            transport_error = error
            failures = int(state.get("consecutive_failures") or 0) + 1 if error else 0
            series[name] = merge_rows(series.get(name, []), rows)
            stale = None
            if name == "oil" and not transport_error:
                stale = oil_freshness_error(series[name], max(end_ms, received))
                error = stale
            if error:
                errors[name] = error
            else:
                errors.pop(name, None)
            delay = (retry_delay_ms(failures) if transport_error else
                     STALE_RETRY_SECONDS * 1000 if stale else REFRESH_SECONDS * 1000)
            source_status[name] = {
                **state,
                "ok": error is None, "error": error, "last_attempt_ms": received,
                "status": "stale" if stale else "error" if error else "healthy",
                "last_success_ms": state.get("last_success_ms") if error else received,
                "consecutive_failures": failures,
                "next_retry_at_ms": max(received + delay, provider_retry),
                "provider_retry_at_ms": provider_retry or None,
                "latest_observed_at_ms": max((int(row[0]) for row in series[name]), default=None),
            }
            detail = 'ERROR ' + error if error else str(len(rows)) + ' observations'
            print(f"{name}: {detail}; failures={failures} next_retry_at_ms="
                  f"{source_status[name]['next_retry_at_ms']}", flush=True)
    payload = {
        "schema_version": 1, "series": series,
        "metadata": {
            **metadata,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "errors": errors, "source_status": source_status, "fred_series": FRED,
            "refresh_policy": {"success_seconds": REFRESH_SECONDS,
                               "stale_retry_seconds": STALE_RETRY_SECONDS,
                               "retry_base_seconds": RETRY_BASE_SECONDS,
                               "retry_max_seconds": RETRY_MAX_SECONDS,
                               "binance_cooldown": "shared persisted 418/429 pause takes precedence"},
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
