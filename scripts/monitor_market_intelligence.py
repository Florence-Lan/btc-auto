#!/usr/bin/env python
"""Read-only public market monitor. No keys, signing, orders or account mutations."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import market_intelligence as intelligence

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/market_intelligence"
FUTURES = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
HYPER = "https://api.hyperliquid.xyz/info"


def fetch(spec):
    name, method, url, params = spec
    try:
        response = (requests.get(url, params=params, timeout=(5, 12)) if method == "GET" else
                    requests.post(url, json=params, timeout=(5, 12)))
        response.raise_for_status()
        data = response.json()
        # Validate JSON numeric encoding before persistence.
        json.dumps(data, allow_nan=False)
        return name, {"status": "ok", "received_ms": int(time.time() * 1000), "url": url, "data": data}
    except (requests.RequestException, ValueError, TypeError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return name, {"status": "error", "received_ms": int(time.time() * 1000), "url": url,
                      "error": f"{type(exc).__name__}:{status}" if status else type(exc).__name__, "data": None}


def specifications():
    specs = []
    for venue, host, prefix in (("spot", SPOT, "/api/v3"), ("futures", FUTURES, "/fapi/v1")):
        specs.extend([
            (venue + "_flow", "GET", host + prefix + "/klines", {"symbol": "BTCUSDT", "interval": "5m", "limit": 30}),
            (venue + "_tape", "GET", host + prefix + "/aggTrades", {"symbol": "BTCUSDT", "limit": 1000}),
            (venue + "_book", "GET", host + prefix + "/depth", {"symbol": "BTCUSDT", "limit": 100}),
        ])
    for name, endpoint in (("open_interest", "openInterestHist"), ("global_accounts", "globalLongShortAccountRatio"),
                           ("top_accounts", "topLongShortAccountRatio"), ("top_positions", "topLongShortPositionRatio")):
        specs.append((name, "GET", FUTURES + "/futures/data/" + endpoint,
                      {"symbol": "BTCUSDT", "period": "5m", "limit": 20}))
    specs.extend([
        ("funding_basis", "GET", FUTURES + "/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"}),
        ("hyper_context", "POST", HYPER, {"type": "metaAndAssetCtxs"}),
        ("hyper_tape", "POST", HYPER, {"type": "recentTrades", "coin": "BTC"}),
        ("bitcoin_chain", "GET", "https://mempool.space/api/mempool/recent", None),
    ])
    return specs


def collect_records():
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = dict(pool.map(fetch, specifications()))
        tape = records["hyper_tape"]
        addresses = []
        if tape["status"] == "ok" and isinstance(tape["data"], list):
            try:
                largest = sorted(tape["data"], key=lambda r: -intelligence.number(r["px"], 0) * intelligence.number(r["sz"], 0))
                for trade in largest:
                    for address in trade.get("users", []):
                        if (isinstance(address, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", address)
                                and address.lower() not in addresses and int(address, 16) != 0):
                            addresses.append(address.lower())
                        if len(addresses) == 6:
                            break
                    if len(addresses) == 6:
                        break
            except (ValueError, TypeError, KeyError):
                addresses = []
        position_records = []
        queries = [(address, "POST", HYPER, {"type": "clearinghouseState", "user": address}) for address in addresses]
        for address, row in pool.map(fetch, queries):
            position_records.append({**row, "address": address})
        records["hyper_positions"] = {"status": "ok" if position_records else "unavailable",
            "received_ms": int(time.time() * 1000), "url": HYPER,
            "error": None if position_records else "No valid recent participant sample", "data": position_records}
    return records


def local_context(report):
    import macro_regime
    import research_world_events
    import world_event_risk
    now = report["generated_at_ms"]
    news_path = ROOT / "data/world_events/observations.json"
    try:
        news = world_event_risk.load_snapshot(news_path) if news_path.exists() else world_event_risk.empty_snapshot()
        if not news_path.exists() or time.time() - news_path.stat().st_mtime > 900:
            research_world_events.collect(news, provider="bbc-world")
            world_event_risk.save_snapshot(news_path, news)
        # Include only knowledge available by completion of this monitoring cycle.
        now = int(time.time() * 1000)
        report["news"] = world_event_risk.report_at(news, now)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["news"] = {"feed_status": "error", "error": type(exc).__name__}
    try:
        path = ROOT / "data/snapshots/macro_shadow_latest.json.gz"
        snapshot = macro_regime.load_macro_snapshot(path)
        decision = macro_regime.macro_decision_at(snapshot, now,
                        enabled_factors=("vix", "dollar", "metals", "sentiment"))
        report["macro"] = {"score": decision.score, "asof_ms": decision.asof_ms,
                           "factors": dict(decision.contributions), "allowed": decision.allowed}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["macro"] = {"status": "unavailable", "error": type(exc).__name__}
    report["generated_at_ms"] = int(time.time() * 1000)
    report["generated_at_utc"] = intelligence.iso(report["generated_at_ms"])


class SingleWriter:
    """OS releases the lock on crashes; a stale PID file cannot block recovery."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if os.fstat(self.handle.fileno()).st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Market intelligence monitor is already running")
        return self

    def __exit__(self, *args):
        self.handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error("poll-seconds must be >= 60")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stop_path = directory / "stop.request"
    if args.stop:
        stop_path.write_text("stop", encoding="utf-8")
        print("Stop requested; monitor exits after its current collection cycle.")
        return 0
    with SingleWriter(directory / "monitor.lock"):
        stop_path.unlink(missing_ok=True)
        db = intelligence.open_db(directory / "observations.sqlite3")
        intelligence.write_report(directory / "runtime.json", {"pid": os.getpid(), "running": True,
                      "started_at_utc": intelligence.iso(int(time.time() * 1000)), "poll_seconds": args.poll_seconds})
        try:
            while True:
                started = time.monotonic()
                records = collect_records()
                now = int(time.time() * 1000)
                previous = intelligence.at_time(db, now, max_age_ms=10 * intelligence.FIVE_MIN)
                report = intelligence.build_report(records, now, previous)
                local_context(report)
                report["poll_seconds"] = args.poll_seconds
                intelligence.persist(db, records, report)
                intelligence.write_report(directory / "latest.json", report)
                print(json.dumps({"at": report["generated_at_utc"], "direction": report["direction"],
                      "healthy_sources": sum(r["status"] == "ok" for r in report["health"].values()),
                      "sources": len(report["health"]), "risk_flags": report["risk_flags"]}), flush=True)
                if not args.loop or stop_path.exists():
                    break
                while time.monotonic() - started < args.poll_seconds:
                    if stop_path.exists():
                        break
                    time.sleep(1)
                if stop_path.exists():
                    break
        finally:
            db.close()
            intelligence.write_report(directory / "runtime.json", {"pid": os.getpid(), "running": False,
                "stopped_at_utc": intelligence.iso(int(time.time() * 1000))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
