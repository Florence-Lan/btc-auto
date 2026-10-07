#!/usr/bin/env python
"""Audit archived observations and replay the fixed filter on their actual time range."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import frozen_strategy
import market_intelligence as intelligence
import portfolio_risk
import simulate_range_swing as sim
from download_market_snapshot import validate_contiguous
from validate_world_event_returns import scaled, total_entry_notional, compact

ROOT = Path(__file__).resolve().parents[1]


def audit(reports):
    if not reports:
        raise ValueError("No archived observations")
    counts, healthy = Counter(), Counter()
    for report in reports:
        for source, value in report["health"].items():
            counts[source] += 1
            healthy[source] += value["status"] == "ok"
    gaps = [b["generated_at_ms"] - a["generated_at_ms"] for a, b in zip(reports, reports[1:])]
    return {"cycles": len(reports), "start_utc": reports[0]["generated_at_utc"],
            "end_utc": reports[-1]["generated_at_utc"],
            "hours": (reports[-1]["generated_at_ms"] - reports[0]["generated_at_ms"]) / 3_600_000,
            "maximum_gap_seconds": max(gaps, default=0) / 1000,
            "gaps_over_180_seconds": sum(gap > 180_000 for gap in gaps),
            "source_health_pct": {name: healthy[name] / count * 100 for name, count in counts.items()},
            "direction_states": dict(Counter(r["direction"]["state"] for r in reports)),
            "long_reduction_cycles": sum(intelligence.entry_multiplier(r, "long") == .5 for r in reports),
            "short_reduction_cycles": sum(intelligence.entry_multiplier(r, "short") == .5 for r in reports),
            "note": "Cycles share overlapping input windows; they are not independent trades."}


def fresh_mark(report):
    row = report.get("features", {}).get("funding_basis", {})
    if report.get("health", {}).get("funding_basis", {}).get("status") != "ok":
        return None
    observed = row.get("observed_ms", 0)
    if not 0 <= report["generated_at_ms"] - observed <= 120_000:
        return None
    value = row.get("mark_price")
    return intelligence.number(value, 1e-12) if value is not None else None


def forward_samples(reports, horizon_minutes=60):
    """Non-overlapping, next-observation mark prices. Diagnostic, not executable PnL."""
    times = [r["generated_at_ms"] for r in reports]
    samples = []
    next_signal = -1
    horizon = horizon_minutes * 60_000
    for i, signal in enumerate(reports):
        timestamp = times[i]
        direction = signal.get("direction", {})
        score = direction.get("score")
        if (timestamp < next_signal or score is None or abs(score) <= 20
                or direction.get("state") not in ("buy_pressure", "sell_pressure")):
            continue
        entry_index = bisect_right(times, timestamp)
        if entry_index >= len(reports) or times[entry_index] - timestamp > 120_000:
            continue
        entry = reports[entry_index]
        entry_price = fresh_mark(entry)
        if entry_price is None or entry["features"]["funding_basis"]["observed_ms"] <= timestamp:
            continue
        target = times[entry_index] + horizon
        exit_index = bisect_left(times, target)
        while exit_index < len(reports) and times[exit_index] <= target + 120_000:
            exit_row = reports[exit_index]
            exit_price = fresh_mark(exit_row)
            if exit_price is not None and exit_row["features"]["funding_basis"]["observed_ms"] >= target:
                break
            exit_index += 1
        else:
            continue
        raw_bps = (exit_price / entry_price - 1) * 10000
        samples.append({"signal_utc": intelligence.iso(timestamp), "entry_utc": intelligence.iso(times[entry_index]),
                        "exit_utc": intelligence.iso(times[exit_index]), "score": score,
                        "side": "long" if score > 0 else "short", "entry_mark": entry_price,
                        "exit_mark": exit_price, "btc_return_bps": raw_bps,
                        "signed_return_bps": raw_bps * (1 if score > 0 else -1)})
        next_signal = times[exit_index]
    directional = [r for r in samples if r["signed_return_bps"] != 0]
    return {"horizon_minutes": horizon_minutes, "samples": len(samples),
            "direction_hit_pct": sum(r["signed_return_bps"] > 0 for r in directional) / len(directional) * 100 if directional else None,
            "mean_signed_return_bps": sum(r["signed_return_bps"] for r in samples) / len(samples) if samples else None,
            "flat_outcomes": len(samples) - len(directional), "details": samples,
            "warning": "Non-overlapping mark-price diagnostics, not strategy returns; no fees, funding or fills modeled. Small correlated market sample."}


def archived_macro(sleeves, reports):
    times = [r["generated_at_ms"] for r in reports]
    adjusted, missing = [], 0
    for sleeve in sleeves:
        trades = []
        for raw in sleeve["trades"]:
            timestamp = int(datetime.fromisoformat(raw["entry_time_utc"]).timestamp() * 1000)
            index = bisect_right(times, timestamp) - 1
            observation = reports[index] if index >= 0 and timestamp - times[index] <= intelligence.FIVE_MIN else {}
            macro = observation.get("macro", {})
            if macro.get("score") is None or macro.get("asof_ms") is None:
                missing += 1
                continue
            if not macro.get("allowed"):
                continue
            multiplier = .35 + .65 * ((float(macro["score"]) + 1) / 2)
            trade = scaled([{"trades": [raw]}], multiplier)[0]["trades"][0]
            trade["macro_risk_multiplier"] = multiplier
            trades.append(trade)
        adjusted.append({**sleeve, "trades": trades})
    return adjusted, missing


def replay(cfg, base, trend, funding, reports, db_path, start_ms):
    sleeve_cfg = replace(cfg, max_drawdown_stop_pct=0)
    raw = [sim.simulate(base, replace(sleeve_cfg, strategy_modes=("trend",)), start_ms, None, funding),
           sim.simulate_timeseries_trend(trend, replace(sleeve_cfg, strategy_modes=("timeseries_trend",)), start_ms, funding)]
    baseline, missing_macro = archived_macro(raw, reports)
    candidate, diagnostics = intelligence.apply_shadow_overlay(baseline, db_path)
    denominator = total_entry_notional(baseline)
    factor = total_entry_notional(candidate) / denominator if denominator else 1.0
    results = {}
    for name, sleeves in (("raw_strategy", raw), ("macro_baseline", baseline), ("macro_plus_flow", candidate),
                           ("equal_entry_size_control", scaled(baseline, factor))):
        result = portfolio_risk.combine_sleeves_with_drawdown_policy(base, sleeves, cfg,
                                    portfolio_risk.DrawdownRiskPolicy(8, 15, .35), start_ms)
        results[name] = {"summary": compact(result), "trades": result["trades"]}
    return {"results": results, "filter": diagnostics, "missing_macro_entries": missing_macro,
            "size_control_multiplier": factor,
            "valid_comparison": bool(diagnostics["decisions"] > 0 and missing_macro == 0
                                     and diagnostics["covered"] == diagnostics["decisions"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/market_intelligence/observations.sqlite3")
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/frozen_strategy_candidate_20260917.json")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "data/validation/intelligence_observations")
    parser.add_argument("--replay-directory", type=Path,
                        help="Reuse a previously pinned database and market snapshot without network calls")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    directory = args.output_directory / stamp
    directory.mkdir(parents=True, exist_ok=False)
    db_path = directory / "observations.sqlite3"
    # Online backup pins a consistent database including WAL without stopping monitoring.
    input_db = args.replay_directory / "observations.sqlite3" if args.replay_directory else args.db
    with closing(sqlite3.connect(input_db.resolve().as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(db_path)) as target:
            source.backup(target)
    with closing(sqlite3.connect(db_path)) as db:
        reports = [json.loads(row[0]) for row in db.execute("SELECT content FROM reports ORDER BY received_ms,id")]
    diagnostics = audit(reports)
    manifest, cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    step = sim.interval_to_ms(cfg.timeseries_timeframe)
    end_ms = reports[-1]["generated_at_ms"] // step * step
    start_ms = (reports[0]["generated_at_ms"] // intelligence.FIVE_MIN + 1) * intelligence.FIVE_MIN
    if end_ms <= start_ms:
        raise ValueError("Need at least one complete strategy bar after archive inception")
    warmup = start_ms - 45 * sim.MS_PER_DAY
    if args.replay_directory:
        intervals, funding, _metadata = sim.load_market_snapshot(args.replay_directory / "market.json.gz")
        base, trend = intervals["5m"], intervals[cfg.timeseries_timeframe]
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            a = pool.submit(sim.fetch_futures_klines_range, "BTCUSDT", "5m", warmup, end_ms)
            b = pool.submit(sim.fetch_futures_klines_range, "BTCUSDT", cfg.timeseries_timeframe, warmup, end_ms)
            c = pool.submit(sim.fetch_funding_history, "BTCUSDT", warmup, end_ms)
            base, trend, funding = a.result(), b.result(), c.result()
    base, trend = [c for c in base if c.close_time_ms < end_ms], [c for c in trend if c.close_time_ms < end_ms]
    for interval, candles in (("5m", base), (cfg.timeseries_timeframe, trend)):
        validate_contiguous(interval, candles)
        if not candles or candles[-1].close_time_ms + 1 != end_ms:
            raise ValueError("Market bars do not cover the frozen evaluation end")
    market_path = directory / "market.json.gz"
    sim.save_market_snapshot(market_path, "BTCUSDT", {"5m": base, cfg.timeseries_timeframe: trend}, funding, warmup, end_ms)
    normal = replay(cfg, base, trend, funding, reports, db_path, start_ms)
    stressed = replay(replace(cfg, maker_fee=cfg.maker_fee * 2, taker_fee=cfg.taker_fee * 2,
                      entry_slippage_bps=cfg.entry_slippage_bps * 2, exit_slippage_bps=cfg.exit_slippage_bps * 2,
                      depth_impact_bps=cfg.depth_impact_bps * 2), base, trend, funding, reports, db_path, start_ms)
    forward = forward_samples(reports)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "places_orders": False,
              "archive_audit": diagnostics, "forward_1h": forward,
              "strategy_window": {"start_utc": intelligence.iso(start_ms), "end_exclusive_utc": intelligence.iso(end_ms)},
              "normal_cost": normal, "double_cost": stressed, "profitability_proven": False,
              "inputs": {str(path): frozen_strategy.sha256_file(path) for path in
                         (db_path, market_path, args.manifest, Path(__file__), ROOT / "scripts/market_intelligence.py")},
              "limitations": ["Less than one day is not enough to infer robust profitability.",
                  "Portfolio replay starts flat, uses identical archived observations and market history; terminal exits may be synthetic.",
                  "Forward mark-price checks are not executable returns and do not include trading costs.",
                  "Timestamped reports preserve macro scores, but old macro provider revisions cannot be reconstructed from them.",
                  "Overlay scales pre-generated trades and is not a full execution-path simulation."]}
    sim.save_json(directory / "report.json", report)
    sim.save_json(args.output_directory / "latest.json", {"report_path": str(directory / "report.json"),
                   "archive_audit": diagnostics, "forward_1h": {k: v for k, v in forward.items() if k != "details"},
                   "normal_cost": {name: r["summary"] for name, r in normal["results"].items()},
                   "filter": normal["filter"], "strategy_window": report["strategy_window"]})
    print(json.dumps({"directory": str(directory), "archive_audit": diagnostics,
                       "forward": {k: v for k, v in forward.items() if k != "details"},
                       "summaries": {k: v["summary"] for k, v in normal["results"].items()},
                       "filter": normal["filter"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
