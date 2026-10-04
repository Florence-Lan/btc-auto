#!/usr/bin/env python3
"""Fixed-profile stock diagnostics; no selection, account access or orders."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import backtest_stock_swing_120 as engine
from replay_stock_swing_per_symbol import after_close_summary, combine_buckets


def trade_diagnostics(trades: list[dict], initial: float, samples=2000, seed=20261004) -> dict:
    """Describe recorded closed PnL; resampling is conditional on this sample."""
    profits = [float(t["net_pnl"]) for t in trades]
    winners = sorted((v for v in profits if v > 0), reverse=True)
    gross_profit = sum(winners)
    by_side = {}
    for side in ("long", "short"):
        subset = [t for t in trades if t["direction"] == side]
        by_side[side] = {"closed_trades": len(subset), "net_pnl": sum(t["net_pnl"] for t in subset),
                         "net120_targets": sum(t["net_return_initial_margin_pct"] >= 120 - 1e-7 for t in subset)}
    result = {
        "closed_pnl": sum(profits), "closed_return_pct_initial_capital": sum(profits) / initial * 100,
        "closed_return_without_largest_winner_pct": (sum(profits) - (winners[0] if winners else 0)) / initial * 100,
        "largest_winner_share_gross_profit_pct": winners[0] / gross_profit * 100 if gross_profit else None,
        "by_side": by_side,
        "bootstrap_warning": "Conditional circular trade-block resampling of the selected historical trades, not independent evidence or a forecast of account returns. No compounding, unseen trades or terminal open inventory included.",
    }
    if not profits:
        result["historical_trade_block_bootstrap"] = None
        return result
    count = len(profits)
    block = min(max(1, count // 2), max(1, math.ceil(math.sqrt(count))))
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = []
        while len(draw) < count:
            first = rng.randrange(count)
            draw.extend(profits[(first + offset) % count] for offset in range(block))
        means.append(sum(draw[:count]) / count / initial * 100)
    means.sort()
    result["historical_trade_block_bootstrap"] = {
        "samples": samples, "seed": seed, "block_length": block, "trade_count": count,
        "sample_below_10_closed_trades": count < 10,
        "mean_trade_pnl_pct_initial_capital_p05": means[int((samples - 1) * .05)],
        "mean_trade_pnl_pct_initial_capital_p50": means[int((samples - 1) * .50)],
        "mean_trade_pnl_pct_initial_capital_p95": means[int((samples - 1) * .95)],
    }
    return result


def monthly_contributions(result: dict) -> list[dict]:
    """Continuous mark-to-market monthly changes, retaining open inventory."""
    initial = result["summary"]["initial_equity"]
    previous = initial
    months = {}
    for point in result["equity_path"]:
        # The point represents this bar's close; midnight belongs to prior month.
        month = engine.iso(engine.parse_time(point["time_utc"]) - 1)[:7]
        item = months.setdefault(month, {"month_utc": month, "start_mark_equity": previous,
                                         "end_mark_equity": previous, "closed_trades": 0,
                                         "net120_targets": 0})
        item["end_mark_equity"] = point["equity"]
        previous = point["equity"]
    for trade in result["trades"]:
        month = trade["exit_utc"][:7]
        item = months[month]
        item["closed_trades"] += 1
        item["net120_targets"] += trade["net_return_initial_margin_pct"] >= 120 - 1e-7
    for item in months.values():
        item["marked_pnl"] = item["end_mark_equity"] - item["start_mark_equity"]
        item["marked_return_pct"] = item["marked_pnl"] / item["start_mark_equity"] * 100
    return list(months.values())


def audit_snapshot(snapshot: dict) -> dict:
    if snapshot.get("execution_step_ms") != 300_000:
        raise ValueError("Uniform five-minute execution history required")
    report = {}
    end = snapshot["end_ms_exclusive"]
    for symbol, source in snapshot["symbols"].items():
        checks = {}
        for label in ("trade", "mark"):
            rows = source[label + "_5m"]
            expected = list(range(source["start_ms"], end, 300_000))
            if [int(row[0]) for row in rows] != expected:
                raise ValueError(f"Missing, duplicate or misordered {symbol} {label} five-minute candles")
            for row in rows:
                o, h, l, c = map(float, row[1:5])
                if not all(math.isfinite(v) and v > 0 for v in (o, h, l, c)) or l > min(o, c) or h < max(o, c) or int(row[6]) >= end:
                    raise ValueError(f"Invalid completed OHLC: {symbol} {label} {row[0]}")
            hourly = {int(row[0]): row for row in source[label + "_1h"]}
            differences = []
            for index in range(0, len(rows), 12):
                group = rows[index:index + 12]
                old = hourly[int(group[0][0])]
                aggregate = [float(group[0][1]), max(float(r[2]) for r in group),
                             min(float(r[3]) for r in group), float(group[-1][4])]
                differences.append(max(abs(a - float(b)) for a, b in zip(aggregate, old[1:5])))
            checks[label] = {"rows": len(rows), "hourly_ohlc_mismatches_tolerance_1e_7": sum(v > 1e-7 for v in differences),
                             "max_hourly_ohlc_difference": max(differences, default=0)}
        report[symbol] = checks
    return report


def complete_suffix_snapshot(snapshot: dict) -> tuple[dict, dict]:
    """Discard all history through the last gap; never reconstruct missing OHLC."""
    missing = {}
    for symbol, source in snapshot["symbols"].items():
        expected = set(range(source["start_ms"], snapshot["end_ms_exclusive"], 300_000))
        for label in ("trade_5m", "mark_5m"):
            gaps = sorted(expected - {int(row[0]) for row in source[label]})
            if gaps:
                missing[f"{symbol}:{label}"] = gaps
    if not missing:
        return snapshot, {"trimmed": False, "missing": {}}
    last_gap = max(time for gaps in missing.values() for time in gaps)
    cutoff = ((last_gap + 300_000 + engine.FOUR_HOURS - 1) // engine.FOUR_HOURS) * engine.FOUR_HOURS
    if cutoff >= snapshot["end_ms_exclusive"]:
        raise ValueError("No complete suffix remains after unavailable public data")
    sources = {}
    for symbol, source in snapshot["symbols"].items():
        first = max(source["start_ms"], cutoff)
        sources[symbol] = {**source, "start_ms": first, "start_utc": engine.iso(first)}
        for label in ("trade_1h", "mark_1h", "index_1h", "trade_5m", "mark_5m"):
            sources[symbol][label] = [row for row in source[label] if int(row[0]) >= first]
    return {**snapshot, "symbols": sources, "execution_gaps": {}, "execution_coverage_complete": True}, {
        "trimmed": True, "missing": missing, "common_cutoff_utc": engine.iso(cutoff),
        "method": "Discard execution AND indicator history through the last unavailable bar, aligned to next 4h boundary. Rewarm all signals on the remaining actual history; do not interpolate.",
        "warning": "Full means the complete remaining suffix, not each symbol's listing-to-date history. Earlier trades and indicator warmup are excluded.",
    }


def hourly_from_five_minute(rows: list) -> list:
    """Aggregate actual complete bars into one consistent signal price history."""
    result = []
    for offset in range(0, len(rows), 12):
        group = rows[offset:offset + 12]
        first = int(group[0][0])
        if len(group) != 12 or first % engine.HOUR or [int(r[0]) for r in group] != list(range(first, first + engine.HOUR, 300_000)):
            raise ValueError("An hourly signal needs twelve complete aligned five-minute bars")
        result.append([first, float(group[0][1]), max(float(r[2]) for r in group),
                       min(float(r[3]) for r in group), float(group[-1][4]),
                       sum(float(r[5]) for r in group), first + engine.HOUR - 1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/stock_swing_per_symbol_120_candidate_20261004.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trim-to-complete-suffix", action="store_true",
                        help="Explicitly discard history through public source gaps")
    parser.add_argument("--derive-hourly-signals-from-5m", action="store_true",
                        help="Use the same five-minute price history for signals and fills")
    parser.add_argument("--liquidity-experiment", action="store_true",
                        help="One declared experiment: cap entry at 10% of prior completed 5m volume")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new directory to preserve previous research")
    profile = json.loads(args.config.read_text())
    base_path = Path(profile["base_config"])
    base = json.loads(base_path.read_text())
    snapshot = json.loads(gzip.decompress(args.snapshot.read_bytes()))
    coverage_choice = {"trimmed": False}
    if args.trim_to_complete_suffix:
        snapshot, coverage_choice = complete_suffix_snapshot(snapshot)
    if profile["status"] != "research_only" or snapshot["venue"] != base["venue"]:
        raise ValueError("Research profile and actual venue must agree")
    weights = profile["capital_weights"]
    if set(weights) != set(base["symbols"]) or set(profile["symbol_profiles"]) != set(weights) or any(v <= 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError("Fixed capital buckets must cover every stock and sum to one")
    published_hourly_audit = audit_snapshot(snapshot)
    published_snapshot = snapshot
    if args.derive_hourly_signals_from_5m:
        snapshot = {**snapshot, "symbols": {symbol: {
            **source, "trade_1h": hourly_from_five_minute(source["trade_5m"]),
            "mark_1h": hourly_from_five_minute(source["mark_5m"])
        } for symbol, source in snapshot["symbols"].items()}}
    audit = audit_snapshot(snapshot)
    args.output_dir.mkdir(parents=True)
    declaration = {
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_parameters_refit": False,
        "liquidity_experiment": {
            "enabled": args.liquidity_experiment, "entry_max_previous_bar_participation_fraction": .1,
            "hypothesis": "Use prior completed five-minute volume to cap quantity; reject unavailable or zero volume. No signal-date exclusions, current-bar final-volume lookahead or parameter grid.",
            "warning": "A proxy for admission and sizing, not proof of opening-tick order book depth, actual fills or subsequent stop liquidity.",
        },
    }
    (args.output_dir / "declaration.json").write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n")
    effective_snapshot = args.output_dir / "effective_snapshot.json.gz"
    effective_snapshot.write_bytes(gzip.compress(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    end = snapshot["end_ms_exclusive"]
    outputs = {}
    combined = defaultdict(dict)
    for symbol, weight in weights.items():
        config = {**base, "candidate_id": profile["candidate_id"], "symbols": [symbol],
                  "symbol_profiles": {symbol: profile["symbol_profiles"][symbol]},
                  "initial_equity_usdt": base["initial_equity_usdt"] * weight,
                  "max_positions": profile["max_positions_per_capital_bucket"],
                  "risk_fraction_per_trade": profile["risk_fraction_per_bucket_trade"]}
        config["execution_timeframe"] = "5m"
        first = snapshot["symbols"][symbol]["start_ms"]
        scenarios = []
        for window, start in (("full", first), ("recent60d", max(first, end - 60 * engine.DAY)),
                              ("recent30d", max(first, end - 30 * engine.DAY))):
            for cost in (1, 2):
                scenarios.append((f"{window}_cost{cost}", start, config, cost, snapshot))
        scenarios.extend([
            ("full_cost4", first, config, 4, snapshot),
            ("full_margin5pct", first, {**config, "maintenance_margin_fraction_assumption": .05}, 1, snapshot),
            ("full_slippage50bps", first, {**config, "adverse_slippage_fraction_assumption": .005}, 1, snapshot),
        ])
        hourly = {**snapshot, "execution_step_ms": engine.HOUR, "symbols": {
            symbol: {k: v for k, v in snapshot["symbols"][symbol].items() if k not in {"trade_5m", "mark_5m"}}
        }}
        scenarios.append(("full_hourly_resolution_comparison", first, config, 1, hourly))
        if args.derive_hourly_signals_from_5m:
            for cost in (1, 2):
                scenarios.append((f"full_published_hourly_signals_cost{cost}", first, config, cost, published_snapshot))
        if args.liquidity_experiment:
            settings = {**config, "entry_max_previous_bar_participation_fraction": .1}
            for window, start in (("full", first), ("recent30d", max(first, end - 30 * engine.DAY))):
                for cost in (1, 2):
                    scenarios.append((f"{window}_prior5m_volume10pct_cost{cost}", start, settings, cost, snapshot))
        runs = {}
        for name, start, settings, cost, source in scenarios:
            result = engine.simulate(source, settings, start, end, cost)
            summary = after_close_summary(result)
            summary["trade_diagnostics"] = trade_diagnostics(result["trades"], summary["initial_equity"])
            bars = {int(row[0]): row for row in snapshot["symbols"][symbol]["trade_5m"]}
            summary["execution_volume_diagnostics"] = {
                "entry_bars_zero_reported_volume": sum(float(bars[engine.parse_time(t["entry_utc"])][5]) == 0 for t in result["trades"]),
                "exit_bars_zero_reported_volume": sum(float(bars[engine.parse_time(t["exit_utc"])][5]) == 0 for t in result["trades"]),
                "warning": "Execution-bar final volume is known only after its close. Descriptive plausibility check, never an available-at-opening filter.",
            }
            if name in {"full_cost1", "full_cost2"}:
                summary["continuous_monthly_mark_contributions"] = monthly_contributions(result)
            runs[name] = summary
            engine.write_csv(args.output_dir / f"{symbol}_{name}_trades.csv", result["trades"])
            # Retain continuous mark paths for independently checking account DD.
            engine.write_csv(args.output_dir / f"{symbol}_{name}_equity.csv", result["equity_path"])
            if name != "full_hourly_resolution_comparison":
                combined[name][symbol] = result
            print(symbol, name, json.dumps({key: summary[key] for key in (
                "estimated_close_return_pct", "closed_trades", "target_trades_net_at_least_120pct_margin",
                "max_sampled_drawdown_pct", "liquidation_stress_count")}), flush=True)
        outputs[symbol] = runs
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
        "research_stock_swing_robustness.py", "replay_stock_swing_per_symbol.py", "backtest_stock_swing_120.py",
        "stock_swing_signals.py", "stock_swing_profiles.py")}
    artifact = {
        "profile": profile, "base_config": base, "snapshot_audit": audit,
        "published_hourly_audit": published_hourly_audit,
        "hourly_signals_derived_from_five_minute": args.derive_hourly_signals_from_5m,
        "declaration": declaration,
        "source_snapshot_path": str(args.snapshot), "source_snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "snapshot_path": str(effective_snapshot), "snapshot_sha256": hashlib.sha256(effective_snapshot.read_bytes()).hexdigest(),
        "coverage_choice": coverage_choice,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "base_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(), "source_sha256": sources,
        "runs": outputs, "combined_equal_capital_buckets": {name: combine_buckets(results, base["initial_equity_usdt"]) for name, results in combined.items()},
        "forward_validated": False, "places_orders": False,
        "limitations": [
            "Fixed selected profiles; no additional parameter optimization. All history has been inspected and is retrospective.",
            "Recent windows start flat, with prior causal indicator warmup; monthly contributions use continuous full-history inventory and are mark-to-market, not realized PnL.",
            "Trade/mark five-minute paths still leave unknown intrabar order and assume stop execution; liquidation-first when both cross inside the same bar.",
            "Current rules, assumed fees/slippage and approximate funding settlement marks cannot establish historical executable fills.",
            "Stress sizing responds to changed costs; slippage stress is a uniform fill assumption, not measured depth or a delayed-stop simulation.",
            "Each symbol's capital stays separate. End-close is an estimate and net120 counts include closed trades only.",
            "Historical block resampling is descriptive of the selected finite sample and omits uncertainty from candidate selection.",
        ],
    }
    (args.output_dir / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print("COMBINED", json.dumps(artifact["combined_equal_capital_buckets"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
