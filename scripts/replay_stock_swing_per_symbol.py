#!/usr/bin/env python3
"""Replay separately funded stock profiles and require every stock to pass."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import backtest_stock_swing_120 as engine


def after_close_summary(result: dict) -> dict:
    summary = dict(result["summary"])
    closed = sum(trade["net_pnl"] for trade in result["trades"])
    open_net = sum(position["estimated_close_net_pnl"] for position in summary["open_positions"])
    summary["estimated_close_equity"] = summary["initial_equity"] + closed + open_net
    summary["estimated_close_return_pct"] = (closed + open_net) / summary["initial_equity"] * 100
    summary["net_closed_pnl"] = closed
    summary["estimated_open_close_net_pnl"] = open_net
    return summary


def assess_stock(runs: dict, criteria: dict) -> dict:
    failures = []
    for window in ("full", "development", "validation"):
        for cost in (1, 2):
            name = f"{window}_cost{cost}"
            summary = runs[name]
            if summary["estimated_close_return_pct"] <= 0:
                failures.append(f"{name}: nonpositive net return")
            minimum = criteria["minimum_development_closed_trades"] if window == "development" else criteria["minimum_validation_closed_trades"] if window == "validation" else 0
            if summary["closed_trades"] < minimum:
                failures.append(f"{name}: insufficient closed-trade sample")
            if window == "full" and summary["target_trades_net_at_least_120pct_margin"] < criteria["minimum_full_net120_trades"]:
                failures.append(f"{name}: insufficient net120 targets")
            if criteria["zero_current_maintenance_liquidation_stress"] and summary["liquidation_stress_count"]:
                failures.append(f"{name}: liquidation stress observed")
    return {
        "historical_returns_all_positive": all(runs[f"{w}_cost{c}"]["estimated_close_return_pct"] > 0 for w in ("full", "development", "validation") for c in (1, 2)),
        "passed_retrospective_gate": not failures,
        "failure_reasons": failures,
        "forward_validated": False,
    }


def combine_buckets(results: dict, total_initial: float) -> dict:
    latest = {symbol: result["summary"]["initial_equity"] for symbol, result in results.items()}
    changes = {}
    for symbol, result in results.items():
        for point in result["equity_path"]:
            changes.setdefault(point["time_utc"], {})[symbol] = point["equity"]
    peak = total_initial
    worst = 0.0
    for timestamp in sorted(changes):
        latest.update(changes[timestamp])
        equity = sum(latest.values())
        peak = max(peak, equity)
        worst = max(worst, max(0, (peak - equity) / peak))
    final = sum(after_close_summary(result)["estimated_close_equity"] for result in results.values())
    return {"initial_equity": total_initial, "estimated_close_equity": final,
            "estimated_close_return_pct": (final / total_initial - 1) * 100,
            "max_sampled_mark_drawdown_pct": worst * 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/stock_swing_per_symbol_120_candidate_20261004.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.config.read_text())
    base = json.loads(Path(profile["base_config"]).read_text())
    snapshot = json.loads(gzip.decompress(args.snapshot.read_bytes()))
    if snapshot.get("execution_step_ms") != 300_000:
        raise ValueError("All stocks require uniform five-minute execution history")
    weights = profile["capital_weights"]
    if abs(sum(weights.values()) - 1) > 1e-12 or any(value <= 0 for value in weights.values()):
        raise ValueError("Capital allocation must be positive and sum to one")
    if set(weights) != set(base["symbols"]) or set(profile["symbol_profiles"]) != set(weights):
        raise ValueError("Every requested stock needs its own capital and profile")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = engine.parse_time(base["evaluation_split_utc"])
    end = snapshot["end_ms_exclusive"]
    all_summaries = {}
    assessments = {}
    portfolio_results = {}
    for symbol, weight in weights.items():
        config = {**base, "candidate_id": profile["candidate_id"], "symbols": [symbol],
                  "symbol_profiles": {symbol: profile["symbol_profiles"][symbol]},
                  "initial_equity_usdt": base["initial_equity_usdt"] * weight,
                  "max_positions": profile["max_positions_per_capital_bucket"],
                  "risk_fraction_per_trade": profile["risk_fraction_per_bucket_trade"]}
        start = snapshot["symbols"][symbol]["start_ms"]
        summaries = {}
        for window, first, last in (("full", start, end), ("development", start, split), ("validation", split, end)):
            for cost in (1, 2):
                name = f"{window}_cost{cost}"
                result = engine.simulate(snapshot, config, first, last, cost)
                summaries[name] = after_close_summary(result)
                engine.write_csv(args.output_dir / f"{symbol}_{name}_trades.csv", result["trades"])
                if window in ("full", "validation"):
                    portfolio_results.setdefault(name, {})[symbol] = result
                print(symbol, name, json.dumps({key: summaries[name][key] for key in (
                    "estimated_close_return_pct", "closed_trades", "target_trades_net_at_least_120pct_margin", "max_sampled_drawdown_pct", "liquidation_stress_count")}), flush=True)
        for window, first, last in (("full", start, end), ("validation", split, end)):
            stress = {**config, "maintenance_margin_fraction_assumption": base["maintenance_margin_fraction_stress"]}
            result = engine.simulate(snapshot, stress, first, last, 1)
            name = f"{window}_margin_stress"
            summaries[name] = after_close_summary(result)
            engine.write_csv(args.output_dir / f"{symbol}_{name}_trades.csv", result["trades"])
        all_summaries[symbol] = summaries
        assessments[symbol] = assess_stock(summaries, profile["individual_acceptance"])
    combined = {name: combine_buckets(results, base["initial_equity_usdt"]) for name, results in portfolio_results.items()}
    artifact = {
        "candidate": profile, "base_config": base,
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "base_config_sha256": hashlib.sha256(Path(profile["base_config"]).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
            "replay_stock_swing_per_symbol.py", "backtest_stock_swing_120.py", "stock_swing_signals.py", "stock_swing_profiles.py")},
        "stock_runs": all_summaries, "individual_assessments": assessments,
        "all_stocks_historical_returns_positive": all(item["historical_returns_all_positive"] for item in assessments.values()),
        "all_stocks_pass_retrospective_gate": all(item["passed_retrospective_gate"] for item in assessments.values()),
        "forward_validated": False, "combined_equal_capital_buckets": combined,
        "limitations": [
            "Family composition is retrospective after reviewing these same historical windows; this is not independent unseen validation.",
            "Actual historical funding rates use five-minute opening mark as a settlement-price approximation.",
            "Current quantity/tick/maintenance tiers and conservative assumed fees; historical changes/order-book fills unavailable.",
            "Unknown order inside each five-minute bar uses liquidation-stress before stop before target; no single-trade overrides.",
            "An estimated end-close includes costs but remains an estimate for open inventory, not a realized fill.",
            "Individual capital buckets cannot move profits to another stock; each must pass separately.",
        ],
    }
    (args.output_dir / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print("ASSESSMENTS", json.dumps(assessments, ensure_ascii=False))
    print("COMBINED", json.dumps(combined, ensure_ascii=False))


if __name__ == "__main__":
    main()
