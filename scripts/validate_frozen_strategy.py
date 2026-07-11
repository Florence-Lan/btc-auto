#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frozen_strategy
import simulate_range_swing as sim
import validate_strategies as validation


def bootstrap_distribution(
    returns: list[float],
    *,
    block_days: int,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    if len(returns) < block_days:
        return {
            "samples": samples,
            "block_days": block_days,
            "annualized_p05_pct": None,
            "annualized_p50_pct": None,
            "annualized_p95_pct": None,
            "positive_probability_pct": None,
        }
    blocks = [
        returns[index : index + block_days]
        for index in range(0, len(returns) - block_days + 1, block_days)
    ]
    rng = random.Random(seed)
    annualized: list[float] = []
    for _ in range(samples):
        sample: list[float] = []
        while len(sample) < len(returns):
            sample.extend(rng.choice(blocks))
        sample = sample[: len(returns)]
        growth = math.prod(1 + value for value in sample)
        years = max(len(sample) / 365, 1 / 365)
        annualized.append((growth ** (1 / years) - 1) * 100 if growth > 0 else -100.0)
    annualized.sort()

    def percentile(fraction: float) -> float:
        return annualized[round((len(annualized) - 1) * fraction)]

    return {
        "samples": samples,
        "block_days": block_days,
        "annualized_p05_pct": percentile(0.05),
        "annualized_p50_pct": percentile(0.50),
        "annualized_p95_pct": percentile(0.95),
        "positive_probability_pct": sum(value > 0 for value in annualized) / len(annualized) * 100,
    }


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(
        description="Validate one immutable strategy without refitting parameters.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config/frozen_strategy_20260711.json",
    )
    parser.add_argument("--data-snapshot", type=Path)
    parser.add_argument("--fold-days", type=int, default=90)
    parser.add_argument("--bootstrap-block-days", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "data/validation/frozen_strategy_20260711.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "data/validation/frozen_strategy_20260711_folds.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fold_days < 30:
        raise ValueError("--fold-days must be >= 30")
    if args.bootstrap_block_days < 2 or args.bootstrap_samples < 100:
        raise ValueError("Bootstrap requires block-days >= 2 and samples >= 100")

    manifest, cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    snapshot = args.data_snapshot or sim.repo_root() / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, snapshot)
    intervals, funding, metadata = sim.load_market_snapshot(snapshot)
    base_candles = intervals["5m"]
    trend_candles = intervals[cfg.timeseries_timeframe]
    first_test_ms = base_candles[0].open_time_ms + 365 * sim.MS_PER_DAY
    last_end_ms = min(base_candles[-1].close_time_ms, trend_candles[-1].close_time_ms)
    fold_ms = args.fold_days * sim.MS_PER_DAY
    fold_starts: list[int] = []
    cursor = first_test_ms
    while cursor + fold_ms <= last_end_ms:
        fold_starts.append(cursor)
        cursor += fold_ms
    if not fold_starts:
        raise RuntimeError("Snapshot does not contain a complete validation fold")

    folds: list[dict[str, Any]] = []
    for number, start_ms in enumerate(fold_starts, start=1):
        end_ms = start_ms + fold_ms
        result = validation.run_fold(
            base_candles,
            trend_candles,
            funding,
            cfg,
            start_ms,
            end_ms,
        )
        fold = validation.compact_fold(result, start_ms, end_ms)
        folds.append(fold)
        print(
            f"Fold {number}/{len(fold_starts)} "
            f"return={fold['summary']['total_return_pct']:.2f}% "
            f"trades={fold['summary']['trades']}"
        )

    aggregate = validation.aggregate_folds(folds, cfg.initial_equity)
    returns = [value for fold in folds for value in fold["daily_returns"]]
    bootstrap = bootstrap_distribution(
        returns,
        block_days=args.bootstrap_block_days,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    stressed_cfg = replace(
        cfg,
        maker_fee=cfg.maker_fee * 2,
        taker_fee=cfg.taker_fee * 2,
        entry_slippage_bps=cfg.entry_slippage_bps * 2,
        exit_slippage_bps=cfg.exit_slippage_bps * 2,
        depth_impact_bps=cfg.depth_impact_bps * 2,
    )
    stressed_folds: list[dict[str, Any]] = []
    for start_ms in fold_starts:
        end_ms = start_ms + fold_ms
        stressed_result = validation.run_fold(
            base_candles,
            trend_candles,
            funding,
            stressed_cfg,
            start_ms,
            end_ms,
        )
        stressed_folds.append(validation.compact_fold(stressed_result, start_ms, end_ms))
    double_cost_stress = validation.aggregate_folds(stressed_folds, cfg.initial_equity)

    targets = manifest.get("validation_targets", {})
    gates = {
        "cagr_pct_min": float(targets.get("cagr_pct_min", 0.0)),
        "max_drawdown_pct_max": float(targets.get("max_drawdown_pct_max", 15.0)),
        "profit_factor_min": float(targets.get("profit_factor_min", 1.3)),
        "trades_per_year_min": float(targets.get("trades_per_year_min", 20.0)),
        "profitable_fold_pct_min": float(targets.get("profitable_fold_pct_min", 60.0)),
        "bootstrap_annualized_p05_pct_min": float(
            targets.get("bootstrap_annualized_p05_pct_min", 0.0)
        ),
        "double_cost_cagr_pct_min": float(targets.get("double_cost_cagr_pct_min", 0.0)),
        "double_cost_max_drawdown_pct_max": float(
            targets.get("double_cost_max_drawdown_pct_max", 15.0)
        ),
        "double_cost_profit_factor_min": float(
            targets.get("double_cost_profit_factor_min", 1.3)
        ),
    }
    evidence_pass = bool(
        aggregate["cagr_pct"] > gates["cagr_pct_min"]
        and aggregate["max_drawdown_pct"] <= gates["max_drawdown_pct_max"]
        and aggregate["profit_factor"] is not None
        and aggregate["profit_factor"] >= gates["profit_factor_min"]
        and aggregate["trades_per_year"] >= gates["trades_per_year_min"]
        and aggregate["profitable_fold_pct"] >= gates["profitable_fold_pct_min"]
        and bootstrap["annualized_p05_pct"] is not None
        and float(bootstrap["annualized_p05_pct"]) > gates["bootstrap_annualized_p05_pct_min"]
        and double_cost_stress["cagr_pct"] > gates["double_cost_cagr_pct_min"]
        and double_cost_stress["max_drawdown_pct"] <= gates["double_cost_max_drawdown_pct_max"]
        and double_cost_stress["profit_factor"] is not None
        and double_cost_stress["profit_factor"] >= gates["double_cost_profit_factor_min"]
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "post_selection_fixed_parameter_rolling_quarters",
        "independence_warning": (
            "These historical folds were visible during parameter research and are pseudo-OOS. "
            "Only observations after frozen_at_utc are genuinely out of sample."
        ),
        "manifest": str(args.manifest),
        "freeze_id": manifest["freeze_id"],
        "config_sha256": manifest["config_sha256"],
        "snapshot": str(snapshot),
        "snapshot_metadata": metadata,
        "fold_days": args.fold_days,
        "fold_count": len(folds),
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "double_cost_stress": double_cost_stress,
        "gates": gates,
        "evidence_pass": evidence_pass,
        "folds": [
            {
                "window": fold["window"],
                "profitable": fold["profitable"],
                "total_return_pct": fold["summary"]["total_return_pct"],
                "max_drawdown_pct": fold["summary"]["max_drawdown_pct"],
                "profit_factor": fold["summary"]["profit_factor"],
                "trades": fold["summary"]["trades"],
            }
            for fold in folds
        ],
    }
    sim.save_json(args.output_json, report)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "start_utc",
            "end_utc",
            "profitable",
            "total_return_pct",
            "max_drawdown_pct",
            "profit_factor",
            "trades",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold in report["folds"]:
            writer.writerow(
                {
                    "start_utc": fold["window"]["start_utc"],
                    "end_utc": fold["window"]["end_utc"],
                    **{key: fold[key] for key in fieldnames[2:]},
                }
            )
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "bootstrap": bootstrap,
                "double_cost_stress": double_cost_stress,
                "gates": gates,
                "evidence_pass": evidence_pass,
            },
            indent=2,
        )
    )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")
    return 0 if evidence_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
