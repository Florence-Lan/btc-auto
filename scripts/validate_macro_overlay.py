#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import frozen_strategy
import macro_regime
import simulate_range_swing as sim


FACTOR_SETS: dict[str, tuple[str, ...]] = {
    "equities": ("equities",),
    "vix": ("vix",),
    "dollar": ("dollar",),
    "metals": ("metals",),
    "all": ("equities", "vix", "dollar", "metals"),
}


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "trades",
        "trades_per_year",
        "win_rate_pct",
        "profit_factor",
        "max_consecutive_losses",
        "total_costs",
        "cost_to_gross_profit_pct",
        "largest_profit_share_pct",
        "by_strategy",
        "by_side",
    )
    return {key: summary.get(key) for key in keys}


def combine_with_overlay(
    base_candles: Sequence[sim.Candle],
    sleeves: Sequence[dict[str, Any]],
    cfg: sim.StrategyConfig,
    evaluation_start_ms: int,
    snapshot: macro_regime.MacroSnapshot,
    factors: Sequence[str],
    min_multiplier: float,
    block_score: float,
    max_staleness_days: int,
) -> dict[str, Any]:
    adjusted, diagnostics = macro_regime.apply_macro_overlay(
        sleeves,
        snapshot,
        enabled_factors=factors,
        min_multiplier=min_multiplier,
        block_score=block_score,
        max_staleness_days=max_staleness_days,
    )
    result = sim.combine_sleeve_results(base_candles, adjusted, cfg, evaluation_start_ms)
    return {"summary": compact_summary(result["summary"]), "macro": diagnostics}


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(
        description="Continuously validate the frozen BTC portfolio with a macro risk overlay.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config/frozen_strategy_active_20260711.json",
    )
    parser.add_argument("--data-snapshot", type=Path)
    parser.add_argument(
        "--macro-snapshot",
        type=Path,
        default=root / "data/snapshots/macro_20191001_20260628.json.gz",
    )
    parser.add_argument("--min-multiplier", type=float, default=0.35)
    parser.add_argument("--block-score", type=float, default=-0.80)
    parser.add_argument("--max-staleness-days", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "data/validation/macro_overlay_20260711.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, frozen_cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    market_path = args.data_snapshot or sim.repo_root() / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, market_path)
    intervals, funding, market_metadata = sim.load_market_snapshot(market_path)
    macro_snapshot = macro_regime.load_macro_snapshot(args.macro_snapshot)
    base_candles = intervals["5m"]
    trend_candles = intervals[frozen_cfg.timeseries_timeframe]
    evaluation_start_ms = base_candles[0].open_time_ms + 365 * sim.MS_PER_DAY

    # Research the strategy continuously with one portfolio state and no automatic resume.
    research_cfg = replace(frozen_cfg, max_drawdown_stop_pct=0.0)
    tactical_modes = tuple(
        mode for mode in research_cfg.strategy_modes if mode != "timeseries_trend"
    )
    tactical = sim.simulate(
        base_candles,
        replace(research_cfg, strategy_modes=tactical_modes),
        evaluation_start_ms,
        None,
        funding,
    )
    core = sim.simulate_timeseries_trend(
        trend_candles,
        replace(research_cfg, strategy_modes=("timeseries_trend",)),
        evaluation_start_ms,
        funding,
    )
    sleeves = [tactical, core]
    baseline = sim.combine_sleeve_results(
        base_candles,
        sleeves,
        research_cfg,
        evaluation_start_ms,
    )

    ablations: dict[str, Any] = {}
    for name, factors in FACTOR_SETS.items():
        ablations[name] = combine_with_overlay(
            base_candles,
            sleeves,
            research_cfg,
            evaluation_start_ms,
            macro_snapshot,
            factors,
            args.min_multiplier,
            args.block_score,
            args.max_staleness_days,
        )
        summary = ablations[name]["summary"]
        print(
            f"{name}: CAGR={summary['cagr_pct']:.2f}% "
            f"DD={summary['max_drawdown_pct']:.2f}% PF={summary['profit_factor']:.2f} "
            f"trades={summary['trades']}"
        )

    halted = combine_with_overlay(
        base_candles,
        sleeves,
        frozen_cfg,
        evaluation_start_ms,
        macro_snapshot,
        FACTOR_SETS["all"],
        args.min_multiplier,
        args.block_score,
        args.max_staleness_days,
    )
    all_summary = ablations["all"]["summary"]
    candidate_pass = bool(
        all_summary["cagr_pct"] is not None
        and all_summary["cagr_pct"] > 0
        and all_summary["max_drawdown_pct"] <= 15.0
        and all_summary["profit_factor"] is not None
        and all_summary["profit_factor"] >= 1.3
        and all_summary["trades_per_year"] >= 20.0
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "continuous_fixed_parameter_macro_ablation",
        "promotion_status": "research_only",
        "freeze_id": manifest["freeze_id"],
        "market_snapshot": str(market_path),
        "market_snapshot_metadata": market_metadata,
        "macro_snapshot": str(args.macro_snapshot),
        "macro_snapshot_metadata": macro_snapshot.metadata,
        "parameters": {
            "min_multiplier": args.min_multiplier,
            "block_score": args.block_score,
            "max_staleness_days": args.max_staleness_days,
            "availability_policy": "provider daily timestamp plus 24 hours",
        },
        "gates": {
            "cagr_pct_min": 0.0,
            "max_drawdown_pct_max": 15.0,
            "profit_factor_min": 1.3,
            "trades_per_year_min": 20.0,
        },
        "baseline_no_halt": compact_summary(baseline["summary"]),
        "ablations_no_halt": ablations,
        "all_factors_with_frozen_drawdown_halt": halted,
        "candidate_pass": candidate_pass,
        "independence_warning": (
            "This is an in-sample factor design experiment on already-visible history. "
            "A new immutable freeze and prospective shadow period are required before promotion."
        ),
    }
    sim.save_json(args.output_json, report)
    print(json.dumps({"candidate_pass": candidate_pass, "output": str(args.output_json)}, indent=2))
    return 0 if candidate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
