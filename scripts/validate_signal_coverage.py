#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frozen_strategy
import simulate_range_swing as sim
import validate_strategies as validation


CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "balanced": {
        "trend_min_signal_score": 0.78,
        "trend_min_adx": 30.0,
        "trend_min_ema_spread": 0.0025,
        "trend_min_drift_pct": 0.0012,
    },
    "balanced_two_tf": {
        "trend_min_signal_score": 0.78,
        "trend_min_adx": 30.0,
        "trend_min_ema_spread": 0.0025,
        "trend_min_drift_pct": 0.0012,
        "trend_confirm_timeframes": ("15m", "1h"),
    },
    "active_two_tf": {
        "trend_min_signal_score": 0.75,
        "trend_min_adx": 28.0,
        "trend_min_ema_spread": 0.0020,
        "trend_min_drift_pct": 0.0010,
        "trend_confirm_timeframes": ("15m", "1h"),
    },
    "baseline_with_range": {
        "strategy_modes": ("trend", "range"),
    },
    "balanced_with_range": {
        "strategy_modes": ("trend", "range"),
        "trend_min_signal_score": 0.78,
        "trend_min_adx": 30.0,
        "trend_min_ema_spread": 0.0025,
        "trend_min_drift_pct": 0.0012,
    },
    "balanced_half_risk": {
        "risk_per_trade": 0.0075,
        "trend_min_signal_score": 0.78,
        "trend_min_adx": 30.0,
        "trend_min_ema_spread": 0.0025,
        "trend_min_drift_pct": 0.0012,
    },
    "baseline_with_range_half_risk": {
        "risk_per_trade": 0.0075,
        "strategy_modes": ("trend", "range"),
    },
    "balanced_with_range_half_risk": {
        "risk_per_trade": 0.0075,
        "strategy_modes": ("trend", "range"),
        "trend_min_signal_score": 0.78,
        "trend_min_adx": 30.0,
        "trend_min_ema_spread": 0.0025,
        "trend_min_drift_pct": 0.0012,
    },
}


DEV_GATES = {
    "cagr_pct_min": 0.0,
    "max_drawdown_pct_max": 18.0,
    "profit_factor_min": 1.5,
    "trades_per_year_min": 45.0,
    "profitable_fold_pct_min": 60.0,
}

HOLDOUT_GATES = {
    "total_return_pct_min": 0.0,
    "max_drawdown_pct_max": 18.0,
    "profit_factor_min": 1.2,
}


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(description="Validate higher signal coverage without changing risk limits.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config/frozen_strategy_20260711.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "data/validation/signal_coverage_20260711.json",
    )
    return parser.parse_args()


def candidate_config(base: sim.StrategyConfig, changes: dict[str, Any]) -> sim.StrategyConfig:
    modes = changes.get("strategy_modes", ("trend",))
    return replace(base, **{**changes, "strategy_modes": modes})


def run_fold(
    base_candles: list[sim.Candle],
    trend_candles: list[sim.Candle],
    funding: sim.FundingHistory,
    cfg: sim.StrategyConfig,
    start_ms: int,
    end_ms: int,
    core: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warmup_start_ms = start_ms - 45 * sim.MS_PER_DAY
    base_slice = validation.slice_window(base_candles, warmup_start_ms, end_ms)
    trend_slice = validation.slice_window(trend_candles, warmup_start_ms, end_ms)
    tactical = sim.simulate(base_slice, cfg, start_ms, None, funding)
    if core is None:
        core = sim.simulate_timeseries_trend(
            trend_slice,
            replace(cfg, strategy_modes=("timeseries_trend",)),
            start_ms,
            funding,
        )
    combined = sim.combine_sleeve_results(base_slice, [tactical, core], cfg, start_ms)
    return validation.compact_fold(combined, start_ms, end_ms)


def passes(summary: dict[str, Any], gates: dict[str, float]) -> bool:
    profit_factor = summary.get("profit_factor")
    return bool(
        summary.get("cagr_pct", 0.0) > gates["cagr_pct_min"]
        and summary["max_drawdown_pct"] <= gates["max_drawdown_pct_max"]
        and profit_factor is not None
        and profit_factor >= gates["profit_factor_min"]
        and summary["trades_per_year"] >= gates["trades_per_year_min"]
        and summary["profitable_fold_pct"] >= gates["profitable_fold_pct_min"]
    )


def holdout_passes(summary: dict[str, Any], gates: dict[str, float]) -> bool:
    profit_factor = summary.get("profit_factor")
    return bool(
        summary["total_return_pct"] > gates["total_return_pct_min"]
        and summary["max_drawdown_pct"] <= gates["max_drawdown_pct_max"]
        and (profit_factor is None or profit_factor >= gates["profit_factor_min"])
    )


def main() -> int:
    args = parse_args()
    manifest, base_cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    snapshot = sim.repo_root() / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, snapshot)
    intervals, funding, metadata = sim.load_market_snapshot(snapshot)
    base_candles = intervals["5m"]
    trend_candles = intervals[base_cfg.timeseries_timeframe]

    fold_ms = 90 * sim.MS_PER_DAY
    first_ms = base_candles[0].open_time_ms + 365 * sim.MS_PER_DAY
    last_ms = min(base_candles[-1].close_time_ms, trend_candles[-1].close_time_ms)
    fold_starts = list(range(first_ms, last_ms - fold_ms + 1, fold_ms))
    if len(fold_starts) < 8:
        raise RuntimeError("Need at least eight quarterly folds")
    dev_count = len(fold_starts) - 4

    folds_by_candidate: dict[str, list[dict[str, Any]]] = {name: [] for name in CANDIDATES}
    for number, start_ms in enumerate(fold_starts, start=1):
        end_ms = start_ms + fold_ms
        warmup_start_ms = start_ms - 45 * sim.MS_PER_DAY
        trend_slice = validation.slice_window(trend_candles, warmup_start_ms, end_ms)
        core = sim.simulate_timeseries_trend(
            trend_slice,
            replace(base_cfg, strategy_modes=("timeseries_trend",)),
            start_ms,
            funding,
        )
        for name, changes in CANDIDATES.items():
            cfg = candidate_config(base_cfg, changes)
            folds_by_candidate[name].append(
                run_fold(base_candles, trend_candles, funding, cfg, start_ms, end_ms, core)
            )
        print(f"Completed fold {number}/{len(fold_starts)}")

    rows: list[dict[str, Any]] = []
    for name, folds in folds_by_candidate.items():
        development = validation.aggregate_folds(folds[:dev_count], base_cfg.initial_equity)
        holdout = validation.aggregate_folds(folds[dev_count:], base_cfg.initial_equity)
        overall = validation.aggregate_folds(folds, base_cfg.initial_equity)
        row = {
            "name": name,
            "changes": CANDIDATES[name],
            "development": development,
            "holdout": holdout,
            "overall": overall,
            "development_pass": passes(development, DEV_GATES),
        }
        rows.append(row)

    eligible = [row for row in rows if row["development_pass"]]
    selected = max(eligible, key=lambda row: row["development"]["cagr_pct"], default=None)
    double_cost_holdout = None
    holdout_pass = False
    if selected is not None:
        changes = CANDIDATES[selected["name"]]
        selected_cfg = candidate_config(base_cfg, changes)
        stressed_cfg = replace(
            selected_cfg,
            maker_fee=selected_cfg.maker_fee * 2,
            taker_fee=selected_cfg.taker_fee * 2,
            entry_slippage_bps=selected_cfg.entry_slippage_bps * 2,
            exit_slippage_bps=selected_cfg.exit_slippage_bps * 2,
            depth_impact_bps=selected_cfg.depth_impact_bps * 2,
        )
        stressed_folds = [
            run_fold(
                base_candles,
                trend_candles,
                funding,
                stressed_cfg,
                start_ms,
                start_ms + fold_ms,
            )
            for start_ms in fold_starts[dev_count:]
        ]
        double_cost_holdout = validation.aggregate_folds(stressed_folds, base_cfg.initial_equity)
        holdout_pass = bool(
            holdout_passes(selected["holdout"], HOLDOUT_GATES)
            and holdout_passes(double_cost_holdout, HOLDOUT_GATES)
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "preregistered_signal_coverage_grid_with_last_four_fold_holdout",
        "independence_warning": (
            "The temporal holdout reduces selection leakage but is not genuinely independent because the historical "
            "market data existed before this run. Prospective paper observations remain required."
        ),
        "snapshot_metadata": metadata,
        "base_freeze_id": manifest["freeze_id"],
        "development_folds": dev_count,
        "holdout_folds": 4,
        "development_gates": DEV_GATES,
        "holdout_gates": HOLDOUT_GATES,
        "candidates": rows,
        "selected": selected["name"] if selected else None,
        "selected_double_cost_holdout": double_cost_holdout,
        "promotion_ready": bool(selected is not None and holdout_pass),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    sim.save_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}")
    return 0 if report["promotion_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
