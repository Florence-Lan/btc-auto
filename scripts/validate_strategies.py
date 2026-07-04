#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import simulate_range_swing as sim


TARGET_VOLS = (0.08, 0.10, 0.12)
TACTICAL_RISKS = (0.015, 0.020, 0.025)
LEVERAGE_CAPS = (1.5, 2.0)


def build_base_config(initial_equity: float) -> sim.StrategyConfig:
    original_argv = sys.argv
    try:
        sys.argv = [
            "simulate_range_swing.py",
            "--strategy-modes",
            "trend,timeseries_trend",
            "--portfolio-mode",
            "sleeves",
            "--initial-equity",
            str(initial_equity),
            "--max-drawdown-stop-pct",
            "0",
            "--no-market-context-enabled",
        ]
        return sim.config_from_args(sim.parse_args())
    finally:
        sys.argv = original_argv


def slice_until(candles: list[sim.Candle], end_ms: int) -> list[sim.Candle]:
    return [candle for candle in candles if candle.open_time_ms <= end_ms]


def slice_window(candles: list[sim.Candle], start_ms: int, end_ms: int) -> list[sim.Candle]:
    return [
        candle
        for candle in candles
        if start_ms <= candle.open_time_ms <= end_ms
    ]


def daily_returns(equity_curves: Iterable[list[dict[str, float]]]) -> list[float]:
    values: list[float] = []
    for curve in equity_curves:
        by_day: dict[str, float] = {}
        for point in curve:
            day = datetime.fromtimestamp(
                float(point["time_ms"]) / 1000,
                tz=timezone.utc,
            ).date().isoformat()
            by_day[day] = float(point["equity"])
        daily = list(by_day.values())
        values.extend(
            daily[index] / daily[index - 1] - 1
            for index in range(1, len(daily))
            if daily[index - 1] > 0
        )
    return values


def compact_fold(result: dict[str, Any], start_ms: int, end_ms: int) -> dict[str, Any]:
    return {
        "window": {
            "start_utc": sim.iso_utc_from_ms(start_ms),
            "end_utc": sim.iso_utc_from_ms(end_ms),
        },
        "summary": result["summary"],
        "trades": result["trades"],
        "daily_returns": daily_returns([result["equity_curve"]]),
        "profitable": result["summary"]["total_return_pct"] > 0,
    }


def bootstrap_fifth_percentile(
    returns: list[float],
    block_days: int = 7,
    samples: int = 1000,
    seed: int = 20260628,
) -> float | None:
    if len(returns) < block_days:
        return None
    blocks = [
        returns[index : index + block_days]
        for index in range(0, len(returns) - block_days + 1, block_days)
    ]
    rng = random.Random(seed)
    annualized: list[float] = []
    target_length = len(returns)
    for _ in range(samples):
        sample: list[float] = []
        while len(sample) < target_length:
            sample.extend(rng.choice(blocks))
        sample = sample[:target_length]
        growth = math.prod(1 + value for value in sample)
        years = max(len(sample) / 365, 1 / 365)
        annualized.append((growth ** (1 / years) - 1) * 100 if growth > 0 else -100.0)
    annualized.sort()
    return annualized[int(0.05 * (len(annualized) - 1))]


def aggregate_folds(folds: list[dict[str, Any]], initial_equity: float) -> dict[str, Any]:
    growth = math.prod(1 + fold["summary"]["total_return_pct"] / 100 for fold in folds)
    years = len(folds) * 90 / 365
    cagr = (growth ** (1 / years) - 1) * 100 if growth > 0 and years > 0 else -100.0
    gross_profit = sum(fold["summary"]["gross_profit"] for fold in folds)
    gross_loss = sum(fold["summary"]["gross_loss"] for fold in folds)
    trades = sum(int(fold["summary"]["trades"]) for fold in folds)
    all_trades = [trade for fold in folds for trade in fold["trades"]]
    total_positive = sum(max(float(trade["net_pnl"]), 0.0) for trade in all_trades)
    largest_share = (
        max((float(trade["net_pnl"]) for trade in all_trades), default=0.0)
        / total_positive
        * 100
        if total_positive > 0
        else None
    )
    return {
        "cagr_pct": cagr,
        "total_return_pct": (growth - 1) * 100,
        "max_drawdown_pct": max((fold["summary"]["max_drawdown_pct"] for fold in folds), default=0.0),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "trades": trades,
        "trades_per_year": trades / years if years > 0 else 0.0,
        "profitable_fold_pct": (
            sum(fold["summary"]["total_return_pct"] > 0 for fold in folds) / len(folds) * 100
            if folds
            else 0.0
        ),
        "turnover_notional": sum(fold["summary"]["turnover_notional"] for fold in folds),
        "largest_profit_share_pct": largest_share,
        "initial_equity": initial_equity,
    }


def passes_base_gates(summary: dict[str, Any]) -> bool:
    profit_factor = summary.get("profit_factor")
    largest_share = summary.get("largest_profit_share_pct")
    return bool(
        summary["cagr_pct"] >= 30.0
        and summary["max_drawdown_pct"] <= 12.0
        and profit_factor is not None
        and profit_factor >= 1.5
        and summary["trades_per_year"] >= 30.0
        and summary["profitable_fold_pct"] >= 70.0
        and largest_share is not None
        and largest_share <= 20.0
    )


def run_fold(
    base_candles: list[sim.Candle],
    trend_candles: list[sim.Candle],
    funding: sim.FundingHistory,
    cfg: sim.StrategyConfig,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    warmup_start_ms = start_ms - 45 * sim.MS_PER_DAY
    base_slice = slice_window(base_candles, warmup_start_ms, end_ms)
    trend_slice = slice_window(trend_candles, warmup_start_ms, end_ms)
    tactical = sim.simulate(base_slice, replace(cfg, strategy_modes=("trend",)), start_ms, None, funding)
    core = sim.simulate_timeseries_trend(trend_slice, cfg, start_ms, funding)
    result = sim.combine_sleeve_results(base_slice, [tactical, core], cfg, start_ms)
    result["window"] = {
        "start_utc": sim.iso_utc_from_ms(start_ms),
        "end_utc": sim.iso_utc_from_ms(end_ms),
    }
    return result


def evaluate_candidate(
    base_candles: list[sim.Candle],
    trend_candles: list[sim.Candle],
    funding: sim.FundingHistory,
    cfg: sim.StrategyConfig,
    fold_starts: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = [
        run_fold(
            base_candles,
            trend_candles,
            funding,
            cfg,
            start_ms,
            start_ms + 90 * sim.MS_PER_DAY,
        )
        for start_ms in fold_starts
    ]
    return aggregate_folds(folds, cfg.initial_equity), folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered BTC strategy walk-forward validation.")
    parser.add_argument("--data-snapshot", type=Path, required=True)
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-folds", type=int, help="Use only the newest N folds for a quicker diagnostic run.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=sim.repo_root() / "data/validation/strategy_validation.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=sim.repo_root() / "data/validation/strategy_validation.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intervals, funding, metadata = sim.load_market_snapshot(args.data_snapshot)
    if "5m" not in intervals or "6h" not in intervals:
        raise ValueError("Validation snapshot must contain 5m and 6h candles")
    base_candles = intervals["5m"]
    trend_candles = intervals["6h"]
    first_test_ms = base_candles[0].open_time_ms + 365 * sim.MS_PER_DAY
    last_end_ms = min(base_candles[-1].close_time_ms, trend_candles[-1].close_time_ms)
    fold_starts: list[int] = []
    cursor = first_test_ms
    while cursor + 90 * sim.MS_PER_DAY <= last_end_ms:
        fold_starts.append(cursor)
        cursor += 90 * sim.MS_PER_DAY
    if args.max_folds:
        fold_starts = fold_starts[-args.max_folds :]
    if not fold_starts:
        raise ValueError("Snapshot needs at least 15 months for walk-forward validation")

    base_cfg = build_base_config(args.initial_equity)
    rows: list[dict[str, Any]] = []
    fold_cache: dict[tuple[float, float, float], list[dict[str, Any]]] = {
        key: []
        for key in (
            (target_vol, tactical_risk, leverage_cap)
            for target_vol in TARGET_VOLS
            for tactical_risk in TACTICAL_RISKS
            for leverage_cap in LEVERAGE_CAPS
        )
    }
    for fold_number, start_ms in enumerate(fold_starts, start=1):
        end_ms = start_ms + 90 * sim.MS_PER_DAY
        warmup_start_ms = start_ms - 45 * sim.MS_PER_DAY
        base_slice = slice_window(base_candles, warmup_start_ms, end_ms)
        trend_slice = slice_window(trend_candles, warmup_start_ms, end_ms)
        tactical_cache: dict[float, dict[str, Any]] = {}
        core_cache: dict[float, dict[str, Any]] = {}
        for tactical_risk in TACTICAL_RISKS:
            tactical_cfg = replace(
                base_cfg,
                risk_per_trade=tactical_risk,
                strategy_modes=("trend",),
            )
            tactical_cache[tactical_risk] = sim.simulate(
                base_slice,
                tactical_cfg,
                start_ms,
                None,
                funding,
            )
        for target_vol in TARGET_VOLS:
            core_cfg = replace(base_cfg, timeseries_target_vol=target_vol)
            core_cache[target_vol] = sim.simulate_timeseries_trend(
                trend_slice,
                core_cfg,
                start_ms,
                funding,
            )
        for target_vol in TARGET_VOLS:
            for tactical_risk in TACTICAL_RISKS:
                for leverage_cap in LEVERAGE_CAPS:
                    cfg = replace(
                        base_cfg,
                        timeseries_target_vol=target_vol,
                        risk_per_trade=tactical_risk,
                        portfolio_leverage_cap=leverage_cap,
                    )
                    combined = sim.combine_sleeve_results(
                        base_slice,
                        [tactical_cache[tactical_risk], core_cache[target_vol]],
                        cfg,
                        start_ms,
                    )
                    fold_cache[(target_vol, tactical_risk, leverage_cap)].append(
                        compact_fold(combined, start_ms, end_ms)
                    )
        print(f"Completed fold {fold_number}/{len(fold_starts)}")

    for key, folds in fold_cache.items():
        target_vol, tactical_risk, leverage_cap = key
        summary = aggregate_folds(folds, base_cfg.initial_equity)
        row = {
            "target_vol": target_vol,
            "tactical_risk": tactical_risk,
            "leverage_cap": leverage_cap,
            **summary,
        }
        row["base_pass"] = passes_base_gates(row)
        rows.append(row)
        print(
            f"vol={target_vol:.3f} risk={tactical_risk:.3f} cap={leverage_cap:.1f} "
            f"CAGR={row['cagr_pct']:.2f}% DD={row['max_drawdown_pct']:.2f}% "
            f"PF={row['profit_factor']} trades/y={row['trades_per_year']:.1f} pass={row['base_pass']}"
        )

    passing = [row for row in rows if row["base_pass"]]
    passing.sort(key=lambda row: (row["max_drawdown_pct"], row["turnover_notional"]))
    selected = passing[0] if passing else None
    diagnostic_candidate = min(
        rows,
        key=lambda row: (
            max(0.0, row["max_drawdown_pct"] - 12.0),
            max(0.0, 30.0 - row["cagr_pct"]),
            row["turnover_notional"],
        ),
    )
    stress_candidate = selected or diagnostic_candidate
    positive_ratio = sum(row["total_return_pct"] > 0 for row in rows) / len(rows)
    stability_pass = positive_ratio >= 0.70
    bootstrap_p05 = None
    cost_stress = None
    final_pass = False

    if stress_candidate is not None:
        key = (
            stress_candidate["target_vol"],
            stress_candidate["tactical_risk"],
            stress_candidate["leverage_cap"],
        )
        bootstrap_p05 = bootstrap_fifth_percentile(
            [
                value
                for fold in fold_cache[key]
                for value in fold["daily_returns"]
            ],
            samples=args.bootstrap_samples,
        )
        stressed_cfg = replace(
            base_cfg,
            timeseries_target_vol=stress_candidate["target_vol"],
            risk_per_trade=stress_candidate["tactical_risk"],
            portfolio_leverage_cap=stress_candidate["leverage_cap"],
            maker_fee=base_cfg.maker_fee * 2,
            taker_fee=base_cfg.taker_fee * 2,
            entry_slippage_bps=base_cfg.entry_slippage_bps * 2,
            exit_slippage_bps=base_cfg.exit_slippage_bps * 2,
            depth_impact_bps=base_cfg.depth_impact_bps * 2,
        )
        cost_stress, _ = evaluate_candidate(
            base_candles,
            trend_candles,
            funding,
            stressed_cfg,
            fold_starts,
        )
        stress_pf = cost_stress.get("profit_factor")
        stress_pass = (
            cost_stress["cagr_pct"] > 0
            and stress_pf is not None
            and stress_pf >= 1.2
            and cost_stress["max_drawdown_pct"] <= 15.0
        )
        final_pass = bool(
            selected is not None
            and stability_pass
            and bootstrap_p05 is not None
            and bootstrap_p05 > 0
            and stress_pass
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(args.data_snapshot),
        "snapshot_metadata": metadata,
        "folds": len(fold_starts),
        "preregistered_grid": {
            "target_vols": TARGET_VOLS,
            "tactical_risks": TACTICAL_RISKS,
            "leverage_caps": LEVERAGE_CAPS,
        },
        "gates": {
            "cagr_min_pct": 30.0,
            "max_drawdown_pct": 12.0,
            "profit_factor_min": 1.5,
            "trades_per_year_min": 30.0,
            "profitable_fold_pct_min": 70.0,
            "largest_profit_share_pct_max": 20.0,
            "bootstrap_p05_min_pct": 0.0,
        },
        "candidates": rows,
        "walk_forward": [
            {
                "target_vol": key[0],
                "tactical_risk": key[1],
                "leverage_cap": key[2],
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
            for key, folds in fold_cache.items()
        ],
        "positive_candidate_ratio": positive_ratio,
        "stability_pass": stability_pass,
        "selected": selected,
        "diagnostic_candidate": diagnostic_candidate,
        "bootstrap_annualized_p05_pct": bootstrap_p05,
        "double_cost_stress": cost_stress,
        "historical_acceptance_pass": final_pass,
        "promotion": (
            "shadow_only_90d_min_8_trades"
            if final_pass
            else "rejected_keep_current_default"
        ),
    }
    sim.save_json(args.output_json, report)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Historical acceptance: {final_pass}")
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")
    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
