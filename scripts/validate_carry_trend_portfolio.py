#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import backtest_funding_carry as carry
import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
import validate_candidate_portfolio as candidate


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(
        description="Validate a segregated funding-carry core and frozen trend satellite.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config/frozen_strategy_active_20260720.json",
    )
    parser.add_argument(
        "--macro-snapshot",
        type=Path,
        default=root / "data/snapshots/macro_20191001_20260628.json.gz",
    )
    parser.add_argument("--carry-report", type=Path, required=True)
    parser.add_argument("--carry-weight", type=float, default=0.85)
    parser.add_argument("--double-cost", action="store_true")
    parser.add_argument("--soft-drawdown-start-pct", type=float, default=8.0)
    parser.add_argument("--hard-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--drawdown-min-multiplier", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def combine_equity_curves(
    carry_curve: Sequence[dict[str, float]],
    trend_curve: Sequence[dict[str, float]],
    carry_weight: float,
    initial_equity: float = 100.0,
) -> list[dict[str, float]]:
    if not carry_curve or not trend_curve:
        raise ValueError("Both carry and trend equity curves are required")
    carry_times = [int(point["time_ms"]) for point in carry_curve]
    carry_initial = initial_equity
    trend_initial = initial_equity
    combined: list[dict[str, float]] = []
    for point in trend_curve:
        timestamp_ms = int(point["time_ms"])
        carry_index = bisect.bisect_right(carry_times, timestamp_ms) - 1
        carry_equity = (
            float(carry_curve[carry_index]["equity"])
            if carry_index >= 0
            else carry_initial
        )
        trend_equity = float(point["equity"])
        equity = initial_equity * (
            carry_weight * carry_equity / carry_initial
            + (1.0 - carry_weight) * trend_equity / trend_initial
        )
        combined.append({"time_ms": timestamp_ms, "equity": equity})
    return combined


def summarize(curve: Sequence[dict[str, float]], initial_equity: float = 100.0) -> dict[str, Any]:
    if not curve:
        raise ValueError("Combined equity curve is empty")
    peak = initial_equity
    max_drawdown = 0.0
    for point in curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
    start_ms = int(curve[0]["time_ms"])
    end_ms = int(curve[-1]["time_ms"])
    years = max((end_ms - start_ms) / (365.25 * sim.MS_PER_DAY), 1 / 365.25)
    final_equity = float(curve[-1]["equity"])
    multiple = final_equity / initial_equity
    annual = carry.annual_returns(curve, initial_equity)
    quarters = candidate.quarterly_returns(curve)
    return {
        "start_utc": sim.iso_utc_from_ms(start_ms),
        "end_utc": sim.iso_utc_from_ms(end_ms),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": (multiple - 1.0) * 100,
        "cagr_pct": (multiple ** (1 / years) - 1.0) * 100 if multiple > 0 else -100.0,
        "max_drawdown_pct": max_drawdown * 100,
        "annual_returns_pct": annual,
        "profitable_year_pct": sum(value > 0 for value in annual.values()) / len(annual) * 100,
        "quarter_count": len(quarters),
        "profitable_quarter_pct": (
            sum(value > 0 for value in quarters) / len(quarters) * 100 if quarters else 0.0
        ),
        "worst_quarter_pct": min(quarters) * 100 if quarters else None,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.carry_weight < 1:
        raise ValueError("--carry-weight must be between 0 and 1")
    carry_report = json.loads(args.carry_report.read_text(encoding="utf-8"))
    if carry_report.get("model") != "synchronized_basis_mark_to_market":
        raise ValueError("Carry report must use synchronized basis mark-to-market")
    expected_cost = "double" if args.double_cost else "normal"
    if carry_report.get("cost_scenario") != expected_cost:
        raise ValueError(
            f"Carry report cost scenario {carry_report.get('cost_scenario')} != {expected_cost}"
        )

    manifest, base_cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    market_path = sim.repo_root() / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, market_path)
    intervals, funding, _metadata = sim.load_market_snapshot(market_path)
    cfg = replace(
        base_cfg,
        strategy_modes=("trend", "timeseries_trend"),
        max_drawdown_stop_pct=args.hard_drawdown_stop_pct,
    )
    if args.double_cost:
        cfg = replace(
            cfg,
            maker_fee=cfg.maker_fee * 2,
            taker_fee=cfg.taker_fee * 2,
            entry_slippage_bps=cfg.entry_slippage_bps * 2,
            exit_slippage_bps=cfg.exit_slippage_bps * 2,
            depth_impact_bps=cfg.depth_impact_bps * 2,
        )
    start_ms = intervals["5m"][0].open_time_ms + 365 * sim.MS_PER_DAY
    sleeves = candidate.simulate_sleeves(
        intervals["5m"],
        intervals[cfg.timeseries_timeframe],
        funding,
        cfg,
        start_ms,
    )
    macro_snapshot = macro_regime.load_macro_snapshot(args.macro_snapshot)
    adjusted, macro_diagnostics = macro_regime.apply_macro_overlay(
        sleeves,
        macro_snapshot,
        enabled_factors=candidate.MACRO_FACTORS,
    )
    trend_result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        intervals["5m"],
        adjusted,
        cfg,
        portfolio_risk.DrawdownRiskPolicy(
            args.soft_drawdown_start_pct,
            args.hard_drawdown_stop_pct,
            args.drawdown_min_multiplier,
        ),
        start_ms,
    )
    combined_curve = combine_equity_curves(
        carry_report["equity_curve"],
        trend_result["equity_curve"],
        args.carry_weight,
    )
    summary = summarize(combined_curve)
    carry_summary = carry_report["summary"]
    gates = {
        "cagr_pct_min": 5.0,
        "max_drawdown_pct_max": 10.0,
        "profitable_year_pct_min": 80.0,
        "profitable_quarter_pct_min": 60.0,
        "carry_min_margin_buffer_pct_min": 10.0,
        "carry_liquidated": False,
    }
    passed = bool(
        summary["cagr_pct"] >= gates["cagr_pct_min"]
        and summary["max_drawdown_pct"] <= gates["max_drawdown_pct_max"]
        and summary["profitable_year_pct"] >= gates["profitable_year_pct_min"]
        and summary["profitable_quarter_pct"] >= gates["profitable_quarter_pct_min"]
        and carry_summary["min_futures_margin_buffer_pct"]
        >= gates["carry_min_margin_buffer_pct_min"]
        and bool(carry_summary["liquidated"]) is gates["carry_liquidated"]
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "places_orders": False,
        "cost_scenario": expected_cost,
        "allocation": {
            "funding_carry_core": args.carry_weight,
            "frozen_trend_satellite": 1.0 - args.carry_weight,
            "segregated_subaccounts_required": True,
        },
        "summary": summary,
        "carry_summary": carry_summary,
        "trend_summary": trend_result["summary"],
        "macro_diagnostics": macro_diagnostics,
        "gates": gates,
        "candidate_pass": passed,
        "inputs": {
            "manifest": str(args.manifest),
            "carry_report": str(args.carry_report),
            "macro_snapshot": str(args.macro_snapshot),
        },
    }
    print("Funding carry core + frozen trend satellite")
    print(f"Cost scenario: {expected_cost}")
    print(
        f"Allocation: carry {args.carry_weight:.0%} / "
        f"trend {1.0 - args.carry_weight:.0%}"
    )
    print(
        f"Return / CAGR: {summary['total_return_pct']:.2f}% / "
        f"{summary['cagr_pct']:.2f}%"
    )
    print(f"Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(
        f"Profitable years / quarters: {summary['profitable_year_pct']:.2f}% / "
        f"{summary['profitable_quarter_pct']:.2f}%"
    )
    print(f"Candidate pass: {passed}")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output_json}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
