#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
import validate_frozen_strategy
import validate_strategies


MACRO_FACTORS = ("vix", "dollar", "metals", "sentiment")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "calmar",
        "trades",
        "trades_per_year",
        "win_rate_pct",
        "profit_factor",
        "max_consecutive_losses",
        "total_costs",
        "cost_to_gross_profit_pct",
        "largest_profit_share_pct",
        "annual_returns_pct",
        "by_strategy",
        "by_side",
    )
    return {key: summary.get(key) for key in keys}


def quarterly_returns(equity_curve: Sequence[dict[str, float]], days: int = 90) -> list[float]:
    if not equity_curve:
        return []
    period_ms = days * sim.MS_PER_DAY
    start_ms = int(equity_curve[0]["time_ms"])
    end_ms = int(equity_curve[-1]["time_ms"])
    values: list[float] = []
    cursor = start_ms
    previous_equity = float(equity_curve[0]["equity"])
    index = 0
    while cursor + period_ms <= end_ms:
        boundary = cursor + period_ms
        while index + 1 < len(equity_curve) and int(equity_curve[index + 1]["time_ms"]) <= boundary:
            index += 1
        end_equity = float(equity_curve[index]["equity"])
        if previous_equity > 0:
            values.append(end_equity / previous_equity - 1.0)
        previous_equity = end_equity
        cursor = boundary
    return values


def simulate_sleeves(
    base_candles: Sequence[sim.Candle],
    trend_candles: Sequence[sim.Candle],
    funding: sim.FundingHistory,
    cfg: sim.StrategyConfig,
    start_ms: int,
) -> list[dict[str, Any]]:
    tactical = sim.simulate(
        base_candles,
        replace(cfg, strategy_modes=("trend",), max_drawdown_stop_pct=0.0),
        start_ms,
        None,
        funding,
    )
    core = sim.simulate_timeseries_trend(
        trend_candles,
        replace(cfg, strategy_modes=("timeseries_trend",), max_drawdown_stop_pct=0.0),
        start_ms,
        funding,
    )
    return [tactical, core]


def evaluate(
    base_candles: Sequence[sim.Candle],
    sleeves: Sequence[dict[str, Any]],
    cfg: sim.StrategyConfig,
    start_ms: int,
    macro_snapshot: macro_regime.MacroSnapshot,
    policy: portfolio_risk.DrawdownRiskPolicy,
) -> dict[str, Any]:
    adjusted, macro_diagnostics = macro_regime.apply_macro_overlay(
        sleeves,
        macro_snapshot,
        enabled_factors=MACRO_FACTORS,
    )
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        base_candles,
        adjusted,
        cfg,
        policy,
        start_ms,
    )
    quarters = quarterly_returns(result["equity_curve"])
    daily_returns = validate_strategies.daily_returns([result["equity_curve"]])
    bootstrap = validate_frozen_strategy.bootstrap_distribution(
        daily_returns,
        block_days=7,
        samples=2000,
        seed=20260711,
    )
    return {
        "summary": compact(result["summary"]),
        "macro": macro_diagnostics,
        "drawdown": result["risk_diagnostics"],
        "quarterly": {
            "count": len(quarters),
            "profitable_pct": (
                sum(value > 0 for value in quarters) / len(quarters) * 100 if quarters else 0.0
            ),
            "worst_pct": min(quarters) * 100 if quarters else None,
            "best_pct": max(quarters) * 100 if quarters else None,
            "returns_pct": [value * 100 for value in quarters],
        },
        "bootstrap": bootstrap,
    }


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(description="Validate the macro, no-range BTC candidate continuously.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config/frozen_strategy_active_20260711.json",
    )
    parser.add_argument(
        "--macro-snapshot",
        type=Path,
        default=root / "data/snapshots/macro_20191001_20260628.json.gz",
    )
    parser.add_argument("--soft-drawdown-start-pct", type=float, default=8.0)
    parser.add_argument("--hard-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--drawdown-min-multiplier", type=float, default=0.35)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "data/validation/candidate_portfolio_20260711.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, base_cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    market_path = sim.repo_root() / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, market_path)
    intervals, funding, market_metadata = sim.load_market_snapshot(market_path)
    macro_snapshot = macro_regime.load_macro_snapshot(args.macro_snapshot)
    base_candles = intervals["5m"]
    trend_candles = intervals[base_cfg.timeseries_timeframe]
    start_ms = base_candles[0].open_time_ms + 365 * sim.MS_PER_DAY
    cfg = replace(
        base_cfg,
        strategy_modes=("trend", "timeseries_trend"),
        max_drawdown_stop_pct=args.hard_drawdown_stop_pct,
    )
    policy = portfolio_risk.DrawdownRiskPolicy(
        args.soft_drawdown_start_pct,
        args.hard_drawdown_stop_pct,
        args.drawdown_min_multiplier,
    )

    normal_sleeves = simulate_sleeves(base_candles, trend_candles, funding, cfg, start_ms)
    normal = evaluate(base_candles, normal_sleeves, cfg, start_ms, macro_snapshot, policy)
    stressed_cfg = replace(
        cfg,
        maker_fee=cfg.maker_fee * 2,
        taker_fee=cfg.taker_fee * 2,
        entry_slippage_bps=cfg.entry_slippage_bps * 2,
        exit_slippage_bps=cfg.exit_slippage_bps * 2,
        depth_impact_bps=cfg.depth_impact_bps * 2,
    )
    stressed_sleeves = simulate_sleeves(
        base_candles,
        trend_candles,
        funding,
        stressed_cfg,
        start_ms,
    )
    stressed = evaluate(
        base_candles,
        stressed_sleeves,
        stressed_cfg,
        start_ms,
        macro_snapshot,
        policy,
    )

    gates = {
        "cagr_pct_min": 5.0,
        "max_drawdown_pct_max": 15.0,
        "profit_factor_min": 1.3,
        "trades_per_year_min": 20.0,
        "profitable_quarter_pct_min": 60.0,
        "bootstrap_annualized_p05_pct_min": 0.0,
        "double_cost_cagr_pct_min": 0.0,
        "double_cost_max_drawdown_pct_max": 15.0,
        "double_cost_profit_factor_min": 1.2,
    }
    summary = normal["summary"]
    stress_summary = stressed["summary"]
    risk_evidence_pass = bool(
        summary["cagr_pct"] >= gates["cagr_pct_min"]
        and summary["max_drawdown_pct"] <= gates["max_drawdown_pct_max"]
        and summary["profit_factor"] is not None
        and summary["profit_factor"] >= gates["profit_factor_min"]
        and summary["trades_per_year"] >= gates["trades_per_year_min"]
        and normal["bootstrap"]["annualized_p05_pct"] is not None
        and normal["bootstrap"]["annualized_p05_pct"] > gates["bootstrap_annualized_p05_pct_min"]
        and stress_summary["cagr_pct"] > gates["double_cost_cagr_pct_min"]
        and stress_summary["max_drawdown_pct"] <= gates["double_cost_max_drawdown_pct_max"]
        and stress_summary["profit_factor"] is not None
        and stress_summary["profit_factor"] >= gates["double_cost_profit_factor_min"]
    )
    candidate_pass = bool(
        risk_evidence_pass
        and normal["quarterly"]["profitable_pct"] >= gates["profitable_quarter_pct_min"]
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "continuous_no_reset_macro_no_range_double_cost",
        "promotion_status": "promotion_candidate" if candidate_pass else (
            "shadow_candidate" if risk_evidence_pass else "research_only"
        ),
        "base_freeze_id": manifest["freeze_id"],
        "market_snapshot": str(market_path),
        "market_snapshot_sha256": manifest["snapshot_sha256"],
        "market_snapshot_metadata": market_metadata,
        "macro_snapshot": str(args.macro_snapshot),
        "macro_snapshot_sha256": sha256_file(args.macro_snapshot),
        "macro_snapshot_metadata": macro_snapshot.metadata,
        "candidate_config": {
            "strategy_modes": ["trend", "timeseries_trend"],
            "risk_per_trade": cfg.risk_per_trade,
            "portfolio_leverage_cap": cfg.portfolio_leverage_cap,
            "macro_factors": list(MACRO_FACTORS),
            "macro_min_multiplier": 0.35,
            "macro_block_score": -0.80,
            "drawdown_policy": {
                "soft_start_pct": policy.soft_start_pct,
                "hard_stop_pct": policy.hard_stop_pct,
                "min_multiplier": policy.min_multiplier,
            },
        },
        "gates": gates,
        "normal": normal,
        "double_cost": stressed,
        "candidate_pass": candidate_pass,
        "shadow_eligible": risk_evidence_pass,
        "independence_warning": (
            "The candidate was designed on visible historical data. Passing permits shadow "
            "tracking only; prospective evidence is required before any live promotion."
        ),
    }
    sim.save_json(args.output_json, report)
    print(json.dumps({
        "normal": summary,
        "normal_quarterly": normal["quarterly"],
        "normal_bootstrap": normal["bootstrap"],
        "double_cost": stress_summary,
        "candidate_pass": candidate_pass,
        "shadow_eligible": risk_evidence_pass,
        "output": str(args.output_json),
    }, indent=2))
    return 0 if candidate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
