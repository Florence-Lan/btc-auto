"""Fixed-plan, time-separated exit and factor evaluation; never promotes live trading."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import diagnose_strategy_losses as diagnosis
import frozen_strategy
import macro_regime
import multifactor
import research_exits
import simulate_range_swing as sim
from validate_macro_overlay import compact_summary


def slice_market(data, funding, end):
    return ({key: [bar for bar in bars if bar.close_time_ms < end] for key, bars in data.items()},
            sim.FundingHistory([t for t in funding.times if t < end],
                               [r for t, r in zip(funding.times, funding.rates) if t < end]))


def describe(result):
    curve, trades = result["equity_curve"], result["trades"]
    return {**compact_summary(result["summary"]),
            "average_gross_leverage": sum(p["gross_qty"] * p["price"] / max(p["equity"], 1e-9) for p in curve) / len(curve) if curve else 0,
            "closed_trades": sum(t["exit_reason"] != "end" for t in trades),
            "synthetic_end_trades": sum(t["exit_reason"] == "end" for t in trades),
            "trade_pnl": [{k: t[k] for k in ("strategy", "entry_time_utc", "exit_time_utc", "exit_reason", "net_pnl")} for t in trades],
            "accounting": diagnosis.accounting(trades),
            "risk_diagnostics": result["risk_diagnostics"]}


def match_constant_exposure(data, sleeves, cfg, start, target):
    """Calibration uses development exposure only, not profit or validation data."""
    low, high = 0.0, 1.0
    for _ in range(16):
        mid = (low + high) / 2
        result = diagnosis.combine(data, diagnosis.constant_scale(sleeves, mid), cfg, start)
        exposure = describe(result)["average_gross_leverage"]
        if exposure < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def select_policy(rows):
    return max(rows, key=lambda name: rows[name]["total_return_pct"] - rows[name]["max_drawdown_pct"])


def validation_gate(candidate, baseline, double_cost):
    checks = {"positive_net_return": candidate["total_return_pct"] > 0,
              "positive_double_cost_return": double_cost["total_return_pct"] > 0,
              "return_at_least_baseline": candidate["total_return_pct"] >= baseline["total_return_pct"],
              "drawdown_no_worse_than_baseline": candidate["max_drawdown_pct"] <= baseline["max_drawdown_pct"],
              "minimum_20_closed_trades": candidate["closed_trades"] >= 20}
    return {"checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("config/exit_research_plan_20260927.json"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/exit_research_20260927.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    _, cfg = frozen_strategy.load_frozen_strategy(Path(plan["base_manifest"]))
    cfg = replace(cfg, max_drawdown_stop_pct=0)
    early_path = Path("data/snapshots/btc_exit_validation_20251101_20260415.json.gz")
    recent_path = Path("data/snapshots/btc_diagnosis_20260927_90d.json.gz")
    early_data, early_funding, _ = sim.load_market_snapshot(early_path)
    recent_data, recent_funding, _ = sim.load_market_snapshot(recent_path)
    early_macro_path = Path("data/snapshots/macro_exit_validation_20251101_20260415.json.gz")
    recent_macro_path = Path("data/snapshots/macro_backtest_20260927_0900.json.gz")
    macros = [macro_regime.load_macro_snapshot(path) for path in (early_macro_path, recent_macro_path)]
    report = {"plan": plan, "plan_sha256": frozen_strategy.sha256_file(args.plan),
              "source_hashes": {str(path): frozen_strategy.sha256_file(path) for path in
                                (early_path, recent_path, early_macro_path, recent_macro_path,
                                 Path("scripts/research_exits.py"), Path("scripts/execution_ledger.py"),
                                 Path("scripts/portfolio_risk.py"))}, "windows": {}}
    constant_scale = None
    selected = None
    for window in ("development", "validation", "known_stress", "known_recent"):
        start, end = map(sim._utc_ms, plan[window])
        recent = window.startswith("known")
        data, funding = slice_market(recent_data if recent else early_data, recent_funding if recent else early_funding, end)
        tactical = sim.simulate(data["5m"], replace(cfg, strategy_modes=("trend",)), start, None, funding)
        hourly = {name: research_exits.simulate_with_exit_policy(data["1h"], cfg, start, funding,
                  research_exits.ExitPolicy(name, plan["fixed_parameters"]["stop_atr"], plan["fixed_parameters"]["trail_atr"]))
                  for name in plan["policies"]}
        # Catch accidental differences between research baseline and immutable engine.
        assert hourly["baseline"] == sim.simulate_timeseries_trend(data["1h"], cfg, start, funding)
        row = {"hourly_only": {}, "portfolio": {}, "factors": {}}
        for name, core in hourly.items():
            row["hourly_only"][name] = describe(diagnosis.combine(data, [core], cfg, start))
            row["portfolio"][name] = describe(diagnosis.combine(data, [tactical, core], cfg, start))
        if window == "development":
            selected = select_policy(row["hourly_only"])
            report["development_selected_policy"] = selected
        # Factors are tested on baseline trades; do not confound them with exit selection.
        sleeves = [tactical, hourly["baseline"]]
        macro, macro_diagnostics = macro_regime.apply_macro_overlay(sleeves, macros[int(recent)],
            enabled_factors=("vix", "dollar", "metals", "sentiment"))
        row["factors"]["legacy_macro"] = describe(diagnosis.combine(data, macro, cfg, start))
        row["legacy_factor_coverage"] = macro_diagnostics
        if window == "development":
            constant_scale = match_constant_exposure(data, sleeves, cfg, start,
                               row["factors"]["legacy_macro"]["average_gross_leverage"])
            report["development_exposure_matched_constant"] = constant_scale
        for name, multiplier in (("constant_0675", .675), ("development_exposure_matched", constant_scale)):
            row["factors"][name] = describe(diagnosis.combine(data, diagnosis.constant_scale(sleeves, multiplier), cfg, start))
        if window == "known_recent":
            factor_path = Path("data/snapshots/multifactor_backtest_20260927_0900.json.gz")
            report["source_hashes"][str(factor_path)] = frozen_strategy.sha256_file(factor_path)
            snap = multifactor.load_snapshot(factor_path, "reconstructed")
            profile = multifactor.load_profile(Path("config/multifactor_candidate_20260926.json"))
            for excluded in (None, *profile["groups"]):
                modified = {**profile, "groups": {k: v for k, v in profile["groups"].items() if k != excluded}}
                adjusted, diagnostics = multifactor.apply_overlay(sleeves, snap, modified)
                name = f"without_{excluded}" if excluded else "six_factors"
                row["factors"][name] = describe(diagnosis.combine(data, adjusted, cfg, start))
                row["factors"][name]["factor_diagnostics"] = diagnostics
            # Descriptive same-window exposure control only, not deployable calibration.
            scale = match_constant_exposure(data, sleeves, cfg, start, row["factors"]["six_factors"]["average_gross_leverage"])
            row["six_factor_same_window_control_scale"] = scale
            row["factors"]["six_factor_same_window_control"] = describe(diagnosis.combine(data, diagnosis.constant_scale(sleeves, scale), cfg, start))
        if window == "validation":
            doubled = replace(cfg, maker_fee=2*cfg.maker_fee, taker_fee=2*cfg.taker_fee,
                              entry_slippage_bps=2*cfg.entry_slippage_bps,
                              exit_slippage_bps=2*cfg.exit_slippage_bps, depth_impact_bps=2*cfg.depth_impact_bps)
            row["double_cost_hourly"] = {}
            for name in dict.fromkeys(("baseline", selected)):
                sleeve = research_exits.simulate_with_exit_policy(data["1h"], doubled, start, funding,
                            research_exits.ExitPolicy(name, plan["fixed_parameters"]["stop_atr"], plan["fixed_parameters"]["trail_atr"]))
                row["double_cost_hourly"][name] = describe(diagnosis.combine(data, [sleeve], doubled, start))
            report["validation_gate"] = validation_gate(row["hourly_only"][selected], row["hourly_only"]["baseline"], row["double_cost_hourly"][selected])
        report["windows"][window] = row
        print(window, {k: (round(v["total_return_pct"], 4), round(v["max_drawdown_pct"], 4), v["closed_trades"])
                       for k, v in row["hourly_only"].items()}, flush=True)
    report["decision"] = "research_only_no_production_change"
    sim.save_json(args.output, report)
    print("selected", selected, "gate", report["validation_gate"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
