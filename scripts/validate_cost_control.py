"""Reduce reentry churn on 2024, freeze once, then review known historical windows."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import frozen_strategy
import simulate_range_swing as sim
import validate_reentry_research as research

PLAN = Path("config/cost_control_plan_20260928.json")
SELECTED = Path("config/cost_control_selected_20260928.json")
OUTPUT = Path("data/validation/cost_control_20260928")
SNAPSHOTS = {
    "development": research.HISTORY,
    "review_2025": research.HISTORY,
    "review_2026": Path("data/snapshots/btc_exit_validation_20251101_20260415.json.gz"),
    "recent": Path("data/snapshots/btc_diagnosis_20260927_90d.json.gz"),
}


def input_hashes(plan):
    paths = [PLAN, Path(plan["base_manifest"]), Path(plan["reference_profile"])]
    paths += list(dict.fromkeys(SNAPSHOTS.values()))
    paths += [Path("scripts") / name for name in (
        "research_reentry.py", "execution_ledger.py", "portfolio_risk.py",
        "simulate_range_swing.py", "validate_reentry_research.py",
        "validate_cost_control.py", "reentry_candidate.py",
        "diagnose_strategy_losses.py", "validate_exit_research.py")]
    return {str(path): frozen_strategy.sha256_file(path) for path in paths}


def costs(row):
    return row["accounting"]["fees"] + row["accounting"]["slippage_cost"]


def checks(row, reference, stress=None):
    result = {
        "execution_cost_reduction_at_least_25pct": costs(row) <= .75 * costs(reference),
        "net_return_at_least_reference": row["total_return_pct"] >= reference["total_return_pct"],
        "drawdown_within_one_percentage_point": row["max_drawdown_pct"] <= reference["max_drawdown_pct"] + 1,
    }
    if stress is not None:
        result["positive_double_cost_return"] = stress["total_return_pct"] > 0
    return result


def select_variant(rows, reference):
    eligible = [name for name, row in rows.items() if all(checks(row, reference).values())]
    return max(eligible, key=lambda name: rows[name]["total_return_pct"] - rows[name]["max_drawdown_pct"]) if eligible else None


def evaluate(data, funding, cfg, start, policy):
    row, _ = research.evaluate(data, funding, cfg, start, policy, 1.0)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("develop", "review", "diagnose"), required=True)
    parser.add_argument("--variant", help="Explicit exploratory variant; only with --phase diagnose")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (args.phase == "diagnose") != bool(args.variant):
        parser.error("--phase diagnose requires --variant; other phases must omit it")
    if args.variant and args.variant not in plan["policies"]:
        parser.error("Unknown diagnostic variant")
    reference = json.loads(Path(plan["reference_profile"]).read_text())
    _, cfg = frozen_strategy.load_frozen_strategy(Path(plan["base_manifest"]))
    cfg = replace(cfg, max_drawdown_stop_pct=0)
    hashes = input_hashes(plan)
    selected = None
    if args.phase == "review":
        selected = json.loads(SELECTED.read_text())
        if selected["input_hashes"] != hashes:
            raise ValueError("Frozen selection inputs changed; refusing review")
    elif args.phase == "diagnose":
        selected = {"selected_variant": args.variant, "policy": plan["policies"][args.variant]}
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_hashes": hashes, "plan": plan, "phase": args.phase,
              "research_only": True, "places_orders": False, "windows": {}}
    if args.phase == "diagnose":
        report["diagnostic_variant"] = args.variant
        report["selection_warning"] = "Exploratory follow-up to failed development gates; not an accepted or independently selected candidate."
    windows = ("development",) if args.phase == "develop" else ("review_2025", "review_2026", "recent")
    for window in windows:
        data, funding, _ = sim.load_market_snapshot(SNAPSHOTS[window])
        start, end = map(sim._utc_ms, plan[window])
        data, funding = research.slice_market(data, funding, end)
        ref = evaluate(data, funding, cfg, start, reference["policy"])
        policies = plan["policies"] if args.phase == "develop" else {selected["selected_variant"]: selected["policy"]}
        rows = {name: evaluate(data, funding, cfg, start, policy) for name, policy in policies.items()}
        record = {"window": plan[window], "reference": ref, "variants": rows}
        for name, row in {"reference": ref, **rows}.items():
            print(window, name, json.dumps({"return": row["total_return_pct"],
                  "drawdown": row["max_drawdown_pct"], "costs": costs(row),
                  "closed_trades": row["closed_trades"], "reentries": row["reentry_count"]}), flush=True)
        if args.phase == "develop":
            name = select_variant(rows, ref)
            report["selected_variant"] = name
            if name is not None:
                profile = {"candidate_id": "btc_cost_control_selected_20260928",
                           "research_only": True, "live_orders_allowed": False,
                           "status": "development_selected_pending_review",
                           "research_plan": str(PLAN), "base_manifest": plan["base_manifest"],
                           "selected_variant": name + "_tactical_1", "policy": policies[name],
                           "tactical_weight": 1.0, "input_hashes": hashes,
                           "selected_at_utc": datetime.now(timezone.utc).isoformat()}
                if SELECTED.exists():
                    existing = json.loads(SELECTED.read_text())
                    if any(existing[k] != profile[k] for k in ("input_hashes", "selected_variant", "policy")):
                        raise ValueError("Existing selection differs; use a new research ID")
                else:
                    sim.save_json(SELECTED, profile)
            print("SELECTION", name, flush=True)
        else:
            doubled = replace(cfg, **{key: getattr(cfg, key)*2 for key in (
                "maker_fee", "taker_fee", "entry_slippage_bps", "exit_slippage_bps", "depth_impact_bps")})
            stressed = evaluate(data, funding, doubled, start, selected["policy"])
            row = next(iter(rows.values()))
            record["double_cost_selected"] = stressed
            record["checks"] = checks(row, ref, stressed)
            record["cost_reduction_pct"] = (1-costs(row)/costs(ref))*100 if costs(ref) else None
            print(window, "review", record["checks"], "double_cost_return", stressed["total_return_pct"], flush=True)
        report["windows"][window] = record
    if args.phase != "develop":
        report["review_pass"] = all(all(row["checks"].values()) for row in report["windows"].values())
        report["promotion_pass"] = False
    suffix = f"diagnose_{args.variant}" if args.phase == "diagnose" else args.phase
    sim.save_json(Path(f"{OUTPUT}_{suffix}.json"), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
