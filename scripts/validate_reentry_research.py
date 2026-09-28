"""Select on 2024, persist selection, then evaluate 2025 without reselection."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import diagnose_strategy_losses as diagnosis
import frozen_strategy
import research_reentry as research
import simulate_range_swing as sim
from validate_exit_research import describe, slice_market

PLAN = Path("config/reentry_research_plan_20260928.json")
SELECTED = Path("config/reentry_selected_20260928.json")
HISTORY = Path("data/snapshots/btc_reentry_20231101_20260101.json.gz")
OUTPUT = Path("data/validation/reentry_20260928")


def hashes():
    paths = [PLAN, HISTORY, Path("config/frozen_strategy_candidate_20260917.json")]
    paths += [Path("scripts") / filename for filename in
              ("research_reentry.py", "execution_ledger.py", "portfolio_risk.py", "simulate_range_swing.py",
               "validate_reentry_research.py")]
    return {str(path): frozen_strategy.sha256_file(path) for path in paths}


def quarter_returns(curve, initial):
    endings = {}
    for point in curve:
        date = datetime.fromtimestamp(point["time_ms"] / 1000, timezone.utc)
        endings[f"{date.year}-Q{(date.month-1)//3+1}"] = point["equity"]
    previous, result = initial, {}
    for quarter, equity in endings.items():
        result[quarter] = (equity / previous - 1) * 100
        previous = equity
    return result


def make_sleeves(data, funding, cfg, start, policy, tactical_weight):
    sleeves = []
    if tactical_weight > 0:
        # Resimulate actual risk/leverage limits and costs, not retrospective PnL scaling.
        tactical_cfg = replace(cfg, strategy_modes=("trend",), leverage=cfg.leverage*tactical_weight,
                               risk_per_trade=cfg.risk_per_trade*tactical_weight)
        sleeves.append(sim.simulate(data["5m"], tactical_cfg, start, None, funding))
    core = research.simulate_with_exit_policy(data["1h"], cfg, start, funding, research.ExitPolicy(**policy))
    sleeves.append(core)
    return sleeves


def evaluate(data, funding, cfg, start, policy, tactical_weight):
    sleeves = make_sleeves(data, funding, cfg, start, policy, tactical_weight)
    result = diagnosis.combine(data, sleeves, cfg, start)
    row = describe(result)
    row["quarter_returns_pct"] = quarter_returns(result["equity_curve"], cfg.initial_equity)
    row["reentry_count"] = len(sleeves[-1].get("reentry_times", []))
    row["policy"] = policy
    row["tactical_weight"] = tactical_weight
    assert result["risk_diagnostics"]["legacy_endpoint_trades"] == 0
    assert result["risk_diagnostics"]["unsettled_positions"] == 0
    assert abs(sum(t["net_pnl"] for t in result["trades"]) - (result["summary"]["final_equity"]-cfg.initial_equity)) < 1e-7
    return row, result


def gate(candidate, baseline, stressed):
    checks = {
        "positive_net_return": candidate["total_return_pct"] > 0,
        "positive_double_cost_return": stressed["total_return_pct"] > 0,
        "return_at_least_baseline": candidate["total_return_pct"] >= baseline["total_return_pct"],
        "drawdown_no_worse_than_baseline": candidate["max_drawdown_pct"] <= baseline["max_drawdown_pct"],
        "at_least_20_closed_trades": candidate["closed_trades"] >= 20,
        "at_least_3_positive_quarters": sum(v > 0 for v in candidate["quarter_returns_pct"].values()) >= 3,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("develop", "holdout", "recent"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    _, cfg = frozen_strategy.load_frozen_strategy(Path(plan["base_manifest"]))
    cfg = replace(cfg, max_drawdown_stop_pct=0)
    window = {"develop": "development", "holdout": "holdout", "recent": "known_recent"}[args.phase]
    start, end = map(sim._utc_ms, plan[window])
    path = HISTORY if args.phase != "recent" else Path("data/snapshots/btc_diagnosis_20260927_90d.json.gz")
    data, funding, metadata = sim.load_market_snapshot(path)
    data, funding = slice_market(data, funding, end)
    report = {"research_id": plan["research_id"], "phase": args.phase, "window": plan[window],
              "input_hashes": hashes(), "market_metadata": metadata, "variants": {},
              "market_snapshot_sha256": frozen_strategy.sha256_file(path), "limitations": plan["limitations"]}
    if args.phase == "develop":
        for name, policy in plan["policies"].items():
            for weight in plan["tactical_weights"]:
                key = f"{name}_tactical_{weight:g}"
                row, _ = evaluate(data, funding, cfg, start, policy, weight)
                report["variants"][key] = row
                print(key, round(row["total_return_pct"], 4), round(row["max_drawdown_pct"], 4), row["closed_trades"], flush=True)
        selected = max(report["variants"], key=lambda k: report["variants"][k]["total_return_pct"] - report["variants"][k]["max_drawdown_pct"])
        row = report["variants"][selected]
        selection = {"candidate_id": "btc_reentry_selected_20260928", "live_orders_allowed": False,
                     "research_only": True, "status": "development_selected_pending_holdout", "selected_variant": selected,
                     "policy": row["policy"], "tactical_weight": row["tactical_weight"],
                     "base_manifest": plan["base_manifest"], "input_hashes": hashes(),
                     "selected_at_utc": datetime.now(timezone.utc).isoformat()}
        if SELECTED.exists() and json.loads(SELECTED.read_text())["input_hashes"] != selection["input_hashes"]:
            raise RuntimeError("Existing frozen selection has different inputs; use a new research ID")
        sim.save_json(SELECTED, selection)
        report["selected_variant"] = selected
        print("FROZEN SELECTION", selected, flush=True)
    else:
        selection = json.loads(SELECTED.read_text())
        if selection["input_hashes"] != hashes():
            raise RuntimeError("Selection inputs changed after development; refusing holdout")
        report["frozen_selection_sha256"] = frozen_strategy.sha256_file(SELECTED)
        setups = {"baseline": (plan["policies"]["baseline"], 1.0),
                  "selected": (selection["policy"], selection["tactical_weight"])}
        for label, (policy, weight) in setups.items():
            row, result = evaluate(data, funding, cfg, start, policy, weight)
            report["variants"][label] = row
            # Full result retained for attribution and charts. No execution target is exported.
            result.update({"research_only": True, "places_orders": False, "live_orders_allowed": False,
                           "candidate_id": selection["candidate_id"], "phase": args.phase})
            sim.save_json(Path(f"{OUTPUT}_{args.phase}_{label}_full.json"), result)
            print(args.phase, label, round(row["total_return_pct"], 4), round(row["max_drawdown_pct"], 4), row["closed_trades"], flush=True)
        if args.phase == "holdout":
            doubled = replace(cfg, maker_fee=cfg.maker_fee*2, taker_fee=cfg.taker_fee*2,
                              entry_slippage_bps=cfg.entry_slippage_bps*2, exit_slippage_bps=cfg.exit_slippage_bps*2,
                              depth_impact_bps=cfg.depth_impact_bps*2)
            row, _ = evaluate(data, funding, doubled, start, selection["policy"], selection["tactical_weight"])
            report["double_cost_selected"] = row
            report["gate"] = gate(report["variants"]["selected"], report["variants"]["baseline"], row)
            print("HOLDOUT GATE", report["gate"], flush=True)
    sim.save_json(Path(f"{OUTPUT}_{args.phase}.json"), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
