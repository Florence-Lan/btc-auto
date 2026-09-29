"""Load the development-frozen reentry candidate for isolated paper evaluation."""
import json
from pathlib import Path

import frozen_strategy
import simulate_range_swing as sim
from validate_reentry_research import make_sleeves


def load_candidate(path: Path, manifest_path: Path):
    profile = json.loads(path.read_text())
    if profile.get("live_orders_allowed") is not False or profile.get("research_only") is not True:
        raise ValueError("Reentry candidate must be research-only with live orders disabled")
    if not 0 <= profile["tactical_weight"] <= 1:
        raise ValueError("Invalid tactical risk weight")
    root = sim.repo_root()
    if (root / profile["base_manifest"]).resolve() != manifest_path.resolve():
        raise ValueError("Research profile and base manifest do not match")
    plan_path = profile.get("research_plan", "config/reentry_research_plan_20260928.json")
    if plan_path not in ("config/reentry_research_plan_20260928.json", "config/cost_control_plan_20260928.json"):
        raise ValueError("Unknown research plan")
    required = ("scripts/research_reentry.py", "scripts/execution_ledger.py", "scripts/portfolio_risk.py",
                "scripts/simulate_range_swing.py", "scripts/validate_reentry_research.py", profile["base_manifest"],
                plan_path)
    if plan_path == "config/cost_control_plan_20260928.json":
        required += ("scripts/validate_cost_control.py", "scripts/reentry_candidate.py",
                     "scripts/diagnose_strategy_losses.py", "scripts/validate_exit_research.py")
    for filename in required:
        if frozen_strategy.sha256_file(root / filename) != profile["input_hashes"].get(filename):
            raise ValueError(f"Research candidate code/config changed: {filename}")
    plan = json.loads((root / plan_path).read_text())
    if plan["base_manifest"] != profile["base_manifest"]:
        raise ValueError("Research plan and candidate manifest do not match")
    allowed = {f"{name}_tactical_{weight:g}": (policy, weight)
               for name, policy in plan["policies"].items() for weight in plan["tactical_weights"]}
    if allowed.get(profile["selected_variant"]) != (profile["policy"], profile["tactical_weight"]):
        raise ValueError("Candidate parameters do not match the frozen research plan")
    return profile


def build_sleeves(data, funding, cfg, start, profile):
    return make_sleeves(data, funding, cfg, start, profile["policy"], profile["tactical_weight"])
