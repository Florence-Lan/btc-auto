"""Resolve the strategy selected by the simulation terminal, without dated defaults."""
import json
from pathlib import Path

import simulate_range_swing as sim


def candidate_path(selection_path: Path | None = None) -> Path:
    root = sim.repo_root()
    selection_path = selection_path or root / "config/active_simulation_candidate.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("execution_mode") != "simulation":
        raise ValueError("Active strategy selection must use simulation")
    path = (root / selection["candidate_path"]).resolve()
    if path.parent != root / "config" or not path.is_file():
        raise ValueError("Active strategy candidate must be a file in config/")
    return path
