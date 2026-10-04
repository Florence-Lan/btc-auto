"""Resolve the strategy selected by the simulation terminal, without dated defaults."""
import json
import math
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import simulate_range_swing as sim


def _leverage_value(value, *, label, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite and at most 10") from exc
    if (not math.isfinite(result) or result > 10
            or result < 0 or (not allow_zero and result == 0)):
        lower = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite, {lower} and at most 10")
    return result


def _risk_limits(profile):
    if profile is None:
        return {}
    if not isinstance(profile, Mapping):
        raise ValueError("Strategy profile must be a mapping")
    limits = profile.get("risk_limits", {})
    if not isinstance(limits, Mapping):
        raise ValueError("Strategy risk_limits must be a mapping")
    return limits


def profile_leverage_cap(profile, default=2.0):
    """Read the simulation profile's authorized upper limit."""
    limits = _risk_limits(profile)
    return _leverage_value(limits.get("portfolio_leverage_cap", default),
                           label="Strategy leverage cap")


def apply_risk_limits(cfg, profile):
    """Override sizing limits without modifying the frozen signal parameters."""
    if "portfolio_leverage_cap" not in _risk_limits(profile):
        return cfg
    cap = profile_leverage_cap(profile)
    return replace(cfg, leverage=cap, timeseries_max_leverage=cap,
                   portfolio_leverage_cap=cap)


def simulation_leverage_cap(report=None, *, strategy_cap=None):
    """Resolve the strategy cap with an optional, lower environment ceiling."""
    if strategy_cap is None:
        if report is not None and not isinstance(report, Mapping):
            raise ValueError("Strategy report must be a mapping")
        config = (report or {}).get("config") or {}
        if not isinstance(config, Mapping):
            raise ValueError("Strategy report config must be a mapping")
        strategy_cap = config.get("portfolio_leverage_cap", 2.0)
    cap = _leverage_value(strategy_cap, label="Strategy leverage cap")
    configured = os.getenv("SIM_MAX_LEVERAGE")
    if configured is None or not configured.strip():
        return cap
    try:
        environment_cap = float(configured)
    except ValueError as exc:
        raise ValueError("SIM_MAX_LEVERAGE must be numeric") from exc
    environment_cap = _leverage_value(environment_cap, label="SIM_MAX_LEVERAGE", allow_zero=True)
    return min(cap, environment_cap)


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
