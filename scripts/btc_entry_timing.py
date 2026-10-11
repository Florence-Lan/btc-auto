"""Bound first entry by the hourly origin, independently of refreshed 5m targets."""
from collections.abc import Mapping
import math


POLICY_MODE = "origin_signal_window_v1"


def validate_policy(policy):
    """Validate an explicitly selected policy; absent policies retain legacy behavior."""
    if not isinstance(policy, Mapping) or policy.get("mode") != POLICY_MODE:
        raise ValueError("Unsupported BTC entry timing policy")
    seconds = policy.get("max_initial_entry_age_seconds")
    try:
        finite = isinstance(seconds, (int, float)) and math.isfinite(seconds)
    except OverflowError:
        finite = False
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not finite or seconds <= 0):
        raise ValueError("BTC initial entry age must be a finite positive number")
    return {"mode": POLICY_MODE, "max_initial_entry_age_seconds": seconds}


def decision_at(report, target, *, current_qty, now_ms):
    """Assess a raw report point or execution target without changing inventory.

    Existing same-direction exposure and all exits bypass the origin-age gate.
    A blocked reversal may close the old direction, but cannot open the new one.
    The window starts at the hourly executable open and excludes its end.
    """
    report, target = report or {}, target or {}
    shadow = report.get("shadow_profile") or {}
    exposure = float(target.get("target_leverage", target.get("signed_qty", 0)) or 0)
    new_direction = (abs(exposure) > 1e-12
                     and (abs(current_qty) <= 1e-12 or exposure * current_qty < 0))
    origin = target.get("origin_signal_time_ms")
    diagnostics = {
        "enabled": False, "allowed": True, "status": "not_configured",
        "reasons": [], "checked_at_ms": now_ms,
        "requires_new_direction": new_direction,
        "origin_signal_time_ms": origin if isinstance(origin, int) and not isinstance(origin, bool) else None,
        "origin_signal_age_seconds": None,
        "max_initial_entry_age_seconds": None,
    }
    if not isinstance(shadow, Mapping) or "entry_timing_policy" not in shadow:
        return diagnostics
    diagnostics["enabled"] = True
    try:
        policy = validate_policy(shadow["entry_timing_policy"])
        diagnostics.update(policy)
    except (ValueError, OverflowError):
        if new_direction:
            diagnostics.update(allowed=False, status="invalid_policy",
                               reasons=["btc_entry_timing_policy_invalid"])
        else:
            diagnostics["status"] = "existing_direction" if exposure else "exit_or_flat"
        return diagnostics

    valid_origin = isinstance(origin, int) and not isinstance(origin, bool) and origin > 0
    if valid_origin:
        diagnostics["origin_signal_age_seconds"] = (now_ms - origin) / 1000
    if not new_direction:
        diagnostics["status"] = "existing_direction" if exposure else "exit_or_flat"
        return diagnostics
    if origin is None:
        status, reason = "origin_missing", "btc_initial_entry_origin_missing"
    elif not valid_origin:
        status, reason = "origin_invalid", "btc_initial_entry_origin_invalid"
    elif origin > now_ms:
        status, reason = "origin_future", "btc_initial_entry_origin_future"
    elif now_ms - origin >= policy["max_initial_entry_age_seconds"] * 1000:
        status, reason = "expired", "btc_initial_entry_origin_expired"
    else:
        diagnostics["status"] = "within_window"
        return diagnostics
    diagnostics.update(allowed=False, status=status, reasons=[reason])
    return diagnostics
