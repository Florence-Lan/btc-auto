"""Account equity controls, applied independently of the shadow strategy ledger."""
from datetime import datetime, timezone
import math

from portfolio_risk import DrawdownRiskPolicy, drawdown_multiplier


def constrain_target(state, target, equity, current_qty, price, now_ms, report=None):
    if not all(math.isfinite(v) for v in (equity, current_qty, price)) or price <= 0:
        raise ValueError("Account risk requires finite equity, quantity and positive price")
    peak = max(float(state.get("peak_equity") or equity), equity)
    drawdown = max(0.0, (peak - equity) / peak * 100) if peak > 0 else 100.0
    state["peak_equity"] = peak
    state["max_drawdown_pct"] = max(float(state.get("max_drawdown_pct") or 0), drawdown)
    policy = DrawdownRiskPolicy()
    shadow_halt = (report or {}).get("risk_diagnostics", {}).get("hard_halt_time_ms")
    if state["max_drawdown_pct"] >= policy.hard_stop_pct or equity <= 0 or shadow_halt is not None:
        if not state.get("risk_halt_at_utc"):
            state["risk_halt_at_utc"] = datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat()
            state["risk_halt_reason"] = "shadow_hard_halt" if shadow_halt is not None else "account_drawdown"
    halted = bool(state.get("risk_halt_at_utc"))
    multiplier = 0.0 if halted else drawdown_multiplier(drawdown / 100, policy)
    guarded = {**target, "target_leverage": float(target["target_leverage"]) * multiplier}
    diagnostics = {"status": "halted" if halted else "throttled" if multiplier < 1 else "normal",
                   "drawdown_pct": drawdown, "risk_multiplier": multiplier,
                   "soft_limit_pct": policy.soft_start_pct, "hard_limit_pct": policy.hard_stop_pct,
                   "halt_at_utc": state.get("risk_halt_at_utc"), "halt_reason": state.get("risk_halt_reason")}
    state["account_risk"] = diagnostics
    return guarded, diagnostics


def block_increases(target, current_qty, equity, price):
    """Missing funding data may prevent additions but never prevent exits."""
    current = current_qty * price / max(equity, 1e-12)
    requested = float(target["target_leverage"])
    if current * requested < 0:
        requested = 0.0
    elif abs(requested) > abs(current):
        requested = current
    return {**target, "target_leverage": requested}
