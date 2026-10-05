"""Recheck current entry restrictions at execution; exits never depend on data health."""
from dataclasses import asdict
import json
from pathlib import Path

import event_risk
import macro_regime
import multifactor
import public_context


def decision_at(report, timestamp, side):
    report = report or {}
    qualification = report.get('strategy_qualification')
    if qualification is not None and (not isinstance(qualification, dict)
            or qualification.get('approved_for_forward_simulation') is not True):
        return {'allowed': False, 'status': 'blocked', 'checked_at_ms': timestamp,
                'reasons': ['strategy_not_qualified'], 'strategy_qualification': qualification}
    configured = any(report.get(name) is not None for name in
                     ("event_overlay", "multifactor_overlay", "macro_overlay"))
    context = report.get("execution_entry_context")
    if not configured and not context:
        return {"allowed": True, "status": "not_configured", "checked_at_ms": timestamp, "reasons": []}
    details, reasons = {}, []
    if not isinstance(context, dict):
        # Old reports do not carry enough information to reevaluate a calendar
        # boundary or factor expiry. Never treat their cached approval as current.
        return {"allowed": False, "status": "blocked", "checked_at_ms": timestamp,
                "reasons": ["execution_entry_context_missing"]}
    try:
        public = None
        if context.get("event_snapshot"):
            path = Path(context["event_snapshot"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            event = event_risk.event_decision_at(event_risk.events_from_payload(payload), timestamp)
            details["event"] = asdict(event)
            if not event.allowed:
                reasons.append("current_event_blocks_entries")
            if "coverage_checks" in payload:
                public = payload
                healthy, missing = public_context.health_at(public, timestamp)
                details["public_sources"] = {"healthy": healthy, "missing": missing}
                if not healthy:
                    reasons.append("current_public_sources_unavailable")
        elif report.get("event_overlay") is not None:
            reasons.append("event_snapshot_missing")
        if context.get("factor_profile"):
            profile = multifactor.load_profile(Path(context["factor_profile"]))
            if profile["availability_mode"] != "first_seen":
                raise ValueError("Execution requires first-seen factors")
            if multifactor.profile_hash(profile) != context["factor_profile_sha256"]:
                raise ValueError("Execution factor profile changed")
            if profile.get("public_context_enabled") and public is None:
                raise ValueError("Required public context unavailable")
            factor = multifactor.load_snapshot(Path(context["factor_snapshot"]), "first_seen")
            decision = multifactor.decision_at(factor, timestamp, side, profile, public)
            details["factor"] = asdict(decision)
            if not decision.allowed:
                reasons.extend("current_factor:" + reason for reason in decision.reasons)
        elif report.get("multifactor_overlay") is not None:
            reasons.append("factor_profile_missing")
        if context.get("macro_snapshot"):
            snapshot = macro_regime.load_macro_snapshot(Path(context["macro_snapshot"]))
            decision = macro_regime.macro_decision_at(snapshot, timestamp,
                enabled_factors=context["macro_factors"],
                min_multiplier=context["macro_min_multiplier"], block_score=context["macro_block_score"])
            details["macro"] = asdict(decision)
            if not decision.allowed:
                reasons.append("current_macro_blocks_entries")
        elif report.get("macro_overlay") is not None:
            reasons.append("macro_snapshot_missing")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, AttributeError, IndexError) as exc:
        # A broken refresh/read only prevents new exposure, including old-target retries.
        reasons.append("entry_context_unavailable:" + type(exc).__name__)
    return {"allowed": not reasons, "status": "allowed" if not reasons else "blocked",
            "checked_at_ms": timestamp, "side": side, "reasons": reasons, **details}


def constrain_quantity(requested, current_qty, decision):
    # Clamp on the exchange quantity grid, without a leverage/equity round trip.
    if decision["allowed"]:
        return requested
    if requested * current_qty < 0:
        return 0.0  # Close the old direction; do not open the rejected reversal.
    if abs(requested) >= abs(current_qty) - 1e-12:
        return current_qty
    return requested
