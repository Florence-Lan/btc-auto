"""Observe account risk independently of strategy inputs; never fill on an old mark."""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import trading_execution as execution

STATUS_PATH = execution.ROOT / "data/runtime/simulation_risk_monitor.json"
MAX_MARK_AGE_MS = 60_000
MAX_MARK_FUTURE_MS = 5_000


def monitor(client, now_ms, *, clock_available=True, clock_error=None,
            account=None, status_path: Path | None = None):
    account = account or execution.SimulationAccount()
    if account.persist and not account.path.exists():
        return None
    path = status_path or STATUS_PATH
    prior = execution.read_json(path, {}) or {}
    if not isinstance(prior, dict):
        prior = {}
    raw_state = execution.read_json(account.path) if account.persist else account.load()
    state = account.load()
    epoch = state.get("created_at_utc")
    if prior.get("account_epoch") != epoch:
        prior = {}
    started_ms = int(time.time() * 1000)
    health = {
        "mode": "simulation", "places_orders": False, "account_epoch": epoch,
        "checked_at_ms": started_ms, "status": "unavailable", "assessment": "not_evaluated",
        "clock_source": "exchange" if clock_available else "mark_endpoint",
        "last_success_at_ms": prior.get("last_success_at_ms"),
        "consecutive_failures": int(prior.get("consecutive_failures") or 0) + 1,
        "errors": {},
    }
    if clock_error:
        health["errors"]["exchange_clock"] = str(clock_error)[:240]

    def save():
        health["checked_at_ms"] = int(time.time() * 1000)
        health["checked_at_utc"] = datetime.fromtimestamp(
            health["checked_at_ms"] / 1000, timezone.utc).isoformat()
        execution.write_json(path, health)
        # Preserve interruptions as well as recoveries for prospective diagnostics.
        with path.with_suffix(".jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(health, ensure_ascii=False) + "\n")
        return health

    if not isinstance(raw_state, dict) or raw_state.get("mode") != "simulation" or not all(
            key in raw_state for key in ("wallet_balance", "position_qty", "entry_price", "created_at_utc")):
        health["errors"]["account"] = "Simulation account is missing, corrupt or has an unexpected mode"
        return {"monitor": save(), "fill": None}

    try:
        observation = client.mark_price_observation(execution.SYMBOL)
        price, source_ms = float(observation["price"]), int(observation["time_ms"])
        received_ms = int(time.time() * 1000)
        reference_ms = int(now_ms) + max(0, received_ms - started_ms) if clock_available else received_ms
        if not math.isfinite(price) or price <= 0 or source_ms <= 0:
            raise ValueError("Invalid mark-price observation")
        if source_ms > reference_ms + MAX_MARK_FUTURE_MS or reference_ms - source_ms > MAX_MARK_AGE_MS:
            raise ValueError("Mark-price observation is stale or ahead of the monitoring clock")
        # A valid mark endpoint supplies its own exchange clock if /time is down.
        # This fallback is only for risk monitoring, never for strategy entry.
        assessment_ms = reference_ms if clock_available else source_ms
        history = state.get("position_history") or []
        if history and assessment_ms < int(history[-1]["time_ms"]):
            raise ValueError("Risk observation predates the latest account inventory")
    except Exception as exc:
        health["errors"]["mark_price"] = f"{type(exc).__name__}: {exc}"[:240]
        return {"monitor": save(), "fill": None}

    # Apply the hard stop BEFORE any funding request. Full closure needs no new
    # quantity sizing; an exchangeInfo outage cannot prevent this reduction.
    def persisted_exit(previous):
        # observe() saves the account before archiving a fill. If the archive
        # fails, report the verifiable closure instead of claiming no execution.
        updated = execution.read_json(account.path) if account.persist else account.load()
        if not isinstance(updated, dict):
            return None
        fill = (updated.get("fills") or [None])[-1]
        if (previous.get("position_qty") and updated.get("position_qty") == 0
                and updated.get("created_at_utc") == previous.get("created_at_utc")
                and updated.get("risk_halt_at_utc") and isinstance(fill, dict)
                and fill.get("sequence") == int(previous.get("fill_count_total") or 0) + 1
                and fill.get("time_utc") == datetime.fromtimestamp(assessment_ms / 1000, timezone.utc).isoformat()
                and fill.get("mode") == "simulation"):
            return {"account_risk": updated["account_risk"], "fill": fill,
                    "funding_pnl": updated["funding_pnl"]}
        return None

    archive_failed = False
    try:
        result = account.observe(price, assessment_ms, (), False)
    except Exception as exc:
        result = persisted_exit(state)
        if result is None:
            health["errors"]["account"] = f"{type(exc).__name__}: {exc}"[:240]
            return {"monitor": save(), "fill": None}
        archive_failed = True
        health["errors"]["fill_archive"] = f"Exit persisted; fill archive failed: {type(exc).__name__}: {exc}"[:240]
    first_fill = result["fill"]
    # Without /time, defer funding until an authoritative cutoff returns. Its
    # settlement uses inventory history, so a late charge is not lost on closure.
    events, available = execution.simulation_funding(client, account, assessment_ms) if clock_available and not archive_failed else ((), False)
    if available:
        age_after_fetch = reference_ms + max(0, int(time.time() * 1000) - received_ms) - source_ms
        if age_after_fetch > MAX_MARK_AGE_MS:
            available = False
            health["errors"]["funding"] = "Funding fetched after mark expiry; settlement deferred"
        else:
            before_funding = account.load()
            try:
                funded_result = account.observe(price, assessment_ms, events, True)
                funded_result["fill"] = first_fill or funded_result["fill"]
                result = funded_result
            except Exception as exc:
                available = False
                recovered = persisted_exit(before_funding)
                if recovered is not None:
                    result = recovered
                    health["errors"]["fill_archive"] = f"Exit persisted; fill archive failed: {type(exc).__name__}: {exc}"[:240]
                else:
                    health["errors"]["funding"] = f"Settlement deferred: {type(exc).__name__}: {exc}"[:240]
    if not available:
        health["errors"].setdefault("funding", "Funding unavailable; settlement deferred and new risk blocked")
    health.update(
        status="healthy" if clock_available and available else "degraded",
        assessment="evaluated", assessment_time_ms=assessment_ms,
        mark_price=price, mark_time_ms=source_ms,
        last_success_at_ms=int(time.time() * 1000), consecutive_failures=0,
        account_risk=result["account_risk"], hard_stop_fill=bool(result["fill"]),
    )
    result["monitor"] = save()
    return result


def status_view(path=STATUS_PATH, *, now_ms=None, account_epoch=None):
    """A previous healthy heartbeat must not hide a stopped or stalled monitor."""
    status = execution.read_json(path, {}) or {}
    if not isinstance(status, dict) or not status or (account_epoch and status.get("account_epoch") != account_epoch):
        return {"status": "not_observed", "assessment": "not_evaluated"}
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    try:
        age_ms = now_ms - int(status.get("checked_at_ms") or 0)
    except (ValueError, TypeError, OverflowError):
        return {**status, "status": "unavailable", "assessment": "not_evaluated",
                "errors": {"heartbeat": "Invalid monitoring timestamp"}}
    if age_ms < -MAX_MARK_FUTURE_MS or age_ms > 120_000:
        return {**status, "status": "stale", "assessment": "not_evaluated"}
    return status
