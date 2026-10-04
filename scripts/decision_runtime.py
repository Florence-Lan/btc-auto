"""Simulation decision clocks and observable judgment health.

Clock fallbacks can keep a calculation moving; they do not authorize a fill.
Execution must independently validate a timestamped mark and current entry gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data/runtime/strategy_judgment.json"
MAX_MARK_AGE_MS = 60_000
MAX_MARK_FUTURE_MS = 5_000
MAX_ANCHOR_AGE_MS = 60_000
MAX_HEARTBEAT_AGE_MS = 120_000
_ANCHOR_ATTRIBUTE = "_decision_runtime_clock_anchor"


def _timestamp(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Invalid exchange timestamp")
    try:
        result = int(value)
        if isinstance(value, float) and (not math.isfinite(value) or value != result):
            raise ValueError("Invalid exchange timestamp")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid exchange timestamp") from exc
    if result <= 0:
        raise ValueError("Invalid exchange timestamp")
    return result


def validate_mark(observation: Mapping[str, Any], now_ms: int | None = None) -> dict[str, Any]:
    """Return ``{price, time_ms, age_ms}`` only for a usable timestamped mark."""
    if not isinstance(observation, Mapping) or isinstance(observation.get("price"), bool):
        raise ValueError("Invalid mark-price observation")
    try:
        price = float(observation["price"])
        source_ms = _timestamp(observation["time_ms"])
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError("Invalid mark-price observation") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid mark-price observation")
    reference_ms = _timestamp(now_ms if now_ms is not None else int(time.time() * 1000))
    age_ms = reference_ms - source_ms
    if age_ms > MAX_MARK_AGE_MS or age_ms < -MAX_MARK_FUTURE_MS:
        raise ValueError("Mark-price observation is stale or ahead of the decision clock")
    return {"price": price, "time_ms": source_ms, "age_ms": age_ms}


def _error(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", message)
    message = re.sub(
        r"(?i)((?:api[_-]?key|api[_-]?secret|access[_-]?token|token|signature|authorization)\s*[:=]\s*)(?:Bearer\s+)?[^\s&,;]+",
        r"\1[redacted]", message,
    )
    return f"{type(exc).__name__}: {message}"[:240]


@dataclass(frozen=True)
class _Anchor:
    time_ms: int
    monotonic_seconds: float
    base_url: str
    source: str


def _remember(client: Any, timestamp: int, source: str) -> None:
    anchor = _Anchor(timestamp, time.monotonic(), str(getattr(client, "base_url", "")), source)
    try:
        setattr(client, _ANCHOR_ATTRIBUTE, anchor)
    except (AttributeError, TypeError):
        # Clients that deliberately prohibit additional attributes still have
        # the direct exchange and fresh-mark paths.
        pass


def _anchor_view(client: Any) -> dict[str, Any] | None:
    anchor = getattr(client, _ANCHOR_ATTRIBUTE, None)
    if not isinstance(anchor, _Anchor) or anchor.base_url != str(getattr(client, "base_url", "")):
        return None
    elapsed_ms = (time.monotonic() - anchor.monotonic_seconds) * 1000
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0 or elapsed_ms > MAX_ANCHOR_AGE_MS:
        return None
    return {"time_ms": anchor.time_ms + int(elapsed_ms),
            "anchor_age_ms": int(elapsed_ms), "anchor_source": anchor.source}


def resolve_clock(client: Any, mode: str = "simulation", *,
                  mark_observation: Mapping[str, Any] | None = None,
                  allow_anchor: bool = True, now_ms: int | None = None) -> dict[str, Any]:
    """Resolve a clock, retaining the primary error when simulation degrades.

    LIVE calls only the original server clock. Simulation may use a fresh mark
    timestamp or a bounded, process-local monotonic anchor for calculations.
    Only a direct server-time observation renews the anchor. A mark timestamp
    may lag its receipt, so neither it nor an anchor fallback extends the anchor
    lifetime. The anchor is independent of local wall-clock jumps.
    """
    if mode not in {"simulation", "live"}:
        raise ValueError("Execution mode must be simulation or live")
    try:
        timestamp = _timestamp(client.server_time_ms())
    except Exception as exc:
        if mode == "live":
            raise
        primary_exception = exc
        primary_error = _error(exc)
    else:
        if mode == "simulation":
            _remember(client, timestamp, "exchange")
        return {"time_ms": timestamp, "source": "exchange", "degraded": False,
                "primary_error": None, "anchor_age_ms": None, "errors": {}}

    errors = {"exchange_clock": primary_error}
    try:
        observation = mark_observation if mark_observation is not None else client.mark_price_observation()
        anchor = _anchor_view(client)
        reference_ms = (now_ms if now_ms is not None else anchor["time_ms"] if anchor else int(time.time() * 1000))
        mark = validate_mark(observation, reference_ms)
    except Exception as exc:
        errors["mark_price"] = _error(exc)
    else:
        return {"time_ms": mark["time_ms"], "source": "mark_endpoint", "degraded": True,
                "primary_error": primary_error, "anchor_age_ms": None, "errors": errors}

    anchor = _anchor_view(client) if allow_anchor else None
    if anchor is not None:
        return {**anchor, "source": "exchange_anchor", "degraded": True,
                "primary_error": primary_error, "errors": errors}
    raise RuntimeError("No valid simulation decision clock; " + "; ".join(errors.values())) from primary_exception


def write_judgment(payload: Mapping[str, Any], path: Path = STATUS_PATH, *,
                   now_ms: int | None = None) -> dict[str, Any]:
    """Atomically publish a poll heartbeat and the caller's latest judgment."""
    timestamp = _timestamp(now_ms if now_ms is not None else int(time.time() * 1000))
    result = {**payload, "checked_at_ms": timestamp,
              "checked_at_utc": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def status_view(path: Path = STATUS_PATH, *, now_ms: int | None = None,
                report_path: Path | str | None = None,
                account_epoch: str | None = None) -> dict[str, Any]:
    """Show generation identity and poll health without concealing stale data."""
    try:
        status = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"status": "not_observed", "judgment_status": "not_evaluated"}
    if not isinstance(status, dict) or not status:
        return {"status": "not_observed", "judgment_status": "not_evaluated"}
    if account_epoch is not None and status.get("account_epoch") != account_epoch:
        return {"status": "not_observed", "judgment_status": "not_evaluated"}
    if report_path is not None:
        try:
            matches = Path(status["report_path"]).resolve() == Path(report_path).resolve()
        except (ValueError, TypeError, KeyError, OSError):
            matches = False
        if not matches:
            return {"status": "not_observed", "judgment_status": "not_evaluated"}
    try:
        checked_at = _timestamp(status.get("checked_at_ms"))
        current_ms = _timestamp(now_ms if now_ms is not None else int(time.time() * 1000))
        age_ms = current_ms - checked_at
    except ValueError:
        return {**status, "status": "unavailable", "judgment_status": "not_evaluated", "decision_status": "unavailable",
                "errors": {**(status.get("errors") if isinstance(status.get("errors"), dict) else {}),
                           "heartbeat": "Invalid judgment heartbeat timestamp"}}
    result = {**status, "heartbeat_age_seconds": age_ms / 1000}
    decision = status.get("decision_status", status.get("judgment_status", "not_evaluated"))
    result.update(decision_status=decision, judgment_status=decision)
    if decision == "unavailable":
        result["status"] = "unavailable"
    elif decision == "evaluated":
        degraded = ((status.get("clock") or {}).get("degraded")
                    or status.get("execution_status") in {"deferred", "execution_deferred"}
                    or status.get("status") == "degraded")
        result["status"] = "degraded" if degraded else "healthy"
    else:
        result.setdefault("status", "waiting_for_bar")
    if age_ms < -MAX_MARK_FUTURE_MS or age_ms > MAX_HEARTBEAT_AGE_MS:
        result.update(status="stale", judgment_status="not_evaluated", decision_status="source_stale")
    return result


# Kept as a descriptive alias for integrations that publish general status.
write_status = write_judgment
