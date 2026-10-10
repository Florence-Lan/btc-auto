from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests


DecisionProvider = Callable[[dict[str, Any]], dict[str, Any]]
DECISION_CACHE_VERSION = 3
REVIEW_PROMPT_VERSION = 3
REVIEW_CONTEXT_VERSION = 3
ADMISSION_POLICY_VERSION = 3
BASE_BAR_MS = 5 * 60 * 1000


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "allow": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
    },
    "required": ["allow", "confidence", "reason", "risk_flags"],
    "additionalProperties": False,
}


SYSTEM_INSTRUCTIONS = """You are a BTCUSDT futures entry risk reviewer.
Use only the supplied point-in-time JSON. Never assume unpublished news, future prices or missing
indicator values. The information cutoff is decision_clock.decision_asof_ms, not the original entry
time, the signal candle's open time, or the wall clock when this review runs. Information first
available after the candle open but at or before the decision cutoff is valid current information.
Future scheduled event dates are not future information if their schedules were already available.
Missing or excluded inputs are unknown, not evidence of a favorable market.
You may only approve or reject the strategy target, never alter direction, leverage, stops or exits.
Judge current evidence at the strategy's own timeframe and indicator periods. For timeseries trend,
use last_confirmed_side, trend_side_under_exit_rule and target_side_valid_under_exit_rule to assess
present validity across the supplied history, not just the latest neutral bar.
Neutral entry-threshold conditions or mild opposite raw EMA ordering do not themselves invalidate
an existing strategy target: its exit rule requires an opposite confirmed threshold transition.
Respect admission_policy. When historical_performance_required=true, remain conservative: require
sufficient historical evidence and reject unclear edge, sparse critical inputs or elevated risk.
For explicitly authorized forward simulation with historical_performance_required=false, allow
sample collection when the strategy remains currently valid and no concrete material risk conflict
is identified. Missing profitability proof, a small sample, original signal age alone, a near-neutral
factor score alone, or small opposite 5m-to-4h returns alone are not entry vetoes. Do not confuse
short-horizon noise with invalidation of a slower strategy. For delayed entries, evaluate current
trend validity, displacement from the original entry, current volatility and proposed exposure;
do not approve merely because the signal was once valid. Reject concrete sufficiently serious
current conflicts, genuinely missing critical context, or elevated current risk. Never rely on an
excluded post-cutoff input; its exclusion alone is not a veto when it is optional and the remaining
current context is sufficient. This is not mandatory approval and does not bypass any risk gate.
strategy_run describes the signal engine, not execution-account trades. Confidence describes your
review judgment, not a calibrated probability of profit. For a rejection, state the current evidence,
the supplied value and units where available, its relevance to this strategy's horizon or risk, and
why it is material. Do not invent score thresholds. Return a short reason and concrete risk flags."""

REVIEW_PROMPT_SHA256 = hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp_ms(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    numeric = _finite(value)
    if numeric is not None:
        return int(numeric) if numeric >= 0 else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _utc_from_ms(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _decision_clock(report: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    point = report.get("execution_target") or (report.get("summary") or {}).get("last_equity_point") or {}
    cutoff = _timestamp_ms(report.get("decision_asof_ms"))
    source = "report.decision_asof_ms"
    if cutoff is None:
        cutoff = _timestamp_ms(report.get("generated_at_utc"))
        source = "report.generated_at_utc_fallback" if cutoff is not None else "missing"
    signal_open = _timestamp_ms(target.get("signal_time_ms", point.get("time_ms")))
    available = _timestamp_ms(point.get("available_time_ms", target.get("available_time_ms")))
    origin = _timestamp_ms(target.get("origin_signal_time_ms", point.get("origin_signal_time_ms")))
    entry_times = [_timestamp_ms(c.get("entry_time_utc")) for c in point.get("components") or []]
    original_entry = min((t for t in entry_times if t is not None), default=None)
    return {
        "decision_asof_ms": cutoff,
        "decision_asof_utc": _utc_from_ms(cutoff),
        "cutoff_source": source,
        "cutoff_missing": cutoff is None,
        "report_generated_at_utc": report.get("generated_at_utc"),
        "signal_candle_open_ms": signal_open,
        "signal_candle_open_utc": _utc_from_ms(signal_open),
        "signal_available_at_ms": available,
        "signal_available_at_utc": _utc_from_ms(available),
        "origin_signal_time_ms": origin,
        "origin_signal_time_utc": _utc_from_ms(origin),
        "original_component_entry_time_ms": original_entry,
        "original_component_entry_time_utc": _utc_from_ms(original_entry),
        "signal_time_semantics": "closed_5m_candle_open_not_information_cutoff",
    }


# Publication/first-seen clocks establish availability; observed/quote clocks also
# prevent future measurements from entering a context. Meeting/event dates and
# operational next-retry times do not establish information availability.
_INPUT_CLOCK_FIELDS = {
    "available_time_ms", "available_at_ms", "available_at_utc", "available_ms",
    "first_seen_ms", "first_seen_at_ms", "first_seen_utc", "first_seen_at_utc",
    "observed_at_ms", "observed_at_utc", "latest_observed_at_ms", "quote_at_utc",
    "decision_asof_ms", "asof_ms", "asof_utc", "generated_at_utc",
    "close_time_ms",
}


def _causal_input(
    value: Any, cutoff_ms: int | None, path: str, excluded: list[dict[str, Any]],
) -> Any:
    """Copy nested input and omit records demonstrably outside the snapshot."""
    if isinstance(value, Mapping):
        clocks = {key: _timestamp_ms(value[key]) for key in _INPUT_CLOCK_FIELDS if key in value and value[key] is not None}
        if clocks and (cutoff_ms is None or any(t is None or t > cutoff_ms for t in clocks.values())):
            excluded.append({
                "path": path,
                "reason": "missing_decision_cutoff" if cutoff_ms is None else
                          "invalid_source_timestamp" if any(t is None for t in clocks.values()) else
                          "after_decision_cutoff",
                "source_timestamps_ms": clocks,
            })
            return None
        result = {}
        for key, item in value.items():
            filtered = _causal_input(item, cutoff_ms, f"{path}.{key}", excluded)
            if filtered is not None or item is None:
                result[key] = filtered
        return result
    if isinstance(value, list):
        return [filtered for index, item in enumerate(value)
                if (filtered := _causal_input(item, cutoff_ms, f"{path}[{index}]", excluded)) is not None]
    return value


def _recent_price_features(report: Mapping[str, Any], cutoff_ms: int | None = None) -> dict[str, Any]:
    if cutoff_ms is None:
        cutoff_ms = _decision_clock(report, {}).get("decision_asof_ms")
    observations: dict[int, tuple[float, int]] = {}
    excluded = 0
    for point in report.get("equity_curve") or []:
        point = point or {}
        price = _finite(point.get("price"))
        opened = _timestamp_ms(point.get("time_ms"))
        available = _timestamp_ms(point.get("available_time_ms"))
        # Legacy curve points had only the open time of a fixed 5m base bar.
        if available is None and opened is not None and "available_time_ms" not in point:
            available = opened + BASE_BAR_MS
        if (cutoff_ms is None or price is None or price <= 0 or opened is None
                or available is None or available < opened or opened > cutoff_ms or available > cutoff_ms):
            excluded += 1
            continue
        observations[opened] = (price, available)
    times = sorted(observations)[-49:]
    prices = [observations[t][0] for t in times]
    returns: dict[str, float | None] = {}
    for label, bars in (("5m", 1), ("15m", 3), ("1h", 12), ("4h", 48)):
        previous = observations.get(times[-1] - bars * BASE_BAR_MS) if times else None
        returns[label] = (prices[-1] / previous[0] - 1) * 100 if previous is not None else None
    recent = []
    for previous, current, previous_time, current_time in zip(prices, prices[1:], times, times[1:]):
        if current_time - previous_time != BASE_BAR_MS:
            recent = []
        else:
            recent.append(math.log(current / previous))
    recent = recent[-12:]
    volatility = statistics.pstdev(recent) * 100 if len(recent) >= 2 else None
    return {
        "returns_pct": returns,
        "realized_5m_volatility_pct_1h": volatility,
        "observations": len(prices),
        "volatility_return_observations": len(recent),
        "base_bar_interval_ms": BASE_BAR_MS,
        "last_candle_open_ms": times[-1] if times else None,
        "last_price_available_at_ms": observations[times[-1]][1] if times else None,
        "excluded_price_observations": excluded,
    }


def _authorized_paper_sampling(report: Mapping[str, Any], mode: str) -> bool:
    qualification = report.get("strategy_qualification") or {}
    return (
        mode == "simulation"
        and qualification.get("approved_for_forward_simulation") is True
        and qualification.get("historical_performance_required") is False
    )


def build_decision_context(
    report: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    current_leverage: float,
    mode: str,
) -> dict[str, Any]:
    summary = report.get("summary") or {}
    point = report.get("execution_target") or summary.get("last_equity_point") or {}
    source_trades = {
        (trade.get("strategy"), trade.get("side"), trade.get("entry_time_utc")): trade
        for trade in report.get("trades") or []
    }
    qualification = report.get("strategy_qualification") or {}
    paper_sampling = _authorized_paper_sampling(report, mode)
    clock = _decision_clock(report, target)
    cutoff_ms = clock["decision_asof_ms"]
    excluded: list[dict[str, Any]] = []
    causal_point = _causal_input(point, cutoff_ms, "execution_target", excluded)
    point_available = causal_point is not None
    point = causal_point or {}
    review_context = _causal_input(report.get("strategy_review_context") or {}, cutoff_ms,
                                  "strategy_review_context", excluded) or {}
    trend = review_context.get("timeseries_trend") or {}
    raw_trend = (report.get("strategy_review_context") or {}).get("timeseries_trend") or {}
    if raw_trend.get("last_closed_bar") and not trend.get("last_closed_bar"):
        # Indicator values derived from a future candle cannot survive removal
        # of just its provenance record.
        review_context.pop("timeseries_trend", None)
        trend = {}
    components = []
    for component in list(point.get("components") or [])[:6]:
        entry_ms = _timestamp_ms(component.get("entry_time_utc"))
        if cutoff_ms is not None and entry_ms is not None and entry_ms > cutoff_ms:
            excluded.append({"path": "execution_target.components", "reason": "after_decision_cutoff",
                             "source_timestamps_ms": {"entry_time_utc": entry_ms}})
            continue
        source = source_trades.get((component.get("strategy"), component.get("side"),
                                   component.get("entry_time_utc")), {})
        component_context = {
            "strategy": str(component.get("strategy") or "unknown")[:80],
            "side": str(component.get("side") or "unknown")[:16],
            "entry_price": _finite(component.get("entry_price")),
            "signal_reason": str(component.get("signal_reason") or source.get("signal_reason") or "")[:240],
            "entry_time_utc": component.get("entry_time_utc"),
        }
        component_context["entry_signal_age_seconds"] = (
            (cutoff_ms - entry_ms) / 1000 if cutoff_ms is not None and entry_ms is not None else None
        )
        if str(component.get("strategy") or "").startswith("timeseries_trend"):
            component_context["strategy_context"] = trend or {
                "status": "unavailable", "unavailable_reason": "missing_causal_strategy_snapshot",
            }
        components.append(component_context)
    origin_ms = clock["origin_signal_time_ms"] or clock["original_component_entry_time_ms"]
    origin_price = _finite(target.get("origin_entry_price", point.get("origin_entry_price")))
    signal_price = _finite(target.get("signal_price")) if point_available else None
    proposed = float(target.get("target_leverage") or 0)
    proposed_side = "long" if proposed > 0 else "short"
    price_change = (signal_price / origin_price - 1) * 100 if signal_price and origin_price and origin_price > 0 else None
    multifactor = {
        key: value for key, value in (report.get("multifactor_overlay") or {}).items()
        if key in {"candidate_id", "profile_sha256", "availability_mode", "current",
                   "group_coverage_pct", "data_metadata"}
    }
    if multifactor.get("data_metadata"):
        multifactor["data_metadata"] = {
            key: value for key, value in multifactor["data_metadata"].items()
            if key in {"generated_at_utc", "source_status", "availability"}
        }
        excluded_before_metadata = len(excluded)
        metadata = _causal_input(multifactor["data_metadata"], cutoff_ms,
                                 "multifactor_overlay.data_metadata", excluded)
        if metadata is None:
            multifactor = {}
        else:
            multifactor["data_metadata"] = metadata
            if len(excluded) > excluded_before_metadata:
                # Scores cannot be assumed causal when their source snapshot
                # advertises measurements beyond this decision's cutoff.
                multifactor.pop("current", None)
                multifactor.pop("group_coverage_pct", None)
    multifactor = _causal_input(multifactor, cutoff_ms, "multifactor_overlay", excluded) or {}
    macro = _causal_input(report.get("macro_overlay") or {}, cutoff_ms, "macro_overlay", excluded) or {}
    event = _causal_input(report.get("event_overlay") or {}, cutoff_ms, "event_overlay", excluded) or {}
    return {
        "symbol": "BTCUSDT",
        "execution_mode": mode,
        "signal_time_ms": int(target.get("signal_time_ms") or 0),
        "position_id": str(target.get("position_id") or ""),
        "signal_price": signal_price,
        "current_leverage": round(current_leverage, 8),
        "proposed_target_leverage": round(proposed, 8),
        "proposed_side": proposed_side,
        "decision_clock": clock,
        "components": components,
        "strategy_review_context": review_context,
        "delayed_entry": {
            "currently_flat": abs(current_leverage) <= 1e-12,
            "action": "new_entry" if abs(current_leverage) <= 1e-12 else
                      "reversal" if current_leverage * proposed < 0 else "exposure_increase",
            "origin_signal_age_seconds": (cutoff_ms - origin_ms) / 1000
                if cutoff_ms is not None and origin_ms is not None else None,
            "origin_entry_price": origin_price,
            "price_change_from_origin_pct": price_change,
            "adverse_move_from_origin_pct": price_change * (-1 if proposed_side == "long" else 1)
                if price_change is not None else None,
            "age_alone_is_veto": False if paper_sampling else None,
        },
        "admission_policy": {
            "version": ADMISSION_POLICY_VERSION,
            "approved_for_forward_simulation": qualification.get("approved_for_forward_simulation") is True,
            "historical_performance_required": not paper_sampling,
            "forward_validated": qualification.get("forward_validated") is True,
            "review_standard": "material_current_risk_conflict" if paper_sampling else
                               "conservative_historical_and_current_evidence",
            "weak_short_horizon_opposition_alone_is_veto": False if paper_sampling else None,
            "near_neutral_factor_score_alone_is_veto": False if paper_sampling else None,
        },
        "review_audit": {
            "prompt_version": REVIEW_PROMPT_VERSION,
            "prompt_sha256": REVIEW_PROMPT_SHA256,
            "context_version": REVIEW_CONTEXT_VERSION,
            "admission_policy_version": ADMISSION_POLICY_VERSION,
        },
        "input_integrity": {
            "information_cutoff_ms": cutoff_ms,
            "excluded_inputs": excluded,
            "untimestamped_aggregate_semantics": "producer_snapshot_at_decision_cutoff_not_independent_publication_time",
        },
        "recent_market": _recent_price_features(report, cutoff_ms),
        "strategy_run": {
            "performance_scope": "signal_engine_summary_not_execution_account",
            "current_drawdown_pct": _finite((summary.get("last_equity_point") or {}).get("drawdown_pct")),
            "max_drawdown_pct": _finite(summary.get("max_drawdown_pct")),
            "trades": int(summary.get("trades") or 0),
            "win_rate_pct": _finite(summary.get("win_rate_pct")),
            "profit_factor": _finite(summary.get("profit_factor")),
        },
        "risk_diagnostics": _causal_input(report.get("risk_diagnostics") or {}, cutoff_ms,
                                          "risk_diagnostics", excluded) or {},
        "macro_overlay": macro,
        "event_overlay": event,
        "multifactor_overlay": multifactor,
    }


def _response_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
            if content.get("type") == "refusal":
                raise RuntimeError(f"LLM refused the trade review: {content.get('refusal')}")
    raise RuntimeError("LLM response did not contain structured output text")


def request_openai_decision(context: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-5.4-nano").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when LLM_TRADE_GATE_ENABLED=true")
    if not model:
        raise RuntimeError("LLM_MODEL is required when LLM_TRADE_GATE_ENABLED=true")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "12") or 12)
    if timeout <= 0 or timeout > 60:
        raise ValueError("LLM_TIMEOUT_SECONDS must be between 0 and 60")
    body = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        "max_output_tokens": 240,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "trade_entry_decision",
                "strict": True,
                "schema": DECISION_SCHEMA,
            }
        },
    }
    started = time.monotonic()
    response = requests.post(
        f"{base_url}/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    decision = json.loads(_response_text(payload))
    decision["model"] = model
    decision["provider"] = "openai"
    decision["response_id"] = payload.get("id")
    decision["latency_ms"] = round((time.monotonic() - started) * 1000)
    return decision


def request_codex_decision(context: dict[str, Any]) -> dict[str, Any]:
    configured_bin = os.getenv("CODEX_BIN", "").strip()
    codex_bin = configured_bin or shutil.which("codex.exe") or shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("Codex CLI was not found; set CODEX_BIN or install Codex CLI")
    timeout = float(os.getenv("LLM_CODEX_TIMEOUT_SECONDS", "90") or 90)
    if timeout <= 0 or timeout > 300:
        raise ValueError("LLM_CODEX_TIMEOUT_SECONDS must be between 0 and 300")
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "Do not call tools, inspect files, browse, or use information outside the JSON below. "
        "Return only the schema-conforming decision.\n\n"
        f"Point-in-time trade context:\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="btc-auto-llm-") as temporary:
        temporary_path = Path(temporary)
        schema_path = temporary_path / "decision-schema.json"
        output_path = temporary_path / "decision.json"
        schema_path.write_text(json.dumps(DECISION_SCHEMA), encoding="utf-8")
        command = [
            codex_bin,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--config",
            'history.persistence="none"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--cd",
            temporary,
        ]
        model = os.getenv("LLM_CODEX_MODEL", "").strip()
        if model:
            command.extend(["--model", model])
        command.append("-")
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown Codex error").strip()
            raise RuntimeError(f"Codex trade review failed: {detail[-500:]}")
        if not output_path.exists():
            raise RuntimeError("Codex trade review did not write a final decision")
        decision = json.loads(output_path.read_text(encoding="utf-8"))
    decision["model"] = model or "codex-default"
    decision["provider"] = "codex"
    decision["response_id"] = None
    decision["latency_ms"] = round((time.monotonic() - started) * 1000)
    return decision


def request_llm_decision(context: dict[str, Any]) -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "codex").strip().lower()
    if provider == "codex":
        return request_codex_decision(context)
    if provider == "openai":
        return request_openai_decision(context)
    raise ValueError("LLM_PROVIDER must be codex or openai")


def _validated_decision(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw.get("allow"), bool):
        raise ValueError("LLM decision allow must be a boolean")
    confidence = _finite(raw.get("confidence"))
    if confidence is None or not 0 <= confidence <= 1:
        raise ValueError("LLM decision confidence must be between 0 and 1")
    reason = str(raw.get("reason") or "").strip()
    if not reason:
        raise ValueError("LLM decision reason must not be empty")
    flags = raw.get("risk_flags") or []
    if not isinstance(flags, list):
        raise ValueError("LLM decision risk_flags must be an array")
    return {
        "allow": raw["allow"],
        "confidence": confidence,
        "reason": reason[:500],
        "risk_flags": [str(item)[:120] for item in flags[:6]],
        "model": str(raw.get("model") or os.getenv("LLM_MODEL", "")),
        "provider": str(raw.get("provider") or os.getenv("LLM_PROVIDER", "codex")),
        "response_id": raw.get("response_id"),
        "latency_ms": raw.get("latency_ms"),
    }


def _decision_key(target: Mapping[str, Any]) -> str:
    leverage = float(target.get("target_leverage") or 0)
    side = "long" if leverage > 0 else "short"
    identity = target.get("position_id") or target.get("origin_signal_time_ms") or target.get("signal_time_ms")
    return f"{identity}:{side}"


def _safe_target_leverage(current_leverage: float, requested_leverage: float) -> float:
    if current_leverage * requested_leverage < 0:
        return 0.0
    return current_leverage


def _rejection_cache_seconds() -> float:
    seconds = _finite(os.getenv("LLM_REJECTION_CACHE_SECONDS", "900"))
    if seconds is None or not 0 < seconds <= 3600:
        raise ValueError("LLM_REJECTION_CACHE_SECONDS must be between 0 (exclusive) and 3600")
    return seconds


def _review_age_seconds(prior: Mapping[str, Any], checked_at_utc: str) -> float | None:
    # Legacy decided_at_utc was overwritten on every cache hit, so it cannot
    # establish the age of an actual review. Only the new immutable field can.
    if (prior.get("cache_version") != DECISION_CACHE_VERSION
            or prior.get("review_prompt_sha256") != REVIEW_PROMPT_SHA256
            or prior.get("review_context_version") != REVIEW_CONTEXT_VERSION
            or prior.get("admission_policy_version") != ADMISSION_POLICY_VERSION):
        return None
    try:
        reviewed = datetime.fromisoformat(str(prior.get("reviewed_at_utc") or ""))
        checked = datetime.fromisoformat(checked_at_utc)
        if reviewed.tzinfo is None or checked.tzinfo is None:
            return None
        age = (checked - reviewed).total_seconds()
    except (ValueError, TypeError, OverflowError):
        return None
    return age if age >= 0 else None


def _review_metadata(cache_seconds: float | None) -> dict[str, Any]:
    reviewed_at = utc_now()
    expires_at = (
        (datetime.fromisoformat(reviewed_at) + timedelta(seconds=cache_seconds)).isoformat()
        if cache_seconds is not None else None
    )
    return {
        "cache_version": DECISION_CACHE_VERSION,
        "review_prompt_version": REVIEW_PROMPT_VERSION,
        "review_prompt_sha256": REVIEW_PROMPT_SHA256,
        "review_context_version": REVIEW_CONTEXT_VERSION,
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "reviewed_at_utc": reviewed_at,
        "decided_at_utc": reviewed_at,
        "cache_age_seconds": 0.0,
        "rejection_cache_seconds": cache_seconds,
        "rejection_cache_expires_at_utc": expires_at,
    }


def apply_llm_trade_gate(
    report: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    current_qty: float,
    equity: float,
    mark_price: float,
    mode: str,
    previous_decision: Mapping[str, Any] | None = None,
    decision_provider: DecisionProvider | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gated = dict(target)
    requested = float(target.get("target_leverage") or 0)
    current_leverage = current_qty * mark_price / equity if equity > 0 and mark_price > 0 else 0.0
    checked_at = utc_now()
    base = {
        "enabled": env_enabled("LLM_TRADE_GATE_ENABLED"),
        "current_leverage": current_leverage,
        "requested_target_leverage": requested,
        "effective_target_leverage": requested,
        "checked_at_utc": checked_at,
    }
    if not base["enabled"]:
        return gated, {**base, "status": "disabled"}

    same_direction = current_leverage * requested > 0
    increases_same_direction = same_direction and abs(requested) > abs(current_leverage) + 1e-9
    opens_or_reverses = abs(requested) > 1e-12 and (
        abs(current_leverage) <= 1e-12 or current_leverage * requested < 0
    )
    if not (increases_same_direction or opens_or_reverses):
        return gated, {**base, "status": "bypassed_non_increasing"}

    key = _decision_key(target)
    prior = previous_decision or {}
    prior_matches = prior.get("decision_key") == key
    prior_allow = prior.get("allow") is True
    prior_limit = _finite(prior.get("approved_target_leverage"))
    cache_seconds = None
    review_trigger = "new_target"
    context_audit: dict[str, Any] = {}
    try:
        cache_seconds = _rejection_cache_seconds()
        age = _review_age_seconds(prior, checked_at)
        if (prior.get("review_execution_mode") != mode
                or prior.get("review_historical_performance_required") is not
                (not _authorized_paper_sampling(report, mode))):
            age = None
        reusable_approval = (
            prior_allow and age is not None and prior_limit is not None
            and abs(requested) <= abs(prior_limit) + 1e-9
        )
        reusable_rejection = not prior_allow and age is not None and age < cache_seconds
        if prior_matches and (reusable_approval or reusable_rejection):
            effective = requested if prior_allow else _safe_target_leverage(current_leverage, requested)
            gated["target_leverage"] = effective
            return gated, {
                **dict(prior),
                **base,
                "status": "cached_approved" if prior_allow else "cached_rejected",
                "cache_age_seconds": age,
                "rejection_cache_seconds": cache_seconds,
                "rejection_cache_expires_at_utc": (
                    (datetime.fromisoformat(str(prior["reviewed_at_utc"]))
                     + timedelta(seconds=cache_seconds)).isoformat()
                    if not prior_allow else None
                ),
                "effective_target_leverage": effective,
                "decision_key": key,
            }
        if prior_matches:
            review_trigger = (
                "untrusted_approval_cache" if prior_allow and age is None
                else "approved_exposure_increase" if prior_allow
                else "rejection_cache_expired" if age is not None
                else "untrusted_rejection_cache"
            )
        context = build_decision_context(
            report,
            target,
            current_leverage=current_leverage,
            mode=mode,
        )
        context_audit = {
            "review_execution_mode": mode,
            "review_historical_performance_required": context["admission_policy"]["historical_performance_required"],
            "review_context_sha256": hashlib.sha256(json.dumps(
                context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "review_decision_clock": context["decision_clock"],
        }
        provider = decision_provider or request_llm_decision
        decision = _validated_decision(provider(context))
        min_confidence = float(os.getenv("LLM_MIN_CONFIDENCE", "0.70") or 0.70)
        if not 0 <= min_confidence <= 1:
            raise ValueError("LLM_MIN_CONFIDENCE must be between 0 and 1")
        allowed = bool(decision["allow"] and decision["confidence"] >= min_confidence)
        effective = requested if allowed else _safe_target_leverage(current_leverage, requested)
        gated["target_leverage"] = effective
        review_metadata = _review_metadata(cache_seconds)
        return gated, {
            **base,
            **decision,
            **review_metadata,
            **context_audit,
            "allow": allowed,
            "raw_allow": decision["allow"],
            "minimum_confidence": min_confidence,
            "status": "approved" if allowed else "rejected",
            "decision_key": key,
            "approved_target_leverage": requested if allowed else 0.0,
            "effective_target_leverage": effective,
            "review_trigger": review_trigger,
            "rejection_cache_expires_at_utc": (
                None if allowed else review_metadata["rejection_cache_expires_at_utc"]
            ),
        }
    except Exception as exc:
        effective = _safe_target_leverage(current_leverage, requested)
        gated["target_leverage"] = effective
        return gated, {
            **base,
            **_review_metadata(cache_seconds),
            **context_audit,
            "status": "error_blocked",
            "allow": False,
            "decision_key": key,
            "approved_target_leverage": 0.0,
            "effective_target_leverage": effective,
            "reason": str(exc)[:500],
            "risk_flags": ["llm_unavailable_or_invalid"],
            "review_trigger": review_trigger,
        }
