from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests


DecisionProvider = Callable[[dict[str, Any]], dict[str, Any]]


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


SYSTEM_INSTRUCTIONS = """You are a conservative BTCUSDT futures entry gate.
Use only the point-in-time data supplied in the request. Never assume unpublished news or future
prices. Decide whether the proposed strategy entry or exposure increase has enough evidence to be
allowed. Prefer rejection when signals conflict, inputs are sparse, risk is elevated, or the edge is
unclear. You may only approve or reject; you may not change direction, leverage, stops, or exits.
Return a short reason and concrete risk flags."""


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


def _recent_price_features(report: Mapping[str, Any]) -> dict[str, Any]:
    prices = [
        value
        for point in list(report.get("equity_curve") or [])[-49:]
        if (value := _finite((point or {}).get("price"))) is not None and value > 0
    ]
    returns: dict[str, float | None] = {}
    for label, bars in (("5m", 1), ("15m", 3), ("1h", 12), ("4h", 48)):
        returns[label] = (
            (prices[-1] / prices[-1 - bars] - 1) * 100
            if len(prices) > bars
            else None
        )
    log_returns = [math.log(current / previous) for previous, current in zip(prices, prices[1:])]
    recent = log_returns[-12:]
    volatility = statistics.pstdev(recent) * 100 if len(recent) >= 2 else None
    return {
        "returns_pct": returns,
        "realized_5m_volatility_pct_1h": volatility,
        "observations": len(prices),
    }


def build_decision_context(
    report: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    current_leverage: float,
    mode: str,
) -> dict[str, Any]:
    summary = report.get("summary") or {}
    point = report.get("execution_target") or summary.get("last_equity_point") or {}
    components = []
    for component in list(point.get("components") or [])[:6]:
        components.append({
            "strategy": str(component.get("strategy") or "unknown")[:80],
            "side": str(component.get("side") or "unknown")[:16],
            "entry_price": _finite(component.get("entry_price")),
            "signal_reason": str(component.get("signal_reason") or "")[:240],
        })
    return {
        "symbol": "BTCUSDT",
        "execution_mode": mode,
        "signal_time_ms": int(target.get("signal_time_ms") or 0),
        "position_id": str(target.get("position_id") or ""),
        "signal_price": _finite(target.get("signal_price")),
        "current_leverage": round(current_leverage, 8),
        "proposed_target_leverage": round(float(target.get("target_leverage") or 0), 8),
        "proposed_side": "long" if float(target.get("target_leverage") or 0) > 0 else "short",
        "components": components,
        "recent_market": _recent_price_features(report),
        "strategy_run": {
            "current_drawdown_pct": _finite((summary.get("last_equity_point") or {}).get("drawdown_pct")),
            "max_drawdown_pct": _finite(summary.get("max_drawdown_pct")),
            "trades": int(summary.get("trades") or 0),
            "win_rate_pct": _finite(summary.get("win_rate_pct")),
            "profit_factor": _finite(summary.get("profit_factor")),
        },
        "risk_diagnostics": report.get("risk_diagnostics") or {},
        "macro_overlay": report.get("macro_overlay") or {},
        "event_overlay": report.get("event_overlay") or {},
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
    base = {
        "enabled": env_enabled("LLM_TRADE_GATE_ENABLED"),
        "current_leverage": current_leverage,
        "requested_target_leverage": requested,
        "effective_target_leverage": requested,
        "decided_at_utc": utc_now(),
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
    prior_allow = bool(prior.get("allow"))
    prior_limit = abs(float(prior.get("approved_target_leverage") or 0))
    if prior_matches and (not prior_allow or abs(requested) <= prior_limit + 1e-9):
        allowed = prior_allow
        effective = requested if allowed else _safe_target_leverage(current_leverage, requested)
        gated["target_leverage"] = effective
        return gated, {
            **dict(prior),
            **base,
            "status": "cached_approved" if allowed else "cached_rejected",
            "effective_target_leverage": effective,
            "decision_key": key,
        }

    context = build_decision_context(
        report,
        target,
        current_leverage=current_leverage,
        mode=mode,
    )
    provider = decision_provider or request_llm_decision
    try:
        decision = _validated_decision(provider(context))
        min_confidence = float(os.getenv("LLM_MIN_CONFIDENCE", "0.70") or 0.70)
        if not 0 <= min_confidence <= 1:
            raise ValueError("LLM_MIN_CONFIDENCE must be between 0 and 1")
        allowed = bool(decision["allow"] and decision["confidence"] >= min_confidence)
        effective = requested if allowed else _safe_target_leverage(current_leverage, requested)
        gated["target_leverage"] = effective
        return gated, {
            **base,
            **decision,
            "allow": allowed,
            "raw_allow": decision["allow"],
            "minimum_confidence": min_confidence,
            "status": "approved" if allowed else "rejected",
            "decision_key": key,
            "approved_target_leverage": requested if allowed else 0.0,
            "effective_target_leverage": effective,
        }
    except Exception as exc:
        effective = _safe_target_leverage(current_leverage, requested)
        gated["target_leverage"] = effective
        return gated, {
            **base,
            "status": "error_blocked",
            "allow": False,
            "decision_key": key,
            "approved_target_leverage": 0.0,
            "effective_target_leverage": effective,
            "reason": str(exc)[:500],
            "risk_flags": ["llm_unavailable_or_invalid"],
        }
