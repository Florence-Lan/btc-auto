"""Run a synthetic Codex review without loading .env, accounts, or trading code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import llm_trade_gate as gate


def check() -> dict:
    now = datetime.now(timezone.utc)
    result = {"diagnostic_only": True, "places_orders": False,
              "reads_account_ledgers": False, "loads_dotenv": False,
              "checked_at_utc": now.isoformat(), "sandbox": "read-only"}
    started = time.monotonic()
    try:
        settings = gate.review_provider_configuration()
        if settings["provider"] != "codex":
            raise ValueError("This check requires LLM_PROVIDER=codex")
        result["review_configuration"] = settings
        if not settings["executable"]:
            raise RuntimeError("Codex CLI not found")
        version = subprocess.run([settings["executable"], "--version"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=10,
                                 check=False, env=gate._codex_review_environment())
        result["cli_version"] = version.stdout.strip()[:120] if version.returncode == 0 else "unavailable"
        # This deliberately incomplete, high-risk fixture must be rejected.
        # It is never an execution target, signal, cached decision, or account.
        fixture = {
            "symbol": "BTCUSDT", "mode": "simulation", "diagnostic_only": True,
            "fixture_description": "Synthetic health check. No real trade or account exists.",
            "decision_clock": {"decision_asof_ms": int(now.timestamp() * 1000)},
            "admission_policy": {"historical_performance_required": True},
            "requested_target_leverage": 1.0, "components": [],
            "strategy_review_context": {}, "multifactor_overlay": {},
            "risk_diagnostics": {"critical_context_missing": True,
                                 "account_hard_drawdown_stop_triggered": True},
        }
        decision = gate._validated_decision(gate.request_codex_decision(fixture))
        result["decision"] = decision
        result["structured_response_valid"] = True
        result["expected_fixture_rejection"] = decision["allow"] is False
        result["ok"] = result["expected_fixture_rejection"]
        if not result["ok"]:
            result["error"] = "Synthetic missing-context/hard-stop fixture was unexpectedly approved"
    except Exception as exc:
        result.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:500])
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-bin", help="Verify this CLI path rather than the diagnostic process PATH")
    parser.add_argument("--model", help="Override only this diagnostic process's review model")
    parser.add_argument("--reasoning-effort", choices=sorted(gate.CODEX_REASONING_EFFORTS))
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--output", type=Path, help="Optional standalone diagnostic JSON evidence")
    args = parser.parse_args(argv)
    overrides = {"LLM_PROVIDER": "codex"}
    if args.codex_bin is not None:
        overrides["CODEX_BIN"] = args.codex_bin
    if args.model is not None:
        overrides["LLM_CODEX_MODEL"] = args.model
    if args.reasoning_effort is not None:
        overrides["LLM_CODEX_REASONING_EFFORT"] = args.reasoning_effort
    if args.timeout_seconds is not None:
        overrides["LLM_CODEX_TIMEOUT_SECONDS"] = str(args.timeout_seconds)
    prior = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        result = check()
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
