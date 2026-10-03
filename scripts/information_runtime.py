"""Refresh inputs independently of execution/account-risk monitoring."""
from __future__ import annotations

import threading
import time
import traceback

import download_multifactor_snapshot as download
import multifactor
import public_context
import supplemental_market_data as supplemental

_refresh_failures = {}


def refresh_if_needed(factor_path, event_path, supplemental_path, force=False):
    def due(path, seconds):
        return force or not path.exists() or time.time() - path.stat().st_mtime >= seconds

    def run(name, path, fn):
        key = (name, str(path))
        failure = _refresh_failures.get(key, {})
        timestamp = int(time.time() * 1000)
        if not force and timestamp < failure.get("next_retry_at_ms", 0):
            return {"ok": False, "skipped": "retry_wait", **failure}
        try:
            payload = fn()
            _refresh_failures.pop(key, None)
            if payload is None:
                return {"ok": True, "skipped": "not_due"}
            metadata = payload.get("metadata", {})
            states = metadata.get("source_status", payload.get("status", {}))
            errors = metadata.get("errors", {}) or {
                source: state.get("error") for source, state in states.items() if not state.get("ok")}
            return {"ok": not errors, "errors": errors, "source_status": states}
        except Exception as exc:
            failures = int(failure.get("consecutive_failures", 0)) + 1
            state = {"error": f"{type(exc).__name__}: {str(exc)[:220]}",
                     "consecutive_failures": failures,
                     "next_retry_at_ms": int(time.time() * 1000) + download.retry_delay_ms(failures)}
            _refresh_failures[key] = state
            print(f"information_refresh_failed source={name} next_retry_at_ms="
                  f"{state['next_retry_at_ms']} error={state['error']}", flush=True)
            traceback.print_exc()
            return {"ok": False, **state}

    def factors():
        end = int(time.time() * 1000)
        return download.collect(factor_path, end - 180 * multifactor.DAY, end,
                                only_due=True, force=force)

    # Collection failures are independent: a factor/provider error cannot skip
    # news/calendar or supplemental refresh in this worker cycle.
    return {
        "factors": run("factors", factor_path, factors),
        "public_context": run("public_context", event_path, lambda:
                              public_context.collect(event_path, factor_path) if due(event_path, 300) else None),
        "supplemental": run("supplemental", supplemental_path, lambda:
                            supplemental.collect(supplemental_path, force=force)),
    }


def start_worker(factor_path, event_path, supplemental_path):
    def run():
        while True:
            try:
                refresh_if_needed(factor_path, event_path, supplemental_path)
            except Exception:
                traceback.print_exc()
            time.sleep(30)
    worker = threading.Thread(target=run, name="information-refresh", daemon=True)
    worker.start()
    return worker


def paper_command(args):
    import simulate_range_swing as sim
    import sys
    root = sim.repo_root()
    profile = multifactor.load_profile(args.factor_profile)
    return [sys.executable, str(root / "scripts/paper_trade_frozen_portfolio.py"),
            "--manifest", str(root / profile["base_manifest"]),
            "--factor-profile", str(args.factor_profile), "--factor-snapshot", str(args.factor_snapshot),
            "--event-snapshot", str(root / profile["event_snapshot"]),
            "--strategy-modes-override", "trend,timeseries_trend", "--tiered-drawdown",
            "--soft-drawdown-start-pct", "8", "--hard-drawdown-stop-pct", "15",
            "--drawdown-min-multiplier", "0.35", "--state-path", str(args.state_path),
            "--report-path", str(args.report_path), "--trades-path", str(args.trades_path)]
