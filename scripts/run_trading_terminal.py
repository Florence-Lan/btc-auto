#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import macro_regime
from binance_terminal_client import BinanceTerminalClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNTIME_PATH = DATA / "runtime/trading_terminal.json"
EMERGENCY_PATH = DATA / "runtime/emergency_stop.json"
STATE_PATH = DATA / "paper_trading/macro_candidate_state.json"
REPORT_PATH = DATA / "paper_trading/macro_candidate_report.json"
STDOUT_PATH = DATA / "paper_trading/macro_candidate_supervisor_stdout.log"
STDERR_PATH = DATA / "paper_trading/macro_candidate_supervisor_stderr.log"
CANDIDATE_PATH = ROOT / "config/shadow_candidate_macro_20260711.json"
VALIDATION_PATH = DATA / "validation/candidate_portfolio_20260711.json"
MACRO_PATH = DATA / "snapshots/macro_shadow_latest.json.gz"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def tail_lines(path: Path, limit: int = 80) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()[-limit:]
    except OSError:
        return []


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class TerminalController:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None

    def runtime(self) -> dict[str, Any]:
        runtime = read_json(RUNTIME_PATH, {}) or {}
        pid = int(runtime.get("pid", 0) or 0)
        runtime["running"] = process_alive(pid)
        runtime["pid"] = pid or None
        return runtime

    def emergency(self) -> dict[str, Any] | None:
        return read_json(EMERGENCY_PATH)

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.emergency():
                raise RuntimeError("急停锁已生效，必须先解除急停")
            current = self.runtime()
            if current.get("running"):
                return current
            STDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            stdout = STDOUT_PATH.open("a", encoding="utf-8")
            stderr = STDERR_PATH.open("a", encoding="utf-8")
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/run_macro_candidate_shadow.py")],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                creationflags=creationflags,
            )
            runtime = {
                "pid": self.process.pid,
                "running": True,
                "started_at_utc": utc_now(),
                "mode": "shadow",
                "places_orders": False,
            }
            write_json(RUNTIME_PATH, runtime)
            return runtime

    def stop(self) -> dict[str, Any]:
        with self.lock:
            runtime = self.runtime()
            pid = runtime.get("pid")
            if runtime.get("running") and pid:
                if os.name == "nt":
                    completed = subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if completed.returncode != 0 and process_alive(pid):
                        raise RuntimeError(
                            completed.stderr.strip()
                            or completed.stdout.strip()
                            or f"无法终止进程 {pid}"
                        )
                else:
                    os.kill(pid, signal.SIGTERM)
                for _ in range(50):
                    if not process_alive(pid):
                        break
                    time.sleep(0.1)
                if process_alive(pid):
                    raise RuntimeError(f"进程 {pid} 在急停后仍存活")
            runtime.update({"pid": None, "running": False, "stopped_at_utc": utc_now()})
            write_json(RUNTIME_PATH, runtime)
            return runtime

    def run_once(self) -> dict[str, Any]:
        with self.lock:
            if self.emergency():
                raise RuntimeError("急停锁已生效，不能执行策略")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_macro_candidate_shadow.py"), "--once"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1200:] or "单次执行失败")
            return {"completed_at_utc": utc_now(), "output": completed.stdout[-1200:]}

    def emergency_stop(self, reason: str) -> dict[str, Any]:
        stopped = self.stop()
        payload = {
            "active": True,
            "triggered_at_utc": utc_now(),
            "reason": reason or "terminal emergency stop",
            "places_orders": False,
        }
        write_json(EMERGENCY_PATH, payload)
        return {"runtime": stopped, "emergency": payload}

    def reset_emergency(self) -> dict[str, Any]:
        with self.lock:
            if EMERGENCY_PATH.exists():
                EMERGENCY_PATH.unlink()
            return {"active": False, "reset_at_utc": utc_now()}

    def status(self) -> dict[str, Any]:
        runtime = self.runtime()
        emergency = self.emergency()
        state = read_json(STATE_PATH, {}) or {}
        report = read_json(REPORT_PATH, {}) or {}
        candidate = read_json(CANDIDATE_PATH, {}) or {}
        validation = read_json(VALIDATION_PATH, {}) or {}
        summary = report.get("summary") or state.get("summary") or {}
        last_point = summary.get("last_equity_point") or {}
        updated = state.get("updated_at_utc")
        heartbeat_age = None
        if updated:
            try:
                heartbeat_age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(updated.replace("Z", "+00:00"))
                ).total_seconds()
            except ValueError:
                pass
        macro_age = None
        if MACRO_PATH.exists():
            macro_age = max(0.0, datetime.now().timestamp() - MACRO_PATH.stat().st_mtime)
        signed_qty = float(last_point.get("signed_qty", 0) or 0)
        positions = []
        if abs(signed_qty) > 1e-12:
            positions.append({
                "symbol": "BTCUSDT",
                "side": "LONG" if signed_qty > 0 else "SHORT",
                "quantity": abs(signed_qty),
                "mark_price": last_point.get("price"),
                "unrealized_pnl": 0.0,
                "source": "shadow reconstruction",
            })
        trades = report.get("trades") or []
        recent_trades = list(reversed(trades[-20:]))
        macro = report.get("macro_overlay") or {}
        profile = state.get("profile") or report.get("shadow_profile") or {}
        macro_current: dict[str, Any] = {}
        if MACRO_PATH.exists():
            try:
                snapshot = macro_regime.load_macro_snapshot(MACRO_PATH)
                factors = tuple(profile.get("macro_factors") or candidate.get("macro", {}).get("factors", []))
                decision = macro_regime.macro_decision_at(
                    snapshot,
                    int(last_point.get("time_ms") or datetime.now(timezone.utc).timestamp() * 1000),
                    enabled_factors=factors,
                )
                macro_current = {
                    "score": decision.score,
                    "risk_multiplier": decision.risk_multiplier,
                    "allowed": decision.allowed,
                    "available_factors": decision.available_factors,
                    "contributions": decision.contributions,
                    "asof_ms": decision.asof_ms,
                }
            except (OSError, ValueError):
                macro_current = {}
        logs = [line.rstrip() for line in tail_lines(STDOUT_PATH, 70)]
        errors = [line.rstrip() for line in tail_lines(STDERR_PATH, 30)]
        exchange = BINANCE.snapshot("BTCUSDT")
        account = {
            "wallet_balance": summary.get("final_equity", 100.0),
            "available_balance": summary.get("final_equity", 100.0),
            "unrealized_pnl": 0.0,
            "realized_return_pct": summary.get("total_return_pct", 0.0),
            "margin_ratio_pct": 0.0,
            "source": "shadow",
        }
        if exchange.get("connected"):
            account.update(exchange["account"])
            account["realized_return_pct"] = summary.get("total_return_pct", 0.0)
            account["source"] = f"Binance {exchange['environment']} read-only"
            positions = exchange.get("positions", positions)
            open_orders = exchange.get("open_orders", [])
        else:
            open_orders = []
        return {
            "server_time_utc": utc_now(),
            "mode": "SHADOW",
            "execution": {
                "places_orders": False,
                "live_enabled": False,
                "testnet_enabled": False,
                "runtime": runtime,
                "emergency": emergency,
                "heartbeat_age_seconds": heartbeat_age,
            },
            "exchange": exchange,
            "market": {
                "symbol": "BTCUSDT",
                "mark_price": last_point.get("price"),
                "data_time_ms": last_point.get("time_ms"),
                "macro_snapshot_age_seconds": macro_age,
            },
            "account": account,
            "risk": {
                "drawdown_pct": summary.get("max_drawdown_pct", 0.0),
                "drawdown_multiplier": last_point.get("drawdown_risk_multiplier", 1.0),
                "soft_limit_pct": 8.0,
                "hard_limit_pct": 15.0,
                "portfolio_leverage_cap": 2.0,
                "macro": macro,
                "macro_current": macro_current,
                "profile": profile,
            },
            "positions": positions,
            "open_orders": open_orders,
            "recent_trades": recent_trades,
            "equity_curve": report.get("equity_curve", [])[-1000:],
            "strategy": {
                "candidate_id": candidate.get("candidate_id"),
                "status": candidate.get("status"),
                "strategy_modes": candidate.get("strategy_modes", []),
                "macro_factors": candidate.get("macro", {}).get("factors", []),
                "observations": state.get("observations", 0),
                "updated_at_utc": updated,
            },
            "validation": {
                "shadow_eligible": validation.get("shadow_eligible"),
                "candidate_pass": validation.get("candidate_pass"),
                "normal": validation.get("normal", {}).get("summary", {}),
                "double_cost": validation.get("double_cost", {}).get("summary", {}),
                "quarterly": validation.get("normal", {}).get("quarterly", {}),
            },
            "logs": logs,
            "errors": errors,
        }


CONTROLLER = TerminalController()
BINANCE = BinanceTerminalClient()


class TerminalHandler(SimpleHTTPRequestHandler):
    server_version = "BTCTradingTerminal/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/terminal/status":
            self.send_json(CONTROLLER.status())
            return
        if path == "/api/terminal/health":
            self.send_json({"ok": True, "time_utc": utc_now()})
            return
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/terminal/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/terminal/control":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if self.headers.get("X-Terminal-Action") != "1":
            self.send_json({"error": "missing action header"}, HTTPStatus.FORBIDDEN)
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 16_384)
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = payload.get("action")
            if action == "start":
                result = CONTROLLER.start()
            elif action == "pause":
                result = CONTROLLER.stop()
            elif action == "run_once":
                result = CONTROLLER.run_once()
            elif action == "emergency_stop":
                if payload.get("confirm") != "EMERGENCY_STOP":
                    raise ValueError("急停确认词不正确")
                result = CONTROLLER.emergency_stop(str(payload.get("reason", "")))
            elif action == "reset_emergency":
                if payload.get("confirm") != "RESET":
                    raise ValueError("解除确认词不正确")
                result = CONTROLLER.reset_emergency()
            else:
                raise ValueError("unknown control action")
            self.send_json({"ok": True, "result": result})
        except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.CONFLICT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local BTC automated trading terminal.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Trading terminal may only bind to localhost")
    server = ThreadingHTTPServer((args.host, args.port), TerminalHandler)
    print(f"BTC trading terminal: http://{args.host}:{args.port}/terminal/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
