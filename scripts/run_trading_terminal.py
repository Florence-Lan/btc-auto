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
from trading_execution import LIVE_STATE_PATH, SimulationAccount, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNTIME_PATH = DATA / "runtime/trading_terminal.json"
MODE_PATH = DATA / "runtime/execution_mode.json"
EMERGENCY_PATH = DATA / "runtime/emergency_stop.json"
STATE_PATH = DATA / "paper_trading/macro_candidate_v3_state.json"
REPORT_PATH = DATA / "paper_trading/macro_candidate_v3_report.json"
STDOUT_PATH = DATA / "runtime/execution_supervisor_stdout.log"
STDERR_PATH = DATA / "runtime/execution_supervisor_stderr.log"
CANDIDATE_PATH = ROOT / "config/shadow_candidate_macro_20260809.json"
MACRO_PATH = DATA / "snapshots/macro_shadow_latest.json.gz"
SYMBOL = "BTCUSDT"
EXECUTION_CHECK_SECONDS = 30
STRATEGY_BAR_SECONDS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_lines(path: Path, limit: int = 80) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()[-limit:]
    except OSError:
        return []


def execution_point_for_report(
    report: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    point = report.get("execution_target") or summary.get("last_equity_point") or {}
    return point if isinstance(point, dict) else {}


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
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
    def __init__(self, client: BinanceTerminalClient | None = None) -> None:
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.client = client or BinanceTerminalClient()

    def mode(self) -> str:
        payload = read_json(MODE_PATH, {}) or {}
        mode = str(payload.get("mode", "simulation"))
        return mode if mode in {"simulation", "live"} else "simulation"

    def runtime(self) -> dict[str, Any]:
        runtime = read_json(RUNTIME_PATH, {}) or {}
        pid = int(runtime.get("pid", 0) or 0)
        runtime["running"] = process_alive(pid)
        runtime["pid"] = pid or None
        runtime.setdefault("mode", self.mode())
        return runtime

    def emergency(self) -> dict[str, Any] | None:
        payload = read_json(EMERGENCY_PATH)
        return payload if isinstance(payload, dict) and payload.get("active") else None

    def set_mode(self, mode: str, confirm: str = "") -> dict[str, Any]:
        with self.lock:
            if mode not in {"simulation", "live"}:
                raise ValueError("模式必须是 simulation 或 live")
            if self.runtime().get("running"):
                raise RuntimeError("切换模式前必须先暂停自动化")
            if self.emergency():
                raise RuntimeError("急停锁已生效，解除后才能切换模式")
            current = self.mode()
            if mode == "live":
                if confirm != "ENABLE_LIVE_BTCUSDT":
                    raise ValueError("实盘确认词不正确")
                readiness = self.client.validate_live_ready(SYMBOL)
            else:
                readiness = None
                if current == "live" and self.client.configured:
                    snapshot = self.client.account_snapshot(SYMBOL)
                    if snapshot["positions"] or snapshot["open_orders"]:
                        raise RuntimeError("Binance 仍有持仓或挂单，不能直接切换到模拟盘")
            payload = {
                "mode": mode,
                "changed_at_utc": utc_now(),
                "live_confirmed": mode == "live",
            }
            write_json(MODE_PATH, payload)
            return {**payload, "readiness": readiness}

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.emergency():
                raise RuntimeError("急停锁已生效，必须先解除急停")
            current = self.runtime()
            if current.get("running"):
                return current
            mode = self.mode()
            if mode == "live":
                self.client.validate_live_ready(SYMBOL)
            STDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            stdout = STDOUT_PATH.open("a", encoding="utf-8")
            stderr = STDERR_PATH.open("a", encoding="utf-8")
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_execution_supervisor.py"),
                    "--mode",
                    mode,
                ],
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
                "mode": mode,
                "places_orders": mode == "live",
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
                    raise RuntimeError(f"进程 {pid} 在停止后仍存活")
            runtime.update({"pid": None, "running": False, "stopped_at_utc": utc_now()})
            write_json(RUNTIME_PATH, runtime)
            return runtime

    def run_once(self) -> dict[str, Any]:
        with self.lock:
            if self.emergency():
                raise RuntimeError("急停锁已生效，不能执行策略")
            if self.runtime().get("running"):
                raise RuntimeError("自动化正在运行，无需再执行单次循环")
            mode = self.mode()
            if mode == "live":
                self.client.validate_live_ready(SYMBOL)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_execution_supervisor.py"),
                    "--mode",
                    mode,
                    "--once",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1600:] or "单次执行失败")
            return {
                "completed_at_utc": utc_now(),
                "mode": mode,
                "output": completed.stdout[-1600:],
            }

    def reset_simulation(self, initial_balance: float, confirm: str) -> dict[str, Any]:
        with self.lock:
            if self.mode() != "simulation":
                raise RuntimeError("只有模拟盘可以重置初始金额")
            if self.runtime().get("running"):
                raise RuntimeError("重置模拟账户前必须先暂停自动化")
            if self.emergency():
                raise RuntimeError("请先解除急停锁")
            if confirm != "RESET_SIMULATION":
                raise ValueError("重置模拟账户确认词不正确")
            state = SimulationAccount().reset(initial_balance)
            return {
                "mode": "simulation",
                "initial_balance": state["initial_balance"],
                "reset_at_utc": state["updated_at_utc"],
            }

    def emergency_stop(self, reason: str) -> dict[str, Any]:
        with self.lock:
            mode = self.mode()
            stopped = self.stop()
            exchange_actions: dict[str, Any] = {"cancelled_orders": False, "flattened": False}
            execution_error = None
            if mode == "live":
                try:
                    self.client.cancel_all_orders(SYMBOL)
                    exchange_actions["cancelled_orders"] = True
                    flatten = self.client.flatten_position(
                        SYMBOL,
                        f"btcauto-emergency-{int(time.time())}"[:36],
                    )
                    exchange_actions["flattened"] = flatten is not None
                except Exception as exc:
                    execution_error = str(exc)
            payload = {
                "active": True,
                "triggered_at_utc": utc_now(),
                "reason": reason or "terminal emergency stop",
                "mode": mode,
                "exchange_actions": exchange_actions,
                "execution_error": execution_error,
            }
            write_json(EMERGENCY_PATH, payload)
            if execution_error:
                raise RuntimeError(
                    f"自动化已停止，但 Binance 撤单/平仓失败：{execution_error}。请立即登录 Binance 检查。"
                )
            return {"runtime": stopped, "emergency": payload}

    def reset_emergency(self) -> dict[str, Any]:
        with self.lock:
            if EMERGENCY_PATH.exists():
                EMERGENCY_PATH.unlink()
            return {"active": False, "reset_at_utc": utc_now()}

    def _macro_status(
        self,
        report: dict[str, Any],
        state: dict[str, Any],
        candidate: dict[str, Any],
        market_time_ms: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        macro = report.get("macro_overlay") or {}
        profile = state.get("profile") or report.get("shadow_profile") or {}
        current: dict[str, Any] = {}
        if MACRO_PATH.exists():
            try:
                snapshot = macro_regime.load_macro_snapshot(MACRO_PATH)
                factors = tuple(
                    profile.get("macro_factors")
                    or candidate.get("macro", {}).get("factors", [])
                )
                decision = macro_regime.macro_decision_at(
                    snapshot, market_time_ms, enabled_factors=factors
                )
                current = {
                    "score": decision.score,
                    "risk_multiplier": decision.risk_multiplier,
                    "allowed": decision.allowed,
                    "available_factors": decision.available_factors,
                    "contributions": decision.contributions,
                    "asof_ms": decision.asof_ms,
                }
            except (OSError, ValueError):
                pass
        return macro, current

    def status(self) -> dict[str, Any]:
        mode = self.mode()
        runtime = self.runtime()
        emergency = self.emergency()
        state = read_json(STATE_PATH, {}) or {}
        report = read_json(REPORT_PATH, {}) or {}
        candidate = read_json(CANDIDATE_PATH, {}) or {}
        summary = report.get("summary") or state.get("summary") or {}
        summary_point = summary.get("last_equity_point") or {}
        execution_point = execution_point_for_report(report, summary)
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
        macro_age = (
            max(0.0, datetime.now().timestamp() - MACRO_PATH.stat().st_mtime)
            if MACRO_PATH.exists() else None
        )
        try:
            mark_price = self.client.mark_price(SYMBOL)
            market_error = None
        except Exception as exc:
            mark_price = float(execution_point.get("price") or 0) or None
            market_error = str(exc)
        market_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        macro, macro_current = self._macro_status(
            report,
            state,
            candidate,
            int(execution_point.get("time_ms") or market_time_ms),
        )
        exchange = self.client.snapshot(SYMBOL) if mode == "live" else {
            "configured": self.client.configured,
            "connected": market_error is None,
            "environment": "live",
            "read_only": True,
            "live_trading_enabled": self.client.live_trading_enabled,
            "error": market_error,
        }
        execution_state: dict[str, Any] = {}
        if mode == "simulation":
            execution_snapshot = SimulationAccount().snapshot(mark_price)
            execution_state = execution_snapshot["state"]
            account = execution_snapshot["account"]
            positions = execution_snapshot["positions"]
            open_orders = execution_snapshot["open_orders"]
            recent_trades = execution_snapshot["recent_trades"]
            equity_curve = execution_snapshot["equity_curve"]
            drawdown = execution_snapshot["max_drawdown_pct"]
        elif exchange.get("connected"):
            live_state = read_json(LIVE_STATE_PATH, {}) or {}
            execution_state = live_state
            margin_balance = float(exchange["account"].get("margin_balance") or 0)
            initial_live_equity = float(live_state.get("initial_equity") or 0)
            account = {
                **exchange["account"],
                "realized_return_pct": (
                    (margin_balance / initial_live_equity - 1) * 100
                    if initial_live_equity > 0 else None
                ),
                "source": "Binance live",
            }
            positions = exchange.get("positions", [])
            open_orders = exchange.get("open_orders", [])
            try:
                recent_trades = self.client.recent_trades(SYMBOL)
            except Exception:
                recent_trades = []
            equity_curve = live_state.get("equity_curve", [])
            drawdown = live_state.get("max_drawdown_pct", 0.0)
        else:
            account = {
                "wallet_balance": None,
                "available_balance": None,
                "unrealized_pnl": None,
                "realized_return_pct": None,
                "margin_ratio_pct": None,
                "source": "Binance live unavailable",
            }
            positions, open_orders, recent_trades, equity_curve = [], [], [], []
            drawdown = summary.get("max_drawdown_pct", 0.0)
        logs = [line.rstrip() for line in tail_lines(STDOUT_PATH, 70)]
        errors = [line.rstrip() for line in tail_lines(STDERR_PATH, 30)]
        max_notional = (
            float(os.getenv("LIVE_MAX_NOTIONAL_USDT", "0") or 0)
            if mode == "live" else float(os.getenv("SIM_MAX_NOTIONAL_USDT", "0") or 0)
        )
        account_equity = float(account.get("margin_balance") or account.get("wallet_balance") or 0)
        signal_equity = float(execution_point.get("equity") or 0)
        signal_price = float(execution_point.get("price") or 0)
        target_signed_qty = float(execution_point.get("signed_qty") or 0)
        target_leverage = (
            target_signed_qty * signal_price / signal_equity
            if signal_equity > 0 and signal_price > 0 else 0.0
        )
        target_leverage = max(-2.0, min(2.0, target_leverage))
        leverage_limit = (
            int(os.getenv("LIVE_LEVERAGE", "1") or 1)
            if mode == "live"
            else min(float(os.getenv("SIM_MAX_LEVERAGE", "2") or 2), 2.0)
        )
        natural_cap = account_equity * leverage_limit
        effective_cap = min(natural_cap, max_notional) if max_notional > 0 else natural_cap
        target_notional = max(-effective_cap, min(effective_cap, target_leverage * account_equity))
        signal_time_ms = int(execution_point.get("time_ms") or 0)
        signal_age_seconds = (
            max(0.0, (market_time_ms - signal_time_ms) / 1000)
            if signal_time_ms > 0 else None
        )
        sleeve_names = list(candidate.get("strategy_modes", []))
        sleeve_signals = []
        for index, sleeve in enumerate(report.get("sleeves") or []):
            point = sleeve.get("last_equity_point") or {}
            signed_qty = float(point.get("signed_qty") or 0)
            price = float(point.get("price") or 0)
            equity = float(point.get("equity") or 0)
            direction = "LONG" if signed_qty > 0 else "SHORT" if signed_qty < 0 else "FLAT"
            sleeve_signals.append({
                "name": sleeve_names[index] if index < len(sleeve_names) else f"sleeve_{index + 1}",
                "status": "POSITION" if direction != "FLAT" else "MONITORING",
                "direction": direction,
                "target_leverage": signed_qty * price / equity if equity > 0 and price > 0 else 0.0,
                "signed_qty": signed_qty,
                "price": price or None,
                "equity": equity or None,
                "trades": int(sleeve.get("trades") or 0),
                "return_pct": sleeve.get("total_return_pct"),
                "max_drawdown_pct": sleeve.get("max_drawdown_pct"),
                "win_rate_pct": sleeve.get("win_rate_pct"),
                "profit_factor": sleeve.get("profit_factor"),
            })
        position_qty = float(execution_state.get("position_qty") or 0) if mode == "simulation" else (
            float(positions[0].get("signed_quantity") or 0) if positions else 0.0
        )
        entry_price = float(execution_state.get("entry_price") or 0) if mode == "simulation" else (
            float(positions[0].get("entry_price") or 0) if positions else 0.0
        )
        execution_fills = execution_state.get("fills") or execution_state.get("trades") or []
        report_trades = list(report.get("trades") or [])
        closed_strategy_trades = [
            trade for trade in report_trades if str(trade.get("exit_reason")) != "end"
        ]
        open_strategy_positions = [
            trade for trade in report_trades if str(trade.get("exit_reason")) == "end"
        ]
        account_details = {
            "initial_balance": execution_state.get("initial_balance") or execution_state.get("initial_equity"),
            "wallet_balance": account.get("wallet_balance"),
            "margin_balance": account.get("margin_balance"),
            "available_balance": account.get("available_balance"),
            "realized_pnl": execution_state.get("realized_pnl") if mode == "simulation" else None,
            "unrealized_pnl": account.get("unrealized_pnl"),
            "fees_paid": execution_state.get("fees_paid") if mode == "simulation" else None,
            "position_qty": position_qty,
            "entry_price": entry_price or None,
            "position_notional": abs(position_qty * float(mark_price or 0)),
            "fill_count_total": (
                int(execution_state.get("fill_count_total") or len(execution_fills))
                if mode == "simulation"
                else len(recent_trades)
            ),
            "closed_strategy_trade_count": len(closed_strategy_trades),
            "open_strategy_position_count": len(open_strategy_positions),
            "execution_inception_utc": execution_state.get("created_at_utc"),
            "shadow_inception_utc": report.get("paper_inception_utc"),
        }
        return {
            "server_time_utc": utc_now(),
            "mode": mode.upper(),
            "execution": {
                "places_orders": mode == "live",
                "live_enabled": self.client.live_trading_enabled,
                "runtime": runtime,
                "emergency": emergency,
                "heartbeat_age_seconds": heartbeat_age,
                "max_notional_usdt": max_notional,
                "leverage": leverage_limit,
                "simulation_initial_balance": (
                    float(execution_snapshot["state"]["initial_balance"])
                    if mode == "simulation" else None
                ),
                "last_cycle_at_utc": execution_state.get("updated_at_utc"),
                "last_fill": execution_fills[-1] if execution_fills else None,
                "entry_guard": execution_state.get("entry_guard"),
                "check_interval_seconds": EXECUTION_CHECK_SECONDS,
                "strategy_bar_seconds": STRATEGY_BAR_SECONDS,
            },
            "exchange": exchange,
            "market": {
                "symbol": SYMBOL,
                "mark_price": mark_price,
                "data_time_ms": market_time_ms,
                "source": "Binance mainnet realtime",
                "macro_snapshot_age_seconds": macro_age,
                "signal_price": signal_price or None,
                "signal_age_seconds": signal_age_seconds,
            },
            "account": account,
            "account_details": account_details,
            "risk": {
                "drawdown_pct": drawdown,
                "drawdown_multiplier": summary_point.get("drawdown_risk_multiplier", 1.0),
                "soft_limit_pct": 8.0,
                "hard_limit_pct": 15.0,
                "portfolio_leverage_cap": 2.0,
                "macro": macro,
                "macro_current": macro_current,
                "profile": state.get("profile") or report.get("shadow_profile") or {},
            },
            "positions": positions,
            "open_orders": open_orders,
            "recent_trades": recent_trades,
            "equity_curve": equity_curve,
            "strategy": {
                "candidate_id": candidate.get("candidate_id"),
                "strategy_modes": candidate.get("strategy_modes", []),
                "macro_factors": candidate.get("macro", {}).get("factors", []),
                "observations": state.get("observations", 0),
                "updated_at_utc": updated,
                "signal_time_ms": execution_point.get("time_ms"),
                "position_id": execution_point.get("position_id"),
                "origin_signal_time_ms": execution_point.get("origin_signal_time_ms"),
                "origin_entry_price": execution_point.get("origin_entry_price"),
                "target_signed_qty": target_signed_qty,
                "target_leverage": target_leverage,
                "target_notional": target_notional,
                "sleeves": sleeve_signals,
            },
            "logs": logs,
            "errors": errors,
        }


BINANCE = BinanceTerminalClient()
CONTROLLER = TerminalController(BINANCE)


class TerminalHandler(SimpleHTTPRequestHandler):
    server_version = "BTCTradingTerminal/2.0"

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
        if urlparse(self.path).path != "/api/terminal/control":
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
            elif action == "set_mode":
                result = CONTROLLER.set_mode(
                    str(payload.get("mode", "")), str(payload.get("confirm", ""))
                )
            elif action == "reset_simulation":
                result = CONTROLLER.reset_simulation(
                    float(payload.get("initial_balance", 0)),
                    str(payload.get("confirm", "")),
                )
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
