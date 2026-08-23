#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import run_macro_candidate_shadow as strategy_supervisor
import simulate_range_swing as sim
from binance_terminal_client import BinanceTerminalClient
from trading_execution import (
    LIVE_STATE_PATH,
    SIMULATION_STATE_PATH,
    execute_report,
    read_json,
)


BAR_INTERVAL_MS = 5 * 60 * 1000
DEFAULT_POLL_SECONDS = 30
BAR_SETTLE_DELAY_SECONDS = 3


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(
        description="Run the BTC strategy against simulation or guarded Binance live execution.",
    )
    parser.add_argument("--mode", choices=("simulation", "live"), required=True)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--bar-settle-delay-seconds",
        type=int,
        default=BAR_SETTLE_DELAY_SECONDS,
    )
    parser.add_argument("--refresh-hours", type=float, default=12.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--macro-snapshot",
        type=Path,
        default=root / "data/snapshots/macro_shadow_latest.json.gz",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_v3_state.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_v3_report.json",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_v3_trades.csv",
    )
    return parser.parse_args()


def strategy_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        poll_seconds=args.poll_seconds,
        refresh_hours=args.refresh_hours,
        once=True,
        macro_snapshot=args.macro_snapshot,
        state_path=args.state_path,
        report_path=args.report_path,
        trades_path=args.trades_path,
    )


def run_cycle(args: argparse.Namespace, client: BinanceTerminalClient) -> dict[str, object]:
    shadow_args = strategy_args(args)
    strategy_supervisor.refresh_macro_if_needed(shadow_args)
    strategy_supervisor.run_shadow_once(shadow_args)
    report = json.loads(args.report_path.read_text(encoding="utf-8"))
    result = execute_report(args.mode, report, client)
    point = (report.get("summary") or {}).get("last_equity_point") or {}
    result["signal_time_ms"] = int(point.get("time_ms") or 0)
    print(
        f"execution_cycle_complete={datetime.now(timezone.utc).isoformat()} "
        f"mode={args.mode} target_leverage={float(result['target_leverage']):.6f} "
        f"llm_gate={(result.get('llm_trade_gate') or {}).get('status', 'unknown')}",
        flush=True,
    )
    return result


def last_processed_signal_ms(mode: str) -> int:
    path = SIMULATION_STATE_PATH if mode == "simulation" else LIVE_STATE_PATH
    state = read_json(path, {}) or {}
    key = "last_signal_time_ms" if mode == "simulation" else "signal_time_ms"
    return int(state.get(key) or 0)


def due_closed_bar_open_ms(
    server_time_ms: int,
    last_processed_ms: int,
    settle_delay_seconds: int = BAR_SETTLE_DELAY_SECONDS,
) -> int | None:
    current_bar_open_ms = server_time_ms // BAR_INTERVAL_MS * BAR_INTERVAL_MS
    phase_ms = server_time_ms - current_bar_open_ms
    if phase_ms < settle_delay_seconds * 1000:
        return None
    latest_closed_bar_open_ms = current_bar_open_ms - BAR_INTERVAL_MS
    return (
        latest_closed_bar_open_ms
        if latest_closed_bar_open_ms > last_processed_ms
        else None
    )


def seconds_until_next_check(
    server_time_ms: int,
    poll_seconds: int,
    settle_delay_seconds: int = BAR_SETTLE_DELAY_SECONDS,
) -> float:
    current_bar_open_ms = server_time_ms // BAR_INTERVAL_MS * BAR_INTERVAL_MS
    phase_ms = server_time_ms - current_bar_open_ms
    if phase_ms < settle_delay_seconds * 1000:
        next_due_ms = current_bar_open_ms + settle_delay_seconds * 1000
    else:
        next_due_ms = current_bar_open_ms + BAR_INTERVAL_MS + settle_delay_seconds * 1000
    until_due = max(0.25, (next_due_ms - server_time_ms) / 1000)
    return min(float(poll_seconds), until_due)


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be >= 5")
    if args.bar_settle_delay_seconds < 1 or args.bar_settle_delay_seconds > 30:
        raise ValueError("--bar-settle-delay-seconds must be between 1 and 30")
    if args.refresh_hours <= 0:
        raise ValueError("--refresh-hours must be > 0")
    client = BinanceTerminalClient()
    if args.mode == "live":
        client.validate_live_ready()
    if args.once:
        run_cycle(args, client)
        return 0
    print(
        f"scheduler_started mode={args.mode} check_seconds={args.poll_seconds} "
        f"bar_seconds={BAR_INTERVAL_MS // 1000} settle_delay_seconds={args.bar_settle_delay_seconds}",
        flush=True,
    )
    while True:
        loop_started = time.time()
        server_time_ms = int(loop_started * 1000)
        try:
            server_time_ms = client.server_time_ms()
            last_processed = last_processed_signal_ms(args.mode)
            due_bar = due_closed_bar_open_ms(
                server_time_ms,
                last_processed,
                args.bar_settle_delay_seconds,
            )
            if due_bar is not None:
                result = run_cycle(args, client)
                actual_signal = int(result.get("signal_time_ms") or 0)
                if actual_signal < due_bar:
                    raise RuntimeError(
                        f"Strategy report signal {actual_signal} is older than due bar {due_bar}"
                    )
        except Exception:
            traceback.print_exc()
            if args.mode == "live":
                raise
        estimated_server_time_ms = server_time_ms + int((time.time() - loop_started) * 1000)
        time.sleep(seconds_until_next_check(
            estimated_server_time_ms,
            args.poll_seconds,
            args.bar_settle_delay_seconds,
        ))


if __name__ == "__main__":
    raise SystemExit(main())
