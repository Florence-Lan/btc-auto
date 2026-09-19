#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import simulate_range_swing as sim


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(description="Maintain the macro BTC candidate in shadow mode.")
    parser.add_argument("--poll-seconds", type=int, default=300)
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
        default=root / "data/paper_trading/macro_candidate_20260917_state.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_20260917_report.json",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_20260917_trades.csv",
    )
    return parser.parse_args()


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=sim.repo_root(), check=True)


def refresh_macro_if_needed(args: argparse.Namespace) -> None:
    age_seconds = float("inf")
    if args.macro_snapshot.exists():
        age_seconds = time.time() - args.macro_snapshot.stat().st_mtime
    if age_seconds < args.refresh_hours * 3600:
        return
    run_checked([
        sys.executable,
        str(sim.repo_root() / "scripts/download_macro_snapshot.py"),
        "--start-utc",
        "2019-10-01T00:00:00Z",
        "--output",
        str(args.macro_snapshot),
        "--force",
    ])


def run_shadow_once(args: argparse.Namespace) -> None:
    run_checked([
        sys.executable,
        str(sim.repo_root() / "scripts/paper_trade_frozen_portfolio.py"),
        "--manifest",
        str(sim.repo_root() / "config/frozen_strategy_candidate_20260917.json"),
        "--state-path",
        str(args.state_path),
        "--report-path",
        str(args.report_path),
        "--trades-path",
        str(args.trades_path),
        "--strategy-modes-override",
        "trend,timeseries_trend",
        "--macro-snapshot",
        str(args.macro_snapshot),
        "--macro-factors",
        "vix,dollar,metals,sentiment",
        "--tiered-drawdown",
        "--soft-drawdown-start-pct",
        "8",
        "--hard-drawdown-stop-pct",
        "15",
        "--drawdown-min-multiplier",
        "0.35",
        "--event-snapshot",
        str(sim.repo_root() / "config/event_risk_template.json"),
    ])


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 60:
        raise ValueError("--poll-seconds must be >= 60")
    if args.refresh_hours <= 0:
        raise ValueError("--refresh-hours must be > 0")
    while True:
        try:
            refresh_macro_if_needed(args)
            run_shadow_once(args)
            print(f"shadow_cycle_complete={datetime.now(timezone.utc).isoformat()}", flush=True)
        except Exception:
            traceback.print_exc()
            if args.once:
                raise
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
