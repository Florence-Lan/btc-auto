"""Refresh inputs independently of execution/account-risk monitoring."""
from __future__ import annotations

import threading
import time
import traceback

import download_multifactor_snapshot as download
import multifactor
import public_context
import supplemental_market_data as supplemental


def refresh_if_needed(factor_path, event_path, supplemental_path, force=False):
    def due(path, seconds):
        return force or not path.exists() or time.time() - path.stat().st_mtime >= seconds
    if due(factor_path, 3600):
        end = int(time.time() * 1000)
        download.collect(factor_path, end - 180 * multifactor.DAY, end)
    if due(event_path, 300):
        public_context.collect(event_path, factor_path)
    supplemental.collect(supplemental_path, force=force)


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
