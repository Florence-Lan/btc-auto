"""Run the expanded candidate in a separate, order-disabled forward account."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import multifactor
import active_strategy
import simulate_range_swing as sim


def record_status(root, candidate, status, stage, returncode):
    sim.save_json(root / f"data/paper_trading/{candidate}_status.json", {
        "candidate_id": candidate, "status": status, "stage": stage,
        "returncode": returncode, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "places_orders": False,
    })


def parse_args(argv=None):
    root = sim.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--profile", type=Path, help="Defaults to the terminal's selected strategy")
    parser.add_argument("--snapshot", type=Path, default=root / "data/snapshots/multifactor_latest.json.gz")
    parser.add_argument("--research-profile", type=Path, help="Optional isolated reentry candidate")
    args = parser.parse_args(argv)
    if args.poll_seconds < 60:
        parser.error("--poll-seconds must be >= 60")
    args.profile = args.profile or active_strategy.candidate_path()
    return args


def main():
    root = sim.repo_root()
    args = parse_args()
    profile = multifactor.load_profile(args.profile)
    # This command observes the selected rules independently; it must never
    # overwrite the terminal's active forward state/report.
    candidate = profile["candidate_id"] + "_shadow"
    if args.research_profile:
        import reentry_candidate
        research = reentry_candidate.load_candidate(args.research_profile, root / profile["base_manifest"])
        candidate += "_" + research["candidate_id"]
    if not candidate.replace("_", "").isalnum():
        raise ValueError("Invalid candidate_id")
    while True:
        if not args.snapshot.exists() or time.time() - args.snapshot.stat().st_mtime >= 3600:
            # Partial outages are persisted as explicit missing coverage; paper entries fail closed.
            completed = subprocess.run([
                sys.executable, str(root / "scripts/download_multifactor_snapshot.py"),
                "--output", str(args.snapshot),
            ], cwd=root)
            if completed.returncode:
                record_status(root, candidate, "degraded", "factor_collection", completed.returncode)
                print("Factor refresh failed; skipping this paper cycle and backing off.", flush=True)
                if args.once:
                    return completed.returncode
                time.sleep(max(args.poll_seconds, 300))
                continue
        event_path = root / profile["event_snapshot"]
        if profile.get("public_context_enabled"):
            if not event_path.exists() or time.time() - event_path.stat().st_mtime >= 300:
                completed = subprocess.run([
                    sys.executable, str(root / "scripts/public_context.py"),
                    "--output", str(event_path), "--factor-snapshot", str(args.snapshot),
                ], cwd=root)
                if completed.returncode:
                    record_status(root, candidate, "degraded", "public_collection", completed.returncode)
                    print("Public context refresh failed; skipping this paper cycle and backing off.", flush=True)
                    if args.once:
                        return completed.returncode
                    time.sleep(max(args.poll_seconds, 300))
                    continue
        command = [
            sys.executable, str(root / "scripts/paper_trade_frozen_portfolio.py"),
            "--manifest", str(root / profile["base_manifest"]),
            "--factor-profile", str(args.profile), "--factor-snapshot", str(args.snapshot),
            "--strategy-modes-override", "trend,timeseries_trend", "--tiered-drawdown",
            "--event-snapshot", str(event_path),
        ]
        if args.research_profile:
            command.extend(["--research-profile", str(args.research_profile)])
        for flag, suffix in (("state", "json"), ("report", "json"), ("trades", "csv")):
            command.extend([f"--{flag}-path", str(root / f"data/paper_trading/{candidate}_{flag}.{suffix}")])
        completed = subprocess.run(command, cwd=root)
        record_status(root, candidate, "ok" if completed.returncode == 0 else "degraded", "paper_evaluation", completed.returncode)
        if args.once:
            return completed.returncode
        time.sleep(args.poll_seconds if completed.returncode == 0 else max(args.poll_seconds, 300))


if __name__ == "__main__":
    raise SystemExit(main())
