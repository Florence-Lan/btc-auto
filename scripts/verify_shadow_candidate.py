#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import simulate_range_swing as sim


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(root: Path, relative_path: str, expected_hash: str) -> None:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Candidate dependency is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise RuntimeError(
            f"Candidate dependency hash mismatch: {relative_path} "
            f"expected={expected_hash} actual={actual}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an immutable BTC shadow candidate.")
    parser.add_argument(
        "--candidate",
        type=Path,
        default=sim.repo_root() / "config/shadow_candidate_macro_20260720.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = sim.repo_root()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if candidate.get("live_orders_allowed") is not False:
        raise RuntimeError("Shadow candidate must explicitly disable live orders")
    for path, expected in candidate["code_sha256"].items():
        verify_file(root, path, expected)
    verify_file(
        root,
        candidate["macro"]["snapshot_path"],
        candidate["macro"]["snapshot_sha256"],
    )
    verify_file(
        root,
        candidate["validation"]["report_path"],
        candidate["validation"]["report_sha256"],
    )
    print(json.dumps({
        "candidate_id": candidate["candidate_id"],
        "status": candidate["status"],
        "shadow_eligible": candidate["validation"]["shadow_eligible"],
        "live_orders_allowed": candidate["live_orders_allowed"],
        "verified": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
