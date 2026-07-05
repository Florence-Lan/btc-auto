from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import simulate_range_swing as sim


TUPLE_FIELDS = {
    "confirm_timeframes",
    "market_context_periods",
    "strategy_modes",
    "trend_confirm_timeframes",
    "trend_stretch_filter_timeframes",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_frozen_strategy(
    manifest_path: Path,
    *,
    verify_engine: bool = True,
) -> tuple[dict[str, Any], sim.StrategyConfig]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    actual_config_hash = canonical_config_hash(config)
    if actual_config_hash != manifest["config_sha256"]:
        raise RuntimeError("Frozen strategy config hash mismatch")

    root = sim.repo_root()
    if verify_engine:
        engine_path = root / manifest["engine_path"]
        if sha256_file(engine_path) != manifest["engine_sha256"]:
            raise RuntimeError("Strategy engine changed after the freeze")

    normalized = {
        key: tuple(value) if key in TUPLE_FIELDS else value
        for key, value in config.items()
    }
    return manifest, sim.StrategyConfig(**normalized)


def verify_snapshot(manifest: dict[str, Any], snapshot_path: Path) -> None:
    if sha256_file(snapshot_path.resolve()) != manifest["snapshot_sha256"]:
        raise RuntimeError("Validation snapshot hash mismatch")
