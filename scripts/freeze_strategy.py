#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import frozen_strategy
import simulate_range_swing as sim


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(description="Create an immutable risk-controlled strategy manifest.")
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=root / "config/frozen_strategy_20260705.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--profile", choices=["risk_controlled", "active"], default="risk_controlled")
    parser.add_argument("--freeze-id")
    parser.add_argument("--timeseries-target-vol", type=float)
    parser.add_argument("--timeseries-min-ema-spread-pct", type=float)
    parser.add_argument("--timeseries-timeframe")
    parser.add_argument("--timeseries-fast-ema", type=int)
    parser.add_argument("--timeseries-slow-ema", type=int)
    parser.add_argument("--timeseries-vol-lookback-bars", type=int)
    parser.add_argument("--strategy-modes")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = sim.repo_root()
    output = args.output or root / (
        "config/frozen_strategy_active_20260711.json"
        if args.profile == "active"
        else "config/frozen_strategy_20260711.json"
    )
    freeze_id = args.freeze_id or (
        "btc_active_20260711_v1"
        if args.profile == "active"
        else "btc_risk_controlled_20260711_v1"
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")

    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    config = dict(base["config"])
    config.update(
        {
            "leverage": 2.0,
            "risk_per_trade": 0.015,
            "max_drawdown_stop_pct": 12.0,
            "timeseries_target_vol": 0.12,
            "timeseries_max_leverage": 2.0,
            "portfolio_leverage_cap": 2.0,
        }
    )
    if args.profile == "active":
        config.update(
            {
                "risk_per_trade": 0.0075,
                "strategy_modes": ["trend", "range", "timeseries_trend"],
            }
        )
    if args.timeseries_target_vol is not None:
        if args.timeseries_target_vol <= 0:
            raise ValueError("--timeseries-target-vol must be > 0")
        config["timeseries_target_vol"] = args.timeseries_target_vol
    if args.timeseries_min_ema_spread_pct is not None:
        if args.timeseries_min_ema_spread_pct < 0:
            raise ValueError("--timeseries-min-ema-spread-pct must be >= 0")
        config["timeseries_min_ema_spread_pct"] = args.timeseries_min_ema_spread_pct
    if args.timeseries_timeframe is not None:
        sim.interval_to_ms(args.timeseries_timeframe)
        config["timeseries_timeframe"] = args.timeseries_timeframe
    if args.timeseries_fast_ema is not None:
        config["timeseries_fast_ema"] = args.timeseries_fast_ema
    if args.timeseries_slow_ema is not None:
        config["timeseries_slow_ema"] = args.timeseries_slow_ema
    if args.timeseries_vol_lookback_bars is not None:
        config["timeseries_vol_lookback_bars"] = args.timeseries_vol_lookback_bars
    if args.strategy_modes is not None:
        strategy_modes = [item.strip() for item in args.strategy_modes.split(",") if item.strip()]
        allowed_modes = {"trend", "range", "timeseries_trend"}
        if not strategy_modes or any(item not in allowed_modes for item in strategy_modes):
            raise ValueError("--strategy-modes contains an unsupported strategy module")
        config["strategy_modes"] = strategy_modes
    if config["timeseries_fast_ema"] < 1 or config["timeseries_slow_ema"] <= config["timeseries_fast_ema"]:
        raise ValueError("timeseries EMA periods must satisfy 1 <= fast < slow")
    if config["timeseries_vol_lookback_bars"] < 2:
        raise ValueError("timeseries volatility lookback must be >= 2")

    engine_path = root / base["engine_path"]
    snapshot_path = root / base["snapshot_path"]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "freeze_id": freeze_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "based_on_freeze_id": base["freeze_id"],
        "git_head": git_head,
        "engine_path": base["engine_path"],
        "engine_sha256": frozen_strategy.sha256_file(engine_path),
        "snapshot_path": base["snapshot_path"],
        "snapshot_sha256": frozen_strategy.sha256_file(snapshot_path),
        "config_sha256": frozen_strategy.canonical_config_hash(config),
        "validation_targets": {
            "max_drawdown_pct_max": 15.0,
            "profit_factor_min": 1.3,
            "profitable_fold_pct_min": 60.0,
            "double_cost_cagr_pct_min": 0.0,
            "trades_per_year_min": 45.0 if args.profile == "active" else 20.0,
        },
        "config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen manifest: {output}")
    print(f"Config SHA-256: {manifest['config_sha256']}")
    print(f"Engine SHA-256: {manifest['engine_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
