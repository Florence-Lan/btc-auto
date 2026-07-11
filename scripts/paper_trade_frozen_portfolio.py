#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frozen_strategy
import event_risk
import macro_regime
import portfolio_risk
import simulate_range_swing as sim


def parse_utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(
        description="Track the frozen portfolio prospectively without placing orders.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config/frozen_strategy_20260711.json",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=root / "data/paper_trading/frozen_portfolio_20260711_state.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "data/paper_trading/frozen_portfolio_20260711_report.json",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=root / "data/paper_trading/frozen_portfolio_20260711_trades.csv",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--macro-snapshot", type=Path)
    parser.add_argument("--macro-min-multiplier", type=float, default=0.35)
    parser.add_argument("--macro-block-score", type=float, default=-0.80)
    parser.add_argument(
        "--macro-factors",
        default="equities,vix,dollar,metals,sentiment",
    )
    parser.add_argument("--strategy-modes-override")
    parser.add_argument("--tiered-drawdown", action="store_true")
    parser.add_argument("--soft-drawdown-start-pct", type=float, default=8.0)
    parser.add_argument("--hard-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--drawdown-min-multiplier", type=float, default=0.35)
    parser.add_argument("--event-snapshot", type=Path)
    return parser.parse_args()


def load_or_create_state(
    path: Path,
    manifest: dict[str, Any],
    symbol: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["freeze_id"] != manifest["freeze_id"]:
            raise RuntimeError("Paper state belongs to a different frozen strategy")
        if state["config_sha256"] != manifest["config_sha256"]:
            raise RuntimeError("Paper state config hash mismatch")
        if profile is not None and state.get("profile") != profile:
            raise RuntimeError("Paper state belongs to a different shadow profile")
        return state
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "version": 1,
        "mode": "frozen_portfolio_shadow",
        "places_orders": False,
        "symbol": symbol,
        "freeze_id": manifest["freeze_id"],
        "config_sha256": manifest["config_sha256"],
        "created_at_utc": now,
        "updated_at_utc": None,
        "observations": 0,
        "summary": None,
        "profile": profile,
    }
    save_state(path, state)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    manifest, cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    symbol = sim.normalize_symbol(args.symbol)
    strategy_modes = (
        tuple(part.strip() for part in args.strategy_modes_override.split(",") if part.strip())
        if args.strategy_modes_override
        else cfg.strategy_modes
    )
    allowed_modes = {"trend", "range", "timeseries_trend"}
    if not strategy_modes or any(mode not in allowed_modes for mode in strategy_modes):
        raise ValueError("Invalid --strategy-modes-override")
    macro_factors = tuple(
        part.strip() for part in args.macro_factors.split(",") if part.strip()
    )
    if any(factor not in macro_regime.DEFAULT_WEIGHTS for factor in macro_factors):
        raise ValueError("Invalid --macro-factors")
    profile = {
        "strategy_modes": list(strategy_modes),
        "macro_snapshot": str(args.macro_snapshot.resolve()) if args.macro_snapshot else None,
        "macro_min_multiplier": args.macro_min_multiplier,
        "macro_block_score": args.macro_block_score,
        "macro_factors": list(macro_factors),
        "tiered_drawdown": args.tiered_drawdown,
        "soft_drawdown_start_pct": args.soft_drawdown_start_pct,
        "hard_drawdown_stop_pct": args.hard_drawdown_stop_pct,
        "drawdown_min_multiplier": args.drawdown_min_multiplier,
        "event_snapshot": str(args.event_snapshot.resolve()) if args.event_snapshot else None,
    }
    custom_profile = bool(
        args.strategy_modes_override
        or args.macro_snapshot
        or args.tiered_drawdown
        or args.event_snapshot
    )
    state = load_or_create_state(
        args.state_path,
        manifest,
        symbol,
        profile if custom_profile else None,
    )
    if state["symbol"] != symbol:
        raise RuntimeError("Paper state symbol does not match --symbol")

    evaluation_start_ms = parse_utc_ms(state["created_at_utc"])
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    warmup_days = max(
        45.0,
        cfg.timeseries_slow_ema
        * sim.interval_to_ms(cfg.timeseries_timeframe)
        / sim.MS_PER_DAY
        + 2,
    )
    fetch_start_ms = evaluation_start_ms - int(warmup_days * sim.MS_PER_DAY)
    base_candles = sim.fetch_futures_klines_range(symbol, "5m", fetch_start_ms, now_ms)
    trend_candles = sim.fetch_futures_klines_range(
        symbol,
        cfg.timeseries_timeframe,
        fetch_start_ms,
        now_ms,
    )
    funding = sim.fetch_funding_history(symbol, fetch_start_ms, now_ms)
    cfg = replace(cfg, strategy_modes=strategy_modes)
    sleeve_cfg = replace(cfg, max_drawdown_stop_pct=0.0) if args.tiered_drawdown else cfg
    tactical_modes = tuple(mode for mode in strategy_modes if mode != "timeseries_trend")
    if not tactical_modes:
        raise ValueError("Frozen portfolio must include at least one tactical strategy")
    tactical = sim.simulate(
        base_candles,
        replace(sleeve_cfg, strategy_modes=tactical_modes),
        evaluation_start_ms,
        None,
        funding,
    )
    core = sim.simulate_timeseries_trend(
        trend_candles,
        replace(sleeve_cfg, strategy_modes=("timeseries_trend",)),
        evaluation_start_ms,
        funding,
    )
    sleeves = [tactical, core]
    macro_diagnostics = None
    if args.macro_snapshot:
        snapshot = macro_regime.load_macro_snapshot(args.macro_snapshot)
        sleeves, macro_diagnostics = macro_regime.apply_macro_overlay(
            sleeves,
            snapshot,
            enabled_factors=macro_factors,
            min_multiplier=args.macro_min_multiplier,
            block_score=args.macro_block_score,
        )
    event_diagnostics = None
    if args.event_snapshot:
        events = event_risk.load_event_snapshot(args.event_snapshot)
        sleeves, event_diagnostics = event_risk.apply_event_overlay(sleeves, events)
    if args.tiered_drawdown:
        policy = portfolio_risk.DrawdownRiskPolicy(
            args.soft_drawdown_start_pct,
            args.hard_drawdown_stop_pct,
            args.drawdown_min_multiplier,
        )
        result = portfolio_risk.combine_sleeves_with_drawdown_policy(
            base_candles, sleeves, cfg, policy, evaluation_start_ms
        )
    else:
        result = sim.combine_sleeve_results(
            base_candles,
            sleeves,
            cfg,
            evaluation_start_ms,
        )
    result.update(
        {
            "mode": "frozen_portfolio_shadow",
            "places_orders": False,
            "freeze_id": manifest["freeze_id"],
            "config_sha256": manifest["config_sha256"],
            "paper_inception_utc": state["created_at_utc"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "macro_overlay": macro_diagnostics,
            "shadow_profile": profile,
            "event_overlay": event_diagnostics,
        }
    )
    state["observations"] = int(state["observations"]) + 1
    state["summary"] = result["summary"]
    save_state(args.state_path, state)
    sim.save_json(args.report_path, result)
    sim.save_trades_csv(args.trades_path, result["trades"])
    sim.print_summary(result["summary"])
    print(f"Freeze: {manifest['freeze_id']}")
    print(f"Observations: {state['observations']}")
    print("Orders: disabled")
    return result


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 60:
        raise ValueError("--poll-seconds must be >= 60")
    while True:
        try:
            run_once(args)
        except Exception:
            traceback.print_exc()
            if not args.loop:
                raise
        if not args.loop:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
