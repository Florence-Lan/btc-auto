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
import execution_targets
import execution_portfolio
import forward_macro
import macro_regime
import multifactor
import portfolio_risk
import public_context
import simulate_range_swing as sim
import timeseries_execution


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
    parser.add_argument("--factor-profile", type=Path)
    parser.add_argument("--factor-snapshot", type=Path)
    parser.add_argument("--research-profile", type=Path, help="Frozen reentry candidate; isolated paper state only")
    parser.add_argument("--market-cache", type=Path, help="Incremental verified public market-data cache")
    parser.add_argument("--asof-ms", type=int, help="Validated simulation decision cutoff")
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


def annotate_open_position_fractions(sleeve_results: list[dict[str, Any]]) -> None:
    """Attach the still-open fraction to each synthetic end-of-window trade."""
    for sleeve in sleeve_results:
        curve = sleeve.get("equity_curve") or []
        point = curve[-1] if curve else {}
        signed_qty = float(point.get("signed_qty") or 0.0)
        if abs(signed_qty) <= 1e-12:
            continue
        side = "long" if signed_qty > 0 else "short"
        candidates = [
            trade
            for trade in sleeve.get("trades", [])
            if str(trade.get("exit_reason")) == "end"
            and str(trade.get("side")) == side
        ]
        if not candidates:
            raise RuntimeError(
                "Open sleeve position is missing its synthetic end-of-window trade"
            )
        trade = candidates[-1]
        initial_qty = float(trade.get("initial_qty") or 0.0)
        if initial_qty <= 0:
            raise RuntimeError("Synthetic end-of-window trade has invalid quantity")
        trade["_open_qty_fraction"] = sim.clamp(
            abs(signed_qty) / initial_qty,
            0.0,
            1.0,
        )


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    manifest, cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    research_profile_path = getattr(args, "research_profile", None)
    research_profile = None
    if research_profile_path:
        import reentry_candidate
        research_profile = reentry_candidate.load_candidate(research_profile_path, args.manifest)
        if not args.tiered_drawdown:
            raise ValueError("Research profile requires the corrected tiered portfolio ledger")
    factor_profile_path = getattr(args, "factor_profile", None)
    factor_snapshot_path = getattr(args, "factor_snapshot", None)
    if bool(factor_profile_path) != bool(factor_snapshot_path):
        raise ValueError("--factor-profile and --factor-snapshot must be supplied together")
    factor_profile = multifactor.load_profile(factor_profile_path) if factor_profile_path else None
    if factor_profile and factor_profile["availability_mode"] != "first_seen":
        raise ValueError("Forward shadow requires first_seen factor availability")
    if factor_profile and args.macro_snapshot:
        raise ValueError("Use either legacy macro or multifactor overlay, not both")
    public = None
    if factor_profile and factor_profile.get("public_context_enabled"):
        if not args.event_snapshot:
            raise ValueError("Public context requires --event-snapshot")
        public = json.loads(args.event_snapshot.read_text(encoding="utf-8"))
        if "coverage_checks" not in public:
            raise ValueError("Public context snapshot is missing source coverage history")
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
        or factor_profile
        or research_profile
    )
    if factor_profile:
        profile["factor_profile_sha256"] = multifactor.profile_hash(factor_profile)
        profile["factor_snapshot"] = str(factor_snapshot_path.resolve())
        profile["factor_engine_sha256"] = frozen_strategy.sha256_file(Path(multifactor.__file__))
        profile["portfolio_risk_sha256"] = frozen_strategy.sha256_file(Path(portfolio_risk.__file__))
        if public is not None:
            profile["public_context_sha256"] = frozen_strategy.sha256_file(Path(public_context.__file__))
            profile["event_risk_sha256"] = frozen_strategy.sha256_file(Path(event_risk.__file__))
    causal_hourly = args.tiered_drawdown and research_profile is None
    startup_hourly = timeseries_execution.startup_enabled(factor_profile)
    if startup_hourly and not causal_hourly:
        raise ValueError("Hourly startup requires the corrected non-research hourly execution model")
    if causal_hourly:
        profile["hourly_execution_model"] = (timeseries_execution.STARTUP_MODEL if startup_hourly
                                              else timeseries_execution.MODEL)
        profile["hourly_execution_sha256"] = frozen_strategy.sha256_file(Path(timeseries_execution.__file__))
        profile["execution_portfolio_sha256"] = frozen_strategy.sha256_file(Path(execution_portfolio.__file__))
        if startup_hourly:
            import research_signal_engine
            profile["hourly_startup_enabled"] = True
            profile["hourly_signal_replay_sha256"] = frozen_strategy.sha256_file(Path(research_signal_engine.__file__))
    if research_profile:
        profile["research_profile_sha256"] = frozen_strategy.sha256_file(research_profile_path)
        profile["research_candidate_id"] = research_profile["candidate_id"]
    state = load_or_create_state(
        args.state_path,
        manifest,
        symbol,
        profile if custom_profile else None,
    )
    if state["symbol"] != symbol:
        raise RuntimeError("Paper state symbol does not match --symbol")

    evaluation_start_ms = parse_utc_ms(state["created_at_utc"])
    now_ms = getattr(args, "asof_ms", None)
    if now_ms is None:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if now_ms <= 0:
        raise ValueError("Decision cutoff must be a positive timestamp")
    warmup_days = max(
        45.0,
        cfg.timeseries_slow_ema
        * sim.interval_to_ms(cfg.timeseries_timeframe)
        / sim.MS_PER_DAY
        + 2,
    )
    fetch_start_ms = evaluation_start_ms - int(warmup_days * sim.MS_PER_DAY)
    market_data = None
    if getattr(args, "market_cache", None):
        import market_data_runtime
        market_data = market_data_runtime.load_market_data(
            symbol, fetch_start_ms, now_ms, cache_dir=args.market_cache,
            intervals=tuple(dict.fromkeys(("5m", cfg.timeseries_timeframe))))
        base_candles = market_data.candles["5m"]
        trend_candles = market_data.candles[cfg.timeseries_timeframe]
        funding = market_data.funding
    else:
        base_candles = sim.fetch_futures_klines_range(symbol, "5m", fetch_start_ms, now_ms)
        trend_candles = sim.fetch_futures_klines_range(
            symbol, cfg.timeseries_timeframe, fetch_start_ms, now_ms)
        funding = sim.fetch_funding_history(symbol, fetch_start_ms, now_ms)
    cfg = replace(cfg, strategy_modes=strategy_modes)
    sleeve_cfg = replace(cfg, max_drawdown_stop_pct=0.0) if args.tiered_drawdown else cfg
    tactical_modes = tuple(mode for mode in strategy_modes if mode != "timeseries_trend")
    if not tactical_modes:
        raise ValueError("Frozen portfolio must include at least one tactical strategy")
    if research_profile:
        sleeves = reentry_candidate.build_sleeves(
            {"5m": base_candles, "1h": trend_candles}, funding, sleeve_cfg,
            evaluation_start_ms, research_profile,
        )
    else:
        tactical = sim.simulate(
            base_candles,
            replace(sleeve_cfg, strategy_modes=tactical_modes),
            evaluation_start_ms,
            None,
            funding,
        )
        core_cfg = replace(sleeve_cfg, strategy_modes=("timeseries_trend",))
        if causal_hourly:
            opening_time = trend_candles[-1].close_time_ms + 1
            step = sim.interval_to_ms(cfg.timeseries_timeframe)
            if opening_time != now_ms // step * step:
                raise ValueError("Closed hourly history is stale at the decision boundary")
            opening = timeseries_execution.opening_from_base(base_candles, opening_time, now_ms)
            if opening is None and market_data is not None:
                opening = market_data.opening(cfg.timeseries_timeframe, opening_time)
            if opening is None:
                opening = timeseries_execution.fetch_opening(symbol, cfg.timeseries_timeframe, opening_time, now_ms)
            core = timeseries_execution.build_sleeve(trend_candles, core_cfg, evaluation_start_ms,
                funding, opening=opening, asof_ms=now_ms,
                activation_ms=evaluation_start_ms if startup_hourly else None)
        else:
            core = sim.simulate_timeseries_trend(trend_candles, core_cfg, evaluation_start_ms, funding)
        sleeves = [tactical, core]
    annotate_open_position_fractions(sleeves)
    execution_targets.prepare_sleeves(sleeves)
    macro_diagnostics = None
    factor_diagnostics = None
    if factor_profile:
        factor_snapshot = multifactor.load_snapshot(factor_snapshot_path)
        sleeves, factor_diagnostics = multifactor.apply_overlay(sleeves, factor_snapshot, factor_profile, public)
        factor_diagnostics["current"] = {
            side: multifactor.asdict(multifactor.decision_at(factor_snapshot, now_ms, side, factor_profile, public))
            for side in ("long", "short")
        }
        factor_diagnostics["data_metadata"] = factor_snapshot.metadata
    if args.macro_snapshot:
        snapshot = macro_regime.load_macro_snapshot(args.macro_snapshot)
        sleeves, macro_diagnostics = forward_macro.apply_overlay(
            sleeves,
            args.macro_snapshot,
            args.state_path.with_suffix(".macro_decisions.json"),
            now_ms,
            factors=macro_factors,
            min_multiplier=args.macro_min_multiplier,
            block_score=args.macro_block_score,
        )
    event_diagnostics = None
    if args.event_snapshot:
        events = event_risk.load_event_snapshot(args.event_snapshot)
        sleeves, event_diagnostics = event_risk.apply_event_overlay(sleeves, events)
        if factor_profile:
            event_diagnostics["coverage_status"] = "loaded" if events else "not_configured"
        if public is not None:
            sleeves, blocked = public_context.apply_entry_coverage(sleeves, public)
            healthy, missing = public_context.health_at(public, now_ms)
            event_diagnostics.update({
                "coverage_status": "healthy" if healthy else "degraded",
                "coverage_blocked_entries": blocked, "missing_sources": missing,
                "current": multifactor.asdict(event_risk.event_decision_at(events, now_ms)),
                "source_status": public.get("metadata", {}).get("source_status", {}),
                "news_items_archived": len(public.get("news", [])),
                "calendar_versions_archived": len(public.get("calendars", [])),
                "rate_expectation": public_context.expectation_at(public, now_ms),
                "policy_surprises": public.get("policy_surprises", []),
                "economic_consensus_surprises": public.get("metadata", {}).get("economic_consensus_surprises"),
            })
    if args.tiered_drawdown:
        policy = portfolio_risk.DrawdownRiskPolicy(
            args.soft_drawdown_start_pct,
            args.hard_drawdown_stop_pct,
            args.drawdown_min_multiplier,
        )
        result = execution_portfolio.combine(
            base_candles,
            sleeves,
            cfg,
            policy,
            evaluation_start_ms,
            include_execution_target=True,
            decision_open_prices=(timeseries_execution.decision_open_prices(base_candles, sleeves)
                                  if causal_hourly else None),
        )
        # Position sizing uses the open mark, without fictitious terminal liquidation costs.
        result["execution_target"] = execution_targets.current_target(base_candles, sleeves, result, cfg)
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
            "multifactor_overlay": factor_diagnostics,
            "research_only": bool(factor_profile or research_profile),
            "research_candidate": research_profile["candidate_id"] if research_profile else None,
            "execution_entry_context": {
                "factor_profile": str(factor_profile_path.resolve()) if factor_profile_path else None,
                "factor_profile_sha256": multifactor.profile_hash(factor_profile) if factor_profile else None,
                "factor_snapshot": str(factor_snapshot_path.resolve()) if factor_snapshot_path else None,
                "event_snapshot": str(args.event_snapshot.resolve()) if args.event_snapshot else None,
                "macro_snapshot": str(args.macro_snapshot.resolve()) if args.macro_snapshot else None,
                "macro_factors": list(macro_factors), "macro_min_multiplier": args.macro_min_multiplier,
                "macro_block_score": args.macro_block_score,
            },
            "execution_model": core["execution_timing"]["model"] if causal_hourly else "legacy_research",
            "hourly_startup": core.get("hourly_startup") if causal_hourly else None,
            "market_data": market_data.diagnostics if market_data is not None else None,
            "decision_asof_ms": now_ms,
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
