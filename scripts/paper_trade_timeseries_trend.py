#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import simulate_range_swing as sim


def build_config(args: argparse.Namespace) -> sim.StrategyConfig:
    original_argv = sys.argv
    try:
        sys.argv = [
            "simulate_range_swing.py",
            "--strategy-modes",
            "timeseries_trend",
            "--initial-equity",
            str(args.initial_equity),
            "--max-drawdown-stop-pct",
            str(args.max_drawdown_stop_pct),
            "--timeseries-target-vol",
            str(args.target_vol),
            "--timeseries-max-leverage",
            str(args.max_leverage),
            "--no-market-context-enabled",
        ]
        return sim.config_from_args(sim.parse_args())
    finally:
        sys.argv = original_argv


def load_state(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    created = args.start_utc or datetime.now(timezone.utc).isoformat()
    return {
        "version": 1,
        "mode": "timeseries_trend_shadow",
        "symbol": sim.normalize_symbol(args.symbol),
        "created_at_utc": created,
        "updated_at_utc": None,
        "initial_equity": args.initial_equity,
        "generation": 1,
        "halted": False,
        "summary": None,
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_utc_ms(value: str) -> int:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 6h time-series trend strategy in shadow mode only.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--max-drawdown-stop-pct", type=float, default=12.0)
    parser.add_argument("--start-utc", help="Optional shadow inception time; defaults to now.")
    parser.add_argument("--resume-after-drawdown", action="store_true")
    parser.add_argument("--loop", action="store_true", help="Keep refreshing the shadow report without placing orders.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--state-path",
        type=Path,
        default=sim.repo_root() / "data/paper_trading/timeseries_trend_shadow_state.json",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=sim.repo_root() / "data/paper_trading/timeseries_trend_shadow_trades.csv",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=sim.repo_root() / "data/paper_trading/timeseries_trend_shadow_report.json",
    )
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> int:
    state = load_state(args.state_path, args)
    symbol = sim.normalize_symbol(args.symbol)
    if state.get("symbol", symbol) != symbol:
        raise RuntimeError("Shadow state symbol does not match --symbol")
    if args.resume_after_drawdown:
        previous = state.get("summary") or {}
        state["initial_equity"] = float(previous.get("final_equity", state["initial_equity"]))
        state["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["generation"] = int(state.get("generation", 1)) + 1
        state["halted"] = False
    if state.get("halted"):
        raise RuntimeError("Shadow strategy is halted; use --resume-after-drawdown to start a new generation")

    args.initial_equity = float(state["initial_equity"])
    cfg = build_config(args)
    start_ms = parse_utc_ms(str(state["created_at_utc"]))
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    warmup_days = max(cfg.timeseries_slow_ema, cfg.timeseries_vol_lookback_bars) / 4 + 2
    start_fetch_ms = start_ms - int(warmup_days * sim.MS_PER_DAY)
    candles = sim.fetch_futures_klines_range(symbol, cfg.timeseries_timeframe, start_fetch_ms, now_ms)
    funding = sim.fetch_funding_history(symbol, start_fetch_ms, now_ms)
    if len(candles) <= max(cfg.timeseries_slow_ema, cfg.timeseries_vol_lookback_bars) + 1:
        raise RuntimeError("Not enough closed 6h candles for the shadow warmup")

    result = sim.simulate_timeseries_trend(candles, cfg, start_ms, funding)
    result["mode"] = "shadow"
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["summary"] = result["summary"]
    state["halted"] = result["summary"]["max_drawdown_pct"] >= args.max_drawdown_stop_pct
    save_state(args.state_path, state)
    sim.save_json(args.report_path, result)
    sim.save_trades_csv(args.trades_path, result["trades"])
    sim.print_summary(result["summary"])
    print(f"Shadow generation: {state['generation']}")
    print(f"Halted: {state['halted']}")
    print(f"State: {args.state_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 60:
        raise ValueError("--poll-seconds must be >= 60")
    while True:
        run_once(args)
        if not args.loop:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
