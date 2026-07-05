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
        default=root / "config/frozen_strategy_20260705.json",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=root / "data/paper_trading/frozen_portfolio_state.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "data/paper_trading/frozen_portfolio_report.json",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=root / "data/paper_trading/frozen_portfolio_trades.csv",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def load_or_create_state(
    path: Path,
    manifest: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["freeze_id"] != manifest["freeze_id"]:
            raise RuntimeError("Paper state belongs to a different frozen strategy")
        if state["config_sha256"] != manifest["config_sha256"]:
            raise RuntimeError("Paper state config hash mismatch")
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
    state = load_or_create_state(args.state_path, manifest, symbol)
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
    tactical = sim.simulate(
        base_candles,
        replace(cfg, strategy_modes=("trend",)),
        evaluation_start_ms,
        None,
        funding,
    )
    core = sim.simulate_timeseries_trend(
        trend_candles,
        replace(cfg, strategy_modes=("timeseries_trend",)),
        evaluation_start_ms,
        funding,
    )
    result = sim.combine_sleeve_results(
        base_candles,
        [tactical, core],
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
