#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import simulate_range_swing as sim


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config(args: argparse.Namespace) -> sim.StrategyConfig:
    return sim.StrategyConfig(
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
        taker_fee=args.taker_fee,
        maker_fee=args.maker_fee,
        bb_period=36,
        bb_std=2.0,
        rsi_period=14,
        atr_period=14,
        adx_period=14,
        ema_fast=48,
        ema_slow=144,
        max_adx=29.0,
        max_bandwidth=0.040,
        max_ema_spread=0.018,
        side_mode="auto",
        drift_lookback_bars=144,
        countertrend_drift_limit_pct=0.0015,
        long_rsi=39.0,
        short_rsi=67.0,
        reclaim_buffer_pct=0.00012,
        min_band_excursion_pct=0.0,
        entry_pullback_atr=0.85,
        entry_edge_ratio=0.18,
        min_pullback_pct=0.0008,
        max_wait_bars=2,
        stop_atr=1.15,
        min_stop_pct=0.0055,
        min_reward_risk=1.30,
        tp1_close_ratio=0.35,
        tp2_close_ratio=0.30,
        break_even_buffer_pct=0.00035,
        trail_atr=1.2,
        max_hold_bars=42,
        cooldown_bars=4,
        max_drawdown_stop_pct=args.max_drawdown_stop_pct,
        high_adx_drift_threshold=26.0,
        min_high_adx_drift_pct=0.0012,
        min_signal_score=0.50,
        adaptive_risk_enabled=True,
        min_risk_multiplier=0.95,
        max_risk_multiplier=1.10,
        maintenance_margin_pct=args.maintenance_margin_pct,
        liquidation_fee_pct=args.liquidation_fee_pct,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        depth_impact_bps=args.depth_impact_bps,
        depth_impact_exponent=args.depth_impact_exponent,
        min_depth_quote=args.min_depth_quote,
    )


def state_template(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "version": 1,
        "mode": "paper",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": None,
        "symbol": sim.normalize_symbol(args.symbol),
        "interval": args.interval,
        "equity": args.initial_equity,
        "peak_equity": args.initial_equity,
        "max_drawdown_pct": 0.0,
        "last_processed_open_time_ms": None,
        "cooldown_until_index": 0,
        "processed_bars": 0,
        "pending": None,
        "position": None,
        "position_equity_base": args.initial_equity,
        "position_signal_reason": "",
        "closed_trades": 0,
        "liquidations": 0,
        "events": [],
    }


def load_state(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = state_template(args)
    save_state(path, state)
    return state


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def pending_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[sim.PendingEntry]:
    if payload is None:
        return None
    return sim.PendingEntry(**payload)


def position_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[sim.Position]:
    if payload is None:
        return None
    return sim.Position(**payload)


def trade_to_row(trade: sim.Trade) -> Dict[str, Any]:
    return asdict(trade)


def append_trade(path: Path, trade: sim.Trade) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = trade_to_row(trade)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def add_event(state: Dict[str, Any], event: str, payload: Dict[str, Any]) -> None:
    item = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    state.setdefault("events", []).append(item)
    state["events"] = state["events"][-100:]


def mark_equity(equity: float, position: Optional[sim.Position], candle: sim.Candle, cfg: sim.StrategyConfig) -> float:
    if position is None:
        return equity
    side_dir = sim.direction(position.side)
    unrealized = (candle.close - position.entry_price) * position.qty * side_dir
    close_fee_estimate = abs(candle.close * position.qty) * cfg.taker_fee
    return equity + position.realized_pnl + unrealized - position.fees_paid - close_fee_estimate


def close_position_if_needed(
    state: Dict[str, Any],
    position: sim.Position,
    candle: sim.Candle,
    paper_index: int,
    atr_value: Optional[float],
    cfg: sim.StrategyConfig,
    trades_path: Path,
) -> Optional[sim.Position]:
    exit_reason = ""

    if position.side == "long":
        liquidation_hit = candle.low <= position.liquidation_price
        stop_hit = candle.low <= position.stop_price
        tp1_hit = (not position.tp1_hit) and candle.high >= position.tp1
        tp2_hit = (not position.tp2_hit) and candle.high >= position.tp2
        tp3_hit = candle.high >= position.tp3
        max_hold_hit = paper_index - position.entry_index >= cfg.max_hold_bars

        if liquidation_hit:
            fill = sim.execution_price(position.liquidation_price, position.side, False, position.qty, candle, cfg)
            sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "liquidation")
            exit_reason = "liquidation"
        elif stop_hit:
            fill = sim.execution_price(position.stop_price, position.side, False, position.qty, candle, cfg)
            sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "stop")
            exit_reason = "stop"
        else:
            if tp1_hit:
                qty = position.initial_qty * cfg.tp1_close_ratio
                fill = sim.execution_price(position.tp1, position.side, False, qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, qty, cfg.taker_fee, "tp1")
                position.tp1_hit = True
                position.stop_price = max(position.stop_price, position.entry_price * (1 + cfg.break_even_buffer_pct))
            if tp2_hit and position.qty > 0:
                qty = position.initial_qty * cfg.tp2_close_ratio
                fill = sim.execution_price(position.tp2, position.side, False, qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, qty, cfg.taker_fee, "tp2")
                position.tp2_hit = True
            if tp3_hit and position.qty > 0:
                fill = sim.execution_price(position.tp3, position.side, False, position.qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "tp3")
                exit_reason = "tp3"
            elif max_hold_hit and position.qty > 0:
                fill = sim.execution_price(candle.close, position.side, False, position.qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "max_hold")
                exit_reason = "max_hold"
    else:
        liquidation_hit = candle.high >= position.liquidation_price
        stop_hit = candle.high >= position.stop_price
        tp1_hit = (not position.tp1_hit) and candle.low <= position.tp1
        tp2_hit = (not position.tp2_hit) and candle.low <= position.tp2
        tp3_hit = candle.low <= position.tp3
        max_hold_hit = paper_index - position.entry_index >= cfg.max_hold_bars

        if liquidation_hit:
            fill = sim.execution_price(position.liquidation_price, position.side, False, position.qty, candle, cfg)
            sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "liquidation")
            exit_reason = "liquidation"
        elif stop_hit:
            fill = sim.execution_price(position.stop_price, position.side, False, position.qty, candle, cfg)
            sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "stop")
            exit_reason = "stop"
        else:
            if tp1_hit:
                qty = position.initial_qty * cfg.tp1_close_ratio
                fill = sim.execution_price(position.tp1, position.side, False, qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, qty, cfg.taker_fee, "tp1")
                position.tp1_hit = True
                position.stop_price = min(position.stop_price, position.entry_price * (1 - cfg.break_even_buffer_pct))
            if tp2_hit and position.qty > 0:
                qty = position.initial_qty * cfg.tp2_close_ratio
                fill = sim.execution_price(position.tp2, position.side, False, qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, qty, cfg.taker_fee, "tp2")
                position.tp2_hit = True
            if tp3_hit and position.qty > 0:
                fill = sim.execution_price(position.tp3, position.side, False, position.qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "tp3")
                exit_reason = "tp3"
            elif max_hold_hit and position.qty > 0:
                fill = sim.execution_price(candle.close, position.side, False, position.qty, candle, cfg)
                sim.exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "max_hold")
                exit_reason = "max_hold"

    if position.qty > 0 and not exit_reason:
        sim.update_trailing_stop(position, candle, atr_value, cfg)

    if position.qty <= 1e-12:
        trade = sim.close_trade_record(
            position,
            candle,
            float(state["position_equity_base"]),
            exit_reason or position.exit_notes or "exit",
            str(state.get("position_signal_reason", "")),
        )
        trade.bars_held = paper_index - position.entry_index
        state["equity"] = float(state["equity"]) + trade.net_pnl
        state["closed_trades"] = int(state.get("closed_trades", 0)) + 1
        if trade.exit_reason == "liquidation":
            state["liquidations"] = int(state.get("liquidations", 0)) + 1
        state["cooldown_until_index"] = paper_index + cfg.cooldown_bars
        append_trade(trades_path, trade)
        add_event(
            state,
            "close",
            {
                "side": trade.side,
                "entry_time_utc": trade.entry_time_utc,
                "exit_time_utc": trade.exit_time_utc,
                "net_pnl": trade.net_pnl,
                "exit_reason": trade.exit_reason,
            },
        )
        return None
    return position


def fill_pending_if_possible(
    state: Dict[str, Any],
    pending: sim.PendingEntry,
    candle: sim.Candle,
    paper_index: int,
    cfg: sim.StrategyConfig,
) -> Optional[sim.Position]:
    if not (candle.low <= pending.target_price <= candle.high):
        return None

    equity = max(float(state["equity"]), 0.0)
    risk_capital = equity * cfg.risk_per_trade * pending.risk_multiplier
    raw_risk_per_unit = abs(pending.target_price - pending.stop_price)
    rough_qty_by_risk = risk_capital / raw_risk_per_unit if raw_risk_per_unit > 0 else 0.0
    rough_qty_by_leverage = (equity * cfg.leverage) / pending.target_price
    rough_qty = min(rough_qty_by_risk, rough_qty_by_leverage)
    entry_price = sim.execution_price(pending.target_price, pending.side, True, rough_qty, candle, cfg)
    risk_per_unit = abs(entry_price - pending.stop_price)
    qty_by_risk = risk_capital / risk_per_unit if risk_per_unit > 0 else 0.0
    qty_by_leverage = (equity * cfg.leverage) / entry_price
    qty = min(qty_by_risk, qty_by_leverage)
    if qty <= 0:
        return None

    entry_price = sim.execution_price(pending.target_price, pending.side, True, qty, candle, cfg)
    entry_fee = abs(entry_price * qty) * cfg.maker_fee
    liq_price = sim.liquidation_price(entry_price, pending.side, cfg.leverage, cfg)
    state["position_equity_base"] = float(state["equity"])
    state["position_signal_reason"] = pending.signal_reason
    add_event(
        state,
        "open",
        {
            "side": pending.side,
            "time_utc": candle.open_time_utc,
            "entry_price": entry_price,
            "qty": qty,
            "liquidation_price": liq_price,
            "risk_multiplier": pending.risk_multiplier,
        },
    )
    return sim.Position(
        side=pending.side,
        entry_index=paper_index,
        entry_time_utc=candle.open_time_utc,
        entry_price=entry_price,
        qty=qty,
        initial_qty=qty,
        stop_price=pending.stop_price,
        tp1=pending.tp1,
        tp2=pending.tp2,
        tp3=pending.tp3,
        liquidation_price=liq_price,
        best_price=candle.high if pending.side == "long" else candle.low,
        fees_paid=entry_fee,
    )


def process_candle(
    state: Dict[str, Any],
    candles: Sequence[sim.Candle],
    index: int,
    cfg: sim.StrategyConfig,
    trades_path: Path,
) -> None:
    candle = candles[index]
    ind = sim.indicators(candles[: index + 1], cfg)
    paper_index = int(state.get("processed_bars", 0))
    atr_value = ind["atr"][index]
    position = position_from_dict(state.get("position"))
    pending = pending_from_dict(state.get("pending"))

    if position is not None:
        position = close_position_if_needed(state, position, candle, paper_index, atr_value, cfg, trades_path)

    marked = mark_equity(float(state["equity"]), position, candle, cfg)
    state["peak_equity"] = max(float(state.get("peak_equity", marked)), marked)
    peak = float(state["peak_equity"])
    drawdown_pct = ((peak - marked) / peak * 100) if peak else 0.0
    state["max_drawdown_pct"] = max(float(state.get("max_drawdown_pct", 0.0)), drawdown_pct)

    if position is None:
        trading_halted = cfg.max_drawdown_stop_pct > 0 and float(state["max_drawdown_pct"]) >= cfg.max_drawdown_stop_pct
        if trading_halted:
            pending = None

        if not trading_halted and pending is not None and paper_index > pending.expires_index:
            add_event(state, "pending_expired", {"side": pending.side, "target_price": pending.target_price})
            pending = None

        if not trading_halted and pending is not None and paper_index >= pending.created_index:
            filled_position = fill_pending_if_possible(state, pending, candle, paper_index, cfg)
            if filled_position is not None:
                position = filled_position
                pending = None

        if (
            not trading_halted
            and position is None
            and pending is None
            and paper_index >= int(state.get("cooldown_until_index", 0))
        ):
            signal_index = index - 1
            signal = sim.signal_for_index(candles[: index + 1], ind, signal_index, cfg)
            if signal is not None:
                side, reason, score = signal
                pending = sim.build_pending_entry(candles[: index + 1], ind, signal_index, index, side, reason, score, cfg)
                if pending is not None:
                    pending.signal_index = max(paper_index - 1, 0)
                    pending.created_index = paper_index
                    pending.expires_index = paper_index + cfg.max_wait_bars
                    add_event(
                        state,
                        "pending_created",
                        {
                            "side": pending.side,
                            "signal_time_utc": candles[signal_index].open_time_utc,
                            "target_price": pending.target_price,
                            "stop_price": pending.stop_price,
                            "signal_score": pending.signal_score,
                            "risk_multiplier": pending.risk_multiplier,
                        },
                    )

    state["position"] = asdict(position) if position is not None else None
    state["pending"] = asdict(pending) if pending is not None else None
    state["last_processed_open_time_ms"] = candle.open_time_ms
    state["processed_bars"] = int(state.get("processed_bars", 0)) + 1
    state["last_marked_equity"] = marked
    state["last_drawdown_pct"] = drawdown_pct
    state["last_price"] = candle.close


def closed_candles(candles: Sequence[sim.Candle]) -> List[sim.Candle]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return [candle for candle in candles if candle.close_time_ms <= now_ms]


def run_once(args: argparse.Namespace, cfg: sim.StrategyConfig, state: Dict[str, Any], state_path: Path, trades_path: Path) -> None:
    symbol = sim.normalize_symbol(args.symbol)
    fetch_days = max(args.fetch_days, 2.0)
    candles = closed_candles(sim.fetch_futures_klines(symbol, args.interval, fetch_days))
    warmup = max(cfg.bb_period, cfg.rsi_period + 1, cfg.atr_period, cfg.adx_period * 2, cfg.ema_slow) + 2
    if len(candles) <= warmup:
        raise RuntimeError(f"not enough candles for paper trading warmup: {len(candles)}")

    last_processed = state.get("last_processed_open_time_ms")
    if last_processed is None:
        start_index = max(warmup, len(candles) - max(args.backfill_bars, 1))
    else:
        start_index = next((i for i, candle in enumerate(candles) if candle.open_time_ms > int(last_processed)), len(candles))

    processed = 0
    for index in range(start_index, len(candles)):
        process_candle(state, candles, index, cfg, trades_path)
        processed += 1

    save_state(state_path, state)
    print_status(state, processed, state_path, trades_path)


def print_status(state: Dict[str, Any], processed: int, state_path: Path, trades_path: Path) -> None:
    position = state.get("position")
    pending = state.get("pending")
    print("Paper trading status")
    print(f"Processed new bars: {processed}")
    print(f"Equity: {float(state['equity']):.2f}")
    print(f"Marked equity: {float(state.get('last_marked_equity', state['equity'])):.2f}")
    print(f"Max drawdown: {float(state.get('max_drawdown_pct', 0.0)):.2f}%")
    print(f"Closed trades: {state.get('closed_trades', 0)}")
    print(f"Liquidations: {state.get('liquidations', 0)}")
    print(f"Position: {position['side']} qty={position['qty']:.6f} entry={position['entry_price']:.2f} liq={position['liquidation_price']:.2f}" if position else "Position: none")
    print(f"Pending: {pending['side']} target={pending['target_price']:.2f} score={pending['signal_score']:.3f}" if pending else "Pending: none")
    print(f"State: {state_path}")
    print(f"Trades: {trades_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live paper trading for the range-swing strategy without real orders.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--leverage", type=float, default=30.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.18)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--taker-fee", type=float, default=0.00045)
    parser.add_argument("--max-drawdown-stop-pct", type=float, default=50.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.004)
    parser.add_argument("--liquidation-fee-pct", type=float, default=0.001)
    parser.add_argument("--entry-slippage-bps", type=float, default=0.5)
    parser.add_argument("--exit-slippage-bps", type=float, default=1.5)
    parser.add_argument("--depth-impact-bps", type=float, default=8.0)
    parser.add_argument("--depth-impact-exponent", type=float, default=0.5)
    parser.add_argument("--min-depth-quote", type=float, default=2_000_000.0)
    parser.add_argument("--fetch-days", type=float, default=3.0)
    parser.add_argument("--backfill-bars", type=int, default=1, help="Replay this many latest closed bars when creating new state.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--state-path", type=Path, default=repo_root() / "data/paper_trading/state.json")
    parser.add_argument("--trades-path", type=Path, default=repo_root() / "data/paper_trading/trades.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = default_config(args)
    if args.reset_state and args.state_path.exists():
        args.state_path.unlink()

    state = load_state(args.state_path, args)
    if state.get("symbol") != sim.normalize_symbol(args.symbol) or state.get("interval") != args.interval:
        raise RuntimeError("state symbol/interval does not match args; use --reset-state to start a new paper account")

    while True:
        run_once(args, cfg, state, args.state_path, args.trades_path)
        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    raise SystemExit(main())
