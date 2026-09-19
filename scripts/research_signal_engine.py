"""Research-only signal replay; copied from the frozen v3 execution model.

Signals use completed candles and execute on the next candle open. None holds;
flat closes. No changes to the active engine. Costs/funding/position sizing match
its timeseries sleeve. Like that engine, depth impact uses execution-bar volume.
"""
from __future__ import annotations
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence
from simulate_range_swing import (
    Candle, StrategyConfig, FundingHistory, Trade, Position, MS_PER_DAY,
    ema, interval_to_ms, execution_price, exit_position_part, close_trade_record,
    drawdown_halted, stddev, direction, liquidation_price, settle_position_funding,
    liquidation_hit_for_position, summarize_results,
)

def simulate_signals(
    candles: Sequence[Candle],
    signals: Sequence[Optional[str]],
    strategy_name: str,
    cfg: StrategyConfig,
    evaluation_start_ms: Optional[int] = None,
    funding_history: Optional[FundingHistory] = None,
) -> Dict[str, Any]:
    if cfg.timeseries_fast_ema >= cfg.timeseries_slow_ema:
        raise ValueError("timeseries fast EMA must be less than slow EMA")
    closes = [candle.close for candle in candles]
    fast = ema(closes, cfg.timeseries_fast_ema)
    slow = ema(closes, cfg.timeseries_slow_ema)
    log_returns = [0.0]
    for index in range(1, len(closes)):
        log_returns.append(math.log(closes[index] / closes[index - 1]) if closes[index - 1] > 0 else 0.0)

    warmup = max(cfg.timeseries_slow_ema, cfg.timeseries_vol_lookback_bars) + 1
    start_index = warmup
    if evaluation_start_ms is not None:
        start_index = max(
            warmup,
            next(
                (index for index, candle in enumerate(candles) if candle.open_time_ms >= evaluation_start_ms),
                len(candles),
            ),
        )

    equity = cfg.initial_equity
    peak_equity = equity
    max_drawdown = 0.0
    trades: List[Trade] = []
    equity_curve: List[Dict[str, float]] = []
    position: Optional[Position] = None
    position_equity_base = equity
    pending_side: Optional[str] = None
    periods_per_year = 365 * (MS_PER_DAY / interval_to_ms(cfg.timeseries_timeframe))

    for index in range(start_index, len(candles)):
        candle = candles[index]

        if pending_side is not None:
            if position is not None:
                raw_exit = candle.open
                fill = execution_price(raw_exit, position.side, False, position.qty, candle, cfg)
                exit_position_part(
                    position,
                    candle,
                    fill,
                    position.qty,
                    cfg.taker_fee,
                    "signal_change",
                    raw_exit,
                )
                trade = close_trade_record(
                    position,
                    candle,
                    position_equity_base,
                    "signal_change",
                    f"timeseries_trend_{position.side}",
                )
                trade.bars_held = index - position.entry_index
                trade.strategy = strategy_name
                equity += trade.net_pnl
                trades.append(trade)
                position = None

            desired_side = pending_side
            pending_side = None
            allowed = desired_side != "flat" and cfg.side_mode in ("auto", "both", desired_side)
            halted = drawdown_halted(max_drawdown, cfg)
            if allowed and not halted and equity > 0:
                window = log_returns[index - cfg.timeseries_vol_lookback_bars : index]
                realized_vol = stddev(window) * math.sqrt(periods_per_year) if len(window) > 1 else 0.0
                target_leverage = min(
                    cfg.timeseries_max_leverage,
                    cfg.timeseries_target_vol / max(realized_vol, 0.05),
                )
                side_dir = direction(desired_side)
                raw_entry = candle.open
                rough_qty = equity * target_leverage / raw_entry
                entry = execution_price(raw_entry, desired_side, True, rough_qty, candle, cfg)
                qty = equity * target_leverage / entry if entry > 0 else 0.0
                if qty > 0:
                    entry_fee = abs(entry * qty) * cfg.taker_fee
                    position_equity_base = equity
                    position = Position(
                        side=desired_side,
                        entry_index=index,
                        entry_time_utc=candle.open_time_utc,
                        entry_price=entry,
                        qty=qty,
                        initial_qty=qty,
                        stop_price=0.0 if side_dir > 0 else float("inf"),
                        tp1=0.0,
                        tp2=0.0,
                        tp3=0.0,
                        liquidation_price=liquidation_price(
                            entry,
                            desired_side,
                            max(target_leverage, 1e-6),
                            cfg,
                        ),
                        best_price=entry,
                        fees_paid=entry_fee,
                        slippage_cost=abs(entry - raw_entry) * qty,
                        last_funding_time_ms=candle.open_time_ms - 1,
                    )

        if position is not None:
            settle_position_funding(position, candle, funding_history)

            if liquidation_hit_for_position(position, candle):
                raw_exit = position.liquidation_price
                fill = execution_price(raw_exit, position.side, False, position.qty, candle, cfg)
                exit_position_part(
                    position,
                    candle,
                    fill,
                    position.qty,
                    cfg.taker_fee + cfg.liquidation_fee_pct,
                    "liquidation",
                    raw_exit,
                )
                trade = close_trade_record(
                    position,
                    candle,
                    position_equity_base,
                    "liquidation",
                    f"timeseries_trend_{position.side}",
                )
                trade.bars_held = index - position.entry_index
                trade.strategy = strategy_name
                equity += trade.net_pnl
                trades.append(trade)
                position = None

        target = signals[index]
        if target not in (None, "flat", "long", "short"):
            raise ValueError("Invalid signal")
        current_side = position.side if position else "flat"
        if target is not None and target != current_side:
            pending_side = target

        marked_equity = equity
        signed_qty = 0.0
        if position is not None:
            signed_qty = position.qty * direction(position.side)
            unrealized = (candle.close - position.entry_price) * signed_qty
            close_fee = abs(candle.close * position.qty) * cfg.taker_fee
            marked_equity = equity + position.realized_pnl + unrealized - position.fees_paid - close_fee
        peak_equity = max(peak_equity, marked_equity)
        drawdown = (peak_equity - marked_equity) / peak_equity if peak_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        equity_curve.append(
            {
                "time_ms": candle.open_time_ms,
                "equity": marked_equity,
                "drawdown_pct": drawdown * 100,
                "exposure": 1.0 if position is not None else 0.0,
                "signed_qty": signed_qty,
                "price": candle.close,
            }
        )

    if position is not None:
        candle = candles[-1]
        raw_exit = candle.close
        fill = execution_price(raw_exit, position.side, False, position.qty, candle, cfg)
        exit_position_part(position, candle, fill, position.qty, cfg.taker_fee, "end", raw_exit)
        trade = close_trade_record(
            position,
            candle,
            position_equity_base,
            "end",
            f"timeseries_trend_{position.side}",
        )
        trade.bars_held = len(candles) - 1 - position.entry_index
        trade.strategy = strategy_name
        equity += trade.net_pnl
        trades.append(trade)

    summary_candles = candles[start_index:] if start_index < len(candles) else candles[-1:]
    summary = summarize_results(summary_candles, trades, equity_curve, cfg, equity, max_drawdown)
    return {
        "summary": summary,
        "trades": [asdict(trade) for trade in trades],
        "equity_curve": equity_curve,
        "config": asdict(cfg),
    }
