"""Research-only hourly exits; original frozen engine and manifests stay intact.

Baseline loop derived from simulate_range_swing.simulate_timeseries_trend.
No CLI or live execution integration. All decisions use completed bars.
"""
from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence
from simulate_range_swing import (
    Candle, StrategyConfig, FundingHistory, Position, Trade, MS_PER_DAY,
    ema, atr, interval_to_ms, stddev, direction, execution_price,
    exit_position_part, close_trade_record, drawdown_halted, liquidation_price,
    settle_position_funding, liquidation_hit_for_position, summarize_results,
)


@dataclass(frozen=True)
class ExitPolicy:
    name: str = "baseline"
    stop_atr: float = 2.0
    trail_atr: float = 3.0

    def __post_init__(self):
        if self.name not in ("baseline", "neutral", "volatility"):
            raise ValueError("Unknown exit policy")
        if self.stop_atr <= 0 or self.trail_atr <= 0:
            raise ValueError("ATR distances must be positive")


def exit_reason_at_close(policy, side, entry, close, best_close, entry_atr, spread):
    side_dir = direction(side)
    if policy.name == "neutral" and spread * side_dir <= 0:
        return "neutral_exit"
    if policy.name == "volatility" and entry_atr > 0:
        loss = (entry - close) * side_dir
        giveback = (best_close - close) * side_dir
        if loss >= policy.stop_atr * entry_atr or giveback >= policy.trail_atr * entry_atr:
            return "volatility_exit"
    return None


def simulate_with_exit_policy(
    candles: Sequence[Candle],
    cfg: StrategyConfig,
    evaluation_start_ms: Optional[int] = None,
    funding_history: Optional[FundingHistory] = None,
    policy: ExitPolicy = ExitPolicy(),
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

    atr_values = atr(candles, 14)
    entry_atr = 0.0
    best_close = 0.0
    pending_exit: Optional[str] = None

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

        if pending_side is not None or pending_exit is not None:
            exit_reason = pending_exit or "ema_cross"
            if position is not None:
                raw_exit = candle.open
                fill = execution_price(raw_exit, position.side, False, position.qty, candle, cfg)
                exit_position_part(
                    position,
                    candle,
                    fill,
                    position.qty,
                    cfg.taker_fee,
                    exit_reason,
                    raw_exit,
                )
                trade = close_trade_record(
                    position,
                    candle,
                    position_equity_base,
                    exit_reason,
                    f"timeseries_trend_{position.side}",
                )
                trade.bars_held = index - position.entry_index
                trade.strategy = "timeseries_trend_6h"
                equity += trade.net_pnl
                trades.append(trade)
                position = None

            desired_side = pending_side
            pending_side = None
            pending_exit = None
            allowed = cfg.side_mode in ("auto", "both", desired_side)
            halted = drawdown_halted(max_drawdown, cfg)
            if desired_side is not None and allowed and not halted and equity > 0:
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
                    entry_atr = float(atr_values[index - 1] or 0.0)
                    best_close = entry
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
                trade.strategy = "timeseries_trend_6h"
                equity += trade.net_pnl
                trades.append(trade)
                position = None

        previous_candle = candles[index - 1]
        if (
            fast[index] is not None
            and slow[index] is not None
            and fast[index - 1] is not None
            and slow[index - 1] is not None
            and candle.close > 0
            and previous_candle.close > 0
        ):
            spread = (fast[index] - slow[index]) / candle.close
            previous_spread = (
                fast[index - 1] - slow[index - 1]
            ) / previous_candle.close
            threshold = cfg.timeseries_min_ema_spread_pct
            confirmed_side = (
                "long" if spread >= threshold
                else "short" if spread <= -threshold
                else None
            )
            previous_confirmed_side = (
                "long" if previous_spread >= threshold
                else "short" if previous_spread <= -threshold
                else None
            )
            if (
                confirmed_side is not None
                and confirmed_side != previous_confirmed_side
                and (position is None or position.side != confirmed_side)
            ):
                pending_side = confirmed_side

        # Only CLOSED hourly bars decide an exit; execute at the next open.
        # Re-entry still requires a fresh threshold crossing, as in the baseline.
        if position is not None and pending_side is None:
            best_close = max(best_close, candle.close) if position.side == "long" else min(best_close, candle.close)
            spread = (fast[index] - slow[index]) / candle.close
            pending_exit = exit_reason_at_close(policy, position.side, position.entry_price,
                                                candle.close, best_close, entry_atr, spread)

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
        trade.strategy = "timeseries_trend_6h"
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
