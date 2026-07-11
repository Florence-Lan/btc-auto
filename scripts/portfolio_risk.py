from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import simulate_range_swing as sim


@dataclass(frozen=True)
class DrawdownRiskPolicy:
    soft_start_pct: float = 8.0
    hard_stop_pct: float = 15.0
    min_multiplier: float = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.soft_start_pct < self.hard_stop_pct:
            raise ValueError("drawdown policy requires 0 <= soft start < hard stop")
        if not 0 < self.min_multiplier <= 1:
            raise ValueError("drawdown min multiplier must be in (0, 1]")


def drawdown_multiplier(drawdown_fraction: float, policy: DrawdownRiskPolicy) -> float:
    drawdown_pct = max(drawdown_fraction, 0.0) * 100
    if drawdown_pct >= policy.hard_stop_pct:
        return 0.0
    if drawdown_pct <= policy.soft_start_pct:
        return 1.0
    progress = (drawdown_pct - policy.soft_start_pct) / (
        policy.hard_stop_pct - policy.soft_start_pct
    )
    return 1.0 - progress * (1.0 - policy.min_multiplier)


def combine_sleeves_with_drawdown_policy(
    candles: Sequence[sim.Candle],
    sleeve_results: Sequence[Mapping[str, Any]],
    cfg: sim.StrategyConfig,
    policy: DrawdownRiskPolicy,
    evaluation_start_ms: Optional[int] = None,
) -> dict[str, Any]:
    entries: dict[int, list[dict[str, Any]]] = {}
    exits: dict[int, list[dict[str, Any]]] = {}
    for result in sleeve_results:
        for raw_trade in result.get("trades", []):
            trade = dict(raw_trade)
            entry_ms = sim._utc_ms(trade["entry_time_utc"])
            exit_ms = sim._utc_ms(trade["exit_time_utc"])
            trade["_immediate"] = exit_ms <= entry_ms
            entries.setdefault(entry_ms, []).append(trade)
            if not trade["_immediate"]:
                exits.setdefault(exit_ms, []).append(trade)

    start_index = 0
    if evaluation_start_ms is not None:
        start_index = next(
            (
                index
                for index, candle in enumerate(candles)
                if candle.open_time_ms >= evaluation_start_ms
            ),
            len(candles),
        )

    equity = cfg.initial_equity
    peak = equity
    max_drawdown = 0.0
    current_drawdown = 0.0
    active: dict[int, dict[str, Any]] = {}
    scaled_trades: list[sim.Trade] = []
    equity_curve: list[dict[str, float]] = []
    trade_id = 0
    throttled_entries = 0
    blocked_entries = 0
    hard_halt_time_ms: int | None = None
    multipliers: list[float] = []

    for candle in candles[start_index:]:
        timestamp = candle.open_time_ms
        for raw in exits.get(timestamp, []):
            matching_id = next(
                (key for key, value in active.items() if value["raw"] is raw),
                None,
            )
            if matching_id is None:
                continue
            item = active.pop(matching_id)
            scaled_trade = sim.scaled_trade_from_raw(
                raw,
                item["scale"],
                item["equity_at_entry"],
            )
            equity += scaled_trade.net_pnl
            scaled_trades.append(scaled_trade)

        risk_multiplier = drawdown_multiplier(current_drawdown, policy)
        if risk_multiplier <= 0 and hard_halt_time_ms is None:
            hard_halt_time_ms = timestamp
        for raw in entries.get(timestamp, []):
            if risk_multiplier <= 0:
                blocked_entries += 1
                continue
            desired_signed_notional = (
                float(raw["entry_price"])
                * float(raw["initial_qty"])
                * sim.direction(str(raw["side"]))
            )
            current_gross_notional = sum(
                abs(value["signed_qty"] * candle.open) for value in active.values()
            )
            cap = max(equity, 0.0) * cfg.portfolio_leverage_cap
            desired_gross_notional = abs(desired_signed_notional)
            available_gross_notional = max(cap - current_gross_notional, 0.0)
            capacity_scale = sim.clamp(
                available_gross_notional / desired_gross_notional
                if desired_gross_notional
                else 0.0,
                0.0,
                1.0,
            )
            scale = capacity_scale * risk_multiplier
            if scale <= 0:
                blocked_entries += 1
                continue
            if risk_multiplier < 1:
                throttled_entries += 1
            multipliers.append(risk_multiplier)
            if raw.get("_immediate"):
                scaled_trade = sim.scaled_trade_from_raw(raw, scale, equity)
                equity += scaled_trade.net_pnl
                scaled_trades.append(scaled_trade)
                continue
            active[trade_id] = {
                "raw": raw,
                "scale": scale,
                "signed_qty": (
                    float(raw["initial_qty"])
                    * sim.direction(str(raw["side"]))
                    * scale
                ),
                "equity_at_entry": equity,
            }
            trade_id += 1

        unrealized = sum(
            (candle.close - float(item["raw"]["entry_price"])) * item["signed_qty"]
            for item in active.values()
        )
        marked = equity + unrealized
        peak = max(peak, marked)
        current_drawdown = (peak - marked) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)
        net_qty = sum(item["signed_qty"] for item in active.values())
        equity_curve.append(
            {
                "time_ms": timestamp,
                "equity": marked,
                "drawdown_pct": current_drawdown * 100,
                "exposure": 1.0 if active else 0.0,
                "signed_qty": net_qty,
                "price": candle.close,
                "drawdown_risk_multiplier": drawdown_multiplier(current_drawdown, policy),
            }
        )

    summary_candles = candles[start_index:] if start_index < len(candles) else candles[-1:]
    summary = sim.summarize_results(
        summary_candles,
        scaled_trades,
        equity_curve,
        cfg,
        equity,
        max_drawdown,
    )
    diagnostics = {
        "policy": asdict(policy),
        "throttled_entries": throttled_entries,
        "blocked_entries": blocked_entries,
        "hard_halt_time_ms": hard_halt_time_ms,
        "average_drawdown_multiplier_at_entry": (
            sum(multipliers) / len(multipliers) if multipliers else None
        ),
        "minimum_drawdown_multiplier_at_entry": min(multipliers) if multipliers else None,
    }
    return {
        "summary": summary,
        "trades": [asdict(trade) for trade in scaled_trades],
        "equity_curve": equity_curve,
        "config": asdict(cfg),
        "risk_diagnostics": diagnostics,
        "sleeves": [result.get("summary", {}) for result in sleeve_results],
    }
