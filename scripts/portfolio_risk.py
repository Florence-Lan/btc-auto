from __future__ import annotations

import hashlib
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
    *,
    include_execution_target: bool = False,
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
    end_position_targets: list[dict[str, Any]] = []

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
            if include_execution_target and str(raw.get("exit_reason")) == "end":
                if "_open_qty_fraction" not in raw:
                    raise RuntimeError(
                        "Synthetic end-of-window trade is missing open quantity metadata"
                    )
                open_fraction = sim.clamp(
                    float(raw["_open_qty_fraction"]),
                    0.0,
                    1.0,
                )
                end_position_targets.append({
                    "signed_qty": (
                        float(raw["initial_qty"])
                        * open_fraction
                        * sim.direction(str(raw["side"]))
                        * scale
                    ),
                    "strategy": str(raw.get("strategy") or "unknown"),
                    "side": str(raw["side"]),
                    "entry_time_utc": str(raw["entry_time_utc"]),
                    "entry_price": float(raw["entry_price"]),
                    "signal_reason": str(raw.get("signal_reason") or ""),
                })
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
    result = {
        "summary": summary,
        "trades": [asdict(trade) for trade in scaled_trades],
        "equity_curve": equity_curve,
        "config": asdict(cfg),
        "risk_diagnostics": diagnostics,
        "sleeves": [result.get("summary", {}) for result in sleeve_results],
    }
    if include_execution_target:
        point = summary.get("last_equity_point") or {}
        target_price = float(point.get("price") or 0.0)
        target_equity = float(point.get("equity") or 0.0)
        gross_qty = sum(abs(float(item["signed_qty"])) for item in end_position_targets)
        max_gross_qty = (
            max(target_equity, 0.0) * cfg.portfolio_leverage_cap / target_price
            if target_price > 0
            else 0.0
        )
        final_scale = min(1.0, max_gross_qty / gross_qty) if gross_qty > 0 else 0.0
        components = [
            {**item, "signed_qty": float(item["signed_qty"]) * final_scale}
            for item in end_position_targets
        ]
        identity = "|".join(sorted(
            f"{item['strategy']}:{item['side']}:{item['entry_time_utc']}"
            for item in components
        ))
        net_qty = sum(float(item["signed_qty"]) for item in components)
        gross_component_qty = sum(abs(float(item["signed_qty"])) for item in components)
        origin_entry_price = (
            sum(
                abs(float(item["signed_qty"])) * float(item["entry_price"])
                for item in components
            ) / gross_component_qty
            if gross_component_qty > 0
            else None
        )
        result["execution_target"] = {
            "time_ms": int(point.get("time_ms") or 0),
            "equity": target_equity,
            "price": target_price,
            "signed_qty": net_qty,
            "position_id": (
                hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                if identity
                else "flat"
            ),
            "origin_signal_time_ms": (
                min(sim._utc_ms(str(item["entry_time_utc"])) for item in components)
                if components
                else None
            ),
            "origin_entry_price": origin_entry_price,
            "components": components,
        }
    return result
