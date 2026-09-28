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
    from execution_ledger import attach_ledger

    timeline: list[tuple[int, int, int, str, dict, dict | None]] = []
    ledger_trades = 0
    legacy_trades = 0
    for result in sleeve_results:
        for raw_trade in attach_ledger(result).get("trades", []):
            raw = dict(raw_trade)
            identity = ledger_trades + legacy_trades
            entry_ms = sim._utc_ms(raw["entry_time_utc"])
            timeline.append((entry_ms, 1, identity, "entry", raw, None))
            if "_ledger" in raw:
                ledger_trades += 1
                for event in raw["_ledger"]:
                    # Existing exits settle before a reversal's new entry. An
                    # immediate trade must first enter, then settle its cash.
                    priority = 1 if event["time_ms"] == entry_ms else 0
                    timeline.append((int(event["time_ms"]), priority, identity, "cash", raw, event))
            else:
                legacy_trades += 1
                exit_ms = max(entry_ms, sim._utc_ms(raw["exit_time_utc"]))
                event = {"time_ms": exit_ms, "remaining_fraction": 0.0,
                         "cash_per_unit": float(raw["net_pnl"]) / float(raw["initial_qty"])}
                timeline.append((exit_ms, 1 if exit_ms == entry_ms else 0, identity, "cash", raw, event))
    # Stable sorting preserves multiple events within one trade/bar.
    timeline.sort(key=lambda item: item[:3])
    start_index = next((i for i, c in enumerate(candles)
                        if evaluation_start_ms is None or c.open_time_ms >= evaluation_start_ms), len(candles))
    equity = cfg.initial_equity  # Cash including realized partial exits and fees.
    peak = equity
    max_drawdown = current_drawdown = 0.0
    active: dict[int, dict[str, Any]] = {}
    scaled_trades: list[sim.Trade] = []
    equity_curve: list[dict[str, float]] = []
    throttled_entries = blocked_entries = 0
    hard_halt_time_ms: int | None = None
    multipliers: list[float] = []
    end_position_targets: list[dict[str, Any]] = []
    cursor = 0

    def marked_equity(price: float) -> float:
        return equity + sum(
            (price - float(item["raw"]["entry_price"])) * item["signed_qty"]
            - (abs(price * item["signed_qty"]) * cfg.taker_fee if item["ledger"] else 0.0)
            for item in active.values())

    def update_risk(price: float, timestamp: int) -> None:
        nonlocal peak, current_drawdown, max_drawdown, hard_halt_time_ms
        marked = marked_equity(price)
        peak = max(peak, marked)
        current_drawdown = max(0.0, (peak - marked) / peak) if peak else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)
        if drawdown_multiplier(current_drawdown, policy) <= 0 and hard_halt_time_ms is None:
            hard_halt_time_ms = timestamp

    def process_until(until: int, price: float) -> None:
        nonlocal cursor, equity, blocked_entries, throttled_entries
        while cursor < len(timeline) and timeline[cursor][0] <= until:
            timestamp, _, identity, kind, raw, event = timeline[cursor]
            cursor += 1
            if evaluation_start_ms is not None and timestamp < evaluation_start_ms:
                continue
            if kind == "cash":
                item = active.get(identity)
                if item is None:
                    continue
                cash = float(event["cash_per_unit"]) * float(raw["initial_qty"]) * item["scale"]
                equity += cash - item["booked_cash"]
                item["booked_cash"] = cash
                item["signed_qty"] = (float(raw["initial_qty"]) * sim.direction(str(raw["side"]))
                                      * float(event["remaining_fraction"]) * item["scale"])
                if float(event["remaining_fraction"]) <= 0:
                    active.pop(identity)
                    scaled_trades.append(sim.scaled_trade_from_raw(raw, item["scale"], item["equity_at_entry"]))
                continue
            update_risk(price, timestamp)
            risk_multiplier = 0.0 if hard_halt_time_ms is not None else drawdown_multiplier(current_drawdown, policy)
            if risk_multiplier <= 0:
                blocked_entries += 1
                continue
            desired = abs(float(raw["entry_price"]) * float(raw["initial_qty"]))
            current_gross = sum(abs(item["signed_qty"] * price) for item in active.values())
            cap = max(marked_equity(price), 0.0) * cfg.portfolio_leverage_cap
            capacity = sim.clamp((cap - current_gross) / desired if desired else 0.0, 0.0, 1.0)
            scale = capacity * risk_multiplier
            if scale <= 0:
                blocked_entries += 1
                continue
            throttled_entries += int(risk_multiplier < 1)
            multipliers.append(risk_multiplier)
            if include_execution_target and str(raw.get("exit_reason")) == "end":
                if "_open_qty_fraction" not in raw:
                    raise RuntimeError("Synthetic end-of-window trade is missing open quantity metadata")
                end_position_targets.append({
                    "signed_qty": float(raw["initial_qty"]) * sim.clamp(float(raw["_open_qty_fraction"]), 0.0, 1.0)
                                  * sim.direction(str(raw["side"])) * scale,
                    "strategy": str(raw.get("strategy") or "unknown"), "side": str(raw["side"]),
                    "entry_time_utc": str(raw["entry_time_utc"]), "entry_price": float(raw["entry_price"]),
                    "signal_reason": str(raw.get("signal_reason") or ""),
                })
            active[identity] = {"raw": raw, "scale": scale, "ledger": "_ledger" in raw,
                                "booked_cash": 0.0, "equity_at_entry": marked_equity(price),
                                "signed_qty": float(raw["initial_qty"]) * sim.direction(str(raw["side"])) * scale}

    for candle in candles[start_index:]:
        process_until(candle.open_time_ms, candle.open)
        # Sub-bar settlement is only used once observable at the base bar close.
        process_until(candle.close_time_ms, candle.close)
        update_risk(candle.close, candle.close_time_ms)
        equity_curve.append({
            "time_ms": candle.open_time_ms, "equity": marked_equity(candle.close),
            "cash_equity": equity, "drawdown_pct": current_drawdown * 100,
            "exposure": 1.0 if active else 0.0,
            "signed_qty": sum(item["signed_qty"] for item in active.values()),
            "gross_qty": sum(abs(item["signed_qty"]) for item in active.values()),
            "price": candle.close,
            "drawdown_risk_multiplier": 0.0 if hard_halt_time_ms is not None else drawdown_multiplier(current_drawdown, policy),
        })

    summary_candles = candles[start_index:] if start_index < len(candles) else candles[-1:]
    summary = sim.summarize_results(
        summary_candles,
        scaled_trades,
        equity_curve,
        cfg,
        marked_equity(candles[-1].close) if candles else equity,
        max_drawdown,
    )
    diagnostics = {
        "policy": asdict(policy),
        "accounting_version": "bar_close_cash_inventory_v1",
        "ledger_trades": ledger_trades,
        "legacy_endpoint_trades": legacy_trades,
        "unsettled_positions": len(active),
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
        point = summary.get("last_equity_point") or (
            {
                "time_ms": candles[-1].open_time_ms,
                "equity": float(summary.get("final_equity") or cfg.initial_equity),
                "price": candles[-1].close,
                "signed_qty": 0.0,
            }
            if candles
            else {}
        )
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
