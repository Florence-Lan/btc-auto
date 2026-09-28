"""Reconstruct trade cash/inventory from frozen sleeve close-of-bar observations.

Amounts are per initial unit, so entry overlays can scale a trade without
scaling its ledger twice. Intrabar events become available at bar close; this
does not invent a tick sequence or read an hourly close at the hour's open.
"""
from bisect import bisect_left
from typing import Any, Mapping

import simulate_range_swing as sim


def attach_ledger(sleeve: Mapping[str, Any]) -> dict[str, Any]:
    trades = sleeve.get("trades", [])
    if not trades or all("_ledger" in t for t in trades):
        return dict(sleeve)
    curve, cfg = sleeve.get("equity_curve"), sleeve.get("config")
    if not curve or not cfg:
        # Historic hand-written trade records have only endpoint information.
        # The combiner reports this explicitly instead of claiming exact cashflow.
        return dict(sleeve)
    times = [int(p["time_ms"]) for p in curve]
    initial = float(cfg["initial_equity"])
    completed_pnl = 0.0
    result = []
    for raw in sorted(trades, key=lambda t: t["entry_time_utc"]):
        trade = dict(raw)
        if "_ledger" in trade:
            raise ValueError("Cannot mix reconstructed and unannotated trades in a sleeve")
        qty = float(trade["initial_qty"])
        if qty <= 0:
            raise ValueError("Ledger requires positive initial quantity")
        core = str(trade.get("strategy", "")).startswith("timeseries")
        interval = sim.interval_to_ms(cfg["timeseries_timeframe"] if core else "5m")
        entry, exit_bar = sim._utc_ms(trade["entry_time_utc"]), sim._utc_ms(trade["exit_time_utc"])
        at_open = core and trade["exit_reason"] in ("ema_cross", "risk_exit", "neutral_exit", "volatility_exit")
        exit_ms = exit_bar if at_open else exit_bar + interval - 1
        entry_fee = float(cfg["taker_fee"] if core else cfg["maker_fee"]) * float(trade["entry_price"])
        events = [{"time_ms": entry, "remaining_fraction": 1.0, "cash_per_unit": -entry_fee}]
        for p in curve[bisect_left(times, entry):bisect_left(times, exit_bar + 1)]:
            available = int(p["time_ms"]) + interval - 1
            if available >= exit_ms:
                continue
            signed = float(p["signed_qty"])
            remaining = abs(signed) / qty
            if remaining > 1.000001 or signed * sim.direction(trade["side"]) < -1e-12:
                raise ValueError("Sleeve inventory does not match original unscaled trade")
            unrealized = (float(p["price"]) - float(trade["entry_price"])) * signed
            estimated_close_fee = abs(signed * float(p["price"])) * float(cfg["taker_fee"])
            cash = (float(p["equity"]) - initial - completed_pnl - unrealized + estimated_close_fee) / qty
            event = {"time_ms": available, "remaining_fraction": remaining, "cash_per_unit": cash}
            prior = events[-1]
            if abs(cash - prior["cash_per_unit"]) > 1e-8 or abs(remaining - prior["remaining_fraction"]) > 1e-10:
                events.append(event)
        events.append({"time_ms": exit_ms, "remaining_fraction": 0.0,
                       "cash_per_unit": float(trade["net_pnl"]) / qty})
        trade["_ledger"] = events
        trade["_ledger_exit_ms"] = exit_ms
        result.append(trade)
        completed_pnl += float(trade["net_pnl"])
    return {**sleeve, "trades": result}
