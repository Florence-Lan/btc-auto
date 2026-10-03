"""Causal bar-close position targets; terminal settlement is kept out of open targets.

Signals still come from the frozen research engine. Only ledger events observable
at the decision bar are used; an end-of-window liquidation is not a trading signal.
"""
import hashlib

from execution_ledger import attach_ledger
import simulate_range_swing as sim


def prepare_sleeves(sleeves):
    """Retain the last observable cash/inventory before synthetic liquidation."""
    for sleeve in sleeves:
        completed = 0.0
        cfg, curve = sleeve.get("config"), sleeve.get("equity_curve")
        if not cfg or not curve:
            continue
        for trade in sorted(sleeve["trades"], key=lambda t: t["entry_time_utc"]):
            if trade["exit_reason"] == "end":
                point = curve[-1]
                qty, signed = float(trade["initial_qty"]), float(point["signed_qty"])
                unrealized = (point["price"] - trade["entry_price"]) * signed
                cash = (point["equity"] - cfg["initial_equity"] - completed - unrealized
                        + abs(signed * point["price"]) * cfg["taker_fee"])
                interval = sim.interval_to_ms(cfg["timeseries_timeframe"] if trade["strategy"].startswith("timeseries") else "5m")
                trade["_execution_terminal_observation"] = {
                    "time_ms": int(point.get("available_time_ms", int(point["time_ms"]) + interval - 1)),
                    "remaining_fraction": abs(signed) / qty, "cash_per_unit": cash / qty}
            completed += trade["net_pnl"]


def trade_key(trade):
    return trade["strategy"], trade["side"], trade["entry_time_utc"]


def target_stream(base, sleeves, result, cfg):
    source = {trade_key(t): t for sleeve in sleeves for t in attach_ledger(sleeve)["trades"]}
    events = []
    for identity, scaled in enumerate(result["trades"]):
        raw = source[trade_key(scaled)]
        qty = float(scaled["initial_qty"])
        entry = sim._utc_ms(raw["entry_time_utc"])
        events.append((entry, 1, identity, "entry", scaled, None))
        for event in raw["_ledger"]:
            if raw["exit_reason"] == "end" and event["remaining_fraction"] == 0:
                continue
            events.append((int(event["time_ms"]), 1 if event["time_ms"] == entry else 0,
                           identity, "inventory", scaled, event))
        observation = raw.get("_execution_terminal_observation")
        if observation:
            events.append((observation["time_ms"], 1 if observation["time_ms"] == entry else 0,
                           identity, "inventory", scaled, observation))
    events.sort(key=lambda item: item[:3])
    active, cursor = {}, 0
    for point in result["equity_curve"]:
        close_ms = int(point.get("available_time_ms", int(point["time_ms"]) + 300_000 - 1))
        while cursor < len(events) and events[cursor][0] <= close_ms:
            _, _, identity, kind, trade, event = events[cursor]
            cursor += 1
            if kind == "entry":
                active[identity] = {"trade": trade, "remaining": 1.0, "cash_per_unit": 0.0}
            elif identity in active:
                active[identity].update(remaining=float(event["remaining_fraction"]),
                                        cash_per_unit=float(event["cash_per_unit"]))
                if event["remaining_fraction"] <= 0:
                    del active[identity]
        equity = float(point["equity"])
        components = []
        for item in active.values():
            trade, fraction = item["trade"], item["remaining"]
            qty = float(trade["initial_qty"])
            signed = qty * fraction * sim.direction(trade["side"])
            components.append({"strategy": trade["strategy"], "side": trade["side"],
                               "entry_time_utc": trade["entry_time_utc"], "entry_price": trade["entry_price"],
                               "signed_qty": signed})
            raw = source[trade_key(trade)]
            if raw["exit_reason"] == "end" and raw["_ledger_exit_ms"] <= close_ms:
                # Undo synthetic closing cash and retain known mark-to-market inventory.
                equity += (item["cash_per_unit"] * qty
                           + (point["price"] - trade["entry_price"]) * signed
                           - abs(signed * point["price"]) * cfg.taker_fee - trade["net_pnl"])
        identity = "|".join(sorted(f"{c['strategy']}:{c['side']}:{c['entry_time_utc']}" for c in components))
        signed_qty = sum(c["signed_qty"] for c in components)
        gross = sum(abs(c["signed_qty"]) for c in components)
        cap = max(equity, 0.0) * cfg.portfolio_leverage_cap / point["price"]
        capacity = min(1.0, cap / gross) if gross else 0.0
        for c in components:
            c["signed_qty"] *= capacity
        yield {"time_ms": int(point["time_ms"]), "available_time_ms": close_ms,
               "equity": equity, "price": point["price"], "signed_qty": signed_qty * capacity,
               "position_id": hashlib.sha256(identity.encode()).hexdigest()[:16] if identity else "flat",
               "origin_signal_time_ms": min(sim._utc_ms(c["entry_time_utc"]) for c in components) if components else None,
               "components": components}


def current_target(base, sleeves, result, cfg):
    target = None
    for target in target_stream(base, sleeves, result, cfg):
        pass
    if target is None:
        return result["execution_target"]
    components = target["components"]
    weight = sum(abs(c["signed_qty"]) for c in components)
    target["origin_entry_price"] = (sum(abs(c["signed_qty"]) * c["entry_price"] for c in components) / weight
                                    if weight else None)
    return target
