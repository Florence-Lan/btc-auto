#!/usr/bin/env python3
"""Causal, isolated public-data replay for a memory-stock swing candidate.

This research program has no account client and cannot submit orders. Margin
tiers, fees and intrabar execution are explicit assumptions, not live evidence.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

from stock_swing_signals import (
    Candle, compute_indicators, entry_gap_allowed, initial_stop,
    per_unit_stop_risk, position_size, signal_at, target_exit_price,
)


HOUR = 3_600_000
FOUR_HOURS = 4 * HOUR
DAY = 24 * HOUR


def iso(time_ms: int) -> str:
    return datetime.fromtimestamp(time_ms / 1000, timezone.utc).isoformat()


def parse_time(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def candle(row) -> Candle:
    return Candle(int(row[0]), *[float(row[i]) for i in range(1, 6)])


def aggregate_4h(rows: list[Candle]) -> list[Candle]:
    groups = {}
    for row in rows:
        groups.setdefault(row.time_ms // FOUR_HOURS * FOUR_HOURS, []).append(row)
    result = []
    for start, group in sorted(groups.items()):
        if [row.time_ms for row in group] != [start + i * HOUR for i in range(4)]:
            continue
        result.append(Candle(start, group[0].open, max(row.high for row in group),
                             min(row.low for row in group), group[-1].close,
                             sum(row.volume for row in group)))
    if any(right.time_ms - left.time_ms != FOUR_HOURS for left, right in zip(result, result[1:])):
        raise ValueError("Missing complete 4h signal candles")
    return result


def round_tick(price: float, tick: float, up: bool) -> float:
    step = Decimal(str(tick))
    number = Decimal(str(price)) / step
    return float(number.to_integral_value(rounding=ROUND_UP if up else ROUND_DOWN) * step)


def liquidation_price(position: dict, maintenance_fraction: float) -> float:
    """Assumed isolated boundary; no inferred tiers from exchange placeholders."""
    entry, side, qty = position["entry"], position["direction"], position["qty"]
    remaining_margin_per_unit = (position["margin"] - position["entry_fee"] - position["funding"]) / qty
    return (side * entry - remaining_margin_per_unit) / (side - maintenance_fraction)


def take_profit_trigger(position: dict, config: dict, fee: float, slippage: float, tick: float) -> float:
    fill = target_exit_price(position["entry"], position["direction"],
                             position["funding"] / position["qty"], fee,
                             config["target_margin_return"], config["leverage"])
    side = position["direction"]
    # A target reached by last price is filled adversely, without double charging.
    trigger = fill / (1 - side * slippage)
    return round_tick(trigger, tick, up=side == 1)


def prepare(snapshot: dict, config: dict) -> dict:
    prepared = {}
    step = int(snapshot.get("execution_step_ms", HOUR))
    for symbol in config["symbols"]:
        source = snapshot["symbols"][symbol]
        profile = {**config, **config.get("symbol_profiles", {}).get(symbol, {})}
        hourly_trades = [candle(row) for row in source["trade_1h"]]
        trades = [candle(row) for row in source.get("trade_5m", source["trade_1h"])]
        marks = {int(row[0]): candle(row) for row in source.get("mark_5m", source["mark_1h"])}
        indexes = {int(row[0]): candle(row) for row in source["index_1h"]}
        if set(row.time_ms for row in trades) != set(marks) or not set(indexes).issubset(marks):
            raise ValueError(f"Incomplete trade/mark/index alignment for {symbol}")
        if any(right.time_ms - left.time_ms != step for left, right in zip(trades, trades[1:])):
            raise ValueError(f"Missing execution candles for {symbol}")
        four = aggregate_4h(hourly_trades)
        indicators = compute_indicators(four, profile)
        # All arrays are causal. An entry at t uses only the bar ending at t.
        signals = {}
        closed_updates = {}
        for index, row in enumerate(four):
            end = row.time_ms + FOUR_HOURS
            closed_updates[end] = (row.close, indicators["atr"][index])
            if "symbol_profiles" in config:
                from stock_swing_profiles import signal_at as profile_signal_at
                signal = profile_signal_at(four, indicators, index, profile)
            else:
                signal = signal_at(four, indicators, index, config)
            if signal is not None:
                signals[end] = signal
        events = source["funding"]
        times = [int(event["fundingTime"]) for event in events]
        if times != sorted(times) or not times:
            raise ValueError(f"Invalid funding coverage for {symbol}")
        filters = {rule["filterType"]: rule for rule in source["rules_current"]["filters"]}
        lot = filters.get("MARKET_LOT_SIZE", filters["LOT_SIZE"])
        prepared[symbol] = {
            "trade": {row.time_ms: row for row in trades}, "mark": marks, "index": indexes,
            "signals": signals, "closed_updates": closed_updates,
            "funding": events, "funding_times": times,
            "funding_by_hour": {},
            "tick": float(filters["PRICE_FILTER"]["tickSize"]),
            "step": float(lot["stepSize"]), "min_qty": float(lot["minQty"]),
            "max_qty": float(lot["maxQty"]),
            "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("notional", 5)),
            "first_signal_time": min(signals, default=None),
            "profile_config": profile,
        }
        for event in events:
            prepared[symbol]["funding_by_hour"].setdefault(int(event["fundingTime"]) // step * step, []).append(event)
    return prepared


def simulate(snapshot: dict, config: dict, start_ms: int, end_ms: int,
             cost_multiplier: float = 1.0) -> dict:
    data = prepare(snapshot, config)
    step = int(snapshot.get("execution_step_ms", HOUR))
    fee = config["taker_fee_rate_assumption"] * cost_multiplier
    slip = config["adverse_slippage_fraction_assumption"] * cost_multiplier
    cash = float(config["initial_equity_usdt"])
    peak = cash
    max_drawdown = 0.0
    positions = {}
    trades = []
    equity_path = []
    skipped = Counter()
    cooldown = {}
    halted = False
    halt_time = None

    def current_equity(timestamp: int, close: bool = False) -> float:
        total = cash
        for symbol, position in positions.items():
            mark = data[symbol]["mark"].get(timestamp)
            if mark is None:
                raise ValueError(f"Missing mark for open position {symbol} at {iso(timestamp)}")
            price = mark.close if close else mark.open
            total += position["direction"] * position["qty"] * (price - position["entry"])
        return total

    def close_position(symbol: str, timestamp: int, reference: float, reason: str):
        nonlocal cash
        pos = positions.pop(symbol)
        side, qty = pos["direction"], pos["qty"]
        exit_fill = reference * (1 - side * slip)
        exit_fee = qty * exit_fill * fee
        gross = side * qty * (exit_fill - pos["entry"])
        if reason == "liquidation_stress":
            # Stress convention: exhaust isolated initial margin, including fees
            # and funding already debited. This is not a venue fill reconstruction.
            exit_fee = 0.0
            gross = -pos["margin"] + pos["entry_fee"] + pos["funding"]
        cash += gross - exit_fee
        net = gross - exit_fee - pos["entry_fee"] - pos["funding"]
        trades.append({
            "symbol": symbol, "direction": "long" if side == 1 else "short",
            "signal_utc": iso(pos["signal_time"]), "entry_utc": iso(pos["entry_time"]),
            "exit_hour_utc": iso(timestamp), "entry_fill": pos["entry"], "exit_fill": exit_fill,
            "exit_utc": iso(timestamp),
            "qty": qty, "initial_margin": pos["margin"], "initial_stop": pos["initial_stop"],
            "exit_reason": reason, "gross_pnl": gross, "entry_fee": pos["entry_fee"],
            "exit_fee": exit_fee, "funding_debit": pos["funding"], "net_pnl": net,
            "funding_mark_proxy_events": pos["funding_mark_proxy_events"],
            "uncredited_ambiguous_funding_events": len(pos.get("pending_funding_credits", [])),
            "net_return_initial_margin_pct": net / pos["margin"] * 100,
            "hours_held": (timestamp - pos["entry_time"]) / HOUR,
        })
        cooldown[symbol] = timestamp + config["cooldown_4h_bars"] * FOUR_HOURS

    all_times = sorted({time for source in data.values() for time in source["trade"]
                        if start_ms <= time < end_ms})
    for timestamp in all_times:
        # Existing inventory is present at an exact-open funding boundary.
        # Settle before reacting to new opening prices; an entry made afterward
        # cannot receive a settlement that has already occurred.
        for symbol, position in positions.items():
            source = data[symbol]
            for event in source["funding_by_hour"].get(timestamp, []):
                if int(event["fundingTime"]) != timestamp:
                    continue
                settlement_mark = event.get("markPrice")
                if settlement_mark is None:
                    settlement_mark = source["mark"][timestamp].open
                    position["funding_mark_proxy_events"] += 1
                debit = position["direction"] * position["qty"] * float(settlement_mark) * float(event["fundingRate"])
                position["funding"] += debit
                cash -= debit
        # First honor gaps and exits scheduled using earlier completed data.
        for symbol, position in list(positions.items()):
            source = data[symbol]
            bar, mark = source["trade"][timestamp], source["mark"][timestamp]
            side = position["direction"]
            liq = liquidation_price(position, config["maintenance_margin_fraction_assumption"])
            target = take_profit_trigger(position, config, fee, slip, source["tick"])
            if side * (mark.open - liq) <= 0:
                close_position(symbol, timestamp, bar.open, "liquidation_stress")
            elif side * (bar.open - position["stop"]) <= 0:
                close_position(symbol, timestamp, bar.open, "stop_gap")
            elif side * (bar.open - target) >= 0:
                close_position(symbol, timestamp, bar.open, "target_gap")
            elif position.get("scheduled_exit"):
                close_position(symbol, timestamp, bar.open, position["scheduled_exit"])
            elif timestamp - position["entry_time"] >= config["max_holding_calendar_days"] * DAY:
                close_position(symbol, timestamp, bar.open, "time_stop")

        equity = current_equity(timestamp)
        peak = max(peak, equity)
        drawdown = max(0.0, (peak - equity) / peak)
        max_drawdown = max(max_drawdown, drawdown)
        if drawdown >= config["account_hard_drawdown_fraction"] or equity <= 0:
            halted = True
            halt_time = halt_time or iso(timestamp)
            for symbol in list(positions):
                close_position(symbol, timestamp, data[symbol]["trade"][timestamp].open, "account_hard_stop")
            equity = cash

        ranked = sorted([(source["signals"][timestamp].strength, symbol, source["signals"][timestamp])
                         for symbol, source in data.items() if timestamp in source["signals"]],
                        key=lambda item: (-item[0], item[1]))
        for _, symbol, signal in ranked:
            source = data[symbol]
            profile = source.get("profile_config", config)
            if symbol in positions:
                skipped["already_open"] += 1
                continue
            if halted or drawdown >= config["account_soft_drawdown_fraction"]:
                skipped["account_drawdown"] += 1
                continue
            if len(positions) >= config["max_positions"]:
                skipped["position_cap"] += 1
                continue
            if timestamp < cooldown.get(symbol, 0):
                skipped["cooldown"] += 1
                continue
            funding_index = bisect.bisect_right(source["funding_times"], timestamp) - 1
            if funding_index < 0 or timestamp - source["funding_times"][funding_index] > 9 * HOUR:
                skipped["stale_or_missing_funding"] += 1
                continue
            rate = float(source["funding"][funding_index]["fundingRate"])
            if signal.direction * rate > config["entry_max_adverse_funding_rate"]:
                skipped["funding_rate"] += 1
                continue
            bar, mark, index = source["trade"][timestamp], source["mark"][timestamp], source["index"][timestamp]
            if abs(mark.open / index.open - 1) > config["max_mark_index_basis_fraction"] or abs(bar.open / mark.open - 1) > config["max_contract_mark_basis_fraction"]:
                skipped["basis"] += 1
                continue
            if not entry_gap_allowed(signal, bar.open, profile):
                skipped["entry_gap"] += 1
                continue
            entry = bar.open * (1 + signal.direction * slip)
            stop = initial_stop(entry, signal.direction, signal.atr, profile)
            if stop is None:
                skipped["stop_too_wide"] += 1
                continue
            stop = round_tick(stop, source["tick"], up=signal.direction == -1)
            if abs(stop / entry - 1) > config["max_stop_fraction"] + 1e-12:
                skipped["stop_too_wide_after_rounding"] += 1
                continue
            qty = position_size(equity, entry, stop, source["step"],
                                config["risk_fraction_per_trade"], fee, slip)
            existing_risk = sum(p["planned_loss"] for p in positions.values())
            gross_notional = sum(p["qty"] * data[s]["mark"][timestamp].open for s, p in positions.items())
            risk_per_unit = per_unit_stop_risk(entry, stop, fee, slip)
            max_risk_qty = max(0, equity * config["portfolio_stop_risk_fraction"] - existing_risk) / risk_per_unit
            max_notional_qty = max(0, equity * config["portfolio_gross_notional_fraction"] - gross_notional) / entry
            reserved_margin = sum(p["margin"] for p in positions.values())
            free_margin = max(0, equity - reserved_margin)
            max_margin_qty = free_margin / (entry / config["leverage"] + entry * fee)
            qty = min(qty, max_risk_qty, max_notional_qty, max_margin_qty, source["max_qty"])
            qty = math.floor((qty + 1e-12) / source["step"]) * source["step"]
            if qty < source["min_qty"] or qty * entry < source["min_notional"]:
                skipped["quantity_or_risk_cap"] += 1
                continue
            position = {
                "entry": entry, "direction": signal.direction, "qty": qty,
                "entry_time": timestamp, "signal_time": signal.time_ms + FOUR_HOURS,
                "stop": stop, "initial_stop": stop, "atr": signal.atr,
                "margin": qty * entry / config["leverage"], "entry_fee": qty * entry * fee,
                "funding": 0.0, "planned_loss": qty * risk_per_unit, "best_close_return": 0.0,
                "funding_mark_proxy_events": 0,
            }
            liq = liquidation_price(position, config["maintenance_margin_fraction_assumption"])
            if signal.direction * (stop - liq) / entry < config["minimum_liquidation_distance_buffer"]:
                skipped["liquidation_buffer"] += 1
                continue
            cash -= position["entry_fee"]
            positions[symbol] = position
            equity = current_equity(timestamp)

        for symbol, position in list(positions.items()):
            source = data[symbol]
            bar, mark = source["trade"][timestamp], source["mark"][timestamp]
            side = position["direction"]
            # Exact-open settlements are known before path evaluation. For later
            # events, exit timing is unknown: charge adverse debits, defer credits
            # until the hour ends, and withhold them if the position exits sooner.
            position["pending_funding_credits"] = []
            for event in source["funding_by_hour"].get(timestamp, []):
                event_time = int(event["fundingTime"])
                if event_time <= timestamp:
                    continue
                settlement_mark = event.get("markPrice")
                if settlement_mark is None:
                    # Aster's public rate history omits exact settlement marks.
                    # Use known hour-open mark, and record this approximation.
                    settlement_mark = mark.open
                debit = side * position["qty"] * float(settlement_mark) * float(event["fundingRate"])
                if event_time > timestamp and debit < 0:
                    position["pending_funding_credits"].append((debit, event.get("markPrice") is None))
                    continue
                if event.get("markPrice") is None:
                    position["funding_mark_proxy_events"] += 1
                position["funding"] += debit
                cash -= debit
            liq = liquidation_price(position, config["maintenance_margin_fraction_assumption"])
            target = take_profit_trigger(position, config, fee, slip, source["tick"])
            mark_worst = mark.low if side == 1 else mark.high
            trade_worst = bar.low if side == 1 else bar.high
            trade_best = bar.high if side == 1 else bar.low
            if side * (mark_worst - liq) <= 0:
                close_position(symbol, timestamp, liq, "liquidation_stress")
            elif side * (trade_worst - position["stop"]) <= 0:
                close_position(symbol, timestamp, position["stop"], "stop")
            elif side * (trade_best - target) >= 0:
                close_position(symbol, timestamp, target, "target")
            else:
                for credit, used_proxy in position["pending_funding_credits"]:
                    position["funding"] += credit
                    cash -= credit
                    if used_proxy:
                        position["funding_mark_proxy_events"] += 1
                position["pending_funding_credits"] = []
                next_open = timestamp + step
                update = source["closed_updates"].get(next_open)
                if update is not None:
                    close_price, atr = update
                    gain = side * (close_price / position["entry"] - 1)
                    position["best_close_return"] = max(position["best_close_return"], gain)
                    if position["best_close_return"] >= config["trail_activation_underlying_return"] and atr is not None:
                        lock = position["entry"] * (1 + side * config["trail_locked_underlying_return"])
                        trailing = close_price - side * config["trail_atr"] * atr
                        candidate = max(lock, trailing) if side == 1 else min(lock, trailing)
                        if side * (candidate - position["stop"]) > 0:
                            position["stop"] = round_tick(candidate, source["tick"], up=side == 1)
                if position["funding"] > position["margin"] * config["max_funding_debit_fraction_initial_margin"]:
                    position["scheduled_exit"] = "funding_budget"

        equity = current_equity(timestamp, close=True)
        peak = max(peak, equity)
        drawdown = max(0, (peak - equity) / peak)
        max_drawdown = max(max_drawdown, drawdown)
        equity_path.append({"time_utc": iso(timestamp + step), "equity": equity,
                            "drawdown_pct": drawdown * 100, "positions": len(positions)})
        if drawdown >= config["account_hard_drawdown_fraction"] or equity <= 0:
            halted = True
            halt_time = halt_time or iso(timestamp + step)
            for position in positions.values():
                position["scheduled_exit"] = "account_hard_stop"

    final_mark_equity = equity_path[-1]["equity"] if equity_path else cash
    remaining = []
    for symbol, position in positions.items():
        last_timestamp = max(time for time in data[symbol]["trade"] if time < end_ms)
        last = data[symbol]["trade"][last_timestamp]
        estimate_fill = last.close * (1 - position["direction"] * slip)
        estimate_net = position["direction"] * position["qty"] * (estimate_fill - position["entry"]) - position["entry_fee"] - position["funding"] - position["qty"] * estimate_fill * fee
        remaining.append({"symbol": symbol, "entry_utc": iso(position["entry_time"]),
                          "estimated_close_net_pnl": estimate_net,
                          "estimated_net_return_initial_margin_pct": estimate_net / position["margin"] * 100})
    gains = sum(max(0, trade["net_pnl"]) for trade in trades)
    losses = -sum(min(0, trade["net_pnl"]) for trade in trades)
    summary = {
        "start_utc": iso(start_ms), "end_utc_exclusive": iso(end_ms),
        "cost_multiplier": cost_multiplier, "initial_equity": config["initial_equity_usdt"],
        "execution_step_ms": step,
        "final_mark_equity": final_mark_equity,
        "account_return_pct": (final_mark_equity / config["initial_equity_usdt"] - 1) * 100,
        "max_sampled_drawdown_pct": max_drawdown * 100,
        "closed_trades": len(trades),
        "target_trades_net_at_least_120pct_margin": sum(t["net_return_initial_margin_pct"] >= 120 - 1e-7 for t in trades),
        "win_rate_pct": sum(t["net_pnl"] > 0 for t in trades) / len(trades) * 100 if trades else None,
        "profit_factor": gains / losses if losses > 0 else None,
        "liquidation_stress_count": sum(t["exit_reason"] == "liquidation_stress" for t in trades),
        "exit_reasons": dict(Counter(t["exit_reason"] for t in trades)),
        "fees_closed_trades": sum(t["entry_fee"] + t["exit_fee"] for t in trades),
        "funding_debit_closed_trades": sum(t["funding_debit"] for t in trades),
        "funding_mark_proxy_events_closed_trades": sum(t["funding_mark_proxy_events"] for t in trades),
        "uncredited_ambiguous_funding_events": sum(t["uncredited_ambiguous_funding_events"] for t in trades),
        "skipped_signals": dict(skipped), "halt_time": halt_time,
        "open_positions": remaining,
        "symbol_summaries": {},
    }
    for symbol in config["symbols"]:
        subset = [trade for trade in trades if trade["symbol"] == symbol]
        summary["symbol_summaries"][symbol] = {
            "closed_trades": len(subset),
            "target_trades": sum(t["net_return_initial_margin_pct"] >= 120 - 1e-7 for t in subset),
            "net_closed_pnl": sum(t["net_pnl"] for t in subset),
        }
    return {"summary": summary, "trades": trades, "equity_path": equity_path}


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/stock_swing_120_candidate_20261004.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(gzip.decompress(args.snapshot.read_bytes()))
    config = json.loads(args.config.read_text())
    if snapshot["venue"] != config["venue"]:
        raise ValueError("Strategy venue must match the actual snapshot venue")
    end = snapshot["end_ms_exclusive"]
    start = min(source["start_ms"] for source in snapshot["symbols"].values())
    split = parse_time(config["evaluation_split_utc"])
    outputs = {}
    for label, first, last, multiple in (
        ("full", start, end, 1), ("full_double_cost", start, end, 2),
        ("full_margin_stress", start, end, 1),
        ("before_september", start, split, 1), ("september_onward", split, end, 1),
        ("september_onward_double_cost", split, end, 2),
        ("september_onward_margin_stress", split, end, 1),
    ):
        run_config = dict(config)
        if "margin_stress" in label:
            run_config["maintenance_margin_fraction_assumption"] = config["maintenance_margin_fraction_stress"]
        result = simulate(snapshot, run_config, first, last, multiple)
        outputs[label] = result["summary"]
        write_csv(args.output_dir / f"{label}_trades.csv", result["trades"])
        write_csv(args.output_dir / f"{label}_equity.csv", result["equity_path"])
        print(label, json.dumps(result["summary"], ensure_ascii=False), flush=True)
    artifact = {
        "candidate": config, "snapshot_path": str(args.snapshot),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "backtest_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "signal_source_sha256": hashlib.sha256(Path(__file__).with_name("stock_swing_signals.py").read_bytes()).hexdigest(),
        "limitations": [
            "Historical replay, not independent forward evidence; daily hindsight intervals are not strategy trades.",
            "Fees and maintenance margin are assumptions; no verified historical leverage brackets or order-book fills.",
            "Aster funding history omits settlement mark prices: actual rates/timestamps are used with hour-open mark price as an explicit approximation, not exact settlement amounts.",
            "One-hour intrabar paths are unknown: liquidation stress before stop before target. Inhour funding debits are charged upfront; credits are deferred and omitted for ambiguous exits.",
            "Portfolio risk and account drawdowns are sampled hourly; gaps can exceed planned stop loss.",
            "Current quantity/tick rules used historically; historical rule changes not reconstructed.",
            "No historical earnings/news halt calendar filter; index and contract bases checked at entry.",
            "Open positions are marked and reported separately, not counted as closed target trades.",
        ],
        "runs": outputs,
    }
    (args.output_dir / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
