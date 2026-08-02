#!/usr/bin/env python
"""Backtest a static delta-neutral BTC spot/perpetual funding carry.

The portfolio buys BTC spot and shorts the same BTC notional in the perpetual.
Directional BTC exposure is approximately neutral, so the modeled return is
funding received by the short perpetual less execution costs. This is a
screening model: the frozen dataset does not contain spot/perpetual basis, so
the report must not be treated as live-ready.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import download_spot_snapshot as spot_snapshot
import frozen_strategy
import simulate_range_swing as sim


MS_PER_DAY = 86_400_000


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(description="Backtest static BTC spot/perpetual funding carry.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "config" / "frozen_strategy_active_20260720.json",
    )
    parser.add_argument("--days", type=float, default=2000.0)
    parser.add_argument(
        "--spot-snapshot",
        type=Path,
        help="Immutable Binance spot snapshot used for synchronized basis mark-to-market.",
    )
    parser.add_argument("--basis-interval", default="1h")
    parser.add_argument(
        "--notional-fraction",
        type=float,
        default=0.5,
        help="Matched notional per leg as a fraction of total capital.",
    )
    parser.add_argument("--spot-fee", type=float, default=0.001)
    parser.add_argument("--futures-fee", type=float, default=0.00045)
    parser.add_argument("--slippage-bps-per-leg", type=float, default=1.5)
    parser.add_argument(
        "--basis-stress-bps",
        type=float,
        default=20.0,
        help="Additional adverse terminal-basis stress charged after observed basis PnL.",
    )
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.004)
    parser.add_argument(
        "--rebalance-days",
        type=float,
        default=0.0,
        help="Close and reopen both legs on this cadence; 0 keeps the static research screen.",
    )
    parser.add_argument(
        "--rebalance-margin-buffer-pct",
        type=float,
        default=0.0,
        help="Rebalance when the isolated futures margin buffer falls to this percentage; 0 disables.",
    )
    parser.add_argument("--double-cost", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def round_trip_cost(
    notional: float,
    spot_fee: float,
    futures_fee: float,
    slippage_bps_per_leg: float,
    basis_stress_bps: float,
) -> float:
    fees = notional * 2 * (spot_fee + futures_fee)
    slippage = notional * 4 * slippage_bps_per_leg / 10_000
    basis_stress = notional * basis_stress_bps / 10_000
    return fees + slippage + basis_stress


def funding_credit(notional: float, rate: float) -> float:
    # A positive Binance funding rate is paid by longs and received by shorts.
    return notional * rate


def execution_cost(
    spot_notional: float,
    futures_notional: float,
    *,
    spot_fee: float,
    futures_fee: float,
    slippage_bps_per_leg: float,
) -> float:
    slippage = slippage_bps_per_leg / 10_000
    return (
        spot_notional * (spot_fee + slippage)
        + futures_notional * (futures_fee + slippage)
    )


def basis_bps(spot_price: float, futures_price: float) -> float:
    if spot_price <= 0 or futures_price <= 0:
        raise ValueError("spot and futures prices must be positive")
    return (futures_price / spot_price - 1.0) * 10_000


def synchronized_events(
    times: Sequence[int],
    rates: Sequence[float],
    spot_candles: Sequence[sim.Candle],
    futures_candles: Sequence[sim.Candle],
    *,
    evaluation_start_ms: int,
    interval_ms: int,
) -> list[tuple[int, float, float, float]]:
    spot_open = {candle.open_time_ms: candle.open for candle in spot_candles}
    futures_open = {candle.open_time_ms: candle.open for candle in futures_candles}
    events: list[tuple[int, float, float, float]] = []
    for timestamp_ms, rate in zip(times, rates):
        if timestamp_ms < evaluation_start_ms:
            continue
        candle_open_ms = timestamp_ms // interval_ms * interval_ms
        spot_price = spot_open.get(candle_open_ms)
        futures_price = futures_open.get(candle_open_ms)
        if spot_price is None or futures_price is None:
            continue
        events.append((timestamp_ms, float(rate), spot_price, futures_price))
    return events


def annual_returns(curve: Sequence[dict[str, float]], initial_equity: float) -> dict[str, float]:
    year_end: dict[str, float] = {}
    for point in curve:
        year = str(datetime.fromtimestamp(point["time_ms"] / 1000, tz=timezone.utc).year)
        year_end[year] = point["equity"]
    result: dict[str, float] = {}
    previous = initial_equity
    for year in sorted(year_end):
        ending = year_end[year]
        result[year] = (ending / previous - 1.0) * 100 if previous else -100.0
        previous = ending
    return result


def backtest_with_basis(
    times: Sequence[int],
    rates: Sequence[float],
    spot_candles: Sequence[sim.Candle],
    futures_candles: Sequence[sim.Candle],
    *,
    evaluation_start_ms: int,
    interval_ms: int,
    initial_equity: float = 100.0,
    notional_fraction: float = 0.5,
    spot_fee: float = 0.001,
    futures_fee: float = 0.00045,
    slippage_bps_per_leg: float = 1.5,
    basis_stress_bps: float = 20.0,
    maintenance_margin_pct: float = 0.004,
    rebalance_days: float = 0.0,
    rebalance_margin_buffer_pct: float = 0.0,
) -> dict[str, Any]:
    events = synchronized_events(
        times,
        rates,
        spot_candles,
        futures_candles,
        evaluation_start_ms=evaluation_start_ms,
        interval_ms=interval_ms,
    )
    if len(events) < 2:
        raise ValueError("At least two synchronized spot/futures funding events are required")
    if not 0 < notional_fraction < 1:
        raise ValueError("notional_fraction must be between 0 and 1 for separate-leg margin")
    if rebalance_days < 0:
        raise ValueError("rebalance_days must be >= 0")
    if rebalance_margin_buffer_pct < 0:
        raise ValueError("rebalance_margin_buffer_pct must be >= 0")

    entry_time, _entry_rate, first_spot, first_futures = events[0]
    cycle_start_equity = initial_equity
    entry_spot = first_spot
    entry_futures = first_futures
    spot_entry_notional = cycle_start_equity * notional_fraction
    quantity = spot_entry_notional / entry_spot
    futures_entry_notional = quantity * entry_futures
    cycle_entry_cost = execution_cost(
        spot_entry_notional,
        futures_entry_notional,
        spot_fee=spot_fee,
        futures_fee=futures_fee,
        slippage_bps_per_leg=slippage_bps_per_leg,
    )
    futures_collateral = cycle_start_equity - spot_entry_notional - cycle_entry_cost
    if futures_collateral <= 0:
        raise ValueError("notional_fraction leaves no futures collateral after entry costs")

    total_funding_pnl = 0.0
    cycle_funding_pnl = 0.0
    realized_basis_pnl = 0.0
    total_costs = cycle_entry_cost
    funding_returns: list[float] = []
    peak = initial_equity
    max_drawdown = 0.0
    min_margin_buffer_pct = float("inf")
    liquidation_time_ms: int | None = None
    annual_funding: dict[str, float] = defaultdict(float)
    positive_events = 0
    curve: list[dict[str, float]] = []
    basis_values: list[float] = []
    entry_basis_values = [basis_bps(entry_spot, entry_futures)]
    rebalance_count = 0
    rebalance_ms = int(rebalance_days * MS_PER_DAY) if rebalance_days else 0
    next_rebalance_ms = entry_time + rebalance_ms if rebalance_ms else 0

    for index, (timestamp_ms, rate, spot_price, futures_price) in enumerate(events):
        if index:
            credit = funding_credit(quantity * futures_price, rate)
            cycle_funding_pnl += credit
            total_funding_pnl += credit
            funding_returns.append(credit / initial_equity)
            annual_funding[str(datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).year)] += credit
            if rate > 0:
                positive_events += 1
        spot_pnl = quantity * (spot_price - entry_spot)
        futures_pnl = quantity * (entry_futures - futures_price)
        equity = cycle_start_equity - cycle_entry_cost + spot_pnl + futures_pnl + cycle_funding_pnl
        futures_equity = futures_collateral + futures_pnl + cycle_funding_pnl
        futures_notional = quantity * futures_price
        maintenance = futures_notional * maintenance_margin_pct
        margin_buffer_pct = (futures_equity - maintenance) / futures_notional * 100
        min_margin_buffer_pct = min(min_margin_buffer_pct, margin_buffer_pct)
        if liquidation_time_ms is None and futures_equity <= maintenance:
            liquidation_time_ms = timestamp_ms
        current_basis_bps = basis_bps(spot_price, futures_price)
        basis_values.append(current_basis_bps)
        rebalanced = False
        scheduled_rebalance = rebalance_ms and timestamp_ms >= next_rebalance_ms
        margin_rebalance = (
            rebalance_margin_buffer_pct > 0
            and margin_buffer_pct <= rebalance_margin_buffer_pct
        )
        if index < len(events) - 1 and (scheduled_rebalance or margin_rebalance):
            close_cost = execution_cost(
                quantity * spot_price,
                quantity * futures_price,
                spot_fee=spot_fee,
                futures_fee=futures_fee,
                slippage_bps_per_leg=slippage_bps_per_leg,
            )
            equity -= close_cost
            total_costs += close_cost
            realized_basis_pnl += spot_pnl + futures_pnl
            cycle_start_equity = equity
            entry_spot = spot_price
            entry_futures = futures_price
            spot_entry_notional = cycle_start_equity * notional_fraction
            quantity = spot_entry_notional / entry_spot
            futures_entry_notional = quantity * entry_futures
            cycle_entry_cost = execution_cost(
                spot_entry_notional,
                futures_entry_notional,
                spot_fee=spot_fee,
                futures_fee=futures_fee,
                slippage_bps_per_leg=slippage_bps_per_leg,
            )
            total_costs += cycle_entry_cost
            futures_collateral = cycle_start_equity - spot_entry_notional - cycle_entry_cost
            cycle_funding_pnl = 0.0
            equity = cycle_start_equity - cycle_entry_cost
            futures_notional = quantity * futures_price
            maintenance = futures_notional * maintenance_margin_pct
            futures_equity = futures_collateral
            margin_buffer_pct = (futures_equity - maintenance) / futures_notional * 100
            min_margin_buffer_pct = min(min_margin_buffer_pct, margin_buffer_pct)
            entry_basis_values.append(current_basis_bps)
            rebalance_count += 1
            next_rebalance_ms = timestamp_ms + rebalance_ms
            rebalanced = True
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
        curve.append(
            {
                "time_ms": timestamp_ms,
                "equity": equity,
                "spot_price": spot_price,
                "futures_price": futures_price,
                "basis_bps": current_basis_bps,
                "funding_rate": rate,
                "futures_margin_buffer_pct": margin_buffer_pct,
                "rebalanced": rebalanced,
            }
        )

    exit_time, _exit_rate, exit_spot, exit_futures = events[-1]
    spot_exit_notional = quantity * exit_spot
    futures_exit_notional = quantity * exit_futures
    final_exit_cost = execution_cost(
        spot_exit_notional,
        futures_exit_notional,
        spot_fee=spot_fee,
        futures_fee=futures_fee,
        slippage_bps_per_leg=slippage_bps_per_leg,
    )
    stress_cost = spot_exit_notional * basis_stress_bps / 10_000
    final_equity = curve[-1]["equity"] - final_exit_cost - stress_cost
    total_costs += final_exit_cost + stress_cost
    curve[-1]["equity"] = final_equity
    max_drawdown = max(max_drawdown, (peak - final_equity) / peak if peak else 0.0)
    years = max((exit_time - entry_time) / (365.25 * MS_PER_DAY), 1 / 365.25)
    final_multiple = final_equity / initial_equity
    observed_basis_pnl = realized_basis_pnl + quantity * (
        (entry_futures - entry_spot) - (exit_futures - exit_spot)
    )
    funding_vol = pstdev(funding_returns) if len(funding_returns) > 1 else 0.0
    periods_per_year = len(funding_returns) / years
    sharpe = mean(funding_returns) / funding_vol * math.sqrt(periods_per_year) if funding_vol else None
    yearly = annual_returns(curve, initial_equity)
    profitable_year_pct = (
        sum(value > 0 for value in yearly.values()) / len(yearly) * 100 if yearly else 0.0
    )
    return {
        "summary": {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return_pct": (final_multiple - 1.0) * 100,
            "cagr_pct": (final_multiple ** (1 / years) - 1.0) * 100 if final_multiple > 0 else -100.0,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_funding_events": sharpe,
            "funding_events": len(events) - 1,
            "positive_funding_pct": positive_events / (len(events) - 1) * 100,
            "gross_funding_pnl": total_funding_pnl,
            "observed_basis_pnl": observed_basis_pnl,
            "total_costs": total_costs,
            "entry_basis_bps": entry_basis_values[0],
            "exit_basis_bps": basis_values[-1],
            "average_entry_basis_bps": mean(entry_basis_values),
            "min_basis_bps": min(basis_values),
            "max_basis_bps": max(basis_values),
            "min_futures_margin_buffer_pct": min_margin_buffer_pct,
            "liquidated": liquidation_time_ms is not None,
            "liquidation_time_utc": (
                sim.iso_utc_from_ms(liquidation_time_ms) if liquidation_time_ms is not None else None
            ),
            "annual_funding_pnl": dict(annual_funding),
            "annual_returns_pct": yearly,
            "profitable_year_pct": profitable_year_pct,
            "rebalances": rebalance_count,
        },
        "equity_curve": curve,
        "config": {
            "structure": "long_spot_short_usdm_perpetual",
            "spot_entry_notional": spot_entry_notional,
            "futures_entry_notional": futures_entry_notional,
            "quantity_btc": quantity,
            "notional_fraction": notional_fraction,
            "spot_fee": spot_fee,
            "futures_fee": futures_fee,
            "slippage_bps_per_leg": slippage_bps_per_leg,
            "basis_stress_bps": basis_stress_bps,
            "maintenance_margin_pct": maintenance_margin_pct,
            "rebalance_days": rebalance_days,
            "rebalance_margin_buffer_pct": rebalance_margin_buffer_pct,
            "margin_model": "separate spot capital and USD-M futures collateral; no automatic transfer",
        },
        "limitations": [
            "Hourly opening prices approximate synchronized executable spot and perpetual marks.",
            "Order-book impact, margin transfers, collateral yield, tax, and custody risk are not modeled.",
            "A recorded margin breach invalidates the implementation even if combined portfolio equity stays positive.",
            "This report is research-only and cannot authorize live orders.",
        ],
    }


def backtest(
    times: Sequence[int],
    rates: Sequence[float],
    *,
    evaluation_start_ms: int,
    initial_equity: float = 100.0,
    notional_fraction: float = 0.5,
    spot_fee: float = 0.001,
    futures_fee: float = 0.00045,
    slippage_bps_per_leg: float = 1.5,
    basis_stress_bps: float = 20.0,
) -> dict[str, Any]:
    events = [(time_ms, rate) for time_ms, rate in zip(times, rates) if time_ms >= evaluation_start_ms]
    if not events:
        raise ValueError("No funding events in evaluation window")
    notional = initial_equity * notional_fraction
    costs = round_trip_cost(
        notional,
        spot_fee,
        futures_fee,
        slippage_bps_per_leg,
        basis_stress_bps,
    )
    equity = initial_equity - costs / 2
    peak = initial_equity
    max_drawdown = (peak - equity) / peak
    curve: list[dict[str, float]] = []
    credits: list[float] = []
    annual_funding: dict[str, float] = {}
    positive_events = 0

    for time_ms, rate in events:
        credit = funding_credit(notional, rate)
        credits.append(credit)
        equity += credit
        if rate > 0:
            positive_events += 1
        year = str(datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).year)
        annual_funding[year] = annual_funding.get(year, 0.0) + credit
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
        curve.append({"time_ms": time_ms, "equity": equity, "funding_rate": rate})

    equity -= costs / 2
    curve[-1]["equity"] = equity
    max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
    years = max((events[-1][0] - events[0][0]) / (365.25 * MS_PER_DAY), 1 / 365.25)
    event_returns = [credit / initial_equity for credit in credits]
    event_vol = pstdev(event_returns) if len(event_returns) > 1 else 0.0
    periods_per_year = len(events) / years
    sharpe = mean(event_returns) / event_vol * math.sqrt(periods_per_year) if event_vol else None
    final_multiple = equity / initial_equity
    summary = {
        "initial_equity": initial_equity,
        "final_equity": equity,
        "total_return_pct": (final_multiple - 1) * 100,
        "cagr_pct": (final_multiple ** (1 / years) - 1) * 100 if final_multiple > 0 else -100.0,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_funding_events": sharpe,
        "funding_events": len(events),
        "positive_funding_pct": positive_events / len(events) * 100,
        "gross_funding_pnl": sum(credits),
        "total_costs": costs,
        "annual_funding_pnl": annual_funding,
    }
    return {
        "summary": summary,
        "equity_curve": curve,
        "config": {
            "structure": "long_spot_short_perpetual",
            "notional_per_leg": notional,
            "notional_fraction": notional_fraction,
            "spot_fee": spot_fee,
            "futures_fee": futures_fee,
            "slippage_bps_per_leg": slippage_bps_per_leg,
            "basis_stress_bps": basis_stress_bps,
        },
        "limitations": [
            "Frozen data contains Binance perpetual funding but no synchronized BTC spot prices.",
            "Directional PnL is assumed to cancel between equal spot and perpetual notionals.",
            "Basis path, spot custody, transfer, borrow, and capital opportunity costs are not modeled.",
            "This report is research-only and cannot authorize live orders.",
        ],
    }


def main() -> int:
    args = parse_args()
    if args.days <= 0 or not 0 < args.notional_fraction <= 1:
        raise ValueError("--days must be > 0 and --notional-fraction must be within (0, 1]")
    if args.maintenance_margin_pct < 0:
        raise ValueError("--maintenance-margin-pct must be >= 0")
    if args.rebalance_days < 0:
        raise ValueError("--rebalance-days must be >= 0")
    if args.rebalance_margin_buffer_pct < 0:
        raise ValueError("--rebalance-margin-buffer-pct must be >= 0")
    manifest, _ = frozen_strategy.load_frozen_strategy(args.manifest.resolve())
    snapshot_path = (sim.repo_root() / manifest["snapshot_path"]).resolve()
    frozen_strategy.verify_snapshot(manifest, snapshot_path)
    intervals, funding, metadata = sim.load_market_snapshot(snapshot_path)
    evaluation_start_ms = max(
        funding.times[0],
        funding.times[-1] - int(args.days * MS_PER_DAY),
    )
    multiplier = 2.0 if args.double_cost else 1.0
    if args.spot_snapshot:
        spot_candles, spot_metadata = spot_snapshot.load_spot_snapshot(args.spot_snapshot)
        spot_symbol = sim.normalize_symbol(str(spot_metadata.get("symbol", "")))
        futures_symbol = sim.normalize_symbol(str(metadata.get("symbol", "")))
        if spot_symbol != futures_symbol:
            raise ValueError(
                f"Spot/futures symbol mismatch: {spot_symbol or 'unknown'} != {futures_symbol or 'unknown'}"
            )
        spot_interval = str(spot_metadata.get("interval", ""))
        if spot_interval != args.basis_interval:
            raise ValueError(
                f"Spot snapshot interval {spot_interval or 'unknown'} != --basis-interval {args.basis_interval}"
            )
        futures_candles = intervals.get(args.basis_interval)
        if not futures_candles:
            raise ValueError(f"Futures snapshot has no {args.basis_interval} candles")
        result = backtest_with_basis(
            funding.times,
            funding.rates,
            spot_candles,
            futures_candles,
            evaluation_start_ms=evaluation_start_ms,
            interval_ms=sim.interval_to_ms(args.basis_interval),
            notional_fraction=args.notional_fraction,
            spot_fee=args.spot_fee * multiplier,
            futures_fee=args.futures_fee * multiplier,
            slippage_bps_per_leg=args.slippage_bps_per_leg * multiplier,
            basis_stress_bps=args.basis_stress_bps,
            maintenance_margin_pct=args.maintenance_margin_pct,
            rebalance_days=args.rebalance_days,
            rebalance_margin_buffer_pct=args.rebalance_margin_buffer_pct,
        )
        result["spot_snapshot"] = str(args.spot_snapshot.resolve())
        result["spot_snapshot_metadata"] = spot_metadata
        result["model"] = "synchronized_basis_mark_to_market"
    else:
        result = backtest(
            funding.times,
            funding.rates,
            evaluation_start_ms=evaluation_start_ms,
            notional_fraction=args.notional_fraction,
            spot_fee=args.spot_fee * multiplier,
            futures_fee=args.futures_fee * multiplier,
            slippage_bps_per_leg=args.slippage_bps_per_leg * multiplier,
            basis_stress_bps=args.basis_stress_bps,
        )
        result["model"] = "funding_only_screen"
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["source"] = {
        "title": "Fundamentals of Perpetual Futures",
        "authors": "Songrun He, Asaf Manela, Omri Ross, Victor von Wachter",
        "ssrn": "4301150",
    }
    result["snapshot_metadata"] = metadata
    result["cost_scenario"] = "double" if args.double_cost else "normal"
    summary = result["summary"]
    print("Static BTC spot/perpetual funding carry")
    print(f"Model: {result['model']}")
    print(f"Return / CAGR: {summary['total_return_pct']:.2f}% / {summary['cagr_pct']:.2f}%")
    print(f"Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"Positive funding events: {summary['positive_funding_pct']:.2f}%")
    print(f"Gross funding / costs: {summary['gross_funding_pnl']:.2f} / {summary['total_costs']:.2f}")
    if result["model"] == "synchronized_basis_mark_to_market":
        print(
            f"Observed basis PnL: {summary['observed_basis_pnl']:.2f}; "
            f"entry / exit basis: {summary['entry_basis_bps']:.2f} / {summary['exit_basis_bps']:.2f} bps"
        )
        print(
            f"Futures margin breach: {summary['liquidated']}; "
            f"minimum buffer: {summary['min_futures_margin_buffer_pct']:.2f}%"
        )
        print(f"Rebalances: {summary['rebalances']}")
    else:
        print("Warning: synchronized spot/perpetual basis is not available; screening result is incomplete.")
    print("Research only: this module never places orders.")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
