#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


KNOWN_PREFIXES = (
    "fake_breakout",
    "wide_failure",
    "exhaustion",
    "squeeze",
    "shock",
    "sweep",
    "trend",
    "range",
)


@dataclass
class StrategyStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    fees: float = 0.0
    best: float = 0.0
    worst: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0


def strategy_name(signal_reason: str) -> str:
    for prefix in KNOWN_PREFIXES:
        if signal_reason.startswith(prefix + "_"):
            return prefix
    return signal_reason.split("_", 1)[0] if signal_reason else "unknown"


def read_trades(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def update_stats(stats: StrategyStats, pnl: float, fees: float) -> None:
    stats.trades += 1
    stats.net_pnl += pnl
    stats.fees += fees
    stats.best = pnl if stats.trades == 1 else max(stats.best, pnl)
    stats.worst = pnl if stats.trades == 1 else min(stats.worst, pnl)
    if pnl > 0:
        stats.wins += 1
        stats.gross_profit += pnl
        stats.consecutive_losses = 0
    elif pnl < 0:
        stats.losses += 1
        stats.gross_loss += abs(pnl)
        stats.consecutive_losses += 1
        stats.max_consecutive_losses = max(stats.max_consecutive_losses, stats.consecutive_losses)
    else:
        stats.consecutive_losses = 0
    stats.equity += pnl
    stats.peak_equity = max(stats.peak_equity, stats.equity)
    stats.max_drawdown = max(stats.max_drawdown, stats.peak_equity - stats.equity)


def classify(stats: StrategyStats, min_trades: int, min_profit_factor: float) -> str:
    if stats.trades < min_trades:
        return "insufficient"
    profit_factor = stats.gross_profit / stats.gross_loss if stats.gross_loss else None
    expectancy = stats.net_pnl / stats.trades if stats.trades else 0.0
    if expectancy > 0 and (profit_factor is None or profit_factor >= min_profit_factor):
        return "positive"
    if expectancy < 0:
        return "negative"
    return "mixed"


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def analyze(paths: Iterable[Path], initial_equity: float, min_trades: int, min_profit_factor: float) -> List[Dict[str, Any]]:
    grouped: Dict[str, StrategyStats] = defaultdict(StrategyStats)
    for path in paths:
        for row in read_trades(path):
            name = strategy_name(str(row.get("signal_reason", "")))
            pnl = float(row.get("net_pnl", 0.0) or 0.0)
            fees = float(row.get("fees", 0.0) or 0.0)
            update_stats(grouped[name], pnl, fees)

    rows: List[Dict[str, Any]] = []
    for name, stats in sorted(grouped.items()):
        profit_factor = stats.gross_profit / stats.gross_loss if stats.gross_loss else None
        win_rate = stats.wins / stats.trades * 100 if stats.trades else 0.0
        expectancy = stats.net_pnl / stats.trades if stats.trades else 0.0
        fee_drag = stats.fees / stats.gross_profit * 100 if stats.gross_profit else None
        rows.append(
            {
                "strategy": name,
                "class": classify(stats, min_trades, min_profit_factor),
                "trades": stats.trades,
                "net_pnl": stats.net_pnl,
                "return_pct": stats.net_pnl / initial_equity * 100 if initial_equity else 0.0,
                "expectancy": expectancy,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "best": stats.best,
                "worst": stats.worst,
                "max_dd": stats.max_drawdown,
                "max_consec_losses": stats.max_consecutive_losses,
                "fees": stats.fees,
                "fee_drag_pct": fee_drag,
            }
        )
    rows.sort(key=lambda item: (item["class"] != "positive", -item["net_pnl"]))
    return rows


def print_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No trades found.")
        return
    headers = [
        "strategy",
        "class",
        "trades",
        "net",
        "ret%",
        "exp/trade",
        "win%",
        "PF",
        "worst",
        "maxDD",
        "maxL",
        "fees",
        "fee/gp%",
    ]
    print(",".join(headers))
    for row in rows:
        print(
            ",".join(
                [
                    str(row["strategy"]),
                    str(row["class"]),
                    str(row["trades"]),
                    fmt(row["net_pnl"]),
                    fmt(row["return_pct"]),
                    fmt(row["expectancy"]),
                    fmt(row["win_rate_pct"]),
                    fmt(row["profit_factor"]),
                    fmt(row["worst"]),
                    fmt(row["max_dd"]),
                    str(row["max_consec_losses"]),
                    fmt(row["fees"]),
                    fmt(row["fee_drag_pct"]),
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-strategy expectancy from one or more trades CSV files.")
    parser.add_argument("trades_csv", nargs="+", type=Path)
    parser.add_argument("--initial-equity", type=float, default=100.0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-profit-factor", type=float, default=1.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze(args.trades_csv, args.initial_equity, args.min_trades, args.min_profit_factor)
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
