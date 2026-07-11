from __future__ import annotations

import bisect
import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class MacroSeries:
    available_times_ms: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class MacroSnapshot:
    series: Mapping[str, MacroSeries]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MacroDecision:
    score: float
    risk_multiplier: float
    allowed: bool
    available_factors: tuple[str, ...]
    contributions: Mapping[str, float]
    asof_ms: int | None


DEFAULT_WEIGHTS: dict[str, float] = {
    "equities": 0.40,
    "vix": 0.25,
    "dollar": 0.20,
    "metals": 0.15,
    "sentiment": 0.10,
}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


def save_macro_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def load_macro_snapshot(path: Path) -> MacroSnapshot:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_series = payload.get("series")
    if not isinstance(raw_series, dict) or not raw_series:
        raise ValueError("Macro snapshot must contain non-empty series")
    parsed: dict[str, MacroSeries] = {}
    for name, rows in raw_series.items():
        if not isinstance(rows, list):
            raise ValueError(f"Macro series {name} must be a list")
        clean = sorted(
            (int(row[0]), float(row[1]))
            for row in rows
            if isinstance(row, list) and len(row) == 2 and math.isfinite(float(row[1]))
        )
        if not clean:
            raise ValueError(f"Macro series {name} has no valid observations")
        times, values = zip(*clean)
        if len(set(times)) != len(times):
            raise ValueError(f"Macro series {name} contains duplicate timestamps")
        parsed[name] = MacroSeries(tuple(times), tuple(values))
    return MacroSnapshot(parsed, payload.get("metadata", {}))


def _index_at(series: MacroSeries, timestamp_ms: int) -> int:
    return bisect.bisect_right(series.available_times_ms, timestamp_ms) - 1


def _return_at(
    series: MacroSeries,
    timestamp_ms: int,
    lookback: int,
    max_staleness_days: int,
) -> tuple[float, int] | None:
    index = _index_at(series, timestamp_ms)
    previous = index - lookback
    if previous < 0:
        return None
    asof = series.available_times_ms[index]
    if timestamp_ms - asof > max_staleness_days * MS_PER_DAY:
        return None
    old = series.values[previous]
    new = series.values[index]
    if old <= 0 or new <= 0:
        return None
    return new / old - 1.0, asof


def _level_at(
    series: MacroSeries,
    timestamp_ms: int,
    max_staleness_days: int,
) -> tuple[float, int] | None:
    index = _index_at(series, timestamp_ms)
    if index < 0:
        return None
    asof = series.available_times_ms[index]
    if timestamp_ms - asof > max_staleness_days * MS_PER_DAY:
        return None
    return series.values[index], asof


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def macro_decision_at(
    snapshot: MacroSnapshot,
    timestamp_ms: int,
    *,
    enabled_factors: Sequence[str] = tuple(DEFAULT_WEIGHTS),
    min_multiplier: float = 0.35,
    block_score: float = -0.80,
    max_staleness_days: int = 5,
) -> MacroDecision:
    if not 0 < min_multiplier <= 1:
        raise ValueError("min_multiplier must be in (0, 1]")
    if not -1 <= block_score <= 0:
        raise ValueError("block_score must be in [-1, 0]")
    enabled = set(enabled_factors)
    contributions: dict[str, float] = {}
    asofs: list[int] = []

    if "equities" in enabled:
        equity_values: list[float] = []
        for name in ("sp500", "nasdaq"):
            series = snapshot.series.get(name)
            result = _return_at(series, timestamp_ms, 5, max_staleness_days) if series else None
            if result is not None:
                value, asof = result
                equity_values.append(_clamp(value / 0.05))
                asofs.append(asof)
        if equity_values:
            contributions["equities"] = _mean(equity_values)

    if "vix" in enabled:
        series = snapshot.series.get("vix")
        change = _return_at(series, timestamp_ms, 5, max_staleness_days) if series else None
        level = _level_at(series, timestamp_ms, max_staleness_days) if series else None
        if change is not None and level is not None:
            change_value, change_asof = change
            level_value, level_asof = level
            change_score = -_clamp(change_value / 0.30)
            level_score = -_clamp((level_value - 20.0) / 15.0)
            contributions["vix"] = 0.65 * change_score + 0.35 * level_score
            asofs.extend((change_asof, level_asof))

    if "dollar" in enabled:
        series = snapshot.series.get("dollar")
        result = _return_at(series, timestamp_ms, 5, max_staleness_days) if series else None
        if result is not None:
            value, asof = result
            contributions["dollar"] = -_clamp(value / 0.02)
            asofs.append(asof)

    if "metals" in enabled:
        gold = snapshot.series.get("gold")
        silver = snapshot.series.get("silver")
        gold_result = _return_at(gold, timestamp_ms, 5, max_staleness_days) if gold else None
        silver_result = _return_at(silver, timestamp_ms, 5, max_staleness_days) if silver else None
        if gold_result is not None and silver_result is not None:
            gold_return, gold_asof = gold_result
            silver_return, silver_asof = silver_result
            # Silver outperforming gold is treated as cyclical/risk-on; gold leadership as defensive.
            contributions["metals"] = _clamp((silver_return - gold_return) / 0.05)
            asofs.extend((gold_asof, silver_asof))

    if "sentiment" in enabled:
        series = snapshot.series.get("fear_greed")
        change = _return_at(series, timestamp_ms, 5, max_staleness_days) if series else None
        level = _level_at(series, timestamp_ms, max_staleness_days) if series else None
        if change is not None and level is not None:
            change_value, change_asof = change
            level_value, level_asof = level
            level_score = _clamp((level_value - 50.0) / 35.0)
            change_score = _clamp(change_value / 0.50)
            contributions["sentiment"] = 0.70 * level_score + 0.30 * change_score
            asofs.extend((change_asof, level_asof))

    weights = {name: DEFAULT_WEIGHTS[name] for name in contributions}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return MacroDecision(0.0, min_multiplier, False, (), {}, None)
    weighted_score = sum(
        contributions[name] * weights[name] for name in contributions
    ) / total_weight
    # Cross-market stress must not be averaged away by one temporarily strong market.
    # A downside-veto component keeps the overlay conservative while preserving the
    # single-factor interpretation used by the ablation report.
    worst_score = min(contributions.values())
    score = _clamp(
        0.50 * weighted_score + 0.50 * worst_score
        if len(contributions) > 1
        else weighted_score
    )
    risk_multiplier = min_multiplier + (1.0 - min_multiplier) * ((score + 1.0) / 2.0)
    return MacroDecision(
        score=score,
        risk_multiplier=_clamp(risk_multiplier, min_multiplier, 1.0),
        allowed=score > block_score,
        available_factors=tuple(sorted(contributions)),
        contributions=contributions,
        asof_ms=min(asofs) if asofs else None,
    )


def _scaled_trade(raw: Mapping[str, Any], scale: float, decision: MacroDecision) -> dict[str, Any]:
    trade = dict(raw)
    for field in (
        "initial_qty",
        "pnl",
        "fees",
        "net_pnl",
        "funding_pnl",
        "slippage_cost",
    ):
        if field in trade:
            trade[field] = float(trade[field]) * scale
    reason = str(trade.get("signal_reason", ""))
    trade["signal_reason"] = f"{reason} macro={decision.score:.3f} x={scale:.3f}".strip()
    trade["macro_score"] = decision.score
    trade["macro_risk_multiplier"] = scale
    trade["macro_factors"] = list(decision.available_factors)
    return trade


def apply_macro_overlay(
    sleeve_results: Sequence[Mapping[str, Any]],
    snapshot: MacroSnapshot,
    *,
    enabled_factors: Sequence[str] = tuple(DEFAULT_WEIGHTS),
    min_multiplier: float = 0.35,
    block_score: float = -0.80,
    max_staleness_days: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    decisions = 0
    blocked = 0
    multipliers: list[float] = []
    factor_coverage: dict[str, int] = {name: 0 for name in enabled_factors}
    for sleeve in sleeve_results:
        trades: list[dict[str, Any]] = []
        for raw in sleeve.get("trades", []):
            entry_ms = int(
                datetime.fromisoformat(str(raw["entry_time_utc"]).replace("Z", "+00:00")).timestamp()
                * 1000
            )
            decision = macro_decision_at(
                snapshot,
                entry_ms,
                enabled_factors=enabled_factors,
                min_multiplier=min_multiplier,
                block_score=block_score,
                max_staleness_days=max_staleness_days,
            )
            decisions += 1
            for factor in decision.available_factors:
                factor_coverage[factor] = factor_coverage.get(factor, 0) + 1
            if not decision.allowed:
                blocked += 1
                continue
            multipliers.append(decision.risk_multiplier)
            trades.append(_scaled_trade(raw, decision.risk_multiplier, decision))
        item = dict(sleeve)
        item["trades"] = trades
        adjusted.append(item)
    diagnostics = {
        "decisions": decisions,
        "blocked": blocked,
        "average_risk_multiplier": _mean(multipliers),
        "min_risk_multiplier": min(multipliers) if multipliers else None,
        "max_risk_multiplier": max(multipliers) if multipliers else None,
        "factor_coverage_pct": {
            name: count / decisions * 100 if decisions else 0.0
            for name, count in factor_coverage.items()
        },
    }
    return adjusted, diagnostics
