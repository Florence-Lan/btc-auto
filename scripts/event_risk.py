from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RiskEvent:
    event_id: str
    published_at_ms: int
    starts_at_ms: int
    ends_at_ms: int
    severity: float
    block_entries: bool
    category: str
    headline: str


@dataclass(frozen=True)
class EventDecision:
    risk_multiplier: float
    allowed: bool
    active_event_ids: tuple[str, ...]


def _utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def load_event_snapshot(path: Path) -> tuple[RiskEvent, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported event snapshot schema")
    events: list[RiskEvent] = []
    seen: set[str] = set()
    for raw in payload.get("events", []):
        event_id = str(raw["event_id"])
        if event_id in seen:
            raise ValueError(f"Duplicate event_id: {event_id}")
        seen.add(event_id)
        published = _utc_ms(str(raw["published_at_utc"]))
        starts = _utc_ms(str(raw["starts_at_utc"]))
        ends = _utc_ms(str(raw["ends_at_utc"]))
        severity = float(raw["severity"])
        if not 0 <= severity <= 1:
            raise ValueError(f"Event severity must be in [0, 1]: {event_id}")
        if ends <= starts or published > ends:
            raise ValueError(f"Invalid event time ordering: {event_id}")
        events.append(
            RiskEvent(
                event_id=event_id,
                published_at_ms=published,
                starts_at_ms=starts,
                ends_at_ms=ends,
                severity=severity,
                block_entries=bool(raw.get("block_entries", False)),
                category=str(raw.get("category", "unknown")),
                headline=str(raw.get("headline", "")),
            )
        )
    return tuple(sorted(events, key=lambda event: event.starts_at_ms))


def event_decision_at(events: Sequence[RiskEvent], timestamp_ms: int) -> EventDecision:
    active = tuple(
        event
        for event in events
        if event.published_at_ms <= timestamp_ms <= event.ends_at_ms
        and event.starts_at_ms <= timestamp_ms
    )
    if not active:
        return EventDecision(1.0, True, ())
    allowed = not any(event.block_entries for event in active)
    worst_severity = max(event.severity for event in active)
    multiplier = max(0.25, 1.0 - 0.75 * worst_severity)
    return EventDecision(
        risk_multiplier=multiplier,
        allowed=allowed,
        active_event_ids=tuple(event.event_id for event in active),
    )


def apply_event_overlay(
    sleeve_results: Sequence[Mapping[str, Any]],
    events: Sequence[RiskEvent],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    decisions = 0
    blocked = 0
    throttled = 0
    for sleeve in sleeve_results:
        trades: list[dict[str, Any]] = []
        for raw in sleeve.get("trades", []):
            timestamp_ms = _utc_ms(str(raw["entry_time_utc"]))
            decision = event_decision_at(events, timestamp_ms)
            decisions += 1
            if not decision.allowed:
                blocked += 1
                continue
            trade = dict(raw)
            if decision.risk_multiplier < 1:
                throttled += 1
                for field in (
                    "initial_qty",
                    "pnl",
                    "fees",
                    "net_pnl",
                    "funding_pnl",
                    "slippage_cost",
                ):
                    if field in trade:
                        trade[field] = float(trade[field]) * decision.risk_multiplier
                reason = str(trade.get("signal_reason", ""))
                ids = ",".join(decision.active_event_ids)
                trade["signal_reason"] = (
                    f"{reason} event={ids} x={decision.risk_multiplier:.3f}"
                ).strip()
            trades.append(trade)
        item = dict(sleeve)
        item["trades"] = trades
        adjusted.append(item)
    return adjusted, {
        "events_loaded": len(events),
        "decisions": decisions,
        "blocked": blocked,
        "throttled": throttled,
    }
