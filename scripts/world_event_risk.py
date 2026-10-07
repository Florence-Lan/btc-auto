"""Point-in-time, research-only geopolitical event overlay.

News discovery is not fact verification. Only reviewed primary evidence or two
reviewed independent source groups can change the shadow portfolio's entry size.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

HOUR_MS = 3_600_000
RULE_VERSION = "world-events-v1"
SCALE_FIELDS = ("initial_qty", "pnl", "fees", "net_pnl", "funding_pnl", "slippage_cost")


def utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("World event timestamps must include a timezone")
    return int(parsed.timestamp() * 1000)


def iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()


def empty_snapshot() -> dict[str, Any]:
    return {"schema_version": 1, "rule_version": RULE_VERSION,
            "research_only": True, "observations": [], "polls": []}


def validate_snapshot(payload: Mapping[str, Any]) -> None:
    if (payload.get("schema_version") != 1 or payload.get("rule_version") != RULE_VERSION
            or payload.get("research_only") is not True):
        raise ValueError("Unsupported world event snapshot")
    if not isinstance(payload.get("observations"), list) or not isinstance(payload.get("polls"), list):
        raise ValueError("observations and polls must be lists")
    seen: set[str] = set()
    for row in payload["observations"]:
        for field in ("id", "event_id", "source_group", "source_url", "headline", "category"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field}")
        if row["id"] in seen:
            raise ValueError("Duplicate observation id")
        seen.add(row["id"])
        if not row["source_url"].startswith(("https://", "http://")):
            raise ValueError("Evidence requires an HTTP(S) source URL")
        first = utc_ms(row["first_seen_at_utc"])
        assessed = utc_ms(row["assessed_at_utc"])
        if assessed < first:
            raise ValueError("Assessment precedes first receipt")
        for field in ("severity", "surprise", "novelty", "confidence"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {field}")
        half_life = row["half_life_hours"]
        if isinstance(half_life, bool) or not isinstance(half_life, (float, int)) or not math.isfinite(half_life) or not 0 < half_life <= 168:
            raise ValueError("Invalid half_life_hours")
        for field in ("verified", "primary_source"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        if row["verified"] and not str(row.get("evidence_note", "")).strip():
            raise ValueError("Reviewed evidence requires a note")
    for poll in payload["polls"]:
        utc_ms(poll["available_at_utc"])
        if poll["status"] not in ("ok", "error", "truncated"):
            raise ValueError("Invalid feed status")


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_snapshot(payload)
    return payload


def save_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    validate_snapshot(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class WorldDecision:
    score: float
    watch_score: float
    risk_multiplier: float
    allowed: bool
    feed_status: str
    active_event_ids: tuple[str, ...]
    confirmed_event_ids: tuple[str, ...]
    candidate_count: int
    feed_provider: str | None
    feed_coverage: str | None


def decision_at(snapshot: Mapping[str, Any], timestamp_ms: int) -> WorldDecision:
    # Assessments are append-only revisions. A later review never rewrites history.
    latest: dict[str, Mapping[str, Any]] = {}
    anchors: dict[str, int] = {}
    for row in sorted(snapshot["observations"], key=lambda r: utc_ms(r["assessed_at_utc"])):
        if max(utc_ms(row["first_seen_at_utc"]), utc_ms(row["assessed_at_utc"])) <= timestamp_ms:
            latest[row["source_url"]] = row
            first_seen = utc_ms(row["first_seen_at_utc"])
            anchors[row["event_id"]] = min(anchors.get(row["event_id"], first_seen), first_seen)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in latest.values():
        grouped.setdefault(row["event_id"], []).append(row)
    scores: list[float] = []
    watches: list[float] = []
    active: list[str] = []
    confirmed: list[str] = []
    for event_id, rows in grouped.items():
        # Reposts/confirmations do not restart the clock for the same event.
        anchor = anchors[event_id]
        age_hours = (timestamp_ms - anchor) / HOUR_MS
        if age_hours >= 168:
            continue
        def strength(row: Mapping[str, Any]) -> float:
            return (100 * row["severity"] * row["surprise"] * row["novelty"]
                    * row["confidence"] * 2 ** (-age_hours / row["half_life_hours"]))
        watch = max(strength(row) for row in rows)
        if watch < 1:
            continue
        watches.append(watch)
        active.append(event_id)
        reviewed = [row for row in rows if row["verified"] and row["confidence"] >= 0.7]
        independent = {row["source_group"] for row in reviewed}
        if any(row["primary_source"] for row in reviewed) or len(independent) >= 2:
            confirmed.append(event_id)
            scores.append(max(strength(row) for row in reviewed))
    # Max aggregation avoids escalating risk merely because of article volume.
    score = max(scores, default=0.0)
    watch_score = max(watches, default=0.0)
    polls = [p for p in snapshot["polls"] if utc_ms(p["available_at_utc"]) <= timestamp_ms]
    latest_poll = max(polls, key=lambda p: utc_ms(p["available_at_utc"]), default=None)
    status = "unavailable"
    if latest_poll is not None:
        status = latest_poll["status"]
        if timestamp_ms - utc_ms(latest_poll["available_at_utc"]) > 2 * HOUR_MS:
            status = "stale"
    # Prototype: unknown feed blocks new shadow entries; it never closes a position.
    # Event risk itself only reduces size. No directional trades or news-only halt.
    allowed = status == "ok"
    multiplier = max(0.25, 1 - 0.75 * score / 100) if allowed else 0.0
    return WorldDecision(score, watch_score, multiplier, allowed, status,
                         tuple(sorted(active)), tuple(sorted(confirmed)), len(latest),
                         latest_poll.get("provider") if latest_poll else None,
                         latest_poll.get("coverage") if latest_poll else None)


def apply_overlay(
    sleeves: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    blocked = throttled = decisions = 0
    for sleeve in sleeves:
        from execution_ledger import attach_ledger
        sleeve = attach_ledger(sleeve)
        trades: list[dict[str, Any]] = []
        for raw in sleeve.get("trades", []):
            decision = decision_at(snapshot, utc_ms(str(raw["entry_time_utc"])))
            decisions += 1
            if not decision.allowed:
                blocked += 1
                continue
            trade = dict(raw)
            previous = min(float(trade.get("macro_risk_multiplier", 1.0)),
                           float(trade.get("multifactor_risk_multiplier", 1.0)))
            if not math.isfinite(previous) or not 0 < previous <= 1:
                raise ValueError("Invalid upstream macro multiplier")
            combined = min(previous, decision.risk_multiplier)
            ratio = combined / previous
            if ratio < 1:
                throttled += 1
                for field in SCALE_FIELDS:
                    if field in trade:
                        trade[field] = float(trade[field]) * ratio
            trade["world_event_score"] = decision.score
            trade["world_event_ids"] = list(decision.confirmed_event_ids)
            trade["world_event_risk_multiplier"] = decision.risk_multiplier
            trade["macro_world_risk_multiplier"] = combined
            trade["signal_reason"] = (str(trade.get("signal_reason", ""))
                                      + f" world={decision.score:.2f} combined={combined:.3f}").strip()
            trades.append(trade)
        adjusted.append({**sleeve, "trades": trades})
    return adjusted, {"rule_version": RULE_VERSION, "research_only": True,
                      "decisions": decisions, "blocked_feed": blocked, "throttled": throttled,
                      "observations_loaded": len(snapshot["observations"])}


def report_at(snapshot: Mapping[str, Any], timestamp_ms: int) -> dict[str, Any]:
    return {"rule_version": RULE_VERSION, "research_only": True, "places_orders": False,
            "asof_utc": iso(timestamp_ms), **asdict(decision_at(snapshot, timestamp_ms))}
