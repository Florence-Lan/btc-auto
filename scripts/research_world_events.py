#!/usr/bin/env python
"""Collect, review and replay world-event risk without exchange credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

import world_event_risk as risk

ROOT = Path(__file__).resolve().parents[1]
QUERY = '("military strike" OR "new sanctions" OR "tariff hike" OR "oil supply disruption" OR "emergency rate") sourcelang:english'
ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
BBC_ENDPOINT = "https://feeds.bbci.co.uk/news/world/rss.xml"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid article URL")
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, urlencode(sorted(query)), ""))


def ingest_articles(snapshot: dict, articles: list, received_ms: int) -> int:
    seen = {canonical_url(row["source_url"]) for row in snapshot["observations"]}
    added = 0
    for article in articles:
        url = canonical_url(article["url"])
        headline = str(article["title"]).strip()
        if not headline:
            raise ValueError("Missing article title")
        if url in seen:
            continue
        seen.add(url)
        normalized = " ".join(re.findall(r"\w+", headline.lower()))
        lowered = headline.lower()
        category = "other"
        for label, words in (("conflict", ("strike", "military", "missile", "war")),
                             ("trade_policy", ("tariff", "sanction")),
                             ("energy", ("oil", "gas", "energy")),
                             ("monetary", ("rate", "central bank"))):
            if any(re.search(r"\b" + re.escape(word) + r"s?\b", lowered) for word in words):
                category = label
                break
        snapshot["observations"].append({
            "id": digest(url), "event_id": "candidate-" + digest(normalized),
            "source_url": url, "source_group": urlsplit(url).hostname,
            "headline": headline, "category": category,
            "first_seen_at_utc": risk.iso(received_ms),
            "assessed_at_utc": risk.iso(received_ms),
            "provider_seen_at": article.get("seendate"),
            "severity": 0.0 if category == "other" else 0.5, "surprise": 0.5, "novelty": 0.5,
            "confidence": 0.3, "half_life_hours": 12,
            "verified": False, "primary_source": False,
            "evidence_note": "Unreviewed search hit; provisional watch score only.",
        })
        added += 1
    return added


def collect(snapshot: dict, session=None, provider: str = "gdelt") -> dict:
    session = session or requests
    if provider not in ("gdelt", "bbc-world"):
        raise ValueError("Unknown provider")
    status, count, added, error = "error", 0, 0, None
    try:
        params = {"query": QUERY, "mode": "artlist",
                               "format": "json", "timespan": "6h", "maxrecords": 250,
                               "sort": "datedesc"} if provider == "gdelt" else None
        response = session.get(ENDPOINT if provider == "gdelt" else BBC_ENDPOINT,
                               params=params, timeout=(10, 25),
                               headers={"User-Agent": "btc-auto-world-event-research/1.0"})
        response.raise_for_status()
        if provider == "gdelt":
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("articles"), list):
                raise ValueError("Unexpected GDELT response schema")
        else:
            root = ET.fromstring(response.content)
            if root.find("channel") is None:
                raise ValueError("Unexpected RSS response schema")
            body = {"articles": [{"url": item.findtext("link"), "title": item.findtext("title"),
                                   "seendate": item.findtext("pubDate")}
                                  for item in root.findall("./channel/item")]}
        count = len(body["articles"])
        # Only commit a complete validated batch. Never stamp historical publication
        # dates as local availability; even old articles first become usable now.
        staged = {**snapshot, "observations": list(snapshot["observations"])}
        received_ms = int(time.time() * 1000)
        added = ingest_articles(staged, body["articles"], received_ms)
        risk.validate_snapshot(staged)
        snapshot["observations"] = staged["observations"]
        status = "truncated" if provider == "gdelt" and count >= 250 else "ok"
    except (requests.RequestException, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        # Do not include arbitrary server response text in logs.
        error = type(exc).__name__
        added = 0
    poll = {"available_at_utc": risk.iso(int(time.time() * 1000)),
            "status": status, "articles_returned": count, "added": added,
            "error": error, "provider": provider,
            "query": QUERY if provider == "gdelt" else "BBC World RSS",
            "coverage": "sampled search results" if provider == "gdelt" else "single publisher RSS"}
    snapshot["polls"].append(poll)
    return poll


def review(snapshot: dict, args: argparse.Namespace) -> None:
    source = next((row for row in snapshot["observations"] if row["id"] == args.observation_id), None)
    if source is None:
        raise ValueError("Unknown observation id")
    assessed = risk.iso(int(time.time() * 1000))
    row = {**source, "id": digest(source["source_url"] + assessed),
           "assessed_at_utc": assessed, "event_id": args.event_id,
           "source_group": args.source_group, "severity": args.severity,
           "surprise": args.surprise, "novelty": args.novelty,
           "confidence": args.confidence, "half_life_hours": args.half_life_hours,
           "verified": not args.retract, "primary_source": args.primary_source,
           "evidence_note": args.evidence_note}
    snapshot["observations"].append(row)
    risk.validate_snapshot(snapshot)


def demo() -> tuple[dict, dict]:
    """Synthetic mechanism check, deliberately not a market backtest."""
    start = risk.utc_ms("2026-09-01T00:00:00Z")
    snapshot = risk.empty_snapshot()
    snapshot["synthetic"] = True
    base = {"id": "demo-a", "event_id": "synthetic-supply-shock", "source_group": "demo-primary",
            "source_url": "https://example.com/synthetic-a", "headline": "SYNTHETIC supply shock",
            "category": "energy", "first_seen_at_utc": risk.iso(start),
            "assessed_at_utc": risk.iso(start), "severity": 0.9, "surprise": 0.9,
            "novelty": 1.0, "confidence": 0.95, "half_life_hours": 6,
            "verified": True, "primary_source": True, "evidence_note": "Synthetic fixture only"}
    snapshot["observations"].append(base)
    snapshot["observations"].append({**base, "id": "demo-repost",
        "source_url": "https://example.com/synthetic-repost", "primary_source": False,
        "first_seen_at_utc": risk.iso(start + risk.HOUR_MS),
        "assessed_at_utc": risk.iso(start + risk.HOUR_MS)})
    for hour in (-1, 0, 1, 6, 12):
        snapshot["polls"].append({"available_at_utc": risk.iso(start + hour * risk.HOUR_MS), "status": "ok"})
    samples = [risk.report_at(snapshot, start + hour * risk.HOUR_MS) for hour in (-1, 0, 1, 6, 12, 15)]
    raw = {"entry_time_utc": risk.iso(start), "initial_qty": 1.0,
           "net_pnl": -100.0, "fees": 5.0, "pnl": -95.0, "signal_reason": "synthetic"}
    macro = {**raw, "initial_qty": 0.6, "net_pnl": -60.0,
             "fees": 3.0, "pnl": -57.0, "macro_risk_multiplier": 0.6}
    adjusted, diagnostics = risk.apply_overlay([{"trades": [macro]}], snapshot)
    return snapshot, {"synthetic": True, "places_orders": False,
                      "warning": "Mechanism demonstration only; no evidence of trading profitability.",
                      "samples": samples, "comparison": {"baseline": raw, "macro_only": macro,
                      "macro_plus_event": adjusted[0]["trades"][0]}, "diagnostics": diagnostics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "report", "review", "demo"))
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/world_events/observations.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/world_events/report.json")
    parser.add_argument("--asof")
    parser.add_argument("--provider", choices=("gdelt", "bbc-world"), default="gdelt")
    parser.add_argument("--observation-id")
    parser.add_argument("--event-id")
    parser.add_argument("--source-group")
    for field in ("severity", "surprise", "novelty", "confidence", "half-life-hours"):
        parser.add_argument("--" + field, type=float)
    parser.add_argument("--primary-source", action="store_true")
    parser.add_argument("--retract", action="store_true")
    parser.add_argument("--evidence-note")
    args = parser.parse_args()
    if args.snapshot.resolve() == args.output.resolve():
        parser.error("Snapshot and report must have different paths")
    exit_code = 0
    if args.command == "demo":
        # Never put synthetic observations in the operational journal.
        _, report = demo()
    else:
        if args.command != "collect" and not args.snapshot.exists():
            parser.error("Snapshot does not exist; collect first")
        snapshot = risk.load_snapshot(args.snapshot) if args.snapshot.exists() else risk.empty_snapshot()
        if args.command == "collect":
            poll = collect(snapshot, provider=args.provider)
            risk.save_snapshot(args.snapshot, snapshot)
            exit_code = 0 if poll["status"] == "ok" else 2
        elif args.command == "review":
            required = ("observation_id", "event_id", "source_group", "severity", "surprise",
                        "novelty", "confidence", "half_life_hours", "evidence_note")
            if any(getattr(args, name) is None for name in required):
                parser.error("review requires observation, event, source group, all scores, half life and evidence note")
            review(snapshot, args)
            risk.save_snapshot(args.snapshot, snapshot)
        timestamp = risk.utc_ms(args.asof) if args.asof else int(time.time() * 1000)
        report = risk.report_at(snapshot, timestamp)
        report["latest_poll"] = max((p for p in snapshot["polls"] if risk.utc_ms(p["available_at_utc"]) <= timestamp),
                                    key=lambda p: risk.utc_ms(p["available_at_utc"]), default=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
