"""Public news/calendar ingestion with first-seen archives and explicit coverage."""
from __future__ import annotations

import argparse
import bisect
import calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import requests

import event_risk

UTC = timezone.utc
ET_ZONE = ZoneInfo("America/New_York")
FED_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
NYFED_CALENDAR = "https://www.newyorkfed.org/research/calendars/nationalecon_cal"
BLS_CALENDAR = "https://www.bls.gov/schedule/news_release/bls.ics"
RSS = {
    "fed_news": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "ecb_news": "https://www.ecb.europa.eu/rss/press.html",
    "un_news": "https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml",
}
REQUIRED = (*RSS, "fomc_calendar", "economic_calendar", "rate_expectations")


def iso(value):
    return value.astimezone(UTC).isoformat()


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def text_only(value):
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def fetch(url):
    response = requests.get(url, headers={"User-Agent": "btc-auto-public-research/1.0"}, timeout=20)
    response.raise_for_status()
    return response.content.decode("utf-8-sig")


def news_category(source, title):
    if source in {"fed_news", "ecb_news"}:
        if re.search(r"FOMC statement|monetary policy decisions|interest rates|economic projections", title, re.I):
            return "monetary_policy", .6
        return "central_bank_communication", .25
    if re.search(r"\b(war|attack\w*|airstrike\w*|missile\w*|sanction\w*|fighting|conflict|nuclear|blockade|strait)\b", title, re.I):
        return "geopolitical_headline", .35
    return "international_news", 0.0


def parse_rss(body, source, url, now):
    root = ET.fromstring(body)
    records = []
    for item in root.findall("./channel/item"):
        title = text_only(item.findtext("title") or "")[:500]
        link = (item.findtext("link") or "").strip()
        published_text = item.findtext("pubDate")
        if not title or not link.startswith("https://") or not published_text:
            continue
        published = parsedate_to_datetime(published_text)
        if published.tzinfo is None:
            raise ValueError("RSS publication time must include timezone")
        if published > now + timedelta(minutes=5):
            continue
        category, severity = news_category(source, title)
        identity = digest(source + ":" + (item.findtext("guid") or link))
        records.append({"news_id": identity, "source": source, "source_url": url,
                        "article_url": link, "headline": title, "published_at_utc": iso(published),
                        "first_seen_at_utc": iso(now), "category": category, "severity": severity})
    if not records:
        raise ValueError("RSS contained no dated news items")
    return records


def calendar_item(identity, headline, scheduled, source, url, now, *, time_basis="source"):
    return {"calendar_id": identity, "headline": headline, "scheduled_at_utc": iso(scheduled),
            "source": source, "source_url": url, "first_seen_at_utc": iso(now), "time_basis": time_basis}


def parse_fomc(body, now):
    headings = list(re.finditer(r"(20\d{2}) FOMC Meetings", body))
    records = []
    for index, heading in enumerate(headings):
        year = int(heading.group(1))
        if year < now.year - 1 or year > now.year + 1:
            continue
        section = body[heading.end():headings[index + 1].start() if index + 1 < len(headings) else len(body)]
        pattern = r'fomc-meeting__month[^>]*>(.*?)</div>\s*<div[^>]*fomc-meeting__date[^>]*>(.*?)</div>'
        for month_html, days_html in re.findall(pattern, section, re.S):
            month = text_only(month_html).split("/")[-1]
            days = text_only(days_html)
            if "notation" in days.lower() or "cancel" in days.lower():
                continue
            numbers = re.findall(r"\d+", days)
            if not numbers:
                continue
            try:
                month_number = list(calendar.month_name).index(month)
                scheduled = datetime(year, month_number, int(numbers[-1]), 14, tzinfo=ET_ZONE)
            except ValueError:
                continue
            date = scheduled.date().isoformat()
            records.append(calendar_item("fomc:" + date, "FOMC policy decision", scheduled,
                           "fomc_calendar", FED_CALENDAR, now, time_basis="regular_meeting_14_ET_convention"))
    if not any(utc(row["scheduled_at_utc"]) > now for row in records):
        raise ValueError("FOMC calendar contains no future meetings")
    return records


ECONOMIC_NAMES = r"Consumer Price Index|Employment Situation|Producer Price Index|Job Openings and Labor Turnover|Personal Income|Gross Domestic Product"


def parse_nyfed(body, url, now):
    match = re.search(r'class="ts-data-table-head"[^>]*>\s*<div[^>]*>([A-Za-z]+) (20\d{2})</div>', body)
    if not match:
        raise ValueError("NY Fed calendar month not found")
    month = list(calendar.month_name).index(match.group(1))
    year = int(match.group(2))
    records = []
    for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", body, re.S | re.I):
        day = re.search(r"<div[^>]*>\s*(\d{1,2})\s*<br", cell)
        if not day:
            continue
        for title, hour, minute in re.findall(r"<a\b[^>]*>(.*?)</a>\s*<br\s*/?>\s*\((\d{2}):(\d{2})\)", cell, re.S):
            title = text_only(title)
            if not re.search(ECONOMIC_NAMES, title, re.I):
                continue
            scheduled = datetime(year, month, int(day.group(1)), int(hour), int(minute), tzinfo=ET_ZONE)
            identity = "econ:" + digest(title + ":" + scheduled.date().isoformat())
            records.append(calendar_item(identity, title, scheduled, "economic_calendar", url, now))
    if not records:
        raise ValueError("NY Fed calendar contains no recognized timed releases")
    return records


def parse_bls_ics(body, now):
    # Unfold RFC5545 lines; BLS does not need recurrence expansion.
    body = re.sub(r"\r?\n[ \t]", "", body)
    records = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", body, re.S):
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        title = fields.get("SUMMARY", "").replace("\\,", ",")
        if not re.search(ECONOMIC_NAMES, title, re.I):
            continue
        starts = next(((k, v) for k, v in fields.items() if k.split(";")[0] == "DTSTART"), None)
        if not starts:
            continue
        key, value = starts
        if value.endswith("Z"):
            scheduled = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        elif "TZID=" in key:
            tz = key.split("TZID=", 1)[1].split(";")[0].strip('"')
            scheduled = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=ZoneInfo(tz))
        else:
            raise ValueError("BLS calendar event has no explicit timezone")
        identity = "bls:" + fields.get("UID", digest(title + value))
        records.append(calendar_item(identity, title, scheduled, "economic_calendar", BLS_CALENDAR, now))
    if not records:
        raise ValueError("BLS calendar has no supported timed releases")
    return records


def economic_calendar(now):
    errors = []
    try:
        rows = parse_bls_ics(fetch(BLS_CALENDAR), now)
        if any(utc(row["scheduled_at_utc"]) > now for row in rows):
            return rows, errors
        errors.append("BLS calendar has no upcoming supported releases")
    except (requests.RequestException, ValueError, KeyError) as exc:
        errors.append(f"BLS: {type(exc).__name__}: {str(exc)[:160]}")
    # Official alternate calendar, not a fabricated release-date recurrence.
    first = now.astimezone(ET_ZONE).replace(day=1)
    next_month = (first + timedelta(days=32)).replace(day=1)
    rows = []
    for month in (first, next_month):
        url = "https://www.newyorkfed.org/research/calendars/i-" + month.strftime("%b%y").lower() + ".html"
        rows.extend(parse_nyfed(fetch(url), url, now))
    if not any(utc(row["scheduled_at_utc"]) > now for row in rows):
        raise ValueError("Economic calendar has no upcoming supported releases")
    return rows, errors


def merge_calendar(previous, incoming, now):
    """Append revisions; supersede only from the time a change was observed."""
    output = [dict(row) for row in previous]
    incoming_ids = {row["calendar_id"] for row in incoming}
    successful_urls = {row["source_url"] for row in incoming}
    for row in output:
        if (not row.get("superseded_at_utc") and row["source_url"] in successful_urls
                and row["calendar_id"] not in incoming_ids and utc(row["scheduled_at_utc"]) > now):
            row["superseded_at_utc"] = iso(now)
    for raw in incoming:
        old = next((row for row in reversed(output) if row["calendar_id"] == raw["calendar_id"]
                    and not row.get("superseded_at_utc")), None)
        if old and old["scheduled_at_utc"] == raw["scheduled_at_utc"] and old["headline"] == raw["headline"]:
            continue
        if old:
            old["superseded_at_utc"] = iso(now)
        output.append({**raw, "revision_id": digest(raw["calendar_id"] + raw["scheduled_at_utc"] + iso(now))})
    return output


def make_events(news, calendars):
    events = []
    for row in news:
        if row["severity"] <= 0:
            continue
        published = utc(row["published_at_utc"])
        events.append({"event_id": "news:" + row["news_id"], "published_at_utc": iso(published),
                       "available_at_utc": row["first_seen_at_utc"], "starts_at_utc": iso(published),
                       "ends_at_utc": iso(published + timedelta(hours=6)),
                       "severity": row["severity"], "block_entries": False,
                       "category": row["category"], "headline": row["headline"],
                       "source_url": row["article_url"]})
    for row in calendars:
        scheduled = utc(row["scheduled_at_utc"])
        # Previously unseen historic calendars cannot act on earlier trades.
        if utc(row["first_seen_at_utc"]) >= scheduled + timedelta(minutes=90):
            continue
        events.append({"event_id": "calendar:" + row["revision_id"],
                       "published_at_utc": row["first_seen_at_utc"],
                       "available_at_utc": row["first_seen_at_utc"],
                       "starts_at_utc": iso(scheduled - timedelta(minutes=30)),
                       "ends_at_utc": iso(scheduled + timedelta(minutes=90)),
                       "superseded_at_utc": row.get("superseded_at_utc"),
                       "severity": .8, "block_entries": True, "category": "scheduled_macro",
                       "headline": row["headline"], "source_url": row["source_url"]})
    return events


def quote_contract(year, month, now):
    symbol = f"ZQ{'FGHJKMNQUVXZ'[month - 1]}{year % 100:02d}.CBT"
    last_error = None
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            data = json.loads(fetch(f"https://{host}/v8/finance/chart/{symbol}?interval=1d&range=5d"))
            chart = data["chart"]["result"][0]
            meta = chart["meta"]
            if meta["symbol"] != symbol or meta["instrumentType"] != "FUTURE":
                raise ValueError("Unexpected futures contract identity")
            price = float(meta["regularMarketPrice"])
            asof = datetime.fromtimestamp(int(meta["regularMarketTime"]), UTC)
            if not math.isfinite(price) or not 80 < price < 105 or not timedelta(0) <= now - asof <= timedelta(days=5):
                raise ValueError("Futures quote invalid, future-dated or stale")
            return {"contract": symbol, "contract_month": f"{year}-{month:02d}", "price": price,
                    "quote_at_utc": iso(asof), "implied_monthly_effr_pct": 100 - price,
                    "source_url": f"https://finance.yahoo.com/quote/{symbol}/"}
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            last_error = exc
    raise ValueError(f"Futures quote unavailable: {type(last_error).__name__}")


def last_factor(payload, name, now):
    timestamp = int(now.timestamp() * 1000)
    rows = [r for r in payload.get("series", {}).get(name, []) if max(r[1], r[3]) <= timestamp]
    if not rows:
        raise ValueError(f"No available {name}")
    row = max(rows, key=lambda r: r[0])
    if timestamp - row[0] > 7 * 86_400_000:
        raise ValueError(f"Stale {name}")
    return float(row[2])


def rate_expectation(calendars, factor_payload, now):
    meetings = sorted(utc(row["scheduled_at_utc"]) for row in calendars
                      if row["source"] == "fomc_calendar" and not row.get("superseded_at_utc")
                      and utc(row["scheduled_at_utc"]) > now)
    if not meetings:
        raise ValueError("No next FOMC meeting")
    meeting = meetings[0]
    local = meeting.astimezone(ET_ZONE)
    next_month = (local.replace(day=1) + timedelta(days=32)).replace(day=1)
    effr = last_factor(factor_payload, "fed_effective", now)
    target = last_factor(factor_payload, "fed_target", now)
    # Prefer a full nonmeeting month anchor. Otherwise solve the single next meeting's
    # monthly weighted average. Both are assumptions, not CME probability distributions.
    following_has_meeting = any((m.astimezone(ET_ZONE).year, m.astimezone(ET_ZONE).month) ==
                                (next_month.year, next_month.month) for m in meetings)
    fallback_reason = None
    if not following_has_meeting:
        try:
            quote = quote_contract(next_month.year, next_month.month, now)
        except ValueError as exc:
            fallback_reason = str(exc)
    if following_has_meeting or fallback_reason:
        same_month = [row for row in calendars if row["source"] == "fomc_calendar"
                      and not row.get("superseded_at_utc")
                      and utc(row["scheduled_at_utc"]).astimezone(ET_ZONE).strftime("%Y-%m") == local.strftime("%Y-%m")]
        total_days = calendar.monthrange(local.year, local.month)[1]
        after_days = total_days - local.day
        if len(same_month) != 1 or after_days < 3:
            raise ValueError("Cannot reliably isolate this meeting from its monthly futures quote")
        quote = quote_contract(local.year, local.month, now)
        implied_end = (quote["implied_monthly_effr_pct"] * total_days - effr * local.day) / after_days
        method = "single_meeting_month_day_weighted_effr"
    else:
        implied_end = quote["implied_monthly_effr_pct"]
        method = "next_full_nonmeeting_month_effr_anchor"
    if not -5 < implied_end < 20:
        raise ValueError("Implausible implied postmeeting rate")
    return {**quote, "meeting_at_utc": iso(meeting), "available_at_utc": iso(now),
            "reference_effr_pct": effr, "reference_target_upper_pct": target,
            "implied_postmeeting_effr_pct": implied_end,
            "expected_change_bps": (implied_end - effr) * 100,
            "method": method, "fallback_reason": fallback_reason,
            "limitations": "Delayed vendor quote; assumes no unscheduled policy move or EFFR/target basis shift. Not CME FedWatch probabilities."}


def expectation_at(payload, timestamp_ms):
    now = datetime.fromtimestamp(timestamp_ms / 1000, UTC)
    rows = [r for r in payload.get("rate_expectations", []) if utc(r["available_at_utc"]) <= now
            and utc(r["meeting_at_utc"]) > now and now - utc(r["quote_at_utc"]) <= timedelta(days=5)
            and now - utc(r["available_at_utc"]) <= timedelta(hours=2)]
    return max(rows, key=lambda r: r["available_at_utc"]) if rows else None


def policy_surprises(expectations, factors, now):
    """Compare an archived PRE-meeting expected change with the released target change."""
    result = []
    for meeting in sorted({r["meeting_at_utc"] for r in expectations if utc(r["meeting_at_utc"]) <= now}):
        candidates = [r for r in expectations if r["meeting_at_utc"] == meeting
                      and utc(r["available_at_utc"]) < utc(meeting)
                      and utc(meeting) - utc(r["quote_at_utc"]) <= timedelta(days=5)]
        if not candidates:
            continue
        expected = max(candidates, key=lambda r: r["available_at_utc"])
        decision_ms = int(utc(meeting).timestamp() * 1000)
        released = [r for r in factors.get("series", {}).get("fed_target", [])
                    if decision_ms < r[0] <= decision_ms + 4 * 86_400_000
                    and max(r[1], r[3]) <= int(now.timestamp() * 1000)]
        if not released:
            continue
        actual = min(released, key=lambda r: r[0])
        actual_change = (actual[2] - expected["reference_target_upper_pct"]) * 100
        result.append({"meeting_at_utc": meeting,
                       "available_at_utc": iso(datetime.fromtimestamp(max(actual[1], actual[3]) / 1000, UTC)),
                       "expected_change_bps": expected["expected_change_bps"],
                       "actual_target_change_bps": actual_change,
                       "surprise_bps": actual_change - expected["expected_change_bps"],
                       "expectation_available_at_utc": expected["available_at_utc"],
                       "method": "released_target_change_minus_archived_expected_effr_change"})
    return result


def health_at(payload, timestamp_ms):
    checks = payload.get("coverage_checks", [])
    times = [event_risk._utc_ms(row["available_at_utc"]) for row in checks]
    index = bisect.bisect_right(times, timestamp_ms) - 1
    if index < 0:
        return False, ["coverage_not_observed"]
    row = checks[index]
    if timestamp_ms - times[index] > 60 * 60_000:
        return False, ["public_context_stale"]
    missing = [name for name in REQUIRED if not row["sources"].get(name, {}).get("ok")]
    return not missing, missing


def apply_entry_coverage(sleeves, payload):
    adjusted, blocked = [], 0
    for sleeve in sleeves:
        trades = []
        for trade in sleeve.get("trades", []):
            healthy, _ = health_at(payload, event_risk._utc_ms(trade["entry_time_utc"]))
            if healthy:
                trades.append(trade)
            else:
                blocked += 1
        adjusted.append({**sleeve, "trades": trades})
    return adjusted, blocked


def collect(output, factor_path):
    previous = json.loads(output.read_text()) if output.exists() else {}
    now = datetime.now(UTC)
    jobs = {name: (lambda n=name, u=url: parse_rss(fetch(u), n, u, datetime.now(UTC))) for name, url in RSS.items()}
    jobs["fomc_calendar"] = lambda: parse_fomc(fetch(FED_CALENDAR), datetime.now(UTC))
    jobs["economic_calendar"] = lambda: economic_calendar(datetime.now(UTC))
    status, collected = {}, {}

    def run(item):
        name, fn = item
        try:
            return name, fn(), None
        except (requests.RequestException, ValueError, KeyError, ET.ParseError) as exc:
            return name, None, f"{type(exc).__name__}: {str(exc)[:240]}"

    with ThreadPoolExecutor(max_workers=5) as pool:
        for name, value, error in pool.map(run, jobs.items()):
            status[name] = {"ok": error is None, "error": error}
            if error is None:
                collected[name] = value
    now = datetime.now(UTC)
    news = {row["news_id"]: row for row in previous.get("news", [])}
    for source in RSS:
        for row in collected.get(source, []):
            news.setdefault(row["news_id"], row)
    economic, fallback_errors = collected.get("economic_calendar", ([], []))
    status["economic_calendar"]["fallback_errors"] = fallback_errors
    calendars = merge_calendar(previous.get("calendars", []), collected.get("fomc_calendar", []) + economic, now)
    expectations = list(previous.get("rate_expectations", []))
    factors = {}
    try:
        with gzip.open(factor_path, "rt", encoding="utf-8") as handle:
            factors = json.load(handle)
        expectation = rate_expectation(calendars, factors, now)
        # Receipt, not request start, determines first possible forward use.
        expectation["available_at_utc"] = iso(datetime.now(UTC))
        expectations.append(expectation)
        status["rate_expectations"] = {"ok": True, "contract": expectation["contract"]}
    except (OSError, ValueError, KeyError) as exc:
        status["rate_expectations"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    now = datetime.now(UTC)
    payload = {"schema_version": 1, "generated_at_utc": iso(now), "news": list(news.values()),
               "calendars": calendars, "rate_expectations": expectations,
               "policy_surprises": policy_surprises(expectations, factors, now),
               "events": make_events(list(news.values()), calendars),
               "coverage_checks": previous.get("coverage_checks", []) + [{"available_at_utc": iso(now), "sources": status}],
               "metadata": {"source_status": status, "news_classification": "headline_rules_risk_only",
                            "economic_consensus_surprises": "unavailable_no_verified_consensus_feed",
                            "policy_surprise_status": "requires_archived_premeeting_expectation_and_released_actual",
                            "sources": {**RSS, "fomc_calendar": FED_CALENDAR, "economic_calendar": NYFED_CALENDAR}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    event_risk.load_event_snapshot(temporary)
    os.replace(temporary, output)
    for name, state in status.items():
        print(f"{name}: {'ok' if state['ok'] else state.get('error')}", flush=True)
    print(f"news={len(news)} calendar_versions={len(calendars)} events={len(payload['events'])}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/snapshots/public_context_latest.json"))
    parser.add_argument("--factor-snapshot", type=Path, default=Path("data/snapshots/multifactor_latest.json.gz"))
    args = parser.parse_args()
    payload = collect(args.output, args.factor_snapshot)
    healthy, _ = health_at(payload, int(datetime.now(UTC).timestamp() * 1000))
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
