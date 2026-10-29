"""Source failures may use verified observations only until their original expiry."""
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_context as pc

NOW = datetime(2026, 10, 19, 16, tzinfo=timezone.utc)


def ms(t):
    return int(t.timestamp() * 1000)


def previous_snapshot():
    news = [{"news_id": name, "source": name, "headline": name,
             "source_url": url, "article_url": url,
             "published_at_utc": pc.iso(NOW - timedelta(minutes=10)),
             "first_seen_at_utc": pc.iso(NOW), "category": "test", "severity": 0}
            for name, url in pc.RSS.items()]
    calendars = [pc.calendar_item("fomc", "FOMC", NOW + timedelta(days=9),
                                  "fomc_calendar", pc.FED_CALENDAR, NOW),
                 pc.calendar_item("cpi", "Consumer Price Index", NOW + timedelta(days=2),
                                  "economic_calendar", pc.NYFED_CALENDAR, NOW)]
    calendars = pc.merge_calendar([], calendars, NOW)
    expectation = {"meeting_at_utc": pc.iso(NOW + timedelta(days=9)),
                   "available_at_utc": pc.iso(NOW), "quote_at_utc": pc.iso(NOW),
                   "expected_change_bps": 5, "contract": "ZQX26.CBT"}
    states = {name: {"ok": True} for name in pc.REQUIRED}
    states["fomc_calendar"]["last_success_at_utc"] = pc.iso(NOW)
    states["fomc_calendar"]["verified_month_pages"] = [url for _, _, url in pc._fed_month_pages(NOW)]
    return {"schema_version": 1, "news": news, "calendars": calendars,
            "rate_expectations": [expectation],
            "coverage_checks": [{"available_at_utc": pc.iso(NOW),
                                  "sources": states}]}


def outage_collect(tmp_path, monkeypatch, current, previous=None):
    snapshot = previous_snapshot() if previous is None else previous
    output = tmp_path / "context.json"
    output.write_text(json.dumps(snapshot), encoding="utf-8")
    factors = tmp_path / "factors.json.gz"
    stamp = ms(current - timedelta(days=2))
    with gzip.open(factors, "wt", encoding="utf-8") as handle:
        json.dump({"series": {name: [[stamp, stamp, value, stamp]] for name, value in
                              [("fed_effective", 3.88), ("fed_target", 4.0)]}}, handle)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current.astimezone(tz)

    def fail(url):
        raise requests.exceptions.ReadTimeout("https://user:private@proxy.invalid/?secret=private")

    monkeypatch.setattr(pc, "datetime", Clock)
    monkeypatch.setattr(pc, "fetch", fail)
    return pc.collect(output, factors)


def test_refresh_outage_preserves_unexpired_source_observations_without_renewing_them(tmp_path, monkeypatch):
    original = previous_snapshot()
    payload = outage_collect(tmp_path, monkeypatch, NOW + timedelta(minutes=5), original)
    assert pc.health_at(payload, ms(NOW + timedelta(minutes=5))) == (True, [])
    for source in pc.REQUIRED:
        status = payload["metadata"]["source_status"][source]
        assert status["ok"] and status["status"] == "cached"
        assert status["fetch_ok"] is False
        assert status["last_success_at_utc"] == pc.iso(NOW)
        assert "private" not in status["error"]
    assert payload["news"] == original["news"]
    assert payload["rate_expectations"] == original["rate_expectations"]
    assert not pc.health_at(payload, ms(NOW + timedelta(hours=1, milliseconds=1)))[0]


def test_repeated_failed_refresh_cannot_slide_cache_expiry(tmp_path, monkeypatch):
    payload = outage_collect(tmp_path, monkeypatch, NOW + timedelta(minutes=40))
    again = outage_collect(tmp_path, monkeypatch, NOW + timedelta(minutes=61), payload)
    assert not pc.health_at(again, ms(NOW + timedelta(minutes=61)))[0]
    assert all(not state["ok"] for state in again["metadata"]["source_status"].values())
    assert all(state["last_success_at_utc"] == pc.iso(NOW)
               for state in again["metadata"]["source_status"].values())


@pytest.mark.parametrize("mutation", ["missing_news", "missing_calendars", "future_success", "stale_quote"])
def test_cache_requires_real_available_and_unexpired_source_content(tmp_path, monkeypatch, mutation):
    previous = previous_snapshot()
    expected = None
    if mutation == "missing_news":
        previous["news"] = [r for r in previous["news"] if r["source"] != "fed_news"]
        expected = "fed_news"
    elif mutation == "missing_calendars":
        previous["calendars"] = []
        expected = "fomc_calendar"
    elif mutation == "future_success":
        previous["coverage_checks"][0]["sources"]["fed_news"]["last_success_at_utc"] = pc.iso(NOW + timedelta(hours=1))
        expected = "fed_news"
    else:
        previous["rate_expectations"][0]["quote_at_utc"] = pc.iso(NOW - timedelta(days=6))
        expected = "rate_expectations"
    payload = outage_collect(tmp_path, monkeypatch, NOW + timedelta(minutes=5), previous)
    assert expected in pc.health_at(payload, ms(NOW + timedelta(minutes=5)))[1]


def test_health_checks_each_cached_sources_original_success_time():
    previous = previous_snapshot()
    sources = {name: {"ok": True, "status": "cached", "fetch_ok": False,
                      "last_success_at_utc": pc.iso(NOW)} for name in pc.REQUIRED}
    sources["fed_news"]["last_success_at_utc"] = pc.iso(NOW - timedelta(minutes=59))
    previous["coverage_checks"].append({"available_at_utc": pc.iso(NOW + timedelta(seconds=30)),
                                         "sources": sources})
    assert pc.health_at(previous, ms(NOW + timedelta(seconds=30)))[0]
    assert pc.health_at(previous, ms(NOW + timedelta(minutes=2))) == (False, ["fed_news"])


def test_future_dated_futures_quote_is_not_usable_even_when_observed_now():
    snapshot = previous_snapshot()
    snapshot["rate_expectations"][0]["quote_at_utc"] = pc.iso(NOW + timedelta(seconds=1))
    assert pc.expectation_at(snapshot, ms(NOW)) is None


def test_claimed_cache_without_original_success_clock_cannot_pass_gate():
    snapshot = previous_snapshot()
    snapshot["coverage_checks"][0]["sources"]["fed_news"] = {
        "ok": True, "fetch_ok": False, "status": "cached"}
    assert pc.health_at(snapshot, ms(NOW)) == (False, ["fed_news"])
