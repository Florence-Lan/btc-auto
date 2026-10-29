"""FOMC press conferences require their own official, causally available row."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import event_risk
import public_context as pc
from test_public_context import factor_rows, NOW as FACTOR_NOW
from test_public_context_reliability import NOW as CACHE_NOW, previous_snapshot, outage_collect

UTC = timezone.utc
NOW = datetime(2026, 10, 28, 17, 24, tzinfo=UTC)


def month_html(month="October", year=2026, *, day="28", time="2:30 p.m.", extra=""):
    # Same column boundaries and title as the 2026-10-28 official HTML.
    return f'''<title>Federal Reserve Board - Calendar: {month} {year}</title>
    <div class="row"><div class="panel panel-unstyled col-xs-12"><div class="panel-body">
    <div class="row"><div class="col-xs-2"><p>{time}</p></div>
    <div class="col-xs-7"><p><a href="https://www.federalreserve.gov/live-broadcast.htm">FOMC Press Conference</a></p></div>
    <div class="col-xs-3"><p>{day}</p></div></div></div></div></div>{extra}'''


def decision(date="2026-10-28T18:00:00+00:00"):
    scheduled = pc.utc(date)
    return pc.calendar_item("fomc:" + scheduled.date().isoformat(), "FOMC policy decision",
                            scheduled, "fomc_calendar", pc.FED_CALENDAR, NOW - timedelta(days=10))


def parse(body, *, year=2026, month=10, rows=None):
    url = pc.FED_MONTH_CALENDAR.format(year=year, month=pc.calendar.month_name[month].lower())
    return pc.parse_fomc_press_conferences(body, year, month, url, rows or [decision()], NOW)


def test_real_row_creates_independent_event_without_changing_decision_or_first_seen():
    policy = decision()
    old = pc.merge_calendar([], [policy], NOW - timedelta(days=10))
    conference = parse(month_html())[0]
    assert conference["calendar_id"] == "fomc_press_conference:2026-10-28"
    assert conference["event_type"] == "press_conference"
    assert conference["scheduled_at_utc"] == "2026-10-28T18:30:00+00:00"
    assert conference["first_seen_at_utc"] == pc.iso(NOW)
    merged = pc.merge_calendar(old, [policy, conference], NOW)
    assert merged[0] == old[0]
    assert len(merged) == 2
    event = next(e for e in pc.make_events([], merged) if e["event_type"] == "press_conference")
    assert event["starts_at_utc"] == "2026-10-28T18:00:00+00:00"
    assert event["ends_at_utc"] == "2026-10-28T20:00:00+00:00"
    assert event["available_at_utc"] == pc.iso(NOW)
    assert len(pc.merge_calendar(merged, [policy, conference], NOW + timedelta(minutes=5))) == 2


def test_svg_accessibility_title_does_not_change_document_calendar_identity():
    body = month_html() + '<svg role="img"><title>Lock</title><path d=""/></svg>'
    assert parse(body)[0]["scheduled_at_utc"] == "2026-10-28T18:30:00+00:00"


def test_new_event_never_backfills_a_trade_before_its_real_first_seen(tmp_path):
    seen = datetime(2026, 10, 28, 19, 45, tzinfo=UTC)
    row = parse(month_html())[0]
    row["first_seen_at_utc"] = pc.iso(seen)
    merged = pc.merge_calendar([], [row], seen)
    path = tmp_path / "events.json"
    import json
    path.write_text(json.dumps({"schema_version": 1, "events": pc.make_events([], merged)}), encoding="utf-8")
    events = event_risk.load_event_snapshot(path)
    assert event_risk.event_decision_at(events, int((seen - timedelta(seconds=1)).timestamp() * 1000)).allowed
    assert not event_risk.event_decision_at(events, int(seen.timestamp() * 1000)).allowed
    assert event_risk.event_decision_at(events, int((seen + timedelta(minutes=15, milliseconds=1)).timestamp() * 1000)).allowed
    row["first_seen_at_utc"] = "2026-10-28T20:00:00+00:00"
    assert pc.make_events([], pc.merge_calendar([], [row], pc.utc(row["first_seen_at_utc"]))) == []


@pytest.mark.parametrize("month,day,year,clock,expected", [
    (10, "28", 2026, "2:30 p.m.", "2026-10-28T18:30:00+00:00"),
    (12, "9", 2026, "2:30 p.m. EST", "2026-12-09T19:30:00+00:00"),
    (3, "18", 2026, "2:30 p.m. EDT", "2026-03-18T18:30:00+00:00"),
    (12, "9", 2026, "19:30 UTC", "2026-12-09T19:30:00+00:00"),
])
def test_explicit_clocks_and_et_dst(month, day, year, clock, expected):
    date = f"{year}-{month:02d}-{int(day):02d}T19:00:00+00:00"
    body = month_html(pc.calendar.month_name[month], year, day=day, time=clock)
    assert parse(body, year=year, month=month, rows=[decision(date)])[0]["scheduled_at_utc"] == expected


@pytest.mark.parametrize("time", ["", "2:30", "2:30 p.m. EST", "2:30 p.m. PST", "25:30 UTC", "2:61 p.m."])
def test_missing_malformed_or_dst_conflicting_time_is_not_guessed(time):
    with pytest.raises(ValueError):
        parse(month_html(time=time))


@pytest.mark.parametrize("mutation", ["missing_title", "wrong_month", "missing_row", "wrong_day", "day_range", "duplicate", "wrong_column"])
def test_identity_and_same_row_date_are_required(mutation):
    body = month_html()
    if mutation == "missing_title": body = body.replace("<title>", "<div>").replace("</title>", "</div>")
    elif mutation == "wrong_month": body = body.replace("October 2026", "November 2026")
    elif mutation == "missing_row": body = '<title>Federal Reserve Board - Calendar: October 2026</title>'
    elif mutation == "wrong_day": body = body.replace("<p>28</p>", "<p>29</p>")
    elif mutation == "day_range": body = body.replace("<p>28</p>", "<p>27-28</p>")
    elif mutation == "duplicate": body = body + body
    elif mutation == "wrong_column": body = body.replace('class="col-xs-3"', 'class="col-xs-4"')
    with pytest.raises(ValueError): parse(body)


def test_neighbor_panel_time_and_other_meeting_text_do_not_supply_missing_clock():
    body = month_html(time="") + '''<div class="row"><div class="col-xs-2">2:30 p.m.</div>
    <div class="col-xs-7">Other event</div><div class="col-xs-3">28</div></div>'''
    with pytest.raises(ValueError): parse(body)
    body = month_html().replace("FOMC Press Conference", "FOMC Meeting Press Conference")
    with pytest.raises(ValueError): parse(body)


def test_no_meeting_month_may_be_empty_but_identity_still_verified():
    body = '<title>Federal Reserve Board - Calendar: November 2026</title>'
    assert pc.parse_fomc_press_conferences(body, 2026, 11, "https://fed/nov", [decision()], NOW) == []


@pytest.mark.parametrize("day_weighted", [False, True])
def test_press_conference_cannot_change_next_policy_meeting_or_month_weight(monkeypatch, day_weighted):
    policy = decision()
    # Preserve old rows without event_type and add press conferences in BOTH
    # the meeting month and anchor month; neither is a second policy change.
    next_policy = decision("2026-11-28T19:00:00+00:00") if day_weighted else decision("2026-12-09T19:00:00+00:00")
    rows = [policy, next_policy]
    quote = {"implied_monthly_effr_pct": 3.9}
    monkeypatch.setattr(pc, "quote_contract", lambda *args: quote)
    base = pc.rate_expectation(rows, factor_rows(), FACTOR_NOW)
    press = parse(month_html())[0]
    extra = {**press, "calendar_id": "extra", "scheduled_at_utc": "2026-11-05T19:30:00+00:00"}
    assert pc.rate_expectation(rows + [press, extra], factor_rows(), FACTOR_NOW) == base


def test_between_decision_and_conference_next_meeting_is_december(monkeypatch):
    now = datetime(2026, 10, 28, 18, 15, tzinfo=UTC)
    rows = [decision(), decision("2026-12-09T19:00:00+00:00"), parse(month_html())[0]]
    stamp = int((now - timedelta(days=1)).timestamp() * 1000)
    factors = {"series": {k: [[stamp, stamp, v, stamp]] for k, v in [("fed_effective", 3.88), ("fed_target", 4.0)]}}
    monkeypatch.setattr(pc, "quote_contract", lambda *args: {"implied_monthly_effr_pct": 3.9})
    assert pc.rate_expectation(rows, factors, now)["meeting_at_utc"] == "2026-12-09T19:00:00+00:00"


def test_partial_month_fetch_fails_whole_source_without_returning_partial_rows(monkeypatch):
    general = '''2026 FOMC Meetings<div class="fomc-meeting__month">October</div>
    <div class="fomc-meeting__date">27-28</div><div class="fomc-meeting__month">December</div>
    <div class="fomc-meeting__date">8-9</div>'''
    calls = []
    def fetch(url):
        calls.append(url)
        if url == pc.FED_CALENDAR: return general
        if "october" in url: return month_html()
        raise ValueError("Second month unavailable")
    monkeypatch.setattr(pc, "fetch", fetch)
    with pytest.raises(ValueError, match="unavailable"): pc.fomc_calendar(NOW)
    assert len(calls) == 3


def test_legacy_decision_only_cache_cannot_claim_new_conference_coverage(tmp_path, monkeypatch):
    previous = previous_snapshot()
    del previous["coverage_checks"][0]["sources"]["fomc_calendar"]["verified_month_pages"]
    payload = outage_collect(tmp_path, monkeypatch, CACHE_NOW + timedelta(minutes=5), previous)
    status = payload["metadata"]["source_status"]["fomc_calendar"]
    assert not status["ok"] and status["status"] == "unavailable"
    assert status["last_success_at_utc"] == pc.iso(CACHE_NOW)
    assert payload["calendars"] == previous["calendars"]


def test_complete_month_cache_keeps_original_verification_clock_and_scope(tmp_path, monkeypatch):
    previous = previous_snapshot()
    payload = outage_collect(tmp_path, monkeypatch, CACHE_NOW + timedelta(minutes=40), previous)
    state = payload["metadata"]["source_status"]["fomc_calendar"]
    assert state["ok"] and state["status"] == "cached" and not state["fetch_ok"]
    assert state["last_success_at_utc"] == pc.iso(CACHE_NOW)
    assert state["verified_month_pages"] == previous["coverage_checks"][0]["sources"]["fomc_calendar"]["verified_month_pages"]
    after = outage_collect(tmp_path, monkeypatch, CACHE_NOW + timedelta(minutes=61), payload)
    assert not after["metadata"]["source_status"]["fomc_calendar"]["ok"]


def test_month_rollover_cannot_reuse_a_cache_without_new_following_month():
    previous = previous_snapshot()
    success = datetime(2026, 10, 31, 23, 50, tzinfo=pc.ET_ZONE).astimezone(UTC)
    state = previous["coverage_checks"][0]["sources"]["fomc_calendar"]
    state["last_success_at_utc"] = pc.iso(success)
    state["verified_month_pages"] = [url for _, _, url in pc._fed_month_pages(success)]
    previous["coverage_checks"][0]["available_at_utc"] = pc.iso(success)
    assert pc._cached_fomc_month_pages(previous, success, success + timedelta(minutes=5))
    assert not pc._cached_fomc_month_pages(previous, success, success + timedelta(minutes=20))


def test_existing_decision_type_is_not_revised_or_re_seen_only_for_new_explicit_field():
    old = pc.merge_calendar([], [decision()], NOW - timedelta(days=10))
    new = {**decision(), "event_type": "policy_decision", "first_seen_at_utc": pc.iso(NOW)}
    assert pc.merge_calendar(old, [new], NOW) == old
