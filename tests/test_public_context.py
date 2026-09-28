import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import event_risk
import multifactor
import public_context as pc

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


def ms(t):
    return int(t.timestamp() * 1000)


def test_rss_keeps_source_publication_and_first_seen_separate():
    body = '''<rss><channel><item><title>New missile attack</title>
    <link>https://news.un.org/a</link><guid>1</guid>
    <pubDate>Sat, 26 Sep 2026 23:00:00 GMT</pubDate></item></channel></rss>'''
    row = pc.parse_rss(body, "un_news", pc.RSS["un_news"], NOW)[0]
    assert row["first_seen_at_utc"] == pc.iso(NOW)
    assert row["published_at_utc"] < row["first_seen_at_utc"]
    assert row["severity"] > 0 and row["category"] == "geopolitical_headline"
    assert row["article_url"] == "https://news.un.org/a"


def test_undated_rss_is_not_healthy():
    with pytest.raises(ValueError, match="no dated"):
        pc.parse_rss('<rss><channel><item><title>x</title></item></channel></rss>', "fed_news", "https://x", NOW)


def test_event_cannot_act_before_observed_or_after_superseded(tmp_path):
    path = tmp_path / "events.json"
    raw = {"event_id": "x", "published_at_utc": pc.iso(NOW - timedelta(hours=2)),
           "available_at_utc": pc.iso(NOW), "starts_at_utc": pc.iso(NOW - timedelta(hours=2)),
           "ends_at_utc": pc.iso(NOW + timedelta(hours=2)),
           "superseded_at_utc": pc.iso(NOW + timedelta(hours=1)),
           "severity": .8, "block_entries": True}
    path.write_text(json.dumps({"schema_version": 1, "events": [raw]}))
    events = event_risk.load_event_snapshot(path)
    assert event_risk.event_decision_at(events, ms(NOW - timedelta(seconds=1))).allowed
    assert not event_risk.event_decision_at(events, ms(NOW)).allowed
    assert event_risk.event_decision_at(events, ms(NOW + timedelta(hours=1))).allowed


def test_fomc_schedule_handles_dst():
    body = '''2026 FOMC Meetings
    <div class="fomc-meeting__month"><strong>October</strong></div>
    <div class="fomc-meeting__date">27-28</div>
    <div class="fomc-meeting__month"><strong>December</strong></div>
    <div class="fomc-meeting__date">8-9*</div>'''
    rows = pc.parse_fomc(body, NOW)
    assert rows[0]["scheduled_at_utc"] == "2026-10-28T18:00:00+00:00"
    assert rows[1]["scheduled_at_utc"] == "2026-12-09T19:00:00+00:00"
    assert rows[0]["time_basis"] == "regular_meeting_14_ET_convention"


def test_nyfed_calendar_uses_explicit_date_and_time():
    body = '''<td class="ts-data-table-head"><div>October 2026</div></td>
    <td><div>02<br/><span><a href="https://bls.gov/jobs">Employment Situation</a><br/>(08:30)</span></div></td>'''
    rows = pc.parse_nyfed(body, "https://www.newyorkfed.org/cal", NOW)
    assert rows[0]["scheduled_at_utc"] == "2026-10-02T12:30:00+00:00"


def test_bls_ics_unfolds_lines_and_handles_timezone():
    body = '''BEGIN:VCALENDAR
BEGIN:VEVENT
UID:cpi-october
SUMMARY:Consumer Price
  Index
DTSTART;TZID=America/New_York:20261014T083000
END:VEVENT
END:VCALENDAR'''
    rows = pc.parse_bls_ics(body, NOW)
    assert rows[0]["scheduled_at_utc"] == "2026-10-14T12:30:00+00:00"


def test_calendar_revision_keeps_previous_history():
    old = pc.calendar_item("cpi", "CPI", NOW + timedelta(days=10), "economic_calendar", "https://a", NOW)
    first = pc.merge_calendar([], [old], NOW)
    revised = {**old, "scheduled_at_utc": pc.iso(NOW + timedelta(days=11)), "first_seen_at_utc": pc.iso(NOW + timedelta(days=1))}
    result = pc.merge_calendar(first, [revised], NOW + timedelta(days=1))
    assert len(result) == 2
    assert result[0]["superseded_at_utc"] == pc.iso(NOW + timedelta(days=1))
    assert not result[1].get("superseded_at_utc")
    assert not first[0].get("superseded_at_utc")


def test_removed_future_calendar_event_expires_after_new_snapshot():
    old = pc.calendar_item("cpi", "CPI", NOW + timedelta(days=10), "economic_calendar", "https://a", NOW)
    other = pc.calendar_item("jobs", "jobs", NOW + timedelta(days=12), "economic_calendar", "https://a", NOW)
    first = pc.merge_calendar([], [old, other], NOW)
    result = pc.merge_calendar(first, [other], NOW + timedelta(days=1))
    assert result[0]["superseded_at_utc"]
    assert not result[1].get("superseded_at_utc")


def test_old_news_does_not_get_a_new_risk_window(tmp_path):
    news = [{"news_id": "n", "published_at_utc": pc.iso(NOW - timedelta(days=10)),
             "first_seen_at_utc": pc.iso(NOW), "severity": .5, "headline": "old", "category": "test", "article_url": "https://a"}]
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"schema_version": 1, "events": pc.make_events(news, [])}))
    assert event_risk.event_decision_at(event_risk.load_event_snapshot(path), ms(NOW)).allowed


def test_health_is_point_in_time_and_outage_blocks_entries():
    good = {k: {"ok": True} for k in pc.REQUIRED}
    bad = {**good, "fed_news": {"ok": False}}
    p = {"coverage_checks": [{"available_at_utc": pc.iso(NOW), "sources": good},
                              {"available_at_utc": pc.iso(NOW + timedelta(minutes=5)), "sources": bad}]}
    assert not pc.health_at(p, ms(NOW - timedelta(seconds=1)))[0]
    assert pc.health_at(p, ms(NOW))[0]
    assert not pc.health_at(p, ms(NOW + timedelta(minutes=5)))[0]
    assert pc.health_at(p, ms(NOW + timedelta(hours=2)))[1] == ["public_context_stale"]
    sleeves = [{"trades": [{"entry_time_utc": pc.iso(NOW)}, {"entry_time_utc": pc.iso(NOW + timedelta(minutes=5))}]}]
    adjusted, blocked = pc.apply_entry_coverage(sleeves, p)
    assert blocked == 1 and len(adjusted[0]["trades"]) == 1
    assert len(sleeves[0]["trades"]) == 2


def factor_rows():
    stamp = ms(NOW - timedelta(days=2))
    return {"series": {"fed_effective": [[stamp, stamp, 3.88, stamp]], "fed_target": [[stamp, stamp, 4, stamp]]}}


def expectation():
    return {"meeting_at_utc": "2026-10-28T18:00:00+00:00", "available_at_utc": pc.iso(NOW),
            "quote_at_utc": pc.iso(NOW - timedelta(days=1)), "expected_change_bps": 15.5,
            "reference_target_upper_pct": 4, "reference_effr_pct": 3.88}


def test_rate_expectation_uses_named_nonmeeting_month():
    meetings = [pc.calendar_item("fomc:oct", "FOMC", pc.utc("2026-10-28T18:00:00+00:00"), "fomc_calendar", pc.FED_CALENDAR, NOW),
                pc.calendar_item("fomc:dec", "FOMC", pc.utc("2026-12-09T19:00:00+00:00"), "fomc_calendar", pc.FED_CALENDAR, NOW)]
    quote = {"implied_monthly_effr_pct": 4.035, "quote_at_utc": pc.iso(NOW - timedelta(days=1))}
    with patch.object(pc, "quote_contract", return_value=quote) as fetch:
        row = pc.rate_expectation(meetings, factor_rows(), NOW)
    fetch.assert_called_once_with(2026, 11, NOW)
    assert row["expected_change_bps"] == pytest.approx(15.5)


def test_rate_expectation_uses_day_weighted_month_when_next_month_has_meeting():
    meetings = [pc.calendar_item(str(month), "FOMC", datetime(2026, month, 28, 18, tzinfo=timezone.utc), "fomc_calendar", pc.FED_CALENDAR, NOW)
                for month in (10, 11)]
    quote = {"implied_monthly_effr_pct": 3.9}
    with patch.object(pc, "quote_contract", return_value=quote) as fetch:
        row = pc.rate_expectation(meetings, factor_rows(), NOW)
    fetch.assert_called_once_with(2026, 10, NOW)
    assert row["implied_postmeeting_effr_pct"] == pytest.approx((3.9 * 31 - 3.88 * 28) / 3)
    assert row["method"] == "single_meeting_month_day_weighted_effr"


def test_future_or_stale_expectation_not_used():
    p = {"rate_expectations": [expectation()]}
    assert pc.expectation_at(p, ms(NOW - timedelta(seconds=1))) is None
    assert pc.expectation_at(p, ms(NOW))
    assert pc.expectation_at(p, ms(NOW + timedelta(hours=3))) is None


def test_policy_surprise_needs_premeeting_archive_and_released_actual():
    meeting = pc.utc("2026-10-28T18:00:00+00:00")
    e = {**expectation(), "available_at_utc": pc.iso(meeting - timedelta(hours=1)),
         "quote_at_utc": pc.iso(meeting - timedelta(hours=2))}
    observed = ms(meeting + timedelta(hours=6))
    available = ms(meeting + timedelta(days=2))
    factors = {"series": {"fed_target": [[observed, available, 4.25, available]]}}
    assert not pc.policy_surprises([e], factors, meeting + timedelta(days=1))
    rows = pc.policy_surprises([e], factors, meeting + timedelta(days=3))
    assert rows[0]["surprise_bps"] == pytest.approx(9.5)
    assert not pc.policy_surprises([{**e, "available_at_utc": pc.iso(meeting + timedelta(seconds=1))}], factors, meeting + timedelta(days=3))


def test_missing_expected_rate_removes_fed_group():
    p = multifactor.load_profile(ROOT / "config/multifactor_candidate_20260927.json")
    p["groups"] = {"fed": 1}
    series = {name: [[i * multifactor.DAY, i * multifactor.DAY, 100, i * multifactor.DAY] for i in range(61)]
              for name in ("fed_target", "fed_assets")}
    s = multifactor.Snapshot({"schema_version": 1, "series": series})
    d = multifactor.decision_at(s, 60 * multifactor.DAY, "long", p, {})
    assert not d.allowed and "fed" in d.missing_groups
