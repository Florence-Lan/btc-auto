"""European brief identity/time/unit checks; no HTTP calls or trading imports."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import europe_bond_observations as eb

URL = ("https://www.borsaitaliana.it/borsa/notizie/radiocor/finance/dettaglio/"
       "euro-zone-benchmark-10year-sovereign-bond-yields-nRC_23102026_1734_551141600.html?lang=en")


def ms(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


RECEIVED = ms("2026-10-25T13:46:01Z")
BRIEF = """<html><h1></h1><h1>Euro zone benchmark 10-year sovereign bond yields</h1>
<span class="t-text -block -line-1-3em">
<p>(Il Sole 24 Ore Radiocor) - Milano, 23 ott - Euro zone
benchmark 10-year sovereign bond yields at 1505 GMT.</p>
<p>Germany - yield 3.35%; up 0.01 percentage points from the previous close.</p>
<p>France - yield 5.04%; down 0.04 percentage points from the previous close.</p>
<p>Belgium - yield 4.44%; down 0.01 percentage points from the previous close.</p>
<p>Italy - yield 4.59%; down 0.01 percentage points from the previous close.</p>
<p>Spain - yield 4.05%; unchanged from the previous close</p>
<p>(RADIOCOR) 23-10-26 17:34:49 (0551) 5 NNNN</p></span></html>"""


def parse(body=BRIEF, **kwargs):
    return eb.parse_europe_bond_brief(body, source_url=kwargs.pop("source_url", URL),
        received_at_ms=kwargs.pop("received_at_ms", RECEIVED),
        asof_ms=kwargs.pop("asof_ms", RECEIVED), **kwargs)


def index(*, article_url=URL, label=eb.TITLE, next_href="?lang=en&page=2"):
    return (f'<ul class="itemListArchive" id="itemListArchive"><li><h3>'
            f'<a href="{article_url}">{label}</a></h3></li></ul>'
            f'<a title="Next" href="{next_href}"><span>next</span></a>')


def test_real_report_shape_preserves_gmt_and_comparable_daily_basis():
    snapshot = parse(BRIEF.encode())
    assert set(snapshot["observations"]) == {"de_10y_pct", "fr_10y_pct", "it_10y_pct", "es_10y_pct", "be_10y_pct"}
    assert snapshot["observed_at_ms"] == ms("2026-10-23T15:05:00Z")
    assert snapshot["status"] == "daily_current"
    assert snapshot["freshness"]["age_days"] == 2
    assert snapshot["freshness"]["valid_until_ms"] == ms("2026-10-28T00:00:00Z")
    row = snapshot["observations"]["fr_10y_pct"]
    assert row["value"] == 5.04 and row["unit"] == "percent"
    assert row["instrument"]["isin"] is None
    assert row["instrument"]["kind"] == "provider_generic_sovereign_benchmark_yield"
    assert row["frequency"] == "daily" and not row["eligible_intraday"]
    assert not row["freshness"]["eligible_intraday"]
    assert row["change_from_previous_close_pp"] == -0.04
    assert snapshot["derived_spreads"]["fr_de_10y_spread_bp"]["value"] == 169
    assert snapshot["derived_spreads"]["it_de_10y_spread_bp"]["value"] == 124
    assert snapshot["derived_spreads"]["es_de_10y_spread_bp"]["value"] == 70
    assert snapshot["provenance"]["published_at_ms"] is None
    assert snapshot["provenance"]["publication_wall_time"] == "2026-10-23T17:34:49"
    assert not snapshot["places_orders"]


def test_refetch_never_renews_availability_or_snapshot_age():
    first = parse()
    later = ms("2026-10-28T00:00:00Z")
    second = parse(previous=first, received_at_ms=later, asof_ms=later)
    assert second["snapshot_id"] == first["snapshot_id"]
    assert second["first_seen_at_ms"] == RECEIVED
    assert second["status"] == "stale"
    assert not second["freshness"]["eligible_daily"]
    for row in second["observations"].values():
        assert row["available_at_ms"] == row["received_at_ms"] == RECEIVED
        assert row["provenance"]["retrieved_at_ms"] == later
        assert not row["eligible_intraday"]
    assert all(not row["freshness"]["eligible_daily"] for row in second["derived_spreads"].values())


def test_changed_nav_does_not_create_a_new_print_but_changed_yield_does():
    first = parse()
    nav_changed = parse(BRIEF.replace("<html>", "<html><nav>new news</nav>"), previous=first,
                        received_at_ms=RECEIVED + 1000, asof_ms=RECEIVED + 1000)
    assert nav_changed["snapshot_id"] == first["snapshot_id"]
    assert nav_changed["provenance"]["source_body_sha256"] != first["provenance"]["source_body_sha256"]
    corrected = parse(BRIEF.replace("5.04%", "5.05%"), previous=first,
                      received_at_ms=RECEIVED + 1000, asof_ms=RECEIVED + 1000)
    assert corrected["snapshot_id"] != first["snapshot_id"]
    assert corrected["first_seen_at_ms"] == RECEIVED + 1000


def test_rows_do_not_share_mutable_freshness_or_provenance():
    snapshot = parse()
    snapshot["observations"]["fr_10y_pct"]["freshness"]["eligible_daily"] = False
    snapshot["observations"]["fr_10y_pct"]["provenance"]["source_url"] = "changed"
    assert snapshot["observations"]["de_10y_pct"]["freshness"]["eligible_daily"]
    assert snapshot["derived_spreads"]["fr_de_10y_spread_bp"]["freshness"]["eligible_daily"]
    assert snapshot["provenance"]["source_url"] == URL


@pytest.mark.parametrize("before,after", [
    ("10-year sovereign", "2-year sovereign"),
    ("1505 GMT", "1505 CET"),
    ("1505 GMT", "2505 GMT"),
    ("23-10-26", "22-10-26"),
    ("23 ott", "23 set"),
    ("17:34:49", "17:35:49"),
    ("3.35%;", "3.35bp;"),
    ("5.04%", "50.04%"),
    ("France - yield", "France 10Y auction - yield"),
    ("previous close", "monthly average"),
    ('class="t-text -block -line-1-3em"', 'class="new-structure"'),
])
def test_changed_identity_unit_dates_or_structure_fail_closed(before, after):
    with pytest.raises(eb.EuropeBondError):
        parse(BRIEF.replace(before, after))


def test_optional_belgium_only_and_no_duplicate_or_partial_required_countries():
    belgium = "<p>Belgium - yield 4.44%; down 0.01 percentage points from the previous close.</p>"
    assert "be_10y_pct" not in parse(BRIEF.replace(belgium, ""))["observations"]
    with pytest.raises(eb.EuropeBondError):
        parse(BRIEF.replace("Belgium - yield", "France - yield"))
    with pytest.raises(eb.EuropeBondError):
        parse(BRIEF.replace("Spain - yield", "Belgium - yield"))


@pytest.mark.parametrize("url", [URL.replace("https:", "http:"), URL.replace("www.borsaitaliana.it", "evil.test"),
    URL.replace(".it/", ".it:443/"), URL + "#fragment", URL + "&lang=en", URL.replace("lang=en", "lang=it"),
    URL.replace("www.borsaitaliana.it", "user@www.borsaitaliana.it"), URL.replace("23102026", "32102026"),
    " " + URL, URL.replace("https://", "https://\n")])
def test_strict_source_allowlist(url):
    with pytest.raises(eb.EuropeBondError):
        parse(source_url=url)


@pytest.mark.parametrize("bad", [True, False, -1, 0, 1.5, "2026-10-25T00:00:00", None])
def test_explicit_positive_millisecond_clocks_only(bad):
    with pytest.raises(eb.EuropeBondError):
        parse(asof_ms=bad)


def test_future_observation_and_future_receipt_rejected():
    with pytest.raises(eb.EuropeBondError, match="future_observation"):
        parse(received_at_ms=ms("2026-10-23T15:04:59Z"), asof_ms=ms("2026-10-23T15:04:59Z"))
    with pytest.raises(eb.EuropeBondError, match="receipt_after_asof"):
        parse(asof_ms=RECEIVED - 1)
    with pytest.raises(eb.EuropeBondError, match="invalid_previous_availability"):
        parse(previous={"snapshot_id": parse()["snapshot_id"], "first_seen_at_ms": RECEIVED + 1})
    with pytest.raises(eb.EuropeBondError, match="previous_snapshot_object_required"):
        parse(previous=[])


@pytest.mark.parametrize("age", [True, -1, 8, 2.5])
def test_freshness_policy_is_finite_calendar_days(age):
    with pytest.raises(eb.EuropeBondError):
        parse(max_age_days=age)


def test_body_limit_and_encoding():
    for body in [b"", b"\xff", "x" * (eb.MAX_BODY_BYTES + 1), "\ud800", {"html": BRIEF}]:
        with pytest.raises(eb.EuropeBondError):
            parse(body)


def test_discovery_uses_actual_links_and_actual_next_page_only():
    result = eb.discover_europe_bond_briefs(index(), asof_ms=RECEIVED)
    assert result["source_status"] == "discovered"
    assert result["candidates"][0]["url"] == URL
    assert result["candidates"][0]["article_date"] == "2026-10-23"
    assert result["next_page_url"] == eb.INDEX_URL + "&page=2"
    assert not result["automatic_url_guessing"]
    no_next = index().split('<a title="Next"')[0]
    assert eb.discover_europe_bond_briefs(no_next, asof_ms=RECEIVED)["next_page_url"] is None


def test_index_without_brief_is_missing_not_evergreen_snapshot():
    html = index(article_url="/borsa/notizie/radiocor/finance/dettaglio/other.html?lang=en", label="Other news")
    result = eb.discover_europe_bond_briefs(html, asof_ms=RECEIVED)
    assert result["source_status"] == "missing" and not result["candidates"]


@pytest.mark.parametrize("html", ["<p>News</p>", index(article_url=URL.replace("www.borsaitaliana.it", "evil.test")),
    index(next_href="?lang=en&page=50"), index(next_href="https://evil.test/index"),
    index(label="Daily government auction yields"), index(article_url=URL.replace("23102026", "26102026"))])
def test_discovery_rejects_changed_shape_foreign_links_future_and_page_jumps(html):
    with pytest.raises(eb.EuropeBondError):
        eb.discover_europe_bond_briefs(html, asof_ms=RECEIVED)


def test_pagination_is_sequential_and_dated_candidates_sort_without_guessing():
    older = URL.replace("23102026", "22102026")
    html = index(article_url=older).replace("</ul>", f'<li><a href="{URL}">{eb.TITLE}</a></li></ul>')
    result = eb.discover_europe_bond_briefs(html, asof_ms=RECEIVED)
    assert [row["url"] for row in result["candidates"]] == [URL, older]
    page2 = eb.discover_europe_bond_briefs(index(next_href="?lang=en&page=3"),
        index_url=eb.INDEX_URL + "&page=2", asof_ms=RECEIVED)
    assert page2["next_page_url"] == eb.INDEX_URL + "&page=3"
