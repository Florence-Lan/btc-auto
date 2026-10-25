"""Point-in-time and transport invariants for the isolated market observer."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cross_market_observations as co
import cross_market_risk as risk

NOW = int(datetime(2026, 10, 23, 1, 0, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.fixture(autouse=True)
def separate_request_deadlines():
    # Independent tests must not share intentionally abandoned deadline workers.
    co._DEADLINES.clear()
    yield
    co._DEADLINES.clear()


def quote_source():
    return copy.deepcopy(next(s for s in co.load_config()["sources"] if s["id"] == "yahoo_nq_dec_2026"))


def config_for(*sources):
    # These historical transport fixtures predate the maintained reference
    # calendar; session causality is exercised separately by integration tests.
    sources = copy.deepcopy(sources)
    for source in sources:
        source.pop("session_calendar", None)
    return {"schema_version": 1, "mode": "OBSERVATION_ONLY",
            "collection_timeout_seconds": 1, "sources": [
                {**s, "timeout_seconds": .5} for s in sources]}


def quote(source=None, *, at_ms=NOW, value=30000, previous=29700):
    source = source or quote_source()
    return {"chart": {"result": [{"meta": {
        "symbol": source["expected_symbol"], "instrumentType": source["expected_type"],
        "currency": source["expected_currency"], "exchangeTimezoneName": source["timezone"],
        "regularMarketTime": at_ms // 1000, "regularMarketPrice": value,
        "previousClose": previous, "longName": source["instrument"]["name"]}}], "error": None}}


def csv_source(*, source_id="daily", field="yield_pct", unit="percent", poll=21600):
    return {"id": source_id, "parser": "dated_csv", "provider": "Official fixture",
            "url": "https://example.org/public.csv", "timezone": "Europe/Brussels",
            "date_column": "Date", "date_format": "%Y-%m-%d", "poll_seconds": poll,
            "timeout_seconds": .5, "instrument": {"country": source_id, "tenor": "10Y"},
            "fields": {field: {"unit": unit, "column": "10Y", "max_age_days": 4}}}


def test_default_config_is_isolated_explicit_and_has_all_requested_keys():
    config = co.load_config()
    keys = {key for source in config["sources"] for key in source["fields"]}
    keys.update(config["derived_spreads"])
    assert keys == {"wti_usd_per_barrel", "brent_usd_per_barrel", "nq_dec_2026", "usdjpy",
                    "us_2y_pct", "us_10y_pct", "us_10y_real_pct", "jp_2y_pct", "jp_10y_pct",
                    "jp_30y_pct", "de_10y_pct", "fr_10y_pct", "it_10y_pct", "uk_10y_pct",
                    "fr_de_10y_spread_bp", "it_de_10y_spread_bp", "es_10y_pct", "be_10y_pct",
                    "es_de_10y_spread_bp", "be_de_10y_spread_bp"}
    assert config["places_orders"] is False
    futures = [s for s in config["sources"] if s.get("expected_type") == "FUTURE"]
    assert all(s["instrument"]["automatic_roll"] is False for s in futures)
    assert all("=F" not in s["expected_symbol"] for s in futures)
    assert all(s["poll_seconds"] == 21600 for s in config["sources"] if s["parser"] == "dated_csv")


def test_yahoo_value_keeps_observation_first_receipt_and_denominator_separate():
    source = quote_source()
    at = NOW - 600000  # A ten-minute-delayed future, not a real-time assertion.
    snapshot = co.collect_observations(config_for(source), now_ms=NOW,
                                       fetcher=lambda s, t: quote(s, at_ms=at))
    row = snapshot["observations"]["nq_dec_2026"]
    assert row["observed_at_ms"] == at
    assert row["first_seen_at_ms"] == row["available_at_ms"] == NOW
    assert row["freshness"]["age_ms"] == 600000
    assert row["eligible_intraday"] and row["freshness"]["eligible_intraday"]
    assert row["return_pct"] == pytest.approx((30000 / 29700 - 1) * 100)
    assert row["provider_previous_close"] == 29700
    assert row["previous_close_observed_at_ms"] is None
    assert row["return_basis"] == "same_named_instrument_provider_previousClose"
    assert snapshot["places_orders"] is False


def test_identical_successful_fetch_never_renews_age_or_first_availability():
    source = quote_source()
    conf = config_for(source)
    fetcher = lambda s, t: quote(s, at_ms=NOW - 600000)
    first = co.collect_observations(conf, now_ms=NOW, fetcher=fetcher)
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 700000, fetcher=fetcher)
    a, b = (s["observations"]["nq_dec_2026"] for s in (first, second))
    assert b["fetch"]["last_success_at_ms"] == NOW + 700000
    assert b["first_seen_at_ms"] == b["available_at_ms"] == b["received_at_ms"] == NOW
    assert b["observed_at_ms"] == a["observed_at_ms"]
    assert b["freshness"]["age_ms"] == 1300000
    assert b["freshness"]["status"] == "stale"
    assert b["eligible_intraday"] is False
    assert second["status"] == "degraded"


def test_independent_daily_cache_does_not_refetch_on_fast_quote_poll():
    source, daily = quote_source(), csv_source()
    conf, called = config_for(source, daily), []
    def fetcher(s, t):
        called.append(s["id"])
        return quote(s, at_ms=NOW - 1000) if s["parser"] == "yahoo_chart" else "Date,10Y\n2026-10-22,3.12\n"
    first = co.collect_observations(conf, now_ms=NOW, fetcher=fetcher)
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 61000, fetcher=fetcher)
    assert called.count(source["id"]) == 2
    assert called.count(daily["id"]) == 1
    row = second["observations"]["yield_pct"]
    assert row["fetch"]["status"] == "cached"
    assert row["first_seen_at_ms"] == row["available_at_ms"] == NOW
    assert row["observed_at_ms"] is None and row["precision"] == "date"
    assert row["freshness"]["eligible_daily"] and not row["eligible_intraday"]


def test_revised_previous_close_is_not_backdated_into_the_price_receipt():
    conf = config_for(quote_source())
    first = co.collect_observations(conf, now_ms=NOW,
                                   fetcher=lambda s, t: quote(s, previous=29700))
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 61000,
                                    fetcher=lambda s, t: quote(s, previous=29800))
    row = second["observations"]["nq_dec_2026"]
    assert row["observed_at_ms"] == row["first_seen_at_ms"] == row["available_at_ms"] == NOW
    assert row["provider_reference_available_at_ms"] == NOW + 61000
    assert row["return_pct"] == pytest.approx((30000 / 29800 - 1) * 100)


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m.update(regularMarketTime=NOW // 1000 + 1), "future_observation"),
    (lambda m: m.pop("exchangeTimezoneName"), "observation_timezone_required"),
    (lambda m: m.update(exchangeTimezoneName="EDT"), "invalid_observation_timezone"),
    (lambda m: m.update(symbol="NQH27.CME"), "instrument_identity_mismatch"),
    (lambda m: m.update(instrumentType="INDEX"), "instrument_identity_mismatch"),
    (lambda m: m.update(currency="JPY"), "instrument_identity_mismatch"),
    (lambda m: m.update(regularMarketPrice=float("nan")), "non_finite_value"),
])
def test_future_timezone_identity_and_nonfinite_rejections(mutate, reason):
    source, payload = quote_source(), quote()
    mutate(payload["chart"]["result"][0]["meta"])
    with pytest.raises(co.ObservationError, match=reason):
        co.parse_yahoo_chart(payload, source, NOW)


def test_naive_iso_timestamp_is_never_assumed_to_be_utc():
    with pytest.raises(co.ObservationError, match="timestamp_timezone_required"):
        co.parse_timestamp_ms("2026-10-23T09:00:00")
    assert co.parse_timestamp_ms("2026-10-23T09:00:00+08:00") == NOW
    assert co.parse_timestamp_ms("2026-10-23T01:00:00Z") == NOW


def test_date_only_rows_keep_latest_date_and_missing_tenor_is_not_backfilled():
    source = csv_source()
    source["fields"]["short_pct"] = {"column": "2Y", "unit": "percent"}
    body = "Interest Rate (October 2026),,,\nDate,2Y,10Y\n2026-10-21,1.914,2.923\n2026-10-22,,2.950\n,footnote,\n"
    values = co.parse_dated_csv(body, source, NOW)
    assert values["yield_pct"]["observed_date"] == "2026-10-22"
    assert values["yield_pct"]["observed_at_ms"] is None
    assert "short_pct" not in values
    assert values["yield_pct"]["value"] == 2.95
    with pytest.raises(co.ObservationError, match="future_observation_date"):
        co.parse_dated_csv("Date,10Y\n2026-10-24,2.95\n", source, NOW)


def test_mof_shift_jis_footer_and_boe_date_format_do_not_invent_times():
    source = csv_source()
    source.update(timezone="Asia/Tokyo", date_format="%Y/%m/%d")
    body = "Interest Rate (October 2026),\r\r\nDate,10Y\r\r\n2026/10/21,2.923\r\r\n最新の注記,\r\r\n".encode("shift_jis")
    row = co.parse_dated_csv(body, source, NOW)["yield_pct"]
    assert row["observed_date"] == "2026-10-21" and row["observed_at_ms"] is None
    source.update(date_column="DATE", date_format="%d %b %Y")
    row = co.parse_dated_csv("DATE,10Y\n20 Oct 2026,5.3516\n", source, NOW)["yield_pct"]
    assert row["value"] == 5.3516 and row["precision"] == "date"


def test_source_dated_revision_gets_new_availability_but_not_a_new_observation_date():
    source = csv_source(poll=30)
    conf = config_for(source)
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: "Date,10Y\n2026-10-22,3.10\n")
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 31000,
                                    fetcher=lambda s, t: "Date,10Y\n2026-10-22,3.11\n")
    a, b = first["observations"]["yield_pct"], second["observations"]["yield_pct"]
    assert a["observed_date"] == b["observed_date"] == "2026-10-22"
    assert a["available_at_ms"] == NOW and b["available_at_ms"] == NOW + 31000
    assert a["value"] == 3.10  # Caller-owned previous snapshot was not mutated.


def test_regressing_quote_cannot_overwrite_a_newer_print():
    source, conf = quote_source(), config_for(quote_source())
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: quote(s, at_ms=NOW - 1000))
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 61000,
                                    fetcher=lambda s, t: quote(s, at_ms=NOW - 2000, value=1))
    row = second["observations"]["nq_dec_2026"]
    assert row["value"] == 30000 and row["observed_at_ms"] == NOW - 1000
    assert row["first_seen_at_ms"] == NOW
    assert row["fetch"]["status"] == "regressed_observation"


def test_cached_future_first_receipt_is_not_causally_available():
    conf = config_for(quote_source())
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: quote(s, at_ms=NOW - 10000))
    row = first["observations"]["nq_dec_2026"]
    row["first_seen_at_ms"] = row["available_at_ms"] = NOW + 1000
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 1,
                                    fetcher=lambda s, t: pytest.fail("cache should suppress fetch"))
    row = second["observations"]["nq_dec_2026"]
    assert row["freshness"]["status"] == "not_yet_available"
    assert row["eligible_intraday"] is False
    with pytest.raises(ValueError, match="not_causally_available"):
        risk._point(row, NOW + 1)


def test_cached_observation_cannot_precede_its_claimed_first_receipt():
    conf = config_for(quote_source())
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: quote(s, at_ms=NOW - 1000))
    row = first["observations"]["nq_dec_2026"]
    row["first_seen_at_ms"] = row["available_at_ms"] = NOW - 2000
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 1,
                                    fetcher=lambda s, t: pytest.fail("cache should suppress fetch"))
    assert second["observations"]["nq_dec_2026"]["freshness"]["status"] == "observation_after_first_receipt"
    assert second["observations"]["nq_dec_2026"]["eligible_intraday"] is False


def test_freshness_policy_change_does_not_reset_first_receipt_identity():
    source = quote_source()
    conf = config_for(source)
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: quote(s))
    changed = copy.deepcopy(conf)
    changed["sources"][0]["fields"]["nq_dec_2026"]["max_age_seconds"] = 30
    changed["sources"][0]["poll_seconds"] = 30
    second = co.collect_observations(changed, previous=first, now_ms=NOW + 61000,
                                    fetcher=lambda s, t: quote(s))
    row = second["observations"]["nq_dec_2026"]
    assert row["first_seen_at_ms"] == NOW and row["eligible_intraday"] is False


def test_roll_review_cutoff_invalidates_cache_without_a_silent_new_contract():
    source = quote_source()
    source["contract_review_cutoff_utc"] = "2026-10-23T01:01:00Z"
    conf = config_for(source)
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: quote(s))
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 60000,
                                    fetcher=lambda s, t: quote(s, at_ms=NOW + 60000))
    row = second["observations"]["nq_dec_2026"]
    assert row["freshness"]["status"] == "contract_review_due"
    assert row["eligible_intraday"] is False and row["first_seen_at_ms"] == NOW
    assert row["instrument"]["symbol"] == "NQZ26.CME"
    assert row["fetch"]["error"]["reason"] == "contract_review_due"


def test_spread_is_basis_points_same_date_and_cannot_refresh_unchanged_inputs():
    sources = [csv_source(source_id=k, field=k + "_10y_pct", poll=30) for k in ("de", "fr", "it")]
    conf = config_for(*sources)
    conf["derived_spreads"] = {
        "fr_de_10y_spread_bp": {"left": "fr_10y_pct", "right": "de_10y_pct"},
        "it_de_10y_spread_bp": {"left": "it_10y_pct", "right": "de_10y_pct"}}
    values = {"de": 3.2, "fr": 4.1, "it": 3.9}
    fetcher = lambda s, t: f"Date,10Y\n2026-10-22,{values[s['id']]}\n"
    first = co.collect_observations(conf, now_ms=NOW, fetcher=fetcher)
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 31000, fetcher=fetcher)
    assert first["observations"]["fr_de_10y_spread_bp"]["value"] == 90
    assert first["observations"]["it_de_10y_spread_bp"]["value"] == 70
    row = second["observations"]["fr_de_10y_spread_bp"]
    assert row["unit"] == "bp" and row["available_at_ms"] == row["first_seen_at_ms"] == NOW
    assert row["precision"] == "date" and not row["eligible_intraday"]
    # A one-day mismatch is missing, rather than a spurious current spread.
    different = co.collect_observations(conf, previous=second, now_ms=NOW + 62000,
        fetcher=lambda s, t: f"Date,10Y\n{'2026-10-21' if s['id'] == 'fr' else '2026-10-22'},{values[s['id']]}\n")
    # An older source print is rejected, so the previous synchronized row survives.
    assert different["sources"]["fr"]["status"] == "regressed_observation"
    blank_old = co.collect_observations(conf, now_ms=NOW,
        fetcher=lambda s, t: f"Date,10Y\n{'2026-10-21' if s['id'] == 'fr' else '2026-10-22'},{values[s['id']]}\n")
    assert blank_old["observations"]["fr_de_10y_spread_bp"]["value"] is None


def test_missing_europe_is_explicit_and_does_not_invent_zero_spreads():
    conf = co.load_config()
    missing = copy.deepcopy(next(s for s in conf["sources"] if s["parser"] == "europe_bond_brief"))
    missing.update(parser="unavailable", reason="daily_country_specific_source_not_verified")
    conf = {**config_for(missing), "derived_spreads": conf["derived_spreads"]}
    snapshot = co.collect_observations(conf, now_ms=NOW,
                                       fetcher=lambda s, t: pytest.fail("unavailable source must not fetch"))
    for row in snapshot["observations"].values():
        assert row["value"] is None and row["first_seen_at_ms"] is None
        assert row["freshness"]["status"] == "missing" and not row["eligible_intraday"]


@pytest.mark.parametrize("status", [403, 429])
def test_http_denial_is_not_retried_or_bypassed_and_has_backoff(monkeypatch, status):
    calls = []
    class Response:
        status_code = status
        headers = {"Retry-After": "7200"}
        def close(self):
            pass
    def get(url, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is False
        return Response()
    monkeypatch.setattr(co.requests, "get", get)
    conf = config_for(quote_source())
    first = co.collect_observations(conf, now_ms=NOW)
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 61000)
    assert len(calls) == 1
    state = first["sources"]["yahoo_nq_dec_2026"]
    assert state["error"]["http_status"] == status
    assert state["next_due_at_ms"] == NOW + 7200000
    assert second["observations"]["nq_dec_2026"]["value"] is None


def test_errors_preserve_previous_price_without_claiming_new_availability():
    conf = config_for(quote_source())
    first = co.collect_observations(conf, now_ms=NOW, fetcher=lambda s, t: quote(s))
    def fail(s, t):
        raise co.requests.ConnectionError("secret proxy credential should never be logged")
    second = co.collect_observations(conf, previous=first, now_ms=NOW + 61000, fetcher=fail)
    row = second["observations"]["nq_dec_2026"]
    assert row["value"] == 30000 and row["available_at_ms"] == row["first_seen_at_ms"] == NOW
    assert row["fetch"]["status"] == "error"
    assert "secret proxy" not in json.dumps(second)


def test_total_deadline_discards_late_data_and_bounds_repeat_workers():
    source = quote_source()
    conf = config_for(source)
    conf.update(collection_timeout_seconds=.08)
    conf["sources"][0]["timeout_seconds"] = .03
    release, calls = threading.Event(), []
    def slow(s, t):
        calls.append(s["id"])
        release.wait(.5)
        return quote(s)
    try:
        start = time.monotonic()
        first = co.collect_observations(conf, now_ms=NOW, fetcher=slow)
        assert time.monotonic() - start < .3
        assert first["observations"]["nq_dec_2026"]["value"] is None
        second = co.collect_observations(conf, previous=first, now_ms=NOW + 61000, fetcher=slow)
        assert second["observations"]["nq_dec_2026"]["value"] is None
        assert len(calls) == 1  # The timed-out DNS/socket worker is not duplicated.
    finally:
        release.set()


def test_three_quote_output_is_compatible_with_risk_point_and_forward_history():
    default = co.load_config()
    sources = [s for s in default["sources"] if next(iter(s["fields"])) in
               {"nq_dec_2026", "brent_usd_per_barrel", "usdjpy"}]
    conf = config_for(*sources)
    snapshot = co.collect_observations(conf, now_ms=NOW,
                                       fetcher=lambda s, t: quote(s, at_ms=NOW - 1000))
    for key in risk.DEFAULT_MARKET_POLICY["required_features"]:
        point = risk._point(snapshot["observations"][key], NOW)
        assert point["observed_at_ms"] <= point["first_seen_at_ms"] <= point["available_at_ms"] <= NOW
        assert point["instrument"]
    history = risk.update_forward_history(snapshot, {}, asof_ms=NOW, inception_ms=NOW - 2000)
    assert all(len(history[k]) == 1 for k in risk.DEFAULT_MARKET_POLICY["required_features"])
    decision = risk.evaluate_cross_market_risk(snapshot, {}, asof_ms=NOW, history=history)
    assert decision["status"] == "warming_up" and not decision["allowed"]


def test_cli_refuses_paper_trading_output_before_any_network(monkeypatch):
    monkeypatch.setattr(co, "collect_observations", lambda *a, **kw: pytest.fail("must reject before fetching"))
    with pytest.raises(ValueError, match="must not overwrite"):
        co.main(["--output-dir", str(ROOT / "data/paper_trading/forbidden")])


def test_cli_writes_only_isolated_latest_and_immutable_archive(tmp_path, monkeypatch):
    snapshot = {"schema_version": 1, "generated_at_ms": NOW, "status": "healthy",
                "required_intraday_missing_or_stale": [], "observations": {}, "sources": {}}
    monkeypatch.setattr(co, "collect_observations", lambda *a, **kw: copy.deepcopy(snapshot))
    co.main(["--output-dir", str(tmp_path), "--once"])
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == snapshot
    archive = tmp_path / "snapshots" / f"{NOW}.json"
    assert json.loads(archive.read_text(encoding="utf-8")) == snapshot
    with pytest.raises(FileExistsError, match="do not overwrite"):
        co.main(["--output-dir", str(tmp_path), "--once"])
