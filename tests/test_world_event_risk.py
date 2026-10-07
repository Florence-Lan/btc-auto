import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import research_world_events as runner
import world_event_risk as risk

T = risk.utc_ms("2026-09-01T00:00:00Z")


def observation(**updates):
    return {"id": "a", "event_id": "shock", "source_group": "wire-a",
            "source_url": "https://example.com/a", "headline": "Test shock",
            "category": "energy", "first_seen_at_utc": risk.iso(T),
            "assessed_at_utc": risk.iso(T), "severity": 1.0, "surprise": 1.0,
            "novelty": 1.0, "confidence": 1.0, "half_life_hours": 6.0,
            "verified": True, "primary_source": True, "evidence_note": "test fixture",
            **updates}


def snapshot(*rows, at=T):
    value = risk.empty_snapshot()
    value["observations"] = list(rows)
    value["polls"] = [{"available_at_utc": risk.iso(at), "status": "ok"}]
    return value


def test_no_future_news_or_future_review():
    raw = observation(verified=False, primary_source=False)
    reviewed = observation(id="review", assessed_at_utc=risk.iso(T + risk.HOUR_MS))
    data = snapshot(raw, reviewed, at=T - 1)
    assert risk.decision_at(data, T - 1).watch_score == 0
    assert risk.decision_at(data, T).score == 0
    assert risk.decision_at(data, T).watch_score == 100
    assert risk.decision_at(data, T + risk.HOUR_MS).score == pytest.approx(100 * 2 ** (-1 / 6))


def test_two_reviewed_independent_sources_required_without_primary():
    first = observation(primary_source=False)
    repost = observation(id="b", source_url="https://example.com/b", primary_source=False)
    independent = observation(id="c", source_url="https://independent.example/c",
                              source_group="wire-b", primary_source=False,
                              assessed_at_utc=risk.iso(T + 1000))
    data = snapshot(first, repost, independent)
    assert risk.decision_at(data, T).score == 0
    assert risk.decision_at(data, T + 1000).score > 99


def test_unreviewed_or_low_confidence_sources_cannot_confirm():
    for extra in ({"verified": False}, {"confidence": 0.6}):
        data = snapshot(observation(primary_source=False), observation(
            id="b", source_url="https://other.example/b", source_group="other",
            primary_source=False, **extra))
        assert risk.decision_at(data, T).score == 0


def test_half_life_and_repost_does_not_reset_or_amplify():
    later = T + 6 * risk.HOUR_MS
    data = snapshot(observation(), observation(id="b", source_url="https://example.com/b",
                    first_seen_at_utc=risk.iso(later), assessed_at_utc=risk.iso(later)), at=later)
    decision = risk.decision_at(data, later)
    assert decision.score == pytest.approx(50)
    assert decision.risk_multiplier == pytest.approx(0.625)


def test_retraction_is_point_in_time():
    data = snapshot(observation(), observation(id="retracted", verified=False,
                    assessed_at_utc=risk.iso(T + 1000)))
    assert risk.decision_at(data, T).score == 100
    assert risk.decision_at(data, T + 1000).score == 0


@pytest.mark.parametrize("status", ["error", "truncated"])
def test_feed_failure_is_not_calm_and_future_success_does_not_fix_history(status):
    data = snapshot()
    data["polls"][0]["status"] = status
    data["polls"].append({"available_at_utc": risk.iso(T + 1000), "status": "ok"})
    assert not risk.decision_at(data, T).allowed
    assert risk.decision_at(data, T + 1000).allowed


def test_stale_unknown_and_empty_healthy_are_distinct():
    data = snapshot()
    assert risk.decision_at(data, T).allowed
    assert risk.decision_at(data, T - 1).feed_status == "unavailable"
    assert risk.decision_at(data, T + 2 * risk.HOUR_MS + 1).feed_status == "stale"
    assert not risk.decision_at(data, T + 2 * risk.HOUR_MS + 1).allowed


@pytest.mark.parametrize("field,value", [("severity", float("nan")), ("confidence", 2),
    ("half_life_hours", 0), ("half_life_hours", float("inf")), ("verified", "false"),
    ("first_seen_at_utc", "2026-09-01T00:00:00"), ("evidence_note", "")])
def test_invalid_inputs_rejected(field, value):
    with pytest.raises(ValueError):
        risk.validate_snapshot(snapshot(observation(**{field: value})))


@pytest.mark.parametrize("macro", [1.0, 0.6, 0.2])
def test_overlay_uses_stricter_limit_without_double_discount_and_preserves_inputs(macro):
    trade = {"entry_time_utc": risk.iso(T), "initial_qty": 10 * macro,
             "net_pnl": -100 * macro, "fees": 5 * macro, "macro_risk_multiplier": macro}
    sleeves = [{"trades": [trade]}]
    saved = copy.deepcopy(sleeves)
    adjusted, diagnostics = risk.apply_overlay(sleeves, snapshot(observation()))
    result = adjusted[0]["trades"][0]
    assert result["initial_qty"] == pytest.approx(10 * min(macro, 0.25))
    assert result["net_pnl"] == pytest.approx(-100 * min(macro, 0.25))
    assert result["fees"] == pytest.approx(5 * min(macro, 0.25))
    assert sleeves == saved
    assert diagnostics["blocked_feed"] == 0


def test_unavailable_blocks_entries_only():
    sleeves = [{"trades": [{"entry_time_utc": risk.iso(T)}]}]
    adjusted, diagnostics = risk.apply_overlay(sleeves, risk.empty_snapshot())
    assert adjusted[0]["trades"] == []
    assert diagnostics["blocked_feed"] == 1


def test_collector_deduplicates_urls_and_uses_actual_receipt_time():
    data = risk.empty_snapshot()
    articles = [{"url": "https://example.com/a?utm_source=test", "title": "New sanctions",
                 "seendate": "20200101T000000Z"},
                {"url": "https://example.com/a", "title": "New sanctions"}]
    assert runner.ingest_articles(data, articles, T) == 1
    assert runner.ingest_articles(data, articles, T + risk.HOUR_MS) == 0
    row = data["observations"][0]
    assert risk.utc_ms(row["first_seen_at_utc"]) == T
    assert not row["verified"]
    assert row["category"] == "trade_policy"
    risk.validate_snapshot(data)


def test_network_failure_retains_evidence_and_records_failure():
    data = snapshot(observation())
    client = Mock()
    client.get.side_effect = requests.Timeout()
    poll = runner.collect(data, client)
    assert poll["status"] == "error"
    assert len(data["observations"]) == 1
    assert len(data["polls"]) == 2


def test_bbc_rss_is_unverified_and_preserves_source_coverage():
    client = Mock()
    client.get.return_value.content = b'<rss><channel><item><title>New sanctions</title><link>https://www.bbc.com/news/a</link><pubDate>Tue, 01 Sep 2026 00:00:00 GMT</pubDate></item></channel></rss>'
    data = risk.empty_snapshot()
    poll = runner.collect(data, client, provider="bbc-world")
    assert poll["status"] == "ok"
    assert poll["coverage"] == "single publisher RSS"
    assert len(data["observations"]) == 1
    assert not data["observations"][0]["verified"]


def test_truncation_and_malformed_response_are_not_healthy():
    client = Mock()
    client.get.return_value.json.return_value = {"articles": [
        {"url": "https://example.com/a", "title": "Test"}] * 250}
    assert runner.collect(risk.empty_snapshot(), client)["status"] == "truncated"
    data = risk.empty_snapshot()
    client.get.return_value.json.return_value = {"articles": [
        {"url": "https://example.com/a", "title": "Test"}, {"title": "bad"}]}
    assert runner.collect(data, client)["status"] == "error"
    assert data["observations"] == []


def test_review_appends_instead_of_backdating(monkeypatch):
    data = snapshot(observation(verified=False))
    before = copy.deepcopy(data["observations"][0])
    monkeypatch.setattr(runner.time, "time", lambda: (T + 1000) / 1000)
    runner.review(data, SimpleNamespace(observation_id="a", event_id="reviewed-event",
        source_group="original-wire", severity=0.9, surprise=0.8, novelty=1,
        confidence=0.95, half_life_hours=6, retract=False, primary_source=True,
        evidence_note="Checked primary release"))
    assert data["observations"][0] == before
    assert len(data["observations"]) == 2
    assert risk.decision_at(data, T).score == 0
    assert risk.decision_at(data, T + 1000).score > 0


def test_snapshot_roundtrip_and_duplicate_id(tmp_path):
    path = tmp_path / "snapshot.json"
    original = snapshot(observation())
    risk.save_snapshot(path, original)
    assert risk.load_snapshot(path) == original
    original["observations"].append(observation())
    with pytest.raises(ValueError, match="Duplicate"):
        risk.save_snapshot(path, original)
