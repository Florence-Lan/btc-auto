import gzip
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_multifactor_snapshot as download
import multifactor as mf
from factor_data_freshness import OIL_MAX_AGE_MS


@pytest.fixture
def clock(monkeypatch):
    value = [1_791_382_800_000]
    monkeypatch.setattr(download, "now_ms", lambda: value[0])
    return value


def oil_row(observed, value=100, first_seen=None):
    available = observed + 3 * mf.DAY
    return [observed, available, value, first_seen or available]


def collect(path, clock, **kwargs):
    return download.collect(path, clock[0] - 180 * mf.DAY, clock[0],
                            only_due=True, sources=["oil"], **kwargs)


def write_archive(path, rows, state, **metadata):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "series": {"oil": rows},
                   "metadata": {"source_status": {"oil": state}, **metadata}}, handle)


def test_http_success_with_stale_oil_retries_until_fresh_without_backfilling(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    old = oil_row(clock[0] - 9 * mf.DAY)
    fetch = Mock(side_effect=lambda *args: [old.copy()])
    monkeypatch.setattr(download, "fetch_fred", fetch)
    stale = collect(path, clock)
    original = stale["series"]["oil"][0].copy()
    state = stale["metadata"]["source_status"]["oil"]
    assert not state["ok"] and state["status"] == "stale"
    assert "age_days=9.000000" in state["error"]
    assert "max_age_days=7" in state["error"]
    assert state["last_success_ms"] is None
    assert state["next_retry_at_ms"] == clock[0] + 60_000

    before = path.read_bytes()
    clock[0] += 59_999
    collect(path, clock)
    assert fetch.call_count == 1 and path.read_bytes() == before
    clock[0] += 1
    again = collect(path, clock)
    assert fetch.call_count == 2
    assert again["metadata"]["source_status"]["oil"]["next_retry_at_ms"] == clock[0] + 60_000
    assert again["series"]["oil"] == [original]

    clock[0] += 60_000
    observed = clock[0] - 3 * mf.DAY
    # Revised overlap cannot replace the value or first collection time already
    # archived. The new date becomes available only at this actual receipt.
    fetch.side_effect = lambda *args: [oil_row(original[0], 999), oil_row(observed, 110)]
    recovered = collect(path, clock)
    state = recovered["metadata"]["source_status"]["oil"]
    assert state["ok"] and state["status"] == "healthy" and state["error"] is None
    assert state["consecutive_failures"] == 0 and state["last_success_ms"] == clock[0]
    assert state["next_retry_at_ms"] == clock[0] + 3_600_000
    assert recovered["metadata"]["errors"] == {}
    assert recovered["series"]["oil"] == [original, oil_row(observed, 110, clock[0])]
    snapshot = mf.Snapshot(recovered)
    assert snapshot.window("oil", clock[0] - 1, 20 * mf.DAY) == 100
    assert snapshot.window("oil", clock[0], OIL_MAX_AGE_MS) == 110


def test_legacy_healthy_but_expired_oil_is_immediately_due_and_preserves_metadata(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    old = oil_row(clock[0] - 8 * mf.DAY)
    previous_success = clock[0] - 3_600_000
    write_archive(path, [old], {"ok": True, "error": None,
                  "next_retry_at_ms": clock[0] + 3_600_000,
                  "last_success_ms": previous_success, "provider_note": "keep me"},
                  archive_note="also keep me", errors={})
    fetch = Mock(side_effect=lambda *args: [old.copy()])
    monkeypatch.setattr(download, "fetch_fred", fetch)
    migrated = collect(path, clock)
    fetch.assert_called_once()
    state = migrated["metadata"]["source_status"]["oil"]
    assert state["status"] == "stale" and not state["ok"]
    assert state["provider_note"] == "keep me"
    assert state["last_success_ms"] == previous_success
    assert migrated["metadata"]["archive_note"] == "also keep me"
    assert migrated["series"]["oil"] == [old]


def test_fresh_merged_archive_is_not_made_stale_by_older_incoming_rows(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    fresh = oil_row(clock[0] - 3 * mf.DAY)
    write_archive(path, [fresh], {"ok": True, "error": None, "next_retry_at_ms": 0,
                                "provider_note": "preserved"}, errors={})
    old = oil_row(clock[0] - 10 * mf.DAY)
    monkeypatch.setattr(download, "fetch_fred", Mock(side_effect=lambda *args: [old.copy()]))
    result = collect(path, clock)
    state = result["metadata"]["source_status"]["oil"]
    assert state["ok"] and state["status"] == "healthy"
    assert state["provider_note"] == "preserved"
    assert state["latest_observed_at_ms"] == fresh[0]
    assert result["series"]["oil"][-1] == fresh


def test_oil_network_failures_keep_bounded_backoff_and_archive(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    old = oil_row(clock[0] - 9 * mf.DAY)
    write_archive(path, [old], {"ok": False, "status": "stale", "error": "stale",
                  "next_retry_at_ms": 0, "consecutive_failures": 0}, errors={"oil": "stale"})
    fetch = Mock(side_effect=requests.exceptions.ConnectionError("offline"))
    monkeypatch.setattr(download, "fetch_fred", fetch)
    for count, seconds in enumerate((60, 120, 240, 480, 900, 900), 1):
        failed = collect(path, clock)
        state = failed["metadata"]["source_status"]["oil"]
        assert state["status"] == "error" and not state["ok"]
        assert state["consecutive_failures"] == count
        assert state["next_retry_at_ms"] == clock[0] + seconds * 1000
        assert failed["series"]["oil"] == [old]
        clock[0] += seconds * 1000 - 1
        collect(path, clock)
        assert fetch.call_count == count
        clock[0] += 1


def test_freshness_has_same_inclusive_seven_day_limit_and_ignores_future_availability(clock):
    boundary = oil_row(clock[0] - OIL_MAX_AGE_MS)
    assert download.oil_freshness_error([boundary], clock[0]) is None
    assert download.oil_freshness_error([boundary], clock[0] + 1) is not None
    future = oil_row(clock[0] - 3 * mf.DAY, first_seen=clock[0] + 1)
    assert download.oil_freshness_error([boundary, future], clock[0] + 1) is None
    assert download.oil_freshness_error([boundary, future], clock[0]) is None
    # At one millisecond past the old boundary the future receipt is still
    # excluded and cannot make the source look usable ahead of collection.
    future[3] = clock[0] + 2
    assert download.oil_freshness_error([boundary, future], clock[0] + 1) is not None
