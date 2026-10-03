import gzip
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import binance_terminal_client as binance
import download_multifactor_snapshot as download
import information_runtime as runtime
import multifactor


@pytest.fixture
def clock(monkeypatch):
    value = [1_790_000_000_000]
    monkeypatch.setattr(download, "now_ms", lambda: value[0])
    monkeypatch.setattr(runtime.time, "time", lambda: value[0] / 1000)
    runtime._refresh_failures.clear()
    yield value
    runtime._refresh_failures.clear()


def collect(path, clock, sources, **kwargs):
    return download.collect(path, clock[0] - 180 * multifactor.DAY, clock[0],
                            sources=sources, only_due=True, **kwargs)


def row(clock, value=100):
    return [clock[0] - 7200_000, clock[0] - 3600_000, value, clock[0]]


def test_failed_source_recovers_early_without_refetching_healthy_source(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    original = row(clock)
    fred = Mock(return_value=[original.copy()])
    failed = Mock(side_effect=requests.exceptions.SSLError("certificate error"))
    monkeypatch.setattr(download, "fetch_fred", fred)
    monkeypatch.setattr(download, "fetch_derivatives", failed)
    payload = collect(path, clock, ["fed_effective", "open_interest"])
    assert payload["metadata"]["source_status"]["open_interest"]["next_retry_at_ms"] == clock[0] + 60_000
    assert payload["metadata"]["source_status"]["fed_effective"]["next_retry_at_ms"] == clock[0] + 3600_000
    archive = path.read_bytes()

    clock[0] += 30_000
    collect(path, clock, ["fed_effective", "open_interest"])
    assert path.read_bytes() == archive
    assert failed.call_count == 1
    assert fred.call_count == 1

    clock[0] += 30_000
    recovered = Mock(return_value=[row(clock, 123)])
    monkeypatch.setattr(download, "fetch_derivatives", recovered)
    # A new call reads persisted schedules, as a restarted worker would.
    payload = collect(path, clock, ["fed_effective", "open_interest"])
    assert recovered.call_count == 1
    assert fred.call_count == 1
    assert payload["series"]["fed_effective"] == [original]
    assert payload["metadata"]["errors"] == {}
    state = payload["metadata"]["source_status"]["open_interest"]
    assert state["consecutive_failures"] == 0
    assert state["last_success_ms"] == clock[0]
    assert state["next_retry_at_ms"] == clock[0] + 3600_000


def test_failure_preserves_rows_and_recovery_never_rewrites_first_seen(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    original = row(clock, 100)
    fetch = Mock(return_value=[original.copy()])
    monkeypatch.setattr(download, "fetch_derivatives", fetch)
    collect(path, clock, ["open_interest"])
    clock[0] += 3600_000
    fetch.side_effect = requests.exceptions.SSLError("certificate error")
    failed = collect(path, clock, ["open_interest"])
    assert failed["series"]["open_interest"] == [original]
    assert failed["metadata"]["source_status"]["open_interest"]["last_success_ms"] == original[3]

    clock[0] += 60_000
    fetch.side_effect = None
    incoming = row(clock, 110)
    fetch.return_value = [[original[0], original[1], 999, clock[0]], incoming]
    recovered = collect(path, clock, ["open_interest"])
    assert recovered["series"]["open_interest"] == [original, incoming]
    assert multifactor.Snapshot(recovered).window("open_interest", clock[0] - 1, 4 * multifactor.HOUR) == 100
    assert multifactor.Snapshot(recovered).window("open_interest", clock[0], 4 * multifactor.HOUR) == 110


def test_backoff_is_bounded_and_repeated_worker_polls_do_not_retry(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    fetch = Mock(side_effect=requests.exceptions.ConnectionError("offline"))
    monkeypatch.setattr(download, "fetch_derivatives", fetch)
    for failures, delay in enumerate([60, 120, 240, 480, 900, 900], 1):
        payload = collect(path, clock, ["open_interest"])
        state = payload["metadata"]["source_status"]["open_interest"]
        assert state["consecutive_failures"] == failures
        assert state["next_retry_at_ms"] == clock[0] + delay * 1000
        clock[0] += delay * 1000 - 1
        collect(path, clock, ["open_interest"])
        assert fetch.call_count == failures
        clock[0] += 1


def test_legacy_partial_archive_retries_error_without_mtime_wait_for_all_sources(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"schema_version": 1, "series": {"fed_effective": [row(clock)], "open_interest": [row(clock)]},
                   "metadata": {"errors": {"open_interest": "SSLError: certificate error"}}}, handle)
    os.utime(path, ((clock[0] - 61_000) / 1000,) * 2)
    fred = Mock(return_value=[row(clock)])
    derivative = Mock(return_value=[row(clock)])
    monkeypatch.setattr(download, "fetch_fred", fred)
    monkeypatch.setattr(download, "fetch_derivatives", derivative)
    payload = collect(path, clock, ["fed_effective", "open_interest"])
    derivative.assert_called_once()
    fred.assert_not_called()
    assert payload["metadata"]["errors"] == {}


def test_shared_binance_ban_serializes_sources_and_force_cannot_bypass_pause(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    monkeypatch.setattr(binance, "COOLDOWN_PATH", tmp_path / "cooldown.json")
    limited = Mock(ok=False, status_code=429, headers={"Retry-After": "600"})
    limited.json.return_value = {"code": -1003, "msg": "Too many requests"}
    success = Mock(ok=True)
    success.json.side_effect = lambda: [row(clock)]
    session = Mock()
    session.get.side_effect = [limited, success, success]
    monkeypatch.setattr(binance.requests, "Session", Mock(return_value=session))
    monkeypatch.setattr(download, "fetch_derivatives", lambda name, start, end:
                        download.binance_json("/futures/data/" + name, {}))
    sources = ["open_interest", "taker_buy_sell"]
    failed = collect(path, clock, sources)
    until = clock[0] + 605_000
    assert session.get.call_count == 1
    for name in sources:
        state = failed["metadata"]["source_status"][name]
        assert state["provider_retry_at_ms"] == until
        assert state["next_retry_at_ms"] == until

    clock[0] += 60_000
    collect(path, clock, sources)
    collect(path, clock, sources, force=True)
    assert session.get.call_count == 1
    assert json.loads(binance.COOLDOWN_PATH.read_text())["retry_at_ms"] == until
    clock[0] = until + 1
    recovered = collect(path, clock, sources)
    assert session.get.call_count == 3
    assert recovered["metadata"]["errors"] == {}


def test_malformed_provider_is_isolated_from_successful_archive_update(tmp_path, monkeypatch, clock):
    path = tmp_path / "factors.json.gz"
    monkeypatch.setattr(download, "fetch_derivatives", lambda *args: [[1, 2, 3]])
    monkeypatch.setattr(download, "fetch_fred", lambda *args: [row(clock)])
    payload = collect(path, clock, ["open_interest", "fed_effective"])
    assert "open_interest" in payload["metadata"]["errors"]
    assert payload["series"]["open_interest"] == []
    assert payload["series"]["fed_effective"] == [row(clock)]
    multifactor.Snapshot(payload)


def test_runtime_failure_does_not_skip_other_inputs_and_uses_outer_backoff(tmp_path, monkeypatch, clock):
    factor = Mock(side_effect=OSError("archive unavailable"))
    public = Mock(side_effect=ValueError("calendar unavailable"))
    supplement = Mock(return_value={"status": {"options": {"ok": True}}})
    monkeypatch.setattr(runtime.download, "collect", factor)
    monkeypatch.setattr(runtime.public_context, "collect", public)
    monkeypatch.setattr(runtime.supplemental, "collect", supplement)
    paths = [tmp_path / "factors.gz", tmp_path / "public.json", tmp_path / "supplemental.json"]
    first = runtime.refresh_if_needed(*paths)
    assert not first["factors"]["ok"] and not first["public_context"]["ok"]
    assert first["supplemental"]["ok"]
    factor.assert_called_once()
    public.assert_called_once()
    supplement.assert_called_once()

    clock[0] += 30_000
    waiting = runtime.refresh_if_needed(*paths)
    assert waiting["factors"]["skipped"] == "retry_wait"
    assert factor.call_count == 1 and public.call_count == 1
    assert supplement.call_count == 2
    clock[0] += 30_000
    factor.side_effect = public.side_effect = None
    factor.return_value = public.return_value = {"metadata": {"errors": {}}}
    recovered = runtime.refresh_if_needed(*paths)
    assert recovered["factors"]["ok"] and recovered["public_context"]["ok"]
    assert runtime._refresh_failures == {}


def test_runtime_does_not_rely_on_new_partial_factor_archive_mtime(tmp_path, monkeypatch, clock):
    paths = [tmp_path / "factors.gz", tmp_path / "public.json", tmp_path / "supplemental.json"]
    for path in paths:
        path.touch()
    for path in paths:
        os.utime(path, (clock[0] / 1000,) * 2)
    factor = Mock(return_value={"metadata": {"errors": {"open_interest": "SSLError"}}})
    public = Mock()
    monkeypatch.setattr(runtime.download, "collect", factor)
    monkeypatch.setattr(runtime.public_context, "collect", public)
    monkeypatch.setattr(runtime.supplemental, "collect", Mock(return_value={}))
    result = runtime.refresh_if_needed(*paths)
    assert not result["factors"]["ok"]
    factor.assert_called_once_with(paths[0], clock[0] - 180 * multifactor.DAY, clock[0],
                                   only_due=True, force=False)
    public.assert_not_called()
