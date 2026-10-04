"""A degraded decision clock never turns stale prices into valid executions."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decision_runtime as runtime


NOW = 1_791_027_000_000


@pytest.fixture
def clock(monkeypatch):
    value = {"wall": NOW / 1000, "monotonic": 1000.0}
    monkeypatch.setattr(runtime.time, "time", lambda: value["wall"])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: value["monotonic"])
    return value


def client():
    return SimpleNamespace(base_url="https://fapi.binance.com",
                           server_time_ms=Mock(return_value=NOW),
                           mark_price_observation=Mock(return_value={"price": 80_000, "time_ms": NOW - 1000}))


@pytest.mark.parametrize("age", [-5000, 0, 60_000])
def test_mark_age_boundaries_are_inclusive(age):
    assert runtime.validate_mark({"price": "80000", "time_ms": NOW - age}, NOW) == {
        "price": 80000.0, "time_ms": NOW - age, "age_ms": age}


@pytest.mark.parametrize("observation", [
    {"price": 80000, "time_ms": NOW - 60001},
    {"price": 80000, "time_ms": NOW + 5001},
    {"price": float("nan"), "time_ms": NOW},
    {"price": float("inf"), "time_ms": NOW},
    {"price": 0, "time_ms": NOW},
    {"price": True, "time_ms": NOW},
    {"price": 80000, "time_ms": 0},
    {"price": 80000, "time_ms": True},
    {"price": 80000, "time_ms": NOW + .5},
    {"price": 80000},
    None,
])
def test_bad_marks_are_rejected(observation):
    with pytest.raises(ValueError):
        runtime.validate_mark(observation, NOW)


def test_primary_clock_never_fetches_mark(clock):
    source = client()
    reading = runtime.resolve_clock(source)
    assert reading["time_ms"] == NOW
    assert reading["source"] == "exchange"
    assert reading["primary_error"] is None
    assert reading["degraded"] is False
    source.mark_price_observation.assert_not_called()


def test_time_outage_uses_fresh_mark_and_preserves_reason(clock):
    source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    reading = runtime.resolve_clock(source)
    assert reading["time_ms"] == NOW - 1000
    assert reading["source"] == "mark_endpoint"
    assert reading["degraded"] is True
    assert reading["primary_error"] == "OSError: time disconnected"
    assert reading["errors"] == {"exchange_clock": reading["primary_error"]}


def test_validated_existing_mark_can_avoid_a_second_request(clock):
    source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    reading = runtime.resolve_clock(source, mark_observation={"price": 80000, "time_ms": NOW})
    assert reading["source"] == "mark_endpoint"
    source.mark_price_observation.assert_not_called()


def test_all_clock_sources_missing_raise_without_inventing_local_time(clock):
    source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    with pytest.raises(RuntimeError, match="No valid simulation decision clock"):
        runtime.resolve_clock(source)


@pytest.mark.parametrize("age", [-5001, 60001])
def test_invalid_fallback_mark_cannot_create_an_anchor(clock, age):
    source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    source.mark_price_observation.return_value = {"price": 80000, "time_ms": NOW - age}
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source)
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source)


def test_anchor_advances_monotonically_despite_wall_clock_jump(clock):
    source = client()
    runtime.resolve_clock(source)
    source.server_time_ms.side_effect = OSError("time disconnected")
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    clock.update(wall=clock["wall"] - 3600, monotonic=1005.0)
    reading = runtime.resolve_clock(source)
    assert reading["time_ms"] == NOW + 5000
    assert reading["source"] == "exchange_anchor"
    assert reading["anchor_age_ms"] == 5000
    assert set(reading["errors"]) == {"exchange_clock", "mark_price"}


def test_failed_anchor_reads_do_not_extend_the_expiry(clock):
    source = client()
    runtime.resolve_clock(source)
    source.server_time_ms.side_effect = OSError("time disconnected")
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    clock["monotonic"] = 1060.0
    assert runtime.resolve_clock(source)["anchor_age_ms"] == 60000
    clock["monotonic"] = 1060.001
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source)


def test_mark_fallback_does_not_renew_server_anchor(clock):
    source = client()
    runtime.resolve_clock(source)
    source.server_time_ms.side_effect = OSError("time disconnected")
    clock["monotonic"] = 1030.0
    source.mark_price_observation.return_value = {"price": 80000, "time_ms": NOW + 30000}
    assert runtime.resolve_clock(source)["source"] == "mark_endpoint"
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    clock["monotonic"] = 1060.001
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source)


def test_duplicate_old_marks_cannot_create_an_indefinitely_renewing_anchor(clock):
    source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    assert runtime.resolve_clock(source)["source"] == "mark_endpoint"
    clock.update(wall=(NOW + 60001) / 1000, monotonic=1060.001)
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source)


@pytest.mark.parametrize("mutation", ["source_change", "monotonic_backwards", "new_client"])
def test_invalid_anchor_cannot_be_used(clock, mutation):
    source = client()
    runtime.resolve_clock(source)
    if mutation == "source_change":
        source.base_url = "https://another.invalid"
    elif mutation == "monotonic_backwards":
        clock["monotonic"] = 999.0
    else:
        source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source)


def test_anchor_can_be_explicitly_disabled(clock):
    source = client()
    runtime.resolve_clock(source)
    source.server_time_ms.side_effect = OSError("time disconnected")
    source.mark_price_observation.side_effect = OSError("mark disconnected")
    with pytest.raises(RuntimeError):
        runtime.resolve_clock(source, allow_anchor=False)


def test_mark_fallback_uses_trusted_anchor_when_local_wall_clock_is_wrong(clock):
    source = client()
    runtime.resolve_clock(source)
    source.server_time_ms.side_effect = OSError("time disconnected")
    clock.update(wall=clock["wall"] + 3600, monotonic=1001.0)
    source.mark_price_observation.return_value = {"price": 80000, "time_ms": NOW + 1000}
    assert runtime.resolve_clock(source)["source"] == "mark_endpoint"


def test_recovery_restores_normal_primary_clock(clock):
    source = client()
    source.server_time_ms.side_effect = OSError("time disconnected")
    assert runtime.resolve_clock(source)["degraded"] is True
    source.server_time_ms.side_effect = None
    source.server_time_ms.return_value = NOW + 5000
    reading = runtime.resolve_clock(source)
    assert reading["source"] == "exchange"
    assert reading["degraded"] is False
    assert reading["errors"] == {}


def test_live_never_uses_existing_anchor_or_mark_fallback(clock):
    source = client()
    runtime.resolve_clock(source)
    failure = OSError("time disconnected")
    source.server_time_ms.side_effect = failure
    with pytest.raises(OSError) as caught:
        runtime.resolve_clock(source, "live")
    assert caught.value is failure
    source.mark_price_observation.assert_not_called()


def test_error_messages_redact_common_credentials(clock):
    source = client()
    source.server_time_ms.side_effect = OSError("https://user:password@example.invalid/time?api_key=secret&signature=signed")
    reading = runtime.resolve_clock(source)
    assert "password" not in reading["primary_error"]
    assert "secret" not in reading["primary_error"]
    assert "signed" not in reading["primary_error"]


def test_judgment_heartbeat_is_separate_from_last_decision_time(tmp_path):
    path = tmp_path / "strategy_judgment.json"
    payload = {"status": "degraded", "judgment_status": "evaluated",
               "last_judged_at_ms": NOW - 300000, "clock_source": "mark_endpoint",
               "execution_status": "execution_deferred", "errors": {"price": "unavailable"}}
    written = runtime.write_judgment(payload, path, now_ms=NOW)
    assert written["checked_at_ms"] == NOW
    assert not list(tmp_path.glob("*.tmp"))
    view = runtime.status_view(path, now_ms=NOW + 120000)
    assert view["status"] == "degraded"
    assert view["judgment_status"] == "evaluated"
    assert view["last_judged_at_ms"] == NOW - 300000
    assert view["heartbeat_age_seconds"] == 120
    assert runtime.status_view(path, now_ms=NOW + 120001)["status"] == "stale"


def test_wrong_generation_cannot_display_a_previous_healthy_judgment(tmp_path):
    path, report = tmp_path / "judgment.json", tmp_path / "report.json"
    runtime.write_judgment({"status": "healthy", "report_path": str(report),
                            "account_epoch": "original"}, path, now_ms=NOW)
    assert runtime.status_view(path, now_ms=NOW, report_path=report,
                               account_epoch="original")["status"] == "healthy"
    assert runtime.status_view(path, now_ms=NOW, report_path=tmp_path / "other.json")["status"] == "not_observed"
    assert runtime.status_view(path, now_ms=NOW, account_epoch="reset")["status"] == "not_observed"


@pytest.mark.parametrize("value", ["{broken", "[]", '{"checked_at_ms": "bad", "errors": []}'])
def test_damaged_judgment_status_is_readable(tmp_path, value):
    path = tmp_path / "judgment.json"
    path.write_text(value)
    assert runtime.status_view(path, now_ms=NOW)["status"] in {"not_observed", "unavailable"}


def test_future_heartbeat_is_not_displayed_as_current(tmp_path):
    path = tmp_path / "judgment.json"
    runtime.write_judgment({"status": "healthy"}, path, now_ms=NOW)
    assert runtime.status_view(path, now_ms=NOW - 5001)["status"] == "stale"


def test_root_judgment_status_derives_health_and_invalidates_stale_decision(tmp_path):
    path = tmp_path / "judgment.json"
    runtime.write_judgment({"decision_status": "evaluated", "execution_status": "completed",
                           "clock": {"degraded": False}, "signal_time_ms": NOW-300000}, path, now_ms=NOW)
    current = runtime.status_view(path, now_ms=NOW)
    assert current["status"] == "healthy"
    assert current["decision_status"] == "evaluated"
    stale = runtime.status_view(path, now_ms=NOW+120001)
    assert stale["status"] == "stale"
    assert stale["decision_status"] == "source_stale"
    assert stale["signal_time_ms"] == NOW-300000


def test_execution_deferral_keeps_evaluated_judgment_visible(tmp_path):
    path = tmp_path / "judgment.json"
    runtime.write_judgment({"decision_status": "evaluated", "execution_status": "deferred"}, path, now_ms=NOW)
    view = runtime.status_view(path, now_ms=NOW)
    assert view["status"] == "degraded"
    assert view["decision_status"] == "evaluated"
