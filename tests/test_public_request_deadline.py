"""A hung public read must never hold an account worker or leak retry threads."""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_request_deadline import PublicRequestDeadline
import public_request_deadline as public_deadline


def test_hung_request_returns_on_total_deadline_and_retries_do_not_spawn_workers():
    deadline = PublicRequestDeadline(timeout_seconds=0.03)
    release = threading.Event()
    entered = threading.Event()
    finished = threading.Event()
    calls = []

    def hanging_read():
        calls.append(1)
        entered.set()
        release.wait(2)
        finished.set()
        return "expired data"

    started = time.monotonic()
    try:
        with pytest.raises(requests.Timeout, match="total deadline"):
            deadline.run("mark", hanging_read)
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        for _ in range(5):
            with pytest.raises(requests.Timeout, match="still pending"):
                deadline.run("mark", hanging_read)
        assert calls == [1]
        assert deadline.run("clock", lambda: 123) == 123
    finally:
        release.set()
        assert finished.wait(1)
    # Acquire the helper's lock to wait for the completed worker's cleanup.
    for _ in range(100):
        with deadline._lock:
            if "mark" not in deadline._pending:
                break
        time.sleep(0.001)
    assert deadline.run("mark", lambda: "fresh data") == "fresh data"


def test_worker_cap_bounds_different_stalled_endpoints():
    deadline = PublicRequestDeadline(timeout_seconds=0.01, max_pending=1)
    release = threading.Event()
    try:
        with pytest.raises(requests.Timeout, match="total deadline"):
            deadline.run("mark", lambda: release.wait(2))
        other = Mock()
        with pytest.raises(requests.Timeout, match="worker limit"):
            deadline.run("another", other)
        other.assert_not_called()
    finally:
        release.set()


def test_process_worker_cap_also_bounds_recreated_clients(monkeypatch):
    monkeypatch.setattr(public_deadline, "_PUBLIC_WORKER_SLOTS", threading.BoundedSemaphore(1))
    release = threading.Event()
    first = PublicRequestDeadline(timeout_seconds=0.01)
    try:
        with pytest.raises(requests.Timeout, match="total deadline"):
            first.run("mark", lambda: release.wait(2))
        restarted = PublicRequestDeadline(timeout_seconds=0.01)
        callback = Mock()
        with pytest.raises(requests.Timeout, match="Process public request worker limit"):
            restarted.run("clock", callback)
        callback.assert_not_called()
    finally:
        release.set()


def test_public_read_exceptions_propagate_without_waiting_for_deadline():
    deadline = PublicRequestDeadline(timeout_seconds=1)
    failure = requests.ConnectionError("offline")
    callback = Mock(side_effect=failure)
    with pytest.raises(requests.ConnectionError) as result:
        deadline.run("mark", callback)
    assert result.value is failure
    assert deadline.run("mark", lambda: "recovered") == "recovered"


def test_completed_response_past_deadline_is_discarded_even_if_waiter_wakes_later(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(public_deadline.time, "monotonic", lambda: clock[0])
    deadline = PublicRequestDeadline(timeout_seconds=1)
    def late_response():
        clock[0] = 102.0
        return "stale response"
    with pytest.raises(requests.Timeout, match="total deadline"):
        deadline.run("mark", late_response)


def test_joining_concurrent_public_read_shares_one_network_request(monkeypatch):
    deadline = PublicRequestDeadline(timeout_seconds=1)
    release, entered, joined = threading.Event(), threading.Event(), threading.Event()
    def read_clock():
        entered.set()
        assert release.wait(2)
        return 123
    duplicate = Mock(side_effect=AssertionError("Joined callback must not run"))
    joining_thread = []
    def join_clock():
        joining_thread.append(threading.get_ident())
        return deadline.run("clock", duplicate, join_pending=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        original = pool.submit(deadline.run, "clock", read_clock)
        assert entered.wait(1)
        with deadline._lock:
            pending = deadline._pending["clock"]
        original_wait = pending.done.wait
        def tracked_wait(seconds):
            if threading.get_ident() in joining_thread:
                joined.set()  # the join already selected the original request
            return original_wait(seconds)
        monkeypatch.setattr(pending.done, "wait", tracked_wait)
        join = pool.submit(join_clock)
        assert joined.wait(1)
        assert not join.done()
        release.set()
        assert original.result(timeout=1) == 123
        assert join.result(timeout=1) == 123
    duplicate.assert_not_called()


def test_expired_pending_read_cannot_be_joined_or_renewed():
    deadline = PublicRequestDeadline(timeout_seconds=0.01)
    release = threading.Event()
    try:
        with pytest.raises(requests.Timeout, match="total deadline"):
            deadline.run("clock", lambda: release.wait(2))
        with pytest.raises(requests.Timeout, match="still pending"):
            deadline.run("clock", Mock(), join_pending=True)
    finally:
        release.set()
