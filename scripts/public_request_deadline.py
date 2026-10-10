"""Bound public read latency without accumulating workers for a stalled endpoint."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Hashable

import requests


_PUBLIC_WORKER_SLOTS = threading.BoundedSemaphore(64)


@dataclass
class _Pending:
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: BaseException | None = None
    timed_out: bool = False


class PublicRequestDeadline:
    """Discard late public reads; at most one outstanding worker per key.

    Requests' socket timeout is not a total request deadline. A daemon worker
    keeps a slow DNS lookup or trickling response from blocking account polling.
    It cannot cancel that socket, so repeat calls for its key fail immediately
    until it finishes. Local and process-wide caps bound abandoned workers,
    including clients recreated on successive refresh cycles. Callbacks
    must perform public reads only, close responses, and decode rate limits
    before returning. Never use this wrapper to retry or abandon signed writes.
    """

    def __init__(self, timeout_seconds: float = 17, max_pending: int = 32) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or max_pending < 1:
            raise ValueError("Invalid public request deadline or worker limit")
        self.timeout_seconds = timeout_seconds
        self.max_pending = max_pending
        self._lock = threading.Lock()
        self._pending: dict[Hashable, _Pending] = {}

    def run(self, key: Hashable, callback: Callable[[], Any], *, join_pending: bool = False) -> Any:
        """Optionally share an in-flight read without renewing its deadline.

        Joined callers receive the same value object; copy mutable observations
        before adding caller-local receipt metadata. Expired reads never join.
        """
        start_worker = False
        worker_slots = _PUBLIC_WORKER_SLOTS
        with self._lock:
            pending = self._pending.get(key)
            if pending is not None:
                if not join_pending or pending.timed_out:
                    raise requests.Timeout("Previous public request is still pending")
            else:
                if len(self._pending) >= self.max_pending:
                    raise requests.Timeout("Public request worker limit reached")
                if not worker_slots.acquire(blocking=False):
                    raise requests.Timeout("Process public request worker limit reached")
                pending = _Pending(deadline=time.monotonic() + self.timeout_seconds)
                self._pending[key] = pending
                start_worker = True

        def fetch() -> None:
            try:
                pending.value = callback()
            except BaseException as exc:
                pending.error = exc
            finally:
                with self._lock:
                    pending.timed_out = pending.timed_out or time.monotonic() > pending.deadline
                    self._pending.pop(key, None)
                    worker_slots.release()
                    pending.done.set()

        if start_worker:
            worker = threading.Thread(target=fetch, name="public-request", daemon=True)
            try:
                worker.start()
            except BaseException:
                with self._lock:
                    self._pending.pop(key, None)
                    worker_slots.release()
                raise
        if not pending.done.wait(max(0, pending.deadline - time.monotonic())):
            with self._lock:
                pending.timed_out = True
            raise requests.Timeout("Public request exceeded its total deadline")
        if pending.timed_out:
            raise requests.Timeout("Public request exceeded its total deadline")
        if pending.error is not None:
            raise pending.error
        return pending.value
