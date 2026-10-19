"""Bounded, credential-free HTTPS text reads for official feeds and calendars."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urljoin, urlsplit

import requests

from public_request_deadline import PublicRequestDeadline
from public_transport import (InvalidPublicResponse, PublicCurlError,
                              PublicTransportError, public_curl_environment,
                              safe_transport_error)

USER_AGENT = "btc-auto-public-research/1.0"
MAX_BODY_BYTES = 4 * 1024 * 1024


class PublicTextHttpError(requests.HTTPError):
    def __init__(self, status):
        self.status = status
        self.transport_category = "http"
        super().__init__(f"Public text endpoint HTTP {status}")


def _https_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment):
        raise ValueError("Public text reads require HTTPS without credentials")
    return parsed


def _decode(body):
    if not body or len(body) > MAX_BODY_BYTES:
        raise InvalidPublicResponse("Public text body is empty or exceeds its limit")
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InvalidPublicResponse("Public text body is not UTF-8") from None


class PublicTextTransport:
    """Try two TLS stacks on the same URL without bypassing HTTP denials.

    Both transports retain configured proxy and CA choices. A requests total
    deadline discards late data and bounds hung-worker retries. An HTTP 403/429
    is a provider response, not a reason to try a different network route.
    """

    def __init__(self):
        self.deadline = PublicRequestDeadline(timeout_seconds=9, max_pending=12)
        self._lock = threading.Lock()
        self._curl_preferred_until = {}

    def _requests_get(self, url):
        current = url
        for _ in range(5):
            _https_url(current)
            response = requests.get(current, headers={"User-Agent": USER_AGENT},
                                    timeout=(4, 7), allow_redirects=False, stream=True)
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    current = urljoin(current, response.headers.get("Location", ""))
                    _https_url(current)  # reject a downgrade before following it
                    continue
                if response.status_code != 200:
                    raise PublicTextHttpError(response.status_code)
                chunks, size = [], 0
                for chunk in response.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > MAX_BODY_BYTES:
                        raise InvalidPublicResponse("Public text body exceeds its limit")
                    chunks.append(chunk)
                return _decode(b"".join(chunks))
            finally:
                response.close()
        raise InvalidPublicResponse("Public text redirect limit exceeded")

    def _curl_get(self, url):
        executable = shutil.which("curl.exe") or shutil.which("curl")
        if not executable:
            raise PublicCurlError("Public text fallback is unavailable", category="unavailable")
        command = [executable, "-q", "--silent", "--show-error", "--location",
                   "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "4",
                   "--connect-timeout", "4", "--max-time", "12", "--user-agent", USER_AGENT,
                   "--write-out", "\n%{http_code}", "--url", url]
        # requests and curl name the CA override differently. Preserve the
        # existing requests trust choice; never disable TLS to recover a read.
        ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE")
        if ca_bundle:
            command.extend(["--capath" if Path(ca_bundle).is_dir() else "--cacert", ca_bundle])
        try:
            result = subprocess.run(command, capture_output=True, timeout=14, check=False,
                                    env=public_curl_environment(),
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired:
            raise PublicCurlError("Public text fallback exceeded its deadline", category="timeout") from None
        except OSError:
            raise PublicCurlError("Public text fallback could not start", category="process") from None
        if result.returncode:
            raise PublicCurlError("Public text fallback failed", curl_exit_code=result.returncode)
        body, separator, status = result.stdout.rpartition(b"\n")
        if not separator or not status.isdigit() or not 100 <= int(status) <= 599:
            raise InvalidPublicResponse("Public text fallback has no HTTP status")
        if int(status) != 200:
            raise PublicTextHttpError(int(status))
        return _decode(body)

    def get(self, url):
        origin = _https_url(url).netloc
        with self._lock:
            preferred = time.monotonic() < self._curl_preferred_until.get(origin, 0)
        transports = ("curl", "requests") if preferred else ("requests", "curl")
        failures = []
        for transport in transports:
            try:
                value = (self.deadline.run(url, lambda: self._requests_get(url), join_pending=True)
                         if transport == "requests" else self._curl_get(url))
            except requests.RequestException as exc:
                if isinstance(exc, PublicTextHttpError) and exc.status not in {408, 500, 502, 503, 504}:
                    raise exc from None
                failures.append({"transport": transport, **safe_transport_error(exc)})
                with self._lock:
                    if transport == "curl":
                        self._curl_preferred_until.pop(origin, None)
                continue
            with self._lock:
                if transport == "curl":
                    self._curl_preferred_until[origin] = time.monotonic() + 300
                else:
                    self._curl_preferred_until.pop(origin, None)
            return value
        raise PublicTransportError("Both public text transports failed", failures) from None
