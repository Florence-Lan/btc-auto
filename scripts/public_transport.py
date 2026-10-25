"""Safe diagnostics shared by credential-free public HTTP transports.

Network exception messages can contain proxy credentials and query strings.
Only class names, fixed diagnostic enums and bounded numbers reach status/logs.
This module preserves the configured network route and TLS verification.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

import requests


_CURL_CATEGORIES = {
    5: "proxy_resolution", 6: "dns", 7: "connection", 18: "incomplete_response",
    28: "timeout", 35: "tls", 51: "tls_verification", 52: "empty_response",
    56: "receive", 58: "tls_certificate", 60: "tls_verification", 77: "tls_certificate",
}

INVALID_PUBLIC_RESPONSE_REASONS = frozenset({
    "clock_payload_invalid", "clock_timestamp_invalid", "clock_timing_invalid",
    "clock_roundtrip_exceeded", "clock_receipt_stale", "api_code_invalid",
    "response_not_json", "response_null",
})
MAX_DIAGNOSTIC_ELAPSED_MS = 86_400_000
PUBLIC_ELAPSED_SCOPES = frozenset({"network", "caller_wait"})
_SAFE_CATEGORIES = frozenset(_CURL_CATEGORIES.values()) | {
    "invalid_response", "proxy", "request", "transport_unavailable", "curl",
}


def _reason_code(value):
    # Exact primitive types also reject objects with hostile __str__/__eq__.
    return value if type(value) is str and value in INVALID_PUBLIC_RESPONSE_REASONS else None


def _elapsed_ms(value):
    if type(value) not in (int, float):
        return None
    try:
        valid = math.isfinite(value) and 0 <= value <= MAX_DIAGNOSTIC_ELAPSED_MS
    except OverflowError:
        return None
    return value if valid else None


def _elapsed_scope(value):
    return value if type(value) is str and value in PUBLIC_ELAPSED_SCOPES else None


def _identifier(value, default=None):
    return value if type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value) else default


def _code(value):
    return value if type(value) is int and -(2**31) <= value < 2**31 else None


def _safe_failure(failure):
    """Keep only safe existing fields and validated optional diagnostics."""
    if type(failure) is not dict:
        failure = {}
    transport = failure.get("transport")
    category = failure.get("category")
    result = {"transport": transport if type(transport) is str and transport in ("requests", "curl") else "unknown",
              "error_type": _identifier(failure.get("error_type"), "UnknownError"),
              "category": category if type(category) is str and category in _SAFE_CATEGORIES else "request"}
    chain = failure.get("error_chain")
    if type(chain) in (list, tuple):
        result["error_chain"] = [name for raw in chain[:16] if (name := _identifier(raw)) is not None]
    for field in ("errno", "winerror"):
        values = failure.get(field)
        if type(values) in (list, tuple):
            codes = [number for raw in values[:16] if (number := _code(raw)) is not None]
            if codes:
                result[field] = codes
    number = _code(failure.get("curl_exit_code"))
    if number is not None:
        result["curl_exit_code"] = number
    reason, elapsed = _reason_code(failure.get("reason_code")), _elapsed_ms(failure.get("elapsed_ms"))
    if reason is not None:
        result["reason_code"] = reason
    if elapsed is not None:
        result["elapsed_ms"] = elapsed
        scope = _elapsed_scope(failure.get("elapsed_scope"))
        if scope is not None:
            result["elapsed_scope"] = scope
    return result


class InvalidPublicResponse(requests.RequestException):
    """A public transport returned a response that cannot supply usable JSON."""

    def __init__(self, *args, reason_code=None, elapsed_ms=None, elapsed_scope=None, **kwargs) -> None:
        # Preserve RequestException construction/requests response metadata. Raw
        # messages are never read by safe_transport_error or aggregate details.
        super().__init__(*args, **kwargs)
        self.reason_code = _reason_code(reason_code)
        self.elapsed_ms = _elapsed_ms(elapsed_ms)
        self.elapsed_scope = _elapsed_scope(elapsed_scope)


class PublicCurlError(requests.ConnectionError):
    def __init__(self, message: str, *, curl_exit_code: int | None = None,
                 category: str | None = None) -> None:
        super().__init__(message)
        self.curl_exit_code = curl_exit_code
        self.transport_category = category or _CURL_CATEGORIES.get(curl_exit_code, "curl")


def safe_transport_error(exc: BaseException) -> dict[str, Any]:
    """Describe an error without rendering any exception messages or URLs."""
    pending = [exc]
    seen: set[int] = set()
    names: list[str] = []
    codes: dict[str, list[int]] = {"errno": [], "winerror": []}
    while pending and len(seen) < 16:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = _identifier(type(current).__name__, "UnknownError")
        if name not in names:
            names.append(name)
        for field, values in codes.items():
            value = getattr(current, field, None)
            if _code(value) is not None and value not in values:
                values.append(value)
        # urllib3 often stores the socket error in reason/args rather than cause.
        for child in (current.__cause__, current.__context__, getattr(current, "reason", None),
                      *current.args):
            if isinstance(child, BaseException):
                pending.append(child)

    category = getattr(exc, "transport_category", None)
    if type(category) is not str or category not in _SAFE_CATEGORIES:
        category = None
    if category is None:
        if isinstance(exc, InvalidPublicResponse):
            category = "invalid_response"
        elif any("Proxy" in name for name in names):
            category = "proxy"
        elif any("SSL" in name or "Certificate" in name for name in names):
            category = "tls"
        elif any("Timeout" in name for name in names):
            category = "timeout"
        elif "ChunkedEncodingError" in names:
            category = "incomplete_response"
        elif any("gaierror" == name or "NameResolution" in name for name in names):
            category = "dns"
        else:
            category = "connection" if isinstance(exc, requests.ConnectionError) else "request"
    result: dict[str, Any] = {"error_type": _identifier(type(exc).__name__, "UnknownError"), "error_chain": names,
                              "category": category}
    result.update({key: values for key, values in codes.items() if values})
    curl_code = getattr(exc, "curl_exit_code", None)
    if _code(curl_code) is not None:
        result["curl_exit_code"] = curl_code
    reason, elapsed = _reason_code(getattr(exc, "reason_code", None)), _elapsed_ms(getattr(exc, "elapsed_ms", None))
    if reason is not None:
        result["reason_code"] = reason
    if elapsed is not None:
        result["elapsed_ms"] = elapsed
        scope = _elapsed_scope(getattr(exc, "elapsed_scope", None))
        if scope is not None:
            result["elapsed_scope"] = scope
    return result


class PublicTransportError(requests.ConnectionError):
    """A bounded sequence of public transport attempts failed."""

    def __init__(self, message: str, failures: list[dict[str, Any]]) -> None:
        # `message` is the caller's fixed aggregate label, never str(cause).
        self.failures = [_safe_failure(failure) for failure in failures]
        details = []
        for failure in self.failures:
            value = f"{failure['transport']}={failure['error_type']}:{failure['category']}"
            if "curl_exit_code" in failure:
                value += f"(curl {failure['curl_exit_code']})"
            if failure.get("errno"):
                value += f"(errno {','.join(map(str, failure['errno']))})"
            if failure.get("winerror"):
                value += f"(winerror {','.join(map(str, failure['winerror']))})"
            if "reason_code" in failure:
                value += f"(reason={failure['reason_code']})"
            if "elapsed_ms" in failure:
                scope = f",elapsed_scope={failure['elapsed_scope']}" if "elapsed_scope" in failure else ""
                value += f"(elapsed_ms={failure['elapsed_ms']}{scope})"
            details.append(value)
        super().__init__(f"{message} [{'; '.join(details)}]")


def public_curl_environment() -> dict[str, str]:
    """Keep proxy/CA choices and Windows runtime variables, never API secrets.

    Windows commonly exposes Path rather than PATH; matching is case-insensitive
    while preserving original key spelling. No proxy is added or bypassed.
    """
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP",
               "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE",
               "SSL_CERT_DIR", "CURL_CA_BUNDLE"}
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}
