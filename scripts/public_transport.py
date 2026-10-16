"""Safe diagnostics shared by credential-free public HTTP transports.

Network exception messages can contain proxy credentials and query strings.
Only exception class names and numeric error codes may reach status/log output.
This module preserves the configured network route and TLS verification.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_CURL_CATEGORIES = {
    5: "proxy_resolution", 6: "dns", 7: "connection", 18: "incomplete_response",
    28: "timeout", 35: "tls", 51: "tls_verification", 52: "empty_response",
    56: "receive", 58: "tls_certificate", 60: "tls_verification", 77: "tls_certificate",
}


class InvalidPublicResponse(requests.RequestException):
    """A public transport returned a response that cannot supply usable JSON."""


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
        name = type(current).__name__
        if name not in names:
            names.append(name)
        for field, values in codes.items():
            value = getattr(current, field, None)
            if isinstance(value, int) and not isinstance(value, bool) and value not in values:
                values.append(value)
        # urllib3 often stores the socket error in reason/args rather than cause.
        for child in (current.__cause__, current.__context__, getattr(current, "reason", None),
                      *current.args):
            if isinstance(child, BaseException):
                pending.append(child)

    category = getattr(exc, "transport_category", None)
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
    result: dict[str, Any] = {"error_type": type(exc).__name__, "error_chain": names,
                              "category": category}
    result.update({key: values for key, values in codes.items() if values})
    curl_code = getattr(exc, "curl_exit_code", None)
    if isinstance(curl_code, int) and not isinstance(curl_code, bool):
        result["curl_exit_code"] = curl_code
    return result


class PublicTransportError(requests.ConnectionError):
    """A bounded sequence of public transport attempts failed."""

    def __init__(self, message: str, failures: list[dict[str, Any]]) -> None:
        self.failures = [dict(failure) for failure in failures]
        details = []
        for failure in self.failures:
            value = f"{failure['transport']}={failure['error_type']}:{failure['category']}"
            if "curl_exit_code" in failure:
                value += f"(curl {failure['curl_exit_code']})"
            if failure.get("errno"):
                value += f"(errno {','.join(map(str, failure['errno']))})"
            if failure.get("winerror"):
                value += f"(winerror {','.join(map(str, failure['winerror']))})"
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
