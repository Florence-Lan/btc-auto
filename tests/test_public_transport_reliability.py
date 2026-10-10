import json
import subprocess
import sys
import threading
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import binance_terminal_client as binance
from public_request_deadline import PublicRequestDeadline


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(binance, "COOLDOWN_PATH", tmp_path / "cooldown.json")
    monkeypatch.setattr(binance, "RULES_CACHE_PATH", tmp_path / "rules.json")
    monkeypatch.setattr(binance.time, "time", lambda: 1_790_000_000.0)
    monkeypatch.setattr(binance.time, "monotonic", lambda: 100.0)
    result = binance.BinanceTerminalClient()
    result.session = Mock()
    return result


def response(payload, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    result.headers.update(headers or {})
    return result


def exchange_info(step="0.001"):
    return {"symbols": [{"symbol": "BTCUSDT", "filters": [
        {"filterType": "LOT_SIZE", "stepSize": step, "minQty": "0.001", "maxQty": "1000"},
        {"filterType": "MIN_NOTIONAL", "notional": "100"},
    ]}]}


@pytest.mark.parametrize("failure", [requests.exceptions.SSLError("EOF"),
                                     requests.exceptions.ConnectionError("reset"),
                                     requests.exceptions.Timeout("timeout")])
def test_connection_failure_uses_same_public_endpoint_and_prefers_working_transport(client, monkeypatch, failure):
    client.session.get.side_effect = failure
    fallback = Mock(return_value=response({"serverTime": 1_790_000_000_000}))
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    assert client.server_time_ms() == 1_790_000_000_000
    fallback.assert_called_once_with("/fapi/v1/time", None)
    assert client.last_public_transport == {
        "transport": "curl", "endpoint": "/fapi/v1/time", "observed_at_ms": 1_790_000_000_000,
        "ok": True, "fallback": True, "primary_error_type": type(failure).__name__,
    }
    assert client.server_time_ms() == 1_790_000_000_000
    assert client.session.get.call_count == 1
    assert fallback.call_count == 2
    monkeypatch.setattr(binance.time, "monotonic", lambda: 401.0)
    client.server_time_ms()
    assert client.session.get.call_count == 2


def test_preferred_curl_connection_failure_can_recover_via_requests(client, monkeypatch):
    client._curl_preferred_until = 500
    fallback = Mock(side_effect=requests.exceptions.ConnectionError("curl offline"))
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    client.session.get.return_value = response({"serverTime": 1_790_000_000_000})
    assert client.server_time_ms() == 1_790_000_000_000
    assert client.last_public_transport["transport"] == "requests"
    assert client._curl_preferred_until == 0


def test_hung_requests_transport_falls_back_and_never_returns_its_late_data(client, monkeypatch):
    client._public_request_deadline = PublicRequestDeadline(timeout_seconds=0.02)
    release = threading.Event()
    def hanging_response(*args, **kwargs):
        release.wait(2)
        return response({"serverTime": 1})
    client.session.get.side_effect = hanging_response
    fallback = Mock(return_value=response({"serverTime": 1_790_000_000_000}))
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    try:
        assert client.server_time_ms() == 1_790_000_000_000
        assert client.last_public_transport["primary_error_type"] == "Timeout"
        # Retrying the primary cannot create another abandoned socket worker.
        client._curl_preferred_until = 0
        assert client.server_time_ms() == 1_790_000_000_000
        assert client.session.get.call_count == 1
    finally:
        release.set()
    _wait_for_public_workers(client)
    client.session.get.side_effect = None
    client.session.get.return_value = response({"serverTime": 1_790_000_001_000})
    client._curl_preferred_until = 0
    assert client.server_time_ms() == 1_790_000_001_000
    assert client.last_public_transport["transport"] == "requests"


def _wait_for_public_workers(client):
    for _ in range(1000):
        with client._public_request_deadline._lock:
            if not client._public_request_deadline._pending:
                return
        # The fixture freezes binance.time.monotonic, so use a bounded event wait.
        threading.Event().wait(0.001)
    pytest.fail("Public transport worker did not finish")


def test_late_rate_limit_from_abandoned_request_still_blocks_future_requests(client, monkeypatch):
    client._public_request_deadline = PublicRequestDeadline(timeout_seconds=0.02)
    release = threading.Event()
    def delayed_ban(*args, **kwargs):
        release.wait(2)
        return response({"code": -1003, "msg": "Too many requests"}, 429, {"Retry-After": "120"})
    client.session.get.side_effect = delayed_ban
    fallback = Mock(return_value=response({"serverTime": 1_790_000_000_000}))
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    try:
        assert client.server_time_ms() == 1_790_000_000_000
    finally:
        release.set()
    _wait_for_public_workers(client)
    with pytest.raises(binance.BinanceApiError, match="paused until"):
        client.server_time_ms()
    fallback.assert_called_once()
    assert client.session.get.call_count == 1


@pytest.mark.parametrize("transport", ["requests", "curl"])
@pytest.mark.parametrize("status", [418, 429])
def test_rate_limit_is_persisted_and_never_bypassed_by_other_transport(client, monkeypatch, transport, status):
    limited = response({"code": -1003, "msg": "Too many requests"}, status, {"Retry-After": "120"})
    client.session.get.return_value = limited
    fallback = Mock(return_value=limited)
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    client._curl_preferred_until = 500 if transport == "curl" else 0
    with pytest.raises(binance.BinanceApiError) as error:
        client.server_time_ms()
    assert error.value.status == status
    assert error.value.retry_at_ms == 1_790_000_125_000
    if transport == "requests":
        fallback.assert_not_called()
    else:
        client.session.get.assert_not_called()
    restarted = binance.BinanceTerminalClient()
    restarted.session = Mock()
    with pytest.raises(binance.BinanceApiError, match="paused until"):
        restarted.server_time_ms()
    restarted.session.get.assert_not_called()


def test_new_shared_cooldown_between_transport_attempts_prevents_fallback(client, monkeypatch):
    def blocked(*args, **kwargs):
        client.record_cooldown("Too many requests", "120")
        raise requests.exceptions.SSLError("EOF")
    client.session.get.side_effect = blocked
    fallback = Mock()
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    with pytest.raises(binance.BinanceApiError, match="paused until"):
        client.server_time_ms()
    fallback.assert_not_called()


def test_http_errors_do_not_retry_and_total_transport_failure_returns_no_data(client, monkeypatch):
    fallback = Mock(side_effect=requests.exceptions.ConnectionError("curl offline"))
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    client.session.get.return_value = response({"code": -1121, "msg": "Invalid symbol"}, 400)
    with pytest.raises(binance.BinanceApiError) as error:
        client.mark_price()
    assert error.value.code == -1121
    fallback.assert_not_called()
    client.session.get.side_effect = requests.exceptions.SSLError("EOF")
    with pytest.raises(requests.exceptions.ConnectionError, match="Both Binance public transports failed"):
        client.mark_price()
    assert not client.last_public_transport["ok"]


@pytest.mark.parametrize("path, params", [
    ("/fapi/v1/order", {}), ("/fapi/v2/account", {}),
    ("/fapi/v1/time?signature=secret", {}),
    ("https://other.invalid/fapi/v1/time", {}),
    ("/fapi/v1/time", {"signature": "secret"}),
    ("/fapi/v1/time", {"API_KEY": "secret"}),
    ("/fapi/v1/time", {"X-MBX-APIKEY": "secret"}),
])
def test_public_get_rejects_signed_paths_or_credential_queries(client, monkeypatch, path, params):
    fallback = Mock()
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    with pytest.raises(ValueError):
        client.public_get(path, params)
    client.session.get.assert_not_called()
    fallback.assert_not_called()


def install_fake_curl(monkeypatch, payload, status=200, header="", returncode=0):
    calls = []
    monkeypatch.setattr(binance.shutil, "which", lambda name: "/usr/bin/curl")
    def run(command, **kwargs):
        calls.append((command, kwargs))
        header_path = Path(command[command.index("--dump-header") + 1])
        header_path.write_text("HTTP/1.1 200 Connection established\r\n\r\n"
                               f"HTTP/2 {status}\r\n{header}\r\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, json.dumps(payload) + f"\n{status}", "local proxy detail")
    monkeypatch.setattr(binance.subprocess, "run", run)
    return calls


def test_curl_is_public_https_only_and_keeps_retry_after_header(client, monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test-credential-must-not-leak")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret-must-not-leak")
    calls = install_fake_curl(monkeypatch, {"code": -1003, "msg": "limit"}, 429, "retry-after: 120\r\n")
    client.session.get.side_effect = requests.exceptions.SSLError("EOF")
    with pytest.raises(binance.BinanceApiError) as error:
        client.public_get("/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
    assert error.value.retry_at_ms == 1_790_000_125_000
    command, kwargs = calls[0]
    assert command[1] == "-q"
    assert command[command.index("--url") + 1] == "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
    assert "--insecure" not in command and "-k" not in command and "--location" not in command
    assert command[command.index("--proto") + 1] == "=https"
    assert "BINANCE_API_KEY" not in kwargs["env"] and "BINANCE_API_SECRET" not in kwargs["env"]
    assert "test-secret-must-not-leak" not in str(command)


def test_curl_transport_error_does_not_include_tool_output(client, monkeypatch):
    install_fake_curl(monkeypatch, {}, returncode=35)
    with pytest.raises(requests.ConnectionError, match=r"curl 35") as error:
        client._curl_public_get("/fapi/v1/time", None)
    assert "local proxy detail" not in str(error.value)


@pytest.mark.parametrize("origin", ["http://fapi.binance.com", "https://user:secret@fapi.binance.com",
                                    "https://fapi.binance.com/path", "https://fapi.binance.com?secret=1"])
def test_curl_rejects_insecure_or_credential_origins(client, monkeypatch, origin):
    client.base_url = origin
    run = Mock()
    monkeypatch.setattr(binance.subprocess, "run", run)
    with pytest.raises(requests.ConnectionError, match="HTTPS origin"):
        client._curl_public_get("/fapi/v1/time", None)
    run.assert_not_called()


def test_recent_rules_persist_across_client_restarts_without_network(client):
    client.public_get = Mock(return_value=exchange_info())
    expected = client.symbol_rules()
    client.public_get.assert_called_once_with("/fapi/v1/exchangeInfo")
    stored = json.loads(binance.RULES_CACHE_PATH.read_text())
    assert stored["symbols"]["BTCUSDT"]["fetched_at_ms"] == 1_790_000_000_000
    assert expected["step_size"] == Decimal("0.001")
    restarted = binance.BinanceTerminalClient()
    restarted.public_get = Mock(side_effect=AssertionError("TTL cache should avoid network"))
    assert restarted.symbol_rules() == expected
    assert restarted.quantize_quantity(0.0025) == 0.002
    restarted.public_get.assert_not_called()


def test_rules_expire_in_memory_and_on_disk_and_failure_cannot_serve_expired_rules(client, monkeypatch):
    client.public_get = Mock(return_value=exchange_info())
    client.symbol_rules()
    monkeypatch.setattr(binance.time, "time", lambda: 1_790_000_000 + binance.RULES_CACHE_TTL_SECONDS)
    client.public_get.side_effect = requests.exceptions.ConnectionError("offline")
    with pytest.raises(requests.exceptions.ConnectionError):
        client.symbol_rules()
    restarted = binance.BinanceTerminalClient()
    restarted.public_get = Mock(return_value=exchange_info("0.002"))
    assert restarted.symbol_rules()["step_size"] == Decimal("0.002")
    restarted.public_get.assert_called_once()


@pytest.mark.parametrize("mutation", ["wrong_origin", "future", "invalid_rules", "bad_json", "scalar"])
def test_invalid_rule_cache_fetches_fresh_exchange_rules(client, monkeypatch, mutation):
    client.public_get = Mock(return_value=exchange_info())
    client.symbol_rules()
    payload = json.loads(binance.RULES_CACHE_PATH.read_text())
    if mutation == "wrong_origin":
        payload["base_url"] = "https://other.invalid"
    elif mutation == "future":
        payload["symbols"]["BTCUSDT"]["fetched_at_ms"] += 1
    elif mutation == "invalid_rules":
        payload["symbols"]["BTCUSDT"]["rules"]["step_size"] = "NaN"
    elif mutation == "scalar":
        payload = []
    binance.RULES_CACHE_PATH.write_text("{" if mutation == "bad_json" else json.dumps(payload))
    restarted = binance.BinanceTerminalClient()
    restarted.public_get = Mock(return_value=exchange_info("0.003"))
    assert restarted.symbol_rules()["step_size"] == Decimal("0.003")
    restarted.public_get.assert_called_once()


def test_rule_cache_write_failure_does_not_block_fresh_rules(client, monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("occupied")
    monkeypatch.setattr(binance, "RULES_CACHE_PATH", blocker / "rules.json")
    client.public_get = Mock(return_value=exchange_info())
    assert client.symbol_rules()["step_size"] == Decimal("0.001")


def test_signed_order_transport_is_never_replayed_by_public_fallback(client, monkeypatch):
    client.api_key = "dummy-test-key"
    client.api_secret = "dummy-test-secret"
    client.public_get = Mock(return_value={"serverTime": 1_790_000_000_000})
    client.session.request.side_effect = requests.exceptions.SSLError("connection broke after write")
    fallback = Mock()
    monkeypatch.setattr(client, "_curl_public_get", fallback)
    with pytest.raises(requests.exceptions.SSLError):
        client.signed_request("POST", "/fapi/v1/order", {"symbol": "BTCUSDT", "side": "BUY"})
    client.session.request.assert_called_once()
    client.session.get.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("funding_field", [{}, {"nextFundingTime": None},
                                         {"nextFundingTime": "1790003600000"}])
def test_mark_observation_optional_funding_boundary_preserves_legacy_shape(client, funding_field):
    client.public_get = Mock(return_value={
        "symbol": "BTCUSDT", "markPrice": "80000", "time": 1_790_000_000_000,
        **funding_field,
    })
    expected = {"price": 80000.0, "time_ms": 1_790_000_000_000}
    if funding_field.get("nextFundingTime") is not None:
        expected["next_funding_time_ms"] = 1_790_003_600_000
    assert client.mark_price_observation() == expected
    client.public_get.assert_called_once_with("/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
