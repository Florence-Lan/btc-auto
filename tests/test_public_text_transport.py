import json
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_text_transport as pt
from public_request_deadline import PublicRequestDeadline
from public_transport import PublicCurlError, PublicTransportError

URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"


def response(content=b"<rss>test</rss>", status=200, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.iter_content.return_value = iter([content])
    return result


def test_tls_failure_recovers_using_same_verified_public_url_and_remembers_working_stack(monkeypatch):
    client = pt.PublicTextTransport()
    primary = Mock(side_effect=requests.exceptions.SSLError("proxy-password"))
    fallback = Mock(return_value="<rss>valid</rss>")
    monkeypatch.setattr(client, "_requests_get", primary)
    monkeypatch.setattr(client, "_curl_get", fallback)
    assert client.get(URL) == "<rss>valid</rss>"
    assert client.get(URL) == "<rss>valid</rss>"
    primary.assert_called_once_with(URL)
    assert fallback.call_args_list == [((URL,),), ((URL,),)]


@pytest.mark.parametrize("status", [403, 404, 429])
def test_provider_denial_does_not_trigger_second_transport(monkeypatch, status):
    client = pt.PublicTextTransport()
    primary = response(status=status)
    monkeypatch.setattr(pt.requests, "get", Mock(return_value=primary))
    fallback = Mock()
    monkeypatch.setattr(client, "_curl_get", fallback)
    with pytest.raises(pt.PublicTextHttpError) as error:
        client.get(URL)
    assert error.value.status == status
    fallback.assert_not_called()
    primary.close.assert_called_once()


def test_hung_primary_is_bounded_and_its_late_body_is_discarded(monkeypatch):
    client = pt.PublicTextTransport()
    client.deadline = PublicRequestDeadline(timeout_seconds=.02)
    release, finished = threading.Event(), threading.Event()
    def stalled(url):
        release.wait(2)
        finished.set()
        return "late news"
    primary = Mock(side_effect=stalled)
    monkeypatch.setattr(client, "_requests_get", primary)
    monkeypatch.setattr(client, "_curl_get", Mock(return_value="fresh news"))
    try:
        assert client.get(URL) == "fresh news"
        client._curl_preferred_until.clear()
        assert client.get(URL) == "fresh news"
        assert primary.call_count == 1
    finally:
        release.set()
        assert finished.wait(1)


def test_errors_never_expose_proxy_or_response_secrets(monkeypatch):
    client = pt.PublicTextTransport()
    monkeypatch.setattr(client, "_requests_get", Mock(side_effect=requests.exceptions.ProxyError(
        "https://user:private@proxy.invalid/?api_key=private")))
    monkeypatch.setattr(client, "_curl_get", Mock(side_effect=PublicCurlError(
        "curl failed", curl_exit_code=60)))
    with pytest.raises(PublicTransportError) as error:
        client.get(URL)
    rendered = "".join(traceback.format_exception(error.value)) + json.dumps(error.value.failures)
    assert "private" not in rendered and "proxy.invalid" not in rendered
    assert error.value.failures[-1]["category"] == "tls_verification"


def test_https_redirect_is_validated_before_following_and_responses_are_closed(monkeypatch):
    client = pt.PublicTextTransport()
    first = response(status=302, headers={"Location": "http://insecure.invalid/feed"})
    fetch = Mock(return_value=first)
    monkeypatch.setattr(pt.requests, "get", fetch)
    with pytest.raises(ValueError, match="HTTPS"):
        client.get(URL)
    fetch.assert_called_once()
    first.close.assert_called_once()


def test_curl_preserves_tls_proxy_and_ca_choices_without_user_config_or_secrets(monkeypatch, tmp_path):
    client = pt.PublicTextTransport()
    monkeypatch.setattr(pt.shutil, "which", lambda _: "curl.exe")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "trusted.pem"))
    monkeypatch.setenv("HTTPS_PROXY", "https://user:private@proxy.invalid")
    monkeypatch.setenv("BINANCE_API_SECRET", "must-not-reach-curl")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, b"<rss>valid</rss>\n200", b""))
    monkeypatch.setattr(pt.subprocess, "run", run)
    assert client._curl_get(URL) == "<rss>valid</rss>"
    command = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert command[1] == "-q"
    assert command[command.index("--cacert") + 1] == str(tmp_path / "trusted.pem")
    assert command[command.index("--proto") + 1] == "=https"
    assert command[command.index("--proto-redir") + 1] == "=https"
    assert "--insecure" not in command
    assert kwargs["env"]["HTTPS_PROXY"] == "https://user:private@proxy.invalid"
    assert "BINANCE_API_SECRET" not in kwargs["env"]


def test_truncated_or_empty_primary_body_can_recover_but_never_return_empty_news(monkeypatch):
    client = pt.PublicTextTransport()
    empty = response(b"")
    monkeypatch.setattr(pt.requests, "get", Mock(return_value=empty))
    monkeypatch.setattr(client, "_curl_get", Mock(return_value="<rss>valid</rss>"))
    assert client.get(URL) == "<rss>valid</rss>"
    empty.close.assert_called_once()
