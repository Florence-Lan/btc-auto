import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import binance_terminal_client as binance


@pytest.fixture
def cooldown(monkeypatch, tmp_path):
    monkeypatch.setattr(binance, "COOLDOWN_PATH", tmp_path / "cooldown.json")
    monkeypatch.setattr(binance.time, "time", lambda: 1_790_868_600.0)
    return binance.BinanceTerminalClient()


def test_http_ban_persists_and_other_client_skips_network(cooldown):
    response = Mock(ok=False, status_code=418, headers={})
    response.json.return_value = {"code": -1003, "msg": "IP banned until 1790889479798."}
    with pytest.raises(binance.BinanceApiError) as error:
        cooldown._decode_response(response)
    assert error.value.retry_at_ms == 1_790_889_484_798
    restarted = binance.BinanceTerminalClient()
    restarted.session = Mock()
    with pytest.raises(binance.BinanceApiError, match="paused until"):
        restarted.mark_price()
    restarted.session.get.assert_not_called()


def test_retry_after_respected_and_expired_cooldown_resumes(cooldown, monkeypatch):
    response = Mock(ok=False, status_code=429, headers={"Retry-After": "120"})
    response.json.return_value = {"code": -1003, "msg": "Too many requests"}
    with pytest.raises(binance.BinanceApiError) as error:
        cooldown._decode_response(response)
    assert error.value.retry_at_ms == 1_790_868_725_000
    monkeypatch.setattr(binance.time, "time", lambda: 1_790_868_726.0)
    cooldown.session = Mock()
    ok = Mock(ok=True)
    ok.json.return_value = {"serverTime": 1_790_868_726_000}
    cooldown.session.get.return_value = ok
    assert cooldown.server_time_ms() == 1_790_868_726_000
    cooldown.session.get.assert_called_once()


def test_shorter_retry_cannot_reduce_existing_pause(cooldown):
    until = cooldown.record_cooldown("banned until 1790889479798")
    assert cooldown.record_cooldown("Too many requests", "60") == until
