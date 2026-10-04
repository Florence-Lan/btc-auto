"""A failed transport may reuse complete inputs, never stale decision history."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import binance_terminal_client as binance
import market_data_runtime as market
import simulate_range_swing as sim

FIVE = 300_000
HOUR = 3_600_000


def kline(timestamp, step, price=100):
    return [timestamp, str(price), str(price + 2), str(price - 2), str(price + 1),
            "10", timestamp + step - 1, "1000", 1, "0", "0", "0"]


class PublicHistory:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.missing = set()
        self.transform = None
        self.funding = []

    def public_get(self, path, params):
        self.calls.append((path, dict(params)))
        if self.failure:
            raise self.failure
        if path == "/fapi/v1/fundingRate":
            return [row for row in self.funding
                    if params["startTime"] <= row["fundingTime"] <= params["endTime"]][:params["limit"]]
        assert path == "/fapi/v1/klines"
        step = sim.interval_to_ms(params["interval"])
        start = (params["startTime"] + step - 1) // step * step
        rows = [kline(t, step) for t in range(start, params["endTime"] + 1, step)
                if (params["interval"], t) not in self.missing][:params["limit"]]
        return self.transform(rows, params) if self.transform else rows


def load(client, tmp_path, asof=2 * HOUR + 1000, start=0, intervals=("5m", "1h")):
    return market.load_market_data("BTCUSDT", start, asof, client=client,
                                   cache_dir=tmp_path, intervals=intervals)


def test_bootstrap_has_complete_closed_history_and_known_open_only(tmp_path):
    client = PublicHistory()
    asof = 2 * HOUR + 1000
    result = load(client, tmp_path, asof)
    assert len(result.candles["5m"]) == 24
    assert len(result.candles["1h"]) == 2
    assert all(bar.close_time_ms <= asof for bars in result.candles.values() for bar in bars)
    assert result.opening("1h", 2 * HOUR).price == 100
    assert result.opening("1h", 2 * HOUR).observed_at_ms == asof
    payload = json.loads((tmp_path / "BTCUSDT_1h.json").read_text())
    assert len(payload["candles"]) == 2
    assert payload["openings"] == [[2 * HOUR, 100.0, asof]]
    assert result.diagnostics["complete"]


def test_forming_high_low_close_and_volume_are_never_parsed(tmp_path):
    client = PublicHistory()
    asof = 2 * HOUR + 1000

    def unfinished(rows, params):
        if rows and rows[-1][6] > asof:
            rows[-1] = [rows[-1][0], "123", "bad high", "bad low", "bad close",
                        "bad volume", "bad timestamp", "bad quote volume"]
        return rows

    client.transform = unfinished
    result = load(client, tmp_path, asof)
    assert result.opening("1h", 2 * HOUR).price == 123
    assert result.candles["1h"][-1].close == 101


def test_next_cycle_fetches_tail_instead_of_reloading_warmup(tmp_path):
    client = PublicHistory()
    first = load(client, tmp_path)
    client.calls.clear()
    result = load(client, tmp_path, 2 * HOUR + FIVE + 1000)
    klines = [params for path, params in client.calls if path.endswith("klines")]
    assert len(klines) == 1
    assert klines[0]["interval"] == "5m"
    assert klines[0]["startTime"] == 2 * HOUR
    assert len(result.candles["5m"]) == len(first.candles["5m"]) + 1
    funding_calls = [params for path, params in client.calls if path.endswith("fundingRate")]
    assert funding_calls[0]["startTime"] == 0  # One-day publication-delay overlap.
    assert result.opening("1h", 2 * HOUR).price == 100


def test_large_bootstrap_paginates_and_followup_only_fetches_tail(tmp_path):
    client = PublicHistory()
    first_asof = 45 * sim.MS_PER_DAY + 1000
    result = load(client, tmp_path, first_asof)
    assert len(result.candles["5m"]) == 45 * 288
    assert len([1 for path, params in client.calls
                if path.endswith("klines") and params["interval"] == "5m"]) == 9
    client.calls.clear()
    load(client, tmp_path, first_asof + FIVE)
    assert all(params["startTime"] >= 45 * sim.MS_PER_DAY
               for path, params in client.calls if path.endswith("klines"))


def test_same_decision_bar_uses_complete_cache_during_transport_failure(tmp_path):
    client = PublicHistory()
    initial = load(client, tmp_path)
    client.failure = ConnectionError("offline")
    fallback = load(client, tmp_path, 2 * HOUR + 2000)
    assert fallback.candles == initial.candles
    assert fallback.funding == initial.funding
    assert fallback.diagnostics["degraded"]
    assert fallback.diagnostics["funding"]["source"] == "cache_fallback"
    assert fallback.opening("1h", 2 * HOUR).observed_at_ms == 2 * HOUR + 1000


def test_new_decision_bar_rejects_stale_cache_when_transport_fails(tmp_path):
    client = PublicHistory()
    load(client, tmp_path)
    client.failure = ConnectionError("offline")
    with pytest.raises(market.MarketDataUnavailable, match="incomplete or stale") as error:
        load(client, tmp_path, 2 * HOUR + FIVE + 1000)
    details = error.value.diagnostics["intervals"]["5m"]
    assert details["required_last_close_ms"] == 2 * HOUR + FIVE - 1
    assert details["cached_last_close_ms"] == 2 * HOUR - 1
    assert not details["complete"]


def test_gap_in_warmup_is_rejected_and_then_repaired_incrementally(tmp_path):
    client = PublicHistory()
    client.missing.add(("5m", 3 * FIVE))
    with pytest.raises(market.MarketDataUnavailable) as error:
        load(client, tmp_path)
    assert error.value.diagnostics["intervals"]["5m"]["missing_spans"] == [(3 * FIVE, 3 * FIVE)]
    client.missing.clear()
    client.calls.clear()
    result = load(client, tmp_path)
    assert result.candles["5m"][3].open_time_ms == 3 * FIVE
    repair = next(params for path, params in client.calls if path.endswith("klines"))
    assert repair["startTime"] == 3 * FIVE
    assert repair["endTime"] == 4 * FIVE - 1


@pytest.mark.parametrize("column,bad", [(1, "nan"), (2, "90"), (3, "110"),
                                       (5, "-1"), (6, FIVE), (7, "inf")])
def test_invalid_closed_ohlcv_is_not_used(tmp_path, column, bad):
    client = PublicHistory()

    def corrupt(rows, params):
        if rows and params["interval"] == "5m":
            rows[0][column] = bad
        return rows

    client.transform = corrupt
    with pytest.raises(market.MarketDataUnavailable):
        load(client, tmp_path)


def test_future_response_candle_is_rejected(tmp_path):
    client = PublicHistory()
    client.transform = lambda rows, params: rows + [kline(3 * HOUR, sim.interval_to_ms(params["interval"]))]
    with pytest.raises(market.MarketDataUnavailable) as error:
        load(client, tmp_path)
    # The valid prior rows have no missing bar, but future rows must not become
    # inputs even when the retrieval itself is marked degraded.
    assert "funding" in error.value.diagnostics or "intervals" in error.value.diagnostics


def test_earlier_asof_never_uses_future_funding_or_opening_observation(tmp_path):
    client = PublicHistory()
    client.funding = [{"symbol": "BTCUSDT", "fundingTime": HOUR, "fundingRate": "0.001"},
                      {"symbol": "BTCUSDT", "fundingTime": 2 * HOUR, "fundingRate": "-0.001"}]
    load(client, tmp_path)
    result = load(client, tmp_path, 2 * HOUR - 1)
    assert result.funding.times == [HOUR]
    assert result.opening("1h", 2 * HOUR) is None
    assert all(t < 2 * HOUR for ticks in result.openings.values() for t in ticks)


def test_insufficient_funding_watermark_cannot_claim_fresh_input(tmp_path):
    client = PublicHistory()
    load(client, tmp_path)
    # Closed bars succeed, but funding fails after the prior covered decision.
    original = client.public_get

    def funding_offline(path, params):
        if path.endswith("fundingRate"):
            raise ConnectionError("funding unavailable")
        return original(path, params)

    client.public_get = funding_offline
    with pytest.raises(market.MarketDataUnavailable, match="Funding history") as error:
        load(client, tmp_path, 2 * HOUR + FIVE + 1000)
    assert not error.value.diagnostics["funding"]["complete"]


def test_late_boundary_funding_is_recovered_after_empty_complete_fetch(tmp_path):
    client = PublicHistory()
    initial = load(client, tmp_path)
    assert initial.funding.times == []
    assert initial.diagnostics["funding"]["coverage_end_ms"] == 2 * HOUR + 1000
    # The exchange publishes the 02:00 event after the first query finished.
    client.funding = [{"symbol": "BTCUSDT", "fundingTime": 2 * HOUR, "fundingRate": "0.001"}]
    asof = 2 * HOUR + FIVE + 1000
    result = load(client, tmp_path, asof)
    assert result.funding.times == [2 * HOUR]
    assert result.funding.rates == [0.001]
    assert result.diagnostics["funding"]["coverage_end_ms"] == asof
    client.calls.clear()
    client.failure = ConnectionError("fully offline")
    offline = load(client, tmp_path, asof)
    assert offline.funding == result.funding
    assert client.calls == []


def test_funding_overlap_covers_a_day_and_overwrites_rates_without_duplicates(tmp_path):
    client = PublicHistory()
    initial_asof = 2 * sim.MS_PER_DAY + 1000
    prior_event = sim.MS_PER_DAY + 2 * HOUR
    client.funding = [{"symbol": "BTCUSDT", "fundingTime": prior_event, "fundingRate": "0.001"}]
    load(client, tmp_path, initial_asof)
    client.calls.clear()
    client.funding = [{"symbol": "BTCUSDT", "fundingTime": prior_event, "fundingRate": "0.002"},
                      {"symbol": "BTCUSDT", "fundingTime": 2 * sim.MS_PER_DAY, "fundingRate": "-0.001"}]
    result = load(client, tmp_path, initial_asof + FIVE)
    call = next(params for path, params in client.calls if path.endswith("fundingRate"))
    assert call["startTime"] == initial_asof - sim.MS_PER_DAY
    assert result.funding.times == [prior_event, 2 * sim.MS_PER_DAY]
    assert result.funding.rates == [0.002, -0.001]


def test_missing_earlier_warmup_is_not_fabricated(tmp_path):
    client = PublicHistory()
    load(client, tmp_path, start=HOUR)
    client.failure = ConnectionError("offline")
    with pytest.raises(market.MarketDataUnavailable) as error:
        load(client, tmp_path, start=0)
    assert error.value.diagnostics["intervals"]["5m"]["missing_spans"] == [(0, HOUR - FIVE)]


def test_shared_cooldown_prevents_any_network_attempt(tmp_path, monkeypatch):
    client = PublicHistory()
    load(client, tmp_path)
    monkeypatch.setattr(binance, "COOLDOWN_PATH", tmp_path / "cooldown.json")
    monkeypatch.setattr(binance.time, "time", lambda: (2 * HOUR + 2000) / 1000)
    real = binance.BinanceTerminalClient()
    real.session = Mock()
    (tmp_path / "cooldown.json").write_text(json.dumps({
        "base_url": real.base_url, "retry_at_ms": 3 * HOUR,
    }))
    result = load(real, tmp_path, 2 * HOUR + 2000)
    assert result.diagnostics["degraded"]
    real.session.get.assert_not_called()


def test_last_millisecond_of_interval_counts_as_closed(tmp_path):
    result = load(PublicHistory(), tmp_path, 2 * HOUR - 1)
    assert len(result.candles["1h"]) == 2
    assert result.candles["1h"][-1].close_time_ms == 2 * HOUR - 1


def test_corrupt_cache_is_rebuilt_without_trusting_bad_inputs(tmp_path):
    client = PublicHistory()
    load(client, tmp_path)
    (tmp_path / "BTCUSDT_5m.json").write_text("{broken")
    result = load(client, tmp_path)
    assert result.diagnostics["intervals"]["5m"]["cache_error"]
    assert len(result.candles["5m"]) == 24


def test_live_and_testnet_histories_cannot_share_cache(tmp_path):
    live = PublicHistory()
    live.environment = "live"
    load(live, tmp_path)
    testnet = PublicHistory()
    testnet.environment = "testnet"
    testnet.failure = ConnectionError("offline")
    with pytest.raises(market.MarketDataUnavailable):
        load(testnet, tmp_path)
    testnet.failure = None
    testnet.transform = lambda rows, params: [kline(row[0], sim.interval_to_ms(params["interval"]), 200)
                                              for row in rows]
    result = load(testnet, tmp_path)
    assert result.candles["5m"][0].open == 200
    assert (tmp_path / "testnet_BTCUSDT_5m.json").exists()
    assert load(live, tmp_path).candles["5m"][0].open == 100
