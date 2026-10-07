import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import market_intelligence as m
import monitor_market_intelligence as monitor

T = 1_790_000_000_000 // m.FIVE_MIN * m.FIVE_MIN


def bars(buy=60):
    return [[T + i * m.FIVE_MIN, "100", "101", "99", "100", "1",
             T + (i + 1) * m.FIVE_MIN - 1, "100", 1, ".6", str(buy), "0"] for i in range(13)]


def record(data, timestamp):
    return {"status": "ok", "received_ms": timestamp, "url": "https://example.com", "data": data}


def test_forming_candle_cannot_change_closed_flow():
    rows = bars()
    rows[-1][10] = "9999999"
    result = m.closed_flow(rows, T + 12 * m.FIVE_MIN)
    assert result["bars"] == 12
    assert result["buy_quote"] == 720
    assert result["imbalance_1h"] == pytest.approx(.2)


@pytest.mark.parametrize("kind", ["gap", "stale", "invalid_volume"])
def test_bad_flow_is_rejected(kind):
    rows = bars()
    now = T + 13 * m.FIVE_MIN
    if kind == "gap":
        del rows[5]
    elif kind == "stale":
        now += 3 * m.FIVE_MIN
    else:
        rows[-1][10] = "101"
    with pytest.raises(ValueError):
        m.closed_flow(rows, now)


def test_oi_uses_btc_quantity_instead_of_usd_price_effect():
    rows = [{"timestamp": T + i * m.FIVE_MIN, "sumOpenInterest": "10",
             "sumOpenInterestValue": str(1000 + i * 100)} for i in range(13)]
    assert m.oi_features(rows, T + 14 * m.FIVE_MIN)["change_1h_pct"] == 0


def test_buyer_maker_means_aggressive_selling_and_reposts_are_deduplicated():
    trade = {"a": 1, "p": "50000", "q": "6", "T": T, "m": True}
    report = m.tape_features([trade, trade], T + 1, "futures")
    assert report["trades"] == 1
    assert report["large_sell_notional"] == 300000
    assert report["large_buy_notional"] == 0
    assert report["sample_only"]


def test_hyperliquid_buy_side_and_compound_trade_identity():
    trade = {"tid": 1, "time": T, "px": "50000", "sz": "6", "side": "B"}
    result = m.tape_features([trade, {**trade, "time": T + 1}], T + 2, "hyperliquid")
    assert result["trades"] == 2
    assert result["large_buy_notional"] == 600000


def test_tape_rejects_stale_future_and_invalid_maker_flag():
    trade = {"a": 1, "p": "50000", "q": "6", "T": T, "m": True}
    for row, now in [(trade, T + m.FIVE_MIN + 1), (trade, T - 6000), ({**trade, "m": "false"}, T)]:
        with pytest.raises(ValueError):
            m.tape_features([row], now, "spot")


def test_order_book_reports_truncated_band_instead_of_full_liquidity():
    result = m.book_features({"bids": [["99.99", "2"]], "asks": [["100.01", "2"]]}, T)
    assert not result["full_10bps_band"]
    assert result["spread_bps"] == pytest.approx(2)


def test_chain_transfers_never_become_buy_sell_signals():
    tx = {"txid": "example", "value": 100 * 100_000_000}
    result = m.chain_features([tx, tx])
    assert result["sample_size"] == 1
    assert result["large_output_transactions"][0]["output_btc"] == 100
    assert result["direction"] is None


def test_hyperliquid_asset_mapping_uses_name_not_fixed_position():
    data = [{"universe": [{"name": "ETH"}, {"name": "BTC"}]}, [{},
            {"markPx": "50000", "openInterest": "20", "funding": ".00001", "dayNtlVlm": "100000"}]]
    assert m.hyper_context(data)["oi_usd"] == 1000000


def test_signed_onchain_position_sample_and_partial_failure():
    records = [{"status": "ok", "address": "public-a", "data": {"assetPositions": [
        {"position": {"coin": "BTC", "szi": "-2", "positionValue": "100000", "liquidationPx": None}}]}},
        {"status": "error", "address": "public-b"}]
    result = m.position_sample(records)
    assert result["short_usd"] == 100000 and result["long_usd"] == 0
    assert result["successful_addresses"] == 1
    assert result["partial"] and result["sample_only"]
    with pytest.raises(ValueError):
        m.position_sample([records[1]])


def test_missing_core_data_does_not_produce_neutral_score():
    report = m.build_report({"spot_flow": record(bars(), T + 13 * m.FIVE_MIN)}, T + 13 * m.FIVE_MIN)
    assert report["direction"]["score"] is None
    assert report["direction"]["state"] == "insufficient_data"
    assert not report["places_orders"]


def test_spot_futures_disagreement_is_explicit():
    now = T + 14 * m.FIVE_MIN
    oi = [{"timestamp": T + i * m.FIVE_MIN, "sumOpenInterest": "10", "sumOpenInterestValue": "1000"} for i in range(13)]
    report = m.build_report({"spot_flow": record(bars(70), now), "futures_flow": record(bars(30), now),
                             "open_interest": record(oi, now)}, now)
    assert report["direction"]["state"] == "conflicting"
    assert "spot_futures_flow_disagree" in report["risk_flags"]


def test_tape_gap_between_polls_is_exposed():
    trade = {"a": 20, "p": "50000", "q": "6", "T": T, "m": False}
    previous = {"features": {"spot_tape": {"max_id": 10}}}
    report = m.build_report({"spot_tape": record([trade], T)}, T, previous)
    assert report["features"]["spot_tape"]["gap_since_previous_poll"]


def test_storage_dedup_and_point_in_time_lookup(tmp_path):
    db = m.open_db(tmp_path / "test.sqlite3")
    try:
        records = {"spot_flow": record(bars(), T)}
        report = {"generated_at_ms": T, "features": {}, "places_orders": False}
        m.persist(db, records, report)
        m.persist(db, records, {**report, "generated_at_ms": T + 1000})
        assert db.execute("SELECT COUNT(*) FROM payloads").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 2
        assert m.at_time(db, T - 1) is None
        assert m.at_time(db, T + 500)["generated_at_ms"] == T
        assert m.at_time(db, T + 1001 + m.FIVE_MIN) is None
    finally:
        db.close()


def test_network_error_does_not_return_old_or_zero_data(monkeypatch):
    get = Mock(side_effect=requests.Timeout())
    monkeypatch.setattr(monitor.requests, "get", get)
    name, result = monitor.fetch(("source", "GET", "https://example.com", {}))
    assert result["status"] == "error"
    assert result["data"] is None
    assert "headers" not in get.call_args.kwargs


def test_single_writer_lock(tmp_path):
    path = tmp_path / "writer.lock"
    with monitor.SingleWriter(path):
        with pytest.raises(RuntimeError, match="already running"):
            with monitor.SingleWriter(path):
                pass
    with monitor.SingleWriter(path):
        pass


def test_entry_filter_requires_two_adverse_flows_and_respects_side():
    report = {"rule_version": m.VERSION, "direction": {"score": -60}, "features": {
        "spot_flow": {"imbalance_1h": -.1}, "futures_flow": {"imbalance_1h": -.1}}}
    assert m.entry_multiplier(report, "long") == .5
    assert m.entry_multiplier(report, "short") == 1
    report["features"]["spot_flow"]["imbalance_1h"] = .1
    assert m.entry_multiplier(report, "long") == 1
    assert m.entry_multiplier(None, "long") is None


def test_shadow_filter_cannot_use_future_observations_or_double_discount(tmp_path):
    path = tmp_path / "research.sqlite3"
    db = m.open_db(path)
    observation = {"generated_at_ms": T, "rule_version": m.VERSION, "direction": {"score": -60}, "features": {
        "spot_flow": {"imbalance_1h": -.1}, "futures_flow": {"imbalance_1h": -.1}}}
    m.persist(db, {}, observation)
    db.close()
    raw = {"entry_time_utc": m.iso(T), "side": "long", "initial_qty": .6,
           "net_pnl": 6, "fees": .6, "macro_risk_multiplier": .6}
    earlier = {**raw, "entry_time_utc": m.iso(T - 1)}
    sleeves = [{"trades": [earlier, raw]}]
    saved = copy.deepcopy(sleeves)
    result, metrics = m.apply_shadow_overlay(sleeves, path)
    assert result[0]["trades"][0]["initial_qty"] == .6
    assert result[0]["trades"][0]["intelligence_multiplier"] is None
    assert result[0]["trades"][1]["initial_qty"] == pytest.approx(.5)
    assert metrics["covered"] == 1 and metrics["throttled"] == 1
    assert sleeves == saved
