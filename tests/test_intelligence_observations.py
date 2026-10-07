import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import market_intelligence as m
import validate_intelligence_observations as validation

T = 1790000000000


def reports(minutes=190):
    return [{"generated_at_ms": T + i * 60000, "generated_at_utc": m.iso(T + i * 60000),
             "rule_version": m.VERSION,
             "health": {"funding_basis": {"status": "ok"}},
             "direction": {"score": 50, "state": "buy_pressure"},
             "features": {"funding_basis": {"mark_price": 100 + i, "observed_ms": T + i * 60000},
                          "spot_flow": {"imbalance_1h": .1}, "futures_flow": {"imbalance_1h": .1}}}
            for i in range(minutes)]


def test_forward_checks_use_next_observation_and_nonoverlapping_windows():
    result = validation.forward_samples(reports())
    assert result["samples"] == 3
    assert result["direction_hit_pct"] == 100
    rows = result["details"]
    assert rows[0]["entry_mark"] == 101
    assert rows[0]["exit_mark"] == 161
    assert rows[0]["signed_return_bps"] == pytest.approx((161 / 101 - 1) * 10000)
    assert all(a["exit_utc"] < b["entry_utc"] for a, b in zip(rows, rows[1:]))


def test_short_direction_uses_signed_price_change():
    data = reports()
    for row in data:
        row["direction"] = {"score": -50, "state": "sell_pressure"}
    result = validation.forward_samples(data)
    assert result["direction_hit_pct"] == 0
    assert result["mean_signed_return_bps"] < 0


def test_no_future_exit_or_stale_quotes_are_invented():
    assert validation.forward_samples(reports(30))["samples"] == 0
    data = reports()
    for row in data:
        row["features"]["funding_basis"]["observed_ms"] -= 180000
    assert validation.forward_samples(data)["samples"] == 0


def test_conflicting_direction_does_not_become_a_trade():
    data = reports()
    for row in data:
        row["direction"]["state"] = "conflicting"
    assert validation.forward_samples(data)["samples"] == 0


def test_archived_macro_never_uses_later_score():
    rows = reports(2)
    rows[1]["macro"] = {"score": 1, "allowed": True, "asof_ms": T}
    raw = {"entry_time_utc": m.iso(T), "initial_qty": 1, "entry_price": 100, "net_pnl": 2}
    adjusted, missing = validation.archived_macro([{"trades": [raw]}], rows)
    assert missing == 1 and adjusted[0]["trades"] == []
    raw["entry_time_utc"] = m.iso(T + 60000)
    adjusted, missing = validation.archived_macro([{"trades": [raw]}], rows)
    assert missing == 0 and adjusted[0]["trades"][0]["initial_qty"] == 1


def test_health_audit_exposes_gaps_and_failed_sources():
    rows = reports(2)
    rows[1]["generated_at_ms"] = T + 300000
    rows[1]["health"]["funding_basis"]["status"] = "error"
    result = validation.audit(rows)
    assert result["source_health_pct"]["funding_basis"] == 50
    assert result["gaps_over_180_seconds"] == 1
