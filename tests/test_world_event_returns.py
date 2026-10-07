import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_world_event_returns as validation
import world_event_risk as world

T = world.utc_ms("2026-09-01T00:00:00Z")


def trade(at=T):
    return {"entry_time_utc": world.iso(at), "initial_qty": 2.0, "entry_price": 100.0,
            "pnl": 12.0, "net_pnl": 10.0, "fees": 2.0}


def test_historical_missing_feed_is_not_comparable_even_if_later_healthy():
    snapshot = world.empty_snapshot()
    snapshot["polls"] = [{"available_at_utc": world.iso(T + 2 * world.HOUR_MS), "status": "ok"}]
    result = validation.coverage(snapshot, [{"trades": [trade()]}], T, T + world.HOUR_MS)
    assert result["healthy_time_pct"] == 0
    assert result["healthy_entries"] == 0
    assert not result["can_compare"]


def test_outage_does_not_disappear_after_recovery():
    snapshot = world.empty_snapshot()
    snapshot["polls"] = [{"available_at_utc": world.iso(T), "status": "ok"},
                         {"available_at_utc": world.iso(T + world.HOUR_MS), "status": "error"},
                         {"available_at_utc": world.iso(T + 2 * world.HOUR_MS), "status": "ok"}]
    result = validation.coverage(snapshot, [{"trades": [trade()]}], T, T + 3 * world.HOUR_MS)
    assert result["healthy_time_pct"] == pytest.approx(200 / 3)
    assert result["healthy_entry_pct"] == 100
    assert not result["can_compare"]


def test_healthy_empty_feed_can_be_compared_but_has_no_event_evidence():
    snapshot = world.empty_snapshot()
    snapshot["polls"] = [{"available_at_utc": world.iso(T), "status": "ok"}]
    result = validation.coverage(snapshot, [{"trades": [trade()]}], T, T + world.HOUR_MS)
    assert result["can_compare"]
    assert result["confirmed_event_entries"] == 0
    assert result["changed_entries"] == 0


def test_control_matches_total_entry_amount_and_preserves_original_inputs():
    original = [{"trades": [trade(), {**trade(), "initial_qty": 3.0}]}]
    saved = copy.deepcopy(original)
    reduced = validation.scaled(original, 0.6)
    assert validation.total_entry_notional(reduced) == pytest.approx(300)
    assert reduced[0]["trades"][0]["net_pnl"] == pytest.approx(6)
    assert reduced[0]["trades"][0]["fees"] == pytest.approx(1.2)
    assert original == saved


def curve(days=21):
    return {"summary": {"initial_equity": 100.0, "total_return_pct": 5.0},
            "equity_curve": [{"time_ms": T + i * 86400000, "equity": 100 + i * .25} for i in range(days)]}


def test_paired_identical_curves_have_zero_increment():
    result = validation.paired_bootstrap(curve(), curve(), samples=100)
    assert result["return_delta_pp"] == 0
    assert result["ci95_return_delta_pp"] == [0, 0]


def test_paired_bootstrap_rejects_different_windows_and_short_samples():
    with pytest.raises(ValueError, match="identical"):
        validation.paired_bootstrap(curve(), curve(20))
    assert validation.paired_bootstrap(curve(3), curve(3))["ci95_return_delta_pp"] is None


def test_daily_returns_include_the_first_day_and_reconcile_terminal_equity():
    result = curve()
    result["equity_curve"][0]["equity"] = 99.0
    values = validation.daily_returns(result)
    assert next(iter(values.values())) == pytest.approx(-0.01)
    growth = 1.0
    for value in values.values():
        growth *= 1 + value
    assert growth * 100 == pytest.approx(result["equity_curve"][-1]["equity"])
