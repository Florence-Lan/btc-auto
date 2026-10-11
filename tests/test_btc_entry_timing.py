"""The hourly entry window cannot be refreshed by five-minute reports or reviews."""
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import btc_entry_timing
import trading_execution as execution


HOUR = 3_600_000
ORIGIN = 1_790_000_000_000
MISSING = object()


@pytest.fixture(autouse=True)
def execution_environment(monkeypatch):
    for key, value in {
        "LLM_TRADE_GATE_ENABLED": "false", "SIM_MAX_LEVERAGE": "2",
        "SIM_MAX_NOTIONAL_USDT": "0", "SIM_TAKER_FEE": "0.00045",
        "SIM_SLIPPAGE_BPS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    # Ordinary checks have no elapsed time; crossing tests advance it explicitly.
    monkeypatch.setattr(execution.time, "monotonic", lambda: 10.0)


def report():
    return {"shadow_profile": {"entry_timing_policy": {
        "mode": "origin_signal_window_v1", "max_initial_entry_age_seconds": 3600,
    }}}


def target(leverage=0.5, origin=ORIGIN, now=ORIGIN + 1_000):
    value = {"target_leverage": leverage, "signal_time_ms": now,
             "signal_price": 100_000, "position_id": "hourly-origin"}
    if origin is not MISSING:
        value["origin_signal_time_ms"] = origin
    return value


def account(tmp_path):
    value = execution.SimulationAccount(tmp_path / "account.json")
    state = value.reset(10_000, now_ms=ORIGIN - 1_000)
    state["observed_flat_target"] = True
    value._save(state)
    return value


def test_fresh_five_minute_report_cannot_reopen_expired_hourly_origin(tmp_path, monkeypatch):
    now = ORIGIN + 60 * HOUR
    value = account(tmp_path)
    selected = report()
    selected["execution_target"] = {
        "time_ms": now - 300_000, "available_time_ms": now - 1,
        "equity": 100, "price": 100_000, "signed_qty": -.0005,
        "position_id": "hourly-origin", "origin_signal_time_ms": ORIGIN,
    }
    requested = execution.target_from_report(selected, now_ms=now)
    assert requested["age_seconds"] == 300  # Report freshness itself passes.
    monkeypatch.setenv("LLM_TRADE_GATE_ENABLED", "true")
    with patch("llm_trade_gate.request_llm_decision") as provider:
        result = value.reconcile(requested, 100_000, report=selected, now_ms=now)
    provider.assert_not_called()
    assert result["fill"] is None
    assert value.load()["position_qty"] == 0
    assert result["btc_entry_timing"]["status"] == "expired"
    assert value.load()["btc_entry_timing"]["origin_signal_age_seconds"] == 216_000


@pytest.mark.parametrize("age,allowed", [(0, True), (HOUR - 1, True), (HOUR, False), (HOUR + 1, False)])
def test_window_includes_origin_and_excludes_exact_expiry(tmp_path, age, allowed):
    value = account(tmp_path)
    result = value.reconcile(target(now=ORIGIN + age), 100_000,
                             report=report(), now_ms=ORIGIN + age)
    assert bool(result["fill"]) is allowed
    assert result["btc_entry_timing"]["allowed"] is allowed
    assert result["btc_entry_timing"]["status"] == ("within_window" if allowed else "expired")


def test_llm_review_crossing_window_cannot_fill_even_with_approval(tmp_path, monkeypatch):
    value = account(tmp_path)
    now = ORIGIN + HOUR - 1_000
    elapsed = {"seconds": 0.0}
    monkeypatch.setattr(execution.time, "monotonic", lambda: 10.0 + elapsed["seconds"])

    def approve(selected, requested, **kwargs):
        assert requested["target_leverage"] == .5
        elapsed["seconds"] = 1.0
        return dict(requested), {"enabled": True, "status": "cached_approved", "allow": True}

    with patch.object(execution, "apply_llm_trade_gate", side_effect=approve):
        result = value.reconcile(target(now=now), 100_000, report=report(), now_ms=now)
    assert result["llm_trade_gate"]["status"] == "cached_approved"
    assert result["btc_entry_timing"]["checked_at_ms"] == ORIGIN + HOUR
    assert result["btc_entry_timing"]["status"] == "expired"
    assert result["target_qty"] == 0
    assert result["fill"] is None


@pytest.mark.parametrize("operation", ["hold", "increase", "reduce", "close"])
def test_old_or_missing_origin_does_not_block_existing_direction_management(tmp_path, operation):
    value = account(tmp_path)
    value.reconcile(target(), 100_000, report=report(), now_ms=ORIGIN + 1_000)
    quantity = value.load()["position_qty"]
    equity = value.snapshot(100_000)["account"]["margin_balance"]
    exposure = quantity * 100_000 / equity
    requested = {"hold": exposure, "increase": exposure * 1.5,
                 "reduce": exposure / 2, "close": 0}[operation]
    result = value.reconcile(target(requested, origin=MISSING, now=ORIGIN + 60 * HOUR),
                             100_000, report=report(), now_ms=ORIGIN + 60 * HOUR)
    assert result["btc_entry_timing"]["allowed"] is True
    assert result["btc_entry_timing"]["requires_new_direction"] is False
    after = value.load()["position_qty"]
    if operation == "hold":
        assert after == quantity
        assert result["fill"] is None
    elif operation == "increase":
        assert after > quantity
    elif operation == "reduce":
        assert 0 < after < quantity
    else:
        assert after == 0


@pytest.mark.parametrize("origin,status", [(ORIGIN, "expired"), (MISSING, "origin_missing")])
def test_blocked_reversal_closes_old_direction_without_opening_new(tmp_path, origin, status):
    value = account(tmp_path)
    value.reconcile(target(), 100_000, report=report(), now_ms=ORIGIN + 1_000)
    held = value.load()["position_qty"]
    now = ORIGIN + HOUR
    result = value.reconcile(target(-.5, origin=origin, now=now), 100_000,
                             report=report(), now_ms=now)
    assert result["btc_entry_timing"]["status"] == status
    assert result["fill"]["side"] == "SELL"
    assert result["fill"]["quantity"] == held
    assert value.load()["position_qty"] == 0


def test_fresh_reversal_still_opens_new_direction(tmp_path):
    value = account(tmp_path)
    value.reconcile(target(), 100_000, report=report(), now_ms=ORIGIN + 1_000)
    now = ORIGIN + 2 * HOUR
    result = value.reconcile(target(-.5, origin=now - 1_000, now=now), 100_000,
                             report=report(), now_ms=now)
    assert result["btc_entry_timing"]["status"] == "within_window"
    assert value.load()["position_qty"] < 0


@pytest.mark.parametrize("origin,status", [
    (MISSING, "origin_missing"), (None, "origin_missing"), (0, "origin_invalid"),
    (-1, "origin_invalid"), (True, "origin_invalid"), (str(ORIGIN), "origin_invalid"),
    (float(ORIGIN), "origin_invalid"), (ORIGIN + 2_000, "origin_future"),
])
def test_missing_invalid_or_future_source_blocks_new_direction(tmp_path, origin, status):
    value = account(tmp_path)
    result = value.reconcile(target(origin=origin), 100_000, report=report(), now_ms=ORIGIN + 1_000)
    assert result["btc_entry_timing"]["status"] == status
    assert result["btc_entry_timing"]["allowed"] is False
    assert result["fill"] is None


@pytest.mark.parametrize("policy", [None, {}, {"mode": "unknown"},
    {"mode": btc_entry_timing.POLICY_MODE, "max_initial_entry_age_seconds": True},
    {"mode": btc_entry_timing.POLICY_MODE, "max_initial_entry_age_seconds": float("nan")},
    {"mode": btc_entry_timing.POLICY_MODE, "max_initial_entry_age_seconds": 0},
])
def test_invalid_policy_blocks_new_risk_but_never_account_hard_stop(tmp_path, policy):
    value = account(tmp_path)
    selected = {"shadow_profile": {"entry_timing_policy": policy}}
    first = value.reconcile(target(), 100_000, report=selected, now_ms=ORIGIN + 1_000)
    assert first["btc_entry_timing"]["status"] == "invalid_policy"
    assert first["fill"] is None
    value.reconcile(target(), 100_000, report=report(), now_ms=ORIGIN + 2_000)
    state = value.load()
    state["wallet_balance"] = 8_000
    value._save(state)
    result = value.reconcile(target(origin=MISSING, now=ORIGIN + 3_000), 100_000,
                             report=selected, now_ms=ORIGIN + 3_000)
    assert result["account_risk"]["status"] == "halted"
    assert result["fill"]["side"] == "SELL"
    assert value.load()["position_qty"] == 0


@pytest.mark.parametrize("selected", [None, {}, {"shadow_profile": {}}, {"shadow_profile": {"strategy_modes": ["timeseries_trend"]}}])
def test_legacy_reports_keep_existing_behavior_without_policy(tmp_path, selected):
    value = account(tmp_path)
    now = ORIGIN + 60 * HOUR
    result = value.reconcile(target(origin=MISSING, now=now), 100_000,
                             report=selected, now_ms=now)
    assert result["btc_entry_timing"]["status"] == "not_configured"
    assert result["fill"]["side"] == "BUY"


def test_runner_diagnostics_can_inspect_raw_report_point():
    point = {"signed_qty": -.0005, "origin_signal_time_ms": ORIGIN}
    decision = btc_entry_timing.decision_at(report(), point, current_qty=0, now_ms=ORIGIN + HOUR)
    assert decision["status"] == "expired"
    assert decision["reasons"] == ["btc_initial_entry_origin_expired"]
