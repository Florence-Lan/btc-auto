"""The selected generation and simulation account share the startup contract."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

from test_hourly_execution_timing import HOUR, cfg, inputs
from test_hourly_startup import ACTIVATION, TREND
from test_selected_strategy_backtest import selected_inputs
import backtest_execution as backtest
import frozen_strategy
import multifactor
import paper_trade_frozen_portfolio as paper
import research_signal_engine
import simulate_range_swing as sim
import timeseries_execution as hourly
import trading_execution


@pytest.fixture(autouse=True)
def execution_environment(monkeypatch):
    for key, value in {
        "LLM_TRADE_GATE_ENABLED": "false", "SIM_TAKER_FEE": "0.00045",
        "SIM_SLIPPAGE_BPS": "1", "SIM_MAX_LEVERAGE": "2",
        "SIM_MAX_NOTIONAL_USDT": "0",
    }.items():
        monkeypatch.setenv(key, value)


def enable_startup(path, enabled=True):
    profile = json.loads(path.read_text())
    profile["hourly_startup_enabled"] = enabled
    path.write_text(json.dumps(profile))
    return profile


def test_forward_runner_keeps_inception_activation_and_hashes_across_refreshes(tmp_path):
    selected = selected_inputs(tmp_path)
    profile = enable_startup(selected.profile)
    start = sim._utc_ms(selected.start_utc)
    activation = start + 8 * HOUR
    market, _, _ = sim.load_market_snapshot(selected.market_snapshot)
    state = {
        "symbol": "BTCUSDT", "created_at_utc": sim.iso_utc_from_ms(activation),
        "observations": 0,
    }
    with patch.object(sys, "argv", [
        "paper", "--tiered-drawdown", "--manifest", profile["base_manifest"],
        "--factor-profile", str(selected.profile),
        "--factor-snapshot", str(selected.factor_snapshot),
        "--event-snapshot", profile["event_snapshot"],
        "--state-path", str(tmp_path / "paper_state.json"),
        "--report-path", str(tmp_path / "paper_report.json"),
        "--trades-path", str(tmp_path / "paper_trades.csv"),
    ]):
        args = paper.parse_args()
    clock_ms = {"now": start + 9 * HOUR + 3000}

    def fetch(symbol, interval, fetch_start, end):
        assert end == clock_ms["now"]
        return [bar for bar in market[interval] if bar.close_time_ms <= end]

    def opening(symbol, interval, timestamp, asof):
        bar = next(bar for bar in market[interval] if bar.open_time_ms == timestamp)
        return hourly.Opening(timestamp, bar.open, asof)

    with patch.object(paper, "load_or_create_state", return_value=state) as load, \
         patch.object(paper.sim, "fetch_futures_klines_range", side_effect=fetch), \
         patch.object(paper.sim, "fetch_funding_history", return_value=sim.FundingHistory([], [])), \
         patch.object(hourly, "fetch_opening", side_effect=opening), \
         patch.object(hourly, "build_sleeve", wraps=hourly.build_sleeve) as build, \
         patch.object(paper, "datetime", wraps=datetime) as clock:
        clock.now.side_effect = lambda *a, **kw: datetime.fromtimestamp(clock_ms["now"] / 1000, timezone.utc)
        first = paper.run_once(args)
        clock_ms["now"] = start + 10 * HOUR + 3000
        second = paper.run_once(args)

    assert [call.kwargs["activation_ms"] for call in build.call_args_list] == [activation, activation]
    assert load.call_count == 2
    assert state["observations"] == 2
    for report in (first, second):
        assert report["paper_inception_utc"] == state["created_at_utc"]
        assert report["execution_model"] == hourly.STARTUP_MODEL
        assert report["hourly_startup"]["activation_ms"] == activation
        assert report["hourly_startup"]["first_open_ms"] == activation
        assert report["hourly_startup"]["signal_close_ms"] == activation - 1
        assert report["hourly_startup"]["initial_side"] == "long"
        shadow = report["shadow_profile"]
        assert shadow["hourly_startup_enabled"] is True
        assert shadow["hourly_execution_model"] == hourly.STARTUP_MODEL
        assert shadow["factor_profile_sha256"] == multifactor.profile_hash(profile)
        assert shadow["hourly_execution_sha256"] == frozen_strategy.sha256_file(Path(hourly.__file__))
        assert shadow["hourly_signal_replay_sha256"] == frozen_strategy.sha256_file(Path(research_signal_engine.__file__))


@pytest.mark.parametrize("enabled", [False, True])
def test_selected_backtest_honors_profile_startup_flag(tmp_path, enabled):
    args = selected_inputs(tmp_path)
    profile = enable_startup(args.profile, enabled)
    report = backtest.run_selected_strategy(args)
    assert report["execution_model"] == (hourly.STARTUP_MODEL if enabled else hourly.MODEL)
    assert report["profile_sha256"] == multifactor.profile_hash(profile)
    assert report["places_orders"] is False
    if enabled:
        assert report["hourly_startup"]["activation_ms"] == sim._utc_ms(args.start_utc)
        assert report["hourly_startup"]["first_open_ms"] == sim._utc_ms(args.start_utc)
        assert report["code_sha256"]["research_signal_engine.py"] == frozen_strategy.sha256_file(Path(research_signal_engine.__file__))
    else:
        assert report["hourly_startup"] is None


def test_startup_replay_recognizes_known_open_and_keeps_open_fill_boundary():
    hours, base = inputs(TREND)
    sleeve = hourly.build_sleeve(hours, cfg(), 4 * HOUR, activation_ms=ACTIVATION)
    report = backtest.replay(base, [sleeve], cfg(), 4 * HOUR, [], initial=10000)
    assert report["execution_model"] == hourly.STARTUP_MODEL
    assert report["fills"][0]["time_utc"] == sim.iso_utc_from_ms(ACTIVATION + 3000)
    assert report["fills"][0]["price"] == pytest.approx(TREND[9] * 1.0001)


def test_replay_accepts_first_startup_target_when_account_starts_at_activation():
    hours, base = inputs(TREND)
    sleeve = hourly.build_sleeve(hours, cfg(), ACTIVATION, activation_ms=ACTIVATION)
    report = backtest.replay(base, [sleeve], cfg(), ACTIVATION, [], initial=10000)
    assert report["execution_model"] == hourly.STARTUP_MODEL
    assert report["summary"]["fills"] == 1
    assert report["summary"]["open_quantity"] > 0
    assert sim._utc_ms(report["fills"][0]["time_utc"]) <= ACTIVATION + 300_000 + 3000


def test_replay_rejects_mixed_known_open_generations():
    hours, base = inputs(TREND)
    old = hourly.build_sleeve(hours, cfg(), 4 * HOUR)
    activated = hourly.build_sleeve(hours, cfg(), 4 * HOUR, activation_ms=ACTIVATION)
    with pytest.raises(ValueError, match="different hourly execution models"):
        backtest.replay(base, [old, activated], cfg(), 4 * HOUR, [])


CREATED = 9 * HOUR
NOW = CREATED + 3000


def new_account(tmp_path):
    account = trading_execution.SimulationAccount(tmp_path / "account.json")
    account.reset(1000, now_ms=CREATED)
    return account


def target(origin=CREATED):
    return {
        "target_leverage": .115, "signal_time_ms": CREATED,
        "signal_price": 50000, "position_id": "new-hourly-generation",
        "origin_signal_time_ms": origin,
    }


@pytest.mark.parametrize("origin", [CREATED, CREATED + 1, NOW])
def test_simulation_accepts_first_fresh_startup_target_only_with_observed_origin(tmp_path, origin):
    account = new_account(tmp_path)
    report = {"execution_model": hourly.STARTUP_MODEL}
    first = account.reconcile(target(origin), 50000, report=report, now_ms=NOW)
    assert first["entry_guard"]["status"] == "fresh_signal_allowed"
    assert first["fill"]["side"] == "BUY"
    assert account.load()["fill_count_total"] == 1

    restarted = trading_execution.SimulationAccount(account.path)
    repeated = restarted.reconcile(target(origin), 50000, report=report, now_ms=NOW + 30_000)
    assert repeated["fill"] is None
    assert restarted.load()["fill_count_total"] == 1


@pytest.mark.parametrize("origin", [None, CREATED - 1, NOW + 1, "32400000", True])
def test_startup_model_still_blocks_missing_old_future_or_invalid_origin(tmp_path, origin):
    account = new_account(tmp_path)
    signal = target(origin)
    if origin is None:
        signal.pop("origin_signal_time_ms")
    result = account.reconcile(signal, 50000, report={"execution_model": hourly.STARTUP_MODEL}, now_ms=NOW)
    assert result["entry_guard"]["status"] == "waiting_for_next_signal"
    assert result["fill"] is None
    assert account.load()["position_qty"] == 0


@pytest.mark.parametrize("model", [None, hourly.MODEL, "legacy_research"])
def test_other_models_keep_first_target_guard_even_with_fresh_origin(tmp_path, model):
    account = new_account(tmp_path)
    report = {} if model is None else {"execution_model": model}
    result = account.reconcile(target(), 50000, report=report, now_ms=NOW)
    assert result["entry_guard"]["status"] == "waiting_for_next_signal"
    assert result["fill"] is None


def test_fresh_startup_permission_cannot_bypass_event_entry_block(tmp_path):
    account = new_account(tmp_path)
    event_path = tmp_path / "events.json"
    event_path.write_text(json.dumps({"schema_version": 1, "events": [{
        "event_id": "release", "published_at_utc": sim.iso_utc_from_ms(CREATED - HOUR),
        "starts_at_utc": sim.iso_utc_from_ms(CREATED),
        "ends_at_utc": sim.iso_utc_from_ms(CREATED + HOUR - 1),
        "severity": 1, "block_entries": True,
    }]}))
    report = {
        "execution_model": hourly.STARTUP_MODEL,
        "execution_entry_context": {"event_snapshot": str(event_path)},
    }
    result = account.reconcile(target(), 50000, report=report, now_ms=NOW)
    assert result["entry_guard"]["status"] == "fresh_signal_allowed"
    assert result["execution_entry_gate"]["status"] == "blocked"
    assert "current_event_blocks_entries" in result["execution_entry_gate"]["reasons"]
    assert result["fill"] is None
    assert account.load()["fill_count_total"] == 0


def test_live_executor_keeps_default_first_target_guard_for_startup_report(tmp_path):
    client = Mock()
    client.validate_live_ready.return_value = {"leverage": 2, "max_notional_usdt": 2000}
    client.account_snapshot.return_value = {
        "account": {"wallet_balance": 1000, "margin_balance": 1000}, "positions": [],
    }
    client.quantize_quantity.side_effect = lambda quantity, symbol: round(abs(quantity), 3)
    path = tmp_path / "live.json"
    trading_execution.write_json(path, {"created_at_utc": sim.iso_utc_from_ms(CREATED)})
    result = trading_execution.LiveExecutor(client, path).reconcile(
        target(), 50000, report={"execution_model": hourly.STARTUP_MODEL},
    )
    assert result["entry_guard"]["status"] == "waiting_for_next_signal"
    assert result["orders"] == []
    client.market_order.assert_not_called()
