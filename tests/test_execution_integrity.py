"""Execution-boundary regressions: old targets must obey current entry restrictions."""
from pathlib import Path
import json
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trading_execution import SimulationAccount, LiveExecutor
import multifactor
import public_context
import simulate_range_swing as sim


@pytest.fixture(autouse=True)
def execution_environment(monkeypatch):
    for key, value in {"LLM_TRADE_GATE_ENABLED": "false", "SIM_MAX_LEVERAGE": "2",
                       "SIM_MAX_NOTIONAL_USDT": "0", "SIM_TAKER_FEE": "0.00045",
                       "SIM_SLIPPAGE_BPS": "1"}.items():
        monkeypatch.setenv(key, value)


def account():
    result = SimulationAccount(Path("unused.json"), persist=False, allow_llm=False)
    state = result.reset(10000, now_ms=0)
    state["observed_flat_target"] = True
    result._save(state)
    return result


def target(leverage, time=1):
    return {"signal_time_ms": time, "target_leverage": leverage,
            "signal_price": 100000, "position_id": "same_old_target"}


def blocked_report(reason, side):
    report = {"event_overlay": {"current": {"allowed": True}, "coverage_status": "healthy"},
              "multifactor_overlay": {"current": {"long": {"allowed": True}, "short": {"allowed": True}}}}
    if reason == "event":
        report["event_overlay"]["current"]["allowed"] = False
    elif reason == "factor":
        report["multifactor_overlay"]["current"][side]["allowed"] = False
    else:
        report["event_overlay"]["coverage_status"] = "degraded"
    return report


@pytest.mark.parametrize("reason", ["event", "factor", "source"])
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("operation", ["entry", "increase", "hold", "reduce", "exit", "reverse"])
def test_current_gate_applies_to_same_old_target_and_preserves_exits(reason, side, operation):
    a = account()
    if operation != "entry":
        a.reconcile(target(side*.1), 100000, now_ms=1)
    before = a.load()["position_qty"]
    desired = side * {"entry": .2, "increase": .2, "hold": .1,
                      "reduce": .05, "exit": 0, "reverse": -.2}[operation]
    if operation == "hold":
        desired = before * 100000 / a.snapshot(100000)["account"]["margin_balance"]
    result = a.reconcile(target(desired, 2), 100000,
                         report=blocked_report(reason, "long" if desired>0 else "short"), now_ms=2)
    after = a.load()["position_qty"]
    if operation in ("entry", "increase", "hold"):
        assert after == before
        assert result["fill"] is None
    elif operation in ("exit", "reverse"):
        assert after == 0
        assert result["fill"] is not None
    else:
        assert 0 < abs(after) < abs(before)


def test_old_target_below_lot_limit_cannot_enter_later_during_block():
    a = account()
    assert a.reconcile(target(.005), 100000, now_ms=1)["fill"] is None
    assert a.load()["position_qty"] == 0
    assert a.reconcile(target(.2,2), 100000, report=blocked_report("event","long"), now_ms=2)["fill"] is None


def test_blocked_live_reversal_only_closes_old_inventory(tmp_path):
    client = Mock()
    client.validate_live_ready.return_value = {"leverage":2, "max_notional_usdt":20000}
    held = {"account":{"wallet_balance":10000,"margin_balance":10000},
            "positions":[{"signed_quantity":.01}]}
    flat = {"account":{"wallet_balance":10000,"margin_balance":10000}, "positions":[]}
    client.account_snapshot.side_effect = [held, flat]
    client.quantize_quantity.side_effect = lambda qty,symbol: round(abs(qty),3)
    client.market_order.return_value = {"status":"FILLED"}
    result = LiveExecutor(client,tmp_path/"live.json").reconcile(target(-.2),100000,
                                                               blocked_report("event","short"))
    assert result["target_qty"] == 0
    client.market_order.assert_called_once()
    assert client.market_order.call_args.kwargs["reduce_only"] is True


def event_context(tmp_path, *, health=True):
    path=tmp_path/"events.json"
    payload={"schema_version":1,"events":[{"event_id":"announced_release",
        "published_at_utc":sim.iso_utc_from_ms(0),"starts_at_utc":sim.iso_utc_from_ms(1000),
        "ends_at_utc":sim.iso_utc_from_ms(2000),"severity":1,"block_entries":True}],
        "coverage_checks":[{"available_at_utc":sim.iso_utc_from_ms(0),
            "sources":{name:{"ok":health} for name in public_context.REQUIRED}}]}
    path.write_text(json.dumps(payload))
    return {"event_overlay":{"current":{"allowed":True},"coverage_status":"healthy"},
            "execution_entry_context":{"event_snapshot":str(path)}}


def test_calendar_is_recomputed_at_execution_not_cached_report_time(tmp_path):
    report=event_context(tmp_path)
    a=account()
    assert a.reconcile(target(.1),100000,report=report,now_ms=999)["fill"] is not None
    before=a.load()["position_qty"]
    result=a.reconcile(target(.2,2),100000,report=report,now_ms=1000)
    assert a.load()["position_qty"] == before
    assert "current_event_blocks_entries" in result["execution_entry_gate"]["reasons"]
    assert a.reconcile(target(0,3),100000,report=report,now_ms=1001)["fill"] is not None


def test_fresh_clock_after_provider_crosses_calendar_boundary(tmp_path):
    report=event_context(tmp_path)
    a=account()
    result=a.reconcile(target(.2),100000,report=report,now_ms=999,execution_clock=lambda:1000)
    assert result["fill"] is None
    assert result["execution_entry_gate"]["checked_at_ms"] == 1000


def test_source_health_is_reloaded_even_when_report_cache_is_healthy(tmp_path):
    report=event_context(tmp_path,health=False)
    a=account()
    result=a.reconcile(target(.2),100000,report=report,now_ms=500)
    assert result["fill"] is None
    assert "current_public_sources_unavailable" in result["execution_entry_gate"]["reasons"]


def factor_context(tmp_path,price=100):
    now=8*sim.MS_PER_DAY
    profile={"candidate_id":"test_first_seen","live_orders_allowed":False,
             "groups":{"btc_momentum":1},"minimum_group_coverage":1,
             "minimum_risk_multiplier":.35,"block_alignment_below":-.35,
             "block_stress_at":.9,"availability_mode":"first_seen"}
    profile_path=tmp_path/"profile.json"
    profile_path.write_text(json.dumps(profile))
    factor_path=tmp_path/"factors.json"
    factor_path.write_text(json.dumps({"schema_version":1,"series":{"btc_close":[
        [sim.MS_PER_DAY,sim.MS_PER_DAY,100,sim.MS_PER_DAY],
        [7*sim.MS_PER_DAY,7*sim.MS_PER_DAY,100,7*sim.MS_PER_DAY],
        [now,now,price,now]]}}))
    return now,{"multifactor_overlay":{"current":{"long":{"allowed":True},"short":{"allowed":True}}},
                "execution_entry_context":{"factor_profile":str(profile_path),
                    "factor_profile_sha256":multifactor.profile_hash(profile),"factor_snapshot":str(factor_path)}}


def test_current_factor_direction_and_same_target_retry_after_recovery(tmp_path):
    now,report=factor_context(tmp_path,90)
    a=account()
    blocked=a.reconcile(target(.2),100000,report=report,now_ms=now)
    assert blocked["fill"] is None
    assert "current_factor:direction_conflict" in blocked["execution_entry_gate"]["reasons"]
    now,healthy=factor_context(tmp_path,100)
    assert a.reconcile(target(.2,2),100000,report=healthy,now_ms=now+1)["fill"] is not None


def test_expired_factors_block_even_if_cached_decision_allowed(tmp_path):
    now,report=factor_context(tmp_path)
    result=account().reconcile(target(.2),100000,report=report,now_ms=now+3*3600000+1)
    assert result["fill"] is None
    assert "current_factor:missing_or_stale_factors" in result["execution_entry_gate"]["reasons"]


def test_broken_context_and_clock_do_not_block_exit(tmp_path):
    a=account()
    a.reconcile(target(.1),100000,now_ms=1)
    report=event_context(tmp_path)
    Path(report["execution_entry_context"]["event_snapshot"]).write_text("{broken")
    result=a.reconcile(target(.2,2),100000,report=report,now_ms=2)
    assert result["fill"] is None
    def broken_clock():
        raise RuntimeError("exchange time unavailable")
    exited=a.reconcile(target(0,3),100000,report=report,now_ms=3,execution_clock=broken_clock)
    assert a.load()["position_qty"] == 0
    assert exited["fill"] is not None
