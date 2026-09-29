import json
from unittest.mock import patch

import pytest

from test_strategy_engine import ROOT, candle, config
import reentry_candidate
import research_reentry as engine
import validate_cost_control as validation
import frozen_strategy


def metric(cost, net, drawdown):
    return {"accounting": {"fees": cost, "slippage_cost": 0},
            "total_return_pct": net, "max_drawdown_pct": drawdown}


def test_lower_cost_alone_cannot_select_worse_return_or_excess_drawdown():
    reference = metric(4, 10, 5)
    rows = {"lower_return": metric(2, 9, 4), "too_risky": metric(2, 11, 7),
            "too_expensive": metric(3.1, 12, 4)}
    assert validation.select_variant(rows, reference) is None
    rows["eligible"] = metric(3, 10, 6)
    assert validation.select_variant(rows, reference) == "eligible"
    assert not all(validation.checks(rows["eligible"], reference, metric(6, -1, 6)).values())


def test_new_profile_loads_and_rejects_parameter_drift(tmp_path):
    plan = json.loads((ROOT / validation.PLAN).read_text())
    name, policy = next(iter(plan["policies"].items()))
    files = [validation.PLAN, ROOT / plan["base_manifest"]]
    files += list((ROOT / "scripts").glob("*.py"))
    hashes = {str(path.relative_to(ROOT) if path.is_absolute() else path):
              frozen_strategy.sha256_file(path) for path in files}
    profile = {"candidate_id": "test", "research_only": True, "live_orders_allowed": False,
               "research_plan": str(validation.PLAN), "base_manifest": plan["base_manifest"],
               "policy": policy, "tactical_weight": 1.0,
               "selected_variant": name + "_tactical_1", "input_hashes": hashes}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert reentry_candidate.load_candidate(path, ROOT / plan["base_manifest"])["policy"] == policy
    profile["policy"]["cooldown_bars"] = 1
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="parameters"):
        reentry_candidate.load_candidate(path, ROOT / plan["base_manifest"])


@pytest.mark.parametrize("side", ["long", "short"])
def test_longer_confirmation_blocks_recross_and_fills_next_open(side):
    prices = [100, 100, 100, 100, 104, 100, 101] + list(range(105, 165))
    fast = [99, 99, 99, 103, 103, 102, 100] + list(range(104, 164))
    if side == "short":
        prices = [200-p for p in prices]
        fast = [200-p for p in fast]
    bars = [candle(i, p, p+1, p-1, p, 3600000) for i, p in enumerate(prices)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=1, timeseries_slow_ema=2,
                 timeseries_vol_lookback_bars=2, timeseries_min_ema_spread_pct=.02,
                 entry_slippage_bps=0, exit_slippage_bps=0, depth_impact_bps=0)
    policy = engine.ExitPolicy("volatility", reentry_enabled=True, cooldown_bars=24, breakout_bars=48)
    with patch.object(engine, "ema", side_effect=[fast, [100]*len(bars)]), \
         patch.object(engine, "atr", return_value=[1]*len(bars)):
        result = engine.simulate_with_exit_policy(bars, cfg, policy=policy)
    assert result["trades"][0]["exit_time_utc"] == bars[6].open_time_utc
    assert result["reentry_times"] == [bars[49].open_time_utc]
    assert result["trades"][1]["entry_price"] == bars[49].open
