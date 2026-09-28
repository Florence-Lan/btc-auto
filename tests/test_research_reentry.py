from unittest.mock import patch

import pytest

from test_strategy_engine import candle, config
import research_reentry as r
import simulate_range_swing as sim


def test_baseline_preserves_frozen_engine_behavior():
    prices = [100]*20 + list(range(100, 130)) + list(range(130, 90, -1))
    bars = [candle(i, p, p+1, p-1, p, 3600000) for i, p in enumerate(prices)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=3, timeseries_slow_ema=10,
                 timeseries_vol_lookback_bars=10, timeseries_min_ema_spread_pct=.001)
    assert r.simulate_with_exit_policy(bars, cfg) == sim.simulate_timeseries_trend(bars, cfg)


@pytest.mark.parametrize("side,prices,fast", [("long", [100, 101, 103], [99, 100, 102]),
                                            ("short", [103, 102, 100], [104, 103, 101])])
def test_breakout_requires_cooldown_and_direction(side, prices, fast):
    policy = r.ExitPolicy("volatility", reentry_enabled=True, cooldown_bars=2, breakout_bars=2)
    assert r.reentry_allowed(policy, side, side, 0, 2, prices, fast, side)
    assert not r.reentry_allowed(policy, side, side, 1, 2, prices, fast, side)
    assert not r.reentry_allowed(policy, side, side, 0, 2, prices, fast, None)
    assert not r.reentry_allowed(policy, side, side, 0, 2, prices, [fast[2]]*3, side)


def test_stop_distance_sizing_never_increases_leverage_and_blocks_missing_atr():
    policy = r.ExitPolicy("volatility", stop_risk_fraction=.005)
    assert r.entry_leverage_limit(policy, 2, 1, 100) == .25
    assert r.entry_leverage_limit(policy, .1, 1, 100) == .1
    assert r.entry_leverage_limit(policy, 2, 0, 100) == 0


def test_reentry_after_protection_fills_next_open_and_recross_cannot_bypass_cooldown():
    prices = [100, 100, 100, 100, 104, 100, 101, 105, 106, 107, 108]
    bars = [candle(i, p, p+1, p-1, p, 3600000) for i, p in enumerate(prices)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=1, timeseries_slow_ema=2,
                 timeseries_vol_lookback_bars=2, timeseries_min_ema_spread_pct=.02,
                 entry_slippage_bps=0, exit_slippage_bps=0, depth_impact_bps=0)
    # Initial cross at 3 -> entry 4; adverse close 5 -> exit open 6.
    # A recross at 7 must wait until 8 (2 bars) -> re-entry at open 9.
    fast = [99, 99, 99, 103, 103, 102, 100, 104, 105, 106, 107]
    policy = r.ExitPolicy("volatility", reentry_enabled=True, cooldown_bars=2, breakout_bars=2)
    with patch.object(r, "ema", side_effect=[fast, [100]*len(bars)]), patch.object(r, "atr", return_value=[1]*len(bars)):
        result = r.simulate_with_exit_policy(bars, cfg, policy=policy)
    assert result["trades"][0]["exit_reason"] == "volatility_exit"
    assert result["trades"][0]["exit_time_utc"] == bars[6].open_time_utc
    assert result["reentry_times"] == [bars[9].open_time_utc]
    assert result["trades"][1]["entry_price"] == bars[9].open


def test_future_extension_does_not_change_closed_reentry_trades():
    prices = [100]*20 + list(range(100, 130)) + list(range(130, 115, -1)) + list(range(115, 140)) + list(range(140, 100, -1))
    bars = [candle(i, p, p+1, p-1, p, 3600000) for i, p in enumerate(prices)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=3, timeseries_slow_ema=10,
                 timeseries_vol_lookback_bars=10, timeseries_min_ema_spread_pct=.001)
    policy = r.ExitPolicy("volatility", reentry_enabled=True, stop_risk_fraction=.005)
    past = r.simulate_with_exit_policy(bars, cfg, policy=policy)
    future = bars + [candle(len(bars)+i, 200, 210, 190, 200, 3600000) for i in range(20)]
    extended = r.simulate_with_exit_policy(future, cfg, policy=policy)
    closed = [t for t in past["trades"] if t["exit_reason"] != "end"]
    assert closed
    assert extended["trades"][:len(closed)] == closed
