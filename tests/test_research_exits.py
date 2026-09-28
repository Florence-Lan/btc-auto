from unittest.mock import patch

import pytest

from test_strategy_engine import candle, config
import simulate_range_swing as sim
import research_exits as research


@pytest.mark.parametrize("side,close,best,spread", [("long", 96, 101, .01), ("short", 104, 99, -.01)])
def test_volatility_loss_exit_is_symmetric(side, close, best, spread):
    assert research.exit_reason_at_close(research.ExitPolicy("volatility"), side, 100, close, best, 2, spread) == "volatility_exit"
    assert research.exit_reason_at_close(research.ExitPolicy(), side, 100, close, best, 2, spread) is None


@pytest.mark.parametrize("side,close,best", [("long", 108, 114), ("short", 92, 86)])
def test_volatility_trail_protects_profit_symmetrically(side, close, best):
    assert research.exit_reason_at_close(research.ExitPolicy("volatility"), side, 100, close, best, 2, .01) == "volatility_exit"


def test_neutral_exit_uses_next_open_and_does_not_immediately_reenter():
    bars = [candle(i, 105 if i == 5 else 100, 110, 90, 100, 3600000) for i in range(8)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=1, timeseries_slow_ema=2,
                 timeseries_vol_lookback_bars=2, timeseries_min_ema_spread_pct=.02,
                 entry_slippage_bps=0, exit_slippage_bps=0, depth_impact_bps=0)
    with patch.object(research, "ema", side_effect=[[100, 100, 100, 103, 99, 99, 99, 99], [100]*8]):
        result = research.simulate_with_exit_policy(bars, cfg, policy=research.ExitPolicy("neutral"))
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["entry_time_utc"] == bars[4].open_time_utc
    assert trade["exit_time_utc"] == bars[5].open_time_utc
    assert trade["avg_exit_price"] == 105
    assert trade["exit_reason"] == "neutral_exit"


def test_research_baseline_is_exactly_equal_to_frozen_engine():
    prices = [100]*20 + list(range(100, 130)) + list(range(130, 90, -1)) + list(range(90, 120))
    bars = [candle(i, p, p+1, p-1, p, 3600000) for i, p in enumerate(prices)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=3, timeseries_slow_ema=10,
                 timeseries_vol_lookback_bars=10, timeseries_min_ema_spread_pct=.001)
    assert research.simulate_with_exit_policy(bars, cfg) == sim.simulate_timeseries_trend(bars, cfg)


def test_future_prices_do_not_change_already_closed_trades():
    prices = [100]*20 + list(range(100, 130)) + list(range(130, 90, -1))
    bars = [candle(i, p, p+1, p-1, p, 3600000) for i, p in enumerate(prices)]
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=3, timeseries_slow_ema=10,
                 timeseries_vol_lookback_bars=10, timeseries_min_ema_spread_pct=.001)
    past = research.simulate_with_exit_policy(bars, cfg, policy=research.ExitPolicy("volatility"))
    future = bars + [candle(len(bars)+i, 200, 210, 190, 200, 3600000) for i in range(20)]
    extended = research.simulate_with_exit_policy(future, cfg, policy=research.ExitPolicy("volatility"))
    closed = [t for t in past["trades"] if t["exit_reason"] != "end"]
    assert closed
    assert extended["trades"][:len(closed)] == closed
