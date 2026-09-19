"""Causality and execution checks for research-only experiments."""
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import frozen_strategy
import simulate_range_swing as sim
from research_signal_engine import simulate_signals
from research_external_signals import ema_signals, donchian_signals, vwap_rsi_signals


def fixtures():
    _, cfg = frozen_strategy.load_frozen_strategy(sim.repo_root()/"config/frozen_strategy_active_20260809.json")
    candles = []
    for i in range(900):
        price = 100 + 6*math.sin(i/19) + .003*i
        t = 1704067200000 + i*3600000
        candles.append(sim.Candle(t,sim.iso_utc_from_ms(t),price,price+1,price-1,price+.2,100,100*(price+.1),t+3599999))
    return candles, replace(cfg,max_drawdown_stop_pct=0)


def test_signals_do_not_change_when_future_candles_are_added():
    candles,cfg = fixtures()
    start = candles[300].open_time_ms
    generators = [lambda c:ema_signals(c,cfg,start,.5),lambda c:ema_signals(c,cfg,start,1),lambda c:donchian_signals(c,start),lambda c:donchian_signals(c,start,"atr"),lambda c:donchian_signals(c,start,"volume"),lambda c:vwap_rsi_signals(c,start)]
    for generate in generators:
        full = generate(candles)
        for length in (400,550,730):
            assert generate(candles[:length]) == full[:length]


def test_signal_executes_next_open_and_flat_only_closes():
    candles,cfg = fixtures()
    signals = [None]*len(candles)
    signals[310],signals[320] = "long","flat"
    result = simulate_signals(candles,signals,"test",cfg,candles[300].open_time_ms,None)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["entry_time_utc"] == candles[311].open_time_utc
    assert trade["exit_time_utc"] == candles[321].open_time_utc
    assert trade["fees"] > 0
    assert trade["side"] == "long"


def test_replay_matches_frozen_ema_engine():
    candles,cfg = fixtures()
    start = candles[300].open_time_ms
    expected = sim.simulate_timeseries_trend(candles,cfg,start,None)
    actual = simulate_signals(candles,ema_signals(candles,cfg,start),"parity",cfg,start,None)
    assert actual["summary"]["final_equity"] == expected["summary"]["final_equity"]
    assert len(actual["trades"]) == len(expected["trades"])
    assert len(actual["trades"]) > 0
    for a,b in zip(actual["trades"],expected["trades"]):
        for key in ("entry_time_utc","exit_time_utc","net_pnl","initial_qty","fees"):
            assert a[key] == b[key]
