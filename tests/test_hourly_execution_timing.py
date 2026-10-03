"""Known-open targets must match full replay at the same information cutoff."""
from dataclasses import replace
from unittest.mock import Mock
from datetime import datetime, timezone
from unittest.mock import patch
import sys

import pytest

from test_strategy_engine import candle, config
import backtest_execution
import execution_targets
import execution_portfolio
from paper_trade_frozen_portfolio import annotate_open_position_fractions
import portfolio_risk
import simulate_range_swing as sim
import timeseries_execution as hourly
import paper_trade_frozen_portfolio as paper

HOUR = 3_600_000


def inputs(values):
    hours = [replace(candle(i,p,p+1,p-1,p,HOUR), quote_volume=2_000_000*(i+1))
             for i,p in enumerate(values)]
    base = [candle(i,values[i//12],values[i//12]+1,values[i//12]-1,values[i//12])
            for i in range(len(values)*12)]
    return hours, base


def cfg():
    return config(timeseries_timeframe="1h", timeseries_fast_ema=2, timeseries_slow_ema=4,
                  timeseries_vol_lookback_bars=4, timeseries_min_ema_spread_pct=.005,
                  max_drawdown_stop_pct=0)


def stream(hours,base,opening=None,funding=None):
    settings = cfg()
    asof = opening.observed_at_ms if opening else hours[-1].close_time_ms
    sleeve = hourly.build_sleeve(hours,settings,4*HOUR,funding,opening=opening,asof_ms=asof)
    annotate_open_position_fractions([sleeve])
    execution_targets.prepare_sleeves([sleeve])
    prices = hourly.decision_open_prices(base,[sleeve])
    result = execution_portfolio.combine(base,[sleeve],settings,
             portfolio_risk.DrawdownRiskPolicy(),4*HOUR,include_execution_target=True,decision_open_prices=prices)
    return list(execution_targets.target_stream(base,[sleeve],result,settings)), sleeve


def test_prior_closed_signal_is_executable_at_next_known_open():
    hours,base = inputs([100]*6+[107,110,112,114])
    old = sim.simulate_timeseries_trend(hours[:7],cfg(),4*HOUR)
    assert not old["trades"]  # Original missing-next-hour regression.
    prefix,sleeve = stream(hours[:7],base[:84],hourly.Opening(7*HOUR,110,7*HOUR+3000))
    assert prefix[-1]["signed_qty"] > 0
    assert prefix[-1]["available_time_ms"] == 7*HOUR
    assert sleeve["trades"][0]["entry_time_utc"] == sim.iso_utc_from_ms(7*HOUR)
    full,_ = stream(hours,base)
    observed = next(p for p in full if p["time_ms"] == prefix[-1]["time_ms"])
    assert prefix[-1]["signed_qty"] == pytest.approx(observed["signed_qty"])
    assert prefix[-1]["equity"] == pytest.approx(observed["equity"])
    assert prefix[-1]["position_id"] == observed["position_id"]


def test_each_hourly_prefix_matches_history_through_reversal_and_funding():
    values = [100]*6+[107,110,112,114,105,90,85,80,95,115]
    hours,base = inputs(values)
    funding = sim.FundingHistory([8*HOUR+HOUR//2,12*HOUR],[.001,-.001])
    full,_ = stream(hours,base,funding=funding)
    lookup = {p["time_ms"]:p for p in full}
    for index in range(6,len(values)):
        partial,_ = stream(hours[:index],base[:index*12],
                           hourly.Opening(index*HOUR,values[index],index*HOUR+3000),funding)
        for p in partial:
            expected = lookup[p["time_ms"]]
            assert p["signed_qty"] == pytest.approx(expected["signed_qty"])
            assert p["equity"] == pytest.approx(expected["equity"])
            assert p["position_id"] == expected["position_id"]


@pytest.mark.parametrize("future_low",[80,1])
def test_forming_hour_future_ohlc_and_volume_cannot_change_open_target(future_low):
    hours,base = inputs([100]*6+[107,110,112,114])
    original,_ = stream(hours,base)
    altered=list(hours)
    altered[7] = replace(altered[7],high=10000,low=future_low,close=80,volume=1e20,quote_volume=1e30)
    changed_base=list(base)
    for i in range(7*12,len(base)):
        changed_base[i] = replace(base[i],high=10000,low=future_low,close=80,volume=1e20,quote_volume=1e30)
    changed,_ = stream(altered,changed_base)
    at = 7*HOUR-300_000
    before = next(p for p in original if p["time_ms"]==at)
    after = next(p for p in changed if p["time_ms"]==at)
    assert before["signed_qty"] == pytest.approx(after["signed_qty"])
    assert before["equity"] == pytest.approx(after["equity"])
    assert before["position_id"] == after["position_id"]


def test_opening_fetch_does_not_parse_unfinished_candle_fields():
    client=Mock()
    client.public_get.return_value=[[7*HOUR,"110","invalid high","invalid low","invalid close"]]
    assert hourly.fetch_opening("BTCUSDT","1h",7*HOUR,7*HOUR+3000,client).price == 110
    client.public_get.assert_called_once()
    with pytest.raises(ValueError,match="not yet observable"):
        hourly.fetch_opening("BTCUSDT","1h",7*HOUR,7*HOUR-1,client)


@pytest.mark.parametrize("opening", [hourly.Opening(8*HOUR,110,8*HOUR),
                                    hourly.Opening(7*HOUR,110,9*HOUR),
                                    hourly.Opening(7*HOUR,float("nan"),7*HOUR)])
def test_bad_or_future_opening_is_rejected(opening):
    hours,_=inputs([100]*6+[107])
    with pytest.raises(ValueError,match="Opening"):
        hourly.build_sleeve(hours,cfg(),4*HOUR,opening=opening,asof_ms=7*HOUR+3000)


def test_account_replay_fills_at_same_known_open_boundary():
    hours,base=inputs([100]*6+[107,110,112,114])
    _,sleeve=stream(hours,base)
    replay=backtest_execution.replay(base,[sleeve],cfg(),4*HOUR,[],initial=10000)
    assert replay["fills"][0]["time_utc"] == sim.iso_utc_from_ms(7*HOUR+3000)
    assert replay["execution_model"] == hourly.MODEL


def test_forward_runner_uses_known_open_and_preserves_closed_signal_identity(tmp_path):
    hours,base=inputs([100]*6+[107,110])
    now=7*HOUR+3000
    with patch.object(sys,"argv",["paper","--tiered-drawdown"]):
        args=paper.parse_args()
    args.state_path=tmp_path/"state.json"
    args.report_path=tmp_path/"report.json"
    args.trades_path=tmp_path/"trades.csv"
    state={"symbol":"BTCUSDT","created_at_utc":sim.iso_utc_from_ms(4*HOUR),"observations":0}
    def fetch(symbol,interval,start,end):
        return base[:84] if interval=="5m" else hours[:7]
    with patch.object(paper.frozen_strategy,"load_frozen_strategy",return_value=(
            {"freeze_id":"test","config_sha256":"test"},cfg())), \
         patch.object(paper,"load_or_create_state",return_value=state), \
         patch.object(paper.sim,"fetch_futures_klines_range",side_effect=fetch), \
         patch.object(paper.sim,"fetch_funding_history",return_value=sim.FundingHistory([],[])), \
         patch.object(hourly,"fetch_opening",return_value=hourly.Opening(7*HOUR,110,now)) as tick, \
         patch.object(paper,"datetime",wraps=datetime) as clock:
        clock.now.return_value=datetime.fromtimestamp(now/1000,timezone.utc)
        result=paper.run_once(args)
    assert result["execution_target"]["signed_qty"] > 0
    assert result["execution_target"]["time_ms"] == 7*HOUR-300000
    assert result["execution_target"]["available_time_ms"] == 7*HOUR
    assert result["execution_model"] == hourly.MODEL
    assert result["execution_entry_context"]["factor_profile"] is None
    tick.assert_called_once_with("BTCUSDT","1h",7*HOUR,now)
