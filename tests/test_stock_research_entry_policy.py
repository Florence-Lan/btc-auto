"""Research restrictions stay causal and a small or unrealized profit cannot pass."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stock_research_entry_policy as policy
from stock_swing_signals import Candle
from research_stock_mechanisms import assessment
from backtest_stock_swing_120 import parse_time


@pytest.mark.parametrize("date,allowed", [
    ("2026-03-06T14:30:00Z", True), ("2026-03-09T13:30:00Z", True),
    ("2026-03-09T13:25:00Z", False), ("2026-03-09T20:00:00Z", False),
    ("2026-03-08T16:00:00Z", False),
])
def test_us_clock_observes_dst_weekends_and_end_exclusion(date, allowed):
    assert policy.regular_clock("SNDKUSDT", parse_time(date)) is allowed


@pytest.mark.parametrize("date,allowed", [
    ("2026-06-01T00:00:00Z", True), ("2026-06-01T06:25:00Z", True),
    ("2026-06-01T06:30:00Z", False), ("2026-05-31T01:00:00Z", False),
])
def test_korean_clock(date, allowed):
    assert policy.regular_clock("SKHYNIXUSDT", parse_time(date)) is allowed


def test_minimum_of_prior_contiguous_bars_never_reads_current_or_future():
    step, t = 300000, 3000000
    bars = {t - i * step: Candle(t - i * step, 100, 101, 99, 100, i * 10) for i in range(1, 7)}
    bars[t] = Candle(t, 100, 101, 99, 100, 0)
    source = {"trade": bars}
    assert policy.preceding_volume(source, t, step, 6) == 10
    bars[t] = Candle(t, 100, 101, 99, 100, 1000000)
    assert policy.preceding_volume(source, t, step, 6) == 10
    bars.pop(t - 4 * step)
    assert policy.preceding_volume(source, t, step, 6) is None


@pytest.mark.parametrize("volume", [0, -1, float("nan"), float("inf")])
def test_invalid_prior_volume_cannot_authorize_an_entry(volume):
    source = {"trade": {0: SimpleNamespace(volume=volume)}}
    assert policy.preceding_volume(source, 300000, 300000, 1) is None


@pytest.mark.parametrize("lookback", [True, 0, -1, 13, 1.5, "6"])
def test_rejects_invalid_lookback(lookback):
    with pytest.raises(ValueError, match="lookback"):
        policy.validate({"entry_volume_lookback_bars": lookback}, 300000)


@pytest.mark.parametrize("validity", [True, -1, 241, 1.5, "240", 7])
def test_rejects_invalid_deferred_expiry(validity):
    with pytest.raises(ValueError, match="validity|expiry"):
        policy.validate({"entry_signal_validity_minutes": validity}, 300000)


def passing_runs():
    row = {"net_closed_pnl": 10, "closed_trades": 10, "max_sampled_drawdown_pct": 2,
           "liquidation_stress_count": 0, "target_trades_net_at_least_120pct_margin": 1,
           "execution_volume_diagnostics": {"entry_bars_zero_reported_volume": 0, "exit_bars_zero_reported_volume": 0},
           "trade_diagnostics": {"closed_return_without_largest_winner_pct": .1}}
    return {name: copy.deepcopy(row) for name in [
        *(f"{w}_cost{c}" for w in ("full", "development", "validation", "recent30d") for c in (1, 2)),
        "full_slippage50bps", "full_margin5pct"]}


def test_gate_never_promotes_even_when_retrospective_screen_passes():
    result = assessment(passing_runs())
    assert result["passed_retrospective_screen"]
    assert not result["execution_qualified"] and not result["forward_validated"]


@pytest.mark.parametrize("key,value", [("net_closed_pnl", 0), ("closed_trades", 2),
                                      ("liquidation_stress_count", 1), ("max_sampled_drawdown_pct", 9)])
def test_positive_unrealized_return_does_not_hide_failed_closed_pnl_sample_or_risk(key, value):
    runs = passing_runs()
    runs["validation_cost2"][key] = value
    runs["validation_cost2"]["estimated_close_return_pct"] = 100
    assert not assessment(runs)["passed_retrospective_screen"]


def test_missing_stop_liquidity_and_winner_dependency_fail_screen():
    runs = passing_runs()
    runs["full_cost2"]["execution_volume_diagnostics"]["exit_bars_zero_reported_volume"] = 1
    runs["full_cost2"]["trade_diagnostics"]["closed_return_without_largest_winner_pct"] = -.01
    result = assessment(runs)
    assert not result["passed_retrospective_screen"]
    assert any("zero-volume" in reason for reason in result["failure_reasons"])
    assert any("best trade" in reason for reason in result["failure_reasons"])
