"""Causal multi-timeframe confirmation and the user's joint outcome metric."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_selective_stock_entries as research
from stock_swing_signals import Candle


def test_higher_timeframe_visibility_uses_close_boundary():
    ends = [4, 8, 12]
    assert research.completed_higher_index(ends, 3) == -1
    assert research.completed_higher_index(ends, 4) == 0
    assert research.completed_higher_index(ends, 7) == 0
    assert research.completed_higher_index(ends, 8) == 1


def test_volume_confirmation_ignores_future_bars_and_rejects_sparse_activity():
    bars = [Candle(i * research.engine.HOUR, 100, 101, 99, 100, 10) for i in range(25)]
    bars[20] = Candle(20 * research.engine.HOUR, 100, 101, 99, 100, 20)
    assert research.volume_confirmed(bars, 20)
    bars[21] = Candle(21 * research.engine.HOUR, 100, 101, 99, 100, 100000)
    assert research.volume_confirmed(bars, 20)
    for i in (15, 16, 17):
        bars[i] = Candle(i * research.engine.HOUR, 100, 101, 99, 100, 0)
    assert not research.volume_confirmed(bars, 20)


@pytest.mark.parametrize('side', [1, -1])
def test_retest_requires_later_touch_and_close_confirmation_for_both_directions(side):
    def reflected(i, o, h, l, c):
        prices = (o, h, l, c) if side == 1 else (200 - o, 200 - l, 200 - h, 200 - c)
        return Candle(i * research.engine.HOUR, *prices, 20)
    bars = [reflected(0, 100, 101, 99, 100), reflected(1, 100.1, 100.2, 100, 100.1),
            reflected(2, 100.1, 100.6, 99.9, 100.5)]
    assert research.retest_confirmed(bars, 2, 1, side, 100, 1)
    assert not research.retest_confirmed(bars, 2, 2, side, 100, 1)
    # Moving the boundary away rejects a bar that never revisited it.
    assert not research.retest_confirmed(bars, 2, 1, side, 100 - side * 2, 1)


def test_small_winners_cannot_count_as_net60_success():
    trades = [{'net_pnl': 1, 'net_return_initial_margin_pct': 1},
              {'net_pnl': 60, 'net_return_initial_margin_pct': 60},
              {'net_pnl': -10, 'net_return_initial_margin_pct': -10}]
    result = research.outcome_metrics(trades)
    assert result['net_winners'] == 2
    assert result['net60_trades'] == 1
    assert result['profitable_trades_below_net60'] == 1
    assert result['net_win_rate_pct'] == pytest.approx(200 / 3)
    assert result['net60_success_rate_pct'] == pytest.approx(100 / 3)


def test_zero_trades_and_unrealized_losses_cannot_pass_the_screen():
    summary = {'user_outcomes': research.outcome_metrics([]), 'net_closed_pnl': 0,
               'estimated_close_return_pct': 0, 'max_sampled_drawdown_pct': 0,
               'liquidation_stress_count': 0}
    assert set(research.qualification_failures(summary, 30)) == {
        'insufficient_closed_trade_sample', 'net_win_rate_below_70pct',
        'net60_success_rate_below_70pct', 'nonpositive_net_profit'}
    summary.update(user_outcomes=research.outcome_metrics([
        {'net_pnl': 60, 'net_return_initial_margin_pct': 60} for _ in range(30)]),
        net_closed_pnl=1800, estimated_close_return_pct=-1)
    assert research.qualification_failures(summary, 30) == ['nonpositive_net_profit']


def fixture_bars():
    bars = []
    for i in range(80):
        close = 100 + i * .1
        bars.append(Candle(i * research.engine.HOUR, close - .03, close + .05, close - .15, close, 10))
    bars[60] = Candle(60 * research.engine.HOUR, 105.95, 106.2, 105.9, 106.15, 30)
    bars[61] = Candle(61 * research.engine.HOUR, 106.1, 106.14, 105.95, 106.05, 10)
    bars[62] = Candle(62 * research.engine.HOUR, 106.05, 106.18, 106, 106.15, 30)
    return bars


def test_multi_timeframe_signal_prefix_is_unchanged_by_future_prices_and_volumes():
    bars = fixture_bars()
    cfg = {'ema_fast': 2, 'ema_mid': 3, 'ema_slow': 4, 'ema_slope_bars': 1,
           'warmup_bars': 4, 'breakout_bars': 3}
    higher = research.engine.aggregate_4h(bars)
    before, checkpoints = research.build_signals(bars, higher, cfg, research.MODES[2])
    cutoff = 64 * research.engine.HOUR
    prefix = {t: s for t, s in before.items() if t < cutoff}
    assert prefix  # Exercise a real accepted retest, rather than an empty equality.
    setups = [p['setup_close_ms'] for p in checkpoints]
    assert len(setups) == len(set(setups))
    for i in range(64, len(bars)):
        bars[i] = Candle(i * research.engine.HOUR, 1000, 1100, 900, 1000, 100000)
    after, _ = research.build_signals(bars, research.engine.aggregate_4h(bars), cfg, research.MODES[2])
    assert {t: s for t, s in after.items() if t < cutoff} == prefix
