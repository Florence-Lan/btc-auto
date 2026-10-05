"""Meaningful causal and admission checks for the fixed net60 revision."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stock_swing_signals import Candle, DEFAULT_CONFIG, compute_indicators
from stock_profit_candidate import FAST_CONFIG, build_signals, signal_at, training_eligible


def candles(side=1):
    rows = []
    for i in range(100):
        close = 100 + .2 * i
        prices = (close - .08, close + .05, close - 1.5, close)
        if side == -1:
            o, h, l, c = prices
            prices = (400-o, 400-l, 400-h, 400-c)
        rows.append(Candle(i*3_600_000, *prices, 10))
    return rows


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('arm', ['trend_pullback', 'compression_breakout'])
def test_real_signals_ignore_future_prices_and_volumes(side, arm):
    bars = candles(side)
    cfg = {**DEFAULT_CONFIG, **FAST_CONFIG}
    cutoff = 82*3_600_000
    before = {t:s for t,s in build_signals(bars, [], cfg, arm).items() if t<cutoff}
    assert before and all(s.direction == side for s in before.values())
    for i in range(82, len(bars)):
        bars[i] = Candle(i*3_600_000, 900, 1000, 800, 950, 1_000_000)
    after = {t:s for t,s in build_signals(bars, [], cfg, arm).items() if t<cutoff}
    assert before == after
    assert all(s.time_ms + 3_600_000 <= t < s.time_ms + 7_200_000 for t,s in before.items())


def test_pullback_requires_actual_touch_and_closed_activity():
    cfg = {**DEFAULT_CONFIG, **FAST_CONFIG}
    bars = candles()
    indicators = compute_indicators(bars, cfg)
    assert signal_at(bars, indicators, 80, cfg, 'trend_pullback')
    for i in (77, 78, 79):
        b = bars[i]
        bars[i] = Candle(b.time_ms, b.open, b.high, b.open-.01, b.close, b.volume)
    assert signal_at(bars, compute_indicators(bars,cfg), 80, cfg, 'trend_pullback') is None
    bars = candles()
    for i in (75, 76, 77):
        b = bars[i]
        bars[i] = Candle(b.time_ms,b.open,b.high,b.low,b.close,0)
    assert signal_at(bars, compute_indicators(bars,cfg), 80, cfg, 'trend_pullback') is None


def test_monthly_admission_rejects_small_sample_outlier_and_open_loss():
    summary = {'estimated_close_return_pct':1,'max_sampled_drawdown_pct':1,'liquidation_stress_count':0}
    metrics = {'closed_trades':20,'mean_closed_net_pnl_usdt':1,'mean_after_removing_best_winner_usdt':.1}
    assert training_eligible(summary, metrics)
    assert not training_eligible(summary, {**metrics,'closed_trades':19})
    assert not training_eligible(summary, {**metrics,'mean_after_removing_best_winner_usdt':-.1})
    assert not training_eligible({**summary,'estimated_close_return_pct':-1}, metrics)
    assert not training_eligible({**summary,'liquidation_stress_count':1}, metrics)
