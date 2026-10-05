"""Fixed causal entry hypotheses for net60 research, never order submission."""
from stock_swing_signals import Signal, compute_indicators
from research_selective_stock_entries import build_signals as old_signals

ARMS = ('tight_breakout', 'trend_pullback', 'compression_breakout')
FAST_CONFIG = {'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
               'stop_atr': 1.25, 'min_stop_fraction': .0075, 'max_stop_fraction': .03}


def signal_at(bars, indicators, index, config, arm):
    if arm not in ARMS[1:]:
        raise ValueError('Unknown fast entry arm')
    if index < max(config['ema_slow'] - 1, config['ema_slope_bars'], 20):
        return None
    fast, mid, slow, atr = [indicators[k][index] for k in ('ema_fast', 'ema_mid', 'ema_slow', 'atr')]
    prior_mid = indicators['ema_mid'][index - config['ema_slope_bars']]
    prior_slow = indicators['ema_slow'][index - config['ema_slope_bars']]
    if any(v is None for v in (fast, mid, slow, atr, prior_mid, prior_slow)):
        return None
    bar = bars[index]
    if atr <= 0 or atr / bar.close > config['max_atr_fraction'] or bar.volume <= 0:
        return None
    # Activity uses completed bars only; no eventual entry-bar volume is visible.
    if sum(b.volume > 0 for b in bars[index - 5:index + 1]) < 4:
        return None
    side = (1 if fast > mid > slow and mid > prior_mid and slow > prior_slow else
            -1 if fast < mid < slow and mid < prior_mid and slow < prior_slow else 0)
    if not side or side * (bar.close - bar.open) <= 0:
        return None
    previous = bars[index - 1]
    if arm == 'trend_pullback':
        touched = any(side * ((b.low if side == 1 else b.high) - indicators['ema_fast'][i]) <= 0
                      for i, b in enumerate(bars[index - 3:index], index - 3))
        boundary = previous.high if side == 1 else previous.low
        extension = side * (bar.close - boundary)
        if not touched or extension <= 0 or side * (bar.close - mid) > 2 * atr:
            return None
    else:
        prior = bars[index - 12:index]
        boundary = max(b.high for b in prior) if side == 1 else min(b.low for b in prior)
        extension = side * (bar.close - boundary)
        width = max(b.high for b in bars[index - 6:index]) - min(b.low for b in bars[index - 6:index])
        if width > 3 * atr or not 0 < extension <= atr:
            return None
    return Signal(side, atr, bar.close, extension / atr, bar.time_ms)


def build_signals(hourly, higher, config, arm):
    if arm == ARMS[0]:
        return old_signals(hourly, higher, config, 'trend_volume_breakout_net60')[0]
    if arm not in ARMS:
        raise ValueError('Unknown entry arm')
    indicators = compute_indicators(hourly, config)
    signals = {}
    for index, bar in enumerate(hourly):
        signal = signal_at(hourly, indicators, index, config, arm)
        if signal:
            end = bar.time_ms + 3_600_000
            for retry in range(end, end + 3_600_000, 300_000):
                signals[retry] = signal
    return signals


def training_eligible(summary, metrics):
    """Monthly decisions require only information available before that month."""
    after_best = metrics['mean_after_removing_best_winner_usdt']
    return (metrics['closed_trades'] >= 20 and metrics['mean_closed_net_pnl_usdt'] > 0
            and after_best is not None and after_best > 0
            and summary['estimated_close_return_pct'] > 0
            and summary['max_sampled_drawdown_pct'] <= 6
            and summary['liquidation_stress_count'] == 0)
