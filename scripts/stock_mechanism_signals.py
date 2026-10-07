"""Causal, symmetric price mechanisms for predeclared paper research."""
import math
from stock_swing_signals import Signal

FAMILIES = {'donchian20', 'dual_momentum', 'rsi_reclaim', 'shock_reclaim'}


def rsi(closes, period=14):
    if len(closes) <= period:
        return None
    changes = [b-a for a,b in zip(closes, closes[1:])]
    gain = sum(max(x,0) for x in changes[:period])/period
    loss = sum(max(-x,0) for x in changes[:period])/period
    for change in changes[period:]:
        gain = (gain*(period-1)+max(change,0))/period
        loss = (loss*(period-1)+max(-change,0))/period
    if gain+loss == 0:
        return 50.
    return 100*gain/(gain+loss)


def signal_at(bars, indicators, index, cfg):
    family = cfg['signal_family']
    if family not in FAMILIES:
        raise ValueError('Unknown declared price mechanism')
    if isinstance(index,bool) or not isinstance(index, int) or not 0 <= index < len(bars):
        raise ValueError('Invalid signal index')
    minimum = 170 if family == 'dual_momentum' else 60
    if index+1 < minimum:
        return None
    atr = indicators['atr'][index]
    if atr is None or not math.isfinite(atr) or atr <= 0 or atr/bars[index].close > cfg['max_atr_fraction']:
        return None
    bar, previous = bars[index], bars[index-1]
    side = None
    if family == 'donchian20':
        prior = bars[index-20:index]
        upper,lower = max(b.high for b in prior),min(b.low for b in prior)
        if bar.close > upper and previous.close <= upper:
            side = 1
        elif bar.close < lower and previous.close >= lower:
            side = -1
        if side and abs(bar.close-(upper if side==1 else lower)) > atr:
            return None  # Avoid chasing an extended breakout.
        if side and cfg.get('require_price_slow', False):
            slow = indicators['ema_slow'][index]
            if slow is None or not math.isfinite(slow) or side*(bar.close-slow) <= 0:
                return None
    elif family == 'dual_momentum':
        previous_atr = indicators['atr'][index-1]
        if previous_atr is None or not math.isfinite(previous_atr) or previous_atr <= 0:
            return None
        def state(end):
            price=bars[end].close
            fast=price/bars[end-24].close-1
            slow=price/bars[end-168].close-1
            threshold=indicators['atr'][end]/price
            return 1 if fast>threshold and slow>0 else -1 if fast < -threshold and slow<0 else 0
        current, old = state(index),state(index-1)
        if current and current != old:
            side=current
    elif family == 'rsi_reclaim':
        # Same fixed prefix length in screening and the production 500-bar feed.
        old=rsi([b.close for b in bars[max(0,index-140):index]])
        new=rsi([b.close for b in bars[max(0,index-139):index+1]])
        if old is not None and old<=30<new and bar.close>previous.close:
            side=1
        elif old is not None and old>=70>new and bar.close<previous.close:
            side=-1
    else:
        shock=previous.close-bars[index-2].close
        if shock <= -1.5*atr and bar.close>previous.close and bar.low>=previous.low:
            side=1
        elif shock >= 1.5*atr and bar.close<previous.close and bar.high<=previous.high:
            side=-1
    if side is None:
        return None
    return Signal(side, atr,bar.close,abs(bar.close-previous.close)/atr,bar.time_ms)
