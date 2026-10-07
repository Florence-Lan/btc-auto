import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_entry_quality as quality
from stock_swing_signals import Candle


def case(side=1, choppy=False):
    closes = [100+side*i for i in range(14)]
    if choppy:
        closes = [100+(i%2)*10 for i in range(14)]
    bars = [Candle(i*300000, p, p+1, p-1, p, 100) for i,p in enumerate(closes)]
    indicators = {'ema_fast':[100+side*2]*14, 'ema_mid':[100+side]*14,
                  'ema_slow':[100]*14, 'atr':[1.0]*14}
    cfg = {'entry_quality_policy': quality.POLICY.copy(),
           'taker_fee_rate_assumption': .000125, 'adverse_slippage_fraction_assumption': .0002}
    return bars, indicators, cfg


@pytest.mark.parametrize('side', [-1,1])
def test_directional_trend_allowed_and_future_bars_cannot_change_decision(side):
    bars, indicators, cfg = case(side)
    result = quality.evaluate(bars, indicators, 12, cfg, side)
    assert result['allowed'] and result['directional_efficiency'] == 1
    bars[13] = Candle(13*300000,10000,10000,10000,10000,0)
    indicators['atr'][13] = .00001
    assert quality.evaluate(bars, indicators, 12, cfg, side) == result
    assert not quality.evaluate(bars, indicators, 12, cfg, -side)['allowed']


def test_chop_and_costs_are_entry_rejections_not_predictions():
    bars, indicators, cfg = case(choppy=True)
    assert 'quality_choppy_or_opposite_direction' in quality.evaluate(bars, indicators, 12, cfg, 1)['reasons']
    bars, indicators, cfg = case()
    indicators['atr'][12] = .01
    assert 'quality_volatility_below_cost' in quality.evaluate(bars, indicators, 12, cfg, 1)['reasons']


def test_disabled_policy_preserves_original_signal_permissions():
    assert quality.evaluate([], {}, 0, {}, 1) == {'allowed':True,'reasons':[]}


@pytest.mark.parametrize('patch', [{'lookback_bars':True}, {'minimum_directional_efficiency': .4},
                                 {'minimum_atr_roundtrip_cost_ratio':True}, {'mode':'other'}])
def test_invalid_frozen_policy_rejected(patch):
    with pytest.raises(ValueError):
        quality.validate({'entry_quality_policy':{**quality.POLICY,**patch}})
