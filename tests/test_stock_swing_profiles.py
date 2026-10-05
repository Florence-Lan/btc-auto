from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stock_swing_profiles as profiles
import stock_swing_signals as baseline


TRANSITION = {
    "signal_family": "ema_transition", "ema_fast": 8, "ema_mid": 24,
    "ema_slow": 60, "warmup_bars": 60, "require_price_slow": True,
}


@pytest.mark.parametrize('direction', [1, -1])
def test_trend_pullback_is_symmetric_and_requires_a_fresh_reclaim(direction):
    cfg = {'signal_family':'trend_pullback','ema_fast':20,'ema_mid':50,
           'ema_slow':200,'warmup_bars':200}
    bars = [baseline.Candle(i * 900_000,100,102,98,100) for i in range(202)]
    indicators = {'ema_fast':[100.0]*202,'ema_mid':[100-direction]*202,
                  'ema_slow':[100-2*direction]*202,'atr':[1.0]*202}
    indicators['ema_mid'][193]=100-2*direction
    price=100+.5*direction
    bars[199]=baseline.Candle(199*900_000,price,price+.2,price-.2,price)
    signal=profiles.signal_at(bars,indicators,199,cfg)
    assert signal and signal.direction==direction
    assert signal.time_ms==bars[199].time_ms
    # A trend already above/below fast on the previous close is not another reclaim.
    bars[198]=bars[199].__class__(198*900_000,price,price+.2,price-.2,price)
    assert profiles.signal_at(bars,indicators,199,cfg) is None


def test_trend_pullback_prefix_is_unchanged_by_future_candles():
    cfg={'signal_family':'trend_pullback','ema_fast':20,'ema_mid':50,'ema_slow':200,'warmup_bars':200}
    bars=[baseline.Candle(i*900_000,100+i*.1,100+i*.1+.2,100+i*.1-.2,100+i*.1) for i in range(240)]
    for i, price in [(209,119.5),(210,120.3)]:
        bars[i]=baseline.Candle(i*900_000,price,price+.2,price-.2,price)
    ind=baseline.compute_indicators(bars,cfg)
    expected=profiles.signal_at(bars,ind,210,cfg)
    assert expected is not None
    bars[220]=baseline.Candle(220*900_000,1000,1001,999,1000)
    assert profiles.signal_at(bars,baseline.compute_indicators(bars,cfg),210,cfg)==expected


@pytest.mark.parametrize('direction', [1, -1])
def test_range_reversion_requires_reentry_and_flat_regime(direction):
    cfg={'signal_family':'range_reversion','ema_fast':20,'ema_mid':50,'ema_slow':200,'warmup_bars':200}
    bars=[baseline.Candle(i*900_000,100,100.1,99.9,100) for i in range(201)]
    for i,price in [(198,100-5*direction),(199,100-direction)]:
        bars[i]=baseline.Candle(i*900_000,price,price+.1,price-.1,price)
    ind={k:[100.0]*201 for k in ('ema_fast','ema_mid','ema_slow')};ind['atr']=[1.0]*201
    signal=profiles.signal_at(bars,ind,199,cfg)
    assert signal and signal.direction==direction
    ind['ema_slow'][193]=99
    assert profiles.signal_at(bars,ind,199,cfg) is None


def crossing_case(direction: int, length: int = 61, index: int = 59):
    bars = [baseline.Candle(i * 14_400_000, 100, 100.2, 99.8, 100) for i in range(length)]
    values = {key: [100.0] * length for key in ("ema_fast", "ema_mid", "ema_slow")}
    values["atr"] = [1.0] * length
    values["ema_fast"][index - 1] = 100 - direction
    values["ema_fast"][index] = 100 + direction
    values["ema_slow"][index] = 100 - direction * 2
    return bars, values


@pytest.mark.parametrize("direction", [1, -1])
def test_breakout_default_and_explicit_dispatch_preserve_original_signal(direction):
    bars = []
    for i in range(80):
        close = 150 + direction * i * 0.1
        bars.append(baseline.Candle(i * 14_400_000, close, close + 0.05, close - 0.05, close))
    cfg = {"ema_fast": 8, "ema_mid": 24, "ema_slow": 60, "warmup_bars": 60, "breakout_bars": 6}
    indicators = baseline.compute_indicators(bars, cfg)
    expected = baseline.signal_at(bars, indicators, 70, cfg)
    assert expected is not None and expected.direction == direction
    assert profiles.signal_at(bars, indicators, 70, cfg) == expected
    assert profiles.signal_at(bars, indicators, 70, cfg | {"signal_family": "breakout"}) == expected


@pytest.mark.parametrize("direction", [1, -1])
def test_transition_requires_fresh_cross_and_reports_signal_bar(direction):
    bars, indicators = crossing_case(direction)
    signal = profiles.signal_at(bars, indicators, 59, TRANSITION)
    assert signal is not None
    assert (signal.direction, signal.close, signal.time_ms) == (direction, 100, bars[59].time_ms)
    assert signal.atr == 1
    assert signal.strength == 1
    indicators["ema_fast"][60] = 100 + direction * 2
    indicators["ema_slow"][60] = 100 - direction * 2
    assert profiles.signal_at(bars, indicators, 60, TRANSITION) is None


@pytest.mark.parametrize("direction", [1, -1])
def test_prior_ema_equality_allows_cross_but_current_equality_does_not(direction):
    bars, indicators = crossing_case(direction)
    indicators["ema_fast"][58] = indicators["ema_mid"][58]
    assert profiles.signal_at(bars, indicators, 59, TRANSITION).direction == direction
    indicators["ema_fast"][59] = indicators["ema_mid"][59]
    assert profiles.signal_at(bars, indicators, 59, TRANSITION) is None


@pytest.mark.parametrize("direction", [1, -1])
def test_slow_gate_is_strict_optional_and_defaults_on(direction):
    bars, indicators = crossing_case(direction)
    for slow in (100, 100 + direction * 2):
        indicators["ema_slow"][59] = slow
        assert profiles.signal_at(bars, indicators, 59, TRANSITION) is None
        assert profiles.signal_at(bars, indicators, 59, {k: v for k, v in TRANSITION.items() if k != "require_price_slow"}) is None
        assert profiles.signal_at(bars, indicators, 59, TRANSITION | {"require_price_slow": False}).direction == direction


def test_transition_warmup_includes_slow_period_even_when_gate_is_off():
    bars, indicators = crossing_case(1, index=58)
    assert profiles.signal_at(bars, indicators, 58, TRANSITION | {"warmup_bars": 1, "require_price_slow": False}) is None
    bars, indicators = crossing_case(1)
    assert profiles.signal_at(bars, indicators, 59, TRANSITION) is not None
    assert profiles.signal_at(bars, indicators, 59, TRANSITION | {"warmup_bars": 61}) is None


@pytest.mark.parametrize("direction", [1, -1])
def test_transition_has_no_mid_slope_or_triple_ema_order_requirement(direction):
    bars, indicators = crossing_case(direction)
    indicators["ema_mid"][53] = 100 + direction * 10
    indicators["ema_slow"][59] = 100 + direction * 10
    # The selected slow-off profile can cross before its longer trend turns.
    signal = profiles.signal_at(bars, indicators, 59, TRANSITION | {"require_price_slow": False})
    assert signal is not None and signal.direction == direction


def test_transition_atr_fraction_cap_includes_boundary():
    bars, indicators = crossing_case(1)
    indicators["atr"][59] = 3
    assert profiles.signal_at(bars, indicators, 59, TRANSITION) is not None
    indicators["atr"][59] = 3.00001
    assert profiles.signal_at(bars, indicators, 59, TRANSITION) is None
    indicators["atr"][59] = 0
    assert profiles.signal_at(bars, indicators, 59, TRANSITION) is None


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("require_slow", [True, False])
@pytest.mark.parametrize("periods", [(8, 24, 60), (12, 36, 120)])
def test_computed_transition_is_causal_under_prefix_and_future_changes(direction, require_slow, periods):
    bars = []
    for i in range(230):
        # A short pullback after a flat lead-in yields a computed reversal after
        # the slow EMA is initialized, in either direction and with either gate.
        value = 100 + direction * (-0.08 * min(max(i - 140, 0), 10) + 0.25 * max(i - 150, 0))
        bars.append(baseline.Candle(i * 14_400_000, value, value + 0.05, value - 0.05, value))
    fast, mid, slow = periods
    cfg = TRANSITION | {"require_price_slow": require_slow, "ema_fast": fast, "ema_mid": mid,
                        "ema_slow": slow, "warmup_bars": slow}
    indicators = baseline.compute_indicators(bars, cfg)
    found = [i for i in range(slow, len(bars))
             if (signal := profiles.signal_at(bars, indicators, i, cfg)) is not None
             and signal.direction == direction]
    assert found, "The causal test must include a real computed signal"
    index = found[0]
    expected = profiles.signal_at(bars, indicators, index, cfg)
    assert expected.direction == direction
    prefix = bars[:index + 1]
    assert profiles.signal_at(prefix, baseline.compute_indicators(prefix, cfg), index, cfg) == expected
    future = prefix + [baseline.Candle(bar.time_ms, 500, 510, 490, 500) for bar in bars[index + 1:]]
    assert profiles.signal_at(future, baseline.compute_indicators(future, cfg), index, cfg) == expected


@pytest.mark.parametrize("field", ["ema_fast", "ema_mid", "ema_slow", "atr"])
def test_missing_transition_indicator_is_not_a_signal(field):
    bars, indicators = crossing_case(1)
    indicators[field][59] = None
    assert profiles.signal_at(bars, indicators, 59, TRANSITION) is None


def test_transition_rejects_unknown_family_invalid_gate_index_and_nonfinite_data():
    bars, indicators = crossing_case(1)
    with pytest.raises(ValueError, match="Unknown signal family"):
        profiles.signal_at(bars, indicators, 59, TRANSITION | {"signal_family": "typo"})
    with pytest.raises(ValueError, match="must be a boolean"):
        profiles.signal_at(bars, indicators, 59, TRANSITION | {"require_price_slow": "false"})
    with pytest.raises(IndexError):
        profiles.signal_at(bars, indicators, -1, TRANSITION)
    indicators["ema_fast"][59] = math.nan
    with pytest.raises(ValueError, match="finite"):
        profiles.signal_at(bars, indicators, 59, TRANSITION)
