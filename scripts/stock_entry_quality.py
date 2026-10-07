"""Closed-bar direction efficiency and volatility/cost checks for paper entries."""
import math

POLICY = {'mode': 'directional_efficiency_cost_v1', 'lookback_bars': 12,
          'minimum_directional_efficiency': .35, 'minimum_atr_roundtrip_cost_ratio': 1.0,
          'require_ema_ordering': True}


def validate(config):
    policy = config.get('entry_quality_policy')
    if policy is None:
        return
    if (policy != POLICY or type(policy.get('lookback_bars')) is not int
            or policy.get('require_ema_ordering') is not True
            or any(isinstance(policy.get(key), bool) for key in (
                'minimum_directional_efficiency', 'minimum_atr_roundtrip_cost_ratio'))):
        raise ValueError('Entry quality requires the frozen directional efficiency/cost policy')


def evaluate(candles, indicators, index, config, direction):
    validate(config)
    if config.get('entry_quality_policy') is None:
        return {'allowed': True, 'reasons': []}
    if isinstance(direction, bool) or direction not in (-1, 1) or isinstance(index, bool) or not 0 <= index < len(candles):
        raise ValueError('Invalid entry quality direction or index')
    n = POLICY['lookback_bars']
    if index < n:
        return {'allowed': False, 'reasons': ['quality_warmup']}
    closes = [b.close for b in candles[index-n:index+1]]
    path = sum(abs(b-a) for a, b in zip(closes, closes[1:]))
    efficiency = direction * (closes[-1]-closes[0]) / path if path > 0 else 0.
    fast, mid, slow, atr = [indicators[k][index] for k in ('ema_fast', 'ema_mid', 'ema_slow', 'atr')]
    if any(v is None or not math.isfinite(v) for v in (fast, mid, slow, atr)) or atr <= 0:
        return {'allowed': False, 'reasons': ['quality_indicators_unavailable']}
    fee, slip = config['taker_fee_rate_assumption'], config['adverse_slippage_fraction_assumption']
    if any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v < .1 for v in (fee, slip)):
        raise ValueError('Invalid fee or adverse slippage')
    roundtrip = 2 * (fee + slip) * closes[-1]
    ratio = atr / roundtrip if roundtrip > 0 else None
    reasons = []
    if efficiency < POLICY['minimum_directional_efficiency']:
        reasons.append('quality_choppy_or_opposite_direction')
    if direction*(fast-mid) <= 0 or direction*(mid-slow) <= 0:
        reasons.append('quality_ema_ordering')
    if ratio is not None and ratio < POLICY['minimum_atr_roundtrip_cost_ratio']:
        reasons.append('quality_volatility_below_cost')
    return {'allowed': not reasons, 'reasons': reasons, 'directional_efficiency': efficiency,
            'atr_roundtrip_cost_ratio': ratio, 'closed_signal_time_ms': candles[index].time_ms}
