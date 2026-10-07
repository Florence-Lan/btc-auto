import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import stock_mechanism_signals as signals
import research_broad_mechanisms as research
import validate_broad_mechanism as confirmation
from stock_swing_signals import Candle


def fixture(family, side):
    closes = [100.] * 200
    if family == 'donchian20':
        closes[-1] = 102.
    elif family == 'dual_momentum':
        closes[-1] = 103.
    elif family == 'rsi_reclaim':
        closes = [200-i*.2 for i in range(200)]
        closes[-1] = closes[-2]+3
    else:
        closes[-2], closes[-1] = 96., 98.
    if side == -1:
        closes = [300-p for p in closes]
    bars = [Candle(i*3600000, p, p+.5, p-.5, p, 100) for i,p in enumerate(closes)]
    return bars, {'atr':[2.]*len(bars)}, {'signal_family':family, 'max_atr_fraction':.03}


@pytest.mark.parametrize('family', sorted(signals.FAMILIES))
@pytest.mark.parametrize('side', [-1,1])
def test_both_directions_and_no_future_observations(family, side):
    bars, indicators, cfg = fixture(family, side)
    observed = signals.signal_at(bars, indicators, len(bars)-1, cfg)
    assert observed and observed.direction == side
    index = len(bars)-1
    bars.append(Candle(bars[-1].time_ms+3600000, 10000, 10001, 9999, 10000, 100))
    indicators['atr'].append(.00001)
    assert signals.signal_at(bars, indicators, index, cfg) == observed


def test_breakout_excludes_current_high_and_rejects_chasing():
    bars, indicators, cfg = fixture('donchian20', 1)
    bars[-1] = replace(bars[-1], high=1000)
    assert signals.signal_at(bars, indicators, 199, cfg).direction == 1
    bars[-1] = replace(bars[-1], close=110)
    assert signals.signal_at(bars, indicators, 199, cfg) is None


@pytest.mark.parametrize('side', [-1,1])
def test_optional_slow_filter_blocks_only_opposite_trend_breakouts(side):
    bars, indicators, cfg = fixture('donchian20', side)
    cfg['require_price_slow'] = True
    indicators['ema_slow'] = [bars[-1].close+side]*len(bars)
    assert signals.signal_at(bars, indicators, 199, cfg) is None
    indicators['ema_slow'][-1] = bars[-1].close-side
    assert signals.signal_at(bars, indicators, 199, cfg).direction == side


def test_dual_momentum_is_a_transition_not_a_continuing_entry():
    bars, indicators, cfg = fixture('dual_momentum', 1)
    bars.append(Candle(200*3600000, 104, 104.5, 103.5, 104, 100))
    indicators['atr'].append(2.)
    assert signals.signal_at(bars, indicators, 200, cfg) is None


@pytest.mark.parametrize('family', sorted(signals.FAMILIES))
def test_invalid_atr_and_insufficient_history(family):
    bars, indicators, cfg = fixture(family, 1)
    indicators['atr'][-1] = float('nan')
    assert signals.signal_at(bars, indicators, 199, cfg) is None
    assert signals.signal_at(bars, indicators, 10, cfg) is None


def test_dual_momentum_requires_a_valid_previous_atr():
    bars, indicators, cfg = fixture('dual_momentum', 1)
    indicators['atr'][-2] = None
    assert signals.signal_at(bars, indicators, 199, cfg) is None
    with pytest.raises(ValueError):
        signals.signal_at(bars, indicators, 199., cfg)


def metrics(pnl=10):
    return {'net_closed_pnl':pnl, 'estimated_close_return_pct':pnl/10, 'closed_trades':12,
            'profit_factor':1.5, 'max_sampled_drawdown_pct':2, 'liquidation_stress_count':0}


def test_selection_uses_only_development_and_has_no_audit_fallback():
    first = {'development_cost1':metrics(30), 'development_cost2':metrics(20),
             'validation_cost2':metrics(-100)}
    second = {'development_cost1':metrics(15), 'development_cost2':metrics(10),
              'validation_cost2':metrics(200)}
    skipped = {'development_cost1':None, 'development_cost2':metrics(-1),
               'normal_cost_run_skipped_after_strict_cost_failure':True}
    assert research.select({'first':first, 'second':second, 'skipped':skipped}) == 'first'
    assert len(research.choices()) == 16


def test_production_confirmation_rejects_a_gain_from_one_winner():
    old = {'total_return_pct':-.5, 'max_drawdown_pct':1.7}
    current = {'total_return_pct':3., 'estimated_liquidated_return_pct':3.,
               'max_drawdown_pct':1., 'closed_positions':14}
    audit = {**current, 'closed_net_pnl':20, 'net_without_best_winner':-1}
    costs = lambda x: {'cost1':x, 'cost2':x}
    result = confirmation.assess(costs(old), {'recent30d':costs(current), 'validation':costs(audit)})
    assert not result['production_replay_passed']
    assert 'validation:cost2:nonpositive_without_best_winner' in result['failures']


def test_complete_trade_net_aggregates_ioc_legs_and_late_funding():
    report = {'closed_trades':[
        {'entry_time_ms':1, 'net_pnl':3., 'position_closed':False},
        {'entry_time_ms':1, 'net_pnl':4., 'position_closed':True},
        {'entry_time_ms':2, 'net_pnl':2., 'position_closed':False}],
        'final_state':{'wallet_balance':1008.5, 'position':{'entry_time':2, 'entry_fee':.2, 'funding':-.1}},
        'funding_settlements':[{'debit':-.5}]}
    stats = confirmation.closed_statistics(report)
    assert stats['close_time_journal_net_pnl'] == 7.
    assert stats['closed_net_pnl'] == pytest.approx(6.6)
    assert stats['net_without_best_winner'] == pytest.approx(-.9)
