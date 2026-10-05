"""Replacement admission must reject one-sided or outlier-dependent gains."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_bidirectional_regimes import refinement_candidates, refinement_failures
from stock_research_entry_policy import validate


def passing_summary():
    return {'net_closed_pnl': 20, 'estimated_close_return_pct': 2,
        'closed_trades': 30, 'profit_factor': 1.3, 'max_sampled_drawdown_pct': 2,
        'liquidation_stress_count': 0, 'trade_diagnostics': {
            'by_side': {'long': {'closed_trades': 15, 'net_pnl': 10},
                        'short': {'closed_trades': 15, 'net_pnl': 10}},
            'closed_return_without_largest_winner_pct': 1}}


def test_replacement_requires_net_improvement_and_bounded_drawdown_at_each_cost():
    candidate = passing_summary()
    baseline = {**candidate, 'net_closed_pnl': 10}
    assert not refinement_failures(candidate, 20, 1.15, 5, baseline)
    baseline['net_closed_pnl'] = 21
    baseline['max_sampled_drawdown_pct'] = 1
    assert set(refinement_failures(candidate, 20, 1.15, 5, baseline)) == {
        'no_closed_net_improvement_over_control', 'drawdown_more_than_10pct_above_control'}


def test_total_profit_cannot_hide_losing_direction_or_best_trade_dependence():
    candidate = passing_summary()
    candidate['trade_diagnostics']['by_side']['short'].update(closed_trades=2, net_pnl=-1)
    candidate['trade_diagnostics']['closed_return_without_largest_winner_pct'] = -1
    assert set(refinement_failures(candidate, 20, 1.15, 5)) == {
        'short_insufficient_sample', 'short_net_pnl_nonpositive', 'nonpositive_without_largest_winner'}


def test_declared_ablation_does_not_mutate_control_or_drop_short_entries():
    control = {'signal_family': 'breakout', 'entry_direction': 'both', 'min_stop_fraction': .02,
               'target_margin_return': 1.2, 'entry_enabled': False}
    frozen = copy.deepcopy(control)
    choices = refinement_candidates(control)
    assert control == frozen
    assert len(choices) == 6
    assert all(c['entry_direction'] == 'both' for c in choices.values())
    assert choices['current_control']['min_stop_fraction'] == .02
    assert choices['tactical_active']['min_stop_fraction'] == .005
    assert choices['tactical_active']['signal_activity_lookback_bars'] == 12


@pytest.mark.parametrize('cap', [True, 0, -.001, .02, float('nan'), float('inf'), '0.001'])
def test_spread_cap_rejects_invalid_settings(cap):
    with pytest.raises(ValueError):
        validate({'max_entry_spread_fraction': cap}, 300_000)
