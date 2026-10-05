from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import execution_entry_gate as gate
import research_bidirectional_regimes as research


@pytest.mark.parametrize('qualification', [False, {}, {'approved_for_forward_simulation':False}])
def test_unqualified_report_blocks_new_exposure_but_allows_reduction(qualification):
    decision=gate.decision_at({'strategy_qualification':qualification},123,'short')
    assert not decision['allowed']
    assert gate.constrain_quantity(-1,0,decision)==0
    assert gate.constrain_quantity(2,1,decision)==1
    assert gate.constrain_quantity(-1,1,decision)==0
    assert gate.constrain_quantity(.5,1,decision)==.5
    assert gate.constrain_quantity(0,1,decision)==0


def test_qualification_approval_keeps_other_execution_restrictions():
    report={'strategy_qualification':{'approved_for_forward_simulation':True},'event_overlay':{}}
    assert not gate.decision_at(report,123,'long')['allowed']


def test_profitability_screen_rejects_tiny_samples_and_negative_net_cost_results():
    summary={'net_closed_pnl':1,'estimated_close_return_pct':.1,'closed_trades':2,
             'profit_factor':1.05,'max_sampled_drawdown_pct':1,'liquidation_stress_count':0}
    assert research.failures(summary,20,1.15)==['insufficient_closed_trades','profit_factor_below_gate']
    summary.update(net_closed_pnl=-1,estimated_close_return_pct=-.1,closed_trades=25,profit_factor=.9)
    assert 'closed_net_pnl_nonpositive' in research.failures(summary,20,1.15)


def test_declared_alternatives_all_allow_both_sides():
    choices=research.candidates(None)
    assert len(choices)==6
    assert all(c['entry_direction']=='both' for c in choices.values())
