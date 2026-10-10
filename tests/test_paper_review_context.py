"""Entry reviews must retain causal evidence and the paper admission policy."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import execution_targets
import llm_trade_gate
import portfolio_risk
from test_execution_ledger import sleeve_fixture


def report():
    component = {'strategy': 'timeseries_trend_6h', 'side': 'short',
                 'entry_time_utc': '2026-10-08T04:00:00+00:00', 'entry_price': 100}
    return {'execution_target': {'components': [component]},
            'trades': [{**component, 'signal_reason': 'timeseries_6h_momentum_short'}],
            'strategy_qualification': {'approved_for_forward_simulation': True,
                'historical_performance_required': False, 'forward_validated': False}}


def context(source, mode='simulation'):
    return llm_trade_gate.build_decision_context(source,
        {'signal_time_ms': 1, 'target_leverage': -.2}, current_leverage=0, mode=mode)


def test_execution_target_preserves_original_entry_reason_through_partial_exits():
    bars, cfg, sleeve = sleeve_fixture()
    original = copy.deepcopy(sleeve)
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        bars, [sleeve], cfg, portfolio_risk.DrawdownRiskPolicy(20, 30))
    points = list(execution_targets.target_stream(bars, [sleeve], result, cfg))
    active = [p for p in points if p['components']]
    assert active
    assert all(c['signal_reason'] == 'trend_test' for p in active for c in p['components'])
    assert sleeve == original


def test_old_report_reason_is_recovered_by_exact_entry_identity():
    source = report()
    original = copy.deepcopy(source)
    assert context(source)['components'][0]['signal_reason'] == 'timeseries_6h_momentum_short'
    assert source == original
    source['execution_target']['components'][0]['entry_time_utc'] = '2026-10-09T04:00:00+00:00'
    assert context(source)['components'][0]['signal_reason'] == ''


def test_component_reason_takes_precedence_over_legacy_fallback():
    source = report()
    source['execution_target']['components'][0]['signal_reason'] = 'entry_observed_reason'
    assert context(source)['components'][0]['signal_reason'] == 'entry_observed_reason'


def test_paper_sampling_policy_is_explicit_and_does_not_apply_to_live():
    source = report()
    paper = context(source)
    assert paper['admission_policy']['historical_performance_required'] is False
    assert paper['admission_policy']['forward_validated'] is False
    assert paper['strategy_run']['performance_scope'] == 'signal_engine_summary_not_execution_account'
    assert context(source, 'live')['admission_policy']['historical_performance_required'] is True
    source['strategy_qualification'].pop('historical_performance_required')
    assert context(source)['admission_policy']['historical_performance_required'] is True


def test_authorized_sampling_does_not_override_llm_market_risk_rejection(monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED', 'true')
    captured = []
    def reject(ctx):
        captured.append(ctx)
        return {'allow': False, 'confidence': .95, 'reason': 'Conflicting current signals',
                'risk_flags': ['direction_conflict']}
    gated, decision = llm_trade_gate.apply_llm_trade_gate(report(),
        {'signal_time_ms': 1, 'target_leverage': -.2, 'position_id': 'entry'},
        current_qty=0, equity=1000, mark_price=100, mode='simulation', decision_provider=reject)
    assert captured[0]['admission_policy']['historical_performance_required'] is False
    assert gated['target_leverage'] == 0
    assert decision['status'] == 'rejected'


def test_reviews_from_before_context_fix_are_not_reused(monkeypatch):
    monkeypatch.setenv('LLM_TRADE_GATE_ENABLED', 'true')
    previous = {'cache_version': 1, 'allow': True, 'decision_key': 'entry:short',
                'approved_target_leverage': -.2, 'reviewed_at_utc': llm_trade_gate.utc_now()}
    captured = []
    def review(ctx):
        captured.append(ctx)
        return {'allow': True, 'confidence': .95, 'reason': 'Reviewed new context', 'risk_flags': []}
    _, decision = llm_trade_gate.apply_llm_trade_gate(report(),
        {'signal_time_ms': 1, 'target_leverage': -.2, 'position_id': 'entry'},
        current_qty=0, equity=1000, mark_price=100, mode='simulation',
        previous_decision=previous, decision_provider=review)
    assert len(captured) == 1
    assert decision['cache_version'] == llm_trade_gate.DECISION_CACHE_VERSION
    assert decision['review_trigger'] == 'untrusted_approval_cache'
