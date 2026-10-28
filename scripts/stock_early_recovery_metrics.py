"""Independent-account outcomes for the new early-recovery paper experiment.

Pure functions: no files, market requests, engine imports, fills or promotion.
The caller must reconcile each account's journals and attest financial_integrity.
Different entry rules may produce different entries, holdings and trade counts;
account-level deltas are deliberately NOT matched-trade treatment effects.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')
ARMS = ('baseline_normal', 'candidate_normal', 'baseline_double', 'candidate_double')
INITIAL_BALANCE = 1000
DEFAULT_MAX_HEARTBEAT_AGE_MS = 180_000
DAY_MS = 86_400_000
MAX_TIMESTAMP_MS = 253_402_300_799_999
COMPARISON_BASIS = ('Independent cost-matched accounts over one prospective epoch; '
                    'entry times, holding periods and closed-position counts may differ.')


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _count(value):
    return type(value) is int and 0 <= value <= 2**31 - 1


def _context(asof_ms, inception_ms, max_age):
    if (type(asof_ms) is not int or not 0 < asof_ms <= MAX_TIMESTAMP_MS
            or type(inception_ms) is not int or not 0 < inception_ms <= MAX_TIMESTAMP_MS
            or type(max_age) is not int or not 0 < max_age <= 3_600_000):
        raise ValueError('Explicit positive millisecond clocks and bounded heartbeat policy required')
    return ['inception_after_review_clock'] if inception_ms > asof_ms else []


def _arm_issues(arm, *, asof_ms, inception_ms, max_age):
    if not isinstance(arm, Mapping):
        return ['account_summary_unavailable']
    issues = []
    if arm.get('financial_integrity') is not True:
        issues.append('financial_integrity_unavailable')
    if type(arm.get('start_ms')) is not int or arm['start_ms'] != inception_ms:
        issues.append('account_epoch_mismatch_or_missing')
    if not _number(arm.get('initial_balance')) or arm['initial_balance'] != INITIAL_BALANCE:
        issues.append('independent_initial_balance_mismatch_or_missing')
    heartbeat = arm.get('heartbeat_at_ms')
    if type(heartbeat) is not int or heartbeat <= 0 or not 0 <= asof_ms - heartbeat <= max_age:
        issues.append('heartbeat_missing_future_or_stale')
    if arm.get('status') != 'healthy':
        issues.append('account_not_currently_healthy')
    for name in ('equity', 'wallet_balance', 'fees_paid', 'funding_pnl', 'position_qty'):
        if not _number(arm.get(name)) or (name == 'fees_paid' and arm[name] < 0):
            issues.append('invalid_' + name)
    metrics = arm.get('metrics')
    if not isinstance(metrics, Mapping):
        return issues + ['closed_metrics_unavailable']
    for name in ('closed_positions', 'net_wins', 'net_losses', 'net_flat', 'open_or_partial_positions_excluded'):
        if not _count(metrics.get(name)):
            issues.append('invalid_metric_' + name)
    for name in ('closed_net_pnl_usdt', 'closed_gross_pnl_usdt', 'closed_fees_usdt',
                 'closed_funding_pnl_usdt', 'closed_net_excluding_best_winner_usdt', 'max_sampled_drawdown_pct',
                 'unallocated_late_closed_funding_debit_usdt'):
        if not _number(metrics.get(name)):
            issues.append('invalid_metric_' + name)
    if any(reason.startswith('invalid_metric_') for reason in issues):
        return issues
    if metrics['net_wins'] + metrics['net_losses'] + metrics['net_flat'] != metrics['closed_positions']:
        issues.append('closed_position_count_identity_failed')
    # A real negative equity can produce drawdown above 100%; do not hide that
    # valid adverse outcome by treating it as financial corruption.
    if metrics['closed_fees_usdt'] < 0 or metrics['max_sampled_drawdown_pct'] < 0:
        issues.append('closed_metric_range_invalid')
    if abs(metrics['unallocated_late_closed_funding_debit_usdt']) > 1e-9:
        issues.append('unallocated_closed_funding')
    expected_net = metrics['closed_gross_pnl_usdt'] - metrics['closed_fees_usdt'] + metrics['closed_funding_pnl_usdt']
    if not _number(expected_net) or not math.isclose(expected_net, metrics['closed_net_pnl_usdt'], rel_tol=1e-10, abs_tol=1e-7):
        issues.append('closed_net_cost_funding_identity_failed')
    best = metrics.get('best_closed_net_pnl_usdt')
    if metrics['net_wins']:
        if not _number(best) or best <= 0:
            issues.append('best_winner_missing_or_invalid')
    elif best is not None:
        issues.append('best_winner_without_net_win')
    if not issues or 'best_winner_missing_or_invalid' not in issues:
        if best is None or _number(best):
            without_best = metrics['closed_net_pnl_usdt'] - (best or 0)
            if not _number(without_best) or not math.isclose(without_best, metrics['closed_net_excluding_best_winner_usdt'], rel_tol=1e-10, abs_tol=1e-7):
                issues.append('best_winner_exclusion_identity_failed')
    if metrics['closed_positions'] == 0 and any(abs(metrics[name]) > 1e-7 for name in (
            'closed_net_pnl_usdt', 'closed_gross_pnl_usdt', 'closed_fees_usdt', 'closed_funding_pnl_usdt')):
        issues.append('nonzero_closed_result_without_closed_position')
    return issues


def _estimated_close(arm):
    estimate = arm.get('estimated_close')
    if not isinstance(estimate, Mapping) or estimate.get('status') != 'available':
        return None
    value = estimate.get('equity_after_estimated_close_usdt')
    return value if _number(value) else None


def compare_independent_arms(baseline, candidate, *, asof_ms, inception_ms,
                             max_heartbeat_age_ms=DEFAULT_MAX_HEARTBEAT_AGE_MS):
    """Compare account outcomes, never pair different-entry trades by time/order.

    Summaries require start_ms, initial_balance and a True journal-integrity
    attestation in addition to shared.tick's metrics and heartbeat fields.
    Missing, stale or quarantined inputs suppress every numeric delta.
    """
    issues = _context(asof_ms, inception_ms, max_heartbeat_age_ms)
    per_arm = {name: _arm_issues(value, asof_ms=asof_ms, inception_ms=inception_ms,
                               max_age=max_heartbeat_age_ms)
               for name, value in (('baseline', baseline), ('candidate', candidate))}
    valid = not issues and not any(per_arm.values())
    result = {'schema_version': 1, 'comparison_valid': valid,
        'status': 'available' if valid else 'unavailable', 'checked_at_ms': asof_ms,
        'inception_ms': inception_ms, 'comparison_basis': COMPARISON_BASIS,
        'paired_comparison': False, 'same_entry_required': False,
        'counterfactual_pnl_simulated': False, 'automatic_strategy_promotion': False,
        'source_issues': issues, 'account_issues': per_arm,
        'equity_delta_usdt': None, 'wallet_balance_delta_usdt': None,
        'closed_net_delta_usdt': None, 'closed_net_excluding_best_winner_delta_usdt': None,
        'fees_paid_delta_usdt': None, 'funding_pnl_delta_usdt': None,
        'closed_positions_delta': None, 'max_sampled_drawdown_delta_pp': None,
        'estimated_close_delta_usdt': None, 'estimated_close_comparison_valid': False,
        'estimated_close_note': 'Current visible-depth estimate only; not a guaranteed fill or matched-trade return.',
        'unrealized_exposure_note': 'Equity can contain different live positions; closed-net deltas exclude open or partial positions.'}
    if not valid:
        return result
    for output, field in (('equity_delta_usdt', 'equity'), ('wallet_balance_delta_usdt', 'wallet_balance'),
                          ('fees_paid_delta_usdt', 'fees_paid'), ('funding_pnl_delta_usdt', 'funding_pnl')):
        result[output] = candidate[field] - baseline[field]
    for output, field in (('closed_net_delta_usdt', 'closed_net_pnl_usdt'),
            ('closed_net_excluding_best_winner_delta_usdt', 'closed_net_excluding_best_winner_usdt'),
            ('closed_positions_delta', 'closed_positions'), ('max_sampled_drawdown_delta_pp', 'max_sampled_drawdown_pct')):
        result[output] = candidate['metrics'][field] - baseline['metrics'][field]
    left, right = _estimated_close(baseline), _estimated_close(candidate)
    if left is not None and right is not None:
        result.update(estimated_close_delta_usdt=right - left, estimated_close_comparison_valid=True)
    numeric_deltas = [name for name in result if '_delta_' in name or name == 'closed_positions_delta']
    if any(result[name] is not None and not _number(result[name]) for name in numeric_deltas):
        result.update(comparison_valid=False, status='unavailable', estimated_close_comparison_valid=False,
                      source_issues=['non_finite_comparison_result'])
        for name in numeric_deltas:
            result[name] = None
    return result


def review_readiness(by_symbol, *, asof_ms, inception_ms, minimum_elapsed_days=14,
                     minimum_closed_positions_each=10,
                     max_heartbeat_age_ms=DEFAULT_MAX_HEARTBEAT_AGE_MS):
    """Require all twelve independent accounts; readiness never means approval.

    The exit-response experiment's matched early-exit follow-up threshold is
    inapplicable when baseline and candidate have different entry rules.
    """
    issues = _context(asof_ms, inception_ms, max_heartbeat_age_ms)
    if (type(minimum_elapsed_days) is not int or not 14 <= minimum_elapsed_days <= 3650
            or type(minimum_closed_positions_each) is not int or not 10 <= minimum_closed_positions_each <= 1_000_000):
        raise ValueError('Review requirements may not be weaker than 14 days / 10 closed positions per account')
    if not isinstance(by_symbol, Mapping):
        by_symbol = {}
        issues.append('account_set_unavailable')
    if set(by_symbol) != set(SYMBOLS):
        issues.append('symbol_set_incomplete_or_unknown')
    rows = {}
    for symbol in SYMBOLS:
        arms = by_symbol.get(symbol)
        if not isinstance(arms, Mapping):
            arms = {}
        if set(arms) != set(ARMS):
            issues.append('arm_set_incomplete_or_unknown:' + symbol)
        for arm in ARMS:
            value = arms.get(arm)
            reasons = _arm_issues(value, asof_ms=asof_ms, inception_ms=inception_ms,
                                  max_age=max_heartbeat_age_ms)
            count = value['metrics']['closed_positions'] if not reasons else None
            rows[symbol + '/' + arm] = {'valid': not reasons, 'reasons': reasons,
                'closed_positions': count,
                'sample_sufficient': not reasons and count >= minimum_closed_positions_each}
    elapsed = max(0, asof_ms - inception_ms) / DAY_MS
    valid = not issues and all(row['valid'] for row in rows.values())
    sample_sufficient = valid and all(row['sample_sufficient'] for row in rows.values())
    return {'schema_version': 1, 'elapsed_days': elapsed,
        'minimum_elapsed_days': minimum_elapsed_days,
        'minimum_closed_positions_each': minimum_closed_positions_each,
        'expected_account_count': 12, 'declared_independent_initial_capital_usdt': 12000,
        'all_accounts_valid': valid, 'closed_trade_sample_sufficient': sample_sufficient,
        'ready_for_first_review': valid and sample_sufficient and elapsed >= minimum_elapsed_days,
        'account_checks': rows, 'source_issues': issues,
        'paired_comparison': False, 'matched_early_exit_pair_threshold_applicable': False,
        'readiness_is_pass': False, 'forward_validated': False, 'statistical_significance_assessed': False,
        'automatic_strategy_promotion': False,
        'interpretation': 'Review readiness is not improvement. Compare cost-matched independent account outcomes, exposure, churn and drawdown; never force same-entry pairs or auto-promote.'}
