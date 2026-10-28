"""Independent entry-rule metrics must not invent same-entry pairs or usable stale data."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_early_recovery_metrics as metrics

START = 1_800_000_000_000
NOW = START + 14 * metrics.DAY_MS


def account(*, net=5.0, equity=1008.0, closed=10, winners=5, best=3.0, heartbeat=NOW):
    fees, funding = 2.0, -.5
    if closed == 0:
        net = fees = funding = 0.0
        winners, best = 0, None
    return {'start_ms': START, 'initial_balance': 1000., 'financial_integrity': True,
        'heartbeat_at_ms': heartbeat, 'status': 'healthy', 'equity': equity,
        'wallet_balance': 1000 + net, 'fees_paid': fees, 'funding_pnl': funding, 'position_qty': .1,
        'metrics': {'closed_positions': closed, 'net_wins': winners, 'net_losses': closed - winners,
            'net_flat': 0, 'closed_net_pnl_usdt': net, 'closed_gross_pnl_usdt': net + fees - funding,
            'closed_fees_usdt': fees, 'closed_funding_pnl_usdt': funding,
            'best_closed_net_pnl_usdt': best, 'closed_net_excluding_best_winner_usdt': net - (best or 0),
            'max_sampled_drawdown_pct': 2., 'open_or_partial_positions_excluded': 0,
            'unallocated_late_closed_funding_debit_usdt': 0.},
        'estimated_close': {'status': 'available', 'equity_after_estimated_close_usdt': equity - .5}}


def compare(a=None, b=None, **kwargs):
    return metrics.compare_independent_arms(a or account(), b or account(net=8, equity=1012, closed=12, best=4),
        asof_ms=kwargs.pop('asof_ms', NOW), inception_ms=kwargs.pop('inception_ms', START), **kwargs)


def accounts():
    return {symbol: {arm: account() for arm in metrics.ARMS} for symbol in metrics.SYMBOLS}


def test_different_trade_counts_and_positions_compare_account_outcomes_without_pairing():
    a, b = account(), account(net=8, equity=1012, closed=12, best=4)
    b['position_qty'] = -.3
    result = compare(a, b)
    assert result['comparison_valid']
    assert result['equity_delta_usdt'] == 4
    assert result['closed_net_delta_usdt'] == 3
    assert result['closed_net_excluding_best_winner_delta_usdt'] == 2
    assert result['closed_positions_delta'] == 2
    assert result['estimated_close_delta_usdt'] == 4
    assert not result['paired_comparison'] and not result['same_entry_required']
    assert 'paired' not in result and not result['counterfactual_pnl_simulated']


def test_estimated_close_is_optional_not_a_guarantee_or_actual_fill():
    a = account(); a['estimated_close'] = {'status': 'unavailable', 'reason': 'depth'}
    result = compare(a)
    assert result['comparison_valid'] and result['estimated_close_delta_usdt'] is None
    assert not result['estimated_close_comparison_valid']


def test_real_adverse_negative_equity_and_drawdown_above_100_are_not_hidden():
    a, b = account(), account(equity=-100)
    b['metrics']['max_sampled_drawdown_pct'] = 110
    result = compare(a, b)
    assert result['comparison_valid']
    assert result['equity_delta_usdt'] == -1108
    assert result['max_sampled_drawdown_delta_pp'] == 108


def test_finite_inputs_cannot_emit_nonfinite_difference():
    a, b = account(equity=-1e308), account(equity=1e308)
    result = compare(a, b)
    assert not result['comparison_valid']
    assert result['source_issues'] == ['non_finite_comparison_result']
    assert result['equity_delta_usdt'] is None and result['closed_net_delta_usdt'] is None


@pytest.mark.parametrize('mutation', ['quarantine', 'missing_integrity', 'wrong_epoch', 'missing_epoch',
    'wrong_capital', 'stale_heartbeat', 'future_heartbeat', 'bool_heartbeat', 'degraded',
    'nan_equity', 'infinite_funding', 'negative_fees', 'bool_qty', 'missing_metrics',
    'bad_count', 'count_identity', 'cash_identity', 'best_identity', 'unallocated_funding', 'huge_number'])
def test_unverified_or_corrupt_input_suppresses_every_financial_delta(mutation):
    b = account()
    if mutation == 'quarantine': b['financial_integrity'] = False
    elif mutation == 'missing_integrity': b.pop('financial_integrity')
    elif mutation == 'wrong_epoch': b['start_ms'] += 1
    elif mutation == 'missing_epoch': b.pop('start_ms')
    elif mutation == 'wrong_capital': b['initial_balance'] = 2000
    elif mutation == 'stale_heartbeat': b['heartbeat_at_ms'] = NOW - 180001
    elif mutation == 'future_heartbeat': b['heartbeat_at_ms'] = NOW + 1
    elif mutation == 'bool_heartbeat': b['heartbeat_at_ms'] = True
    elif mutation == 'degraded': b['status'] = 'degraded'
    elif mutation == 'nan_equity': b['equity'] = float('nan')
    elif mutation == 'infinite_funding': b['funding_pnl'] = float('inf')
    elif mutation == 'negative_fees': b['fees_paid'] = -.1
    elif mutation == 'bool_qty': b['position_qty'] = True
    elif mutation == 'missing_metrics': b['metrics'] = None
    elif mutation == 'bad_count': b['metrics']['closed_positions'] = True
    elif mutation == 'count_identity': b['metrics']['net_wins'] += 1
    elif mutation == 'cash_identity': b['metrics']['closed_net_pnl_usdt'] += 1
    elif mutation == 'best_identity': b['metrics']['closed_net_excluding_best_winner_usdt'] += 1
    elif mutation == 'huge_number': b['equity'] = 10**1000
    else: b['metrics']['unallocated_late_closed_funding_debit_usdt'] = .1
    result = compare(b=b)
    assert not result['comparison_valid']
    assert result['account_issues']['candidate']
    assert all(value is None for name, value in result.items() if '_delta_' in name or name == 'closed_positions_delta')


def test_zero_closed_positions_is_valid_but_not_ready_or_a_zero_winrate():
    result = compare(account(closed=0), account(closed=0))
    assert result['comparison_valid'] and result['closed_net_delta_usdt'] == 0
    data = accounts(); data['MUUSDT']['candidate_double'] = account(closed=0)
    review = metrics.review_readiness(data, asof_ms=NOW, inception_ms=START)
    assert review['all_accounts_valid'] and not review['ready_for_first_review']


def test_review_requires_all_twelve_independent_closed_samples_and_time_not_matched_pairs():
    data = accounts()
    review = metrics.review_readiness(data, asof_ms=NOW, inception_ms=START)
    assert review['ready_for_first_review']
    assert review['expected_account_count'] == 12
    assert review['declared_independent_initial_capital_usdt'] == 12000
    assert not review['matched_early_exit_pair_threshold_applicable']
    assert not review['readiness_is_pass'] and not review['forward_validated']
    assert not review['automatic_strategy_promotion']
    data['SNDKUSDT']['candidate_double'] = account(closed=9)
    review = metrics.review_readiness(data, asof_ms=NOW, inception_ms=START)
    assert not review['closed_trade_sample_sufficient']
    assert not review['ready_for_first_review']


def test_elapsed_time_alone_or_normal_cost_only_cannot_make_review_ready():
    data = accounts()
    data['MUUSDT'].pop('baseline_double')
    review = metrics.review_readiness(data, asof_ms=NOW, inception_ms=START)
    assert not review['all_accounts_valid'] and not review['ready_for_first_review']
    data = accounts()
    for arms in data.values():
        for a in arms.values(): a['heartbeat_at_ms'] = NOW - 1
    assert not metrics.review_readiness(data, asof_ms=NOW - 1, inception_ms=START)['ready_for_first_review']


def test_unknown_accounts_or_symbols_and_quarantine_cannot_pass_review():
    for mutation in ('symbol', 'arm', 'integrity'):
        data = accounts()
        if mutation == 'symbol': data['OTHERUSDT'] = data.pop('MUUSDT')
        elif mutation == 'arm': data['MUUSDT']['extra_normal'] = account()
        else: data['MUUSDT']['baseline_normal']['financial_integrity'] = False
        review = metrics.review_readiness(data, asof_ms=NOW, inception_ms=START)
        assert not review['ready_for_first_review']


def test_future_epoch_fails_closed_and_inputs_are_not_modified():
    a, b = account(), account()
    before = deepcopy([a, b])
    result = compare(a, b, asof_ms=START - 1)
    assert not result['comparison_valid'] and 'inception_after_review_clock' in result['source_issues']
    assert [a, b] == before


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '123', None, 10**1000])
def test_clock_types_are_strict(value):
    with pytest.raises(ValueError): compare(asof_ms=value)


@pytest.mark.parametrize('kwargs', [{'minimum_elapsed_days': 13}, {'minimum_closed_positions_each': 9},
    {'minimum_elapsed_days': True}, {'max_heartbeat_age_ms': float('inf')}])
def test_review_policy_cannot_weaken_fixed_minimums(kwargs):
    with pytest.raises(ValueError):
        metrics.review_readiness(accounts(), asof_ms=NOW, inception_ms=START, **kwargs)
