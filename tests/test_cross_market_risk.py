from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cross_market_risk as risk

START = 1_790_000_000_000
NOW = START + 40 * 60_000


def snapshot(time, values=(100, 100, 100)):
    return {'observations': {name: {'value': value, 'observed_at_ms': time,
        'first_seen_at_ms': time + 1, 'available_at_ms': time + 1,
        'instrument': name, 'precision': 'millisecond'}
        for name, value in zip(risk.DEFAULT_MARKET_POLICY['required_features'], values)}}


def history_and_current(values=(100, 100, 100)):
    old = snapshot(NOW - 30 * 60_000)
    current = snapshot(NOW, values)
    history = risk.update_forward_history(old, {}, asof_ms=NOW + 1, inception_ms=START)
    history = risk.update_forward_history(current, history, asof_ms=NOW + 1, inception_ms=START)
    return history, current


def test_clear_changes_require_actual_forward_observations():
    history, current = history_and_current()
    decision = risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, history=history)
    assert decision['allowed'] and decision['status'] == 'clear'
    assert decision['features']['usdjpy']['change_pct'] == 0
    assert decision['pnl_simulated'] is False


@pytest.mark.parametrize('values,reason', [((98, 104, 100), 'energy_and_technology_shock'),
    ((98, 100, 99), 'yen_and_technology_shock'), ((104, 100, 100), 'technology_disorderly_move')])
@pytest.mark.parametrize('side', [-1, 1])
def test_shocks_are_entry_risk_hypotheses_not_direction_switches(values, reason, side):
    history, current = history_and_current(values)
    decision = risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, direction=side, history=history)
    assert not decision['allowed'] and reason in decision['reasons']
    assert decision['protective_exits_allowed']


def test_no_historical_backfill_at_inception():
    old = snapshot(START - 1)
    history = risk.update_forward_history(old, {}, asof_ms=NOW + 1, inception_ms=START)
    assert all(not rows for rows in history.values())
    decision = risk.evaluate_cross_market_risk(snapshot(NOW), {}, asof_ms=NOW + 1, history=history)
    assert not decision['allowed'] and decision['status'] == 'warming_up'


def test_repeated_observation_never_renews_first_seen():
    history, current = history_and_current()
    newer = deepcopy(current)
    for point in newer['observations'].values():
        point['first_seen_at_ms'] += 1000
        point['available_at_ms'] += 1000
    after = risk.update_forward_history(newer, history, asof_ms=NOW + 2000, inception_ms=START)
    assert after == history


def test_revised_same_timestamp_is_not_silently_used():
    history, current = history_and_current()
    current['observations']['usdjpy']['value'] = 98
    decision = risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, history=history)
    assert not decision['allowed'] and 'usdjpy:same_observation_revised' in decision['reasons']


@pytest.mark.parametrize('field', ['observed_at_ms', 'first_seen_at_ms', 'available_at_ms'])
def test_future_inputs_fail_closed(field):
    history, current = history_and_current()
    current['observations']['usdjpy'][field] = NOW + 1000
    assert not risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, history=history)['allowed']


def test_contract_roll_cannot_reuse_another_contract_history():
    history, current = history_and_current()
    current['observations']['nq_dec_2026']['instrument'] = 'new_contract'
    decision = risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, history=history)
    assert decision['status'] == 'warming_up'


def test_stale_and_date_only_quotes_do_not_pass_intraday_gate():
    history, current = history_and_current()
    assert not risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 30 * 60_000, history=history)['allowed']
    current['observations']['usdjpy']['precision'] = 'date'
    assert not risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, history=history)['allowed']


def test_cached_quote_cannot_cross_contract_review_deadline_between_fetches():
    history, current = history_and_current()
    current['observations']['nq_dec_2026']['contract_review_cutoff_at_ms'] = NOW + 1000
    assert risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1, history=history)['allowed']
    decision = risk.evaluate_cross_market_risk(current, {}, asof_ms=NOW + 1000, history=history)
    assert not decision['allowed'] and 'nq_dec_2026:contract_review_due' in decision['reasons']


def states():
    stamp = datetime.fromtimestamp(NOW / 1000, timezone.utc).isoformat()
    return {name: {'equity': 1000, 'position_qty': 0, 'position': None,
                   'checked_at_utc': stamp} for name in ['mu', 'sndk', 'skhynix']}


def test_sector_candidate_does_not_rewrite_ledgers_or_create_counterfactual_pnl():
    book = states()
    before = deepcopy(book)
    exposure = risk.portfolio_exposure(book, asof_ms=NOW)
    decision = risk.evaluate_sector_budget(exposure, account_id='mu', direction=-1, proposed_risk_usdt=2.5)
    assert decision['allowed'] and book == before and not decision['pnl_simulated']


def test_correlated_positions_consume_common_budget_and_count():
    book = states()
    book['mu'].update(position_qty=-1, last_mark_price=100, position={'direction': -1, 'qty': 1,
        'entry': 100, 'stop': 102}, rule={'taker_fee_rate_assumption': .0005,
                                       'adverse_slippage_fraction_assumption': .0002})
    exposure = risk.portfolio_exposure(book, asof_ms=NOW)
    assert exposure['same_direction']['-1']['equity_to_stop_estimate_usdt'] == pytest.approx(2.07)
    decision = risk.evaluate_sector_budget(exposure, account_id='sndk', direction=-1, proposed_risk_usdt=2.5)
    assert not decision['allowed'] and 'correlated_same_direction_position_limit' in decision['reasons']
    assert 'correlated_stop_risk_budget' in decision['reasons']
    assert not risk.evaluate_sector_budget(exposure, account_id='mu', direction=-1)['allowed']
    assert risk.evaluate_sector_budget(exposure, account_id='mu', direction=-1, action='protective_exit')['allowed']


def test_unknown_stop_or_stale_state_cannot_report_zero_risk():
    book = states()
    book['mu']['position_qty'] = 1
    exposure = risk.portfolio_exposure(book, asof_ms=NOW)
    assert exposure['status'] == 'unavailable'
    assert not risk.evaluate_sector_budget(exposure, account_id='sndk', direction=-1)['allowed']
    assert risk.portfolio_exposure(states(), asof_ms=NOW + 181000)['status'] == 'unavailable'


def test_btc_is_outside_stock_sector_hypothesis():
    assert risk.evaluate_sector_budget({}, account_id='btc', direction=-1)['status'] == 'not_applicable'


@pytest.mark.parametrize('policy', [{'stock_accounts': []}, {'maximum_same_direction_positions': True},
    {'maximum_same_direction_positions': 0}, {'maximum_state_age_seconds': 0},
    {'same_direction_stop_risk_fraction_of_largest_account': float('nan')}, {'unknown': 1}])
def test_invalid_sector_policy_cannot_silently_disable_or_loosen_hypothesis(policy):
    with pytest.raises(ValueError):
        risk.portfolio_exposure(states(), asof_ms=NOW, policy=policy)


@pytest.mark.parametrize('quantity,mark', [(0, 100), (-1, 103)])
def test_inconsistent_inventory_or_breached_stop_is_not_reported_as_small_risk(quantity, mark):
    book = states()
    book['mu'].update(position_qty=quantity, last_mark_price=mark,
        position={'direction': -1, 'qty': 1, 'entry': 100, 'stop': 102})
    assert risk.portfolio_exposure(book, asof_ms=NOW)['status'] == 'unavailable'
