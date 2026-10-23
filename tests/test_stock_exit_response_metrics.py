"""Deterministic causal/accounting tests; no network or trading imports."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stock_exit_response_metrics import HOUR, update_diagnostics


def entry(timestamp=HOUR + 100, boundary=HOUR, direction=1, price=100, qty=1, level=99):
    return {'entry_time_ms': timestamp, 'signal_boundary_ms': boundary, 'direction': direction,
            'entry_price': price, 'initial_stop': price - direction * 10, 'initial_qty': qty,
            'entry_trigger': {'breakout_level': level}}


def close(ent, timestamp=2 * HOUR + 100, qty=None, price=105, complete=True, net=4.8,
          reason='hourly_exit_response_giveback', sequence=2):
    return {'entry_time_ms': ent['entry_time_ms'], 'time_ms': timestamp,
            'qty': ent['initial_qty'] if qty is None else qty, 'price': price,
            'position_closed': complete, 'net_pnl': net, 'reason': reason, 'sequence': sequence}


def candle(boundary, price, high=999, low=1):
    return [boundary - HOUR, price, high, low, price, 1, boundary - 1]


def write(root, arm, name, rows, cost='normal'):
    folder = root / 'MUUSDT' / f'{arm}_{cost}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def setup(root, ent=None, cand_closes=None, base_closes=None, cost='normal'):
    ent = ent or entry()
    for arm in ('baseline', 'giveback'):
        for name in ('entries', 'fills', 'closed_trades', 'funding'):
            rows = [ent] if name == 'entries' else []
            write(root, arm, name, rows, cost)
    write(root, 'giveback', 'closed_trades', cand_closes if cand_closes is not None else [close(ent)], cost)
    write(root, 'baseline', 'closed_trades', base_closes or [], cost)
    return ent


def run(root, ent=None, bars=None, asof=5 * HOUR, state=None, cost='normal'):
    ent = ent or entry()
    state = state if state is not None else {'position': {'entry_time': ent['entry_time_ms'], 'qty': ent['initial_qty']}}
    return update_diagnostics(root, 'MUUSDT', cost, 'giveback', state, {'position': None}, bars or [], asof)


def test_matched_censored_close_path_excludes_intrabar_and_future(tmp_path):
    ent = setup(tmp_path)
    result = run(tmp_path, ent, [candle(2 * HOUR, 160), candle(3 * HOUR, 95),
                                 candle(4 * HOUR, 115), candle(6 * HOUR, 200)])
    pair = result['matched_pairs'][0]
    follow = pair['follow_up']
    assert result['status'] == 'ok'
    assert pair['status'] == 'censored' and pair['policy_early_exit'] is True
    assert follow['closed_hour_count'] == 2
    assert follow['max_original_entry_r'] == 1.5
    assert follow['max_favorable_r_from_exit'] == 1
    assert follow['max_adverse_r_from_exit'] == 1
    assert follow['reentered_breakout_side'] is True
    assert follow['missing_hour_count'] == 1
    assert follow['price_basis'].startswith('completed_hourly_close')


def test_completed_baseline_and_future_close_bar_excluded(tmp_path):
    ent = entry()
    setup(tmp_path, ent, base_closes=[close(ent, 4 * HOUR + 500, price=120, net=19.8, reason='net120_target')])
    result = run(tmp_path, ent, [candle(3 * HOUR, 110), candle(4 * HOUR, 120), candle(5 * HOUR, 150)], state={'position': None})
    pair = result['matched_pairs'][0]
    assert pair['status'] == 'complete'
    assert pair['baseline_exit']['net_pnl'] == 19.8
    assert pair['follow_up']['end_ms'] == 4 * HOUR + 500
    assert pair['follow_up']['closed_hour_count'] == 2
    assert result['summary']['completed_followups'] == 1


def test_rolling_feed_idempotence_and_first_observation_retained(tmp_path):
    ent = setup(tmp_path)
    first = run(tmp_path, ent, [candle(3 * HOUR, 110)], asof=3 * HOUR)
    second = run(tmp_path, ent, [candle(3 * HOUR, 110), candle(4 * HOUR, 115)], asof=4 * HOUR)
    third = run(tmp_path, ent, [candle(4 * HOUR, 999)], asof=5 * HOUR)
    a = second['matched_pairs'][0]['follow_up']
    b = third['matched_pairs'][0]['follow_up']
    assert a['closed_hour_count'] == b['closed_hour_count'] == 2
    assert b['max_original_entry_r'] == 1.5
    assert b['observed_hourly_closes'][0]['first_seen_at_ms'] == first['asof_ms']
    assert any(issue['error_type'] == 'revised_close_first_observation_retained' for issue in third['source_issues'])


def test_partial_close_not_early_exit(tmp_path):
    ent = entry()
    setup(tmp_path, ent, cand_closes=[close(ent, qty=.5, complete=False, net=2)])
    pair = run(tmp_path, ent)['matched_pairs'][0]
    assert pair['candidate_early_exit'] is None
    assert pair['status'] == 'candidate_still_open'


def test_split_close_uses_last_leg_and_vwap(tmp_path):
    ent = entry()
    setup(tmp_path, ent, cand_closes=[close(ent, qty=.6, complete=False, price=105, net=2.9),
                                    close(ent, 2 * HOUR + 500, qty=.4, price=106, net=2.3, sequence=3)])
    pair = run(tmp_path, ent)['matched_pairs'][0]
    assert pair['candidate_early_exit']['time_ms'] == 2 * HOUR + 500
    assert pair['candidate_early_exit']['vwap'] == pytest.approx(105.4)
    assert pair['candidate_early_exit']['net_pnl'] == pytest.approx(5.2)


def test_false_complete_quantity_is_not_early_exit(tmp_path):
    ent = entry()
    setup(tmp_path, ent, cand_closes=[close(ent, qty=.5, complete=True)])
    assert run(tmp_path, ent)['matched_pairs'][0]['candidate_early_exit'] is None


def test_baseline_holding_must_be_proven(tmp_path):
    ent = setup(tmp_path)
    pair = run(tmp_path, ent, state={'position': None})['matched_pairs'][0]
    assert pair['status'] == 'baseline_holding_unknown'
    assert pair['follow_up'] is None


def test_same_time_and_later_candidate_exits_not_early(tmp_path):
    ent = entry()
    setup(tmp_path, ent, base_closes=[close(ent)])
    assert run(tmp_path, ent)['matched_pairs'][0]['candidate_early_exit'] is None


def test_non_policy_earlier_exit_is_labelled(tmp_path):
    ent = entry()
    setup(tmp_path, ent, cand_closes=[close(ent, reason='protective_stop')])
    pair = run(tmp_path, ent)['matched_pairs'][0]
    assert pair['candidate_early_exit'] is not None
    assert pair['policy_early_exit'] is False


def test_extra_trades_costs_and_open_position_not_mixed_into_net(tmp_path):
    ent = setup(tmp_path)
    extra = entry(3 * HOUR + 100, 3 * HOUR, -1, 110, level=112)
    still_open = entry(4 * HOUR + 100, 4 * HOUR, 1, 120, level=118)
    write(tmp_path, 'giveback', 'entries', [ent, extra, still_open])
    write(tmp_path, 'giveback', 'closed_trades', [close(ent), close(extra, 4 * HOUR, price=100, net=9.7)])
    write(tmp_path, 'giveback', 'fills', [
        {'time_ms': 2 * HOUR + 100, 'fee': .5},  # Original early exit, outside follow-up window.
        {'time_ms': extra['entry_time_ms'], 'fee': .1},
        {'time_ms': 4 * HOUR, 'fee': .2},
        {'time_ms': still_open['entry_time_ms'], 'fee': .3}])
    write(tmp_path, 'giveback', 'funding', [{'fundingTime': 4 * HOUR, 'debit': -.4}])
    activity = run(tmp_path, ent)['matched_pairs'][0]['candidate_window_activity']
    assert activity['extra_entries'] == 2 and activity['direction_switches'] == 2
    assert activity['fees_paid'] == pytest.approx(.6)
    assert activity['funding_pnl'] == .4
    assert activity['closed_net_pnl'] == 9.7
    assert len(activity['open_positions']) == 1 and activity['incomplete'] is True
    assert activity['extra_entry_notional'] == 230


def test_unmatched_or_ambiguous_entries_never_generate_improvement(tmp_path):
    ent = setup(tmp_path)
    different = entry(2 * HOUR + 1, 2 * HOUR)
    write(tmp_path, 'giveback', 'entries', [different])
    result = run(tmp_path, ent)
    assert not result['matched_pairs'] and len(result['unpaired_entries']) == 2
    write(tmp_path, 'giveback', 'entries', [ent, entry(HOUR + 200)])
    result = run(tmp_path, ent)
    assert not result['matched_pairs']
    assert {row['reason'] for row in result['unpaired_entries']} == {'ambiguous_signal_cohort'}
    assert 'total_improvement' not in result['summary']


def test_missing_cost_logs_unknown_not_zero(tmp_path):
    ent = setup(tmp_path)
    (tmp_path / 'MUUSDT/giveback_normal/funding.jsonl').unlink()
    pair = run(tmp_path, ent)['matched_pairs'][0]
    activity = pair['candidate_window_activity']
    assert activity['funding_pnl'] is None and activity['incomplete']
    assert pair['candidate_exit']['net_pnl'] is None
    assert pair['candidate_exit']['closed_journal_net_pnl'] == 4.8


def test_missing_entry_fill_cannot_claim_zero_cost(tmp_path):
    ent = setup(tmp_path)
    extra = entry(3 * HOUR + 100, 3 * HOUR)
    write(tmp_path, 'giveback', 'entries', [ent, extra])
    activity = run(tmp_path, ent)['matched_pairs'][0]['candidate_window_activity']
    assert activity['fees_paid'] is None and activity['incomplete']


def test_damaged_source_preserves_prior_trigger_without_freshening(tmp_path):
    ent = setup(tmp_path)
    old = run(tmp_path, ent, [candle(3 * HOUR, 110)])
    target = tmp_path / 'MUUSDT/giveback_normal/closed_trades.jsonl'
    target.write_text(target.read_text(encoding='utf-8') + '{partial', encoding='utf-8')
    now = run(tmp_path, ent, [candle(4 * HOUR, 120)], asof=6 * HOUR)
    pair = now['matched_pairs'][0]
    assert pair['status'] == 'source_unavailable' and pair['metrics_fresh'] is False
    assert pair['follow_up'] == old['matched_pairs'][0]['follow_up']


def test_vanished_cohort_retained_as_source_failure(tmp_path):
    ent = setup(tmp_path)
    run(tmp_path, ent)
    write(tmp_path, 'giveback', 'entries', [])
    pair = run(tmp_path, ent, asof=6 * HOUR)['matched_pairs'][0]
    assert pair['status'] == 'source_unavailable'


def test_out_of_order_asof_does_not_overwrite_or_expose_future_metrics(tmp_path):
    ent = setup(tmp_path)
    run(tmp_path, ent, [candle(4 * HOUR, 120)])
    target = tmp_path / 'paired/MUUSDT/giveback_normal.json'
    saved = target.read_bytes()
    earlier = run(tmp_path, ent, asof=3 * HOUR)
    assert earlier['status'] == 'out_of_order' and earlier['matched_pairs'] == []
    assert target.read_bytes() == saved


def test_short_direction_is_symmetric(tmp_path):
    ent = entry(direction=-1, level=101)
    setup(tmp_path, ent, cand_closes=[close(ent, price=95)])
    follow = run(tmp_path, ent, [candle(3 * HOUR, 105), candle(4 * HOUR, 85)])['matched_pairs'][0]['follow_up']
    assert follow['max_original_entry_r'] == 1.5
    assert follow['max_favorable_r_from_exit'] == 1
    assert follow['max_adverse_r_from_exit'] == 1
    assert follow['reentered_breakout_side'] is True


def test_normal_and_double_are_separate_ledgers(tmp_path):
    ent = setup(tmp_path)
    setup(tmp_path, ent, cand_closes=[close(ent, net=4.6)], cost='double')
    normal, double = run(tmp_path, ent), run(tmp_path, ent, cost='double')
    assert normal['matched_pairs'][0]['candidate_exit']['net_pnl'] == 4.8
    assert double['matched_pairs'][0]['candidate_exit']['net_pnl'] == 4.6
    assert (tmp_path / 'paired/MUUSDT/giveback_normal.json').exists()
    assert (tmp_path / 'paired/MUUSDT/giveback_double.json').exists()


def test_future_entries_and_closures_not_visible(tmp_path):
    ent = setup(tmp_path)
    future = entry(7 * HOUR + 100, 7 * HOUR)
    write(tmp_path, 'giveback', 'entries', [ent, future])
    write(tmp_path, 'giveback', 'closed_trades', [close(ent, 7 * HOUR)])
    pair = run(tmp_path, ent)['matched_pairs'][0]
    assert pair['status'] == 'candidate_still_open'


def test_invalid_hourly_alignment_not_used(tmp_path):
    ent = setup(tmp_path)
    invalid = candle(3 * HOUR, 999)
    invalid[6] -= 1
    result = run(tmp_path, ent, [invalid])
    follow = result['matched_pairs'][0]['follow_up']
    assert follow['closed_hour_count'] == 0
    assert follow['max_favorable_r_from_exit'] is None
    assert follow['reentered_breakout_side'] is None
    assert result['status'] == 'degraded'


def test_staying_on_breakout_side_is_not_a_new_crossing(tmp_path):
    ent = setup(tmp_path)
    follow = run(tmp_path, ent, [candle(3 * HOUR, 110), candle(4 * HOUR, 115)])['matched_pairs'][0]['follow_up']
    assert follow['breakout_side_observed'] is True
    assert follow['reentered_breakout_side'] is False


def test_unjournaled_live_extra_position_is_incomplete(tmp_path):
    ent = setup(tmp_path)
    result = update_diagnostics(tmp_path, 'MUUSDT', 'normal', 'giveback',
                                {'position': {'entry_time': ent['entry_time_ms'], 'qty': 1}},
                                {'position': {'entry_time': 3 * HOUR + 100, 'qty': 1}}, [], 5 * HOUR)
    activity = result['matched_pairs'][0]['candidate_window_activity']
    assert activity['untracked_open_position'] is True and activity['incomplete'] is True


def test_late_closed_funding_allocated_once_to_matched_and_extra_trade_net(tmp_path):
    ent = setup(tmp_path)
    extra = entry(3 * HOUR + 100, 3 * HOUR)
    write(tmp_path, 'giveback', 'entries', [ent, extra])
    write(tmp_path, 'giveback', 'closed_trades', [close(ent), close(extra, 4 * HOUR + 100, net=9.7)])
    write(tmp_path, 'giveback', 'fills', [{'time_ms': extra['entry_time_ms'], 'fee': .1},
                                         {'time_ms': 4 * HOUR + 100, 'fee': .2}])
    write(tmp_path, 'giveback', 'funding', [
        {'fundingTime': 2 * HOUR, 'debit': .3, 'closed_funding_debit': .3},
        {'fundingTime': 4 * HOUR, 'debit': -.6, 'closed_funding_debit': -.4}])
    pair = run(tmp_path, ent)['matched_pairs'][0]
    assert pair['candidate_exit']['closed_journal_net_pnl'] == 4.8
    assert pair['candidate_exit']['net_pnl'] == pytest.approx(4.5)
    activity = pair['candidate_window_activity']
    assert activity['closed_net_pnl'] == pytest.approx(10.1)
    assert activity['funding_pnl'] == pytest.approx(.6)
    assert activity['fees_paid'] == pytest.approx(.3)


@pytest.mark.parametrize('symbol,cost,variant', [('..', 'normal', 'giveback'), ('MUUSDT', '../x', 'giveback'),
                                             ('MUUSDT', 'normal', 'baseline')])
def test_invalid_path_identity_rejected(tmp_path, symbol, cost, variant):
    with pytest.raises(ValueError):
        update_diagnostics(tmp_path, symbol, cost, variant, {}, {}, [], 0)
