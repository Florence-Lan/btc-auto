"""Financial crash boundaries and long-running fill tails must not create paper money."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_stock_exit_response_shadow as trial


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def full_history(tmp_path, count=101):
    state = trial.paper.initial_stock('MUUSDT', 1234, {})
    state['signal_active_after_ms'] = 1234
    fills = [{'sequence': i + 1, 'time_ms': 2000 + i, 'side': 'BUY' if i % 2 == 0 else 'SELL',
              'qty': 1, 'price': 100., 'fee': .1, 'reason': 'fresh_closed_1h_signal' if i % 2 == 0 else 'protective_stop'}
             for i in range(count)]
    state.update(fills=fills[-100:], fill_count_total=count, fees_paid=count * .1,
                 wallet_balance=1000 - count * .1, equity=1000 - count * .1,
                 position_qty=count % 2)
    if count % 2:
        state['position'] = {'qty': 1, 'direction': 1}
    for name, rows in {'fills.jsonl': fills, 'funding.jsonl': [], 'entries.jsonl': [],
                       'closed_trades.jsonl': [{'sequence': r['sequence'], 'gross_pnl': 0.}
                                              for r in fills if r['side'] == 'SELL']}.items():
        write_rows(tmp_path / name, rows)
    return state, fills


def test_more_than_one_hundred_fills_uses_full_journal_and_state_tail(tmp_path):
    state, _ = full_history(tmp_path)
    trial.validate_ledger(state, {}, 'MUUSDT', 1234)
    trial.validate_financial_journals(tmp_path, state)
    assert state['fill_count_total'] == 101 and len(state['fills']) == 100


@pytest.mark.parametrize('damage', ['uncommitted_fill', 'missing_early_fill', 'duplicate_settlement', 'wrong_inventory'])
def test_partial_or_corrupted_financial_write_is_detected(tmp_path, damage):
    state, fills = full_history(tmp_path)
    if damage == 'uncommitted_fill':
        write_rows(tmp_path / 'fills.jsonl', fills + [{**fills[-1], 'sequence': 102}])
    elif damage == 'missing_early_fill':
        write_rows(tmp_path / 'fills.jsonl', fills[1:])
    elif damage == 'duplicate_settlement':
        write_rows(tmp_path / 'funding.jsonl', [{'symbol': 'MUUSDT', 'fundingTime': 1, 'debit': 0}] * 2)
    else:
        state['position']['qty'] = 2
    with pytest.raises(trial.LedgerIntegrityError):
        trial.validate_financial_journals(tmp_path, state)


def test_bad_arm_is_quarantined_without_reexecution_or_profit_comparison(tmp_path, monkeypatch):
    spec, rules, fingerprint = trial.load_design()
    inception = spec['effective_from_ms'] + 1
    trial.prepare(tmp_path, spec, rules, fingerprint, inception)
    symbol = 'MUUSDT'
    class Worker:
        candle_bucket = 0
        rules = {}
        signal_bars = []
        activated_at_ms = inception
        def __init__(self, arm):
            self.path = tmp_path / symbol / arm / 'state.json'
            self.config = rules[symbol][arm]
            self.calls = 0
        def now_ms(self):
            return inception + 1000
        def market_data(self, **kwargs):
            return []
        def step(self, responses):
            self.calls += 1
            return trial.paper.read_json(self.path)
    class Venue:
        def begin_tick(self):
            pass
    workers = {a: Worker(a) for a in trial.ARMS}
    bad = workers['breakout_failure_normal']
    write_rows(bad.path.with_name('fills.jsonl'), [{'sequence': 1, 'time_ms': inception,
        'qty': 1, 'side': 'BUY', 'price': 100, 'fee': .01}])
    monkeypatch.setattr(trial.shared, 'checked_snapshot', lambda r: (r, {'source': 'unavailable'}))
    monkeypatch.setattr(trial, 'archive_context', lambda *a: 'snapshot')
    frozen = bad.path.read_bytes()
    result, comparisons, _ = trial.tick_symbol(tmp_path, symbol, workers, Venue(), 'calendar', {})
    assert bad.calls == 0 and bad.path.read_bytes() == frozen
    assert bad.path.with_name('quarantine.json').exists()
    assert result['breakout_failure_normal']['financial_integrity'] is False
    assert result['breakout_failure_normal']['metrics'] is None
    comparison = comparisons['normal']['breakout_failure']
    assert comparison['comparison_valid'] is False and comparison['equity_delta_usdt'] is None
    assert comparison['closed_net_delta_usdt'] is None
    # Even if someone repairs the file outside the process, quarantine stays explicit.
    bad.path.with_name('fills.jsonl').write_text('', encoding='utf-8')
    trial.tick_symbol(tmp_path, symbol, workers, Venue(), 'calendar', {})
    assert bad.calls == 0


def test_review_needs_completed_followups_not_just_time_and_trades(tmp_path):
    spec, rules, fp = trial.load_design()
    inception = spec['effective_from_ms'] + 1
    manifest = trial.prepare(tmp_path, spec, rules, fp, inception)
    accounts = {s: {a: {'status': 'healthy', 'financial_integrity': True,
                        'metrics': {'closed_positions': 10}} for a in trial.ARMS} for s in trial.SYMBOLS}
    comparisons = {s: {c: {v: {'paired': {'summary': {'completed_followups': 9}}}
                            for v in trial.VARIANTS[1:]} for c in trial.COSTS} for s in trial.SYMBOLS}
    manifest['started_at_ms'] = 1
    status = trial.publish(tmp_path, manifest, accounts, comparisons, {}, running=True, cycles=1, calendar_hash='x')
    assert status['review']['closed_trade_sample_sufficient'] is True
    assert status['review']['ready_for_first_review'] is False
    for cost in comparisons.values():
        for variants in cost.values():
            for value in variants.values():
                value['paired']['summary']['completed_followups'] = 10
    status = trial.publish(tmp_path, manifest, accounts, comparisons, {}, running=True, cycles=2, calendar_hash='x')
    assert status['review']['ready_for_first_review'] is True


def test_estimated_close_is_cost_aware_and_never_changes_inventory_or_cash():
    state = {'symbol': 'MUUSDT', 'wallet_balance': 999.9, 'errors': {},
             'position': {'entry': 100., 'direction': 1, 'qty': 1.}}
    frozen = deepcopy(state)
    responses = [('book', {'T': 2000, 'bids': [['103', '100']], 'asks': [['103.1', '100']]}, None)]
    contract = {'symbol': 'MUUSDT', 'filters': [{'filterType': 'LOT_SIZE', 'stepSize': '.01'}]}
    normal = {'adverse_slippage_fraction_assumption': .0002, 'taker_fee_rate_assumption': .001}
    double = {k: v * 2 for k, v in normal.items()}
    a = trial.estimated_close(state, responses, normal, 2000, contract)
    b = trial.estimated_close(state, responses, double, 2000, contract)
    assert a['status'] == b['status'] == 'available'
    assert a['equity_after_estimated_close_usdt'] > b['equity_after_estimated_close_usdt']
    assert a['equity_after_estimated_close_usdt'] < 1002.9
    assert state == frozen
    responses[0][1]['bids'] = [['103', '.1']]
    assert trial.estimated_close(state, responses, normal, 2000, contract)['equity_after_estimated_close_usdt'] is None
    responses[0][1]['T'] = 1
    assert trial.estimated_close(state, responses, normal, 90000, contract)['status'] == 'unavailable'


def test_invalid_clock_cannot_create_hourly_exit_from_local_fallback_time(tmp_path, monkeypatch):
    worker = trial.ExitResponseAccount.__new__(trial.ExitResponseAccount)
    worker.config = {'exit_response_policy': {'mode': 'hourly_exit_response_v1', 'variant': 'breakout_failure'}}
    worker.activated_at_ms = 0
    worker.signal_interval_ms = trial.HOUR
    worker.signal_bars = [[i * trial.HOUR, 100, 101, 99, 100, 1, (i + 1) * trial.HOUR - 1] for i in range(3)]
    worker.path = tmp_path / 'state.json'
    monkeypatch.setattr(worker, 'entry_metadata', lambda state: None)
    pos = {'entry': 100, 'initial_stop': 98, 'direction': 1, 'entry_time': 1,
           'entry_trigger': {'breakout_level': 101, 'direction': 1, 'boundary_ms': 0}}
    state = {'position': pos}
    assert worker.additional_exit_reason(state, pos, {'clock': {'serverTime': 1}},
                                          {'clock': 'unverified_clock'}, 3 * trial.HOUR + 1000) is None
    assert state['exit_response_status']['status'] == 'unavailable'
    assert not worker.path.with_name('exit_decisions.jsonl').exists()
