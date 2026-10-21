from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_sndk_4h_gate_shadow as shadow


def test_rule_difference_and_normal_double_cost_design():
    _, rules, frozen = shadow.load_design()
    for cost in ('normal', 'double'):
        base, candidate = rules['baseline_' + cost], rules['candidate_' + cost]
        assert {k: v for k, v in base.items() if k != 'swing_confirmation_policy'} == candidate
        assert base['swing_confirmation_policy'] == shadow.paper.swing_confirmation.POLICY
        assert candidate['entry_timing_policy']['require_breakout_valid_at_fill'] is True
        assert candidate['entry_signal_validity_minutes'] == 60
        assert candidate['entry_direction'] == 'both'
        assert candidate['risk_fraction_per_trade'] == .0025
        assert candidate['target_margin_return'] == .6
    for family in ('baseline', 'candidate'):
        normal, double = rules[family + '_normal'], rules[family + '_double']
        costs = {'taker_fee_rate_assumption', 'adverse_slippage_fraction_assumption'}
        assert {k: v for k, v in normal.items() if k not in costs} == {
            k: v for k, v in double.items() if k not in costs}
        for key in costs:
            assert double[key] == 2 * normal[key]
    assert frozen['profile_sha256'] and frozen['source_sha256']
    assert 'scripts/run_parallel_simulation.py' in frozen['source_sha256']
    assert 'scripts/stock_entry_timing.py' in frozen['source_sha256']


def test_restart_preserves_balances_and_rejects_code_profile_rules_and_missing_state(tmp_path):
    spec, rules, frozen = shadow.load_design()
    start = 600 * shadow.HOUR + 10_000
    original = shadow.prepare(tmp_path, spec, rules, frozen, start)
    state_path = tmp_path / 'candidate_normal/state.json'
    state = shadow.paper.read_json(state_path)
    state.update(wallet_balance=997., realized_pnl=-3.)
    shadow.paper.write_json(state_path, state)
    assert shadow.prepare(tmp_path, spec, rules, frozen, start + shadow.HOUR, True) == original
    assert shadow.paper.read_json(state_path)['wallet_balance'] == 997.
    for key in ('source_sha256', 'profile_sha256', 'rules_sha256'):
        changed = deepcopy(frozen)
        changed[key]['changed'] = 'different'
        with pytest.raises(ValueError, match='Frozen experiment changed'):
            shadow.prepare(tmp_path, spec, rules, changed, start + shadow.HOUR)
    modified = deepcopy(rules)
    modified['candidate_normal']['risk_fraction_per_trade'] *= 2
    with pytest.raises(ValueError, match='Frozen experiment changed'):
        shadow.prepare(tmp_path, spec, modified, frozen, start)
    state_path.unlink()
    with pytest.raises(ValueError, match='missing ledger'):
        shadow.prepare(tmp_path, spec, rules, frozen, start)
    assert not state_path.exists()


def test_require_existing_and_partial_bootstrap_never_create_or_reset(tmp_path):
    spec, rules, frozen = shadow.load_design()
    with pytest.raises(ValueError, match='Existing experiment required'):
        shadow.prepare(tmp_path, spec, rules, frozen, 1, require_existing=True)
    assert not (tmp_path / 'manifest.json').exists()
    old = tmp_path / 'old-ledger.json'
    old.write_text('{"wallet_balance":987}')
    with pytest.raises(ValueError, match='Incomplete experiment directory'):
        shadow.prepare(tmp_path, spec, rules, frozen, 1)
    assert old.read_text() == '{"wallet_balance":987}'


def test_exchange_clock_uses_successful_transport_receipt_after_slow_failed_attempt(monkeypatch):
    monotonic = iter((10., 30., 31.))
    monkeypatch.setattr(shadow.time, 'monotonic', lambda: next(monotonic))
    venue = Mock()
    venue.get.return_value = {'serverTime': 100_000, '_public_clock_timing': {
        'started_monotonic': 28., 'received_monotonic': 29., 'transport': 'curl'}}
    clock, info = shadow.exchange_clock(venue)
    assert info['observed_at_ms'] == 100_000 and info['roundtrip_ms'] == 1000
    assert clock() == 102_000


@pytest.mark.parametrize('payload,outer_receipt,reason', [
    ({'serverTime': 100_000}, 16., 'roundtrip exceeded five seconds'),
    ({'serverTime': 100_000, '_public_clock_timing': {
        'started_monotonic': 10., 'received_monotonic': 11., 'transport': 'curl'}},
     27., 'Stale public exchange clock receipt')])
def test_exchange_clock_rejects_slow_or_stale_receipt(monkeypatch, payload, outer_receipt, reason):
    monotonic = iter((10., outer_receipt))
    monkeypatch.setattr(shadow.time, 'monotonic', lambda: next(monotonic))
    venue = Mock()
    venue.get.return_value = payload
    with pytest.raises(ValueError, match=reason):
        shadow.exchange_clock(venue)


def fixture(tmp_path, monkeypatch, *, before_inception=False):
    boundary = 600 * shadow.HOUR
    stamp = boundary + 60_000
    monkeypatch.setattr(shadow.paper, 'now_ms', lambda: stamp)
    monkeypatch.setattr(shadow.time, 'monotonic', lambda: stamp / 1000)
    spec, rules, frozen = shadow.load_design()
    manifest = shadow.prepare(tmp_path, spec, rules, frozen,
                              stamp if before_inception else boundary - 1)
    rows = []
    for i in range(500):
        price = 200 - i * (100 / 479) if i < 480 else 100 + (i - 479) * .65
        start = boundary - (500 - i) * shadow.HOUR
        rows.append([start, price - .2, price + .1, price - .3, price, 1000,
                     start + shadow.HOUR - 1])
    price = rows[-1][4]
    rules_payload = {'symbols': [{'symbol': 'SNDKUSDT', 'status': 'TRADING', 'filters': [
        {'filterType': 'LOT_SIZE', 'stepSize': '.01', 'minQty': '.01', 'maxQty': '1000'},
        {'filterType': 'PRICE_FILTER', 'tickSize': '.01'},
        {'filterType': 'MIN_NOTIONAL', 'notional': '5'}]}]}
    five_start = stamp // shadow.paper.FIVE_MINUTES * shadow.paper.FIVE_MINUTES - shadow.paper.FIVE_MINUTES
    payloads = {'time': {'serverTime': stamp}, 'exchangeInfo': rules_payload,
        'premiumIndex': {'time': stamp, 'markPrice': str(price), 'indexPrice': str(price),
                         'lastFundingRate': '0', 'nextFundingTime': stamp + 10_000_000},
        'depth': {'T': stamp, 'bids': [[str(price - .01), '10000']],
                  'asks': [[str(price + .01), '10000']]},
        'fundingRate': []}
    def get(endpoint, params=None):
        if endpoint == 'klines':
            if params['interval'] == '1h':
                # An unfinished enormous breakout must not reach any arm.
                return deepcopy(rows + [[boundary, price, 10000, price, 10000, 1000,
                                         boundary + shadow.HOUR - 1]])
            return [[five_start, price, price + 1, price - 1, price, 1000,
                     five_start + shadow.paper.FIVE_MINUTES - 1]]
        return deepcopy(payloads[endpoint])
    underlying = Mock()
    underlying.get.side_effect = get
    venue = shadow.SharedPublicVenue(underlying)
    workers = {arm: shadow.paper.StockAccount(tmp_path / arm / 'state.json', venue, 'SNDKUSDT', cfg)
               for arm, cfg in rules.items()}
    return workers, venue, underlying, manifest, stamp, price


def test_real_engine_shared_responses_gate_only_difference_and_actual_costs(tmp_path, monkeypatch):
    workers, venue, raw, _, stamp, price = fixture(tmp_path, monkeypatch)
    snapshots = {}
    for arm, worker in workers.items():
        original_step = worker.step
        def capture(responses, arm=arm, original_step=original_step):
            snapshots[arm] = deepcopy(responses)
            return original_step(responses)
        worker.step = capture
    accounts, clock = shadow.tick(workers, venue)
    assert clock['observed_at_ms'] == stamp
    assert all(snapshots[arm] == snapshots['baseline_normal'] for arm in shadow.ARMS)
    baseline = accounts['baseline_normal']
    candidate = accounts['candidate_normal']
    assert baseline['status'] == candidate['status'] == 'healthy'
    assert baseline['position_qty'] == 0
    assert baseline['signal_status'] == 'swing_confirmation_blocked'
    assert candidate['position_qty'] > 0
    assert candidate['stock_entry_timing']['allowed'] is True
    snapshots['candidate_normal'][0][1]['serverTime'] = 0
    assert snapshots['baseline_normal'][0][1]['serverTime'] == stamp
    assert sum(call.args[0] == 'time' for call in raw.get.call_args_list) == 1
    hourly = [call for call in raw.get.call_args_list
              if call.args[0] == 'klines' and call.args[1]['interval'] == '1h']
    assert len(hourly) == 1
    for cost in ('normal', 'double'):
        arm = 'candidate_' + cost
        state = shadow.paper.read_json(workers[arm].path)
        pos, fill = state['position'], state['fills'][0]
        cfg = state['rule']
        assert pos['signal_time'] == stamp // shadow.HOUR * shadow.HOUR
        assert fill['price'] == pytest.approx((price + .01) * (1 + cfg['adverse_slippage_fraction_assumption']))
        assert fill['fee'] == pytest.approx(fill['qty'] * fill['price'] * cfg['taker_fee_rate_assumption'])
        risk = pos['qty'] * shadow.paper.signals.per_unit_stop_risk(pos['entry'], pos['stop'],
                   cfg['taker_fee_rate_assumption'], cfg['adverse_slippage_fraction_assumption'])
        assert risk <= 1000 * .0025 + 1e-9
        assert state['mode'] == 'simulation' and state['places_orders'] is False
    assert shadow.paper.read_json(workers['candidate_double'].path)['position']['qty'] <= (
        shadow.paper.read_json(workers['candidate_normal'].path)['position']['qty'])


def test_pre_inception_candle_is_warmup_only_no_historical_signal_or_trade(tmp_path, monkeypatch):
    workers, venue, _, manifest, _, _ = fixture(tmp_path, monkeypatch, before_inception=True)
    observed = Mock(side_effect=AssertionError('Historical source evaluated'))
    monkeypatch.setattr(shadow.paper.stock_swing_profiles, 'signal_at', observed)
    monkeypatch.setattr(shadow.paper.stock_swing_profiles, 'signal_with_diagnostics', observed)
    accounts, _ = shadow.tick(workers, venue)
    observed.assert_not_called()
    assert all(a['wallet_balance'] == 1000 and a['fill_count_total'] == 0 for a in accounts.values())
    assert all(a['signal_status'] == 'waiting_for_first_forward_candle' for a in accounts.values())
    for worker in workers.values():
        state = shadow.paper.read_json(worker.path)
        assert state['signal_active_after_ms'] == manifest['started_at_ms']


def test_malformed_or_stale_closed_candle_blocks_entries(tmp_path, monkeypatch):
    workers, venue, raw, _, _, _ = fixture(tmp_path, monkeypatch)
    original = raw.get.side_effect
    def bad(endpoint, params=None):
        result = original(endpoint, params)
        if endpoint == 'klines' and params['interval'] == '1h':
            result[-2][6] += 1
        return result
    raw.get.side_effect = bad
    accounts, _ = shadow.tick(workers, venue)
    assert all(a['status'] == 'degraded' and a['position_qty'] == 0 for a in accounts.values())
    assert all('signal' in a['errors'] for a in accounts.values())


@pytest.mark.parametrize('failure', ['clock', 'signal', 'funding'])
def test_data_failure_blocks_new_entries_but_retains_protective_management(tmp_path, monkeypatch, failure):
    workers, venue, raw, _, stamp, _ = fixture(tmp_path, monkeypatch)
    worker = workers['candidate_normal']
    state = shadow.paper.read_json(worker.path)
    state['position'] = {'entry': 120., 'qty': 1., 'direction': 1, 'entry_time': stamp - 1,
        'signal_time': stamp - 1, 'stop': 118., 'initial_stop': 118., 'entry_fee': 0.,
        'margin': 12., 'funding': 0., 'atr': 1.}
    state['position_history'] = [{'time_ms': stamp - 1, 'signed_qty': 1.}]
    shadow.paper.write_json(worker.path, state)
    original = raw.get.side_effect
    def failed(endpoint, params=None):
        if (endpoint == {'clock': 'time', 'signal': 'klines', 'funding': 'fundingRate'}[failure]
                and (failure != 'signal' or params['interval'] == '1h')):
            raise RuntimeError('Controlled source failure')
        return original(endpoint, params)
    raw.get.side_effect = failed
    accounts, _ = shadow.tick(workers, venue)
    assert accounts['candidate_normal']['position_qty'] == 0
    assert accounts['candidate_normal']['fill_count_total'] == 1
    assert shadow.paper.read_json(worker.path)['fills'][0]['reason'] == 'protective_stop'
    assert all(a['status'] == 'degraded' for a in accounts.values())
    assert accounts['candidate_double']['fill_count_total'] == 0


def test_restart_clock_outage_preserves_ledgers_protects_and_recovers_next_tick(tmp_path, monkeypatch):
    workers, venue, raw, manifest, stamp, _ = fixture(tmp_path, monkeypatch)
    worker = workers['candidate_normal']
    state = shadow.paper.read_json(worker.path)
    state['position'] = {'entry': 120., 'qty': 1., 'direction': 1, 'entry_time': stamp - 1,
        'signal_time': stamp - 1, 'stop': 118., 'initial_stop': 118., 'entry_fee': 0.,
        'margin': 12., 'funding': 0., 'atr': 1.}
    state['position_history'] = [{'time_ms': stamp - 1, 'signed_qty': 1.}]
    shadow.paper.write_json(worker.path, state)
    before = {arm: account.path.read_bytes() for arm, account in workers.items()}
    original_get, original_clock = raw.get.side_effect, shadow.paper.now_ms
    unavailable = {'clock': True}
    def get(endpoint, params=None):
        if endpoint == 'time' and unavailable['clock']:
            raise shadow.paper.PublicTransportError('Both public clock transports unavailable', [])
        return original_get(endpoint, params)
    raw.get.side_effect = get
    resumed, clock = shadow.start_experiment(tmp_path, manifest['design'], manifest['rules'],
        manifest['fingerprint'], venue, require_existing=True)
    assert resumed == manifest and clock['source'] == 'unavailable'
    assert clock['error'] == 'PublicTransportError'
    assert shadow.paper.now_ms is original_clock
    assert {arm: account.path.read_bytes() for arm, account in workers.items()} == before

    snapshots = {}
    for arm, account in workers.items():
        step = account.step
        def capture(responses, arm=arm, step=step):
            snapshots[arm] = deepcopy(responses)
            return step(responses)
        account.step = capture
    degraded, clock = shadow.tick(workers, venue)
    assert clock['source'] == 'unavailable'
    assert all(account['status'] == 'degraded' and 'clock' in account['errors']
               for account in degraded.values())
    assert all(snapshot == snapshots['baseline_normal'] for snapshot in snapshots.values())
    assert degraded['candidate_normal']['position_qty'] == 0
    assert shadow.paper.read_json(worker.path)['fills'][-1]['reason'] == 'protective_stop'
    assert degraded['candidate_double']['fill_count_total'] == 0

    unavailable['clock'] = False
    recovered, clock = shadow.tick(workers, venue)
    assert clock['observed_at_ms'] == stamp
    assert all(account['status'] == 'healthy' for account in recovered.values())
    assert recovered['candidate_double']['position_qty'] > 0
    # One shared failed startup request, one failed tick, one recovered tick.
    assert sum(call.args[0] == 'time' for call in raw.get.call_args_list) == 3


def test_startup_clock_failure_cannot_create_new_inception_or_hide_corrupt_ledger(tmp_path):
    spec, rules, frozen = shadow.load_design()
    venue = Mock()
    venue.get.side_effect = shadow.paper.PublicTransportError('Public clock unavailable', [])
    with pytest.raises(shadow.paper.PublicTransportError, match='Public clock unavailable'):
        shadow.start_experiment(tmp_path, spec, rules, frozen, venue)
    assert list(tmp_path.iterdir()) == []

    shadow.prepare(tmp_path, spec, rules, frozen, 1)
    path = tmp_path / 'candidate_normal/state.json'
    path.write_text('{', encoding='utf-8')
    venue.reset_mock()
    with pytest.raises(ValueError, match='ledger invalid'):
        shadow.start_experiment(tmp_path, spec, rules, frozen, venue, require_existing=True)
    venue.get.assert_not_called()
    assert path.read_text(encoding='utf-8') == '{'


def test_restart_redacts_clock_errors_and_propagates_unexpected_faults(tmp_path):
    spec, rules, frozen = shadow.load_design()
    shadow.prepare(tmp_path, spec, rules, frozen, 1)
    venue = Mock()
    venue.get.side_effect = shadow.paper.requests.ConnectionError('https://user:secret@proxy.invalid')
    _, clock = shadow.start_experiment(tmp_path, spec, rules, frozen, venue, require_existing=True)
    assert clock['error'] == 'ConnectionError'
    assert clock['diagnostics']['category'] == 'connection'
    assert 'secret' not in json.dumps(clock) and 'proxy.invalid' not in json.dumps(clock)
    venue.get.side_effect = RuntimeError('Unexpected strategy configuration failure')
    with pytest.raises(RuntimeError, match='strategy configuration failure'):
        shadow.start_experiment(tmp_path, spec, rules, frozen, venue, require_existing=True)


def test_shared_funding_fallback_is_cached_and_returns_independent_values():
    raw = Mock()
    raw.get.return_value = [[1000, '100']]
    venue = shadow.SharedPublicVenue(raw)
    params = {'symbol': 'SNDKUSDT', 'interval': '1m', 'startTime': 1000, 'limit': 1}
    first = venue.get('markPriceKlines', params)
    first[0][1] = '0'
    assert venue.get('markPriceKlines', params) == [[1000, '100']]
    raw.get.assert_called_once()
    with pytest.raises(ValueError, match='Only public'):
        venue.get('order', {})


def test_metrics_group_exit_legs_exclude_partial_and_apply_late_funding(tmp_path):
    rows = [
        {'entry_time_ms': 10, 'time_ms': 30, 'gross_pnl': 5., 'fee': .1,
         'allocated_entry_fee': .1, 'allocated_funding_debit': .2, 'net_pnl': 4.6, 'position_closed': False},
        {'entry_time_ms': 10, 'time_ms': 40, 'gross_pnl': 3., 'fee': .1,
         'allocated_entry_fee': .1, 'allocated_funding_debit': .2, 'net_pnl': 2.6, 'position_closed': True},
        {'entry_time_ms': 50, 'time_ms': 70, 'gross_pnl': -2., 'fee': .1,
         'allocated_entry_fee': .1, 'allocated_funding_debit': 0., 'net_pnl': -2.2, 'position_closed': True},
        {'entry_time_ms': 80, 'time_ms': 90, 'gross_pnl': 1., 'fee': .1,
         'allocated_entry_fee': .1, 'allocated_funding_debit': 0., 'net_pnl': .8, 'position_closed': False}]
    for row in rows:
        shadow.paper.journal(tmp_path / 'closed_trades.jsonl', row)
    shadow.paper.journal(tmp_path / 'funding.jsonl', {'fundingTime': 35, 'closed_funding_debit': .5})
    result = shadow.closed_metrics(tmp_path, {'max_drawdown_pct': 1.})
    assert result['closed_positions'] == 2
    assert result['net_wins'] == result['net_losses'] == 1
    assert result['net_win_rate_pct'] == 50.
    assert result['closed_net_pnl_usdt'] == pytest.approx(4.5)
    assert result['closed_net_excluding_best_winner_usdt'] == pytest.approx(-2.2)
    assert result['closed_fees_usdt'] == pytest.approx(.6)
    assert result['closed_funding_pnl_usdt'] == pytest.approx(-.9)
    assert result['open_or_partial_positions_excluded'] == 1
    assert result['unallocated_late_closed_funding_debit_usdt'] == 0


def test_metrics_retain_post_fill_drawdown_after_recovery_and_restart(tmp_path):
    first = shadow.closed_metrics(tmp_path, {'start_ms': 1, 'equity': 990., 'peak_equity': 1000.,
                                            'max_drawdown_pct': 0., 'updated_at_utc': 'first'})
    assert first['max_sampled_drawdown_pct'] == pytest.approx(1.)
    second = shadow.closed_metrics(tmp_path, {'start_ms': 1, 'equity': 1010., 'peak_equity': 1010.,
                                             'max_drawdown_pct': 0., 'updated_at_utc': 'second'})
    assert second['max_sampled_drawdown_pct'] == pytest.approx(1.)
    assert second['equity_samples'] == 2
