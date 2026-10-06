from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import run_stock_trend_experiment as experiment
import stock_trend_exits as trend
from test_parallel_simulation import fast_stock_fixture


def bars(boundary, interval, side=1):
    result = []
    for i in range(70):
        close = 169 - i if side == 1 else 31 + i
        start = boundary - (70 - i) * interval
        result.append(paper.signals.Candle(start, close, close + 1, close - 1, close, 1000))
    return result


def config():
    return {'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
            'trend_exit_policy': trend.POLICY.copy()}


@pytest.mark.parametrize('side', [1, -1])
def test_requires_two_post_activation_closed_confirmations_for_both_directions(side):
    interval = 900_000
    boundary = 200 * interval
    position = {'direction': side, 'entry_time': boundary - 10 * interval}
    candles = bars(boundary, interval, side)
    now = boundary + 1000
    one = trend.evaluate(position, candles, config(), now, interval, boundary - interval + 1)
    assert one['confirmation_bars'] == 1
    assert not one['triggered']
    assert trend.evaluate(position, candles, config(), now + 30_000, interval,
                          boundary - interval + 1)['confirmation_bars'] == 1
    two = trend.evaluate(position, candles, config(), now, interval, boundary - 2 * interval - 1)
    assert two['triggered'] and two['confirmation_bars'] == 2
    assert not trend.evaluate(position, candles, config(), now, interval, boundary)['triggered']


def test_stale_unfinished_and_gapped_bars_never_trigger():
    interval = 900_000
    boundary = 200 * interval
    position = {'direction': 1, 'entry_time': 0}
    candles = bars(boundary, interval)
    for data, now, reason in [(candles, boundary - 1, 'unfinished_candle'),
        (candles, boundary + 2 * interval, 'stale_closed_candles'),
        (candles[:-2] + candles[-1:], boundary + 1000, 'nonconsecutive_candles')]:
        result = trend.evaluate(position, data, config(), now, interval, 0)
        assert not result['triggered'] and result['reason'] == reason


def worker_fixture(tmp_path, monkeypatch, opened=True):
    worker, path, clock, signal = fast_stock_fixture(tmp_path, monkeypatch)
    cfg = {**worker.config, 'trend_exit_policy': trend.POLICY.copy()}
    state = paper.read_json(path)
    state['rule'] = cfg
    interval = worker.signal_interval_ms
    boundary = clock['now'] // interval * interval
    if opened:
        state['position'] = {'entry': 100, 'qty': 1, 'direction': 1, 'entry_time': boundary - 10 * interval,
            'signal_time': boundary - 10 * interval, 'initial_stop': 80, 'stop': 80,
            'entry_fee': 0, 'margin': 10, 'funding': 0}
        state['position_history'] = [{'time_ms': state['position']['entry_time'], 'signed_qty': 1}]
    paper.write_json(path, state)
    original = worker.venue.get.side_effect
    def source(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '15m':
            return [[b.time_ms, b.open, b.high, b.low, b.close, b.volume, b.time_ms + interval - 1]
                    for b in bars(boundary, interval)]
        return original(endpoint, params)
    worker.venue.get.side_effect = source
    candidate = experiment.TrendStockAccount(path, worker.venue, 'MUUSDT', cfg,
        activated_at_ms=boundary - 2 * interval - 1)
    return candidate, path, clock, signal


def test_candidate_exits_control_holds_using_identical_responses(tmp_path, monkeypatch):
    candidate, path, clock, observed = worker_fixture(tmp_path, monkeypatch)
    control_path = tmp_path / 'control.json'
    control_state = paper.read_json(path)
    control_state['rule'].pop('trend_exit_policy')
    paper.write_json(control_path, control_state)
    control = paper.StockAccount(control_path, candidate.venue, 'MUUSDT', control_state['rule'])
    responses = candidate.market_data()
    new = candidate.step(responses)
    old = control.step(responses)
    assert new['position'] is None
    assert new['fills'][-1]['reason'] == 'trend_reversal_exit'
    assert new['cooldown_until_ms'] == clock['now'] + 3 * 900_000
    assert new['wallet_balance'] == pytest.approx(1000 + new['realized_pnl'] - new['fees_paid'] + new['funding_pnl'])
    assert old['position'] and old['fill_count_total'] == 0


def test_partial_trend_exit_survives_restart_and_never_reopens_on_same_signal(tmp_path, monkeypatch):
    candidate, path, clock, observed = worker_fixture(tmp_path, monkeypatch)
    original = candidate.venue.get.side_effect
    depth = {'qty': '.2'}
    def source(endpoint, params=None):
        if endpoint == 'depth':
            return {'T': clock['now'], 'bids': [['99.99', depth['qty']]], 'asks': [['100.01', '100']]}
        return original(endpoint, params)
    candidate.venue.get.side_effect = source
    first = candidate.step()
    assert first['position']['qty'] == pytest.approx(.98)
    assert first['position']['pending_exit'] == 'trend_reversal_exit'
    clock['now'] += 30_000
    depth['qty'] = '100'
    restarted = experiment.TrendStockAccount(path, candidate.venue, 'MUUSDT', candidate.config,
        activated_at_ms=candidate.activated_at_ms)
    second = restarted.step()
    assert second['position'] is None and second['fill_count_total'] == 2
    assert all(f['reason'] == 'trend_reversal_exit' for f in second['fills'])
    restarted.step()
    assert paper.read_json(path)['fill_count_total'] == 2


def test_trend_observation_outage_does_not_block_original_protective_exit(tmp_path, monkeypatch):
    candidate, path, clock, observed = worker_fixture(tmp_path, monkeypatch)
    state = paper.read_json(path)
    state['position']['stop'] = 101
    paper.write_json(path, state)
    responses = [(label, None, 'signal unavailable') if label == 'signal' else (label, payload, error)
                 for label, payload, error in candidate.market_data()]
    result = candidate.step(responses)
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'


def test_optional_trend_exits_add_no_entry_admission_gate(tmp_path, monkeypatch):
    candidate, path, clock, observed = worker_fixture(tmp_path, monkeypatch, opened=False)
    clock['volume'] = 1000
    result = candidate.step()
    assert result['position'] and result['fill_count_total'] == 1
    assert result['entry_blockers'] == []
    assert result['trend_exit_status']['status'] == 'waiting_for_closed_bar'


def test_prepare_clones_inventory_preserves_sources_and_restart_never_resets(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    source = paper.initial_stock('MUUSDT', 1000, {'entry_enabled': False, 'entry_qualification': {'approved_for_forward_simulation': False}})
    source['wallet_balance'] = 999
    source['fees_paid'] = 1
    source['position'] = {'qty': 1, 'direction': -1, 'entry': 100}
    path = tmp_path / 'source.json'
    paper.write_json(path, source)
    paper.write_json(tmp_path / 'plan.json', {'accounts': [{'account_id': 'mu', 'symbol': 'MUUSDT', 'state_path': 'source.json'}]})
    for name in ('run_stock_trend_experiment.py', 'stock_trend_exits.py', 'run_parallel_simulation.py',
                 'stock_swing_signals.py', 'stock_swing_profiles.py', 'stock_profit_exits.py'):
        p = tmp_path / 'scripts' / name
        p.parent.mkdir(exist_ok=True)
        p.write_text('source')
    profile = {'experiment_id': 'test', 'mode': 'simulation', 'places_orders': False,
               'source_plan': 'plan.json', 'stock_account_ids': ['mu'], 'trend_exit_policy': trend.POLICY.copy()}
    root = tmp_path / 'experiment'
    manifest = experiment.prepare(profile, root)
    assert paper.read_json(path) == source
    for arm in ('candidate', 'control'):
        state = paper.read_json(root / 'mu' / arm / 'state.json')
        for key in ('wallet_balance', 'fees_paid', 'position', 'fills', 'settled_funding', 'start_ms'):
            assert state[key] == source[key]
        assert state['rule']['entry_enabled'] is True
        assert state['rule']['entry_qualification']['approved_for_forward_simulation'] is True
        assert state['pending_signal'] is None
    target = root / 'mu/candidate/state.json'
    state = paper.read_json(target)
    state['wallet_balance'] = 998
    paper.write_json(target, state)
    assert experiment.prepare(profile, root) == manifest
    assert paper.read_json(target)['wallet_balance'] == 998
    target.unlink()
    with pytest.raises(RuntimeError):
        experiment.prepare(profile, root)


def test_one_arm_failure_does_not_stop_other_arm():
    candidate, control = Mock(), Mock()
    source = [('clock', {'serverTime': 1}, None)]
    candidate.market_data.return_value = source
    candidate.step.side_effect = RuntimeError('candidate outage')
    state = {'fill_count_total': 0, 'fills': []}
    control.step.return_value = state
    candidate.path = Path('/candidate/state.json')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paper, 'read_json', lambda path: state)
        patch.setattr(paper, 'journal', lambda *args: None)
        result = experiment.tick({'candidate': candidate, 'control': control}, {'fill_count_total': 0})
    assert result['candidate']['status'] == 'degraded'
    control.step.assert_called_once_with(source)


def test_existing_stop_has_priority_when_trend_exit_is_also_confirmed(tmp_path, monkeypatch):
    candidate, path, clock, observed = worker_fixture(tmp_path, monkeypatch)
    state = paper.read_json(path)
    state['position']['stop'] = 101
    paper.write_json(path, state)
    result = candidate.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'
