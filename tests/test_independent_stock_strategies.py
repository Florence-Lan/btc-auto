from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import backtest_stock_swing_120 as engine
import research_stock_independent_strategies as research
import run_parallel_simulation as runner
from stock_swing_signals import Signal
from test_stock_swing_backtest_5m import config, real_snapshot, candle_row, bar, source
from test_parallel_simulation import fast_stock_fixture


def test_individual_period_and_exit_override_shared_defaults(tmp_path, monkeypatch):
    original = json.loads((runner.ROOT/'config/stock_swing_per_symbol_15m_forward_20261005.json').read_text())
    original['base_config'] = str(runner.ROOT/original['base_config'])
    original['symbol_profiles']['MUUSDT'].update(signal_timeframe='5m', entry_direction='long', target_margin_return=.3)
    (tmp_path/'profile.json').write_text(json.dumps(original))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    plan = {'stock_research_profile': 'profile.json'}
    mu = runner.stock_config(plan, 'MUUSDT')
    sndk = runner.stock_config(plan, 'SNDKUSDT')
    assert mu['signal_timeframe'] == '5m' and mu['target_margin_return'] == .3
    assert sndk['signal_timeframe'] == '15m' and sndk['target_margin_return'] == 1.2
    assert mu['signal_family'] != sndk['signal_family']
    assert mu['risk_fraction_per_trade'] == sndk['risk_fraction_per_trade'] == .005


def test_short_cycle_replay_uses_closed_5m_groups_without_future_price(monkeypatch):
    snapshot = real_snapshot([candle_row(bar(t)) for t in range(0, 3*engine.HOUR, engine.HOUR)])
    cfg = config() | {'signal_timeframe': '15m', 'entry_signal_validity_minutes': 15}
    monkeypatch.setattr(engine, 'signal_at', lambda bars, indicators, i, settings:
        Signal(1, 1, bars[i].close, 1, bars[i].time_ms) if i == 0 else None)
    before = engine.prepare(snapshot, cfg)['MUUSDT']
    assert sorted(before['signals']) == [900_000, 1_200_000, 1_500_000]
    assert before['first_signal_time'] == 900_000
    for row in snapshot['symbols']['MUUSDT']['trade_5m']:
        if row[0] >= 900_000: row[1:5] = [120, 121, 119, 120]
    after = engine.prepare(snapshot, cfg)['MUUSDT']
    assert before['signals'][900_000] == after['signals'][900_000]
    assert after['closed_updates'][900_000] == before['closed_updates'][900_000]


def test_short_cycle_cache_matches_fresh_preparation(monkeypatch):
    snapshot = real_snapshot([candle_row(bar(t)) for t in range(0, 3*engine.HOUR, engine.HOUR)])
    cfg = config() | {'signal_timeframe': '15m', 'entry_signal_validity_minutes': 15, 'cooldown_signal_bars': 3}
    monkeypatch.setattr(engine, 'signal_at', lambda bars, indicators, i, settings:
        Signal(1, 1, 100, 1, bars[i].time_ms) if i == 0 else None)
    prepared = engine.prepare(snapshot, cfg)
    a = engine.simulate(snapshot, cfg, 900_000, 3*engine.HOUR, 2)
    b = engine.simulate(snapshot, cfg, 900_000, 3*engine.HOUR, 2, prepared_data=prepared)
    assert a == b


def test_stock_direction_constraint_blocks_opposite_entry(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    account.config = {**account.config, 'entry_direction': 'short'}
    clock['volume'] = 1000
    result = account.step()
    assert result['fill_count_total'] == 0
    assert result['pending_signal'] is None
    assert result['signal_status'] == 'direction_filtered'
    rows = [json.loads(line) for line in path.with_name('signals.jsonl').read_text().splitlines()]
    assert rows[-1]['raw_signal']['direction'] == 1
    assert rows[-1]['rejection'] == 'research_direction'


def summary(pnl, trades=10):
    return {'net_closed_pnl': pnl, 'initial_equity': 1000, 'closed_trades': trades,
            'max_sampled_drawdown_pct': 1, 'liquidation_stress_count': 0}


def test_validation_winner_never_changes_development_selection():
    experiments = {
        'a': {'runs': {'development_cost1': summary(20), 'development_cost2': summary(10),
                       'validation_cost1': summary(-20), 'validation_cost2': summary(-30)}},
        'b': {'runs': {'development_cost1': summary(10), 'development_cost2': summary(5),
                       'validation_cost1': summary(200), 'validation_cost2': summary(100)}},
    }
    assert research.select_on_development(experiments) == 'a'
    assert research.screen(experiments['a']['runs'], 'validation', 3)


def test_no_profitable_development_candidate_is_an_explicit_no_selection():
    experiments = {'loss': {'runs': {'development_cost1': summary(1), 'development_cost2': summary(-1)}},
                   'tiny': {'runs': {'development_cost1': summary(20, 1), 'development_cost2': summary(10, 1)}}}
    assert research.select_on_development(experiments) is None


def test_aggregate_signal_history_rejects_internal_missing_bar():
    rows = [bar(t) for t in range(0, 45*60_000, 300_000) if t != 20*60_000]
    with pytest.raises(ValueError, match='Missing complete signal candles'):
        engine.aggregate_signal_bars(rows, 900_000, 300_000)


def test_runtime_five_minute_account_uses_own_cadence_and_cooldown(tmp_path, monkeypatch):
    account, path, clock, observed = fast_stock_fixture(tmp_path, monkeypatch)
    cfg = {**account.config, 'signal_timeframe': '5m'}
    account = runner.StockAccount(path, account.venue, 'MUUSDT', cfg)
    clock['volume'] = 1000
    boundary = clock['now']//300_000*300_000
    observed.return_value = Signal(1, 1, 100, 1, boundary-300_000)
    state = account.step()
    assert state['signal_timeframe'] == '5m'
    assert state['next_signal_time_ms'] == boundary+300_000
    assert state['fills'][0]['reason'] == 'fresh_closed_5m_signal'
    assert all(call.args[1]['interval'] == '5m'
        for call in account.venue.get.call_args_list if call.args[0] == 'klines')
    state['position']['stop'] = 101
    runner.write_json(path, state)
    clock['now'] += 30_000
    after = account.step()
    assert after['position'] is None
    assert after['cooldown_until_ms'] == clock['now']+15*60_000


def test_replay_cooldown_uses_each_accounts_signal_period(monkeypatch):
    from test_stock_swing_backtest_5m import run, ENTRY, STEP
    cfg = config() | {'cooldown_signal_bars': 3, 'signal_timeframe': '15m'}
    prepared = source([bar(ENTRY+i*STEP, high=115) for i in range(11)])
    prepared.update(signal_interval_ms=900_000, profile_config=cfg)
    prepared['signals'] = {t: Signal(1, 1, 100, 1, t-900_000)
        for t in (ENTRY, ENTRY+900_000, ENTRY+2_700_000)}
    result = run(monkeypatch, {'MUUSDT': prepared}, cfg)
    assert len(result['trades']) == 2
    assert result['trades'][1]['entry_utc'] == engine.iso(ENTRY+2_700_000)
    assert result['summary']['skipped_signals']['cooldown'] == 1
