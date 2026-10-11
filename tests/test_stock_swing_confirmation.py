"""Closed 4h admission must be causal, symmetric, and spare protective exits."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_parallel_simulation as paper
import stock_swing_confirmation as confirmation
import stock_swing_profiles as profiles
import stock_swing_signals as signals
from test_parallel_simulation import fast_stock_fixture

HOUR = confirmation.HOUR
CONFIG = {'signal_timeframe': '1h', 'signal_family': 'donchian20',
          'ema_fast': 8, 'ema_mid': 24, 'ema_slow': 60, 'warmup_bars': 60,
          'max_atr_fraction': .03, 'require_price_slow': True,
          'swing_confirmation_policy': confirmation.POLICY.copy()}


def hourly_bars(side=1, length=504, step=.02, start=0):
    return [signals.Candle((start + i) * HOUR, 100 + side*i*step,
                          100 + side*i*step + .005, 100 + side*i*step - .005,
                          100 + side*i*step, 10000) for i in range(length)]


@pytest.mark.parametrize('side', [1, -1])
def test_closed_four_hour_admission_and_signal_are_symmetric(side):
    bars = hourly_bars(side)
    outcome = confirmation.evaluate(bars, 499, CONFIG, side)
    assert outcome['allowed'] and outcome['direction'] == side
    assert outcome['confirmation_close_ms'] == outcome['signal_close_ms'] == 500*HOUR
    assert outcome['ema_spread_fraction'] > .004
    assert 'swing_4h_ema_ordering' in confirmation.evaluate(bars, 499, CONFIG, -side)['reasons']
    indicators = signals.compute_indicators(bars, CONFIG)
    signal = profiles.signal_at(bars, indicators, 499, CONFIG)
    assert signal and signal.direction == side
    # Neither future candles nor a future opposite move can change this decision.
    bars[500:] = hourly_bars(-side, 4, 3, start=500)
    assert confirmation.evaluate(bars, 499, CONFIG, side) == outcome
    assert profiles.signal_at(bars, signals.compute_indicators(bars, CONFIG), 499, CONFIG) == signal


def test_incomplete_four_hour_bucket_is_ignored_even_after_large_hourly_move():
    bars = hourly_bars(length=403)
    expected = confirmation.evaluate(bars, 399, CONFIG, 1)
    for index in range(400, 403):
        bars[index] = signals.Candle(index*HOUR, 10, 10.1, 9.9, 10, 100)
    later = confirmation.evaluate(bars, 402, CONFIG, 1)
    assert later['allowed']
    for key in ('confirmation_close_ms', 'ema_fast', 'ema_mid', 'ema_slow', 'close', 'ema_spread_fraction'):
        assert later[key] == expected[key]
    assert later['confirmation_close_ms'] == 400*HOUR
    assert later['signal_close_ms'] == 403*HOUR


def test_utc_edge_buckets_drop_without_interpolation():
    bars = hourly_bars(length=501, start=1)
    grouped = confirmation.completed_four_hour_bars(bars, 499)
    assert grouped[0].time_ms == 4*HOUR
    assert grouped[-1].time_ms == 496*HOUR
    assert len(grouped) == 124
    assert grouped[0].volume == sum(bar.volume for bar in bars[3:7])
    assert grouped[0].open == bars[3].open and grouped[0].close == bars[6].close


def test_missing_hour_alignment_and_warmup_fail_closed():
    bars = hourly_bars()
    assert confirmation.evaluate(bars, 238, CONFIG, 1)['reasons'] == ['swing_4h_warmup']
    assert confirmation.evaluate(bars, 239, CONFIG, 1)['allowed']
    del bars[101]
    assert confirmation.evaluate(bars, 499, CONFIG, 1)['reasons'] == ['swing_hourly_gap']
    misaligned = [signals.Candle(b.time_ms + 1, b.open, b.high, b.low, b.close, b.volume)
                  for b in hourly_bars()]
    assert confirmation.evaluate(misaligned, 499, CONFIG, 1)['reasons'] == ['swing_hourly_alignment']


def test_small_four_hour_move_cannot_admit_an_hourly_breakout():
    bars = hourly_bars(step=.001)
    # A narrow bar makes the 1h breakout real, but the 4h trend remains too small.
    bars = [signals.Candle(b.time_ms, b.close, b.close+.0001, b.close-.0001, b.close, b.volume)
            for b in bars]
    indicators = signals.compute_indicators(bars, CONFIG)
    assert profiles.raw_signal_at(bars, indicators, 499, CONFIG)
    assert profiles.signal_at(bars, indicators, 499, CONFIG) is None
    assert confirmation.evaluate(bars, 499, CONFIG, 1)['reasons'] == ['swing_4h_spread_too_small']


def test_price_must_remain_on_the_confirmed_slow_ema_side(monkeypatch):
    bars = hourly_bars()
    size = 125
    monkeypatch.setattr(confirmation.signals, 'compute_indicators', lambda *args: {
        'ema_fast': [120.]*size, 'ema_mid': [115.]*size, 'ema_slow': [112.]*size})
    result = confirmation.evaluate(bars, 499, CONFIG, 1)
    assert result['direction'] == 1
    assert result['reasons'] == ['swing_4h_price_slow']


@pytest.mark.parametrize('patch', [
    {'signal_timeframe': '15m'},
    {'swing_confirmation_policy': {**confirmation.POLICY, 'ema_fast': True}},
    {'swing_confirmation_policy': {**confirmation.POLICY, 'warmup_bars': 60.0}},
    {'swing_confirmation_policy': {**confirmation.POLICY, 'minimum_ema_spread_fraction': .005}},
    {'swing_confirmation_policy': {**confirmation.POLICY, 'require_price_slow': False}},
])
def test_frozen_policy_rejects_alternate_settings(patch):
    with pytest.raises(ValueError, match='frozen'):
        confirmation.validate({**CONFIG, **patch})


def test_absent_policy_retains_original_hourly_signal():
    bars = hourly_bars(step=.001)
    bars = [signals.Candle(b.time_ms, b.close, b.close+.0001, b.close-.0001, b.close, b.volume)
            for b in bars]
    original = {key: value for key, value in CONFIG.items() if key != 'swing_confirmation_policy'}
    indicators = signals.compute_indicators(bars, original)
    assert profiles.signal_at(bars, indicators, 499, original) == profiles.raw_signal_at(bars, indicators, 499, original)
    assert confirmation.evaluate([], 0, {}, None) == {'enabled': False, 'allowed': True, 'reasons': []}


def worker_fixture(tmp_path, monkeypatch, side=1, count=500):
    worker, path, clock, _ = fast_stock_fixture(tmp_path, monkeypatch)
    cfg = {**worker.config, **CONFIG}
    cfg.pop('profit_exit_policy', None)
    clock.update(now=600*HOUR + 60_000, volume=10000)
    worker = paper.StockAccount(path, worker.venue, 'MUUSDT', cfg)
    paper.write_json(path, paper.initial_stock('MUUSDT', 600*HOUR-10_000, cfg))
    original_get = worker.venue.get.side_effect
    bars = hourly_bars(side, count, start=600-count)
    price = bars[-1].close
    def get(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '1h':
            return [[b.time_ms, b.open, b.high, b.low, b.close, b.volume, b.time_ms+HOUR-1]
                    for b in bars] + [[600*HOUR, 10000, 10000, 10000, 10000, 10000, 601*HOUR-1]]
        if endpoint == 'premiumIndex':
            return {'time': clock['now'], 'markPrice': str(price), 'indexPrice': str(price),
                    'lastFundingRate': '0', 'nextFundingTime': clock['now']+HOUR}
        if endpoint == 'depth':
            return {'T': clock['now'], 'bids': [[str(price-.0001), '1000']],
                    'asks': [[str(price+.0001), '1000']]}
        return original_get(endpoint, params)
    worker.venue.get.side_effect = get
    return worker, path, clock


@pytest.mark.parametrize('side', [1, -1])
def test_paper_worker_uses_same_admission_and_excludes_open_hour(tmp_path, monkeypatch, side):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch, side)
    result = worker.step()
    assert result['position']['direction'] == side
    assert result['swing_confirmation']['allowed']
    assert result['swing_confirmation']['signal_close_ms'] == 600*HOUR
    assert result['fills'][0]['reason'] == 'fresh_closed_1h_signal'


def test_paper_rejection_persists_between_polls_and_does_not_block_stop(tmp_path, monkeypatch):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch, count=150)
    blocked = worker.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['signal_status'] == 'swing_confirmation_blocked'
    assert blocked['entry_blockers'] == ['swing_4h_warmup']
    clock['now'] += 30_000
    assert worker.step()['entry_blockers'] == ['swing_4h_warmup']
    # The same unavailable confirmation must never prevent protecting existing risk.
    state = json.loads(path.read_text())
    state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
        'entry_time': clock['now']-5000, 'signal_time': clock['now']-5000,
        'initial_stop': 98, 'stop': 104, 'entry_fee': .01, 'margin': 10, 'funding': 0}
    state['position_history'] = [{'time_ms': clock['now']-5000, 'signed_qty': 1}]
    paper.write_json(path, state)
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'


@pytest.mark.parametrize('failure', ['gap', 'outage'])
def test_hourly_source_failure_blocks_entries_but_protective_exit_still_fills(tmp_path, monkeypatch, failure):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch)
    original_get = worker.venue.get.side_effect
    def get(endpoint, params=None):
        if endpoint == 'klines' and params['interval'] == '1h':
            if failure == 'outage':
                raise RuntimeError('Hourly source unavailable')
            rows = original_get(endpoint, params)
            return rows[:101] + rows[102:]
        return original_get(endpoint, params)
    worker.venue.get.side_effect = get
    blocked = worker.step()
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['status'] == 'degraded'
    assert blocked['entry_blockers'] == ['swing_hourly_data_unavailable']
    state = json.loads(path.read_text())
    state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
        'entry_time': clock['now']-5000, 'signal_time': clock['now']-5000,
        'initial_stop': 98, 'stop': 111, 'entry_fee': .01, 'margin': 10, 'funding': 0}
    state['position_history'] = [{'time_ms': clock['now']-5000, 'signed_qty': 1}]
    paper.write_json(path, state)
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'


@pytest.mark.parametrize('forming_interval, error_key', [('1h', 'signal'), ('5m', 'five')])
def test_forming_only_refresh_cannot_fill_pending_entry_but_still_protects_holding(
        tmp_path, monkeypatch, forming_interval, error_key):
    worker, path, clock = worker_fixture(tmp_path, monkeypatch)
    clock['volume'] = 0
    waiting = worker.step()
    assert waiting['pending_signal'] and waiting['position'] is None
    assert waiting['entry_blockers'] == ['prior_5m_volume']
    original_get = worker.venue.get.side_effect
    def get(endpoint, params=None):
        rows = original_get(endpoint, params)
        if endpoint == 'klines' and params['interval'] == forming_interval:
            return rows[-1:]  # Nonempty API response containing only the forming candle.
        return rows
    worker.venue.get.side_effect = get
    clock.update(now=clock['now']+300_000, volume=10000)
    blocked = worker.step()
    assert blocked['pending_signal'] and clock['now'] < blocked['pending_signal']['expires_at_ms']
    assert blocked['position'] is None and blocked['fill_count_total'] == 0
    assert blocked['errors'][error_key] == 'Completed candle response is empty'
    assert error_key in blocked['entry_blockers']
    # The source error must block new risk while an existing stop can still use the book.
    state = json.loads(path.read_text())
    state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
        'entry_time': clock['now']-5000, 'signal_time': clock['now']-5000,
        'initial_stop': 98, 'stop': 111, 'entry_fee': .01, 'margin': 10, 'funding': 0}
    state['position_history'] = [{'time_ms': clock['now']-5000, 'signed_qty': 1}]
    paper.write_json(path, state)
    result = worker.step()
    assert result['position'] is None
    assert result['fills'][-1]['reason'] == 'protective_stop'


def test_stock_config_carries_common_swing_policy_without_changing_risk(tmp_path, monkeypatch):
    profile = json.loads((paper.ROOT/'config/stock_hourly_aligned_breakout_paper_20261007.json').read_text(encoding='utf-8'))
    base = json.loads((paper.ROOT/profile['base_config']).read_text(encoding='utf-8'))
    profile['swing_confirmation_policy'] = confirmation.POLICY.copy()
    for symbol in profile['symbol_profiles']:
        profile['symbol_profiles'][symbol]['signal_timeframe'] = '1h'
    profile['base_config'] = 'base.json'
    (tmp_path/'profile.json').write_text(json.dumps(profile), encoding='utf-8')
    (tmp_path/'base.json').write_text(json.dumps(base), encoding='utf-8')
    monkeypatch.setattr(paper, 'ROOT', tmp_path)
    for symbol in profile['symbol_profiles']:
        cfg = paper.stock_config({'stock_research_profile': 'profile.json'}, symbol)
        assert cfg['swing_confirmation_policy'] == confirmation.POLICY
        assert cfg['risk_fraction_per_trade'] == .0025
