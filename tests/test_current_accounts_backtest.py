import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import backtest_current_accounts as replay


def test_stock_adapter_is_causal_and_uses_current_partial_exit(tmp_path, monkeypatch):
    start = 100 * replay.DAY
    step = replay.STEP
    plan = json.loads((replay.ROOT / 'config/parallel_simulation_plan_20261005.json').read_text(encoding='utf-8'))
    cfg = replay.paper.stock_config(plan, 'MUUSDT')
    cfg.update(signal_timeframe='5m', warmup_bars=60)
    def row(t, p):
        return [t, str(p), str(p + 1), str(p - 1), str(p), '10000', t + step - 1]
    bars = [row(t, 100) for t in range(start - 60 * step, start, step)]
    bars += [row(start + i * step, p) for i, p in enumerate((100, 103, 105, 99))]
    # Final close values are not available during a bar-opening decision.
    marks = [r[:] for r in bars]
    marks[-1][4] = '99.5'
    source = {'symbol': 'MUUSDT', 'base': 'unused', 'trade': bars, 'mark': marks,
        'index': bars, 'gaps': {}, 'rules': {'symbol': 'MUUSDT', 'status': 'TRADING',
        'filters': [{'filterType': 'LOT_SIZE', 'stepSize': '.01', 'minQty': '.01', 'maxQty': '1000'},
                    {'filterType': 'PRICE_FILTER', 'tickSize': '.01'},
                    {'filterType': 'MIN_NOTIONAL', 'notional': '5'}]},
        'funding': [{'fundingTime': start - step, 'fundingRate': '0', 'markPrice': '100'},
                    {'fundingTime': start + 2 * step, 'fundingRate': '.001', 'markPrice': '100'}]}
    monkeypatch.setattr(replay.paper.stock_swing_profiles, 'signal_at',
        lambda candles, indicators, index, config: replay.paper.signals.Signal(1, 1, 100, 1, start - step)
        if candles[-1].time_ms == start - step else None)
    original_step = replay.paper.StockAccount.step
    captured = []
    def checked_step(worker, responses):
        data = {key: value for key, value, error in responses}
        now = data['clock']['serverTime']
        assert all(int(r[6]) < now for r in data['signal'])
        assert all(int(r[6]) < now for r in data['five'])
        assert all(int(e['fundingTime']) <= now for e in data['funding'])
        assert float(data['book']['bids'][0][1]) == float(data['five'][-1][5])
        assert float(data['mark']['markPrice']) == float(source['mark'][60 + len(captured)][1])
        captured.append(now)
        return original_step(worker, responses)
    monkeypatch.setattr(replay.paper.StockAccount, 'step', checked_step)
    original_write, original_clock = replay.paper.write_json, replay.paper.now_ms
    result = replay.stock_replay(source, cfg, start, start + 4 * step, 1, tmp_path)
    artifact = json.loads((tmp_path / 'MUUSDT_cost1.json').read_text(encoding='utf-8'))
    assert captured == [start + i * step for i in range(4)]
    assert any(t['reason'] == 'one_r_partial' for t in artifact['closed_trades'])
    assert artifact['summary']['closed_positions'] == 1
    assert artifact['summary']['fills'] == 3
    assert artifact['final_state']['position'] is None
    assert artifact['summary']['funding_pnl'] < 0
    assert sum(t['net_pnl'] for t in artifact['closed_trades']) == pytest.approx(result['summary']['net_pnl'])
    assert replay.paper.write_json is original_write
    assert replay.paper.now_ms is original_clock


def test_missing_prices_are_unavailable_not_zero_return(tmp_path):
    result = replay.stock_replay({'gaps': {'mark': [123]}}, {}, 0, replay.DAY, 1, tmp_path)
    assert result['status'] == 'unavailable_incomplete_price_history'
    assert 'summary' not in result


def test_invalid_worker_config_cannot_leak_memory_handlers(tmp_path):
    before = (replay.paper.write_json, replay.paper.journal, replay.paper.now_ms,
              replay.paper.signals.compute_indicators)
    with pytest.raises(ValueError):
        replay.stock_replay({'gaps': {}, 'symbol': 'MUUSDT', 'base': 'unused'},
            {'profit_exit_policy': {'mode': 'unknown'}, 'taker_fee_rate_assumption': .001,
             'adverse_slippage_fraction_assumption': .0002}, 0, replay.DAY, 1, tmp_path)
    assert before == (replay.paper.write_json, replay.paper.journal, replay.paper.now_ms,
                      replay.paper.signals.compute_indicators)
