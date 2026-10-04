from decimal import Decimal
from types import SimpleNamespace
import json
from unittest.mock import Mock, patch

import pytest

from test_strategy_engine import candle, config
from trading_execution import SimulationAccount, LiveExecutor, simulation_funding
import trading_execution
from account_risk import constrain_target
import forward_macro
import macro_regime
import execution_targets
import portfolio_risk
import simulate_range_swing as sim


@pytest.fixture(autouse=True)
def execution_environment(monkeypatch):
    for key, value in {"LLM_TRADE_GATE_ENABLED": "false", "SIM_MAX_LEVERAGE": "2",
                       "SIM_MAX_NOTIONAL_USDT": "0", "SIM_TAKER_FEE": "0.00045",
                       "SIM_SLIPPAGE_BPS": "1"}.items():
        monkeypatch.setenv(key, value)


def account(tmp_path, equity=10000):
    a = SimulationAccount(tmp_path / "account.json")
    state = a.reset(equity, now_ms=0)
    state["observed_flat_target"] = True
    a._save(state)
    return a


def target(leverage=1, timestamp=1):
    return {"target_leverage": leverage, "signal_time_ms": timestamp,
            "position_id": "signal", "signal_price": 100000}


def funding(time=100, rate=.001, price=100000):
    return {"fundingTime": time, "fundingRate": rate, "markPrice": price}


def test_twenty_percent_account_loss_blocks_new_position_and_latches(tmp_path):
    a = account(tmp_path)
    state = a.load()
    state.update(wallet_balance=8000)
    a._save(state)
    result = a.reconcile(target(), 100000, now_ms=1)
    assert result["fill"] is None
    assert result["account_risk"]["status"] == "halted"
    state = a.load()
    state["wallet_balance"] = 10000
    a._save(state)
    assert a.reconcile(target(), 100000, now_ms=2)["fill"] is None
    a.reset(10000, now_ms=3)
    assert a.load()["risk_halt_at_utc"] is None


def test_hard_stop_flattens_and_does_not_change_signal_cursor_in_monitor(tmp_path):
    a = account(tmp_path)
    a.reconcile(target(), 100000, now_ms=1)
    state = a.load()
    state["wallet_balance"] = 8000
    a._save(state)
    result = a.observe(100000, 2)
    assert result["fill"]["side"] == "SELL"
    assert a.load()["position_qty"] == 0
    assert a.load()["last_signal_time_ms"] == 1
    assert a.load()["max_drawdown_pct"] >= 20


def test_soft_risk_reduction_and_exits_bypass_llm(tmp_path, monkeypatch):
    a = account(tmp_path)
    a.reconcile(target(), 100000, now_ms=1)
    state = a.load()
    state["wallet_balance"] = 9000
    a._save(state)
    monkeypatch.setenv("LLM_TRADE_GATE_ENABLED", "true")
    with patch("llm_trade_gate.request_llm_decision", side_effect=AssertionError("reduction must bypass LLM")) as provider:
        r = a.reconcile(target(.5), 100000, now_ms=2)
        assert r["account_risk"]["risk_multiplier"] < 1
        assert r["fill"]["side"] == "SELL"
        assert a.reconcile(target(0), 100000, now_ms=3)["fill"] is not None
        provider.assert_not_called()


@pytest.mark.parametrize("side,rate,sign", [(1,.001,-1),(-1,.001,1),(1,-.001,1),(-1,-.001,-1)])
def test_funding_direction_idempotence_and_no_future_charges(tmp_path, side, rate, sign):
    a = account(tmp_path)
    a.reconcile(target(side), 100000, now_ms=1)
    qty = a.load()["position_qty"]
    event = funding(rate=rate)
    a.observe(100000, 99, [event])
    assert a.load()["funding_pnl"] == 0
    a.observe(100000, 101, [event, event])
    assert a.load()["funding_pnl"] == pytest.approx(-qty*100000*rate)
    assert a.load()["funding_pnl"] * sign > 0
    a.observe(100000, 102, [event])
    assert len(a.load()["funding_settlements"]) == 1


def test_late_published_funding_uses_inventory_at_settlement_not_current_position(tmp_path):
    a = account(tmp_path)
    a.reconcile(target(), 100000, now_ms=1)
    qty = a.load()["position_qty"]
    a.reconcile(target(-.5), 100000, now_ms=200)
    a.observe(100000, 300, [funding(100)])
    assert a.load()["funding_pnl"] == pytest.approx(-qty*100000*.001)


def test_funding_failure_blocks_additions_but_allows_flatten(tmp_path):
    a = account(tmp_path)
    assert a.reconcile(target(), 100000, now_ms=1, funding_available=False)["fill"] is None
    a.reconcile(target(), 100000, now_ms=2)
    assert a.reconcile(target(-1), 100000, now_ms=3, funding_available=False)["fill"]["side"] == "SELL"
    assert a.load()["position_qty"] == 0


def test_migration_starts_funding_now_without_retroactive_inventory_invention(tmp_path):
    a = account(tmp_path)
    s = a.load()
    for key in ("funding_tracking_start_ms", "funding_settlements", "funding_pnl", "position_history"):
        del s[key]
    s["position_qty"] = .1
    a._save(s)
    migrated = a.load()
    SimulationAccount.settle_funding(migrated, [funding(100)], migrated["funding_tracking_start_ms"]+1)
    assert migrated["funding_pnl"] == 0
    assert migrated["position_history"][0]["signed_qty"] == .1


def test_successful_fetch_cursor_is_committed_only_with_settlement(tmp_path):
    a = account(tmp_path)
    client = Mock()
    client.funding_history.return_value = [funding()]
    events, ok = simulation_funding(client, a, 200)
    assert ok
    assert "funding_last_fetch_ms" not in a.load()
    a.observe(100000, 200, events, ok)
    assert a.load()["funding_last_fetch_ms"] == 200


def test_risk_reduction_below_min_notional_is_allowed(tmp_path):
    a = account(tmp_path)
    rules = {"step_size": Decimal(".0001"), "min_qty": Decimal(".0001"), "min_notional": Decimal("50")}
    a.reconcile(target(), 100000, rules, now_ms=1)
    # Reduce by one lot, worth only 10 USDT.
    equity = a.snapshot(100000)["account"]["margin_balance"]
    qty = a.load()["position_qty"] - .0001
    r = a.reconcile(target(qty*100000/equity), 100000, rules, now_ms=2)
    assert r["fill"] is not None
    assert r["fill"]["quantity"] == pytest.approx(.0001)


def test_quantity_grid_survives_many_one_lot_rebalances(tmp_path):
    a = account(tmp_path)
    # Use exact grid targets directly to isolate accounting from leverage sizing.
    state = a.load()
    from trading_execution import DEFAULT_SIMULATION_RULES
    for index in range(30):
        requested = .023 if index%2 == 0 else .022
        fill = a._fill_to(state,requested,100000,index+1,index+1,DEFAULT_SIMULATION_RULES)
        assert fill is not None
        assert state['position_qty'] == requested
        if index:
            assert fill['quantity'] == .001


def test_live_account_risk_halt_only_sends_reduce_only_close(tmp_path):
    client = Mock()
    client.validate_live_ready.return_value = {"leverage": 2, "max_notional_usdt": 20000}
    held = {"account": {"wallet_balance": 8000, "margin_balance": 8000},
            "positions": [{"signed_quantity": .1}]}
    flat = {"account": {"wallet_balance": 8000, "margin_balance": 8000}, "positions": []}
    client.account_snapshot.side_effect = [held, flat]
    client.quantize_quantity.side_effect = lambda q,s: round(abs(q),3)
    client.market_order.return_value = {"status": "FILLED"}
    from trading_execution import write_json
    p = tmp_path / "live.json"
    write_json(p, {"peak_equity": 10000})
    result = LiveExecutor(client,p).reconcile(target(),100000)
    assert result["account_risk"]["status"] == "halted"
    client.market_order.assert_called_once()
    assert client.market_order.call_args.kwargs["reduce_only"] is True


def test_target_stream_uses_only_observable_inventory_and_preserves_terminal_position():
    cfg = config(timeseries_timeframe="1h", timeseries_fast_ema=2, timeseries_slow_ema=4,
                 timeseries_vol_lookback_bars=4, timeseries_min_ema_spread_pct=.005,
                 max_drawdown_stop_pct=0)
    prices = [100]*6+[107,110,112,114]
    hours = [candle(i,p,p+1,p-1,p,3600000) for i,p in enumerate(prices)]
    base = [candle(i,prices[i//12],prices[i//12]+1,prices[i//12]-1,prices[i//12]) for i in range(120)]
    sleeve = sim.simulate_timeseries_trend(hours,cfg,4*3600000)
    from paper_trade_frozen_portfolio import annotate_open_position_fractions
    annotate_open_position_fractions([sleeve])
    execution_targets.prepare_sleeves([sleeve])
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        base,[sleeve],cfg,portfolio_risk.DrawdownRiskPolicy(),4*3600000,include_execution_target=True)
    stream = list(execution_targets.target_stream(base,[sleeve],result,cfg))
    assert stream[-1]["signed_qty"] > 0
    assert stream[-1]["equity"] > result["summary"]["final_equity"]
    first_entry = sim._utc_ms(sleeve["trades"][0]["entry_time_utc"])
    assert all(p["signed_qty"] == 0 for p in stream if p["available_time_ms"] < first_entry)


def test_target_prefix_does_not_change_when_future_prices_or_terminal_fills_change():
    from dataclasses import replace
    cfg = config(timeseries_timeframe='1h',timeseries_fast_ema=2,timeseries_slow_ema=4,
                 timeseries_vol_lookback_bars=4,timeseries_min_ema_spread_pct=.005,max_drawdown_stop_pct=0)
    prices = [100]*6+[107,110,112,114]
    def run(values):
        hours = [candle(i,p,p+1,p-1,p,3600000) for i,p in enumerate(values)]
        base = [candle(i,values[i//12],values[i//12]+1,values[i//12]-1,values[i//12]) for i in range(len(values)*12)]
        sleeve = sim.simulate_timeseries_trend(hours,cfg,4*3600000)
        execution_targets.prepare_sleeves([sleeve])
        result = portfolio_risk.combine_sleeves_with_drawdown_policy(base,[sleeve],cfg,portfolio_risk.DrawdownRiskPolicy(),4*3600000)
        return list(execution_targets.target_stream(base,[sleeve],result,cfg))
    short, full, changed = run(prices[:8]),run(prices),run(prices[:8]+[80,75])
    assert len(short) > 0
    for a,b,c in zip(short,full,changed):
        assert a['signed_qty'] == pytest.approx(b['signed_qty'])
        assert a['signed_qty'] == pytest.approx(c['signed_qty'])
        assert a['equity'] == pytest.approx(b['equity'])
        assert a['position_id'] == b['position_id'] == c['position_id']


def test_terminal_target_keeps_last_hour_funding_in_open_equity():
    cfg = config(timeseries_timeframe='1h',timeseries_fast_ema=2,timeseries_slow_ema=4,
                 timeseries_vol_lookback_bars=4,timeseries_min_ema_spread_pct=.005,max_drawdown_stop_pct=0)
    prices = [100]*6+[107,110,112,114]
    hours = [candle(i,p,p+1,p-1,p,3600000) for i,p in enumerate(prices)]
    base = [candle(i,prices[i//12],prices[i//12]+1,prices[i//12]-1,prices[i//12]) for i in range(120)]
    funding = sim.FundingHistory(times=[9*3600000+1800000],rates=[.001])
    sleeve = sim.simulate_timeseries_trend(hours,cfg,4*3600000,funding)
    execution_targets.prepare_sleeves([sleeve])
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(base,[sleeve],cfg,portfolio_risk.DrawdownRiskPolicy(),4*3600000)
    current = list(execution_targets.target_stream(base,[sleeve],result,cfg))[-1]
    assert current['equity'] == pytest.approx(sleeve['equity_curve'][-1]['equity'])
    assert sleeve['trades'][-1]['funding_pnl'] < 0


def test_execution_journal_retains_all_fills_beyond_ui_limit(tmp_path):
    a = account(tmp_path)
    for index in range(205):
        a.reconcile(target(.2 if index%2 == 0 else -.2),100000,now_ms=index+1)
    rows = [json.loads(line) for line in a.path.with_suffix('.fills.jsonl').read_text().splitlines()]
    assert len(rows) == a.load()['fill_count_total'] == 205
    assert len(a.load()['fills']) == 200
    assert [r['sequence'] for r in rows] == list(range(1,206))


def test_forward_macro_pins_decision_and_records_actual_first_observation(tmp_path):
    path = tmp_path/'macro.json'
    payload = {'series':{'vix':[[i*86400000,15] for i in range(10)]}}
    path.write_text(__import__('json').dumps(payload))
    trade = {'strategy':'hourly','side':'long','entry_time_utc':sim.iso_utc_from_ms(7*86400000),
             'initial_qty':1,'net_pnl':0,'_ledger':[]}
    cache = tmp_path/'decisions.json'
    first,_ = forward_macro.apply_overlay([{'trades':[trade]}],path,cache,8*86400000,factors=('vix',))
    payload['series']['vix'] = [[i*86400000,60] for i in range(10)]
    path.write_text(__import__('json').dumps(payload))
    second,_ = forward_macro.apply_overlay([{'trades':[trade]}],path,cache,9*86400000,factors=('vix',))
    assert first[0]['trades'] == second[0]['trades']
    records = __import__('json').loads(cache.read_text())['decisions']
    assert next(iter(records.values()))['first_seen_at_utc'] == sim.iso_utc_from_ms(8*86400000)
    assert len(list((tmp_path/'first_seen_macro').glob('*.receipt.json'))) == 2
    cache.write_text('{corrupt')
    with pytest.raises(ValueError,match='archive'):
        forward_macro.apply_overlay([{'trades':[trade]}],path,cache,10*86400000,factors=('vix',))


def test_failed_live_close_retains_hard_stop_latch(tmp_path):
    client = Mock()
    client.validate_live_ready.return_value = {'leverage':2,'max_notional_usdt':20000}
    client.account_snapshot.return_value = {'account':{'wallet_balance':8000,'margin_balance':8000},
                                           'positions':[{'signed_quantity':.1}]}
    client.quantize_quantity.side_effect = lambda q,s: abs(q)
    client.market_order.side_effect = RuntimeError('exchange unavailable')
    from trading_execution import write_json,read_json
    p = tmp_path/'live.json'
    write_json(p,{'peak_equity':10000})
    with pytest.raises(RuntimeError,match='unavailable'):
        LiveExecutor(client,p).reconcile(target(),100000)
    assert read_json(p)['risk_halt_at_utc']


def test_execute_report_settles_public_funding_without_exchange_orders(tmp_path):
    from datetime import datetime,timezone
    now = int(datetime.now(timezone.utc).timestamp()*1000)
    a = account(tmp_path)
    a.reconcile(target(.2),100000,now_ms=now-20000)
    qty = a.load()['position_qty']
    client = Mock()
    client.server_time_ms.return_value = now
    client.mark_price.return_value = 100000
    client.mark_price_observation.return_value = {"price": 100000, "time_ms": now}
    client.funding_history.return_value = [funding(now-10000)]
    from trading_execution import DEFAULT_SIMULATION_RULES
    client.symbol_rules.return_value = DEFAULT_SIMULATION_RULES
    report = {'execution_target':{'time_ms':now,'equity':100,'price':100000,'signed_qty':.0002}}
    with patch.object(trading_execution,'SimulationAccount',return_value=a):
        result = trading_execution.execute_report('simulation',report,client)
    assert result['funding_pnl'] == pytest.approx(-qty*100000*.001)
    client.market_order.assert_not_called()


def test_funding_api_pagination_preserves_associated_mark_price():
    from binance_terminal_client import BinanceTerminalClient
    client = BinanceTerminalClient()
    batch = [funding(time=i) for i in range(1000)]
    with patch.object(client,'public_get',side_effect=[batch,[funding(1000)]]) as get:
        rows = client.funding_history(0,1000)
    assert len(rows) == 1001
    assert rows[-1]['markPrice'] == 100000
    assert get.call_args_list[1].args[1]['startTime'] == 1000
