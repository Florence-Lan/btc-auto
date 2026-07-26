from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import simulate_range_swing as sim
import frozen_strategy
import event_risk
import macro_regime
import paper_trade_range_swing as paper_range
import paper_trade_frozen_portfolio as paper_frozen
import paper_trade_timeseries_trend as paper_timeseries
import portfolio_risk
import run_trading_terminal as trading_terminal
import run_execution_supervisor as execution_supervisor
import trading_execution
from binance_terminal_client import sign_query
import validate_frozen_strategy as frozen_validation
import validate_strategies as strategy_validation


def candle(index: int, open_: float, high: float, low: float, close: float, interval_ms: int = 300_000) -> sim.Candle:
    open_time = index * interval_ms
    return sim.Candle(
        open_time_ms=open_time,
        open_time_utc=sim.iso_utc_from_ms(open_time),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10.0,
        quote_volume=1_000_000.0,
        close_time_ms=open_time + interval_ms - 1,
    )


def config(**changes: object) -> sim.StrategyConfig:
    original = sys.argv
    try:
        sys.argv = ["simulate_range_swing.py", "--no-market-context-enabled"]
        cfg = sim.config_from_args(sim.parse_args())
    finally:
        sys.argv = original
    return replace(cfg, **changes)


class StrategyEngineTests(unittest.TestCase):
    def test_binance_signature_is_deterministic(self) -> None:
        self.assertEqual(
            sign_query("secret", "symbol=BTCUSDT&timestamp=1"),
            "ef9d3d77a34d9a13a21a4c2d7f3e8cb091888a74ca62b5b62f430e78eded95ba",
        )

    def test_trading_terminal_defaults_to_mainnet_simulation(self) -> None:
        with (
            mock.patch.object(trading_terminal.CONTROLLER, "mode", return_value="simulation"),
            mock.patch.object(trading_terminal.BINANCE, "mark_price", return_value=65_000.0),
        ):
            status = trading_terminal.CONTROLLER.status()
        self.assertEqual(status["mode"], "SIMULATION")
        self.assertFalse(status["execution"]["places_orders"])
        self.assertEqual(status["market"]["source"], "Binance mainnet realtime")
        self.assertIn("target_leverage", status["strategy"])
        self.assertIn("target_notional", status["strategy"])
        self.assertIn("initial_balance", status["account_details"])
        self.assertEqual(status["execution"]["check_interval_seconds"], 30)
        self.assertEqual(status["execution"]["strategy_bar_seconds"], 300)

    def test_execution_scheduler_waits_for_closed_five_minute_bar(self) -> None:
        interval = execution_supervisor.BAR_INTERVAL_MS
        boundary = 10 * interval
        self.assertIsNone(
            execution_supervisor.due_closed_bar_open_ms(boundary + 2_000, 0)
        )
        due = execution_supervisor.due_closed_bar_open_ms(boundary + 3_000, 0)
        self.assertEqual(due, boundary - interval)

    def test_execution_scheduler_deduplicates_processed_bar(self) -> None:
        interval = execution_supervisor.BAR_INTERVAL_MS
        boundary = 10 * interval
        processed = boundary - interval
        self.assertIsNone(
            execution_supervisor.due_closed_bar_open_ms(
                boundary + 30_000,
                processed,
            )
        )

    def test_execution_scheduler_aligns_to_settlement_delay(self) -> None:
        interval = execution_supervisor.BAR_INTERVAL_MS
        boundary = 10 * interval
        self.assertAlmostEqual(
            execution_supervisor.seconds_until_next_check(boundary + 2_000, 30),
            1.0,
        )
        self.assertAlmostEqual(
            execution_supervisor.seconds_until_next_check(boundary + 290_000, 30),
            13.0,
        )

    def test_strategy_target_scales_as_leverage_not_virtual_quantity(self) -> None:
        now_ms = 1_000_000
        report = {
            "summary": {
                "last_equity_point": {
                    "time_ms": now_ms,
                    "equity": 100.0,
                    "price": 50_000.0,
                    "signed_qty": 0.002,
                }
            }
        }
        target = trading_execution.target_from_report(report, now_ms=now_ms)
        self.assertAlmostEqual(target["target_leverage"], 1.0)

    def test_execution_target_overrides_flat_backtest_endpoint(self) -> None:
        now_ms = 1_000_000
        report = {
            "execution_target": {
                "time_ms": now_ms,
                "equity": 100.0,
                "price": 50_000.0,
                "signed_qty": 0.002,
            },
            "summary": {
                "last_equity_point": {
                    "time_ms": now_ms,
                    "equity": 100.0,
                    "price": 50_000.0,
                    "signed_qty": 0.0,
                }
            },
        }
        target = trading_execution.target_from_report(report, now_ms=now_ms)
        self.assertAlmostEqual(target["target_leverage"], 1.0)

    def test_shadow_end_trade_survives_as_execution_target(self) -> None:
        bars = [
            candle(1, 100, 101, 99, 100),
            candle(2, 110, 111, 109, 110),
            candle(3, 120, 121, 119, 120),
        ]
        trade = {
            "side": "long",
            "entry_time_utc": bars[0].open_time_utc,
            "exit_time_utc": bars[1].open_time_utc,
            "entry_price": 100.0,
            "avg_exit_price": 110.0,
            "initial_qty": 1.0,
            "pnl": 10.0,
            "fees": 0.0,
            "net_pnl": 10.0,
            "return_on_equity_pct": 10.0,
            "bars_held": 1,
            "exit_reason": "end",
            "signal_reason": "timeseries_trend_long",
            "liquidation_price": 0.0,
            "funding_pnl": 0.0,
            "slippage_cost": 0.0,
            "strategy": "timeseries_trend_6h",
            "_open_qty_fraction": 0.4,
        }
        cfg = config(portfolio_leverage_cap=2.0, max_drawdown_stop_pct=0.0)
        result = portfolio_risk.combine_sleeves_with_drawdown_policy(
            bars,
            [{"trades": [trade], "summary": {}}],
            cfg,
            portfolio_risk.DrawdownRiskPolicy(),
            bars[0].open_time_ms,
            include_execution_target=True,
        )
        self.assertEqual(result["summary"]["last_equity_point"]["signed_qty"], 0.0)
        self.assertAlmostEqual(result["execution_target"]["signed_qty"], 0.4)
        self.assertTrue(result["execution_target"]["position_id"])
        self.assertEqual(
            result["execution_target"]["origin_signal_time_ms"],
            bars[0].open_time_ms,
        )
        self.assertAlmostEqual(result["execution_target"]["origin_entry_price"], 100.0)
        target = trading_execution.target_from_report(
            result,
            now_ms=bars[-1].open_time_ms,
        )
        self.assertGreater(target["target_leverage"], 0.0)

    def test_new_shadow_window_emits_a_valid_flat_execution_target(self) -> None:
        bars = [candle(1, 100, 101, 99, 100)]
        cfg = config(portfolio_leverage_cap=2.0, max_drawdown_stop_pct=0.0)
        result = portfolio_risk.combine_sleeves_with_drawdown_policy(
            bars,
            [{"trades": [], "summary": {}}],
            cfg,
            portfolio_risk.DrawdownRiskPolicy(),
            bars[-1].open_time_ms + 1,
            include_execution_target=True,
        )
        target = result["execution_target"]
        self.assertEqual(target["time_ms"], bars[-1].open_time_ms)
        self.assertEqual(target["equity"], cfg.initial_equity)
        self.assertEqual(target["price"], bars[-1].close)
        self.assertEqual(target["signed_qty"], 0.0)
        self.assertEqual(target["position_id"], "flat")

    def test_open_position_annotation_preserves_partial_quantity(self) -> None:
        sleeve = {
            "equity_curve": [{"signed_qty": 0.25}],
            "trades": [
                {
                    "side": "long",
                    "exit_reason": "end",
                    "initial_qty": 1.0,
                }
            ],
        }
        paper_frozen.annotate_open_position_fractions([sleeve])
        self.assertAlmostEqual(sleeve["trades"][0]["_open_qty_fraction"], 0.25)

    def test_execution_candidate_excludes_negative_expectancy_range(self) -> None:
        candidate = json.loads(
            (ROOT / "config/shadow_candidate_macro_20260720.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("range", candidate["strategy_modes"])
        self.assertFalse(candidate["live_orders_allowed"])

    def test_stale_strategy_target_is_blocked(self) -> None:
        report = {
            "summary": {
                "last_equity_point": {
                    "time_ms": 1,
                    "equity": 100.0,
                    "price": 50_000.0,
                    "signed_qty": 0.001,
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "stale"):
            trading_execution.target_from_report(
                report,
                now_ms=1_000_000,
                max_age_seconds=10,
            )

    def test_simulation_reconcile_never_calls_binance_order_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            account = trading_execution.SimulationAccount(Path(temporary) / "sim.json")
            target = {
                "signal_time_ms": 1,
                "signal_price": 50_000.0,
                "target_leverage": 1.0,
                "age_seconds": 0.0,
            }
            result = account.reconcile(target, 50_000.0)
            self.assertEqual(result["mode"], "simulation")
            self.assertIsNotNone(result["fill"])
            snapshot = account.snapshot(50_000.0)
            self.assertEqual(len(snapshot["positions"]), 1)
            self.assertEqual(snapshot["positions"][0]["side"], "LONG")

    def test_simulation_quantizes_and_ignores_micro_rebalances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            account = trading_execution.SimulationAccount(Path(temporary) / "sim.json")
            account.reset(1_000.0)
            first = {
                "signal_time_ms": 1,
                "signal_price": 50_000.0,
                "target_leverage": 0.115,
                "age_seconds": 0.0,
            }
            second = {**first, "signal_time_ms": 2, "target_leverage": 0.119}
            first_result = account.reconcile(first, 50_000.0)
            second_result = account.reconcile(second, 50_000.0)
            snapshot = account.snapshot(50_000.0)
        self.assertEqual(first_result["target_qty"], 0.002)
        self.assertIsNotNone(first_result["fill"])
        self.assertEqual(second_result["target_qty"], 0.002)
        self.assertIsNone(second_result["fill"])
        self.assertEqual(snapshot["state"]["fill_count_total"], 1)
        self.assertEqual(snapshot["positions"][0]["signed_quantity"], 0.002)

    def test_new_simulation_waits_for_next_position_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            account = trading_execution.SimulationAccount(Path(temporary) / "sim.json")
            account.reset(1_000.0)
            existing = {
                "signal_time_ms": 1,
                "signal_price": 50_000.0,
                "target_leverage": 1.0,
                "age_seconds": 0.0,
                "position_id": "existing-position",
                "origin_signal_time_ms": 1,
            }
            first = account.reconcile(existing, 50_000.0)
            repeated = account.reconcile(existing, 50_000.0)
            next_signal = account.reconcile(
                {**existing, "signal_time_ms": 2, "position_id": "next-position"},
                50_000.0,
            )
            snapshot = account.snapshot(50_000.0)
        self.assertEqual(first["entry_guard"]["status"], "waiting_for_next_signal")
        self.assertIsNone(first["fill"])
        self.assertEqual(repeated["entry_guard"]["status"], "waiting_for_next_signal")
        self.assertIsNone(repeated["fill"])
        self.assertEqual(next_signal["entry_guard"]["status"], "new_signal_allowed")
        self.assertIsNotNone(next_signal["fill"])
        self.assertEqual(snapshot["state"]["fill_count_total"], 1)

    def test_terminal_prefers_execution_target_over_closed_backtest_point(self) -> None:
        execution_target = {"signed_qty": 0.25, "position_id": "open-position"}
        point = trading_terminal.execution_point_for_report(
            {"execution_target": execution_target},
            {"last_equity_point": {"signed_qty": 0.0}},
        )
        self.assertIs(point, execution_target)

    def test_simulation_initial_balance_can_be_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            account = trading_execution.SimulationAccount(Path(temporary) / "sim.json")
            state = account.reset(12_345.67)
            snapshot = account.snapshot(50_000.0)
        self.assertEqual(state["initial_balance"], 12_345.67)
        self.assertEqual(snapshot["account"]["wallet_balance"], 12_345.67)
        self.assertEqual(snapshot["positions"], [])
        self.assertEqual(snapshot["recent_trades"], [])

    def test_simulation_reset_rejects_invalid_amount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            account = trading_execution.SimulationAccount(Path(temporary) / "sim.json")
            with self.assertRaisesRegex(ValueError, "初始金额"):
                account.reset(0)

    def test_live_mode_requires_exact_confirmation(self) -> None:
        controller = trading_terminal.TerminalController(mock.Mock())
        with (
            mock.patch.object(controller, "runtime", return_value={"running": False}),
            mock.patch.object(controller, "emergency", return_value=None),
            mock.patch.object(controller, "mode", return_value="simulation"),
        ):
            with self.assertRaisesRegex(ValueError, "确认词"):
                controller.set_mode("live", "wrong")

    def test_live_executor_caps_order_notional(self) -> None:
        client = mock.Mock()
        client.validate_live_ready.return_value = {
            "max_notional_usdt": 50.0,
            "leverage": 2,
        }
        client.account_snapshot.return_value = {
            "account": {"wallet_balance": 100.0, "margin_balance": 100.0},
            "positions": [],
            "open_orders": [],
        }
        client.quantize_quantity.side_effect = lambda quantity, symbol: round(abs(quantity), 3)
        client.market_order.return_value = {"status": "FILLED"}
        target = {
            "signal_time_ms": 1_000,
            "signal_price": 50_000.0,
            "target_leverage": 2.0,
            "age_seconds": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            executor = trading_execution.LiveExecutor(
                client,
                Path(temporary) / "live.json",
            )
            result = executor.reconcile(target, 50_000.0)
        self.assertEqual(result["target_qty"], 0.001)
        client.market_order.assert_called_once()
        self.assertEqual(client.market_order.call_args.args[1], 0.001)
        self.assertFalse(client.market_order.call_args.kwargs["reduce_only"])

    def test_live_emergency_cancels_and_flattens(self) -> None:
        client = mock.Mock()
        controller = trading_terminal.TerminalController(client)
        with (
            mock.patch.object(controller, "mode", return_value="live"),
            mock.patch.object(controller, "stop", return_value={"running": False}),
            mock.patch.object(trading_terminal, "write_json"),
        ):
            result = controller.emergency_stop("test")
        client.cancel_all_orders.assert_called_once_with("BTCUSDT")
        client.flatten_position.assert_called_once()
        self.assertTrue(result["emergency"]["exchange_actions"]["cancelled_orders"])

    def test_event_risk_never_uses_unpublished_news(self) -> None:
        event = event_risk.RiskEvent(
            "news-1",
            published_at_ms=200,
            starts_at_ms=100,
            ends_at_ms=400,
            severity=1.0,
            block_entries=True,
            category="regulation",
            headline="test",
        )
        before_publication = event_risk.event_decision_at((event,), 150)
        after_publication = event_risk.event_decision_at((event,), 250)
        self.assertTrue(before_publication.allowed)
        self.assertFalse(after_publication.allowed)

    def test_tiered_drawdown_policy_throttles_before_hard_stop(self) -> None:
        policy = portfolio_risk.DrawdownRiskPolicy(8.0, 15.0, 0.35)
        self.assertEqual(portfolio_risk.drawdown_multiplier(0.07, policy), 1.0)
        middle = portfolio_risk.drawdown_multiplier(0.115, policy)
        self.assertGreater(middle, 0.35)
        self.assertLess(middle, 1.0)
        self.assertEqual(portfolio_risk.drawdown_multiplier(0.15, policy), 0.0)

    def test_macro_snapshot_is_point_in_time_and_scales_without_amplifying(self) -> None:
        day = macro_regime.MS_PER_DAY
        rows = tuple(index * day for index in range(8))
        snapshot = macro_regime.MacroSnapshot(
            {
                "sp500": macro_regime.MacroSeries(rows, (100, 100, 100, 100, 100, 95, 94, 93)),
                "nasdaq": macro_regime.MacroSeries(rows, (100, 100, 100, 100, 100, 94, 92, 90)),
                "vix": macro_regime.MacroSeries(rows, (20, 20, 20, 20, 20, 25, 28, 30)),
                "dollar": macro_regime.MacroSeries(rows, (100, 100, 100, 100, 100, 101, 102, 103)),
                "gold": macro_regime.MacroSeries(rows, (100, 100, 100, 100, 100, 102, 103, 104)),
                "silver": macro_regime.MacroSeries(rows, (100, 100, 100, 100, 100, 99, 98, 97)),
                "fear_greed": macro_regime.MacroSeries(rows, (50, 50, 50, 50, 50, 35, 25, 20)),
            },
            {},
        )
        decision = macro_regime.macro_decision_at(snapshot, 7 * day)
        self.assertLess(decision.score, 0)
        self.assertLessEqual(decision.score, min(decision.contributions.values()) * 0.50)
        self.assertGreaterEqual(decision.risk_multiplier, 0.35)
        self.assertLessEqual(decision.risk_multiplier, 1.0)
        stale = macro_regime.macro_decision_at(snapshot, 20 * day, max_staleness_days=5)
        self.assertFalse(stale.allowed)

    def test_higher_timeframe_aggregation_uses_only_complete_buckets(self) -> None:
        candles = [candle(index, 100, 101, 99, 100 + index / 10) for index in range(13)]
        aggregated = sim.aggregate_candles(candles, "1h")
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0].close_time_ms, 3_600_000 - 1)
        self.assertEqual(aggregated[0].close, candles[11].close)

    def test_positive_funding_charges_long_and_credits_short(self) -> None:
        bar = candle(1, 100, 101, 99, 100)
        funding = sim.FundingHistory(times=[bar.open_time_ms], rates=[0.001])
        long = sim.Position("long", 0, bar.open_time_utc, 100, 1, 1, 95, 0, 0, 0, 0)
        short = sim.Position("short", 0, bar.open_time_utc, 100, 1, 1, 105, 0, 0, 0, 0)
        sim.settle_position_funding(long, bar, funding)
        sim.settle_position_funding(short, bar, funding)
        self.assertAlmostEqual(long.funding_pnl, -0.1)
        self.assertAlmostEqual(short.funding_pnl, 0.1)

    def test_limit_price_and_entry_bar_stop_are_pessimistic(self) -> None:
        bar = candle(1, 100, 102, 97, 101)
        position = sim.Position("long", 0, bar.open_time_utc, 100, 1, 1, 98, 0, 0, 0, 0)
        self.assertTrue(sim.price_within_candle(101.5, bar))
        self.assertFalse(sim.price_within_candle(102.5, bar))
        self.assertTrue(sim.stop_hit_for_position(position, bar))

    def test_liquidation_hit_checks_both_sides(self) -> None:
        bar = candle(1, 100, 106, 94, 100)
        long = sim.Position("long", 0, bar.open_time_utc, 100, 1, 1, 90, 0, 0, 0, 95)
        short = sim.Position("short", 0, bar.open_time_utc, 100, 1, 1, 110, 0, 0, 0, 105)
        self.assertTrue(sim.liquidation_hit_for_position(long, bar))
        self.assertTrue(sim.liquidation_hit_for_position(short, bar))

    def test_drawdown_halt_is_explicit(self) -> None:
        cfg = config(max_drawdown_stop_pct=10.0)
        self.assertFalse(sim.drawdown_halted(0.099, cfg))
        self.assertTrue(sim.drawdown_halted(0.10, cfg))
        self.assertFalse(sim.drawdown_halted(0.50, replace(cfg, max_drawdown_stop_pct=0.0)))

    def test_default_trend_entry_uses_near_touch_limit(self) -> None:
        cfg = config()
        bar = candle(1, 100, 102, 98, 100)
        indicator_values = {"atr": [None, 2.0]}
        pending = sim.build_trend_pending_entry(
            [candle(0, 100, 101, 99, 100), bar],
            indicator_values,
            signal_index=1,
            created_index=2,
            side="long",
            reason="trend_long_pullback",
            signal_score=cfg.trend_min_signal_score,
            cfg=cfg,
        )
        self.assertEqual(cfg.trend_entry_pullback_atr, 0.05)
        self.assertIsNotNone(pending)
        self.assertAlmostEqual(pending.target_price, 99.9)

    def test_default_portfolio_adds_timeseries_trend_sleeve(self) -> None:
        cfg = config()
        self.assertEqual(cfg.strategy_modes, ("trend", "timeseries_trend"))
        self.assertEqual(cfg.portfolio_mode, "sleeves")
        self.assertEqual(cfg.timeseries_fast_ema, 24)
        self.assertEqual(cfg.timeseries_slow_ema, 120)
        self.assertEqual(cfg.timeseries_min_ema_spread_pct, 0.0)
        self.assertEqual(cfg.timeseries_target_vol, 0.12)
        self.assertEqual(cfg.timeseries_max_leverage, 2.0)
        self.assertEqual(cfg.portfolio_leverage_cap, 2.0)

    def test_timeseries_ema_hysteresis_filters_small_crosses(self) -> None:
        bars = [candle(index, 100, 101, 99, 100) for index in range(8)]
        fast = [100.0, 100.0, 100.0, 100.1, 99.9, 100.1, 99.9, 100.1]
        slow = [100.0] * len(bars)
        base_cfg = config(
            timeseries_fast_ema=1,
            timeseries_slow_ema=2,
            timeseries_vol_lookback_bars=2,
        )
        with mock.patch.object(sim, "ema", side_effect=[fast, slow]):
            baseline = sim.simulate_timeseries_trend(bars, base_cfg)
        with mock.patch.object(sim, "ema", side_effect=[fast, slow]):
            filtered = sim.simulate_timeseries_trend(
                bars,
                replace(base_cfg, timeseries_min_ema_spread_pct=0.003),
            )
        self.assertGreater(len(baseline["trades"]), 0)
        self.assertEqual(filtered["trades"], [])

    def test_paper_defaults_match_promoted_strategies(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["paper_trade_range_swing.py"]
            range_args = paper_range.parse_args()
            sys.argv = ["paper_trade_timeseries_trend.py"]
            timeseries_args = paper_timeseries.parse_args()
        finally:
            sys.argv = original
        self.assertEqual(range_args.trend_entry_pullback_atr, 0.05)
        self.assertEqual(range_args.leverage, 2.0)
        self.assertEqual(range_args.risk_per_trade, 0.015)
        self.assertEqual(timeseries_args.target_vol, 0.12)
        self.assertEqual(timeseries_args.max_leverage, 2.0)

    def test_frozen_strategy_manifest_is_valid(self) -> None:
        manifest_path = ROOT / "config/frozen_strategy_20260711.json"
        manifest, cfg = frozen_strategy.load_frozen_strategy(
            manifest_path,
            verify_engine=False,
        )
        self.assertEqual(manifest["freeze_id"], "btc_risk_controlled_20260711_v1")
        self.assertEqual(cfg.timeseries_fast_ema, 24)
        self.assertEqual(cfg.timeseries_slow_ema, 120)
        self.assertEqual(cfg.risk_per_trade, 0.015)
        self.assertEqual(cfg.max_drawdown_stop_pct, 12.0)
        self.assertEqual(cfg.portfolio_leverage_cap, 2.0)
        self.assertEqual(
            frozen_strategy.canonical_config_hash(manifest["config"]),
            manifest["config_sha256"],
        )

    def test_frozen_paper_state_never_places_orders(self) -> None:
        manifest, _ = frozen_strategy.load_frozen_strategy(
            ROOT / "config/frozen_strategy_20260711.json",
            verify_engine=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            state = paper_frozen.load_or_create_state(
                Path(directory) / "state.json",
                manifest,
                "BTCUSDT",
            )
        self.assertFalse(state["places_orders"])
        self.assertEqual(state["freeze_id"], manifest["freeze_id"])

    def test_active_frozen_strategy_manifest_is_valid(self) -> None:
        historical, _ = frozen_strategy.load_frozen_strategy(
            ROOT / "config/frozen_strategy_active_20260711.json",
            verify_engine=False,
        )
        self.assertEqual(historical["freeze_id"], "btc_active_20260711_v1")
        manifest, cfg = frozen_strategy.load_frozen_strategy(
            ROOT / "config/frozen_strategy_active_20260720.json"
        )
        self.assertEqual(manifest["freeze_id"], "btc_active_20260720_v2")
        self.assertEqual(cfg.risk_per_trade, 0.0075)
        self.assertEqual(cfg.strategy_modes, ("trend", "range", "timeseries_trend"))
        self.assertEqual(cfg.timeseries_target_vol, 0.11)
        self.assertEqual(cfg.timeseries_min_ema_spread_pct, 0.003)
        self.assertEqual(manifest["validation_targets"]["trades_per_year_min"], 45.0)

    def test_block_bootstrap_is_deterministic(self) -> None:
        returns = [0.01, -0.005, 0.002, 0.004] * 30
        first = frozen_validation.bootstrap_distribution(
            returns,
            block_days=5,
            samples=100,
            seed=7,
        )
        second = frozen_validation.bootstrap_distribution(
            returns,
            block_days=5,
            samples=100,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertIsNotNone(first["annualized_p05_pct"])

    def test_execution_cost_stress_is_adverse(self) -> None:
        bar = candle(1, 100, 101, 99, 100)
        base = config(depth_impact_bps=0.0)
        stressed = replace(
            base,
            entry_slippage_bps=base.entry_slippage_bps * 2,
            exit_slippage_bps=base.exit_slippage_bps * 2,
        )
        base_entry = sim.execution_price(100, "long", True, 1, bar, base)
        stressed_entry = sim.execution_price(100, "long", True, 1, bar, stressed)
        base_exit = sim.execution_price(100, "long", False, 1, bar, base)
        stressed_exit = sim.execution_price(100, "long", False, 1, bar, stressed)
        self.assertGreater(stressed_entry, base_entry)
        self.assertLess(stressed_exit, base_exit)

    def test_validation_preserves_all_tactical_modes(self) -> None:
        cfg = config(strategy_modes=("trend", "range", "timeseries_trend"))
        captured: dict[str, tuple[str, ...]] = {}

        def fake_tactical(*args: object, **kwargs: object) -> dict[str, object]:
            captured["modes"] = args[1].strategy_modes
            return {"summary": {}, "trades": [], "equity_curve": []}

        with (
            mock.patch.object(strategy_validation.sim, "simulate", side_effect=fake_tactical),
            mock.patch.object(
                strategy_validation.sim,
                "simulate_timeseries_trend",
                return_value={"summary": {}, "trades": [], "equity_curve": []},
            ),
            mock.patch.object(
                strategy_validation.sim,
                "combine_sleeve_results",
                return_value={"summary": {}, "trades": [], "equity_curve": []},
            ),
        ):
            strategy_validation.run_fold([], [], sim.FundingHistory([], []), cfg, 0, 1)
        self.assertEqual(captured["modes"], ("trend", "range"))

    def test_snapshot_round_trip(self) -> None:
        candles = {"5m": [candle(1, 100, 101, 99, 100)]}
        funding = sim.FundingHistory(times=[300_000], rates=[0.0001])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json.gz"
            sim.save_market_snapshot(path, "BTCUSDT", candles, funding, 0, 600_000)
            loaded, loaded_funding, metadata = sim.load_market_snapshot(path)
        self.assertEqual(loaded["5m"][0].close, 100)
        self.assertEqual(loaded_funding.rates, [0.0001])
        self.assertEqual(metadata["symbol"], "BTCUSDT")

    def test_portfolio_caps_same_direction_net_exposure(self) -> None:
        bars = [
            candle(1, 100, 101, 99, 100),
            candle(2, 110, 111, 109, 110),
        ]
        entry_time = bars[0].open_time_utc
        exit_time = bars[1].open_time_utc

        def raw_trade(strategy: str) -> dict[str, object]:
            return {
                "side": "long",
                "entry_time_utc": entry_time,
                "exit_time_utc": exit_time,
                "entry_price": 100.0,
                "avg_exit_price": 110.0,
                "initial_qty": 1.0,
                "pnl": 10.0,
                "fees": 0.0,
                "net_pnl": 10.0,
                "return_on_equity_pct": 10.0,
                "bars_held": 1,
                "exit_reason": "test",
                "signal_reason": strategy,
                "liquidation_price": 0.0,
                "funding_pnl": 0.0,
                "slippage_cost": 0.0,
                "strategy": strategy,
            }

        sleeve_a = {"trades": [raw_trade("a")], "summary": {}}
        sleeve_b = {"trades": [raw_trade("b")], "summary": {}}
        cfg = config(portfolio_leverage_cap=1.5, max_drawdown_stop_pct=0.0)
        result = sim.combine_sleeve_results(bars, [sleeve_a, sleeve_b], cfg, bars[0].open_time_ms)
        total_qty = sum(float(trade["initial_qty"]) for trade in result["trades"])
        self.assertAlmostEqual(total_qty, 1.5)
        self.assertAlmostEqual(result["summary"]["final_equity"], 115.0)

    def test_portfolio_caps_opposite_direction_gross_exposure(self) -> None:
        bars = [
            candle(1, 100, 101, 99, 100),
            candle(2, 110, 111, 89, 110),
        ]
        entry_time = bars[0].open_time_utc
        exit_time = bars[1].open_time_utc

        def raw_trade(side: str) -> dict[str, object]:
            exit_price = 110.0 if side == "long" else 90.0
            return {
                "side": side,
                "entry_time_utc": entry_time,
                "exit_time_utc": exit_time,
                "entry_price": 100.0,
                "avg_exit_price": exit_price,
                "initial_qty": 1.0,
                "pnl": 10.0,
                "fees": 0.0,
                "net_pnl": 10.0,
                "return_on_equity_pct": 10.0,
                "bars_held": 1,
                "exit_reason": "test",
                "signal_reason": f"{side}_test",
                "liquidation_price": 0.0,
                "funding_pnl": 0.0,
                "slippage_cost": 0.0,
                "strategy": side,
            }

        sleeves = [
            {"trades": [raw_trade("long")], "summary": {}},
            {"trades": [raw_trade("short")], "summary": {}},
        ]
        cfg = config(portfolio_leverage_cap=1.5, max_drawdown_stop_pct=0.0)
        result = sim.combine_sleeve_results(bars, sleeves, cfg, bars[0].open_time_ms)
        total_qty = sum(float(trade["initial_qty"]) for trade in result["trades"])
        self.assertAlmostEqual(total_qty, 1.5)
        self.assertAlmostEqual(result["summary"]["final_equity"], 115.0)

    def test_portfolio_keeps_entry_bar_stop_trade(self) -> None:
        bar = candle(1, 100, 101, 95, 96)
        trade = {
            "side": "long",
            "entry_time_utc": bar.open_time_utc,
            "exit_time_utc": bar.open_time_utc,
            "entry_price": 100.0,
            "avg_exit_price": 96.0,
            "initial_qty": 1.0,
            "pnl": -4.0,
            "fees": 0.0,
            "net_pnl": -4.0,
            "return_on_equity_pct": -4.0,
            "bars_held": 0,
            "exit_reason": "entry_bar_stop",
            "signal_reason": "trend_test",
            "liquidation_price": 0.0,
            "funding_pnl": 0.0,
            "slippage_cost": 0.0,
            "strategy": "trend_pullback_5m",
        }
        cfg = config(portfolio_leverage_cap=1.5, max_drawdown_stop_pct=0.0)
        result = sim.combine_sleeve_results(
            [bar],
            [{"trades": [trade], "summary": {}}],
            cfg,
            bar.open_time_ms,
        )
        self.assertEqual(result["summary"]["trades"], 1)
        self.assertAlmostEqual(result["summary"]["final_equity"], 96.0)

    def test_portfolio_drawdown_halt_blocks_later_entries(self) -> None:
        bars = [
            candle(1, 100, 101, 99, 100),
            candle(2, 100, 101, 99, 100),
        ]

        def immediate_trade(index: int, net_pnl: float) -> dict[str, object]:
            return {
                "side": "long",
                "entry_time_utc": bars[index].open_time_utc,
                "exit_time_utc": bars[index].open_time_utc,
                "entry_price": 100.0,
                "avg_exit_price": 100.0,
                "initial_qty": 1.0,
                "pnl": net_pnl,
                "fees": 0.0,
                "net_pnl": net_pnl,
                "return_on_equity_pct": net_pnl,
                "bars_held": 0,
                "exit_reason": "test",
                "signal_reason": "drawdown_test",
                "liquidation_price": 0.0,
                "funding_pnl": 0.0,
                "slippage_cost": 0.0,
                "strategy": "test",
            }

        sleeve = {
            "trades": [immediate_trade(0, -15.0), immediate_trade(1, 20.0)],
            "summary": {},
        }
        cfg = config(portfolio_leverage_cap=1.5, max_drawdown_stop_pct=10.0)
        result = sim.combine_sleeve_results(bars, [sleeve], cfg, bars[0].open_time_ms)
        self.assertEqual(result["summary"]["trades"], 1)
        self.assertAlmostEqual(result["summary"]["final_equity"], 85.0)


if __name__ == "__main__":
    unittest.main()
