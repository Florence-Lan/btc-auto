from __future__ import annotations

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
        self.assertEqual(cfg.timeseries_target_vol, 0.12)
        self.assertEqual(cfg.timeseries_max_leverage, 2.0)
        self.assertEqual(cfg.portfolio_leverage_cap, 2.0)

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
        manifest, cfg = frozen_strategy.load_frozen_strategy(manifest_path)
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
        manifest, cfg = frozen_strategy.load_frozen_strategy(
            ROOT / "config/frozen_strategy_active_20260711.json"
        )
        self.assertEqual(manifest["freeze_id"], "btc_active_20260711_v1")
        self.assertEqual(cfg.risk_per_trade, 0.0075)
        self.assertEqual(cfg.strategy_modes, ("trend", "range", "timeseries_trend"))
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
