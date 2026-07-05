from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import simulate_range_swing as sim
import frozen_strategy
import paper_trade_range_swing as paper_range
import paper_trade_frozen_portfolio as paper_frozen
import paper_trade_timeseries_trend as paper_timeseries
import validate_frozen_strategy as frozen_validation


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
        self.assertEqual(cfg.timeseries_target_vol, 1.0)
        self.assertEqual(cfg.timeseries_max_leverage, 5.0)
        self.assertEqual(cfg.portfolio_leverage_cap, 5.0)

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
        self.assertEqual(timeseries_args.target_vol, 1.0)
        self.assertEqual(timeseries_args.max_leverage, 5.0)

    def test_frozen_strategy_manifest_is_valid(self) -> None:
        manifest_path = ROOT / "config/frozen_strategy_20260705.json"
        manifest, cfg = frozen_strategy.load_frozen_strategy(manifest_path)
        self.assertEqual(manifest["freeze_id"], "btc_default_20260705_v1")
        self.assertEqual(cfg.timeseries_fast_ema, 24)
        self.assertEqual(cfg.timeseries_slow_ema, 120)
        self.assertEqual(
            frozen_strategy.canonical_config_hash(manifest["config"]),
            manifest["config_sha256"],
        )

    def test_frozen_paper_state_never_places_orders(self) -> None:
        manifest, _ = frozen_strategy.load_frozen_strategy(
            ROOT / "config/frozen_strategy_20260705.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            state = paper_frozen.load_or_create_state(
                Path(directory) / "state.json",
                manifest,
                "BTCUSDT",
            )
        self.assertFalse(state["places_orders"])
        self.assertEqual(state["freeze_id"], manifest["freeze_id"])

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


if __name__ == "__main__":
    unittest.main()
