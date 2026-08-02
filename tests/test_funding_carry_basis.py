from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_funding_carry as carry
import download_spot_snapshot as spot_snapshot
import simulate_range_swing as sim
import validate_carry_trend_portfolio as portfolio


HOUR_MS = 3_600_000


def candle(timestamp_ms: int, price: float) -> sim.Candle:
    return sim.Candle(
        open_time_ms=timestamp_ms,
        open_time_utc=sim.iso_utc_from_ms(timestamp_ms),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1.0,
        quote_volume=price,
        close_time_ms=timestamp_ms + HOUR_MS - 1,
    )


class FundingCarryBasisTests(unittest.TestCase):
    def run_basis(self, exit_spot: float, exit_futures: float) -> dict:
        exit_time = 8 * HOUR_MS
        return carry.backtest_with_basis(
            [0, exit_time],
            [0.0, 0.0],
            [candle(0, 100.0), candle(exit_time, exit_spot)],
            [candle(0, 102.0), candle(exit_time, exit_futures)],
            evaluation_start_ms=0,
            interval_ms=HOUR_MS,
            notional_fraction=0.4,
            spot_fee=0.0,
            futures_fee=0.0,
            slippage_bps_per_leg=0.0,
            basis_stress_bps=0.0,
            maintenance_margin_pct=0.0,
        )

    def test_basis_convergence_is_profitable_for_short_perpetual(self) -> None:
        result = self.run_basis(100.0, 100.0)
        self.assertAlmostEqual(result["summary"]["observed_basis_pnl"], 0.8)
        self.assertAlmostEqual(result["summary"]["final_equity"], 100.8)

    def test_basis_widening_is_a_loss_for_short_perpetual(self) -> None:
        result = self.run_basis(100.0, 104.0)
        self.assertAlmostEqual(result["summary"]["observed_basis_pnl"], -0.8)
        self.assertAlmostEqual(result["summary"]["final_equity"], 99.2)

    def test_separate_futures_margin_breach_is_reported(self) -> None:
        exit_time = 8 * HOUR_MS
        result = carry.backtest_with_basis(
            [0, exit_time],
            [0.0, 0.0],
            [candle(0, 100.0), candle(exit_time, 300.0)],
            [candle(0, 100.0), candle(exit_time, 300.0)],
            evaluation_start_ms=0,
            interval_ms=HOUR_MS,
            notional_fraction=0.5,
            spot_fee=0.0,
            futures_fee=0.0,
            slippage_bps_per_leg=0.0,
            basis_stress_bps=0.0,
            maintenance_margin_pct=0.004,
        )
        self.assertTrue(result["summary"]["liquidated"])
        self.assertLess(result["summary"]["min_futures_margin_buffer_pct"], 0.0)

    def test_periodic_rebalance_resets_futures_margin(self) -> None:
        times = [0, 8 * HOUR_MS, 16 * HOUR_MS]
        spot = [candle(times[0], 100.0), candle(times[1], 150.0), candle(times[2], 225.0)]
        futures = [candle(times[0], 100.0), candle(times[1], 150.0), candle(times[2], 225.0)]
        result = carry.backtest_with_basis(
            times,
            [0.0, 0.0, 0.0],
            spot,
            futures,
            evaluation_start_ms=0,
            interval_ms=HOUR_MS,
            notional_fraction=0.5,
            spot_fee=0.0,
            futures_fee=0.0,
            slippage_bps_per_leg=0.0,
            basis_stress_bps=0.0,
            maintenance_margin_pct=0.004,
            rebalance_days=8 / 24,
        )
        self.assertFalse(result["summary"]["liquidated"])
        self.assertEqual(result["summary"]["rebalances"], 1)

    def test_margin_buffer_rebalance_avoids_calendar_turnover(self) -> None:
        times = [0, 8 * HOUR_MS, 16 * HOUR_MS]
        spot = [candle(times[0], 100.0), candle(times[1], 180.0), candle(times[2], 180.0)]
        futures = [candle(times[0], 100.0), candle(times[1], 180.0), candle(times[2], 180.0)]
        result = carry.backtest_with_basis(
            times,
            [0.0, 0.0, 0.0],
            spot,
            futures,
            evaluation_start_ms=0,
            interval_ms=HOUR_MS,
            notional_fraction=0.5,
            spot_fee=0.0,
            futures_fee=0.0,
            slippage_bps_per_leg=0.0,
            basis_stress_bps=0.0,
            maintenance_margin_pct=0.004,
            rebalance_margin_buffer_pct=30.0,
        )
        self.assertFalse(result["summary"]["liquidated"])
        self.assertEqual(result["summary"]["rebalances"], 1)

    def test_spot_snapshot_round_trip(self) -> None:
        rows = [candle(0, 100.0), candle(HOUR_MS, 101.0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spot.json.gz"
            spot_snapshot.save_spot_snapshot(path, "BTCUSDT", "1h", rows, 0, 2 * HOUR_MS)
            loaded, metadata = spot_snapshot.load_spot_snapshot(path)
        self.assertEqual([item.open for item in loaded], [100.0, 101.0])
        self.assertEqual(metadata["symbol"], "BTCUSDT")
        self.assertEqual(metadata["interval"], "1h")

    def test_segregated_portfolio_combines_normalized_equity(self) -> None:
        combined = portfolio.combine_equity_curves(
            [{"time_ms": 0, "equity": 100.0}, {"time_ms": 10, "equity": 110.0}],
            [{"time_ms": 0, "equity": 100.0}, {"time_ms": 10, "equity": 90.0}],
            carry_weight=0.75,
        )
        self.assertAlmostEqual(combined[-1]["equity"], 105.0)


if __name__ == "__main__":
    unittest.main()
