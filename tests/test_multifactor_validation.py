from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import frozen_strategy
import multifactor
import public_context
import validate_multifactor as validation


def test_public_coverage_counts_outage_time_even_with_no_trades():
    start = 1_800_000
    checked = "1970-01-01T00:30:00+00:00"
    public = {
        "coverage_checks": [{"available_at_utc": checked,
                             "sources": {k: {"ok": True} for k in public_context.REQUIRED}}],
        "rate_expectations": [{"available_at_utc": checked, "quote_at_utc": checked,
                               "meeting_at_utc": "1970-01-02T00:00:00+00:00"}],
    }
    bars = [SimpleNamespace(open_time_ms=start - 300_000),
            SimpleNamespace(open_time_ms=start), SimpleNamespace(open_time_ms=start + 3_900_000)]
    coverage = validation.public_coverage(bars, start, public)
    assert coverage["evaluation_bars"] == 2
    assert coverage["covered_bars"] == 1
    assert coverage["covered_pct"] == 50


def test_new_public_health_gate_does_not_contaminate_price_baseline():
    _, cfg = frozen_strategy.load_frozen_strategy(ROOT / "config/frozen_strategy_candidate_20260917.json")
    profile = multifactor.load_profile(ROOT / "config/multifactor_candidate_20260927.json")
    sleeve = {"trades": [{"entry_time_utc": "2026-09-27T00:00:00+00:00", "exit_reason": "test"}]}

    def combine(base, adjusted, *args):
        trades = [trade for item in adjusted for trade in item["trades"]]
        return {"trades": trades, "summary": {"trades": len(trades)}}

    with patch.object(validation.sim, "simulate", return_value=sleeve), \
         patch.object(validation.sim, "simulate_timeseries_trend", return_value={"trades": []}), \
         patch.object(validation.multifactor, "apply_overlay", side_effect=lambda sleeves, *a: (sleeves, {})), \
         patch.object(validation.portfolio_risk, "combine_sleeves_with_drawdown_policy", side_effect=combine):
        result = validation.evaluate([], [], None, cfg, 0, None, profile, (), public={"coverage_checks": []})
    assert result["price_only"]["summary"]["trades"] == 1
    assert result["all_factors"]["summary"]["trades"] == 0
    assert result["all_factors"]["events"]["coverage_blocked_entries"] == 1
