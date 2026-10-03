"""Fixed-parameter, cost-aware ablation; never promotes research to live trading."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import event_risk
import active_strategy
import frozen_strategy
import macro_regime
import multifactor
import portfolio_risk
import public_context
import simulate_range_swing as sim
from validate_macro_overlay import compact_summary


def evaluate(base, trend, funding, cfg, start, snapshot, profile, events, legacy_snapshot=None, public=None):
    sleeves = [
        sim.simulate(base, replace(cfg, strategy_modes=("trend",)), start, None, funding),
        sim.simulate_timeseries_trend(trend, replace(cfg, strategy_modes=("timeseries_trend",)), start, funding),
    ]
    outputs = {}
    variants = {"price_only": None, "all_factors": profile}
    # Leave-one-group-out measures marginal effect while holding all other parameters fixed.
    variants.update({f"without_{name}": {**profile, "groups": {
        k: v for k, v in profile["groups"].items() if k != name
    }} for name in profile["groups"]})
    if legacy_snapshot is not None:
        legacy_adjusted, legacy_diag = macro_regime.apply_macro_overlay(
            sleeves, legacy_snapshot, enabled_factors=("vix", "dollar", "metals", "sentiment"),
        )
        variants["legacy_macro"] = "legacy"
    for name, variant in variants.items():
        diagnostics = None
        if variant == "legacy":
            adjusted, diagnostics = legacy_adjusted, legacy_diag
        elif variant:
            adjusted, diagnostics = multifactor.apply_overlay(sleeves, snapshot, variant, public)
        else:
            adjusted = sleeves
        # Full public context belongs only to the new candidate; applying it to the
        # baselines would no longer compare against the original strategies.
        candidate_public = public if isinstance(variant, dict) else None
        applicable_events = events if public is None or candidate_public is not None else ()
        adjusted, event_diag = event_risk.apply_event_overlay(adjusted, applicable_events)
        if candidate_public is not None:
            adjusted, blocked = public_context.apply_entry_coverage(adjusted, public)
            event_diag["coverage_blocked_entries"] = blocked
        result = portfolio_risk.combine_sleeves_with_drawdown_policy(
            base, adjusted, cfg, portfolio_risk.DrawdownRiskPolicy(), start,
        )
        outputs[name] = {"summary": compact_summary(result["summary"]),
                         "factors": diagnostics, "events": event_diag,
                         "closed_trades": sum(t["exit_reason"] != "end" for t in result["trades"]),
                         "synthetic_end_trades": sum(t["exit_reason"] == "end" for t in result["trades"])}
    return outputs


def public_coverage(base, start, public):
    eligible = [bar for bar in base if bar.open_time_ms >= start]
    covered = sum(public_context.health_at(public, bar.open_time_ms)[0]
                  and public_context.expectation_at(public, bar.open_time_ms) is not None
                  for bar in eligible)
    return {"evaluation_bars": len(eligible), "covered_bars": covered,
            "covered_pct": covered / len(eligible) * 100 if eligible else 0.0,
            "first_check_utc": public["coverage_checks"][0]["available_at_utc"],
            "last_check_utc": public["coverage_checks"][-1]["available_at_utc"],
            "policy": "actual first-seen coverage; outages and gaps remain unavailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, help="Defaults to the terminal's selected strategy")
    parser.add_argument("--factor-snapshot", type=Path, default=Path("data/snapshots/multifactor_latest.json.gz"))
    parser.add_argument("--market-snapshot", type=Path, required=True)
    parser.add_argument("--legacy-macro-snapshot", type=Path)
    parser.add_argument("--event-snapshot", type=Path)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/validation/selected_strategy_factor_validation.json"))
    args = parser.parse_args()
    args.profile = args.profile or active_strategy.candidate_path()
    profile = multifactor.load_profile(args.profile)
    manifest, cfg = frozen_strategy.load_frozen_strategy(sim.repo_root() / profile["base_manifest"])
    cfg = replace(cfg, max_drawdown_stop_pct=0)
    intervals, funding, metadata = sim.load_market_snapshot(args.market_snapshot)
    start = event_risk._utc_ms(args.start_utc)
    base = intervals["5m"]
    trend = intervals[cfg.timeseries_timeframe]
    if start < base[0].open_time_ms + 45 * sim.MS_PER_DAY or start >= base[-1].open_time_ms:
        raise ValueError("Need 45 days warmup before start, plus evaluation candles")
    factor_mode = "first_seen" if profile.get("public_context_enabled") else "reconstructed"
    snapshot = multifactor.load_snapshot(args.factor_snapshot, factor_mode)
    legacy_snapshot = macro_regime.load_macro_snapshot(args.legacy_macro_snapshot) if args.legacy_macro_snapshot else None
    event_path = args.event_snapshot or sim.repo_root() / profile["event_snapshot"]
    events = event_risk.load_event_snapshot(event_path)
    public = None
    if profile.get("public_context_enabled"):
        public = json.loads(event_path.read_text())
        checks = public.get("coverage_checks", [])
        if not checks or start < event_risk._utc_ms(checks[0]["available_at_utc"]):
            raise ValueError("Public news/calendar/expectations history does not cover the requested start; cannot backfill current context into historical decisions")
    results = {}
    for cost in (1, 2):
        local = replace(cfg, **{key: getattr(cfg, key) * cost for key in (
            "maker_fee", "taker_fee", "entry_slippage_bps", "exit_slippage_bps", "depth_impact_bps"
        )})
        results[str(cost)] = evaluate(base, trend, funding, local, start, snapshot, profile, events, legacy_snapshot, public)
        for name in ("price_only", "legacy_macro", "all_factors"):
            if name in results[str(cost)]:
                s = results[str(cost)][name]["summary"]
                print(f"cost={cost} {name}: return={s['total_return_pct']:.4f}% "
                      f"drawdown={s['max_drawdown_pct']:.4f}% trades={s['trades']}", flush=True)
    baseline = results["1"]["price_only"]["summary"]
    candidate = results["1"]["all_factors"]["summary"]
    stress = results["2"]["all_factors"]["summary"]
    diagnostics = results["1"]["all_factors"]["factors"]
    coverage = public_coverage(base, start, public) if public is not None else None
    gates = {
        "positive_net_return": candidate["total_return_pct"] > 0,
        "return_at_least_price_baseline": candidate["total_return_pct"] >= baseline["total_return_pct"],
        "drawdown_no_worse_than_baseline": candidate["max_drawdown_pct"] <= baseline["max_drawdown_pct"],
        "drawdown_under_15pct": candidate["max_drawdown_pct"] <= 15,
        "positive_double_cost_return": stress["total_return_pct"] > 0,
        "profit_factor_at_least_1_3": (candidate["profit_factor"] or 0) >= 1.3,
        "at_least_30_closed_trades": results["1"]["all_factors"]["closed_trades"] >= 30,
        "factor_coverage_at_least_95pct": all(v >= 95 for v in diagnostics["group_coverage_pct"].values()),
        "legacy_comparison_available": legacy_snapshot is not None,
    }
    if coverage is not None:
        gates["public_time_coverage_at_least_95pct"] = coverage["covered_pct"] >= 95
    if legacy_snapshot is not None:
        legacy_summary = results["1"]["legacy_macro"]["summary"]
        gates["return_at_least_legacy_baseline"] = candidate["total_return_pct"] >= legacy_summary["total_return_pct"]
        gates["drawdown_no_worse_than_legacy_baseline"] = candidate["max_drawdown_pct"] <= legacy_summary["max_drawdown_pct"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": profile["candidate_id"], "profile_sha256": multifactor.profile_hash(profile),
        "base_freeze_id": manifest["freeze_id"], "research_only": True, "live_orders_allowed": False,
        "evaluation_type": "first_seen_replay" if public is not None else "reconstructed_factor_research",
        "factor_availability_mode": factor_mode,
        "public_context_coverage": coverage,
        "promotion_pass": False, "research_gates": gates, "research_gates_pass": all(gates.values()),
        "market_sha256": frozen_strategy.sha256_file(args.market_snapshot),
        "evaluation_start_utc": args.start_utc,
        "evaluation_end_utc": sim.iso_utc_from_ms(base[-1].close_time_ms),
        "factor_sha256": frozen_strategy.sha256_file(args.factor_snapshot),
        "event_snapshot_sha256": frozen_strategy.sha256_file(event_path),
        "legacy_macro_sha256": frozen_strategy.sha256_file(args.legacy_macro_snapshot) if args.legacy_macro_snapshot else None,
        "code_sha256": {name: frozen_strategy.sha256_file(sim.repo_root() / "scripts" / name)
                        for name in ("multifactor.py", "portfolio_risk.py", "validate_multifactor.py", "public_context.py", "event_risk.py")},
        "market_metadata": metadata, "factor_metadata": snapshot.metadata,
        "results_by_cost": results,
        "limitations": [
            ("First-seen replay starts only after public-context collection; sparse coverage cannot establish performance."
             if public is not None else "Reconstructed historical availability with latest FRED vintages is not prospective OOS."),
            "Short derivatives history cannot establish a stable edge across market cycles.",
            "Entry overlay scales precomputed sleeve trades; it does not resimulate rejected-signal opportunity paths or capital compounding.",
            "Legacy comparator uses the same frozen price engine, four legacy factors and corrected drawdown latch; it does not receive new public-context gates.",
            "No news coverage is implied by an empty event file. International risk proxies are incomplete.",
        ],
    }
    sim.save_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
