"""Fixed-parameter loss attribution; diagnostic counterfactuals never promote a strategy."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import frozen_strategy
import macro_regime
import multifactor
import portfolio_risk
import simulate_range_swing as sim
from validate_macro_overlay import compact_summary


def trade_key(t):
    return t["strategy"], t["side"], t["entry_time_utc"]


def run_sleeves(data, funding, cfg, start):
    checks, signals = Counter(), Counter()
    confirm = sim.higher_timeframe_trend_allows
    signal = sim.trend_signal_for_index

    def traced_confirm(*args, **kwargs):
        allowed, reason = confirm(*args, **kwargs)
        checks["passed" if allowed else reason.split("=", 1)[0]] += 1
        return allowed, reason

    def traced_signal(*args, **kwargs):
        result = signal(*args, **kwargs)
        signals["evaluations"] += 1
        signals["accepted" if result else "no_signal"] += 1
        return result

    with patch.object(sim, "higher_timeframe_trend_allows", traced_confirm), patch.object(sim, "trend_signal_for_index", traced_signal):
        tactical = sim.simulate(data["5m"], replace(cfg, strategy_modes=("trend",)), start, None, funding)
    core = sim.simulate_timeseries_trend(data["1h"], replace(cfg, strategy_modes=("timeseries_trend",)), start, funding)
    return [tactical, core], {"tactical_signal_calls": dict(signals), "higher_timeframe_checks": dict(checks)}


def constant_scale(sleeves, multiplier):
    result = []
    for sleeve in sleeves:
        from execution_ledger import attach_ledger
        sleeve = attach_ledger(sleeve)
        trades = []
        for raw in sleeve["trades"]:
            trade = dict(raw)
            for key in ("initial_qty", "pnl", "fees", "net_pnl", "funding_pnl", "slippage_cost"):
                trade[key] *= multiplier
            trades.append(trade)
        result.append({**sleeve, "trades": trades})
    return result


def combine(data, sleeves, cfg, start):
    return portfolio_risk.combine_sleeves_with_drawdown_policy(data["5m"], sleeves, cfg,
                                                               portfolio_risk.DrawdownRiskPolicy(), start)


def accounting(trades):
    net = sum(t["net_pnl"] for t in trades)
    fees = sum(t["fees"] for t in trades)
    funding = sum(t["funding_pnl"] for t in trades)
    slippage = sum(t["slippage_cost"] for t in trades)
    return {"net_pnl": net, "fees": fees, "funding_pnl": funding, "slippage_cost": slippage,
            "price_pnl_before_friction": net + fees - funding + slippage,
            "identity": "net = price_before_friction - slippage - fees + funding"}


def excursions(trade, data):
    entry = sim._utc_ms(trade["entry_time_utc"])
    exit_ms = sim._utc_ms(trade["exit_time_utc"])
    is_core = trade["strategy"].startswith("timeseries")
    if trade["exit_reason"] == "end":
        exit_ms += sim.interval_to_ms("1h" if is_core else "5m")
    # Tactical fills/stops can occur inside a bar; omit boundary bars to avoid
    # attributing pre-entry/post-exit extrema. EMA reversals execute at bar open.
    bars = [c for c in data["5m"] if (entry <= c.open_time_ms if is_core else entry < c.open_time_ms)
            and c.open_time_ms < exit_ms]
    if not bars:
        return {}
    low = min(bars, key=lambda c: c.low)
    high = max(bars, key=lambda c: c.high)
    p = trade["entry_price"]
    long = trade["side"] == "long"
    return {"max_favorable_price_pct": (high.high / p - 1) * 100 if long else (1 - low.low / p) * 100,
            "max_adverse_price_pct": (low.low / p - 1) * 100 if long else (1 - high.high / p) * 100,
            "favorable_extreme_time_utc": high.open_time_utc if long else low.open_time_utc,
            "realized_price_change_pct": (trade["avg_exit_price"] / p - 1) * (1 if long else -1) * 100,
            "economic_exit_time_utc": sim.iso_utc_from_ms(exit_ms),
            "holding_hours": (exit_ms - entry) / 3_600_000,
            "warning": "Observed intrabar extremes are not achievable exit prices or an optimized target."}


def inventory_audit(sleeves):
    findings = []
    for sleeve in sleeves:
        curve = sleeve["equity_curve"]
        times = [p["time_ms"] for p in curve]
        for trade in sleeve["trades"]:
            start, end = sim._utc_ms(trade["entry_time_utc"]), sim._utc_ms(trade["exit_time_utc"])
            interior = curve[bisect_left(times, start):bisect_left(times, end)]
            changes = [p for p in interior if 1e-12 < abs(p["signed_qty"]) < trade["initial_qty"] * .999999]
            if changes:
                findings.append({"trade": trade_key(trade), "first_partial_time_utc": sim.iso_utc_from_ms(changes[0]["time_ms"]),
                                 "minimum_remaining_fraction": min(abs(p["signed_qty"]) / trade["initial_qty"] for p in changes)})
    return {"partial_exit_trades": findings,
            "portfolio_marking_method": "Current tiered combiner reconstructs cash and remaining quantity from sleeve observations at bar close; legacy records without config/curve retain endpoint accounting and are counted in risk diagnostics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-snapshot", type=Path, required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--legacy-snapshot", type=Path, default=Path("data/snapshots/macro_backtest_20260927_0900.json.gz"))
    parser.add_argument("--factor-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, cfg = frozen_strategy.load_frozen_strategy(Path("config/frozen_strategy_candidate_20260917.json"))
    cfg = replace(cfg, max_drawdown_stop_pct=0)
    data, funding, metadata = sim.load_market_snapshot(args.market_snapshot)
    start = sim._utc_ms(args.start_utc)
    legacy = macro_regime.load_macro_snapshot(args.legacy_snapshot)
    sleeves, funnel = run_sleeves(data, funding, cfg, start)
    old, old_diagnostics = macro_regime.apply_macro_overlay(sleeves, legacy, enabled_factors=("vix", "dollar", "metals", "sentiment"))
    variants = {"price_only": sleeves, "legacy_macro": old,
                "constant_0675": constant_scale(sleeves, .675), "tactical_only": [sleeves[0]], "hourly_only": [sleeves[1]]}
    factors = None
    if args.factor_snapshot:
        profile = multifactor.load_profile(Path("config/multifactor_candidate_20260926.json"))
        snap = multifactor.load_snapshot(args.factor_snapshot, "reconstructed")
        variants["six_factors"], factors = multifactor.apply_overlay(sleeves, snap, profile)
    outputs = {name: combine(data, variant, cfg, start) for name, variant in variants.items()}
    free_cfg = replace(cfg, maker_fee=0, taker_fee=0, entry_slippage_bps=0, exit_slippage_bps=0, depth_impact_bps=0)
    free_sleeves, _ = run_sleeves(data, funding, free_cfg, start)
    outputs["zero_trading_cost_with_funding"] = combine(data, free_sleeves, free_cfg, start)
    # Fee removal is a diagnostic, not a tradable performance estimate.
    pre_cost = {}
    for name, output in outputs.items():
        pre_cost[name] = {"summary": compact_summary(output["summary"]), "accounting": accounting(output["trades"]),
                          "closed_trades": sum(t["exit_reason"] != "end" for t in output["trades"]),
                          "synthetic_end_trades": sum(t["exit_reason"] == "end" for t in output["trades"]),
                          "risk_diagnostics": output["risk_diagnostics"]}
    lookups = {name: {trade_key(t): t for t in output["trades"]} for name, output in outputs.items()}
    original = [t for sleeve in sleeves for t in sleeve["trades"]]
    per_trade = []
    for raw in sorted(original, key=lambda t: t["entry_time_utc"]):
        row = {**raw, "excursions": excursions(raw, data), "by_variant": {}}
        for name, index in lookups.items():
            trade = index.get(trade_key(raw))
            if trade:
                row["by_variant"][name] = {"net_pnl": trade["net_pnl"], "quantity_scale": trade["initial_qty"] / raw["initial_qty"]}
        per_trade.append(row)
    bars = [c for c in data["5m"] if c.open_time_ms >= start]
    report = {"start_utc": args.start_utc, "end_utc": sim.iso_utc_from_ms(bars[-1].close_time_ms),
              "initial_equity": cfg.initial_equity, "freeze_id": manifest["freeze_id"],
              "market_sha256": frozen_strategy.sha256_file(args.market_snapshot),
              "market_metadata": metadata, "variants": pre_cost, "trades": per_trade,
              "tactical_funnel": funnel, "legacy_diagnostics": old_diagnostics, "factor_diagnostics": factors,
              "inventory_audit": inventory_audit(sleeves),
              "buy_hold_price_return_pct": (bars[-1].close / bars[0].open - 1) * 100,
              "limitations": ["Fixed parameters, no optimization or promotion.",
                              "Factor history reconstructed; short-history overlay is not a full news/expectations backtest.",
                              "Trade-level attribution is descriptive of this sample, not a causal proof of future returns."]}
    sim.save_json(args.output, report)
    for name, result in pre_cost.items():
        s = result["summary"]
        print(name, f"return={s['total_return_pct']:.4f}% dd={s['max_drawdown_pct']:.4f}% trades={s['trades']}")
    print("funnel", funnel)
    print("partial exits", len(report["inventory_audit"]["partial_exit_trades"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
