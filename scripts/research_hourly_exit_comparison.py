"""Offline, fixed-policy hourly exit diagnostics; never touches a runtime account.

Run: .venv/bin/python scripts/research_hourly_exit_comparison.py
All source inputs and loaded code are archived before running. No network calls,
parameter search, live orders, or selected-profile changes are made.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import shutil
import sys

import backtest_execution
from download_market_snapshot import validate_contiguous
import event_risk
from execution_ledger import attach_ledger
import execution_targets
import frozen_strategy
import multifactor
from paper_trade_frozen_portfolio import annotate_open_position_fractions
import public_context
import research_reentry
import simulate_range_swing as sim
import timeseries_execution as hourly


def build_protected_sleeve(closed, cfg, start, funding=None, *, policy,
                           opening=None, asof_ms=None):
    """Research counterpart of the selected hourly known-open constructor."""
    if policy.name == "baseline":
        return hourly.build_sleeve(closed, cfg, start, funding,
                                   opening=opening, asof_ms=asof_ms)
    step = sim.interval_to_ms(cfg.timeseries_timeframe)
    if not closed:
        raise ValueError("Hourly signal input must contain closed candles")
    asof_ms = asof_ms if asof_ms is not None else closed[-1].close_time_ms
    if any(bar.close_time_ms > asof_ms for bar in closed):
        raise ValueError("Hourly signal input must contain only closed candles")
    if any(b.open_time_ms-a.open_time_ms != step for a, b in zip(closed, closed[1:])):
        raise ValueError("Hourly signal input has a gap")
    candles = [replace(bar, volume=closed[max(0, i-1)].volume,
                       quote_volume=closed[max(0, i-1)].quote_volume)
               for i, bar in enumerate(closed)]
    if opening is not None:
        if (opening.time_ms != closed[-1].close_time_ms+1 or opening.time_ms % step
                or opening.time_ms > asof_ms or opening.observed_at_ms > asof_ms
                or opening.observed_at_ms < opening.time_ms
                or not math.isfinite(opening.price) or opening.price <= 0):
            raise ValueError("Opening does not follow the last closed hour")
        p = opening.price
        candles.append(sim.Candle(opening.time_ms, sim.iso_utc_from_ms(opening.time_ms),
            p, p, p, p, closed[-1].volume, closed[-1].quote_volume, opening.time_ms))
        if funding is not None:
            rows = [(t, r) for t, r in zip(funding.times, funding.rates)
                    if t <= closed[-1].close_time_ms]
            funding = sim.FundingHistory([t for t, r in rows], [r for t, r in rows])
    result = research_reentry.simulate_with_exit_policy(candles, cfg, start, funding, policy)
    if opening is not None and result["equity_curve"]:
        point = result["equity_curve"][-1]
        if int(point["time_ms"]) == opening.time_ms:
            point["available_time_ms"] = opening.time_ms
            result = attach_ledger(result)
            for trade in result["trades"]:
                if trade["exit_reason"] == "end":
                    trade["_ledger_exit_ms"] = opening.time_ms
                    trade["_ledger"][-1]["time_ms"] = opening.time_ms
    result["execution_timing"] = {"model": hourly.MODEL,
        "opening": asdict(opening) if opening else None,
        "liquidity": "previous_closed_hour"}
    result["research_exit_policy"] = asdict(policy)
    return result


@contextmanager
def fixed_execution_environment(execution):
    values = {"SIM_TAKER_FEE": str(execution["fee_rate"]),
              "SIM_SLIPPAGE_BPS": str(execution["slippage_bps"]),
              "SIM_MAX_LEVERAGE": str(execution["max_leverage"]),
              "SIM_MAX_NOTIONAL_USDT": str(execution["max_notional_usdt"])}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def closed_execution_episodes(curve):
    """Count flat-to-position-to-flat/sign-flip episodes, not rebalance fills."""
    previous, count = 0, 0
    for point in curve:
        qty = float(point["signed_qty"])
        side = 1 if qty > 1e-12 else -1 if qty < -1e-12 else 0
        if previous and side != previous:
            count += 1
        previous = side
    return count


def checks(candidate, baseline):
    normal, stressed = candidate["1"], candidate["2"]
    reference = baseline["1"]
    return {
        "normal_return_at_least_baseline": normal["total_return_pct"] >= reference["total_return_pct"],
        "normal_drawdown_no_worse_than_baseline": normal["max_drawdown_pct"] <= reference["max_drawdown_pct"],
        "positive_double_cost_return": stressed["total_return_pct"] > 0,
        "normal_fees_no_more_than_baseline": normal["fees"] <= reference["fees"],
        "at_least_20_closed_execution_episodes": normal["closed_execution_episodes"] >= 20,
    }


def capture_inputs(root, plan_path, plan, output):
    sources = {plan_path.resolve()}
    sources.update(root/plan[key] for key in ("profile", "manifest", "factor_snapshot", "event_snapshot"))
    sources.update(root/path for path in plan["policy_sources"])
    for window in plan["windows"].values():
        sources.update((root/window["market_snapshot"], root/window["funding_snapshot"]))
    captured, provenance = {}, {}
    for source in sorted(sources):
        relative = source.relative_to(root)
        dest = output/"inputs"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        digest = frozen_strategy.sha256_file(dest)
        if digest != frozen_strategy.sha256_file(source):
            raise ValueError(f"Input changed while capturing: {relative}")
        captured[str(relative)] = dest.resolve()
        provenance[str(relative)] = {"captured": str(dest.relative_to(root)), "sha256": digest}
    code = {}
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value and Path(value).resolve().parent == root/"scripts" and str(value).endswith(".py"):
            paths.add(Path(value).resolve())
    for source in sorted(paths):
        relative = source.relative_to(root)
        dest = output/"code"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        code[str(relative)] = {"captured": str(dest.relative_to(root)),
                               "sha256": frozen_strategy.sha256_file(dest)}
    return captured, provenance, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=sim.repo_root()/"config/hourly_exit_comparison_20261003.json")
    parser.add_argument("--output-dir", type=Path, default=sim.repo_root()/"data/validation/hourly_exit_comparison_20261003")
    args = parser.parse_args(argv)
    root, output = sim.repo_root(), args.output_dir.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; preserve previous research evidence")
    plan = json.loads(args.plan.read_text())
    if plan["live_orders_allowed"] or not plan["research_only"]:
        raise ValueError("This runner supports research-only plans")
    output.mkdir(parents=True)
    captured, provenance, code = capture_inputs(root, args.plan, plan, output)
    # Require equality to the already fixed policies; no new tuning is accepted.
    old_exit = json.loads(captured[plan["policy_sources"][0]].read_text())
    old_execution = json.loads(captured[plan["policy_sources"][1]].read_text())
    fixed = old_exit["fixed_parameters"]
    if (plan["policies"]["protective_exit_only"] != {"name": "volatility", "stop_atr": fixed["stop_atr"], "trail_atr": fixed["trail_atr"]}
            or plan["policies"]["protective_reentry_fixed"] != old_execution["protected_policy"]):
        raise ValueError("Protective policies differ from the prior fixed plans")
    profile = multifactor.load_profile(captured[plan["profile"]])
    manifest, cfg = frozen_strategy.load_frozen_strategy(captured[plan["manifest"]])
    cfg = replace(cfg, max_drawdown_stop_pct=0)
    factors = multifactor.load_snapshot(captured[plan["factor_snapshot"]], "first_seen")
    public = json.loads(captured[plan["event_snapshot"]].read_text())
    events = event_risk.events_from_payload(public)
    e = plan["execution"]
    rules = {"step_size": Decimal(str(e["quantity_step"])),
             "min_qty": Decimal(str(e["minimum_quantity"])),
             "max_qty": Decimal("120"), "min_notional": Decimal(str(e["minimum_notional_usdt"]))}
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
              "input_sha256": provenance, "code_sha256": code, "windows": {},
              "research_only": True, "places_orders": False, "promotion_pass": False,
              "execution_model": hourly.MODEL, "freeze_id": manifest["freeze_id"],
              "profile_sha256": multifactor.profile_hash(profile),
              "limitations": plan["limitations"], "status": "running"}
    sim.save_json(output/"report.json", report)
    market_cache, funding_cache = {}, {}
    with fixed_execution_environment(e):
        for label, window in plan["windows"].items():
            market_path, funding_path = window["market_snapshot"], window["funding_snapshot"]
            if market_path not in market_cache:
                market_cache[market_path] = sim.load_market_snapshot(captured[market_path])
            if funding_path not in funding_cache:
                funding_cache[funding_path] = json.loads(captured[funding_path].read_text())
            original, funding, _ = market_cache[market_path]
            rates = funding_cache[funding_path]
            start, end = map(sim._utc_ms, window["utc"])
            data = {key: [bar for bar in bars if start-45*sim.MS_PER_DAY <= bar.open_time_ms and bar.close_time_ms < end]
                    for key, bars in original.items() if key in ("5m", cfg.timeseries_timeframe)}
            for key, bars in data.items():
                validate_contiguous(key, bars)
            base, closed = data["5m"], data[cfg.timeseries_timeframe]
            if (not base or not closed or base[-1].close_time_ms+1 != end
                    or base[0].open_time_ms > start-45*sim.MS_PER_DAY
                    or sim._utc_ms(rates["start_utc"]) > start or sim._utc_ms(rates["end_utc"]) < end):
                raise ValueError(f"Incomplete frozen coverage: {label}")
            asof = base[-1].close_time_ms
            opening = hourly.opening_from_base(base, closed[-1].close_time_ms+1, asof)
            tactical = sim.simulate(base, replace(cfg, strategy_modes=("trend",)), start, None, funding)
            rows = {}
            for name, policy_values in plan["policies"].items():
                # Each overlay mutates trade annotations; give variants fresh sleeves.
                sleeves = [json.loads(json.dumps(tactical)), build_protected_sleeve(
                    closed, replace(cfg, strategy_modes=("timeseries_trend",)), start, funding,
                    policy=research_reentry.ExitPolicy(**policy_values), opening=opening, asof_ms=asof)]
                annotate_open_position_fractions(sleeves)
                execution_targets.prepare_sleeves(sleeves)
                raw_signals = sum(len(s["trades"]) for s in sleeves)
                raw_hourly = sleeves[-1]["trades"]
                evidence = {"raw_price_signal_records": raw_signals,
                    "raw_hourly_trades": [{key: t[key] for key in ("side", "entry_time_utc", "exit_time_utc", "exit_reason")}
                                           for t in raw_hourly]}
                entry_report = None
                if window["scope"] == "current_six_factor":
                    if start < event_risk._utc_ms(public["coverage_checks"][0]["available_at_utc"]):
                        raise ValueError("No first-seen public archive at window start")
                    sleeves, fd = multifactor.apply_overlay(sleeves, factors, profile, public)
                    sleeves, ed = event_risk.apply_event_overlay(sleeves, events)
                    sleeves, blocked = public_context.apply_entry_coverage(sleeves, public)
                    evidence.update({"factor_diagnostics": fd, "event_diagnostics": ed,
                                     "public_coverage_blocked_entries": blocked})
                    entry_report = {"execution_entry_context": {
                        "factor_profile": str(captured[plan["profile"]]),
                        "factor_profile_sha256": multifactor.profile_hash(profile),
                        "factor_snapshot": str(captured[plan["factor_snapshot"]]),
                        "event_snapshot": str(captured[plan["event_snapshot"]])}}
                rows[name] = {}
                for cost in plan["cost_multipliers"]:
                    result = backtest_execution.replay(base, sleeves, cfg, start, rates["events"],
                        initial=plan["initial_equity_usdt"], cost_multiplier=cost,
                        rules=rules, lag_ms=3000, entry_report=entry_report)
                    assert result["execution_model"] == hourly.MODEL
                    result.update({"window": window, "variant": name, "policy": policy_values,
                                   "cost_multiplier": cost, "signal_evidence": evidence})
                    result["summary"]["closed_execution_episodes"] = closed_execution_episodes(result["equity_curve"])
                    dest = output/f"{label}_{name}_cost{cost}.json"
                    sim.save_json(dest, result)
                    rows[name][str(cost)] = {**result["summary"],
                        "entry_gate_diagnostics": result["entry_gate_diagnostics"],
                        "full_result": str(dest.relative_to(root)), "sha256": frozen_strategy.sha256_file(dest)}
                    print(label, name, cost, f"return={result['summary']['total_return_pct']:.4f}%",
                          f"dd={result['summary']['max_drawdown_pct']:.4f}%",
                          f"fills={result['summary']['fills']}", flush=True)
            report["windows"][label] = {"scope": window["scope"], "utc": window["utc"],
                "results": rows, "protective_checks": {name: checks(rows[name], rows["baseline"])
                    for name in rows if name != "baseline"}}
            sim.save_json(output/"report.json", report)
    for relative, evidence in code.items():
        if frozen_strategy.sha256_file(root/relative) != evidence["sha256"]:
            raise RuntimeError(f"Code changed during comparison: {relative}; rerun in a new directory")
    report.update({"status": "complete", "qualification_status": "insufficient_prospective_evidence",
                   "code_unchanged_during_run": True,
                   "reproduction_command": ".venv/bin/python scripts/research_hourly_exit_comparison.py --output-dir data/validation/hourly_exit_comparison_20261003_rerun"})
    sim.save_json(output/"report.json", report)
    print("Report:", output/"report.json", flush=True)


if __name__ == "__main__":
    main()
