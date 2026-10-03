"""Replay five-minute target execution with the same simulated account as the terminal."""
import argparse
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import os

import active_strategy
import event_risk
import execution_targets
import execution_portfolio
import frozen_strategy
import multifactor
import portfolio_risk
import public_context
import simulate_range_swing as sim
import timeseries_execution
from trading_execution import SimulationAccount, DEFAULT_SIMULATION_RULES
from paper_trade_frozen_portfolio import annotate_open_position_fractions
from download_market_snapshot import validate_contiguous


def replay(base, sleeves, cfg, start, funding_events, initial=10000, cost_multiplier=1,
           rules=None, lag_ms=3000, entry_report=None):
    causal_hourly = any(s.get("execution_timing", {}).get("model") == timeseries_execution.MODEL for s in sleeves)
    opening_prices = timeseries_execution.decision_open_prices(base, sleeves) if causal_hourly else None
    result = execution_portfolio.combine(
        base, sleeves, cfg, portfolio_risk.DrawdownRiskPolicy(), start, decision_open_prices=opening_prices)
    account = SimulationAccount(Path("unused_execution_replay.json"), persist=False,
                                cost_multiplier=cost_multiplier, allow_llm=False)
    account.reset(initial, now_ms=start)
    bar_lookup = {bar.open_time_ms: bar for bar in base}
    rates = sorted(funding_events, key=lambda row: int(row["fundingTime"]))
    rate_times = [int(row["fundingTime"]) for row in rates]
    cursor = bisect_left(rate_times, start)
    equity_curve, fills = [], []
    gate_checks = gate_allowed = complete_factors = healthy_sources = 0
    gate_reasons = {}
    for point in execution_targets.target_stream(base, sleeves, result, cfg):
        next_bar = bar_lookup.get(point["time_ms"] + 300000)
        if next_bar is None:
            continue  # No price after the final decision; never invent a final execution.
        now_ms = next_bar.open_time_ms + lag_ms
        events = []
        while cursor < len(rates) and rate_times[cursor] <= now_ms:
            events.append(rates[cursor])
            cursor += 1
        # Replay snapshots with the bar-opening trade price as a mark-price proxy.
        equity = float(point["equity"])
        target = {"signal_time_ms": point["time_ms"], "signal_price": point["price"],
                  "target_leverage": point["signed_qty"] * point["price"] / equity if equity > 0 else 0,
                  "position_id": point["position_id"], "origin_signal_time_ms": point["origin_signal_time_ms"]}
        halt = result["risk_diagnostics"]["hard_halt_time_ms"]
        report = {**(entry_report or {}),
                  "risk_diagnostics": {"hard_halt_time_ms": halt if halt is not None and halt <= now_ms else None}}
        execution = account.reconcile(target, next_bar.open, rules or DEFAULT_SIMULATION_RULES, report,
                                      funding_events=events, now_ms=now_ms)
        gate = execution["execution_entry_gate"]
        if gate["status"] != "not_configured":
            gate_checks += 1
            gate_allowed += bool(gate["allowed"])
            complete_factors += gate.get("factor", {}).get("coverage", 0) >= 1 - 1e-12
            healthy_sources += bool(gate.get("public_sources", {}).get("healthy"))
            for reason in gate["reasons"]:
                gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
        for field in ("fill", "risk_fill"):
            if execution[field]:
                fills.append(execution[field])
        state = account.load()
        equity_curve.append({"time_ms": now_ms, "equity": execution["account"]["margin_balance"],
                             "signed_qty": state["position_qty"], "price": next_bar.open,
                             "drawdown_pct": state["account_risk"]["drawdown_pct"]})
    last = base[-1]
    now_ms = last.close_time_ms
    events = rates[cursor:bisect_left(rate_times, now_ms + 1)]
    terminal = account.observe(last.close, now_ms, events, rules=rules)
    if terminal["fill"]:
        fills.append(terminal["fill"])
    snapshot = account.snapshot(last.close)
    state = snapshot["state"]
    equity = snapshot["account"]["margin_balance"]
    # Keep the endpoint open and charge a liquidation estimate only in a separate metric.
    close_fee = abs(state["position_qty"] * last.close) * float(os.getenv("SIM_TAKER_FEE", ".00045")) * cost_multiplier
    close_slip = abs(state["position_qty"] * last.close) * float(os.getenv("SIM_SLIPPAGE_BPS", "1")) / 10000 * cost_multiplier
    equity_curve.append({"time_ms": now_ms, "equity": equity, "signed_qty": state["position_qty"],
                         "price": last.close, "drawdown_pct": state["account_risk"]["drawdown_pct"]})
    assert abs(equity - (initial + state["realized_pnl"] - state["fees_paid"]
                         + state["funding_pnl"] + snapshot["account"]["unrealized_pnl"])) < 1e-7
    gross = [abs(p["signed_qty"] * p["price"]) / max(p["equity"], 1e-12) for p in equity_curve]
    return {"summary": {"initial_equity": initial, "final_equity": equity,
                        "total_return_pct": (equity / initial - 1) * 100,
                        "estimated_liquidated_return_pct": ((equity-close_fee-close_slip)/initial-1)*100,
                        "max_drawdown_pct": state["max_drawdown_pct"],
                        "fills": state["fill_count_total"], "fees": state["fees_paid"],
                        "funding_pnl": state["funding_pnl"], "unrealized_pnl": snapshot["account"]["unrealized_pnl"],
                        "average_gross_leverage": sum(gross)/len(gross) if gross else 0,
                        "open_quantity": state["position_qty"], "account_risk": state["account_risk"]},
            "equity_curve": equity_curve, "fills": fills,
            "funding_settlements": state["funding_settlements"],
            "research_only": True, "places_orders": False,
            "execution_model": timeseries_execution.MODEL if causal_hourly else "legacy_research",
            "entry_gate_diagnostics": {
                "execution_checks": gate_checks, "allowed_checks": gate_allowed,
                "blocked_checks": gate_checks-gate_allowed, "reason_counts": gate_reasons,
                "complete_factor_checks": complete_factors, "healthy_public_checks": healthy_sources,
                "complete_factor_coverage_pct": 100*complete_factors/gate_checks if gate_checks else None,
                "public_health_coverage_pct": 100*healthy_sources/gate_checks if gate_checks else None,
                "permission_side": "requested target direction; long when target is flat",
            },
            "limitations": ["Next five-minute bar open proxies the execution mark price after a three-second settlement delay.",
                            "Risk monitoring is sampled every five minutes; production monitors every thirty seconds.",
                            ("Factors, events and source health use first-seen archives and are rechecked at each execution timestamp."
                             if entry_report else "Sleeves are supplied by the caller; no current entry sources are configured."),
                            "Current lot/minimum-notional rules are applied historically; historical rule changes are unverified.",
                            "Optional LLM decisions and true order-book matching are not replayed."]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-snapshot", type=Path, required=True)
    parser.add_argument("--profile", type=Path, help="Defaults to the terminal's selected strategy")
    parser.add_argument("--factor-snapshot", type=Path,
                        default=sim.repo_root()/"data/snapshots/multifactor_latest.json.gz")
    parser.add_argument("--event-snapshot", type=Path)
    parser.add_argument("--funding-snapshot", type=Path, required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--initial-equity", type=float, help="Defaults to the selected strategy's account size")
    parser.add_argument("--cost-multiplier", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run_selected_strategy(args):
    root = sim.repo_root()
    profile_path = args.profile or active_strategy.candidate_path()
    profile = multifactor.load_profile(profile_path)
    if profile["availability_mode"] != "first_seen":
        raise ValueError("Selected strategy backtest requires first_seen factors")
    if not profile.get("public_context_enabled") or not profile.get("policy_expectations_enabled"):
        raise ValueError("Selected strategy backtest requires public context and policy expectations")
    event_path = args.event_snapshot or root/profile["event_snapshot"]
    manifest_path = root/profile["base_manifest"]
    initial = args.initial_equity if args.initial_equity is not None else profile["risk_limits"]["starting_equity_usdt"]
    if not math.isfinite(initial) or initial <= 0 or not math.isfinite(args.cost_multiplier) or args.cost_multiplier <= 0:
        raise ValueError("Account size and cost multiplier must be positive and finite")
    policy = portfolio_risk.DrawdownRiskPolicy()
    if (profile["risk_limits"]["soft_drawdown_start_pct"] != policy.soft_start_pct
            or profile["risk_limits"]["hard_drawdown_stop_pct"] != policy.hard_stop_pct
            or profile["risk_limits"]["portfolio_leverage_cap"] != 2):
        raise ValueError("Selected strategy risk limits must match terminal execution")
    if tuple(profile["strategy_modes"]) != ("trend", "timeseries_trend"):
        raise ValueError("Unsupported selected strategy modules")
    # Freeze mutable source files before replay; dynamic entry checks read these
    # same captured inputs, never a background refresh halfway through a run.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = args.output.parent / (args.output.stem + "_inputs_" + stamp)
    archive.mkdir(parents=True)
    sources = {"profile": profile_path, "manifest": manifest_path,
               "market": args.market_snapshot, "factor": args.factor_snapshot,
               "events": event_path, "funding": args.funding_snapshot}
    if args.profile is None:
        sources["selection"] = root/"config/active_simulation_candidate.json"
    captured = {}
    for name, path in sources.items():
        destination = archive/(name + "_" + path.name)
        destination.write_bytes(path.read_bytes())
        captured[name] = destination.resolve()
    captured_profile = multifactor.load_profile(captured["profile"])
    if multifactor.profile_hash(captured_profile) != multifactor.profile_hash(profile):
        raise ValueError("Selected strategy changed while capturing inputs")
    profile = captured_profile
    manifest, cfg = frozen_strategy.load_frozen_strategy(captured["manifest"])
    cfg = replace(cfg,max_drawdown_stop_pct=0)
    data,funding,_ = sim.load_market_snapshot(captured["market"])
    start,end = sim._utc_ms(args.start_utc),sim._utc_ms(args.end_utc)
    data = {name:[c for c in bars if start-45*sim.MS_PER_DAY <= c.open_time_ms and c.close_time_ms < end]
            for name,bars in data.items() if name in ("5m",cfg.timeseries_timeframe)}
    for name,bars in data.items():
        validate_contiguous(name,bars)
    if (not data.get('5m') or not data.get(cfg.timeseries_timeframe)
            or start-45*sim.MS_PER_DAY < data['5m'][0].open_time_ms-300000
            or start >= end):
        raise ValueError("Need 45 warmup days and a non-empty evaluation window")
    if data['5m'][-1].close_time_ms + 1 != end:
        raise ValueError("Market snapshot does not cover the requested ending boundary")
    asof = data['5m'][-1].close_time_ms
    opening = timeseries_execution.opening_from_base(data['5m'], data[cfg.timeseries_timeframe][-1].close_time_ms+1, asof)
    sleeves = [sim.simulate(data['5m'],replace(cfg,strategy_modes=('trend',)),start,None,funding),
               timeseries_execution.build_sleeve(data[cfg.timeseries_timeframe],replace(cfg,strategy_modes=('timeseries_trend',)),
                   start,funding,opening=opening,asof_ms=asof)]
    annotate_open_position_fractions(sleeves)
    execution_targets.prepare_sleeves(sleeves)
    factor = multifactor.load_snapshot(captured["factor"], "first_seen")
    public = json.loads(captured["events"].read_text(encoding="utf-8"))
    checks = public.get("coverage_checks", [])
    if not checks or start < event_risk._utc_ms(checks[0]["available_at_utc"]):
        raise ValueError("No first-seen public context at the requested start; missing history cannot be backfilled")
    events = event_risk.events_from_payload(public)
    adjusted,factor_diagnostics = multifactor.apply_overlay(sleeves,factor,profile,public)
    adjusted,event_diagnostics = event_risk.apply_event_overlay(adjusted,events)
    adjusted,coverage_blocked = public_context.apply_entry_coverage(adjusted,public)
    entry_report = {"execution_entry_context": {
        "factor_profile": str(captured["profile"]), "factor_profile_sha256": multifactor.profile_hash(profile),
        "factor_snapshot": str(captured["factor"]), "event_snapshot": str(captured["events"]),
    }}
    funding_payload = json.loads(captured["funding"].read_text())
    if sim._utc_ms(funding_payload['start_utc']) > start or sim._utc_ms(funding_payload['end_utc']) < end:
        raise ValueError("Funding snapshot does not cover the requested window")
    report = replay(data['5m'],adjusted,cfg,start,funding_payload['events'],initial,args.cost_multiplier,
                    entry_report=entry_report)
    report.update({"candidate_id":profile['candidate_id'], "profile_sha256":multifactor.profile_hash(profile),
                   "freeze_id":manifest['freeze_id'],"start_utc":args.start_utc,"end_utc_exclusive":args.end_utc,
                   "factor_groups":profile['groups'], "factor_availability_mode":"first_seen",
                   "factor_diagnostics":factor_diagnostics, "event_diagnostics":event_diagnostics,
                   "public_coverage_blocked_entries":coverage_blocked,
                   "raw_price_signal_records":sum(len(s['trades']) for s in sleeves),
                   "cost_multiplier":args.cost_multiplier,
                   "captured_inputs":{name:str(path) for name,path in captured.items()},
                   "input_sha256":{str(p):frozen_strategy.sha256_file(p) for p in captured.values()},
                   "code_sha256":{name:frozen_strategy.sha256_file(root/'scripts'/name) for name in (
                       'backtest_execution.py','active_strategy.py','multifactor.py','event_risk.py','public_context.py',
                       'timeseries_execution.py','execution_portfolio.py','execution_targets.py','trading_execution.py',
                       'execution_ledger.py','portfolio_risk.py','simulate_range_swing.py')},
                   "limitations":report['limitations'] + [
                       "Missing or expired first-seen inputs block entries; coverage gaps are not profitable-strategy evidence.",
                       "Filtered sleeve trades are precomputed; rejected-entry opportunity paths are not fully resimulated.",
                       "Current parameters applied before activation form a retrospective diagnostic, not actual forward returns."]})
    sim.save_json(args.output,report)
    return report


def main():
    args = parse_args()
    report = run_selected_strategy(args)
    print("Strategy:",report['candidate_id'])
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
