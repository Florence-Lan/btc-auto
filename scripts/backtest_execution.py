"""Replay five-minute target execution with the same simulated account as the terminal."""
import argparse
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path
import json
import os

import execution_targets
import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
from trading_execution import SimulationAccount, DEFAULT_SIMULATION_RULES


def replay(base, sleeves, cfg, start, funding_events, initial=10000, cost_multiplier=1,
           rules=None, lag_ms=3000):
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(
        base, sleeves, cfg, portfolio_risk.DrawdownRiskPolicy(), start)
    account = SimulationAccount(Path("unused_execution_replay.json"), persist=False,
                                cost_multiplier=cost_multiplier, allow_llm=False)
    account.reset(initial, now_ms=start)
    bar_lookup = {bar.open_time_ms: bar for bar in base}
    rates = sorted(funding_events, key=lambda row: int(row["fundingTime"]))
    rate_times = [int(row["fundingTime"]) for row in rates]
    cursor = bisect_left(rate_times, start)
    equity_curve, fills = [], []
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
        report = {"risk_diagnostics": {"hard_halt_time_ms": halt if halt is not None and halt <= now_ms else None}}
        execution = account.reconcile(target, next_bar.open, rules or DEFAULT_SIMULATION_RULES, report,
                                      funding_events=events, now_ms=now_ms)
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
            "limitations": ["Next five-minute bar open proxies the execution mark price after a three-second settlement delay.",
                            "Risk monitoring is sampled every five minutes; production monitors every thirty seconds.",
                            "Signals use the frozen research engine and reconstructed macro history, not archived live decisions.",
                            "Current lot/minimum-notional rules are applied historically; historical rule changes are unverified.",
                            "Optional LLM decisions and true order-book matching are not replayed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-snapshot", type=Path, required=True)
    parser.add_argument("--macro-snapshot", type=Path, required=True)
    parser.add_argument("--funding-snapshot", type=Path, required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--initial-equity", type=float, default=10000)
    parser.add_argument("--cost-multiplier", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = sim.repo_root()/"config/frozen_strategy_candidate_20260917.json"
    manifest, cfg = frozen_strategy.load_frozen_strategy(manifest_path)
    cfg = replace(cfg,max_drawdown_stop_pct=0)
    data,funding,_ = sim.load_market_snapshot(args.market_snapshot)
    start,end = sim._utc_ms(args.start_utc),sim._utc_ms(args.end_utc)
    data = {name:[c for c in bars if c.close_time_ms < end] for name,bars in data.items()}
    if start < data['5m'][0].open_time_ms + 45*sim.MS_PER_DAY or start >= end:
        raise ValueError("Need 45 warmup days and a non-empty evaluation window")
    if data['5m'][-1].close_time_ms + 1 != end:
        raise ValueError("Market snapshot does not cover the requested ending boundary")
    sleeves = [sim.simulate(data['5m'],replace(cfg,strategy_modes=('trend',)),start,None,funding),
               sim.simulate_timeseries_trend(data['1h'],cfg,start,funding)]
    execution_targets.prepare_sleeves(sleeves)
    adjusted,_ = macro_regime.apply_macro_overlay(sleeves,macro_regime.load_macro_snapshot(args.macro_snapshot),
                                                  enabled_factors=('vix','dollar','metals','sentiment'))
    funding_payload = json.loads(args.funding_snapshot.read_text())
    if sim._utc_ms(funding_payload['start_utc']) > start or sim._utc_ms(funding_payload['end_utc']) < end:
        raise ValueError("Funding snapshot does not cover the requested window")
    report = replay(data['5m'],adjusted,cfg,start,funding_payload['events'],args.initial_equity,args.cost_multiplier)
    report.update({"freeze_id":manifest['freeze_id'],"start_utc":args.start_utc,"end_utc_exclusive":args.end_utc,
                   "cost_multiplier":args.cost_multiplier,
                   "input_sha256":{str(p):frozen_strategy.sha256_file(p) for p in
                                   (manifest_path,args.market_snapshot,args.macro_snapshot,args.funding_snapshot)}})
    sim.save_json(args.output,report)
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
