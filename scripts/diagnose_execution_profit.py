"""Attribute existing execution paths; compare fixed controls without changing paper trading."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from dataclasses import replace
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_execution
import diagnose_strategy_losses as diagnosis
import execution_targets
import frozen_strategy
import macro_regime
import simulate_range_swing as sim


def attribution(result):
    summary = result["summary"]
    events = [(int(datetime.fromisoformat(f["time_utc"]).timestamp()*1000), 1, f)
              for f in result["fills"]]
    events += [(f["time_ms"], 0, f) for f in result["funding_settlements"]]
    qty, active, episodes = Decimal(0), None, []
    for timestamp, kind, row in sorted(events, key=lambda x: x[:2]):
        if kind == 0:
            assert abs(float(qty)-row["signed_qty"]) < 1e-8
            if active:
                active["funding"] += row["payment"]
            else:
                assert abs(row["payment"]) < 1e-8
            continue
        delta = Decimal(str(row["quantity"])) * (1 if row["side"] == "BUY" else -1)
        new_qty = qty + delta
        closing = min(abs(qty), abs(delta)) if qty*delta < 0 else Decimal(0)
        fee_close = row["fee"] * float(closing / abs(delta))
        if active:
            active["gross_realized"] += row["realized_pnl"]
            active["fees"] += row["fee"] if qty*new_qty > 0 else fee_close
        if active and qty*new_qty <= 0:
            active.update(exit_ms=timestamp, closed=True)
            episodes.append(active)
            active = None
        if new_qty != 0 and active is None:
            active = {"entry_ms": timestamp, "side": "LONG" if new_qty > 0 else "SHORT",
                      "gross_realized": 0.0, "fees": row["fee"]-fee_close,
                      "funding": 0.0, "unrealized": 0.0, "closed": False}
        qty = new_qty
    if active:
        active["unrealized"] = summary["unrealized_pnl"]
        active["exit_ms"] = result["equity_curve"][-1]["time_ms"]
        episodes.append(active)
    for e in episodes:
        e["net"] = e["gross_realized"] - e["fees"] + e["funding"] + e["unrealized"]
        e["holding_hours"] = (e["exit_ms"]-e["entry_ms"])/3600000
    pnl = summary["final_equity"]-summary["initial_equity"]
    assert abs(sum(e["net"] for e in episodes)-pnl) < 1e-6
    assert abs(sum(e["fees"] for e in episodes)-summary["fees"]) < 1e-6
    closed = [e for e in episodes if e["closed"]]
    wins = sum(e["net"] for e in closed if e["net"]>0)
    losses = -sum(e["net"] for e in closed if e["net"]<0)
    best = max((e["net"] for e in episodes), default=0)
    gross_wins = sum(max(0,e["net"]) for e in episodes)
    curve = result["equity_curve"]
    initial = summary["initial_equity"]
    before_fees = pnl + summary["fees"]
    return {"summary": summary, "episodes": episodes,
            "closed_episodes": len(closed),
            "closed_episode_profit_factor": wins/losses if losses else None,
            "net_before_fees_on_same_fill_path": before_fees,
            "fee_share_of_before_fee_net_pct": 100*summary["fees"]/before_fees if before_fees>0 else None,
            "best_episode_net": best,
            "best_episode_share_of_positive_net_pct": best/gross_wins*100 if gross_wins else None,
            "net_excluding_best_episode": pnl-best,
            "return_to_max_drawdown": summary["total_return_pct"]/summary["max_drawdown_pct"] if summary["max_drawdown_pct"] else None,
            "invested_bar_pct": 100*sum(abs(p["signed_qty"])>1e-10 for p in curve)/len(curve),
            "notes": ["An episode is a continuous same-direction NET account position, not an individual strategy trade.",
                      "Reversal fees are split between old and new inventory; partial exits and funding belong to their inventory episode.",
                      "The final open episode is marked to market. Fee attribution is accounting on the existing fill path, not a zero-cost counterfactual.",
                      "Removing the best episode is a concentration diagnostic, not an independently replayed strategy.",
                      "Return/max drawdown uses the stated window return; 30d/90d values are NOT annualized Calmar ratios."]}


def main():
    plan = json.loads((ROOT/"config/execution_validation_20261001.json").read_text())
    _, cfg = frozen_strategy.load_frozen_strategy(ROOT/plan["manifest"])
    cfg = replace(cfg,max_drawdown_stop_pct=0)
    macro = macro_regime.load_macro_snapshot(ROOT/plan["macro_snapshot"])
    rates = json.loads((ROOT/plan["funding_snapshot"]).read_text())["events"]
    report = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "research_only":True,"paper_account_unchanged":True,"windows":{},
              "comparison": "Fixed 0.60 macro control from the existing October 1 validation plan; no new parameter search.",
              "limitations": "All windows already observed; fixed exposure is not exactly risk-matched; do not promote from these results.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache = {}
    for label,window in plan["windows"].items():
        path = ROOT/window["market_snapshot"]
        if path not in cache:
            cache[path] = sim.load_market_snapshot(path)
        original,funding,_ = cache[path]
        start,end = map(sim._utc_ms,window["utc"])
        data = {k:[c for c in bars if c.close_time_ms<end] for k,bars in original.items()}
        sleeves = [sim.simulate(data["5m"],replace(cfg,strategy_modes=("trend",)),start,None,funding),
                   sim.simulate_timeseries_trend(data["1h"],cfg,start,funding)]
        execution_targets.prepare_sleeves(sleeves)
        adjusted,_ = macro_regime.apply_macro_overlay(sleeves,macro,enabled_factors=("vix","dollar","metals","sentiment"))
        variants = {"current_macro":adjusted,"fixed_060_control":diagnosis.constant_scale(sleeves,.60)}
        rows = {}
        for variant,value in variants.items():
            rows[variant] = {}
            for cost in (1,2):
                result = backtest_execution.replay(data["5m"],value,cfg,start,rates,initial=1000,cost_multiplier=cost)
                rows[variant][str(cost)] = attribution(result)
                s=result["summary"]
                print(label,variant,cost,'net',round(s["total_return_pct"],4),'dd',round(s["max_drawdown_pct"],4),
                      'episodes',rows[variant][str(cost)]["closed_episodes"],flush=True)
        report["windows"][label] = rows
        (ROOT/"data/validation/profit_quality_20261002.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == "__main__":
    main()
