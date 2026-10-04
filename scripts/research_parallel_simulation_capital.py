#!/usr/bin/env python3
"""Recheck all existing stock hypotheses at the requested independent 1000-USDT capital."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import backtest_stock_swing_120 as engine
from replay_stock_swing_per_symbol import after_close_summary
from research_stock_mechanisms import VARIANTS, DEFERRED_VARIANTS, assessment, validate_ledger, volume_diagnostics
from research_stock_swing_robustness import audit_snapshot, trade_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("config/parallel_simulation_plan_20261005.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["execution_mode"] == "simulation" and plan["live_orders_allowed"] is False
    assert len(plan["accounts"]) == 4 and len({a["state_path"] for a in plan["accounts"]}) == 4
    assert sum(a["initial_balance_usdt"] for a in plan["accounts"]) == 4000
    assert all(a["initial_balance_usdt"] == 1000 and a["max_leverage"] == 10 for a in plan["accounts"])
    profile = json.loads(Path(plan["stock_research_profile"]).read_text())
    base = json.loads(Path(profile["base_config"]).read_text())
    snapshot = json.loads(gzip.decompress(args.snapshot.read_bytes()))
    audit_snapshot(snapshot)
    if args.output_dir.exists():
        raise FileExistsError("Preserve existing research in a new directory")
    args.output_dir.mkdir(parents=True)
    variants = {**VARIANTS, **DEFERRED_VARIANTS}
    declaration = {"declared_at_utc": datetime.now(timezone.utc).isoformat(), "places_orders": False,
                   "capital_per_account_usdt": 1000, "max_leverage": 10, "variants": variants,
                   "selection": "Recheck all eight existing rules at user-requested capital; no refitting or new winner selection.",
                   "independent_holdout": False, "forward_started": False,
                   "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                   "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest()}
    (args.output_dir / "declaration.json").write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n")
    results, count, ledger_count = {}, 0, 0
    end = snapshot["end_ms_exclusive"]
    split = engine.parse_time(base["evaluation_split_utc"])
    for account in plan["accounts"]:
        symbol = account["symbol"]
        if symbol == "BTCUSDT":
            continue
        common = {**base, "symbols": [symbol], "symbol_profiles": {symbol: profile["symbol_profiles"][symbol]},
                  "initial_equity_usdt": account["initial_balance_usdt"], "leverage": account["max_leverage"],
                  "max_positions": 1, "risk_fraction_per_trade": profile["risk_fraction_per_bucket_trade"],
                  "entry_max_previous_bar_participation_fraction": .1, "execution_timeframe": "5m"}
        first = snapshot["symbols"][symbol]["start_ms"]
        experiments = {}
        for variant, extras in variants.items():
            config, runs = {**common, **extras}, {}
            cases = [(f"{w}_cost{c}", start, finish, config, c)
                     for w, start, finish in (("full", first, end), ("development", first, split),
                                              ("validation", split, end), ("recent30d", max(first, end - 30 * engine.DAY), end),
                                              ("recent60d", max(first, end - 60 * engine.DAY), end))
                     for c in (1, 2)]
            cases += [("full_slippage50bps", first, end, {**config, "adverse_slippage_fraction_assumption": .005}, 1),
                      ("full_margin5pct", first, end, {**config, "maintenance_margin_fraction_assumption": .05}, 1)]
            for name, start, finish, settings, cost in cases:
                replay = engine.simulate(snapshot, settings, start, finish, cost)
                check = validate_ledger(replay, settings, snapshot["symbols"][symbol], cost)
                summary = after_close_summary(replay)
                summary["trade_diagnostics"] = trade_diagnostics(replay["trades"], 1000)
                summary["execution_volume_diagnostics"] = volume_diagnostics(replay, snapshot["symbols"][symbol])
                runs[name] = summary
                count += 1
                ledger_count += check["closed_trades_checked"]
                # Closed ledgers stay reviewable; large equity CSVs are unnecessary for this capital check.
                engine.write_csv(args.output_dir / f"{symbol}_{variant}_{name}_trades.csv", replay["trades"])
            experiments[variant] = {"settings": extras, "runs": runs, "assessment": assessment(runs)}
            print(symbol, variant, "double_cost_pct", round(runs["full_cost2"]["estimated_close_return_pct"], 4),
                  "passed", experiments[variant]["assessment"]["passed_retrospective_screen"], flush=True)
        results[symbol] = experiments
    artifact = {"plan": plan, "declaration": declaration, "mechanism_experiments": results,
                "scenario_count": count, "closed_ledger_records_checked": ledger_count,
                "places_orders": False, "forward_started": False,
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
                    "research_parallel_simulation_capital.py", "research_stock_mechanisms.py", "stock_research_entry_policy.py",
                    "backtest_stock_swing_120.py", "stock_swing_signals.py", "stock_swing_profiles.py")}}
    (args.output_dir / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
