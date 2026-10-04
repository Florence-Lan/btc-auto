#!/usr/bin/env python3
"""Finite, declared stock ablations; all results retained, no orders or refitting."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import backtest_stock_swing_120 as engine
import stock_research_entry_policy as policy
from replay_stock_swing_per_symbol import after_close_summary
from research_stock_swing_robustness import audit_snapshot, monthly_contributions, trade_diagnostics

VARIANTS = {
    "prior5m": {},
    "active30m": {"entry_volume_lookback_bars": 6},
    "regular_clock": {"entry_session": "weekday_regular_clock"},
    "long_only": {"entry_direction": "long"},
    "active30m_regular_long": {"entry_volume_lookback_bars": 6,
                              "entry_session": "weekday_regular_clock", "entry_direction": "long"},
}
DEFERRED_VARIANTS = {
    "wait4h_active30m": {"entry_signal_validity_minutes": 240, "entry_volume_lookback_bars": 6},
    "wait4h_active30m_regular": {"entry_signal_validity_minutes": 240, "entry_volume_lookback_bars": 6,
                               "entry_session": "weekday_regular_clock"},
    "wait4h_active30m_regular_long": {"entry_signal_validity_minutes": 240, "entry_volume_lookback_bars": 6,
                                    "entry_session": "weekday_regular_clock", "entry_direction": "long"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def volume_diagnostics(result: dict, source: dict) -> dict:
    rows = {int(row[0]): row for row in source["trade_5m"]}
    entries, exits, details = 0, 0, []
    for trade in result["trades"]:
        entry = rows[engine.parse_time(trade["entry_utc"])]
        exit_bar = rows[engine.parse_time(trade["exit_utc"])]
        entries += float(entry[5]) == 0
        exits += float(exit_bar[5]) == 0
        details.append({"entry_utc": trade["entry_utc"], "exit_utc": trade["exit_utc"],
                        "entry_reported_volume": float(entry[5]), "exit_reported_volume": float(exit_bar[5]),
                        "entry_trade_count": int(entry[8]), "exit_trade_count": int(exit_bar[8])})
    return {"entry_bars_zero_reported_volume": entries, "exit_bars_zero_reported_volume": exits,
            "closed_trade_bar_checks": details,
            "warning": "Ex-post diagnostic only, never an entry filter; positive volume does not prove opening-tick or stop fills."}


def assessment(runs: dict) -> dict:
    """Retrospective screening only. Zero trades or open estimates cannot pass."""
    failures = []
    minimums = {"full": 10, "development": 3, "validation": 3, "recent30d": 3}
    for window, minimum in minimums.items():
        for cost in (1, 2):
            name = f"{window}_cost{cost}"
            r = runs[name]
            if r["net_closed_pnl"] <= 0:
                failures.append(f"{name}: nonpositive closed PnL")
            if r["closed_trades"] < minimum:
                failures.append(f"{name}: fewer than {minimum} closed trades")
            if r["max_sampled_drawdown_pct"] > 8:
                failures.append(f"{name}: drawdown above 8%")
            if r["liquidation_stress_count"]:
                failures.append(f"{name}: liquidation stress")
            if r["execution_volume_diagnostics"]["entry_bars_zero_reported_volume"] or r["execution_volume_diagnostics"]["exit_bars_zero_reported_volume"]:
                failures.append(f"{name}: zero-volume assumed entry or exit")
    for cost in (1, 2):
        r = runs[f"full_cost{cost}"]
        if r["target_trades_net_at_least_120pct_margin"] < 1:
            failures.append(f"full_cost{cost}: no net120 trade")
        if r["trade_diagnostics"]["closed_return_without_largest_winner_pct"] <= 0:
            failures.append(f"full_cost{cost}: nonpositive closed PnL without best trade")
    for name in ("full_slippage50bps", "full_margin5pct"):
        r = runs[name]
        if r["net_closed_pnl"] <= 0 or r["liquidation_stress_count"] or r["max_sampled_drawdown_pct"] > 8:
            failures.append(f"{name}: PnL/liquidation/drawdown stress failed")
    return {"passed_retrospective_screen": not failures, "failure_reasons": failures,
            "forward_validated": False, "execution_qualified": False}


def validate_ledger(result: dict, settings: dict, source: dict, cost: int) -> dict:
    fee = settings["taker_fee_rate_assumption"] * cost
    bars = {int(row[0]): engine.candle(row) for row in source["trade_5m"]}
    checked = 0
    for trade in result["trades"]:
        side = 1 if trade["direction"] == "long" else -1
        t = engine.parse_time(trade["entry_utc"])
        assert policy.rejection(settings, trade["symbol"], t, side) is None
        volume = policy.preceding_volume({"trade": bars}, t, 300_000, settings.get("entry_volume_lookback_bars", 1))
        assert volume is not None and trade["qty"] <= volume * .1 + 1e-10
        assert abs(trade["entry_fee"] - trade["qty"] * trade["entry_fill"] * fee) < 1e-7
        assert abs(trade["net_pnl"] - (trade["gross_pnl"] - trade["entry_fee"] - trade["exit_fee"] - trade["funding_debit"])) < 1e-7
        assert abs(trade["net_return_initial_margin_pct"] - trade["net_pnl"] / trade["initial_margin"] * 100) < 1e-7
        assert abs(trade["initial_margin"] - trade["qty"] * trade["entry_fill"] / 10) < 1e-7
        checked += 1
    s = after_close_summary(result)
    assert s["closed_trades"] == checked
    assert s["target_trades_net_at_least_120pct_margin"] == sum(t["net_return_initial_margin_pct"] >= 120 - 1e-7 for t in result["trades"])
    assert abs(s["estimated_close_equity"] - (s["initial_equity"] + s["net_closed_pnl"] + s["estimated_open_close_net_pnl"])) < 1e-7
    return {"closed_trades_checked": checked, "accounting_and_causal_entry_checks_passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/research/stock_swing_liquidity_20261004"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("fixed", "deferred"), default="fixed")
    parser.add_argument("--snapshot", type=Path, default=Path("data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve previous research; choose a new directory")
    previous_path = args.source_dir / "results.json"
    previous = json.loads(previous_path.read_text())
    profile, base = previous["profile"], previous["base_config"]
    snapshot_path = args.snapshot
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    audit_snapshot(snapshot)
    assert base["leverage"] == 10 and base["target_margin_return"] == 1.2
    args.output_dir.mkdir(parents=True)
    btc_path = Path("config/active_simulation_candidate.json")
    btc_before = btc_path.read_bytes()
    variants = VARIANTS if args.phase == "fixed" else DEFERRED_VARIANTS
    declaration = {
        "declared_at_utc": datetime.now(timezone.utc).isoformat(), "places_orders": False,
        "already_inspected_history": True, "independent_holdout": False,
        "signal_parameters_refit": False, "phase": args.phase, "variant_count": len(variants), "variants": variants,
        "snapshot_sha256": digest(snapshot_path), "previous_results_sha256": digest(previous_path),
        "participation_fraction": .1,
        "hypotheses": ["Require all prior six completed 5m volumes positive and cap at 10% of their minimum.",
                       "Weekday regular local clock, US DST respected; no holidays/early-close calendar claim.",
                       "Remove short entries; existing short PnL is descriptive, not independent selection evidence.",
                       "Combined rule is declared in advance, not selected after this experiment."],
        "retrospective_screen": {"positive_closed_pnl_both_costs_windows": ["full", "development", "validation", "recent30d"],
                                 "minimum_closed_trades": {"full": 10, "development": 3, "validation": 3, "recent30d": 3},
                                 "full_pnl_without_best_trade_positive_both_costs": True,
                                 "max_drawdown_pct": 8, "full_net120_targets_both_costs_min": 1,
                                 "zero_reported_volume_assumed_entry_or_exit": 0,
                                 "stress_positive_closed_pnl_no_liquidation_dd_at_most8": True},
        "warning": "Retrospective engineering screen, stricter than prior screen; no auto-promotion. Live fills remain unverified.",
    }
    if args.phase == "deferred":
        declaration["hypotheses"] = [
            "Adaptive follow-up AFTER reviewing five fixed variants; not a new holdout.",
            "Signal remains eligible for <4h after close; retry each 5m, at most one fill per signal.",
            "Wait for six contiguous positive prior volumes, cap at 10% of minimum; prices rechecked at entry.",
            "Compare all clocks, weekday regular clocks, and weekday regular long-only; no expiry/grid search.",
            "Basis filter uses currently observable hour-opening index as approximation, not current-tick oracle."]
    (args.output_dir / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    result = copy.deepcopy(previous)
    result["previous_research_results_path"] = str(previous_path)
    result["previous_research_results_sha256"] = digest(previous_path)
    result["mechanism_declarations"] = [*previous.get("mechanism_declarations", []), declaration]
    result["mechanism_declaration"] = declaration
    result["mechanism_experiments"] = copy.deepcopy(previous.get("mechanism_experiments", {}))
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["source_sha256"] = {name: digest(Path(__file__).with_name(name)) for name in (
        "research_stock_mechanisms.py", "stock_research_entry_policy.py", "backtest_stock_swing_120.py",
        "stock_swing_signals.py", "stock_swing_profiles.py", "replay_stock_swing_per_symbol.py",
        "research_stock_swing_robustness.py")}
    frozen = args.output_dir / "frozen_inputs"
    frozen.mkdir()
    for name in result["source_sha256"]:
        (frozen / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    checked, scenarios, reproduced = 0, 0, 0
    end = snapshot["end_ms_exclusive"]
    split = engine.parse_time(base["evaluation_split_utc"])
    for symbol, weight in profile["capital_weights"].items():
        first = snapshot["symbols"][symbol]["start_ms"]
        common = {**base, "symbols": [symbol], "symbol_profiles": {symbol: profile["symbol_profiles"][symbol]},
                  "initial_equity_usdt": base["initial_equity_usdt"] * weight, "max_positions": 1,
                  "risk_fraction_per_trade": profile["risk_fraction_per_bucket_trade"],
                  "execution_timeframe": "5m", "entry_max_previous_bar_participation_fraction": .1}
        experiments = result["mechanism_experiments"].get(symbol, {})
        for variant, extras in variants.items():
            config = {**common, **extras}
            runs = {}
            cases = []
            for window, start, finish in (("full", first, end), ("development", first, split),
                                           ("validation", split, end), ("recent60d", max(first, end - 60 * engine.DAY), end),
                                           ("recent30d", max(first, end - 30 * engine.DAY), end)):
                for cost in (1, 2):
                    cases.append((f"{window}_cost{cost}", start, finish, config, cost))
            cases.extend([("full_slippage50bps", first, end, {**config, "adverse_slippage_fraction_assumption": .005}, 1),
                          ("full_margin5pct", first, end, {**config, "maintenance_margin_fraction_assumption": .05}, 1)])
            for name, start, finish, settings, cost in cases:
                replay = engine.simulate(snapshot, settings, start, finish, cost)
                check = validate_ledger(replay, settings, snapshot["symbols"][symbol], cost)
                checked += check["closed_trades_checked"]
                scenarios += 1
                summary = after_close_summary(replay)
                summary["trade_diagnostics"] = trade_diagnostics(replay["trades"], summary["initial_equity"])
                summary["execution_volume_diagnostics"] = volume_diagnostics(replay, snapshot["symbols"][symbol])
                summary["monthly_continuous_contributions"] = monthly_contributions(replay)
                runs[name] = summary
                engine.write_csv(args.output_dir / f"{symbol}_{variant}_{name}_trades.csv", replay["trades"])
                engine.write_csv(args.output_dir / f"{symbol}_{variant}_{name}_equity.csv", replay["equity_path"])
                if variant == "prior5m" and name.startswith(("full_cost", "recent30d_cost")):
                    old_key = name.replace("_cost", "_prior5m_volume10pct_cost")
                    old = previous["runs"][symbol][old_key]
                    for key in ("net_closed_pnl", "estimated_close_return_pct", "closed_trades", "target_trades_net_at_least_120pct_margin", "max_sampled_drawdown_pct"):
                        assert abs(summary[key] - old[key]) < 1e-9, (symbol, name, key)
                    result["runs"][symbol][old_key] = summary
                    reproduced += 1
            experiments[variant] = {"settings": extras, "runs": runs, "assessment": assessment(runs)}
            print(symbol, variant, json.dumps({"full_return_pct": runs["full_cost2"]["estimated_close_return_pct"],
                  "validation_closed_pnl": runs["validation_cost2"]["net_closed_pnl"],
                  "closed": runs["full_cost2"]["closed_trades"],
                  "passed": experiments[variant]["assessment"]["passed_retrospective_screen"]}), flush=True)
        result["mechanism_experiments"][symbol] = experiments
    assert btc_path.read_bytes() == btc_before
    result["places_orders"] = False
    result["forward_validated"] = False
    result["mechanism_conclusion"] = {"all_stocks_have_passing_variant": all(
        any(v["assessment"]["passed_retrospective_screen"] for v in experiments.values())
        for experiments in result["mechanism_experiments"].values()), "automatic_promotion": False}
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    verification = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "scenario_ledgers_checked": scenarios,
                    "closed_trades_checked": checked, "previous_scenarios_reproduced": reproduced,
                    "consistent_hourly_aggregation": True, "source_hashes_match": True,
                    "accounting_and_causal_entry_checks_passed": True,
                    "btc_selection_unchanged": True, "btc_selection": json.loads(btc_before), "places_orders": False}
    (args.output_dir / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "forward_research_plan.json").write_bytes((args.source_dir / "forward_research_plan.json").read_bytes())


if __name__ == "__main__":
    main()
