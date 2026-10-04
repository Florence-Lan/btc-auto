#!/usr/bin/env python3
"""Independently check archived stock ledgers and publish a compact review."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig"))) if path.exists() else []


def close(a, b, tolerance=1e-7):
    if abs(a - b) > tolerance:
        raise AssertionError((a, b))


def ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=Path("data/research/stock_mechanisms_20261005"))
    parser.add_argument("--deferred-dir", type=Path, default=Path("data/research/stock_mechanisms_wait_20261005"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("data/research/stock_execution_evidence_20261005"))
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    counts = []
    for folder in (args.fixed_dir, args.deferred_dir):
        artifact = load(folder / "results.json")
        for name, expected in artifact["source_sha256"].items():
            assert hashlib.sha256((folder / "frozen_inputs" / name).read_bytes()).hexdigest() == expected
        checked, scenario_count = 0, 0
        for symbol, experiments in artifact["mechanism_experiments"].items():
            for variant in artifact["mechanism_declaration"]["variants"]:
                for case, summary in experiments[variant]["runs"].items():
                    prefix = folder / f"{symbol}_{variant}_{case}"
                    ledger = rows(Path(str(prefix) + "_trades.csv"))
                    equity = rows(Path(str(prefix) + "_equity.csv"))
                    assert len(ledger) == summary["closed_trades"]
                    close(sum(float(t["net_pnl"]) for t in ledger), summary["net_closed_pnl"])
                    best = max([0, *(float(t["net_pnl"]) for t in ledger)])
                    close((summary["net_closed_pnl"] - best) / summary["initial_equity"] * 100,
                          summary["trade_diagnostics"]["closed_return_without_largest_winner_pct"])
                    for t in ledger:
                        close(float(t["gross_pnl"]) - float(t["entry_fee"]) - float(t["exit_fee"]) - float(t["funding_debit"]), float(t["net_pnl"]))
                        close(float(t["net_pnl"]) / float(t["initial_margin"]) * 100, float(t["net_return_initial_margin_pct"]))
                        close(float(t["qty"]) * float(t["entry_fill"]) / 10, float(t["initial_margin"]))
                        if variant.startswith("wait4h"):
                            delay = ms(t["entry_utc"]) - ms(t["signal_utc"])
                            assert 0 <= delay < 4 * 3600000 and delay % 300000 == 0
                    assert len({t["signal_utc"] for t in ledger}) == len(ledger)
                    assert sum(float(t["net_return_initial_margin_pct"]) >= 120 - 1e-7 for t in ledger) == summary["target_trades_net_at_least_120pct_margin"]
                    close(summary["initial_equity"] + summary["net_closed_pnl"] + summary["estimated_open_close_net_pnl"], summary["estimated_close_equity"])
                    if equity:
                        close(float(equity[-1]["equity"]), summary["final_mark_equity"])
                        close(sum(m["marked_pnl"] for m in summary["monthly_continuous_contributions"]), float(equity[-1]["equity"]) - summary["initial_equity"])
                    checked += len(ledger)
                    scenario_count += 1
        counts.append({"directory": str(folder), "scenario_ledgers_checked": scenario_count, "closed_trades_checked": checked})
        verification = load(folder / "verification.json")
        assert load(Path("config/active_simulation_candidate.json")) == verification["btc_selection"]
        assert checked == verification["closed_trades_checked"] and scenario_count == verification["scenario_ledgers_checked"]
        verification.update({"independent_archived_csv_verification_passed": True, "tests_passed": args.tests_passed,
                             "independent_verified_at_utc": datetime.now(timezone.utc).isoformat()})
        (folder / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    latest = load(args.deferred_dir / "results.json")
    snapshot = json.loads(gzip.decompress(Path("data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz").read_bytes()))
    evidence = load(args.evidence_dir / "evidence.json")
    for request in evidence["requests"]:
        assert hashlib.sha256((args.evidence_dir / "raw" / (request["key"] + ".json")).read_bytes()).hexdigest() == request["sha256"]
    evidence_checks = {}
    for symbol, details in evidence["historical_fill_bar_queries"].items():
        candles = {int(r[0]): r for r in snapshot["symbols"][symbol]["trade_5m"]}
        for item in details:
            assert not item.get("unavailable") and not item["possibly_truncated"]
            close(float(candles[ms(item["bar_start_utc"])][5]), item["bar_quantity_observed"], 1e-6)
        evidence_checks[symbol] = {"bars_checked_against_raw_aggregates": len(details),
                                  "volume_mismatches": 0,
                                  "empty_entry_bars": sum(d["kind"] == "entry" and d["within_bar_count"] == 0 for d in details),
                                  "empty_exit_bars": sum(d["kind"] == "exit" and d["within_bar_count"] == 0 for d in details)}
    summary = {"status": "research_only", "places_orders": False, "forward_validated": False,
               "reviewed_at_utc": datetime.now(timezone.utc).isoformat(), "stocks": {},
               "phases": counts, "scenario_count": sum(c["scenario_ledgers_checked"] for c in counts),
               "historical_fill_bar_evidence": evidence_checks,
               "report_path": "docs/stock_mechanism_review_20261005.md",
               "limitations": ["All inspected history is retrospective; phases are sequential and adaptive.",
                               "Overlapping windows and repeated variants are not independent trade samples.",
                               "Positive reported volume and visible current depth do not prove hypothetical fills."]}
    for symbol, experiments in latest["mechanism_experiments"].items():
        passed = sum(v["assessment"]["passed_retrospective_screen"] for v in experiments.values())
        summary["stocks"][symbol] = {
            "trial_count": len(experiments), "passed_count": passed,
            "finding": f"本轮比较 {len(experiments)} 套规则，{passed} 套通过回顾性筛查；当前股票方案继续不具备执行资格。",
            "variants": {name: {"full_cost1_return_pct": v["runs"]["full_cost1"]["estimated_close_return_pct"],
                                "full_cost2_return_pct": v["runs"]["full_cost2"]["estimated_close_return_pct"],
                                "full_cost2_closed_trades": v["runs"]["full_cost2"]["closed_trades"],
                                "recent30d_cost2_closed_return_pct": v["runs"]["recent30d_cost2"]["net_closed_pnl"] / v["runs"]["recent30d_cost2"]["initial_equity"] * 100,
                                "failure_reasons": v["assessment"]["failure_reasons"]}
                         for name, v in experiments.items()},
        }
    (args.deferred_dir / "mechanism_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"scenario_count": summary["scenario_count"], "archived_closed_trades_checked": sum(c["closed_trades_checked"] for c in counts),
                      "raw_fill_bars_checked": sum(v["bars_checked_against_raw_aggregates"] for v in evidence_checks.values())}))


if __name__ == "__main__":
    main()
