"""Small, preregistered EMA-spread study; never modifies active configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
from validate_candidate_portfolio import compact, quarterly_returns

ROOT = sim.repo_root()
OUT = ROOT / "data/validation/trend_filter_20260917"
SPREADS = {"baseline": 0.003, "spread_004": 0.004, "spread_005": 0.005, "spread_006": 0.006}


def ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cfg = frozen_strategy.load_frozen_strategy(ROOT / "config/frozen_strategy_active_20260809.json")
    cfg = replace(cfg, strategy_modes=("trend", "timeseries_trend"), max_drawdown_stop_pct=0.0)
    plan = {
        "candidates": SPREADS, "changed_parameter": "timeseries_min_ema_spread_pct",
        "development": ["2022-01-01", "2025-01-01"],
        "temporal_validation": ["2025-01-01", "2026-06-28T15:00:00"],
        "selection": "Development only: positive return, PF >= 1.3, drawdown <= baseline and 15%; maximize Calmar above baseline.",
        "acceptance": "Validation return >= baseline, drawdown <= baseline; recent return >= baseline, drawdown <= baseline; double-cost validation positive and drawdown <= 15%.",
        "limitations": "Historical data already existed and informed earlier strategy versions. Temporal validation is not independent prospective OOS. Recent 50 days already inspected. LLM gate and real fills not simulated.",
    }
    sim.save_json(OUT / "plan.json", plan)
    historical_path = ROOT / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, historical_path)
    historical, funding, _ = sim.load_market_snapshot(historical_path)
    recent, recent_funding, recent_meta = sim.load_market_snapshot(ROOT / "data/snapshots/btcusdt_backtest_50d_20260917.json.gz")
    old_macro = macro_regime.load_macro_snapshot(ROOT / "data/snapshots/macro_20191001_20260628.json.gz")
    new_macro = macro_regime.load_macro_snapshot(ROOT / "data/snapshots/macro_backtest_50d_20260917.json.gz")
    end_recent = ms(recent_meta["end_utc"])
    results = {}

    def window(label, data, rates, macro, start, end, candidates, double=False):
        base = [c for c in data["5m"] if start - 45 * sim.MS_PER_DAY <= c.open_time_ms and c.close_time_ms < end]
        trend = [c for c in data["1h"] if start - 45 * sim.MS_PER_DAY <= c.open_time_ms and c.close_time_ms < end]
        local = cfg
        if double:
            local = replace(cfg, **{k: getattr(cfg, k) * 2 for k in ("maker_fee", "taker_fee", "entry_slippage_bps", "exit_slippage_bps", "depth_impact_bps")})
        tactical = sim.simulate(base, replace(local, strategy_modes=("trend",)), start, None, rates)
        rows = {}
        for name in candidates:
            candidate = replace(local, timeseries_min_ema_spread_pct=SPREADS[name])
            core = sim.simulate_timeseries_trend(trend, replace(candidate, strategy_modes=("timeseries_trend",)), start, rates)
            sleeves, diagnostics = macro_regime.apply_macro_overlay([tactical, core], macro, enabled_factors=("vix", "dollar", "metals", "sentiment"))
            result = portfolio_risk.combine_sleeves_with_drawdown_policy(base, sleeves, candidate, portfolio_risk.DrawdownRiskPolicy(8, 15, 0.35), start)
            row = compact(result["summary"])
            row["quarterly_returns_pct"] = [v * 100 for v in quarterly_returns(result["equity_curve"])]
            row["macro"] = diagnostics
            row["risk"] = result["risk_diagnostics"]
            rows[name] = row
            sim.save_trades_csv(OUT / f"{label}_{name}_trades.csv", result["trades"])
            print(label, name, json.dumps({k: row[k] for k in ("total_return_pct", "max_drawdown_pct", "profit_factor", "trades", "calmar")}), flush=True)
        results[label] = rows
        sim.save_json(OUT / "results.json", results)
        return rows

    dev = window("development", historical, funding, old_macro, ms("2022-01-01"), ms("2025-01-01"), SPREADS)
    baseline = dev["baseline"]
    eligible = [name for name, r in dev.items() if name != "baseline" and r["total_return_pct"] > 0 and (r["profit_factor"] or 0) >= 1.3 and r["max_drawdown_pct"] <= min(15, baseline["max_drawdown_pct"]) and r["calmar"] > baseline["calmar"]]
    selected = max(eligible, key=lambda name: dev[name]["calmar"], default=None)
    sim.save_json(OUT / "selection.json", {"selected": selected, "development_only": True})
    names = ["baseline"] + ([selected] if selected else [])
    holdout = window("validation", historical, funding, old_macro, ms("2025-01-01"), ms("2026-06-28T15:00:00"), names)
    latest = window("recent_50d", recent, recent_funding, new_macro, end_recent - 50 * sim.MS_PER_DAY, end_recent, names)
    stress = window("double_cost_validation", historical, funding, old_macro, ms("2025-01-01"), ms("2026-06-28T15:00:00"), names, True)
    passed = bool(selected and all(rows[selected]["total_return_pct"] >= rows["baseline"]["total_return_pct"] and rows[selected]["max_drawdown_pct"] <= rows["baseline"]["max_drawdown_pct"] for rows in (holdout, latest)) and stress[selected]["total_return_pct"] > 0 and stress[selected]["max_drawdown_pct"] <= 15)
    report = {"plan": plan, "selected": selected, "acceptance_pass": passed, "results": results, "active_configuration_changed": False}
    if selected:
        candidate_config = asdict(replace(cfg, timeseries_min_ema_spread_pct=SPREADS[selected], max_drawdown_stop_pct=12.0))
        candidate_manifest = {**manifest, "freeze_id": "btc_trend_filter_research_20260917", "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "based_on_freeze_id": manifest["freeze_id"], "config": candidate_config, "config_sha256": frozen_strategy.canonical_config_hash(candidate_config), "status": "research_candidate", "acceptance_pass": passed}
        sim.save_json(OUT / "candidate_manifest.json", candidate_manifest)
    sim.save_json(OUT / "report.json", report)
    print(json.dumps({"selected": selected, "acceptance_pass": passed}), flush=True)


if __name__ == "__main__":
    main()
