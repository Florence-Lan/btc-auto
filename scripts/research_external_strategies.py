"""Frozen-window experiments for external ideas. Never places orders."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import backtest_funding_carry as carry
import download_spot_snapshot
import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
from research_external_signals import ema_signals, donchian_signals, vwap_rsi_signals
from research_signal_engine import simulate_signals
from validate_candidate_portfolio import compact, quarterly_returns

ROOT = sim.repo_root()
OUT = ROOT / "data/validation/external_experiments_20260917"
EXPERIMENTS = {
    "baseline_003": "Active v3 0.30% threshold",
    "candidate_004": "Prior candidate 0.40% threshold",
    "atr_05": "EMA48/240 difference >= 0.5 ATR14, symmetric hysteresis",
    "atr_10": "EMA48/240 difference >= 1.0 ATR14, symmetric hysteresis",
    "ensemble": "EMA24/120,48/240,96/480; 0.40% threshold; each 1/3 core budget",
    "donchian": "Replace core with long-only 1h prior-20-high entry / prior-10-low exit",
    "donchian_volume": "Same breakout, volume above prior 20-bar mean required at entry",
    "donchian_atr": "Same breakout, ATR14 above its 20-bar mean required at entry",
    "vwap_rsi": "Replace tactical sleeve with UTC session VWAP + RSI14 60/40 crosses; keep 0.40% core",
}


def ms(value):
    dt = datetime.fromisoformat(value)
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def weighted(sleeve, fraction):
    return {**sleeve, "trades": [asdict(sim.scaled_trade_from_raw(t, fraction, 100.0)) for t in sleeve["trades"]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cfg = frozen_strategy.load_frozen_strategy(ROOT / "config/frozen_strategy_active_20260809.json")
    cfg = replace(cfg, strategy_modes=("trend", "timeseries_trend"), max_drawdown_stop_pct=0.0)
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": EXPERIMENTS,
        "rules": "Completed-bar signals, next-open fills. 45d warmup. Each window starts flat. Synthetic terminal closes included. No post-result tuning.",
        "windows": {"development": ["2022-01-01", "2025-01-01"], "validation": ["2025-01-01", "2026-06-28T15:00:00"], "recent_50d": ["2026-07-28T16:20:00", "2026-09-16T16:20:00"]},
        "acceptance": "Research shortlist only: dev/validation/recent positive; validation PF >=1.3; DD <=15% all windows; validation return >= candidate004 and DD <= candidate004; double-cost validation positive. No live promotion.",
        "funding_carry": "Separate long spot / short perp; 50% initial notional per leg, remaining capital futures collateral; rebalance at 40% margin buffer; observed hourly basis and 20bps terminal adverse basis stress.",
        "vwap_rsi": "5m UTC VWAP quote_volume/volume; RSI14 crossing 60 long /40 short aligned with VWAP. Exit on RSI50 or VWAP invalidation or close moving 2 entry-signal ATR against us or 36 bars. Next-open exits. No intrabar stops. 11% vol target, 240-bar lookback, <=2x sleeve leverage.",
        "risk": "Keep tactical unchanged for core experiments. No extra core budget for ensemble. Volatility sizing at entry only. Gross portfolio cap 2x; DD throttle 8%, entry block at 15%, min multiplier .35. Same macro overlay. Event template empty.",
        "sources": ["https://qoppac.blogspot.com/2017/06/some-more-trading-rules.html", "https://concretumgroup.com/wp-content/uploads/2026/02/Catching-Crypto-Trends.pdf", "https://www.traseq.com/blog/donchian-channel-breakout-strategy", "https://www.coinquant.ai/blog/atr-strategy-backtest-on-bitcoin-does-volatility-filtering-improve-entries", "https://www.binance.com/en/square/post/363007142090236", "https://www.binance.com/en/square/post/14991209765562"],
        "limitations": ["Adaptations, not exact reproductions of external claims.", "Historical windows are reused research data, not independent prospective OOS.", "Execution-layer LLM gate not replayed.", "Inherited depth impact uses execution-bar volume; bar replay does not prove executable fills.", "Inherited portfolio drawdown marks price PnL and settles costs/funding at trade close; it is not a tick-accurate account equity model.", "The same risk cap does not guarantee equal realized risk across strategies.", "OI/order-book/X ideas lacking reproducible rules or historical data are not tested."],
    }
    sim.save_json(OUT / "plan.json", plan)
    hist_path = ROOT / manifest["snapshot_path"]
    frozen_strategy.verify_snapshot(manifest, hist_path)
    hist, hist_funding, _ = sim.load_market_snapshot(hist_path)
    recent, recent_funding, _ = sim.load_market_snapshot(ROOT / "data/snapshots/btcusdt_backtest_50d_20260917.json.gz")
    old_macro = macro_regime.load_macro_snapshot(ROOT / "data/snapshots/macro_20191001_20260628.json.gz")
    new_macro = macro_regime.load_macro_snapshot(ROOT / "data/snapshots/macro_backtest_50d_20260917.json.gz")
    hist_spot, _ = download_spot_snapshot.load_spot_snapshot(ROOT / "data/snapshots/btcusdt_spot_1h_20200101_20260628.json.gz")
    recent_spot, _ = download_spot_snapshot.load_spot_snapshot(ROOT / "data/snapshots/btcusdt_spot_external_20260917.json.gz")
    results, carry_results, checks = {}, {}, {}
    for label, bounds in plan["windows"].items():
        start, end = map(ms, bounds)
        data, funding, macro, spot = (recent, recent_funding, new_macro, recent_spot) if label == "recent_50d" else (hist, hist_funding, old_macro, hist_spot)
        base = [c for c in data["5m"] if start-45*sim.MS_PER_DAY <= c.open_time_ms and c.close_time_ms < end]
        hourly = [c for c in data["1h"] if start-45*sim.MS_PER_DAY <= c.open_time_ms and c.close_time_ms < end]
        assert base and hourly and end-base[-1].close_time_ms <= 300000
        for interval, candles in (("5m", base), ("1h", hourly)):
            step = sim.interval_to_ms(interval)
            assert all(b.open_time_ms-a.open_time_ms == step for a,b in zip(candles,candles[1:]))
        for double in (False, True):
            key = label + ("_double_cost" if double else "")
            print("Starting", key, flush=True)
            local = replace(cfg, **{k: getattr(cfg,k)*2 for k in ("maker_fee", "taker_fee", "entry_slippage_bps", "exit_slippage_bps", "depth_impact_bps")}) if double else cfg
            tactical = sim.simulate(base, replace(local, strategy_modes=("trend",)), start, None, funding)
            baseline = sim.simulate_timeseries_trend(hourly, local, start, funding)
            candidate_cfg = replace(local, timeseries_min_ema_spread_pct=.004)
            candidate = sim.simulate_timeseries_trend(hourly, candidate_cfg, start, funding)
            if not double:
                replay = simulate_signals(hourly, ema_signals(hourly, local, start), "parity", local, start, funding)
                assert len(replay["trades"]) == len(baseline["trades"])
                for a,b in zip(replay["trades"],baseline["trades"]):
                    for field in ("entry_time_utc", "exit_time_utc", "net_pnl", "fees", "initial_qty"):
                        assert a[field] == b[field], (label,field,a[field],b[field])
                checks[label] = "Custom signal engine reproduces frozen baseline trade times, quantities, costs and PnL exactly"
            rows = {}
            for name in EXPERIMENTS:
                if name == "baseline_003":
                    sleeves = [tactical, baseline]
                elif name == "candidate_004":
                    sleeves = [tactical, candidate]
                elif name.startswith("atr_"):
                    multiple = .5 if name == "atr_05" else 1.0
                    core = simulate_signals(hourly, ema_signals(hourly,local,start,multiple), name, local, start, funding)
                    sleeves = [tactical, core]
                elif name == "ensemble":
                    cores = []
                    for fast,slow in ((24,120),(48,240),(96,480)):
                        ecfg = replace(candidate_cfg,timeseries_fast_ema=fast,timeseries_slow_ema=slow)
                        core = simulate_signals(hourly,ema_signals(hourly,ecfg,start),f"ema_{fast}_{slow}",ecfg,start,funding)
                        cores.append(weighted(core,1/3))
                    sleeves = [tactical,*cores]
                elif name.startswith("donchian"):
                    filter_name = name.split("_",1)[1] if "_" in name else None
                    core = simulate_signals(hourly,donchian_signals(hourly,start,filter_name),name,local,start,funding)
                    sleeves = [tactical,core]
                else:
                    vcfg = replace(local,timeseries_timeframe="5m")
                    vwap = simulate_signals(base,vwap_rsi_signals(base,start),name,vcfg,start,funding)
                    sleeves = [vwap,candidate]
                adjusted, macro_diag = macro_regime.apply_macro_overlay(sleeves,macro,enabled_factors=("vix","dollar","metals","sentiment"))
                result = portfolio_risk.combine_sleeves_with_drawdown_policy(base,adjusted,local,portfolio_risk.DrawdownRiskPolicy(),start)
                row = compact(result["summary"])
                row["quarters_pct"] = [v*100 for v in quarterly_returns(result["equity_curve"])]
                row["macro"] = macro_diag
                row["risk"] = result["risk_diagnostics"]
                rows[name] = row
                sim.save_trades_csv(OUT/f"{key}_{name}_trades.csv",result["trades"])
                # Full curve retained as a compressed artifact for audit/charting.
                import gzip
                with gzip.open(OUT/f"{key}_{name}_equity.json.gz","wt",encoding="utf-8") as handle:
                    json.dump(result["equity_curve"],handle,separators=(",",":"))
                print(key,name,"return",round(row["total_return_pct"],3),"DD",round(row["max_drawdown_pct"],3),"trades",row["trades"],flush=True)
                results[key] = rows
                sim.save_json(OUT/"results.json",results)
            selected = [(t,r) for t,r in zip(funding.times,funding.rates) if start <= t < end]
            factor = 2 if double else 1
            cr = carry.backtest_with_basis([t for t,r in selected],[r for t,r in selected],spot,hourly,evaluation_start_ms=start,interval_ms=3600000,notional_fraction=.5,spot_fee=.001*factor,futures_fee=.00045*factor,slippage_bps_per_leg=1.5*factor,basis_stress_bps=20,rebalance_margin_buffer_pct=40)
            carry_results[key] = cr["summary"]
            sim.save_json(OUT/f"{key}_funding_carry.json",cr)
    shortlist = []
    for name in EXPERIMENTS:
        if name in ("baseline_003","candidate_004"):
            continue
        h = results["validation"][name]
        reference = results["validation"]["candidate_004"]
        if all(results[w][name]["total_return_pct"] > 0 and results[w][name]["max_drawdown_pct"] <=15 for w in plan["windows"]) and (h["profit_factor"] or 0)>=1.3 and h["total_return_pct"]>=reference["total_return_pct"] and h["max_drawdown_pct"]<=reference["max_drawdown_pct"] and results["validation_double_cost"][name]["total_return_pct"]>0:
            shortlist.append(name)
    files = [hist_path, ROOT/"data/snapshots/btcusdt_backtest_50d_20260917.json.gz", ROOT/"data/snapshots/macro_20191001_20260628.json.gz",ROOT/"data/snapshots/macro_backtest_50d_20260917.json.gz",ROOT/"data/snapshots/btcusdt_spot_1h_20200101_20260628.json.gz",ROOT/"data/snapshots/btcusdt_spot_external_20260917.json.gz",Path(__file__),ROOT/"scripts/research_external_signals.py",ROOT/"scripts/research_signal_engine.py"]
    report = {"plan":plan,"results":results,"funding_carry":carry_results,"verification":checks,"shortlist":shortlist,"input_sha256":{str(p.relative_to(ROOT)):frozen_strategy.sha256_file(p) for p in files},"active_configuration_changed":False}
    sim.save_json(OUT/"report.json",report)
    with (OUT/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        keys = ["window","experiment","total_return_pct","max_drawdown_pct","profit_factor","trades","win_rate_pct","total_costs"]
        writer = csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        for window,rows in results.items():
            for name,r in rows.items():
                writer.writerow({k:({"window":window,"experiment":name}.get(k,r.get(k))) for k in keys})
    print("SHORTLIST",shortlist,flush=True)


if __name__ == "__main__":
    main()
