"""Fixed comparisons in the terminal execution model; never selects a live strategy."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json

import backtest_execution
import diagnose_strategy_losses as diagnosis
import execution_targets
import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
from validate_reentry_research import make_sleeves


def main():
    root = sim.repo_root()
    plan = json.loads((root/"config/execution_validation_20261001.json").read_text())
    manifest, cfg = frozen_strategy.load_frozen_strategy(root/plan["manifest"])
    cfg = replace(cfg,max_drawdown_stop_pct=0)
    macro = macro_regime.load_macro_snapshot(root/plan["macro_snapshot"])
    rates = json.loads((root/plan["funding_snapshot"]).read_text())["events"]
    prefix = root/"data/validation/execution_validation_20261001"
    paths = [root/"config/execution_validation_20261001.json",root/plan["manifest"],
             root/plan["macro_snapshot"],root/plan["funding_snapshot"]]
    paths += [root/row["market_snapshot"] for row in plan["windows"].values()]
    paths += [root/"scripts"/name for name in ("trading_execution.py","account_risk.py","execution_targets.py",
              "backtest_execution.py","validate_execution_model.py","simulate_range_swing.py",
              "portfolio_risk.py","execution_ledger.py","macro_regime.py","research_reentry.py")]
    report = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),"plan":plan,
              "input_sha256":{str(p.relative_to(root)):frozen_strategy.sha256_file(p) for p in set(paths)},
              "research_only":True,"places_orders":False,"promotion_pass":False,"windows":{}}
    cache = {}
    for label, window in plan["windows"].items():
        path = root/window["market_snapshot"]
        if path not in cache:
            cache[path] = sim.load_market_snapshot(path)
        original,funding,_ = cache[path]
        start,end = map(sim._utc_ms,window["utc"])
        data = {key:[c for c in bars if c.close_time_ms < end] for key,bars in original.items()}
        base = data['5m']
        if base[-1].close_time_ms+1 != end or start < base[0].open_time_ms+45*sim.MS_PER_DAY:
            raise ValueError("Invalid evaluation coverage or warmup")
        sleeves = [sim.simulate(base,replace(cfg,strategy_modes=('trend',)),start,None,funding),
                   sim.simulate_timeseries_trend(data['1h'],cfg,start,funding)]
        execution_targets.prepare_sleeves(sleeves)
        adjusted,_ = macro_regime.apply_macro_overlay(sleeves,macro,enabled_factors=('vix','dollar','metals','sentiment'))
        protected = make_sleeves(data,funding,cfg,start,plan["protected_policy"],1.0)
        execution_targets.prepare_sleeves(protected)
        protected,_ = macro_regime.apply_macro_overlay(protected,macro,enabled_factors=('vix','dollar','metals','sentiment'))
        variants = {"current_macro":adjusted,
                    "fixed_risk_control":diagnosis.constant_scale(sleeves,plan["fixed_risk_multiplier"]),
                    "protected_wait12_break24":protected}
        rows = {}
        for name,variant in variants.items():
            rows[name] = {}
            for cost in (1,2):
                result = backtest_execution.replay(base,variant,cfg,start,rates,
                                                   initial=plan["initial_equity"],cost_multiplier=cost)
                output = Path(f"{prefix}_{label}_{name}_cost{cost}.json")
                sim.save_json(output,result)
                rows[name][str(cost)] = {**result["summary"],"full_result":str(output.relative_to(root))}
                s = result["summary"]
                print(label,name,cost,f"return={s['total_return_pct']:.4f}% dd={s['max_drawdown_pct']:.4f}% fills={s['fills']}",flush=True)
        baseline,candidate = rows['current_macro'],rows['protected_wait12_break24']
        checks = {"normal_return_at_least_current":candidate['1']['total_return_pct'] >= baseline['1']['total_return_pct'],
                  "drawdown_no_worse_than_current":candidate['1']['max_drawdown_pct'] <= baseline['1']['max_drawdown_pct'],
                  "positive_double_cost_return":candidate['2']['total_return_pct'] > 0,
                  "execution_fees_no_more_than_current":candidate['1']['fees'] <= baseline['1']['fees']}
        report['windows'][label] = {"results":rows,"protected_checks":checks}
        sizes = {}
        for initial in (100,1000):
            diagnostic = backtest_execution.replay(base,adjusted,cfg,start,rates,initial=initial)
            sizes[str(initial)] = diagnostic['summary']
            print(label,'account_size',initial,'return',round(diagnostic['summary']['total_return_pct'],4),
                  'fills',diagnostic['summary']['fills'],flush=True)
        report['windows'][label]['account_size_diagnostics'] = sizes
        sim.save_json(Path(f"{prefix}.json"),report)
    report['protected_historical_checks_pass'] = all(all(row['protected_checks'].values()) for row in report['windows'].values())
    report['qualification_status'] = 'awaiting_prospective_evidence'
    sim.save_json(Path(f"{prefix}.json"),report)


if __name__ == '__main__':
    main()
