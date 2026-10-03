"""Reproduce the hourly timing regression against frozen local market data, without orders."""
import argparse
from dataclasses import replace
from pathlib import Path

import backtest_execution
import execution_portfolio
import execution_targets
import frozen_strategy
from paper_trade_frozen_portfolio import annotate_open_position_fractions
import portfolio_risk
import simulate_range_swing as sim
import timeseries_execution as hourly


def main():
    root=sim.repo_root()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-snapshot",type=Path,
        default=root/"data/snapshots/btc_strategy_review_20260415_20261001_1200.json.gz")
    parser.add_argument("--funding-snapshot",type=Path,
        default=root/"data/snapshots/funding_execution_20250101_20261001_1200.json")
    parser.add_argument("--output",type=Path,default=root/"data/validation/execution_integrity_fixed_20261003.json")
    args=parser.parse_args()
    manifest_path=root/"config/frozen_strategy_candidate_20260917.json"
    manifest,cfg=frozen_strategy.load_frozen_strategy(manifest_path)
    cfg=replace(cfg,max_drawdown_stop_pct=0)
    original,funding,_=sim.load_market_snapshot(args.market_snapshot)
    start=sim._utc_ms("2026-07-03T12:00:00+00:00")
    stop=sim._utc_ms("2026-07-05T00:00:00+00:00")
    data={k:[b for b in bars if start-45*sim.MS_PER_DAY <= b.open_time_ms and b.close_time_ms < stop]
          for k,bars in original.items()}
    import json
    rates=json.loads(args.funding_snapshot.read_text())["events"]
    legacy=sim.simulate_timeseries_trend(data["1h"],cfg,start,funding)
    execution_targets.prepare_sleeves([legacy])
    legacy_replay=backtest_execution.replay(data["5m"],[legacy],cfg,start,rates,initial=1000)
    full=hourly.build_sleeve(data["1h"],cfg,start,funding,asof_ms=stop-1)
    annotate_open_position_fractions([full])
    execution_targets.prepare_sleeves([full])
    result=execution_portfolio.combine(data["5m"],[full],cfg,portfolio_risk.DrawdownRiskPolicy(),start,
        decision_open_prices=hourly.decision_open_prices(data["5m"],[full]))
    lookup={p["time_ms"]:p for p in execution_targets.target_stream(data["5m"],[full],result,cfg)}
    fixed_replay=backtest_execution.replay(data["5m"],[full],cfg,start,rates,initial=1000)
    checks=[]
    for hour in (15,16,17,18):
        at=sim._utc_ms(f"2026-07-03T{hour:02}:00:03+00:00")
        base=[b for b in data["5m"] if b.close_time_ms <= at]
        closed=[b for b in data["1h"] if b.close_time_ms <= at]
        opening=hourly.opening_from_base(data["5m"],closed[-1].close_time_ms+1,at)
        core=hourly.build_sleeve(closed,cfg,start,funding,opening=opening,asof_ms=at)
        annotate_open_position_fractions([core])
        execution_targets.prepare_sleeves([core])
        partial=execution_portfolio.combine(base,[core],cfg,portfolio_risk.DrawdownRiskPolicy(),start,
            include_execution_target=True,decision_open_prices=hourly.decision_open_prices(base,[core]))
        target=execution_targets.current_target(base,[core],partial,cfg)
        expected=lookup[target["time_ms"]]
        assert abs(target["signed_qty"]-expected["signed_qty"])<1e-10
        assert abs(target["equity"]-expected["equity"])<1e-7
        assert target["position_id"]==expected["position_id"]
        checks.append({"asof_utc":sim.iso_utc_from_ms(at),"target":target,
                       "full_history_quantity":expected["signed_qty"],"matches":True})
    assert checks[0]["target"]["signed_qty"]>0
    assert fixed_replay["fills"][0]["time_utc"]=="2026-07-03T15:00:03+00:00"
    assert legacy_replay["fills"][0]["time_utc"]=="2026-07-03T15:05:03+00:00"
    paths=[args.market_snapshot,args.funding_snapshot,manifest_path]
    paths += [root/"scripts"/n for n in ("timeseries_execution.py","execution_portfolio.py",
        "execution_entry_gate.py","execution_targets.py","trading_execution.py","backtest_execution.py",
        "paper_trade_frozen_portfolio.py","validate_execution_integrity.py")]
    report={"research_only":True,"places_orders":False,"runtime_account_written":False,
        "model":hourly.MODEL,"freeze_id":manifest["freeze_id"],"cutoff_checks":checks,
        "legacy_first_fill":legacy_replay["fills"][0],"fixed_first_fill":fixed_replay["fills"][0],
        "frozen_engine_and_config_verified":True,
        "input_sha256":{str(p.relative_to(root)):frozen_strategy.sha256_file(p) for p in paths},
        "limitation":"Core-only timing regression; not evidence of profitability or real orderbook execution."}
    sim.save_json(args.output,report)
    print("Historical cutoff checks passed:",len(checks))
    print("Legacy first fill:",report["legacy_first_fill"]["time_utc"])
    print("Fixed first fill:",report["fixed_first_fill"]["time_utc"])
    print("Saved",args.output)


if __name__=="__main__":
    main()
