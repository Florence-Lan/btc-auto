#!/usr/bin/env python
"""Compare fixed strategy overlays; refuse to score missing historical news as alpha."""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import frozen_strategy
import macro_regime
import portfolio_risk
import simulate_range_swing as sim
import world_event_risk as world
from download_market_snapshot import validate_contiguous

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ("vix", "dollar", "metals", "sentiment")


def scaled(sleeves, multiplier):
    if not 0 <= multiplier <= 1:
        raise ValueError("Control multiplier must be in [0, 1]")
    result = []
    for sleeve in sleeves:
        trades = []
        for raw in sleeve["trades"]:
            row = dict(raw)
            for field in world.SCALE_FIELDS:
                if field in row:
                    row[field] *= multiplier
            trades.append(row)
        result.append({**sleeve, "trades": trades})
    return result


def total_entry_notional(sleeves):
    return sum(abs(float(t["initial_qty"]) * float(t["entry_price"]))
               for s in sleeves for t in s["trades"])


def coverage(snapshot, sleeves, start_ms, end_ms):
    # Integrate actual healthy poll intervals. A later successful fetch cannot
    # fill historical outages, and healthy-but-unreviewed is not an event sample.
    polls = sorted(snapshot["polls"], key=lambda p: world.utc_ms(p["available_at_utc"]))
    healthy_ms = 0
    for index, poll in enumerate(polls):
        left = world.utc_ms(poll["available_at_utc"])
        next_time = world.utc_ms(polls[index + 1]["available_at_utc"]) if index + 1 < len(polls) else end_ms
        right = min(left + 2 * world.HOUR_MS, next_time, end_ms)
        if poll["status"] == "ok":
            healthy_ms += max(0, right - max(left, start_ms))
    entries = [t for s in sleeves for t in s["trades"]]
    known = confirmed = impacted = 0
    ids = set()
    for trade in entries:
        decision = world.decision_at(snapshot, world.utc_ms(trade["entry_time_utc"]))
        known += int(decision.allowed)
        confirmed += int(decision.allowed and decision.score > 0)
        if decision.allowed and decision.risk_multiplier < trade.get("macro_risk_multiplier", 1.0):
            impacted += 1
            ids.update(decision.confirmed_event_ids)
    return {"healthy_time_pct": healthy_ms / (end_ms - start_ms) * 100,
            "eligible_entries": len(entries), "healthy_entries": known,
            "healthy_entry_pct": known / len(entries) * 100 if entries else 0.0,
            "confirmed_event_entries": confirmed, "changed_entries": impacted,
            "distinct_impact_events": len(ids), "impact_event_ids": sorted(ids),
            "can_compare": bool(entries and known == len(entries)
                                and healthy_ms / (end_ms - start_ms) >= 0.99),
            "reviewed_records_known_by_end": sum(r["verified"] and
                world.utc_ms(r["assessed_at_utc"]) < end_ms for r in snapshot["observations"])}


def daily_returns(result):
    by_day = {}
    for point in result["equity_curve"]:
        by_day[int(point["time_ms"]) // sim.MS_PER_DAY] = float(point["equity"])
    previous = float(result["summary"]["initial_equity"])
    returns = {}
    for day, equity in sorted(by_day.items()):
        if previous <= 0:
            raise ValueError("Nonpositive equity in paired comparison")
        returns[day] = equity / previous - 1
        previous = equity
    return returns


def paired_bootstrap(candidate, control, samples=2000, block_days=7, seed=20260923):
    a, b = daily_returns(candidate), daily_returns(control)
    if set(a) != set(b):
        raise ValueError("Paired comparisons require identical observation dates")
    days = sorted(a)
    if len(days) < block_days * 2:
        return {"status": "insufficient_days", "ci95_return_delta_pp": None}
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = []
        while len(indices) < len(days):
            begin = rng.randrange(len(days) - block_days + 1)
            indices.extend(range(begin, begin + block_days))
        chosen = indices[:len(days)]
        gain_a = math.prod(1 + a[days[i]] for i in chosen)
        gain_b = math.prod(1 + b[days[i]] for i in chosen)
        deltas.append((gain_a - gain_b) * 100)
    deltas.sort()
    return {"status": "descriptive_in_sample", "samples": samples, "block_days": block_days,
            "days_including_partial_endpoints": len(days), "seed": seed,
            "return_delta_pp": candidate["summary"]["total_return_pct"] - control["summary"]["total_return_pct"],
            "ci95_return_delta_pp": [deltas[int((samples - 1) * .025)], deltas[int((samples - 1) * .975)]]}


def compact(result):
    keys = ("total_return_pct", "final_equity", "max_drawdown_pct", "trades", "win_rate_pct",
            "profit_factor", "total_fees", "total_slippage", "total_funding_pnl", "worst_trade")
    summary = {k: result["summary"].get(k) for k in keys}
    losses = sorted(t["net_pnl"] for t in result["trades"])
    summary["worst_5pct_trade_mean_pnl"] = (sum(losses[:max(1, math.ceil(len(losses) * .05))]) /
                                           max(1, math.ceil(len(losses) * .05))) if losses else None
    summary["risk_diagnostics"] = result.get("risk_diagnostics")
    return summary


def run_case(cfg, base, trend, funding, macro, news, start_ms, end_ms):
    sleeve_cfg = replace(cfg, max_drawdown_stop_pct=0.0)
    sleeves = [sim.simulate(base, replace(sleeve_cfg, strategy_modes=("trend",)), start_ms, None, funding),
               sim.simulate_timeseries_trend(trend, replace(sleeve_cfg, strategy_modes=("timeseries_trend",)), start_ms, funding)]
    macro_sleeves, diagnostics = macro_regime.apply_macro_overlay(sleeves, macro, enabled_factors=FACTORS)
    def combine(inputs):
        return portfolio_risk.combine_sleeves_with_drawdown_policy(base, inputs, cfg,
                        portfolio_risk.DrawdownRiskPolicy(8, 15, .35), start_ms)
    denominator = total_entry_notional(sleeves)
    factor = total_entry_notional(macro_sleeves) / denominator if denominator else 0.0
    results = {"baseline": combine(sleeves), "macro": combine(macro_sleeves),
               "constant_size_control": combine(scaled(sleeves, factor))}
    cov = coverage(news, macro_sleeves, start_ms, end_ms)
    world_diagnostics = None
    event_comparisons = None
    event_factor = None
    if cov["can_compare"]:
        adjusted, world_diagnostics = world.apply_overlay(macro_sleeves, news)
        results["macro_world"] = combine(adjusted)
        event_factor = total_entry_notional(adjusted) / total_entry_notional(macro_sleeves)
        results["event_constant_size_control"] = combine(scaled(macro_sleeves, event_factor))
        event_comparisons = {"vs_macro": paired_bootstrap(results["macro_world"], results["macro"]),
                            "vs_size_control": paired_bootstrap(results["macro_world"], results["event_constant_size_control"])}
    return {"summaries": {k: compact(v) for k, v in results.items()},
            "macro_average_entry_notional_scale": factor, "event_average_entry_notional_scale": event_factor,
            "macro_diagnostics": diagnostics, "event_diagnostics": world_diagnostics,
            "news_coverage": cov, "event_comparisons": event_comparisons,
            "macro_vs_baseline": paired_bootstrap(results["macro"], results["baseline"]),
            "macro_vs_size_control": paired_bootstrap(results["macro"], results["constant_size_control"])}


def write_markdown(path, report):
    lines = ["# 全球事件指标收益验证", "", f"结论：**{report['conclusion']}**", "",
             f"区间：{report['start_utc']} 至 {report['end_exclusive_utc']}，{report['days']} 天。初始资金 100 USDT。", "",
             "| 方案 | 扣费收益 | 最大回撤 | 交易数 | 盈利因子 |", "|---|---:|---:|---:|---:|"]
    labels = {"baseline": "原策略＋相同回撤保护", "macro": "加入现有宏观风控",
              "constant_size_control": "同平均入场金额固定减仓", "macro_world": "宏观＋新事件指标",
              "event_constant_size_control": "事件同仓位对照"}
    for name, row in report["normal_cost"]["summaries"].items():
        pf = f"{row['profit_factor']:.3f}" if row['profit_factor'] is not None else "—"
        lines.append(f"| {labels[name]} | {row['total_return_pct']:.3f}% | {row['max_drawdown_pct']:.3f}% | {row['trades']} | {pf} |")
    cov = report["normal_cost"]["news_coverage"]
    if not cov["can_compare"]:
        lines.append("| 宏观＋新事件指标 | 数据不足 | 数据不足 | — | — |")
    comparison = report["normal_cost"]["macro_vs_size_control"]
    interval = comparison["ci95_return_delta_pp"]
    uncertainty = (f"收益差 {comparison['return_delta_pp']:+.3f} 个百分点，95% 区间 [{interval[0]:+.3f}, {interval[1]:+.3f}] 个百分点。"
                   if interval is not None else "样本天数不足，不能估计区间。")
    lines += ["", f"历史新闻健康覆盖 {cov['healthy_time_pct']:.2f}%；已核实事件影响入场 {cov['changed_entries']} 次。", "",
              "## 仓位与不确定性", "",
              f"宏观风控保留原策略总入场名义金额的 {report['normal_cost']['macro_average_entry_notional_scale'] * 100:.2f}%。固定减仓对照使用相同倍率。",
              f"宏观相对固定减仓：{uncertainty}", "",
              "## 双倍手续费和滑点", ""]
    for name, row in report["double_cost"]["summaries"].items():
        lines.append(f"- {labels[name]}：收益 {row['total_return_pct']:.3f}%，最大回撤 {row['max_drawdown_pct']:.3f}%。")
    lines += ["", "## 解释限制", ""] + [f"- {line}" for line in report["limitations"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/frozen_strategy_candidate_20260917.json")
    parser.add_argument("--market", type=Path, default=ROOT / "data/snapshots/btcusdt_backtest_50d_20260917.json.gz")
    parser.add_argument("--macro", type=Path, default=ROOT / "data/snapshots/macro_backtest_50d_20260917.json.gz")
    parser.add_argument("--news", type=Path, default=ROOT / "data/world_events/observations.json")
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/world_event_returns.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/world_event_validation.md")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("days must be positive")
    manifest, cfg = frozen_strategy.load_frozen_strategy(args.manifest)
    intervals, funding, metadata = sim.load_market_snapshot(args.market)
    macro = macro_regime.load_macro_snapshot(args.macro)
    news = world.load_snapshot(args.news)
    if news.get("synthetic"):
        raise ValueError("Synthetic evidence cannot validate profitability")
    base, trend = intervals["5m"], intervals[cfg.timeseries_timeframe]
    for interval in ("5m", cfg.timeseries_timeframe):
        validate_contiguous(interval, intervals[interval])
    # Use the common closed-bar end; never replay incomplete future candles.
    end_ms = min(base[-1].close_time_ms + 1, trend[-1].close_time_ms + 1)
    start_ms = end_ms - args.days * sim.MS_PER_DAY
    if start_ms - base[0].open_time_ms < 45 * sim.MS_PER_DAY - sim.interval_to_ms(cfg.timeseries_timeframe):
        raise ValueError("Insufficient 45-day warmup for requested evaluation window")
    base = [c for c in base if c.close_time_ms < end_ms]
    trend = [c for c in trend if c.close_time_ms < end_ms]
    normal = run_case(cfg, base, trend, funding, macro, news, start_ms, end_ms)
    print("normal_cost_complete", flush=True)
    stressed_cfg = replace(cfg, maker_fee=cfg.maker_fee * 2, taker_fee=cfg.taker_fee * 2,
                           entry_slippage_bps=cfg.entry_slippage_bps * 2,
                           exit_slippage_bps=cfg.exit_slippage_bps * 2, depth_impact_bps=cfg.depth_impact_bps * 2)
    stressed = run_case(stressed_cfg, base, trend, funding, macro, news, start_ms, end_ms)
    cov = normal["news_coverage"]
    enough = (args.days >= 90 and cov["can_compare"] and cov["changed_entries"] >= 30
              and cov["distinct_impact_events"] >= 5)
    conclusion = "尚不能证明新事件指标改善收益：缺少覆盖回测期的事件证据。"
    if cov["can_compare"]:
        conclusion = "事件指标已完成描述性对比；样本或独立性不足，不能确认收益改善。"
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "places_orders": False,
              "conclusion": conclusion, "event_improvement_proven": False,
              "minimum_event_sample_met": enough,
              "minimum_event_sample_policy": {"days": 90, "changed_entries": 30, "distinct_impact_events": 5},
              "start_utc": world.iso(start_ms), "end_exclusive_utc": world.iso(end_ms), "days": args.days,
              "freeze_id": manifest["freeze_id"], "strategy_frozen_at_utc": manifest["frozen_at_utc"],
              "inputs": {str(p): frozen_strategy.sha256_file(p) for p in
                         (args.manifest, args.market, args.macro, args.news, Path(__file__),
                          ROOT / "scripts/world_event_risk.py", ROOT / "scripts/macro_regime.py",
                          ROOT / "scripts/portfolio_risk.py")},
              "market_metadata": metadata, "normal_cost": normal, "double_cost": stressed,
              "limitations": ["本次是冻结参数的历史回放；策略与宏观规则在可见历史上研究过，不是独立样本外证明。",
                  "新事件日志不能回填为当时已知的数据；没有历史覆盖时，事件收益标为不可估计，不把全程停交易记为改善。",
                  "同仓位对照匹配风控前的总入场名义金额，不保证持仓时间加权敞口或最终回撤控制后的敞口完全相同。",
                  "交易成本包含手续费、滑点、资金费；双倍成本场景加倍手续费和滑点，资金费沿用历史值。",
                  "7 天移动区块、2000 次配对重采样的区间只描述当前样本，不是未来获利概率。",
                  "叠加层按已生成交易的入场时间调整规模，未完整重算所有路径依赖；LLM 审核和真实成交约束未回放。",
                  "完整新闻覆盖、至少 90 天/30 次受影响入场/5 个独立事件仅为最低研究样本要求，仍需独立前瞻验证。"]}
    sim.save_json(args.output, report)
    write_markdown(args.markdown, report)
    for name, row in normal["summaries"].items():
        print(f"{name}: return={row['total_return_pct']:.4f}% dd={row['max_drawdown_pct']:.4f}% trades={row['trades']}")
    print(f"event_coverage={cov}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
