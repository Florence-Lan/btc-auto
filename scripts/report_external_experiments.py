"""Build a human-readable report and static comparison chart from frozen runs."""
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/external_experiments_20260917"
NAMES = {
    "baseline_003": "原策略 0.30%", "candidate_004": "现有候选 0.40%",
    "atr_05": "EMA / ATR 门槛 0.5", "atr_10": "EMA / ATR 门槛 1.0",
    "ensemble": "三周期 EMA 等权", "donchian": "Donchian 20/10",
    "donchian_volume": "Donchian + 成交量", "donchian_atr": "Donchian + ATR",
    "vwap_rsi": "VWAP + RSI 替换短线",
}


def main():
    report = json.loads((OUT/"report.json").read_text(encoding="utf-8"))
    results = report["results"]
    # Check old control results before presenting any new experiments.
    previous = json.loads((ROOT/"data/validation/trend_filter_20260917/report.json").read_text(encoding="utf-8"))["results"]
    for window in ("development","validation","recent_50d"):
        for new,old in (("baseline_003","baseline"),("candidate_004","spread_004")):
            for key in ("total_return_pct","max_drawdown_pct","trades"):
                assert abs(results[window][new][key]-previous[window][old][key]) < 1e-9, (window,new,key)
    lines = ["# 外部策略实验 — 2026-09-17", "", "已完成 8 个外部方法实验（含独立资金费率套利），另设 0.30% 和 0.40% 两个对照。运行配置未切换。", "", "## 口径", "", "开发期：2022-01-01 至 2025-01-01 UTC；后续验证：2025-01-01 至 2026-06-28 15:00 UTC；最近 50 天：2026-07-28 16:20 至 2026-09-16 16:20 UTC。所有方向策略均进行正常成本和双倍手续费/滑点回测。", "", "各窗口独立空仓开始，45 天预热；按已收盘 K 线生成信号，下一根开盘执行，含期末模拟平仓、资金费和宏观过滤。方向组合保留 2 倍杠杆上限及原回撤控制。相同上限不代表相同实际风险。", "", "## 完整比较", "", "每格为 **净收益 / 最大回撤**。", "", "| 实验 | 开发期 | 后续验证 | 最近 50 天 | 双倍成本后续验证 |", "|---|---:|---:|---:|---:|"]
    for name,title in NAMES.items():
        cells = []
        for window in ("development","validation","recent_50d","validation_double_cost"):
            s=results[window][name]
            cells.append(f"{s['total_return_pct']:.2f}% / {s['max_drawdown_pct']:.2f}%")
        lines.append("| "+title+" | "+" | ".join(cells)+" |")
    lines += ["", "## 最近 50 天交易质量", "", "| 实验 | 交易数 | 胜率 | 利润因子 | 手续费及滑点（初始 100 USDT） |", "|---|---:|---:|---:|---:|"]
    for name,title in NAMES.items():
        s=results["recent_50d"][name]
        pf=f"{s['profit_factor']:.2f}" if s['profit_factor'] is not None else "无亏损/未定义"
        lines.append(f"| {title} | {s['trades']} | {s['win_rate_pct']:.2f}% | {pf} | {s['total_costs']:.3f} |")
    lines += ["", "## 独立资金费率套利", "", "买入现货、做空等量永续，每条腿初始名义金额为总资金的 50%，剩余资金用于合约保证金。按保证金缓冲触发再平衡；含观察到的现货/合约基差和额外 20bps 期末不利基差压力。以资金费结算时点估值，窗口从首个可用结算点开始、最后结算点结束，不能与 5 分钟方向组合做完全同口径比较；结算间的回撤可能更大。", "", "| 窗口 | 净收益 | 结算时点最大回撤 | 保证金违约 |", "|---|---:|---:|---|"]
    for key,s in report["funding_carry"].items():
        lines.append(f"| {key} | {s['total_return_pct']:.2f}% | {s['max_drawdown_pct']:.2f}% | {s['liquidated']} |")
    shortlist=report["shortlist"]
    lines += ["", "## 筛选结论", "", "通过预设研究筛选条件："+("、".join(NAMES[n] for n in shortlist) if shortlist else "无")+"。", "", "筛选要求开发、后续、近期收益均为正且回撤不超过 15%；后续验证利润因子至少 1.3，收益不低于 0.40% 候选、回撤不高于它；双倍成本后续验证仍盈利。筛选仅代表后续观察资格，不等于实盘批准。未根据结果追加调参。", "", "## 验证与限制", "", "- 两个对照在三个窗口的收益、回撤和交易数均精确复现此前报告。", "- 研究信号引擎在三个实际窗口中与冻结原引擎的 EMA 交易时间、数量、费用及盈亏逐笔一致。", "- 信号前缀不变性、下一根执行、平仓信号不误开仓、原引擎一致性测试通过。", "- 历史窗口已用于过往研究，后续验证不是独立前瞻样本；最近 50 天也已被观察。", "- 这些是外部思路的明确规则化改编，非精确复刻他人策略。VWAP 策略的退出规则由本次实验补全，止损在收盘触发、下根开盘成交。", "- 沿用原引擎的成交量冲击模型与组合记账：成交冲击使用执行 K 线成交量，组合在平仓时结算费用和资金费，不能代替逐笔实盘账本。", "- 没有回放 LLM 审核，也没有订单簿、OI 或盘口历史，故未测试 X 上规则不完整的订单流/套利宣传。", "", "## 复现与文件", "", "运行 `python scripts/research_external_strategies.py`，然后 `python scripts/report_external_experiments.py`。", "", "- `plan.json`：运行前固定的规则、窗口、门槛、来源。", "- `report.json`：全部结果、校验和、实验限制。", "- `comparison.csv`：54 组方向实验汇总。", "- `*_trades.csv`：逐笔交易。", "- `*_equity.json.gz`：完整资金曲线。", "- `comparison.png`：资金曲线及收益/回撤图。", "", "## 外部来源", ""]
    lines += [f"- {url}" for url in report["plan"]["sources"]]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    chosen=["baseline_003","candidate_004","atr_10","ensemble","donchian","vwap_rsi"]
    for ax,window,title in ((axes[0,0],"validation","Validation: 2025-01 to 2026-06"),(axes[0,1],"recent_50d","Recent 50 days")):
        for name in chosen:
            with gzip.open(OUT/f"{window}_{name}_equity.json.gz","rt",encoding="utf-8") as handle:
                curve=json.load(handle)
            step=max(1,len(curve)//2000)
            sample=curve[::step]
            if sample[-1] is not curve[-1]: sample.append(curve[-1])
            ax.plot([datetime.fromtimestamp(p["time_ms"]/1000,timezone.utc) for p in sample],[p["equity"] for p in sample],label=name,linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel("Equity (initial 100 USDT)")
        ax.tick_params(axis="x",rotation=20)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    for ax,metric,title in ((axes[1,0],"total_return_pct","Validation return: normal vs double costs"),(axes[1,1],"max_drawdown_pct","Validation maximum drawdown")):
        names=list(NAMES)
        for offset,window,label in ((-.2,"validation","Normal"),(.2,"validation_double_cost","Double costs")):
            ax.bar([i+offset for i in range(len(names))],[results[window][name][metric] for name in names],width=.4,label=label)
        ax.set_xticks(range(len(names)),names,rotation=45,ha="right",fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.axhline(0,color="black",linewidth=.6)
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("External strategy experiments — historical research, not live results",fontsize=14)
    fig.savefig(OUT/"comparison.png",dpi=160)
    plt.close(fig)
    print(OUT/"summary.md")


if __name__ == "__main__":
    main()
