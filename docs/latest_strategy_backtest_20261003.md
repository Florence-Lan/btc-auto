# 当前策略统一回测（2026-10-03）

运行和回测统一读取 `config/active_simulation_candidate.json`。本次实际选择 `btc_multifactor_trial_20261002`，包含六组因素、新闻及日历、政策预期、账户风控，以及 `closed_signal_known_open_v1` 小时执行修复。选择文件、配置和运行报告的因子配置哈希核对一致：`c940e92df52782daef984eb213597863eab54bbc7acaca7b0c9558cef6a6c6b5`。

## 入口已统一

`scripts/backtest_execution.py` 默认加载当前选定配置、六因素快照和公开上下文，不再强制四因素宏观输入。每笔原始入场经过因素、事件和来源覆盖检查；每个账户执行时点再次检查当前许可，包含旧目标重试。小时目标使用已收盘信号与已知下一根开盘价。账户规模默认从选定配置读取，本次为 1,000 USDT，成本、资金费、数量门槛和回撤控制通过终端相同的 `SimulationAccount` 回放。

`run_multifactor_shadow.py` 与 `validate_multifactor.py` 的默认配置同样读取当前选择。独立观察使用 `*_shadow_*` 文件，保留终端已有的观察状态。README 已更新当前策略说明和回测命令。

## 本次结果

按最新规则回放现有首见归档可用区间：北京时间 **2026-09-27 10:15 至 2026-10-03 11:10**，结束时间不含。所有输入先复制归档，回放期间来源刷新不改变结果。此区间包含正式启用前的日期，属于当前参数的回顾性诊断；真实前瞻起点仍为北京时间 10 月 2 日 19:15:03。

| 成本设置 | 初始资金 | 期末权益 | 净收益 | 最大回撤 | 成交次数 |
|---|---:|---:|---:|---:|---:|
| 正常成本 | 1,000 USDT | 1,000 USDT | 0.0000% | 0.0000% | 0 |
| 双倍手续费及滑点 | 1,000 USDT | 1,000 USDT | 0.0000% | 0.0000% | 0 |

按当前规则重算行情只出现一条价格入场记录，时间为北京时间 9 月 30 日 21:00。该时点缺少 BTC 动量、衍生品持仓、美联储和全球风险组的有效首见数据，且已有组分数与多头方向冲突，因此被拒绝，没有形成账户成交。

回放实际进行了 **1,738 次执行前检查**。六组因素齐全的时点为 175 个，覆盖 **10.07%**；公开来源健康的时点为 166 个，覆盖 **9.55%**。缺失和过期记录按原规则阻止新增风险。不能将这些归档缺口产生的空仓解释为已证明的过滤收益，也不能用零回撤证明风控已通过经济验收。本次没有足够的有效成交样本评价盈利能力。

## 验证与复现

全套 **228 项测试通过**，包含默认入口跟随当前选择、执行时重查事件窗口、配置/来源归档、账户规模读取、拒绝缺失历史，以及独立观察不覆盖终端状态。运行配置、账户资金和原始起点核对通过，账户仍为 1,000 USDT、空仓、零成交。本次回测账户只存在内存中，没有交易所订单。

汇总与逐步账户结果：

- [统一回测汇总](../data/validation/latest_strategy_20261003/results.json)
- [正常成本完整回放](../data/validation/latest_strategy_20261003/cost1_full.json)
- [双倍成本完整回放](../data/validation/latest_strategy_20261003/cost2_full.json)
- [固定输入、实际前瞻起点与 SHA-256](../data/validation/latest_strategy_20261003/captured_inputs.json)
- [两次实际执行命令](../data/validation/latest_strategy_20261003/commands.json)

用当前选定策略再次运行：

```sh
.venv/bin/python scripts/backtest_execution.py \
  --market-snapshot data/validation/current_backtest_20261003/market_extended.json.gz \
  --funding-snapshot data/validation/current_backtest_20261003/funding_extended.json \
  --start-utc 2026-09-27T02:15:00Z --end-utc 2026-10-03T03:10:00Z \
  --output data/validation/latest_strategy_backtest.json
```

添加 `--cost-multiplier 2` 可复现双倍执行成本。命令会自动读取当前选择及其因素、事件路径，并再次归档输入；固定复现本次输入可使用 `commands.json` 中的完整命令。项目默认忽略数据目录，复现须保留本地输入文件。

回放采用开盘价代理、3 秒结算延迟和 5 分钟风险采样，未模拟真实订单簿和可选 LLM 决策；被过滤信号之后的机会路径未完整重算。缺少首见新闻或预期归档的更长历史，不用后来抓取的数据填补。当前模拟调度正在等待已记录的 Binance 冷却期限，北京时间 14:17 后自动重试；本次固定输入行情截至 11:10。
