# 全球事件风险指标：研究原型

本版本只通过显式 `--world-event-snapshot` 参数接入影子回放，不修改实盘配置、现有默认启动命令或事件模板。没有启动常驻采集任务。

## 本轮验证目标

1. 未来新闻、未来审核结论对过去决策的影响必须为 0。
2. 重复报道不能增加事件分数或重置事件衰减时间；最终仓位倍率不超过原仓位。
3. 数据故障必须可识别，不能被解释为风险为 0；模拟数据和真实采集分开。

以上是工程验收目标，不代表收益预测。收益研究还需前瞻记录事件，再比较原策略、仅宏观、宏观加事件的最大回撤、事件后尾部损失、扣费净收益和交易机会损失。应设置等平均仓位的对照组，区分择时效果和单纯减仓效果。历史新闻事后补录不能拿来证明历史策略收益。

## 分数与行为

单条证据分数：

```text
100 × 严重度 × 意外程度 × 新信息比例 × 可信度
    × 2 ^ (-事件首次获知以来的小时数 / 半衰期小时数)
```

四个输入均为 0～1；半衰期为 (0, 168] 小时，事件最长保留 7 天影响。`watch_score` 是观察分数，`score` 是已确认事件分数，均非概率。参数为未经收益优化的研究初值。

- 自动采集只做关键词粗分类。命中候选初始四项为 0.5、0.5、0.5、0.3；未分类消息严重度为 0。因此观察分数只表示发现候选，不能解释为世界局势的真实严重程度。
- 经审核且可信度至少 0.7 的一手证据，或两组独立来源的已审核证据，才能贡献 `score`。`primary_source` 指与事件有关的一手公告/证据，不是媒体品牌知名度。
- 同一个 `event_id` 取最大分数，不对转载数量求和。`source_group` 应按原始采编来源填写：转载同一通讯社的多个网站只能算一组。
- 自动聚类仅识别同标题；跨标题语义聚类由审核时指定 `event_id` 完成。真正新增的事件进展使用新的事件 ID，旧闻转载继续使用原 ID。
- 事件仓位上限为 `max(0.25, 1 - 0.75 × score / 100)`。与宏观上限取较小值；现有独立静态事件风控、回撤控制仍单独生效。
- 此原型没有新闻做多/做空、自动加仓、新闻强制平仓功能。它只调整模拟入场交易的规模；既有交易不会因之后的新闻被重新缩放。
- 数据超过 2 小时未成功检查、接口失败、GDELT 返回达到 250 条上限、尚无采集历史时，暂停该实验的新模拟入场。健康只表示配置的数据源请求成功，不能保证全球事件覆盖完整。

## 采集与查看

无需交易所密钥。每次执行只采集一次，建议实验时每 15 分钟运行，同一日志只保留一个写入进程。

```powershell
python scripts/research_world_events.py collect
# GDELT 不可用时，可显式改用 BBC World 单一来源样本：
python scripts/research_world_events.py collect --provider bbc-world
python scripts/research_world_events.py report
```

默认日志为 `data/world_events/observations.json`，报告为 `data/world_events/report.json`。这些运行数据已受仓库现有 data 忽略规则保护，不自动提交。

GDELT 查询最近 6 小时的英文报道，最多 250 条；BBC 为 World RSS 的当前文章列表，两者都不是完整的全球实时新闻流。日志保留来源 URL、标题、供应商时间、本机首次收取时间、评分时间与每次采集状态。供应商时间不作为本机提前获知事件的证据。

采集会按去除常见跟踪参数后的 URL 去重；重新下载不会刷新首次收取时间。接口失败会保留旧证据并追加失败记录。长时间停机期间的事件不能恢复成可信的实时历史。

## 审核记录

在日志找到实际观察 ID，阅读原始来源并判断事件归类、严重度、意外程度、独立来源后，追加审核版本。例如下面参数只是格式示例，不能直接作为事实判断：

```powershell
python scripts/research_world_events.py review `
  --observation-id <实际ID> --event-id <同一事实共用的ID> `
  --source-group <原始采编来源> `
  --severity 0.8 --surprise 0.6 --novelty 1.0 --confidence 0.9 `
  --half-life-hours 12 --evidence-note "审核依据及不确定性"
```

只有真正一手证据才加 `--primary-source`；撤回错误审核可使用 `--retract`，其余必要参数同上。审核通过脚本追加版本，保留实际审核时间，不要直接改写已有记录。历史决策只取当时已知的最新版本，后来的核实或撤回不回写过去。

## 单独运行影子组合

先采集真实新闻，再执行下列命令；使用独立状态文件，避免混入原有实验。可以重复执行或使用原脚本的 `--loop`，新闻采集需独立定时运行。

```powershell
python scripts/paper_trade_frozen_portfolio.py `
  --manifest config/frozen_strategy_candidate_20260917.json `
  --strategy-modes-override trend,timeseries_trend `
  --macro-snapshot data/snapshots/macro_shadow_latest.json.gz `
  --macro-factors vix,dollar,metals,sentiment `
  --tiered-drawdown `
  --world-event-snapshot data/world_events/observations.json `
  --state-path data/paper_trading/world_events_execution_v2_state.json `
  --report-path data/paper_trading/world_events_execution_v2_report.json `
  --trades-path data/paper_trading/world_events_execution_v2_trades.csv
```

宏观快照需沿用项目现有方式保持新鲜。报告新增 `world_event_overlay`：包括入场时的阻断/减仓统计，以及当前事件分数、采集健康、活跃事件 ID。仅观察到新闻但未审核时，已确认分数为 0；这表示缺少确认，不能据此断言没有风险。

2026-10-07 已整合新版执行路径。世界事件、情报与六因子入场上限取较严格值；缩仓前保存现金与库存账本，保持部分退出、资金费及小时已知开盘目标一致。启用本参数的报告标记为研究用途，不能送入实盘执行器。新版状态配置与旧实验可能不同，示例使用独立 `execution_v2` 路径，旧账本不迁移或覆盖。本次只更新代码和离线测试，未启动研究或交易循环。详见 [更新与验证记录](remote_integration_20261007.md)。

现有组合回放是对已生成的交易按入场时间调整，不能完全模拟事件导致的后续交易路径变化、成交约束或盘中平仓。绩效必须进一步通过前瞻影子观察验证。

## 离线演示与测试

```powershell
python scripts/research_world_events.py demo --output data/world_events/demo_report.json
python -m pytest tests/test_world_event_risk.py tests/test_candidate_execution.py -q
```

演示全部使用虚构事件和虚构交易，报告明确标记 `synthetic`；不会写入真实采集日志。示例风险从 76.95 分开始，6 小时后减半至 38.475；宏观仓位上限为 60% 时，事件发生时组合上限为约 42.29%，而不是重复相乘后的约 25.37%。减仓同样会缩小盈利，这不是盈利回测。

来源说明：[GDELT DOC API 官方文档](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)、[BBC World RSS](https://feeds.bbci.co.uk/news/world/rss.xml)。

## 收益验证

执行 `python scripts/validate_world_event_returns.py`，可复现 [验证结果](world_event_validation.md)。默认使用本地 50 天历史快照，比较冻结策略、现有宏观风控和同平均入场名义金额的固定减仓对照，同时运行双倍手续费/滑点压力测试和配对区块重采样。

事件日志必须覆盖历史区间的至少 99% 时间和全部候选入场，才能生成事件收益对比；否则明确输出不可估计。健康采集但没有已确认事件，只能作为零处理效应对照，不能证明指标有效。报告保留输入和研究脚本哈希，宏观对照收益不能冒充新事件指标的收益。
