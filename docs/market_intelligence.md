# BTC 资金流与链上观察

本轮目标是扩大可追溯的观测范围，并保留可验证的历史，避免仅凭新闻或单一情绪判断方向。

工程验收：未来数据影响过去决策为 0；缺失方向信号不能填成中性；模拟仓位不高于原策略；重复成交不重复计入同一批次；观察进程不包含签名、账户认证或下单接口。

## 查看与运行

已有交易终端运行时，打开 [资金流与链上观察](http://127.0.0.1:8766/terminal/intelligence.html)，也可从 BTC 账户页的“资金流与链上观察”入口进入；观察页返回链接指向 BTC 账户。

```powershell
# 单次采集
python scripts/monitor_market_intelligence.py
# 持续观测，每 60 秒一轮
python scripts/monitor_market_intelligence.py --loop --poll-seconds 60
# 请求本监控完成当前网络请求后退出，不影响交易机器人
python scripts/monitor_market_intelligence.py --stop
```

单写入锁避免重复启动。运行信息、最新报告及日志位于 `data/market_intelligence/`；`observations.sqlite3` 追加记录原始响应和本机获取时间，相同原始响应按哈希去重并压缩。运行期间保留 SQLite 的 WAL 文件，不要只拷贝主文件当作完整备份。

持续任务仅在本机进程存活、联网时工作；没有添加开机启动或外部告警。预计每轮最多 20 次行情请求，新闻另约每 15 分钟更新；数据库随时间增长，不自动删除研究证据。进程意外退出时页面会提示过期，需要重新启动。

## 观测内容及解释

| 类别 | 数据 | 当前用途与边界 |
|---|---|---|
| 现货 / 合约主动成交 | Binance 已收盘 5m K 线中，最近 1h 主动买卖名义金额差 | 方向确认；未收盘 K 线不参与 |
| 持仓量 OI | BTC 数量、USD 金额、1h BTC 数量变化 | 避免将价格上涨引起的 USD 持仓增长误认为新增仓位 |
| 多空拥挤 | 全体账户、大户账户、大户持仓三种比率 | 分开显示；账户人数不等于资金量，不能用人数比推断必涨必跌 |
| 已成交大单 | Binance 现货 / 合约最近各 1000 条聚合成交；Hyperliquid 最近成交 | 25 万 USDT/USD 阈值；显示主动方向、采样时长、跨轮缺口；不是完整流水或单个交易者身份 |
| 盘口 | 当前 100 档买卖价差、±10bp 内买卖深度 | 标明深度带是否完整；挂单可撤销，不纳入第一版方向评分 |
| 资金费率与基差 | Binance 最近资金费率、标记价相对指数价基差 | 拥挤提示；绝对资金费率 >0.1% 为研究告警阈值，不直接反向交易 |
| Hyperliquid 市场 | BTC OI、1h 资金费率、24h 成交量 | 链上衍生品平台公开接口的市场统计；与 Binance 费率周期不同 |
| 链上衍生品持仓 | 从最近 BTC 成交中选取最多 6 个去重参与地址，查询当前 BTC 有符号仓位 | 区分真实持仓多空；公开样本存在明显选择偏差，不等于全市场多空或全体鲸鱼 |
| Bitcoin 链上活动 | 最近内存池交易、总输出 ≥100 BTC 的记录 | 未确认、含找零/自转账；无法推断买卖和交易所净流入 |
| 宏观与新闻 | 原有宏观快照、每 15 分钟 BBC 新闻观察 | 保留来源时间和审核时间；宏观沿用原刷新机制 |

尚未提供可靠数据的项目明确标为不可用：交易所链上净流入、全市场链上多空、完整爆仓流水、ETF 净流入。特别是交易所净流入需要有出处、带历史版本的交易所地址标签与实体去重；不能把 Bitcoin 任意大额转账当成卖压。没有付费链上服务时，本版本也不会伪造这一类指标。

## 固定的第一版研究规则

方向分数范围 -100～100，权重为现货主动流 45%、合约主动流 35%、持仓与价格确认 20%。主动流指标为 `(主动买入金额 - 主动卖出金额) / 总成交金额`，除以 0.15 后截断到 [-1,1]。OI 只在 BTC 数量增加时弱确认已有价格方向，数量下降不自动解读成新空头。

三个方向组件缺任一个，则方向分数不可用；现货和合约相反时单独标注分歧。多空拥挤、资金费率、快速去杠杆作为独立风险提示，不反复重复计权。上述权重/阈值是未优化的研究初值，不是涨跌概率，也尚未证明有收益优势。

## 可选影子过滤器

过滤器只在某方向同时满足以下条件时，将入场上限压至原始规模的 50%：方向分数反向至少 35 分，且现货、合约 1h 净主动流均反向至少 5%。与已有宏观、六因子及世界事件入场上限取较严格值，不重复相乘；不主动创建反向交易。独立的静态事件限制和账户回撤控制仍按原规则执行。

已接入新版现金与库存账本：缩仓前保留原始交易的逐步现金流，部分退出、资金费和手续费仅随最终数量缩放一次。过滤器保留小时已收盘信号在下一已知开盘可执行的时间和目标标识。启用任一情报过滤器的报告均标记为研究用途，实盘执行器拒绝使用。

缺少入场时的归档、数据超过 5 分钟、方向组件不齐全时，保持原策略并记录未覆盖。未覆盖期间不能被算作新过滤器的有效样本。

```powershell
python scripts/paper_trade_frozen_portfolio.py `
  --manifest config/frozen_strategy_candidate_20260917.json `
  --strategy-modes-override trend,timeseries_trend `
  --macro-snapshot data/snapshots/macro_shadow_latest.json.gz `
  --macro-factors vix,dollar,metals,sentiment --tiered-drawdown `
  --intelligence-db data/market_intelligence/observations.sqlite3 `
  --state-path data/paper_trading/intelligence_execution_v2_state.json `
  --report-path data/paper_trading/intelligence_execution_v2_report.json `
  --trades-path data/paper_trading/intelligence_execution_v2_trades.csv
```

新实验使用独立状态，已进行一次接线验证，没有交易样本。持续采集已启动；影子组合当前仅单次运行，原有执行机器人不自动启用此过滤器。

上述采集与接线描述为原实验记录。2026-10-07 的代码整合没有启动或重启进程，也没有迁移或清空历史账本。新版执行模型会校验状态配置；旧实验继续保留，示例改用新的独立 `execution_v2` 路径，避免覆盖原状态。与六因子一起研究时，使用成对的 `--factor-profile` / `--factor-snapshot` 及候选要求的公共事件快照，去掉 `--macro-snapshot`；旧宏观和六因子不能同时启用。详见 [更新与验证记录](remote_integration_20261007.md)。

后续收益验证仍需同一期间、相同成本的基准/过滤器/等平均仓位三组，以及分组剔除各因子的比较。必须报告覆盖率、受影响入场数、独立事件数、最大回撤、扣费收益和区间不确定性。成交采样数据和 6 地址持仓样本目前仅观察，不用于估计全市场资金量。

## 验证

```powershell
python -m pytest tests/test_market_intelligence.py tests/test_world_event_risk.py tests/test_world_event_returns.py tests/test_candidate_execution.py -q
node --check terminal/intelligence.js
```

测试覆盖未收盘数据、时序缺口、OI 数量口径、主动成交方向、成交去重、采样偏差标记、数据错误、未来信息隔离、单写入锁及影子仓位叠加。页面通过本地 HTTP 和脚本语法检查；本会话浏览器自动化未提供可用浏览器，因此未进行视觉截图验收。

## 持续观测的复测

`python scripts/validate_intelligence_observations.py` 会在线备份正在记录的数据库、冻结共同市场区间，再进行采集健康检查、下一观测价入场的非重叠 1h 方向诊断，以及基准/过滤器/同入场金额三组组合回放。不会停止监控或修改模拟账户。运行产物在 `data/validation/intelligence_observations/`。

使用 `--replay-directory <某次产物目录>` 可以直接复用冻结的数据库和行情，离线复现结果。方向诊断未计手续费、滑点、资金费，不是可执行策略的收益；没有入场时，收益对比标记为不可验证。

最近一次完整复测见 [2026-09-24 结果](intelligence_retest_20260924.md)。

官方口径：[Binance 市场数据](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)、[Hyperliquid BTC 市场与账户状态](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)、[Hyperliquid 成交方向与持仓符号](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/notation)、[Bitcoin mempool API](https://mempool.space/docs/api/rest)、[交易所地址指标的标签修订说明](https://docs.glassnode.com/basic-api/metadata)。
