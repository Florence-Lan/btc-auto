# 双向策略重新评估与有条件前瞻模拟

用户要求替换为更有依据、能做多和做空的赚钱策略。当前仅SNDK通过本轮有限回顾性筛查；没有找到四个账户都满足要求的方案，也没有可保证盈利的证据。北京时间 **2026-10-05 17:34:56.278** 应用新准入规则，执行器PID **61363**。

## 已应用的行为

- SNDK保留已有5m双向趋势突破：8/24/60 EMA方向排列、中线斜率同向、收盘突破前6根信号线且突破距离不超过1 ATR。保留原保护止损、保证金净120%目标、15天期限、资金费预算及追踪规则。不是每五分钟必开仓。
- 股票新交易的计划风险由每笔0.5%降至0.25%，上限10x继续只是杠杆上限，总名义仓位上限仍为1倍权益。SNDK按现行RWA单边0.0125%模拟手续费计账；保留可见盘口限量、此前已收盘5m成交量10%限量、价差/资金费/追价检查和每30秒刷新。MU同为0.0125%费率假设，SKHYNIX按0.10%，但均暂停新增。
- MU、SKHYNIX及本轮联合账户中的BTC暂停新增：仍接收行情、记录信号并处理保护退出。配置均保留双向权限；重新批准后的策略需要新的规则生效后信号。
- **SKHYNIX在本次切换前的17:30已按此前策略产生一笔模拟多头成交，数量0.163。** 该仓位及其入场费用保留，继续执行原保护/退出规则，没有因资格变更强制平仓。修复后有实际成交不能证明策略盈利。
- BTC新增限制在最终账户执行门槛生效，禁止开仓、加仓及反向新仓；允许减仓、平仓。实际17:35执行记录已写出`allowed=false`与`strategy_not_qualified`。原先另一个BTC模拟账户和执行主管继续按原授权运行；此次资格限制作用于四账户联合模拟中的BTC。
- 终端每账户显示准入原因，BTC仍在两列布局的右下。历史资金、成交、费用和资金费保持，不重置收益。净目标计算支持实际已付入场费用，避免未来更改费率后抹掉旧入场成本。

## 声明与方法

第一阶段在计算前声明六个替代候选：20/50/200均线的趋势回调、前20根趋势突破、20根两标准差带的震荡回归，各比较15m/1h周期；加各股票当前双向策略对照，共21个候选。全部允许多空，没有网格、加倍补亏或移除亏损方向。

开发期为2026-09-01 UTC之前；只用开发期选择：正常和双倍成本已平仓净收益与期末估计平仓权益均正，至少20笔、净利润因子至少1.15、最大采样回撤不超过6%、无模型强平；按两种成本中较低的已平仓净收益排名。冻结开发期第一名后再看后续区间（9月1日至快照结束）与最近30天；分别至少5/3笔、利润因子至少1.1，两个成本下净收益正、回撤不超过6%、无强平。审查失败不改选后续区间赢家、不降低笔数门槛。

第一阶段统一单边0.10%保守费用，MU、SNDK无合格开发候选，SKHYNIX选出的现行规则在最近30天失败。

随后查到官方费用区别，单独保存**第二阶段当前费率情景**：MU/SNDK按RWA0.0125%，SKHYNIX属于Group B，按0.10%；候选和门槛保持。把今天费率应用于过去价格路径，是前瞻成本假设下的回顾性情景，不是还原过去真实手续费。官方页面列RWA1.25bp，自2026-09-07起生效，并明确列SKHYNIX为Group B：[Aster官方费率](https://docs.asterdex.com/trading/perpetuals/fees-and-specs/fees)。没有把更低费率当作盈利筛查后的参数搜索。

趋势类设计参考双向时间序列动量这一研究方向；长期分散期货研究不能验证这里的单股短周期永续策略：[AQR趋势研究](https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing)。

BTC另评估同六个新候选，既有2026-04-15至2026-10-01 12:00 UTC交易价快照；以交易价作为历史标记价、小时指数开盘代理，费用按项目原单边0.045%和1bp滑点，双倍成本重跑。该研究是价格候选对照，没有历史盘口/基差或六因素首见新闻验证，不能作为当前六因素策略的完整重放。选出的1h突破在后续与最近30天正常/双倍成本均亏损，拒绝替换。

## 筛查结果

股票价格快照结束于北京时间2026-10-04 20:00（不含），历史此前已被研究过，全部为已知历史上的回顾性研究，没有独立未见验证。以下百分比基于每账户初始1,000 USDT，只计已平仓净损益；各窗口独立空仓重置，样本重叠，不能相加。

| 账户 | 开发期冻结选择 | 双倍成本开发期净收益 | 双倍成本后续区间净收益 | 双倍成本最近30天净收益 | 决定 |
|---|---|---:|---:|---:|---|
| MU | 无候选达到开发门槛 | — | 未选后续赢家 | — | 暂停新增 |
| SNDK | 当前5m双向突破 | +5.3728%（122笔） | +1.2616%（10笔） | +1.2844%（13笔） | 小风险前瞻候选 |
| SKHYNIX | 当前15m双向交叉 | +1.9321%（25笔） | +0.3080%（13笔） | −0.2412%（11笔） | 审查失败，暂停新增 |
| BTC | 新1h趋势突破，价格代理模型 | +0.7820%（35笔） | −0.4064%（9笔） | −0.4064%（9笔） | 审查失败，拒绝替换并限制联合账户新增 |

SNDK当前费率情景完整区间双倍成本已平仓净收益+6.1401%（134笔），最大采样回撤5.0403%；后续区间净利润因子1.7887，最近30天1.7305。后续区间10笔中多头净+13.5210 USDT、空头净−0.9047；完整区间双倍成本空头合计仍小亏−0.8399 USDT，最近30天空头为+1.4224 USDT。因此通过的是整个双向规则的有限历史门槛，不是两个方向都已证明持续赚钱。循环区块重采样的完整样本/近30天单笔均值下界仍为负，盈利可信度有限。

五分钟OHLC无法重建盘口、部分成交、零量时准确路径与真实下单可成交性；小时指数开盘为已知代理；缺少结算标记价时按五分钟开盘代理资金费。当前假设费率不能消除价差、滑点、流动性和样本偏差。模拟每30秒观察与回放每5分钟一次尝试会产生不同成交序列。历史成功不是未来收益承诺。

## 验证与证据

300项相关测试通过，包含双向趋势回调/震荡回归对称性、有效信号因果性、净目标保留旧入场费用、新增资格限制不能绕过其他门槛、禁止新增但可减仓/平仓/保护退出、原有库存和资金费账本、数量/杠杆和周期执行检查。Python和JavaScript语法、差异空白检查通过。

从CSV独立重读三阶段120个情景输出、5,764条跨情景重复账本记录，逐项核对笔数、净损益与`净损益=毛损益−开仓费用−平仓费用−资金费`。这些不是5,764笔独立成交。股票状态与停机后备份逐项核对：现金、累计毛损益、费用、资金费、成交、库存及账户开始时间保留；SKHYNIX开放仓位完整保留，浮动权益随后随行情更新。

备份：`data/parallel_simulation/btc_memory_stocks_1000_each_10x_20261005/strategy_screen_backups/1791192896129`。三个股票新规则共同生效时点由rule_revisions自动记录，BTC准入政策另写manifest.entry_policy_revisions。服务首轮四个账户均healthy，错误日志为空；未通过策略的正常采集状态与交易准入状态分开显示。

- [当前股票配置](../config/stock_screened_bidirectional_forward_20261005.json)、[运行计划](../config/parallel_simulation_plan_20261005.json)
- [保守成本结果](../data/research/bidirectional_regimes_20261005/results.json)、[声明](../data/research/bidirectional_regimes_20261005/declaration.json)、[核对](../data/research/bidirectional_regimes_20261005/verification.json)
- [现行费率情景](../data/research/bidirectional_regimes_current_fees_20261005/results.json)、[声明](../data/research/bidirectional_regimes_current_fees_20261005/declaration.json)、[核对](../data/research/bidirectional_regimes_current_fees_20261005/verification.json)
- [BTC价格候选](../data/research/btc_bidirectional_regimes_20261005/results.json)、[声明](../data/research/btc_bidirectional_regimes_20261005/declaration.json)、[核对](../data/research/btc_bidirectional_regimes_20261005/verification.json)

复现使用新目录保留证据，脚本默认读取切换前冻结配置，而不是当前动态运行配置：

```sh
.venv/bin/python scripts/research_bidirectional_regimes.py --output-dir data/research/bidirectional_regimes_reproduction
.venv/bin/python scripts/research_bidirectional_regimes.py --current-fee-scenario --output-dir data/research/bidirectional_regimes_current_fees_reproduction
.venv/bin/python scripts/research_bidirectional_regimes.py --btc-snapshot data/snapshots/btc_strategy_review_20260415_20261001_1200.json.gz --output-dir data/research/btc_bidirectional_regimes_reproduction
```
