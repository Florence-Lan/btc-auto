# 多因素策略候选（2026-09-26）

## 9 月 27 日：公开新闻、经济日历和政策预期已接入

默认 `run_multifactor_shadow.py` 现在选择 `config/multifactor_candidate_20260927.json`，使用独立的 `btc_multifactor_20260927_*` 状态，不沿用 9 月 26 日的模拟记录。以下 9 月 26 日回测结果只属于旧候选，不能当作新版本业绩。

新接入会在该模拟程序运行时每 5 分钟刷新：

| 数据 | 实际来源与用途 |
|---|---|
| 央行新闻 | [美联储货币政策 RSS](https://www.federalreserve.gov/feeds/press_monetary.xml)、[欧洲央行 RSS](https://www.ecb.europa.eu/rss/press.html)，按公开发布时间与首次采集时间归档 |
| 国际新闻 | [联合国和平与安全新闻](https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml)，归档标题和链接；关键词规则识别冲突等风险 |
| FOMC 日历 | [美联储官方会议日历](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)，按纽约时区处理夏令时；决议时间采用常规 14:00 ET，报告标明该时间约定 |
| CPI、非农、PPI、GDP、PCE 等 | 优先 BLS ICS；本次 BLS 返回 403，自动改用[纽约联储官方经济日历](https://www.newyorkfed.org/research/calendars/nationalecon_cal)的当前月和下月实际日期时间 |
| 利率预期 | Yahoo 的明确月份联邦基金期货合约 + FRED DFF 实际有效利率；记录报价时间、合约月份和推导方法 |

首次实际接入取得 60 条新闻和 34 条日历版本记录，全部六类必要来源通过健康检查。首轮前瞻模拟成功，`places_orders=false`；这只是数据与运行验证，不是盈利验证。完整来源状态、已归档记录及预期保存在 `data/snapshots/public_context_latest.json`，模拟报告为 `data/paper_trading/btc_multifactor_20260927_report.json`。

重要日历事件前 30 分钟至后 90 分钟阻止新开仓。新闻只在原发布时间起的 6 小时窗口内降低风险，并且不能早于实际首次采集时间起作用；旧新闻不会因重复下载而延长风险窗口。新闻规则不会生成交易方向，不能等同于完整语义理解，也不代表覆盖所有全球新闻。来源故障或整个上下文超过一小时未更新会阻止新入场。退出不受影响。

日历改期或删除会保存版本，并仅从观察到变动的时间起使旧安排失效；不会改写以前交易时能看到的安排。每次来源健康检查同样有时间戳，当前的故障不会反向影响过去的健康窗口。

期货隐含利率使用 `100 - 合约价格` 得到该合约月份的平均 EFFR。优先用下次会议之后的完整无会议月份作为利率参照；若随后月份也有会议，使用仅包含下一次会议的合约按会议前后日数拆分，日末会议或同月多次会议等不能可靠隔离的情形报告缺失。计算思路参考 [CME 的方法说明](https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html)，当前实现不是 CME FedWatch 概率。供应商报价有延迟，推导假设没有临时政策调整及 EFFR 与政策目标之间的价差变化。

美联储因素现加入隐含预期变化。议息后，在收到实际政策目标值且此前已保存会前预期时，计算“实际目标变化 − 会前预期 EFFR 变化”，再用于新入场评分。**此前未归档的议息预期差不回填，当前暂无这一前瞻样本。** CPI/非农的市场一致预期数值未取得可靠来源，因此尚不计算其“实际值减一致预期”；这不影响已接入的公布时间风控。

```bash
python scripts/public_context.py
python scripts/run_multifactor_shadow.py --once
# 持续前瞻模拟；运行期间自动刷新数据
python scripts/run_multifactor_shadow.py
```

已执行一次数据采集和前瞻模拟验证，没有安装常驻服务。新版本没有足够的历史新闻/会前预期归档，验证程序会拒绝把今天抓到的数据用于此前的历史决策。

---

以下是 9 月 26 日初版研究记录。

目标是兼顾扣除费用后的收益和回撤。当前状态为 **研究未通过**，仅能作为独立模拟候选。
原 9 月 17 日冻结价格引擎、原终端默认策略未切换，也没有启动常驻交易进程。

## 实际接入

| 因素组 | 数据与用途 | 初始研究权重 |
|---|---|---:|
| BTC 价格 | 24 小时、7 天动量；极端单日波动降低风险 | 20% |
| 衍生品持仓 | 全市场多空账户比、头部交易者多空持仓比、主动买卖量比、BTC 数量口径未平仓量变化、已结算资金费率 | 20% |
| 美联储 | 政策利率上限的 30 天变化、资产负债表的 28 天变化 | 15% |
| 美债 | 2 年、10 年名义收益率及 10 年实际收益率的 7 天变化，同时记录曲线变化 | 15% |
| 汇率 | 广义美元指数及 EUR、GBP、JPY、CNY、CHF 对美元汇率；统一报价方向后合并计权 | 15% |
| 全球风险 | 标普、纳斯达克动量；VIX、油价冲击与可用时的黄金防御性信号控制双向风险 | 15% |

这些权重、归一化区间和经济解释都是待验证假设，不代表已证实的因果关系。多空比是账户/持仓结构，不能解释为资金净流入或“多数看多就会上涨”。收益率使用百分点变化，支持负实际利率。

原策略继续产生多空方向和价格止损/退出。新层在每次原始入场时计算：方向支持分数、与当前交易方向的一致度、市场压力、数据覆盖率。宏观利空对多单和空单的影响不同；极端市场压力降低两边风险。新层只减仓或阻止入场，不增加杠杆，不生成方向，不修改原价格引擎，也不重设止损。

新候选要求六组数据全部有效。缺失、过期、方向冲突或压力过高时阻止新入场，并在报告里列明原因。黄金是全球风险组的可选附加指标，其缺失仍会显示在数据诊断中。

## 国际事件的边界

已有 `event_risk.py` 接口继续接收带 `published_at_utc`、生效窗口和严重程度的事件，可表达 FOMC、CPI、非农、战争、制裁等事件风险。事件在发布前不能影响决策，只能降低仓位或禁止入场。

**9 月 26 日初版没有自动新闻源或经济日历，事件模板为空，报告显示 `not_configured`；9 月 27 日已补接，见上文。** 初版的股指、VIX、汇率、油价只能反映部分国际环境，不能声称覆盖了全部国际新闻；也没有将语言模型对新闻的解读当作历史交易信号。

## 数据与时间约束

采集器使用 FRED、Binance 公共接口和 Yahoo 黄金期货行情，不需要账户密钥。每行保存观察时间、估计发布时间、数值、首次采集时间。前瞻决策只使用实际已采集的数据，已归档观察值不会被后续修订覆盖。每小时刷新一次；不同频率数据分别检查过期，不把缺失值填成中性值。

美债/Fed/FRED 最新历史并不是完整的历史版本数据库。回测使用保守发布延迟的 `reconstructed` 模式，明确标记为回溯研究；前瞻模拟强制 `first_seen`。H.10 汇率有周发布滞后，不能按观察当天提前使用。新数据提供者失败会被写入快照；已有数据只能在有效期内使用。

Binance 多空比和相关衍生品接口通常仅有最近约 30 天，需要持续归档；不能凭这段数据验证牛熊周期稳定性。

原始来源与接口说明：

- [Binance USD-M 市场数据接口](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)
- [美联储 H.10 汇率发布规则](https://www.federalreserve.gov/releases/h10/)
- [美联储 H.15 利率数据](https://www.federalreserve.gov/releases/h15/)
- [FRED 政策利率上限 DFEDTARU](https://fred.stlouisfed.org/series/DFEDTARU)
- [FRED 实际收益率 DFII10](https://fred.stlouisfed.org/series/DFII10)
- [FRED 资产负债表 WALCL](https://fred.stlouisfed.org/series/WALCL)

## 本次实际验证

行情预热从 2026-07-10 开始；评估窗口为 2026-08-30 00:00 UTC 至 2026-09-26 11:00 UTC。

| 配置 | 正常成本净收益 | 最大回撤 | 双倍成本净收益 | 双倍成本最大回撤 |
|---|---:|---:|---:|---:|
| 原价格策略，无宏观过滤 | -1.4589% | 3.7877% | -1.7032% | 3.8567% |
| 原四因素宏观策略 | -1.1510% | 2.4525% | -1.3206% | 2.5301% |
| 新六组因素候选 | -1.2234% | 2.5041% | -1.3945% | 2.6555% |

新候选没有超过原四因素策略，也没有盈利，验证结论是不通过。只有 4 条交易记录（报告区分真实退出与期末估值记录），样本过少，不能支持参数优化或推广。各因素组在这些入场点的覆盖率为 100%，这不等于新闻覆盖完整。

报告同时包含逐组去除因素的对照、双倍手续费/滑点/冲击成本、逐笔因素解释、数据和代码哈希。所有组合使用相同价格引擎、事件文件及修复后的回撤控制。没有根据这些结果再调权重。

回测仍沿用项目的交易记录缩放模型：过滤和缩放已生成的交易，未重新模拟被拒绝交易后的机会路径、资本复利或持续调整退出。不能把此报告等同于完整交易所撮合回测或独立样本外表现。未接入真实订单。

同次修复：15% 回撤停止现在锁定，之后即使净值恢复也不会自动重新开仓；这仍是禁止新入场，不是保证损失上限或触线强平。

## 复现与运行

环境：Python 3.10+，安装 `requirements.txt`；测试另需 `pytest`。

```bash
python scripts/download_multifactor_snapshot.py
python scripts/run_multifactor_shadow.py --profile config/multifactor_candidate_20260926.json --once
```

持续前瞻模拟使用 `python scripts/run_multifactor_shadow.py`。其独立状态和报告保存在 `data/paper_trading/btc_multifactor_20260926_*`，不发送交易所订单。更改候选权重、因素代码或组合风控代码后，已有状态的配置校验会拒绝混用，应显式创建新候选代际。

复现本次研究（生成的快照保存在本地，git 默认忽略）：

```bash
python scripts/download_market_snapshot.py \
  --start-utc 2026-07-10T00:00:00Z --end-utc 2026-09-26T11:00:00Z \
  --intervals 5m,1h --output data/snapshots/btc_multifactor_research_20260926.json.gz
python scripts/download_macro_snapshot.py \
  --start-utc 2026-04-01T00:00:00Z --end-utc 2026-09-26T11:00:00Z \
  --output data/snapshots/macro_comparison_20260926.json.gz
python scripts/validate_multifactor.py \
  --market-snapshot data/snapshots/btc_multifactor_research_20260926.json.gz \
  --legacy-macro-snapshot data/snapshots/macro_comparison_20260926.json.gz \
  --start-utc 2026-08-30T00:00:00Z
python -m pytest -q
```

原始快照必须保留才能精确复现：日后重新下载可能修订历史，也无法重新取得过期的多空比数据。候选配置是 `config/multifactor_candidate_20260926.json`；完整报告是 `data/validation/multifactor_20260926.json`。程序不会因为历史指标改善就自动升级实盘。
