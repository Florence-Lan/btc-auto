# 策略评估：2026 年 10 月 23 日凌晨（北京时间）

**判断：需要更新风险信息覆盖、数据时效和执行监测；目前不支持直接修改主策略的均线、四小时确认、止损宽度或风险预算。** 市场表现更偏风险规避，能源冲击、欧洲财政风险和科技融资担忧相互叠加；BTC 又存在 ETF 资金回流，不能把所有信息压成一个永久做空结论。

本轮目标是维持现有 8%／15% 账户软硬回撤控制、股票每笔 0.25% 计划风险预算，以及对新增规则按净收益、成本和回撤进行独立比较。这些是配置和评审目标，不是损失保证。此次只检查、归档和评估，没有修改运行配置、账本、服务或交易模式。

本地证据冻结于 **10 月 23 日 00:14:33 北京时间**；公开行情采样主要在 00:12—00:17，对应美东 **10 月 22 日中午**，美股当天尚未收盘。官方日度利率、盘中债券报价、延迟期货和实时股票样本分开标注。

## 新账户证据

四主账户仍全部空仓，但昨日早上的 4,026.894 USDT 已发生变化。

| 账户 | 权益 USDT | 累计净收益 USDT | 完整平仓：胜／负 |
| --- | ---: | ---: | ---: |
| BTC | 997.781577 | -2.218423 | 0／1 |
| MU | 1,004.764878 | +4.764878 | 2／3 |
| SNDK | 1,010.994641 | +10.994641 | 3／4 |
| SKHYNIX | 1,008.166475 | +8.166475 | 3／1 |
| 合计 | 4,021.707571 | +21.707571 | 8／9 |

总本金 4,000 USDT，累计收益 **+0.542689%**。38 条成交记录归并为 **17 笔完整持仓**，胜率 **47.06%**。毛利 22.116020 − 手续费 1.073479 + 资金费净收入 0.665030 = 净收益 21.707571 USDT；滑点已含在成交价。四账户按原始成交、资金费逐事件重放平账，32 份冻结文件哈希在审计前后通过，另经独立 Decimal 复核。

新增加的两笔均为空单：MU 于 10 月 22 日 16:00:51 入场、21:54:45 保护止损，净 **-2.490226 USDT**；SNDK 于 16:05:18 入场、22:02:04 完整退出，净 **-2.696201 USDT**，其两个退出成交归并为一笔。它们说明宏观偏空并不能保证空单获利；还需分析入场时段、反弹路径及执行。不能仅凭随后价格更低就拉宽止损或补记本来没有的盈利。

10 月 17 日 12:29 后新开且完整退出的自动交易共三笔，均亏，合计 **-7.092447 USDT**。累计样本仍混有旧规则和两笔人工退出；三笔近期交易不足以稳定估计当前版本收益。账本采样最大回撤最高约 1.003%，不代表连续盘中回撤上限或未来风险。

主 BTC 账户在本次冻结点已恢复 healthy，风控有新鲜交易所时间及 mark。昨天早上的 HTTP 418 数据中断不能继续描述为“当前仍完全中断”；独立实验仍有局部 degraded，不能由主账户恢复推断所有来源全天连续可用。

原始证据仅保存在本地 `data/research/strategy_assessment_20261023_161433Z/`，其中 `capture_manifest.json` 为冻结清单、`audit_metrics.json` 为账本复算、`assessment_summary.json` 为因子日期摘要。这些文件按 `.gitignore` 排除，未随仓库发布；远程仓库保留本文汇总，不能仅从仓库复算原始账本。目录中的日期使用北京日期，后缀 161433Z 对应 **10 月 22 日 16:14:33 UTC**，不是 10 月 23 日 16:14 UTC。冻结为逐文件稳定读取，不是跨文件原子检查点。

## 世界局势、美国与欧洲利率

IMO 截至 **10 月 22 日**的官方清单确认中东及霍尔木兹地区持续发生船舶事故和袭击。AP 当天报道伊朗声称一夜七艘油轮受损，但该数量未经独立证实。可据以提高航运和能源冲击监测，不能把每个战事情节都当成已确认事实。[IMO 已确认清单](https://www.imo.org/en/mediacentre/hottopics/pages/middle-east-highlighted-incidents.aspx)、[AP 报道](https://apnews.com/article/iran-united-states-war-hormuz-trump-israel-2abebce2fca2c085e0e0ee717e42cf28)。

| 指标 | 本次可核验值 | 观察日期与口径 |
| --- | ---: | --- |
| 美国 2 年／10 年国债 | 4.81%／5.13% | 10 月 21 日，Treasury 日度恒定期限曲线 |
| 美国 10 年实际利率 | 2.76% | 同日 TIPS 实际曲线 |
| 德国 2 年／10 年 | 3.08%／3.38% | Reuters，10 月 22 日盘中基准券 |
| 法国 10 年 | 5.11% | 同日报道；日内高 5.1415% |
| 法德／意德 10 年利差 | 173.58／123 bp | 同日报道；意德当日扩大约 10.5 bp |
| 英国 10 年 | 5.37% | Trading Economics，10 月 21 日 OTC；未取得同日官方曲线 |

美国来源：[名义曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2026)、[实际曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_real_yield_curve&field_tdr_date_value=2026)。欧洲来源：[Reuters／MarketScreener](https://www.marketscreener.com/news/french-debt-risk-premium-reaches-euro-zone-sovereign-crisis-levels-ce785cded98ff225)。英国辅助来源：[Trading Economics](https://tradingeconomics.com/united-kingdom/government-bond-yield)。以上覆盖主要欧洲市场，不能代表欧洲每个国家，也不是同一时点、同一币种的可直接排名。

Fed 9 月 16 日加息 25bp 至 **3.75%—4.00%**；ECB 9 月 10 日决定加息 25bp，存款便利利率随后为 **2.50%**；BoE 9 月 17 日维持 **3.75%**，三名委员主张加息。[Fed 决议](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm)、[ECB 决议](https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp260910~314e508016.en.html)、[BoE 决议](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/september-2026)。

**推断：**高实际利率对风险资产估值构成约束；法德、意德利差扩大提示欧洲财政风险分化。当天德债收益率下降同时存在，不能写成“所有欧洲债券收益率都上涨”，也不能将避险买债造成的利率下降机械解释为科技股或 BTC 的宽松利好。能源、债券和科技股可能反映同一冲击，新增模型要避免重复计权。

## 日本利率与日元

BoJ 9 月 18 日决定隔夜目标约 **1.25%**，9 月 24 日生效。MOF 官方 CSV 本次最新为 **10 月 21 日**：2 年 **1.914%**、10 年 **2.923%**、30 年 **3.980%**；相较 10 月 1 日分别下降 2.5、16.9、14.2bp。日债绝对收益率较高，但不能描述为“本月正在急升”。[BoJ 决议](https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260918a.pdf)、[MOF 原始曲线](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv)。

BoJ 官方外汇记录：**10 月 22 日 17:00 JST，即北京 16:00**，USD/JPY **157.22—157.24**；前一天同时间为 158.12—158.14，按中值日元升值约 0.57%。[10 月 22 日](https://www.boj.or.jp/en/statistics/market/forex/fxdaily/fxlist/fx261022.pdf)、[10 月 21 日](https://www.boj.or.jp/en/statistics/market/forex/fxdaily/fxlist/fx261021.pdf)。

**推断：**日元融资成本和升值可放大套息组合风险，值得纳入风险监控；上述证据不能证明大规模套息平仓已发生，更不能证明它就是当前存储股跌势的主因。日本全国 9 月 CPI 安排在**今天 07:30 北京时间**，本轮读取时尚未公布，不填入预测或提前写实际值。[官方发布日历](https://www.e-stat.go.jp/release-calendar/statistics/00200573)。

## 美股、纳指期货与科技板块

| 市场 | 样本 | 时间 |
| --- | --- | --- |
| Nasdaq Composite | 26,163.70，-2.02% | 美东 10 月 22 日 12:13，即北京 10 月 23 日 00:13 |
| MU 现货 | 870.26 美元，-4.75% | 美东 12:14 |
| SNDK 现货 | 1,283.28 美元，-6.89% | 美东 12:14 |
| SKHY 美国 ADS | 134.83 美元，-5.88% | 美东 12:15 |
| NQZ6，2026 年 12 月合约 | 29,280.00，-2.42%，前结算 30,007.00 | 页面标 10 月 22 日 10:17 CDT，即北京 23:17；延时报价 |

来源：[Nasdaq 指数](https://api.nasdaq.com/api/quote/COMP/info?assetclass=index)、[MU](https://api.nasdaq.com/api/quote/MU/info?assetclass=stocks)、[SNDK](https://api.nasdaq.com/api/quote/SNDK/info?assetclass=stocks)、[SKHY ADS](https://api.nasdaq.com/api/quote/SKHY/info?assetclass=stocks)、[Webull NQZ6](https://www.webull.com/quote/cme-NQZ6)。NQ 是较早的带时区样本，不是报告完成时的可执行价格；其日变动分母为前结算，与现货指数口径不同。

按 Nasdaq 历史收盘数据，从 **10 月 1 日至 10 月 21 日**，MU、SNDK、SKHY ADS 分别约 **-16.74%、-22.90%、-25.97%**。这支持近期科技／存储板块压力仍在的判断，但不能把一次反弹或一次下跌当作永久方向。美国 ADS、韩国原股与 Aster 永续不能直接比较绝对价格。[MU 历史](https://api.nasdaq.com/api/quote/MU/historical?assetclass=stocks&fromdate=2026-10-01&todate=2026-10-21&limit=30)、[SNDK 历史](https://api.nasdaq.com/api/quote/SNDK/historical?assetclass=stocks&fromdate=2026-10-01&todate=2026-10-21&limit=30)、[SKHY 历史](https://api.nasdaq.com/api/quote/SKHY/historical?assetclass=stocks&fromdate=2026-10-01&todate=2026-10-21&limit=30)。

三主股票账户在冻结点的完整四小时确认已转为 **short**，不是昨天早上“仍全部未确认”的状态。MU 冷却中，SNDK 和 SKHYNIX 当轮无新信号，均空仓。主系统已有双向趋势响应，无需仅凭新闻另加一个固定做空指令。

## BTC 链上、ETF 与杠杆

Farside 完整到美国交易日 **10 月 21 日**：10/19、20、21 净流入分别 **5.267 亿、0.232 亿、5.248 亿美元**，三日合计 **10.747 亿美元**；此前 10/12—16 合计净流出 2.284 亿。10/22 各基金仍为“-”，总计零是未完成占位，不能当作零流入。[Farside 原表](https://farside.co.uk/btc/)。

Binance BTCUSDT OI 从北京 10/22 00:00 的 **98,971.071 BTC** 降至 10/23 00:00 的 **95,561.006 BTC**，下降 **3.45%**；美元名义值下降约 6.70%，其中包含币价下跌，不能全当作平仓。最近三档结算资金费为 **+0.000507%、-0.001401%、+0.001265%／8 小时**，没有持续极端正费率。00:14 左右币价约 81,534 USDT，滚动 24 小时约 -3.57%。[OI](https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=25)、[结算资金费](https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=5)、[24 小时行情](https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT)。这些是单一交易所衍生品数据，不等于链上流量。

Glassnode **10 月 21 日**研究描述获利了结较九月高点减弱，并给出实现成本等背景指标；报告不是本轮实时链上数列，其中 6—12 个月币龄成本不能冒称短期持有者成本。其 ETF 单日数字与 Farside 有差异，本报告统一用 Farside。未取得 10/22 可直接核验的交易所净流入、储备、最新 STH 成本及 MVRV 原始数列，不据此宣称鲸鱼吸筹、储备下降或确定底部。[Glassnode 原始研究](https://research.glassnode.com/the-week-onchain-week-42-2026/)、[CryptoQuant 储备定义页](https://cryptoquant.com/asset/btc/chart/exchange-flows/exchange-reserve?exchange=all_exchange&window=DAY)。

**推断：**ETF 回流是支持因素，价格下降与 BTC 计价 OI 收缩反映短线压力，两者可以同时存在。不能只因 ETF 流入开多，也不能把 OI 下降自动等同爆仓，更不能把衍生品数据冒称已覆盖全部链上信息。

## 策略实际覆盖及更新决定

本地冻结因子显示：美债、股指、VIX 最新观察为 **10/20**；美元及 USD/JPY 等外汇为 **10/09**；油价已经更新到 **10/19**，不能沿用前次“仍停在 10/14、已省略”的描述。源状态 healthy 仅反映抓取成功，不代表盘中时效。当前保守可用时间模型对日频股指／美债估算两天、外汇十天，再与真实 first-seen 取较晚值；更及时来源必须作为新版本验证，不能回填历史决策。

BTC 六组因子没有直接欧债收益率／利差、日本国债或盘中 NQ 输入；NASDAQCOM 日度数据不能替代 NQ 期货。ETF、交易所流量等补充资料标为 `observation_only_no_directional_weights`；本次补充 ETF 来源还报 403。股票主规则仍是价格信号、完整四小时确认、执行风控和公司财报门禁；独立股票宏观实验增加的是事件窗口，不是宏观方向模型。LLM 审核也不能自行联网补齐这些缺口。

| 工作 | 决定 | 理由与边界 |
| --- | --- | --- |
| 事件维护、日期及来源健康展示 | 需要更新维护 | 核查日本 CPI／BoJ、ECB 等覆盖；维护已核实财报通知和首发不确定性，保留真实 first-seen |
| 更及时能源、NQ、日元及欧债风险输入 | 先建立独立候选 | 分别研究变化和共同冲击，避免重复计权；先验证持续取得、延迟、缺失与跨时区处理 |
| 三只存储股的行业共振风险 | 需要组合层研究 | 三账户独立本金不等于三个独立风险；先评估联合止损预算／同向暴露候选，未选具体新阈值 |
| BTC EMA48/240、0.4% 状态转换及结构回踩 | 暂维持 | 当前空仓，旧 origin 不复活；单笔人工退出不足以证明该方向规则失效 |
| 股票 Donchian20 和完整四小时确认 | 暂维持 | 当前已能确认空头；旧 SNDK 删除确认候选仍落后原规则 |
| 止损、追踪、风险预算和 10x 上限 | 暂维持 | 近期止损需要归因，不能以随后方向正确为由扩大亏损容忍或仓位 |

旧 SNDK 删除四小时过滤候选，正常／双倍成本累计净收益仍分别落后基准 **7.045006／6.979105 USDT**。新 BTC 四臂仍零成交；股票宏观对照的 MU、SNDK 每个成本及规则臂各完成一笔亏损交易，候选与基准收益差仍为零，SKHYNIX 零成交。不能继续写所有新实验均零成交，也没有宏观候选改善收益的证据。按既定至少十四天、每臂十笔完整平仓达到首轮评审规模，仍须分别核查成本、回撤、未平仓风险和去掉最大赢家后的表现；不自动晋级。

## 近期事件与后续评审

| 事件 | 北京时间或官方日期 | 处理含义 |
| --- | --- | --- |
| 日本全国 9 月 CPI | 10/23 07:30 | 采样时未公布；核查是否进入现有监控，新增交易规则另行验证 |
| SK hynix Q3 电话会 | 10/28 08:00 | 官方明确新闻稿更早，精确首发时间未知；现有财报门禁从 10/27 08:00 起 |
| FOMC | 美国 10/27—28 会议 | 原 BTC 及股票宏观实验继续按已声明事件窗口；主股票未自动升级该实验 |
| 美国 GDP 初值、个人收入与支出 | 10/29 20:30 | 官方同日发布；继续复核已有事件窗口 |
| ECB 政策会议 | 欧洲 10/28—29 | 核查直接事件覆盖，不能用新闻抓取代替完整政策日历 |
| SNDK 电话会 | 10/30 04:30 | 现有公司门禁从 10/29 04:30 起；最早 07:00 且满足完整事件后小时线再判断 |
| 日本东京 CPI／BoJ | CPI 10/30 07:30；BoJ 10/29—30会议 | BoJ 决议无固定时刻，不填虚构精确时间 |

来源：[日本 CPI](https://www.e-stat.go.jp/release-calendar/statistics/00200573)、[SK hynix 公司邀请](https://news.skhynix.com/en/q3-2026-earnings-conference-call-invitation/)、[Fed 日历](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)、[BEA 日历](https://www.bea.gov/news/schedule)、[ECB 日历](https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html)、[SNDK 公司通知](https://investor.sandisk.com/news-releases/news-release-details/sandisk-report-first-quarter-fiscal-year-2027-results-october-29)、[BoJ 日历](https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm)。公司电话会锚点不是首次业绩发布时间或实际结束时间。财报日历现覆盖至北京 **11 月 1 日 00:00**，需要在到期前例行维护；MU 下一次财报日期本轮仍未核实，不视作没有事件。

本轮没有实施上述新候选，也没有声称市场可穷尽观察或策略已经覆盖所有全球信息。完成的验证是新证据冻结、哈希、账本重放、独立会计复核、配置与来源日期核查及联网资料交叉检查；没有代码变更，因此没有重跑实现回归。
