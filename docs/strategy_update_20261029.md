# 2026-10-29：宏观数据维护与独立入场时机对照

本轮已部署欧债日报解析兼容与 Fed 发布会日历修复，并启动新的 **18 个独立模拟账户**验证更早入场的两项固定假设。**主四账户的技术入场门槛没有放宽，风险预算没有扩大；新实验尚无盈利或优于基线的证明。**

下文时间均为北京时间，另标来源时区。宏观采用 08:24 信息集、08:26–08:30 再核结果；主账户截面为 08:48:01。01:07 的盘中研究保留原截止点，不用事后政策结果回填。

## 已核实的宏观与市场事实

美联储于今日 **02:00** 维持联邦基金目标区间 **3.75%—4.00%**，表决 10:2，两名反对者主张加息25bp。IORB维持3.90%，不是降息。02:30发布会已举行；本轮读取官方页面及文件，没有取得可读完整逐字稿，问答摘要使用路透二级佐证。[Fed声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm)、[实施说明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a1.htm)、[发布会页](https://www.federalreserve.gov/monetarypolicy/fomcpresconf20261028.htm)

| 指标 | 最新已核实观察 | 时间与口径 |
|---|---|---|
| 美国2年／10年／10年实际利率 | 4.67%／5.05%／2.67% | 财政部10/28日数据；较前日−3／+6／+3bp，精确发布时间未知 |
| 德／法／意／西10年国债 | 3.37%／4.93%／4.52%／3.99% | Radiocor 10/28 15:48 GMT，即23:48；通用基准，非具体可执行债券报价 |
| 法德／意德／西德利差 | 156／115／62bp | 同一日报；相对10/27同源15:46 GMT截面−4／0／−2bp |
| 英国10年 | 约5.32%，当日−2bp | Reuters文章更新10/28 23:22:34，未给精确报价采样分钟 |
| 日本2／10／30年国债 | 1.888%／2.829%／3.895% | MOF今晨08:27重新下载后，最新仍10/27，尚无10/28行 |
| 日本政策利率 | 隔夜拆借目标约1.25% | 9/18决议、9/24生效；不是尚未公布的10月会议结果 |
| USDJPY | 158.040日元／美元 | Yahoo外汇08:27:47观察、08:27:52.946接收；较其前收157.762上涨0.1762% |

美债短端下行而长端、实际利率上升，不能概括为全面宽松；欧债风险溢价仍高，但最新可比截面并未显示法国风险加速恶化。英国与德国币种不同，二者收益率差不能直接叫纯信用利差。本轮没有取得当前CDS、iTraxx或公司债IG/HY利差。[美债名义曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=202610)、[实际曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_real_yield_curve&field_tdr_date_value=202610)、[欧债原文](https://www.borsaitaliana.it/borsa/notizie/radiocor/finance/dettaglio/euro-zone-benchmark-10year-sovereign-bond-yields-nRC_28102026_1652_576143484.html?lang=en)、[英国Reuters](https://live.euronext.com/en/financial-news/british-markets-take-healeys-first-budget-their-stride)

MOF 10/27日债相对10/26变化为0／+0.5／+1.1bp，相对10/22为−2.6／−8.5／−5.7bp；没有把旧日频数据改成今早实时值。日元本次相对提供商前收略弱，未取得套息持仓或强平规模，不能宣称已经出现日元升值驱动的套息崩盘。[MOF CSV](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv)、[BoJ政策决议](https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260918a.pdf)、[Yahoo外汇接口](https://query1.finance.yahoo.com/v8/finance/chart/JPY=X?interval=1m&range=1d)

能源采用最新核验的结算报道：Brent 115.32美元／桶，涨1.44美元；WTI 102.78，涨2.88美元。该Reuters版本于今日03:04:10发布，正文未标交割月，不能与机器人具名Brent代理合约拼接计算涨跌，也不是08:24即时价。沙特炼厂袭击仍属胡塞声明，报道未获沙特即时确认；未核实到签署停火协议或确定停产量。[Reuters/Euronext](https://live.euronext.com/en/financial-news/oil-prices-rise-yemen-fighting-saudi-refinery-attack)

美国10/28常规盘已结束：S&P500 7743.98、纳指综合26549.85，均约涨0.3%。MU收983.23美元、+5.86%，SNDK收1404.77、+7.66%；随后同页盘后快照分别958.96、1363.89，较常规收盘−2.47%、−2.91%。这些盘后数值没有独立last-trade时戳，不能称统一盘后收盘或08:30实时成交价。[AP收盘](https://apnews.com/article/wall-street-stocks-dow-nasdaq-7864eac2e13db75f07758ed9c39ad96b)、[MU](https://www.google.com/finance/quote/MU:NASDAQ)、[SNDK](https://www.google.com/finance/quote/SNDK:NASDAQ)

Microsoft与Alphabet财报均已发布。Microsoft营收923亿美元、同比+19%，Azure增长48%；Alphabet营收1275.77亿美元、+25%，GAAP EPS2.12，Cloud约307亿美元、+103%。Alphabet二级盘后快照约跌10%，与Cloud经营增长可以同时成立，不能据单一股价断言AI需求崩溃。Alphabet使用可读的发行人SEC文件索引摘录，本轮直接打开原文件不可用，证据路径已记录。[Microsoft IR](https://www.microsoft.com/en-us/Investor/earnings/FY-2027-Q1/press-release-webcast)、[Alphabet SEC原文件](https://www.sec.gov/Archives/edgar/data/1652044/000165204426000080/googexhibit991q32026.htm)

具名纳指期货仍须与现金指数区分：08:35 observer保存Yahoo `NQZ26.CME`（2026年12月）29725.50点，实际观察08:25:42、首次接收08:35:44.521，约延迟10分钟。研究页Webull的同月NQZ6另报29625.25、相对其前结算−1.18%，页面标10/28 18:42 CDT，确切last-trade未独立确认；不同时间／提供商不能合并成一次涨跌。CME官方页未取得可抽取实时数值。[具名合约页面](https://www.webull.com/quote/cme-NQZ6)、[CME](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.quotes.html)

## BTC与真正链上的边界

Coinbase原始现货ticker为 **81138.80美元**，成交时间今日08:28:34.936，接收08:28:35.892；它不能替代模拟合约的book或mark执行价。[原始ticker](https://api.exchange.coinbase.com/products/BTC-USD/ticker)

Farside最新完整ETF交易日仍为 **10/27，净流入1.760亿美元**。10/28只见BITB +130万美元，IBIT、FBTC、HODL缺失；页面合计130万美元是已报告部分，不能当完整日净流入，也不能将缺失写成零。[Farside](https://farside.co.uk/btc/)

CoinMetrics公共原始exchange-series今晨重取成功，但最新仍10/27：净流入−17262.74652715 BTC、交易所供应2678007.25573681 BTC，状态均为 **flash／未复核**。新抓取时刻不更新观测日期；不据此断言鲸鱼吸筹、抛售意图或底部。MVRV此前403，本轮未换路重取；当前原始SOPR、全面清算和实体级资金流仍缺失。[CoinMetrics原始接口](https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=FlowInExNtv%2CFlowOutExNtv%2CSplyExNtv&frequency=1d&start_time=2026-10-26&end_time=2026-10-29&page_size=100)

## 接下来的官方事件

| 事件 | 北京时间及限制 |
|---|---|
| 美国GDP初值／个人收入与PCE | 今晚10/29 20:30；原公共日历来源为纽约联储，尚未填未来实际值 |
| ECB决议／发布会 | 今晚10/29 21:15／21:45；当地14:15／14:45 CET，欧洲已结束夏令时 |
| BoJ两日会议 | 日本当地10/29–30；10/30决议与展望基本观点，官方发布分钟仍“未定”，不填中午 |
| 东京区部10月CPI初值 | 10/30 07:30，即08:30 JST；不是全国CPI，也未填未来实际值 |
| BoJ植田发布会 | 10/30 14:30，即15:30 JST；文字记录计划11/2，不能混作会见起点 |
| SNDK财报电话会 | 10/30 04:30，即10/29 13:30 PDT；首次业绩发布分钟和会见结束仍未知 |

来源：[纽约联储经济日历](https://www.newyorkfed.org/research/calendars/i-oct26.html)、[ECB本周日程](https://www.ecb.europa.eu/press/calendars/weekly/html/index.en.html)、[BoJ实时日程](https://www.boj.or.jp/about/calendar/index.htm)、[统计局日表](https://www.stat.go.jp/english/data/cpi/1582.htm)、[e-Stat明确08:30计划](https://www.e-stat.go.jp/release-calendar/statistics/00200573)、[Sandisk官方公告](https://investor.sandisk.com/node/8381/pdf)。国际日历只覆盖其声明范围，自动复核不会无依据延长覆盖；MU动态IR未取得新公告的成功鲜度确认，不等于证明没有新事件。

## 主账户与入场迟滞的证据

08:48:01主四账户总权益 **4019.361218 USDT**，相对4000初始本金+19.361218，约+0.4840%。这是含浮动估值的模拟截面，独立实验本金和收益没有加进来。

| 主账户 | 权益USDT | 模拟合约持仓数量 | 累计fills |
|---|---:|---:|---:|
| BTC | 997.781577 | 0 | 2 |
| MU | 1003.155199 | 多0.12 | 14 |
| SNDK | 1008.644585 | 0 | 20 |
| SKHYNIX | 1009.779857 | 0 | 11 |

47笔fills是执行记录数，不等于47笔完整平仓。SNDK当时有原4h排序和财报禁入原因；MU仍持仓。该动态截面来自[主status](../data/parallel_simulation/btc_memory_stocks_latest_1000_each_20261007/status.json)，不能当作未来权益或新规则收益。

MU 10/28 12:00多单的主要迟滞是**技术确认**。10/27 02:00起已有原始多候选，至10/28 12:00前共9个被实际4h日志拒绝；09／10／11点没有原始候选，不能把所有空仓小时都算作4h损失。12:00首次同时通过1h Donchian20与原4h确认，最早journal留痕12:00:13.173，12:00:54.078买入0.12，价格966.053172。约41秒执行等待中，最初所见前5分钟成交量为0，随后重取为1.89；事件检查clear。

从10/27 02:00候选收盘916.66到实际成交，34小时后价格高约5.39%，说明参与滞后确有日志依据。但不能假定当时可在916.66成交、持有期间不止损、最后兑现全部涨幅。原主日志没有独立first_seen字段，13.173秒只能叫最早留痕，不能补造成精确接收时刻。[因果时序复核](../data/research/entry_latency_audit_20261028/mu.md)

旧early recovery分支也没有覆盖这次08／09／10点的趋势延续：原输入下，08点此前EMA8仍高于EMA24，6h回撤不足1ATR；09／10点又不是首次收复。不能由它之前一笔short亏损反推规则无用，也不能称它已经解决所有晚入场问题。[固定三时点复核](../data/research/entry_latency_audit_20261028/early_coverage_gap.md)

## 本轮实际修复

**欧债parser：**真实10/28日报把西班牙一行拆成“from the.”和独立“previous close.”两段。修复只允许唯一report容器内这一精确单次格式，随后继续严格校验标题、国家、单位、日期、GMT、未来观察与receipt；记录归一provenance，保留原body SHA。归档上限仍2页，没有增加网络预算或绕过403／429，也未使用全页面正则捞数字。[修复与139项定向测试](../data/research/strategy_assessment_20261029/europe_maintenance/maintenance_summary.md)

停写维护只将欧债源的next_due提前至真实actor到期时间，保留此前14次失败和旧观察时钟。恢复后 **08:35:46.124**正常GET成功，首次可得才建立在这个时刻；10/28 23:48的观察没有倒填为当时已被系统接收。欧债变为daily_current，仍**不具备intraday资格**。[激活记录](../data/research/strategy_assessment_20261029/activation.json)、[恢复后的真实快照](../data/research/strategy_assessment_20261029/observer_after_capture.json)

**Fed发布会：**公共日历此前只有决议事件，缺独立14:30 EDT发布会。现读取官方月历时间，以`fomc_press_conference`独立事件输出，决议利率预期仍只使用policy_decision，避免混入发布会。相同前30／后90分钟政策下，决议窗口到03:30，独立发布会到04:00。此修复没有给未配置宏观门禁的账户自动新增门禁；部署晚于本次Fed窗口，不声称新代码保护了已发生的交易。[官方月历](https://www.federalreserve.gov/newsevents/2026-october.htm?view=calendar)、[原文离线校验](../data/research/strategy_assessment_20261029/fomc_press_parser_verification.json)

## 新18账户如何检验更早入场

3个品种（MU／SNDK／SKHYNIX）×3个方案×正常／双倍成本=18个独立账户，每个初始1000 USDT；18000是独立研究虚拟本金，不计入主4000，也没有向原账户加资。声明为今日01:30:10.915；真实交易所epoch **08:37:06.343**，08:44终验全flat、0fills、financial_integrity全部true。

| 方案 | 固定假设 |
|---|---|
| baseline | 当前1h Donchian20与严格已收4h确认；另加所有18账户共同使用的宏观事件门禁 |
| turning | 保留原Donchian原始信号；接受原严格4h，或已收4h快线位于中／慢线方向侧、收盘位于慢线方向侧、快慢差仍≥价格0.4%、中线相对第4个已收点同向改善 |
| turn_pullback | 使用turning背景；原Donchian有优先权，只有原始信号不存在时才允许固定trend_pullback的新收复，不让被拒Donchian借另一分支绕门禁 |

pullback保留EMA8／24／60排序、中线六栏方向、上一收盘跨EMA8、最小均线间隔和≤1ATR延展。冻结原EMA8及两栏结构加0.25ATR缓冲；原2ATR／各品种最小／最大止损距离必须覆盖结构，信号、可执行报价、实际fill均检查，保留0.5ATR追价和1ATR延展约束。没有复活过期突破，也没有使用未收4h栏。

三方案共同保留原0.25%权益风险、≤1倍总名义敞口、杠杆上限、原止损／目标／追踪／持仓上限、资金费、回撤约束、深度与费用、3栏冷却、1小时有效期、无自动反手及财报门禁。共同宏观门禁为前30／后90分钟，仅阻新风险；时钟或事件来源失败仍允许原保护退出。

**实验baseline在宏观禁入期间可能不同于实际主账户**，不能隐去这一差异。正常与双倍成本分别定仓成交，双倍手续费和不利滑点、共同真实资金费；不同入场路径的最终收益是独立账户结果，不是配对交易的因果利润。[固定设计](../config/stock_entry_timing_trial_20261029.json)、[独立正确性审查](../data/research/strategy_assessment_20261029/entry_trial_independent_audit.json)

只有严格晚于epoch的已收小时边界可产生前瞻信号；预热只算指标。记录实际first_seen、相对bar边界的lag、variant和不可变上下文hash，使用当下book成交。已知MU／SK案例仅是设计背景或诊断，不计作未知前瞻收益。首次严格前瞻小时边界为今日09:00，不能因等待它而补旧交易。

复核至少等待14天且每相关品种／方案10笔完整平仓，检查正常与双倍成本、未平仓预计退出成本、回撤、频繁进出、错失赢家与单笔大赢家依赖；分别比较turning对baseline、pullback对turning的增量。达到样本门槛也不是盈利通过，更不自动推广。

## 部署、保全与运行验收

08:34:50.482完成维护激活，9把锁下保护487个文件、300778781字节。原60个金融账户（主4＋SNDK4＋BTC reentry4＋macro12＋exit24＋early12）的财务、持仓、止损、规则与epoch保留。独审284条历史journal／CSV前缀无差异；observer恢复后新增的决策历史单列，未回填。旧SNDK frozen 41文件、runtime manifest、实验manifest与绑定保持不变，没有复制root新策略进去。[激活证据](../data/research/strategy_assessment_20261029/activation.json)、[独立保全终验](../data/research/strategy_assessment_20261029/independent_after_activation.json)

全量回归 **2636 passed、2 skipped、43 subtests passed，81.54秒**；覆盖基线金融行为等价、双成本、未来／入场前bar拒绝、保护退出优先、临成交事件再检查和账本恒等式。通过测试证明这些代码约束，不证明赚钱。[测试原日志](../data/research/strategy_assessment_20261029/full_tests.txt)

另增生产器身份回归测试，**22 passed**：旧源码pin必须拒绝且不写state／report；只明确迁移批准的单一pin后可只读加载；其它风险、策略、源码与冻结身份变化仍拒绝。[身份回归](../tests/test_signal_producer_profile_identity.py)、[运行结果](../data/research/strategy_assessment_20261029/producer_identity_tests.txt)

**BTC producer已恢复正常评估。**08:44审计确认独立state.profile的22项中，唯一差异为`public_context_sha256`旧pin；当时最后成功报告仍是今日01:25:27.965。维护前后日志均有身份不匹配，不能由最后成功时间推断首个失败的精确秒。08:51:47.545仅迁移这一metadata值，其余state字节不变，另71个主目录文件的流式SHA不变，身份guard未改。原生产器正常生成新report于 **08:52:09.457**，新PID17440、evaluations=1、error=null；目标为08:45已收5分钟栏，可用时刻08:49:59.999，在15分钟时效内。created／原始信号时间保留，observations正常增加；主账户08:53:38已消费新报告并恢复healthy。原始信号仍是10/24 00:00，初次入场与回踩均继续按origin expired拒绝，没有复活旧信号或补过去成交。[只读定位](../data/research/strategy_assessment_20261029/btc_producer_profile_mismatch.json)、[精确迁移证据](../data/research/strategy_assessment_20261029/producer_pin_activation.json)、[迁移前运行终验](../data/research/strategy_assessment_20261029/runtime_after_activation.md)

后续维护的依赖检查面必须同时覆盖：各shadow manifest的源码／规则pin、主manifest与frozen绑定、**producer独立`state.profile`源码身份**、状态内期望profile及实际成功report时效。此次遗漏说明只验shadow manifests不足够；不能旁路身份guard、重置producer历史，或把主风险监控可用误写成信号源已消费最新宏观。

08:54独立终验确认主BTC及BTC reentry均healthy，主BTC资金、费用、资金费、持仓和成交记录较修复前全等，fill journal字节未变。5个任务均启用并运行，20个Python节点构成10棵唯一venv父子树；SNDK38／BTC61／macro42／exit43／early47／observer10／新trial49个源码闭包逐文件均无hash差异，进程路径与启动时间符合部署。本证据核对磁盘指纹及实际进程，不声称读取了Python内存代码；新18仍0fills。[生产器与运行终验](../data/research/strategy_assessment_20261029/producer_pin_postcheck.md)、[完整记录](../data/research/strategy_assessment_20261029/producer_pin_postcheck.json)

此外，原exit实验六个MU候选的`processed_closed_bar_revised`为既有修订告警，本轮没有清零状态或修改其退出规则。新18账户无成交和较少信号不能解释为策略优劣；当前能确认的是原规则存在参与滞后，保护性过滤也可能必要，提前入场能否改善扣费结果仍需前瞻样本回答。
