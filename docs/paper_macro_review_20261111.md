# 2026-11-11 模拟盘与跨市场维护

本轮保留交易策略和风险阈值，处理三个已确认的运行阻断：公司事件覆盖到期、纳指等参考交易时段到期、升级后原审核程序路径不存在。没有新的成交证据支持按近期宏观走势调整方向权重或放宽入场。用户已授权必要修改、推送远程和恢复原模拟盘。

验证目标：原四账户初始本金合计保持 4000 USDT；股票新仓计划风险保持不超过 min(20 USDT, 权益×2%)，执行压力风险预算保持 min(25 USDT, 权益×2.5%)；净目标至少100 USDT，杠杆上限10倍。技术止损、费用、流动性和资金费检查保留。计划预算不能保证跳空后的实际亏损上限。

## 账本与新规则样本

最后股票截面为北京时间11月11日21:39，BTC金融状态为21:31。所有账户空仓。

| 账户 | 权益 USDT | 累计净收益 USDT | 持久化最大回撤 | 完整平仓胜率 |
| --- | ---: | ---: | ---: | ---: |
| BTC | 997.7816 | -2.2184 | 0.2732% | 0/1 |
| MU | 970.8237 | -29.1763 | 3.9697% | 2/9 |
| SNDK | 1007.8231 | +7.8231 | 0.7702% | 3/9 |
| SK hynix | 1009.7799 | +9.7799 | 0.6970% | 4/5 |

合计权益 **3986.2082047 USDT**，累计 **-13.7917953（-0.34479%）**，与11月6日复盘相同。逐账户核对净收益等于毛实现收益减手续费加资金费；分批退出按入场时间合并，最大回撤取原状态字段。BTC历史信号重放成绩不算执行收益。股票历史平均盈利/平均亏损约为 MU 0.995、SNDK 3.209、SK 1.236，主要属于旧规则，不能用来评价11月7日的新版本。

11月7日22:34启用回踩入场后，三个股票各记录94次小时评估，原始信号分别8、6、10个，新增成交全部为0。公司日历于11月8日00:00北京时间到期；外部参考时段同日13:00到期。停机前日志分别显示 `stock_event_calendar_expired` 和纳指 `session_outside_validity / intraday_eligibility_unavailable`。原始信号不是获准机会或成交，不能把这段空仓称为新规则成功避险，也不能只因零成交放松风险预算。恢复后不补开过去的信号。

## 美股、美债、日债、汇率

研究下载接收时间约北京时间22:09，原始响应、接收时间和SHA256保存在本机研究目录。下列日线、日频曲线和未完成日bar分别解释，不拼成同步盘中收益。

| 因素 | 本次观察 | 策略含义 |
| --- | --- | --- |
| 美股与半导体 | 提供商11月10日完成日线：纳综25065.87（-0.7416%）、SOX9998.33（-0.3548%）、MU725.11（-1.0966%）、SNDK859.99（-2.9915%）；11月9日纳综、SOX分别反弹2.741%、6.855% | 反弹与回撤交替。保留多周期方向、回踩确认和同一具名纳指期货的前瞻30分钟过滤；现金股日涨跌不能直接替代永续合约信号。 |
| 美债 | 官方11月10日2年4.73%、10年4.98%、实际10年2.78%；较11月9日分别+17、+10、+12bp | 名义与实际利率同时上升，对高估值科技及风险资产构成压力。这是机制判断，不是本盘亏损的定量归因。 |
| 日债 | 财政省CSV最新11月10日2年1.784%、10年2.748%、30年3.935%；较11月9日+0.6、+2.4、+2.4bp | 日债也上行，但幅度小于美债。不同期限、估值方法和截面须区分，继续只读诊断，不新增方向权重。没有11月11日官方日频记录。 |
| 汇率 | 约22:09提供商未完成日bar：USDJPY159.555（对前日bar约+0.995%）、DXY102.783（+0.541%）、USDKRW1336.32（-0.560%） | 美元走强而韩元相对美元升值并存，不能概括为所有亚洲货币同向。USDJPY上升代表日元走弱，没有足够持仓或强平证据断言日元套息去杠杆。 |
| 韩国股票 | 提供商11月11日SK现金股日线1277000韩元，较前日+5.363% | 与美股前日窗口不同。韩元折算和现金股表现不能等同USDT永续合约的实际收益。 |

来源：[美债名义曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=202611)、[实际曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_real_yield_curve&field_tdr_date_value=202611)、[财政省日债CSV](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv)、[纳综](https://finance.yahoo.com/quote/%5EIXIC/history/)、[SOX](https://finance.yahoo.com/quote/%5ESOX/history/)、[MU](https://finance.yahoo.com/quote/MU/history/)、[SNDK](https://finance.yahoo.com/quote/SNDK/history/)、[SK](https://finance.yahoo.com/quote/000660.KS/history/)、[USDJPY](https://finance.yahoo.com/quote/JPY=X/history/)、[USDKRW](https://finance.yahoo.com/quote/KRW=X/history/)、[DXY](https://finance.yahoo.com/quote/DX-Y.NYB/history/)。

本轮账户没有新增敞口或损益，不能估计这些因素对收益的独立贡献。BTC原策略已有美债、外汇和全球风险因子；因子分数是准入/风险输入，不是收益贡献。日债与独立跨市场观察器不自动升级为交易信号。研究新下载的数据不回填到过去的可见输入。

## 有限覆盖更新

公司日历仅续审可核实的SNDK、SK官方当前列表，截止 **11月16日00:00北京时间**。SNDK列表显示10月29日财报电话会和11月17日股东会；新覆盖在股东会前结束。SK最新IR列表仍为10月28日业绩及电话会。原已知事件锚点、首次发现、实际发布分钟未知和电话会结束未知均保留。旧日历按原字节哈希存入版本库，核验时刻以前不能使用新快照。

MU页面仍只返回动态空壳，不能证明没有新公告。原有限模拟豁免已到期，本轮不续签；MU继续阻止新增，保护退出正常。来源：[SNDK当前事件](https://investor.sandisk.com/news-events/events)、[SK官方IR](https://news.skhynix.com/en/category/ir/)、[MU事件页面](https://investors.micron.com/events-and-presentations/default.aspx)。

WTI十二月、NYMEX Brent一月、NQ十二月及USDJPY的参考时段续审纽约 **11月11—15日**，截止 **11月16日00:00纽约时间（北京时间13:00）**。新有限窗口从本轮核验时刻开始，保留原首次发现和具名合约身份，不填补11月8日以来缺口。WTI原合约审查截止仍为11月15日00:00 UTC，先于参考时段到期，继续独立约束。

CME标准时段及2026 Globex假期列表已核对；Veterans Day未列入该Globex假期表，其单独结算通知说明常规结算时间。没有将债券现金市场休市误作所有期货停市。参考时段仍不证明实际交易状态或临时无停牌。OANDA外汇标准时段仅作经纪商参考；其假期页没有逐日确认本次2026日期。来源：[CME标准时段](https://www.cmegroup.com/markets/equities/sp-500-and-nasdaq-100-futures.html)、[Globex假期](https://www.cmegroup.com/trading-hours.html)、[11月11日结算通知](https://www.cmegroup.com/tools-information/holiday-calendar/files/2026/veterans-day-holiday-settlement-times-2026.pdf)、[CL规格](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html)、[BZ规格](https://www.cmegroup.com/markets/energy/crude-oil/brent-crude-oil-last-day.contractSpecs.html)、[OANDA时段](https://www.oanda.com/us-en/trading/hours-of-operation/)、[假期页](https://www.oanda.com/us-en/trading/holiday-trading-hours/)。

盘中报价仍需满足原20分钟新鲜度、同一合约、时钟及历史长度条件。独立国际事件模板仍为过期的十月范围；其刷新器不支持自动发现十一月事件，本轮不靠改截止时间虚构覆盖，仍显示不可用。

## 进程恢复与验证

检查时没有Python模拟服务、观察器或终端进程。监督器最后记录21:40停止，Windows System事件1074/6006/6005记录21:40—21:43的系统升级重启，支持本次进程退出与系统更新相关；主盘落盘的 `running=true` 已过时。零星状态发布拒绝访问后有后续新心跳，没有据此认定金融账本损坏。

原审核程序绑定的10月版VS Code扩展路径已不存在。使用11月版实际可执行文件运行现有只读合成诊断：结构化回复有效，并拒绝明确缺上下文、触发硬回撤的请求。仅更新模拟盘专用路径、SHA256、版本；审核模型和策略不变，不修改全局设置或密钥。

操作助手位于本机 `data/research/paper_macro_review_20261111/operate_review.py`，分步记录进程捕获、已停止确认、持锁保存102账户及金融历史前缀、观察器配置指纹维护、远程提交核对与 `--require-existing` 恢复。观察器维持原起点、去重与决策日志；98个取消账户保持原字节，不恢复对照。源代码指纹和账户规则不迁移，只有只读观察器配置指纹有明确维护记录。

相关回归 **449 passed、43 subtests passed**；合成审核诊断通过，停机保全核验通过。旧日历归档增加明确LF规则，避免Windows与其他平台检出后改变按字节命名的历史哈希。运行配置在发布前统一为仓库LF字节，观察器记录单独指纹维护，解析内容、时钟和报价身份不变。

测试和实际恢复结果记录在同目录 `pytest.log`、`verified_stopped.json`、`verified_stopped_normalized.json`、`resumed_main.json`、`resumed_observer.json`、`verified_live.json`。恢复验证必须分别确认监督器/账户心跳、模拟模式、原账本、取消账户、外部报价及仍存在的限制，不能只以进程存在宣称可交易或盈利改善。
