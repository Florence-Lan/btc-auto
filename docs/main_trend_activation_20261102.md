# 主股票趋势修复维护工具

已于北京时间 **2026-11-02 10:51:20.850** 在原 MU、SNDK、SK hynix 主模拟账户启用并恢复原监督服务。实际新主 worker 为 **13060**；恢复后三个股票账户均已写出新规则运行标记、当前趋势与退出诊断。SNDK 原空单 0.04 的资金和旧预算保持，新增前瞻退出与利润保护叠加已生效。

停止状态及恢复后保全核验均为 **PASS**：102 个原账户身份与历史得到核对，98 个已取消对照账户及其元数据字节未变，金融日志原字节前缀保持，主盘初始资金仍为 4000 USDT。监督器仅恢复 `main` 和 `btc_signals`。证据见 `data/research/main_trend_activation_20261102/{activation,verified_stopped,verified_live}.json`。

全量回归 **3111 passed、2 skipped、43 subtests passed**；最后补充的故障隔离、覆盖到期及实际执行边界定向回归 **255 passed**。本地股票页面已核对三个账户的趋势与纳指状态渲染；当前资料不可用时显示明确提示。页面的旧活动对照入口已隐藏，策略说明同步为现行规则。

新规则最早可计入退出确认的完整小时为北京时间 **11:00—12:00**、**12:00—13:00**，是否退出取决于两根实际完成行情与执行条件。板块私有报价从激活后真实观察积累，启动初期为未知并限制新增，已有保护退出保持运行。BTC 原上游冷却与 MU 原公司事件复核限制仍独立存在。

纳指会话参考已根据 [CME 股指期货时段](https://www.cmegroup.com/markets/equities/sp-500-and-nasdaq-100-futures.html)及 [CME 2026 假期列表](https://www.cmegroup.com/trading-hours.html)续审纽约 11 月 2 日至 7 日，有限覆盖截至北京时间 **11 月 8 日 13:00**。保持原首次发现与有效起点、具名合约身份，未把常规时段参考当成实际停牌状态。原始配置字节和观察器审计记录保存在 `nq_session_review/`；观察器于 10:53:27 完成更新恢复，新 worker **4688** 已自然采集并证明加载新覆盖，原 tracker 与旧报价、决定历史前缀保持。原国际事件日历刷新错误仍独立存在。

## 本轮规则

本轮直接修复原主模拟盘三个股票账户，不创建新的对照盘。策略仍未被证明具有稳定盈利能力；本次是明确的前瞻规则修订。

- 每次有效观察更新独立的 `current_trend`，分别呈现已完成 1 小时与 4 小时的价格方向、均线背景及收盘时间。持仓期间也更新，行情或校时失败时明确显示不可用；原入场诊断保留为历史证据。
- 保留原小时 Donchian20 原始信号、入场价格有效性和执行检查。多头新增与已有空头对称的 4 小时价格确认通道：收盘在 EMA8、EMA24 上方，EMA8 较上一根上升，快慢线绝对间距不低于 0.4%。严格确认与价格确认任一通过后仍须满足原质量检查，不制造原始信号。
- 新增主盘独立趋势失效退出：连续两根完整小时线的 EMA8/24 与持仓方向相反，且收盘跨到这两条线的另一侧。只计整根开盘时间不早于本次激活和该仓入场的小时线，无须再等价格越过 EMA60。已有保护止损、账户风险、持仓期限和资金费退出保持优先；部分平仓继续退出剩余仓位。
- 纳指十二月具名期货使用现有只读观察器，主盘各账户独立积累激活后的真实报价。相对报价商前收参考与过去 30—35 分钟同身份报价均上涨时偏强，均下跌时偏弱；分歧为中性，缺数、过期或未完成预热为未知。未知阻止新风险，偏强阻止新增空头、偏弱阻止新增多头。重试、数量预算、最终成交前复核；独立保护退出及技术失效退出不依赖纳指数据。
- 保留各品种原追踪启动和最低锁盈阈值，新增激活后实时、足以退出全部数量的真实盘口观察。达到原阈值即可单调收紧保护价，无须等待小时收盘；不使用激活前盘中极值，不制造按旧价格的回溯成交。SNDK 原阈值仍为 4% 启动、1% 最低标的收益锁定；这不是扣费后保本或成交价格保证。

净目标 100 USDT、单笔计划风险不超过 25 USDT 或权益的 2.5%、至少四倍计划净收益风险比与十倍敞口上限保持。SNDK 原仓沿用原资金和预算规则，仅叠加生效后的趋势失效退出与盘中利润保护。

纳指为风险代理，不是半导体全样本广度或各永续合约的方向预测。新历史通常需 40—50 分钟以上才可用于 30 分钟比较（含报价延迟），缺数不解释为市场走弱。常规日历边界、来源身份或前收版本改变后按既有规则重新积累。

## 维护流程

本文件描述 `scripts/activate_main_trend_20261102.py` 的维护边界。编写及测试该工具不代表已经激活新规则；实际部署以研究目录中的 `activation.json` 和恢复后的心跳核验为准。

工具只将原主盘 MU、SNDK、SK hynix 的规则增加 `main_stock_trend_policy`，原 BTC 规则、四账户各 1000 USDT 初始资金、创建时间、已有资金及成交历史均保留。现有 SNDK 的数量、入场价、止损及旧预算规则快照不在迁移允许修改范围内；其新增前瞻退出由主运行时叠加处理。

工具接受的唯一规则迁移是 `main_trend_only`，对应 `main_trend_active_after_ms`。同一时刻更新 `signal_active_after_ms` 并撤销旧 `pending_signal`，保留已消费的 `last_signal_time_ms`；旧信号不恢复成新的入场机会。原短确认及净 100 / 风险 25 的激活时刻、预算参数保持原值。

所有命令在仓库根目录运行，使用本仓库 Python：

```powershell
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py plan
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py check --reviewed-plan data/research/main_trend_activation_20261102/plan.json
```

`plan`、`check` 只读取生产状态并输出研究证据，不修改配置、账户或服务。计划绑定新 profile、base config、主 plan、manifest、停用配置及源码哈希，不绑定持续变化的钱包或行情估值。修改代码或配置后需要重新产生并审阅具体计划。

明确进入维护后，顺序为：

```powershell
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py capture-control
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py pause --reviewed-plan data/research/main_trend_activation_20261102/plan.json
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py apply --reviewed-plan data/research/main_trend_activation_20261102/plan.json
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py verify
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py resume
.venv/Scripts/python.exe -B scripts/activate_main_trend_20261102.py verify --live
```

`capture-control` 记录精确任务动作及进程创建身份。`pause` 只禁用和停止原主监督器，再按记录身份结束其受控进程树；五个已停用的独立实验任务始终要求保持停用。跨市场只读观察进程不停止。`apply` 再确认监督器与全部受保护 writer 消失，同时持有监督器、主 runner、BTC producer 三把 OS 进程锁，避免维护与恢复任务竞争。

停止状态保存全部 102 个原账户的原始状态备份，以及成交、资金费、余额、performance、规则历史文件的固定字节前缀哈希。迁移后，BTC 与 98 个已取消账户要求状态及字节完全一致；三个主股票账户只允许增加政策及前瞻激活元数据。金融字段、原持仓、原规则历史固定前缀不允许修改。只有三个主股票的规则审计日志允许追加本次迁移记录。

主 manifest 保留最初的 `source_hashes`、创建时间、资本和历史 profile 引用；本次规则变化追加审计链，并把实际本地依赖闭包写入 `current_source_sha256`。已取消实验的 manifest 与 fingerprint 不更新，也不调用它们的 prepare 或重新启动。

`resume` 要求迁移成功证据及停止状态验证均通过，并确认监督器只会恢复 `main`、`btc_signals`。恢复后的 `verify --live` 允许主盘自然行情、成交与资金费变化，继续核对四账户身份、BTC 预算政策、股票政策激活时刻、历史前缀与全部已取消账户。还需检查新 worker 的创建时间、主盘及股票趋势状态心跳；本工具不把进程存在视为策略评估正常。

任何应用失败都会写 `activation_failed.json`，禁止自动恢复；工具不会将账户金融状态或历史回滚到维护前快照。应先检查是否出现部分规则迁移，在停写状态下完成修复及重新审阅。
