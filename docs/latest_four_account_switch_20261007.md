# 四账户最新策略切换准备（2026-10-07）

用户明确要求四个原模拟账户都切换到最新策略。默认启动计划已指向最新股票配置，BTC 保留当前最新多因子 10x 版本；此次授权涵盖模拟规则切换，不发送交易所订单。

| 账户 | 最新规则 |
|---|---|
| BTC | `multifactor_regular_10x_20261004.json`，六因子/first-seen、原资金费与来源门槛 |
| MU | 15m 双向 EMA8/24/60 转换，1R 平半、扣费保本与 1.5 ATR 跟踪 |
| SNDK | 1h Donchian20 突破＋EMA60 方向过滤，2 ATR 止损，净初始保证金 60% 目标与跟踪退出 |
| SKHYNIX | 15m 双向 EMA8/24/60 转换，1R 平半、扣费保本与 1.5 ATR 跟踪 |

默认[四账户计划](../config/parallel_simulation_plan_20261005.json)沿用原 plan ID 和全部原 `state_path`，不新建一轮账户。股票[最新配置](../config/stock_hourly_aligned_breakout_paper_20261007.json)中 MU/SKHYNIX 合并后规则与前版一致；BTC 策略路径与规则不变。SNDK 是本次实际变化的账户。

## 库存与记录

原迁移函数在执行器进程锁内备份旧状态，保留余额、已实现损益、成交序号、费用、资金费、持仓、初始止损、当前止损及库存历史。SNDK 规则切换改为 `entry_and_exit`，清除旧待入场信号，并要求新的收盘信号；既有持仓继续沿用原保护止损距离，后续应用新的目标、持仓期限和跟踪规则。止损不会因改配置直接放宽。原 1R 进度保留在旧库存记录中，但最新 SNDK 规则不再继续分批止盈。

旧退出对照有自己的 5m 信号，而新 SNDK 使用 1h。执行器已补齐不同周期对照的独立信号读取：共享盘口、标记、资金费和 5m 流动性，只单独读取控制组的已完成 5m 信号，避免错误地把 1h K 线当成 5m。

## 时钟与检查

当前 Windows 系统时间比 Aster 慢约 8 秒。股票执行器现在根据自己取得的公开交易所时间响应建立单调时钟锚点，每轮刷新。新鲜度阈值保持不变；超过五秒的时钟请求或超过十五秒的响应年龄被拒绝，过期报价仍阻止执行。历史重放和测试提供的原始时钟响应没有运行时锚点，沿用原确定性时钟。

全量 Python 检查 **894 passed, 2 skipped**。新增覆盖库存/余额/成交保留、重复迁移幂等性、控制组周期、时钟偏差与过期报价拒绝、缺账本时拒绝初始化。

[机器可读核验](latest_four_account_switch_verification_20261007.json)保存了最新策略路径、原账户路径、测试结果、行情检查和服务未切换的阻塞原因。

三个股票最新配置在隔离的一次性诊断账本中取得真实公开行情，均为 `healthy`、零成交，MU/SKHYNIX 为 15m、SNDK 为 1h。诊断不是原账户启动，也不构成前瞻盈利验证；证据见 `data/validation/latest_four_account_switch_20261007/market_probe.json`。

## 原服务仍待接入

本机没有原四账户的 `manifest.json` 或四份 `state.json`，也没有四账户执行进程；用户 SSH 目录没有可用连接配置。没有用另起四个空账户代替原账户。对原目录执行 `--once --require-existing` 已明确拒绝启动，缺少的账户为 BTC、MU、SNDK、SKHYNIX。

此次已完成配置和代码准备；**未宣称原四账户已经运行最新版**。需要原运行机器/目录或 SSH 连接别名，才能核实服务、备份账本、拉取此版本、重启原执行器并检查连续状态更新。若需要更新 BTC 信号主管，应沿用原 supervisor 的 factor-profile、状态/报告/成交路径，避免另建信号账户或重复执行器。

原服务的四份账本和 manifest 齐备后，续接原账户的命令为：

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/run_parallel_simulation.py --require-existing
```

重启前应停止该原执行器实例并等其进程锁释放；此命令不停止其他服务。进程锁继续拒绝重复实例，且 `--require-existing` 在目录创建和 bootstrap 前拒绝缺失的原账本。
