# 四账户继续前瞻模拟验证

用户明确指示：“不要管这个开关，就是要跑验证，模拟盘管他的呢”。北京时间 **2026-10-05 23:25:08**，四账户联合模拟恢复新增开仓权限，执行器 PID **41311**。历史盈利、胜率和样本量筛查不再作为本轮模拟开仓的前提；历史研究结果与未验证状态继续保留，不将用户授权记为盈利验证通过。

## 当前行为

- BTC继续现有多因子信号；MU继续15m双向均线交叉；SNDK继续5m双向突破；SKHYNIX继续15m双向均线交叉。
- 四账户均允许新增模拟交易。开仓仍需实际信号，以及新鲜完整数据、数量与保证金、价差、成交量、资金费、追价和账户风险条件通过。
- 保留每30秒观察、股票每笔0.25%计划风险、10x杠杆上限、手续费/滑点/资金费计账、保护退出和冷却。解除资格限制不强制产生交易。
- 原有共同起点、余额、已实现损益、手续费、资金费、成交次数和仓位保留。SNDK空头仓位与SKHYNIX多头仓位继续原保护参数。
- 股票资格政策变更按现有执行器规则记录新生效时点；空仓MU等待变更后新的已收盘信号，不补做过去信号。BTC等待当前有效策略目标。
- 运行模式仍为本地simulation，`live_orders_allowed=false`，不提交交易所订单。

## 配置与验证

新股票运行配置：[stock_unrestricted_paper_validation_20261005.json](../config/stock_unrestricted_paper_validation_20261005.json)。原观察配置完整保留；[四账户计划](../config/parallel_simulation_plan_20261005.json)切换至新配置，并授权BTC本轮模拟新增。

58项现有测试通过，覆盖模拟执行、双向入场、数据缺失与风险拦截、保护退出，以及策略资格授权不能绕过其他执行限制。重启后四账户均healthy、无当前错误，终端状态HTTP 200。逐项比较停机后备份与恢复后账本，余额、损益、费用、成交和原仓位完全一致。股票运行规则只改变候选记录标识、开仓开关与模拟授权说明。

激活与核验记录：`data/runtime/paper_validation_reenabled_20261005.json`。备份：`data/parallel_simulation/btc_memory_stocks_1000_each_10x_20261005/paper_validation_reenable_backups/1791213907657/`。manifest记录资格政策修订，股票rule_revisions记录规则切换。新执行器日志：`data/runtime/parallel_paper_validation_stdout.log`及`parallel_paper_validation_stderr.log`。

该进程继续使用本次登录会话的后台运行方式；本次没有新增系统启动服务。

```sh
.venv/bin/python -m pytest -q tests/test_parallel_simulation.py tests/test_strategy_qualification.py tests/test_account_execution.py
```
