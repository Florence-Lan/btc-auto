# BTC 过期等待项修复（2026 年 10 月 24 日）

本轮修复 BTC 回撤入场等待项的终止顺序。原始方向的一小时首次入场窗口仍然有效；均线、四小时确认、止损、仓位、费用及自动晋级规则均未调整。

原程序在窗口结束后正确停止请求专用回撤行情，但执行层先检查行情完整性，导致已经存在的 `waiting_retrace`、`waiting_confirmation` 或 `ready` 等待项仍报告 `data_unavailable`、不断重试。新程序允许到期等待项先进入原有 `advance_pending` 验证和终止流程，记录 `expired` 并消费该来源；期限前缺数据仍阻止入场。冻结来源、策略和期限的校验仍执行，损坏记录不会获得入场许可。新来源使用自己的起点和期限，退出及已有仓位管理沿用原规则。

实现：[trading_execution.py](../scripts/trading_execution.py)。回归：[test_btc_pullback_execution.py](../tests/test_btc_pullback_execution.py)。新增 67 项测试覆盖到期前一毫秒、到期等号及之后，缺失／损坏／未来行情、重启和消费幂等、新多空来源恢复入场、平仓／反向减仓／账户硬止损，以及伪造期限仍被拒绝。

以 UTF-8 模式运行全量 Python 回归：**1,972 项通过、2 项跳过、43 个 subtests 通过**。本轮未修改界面。离线复制真实 BTC 状态验证，旧 `data_unavailable` 转为 `expired` 且消费来源，资金字段相同，未写入任何账户文件。

验证证据位于 [本轮运维目录](../data/research/btc_pullback_expiry_20261024/)：`pytest_full.log`、`actual_pending_replay.json`。该目录被 git 忽略，不随源码推送。

本轮还发现一项独立停机：北京时间 **16:39** 原五项服务因审核程序固定路径已消失而关闭，计划任务随后重复校验失败。旧 VS Code 插件版本的 `codex.exe` 已被升级替换。使用现有合成诊断确认新版本返回有效结构并拒绝缺上下文／触发硬止损的测试请求后，只更新本地模拟审核配置的程序路径、SHA256 和 CLI 版本；审核模型 `gpt-6.1-sol`、`medium`、90 秒超时及门槛保持。

诊断记录：[llm_preflight.json](../data/research/btc_pullback_expiry_20261024/llm_preflight.json)。这次诊断没有读取 `.env` 或账户账本，也没有发送交易指令。审核程序升级作为独立工具链修订记录，不能把更新前后的审核结果视为同一固定程序的盈利验证。

上线采用原账本恢复：在全部相关写入停止并持有七把排他锁后备份、校验历史，仅修订主账户审计记录和受影响实验的执行指纹。BTC 再入场实验另记录已诊断的审核工具链变更。原起始时间、策略设计、资本、成交和资金费保持；SNDK 冻结运行时及其清单保持原字节。跨市场观察器不依赖本次修改，继续运行。

北京时间 **16:54:02** 完成版本记录切换，持锁阶段 **384 个文件、122,890,852 字节**完成备份及逐字节哈希校验；除四份获准的 manifest 元数据外均未改动。随后恢复原五项服务任务和独立退出对照任务。迁移证明：[activation.json](../data/research/btc_pullback_expiry_20261024/activation.json)，备份索引：[stopped_before_inventory.json](../data/research/btc_pullback_expiry_20261024/stopped_before_inventory.json)，迁移脚本：[migrate_manifests.py](../data/research/btc_pullback_expiry_20261024/migrate_manifests.py)。执行实现修订保留在同账本审计记录内，不作为一次新实验或新的盈利证据。

恢复后的独立在线验收通过：原五项服务、观察器和退出对照各有一棵有效进程树，心跳有效；受影响的三个实验源码指纹及审核绑定匹配，相关源码修改时间均早于新进程启动。SNDK 的 40 个冻结文件保持原版本，观察器原 PID 保持。按成交及资金费重放主四账户与退出对照 24 账户，共 28 本账对平；205 份历史日志前缀一致。主账户合计钱包仍为 **4,016.97044312 USDT**。

实际 BTC 等待项和决策均已变为 `expired`、来源已消费，`retryable=false`；主账户当前 `healthy / entry_blocked`，阻断原因准确显示原始窗口及回撤等待过期，保护行情监控为 `healthy`。进程版本核验依据是源码哈希、进程路径、创建时间及实时行为，不是 Python 内存镜像证明。

在线证据：[post_activation_audit.json](../data/research/btc_pullback_expiry_20261024/post_activation_audit.json)，核验脚本：[verify_runtime.py](../data/research/btc_pullback_expiry_20261024/verify_runtime.py)。运行正常只表示本次检查截面符合预期，策略收益改善仍需独立前瞻样本。

回滚时只恢复代码、执行指纹及审核程序绑定；不得回拨钱包、成交或资金费。旧审核程序路径已经不存在，恢复该旧路径不能恢复服务，需要可验证的可用程序。
