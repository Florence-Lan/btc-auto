# Binance BTCUSDT Perpetual Strategy (10x)

这个项目包含一个可运行的策略机器人 `bot.py`，用于 Binance USDT 永续（默认 `BTCUSDT`），杠杆默认 `10x`。

策略信号由两部分同时确认：

1. K线趋势：`EMA 快线/慢线 + 斜率 + 价格位置`
2. 大单与盘口：`大额主动买卖失衡 + 深度盘口失衡`

只有当趋势与资金流方向一致时才开仓：

- 趋势多头 + 大单/盘口偏多 -> 开多
- 趋势空头 + 大单/盘口偏空 -> 开空
- 其余情况 -> 不开新仓

## 运行步骤

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 准备环境变量

```bash
cp .env.example .env
```

3. 先用模拟开关验证（强烈建议）

- 保持 `.env` 中 `DRY_RUN=true`
- 运行：

```bash
python3 bot.py
```

4. 实盘前再切换

- 检查参数后，把 `DRY_RUN=false`
- 再运行 `python3 bot.py`

## 关键参数

- `LEVERAGE`：杠杆倍数（默认 10）
- `HEDGE_MODE`：是否双向持仓（默认 `false`，即单向持仓）
- `MARGIN_USDT`：每次开仓使用的保证金（USDT）
- `TIMEFRAME`：K线周期（默认 `5m`）
- `BIG_TRADE_USDT`：定义“大单”的单笔成交额阈值
- `FLOW_SCORE_THRESHOLD`：资金流综合分数阈值
- `TRADE_WEIGHT`：成交流失衡在综合分数中的权重（0~1）

## 风险提示

- 这是可执行下单脚本，不是回测结果保证。
- `DRY_RUN=false` 后会真实下单，请先小仓位验证。
- Binance API Key 建议：
  - 只开交易权限，不开提现权限
  - 绑定固定 IP
  - 定期轮换密钥

## 现在可以怎么看多空力量对比

常见有 8 类（可组合使用）：

1. 主动买卖成交量失衡（Taker Buy/Sell 或 CVD）
2. 大单成交方向失衡（本策略已使用）
3. 盘口挂单失衡（Order Book Imbalance，本策略已使用）
4. 未平仓量 OI 与价格联动
5. 资金费率（Funding Rate）与变化速度
6. 全市场或大户多空账户比（Long/Short Ratio）
7. 爆仓分布（多头爆仓 vs 空头爆仓）
8. 永续-现货基差（Basis / Premium Index）

实战里通常不会只看一种指标，建议至少使用：`趋势 + 成交流 + 盘口 + OI/Funding`。
