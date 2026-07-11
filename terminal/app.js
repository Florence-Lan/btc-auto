const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fmt = (value, digits = 2) => value == null ? "—" : Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const pct = (value, digits = 2) => value == null ? "—" : `${Number(value).toFixed(digits)}%`;
let latest = null;
let logViewCleared = false;

function timeAgo(seconds) {
  if (seconds == null) return "无心跳";
  if (seconds < 60) return `${Math.round(seconds)} 秒前`;
  return `${Math.round(seconds / 60)} 分钟前`;
}

function signedMoney(value) {
  if (value == null) return "—";
  const number = Number(value);
  return `${number > 0 ? "+" : number < 0 ? "−" : ""}${fmt(Math.abs(number))}`;
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3600);
}

async function control(action, extra = {}) {
  const response = await fetch("/api/terminal/control", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Terminal-Action": "1" },
    body: JSON.stringify({ action, ...extra }),
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || "操作失败");
  await refresh();
  return payload.result;
}

function renderChart(points = []) {
  const svg = $("#equityChart");
  const empty = $("#emptyChart");
  if (points.length < 2) {
    svg.innerHTML = "";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  const width = 900, height = 260, pad = 28;
  const values = points.map(point => Number(point.equity));
  const min = Math.min(...values) * .995, max = Math.max(...values) * 1.005;
  const x = index => pad + index / (points.length - 1) * (width - pad * 2);
  const y = value => pad + (max - value) / (max - min || 1) * (height - pad * 2);
  const coords = points.map((point, index) => [x(index), y(Number(point.equity))]);
  const line = coords.map((point, index) => `${index ? "L" : "M"} ${point[0]} ${point[1]}`).join(" ");
  const area = `${line} L ${coords.at(-1)[0]} ${height - pad} L ${coords[0][0]} ${height - pad} Z`;
  svg.innerHTML = `${[0,1,2,3].map(i => `<line class="grid-line" x1="${pad}" x2="${width-pad}" y1="${pad+i*(height-pad*2)/3}" y2="${pad+i*(height-pad*2)/3}"/>`).join("")}<path class="equity-area" d="${area}"/><path class="equity-path" d="${line}"/><text class="axis-label" x="${pad}" y="${height-7}">${new Date(points[0].time_ms).toLocaleTimeString()}</text><text class="axis-label" text-anchor="end" x="${width-pad}" y="${height-7}">${new Date(points.at(-1).time_ms).toLocaleTimeString()}</text>`;
}

function renderRows(body, rows, columns, emptyText) {
  body.innerHTML = rows.length
    ? rows.map(row => `<tr>${columns.map(column => `<td class="${column.className ? column.className(row) : ""}">${column.render(row)}</td>`).join("")}</tr>`).join("")
    : `<tr class="empty-row"><td colspan="${columns.length}">${emptyText}</td></tr>`;
}

function render(data) {
  latest = data;
  const live = data.mode === "LIVE";
  const running = Boolean(data.execution.runtime.running);
  const emergency = data.execution.emergency;
  $("#modeBadge").textContent = data.mode;
  $("#modeBadge").classList.toggle("live", live);
  $("#healthDot").classList.toggle("online", running && !emergency);
  $("#healthText").textContent = emergency ? "急停锁定" : (running ? `${live ? "实盘" : "模拟盘"}运行中` : "自动化已暂停");
  $("#markPrice").textContent = data.market.mark_price ? `$${fmt(data.market.mark_price, 1)}` : "—";
  $("#marketTime").textContent = data.market.data_time_ms ? `Binance · ${new Date(data.market.data_time_ms).toLocaleTimeString("zh-CN")}` : "等待行情";
  $("#candidateId").textContent = data.strategy.candidate_id || "BTC 自动策略";
  $("#runtimeMeta").textContent = running ? `PID ${data.execution.runtime.pid} · ${live ? "真实下单" : "本地模拟成交"}` : `${live ? "实盘" : "模拟盘"}进程未运行`;
  $("#exchangeStatus").textContent = live
    ? (data.exchange.connected ? "Binance 主网账户已连接" : `实盘账户未连接${data.exchange.error ? " · " + data.exchange.error : ""}`)
    : (data.exchange.connected ? "Binance 主网实时行情已连接" : "Binance 主网行情未连接");
  $("#equityLabel").textContent = live ? "实盘钱包余额" : "模拟权益";
  $("#equitySource").textContent = live ? "USDT · Binance" : "USDT · local simulation";
  $("#returnSource").textContent = live ? "由 Binance 账户结算" : "模拟运行期";
  $("#chartTitle").textContent = live ? "实盘账户净值与风险状态" : "模拟净值与风险状态";
  $("#walletBalance").textContent = fmt(live ? data.account.wallet_balance : data.account.margin_balance);
  $("#availableBalance").textContent = fmt(data.account.available_balance);
  $("#unrealizedPnl").textContent = fmt(data.account.unrealized_pnl);
  $("#realizedReturn").textContent = pct(data.account.realized_return_pct);
  $("#drawdown").textContent = pct(data.risk.drawdown_pct);
  $("#drawdownBar").style.width = `${Math.min(100, Number(data.risk.drawdown_pct || 0) / data.risk.hard_limit_pct * 100)}%`;
  $("#targetNotional").textContent = signedMoney(data.strategy.target_notional);
  $("#targetNotional").className = Number(data.strategy.target_notional || 0) > 0 ? "positive" : Number(data.strategy.target_notional || 0) < 0 ? "negative" : "";
  $("#targetLeverage").textContent = `目标杠杆 ${Number(data.strategy.target_leverage || 0).toFixed(3)}×`;
  $("#macroMultiplier").textContent = data.risk.macro_current.risk_multiplier == null ? "—" : `${fmt(data.risk.macro_current.risk_multiplier, 2)}×`;
  const coverage = data.risk.macro.factor_coverage_pct || {};
  $("#macroCoverage").textContent = Object.keys(coverage).length ? `${Object.keys(coverage).length} 组因子在线` : "等待产生信号";
  $("#observations").textContent = data.strategy.observations;
  $("#heartbeat").textContent = `信号心跳 ${timeAgo(data.execution.heartbeat_age_seconds)}`;
  $("#strategyStatus").textContent = emergency ? "EMERGENCY" : (running ? data.mode : "PAUSED");
  $("#strategyStatus").classList.toggle("live", live);
  $("#strategyList").innerHTML = data.strategy.strategy_modes.map(mode => `<div class="strategy-item"><span>${mode === "trend" ? "多周期趋势" : mode === "timeseries_trend" ? "6H 时间序列趋势" : mode}</span><strong>ACTIVE</strong></div>`).join("");
  const strategyNames = { trend: "多周期趋势", timeseries_trend: "6H 时间序列趋势", range: "区间策略" };
  renderRows($("#sleeveSignalsBody"), data.strategy.sleeves || [], [
    { render: row => strategyNames[row.name] || row.name },
    { render: row => row.status === "POSITION" ? "持仓中" : "监控中", className: row => row.status === "POSITION" ? "positive" : "" },
    { render: row => row.direction, className: row => row.direction === "LONG" ? "positive" : row.direction === "SHORT" ? "negative" : "" },
    { render: row => `${Number(row.target_leverage || 0).toFixed(3)}×` },
    { render: row => pct(row.return_pct) },
    { render: row => pct(row.max_drawdown_pct) },
    { render: row => row.trades ? pct(row.win_rate_pct, 1) : "样本不足" },
    { render: row => row.profit_factor == null ? "—" : fmt(row.profit_factor, 2) },
  ], "策略数据尚未生成，请先执行一次策略循环");
  $("#signalFreshness").textContent = data.market.signal_age_seconds == null ? "无信号" : `信号 ${timeAgo(data.market.signal_age_seconds)}`;
  const details = data.account_details || {};
  $("#initialBalanceDetail").textContent = fmt(details.initial_balance);
  $("#walletBalanceDetail").textContent = fmt(details.wallet_balance);
  $("#marginBalanceDetail").textContent = fmt(details.margin_balance);
  $("#realizedPnlDetail").textContent = fmt(details.realized_pnl);
  $("#realizedPnlDetail").className = Number(details.realized_pnl || 0) >= 0 ? "positive" : "negative";
  $("#feesPaidDetail").textContent = fmt(details.fees_paid);
  $("#positionNotionalDetail").textContent = fmt(details.position_notional);
  $("#positionQtyDetail").textContent = fmt(details.position_qty, 6);
  $("#entryPriceDetail").textContent = fmt(details.entry_price, 1);
  $("#lastExecution").textContent = data.execution.last_cycle_at_utc ? `执行 ${new Date(data.execution.last_cycle_at_utc).toLocaleString("zh-CN")}` : "尚未执行";
  const targetQty = Number(data.strategy.target_signed_qty || 0);
  $("#targetPosition").textContent = targetQty > 0 ? "LONG" : targetQty < 0 ? "SHORT" : "FLAT";
  $("#targetPosition").className = targetQty > 0 ? "positive" : targetQty < 0 ? "negative" : "";
  $("#executionPermission").textContent = live ? "BINANCE 实盘下单" : "仅本地模拟";
  $("#executionPermission").className = live ? "negative" : "positive";
  $("#riskMultiplier").textContent = `${fmt(data.risk.drawdown_multiplier, 2)}×`;
  $("#emergencyState").textContent = emergency ? "已锁定" : "未触发";
  $("#emergencyState").className = emergency ? "negative" : "positive";
  const macroAgeHours = data.market.macro_snapshot_age_seconds == null ? null : data.market.macro_snapshot_age_seconds / 3600;
  $("#macroFreshness").textContent = macroAgeHours == null ? "无快照" : `${fmt(macroAgeHours, 1)}h ago`;
  const factorNames = { vix: "VIX 恐慌", dollar: "美元指数", metals: "黄金 / 白银", sentiment: "Crypto Fear & Greed" };
  const contributions = data.risk.macro_current.contributions || {};
  $("#factorList").innerHTML = data.strategy.macro_factors.map(factor => {
    const value = contributions[factor];
    const label = value == null ? "WAIT" : `${value >= 0 ? "+" : ""}${Number(value).toFixed(2)}`;
    return `<div class="factor-item"><span>${factorNames[factor] || factor}</span><strong class="${value == null ? "" : (value >= 0 ? "positive" : "negative")}">${label}</strong></div>`;
  }).join("");
  $("#positionCount").textContent = data.positions.length;
  $("#orderCount").textContent = data.open_orders.length;
  $("#tradeCount").textContent = data.recent_trades.length;
  renderRows($("#positionsBody"), data.positions, [
    { render: row => row.symbol }, { render: row => row.side, className: row => row.side === "LONG" ? "positive" : "negative" },
    { render: row => fmt(row.quantity, 5) }, { render: row => fmt(row.mark_price, 1) }, { render: row => fmt(row.unrealized_pnl) }, { render: row => row.source },
  ], "当前无持仓");
  renderRows($("#ordersBody"), data.open_orders, [
    { render: row => row.symbol }, { render: row => row.type }, { render: row => row.side }, { render: row => fmt(row.price) }, { render: row => fmt(row.quantity) }, { render: row => row.status },
  ], "当前无活动订单");
  renderRows($("#tradesBody"), data.recent_trades, [
    { render: row => (row.time_utc || "").slice(0, 19).replace("T", " ") || "—" }, { render: row => (row.mode || data.mode).toUpperCase() },
    { render: row => row.side || "—" }, { render: row => fmt(row.price, 1) }, { render: row => fmt(row.quantity, 5) },
    { render: row => fmt(row.realized_pnl), className: row => Number(row.realized_pnl || 0) >= 0 ? "positive" : "negative" },
  ], live ? "实盘成交请以 Binance 账户记录为准" : "暂无模拟成交");
  renderChart(data.equity_curve || []);
  if (!logViewCleared) {
    const lines = [...data.logs, ...data.errors.map(line => `[ERROR] ${line}`)];
    $("#logStream").textContent = lines.join("\n") || "暂无运行日志";
    $("#logStream").scrollTop = $("#logStream").scrollHeight;
  }
  $("#startButton").disabled = running || Boolean(emergency);
  $("#pauseButton").disabled = !running;
  $("#onceButton").disabled = running || Boolean(emergency);
  $("#simulationMode").classList.toggle("active", !live);
  $("#liveMode").classList.toggle("active", live);
  $("#simulationMode").disabled = running || Boolean(emergency) || !live;
  $("#liveMode").disabled = running || Boolean(emergency) || live;
  $("#emergencyButton").textContent = emergency ? "解除急停" : "急停";
  $("#guardStatus").textContent = live ? "LIVE ARMED" : "SIM SAFE";
  $("#guardStatus").className = `guard-badge ${live ? "negative" : "positive"}`;
  $("#guardDescription").textContent = live
    ? "实盘会向 Binance 主网提交订单；急停将撤单并以 reduceOnly 市价平仓。"
    : "模拟盘读取 Binance 主网实时行情，成交和资金仅写入本地账本。";
  $("#simulationCapital").classList.toggle("hidden", live);
  const initialBalanceInput = $("#simulationInitialBalance");
  if (!live && document.activeElement !== initialBalanceInput) {
    initialBalanceInput.value = data.execution.simulation_initial_balance ?? 100;
  }
  $("#resetSimulation").disabled = running || Boolean(emergency) || live;
  $("#notionalLimit").textContent = data.execution.max_notional_usdt > 0 ? `${fmt(data.execution.max_notional_usdt)} USDT` : "按权益 × 2";
  $("#leverageLimit").textContent = `${fmt(data.execution.leverage, 1)}×`;
  $("#observationInterval").textContent = `${data.execution.check_interval_seconds}秒检查 / ${Math.round(data.execution.strategy_bar_seconds / 60)}分钟K线`;
  $("#emergencyDescription").textContent = live
    ? "这会停止实盘自动化、撤销 BTCUSDT 挂单，并使用 reduceOnly 市价平掉当前 BTCUSDT 仓位。"
    : "这会停止模拟盘自动化并锁住再次启动，不会向 Binance 提交订单。";
}

async function refresh() {
  try {
    const response = await fetch(`/api/terminal/status?v=${Date.now()}`);
    if (!response.ok) throw new Error(`状态接口 ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("#healthDot").classList.remove("online");
    $("#healthText").textContent = "终端离线";
    showToast(error.message, true);
  }
}

$("#startButton").addEventListener("click", async () => { try { await control("start"); showToast(`${latest?.mode === "LIVE" ? "实盘" : "模拟盘"}自动化已启动`); } catch (e) { showToast(e.message, true); } });
$("#pauseButton").addEventListener("click", async () => { try { await control("pause"); showToast("自动化进程已暂停"); } catch (e) { showToast(e.message, true); } });
$("#onceButton").addEventListener("click", async () => { try { showToast("正在执行一次策略与仓位同步"); await control("run_once"); showToast("单次执行完成"); } catch (e) { showToast(e.message, true); } });
$("#simulationMode").addEventListener("click", async () => { try { await control("set_mode", { mode: "simulation" }); showToast("已切换到模拟盘"); } catch (e) { showToast(e.message, true); } });
$("#resetSimulation").addEventListener("click", async () => {
  const amount = Number($("#simulationInitialBalance").value);
  if (!Number.isFinite(amount) || amount < 1 || amount > 1_000_000_000) {
    showToast("初始金额必须在 1 至 1,000,000,000 USDT 之间", true);
    return;
  }
  if (!window.confirm(`确认以 ${amount.toLocaleString()} USDT 重置模拟账户？现有模拟记录会被清空。`)) return;
  try {
    await control("reset_simulation", { initial_balance: amount, confirm: "RESET_SIMULATION" });
    showToast(`模拟账户已重置为 ${amount.toLocaleString()} USDT`);
  } catch (e) { showToast(e.message, true); }
});
$("#liveMode").addEventListener("click", () => $("#liveModal").classList.remove("hidden"));
$("#cancelLive").addEventListener("click", () => $("#liveModal").classList.add("hidden"));
$("#confirmLive").addEventListener("click", async () => { try { await control("set_mode", { mode: "live", confirm: $("#liveConfirm").value }); $("#liveModal").classList.add("hidden"); $("#liveConfirm").value = ""; showToast("实盘模式已启用", true); } catch (e) { showToast(e.message, true); } });
$("#emergencyButton").addEventListener("click", async () => {
  if (latest?.execution.emergency) {
    try { await control("reset_emergency", { confirm: "RESET" }); showToast("急停锁已解除"); } catch (e) { showToast(e.message, true); }
  } else {
    $("#emergencyModal").classList.remove("hidden");
  }
});
$("#cancelEmergency").addEventListener("click", () => $("#emergencyModal").classList.add("hidden"));
$("#confirmEmergency").addEventListener("click", async () => { try { await control("emergency_stop", { confirm: $("#emergencyConfirm").value, reason: "manual terminal emergency stop" }); $("#emergencyModal").classList.add("hidden"); $("#emergencyConfirm").value = ""; showToast("急停已生效", true); } catch (e) { showToast(e.message, true); await refresh(); } });
$("#clearViewButton").addEventListener("click", () => { logViewCleared = true; $("#logStream").textContent = "视图已清空；刷新页面恢复日志。"; });
$$('.tab').forEach(tab => tab.addEventListener("click", () => { $$('.tab').forEach(item => item.classList.toggle("active", item === tab)); ["positions","orders","trades"].forEach(name => $(`#${name}Pane`).classList.toggle("hidden", name !== tab.dataset.tab)); }));
setInterval(() => $("#clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }), 1000);
setInterval(refresh, 3000);
refresh();
