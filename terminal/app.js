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

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3200);
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
  const running = Boolean(data.execution.runtime.running);
  const emergency = data.execution.emergency;
  $("#healthDot").classList.toggle("online", running && !emergency);
  $("#healthText").textContent = emergency ? "急停锁定" : (running ? "机器人运行中" : "机器人已暂停");
  $("#markPrice").textContent = data.market.mark_price ? `$${fmt(data.market.mark_price, 1)}` : "—";
  $("#marketTime").textContent = data.market.data_time_ms ? new Date(data.market.data_time_ms).toLocaleString("zh-CN") : "等待行情";
  $("#candidateId").textContent = data.strategy.candidate_id || "宏观影子候选";
  $("#runtimeMeta").textContent = running ? `PID ${data.execution.runtime.pid} · 自动轮询` : "进程未运行";
  $("#exchangeStatus").textContent = data.exchange.connected
    ? `Binance ${data.exchange.environment.toUpperCase()} 只读已连接`
    : `Binance ${data.exchange.environment.toUpperCase()} 未连接`;
  $("#walletBalance").textContent = fmt(data.account.wallet_balance);
  $("#unrealizedPnl").textContent = fmt(data.account.unrealized_pnl);
  $("#realizedReturn").textContent = pct(data.account.realized_return_pct);
  $("#drawdown").textContent = pct(data.risk.drawdown_pct);
  $("#drawdownBar").style.width = `${Math.min(100, data.risk.drawdown_pct / data.risk.hard_limit_pct * 100)}%`;
  $("#macroMultiplier").textContent = `${fmt(data.risk.macro_current.risk_multiplier, 2)}×`;
  const coverage = data.risk.macro.factor_coverage_pct || {};
  $("#macroCoverage").textContent = Object.keys(coverage).length ? `${Object.keys(coverage).length} 组因子在线` : "等待产生信号";
  $("#observations").textContent = data.strategy.observations;
  $("#heartbeat").textContent = `心跳 ${timeAgo(data.execution.heartbeat_age_seconds)}`;
  $("#strategyStatus").textContent = emergency ? "EMERGENCY" : (running ? "RUNNING" : "PAUSED");
  $("#strategyList").innerHTML = data.strategy.strategy_modes.map(mode => `<div class="strategy-item"><span>${mode === "trend" ? "多周期趋势" : mode}</span><strong>ACTIVE</strong></div>`).join("");
  $("#shadowEligible").textContent = data.validation.shadow_eligible ? "通过" : "未通过";
  $("#shadowEligible").className = data.validation.shadow_eligible ? "positive" : "negative";
  $("#quarterlyRate").textContent = pct(data.validation.quarterly.profitable_pct, 1);
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
    { render: row => row.exit_time_utc?.slice(0, 19).replace("T", " ") || "—" }, { render: row => row.strategy || "—" }, { render: row => row.side },
    { render: row => fmt(row.entry_price, 1) }, { render: row => fmt(row.avg_exit_price, 1) }, { render: row => fmt(row.net_pnl), className: row => Number(row.net_pnl) >= 0 ? "positive" : "negative" },
  ], "暂无成交记录");
  const reportPoints = data.equity_curve || [];
  renderChart(reportPoints);
  if (!logViewCleared) {
    const lines = [...data.logs, ...data.errors.map(line => `[ERROR] ${line}`)];
    $("#logStream").textContent = lines.join("\n") || "暂无运行日志";
    $("#logStream").scrollTop = $("#logStream").scrollHeight;
  }
  $("#startButton").disabled = running || Boolean(emergency);
  $("#pauseButton").disabled = !running;
  $("#onceButton").disabled = Boolean(emergency);
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

$("#startButton").addEventListener("click", async () => { try { await control("start"); showToast("自动化影子进程已启动"); } catch (e) { showToast(e.message, true); } });
$("#pauseButton").addEventListener("click", async () => { try { await control("pause"); showToast("自动化进程已暂停"); } catch (e) { showToast(e.message, true); } });
$("#onceButton").addEventListener("click", async () => { try { showToast("正在执行一次策略循环"); await control("run_once"); showToast("单次执行完成"); } catch (e) { showToast(e.message, true); } });
$("#emergencyButton").addEventListener("click", () => $("#emergencyModal").classList.remove("hidden"));
$("#cancelEmergency").addEventListener("click", () => $("#emergencyModal").classList.add("hidden"));
$("#confirmEmergency").addEventListener("click", async () => { try { await control("emergency_stop", { confirm: $("#emergencyConfirm").value, reason: "manual terminal emergency stop" }); $("#emergencyModal").classList.add("hidden"); showToast("急停已生效", true); } catch (e) { showToast(e.message, true); } });
$("#clearViewButton").addEventListener("click", () => { logViewCleared = true; $("#logStream").textContent = "视图已清空；刷新页面恢复日志。"; });
$$('.tab').forEach(tab => tab.addEventListener("click", () => { $$('.tab').forEach(item => item.classList.toggle("active", item === tab)); ["positions","orders","trades"].forEach(name => $(`#${name}Pane`).classList.toggle("hidden", name !== tab.dataset.tab)); }));
setInterval(() => $("#clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }), 1000);
setInterval(refresh, 3000);
refresh();
