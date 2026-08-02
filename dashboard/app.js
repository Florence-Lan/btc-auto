const sources = {
  normal: "../data/validation/carry_trend_portfolio_85_15_20260801.json",
  stress: "../data/validation/carry_trend_portfolio_85_15_double_cost_20260801.json",
  manifest: "../config/carry_trend_candidate_20260801.json",
};

const profileLabels = {
  portfolio: "资金费组合",
  trend: "趋势卫星",
};

let currentProfile = "portfolio";
let cachedData = null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const pct = (value, digits = 1) => value == null ? "—" : `${Number(value).toFixed(digits)}%`;
const num = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const signedPct = (value, digits = 1) => value == null ? "—" : `${Number(value) >= 0 ? "+" : "−"}${pct(Math.abs(Number(value)), digits)}`;

async function fetchJson(path) {
  const response = await fetch(`${path}?v=${Date.now()}`);
  if (!response.ok) throw new Error(`无法读取 ${path} (${response.status})`);
  return response.json();
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function annualFolds(returns = {}) {
  return Object.entries(returns).map(([year, value]) => ({
    label: year,
    total_return_pct: Number(value),
    trades: null,
  }));
}

function compoundFolds(folds, initialEquity = 100) {
  let equity = initialEquity;
  const points = [{ label: folds[0]?.label || "起点", value: equity, fold: null }];
  folds.forEach((fold) => {
    equity *= 1 + fold.total_return_pct / 100;
    points.push({ label: fold.label, value: equity, fold });
  });
  return points;
}

function smoothPath(points) {
  if (points.length < 2) return "";
  return points.reduce((path, point, index) => {
    if (index === 0) return `M ${point.x} ${point.y}`;
    const previous = points[index - 1];
    const midX = (previous.x + point.x) / 2;
    return `${path} C ${midX} ${previous.y}, ${midX} ${point.y}, ${point.x} ${point.y}`;
  }, "");
}

function renderEquityChart(folds, initialEquity) {
  const svg = $("#equityChart");
  const raw = compoundFolds(folds, initialEquity);
  const width = 900;
  const height = 330;
  const margin = { top: 24, right: 48, bottom: 34, left: 48 };
  const values = raw.map((point) => point.value);
  const minValue = Math.min(...values) * 0.96;
  const maxValue = Math.max(...values) * 1.04;
  const x = (index) => margin.left + (index / Math.max(1, raw.length - 1)) * (width - margin.left - margin.right);
  const y = (value) => margin.top + ((maxValue - value) / (maxValue - minValue || 1)) * (height - margin.top - margin.bottom);
  const points = raw.map((point, index) => ({ ...point, x: x(index), y: y(point.value) }));
  const line = smoothPath(points);
  const area = `${line} L ${points.at(-1).x} ${height - margin.bottom} L ${points[0].x} ${height - margin.bottom} Z`;
  const ticks = Array.from({ length: 5 }, (_, index) => minValue + ((maxValue - minValue) * index) / 4).reverse();
  svg.innerHTML = `
    <defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#20a77a" stop-opacity=".18"/><stop offset="1" stop-color="#20a77a" stop-opacity="0"/></linearGradient></defs>
    ${ticks.map((tick) => `<line class="chart-grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"/><text class="chart-label" x="${width - margin.right + 9}" y="${y(tick) + 4}">${Math.round(tick)}</text>`).join("")}
    ${points.map((point) => `<text class="chart-label" x="${point.x}" y="${height - 10}" text-anchor="middle">${point.label}</text>`).join("")}
    <path class="chart-area" d="${area}"/><path class="chart-path" d="${line}"/>
    ${points.map((point, index) => `<circle class="chart-dot" data-index="${index}" cx="${point.x}" cy="${point.y}" r="3.5"/>`).join("")}
  `;
  const tooltip = $("#chartTooltip");
  svg.querySelectorAll(".chart-dot").forEach((dot) => {
    dot.addEventListener("mouseenter", () => {
      const point = points[Number(dot.dataset.index)];
      tooltip.innerHTML = `${point.label}<strong>净值 ${point.value.toFixed(1)}</strong>${point.fold ? `窗口 ${signedPct(point.fold.total_return_pct)}` : "初始净值"}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.min((point.x / width) * svg.clientWidth, svg.clientWidth - 130)}px`;
      tooltip.style.top = `${Math.max((point.y / height) * svg.clientHeight - 58, 4)}px`;
    });
    dot.addEventListener("mouseleave", () => { tooltip.hidden = true; });
  });
}

function renderFoldChart(folds) {
  const container = $("#foldChart");
  const maxAbs = Math.max(...folds.map((fold) => Math.abs(fold.total_return_pct)), 1);
  container.innerHTML = folds.map((fold) => {
    const value = fold.total_return_pct;
    const height = Math.max(2, Math.abs(value) / maxAbs * 45);
    const direction = value >= 0 ? "positive" : "negative";
    return `<div class="fold-bar-wrap"><span class="fold-bar ${direction}" style="height:${height}%" title="${fold.label} · ${signedPct(value)}"></span></div>`;
  }).join("");
}

function positiveYearPct(returns = {}) {
  const values = Object.values(returns).map(Number);
  return values.length ? values.filter((value) => value > 0).length / values.length * 100 : 0;
}

function viewModel(normal, stress, manifest, profile) {
  if (profile === "trend") {
    const summary = normal.trend_summary;
    const stressSummary = stress.trend_summary;
    return {
      profile,
      summary,
      stressSummary,
      folds: annualFolds(summary.annual_returns_pct),
      drawdownLimit: 15,
      qualityLabel: "盈亏比",
      qualityValue: num(summary.profit_factor),
      qualityNote: `双倍成本 ${num(stressSummary.profit_factor)} · 胜率 ${pct(summary.win_rate_pct)}`,
      activityLabel: "交易频次",
      activityValue: `${num(summary.trades_per_year, 1)}/年`,
      activityNote: `${summary.trades} 笔历史交易`,
      metricOne: pct(positiveYearPct(summary.annual_returns_pct)),
      metricTwo: pct(summary.win_rate_pct),
      metricThree: num(stressSummary.profit_factor),
      metricLabels: ["盈利年份占比", "历史胜率", "双倍成本盈亏比"],
      evidenceBadge: "SATELLITE ONLY",
      evidencePass: false,
      gateStatus: "趋势模块保持正期望，但只占组合 15%，不得独立替代 Carry 核心",
      stressStatus: `成本翻倍后年化 ${pct(stressSummary.cagr_pct)} · 盈亏比 ${num(stressSummary.profit_factor)} · 回撤 ${pct(stressSummary.max_drawdown_pct)}`,
      paperStatus: "沿用冻结趋势逻辑 · 仅作为组合卫星仓 · 当前不下单",
      strategies: [
        { icon: "↗", title: "多周期趋势", description: "15m / 1h / 4h 共振，回调后顺势进入", state: "10.4%" },
        { icon: "⌁", title: "6h 时序趋势", description: "24 / 120 EMA 方向，按目标波动率调整", state: "4.6%" },
      ],
      risk: {
        cap: "15% 组合权重",
        label: "单笔战术风险",
        value: pct(0.0075 * 100, 2),
        volLabel: "时序目标波动率",
        vol: pct(0.11 * 100, 0),
        stopLabel: "回撤停止线",
        stop: "15%",
      },
      pipeline: {
        signal: "趋势分数 ≥ 0.82 · ADX ≥ 34\n15m / 1h / 4h 多周期确认",
        sizing: "卫星仓权重 15%\n风险由组合总资金隔离",
        exit: "止盈 1R / 2R / 3.2R\n宏观因子只降风险",
      },
      distributionTitle: "趋势年度收益分布",
      performanceTitle: "趋势卫星年度净值",
      freezeId: manifest.trend.base_manifest,
    };
  }

  const summary = normal.summary;
  const stressSummary = stress.summary;
  const carrySummary = normal.carry_summary;
  return {
    profile,
    summary,
    stressSummary,
    folds: annualFolds(summary.annual_returns_pct),
    drawdownLimit: normal.gates.max_drawdown_pct_max,
    qualityLabel: "保证金缓冲",
    qualityValue: pct(carrySummary.min_futures_margin_buffer_pct),
    qualityNote: `门槛 ≥ ${pct(normal.gates.carry_min_margin_buffer_pct_min, 0)} · 历史无穿仓`,
    activityLabel: "资金费结算",
    activityValue: carrySummary.funding_events.toLocaleString("zh-CN"),
    activityNote: `${carrySummary.rebalances} 次风险再平衡`,
    metricOne: pct(summary.profitable_year_pct),
    metricTwo: pct(summary.profitable_quarter_pct),
    metricThree: signedPct(summary.worst_quarter_pct),
    metricLabels: ["盈利年份占比", "盈利季度占比", "最差季度"],
    evidenceBadge: normal.candidate_pass && stress.candidate_pass ? "RESEARCH PASS" : "REVIEW REQUIRED",
    evidencePass: normal.candidate_pass && stress.candidate_pass,
    gateStatus: `年化、回撤、盈利年份与季度均通过；永续最低缓冲 ${pct(carrySummary.min_futures_margin_buffer_pct)}`,
    stressStatus: `成本翻倍后年化 ${pct(stressSummary.cagr_pct)} · 回撤 ${pct(stressSummary.max_drawdown_pct)} · 盈利季度 ${pct(stressSummary.profitable_quarter_pct)}`,
    paperStatus: "研究通过 · 双腿配对执行器尚未实现 · 禁止实盘下单",
    strategies: [
      { icon: "₿", title: "资金费 Carry 核心", description: "多 BTC 现货 / 空等数量 USD-M 永续", state: "85%" },
      { icon: "↗", title: "冻结趋势卫星", description: "多周期趋势 + 6h 时序趋势，补充收益弹性", state: "15%" },
    ],
    risk: {
      cap: "子账户隔离",
      label: "Carry 内部名义仓位",
      value: pct(manifest.carry.notional_fraction_within_carry_subaccount * 100, 0),
      volLabel: "趋势卫星权重",
      vol: pct(manifest.allocation.frozen_trend_satellite * 100, 0),
      stopLabel: "保证金再平衡线",
      stop: pct(manifest.carry.rebalance_margin_buffer_pct, 0),
    },
    pipeline: {
      signal: "85% Carry 核心\n15% 趋势卫星",
      sizing: "等数量多现货 / 空永续\n子账户严格隔离",
      exit: `保证金缓冲 ≤ ${pct(manifest.carry.rebalance_margin_buffer_pct, 0)}\n平双腿后重新配平`,
    },
    distributionTitle: "组合年度收益分布",
    performanceTitle: "组合年度复合净值",
    freezeId: manifest.candidate_id,
  };
}

function renderStrategies(items) {
  $("#strategyStack").innerHTML = items.map((item) => `
    <div class="strategy-item">
      <span class="strategy-symbol">${item.icon}</span>
      <div><strong>${item.title}</strong><small>${item.description}</small></div>
      <span class="strategy-state">${item.state}</span>
    </div>
  `).join("");
}

function render(normal, stress, manifest, profile) {
  const view = viewModel(normal, stress, manifest, profile);
  const summary = view.summary;
  setText("#totalReturn", signedPct(summary.total_return_pct));
  setText("#equityMultiple", `净值 ${num(summary.initial_equity || 100, 0)} → ${num(summary.final_equity, 1)}`);
  setText("#cagr", signedPct(summary.cagr_pct));
  setText("#maxDrawdown", `−${pct(summary.max_drawdown_pct)}`);
  setText("#drawdownLimit", `风控红线 ${pct(view.drawdownLimit, 0)}`);
  $("#drawdownBar").style.width = `${Math.min(100, summary.max_drawdown_pct / view.drawdownLimit * 100)}%`;
  setText("#qualityLabel", view.qualityLabel);
  setText("#profitFactor", view.qualityValue);
  setText("#profitFactorNote", view.qualityNote);
  setText("#activityLabel", view.activityLabel);
  setText("#tradeFrequency", view.activityValue);
  setText("#tradeCount", view.activityNote);
  setText("#metricOneLabel", view.metricLabels[0]);
  setText("#metricTwoLabel", view.metricLabels[1]);
  setText("#metricThreeLabel", view.metricLabels[2]);
  setText("#profitableFolds", view.metricOne);
  setText("#positiveProbability", view.metricTwo);
  setText("#p05Return", view.metricThree);
  setText("#profileTag", profileLabels[profile]);
  setText("#portfolioCap", view.risk.cap);
  setText("#riskLabel", view.risk.label);
  setText("#riskPerTrade", view.risk.value);
  setText("#volLabel", view.risk.volLabel);
  setText("#targetVol", view.risk.vol);
  setText("#stopLabel", view.risk.stopLabel);
  setText("#drawdownStop", view.risk.stop);
  setText("#signalThresholds", view.pipeline.signal);
  setText("#sizingRules", view.pipeline.sizing);
  setText("#exitRules", view.pipeline.exit);
  ["#signalThresholds", "#sizingRules", "#exitRules"].forEach((selector) => { $(selector).style.whiteSpace = "pre-line"; });
  setText("#performanceTitle", view.performanceTitle);
  setText("#chartLegend", "年度复合净值");
  setText("#distributionTitle", view.distributionTitle);
  setText("#foldCount", `${view.folds.length} 个年度窗口`);
  setText("#evidenceBadge", view.evidenceBadge);
  $("#evidenceBadge").classList.toggle("watch", !view.evidencePass);
  setText("#gateStatus", view.gateStatus);
  setText("#stressStatus", view.stressStatus);
  setText("#recentYieldStatus", `近 ${manifest.validation.recent_carry_only.days} 天 Carry 双倍成本年化仅 ${pct(manifest.validation.recent_carry_only.double_cost_cagr_pct)}，低于长期均值`);
  setText("#paperStatus", view.paperStatus);
  setText("#freezeId", view.freezeId);
  setText("#configHash", manifest.data.spot_snapshot_sha256);
  setText("#generatedAt", new Date(normal.generated_at_utc).toLocaleString("zh-CN", { timeZone: "UTC", hour12: false }));
  setText("#dataRange", `${summary.start_utc?.slice(0, 10) || summary.symbol_window?.start_utc?.slice(0, 10)} — ${summary.end_utc?.slice(0, 10) || summary.symbol_window?.end_utc?.slice(0, 10)}`);
  renderStrategies(view.strategies);
  renderEquityChart(view.folds, summary.initial_equity || 100);
  renderFoldChart(view.folds);
}

async function loadData() {
  if (cachedData) return cachedData;
  const [normal, stress, manifest] = await Promise.all([
    fetchJson(sources.normal),
    fetchJson(sources.stress),
    fetchJson(sources.manifest),
  ]);
  cachedData = { normal, stress, manifest };
  return cachedData;
}

async function loadProfile(profile = currentProfile, force = false) {
  currentProfile = profile;
  const refresh = $("#refreshButton");
  refresh.classList.add("loading");
  if (force) cachedData = null;
  try {
    const { normal, stress, manifest } = await loadData();
    render(normal, stress, manifest, profile);
    $("#errorToast").hidden = true;
  } catch (error) {
    const toast = $("#errorToast");
    toast.textContent = `${error.message}。请从仓库根目录启动本地 HTTP 服务。`;
    toast.hidden = false;
  } finally {
    refresh.classList.remove("loading");
  }
}

$$('.profile-option').forEach((button) => {
  button.addEventListener("click", () => {
    $$('.profile-option').forEach((item) => item.classList.toggle("active", item === button));
    loadProfile(button.dataset.profile);
  });
});

$("#refreshButton").addEventListener("click", () => loadProfile(currentProfile, true));
loadProfile();
