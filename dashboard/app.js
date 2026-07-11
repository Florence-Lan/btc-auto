const profiles = {
  controlled: {
    label: "风控优先",
    validation: "../data/validation/frozen_strategy_20260711.json",
    manifest: "../config/frozen_strategy_20260711.json",
    paper: "../data/paper_trading/frozen_portfolio_20260711_report.json",
  },
  active: {
    label: "高覆盖",
    validation: "../data/validation/frozen_strategy_active_20260711.json",
    manifest: "../config/frozen_strategy_active_20260711.json",
    paper: "../data/paper_trading/frozen_portfolio_active_20260711_report.json",
  },
  candidate: {
    label: "宏观影子候选",
    validation: "../data/validation/candidate_portfolio_20260711.json",
    manifest: "../config/frozen_strategy_active_20260711.json",
    candidate: "../config/shadow_candidate_macro_20260711.json",
    paper: "../data/paper_trading/macro_candidate_report.json",
  },
};

let currentProfile = "controlled";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const pct = (value, digits = 1) => `${Number(value).toFixed(digits)}%`;
const num = (value, digits = 2) => Number(value).toFixed(digits);
const shortDate = (value) => new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short" }).format(new Date(value));

async function fetchJson(path) {
  const response = await fetch(`${path}?v=${Date.now()}`);
  if (!response.ok) throw new Error(`无法读取 ${path} (${response.status})`);
  return response.json();
}

function setText(selector, value) { $(selector).textContent = value; }

function compoundFolds(folds, initialEquity = 100) {
  let equity = initialEquity;
  const points = [{ label: shortDate(folds[0].window.start_utc), value: equity, fold: null }];
  folds.forEach((fold) => {
    equity *= 1 + fold.total_return_pct / 100;
    points.push({ label: shortDate(fold.window.end_utc), value: equity, fold });
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
  const width = 900, height = 330, margin = { top: 24, right: 48, bottom: 34, left: 48 };
  const values = raw.map((point) => point.value);
  const minValue = Math.min(...values) * .92;
  const maxValue = Math.max(...values) * 1.06;
  const x = (index) => margin.left + (index / (raw.length - 1)) * (width - margin.left - margin.right);
  const y = (value) => margin.top + ((maxValue - value) / (maxValue - minValue || 1)) * (height - margin.top - margin.bottom);
  const points = raw.map((point, index) => ({ ...point, x: x(index), y: y(point.value) }));
  const line = smoothPath(points);
  const area = `${line} L ${points.at(-1).x} ${height - margin.bottom} L ${points[0].x} ${height - margin.bottom} Z`;
  const ticks = Array.from({ length: 5 }, (_, index) => minValue + ((maxValue - minValue) * index) / 4).reverse();
  const labelStep = Math.max(1, Math.ceil((raw.length - 1) / 6));
  svg.innerHTML = `
    <defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#20a77a" stop-opacity=".18"/><stop offset="1" stop-color="#20a77a" stop-opacity="0"/></linearGradient></defs>
    ${ticks.map((tick) => `<line class="chart-grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"/><text class="chart-label" x="${width - margin.right + 9}" y="${y(tick) + 4}">${Math.round(tick)}</text>`).join("")}
    ${points.filter((_, index) => index % labelStep === 0 || index === points.length - 1).map((point) => `<text class="chart-label" x="${point.x}" y="${height - 10}" text-anchor="middle">${point.label}</text>`).join("")}
    <path class="chart-area" d="${area}"/><path class="chart-path" d="${line}"/>
    ${points.map((point, index) => `<circle class="chart-dot" data-index="${index}" cx="${point.x}" cy="${point.y}" r="3.5"/>`).join("")}
  `;
  const tooltip = $("#chartTooltip");
  svg.querySelectorAll(".chart-dot").forEach((dot) => {
    dot.addEventListener("mouseenter", () => {
      const point = points[Number(dot.dataset.index)];
      tooltip.innerHTML = `${point.label}<strong>净值 ${point.value.toFixed(1)}</strong>${point.fold ? `季度 ${pct(point.fold.total_return_pct)}` : "初始净值"}`;
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
  container.innerHTML = folds.map((fold, index) => {
    const value = fold.total_return_pct;
    const height = Math.max(2, Math.abs(value) / maxAbs * 45);
    const direction = value >= 0 ? "positive" : "negative";
    return `<div class="fold-bar-wrap"><span class="fold-bar ${direction}" style="height:${height}%" title="${shortDate(fold.window.start_utc)} — ${pct(value)} · ${fold.trades} 笔"></span></div>`;
  }).join("");
}

function normalizeCandidate(data, manifest, candidate) {
  const summary = data.normal.summary;
  const start = new Date(data.market_snapshot_metadata.start_utc).getTime() + 365 * 86400000;
  const folds = data.normal.quarterly.returns_pct.map((value, index) => ({
    window: {
      start_utc: new Date(start + index * 90 * 86400000).toISOString(),
      end_utc: new Date(start + (index + 1) * 90 * 86400000).toISOString(),
    },
    total_return_pct: value,
    max_drawdown_pct: null,
    profit_factor: null,
    trades: 0,
  }));
  return {
    data: {
      ...data,
      aggregate: {
        ...summary,
        initial_equity: 100,
        profitable_fold_pct: data.normal.quarterly.profitable_pct,
      },
      bootstrap: data.normal.bootstrap,
      double_cost_stress: data.double_cost.summary,
      fold_count: folds.length,
      folds,
      evidence_pass: data.candidate_pass,
      freeze_id: candidate.candidate_id,
      config_sha256: candidate.macro.snapshot_sha256,
      snapshot_metadata: data.market_snapshot_metadata,
    },
    manifest: {
      ...manifest,
      config: {
        ...manifest.config,
        strategy_modes: candidate.strategy_modes,
        max_drawdown_stop_pct: candidate.drawdown_policy.hard_stop_pct,
      },
    },
  };
}

const strategyInfo = {
  trend: { icon: "↗", title: "多周期趋势", description: "15m / 1h / 4h 共振，回调后顺势进入" },
  range: { icon: "↔", title: "震荡回归", description: "布林带 + RSI，在低趋势强度区间捕捉回归" },
  timeseries_trend: { icon: "⌁", title: "6h 时序趋势", description: "24 / 120 EMA 方向，按目标波动率调整仓位" },
};

function renderStrategies(config) {
  const modes = config.strategy_modes || [];
  $("#strategyStack").innerHTML = modes.map((mode) => {
    const item = strategyInfo[mode] || { icon: "◇", title: mode, description: "策略模块" };
    return `<div class="strategy-item"><span class="strategy-symbol">${item.icon}</span><div><strong>${item.title}</strong><small>${item.description}</small></div><span class="strategy-state">ACTIVE</span></div>`;
  }).join("");
}

function render(data, manifest, paper, profile) {
  const aggregate = data.aggregate;
  const config = manifest.config;
  const gates = data.gates;
  setText("#totalReturn", `+${pct(aggregate.total_return_pct)}`);
  setText("#equityMultiple", `净值 100 → ${(aggregate.initial_equity * (1 + aggregate.total_return_pct / 100)).toFixed(1)}`);
  setText("#cagr", `+${pct(aggregate.cagr_pct)}`);
  setText("#maxDrawdown", `−${pct(aggregate.max_drawdown_pct)}`);
  setText("#drawdownLimit", `风控红线 ${pct(gates.max_drawdown_pct_max, 0)}`);
  $("#drawdownBar").style.width = `${Math.min(100, aggregate.max_drawdown_pct / gates.max_drawdown_pct_max * 100)}%`;
  setText("#profitFactor", num(aggregate.profit_factor));
  setText("#profitFactorNote", `门槛 ≥ ${num(gates.profit_factor_min, 1)} · 每亏 1 对应盈利 ${num(aggregate.profit_factor)}`);
  setText("#tradeFrequency", `${num(aggregate.trades_per_year, 1)}/年`);
  setText("#tradeCount", `${aggregate.trades} 笔历史交易`);
  setText("#profitableFolds", pct(aggregate.profitable_fold_pct, 1));
  setText("#positiveProbability", pct(data.bootstrap.positive_probability_pct, 2));
  setText("#p05Return", `+${pct(data.bootstrap.annualized_p05_pct, 1)}`);
  setText("#profileTag", profiles[profile].label);
  setText("#portfolioCap", `${num(config.portfolio_leverage_cap, 0)}× 总敞口上限`);
  setText("#riskPerTrade", pct(config.risk_per_trade * 100, 2));
  setText("#targetVol", pct(config.timeseries_target_vol * 100, 0));
  setText("#drawdownStop", pct(config.max_drawdown_stop_pct, 0));
  setText("#signalThresholds", `趋势分数 ≥ ${num(config.trend_min_signal_score, 2)} · ADX ≥ ${num(config.trend_min_adx, 0)}\n${config.trend_confirm_timeframes.join(" / ")} 多周期确认`);
  $("#signalThresholds").style.whiteSpace = "pre-line";
  setText("#sizingRules", `单笔风险 ${pct(config.risk_per_trade * 100, 2)}\n组合总敞口 ≤ ${num(config.portfolio_leverage_cap, 0)}×`);
  $("#sizingRules").style.whiteSpace = "pre-line";
  setText("#exitRules", `止盈 ${config.trend_tp1_rr}R / ${config.trend_tp2_rr}R / ${config.trend_tp3_rr}R\n回撤 ${pct(config.max_drawdown_stop_pct, 0)} 自动停止`);
  $("#exitRules").style.whiteSpace = "pre-line";
  setText("#foldCount", `${data.fold_count} 个独立季度窗口`);
  const shadowCandidate = profile === "candidate";
  setText("#evidenceBadge", shadowCandidate ? "SHADOW ONLY" : (data.evidence_pass ? "EVIDENCE PASS" : "REVIEW REQUIRED"));
  setText("#gateStatus", shadowCandidate
    ? `收益与风险门槛通过；盈利季度 ${pct(aggregate.profitable_fold_pct)} 未达 60%，禁止实盘`
    : (data.evidence_pass ? "收益、回撤、盈亏比、频次与稳定性门槛均通过" : "部分历史验收门槛未通过，请检查验证报告"));
  const stress = data.double_cost_stress;
  setText("#stressStatus", `成本翻倍后年化 ${pct(stress.cagr_pct)} · 盈亏比 ${num(stress.profit_factor)} · 回撤 ${pct(stress.max_drawdown_pct)}`);
  setText("#paperStatus", `始于 ${paper.paper_inception_utc.slice(0, 10)} · 当前 ${paper.summary.trades} 笔 · ${paper.places_orders ? "会下单" : "不下单"}`);
  setText("#freezeId", data.freeze_id);
  setText("#configHash", data.config_sha256);
  setText("#generatedAt", new Date(data.generated_at_utc).toLocaleString("zh-CN", { timeZone: "UTC", hour12: false }));
  setText("#dataRange", `${shortDate(data.snapshot_metadata.start_utc)} — ${shortDate(data.snapshot_metadata.end_utc)}`);
  renderStrategies(config);
  renderEquityChart(data.folds, aggregate.initial_equity);
  renderFoldChart(data.folds);
}

async function loadProfile(profile = currentProfile) {
  currentProfile = profile;
  const refresh = $("#refreshButton");
  refresh.classList.add("loading");
  try {
    const paths = profiles[profile];
    const [rawValidation, rawManifest, paper, candidate] = await Promise.all([
      fetchJson(paths.validation), fetchJson(paths.manifest), fetchJson(paths.paper),
      paths.candidate ? fetchJson(paths.candidate) : Promise.resolve(null),
    ]);
    const normalized = candidate
      ? normalizeCandidate(rawValidation, rawManifest, candidate)
      : { data: rawValidation, manifest: rawManifest };
    const validation = normalized.data;
    const manifest = normalized.manifest;
    render(validation, manifest, paper, profile);
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
$("#refreshButton").addEventListener("click", () => loadProfile());
loadProfile();
