const numeric = value => typeof value === "number" && Number.isFinite(value) ? value : null;
const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
const number = (value, digits = 2) => numeric(value) == null ? "—" : value.toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits});
const percent = value => numeric(value) == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
const tone = value => numeric(value) == null || value === 0 ? "" : value > 0 ? "positive" : "negative";
const count = value => numeric(value) == null ? "—" : number(value, 0);

export function beijingTime(value) {
  if (!value || Number.isNaN(new Date(value).getTime())) return "—";
  return new Date(value).toLocaleString("zh-CN", {timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false});
}

export function selectedSummary(stock, cost, family = "liquidity", window = "full") {
  return stock.scenarios?.[family]?.[String(cost)]?.[window] || null;
}

export function researchMarkup(data, cost = 1) {
  const cards = [], rows = [], findings = [];
  for (const stock of data.stocks || []) {
    const full = selectedSummary(stock, cost);
    const recent = selectedSummary(stock, cost, "liquidity", "recent30d");
    const baseline = selectedSummary(stock, cost, "baseline");
    const profile = stock.profile || {};
    const rule = `${profile.signal_family === "breakout" ? "四小时突破" : "四小时均线交叉"} · EMA ${[profile.ema_fast, profile.ema_mid, profile.ema_slow].map(value => numeric(value) == null ? "—" : value).join(" / ")}`;
    const loss = numeric(full?.estimated_close_return_pct) != null && full.estimated_close_return_pct < 0;
    const review = stock.mechanism_review;
    const failedReview = numeric(review?.trial_count) > 0 && review?.passed_count === 0;
    const status = !full ? "结果待补齐" : failedReview ? "本轮研究未通过" : loss ? "收益未通过" : "待成交验证";
    cards.push(`<article class="stock-card panel">
      <div class="stock-card-heading"><div class="stock-identity"><span class="stock-monogram">${escape(stock.symbol?.replace("USDT", "").slice(0, 2))}</span><div><h3>${escape(stock.name)}</h3><span>${escape(stock.symbol)}</span></div></div><span class="stock-status ${loss || failedReview ? "stock-status-failed" : ""}">${status}</span></div>
      <div class="stock-return"><span>独立资金收益 · ${cost === 2 ? "双倍" : "正常"}成本</span><strong class="${tone(full?.estimated_close_return_pct)}">${percent(full?.estimated_close_return_pct)}</strong><small>历史回放，含期末估计平仓损益</small></div>
      <div class="stock-metrics"><div><span>最大采样回撤</span><strong>${numeric(full?.max_sampled_drawdown_pct) == null ? "—" : `${number(full.max_sampled_drawdown_pct)}%`}</strong></div><div><span>已平仓交易</span><strong>${count(full?.closed_trades)} <small>笔</small></strong></div><div><span>净保证金 ≥120%</span><strong>${count(full?.target_trades_net_at_least_120pct_margin)} <small>笔</small></strong></div><div><span>近 30 天收益</span><strong class="${tone(recent?.estimated_close_return_pct)}">${percent(recent?.estimated_close_return_pct)}</strong></div></div>
      <div class="stock-accounting"><span>已平仓资金收益 <b class="${tone(full?.closed_return_pct)}">${percent(full?.closed_return_pct)}</b></span><span>期末估计平仓损益 <b>${number(full?.estimated_open_close_net_pnl)} USDT</b></span></div>
      <div class="stock-card-footer"><strong>${escape(rule)}</strong><span>研究资本 ${number(full?.initial_equity)} USDT</span><span>北京时间 ${beijingTime(full?.start_utc)} — ${beijingTime(full?.end_utc_exclusive)}</span></div>
    </article>`);
    rows.push(`<tr><td><strong>${escape(stock.name)}</strong><small class="stock-table-symbol">${escape(stock.symbol)}</small></td><td class="${tone(baseline?.estimated_close_return_pct)}">${percent(baseline?.estimated_close_return_pct)}</td><td class="${tone(full?.estimated_close_return_pct)}">${percent(full?.estimated_close_return_pct)}</td><td class="${tone(recent?.estimated_close_return_pct)}">${percent(recent?.estimated_close_return_pct)}</td><td>${count(baseline?.zero_volume_entries)} / ${count(baseline?.closed_trades)}</td></tr>`);
    const lines = [];
    if (review?.finding) lines.push(review.finding);
    if (numeric(baseline?.zero_volume_entries) != null && baseline.zero_volume_entries > 0) lines.push(`原策略 ${count(baseline.zero_volume_entries)} / ${count(baseline.closed_trades)} 笔开仓所在五分钟线报告成交量为零。`);
    if (loss) lines.push("成交量限制后区间收益为负，需要继续研究信号与执行行为。");
    if (numeric(full?.zero_volume_entries) != null && full.zero_volume_entries > 0) lines.push(`限量后仍有 ${count(full.zero_volume_entries)} 笔零量开仓；此前成交量不能保证随后可成交。`);
    if (numeric(full?.closed_trades) != null && full.closed_trades < 10) lines.push(`仅 ${count(full.closed_trades)} 笔已平仓，样本仍少。`);
    if (numeric(baseline?.return_without_best_trade_pct) != null && baseline.return_without_best_trade_pct < 0) lines.push("原策略去掉最大一笔盈利后，已平仓收益转负。");
    if (!full) lines.push("该标的研究结果待补齐。");
    findings.push(`<div><h3>${escape(stock.name)} <span>${escape(stock.symbol)}</span></h3><p>${escape(lines.join(" ") || "实际盘口、部分成交与止损执行仍待验证。")}</p></div>`);
  }
  return {cards: cards.join(""), rows: rows.join(""), findings: findings.join("")};
}

export function createResearchView(root, fetcher = globalThis.fetch) {
  const $ = selector => root.querySelector(selector);
  const $$ = selector => [...root.querySelectorAll(selector)];
  let data = null, cost = 1, pending = false;
  const viewWindow = root.defaultView;
  function selectCost(value) {
    cost = Number(value) === 2 ? 2 : 1;
    $$('[data-research-cost]').forEach(button => {
      const selected = Number(button.dataset.researchCost) === cost;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    if (!data) return;
    const markup = researchMarkup(data, cost);
    $("#stockCards").innerHTML = markup.cards;
    $("#stockComparisonBody").innerHTML = markup.rows;
    $("#stockFindings").innerHTML = markup.findings;
    $("#stockCostLabel").textContent = cost === 2 ? "双倍手续费与滑点" : "正常成本";
    const assumptions = data.assumptions || {};
    $("#stockRuleStrip").innerHTML = [
      ["逐仓模型杠杆", `${number(assumptions.leverage, 0)}×`],
      ["单笔保证金净目标", `${number(assumptions.target_margin_return_pct, 0)}%`],
      ["每笔子账户风险", `${number(assumptions.risk_per_bucket_trade_pct)}%`],
      ["单边手续费假设", `${number(numeric(assumptions.taker_fee_pct) == null ? null : assumptions.taker_fee_pct * cost, 2)}%`],
      ["成交滑点假设", `${number(numeric(assumptions.slippage_pct) == null ? null : assumptions.slippage_pct * cost, 2)}%`],
    ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  }
  async function refresh() {
    if (pending) return;
    pending = true;
    $("#retryStockResearch").disabled = true;
    try {
      const response = await fetcher(`/api/terminal/research?v=${Date.now()}`);
      if (!response.ok) throw new Error("股票研究接口暂时无法连接");
      const result = await response.json();
      if (result.status !== "available") throw new Error(result.message || "股票研究结果暂不可用");
      data = result;
      $("#stockResearchContent").classList.remove("hidden");
      $("#stockResearchUnavailable").classList.add("hidden");
      $("#stockDataCutoff").textContent = beijingTime(data.data_end_utc_exclusive);
      const reviewedAt = data.mechanism_reviewed_at_utc || data.reviewed_at_utc;
      $("#stockResearchReviewed").textContent = reviewedAt ? `研究更新 ${beijingTime(reviewedAt)}` : "研究更新时间未记录";
      const rule = numeric(data.liquidity_rule);
      $("#stockLiquidityRule").textContent = rule == null ? "此前已收盘五分钟成交量限制；比例未记录" : `仓位不超过此前已收盘五分钟成交量的 ${(rule * 100).toFixed(0)}%`;
      $("#stockForwardState").textContent = data.forward_start_utc ? `前瞻记录起于 ${beijingTime(data.forward_start_utc)}` : data.forward_status === "prepared_not_started" ? "尚未启动，历史数据不计入前瞻样本" : "尚未取得前瞻运行记录";
      $("#stockResearchContext").textContent = data.forward_start_utc ? "历史回放 · 前瞻记录需独立评估" : data.forward_status === "prepared_not_started" ? "历史回放，前瞻观察尚未启动" : "历史回放 · 前瞻状态未取得";
      try {
        const forwardResponse = await fetcher(`/data/parallel_simulation/btc_memory_stocks_latest_1000_each_20261007/status.json?v=${Date.now()}`);
        if (forwardResponse.ok) {
          const forward = await forwardResponse.json();
          if (forward?.mode !== "SIMULATION" || forward.places_orders !== false
              || typeof forward.running !== "boolean"
              || !["mu", "sndk", "skhynix"].every(id => forward.accounts?.[id])
              || !Number.isFinite(Date.parse(forward.started_at_utc))
              || !Number.isFinite(Date.parse(forward.updated_at_utc))) {
            throw new Error("股票账户状态格式无效");
          }
          const age = (Date.now() - new Date(forward.updated_at_utc).getTime()) / 1000;
          const active = forward.running && age >= -5 && age < 120;
          $("#stockForwardState").textContent = `股票账户${active ? "运行中" : "心跳停止或过期"}；起于北京时间 ${beijingTime(forward.started_at_utc)}，当前账户见本页上方。`;
          $("#stockResearchContext").textContent = "历史回放 · 已开启独立股票前瞻模拟";
        }
      } catch (_) { /* Historical view stays available when forward status is unavailable. */ }
      $("#stockCoverageNote").textContent = data.coverage?.trimmed ? `因公共数据缺口，完整区间从北京时间 ${beijingTime(data.coverage.common_cutoff_utc)} 之后重新预热；较晚上市标的从自身历史起点开始。` : "各标的从其有效历史起点开始预热；以上为回顾性研究结果。";
      const link = $("#stockReportLink");
      const safeLink = typeof data.report_url === "string" && /^\/docs\/[a-zA-Z0-9_./-]+\.md$/.test(data.report_url);
      link.classList.toggle("hidden", !safeLink);
      if (safeLink) link.setAttribute("href", data.report_url);
      selectCost(cost);
    } catch (error) {
      data = null;
      $("#stockResearchContent").classList.add("hidden");
      $("#stockResearchUnavailable").classList.remove("hidden");
      $("#stockReportLink").classList.add("hidden");
      $("#stockDataCutoff").textContent = "—";
      $("#stockResearchContext").textContent = "历史回放 · 研究数据暂不可用";
      $("#stockResearchError").textContent = error.message || "股票研究结果暂不可用";
    } finally {
      $("#stockResearchLoading").classList.add("hidden");
      $("#retryStockResearch").disabled = false;
      pending = false;
    }
  }
  $$('[data-research-cost]').forEach(button => button.addEventListener("click", () => selectCost(button.dataset.researchCost)));
  $("#retryStockResearch").addEventListener("click", refresh);
  const ready = refresh();
  const interval = viewWindow?.setInterval(refresh, 60_000);
  return {selectCost, refresh, ready, destroy: () => viewWindow?.clearInterval(interval)};
}
