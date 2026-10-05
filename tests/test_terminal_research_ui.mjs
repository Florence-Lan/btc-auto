import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { beijingTime, researchMarkup, createResearchView } from "../terminal/research.js";

const html = readFileSync(new URL("../terminal/index.html", import.meta.url), "utf8");
const result = JSON.parse(readFileSync(new URL("../data/research/stock_swing_liquidity_20261004/results.json", import.meta.url), "utf8"));
function data() {
  return {
    status: "available", places_orders: false, forward_status: "prepared_not_started",
    data_end_utc_exclusive: "2026-10-04T12:00:00Z", liquidity_rule: .1,
    assumptions: {leverage: 10, target_margin_return_pct: 120, risk_per_bucket_trade_pct: .5, taker_fee_pct: .1, slippage_pct: .02},
    report_url: "/docs/stock_swing_followup_20261004.md",
    stocks: Object.entries({MUUSDT: "美光", SNDKUSDT: "闪迪", SKHYNIXUSDT: "海力士"}).map(([symbol, name]) => {
      const scenarios = {};
      for (const [family, suffix] of [["baseline", ""], ["liquidity", "_prior5m_volume10pct"]]) {
        scenarios[family] = {};
        for (const cost of [1, 2]) {
          scenarios[family][cost] = {};
          for (const window of ["full", "recent30d"]) {
            const summary = result.runs[symbol][`${window}${suffix}_cost${cost}`];
            scenarios[family][cost][window] = {...summary,
              closed_return_pct: summary.net_closed_pnl / summary.initial_equity * 100,
              zero_volume_entries: summary.execution_volume_diagnostics?.entry_bars_zero_reported_volume};
          }
        }
      }
      return {symbol, name, profile: result.profile.symbol_profiles[symbol], scenarios};
    })
  };
}

class Element {
  constructor(id, classes = "", dataset = {}) {
    this.id = id; this.dataset = dataset; this.attributes = {}; this.events = {};
    this.innerHTML = ""; this.textContent = "";
    this.classes = new Set(classes.split(" ").filter(Boolean));
    this.classList = {
      add: name => this.classes.add(name), remove: name => this.classes.delete(name),
      contains: name => this.classes.has(name),
      toggle: (name, state) => state ? this.classes.add(name) : this.classes.delete(name)
    };
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, fn) { this.events[name] = fn; }
  focus() { this.focused = true; }
}
function root(hash = "") {
  const elements = {};
  for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const classes = match[0].match(/class="([^"]*)"/)?.[1] || "";
    elements[match[1]] = new Element(match[1], classes);
  }
  const tabs = [elements.btcViewTab, elements.stockViewTab];
  tabs[0].dataset.workspace = "btc"; tabs[1].dataset.workspace = "stocks";
  const costs = [1, 2].map(n => new Element(`cost${n}`, n === 1 ? "active" : "", {researchCost: String(n)}));
  return {
    elements, tabs, costs,
    querySelector(selector) { assert.ok(elements[selector.slice(1)], `Missing HTML binding: ${selector}`); return elements[selector.slice(1)]; },
    querySelectorAll(selector) { return selector === "[data-workspace]" ? tabs : selector === "[data-research-cost]" ? costs : []; },
    defaultView: {
      location: {hash}, history: {replaceState(_state, _title, next) { this.hash = next; }},
      addEventListener() {}, setInterval() { return 1; }, clearInterval() {}
    }
  };
}

test("cards and comparison use the same cost scenario and independent ledgers", () => {
  const normal = researchMarkup(data(), 1), double = researchMarkup(data(), 2);
  assert.equal((normal.cards.match(/class="stock-card panel"/g) || []).length, 3);
  for (const value of ["+0.73%", "-0.61%", "+1.58%", "收益未通过", "1.81 USDT"]) assert.ok(normal.cards.includes(value), value);
  for (const value of ["+0.68%", "-0.67%", "+1.43%", "1.71 USDT"]) assert.ok(double.cards.includes(value), value);
  assert.ok(normal.rows.includes("+13.84%"));
  assert.ok(double.rows.includes("+12.06%"));
  assert.ok(normal.findings.includes("此前成交量不能保证随后可成交"));
});

test("missing returns display a dash rather than a fabricated zero", () => {
  const value = data();
  value.stocks = [value.stocks[0]];
  value.stocks[0].scenarios.liquidity[1].full = null;
  const markup = researchMarkup(value);
  assert.ok(markup.cards.includes("结果待补齐"));
  assert.ok(!markup.cards.includes("0.00%"));
  value.stocks[0].scenarios.liquidity[1].full = {estimated_close_return_pct: 0};
  assert.ok(researchMarkup(value).cards.includes("0.00%"));
});

test("artifact text is escaped in generated HTML", () => {
  const value = data();
  value.stocks[0].name = '<img src=x onerror="alert(1)">';
  const markup = researchMarkup(value);
  assert.ok(markup.cards.includes("&lt;img"));
  assert.ok(!markup.cards.includes("<img"));
  assert.ok(!markup.rows.includes("<img"));
});

test("failed mechanism research labels a positive return as unqualified and escapes findings", () => {
  const value = data();
  value.stocks[0].mechanism_review = {trial_count: 8, passed_count: 0, finding: '<script>alert(1)</script> 0 / 8 通过'};
  const markup = researchMarkup(value);
  assert.ok(markup.cards.includes("+0.73%"));
  assert.ok(markup.cards.includes("本轮研究未通过"));
  assert.ok(markup.findings.includes("&lt;script&gt;"));
  assert.ok(!markup.findings.includes("<script>"));
});

test("timestamps explicitly use Beijing time", () => {
  assert.match(beijingTime("2026-10-04T12:00:00Z"), /2026\/10\/04 20:00/);
  assert.equal(beijingTime("invalid"), "—");
  assert.equal(beijingTime(null), "—");
});

test("deep links and tab keyboard navigation show the appropriate controls", async () => {
  const page = root("#stocks"), requests = [];
  const view = createResearchView(page, async url => {requests.push(url); return {ok: true, json: async () => data()};});
  await view.ready;
  assert.ok(page.elements.btcWorkspace.classList.contains("hidden"));
  assert.ok(!page.elements.stockWorkspace.classList.contains("hidden"));
  assert.equal(page.elements.stockViewTab.attributes["aria-selected"], "true");
  assert.equal(page.elements.stockViewTab.tabIndex, 0);
  assert.match(page.elements.stockDataCutoff.textContent, /20:00/);
  page.elements.stockViewTab.events.keydown({key: "Home", preventDefault() {}});
  assert.ok(!page.elements.btcWorkspace.classList.contains("hidden"));
  assert.ok(page.elements.stockWorkspace.classList.contains("hidden"));
  assert.equal(page.defaultView.history.hash, "#btc");
  assert.equal(page.elements.btcViewTab.focused, true);
  assert.equal(requests.length, 2);
  assert.match(requests[0], /^\/api\/terminal\/research\?/);
  assert.match(requests[1], /^\/data\/parallel_simulation\/[^/]+\/status\.json\?/);
  view.destroy();
});

test("cost buttons update cards, comparisons, and fee assumptions without execution calls", async () => {
  const page = root(); let requests = 0;
  const view = createResearchView(page, async () => { requests++; return {ok: true, json: async () => data()}; });
  await view.ready;
  page.costs[1].events.click();
  assert.ok(page.elements.stockCards.innerHTML.includes("-0.67%"));
  assert.ok(page.elements.stockComparisonBody.innerHTML.includes("+12.06%"));
  assert.ok(page.elements.stockRuleStrip.innerHTML.includes("0.20%"));
  assert.equal(page.costs[1].attributes["aria-pressed"], "true");
  assert.equal(page.costs[0].attributes["aria-pressed"], "false");
  assert.equal(requests, 2);
  view.destroy();
});

test("failed refresh hides stale metrics and supports recovery", async () => {
  const page = root(); let failure = false;
  const view = createResearchView(page, async () => ({ok: true, json: async () => failure ? {status: "unavailable", message: "数据缺失"} : data()}));
  await view.ready;
  failure = true;
  await view.refresh();
  assert.ok(page.elements.stockResearchContent.classList.contains("hidden"));
  assert.ok(!page.elements.stockResearchUnavailable.classList.contains("hidden"));
  assert.equal(page.elements.stockResearchError.textContent, "数据缺失");
  assert.equal(page.elements.stockDataCutoff.textContent, "—");
  assert.equal(page.elements.retryStockResearch.disabled, false);
  failure = false;
  await view.refresh();
  assert.ok(!page.elements.stockResearchContent.classList.contains("hidden"));
  assert.ok(page.elements.stockResearchUnavailable.classList.contains("hidden"));
  view.destroy();
});

test("network errors and absent forward metadata have explicit states", async () => {
  const page = root(); let fail = true;
  const view = createResearchView(page, async () => {
    if (fail) throw new Error("连接失败");
    const value = data(); delete value.forward_status; value.report_url = "javascript:alert(1)";
    return {ok: true, json: async () => value};
  });
  await view.ready;
  assert.equal(page.elements.stockResearchError.textContent, "连接失败");
  fail = false;
  await view.refresh();
  assert.equal(page.elements.stockForwardState.textContent, "尚未取得前瞻运行记录");
  assert.ok(page.elements.stockReportLink.classList.contains("hidden"));
  view.destroy();
});

test("valid four-account heartbeat updates forward status without changing historical returns", async () => {
  const page = root();
  const forward = {mode: "SIMULATION", places_orders: false, running: true,
    started_at_utc: "2026-10-05T04:42:03Z", updated_at_utc: new Date().toISOString(),
    accounts: {btc: {status: "degraded"}, mu: {}, sndk: {}, skhynix: {}}};
  const view = createResearchView(page, async url => ({ok: true,
    json: async () => url.startsWith("/data/") ? forward : data()}));
  await view.ready;
  assert.match(page.elements.stockForwardState.textContent, /四账户运行中/);
  assert.match(page.elements.stockForwardState.textContent, /12:42/);
  assert.ok(page.elements.stockCards.innerHTML.includes("+0.73%"));
  forward.updated_at_utc = "2026-01-01T00:00:00Z";
  await view.refresh();
  assert.match(page.elements.stockForwardState.textContent, /心跳停止或过期/);
  view.destroy();
});

test("missing forward endpoint keeps historical research available", async () => {
  const page = root();
  const view = createResearchView(page, async url => url.startsWith("/data/") ? {ok:false} : {ok:true, json:async()=>data()});
  await view.ready;
  assert.match(page.elements.stockForwardState.textContent, /尚未启动/);
  assert.ok(!page.elements.stockResearchContent.classList.contains("hidden"));
  view.destroy();
});
