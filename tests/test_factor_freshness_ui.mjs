import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createPaperAccounts, factorFreshnessMarkup } from '../terminal/parallel.js';

const date = Date.parse('2026-10-20T14:00:00Z');
const diagnostics = () => ({checked_at_ms: date, display_only: true, factors: {
  broad_dollar: {name: '广义美元指数', group: 'fx', group_enabled: true, cadence: 'daily',
    latest_observed_at_ms: Date.parse('2026-10-06T00:00:00Z'), age_ms: 14.583 * 86_400_000,
    max_age_ms: 16 * 86_400_000, status: 'fresh', slow_frequency: true,
    fetch_ok: true, last_fetch_attempt_ms: date, last_fetch_success_ms: date,
    notes: ['按日发布；允许滞后不代表实时行情']},
  oil: {name: '<oil>', group: 'global_risk', cadence: 'daily', optional: true, ignored: true,
    latest_observed_at_ms: Date.parse('2026-10-10T00:00:00Z'), age_ms: 10.583 * 86_400_000,
    max_age_ms: 7 * 86_400_000, status: 'stale', fetch_ok: true,
    last_fetch_attempt_ms: date, last_fetch_success_ms: date, notes: ['<unsafe>']},
}});

test('separates old observation dates from successful collection and escapes display values', () => {
  const markup = factorFreshnessMarkup(diagnostics());
  assert.match(markup, /宏观因子观测日期|来源抓取成功不等于实时观测/);
  assert.match(markup, /2026\/10\/6 08:00:00/);
  assert.match(markup, /14\.6 天|16\.0 天|日数据/);
  assert.match(markup, /现行时限内/);
  assert.match(markup, /观测已过期/);
  assert.match(markup, /最近抓取成功/);
  assert.match(markup, /可选输入|本次判断未计入/);
  assert.match(markup, /&lt;oil&gt;|&lt;unsafe&gt;/);
  assert.doesNotMatch(markup, /<oil>|<unsafe>/);
});

test('missing and legacy collection dates remain explicit, without fabricating an outage', () => {
  const value = diagnostics();
  Object.assign(value.factors.oil, {status: 'missing', latest_observed_at_ms: null,
    age_ms: null, fetch_ok: null, last_fetch_attempt_ms: null, last_fetch_success_ms: null});
  const markup = factorFreshnessMarkup(value);
  assert.match(markup, /观测缺失|未取得可用观测/);
  assert.match(markup, /抓取结果未单独记录|最近尝试 未记录/);
  assert.doesNotMatch(markup, /最近抓取失败/);
  assert.match(factorFreshnessMarkup(null), /尚无逐项日期记录/);
});

function root(group) {
  const fields = Object.fromEntries(['start', 'heartbeat', 'total', 'cards', 'factor-freshness']
    .map(name => [name, {textContent: '', innerHTML: '', classList: {add() {}, toggle() {}}}]));
  return {fields, dataset: {accountGroup: group}, querySelector(selector) {
    return fields[selector.match(/"([^"]+)"/)[1]] || null;
  }};
}

test('BTC account reads current gate diagnostics and clears the table on a failed refresh', async () => {
  const now = new Date().toISOString();
  const page = root('btc');
  let available = true;
  const account = {equity: 1000, status: 'healthy', checked_at_utc: now,
    execution_entry_gate: {by_side: {short: {factor_data_freshness: diagnostics()}}}};
  const fetcher = async () => ({ok: available, json: async () => ({running: true,
    updated_at_utc: now, started_at_utc: now, accounts: {btc: account}})});
  const view = createPaperAccounts(page, fetcher);
  await view.ready;
  assert.match(page.fields['factor-freshness'].innerHTML, /广义美元指数/);
  available = false;
  await view.refresh();
  assert.match(page.fields['factor-freshness'].innerHTML, /尚无逐项日期记录/);
  assert.doesNotMatch(page.fields['factor-freshness'].innerHTML, /广义美元指数/);
  view.destroy();
});

test('dedicated date table is visible only on the BTC page', () => {
  const html = name => readFileSync(new URL(`../terminal/${name}`, import.meta.url), 'utf8');
  assert.match(html('btc.html'), /data-paper-field="factor-freshness"/);
  assert.doesNotMatch(html('stocks.html'), /data-paper-field="factor-freshness"/);
});
