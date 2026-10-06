import test from 'node:test';
import assert from 'node:assert/strict';
import { trendExperimentMarkup, createTrendExperiment } from '../terminal/trend-experiment.js';

function status() {
  const now = new Date().toISOString();
  const arm = equity => ({equity, status: 'healthy', checked_at_utc: now, position: null,
    experiment_fill_count: 0, recent_fills: []});
  return {mode: 'SIMULATION', places_orders: false, running: true, started_at_utc: now,
    updated_at_utc: now, baseline: {mu: {equity: 999}, sndk: {equity: 1000}, skhynix: {equity: 995}},
    accounts: {mu: {candidate: arm(998), control: arm(999)}, sndk: {candidate: arm(1001), control: arm(1000)},
      skhynix: {candidate: arm(995), control: arm(995)}}};
}

test('experiment uses activation deltas and excludes pre-activation transaction counts', () => {
  const value = status();
  value.accounts.mu.candidate.fill_count_total = 100;
  const markup = trendExperimentMarkup(value);
  assert.match(markup.rows, /美光 · MU<\/td><td>-1\.0000<\/td><td>0\.0000<\/td><td>-1\.0000/);
  assert.match(markup.rows, /<td>0 \/ 0<\/td>/);
  assert.doesNotMatch(markup.rows, /100 \/ 0/);
  assert.match(markup.tradeRows, /等待新的模拟成交/);
});

test('stale or degraded experiments never display fresh equity differences', () => {
  const value = status();
  value.accounts.mu.candidate.status = 'degraded';
  assert.match(trendExperimentMarkup(value).rows, /美光 · MU<\/td><td>—<\/td><td>0\.0000<\/td><td>—/);
  value.updated_at_utc = '2026-01-01T00:00:00Z';
  const markup = trendExperimentMarkup(value);
  assert.equal(markup.active, false);
  assert.ok(!markup.rows.includes('1.0000'));
});

test('ledger reasons are escaped and partial exits are shown as exiting', () => {
  const value = status();
  value.accounts.mu.candidate.position = {direction: -1, qty: .02, pending_exit: 'trend_reversal_exit'};
  value.accounts.mu.candidate.recent_fills = [{time_ms: Date.now(), side: 'BUY', price: 1070, qty: .01, fee: .001, reason: '<img onerror=x>'}];
  const markup = trendExperimentMarkup(value);
  assert.match(markup.rows, /反转退出中/);
  assert.match(markup.tradeRows, /&lt;img onerror=x&gt;/);
  assert.doesNotMatch(markup.tradeRows, /<img/);
});

test('experiment endpoint failures are contained and recovery reloads the independent comparison', async () => {
  const fields = Object.fromEntries(['status', 'rows', 'trades'].map(name => [name,
    {textContent: '', innerHTML: '', classList: {add() {}, toggle() {}}}]));
  const root = {querySelector: selector => fields[selector.match(/"([^"]+)"/)[1]]};
  let fail = true; const requests = [];
  const view = createTrendExperiment(root, async url => {requests.push(url); return {ok: !fail, json: async () => status()};});
  await view.ready;
  assert.match(fields.rows.innerHTML, /原账户继续独立运行/);
  fail = false;
  await view.refresh();
  assert.match(fields.rows.innerHTML, /美光 · MU/);
  assert.ok(requests.every(url => url.startsWith('/data/paper_trading/stock_trend_reversal_exit_20261006/status.json')));
  view.destroy();
});
