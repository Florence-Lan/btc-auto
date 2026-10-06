import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { createPaperAccounts, stockTradingMarkup } from '../terminal/parallel.js';

const html = name => readFileSync(new URL(`../terminal/${name}`, import.meta.url), 'utf8');
function root(group) {
  const fields = Object.fromEntries(['start', 'heartbeat', 'total', 'cards', ...(group === 'stocks' ? ['comparison', 'comparison-start', 'comparison-rows', 'positions', 'trades', 'trade-status'] : [])]
    .map(name => [name, {textContent: '', innerHTML: '', hidden: false, classList: {add() {}, toggle() {}}}]));
  return {fields, dataset: {accountGroup: group}, querySelector(selector) {
    return fields[selector.match(/"([^"]+)"/)[1]] || null;
  }};
}
function status() {
  const now = new Date().toISOString();
  const accounts = Object.fromEntries(['btc', 'mu', 'sndk', 'skhynix'].map((id, i) => [id, {
    equity: (i + 1) * 1000, checked_at_utc: now, status: 'healthy', position_qty: 0,
  }]));
  return {running: true, started_at_utc: now, updated_at_utc: now, accounts,
    profit_exit_comparison: {activated_at_utc: now, controls: accounts, baseline: accounts}};
}
const fetcher = value => async () => ({ok: true, json: async () => value});

test('home and former combined page offer two distinct account destinations', () => {
  for (const page of ['index.html', 'parallel.html']) {
    assert.match(html(page), /href="\.\/btc\.html"/);
    assert.match(html(page), /href="\.\/stocks\.html"/);
    assert.equal((html(page).match(/class="panel account-entry"/g) || []).length, 2);
    assert.doesNotMatch(html(page), /src="\.\/(app|parallel|stocks)\.js"/);
  }
  assert.doesNotMatch(html('btc.html'), /id="stockWorkspace"|src="\.\/stocks\.js(?:\?[^"]*)?"/);
  assert.doesNotMatch(html('stocks.html'), /id="btcWorkspace"|id="startButton"|src="\.\/app\.js(?:\?[^"]*)?"/);
});

test('BTC equity and cards exclude every stock account and stock comparison', async () => {
  const page = root('btc');
  const view = createPaperAccounts(page, fetcher(status()));
  await view.ready;
  assert.match(page.fields.total.textContent, /BTC权益 1,000\.0000 USDT \/ 初始 1,000\.0000/);
  assert.equal((page.fields.cards.innerHTML.match(/class="panel parallel-card"/g) || []).length, 1);
  assert.doesNotMatch(page.fields.cards.innerHTML, /MU|SNDK|SKHYNIX/);
  view.destroy();
});

test('stock equity excludes BTC and renders all three independent accounts and controls', async () => {
  const page = root('stocks');
  const view = createPaperAccounts(page, fetcher(status()));
  await view.ready;
  assert.match(page.fields.total.textContent, /股票权益 9,000\.0000 USDT \/ 初始 3,000\.0000/);
  assert.equal((page.fields.cards.innerHTML.match(/class="panel parallel-card"/g) || []).length, 3);
  assert.doesNotMatch(page.fields.cards.innerHTML, /<h2>BTC<\/h2>/);
  assert.equal(page.fields.comparison.hidden, false);
  assert.equal((page.fields['comparison-rows'].innerHTML.match(/<tr>/g) || []).length, 3);
  view.destroy();
});

test('a missing stock hides aggregate equity even when BTC remains available', async () => {
  const value = status();
  delete value.accounts.sndk;
  const page = root('stocks');
  const view = createPaperAccounts(page, fetcher(value));
  await view.ready;
  assert.match(page.fields.total.textContent, /股票权益 —/);
  assert.match(page.fields.cards.innerHTML, /闪迪 · SNDK/);
  view.destroy();
});

test('unavailable account data clears prior values and recovers on refresh', async () => {
  for (const group of ['btc', 'stocks']) {
    const page = root(group); let fail = false;
    const view = createPaperAccounts(page, async () => ({ok: !fail, json: async () => status()}));
    await view.ready;
    fail = true;
    await view.refresh();
    assert.equal(page.fields.total.textContent, '当前权益暂不可用');
    assert.equal(page.fields.cards.textContent, '等待运行记录恢复');
    if (group === 'stocks') assert.equal(page.fields.comparison.hidden, true);
    fail = false;
    await view.refresh();
    assert.match(page.fields.cards.innerHTML, /parallel-card/);
    view.destroy();
  }
});

test('former workspace bookmarks go directly to their dedicated pages', () => {
  const script = html('entry.js');
  for (const [hash, expected] of [['#btc', './btc.html'], ['#stocks', './stocks.html'], ['', null]]) {
    let destination = null;
    runInNewContext(script, {location: {hash, replace(url) {destination = url;}}});
    assert.equal(destination, expected);
  }
});

function ledger(id) {
  return {mode: 'simulation', places_orders: false, symbol: {mu: 'MUUSDT', sndk: 'SNDKUSDT', skhynix: 'SKHYNIXUSDT'}[id],
    equity: 1001, wallet_balance: 1000, last_mark_price: 105,
    position: id === 'mu' ? {direction: -1, qty: .07, entry: 106, stop: 108} : null,
    fills: [{time_ms: 1000, side: 'SELL', price: 106, qty: .07, fee: .01, reason: 'fresh_closed_15m_signal'}]};
}

test('stock page loads its three real ledgers and shows positions and fills', async () => {
  const page = root('stocks'), requests = [];
  const view = createPaperAccounts(page, async url => {
    requests.push(url);
    const id = url.match(/\/(mu|sndk|skhynix)\/state\.json/)?.[1];
    return {ok: true, json: async () => id ? ledger(id) : status()};
  });
  await view.ready;
  assert.equal(requests.length, 4);
  assert.ok(!requests.some(url => url.includes('/btc/state.json') || url.includes('/control')));
  assert.match(page.fields.positions.innerHTML, /做空/);
  assert.match(page.fields.positions.innerHTML, /0\.0700/);
  assert.match(page.fields.trades.innerHTML, /106\.0000/);
  assert.match(page.fields.trades.innerHTML, /15分钟信号入场/);
  assert.match(page.fields['trade-status'].textContent, /最近3笔模拟成交/);
  view.destroy();
});

test('one failed ledger keeps other positions and trades visible without inventing an empty position', async () => {
  const page = root('stocks');
  const view = createPaperAccounts(page, async url => {
    if (url.includes('/sndk/state.json')) return {ok: false};
    const id = url.match(/\/(mu|skhynix)\/state\.json/)?.[1];
    return {ok: true, json: async () => id ? ledger(id) : status()};
  });
  await view.ready;
  assert.match(page.fields.positions.innerHTML, /闪迪 · SNDK<\/td><td colspan="6">持仓明细暂不可用/);
  assert.match(page.fields.positions.innerHTML, /做空/);
  assert.match(page.fields.trades.innerHTML, /美光 · MU/);
  assert.match(page.fields['trade-status'].textContent, /部分账本读取失败/);
  assert.match(page.fields.total.textContent, /9,000/);
  view.destroy();
});

test('trade rows are newest first, escape ledger strings and distinguish absent records', () => {
  const state = ledger('mu');
  state.fills.push({time_ms: 2000, side: 'BUY', price: 99, qty: .02, fee: .01, reason: '<script>x</script>'});
  const markup = stockTradingMarkup([{id: 'mu', state}]);
  assert.ok(markup.trades.indexOf('99.0000') < markup.trades.indexOf('106.0000'));
  assert.ok(markup.trades.includes('&lt;script&gt;'));
  assert.ok(!markup.trades.includes('<script>'));
  state.fills = [];
  assert.match(stockTradingMarkup([{id: 'mu', state}]).trades, /暂无模拟成交记录/);
  assert.match(stockTradingMarkup([{id: 'mu', error: 'unavailable'}]).trades, /成交明细暂不可用/);
});
