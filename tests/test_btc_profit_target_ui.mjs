import test from 'node:test';
import assert from 'node:assert/strict';
import { createPaperAccounts } from '../terminal/parallel.js';

function root(group = 'btc') {
  const fields = Object.fromEntries(['start', 'heartbeat', 'total', 'cards']
    .map(name => [name, {textContent: '', innerHTML: '', classList: {add() {}, toggle() {}}}]));
  return {fields, dataset: {accountGroup: group}, querySelector(selector) {
    return fields[selector.match(/"([^"]+)"/)[1]] || null;
  }};
}

function status(profit) {
  const now = new Date().toISOString();
  return {running: true, started_at_utc: now, updated_at_utc: now,
    accounts: Object.fromEntries(['btc', 'mu', 'sndk', 'skhynix'].map(id => [id, {
      equity: 1000, checked_at_utc: now, status: 'healthy', position_qty: 0,
      profit_exit_status: profit,
    }]))};
}

async function render(profit, group = 'btc', btc = {}) {
  const page = root(group);
  const value = status(profit);
  Object.assign(value.accounts.btc, btc);
  const view = createPaperAccounts(page, async () => ({ok: true, json: async () => value}));
  await view.ready;
  view.destroy();
  return page.fields.cards.innerHTML;
}

test('BTC shows its margin target and cost-adjusted close projection without a stock split label', async () => {
  const cards = await render({enabled: true, target_margin_return: .6,
    projected_net_roi_pct: 12.345, target_mark_price: 94000.25});
  assert.match(cards, /保证金净收益目标60%/);
  assert.match(cards, /预计平仓净ROI 12\.35%/);
  assert.match(cards, /目标标记价 94,000\.2500 USDT/);
  assert.doesNotMatch(cards, /分批止盈|等待达到1R|保证盈利|预期收益60%/);
});

test('a flat BTC account shows the configured target without inventing ROI or target prices', async () => {
  const cards = await render({enabled: true, status: 'flat', target_margin_return: .6});
  assert.match(cards, /保证金净收益目标60%/);
  assert.doesNotMatch(cards, /预计平仓净ROI|目标标记价|NaN|undefined/);
});

test('missing, disabled and invalid BTC policy diagnostics show no target', async () => {
  for (const profit of [undefined, {enabled: false, target_margin_return: .6},
    {enabled: true, target_margin_return: null}, {enabled: true, target_margin_return: NaN},
    {enabled: true, target_margin_return: .5}]) {
    const cards = await render(profit);
    assert.doesNotMatch(cards, /保证金净收益目标|分批止盈|NaN|undefined/);
  }
  const cards = await render({enabled: true, target_margin_return: .6,
    projected_net_roi_pct: Infinity, target_mark_price: -1});
  assert.match(cards, /保证金净收益目标60%/);
  assert.doesNotMatch(cards, /预计平仓净ROI|目标标记价|Infinity/);
});

test('BTC explains unavailable profit checks and completed target exits without presenting stale projections', async () => {
  for (const [status, message] of [
    ['funding_unavailable', '资金费数据暂不可用，等待完整数据后判断止盈'],
    ['ledger_unavailable', '成交账本暂不可核验，等待完整账本后判断止盈'],
    ['closed_target', '已按目标止盈，等待新信号'],
  ]) {
    const cards = await render({enabled: true, status, target_margin_return: .6,
      projected_net_roi_pct: 60, target_mark_price: 94000});
    assert.match(cards, /保证金净收益目标60%/);
    assert.ok(cards.includes(message));
    assert.doesNotMatch(cards, /预计平仓净ROI|目标标记价/);
  }
  const cards = await render({enabled: true, status: 'invalid_policy',
    target_margin_return: .6, projected_net_roi_pct: 60, target_mark_price: 94000});
  assert.match(cards, /止盈配置无效，等待修复/);
  assert.doesNotMatch(cards, /保证金净收益目标|预计平仓净ROI|目标标记价/);
});

test('BTC translates the consumed hourly signal blocker after profit-taking', async () => {
  const cards = await render(undefined, 'btc', {signal_status: 'entry_blocked',
    entry_blockers: ['profit_target_signal_consumed']});
  assert.match(cards, /该小时信号已止盈，等待新信号/);
  assert.doesNotMatch(cards, /profit_target_signal_consumed/);
});

test('stock cards retain their existing split profit protection display', async () => {
  const cards = await render({stage_done: true, filled_qty: .02, stop: 101}, 'stocks');
  assert.match(cards, /分批止盈：首段完成 · 余仓跟踪保护/);
  assert.match(cards, /首段已退出 0\.0200 · 当前保护价 101\.0000/);
  assert.doesNotMatch(cards, /保证金净收益目标|预计平仓净ROI/);
});
