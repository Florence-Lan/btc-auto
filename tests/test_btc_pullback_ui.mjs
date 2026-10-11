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

function status(pullback, account = {}) {
  const now = new Date().toISOString();
  return {running: true, started_at_utc: now, updated_at_utc: now,
    accounts: Object.fromEntries(['btc', 'mu', 'sndk', 'skhynix'].map(id => [id, {
      equity: 1000, checked_at_utc: now, status: 'healthy', position_qty: 0,
      btc_pullback_entry: pullback, ...account,
    }]))};
}

const fetcher = value => async () => ({ok: true, json: async () => value});
async function render(pullback, account = {}, group = 'btc') {
  const page = root(group);
  const view = createPaperAccounts(page, fetcher(status(pullback, account)));
  await view.ready;
  view.destroy();
  return page.fields.cards.innerHTML;
}

function plan(overrides = {}) {
  return {enabled: true, allowed: false, status: 'waiting_retrace', side: 'long',
    reference_price: 83456, candidate_low: 83300, candidate_high: 83322,
    numeric_levels: [83300, {price: 83311, kind: 'repeated_tail'}, 83322],
    projected_fill_price: 83320.25, checked_at_ms: Date.now(),
    expires_at_ms: Date.parse('2026-10-12T01:00:00Z'), ...overrides};
}

test('BTC waits for a lower long entry with distinct strategy, candidate, projected and actual prices', async () => {
  const cards = await render(plan(), {position_qty: .002, entry_price: 83444.5,
    signal_status: 'entry_blocked', profit_exit_status: {enabled: false, target_margin_return: .6}});
  assert.match(cards, /等待回踩/);
  assert.match(cards, /原始策略参考价 83,456\.0000 USDT/);
  assert.match(cards, /候选入场区 83,300\.0000—83,322\.0000 USDT/);
  assert.match(cards, /整数\/对子参考价 83,300\.0000、83,311\.0000、83,322\.0000 USDT/);
  assert.match(cards, /预计执行价（含滑点）83,320\.2500 USDT/);
  assert.match(cards, /实际持仓均价<\/dt><dd>83,444\.5000/);
  assert.match(cards, /有效至北京时间 2026\/10\/12 09:00:00/);
  assert.match(cards, /class="parallel-state ">运行正常/);
  assert.doesNotMatch(cards, /数据降级|保证金净收益目标|预计平仓净ROI|保证盈利|60%/);
});

test('BTC waits for a higher short entry without turning a price wait into a worker failure', async () => {
  const cards = await render(plan({side: 'short', reference_price: 83000,
    candidate_low: 83100, candidate_high: 83122}));
  assert.match(cards, /等待反弹/);
  assert.match(cards, /class="parallel-state ">运行正常/);
  assert.doesNotMatch(cards, /等待回踩|数据降级|parallel-state parallel-warning/);
});

test('BTC translates pullback blockers for both status prefix forms', async () => {
  const cards = await render(plan({side: 'short'}), {signal_status: 'entry_blocked',
    entry_blockers: ['btc_pullback_waiting_retrace', 'btc_pullback_entry_waiting_confirmation']});
  assert.match(cards, /等待反弹/);
  assert.match(cards, /已到候选区，等待止跌\/受阻确认/);
  assert.doesNotMatch(cards, /btc_pullback_/);
});

test('BTC translates module price guards and causal-data blockers without showing raw codes', async () => {
  const cards = await render(plan({status: 'ready', allowed: false}), {
    signal_status: 'entry_blocked', entry_blockers: ['btc_pullback_chase_exceeded',
      'btc_pullback_rejection_not_held', 'btc_pullback_5m_gap', 'btc_pullback_hourly_history_insufficient',
      'btc_pullback_pending_corridor_invalid', 'btc_pullback_policy_unknown_key',
      'btc_pullback_source_consumed']});
  assert.match(cards, /预计执行价超过追价上限，等待价格回归/);
  assert.match(cards, /价格尚未守住确认区，继续等待/);
  assert.match(cards, /5分钟确认行情有缺口，等待完整数据/);
  assert.match(cards, /原始小时行情不足，等待完整历史/);
  assert.match(cards, /回踩计划资料无效，等待新小时信号/);
  assert.match(cards, /回踩入场配置无效，等待修复/);
  assert.match(cards, /该小时入场机会已使用，等待新小时信号/);
  assert.doesNotMatch(cards, /btc_pullback_/);
});

test('BTC distinguishes a zone visit from confirmation and the final execution-price check', async () => {
  const waiting = await render(plan({status: 'waiting_confirmation'}));
  assert.match(waiting, /已到候选区，等待止跌\/受阻确认/);
  assert.doesNotMatch(waiting, /已确认，检查实际成交价/);
  const ready = await render(plan({status: 'ready', allowed: true}));
  assert.match(ready, /已确认，检查实际成交价/);
  assert.doesNotMatch(ready, /已成交|保证盈利/);
});

test('unavailable, invalid, expired and cancelled plans suppress stale candidate and projected prices', async () => {
  for (const [status, message] of [
    ['market_data_unavailable', '入场行情数据暂不可用，等待完整数据'],
    ['data_unavailable', '入场行情数据暂不可用，等待完整数据'],
    ['price_unavailable', '当前执行价暂不可用，等待完整报价'],
    ['invalid_policy', '回踩入场配置无效，等待修复'],
    ['origin_missing', '原始小时信号资料缺失，等待新小时信号'],
    ['no_structure', '暂未找到有效回踩结构，继续等待'],
    ['invalid_origin', '原始小时信号资料无效，等待新小时信号'],
    ['invalid_source', '原始小时信号资料无效，等待新小时信号'],
    ['expired', '回踩计划失效，等待新小时信号'],
    ['cancelled', '回踩计划失效，等待新小时信号'],
    ['structure_invalidated', '回踩计划失效，等待新小时信号'],
  ]) {
    const cards = await render(plan({status}), {position_qty: -.002, entry_price: 83444.5});
    assert.ok(cards.includes(message), status);
    assert.match(cards, /实际持仓均价<\/dt><dd>83,444\.5000/);
    assert.doesNotMatch(cards, /候选入场区|整数\/对子参考价|预计执行价|有效至北京时间/);
    if (['invalid_policy', 'origin_missing', 'invalid_origin', 'invalid_source'].includes(status)) {
      assert.doesNotMatch(cards, /原始策略参考价/);
    }
  }
});

test('BTC displays only valid provided prices and does not invent a fill price for a flat account', async () => {
  const cards = await render(plan({reference_price: Infinity, candidate_low: 83322,
    candidate_high: 83300, numeric_levels: [null, '<script>x</script>', {price: NaN}, -1, 83311, 83311],
    projected_fill_price: NaN, expires_at_ms: 0}), {position_qty: 0, entry_price: 83444.5});
  assert.match(cards, /整数\/对子参考价 83,311\.0000 USDT/);
  assert.doesNotMatch(cards, /原始策略参考价|候选入场区|预计执行价|实际持仓均价|有效至北京时间|<script>|NaN|Infinity/);
});

test('legacy and disabled BTC plans do not show an invented pullback state', async () => {
  for (const value of [undefined, {enabled: false}, plan({enabled: false})]) {
    const cards = await render(value);
    assert.doesNotMatch(cards, /等待回踩|等待反弹|原始策略参考价|候选入场区/);
  }
});

test('a refresh clears candidate prices after the plan loses its structure', async () => {
  const page = root();
  const value = status(plan());
  const view = createPaperAccounts(page, fetcher(value));
  await view.ready;
  assert.match(page.fields.cards.innerHTML, /候选入场区/);
  value.accounts.btc.btc_pullback_entry.status = 'structure_invalidated';
  await view.refresh();
  assert.match(page.fields.cards.innerHTML, /回踩计划失效，等待新小时信号/);
  assert.doesNotMatch(page.fields.cards.innerHTML, /候选入场区|整数\/对子参考价|预计执行价/);
  view.destroy();
});

test('stock cards keep their split profit display and do not display BTC plans or BTC average prices', async () => {
  const cards = await render(plan(), {position_qty: .02, entry_price: 83444.5,
    profit_exit_status: {stage_done: true, filled_qty: .02, stop: 101}}, 'stocks');
  assert.match(cards, /分批止盈：首段完成 · 余仓跟踪保护/);
  assert.doesNotMatch(cards, /等待回踩|原始策略参考价|候选入场区|实际持仓均价/);
});
