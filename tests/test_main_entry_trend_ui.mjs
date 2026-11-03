import test from 'node:test';
import assert from 'node:assert/strict';
import { createPaperAccounts } from '../terminal/parallel.js';

function page() {
  const fields = Object.fromEntries(['start', 'heartbeat', 'total', 'cards']
    .map(name => [name, {textContent: '', innerHTML: '', classList: {add() {}, toggle() {}}}]));
  return {fields, dataset: {accountGroup: 'stocks'}, querySelector(selector) {
    return fields[selector.match(/"([^"]+)"/)[1]] || null;
  }};
}

const close = Date.UTC(2026, 10, 2, 12);
const context = (side, overrides = {}) => ({status: 'healthy', direction: side,
  strict_direction: side, price_led_direction: side,
  ema_spread_fraction: .01, closed_at_ms: close, ...overrides});
const confirmation = (frames = [], overrides = {}) => ({enabled: true, allowed: frames.length > 0,
  status: frames.length ? 'confirmed' : 'no_confirmation', direction: null,
  decision_scope: 'current_context_not_entry_permission', reasons: [],
  admitted_by_timeframes: frames, channels: Object.fromEntries(['1h', '2h', '4h'].map(frame =>
    [frame, {allowed: frames.includes(frame), admitted_by: frames.includes(frame) ? 'strict' : null,
      reasons: frames.includes(frame) ? [] : ['entry_trend_no_confirmation']}])) , ...overrides});

function status(account = {}) {
  const now = new Date().toISOString();
  const accounts = Object.fromEntries(['mu', 'sndk', 'skhynix'].map(id => [id,
    {equity: 1000, checked_at_utc: now, status: 'healthy', position_qty: 0,
      signal_family: 'donchian20', signal_timeframe: '1h', entry_direction: 'both',
      signal_status: 'no_signal', entry_blockers: []}]));
  Object.assign(accounts.mu, {entry_trend_runtime: {enabled: true,
    confirmation_timeframes: ['1h', '2h', '4h']},
    current_trend: {enabled: true, status: 'healthy', hourly: context(1),
      two_hour: context(-1), four_hour: context(-1),
      entry_confirmation: confirmation(['2h', '4h'])}}, account);
  return {running: true, started_at_utc: now, updated_at_utc: now, accounts};
}

async function render(value) {
  const root = page();
  const view = createPaperAccounts(root, async () => ({ok: true, json: async () => value}));
  await view.ready;
  return {root, view, cards: () => root.fields.cards.innerHTML};
}

test('main stock cards show separate 1h, 2h and 4h closed contexts without claiming an entry', async () => {
  const rendered = await render(status());
  try {
    assert.match(rendered.cards(), /1小时价格方向：偏多；均线背景：偏多/);
    assert.match(rendered.cards(), /2小时价格方向：偏空；均线背景：偏空/);
    assert.match(rendered.cards(), /4小时价格方向：偏空；均线背景：偏空/);
    assert.match(rendered.cards(), /入场趋势规则：1h\/2h\/4h任一周期确认/);
    assert.match(rendered.cards(), /趋势背景已形成：2h、4h；入场另需原始信号和执行检查/);
    assert.match(rendered.cards(), /本根未触发信号/);
    assert.doesNotMatch(rendered.cards(), /信号时趋势确认：|已入场|等待4小时波段方向确认/);
  } finally { rendered.view.destroy(); }
});

test('signal-time confirmation names only actual accepted periods and retains other entry checks', async () => {
  const value = status({entry_trend_confirmation: confirmation(['2h', '4h'], {
    direction: -1, decision_scope: 'entry_confirmation_only',
    admitted_by_timeframes: ['2h', '<img src=x onerror=alert(1)>', '4h']})});
  const rendered = await render(value);
  try {
    assert.match(rendered.cards(), /信号时趋势确认：2h、4h；继续检查实际入场条件/);
    assert.doesNotMatch(rendered.cards(), /<img|onerror|已入场/);
    value.accounts.mu.entry_trend_confirmation.allowed = false;
    await rendered.view.refresh();
    assert.match(rendered.cards(), /信号时趋势确认：暂无周期确认/);
    assert.doesNotMatch(rendered.cards(), /信号时趋势确认：2h、4h/);
  } finally { rendered.view.destroy(); }
});

test('unconfirmed directions and unavailable observations have distinct explanations', async () => {
  const value = status({signal_status: 'swing_confirmation_blocked',
    entry_blockers: ['entry_trend_no_confirmation'],
    entry_trend_confirmation: confirmation([], {direction: 1})});
  value.accounts.mu.current_trend.entry_confirmation = confirmation();
  const rendered = await render(value);
  try {
    assert.match(rendered.cards(), /等待趋势方向确认/);
    assert.match(rendered.cards(), /1h、2h、4h暂未确认入场方向/);
    assert.match(rendered.cards(), /趋势背景：当前无周期达到确认条件/);
    assert.match(rendered.cards(), /信号时趋势确认：暂无周期确认/);
    assert.doesNotMatch(rendered.cards(), /entry_trend_no_confirmation/);
    value.accounts.mu.entry_blockers = ['entry_trend_data_unavailable'];
    value.accounts.mu.entry_trend_confirmation = confirmation([], {direction: 1,
      status: 'unavailable', reasons: ['entry_trend_data_unavailable']});
    value.accounts.mu.current_trend.status = 'unavailable';
    await rendered.view.refresh();
    assert.match(rendered.cards(), /趋势确认资料暂不可用，等待完整收盘行情/);
    assert.match(rendered.cards(), /趋势背景：资料暂不可用/);
    assert.match(rendered.cards(), /信号时趋势确认：资料暂不可用/);
    assert.doesNotMatch(rendered.cards(), /2小时价格方向|趋势背景已形成|entry_trend_data_unavailable/);
  } finally { rendered.view.destroy(); }
});

test('a context rejection is not presented as rejection of an absent raw signal', async () => {
  const rendered = await render(status({signal_status: 'swing_confirmation_blocked',
    entry_blockers: ['entry_trend_no_confirmation'], pending_signal: null,
    signal_diagnostics: {admission: {stage: 'no_raw_signal'}, final_stage: 'no_raw_signal'}}));
  try {
    assert.match(rendered.cards(), /本根未触发信号/);
    assert.match(rendered.cards(), /趋势背景：1h、2h、4h暂未确认入场方向/);
    assert.doesNotMatch(rendered.cards(), /等待趋势方向确认|已入场/);
  } finally { rendered.view.destroy(); }
});

test('an unavailable 2h channel is explicit while other healthy contexts remain visible', async () => {
  const value = status();
  value.accounts.mu.current_trend.two_hour = {status: 'unavailable'};
  const rendered = await render(value);
  try {
    assert.match(rendered.cards(), /2小时趋势资料暂不可用/);
    assert.match(rendered.cards(), /1小时价格方向：偏多/);
    assert.match(rendered.cards(), /4小时价格方向：偏空/);
    assert.doesNotMatch(rendered.cards(), /2小时价格方向：偏空/);
  } finally { rendered.view.destroy(); }
});

test('new policy activation waits and legacy accounts keep their original context rendering', async () => {
  const value = status({signal_status: 'swing_confirmation_blocked',
    entry_blockers: ['entry_trend_awaiting_new_signal']});
  const rendered = await render(value);
  try {
    assert.match(rendered.cards(), /等待新趋势规则生效后的小时信号/);
    assert.doesNotMatch(rendered.cards(), /entry_trend_awaiting_new_signal/);
    delete value.accounts.mu.entry_trend_runtime;
    delete value.accounts.mu.current_trend.two_hour;
    delete value.accounts.mu.current_trend.entry_confirmation;
    await rendered.view.refresh();
    assert.match(rendered.cards(), /1小时价格方向|4小时价格方向/);
    assert.doesNotMatch(rendered.cards(), /2小时价格方向|入场趋势规则：1h\/2h\/4h/);
  } finally { rendered.view.destroy(); }
});
