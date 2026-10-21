const names = {mu: '美光 · MU', sndk: '闪迪 · SNDK', skhynix: '海力士 · SKHYNIX', btc: 'BTC'};
const money = n => n == null ? '—' : Number(n).toLocaleString('en-US', {minimumFractionDigits: 4, maximumFractionDigits: 4});
const escape = s => String(s ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const localTime = s => s ? new Date(s).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai', hour12: false}) : '—';
const signalNames = {holding:'持仓保护中', waiting_for_new_closed_bar:'等待下一根收盘线',
  waiting_for_first_forward_candle:'等待新规则首根收盘线', no_signal:'本根未触发信号',
  signal_expired:'上次信号已过期', entry_data_unavailable:'等待完整入场数据',
  risk_halted:'回撤限制新增', cooldown:'退出后冷却中',
  entry_data_gap_basis_funding_or_volume_blocked:'信号有效 · 等待成交量、价差或资金费条件',
  insufficient_observed_liquidity_or_minimum_quantity:'信号有效 · 等待足够可成交数量',
  risk_or_liquidation_buffer_blocked:'信号有效 · 风险或保证金条件未通过',
  stop_distance_blocked:'信号有效 · 止损距离未通过', entered:'已入场',
  entered_partial_ioc_remainder_cancelled:'已部分入场', signal_observed:'新信号已观察',
  no_new_signal:'等待新信号'};
signalNames.strategy_not_qualified = '策略未通过验证 · 暂停新增交易';
signalNames.direction_filtered = '本根信号不符合账户的方向规则';
signalNames.swing_confirmation_blocked = '等待4小时波段方向确认';
signalNames.entry_signal_invalidated = '突破已失效，等待新小时信号';
signalNames.entry_timing_blocked = '入场价格超出信号有效范围';
signalNames.stock_event_blocked = '财报事件限制新增交易';
const blockerNames = {prior_5m_volume:'此前5分钟成交量为0或缺失', funding:'资金费超限',
  mark_index_basis:'标记价与指数价偏离超限', book_mark_basis:'盘口与标记价偏离超限',
  entry_gap:'追价距离超限', bid_ask_spread:'买卖价差超过入场上限',
  swing_4h_warmup:'等待足够的已收盘4小时行情', swing_hourly_alignment:'小时行情时间不完整',
  swing_hourly_data_unavailable:'小时行情暂不可用',
  swing_hourly_gap:'小时行情有缺口', swing_4h_indicators_unavailable:'4小时趋势数据暂不可用',
  swing_4h_ema_ordering:'4小时趋势与入场方向不一致',
  swing_4h_price_slow:'4小时价格尚未确认方向', swing_4h_spread_too_small:'4小时趋势幅度不足',
  breakout_reference_invalidated:'盘口已退回原突破位内，信号撤销',
  breakout_mark_invalidated:'标记价已退回原突破位内，信号撤销',
  breakout_fill_invalidated:'成交价已退回原突破位内，信号撤销',
  entry_timing_reference_gap_exceeded:'盘口追价超过0.5 ATR',
  entry_timing_fill_gap_exceeded:'成交追价超过0.5 ATR',
  entry_timing_reference_extension_exceeded:'盘口距突破位超过1 ATR',
  entry_timing_fill_extension_exceeded:'成交距突破位超过1 ATR',
  entry_timing_trigger_missing:'原突破位资料缺失，等待新信号',
  entry_timing_trigger_invalid:'原突破位资料无效，等待新信号'};
const stockEventReasonName = reason => ({
  stock_event_blackout: '处于财报电话会前后观察窗口',
  stock_event_post_event_bars_unavailable: '等待财报后两根完整小时行情',
  stock_event_calendar_expired: '财报日历已到复核期限，暂停新增',
  stock_event_calendar_unavailable: '财报日历暂不可用，暂停新增',
  stock_event_calendar_not_yet_available: '财报日历尚未到可用时间',
  stock_event_calendar_outside_coverage: '当前时间超出财报日历覆盖期',
  stock_event_symbol_not_covered: '财报日历尚未覆盖此标的',
}[reason] || (String(reason).startsWith('stock_event_') ? '财报日历需要复核，暂停新增' : ''));

function stockEventDetails(account) {
  const gate = account.stock_event_risk;
  if (!gate?.enabled) return '';
  const label = {clear: '当前窗口外', blackout: '暂停新增', unavailable: '日历需要复核',
    not_active_yet: '等待生效'}[gate.status] || '等待检查';
  const pieces = [`财报风控：${label}`];
  if (gate.next_event?.anchor_time_ms) pieces.push(`下一电话会 ${localTime(gate.next_event.anchor_time_ms)}`);
  const resume = Math.max(0, ...(gate.blocked_events || []).map(event => Number(event.earliest_resume_at_ms) || 0));
  if (resume) pieces.push(`最早复核 ${localTime(resume)}，并等待两根完整小时行情`);
  if (gate.valid_until_ms) pieces.push(`日历有效至 ${localTime(gate.valid_until_ms)}`);
  if (!gate.allowed) pieces.push('保护退出继续');
  return pieces.join(' · ');
}
const dataRoot = '/data/parallel_simulation/btc_memory_stocks_latest_1000_each_20261007';
const stockSymbols = {mu: 'MUUSDT', sndk: 'SNDKUSDT', skhynix: 'SKHYNIXUSDT'};
const reasonName = reason => ({protective_stop: '保护止损', one_r_partial: '1R分批止盈',
  profit_target: '目标止盈', fresh_closed_5m_signal: '5分钟信号入场',
  fresh_closed_15m_signal: '15分钟信号入场', fresh_closed_1h_signal: '1小时波段信号入场',
  fresh_closed_4h_signal: '4小时波段信号入场', account_hard_stop: '账户回撤退出'}[reason] || reason || '—');
const btcBlockerName = (reason, account) => ({
  'current_factor:missing_or_stale_factors': '必需因子数据缺失或过期，暂停新增',
  'current_factor:direction_conflict': '因子方向与开仓方向冲突',
  'current_factor:market_stress': '市场风险超过入场上限',
  current_public_sources_unavailable: '新闻或日历来源暂不可用',
  current_event_blocks_entries: '当前事件风险限制新增',
  execution_entry_context_missing: '入场检查资料缺失',
  execution_clock_unavailable: '交易所时间暂不可用',
  llm_trade_gate_rejected: '模型审核未通过，暂停新增',
  llm_trade_gate_unavailable: '模型审核暂不可用，暂停新增',
  btc_initial_entry_origin_expired: '原始小时信号已超过首次入场窗口，等待新信号',
  btc_initial_entry_origin_missing: '原始小时信号时间缺失，暂停首次入场',
  btc_initial_entry_origin_invalid: '原始小时信号时间无效，暂停首次入场',
  btc_initial_entry_origin_future: '原始小时信号尚未可用，暂停首次入场',
  btc_entry_timing_policy_invalid: '入场时效配置无效，暂停首次入场',
  profit_target_signal_consumed: '该小时信号已止盈，等待新信号',
}[reason] || btcPullbackReasonName(reason, account) || signalNames[reason] || reason);

function btcProfitMarkup(account) {
  const p = account.profit_exit_status;
  if (!p?.enabled) return '';
  if (p.status === 'invalid_policy') return '<p class="parallel-age">止盈配置无效，等待修复</p>';
  if (!Number.isFinite(p.target_margin_return) || p.target_margin_return < .6) return '';
  const target = Number((p.target_margin_return * 100).toFixed(2));
  const details = [`保证金净收益目标${target}%`];
  const waiting = {funding_unavailable: '资金费数据暂不可用，等待完整数据后判断止盈',
    ledger_unavailable: '成交账本暂不可核验，等待完整账本后判断止盈',
    closed_target: '已按目标止盈，等待新信号'}[p.status];
  if (waiting) details.push(waiting);
  else {
    if (Number.isFinite(p.projected_net_roi_pct)) details.push(`预计平仓净ROI ${p.projected_net_roi_pct.toFixed(2)}%`);
    if (Number.isFinite(p.target_mark_price) && p.target_mark_price > 0) details.push(`目标标记价 ${money(p.target_mark_price)} USDT`);
  }
  return `<p class="parallel-age">${details.map(escape).join('<br>')}</p>`;
}

function btcPullbackStatus(p) {
  const long = p?.side === 'long' || p?.side === 1;
  return {
    waiting_retrace: long ? '等待回踩' : p?.side === 'short' || p?.side === -1 ? '等待反弹' : '等待回踩或反弹',
    waiting_confirmation: '已到候选区，等待止跌/受阻确认',
    ready: '已确认，检查实际成交价',
    expired: '回踩计划失效，等待新小时信号',
    cancelled: '回踩计划失效，等待新小时信号',
    structure_invalidated: '回踩计划失效，等待新小时信号',
    market_data_unavailable: '入场行情数据暂不可用，等待完整数据',
    data_unavailable: '入场行情数据暂不可用，等待完整数据',
    price_unavailable: '当前执行价暂不可用，等待完整报价',
    invalid_policy: '回踩入场配置无效，等待修复',
    origin_missing: '原始小时信号资料缺失，等待新小时信号',
    no_structure: '暂未找到有效回踩结构，继续等待',
    invalid_origin: '原始小时信号资料无效，等待新小时信号',
    invalid_source: '原始小时信号资料无效，等待新小时信号',
  }[p?.status] || '';
}

function btcPullbackReasonName(reason, account) {
  if (typeof reason !== 'string' || !reason.startsWith('btc_pullback_')) return '';
  const status = reason.replace(/^btc_pullback_entry_/, '').replace(/^btc_pullback_/, '');
  const known = {
    target_flat: '回踩计划失效，等待新小时信号',
    source_changed: '回踩计划失效，等待新小时信号',
    source_consumed: '该小时入场机会已使用，等待新小时信号',
    origin_expired: '回踩计划失效，等待新小时信号',
    origin_future: '原始小时信号尚未可用，等待信号生效',
    no_legal_structure: '暂未找到有效回踩结构，继续等待',
    chase_exceeded: '预计执行价超过追价上限，等待价格回归',
    rejection_not_held: '价格尚未守住确认区，继续等待',
    mark_unavailable: '当前执行价暂不可用，等待完整报价',
    projected_fill_invalid: '预计执行价暂不可用，等待完整报价',
    '5m_gap': '5分钟确认行情有缺口，等待完整数据',
    hourly_gap: '原始小时行情有缺口，等待完整历史',
    hourly_history_insufficient: '原始小时行情不足，等待完整历史',
    original_hourly_prefix_missing: '原始小时行情缺失，等待完整历史',
    atr_invalid: '入场波动数据暂不可用，等待完整数据',
    policy_changed: '回踩入场配置已变化，等待新小时信号',
  }[status] || btcPullbackStatus({side: account?.btc_pullback_entry?.side, status});
  if (known) return known;
  if (status.startsWith('pending_')) return '回踩计划资料无效，等待新小时信号';
  if (/policy|round_steps|repeated_digits|numeric_|tick_size|slippage|price_corridor|original_entry_window/.test(status)) {
    return '回踩入场配置无效，等待修复';
  }
  if (/source|origin|direction|position_id|anchor/.test(status)) return '原始小时信号资料无效，等待新小时信号';
  return '回踩入场资料暂不可核验，等待完整数据';
}

function btcPullbackMarkup(account) {
  const p = account.btc_pullback_entry;
  if (!p?.enabled) return '';
  const label = btcPullbackStatus(p);
  if (!label) return '';
  const details = [label];
  const usable = ['waiting_retrace', 'waiting_confirmation', 'ready'].includes(p.status);
  if (Number.isFinite(p.reference_price) && p.reference_price > 0
      && !['invalid_policy', 'origin_missing', 'invalid_origin', 'invalid_source'].includes(p.status)) {
    details.push(`原始策略参考价 ${money(p.reference_price)} USDT`);
  }
  if (usable) {
    if (Number.isFinite(p.candidate_low) && p.candidate_low > 0
        && Number.isFinite(p.candidate_high) && p.candidate_high >= p.candidate_low) {
      details.push(`候选入场区 ${money(p.candidate_low)}—${money(p.candidate_high)} USDT`);
    }
    const levels = Array.isArray(p.numeric_levels) ? [...new Set(p.numeric_levels
      .map(level => typeof level === 'number' ? level : level?.price)
      .filter(price => Number.isFinite(price) && price > 0))] : [];
    if (levels.length) details.push(`整数/对子参考价 ${levels.map(money).join('、')} USDT`);
    if (Number.isFinite(p.projected_fill_price) && p.projected_fill_price > 0) {
      details.push(`预计执行价（含滑点）${money(p.projected_fill_price)} USDT`);
    }
    if (Number.isFinite(p.expires_at_ms) && p.expires_at_ms > 0) details.push(`有效至北京时间 ${localTime(p.expires_at_ms)}`);
  }
  return `<p class="parallel-age">${details.map(escape).join('<br>')}</p>`;
}

function btcEntryDetails(account) {
  const gate = account.execution_entry_gate;
  if (!gate) return '';
  const groups = {btc_momentum: 'BTC动量', positioning: '持仓与资金流', fed: '美联储',
    treasury: '美债', fx: '汇率', global_risk: '全球风险'};
  const sides = gate.by_side || (gate.side ? {[gate.side]: gate} : {});
  const details = Object.entries(sides).flatMap(([side, decision]) => {
    const factor = decision?.factor;
    const missing = factor?.missing_groups || [];
    const oil = decision?.factor_source_status?.oil || account.entry_source_status?.factors?.oil;
    const oilIgnored = factor?.ignored_features?.includes('oil') || oil?.ignored === true;
    const direction = {long: '多头', short: '空头', both: '多空'}[side] || side;
    const details = [];
    if (missing.length) {
      const missingNames = missing.map(name => {
        if (name !== 'global_risk' || !oilIgnored) return groups[name] || name;
        const features = factor.features || {};
        const required = Object.entries({sp500_change: '标普500变化', nasdaq_change: '纳斯达克变化', vix: 'VIX'})
          .filter(([key]) => Object.hasOwn(features, key) && features[key] === null).map(([, label]) => label);
        return `全球风险（${required.length ? required.join('、') : '其他必需数据'}）`;
      });
      details.push(`${direction}：${missingNames.join('、')}因子缺失或过期`);
    }
    if (oilIgnored || missing.includes('global_risk') && factor.features?.oil_change === null) {
      details.push(oilIgnored ? `${direction}：油价缺失或过期，本轮未计入；其他因子继续判断`
        : oil?.status === 'stale' ? '油价观测已过期' : '油价变化数据缺失或过期');
      if (Number.isFinite(oil?.latest_observed_at_ms)) {
        details.push(`油价最近观测：北京时间 ${localTime(oil.latest_observed_at_ms)}`);
      }
      if (Number.isFinite(oil?.age_ms) && Number.isFinite(oil?.max_age_ms)) {
        const days = ms => Number((ms / 86_400_000).toFixed(1));
        details.push(`观测距今 ${days(oil.age_ms)} 天 · 新鲜度上限 ${days(oil.max_age_ms)} 天`);
      }
      if (oil?.ok === false && oil.status !== 'stale') details.push('油价来源暂不可用');
    }
    return details;
  });
  if ((account.entry_blockers || []).some(reason => reason.startsWith('llm_trade_gate_')) && account.llm_trade_gate?.reason) {
    details.push(`模型审核理由：${account.llm_trade_gate.reason}`);
  }
  const timing = account.btc_entry_timing;
  if (timing?.enabled && timing.requires_new_direction && Number.isFinite(timing.origin_signal_age_seconds)) {
    details.push(`原始小时信号距今 ${(timing.origin_signal_age_seconds / 60).toFixed(1)} 分钟 · 首次入场窗口 ${timing.max_initial_entry_age_seconds / 60} 分钟`);
  }
  return details.join('；');
}

const factorAge = ms => !Number.isFinite(ms) || ms < 0 ? '—'
  : ms >= 86_400_000 ? `${(ms / 86_400_000).toFixed(1)} 天`
  : ms >= 3_600_000 ? `${(ms / 3_600_000).toFixed(1)} 小时`
  : `${(ms / 60_000).toFixed(1)} 分钟`;

export function factorFreshnessMarkup(diagnostics) {
  const factors = diagnostics?.factors;
  const heading = '<div class="panel-heading"><div><small>OBSERVATION / COLLECTION DATES</small><h2>宏观因子观测日期</h2></div></div>';
  if (!factors || typeof factors !== 'object' || Array.isArray(factors) || !Object.keys(factors).length) {
    return `${heading}<p class="experiment-description">尚无逐项日期记录，等待下一次因子检查。抓取成功不能证明观测是实时数据。</p>`;
  }
  const cadence = {hourly: '小时数据', settlement: '结算数据', daily: '日数据', weekly: '周数据', unknown: '频率未记录'};
  const statusName = {fresh: '现行时限内', stale: '观测已过期', missing: '观测缺失', not_used: '未直接用于模型'};
  const rows = Object.entries(factors).map(([id, factor]) => {
    const f = factor && typeof factor === 'object' ? factor : {};
    const status = statusName[f.status] || '状态未记录';
    const role = f.group == null ? '未直接使用' : f.group_enabled === false ? '因子组未启用'
      : f.optional ? '可选输入' : '因子组必需输入';
    const fetch = f.fetch_ok === true ? '最近抓取成功' : f.fetch_ok === false ? '最近抓取失败' : '抓取结果未单独记录';
    const notes = [...(Array.isArray(f.notes) ? f.notes : [])];
    if (f.ignored) notes.push('本次判断未计入');
    const observed = Number.isFinite(f.latest_observed_at_ms) ? localTime(f.latest_observed_at_ms) : '未取得可用观测';
    const dateLabel = f.fetch_success_time_kind === 'legacy_usable_collection_record' ? '成功接收（旧版可用记录）' : '成功接收';
    const dates = [`${dateLabel} ${Number.isFinite(f.last_fetch_success_ms) ? localTime(f.last_fetch_success_ms) : '未记录'}`,
      `最近尝试 ${Number.isFinite(f.last_fetch_attempt_ms) ? localTime(f.last_fetch_attempt_ms) : '未记录'}`];
    return `<tr><td><strong>${escape(f.name || id)}</strong><small class="stock-table-symbol">${escape(cadence[f.cadence] || cadence.unknown)} · ${escape(role)}</small></td>`
      + `<td>${escape(observed)}<small class="stock-table-symbol">距检查时间 ${escape(factorAge(f.age_ms))}</small></td>`
      + `<td>${Number.isFinite(f.max_age_ms) ? escape(factorAge(f.max_age_ms)) : '未设交易阈值'}</td>`
      + `<td class="${['stale', 'missing'].includes(f.status) ? 'parallel-warning' : ''}">${escape(status)}<small class="stock-table-symbol">${notes.map(escape).join('；')}</small></td>`
      + `<td>${escape(fetch)}</td><td>${dates.map(escape).join('<br>')}</td></tr>`;
  }).join('');
  return `${heading}<p class="experiment-description">检查记录：北京时间 ${escape(localTime(diagnostics.checked_at_ms))}。日/周数据含发布延迟；“现行时限内”只表示未超过策略容忍期，来源抓取成功不等于实时观测。下列日期均为北京时间。</p>`
    + `<div class="table-wrap factor-freshness-scroll"><table><thead><tr><th>因子 / 发布频率</th><th>实际观测时间 / 滞后</th><th>现行允许滞后</th><th>数据新鲜度 / 用途</th><th>来源抓取</th><th>抓取日期</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function btcFactorFreshness(account) {
  const gate = account?.execution_entry_gate || {};
  return [gate, ...Object.values(gate.by_side || {})]
    .map(decision => decision?.factor_data_freshness)
    .filter(value => value && typeof value === 'object')
    .sort((a, b) => (b.checked_at_ms || 0) - (a.checked_at_ms || 0))[0];
}

export function stockTradingMarkup(ledgers) {
  const positions = [], trades = [], errors = [];
  for (const {id, state, error} of ledgers) {
    if (error || !state) {
      errors.push(`${names[id]}：${error || '账本不可用'}`);
      positions.push(`<tr><td>${names[id]}</td><td colspan="6">持仓明细暂不可用</td></tr>`);
      continue;
    }
    const p = state.position;
    const pnl = Number.isFinite(state.equity) && Number.isFinite(state.wallet_balance) ? state.equity - state.wallet_balance : null;
    positions.push(p
      ? `<tr><td>${names[id]}</td><td class="${p.direction > 0 ? 'positive' : 'negative'}">${p.direction > 0 ? '做多' : '做空'}</td><td>${money(p.qty)}</td><td>${money(p.entry)}</td><td>${money(state.last_mark_price)}</td><td class="${pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : ''}">${money(pnl)}</td><td>${money(p.stop)}</td></tr>`
      : `<tr><td>${names[id]}</td><td>空仓</td><td>0</td><td>—</td><td>${money(state.last_mark_price)}</td><td>—</td><td>—</td></tr>`);
    for (const fill of state.fills || []) trades.push({...fill, accountId: id});
  }
  trades.sort((a, b) => Number(b.time_ms || Date.parse(b.time_utc)) - Number(a.time_ms || Date.parse(a.time_utc)));
  const rows = trades.slice(0, 50).map(fill => `<tr><td>${localTime(fill.time_utc || fill.time_ms)}</td><td>${names[fill.accountId]}</td><td class="${fill.side === 'BUY' ? 'positive' : 'negative'}">${fill.side === 'BUY' ? '买入' : fill.side === 'SELL' ? '卖出' : escape(fill.side)}</td><td>${money(fill.price)}</td><td>${money(fill.qty)}</td><td>${money(fill.fee)}</td><td>${escape(reasonName(fill.reason))}${fill.partial ? ' · 部分成交' : ''}</td></tr>`).join('');
  return {
    positions: positions.join(''),
    trades: rows || `<tr><td colspan="7">${errors.length ? '成交明细暂不可用' : '三个账户暂无模拟成交记录'}</td></tr>`,
    status: errors.length ? `部分账本读取失败：${errors.join('；')}` : `已读取三个独立账本 · 显示最近${Math.min(trades.length, 50)}笔模拟成交 · 每10秒刷新`,
    unavailable: errors.length > 0,
  };
}

export function createPaperAccounts(root, fetcher = globalThis.fetch) {
  const group = root.dataset.accountGroup;
  if (!["btc", "stocks"].includes(group)) throw new Error("账户分组无效");
  const ids = group === "btc" ? ["btc"] : ["mu", "sndk", "skhynix"];
  const label = group === "btc" ? "BTC" : "股票";
  const $ = selector => root.querySelector(`[data-paper-field="${selector.slice(1)}"]`);
  let pending = false;
  async function refresh() {
    if (pending) return;
    pending = true;
    try {
      const response = await fetcher(`${dataRoot}/status.json?v=${Date.now()}`, {cache:'no-store'});
      if (!response.ok) throw new Error(`${label}账户运行记录暂时无法读取`);
      const result = await response.json();
      const age = (Date.now() - new Date(result.updated_at_utc).getTime()) / 1000;
      const active = result.running && age >= -5 && age < 120;
      $('#start').textContent = `账户起点：北京时间 ${localTime(result.started_at_utc)} · 每账户 1,000 USDT · 上限 10×`;
      $('#heartbeat').textContent = `${active ? '后台运行中' : '进程心跳停止或过期'} · 更新 ${localTime(result.updated_at_utc)}`;
      $('#heartbeat').classList.toggle('parallel-warning', !active);
      const views = ids.map(id => result.accounts?.[id]);
      $('#total').textContent = `${label}权益 ` + (views.every(a => Number.isFinite(a?.equity)) ? money(views.reduce((sum,a)=>sum+a.equity,0)) : '—') + ` USDT / 初始 ${money(ids.length * 1000)}`;
      $('#cards').innerHTML = ids.map(id => {
        const name = names[id];
        const a = result.accounts?.[id] || {};
        const accountAge = (Date.now() - new Date(a.checked_at_utc).getTime()) / 1000;
        const healthy = active && accountAge >= -5 && accountAge < 120 && a.status === 'healthy';
        const status = !active || accountAge > 120 ? '心跳过期' : a.status === 'healthy' ? '运行正常' : a.status === 'waiting_for_first_forward_candle' ? '等待首根前瞻K线' : a.status === 'degraded' ? '数据降级 · 暂停新增' : '初始化中';
        const error = Object.entries(a.errors || {}).map(([k,v]) => `${escape(k)}: ${escape(v)}`).join('<br>');
        const family = {breakout:'趋势突破', donchian20:'20根收盘线突破', ema_transition:'均线交叉', trend_pullback:'趋势回调', range_reversion:'震荡回归'}[a.signal_family] || '策略更新中';
        const direction = {both:'双向', long:'只做多', short:'只做空'}[a.entry_direction] || '—';
        const blockers = (a.entry_blockers || []).map(k => id === 'btc' ? btcBlockerName(k, a) : blockerNames[k] || stockEventReasonName(k) || signalNames[k] || k).join('；');
        const btcDetails = id === 'btc' ? btcEntryDetails(a) : '';
        const signalLabel = id === 'btc' && a.signal_status === 'entry_blocked' ? '入场条件未通过 · 暂停新增' : signalNames[a.signal_status] || '等待状态更新';
        const volume = a.entry_checks ? `此前5分钟成交量 ${money(a.entry_checks.prior_5m_volume)} · 入场检查 ${localTime(a.entry_checks.checked_at_ms)}` : '';
        const spread = a.entry_checks?.max_spread_fraction != null ? `买卖价差 ${(a.entry_checks.spread_fraction * 100).toFixed(4)}% · 入场上限 ${(a.entry_checks.max_spread_fraction * 100).toFixed(4)}%` : '';
        const qualification = a.entry_qualification?.approved_for_forward_simulation === false ? a.entry_qualification.reason : '';
        const confirmation = a.swing_confirmation;
        const swing = confirmation?.enabled ? `4小时确认：${{1:'多头', '-1':'空头', 0:'趋势未形成'}[confirmation.direction] || '等待完整行情'}${Number.isFinite(confirmation.ema_spread_fraction) ? ' · 均线间距 '+(confirmation.ema_spread_fraction * 100).toFixed(2)+'%' : ''}${confirmation.confirmation_close_ms ? ' · 收盘 '+localTime(confirmation.confirmation_close_ms) : ''}` : '';
          const events = id !== 'btc' ? stockEventDetails(a) : '';
          const signal = `<p class="parallel-age">${id !== 'btc' ? escape(family)+' · '+escape(direction)+' · '+escape(a.signal_timeframe || '—')+'<br>' : ''}${escape(signalLabel)}${blockers && !blockers.includes(signalLabel) ? '<br>'+escape(blockers) : ''}${swing ? '<br>'+escape(swing) : ''}${events ? '<br>'+escape(events) : ''}${btcDetails ? '<br>'+escape(btcDetails) : ''}${qualification ? '<br>'+escape(qualification) : ''}${volume ? '<br>'+escape(volume) : ''}${spread ? '<br>'+escape(spread) : ''}${id !== 'btc' ? '<br>下一根收盘 '+localTime(a.next_signal_time_ms) : ''}</p>`;
        const p = a.profit_exit_status;
        const profit = id === 'btc' ? btcProfitMarkup(a) : p ? `<p class="parallel-age">分批止盈：${p.split_skipped ? '数量不足以拆分 · 整仓保护' : p.stage_done ? '首段完成 · 余仓跟踪保护' : p.armed ? '首段部分成交 · 继续完成减仓' : p.trigger_observed_at_ms ? '已触发1R · 等待可成交数量' : '等待达到1R'}<br>首段已退出 ${money(p.filled_qty)} · 当前保护价 ${money(p.stop)}</p>` : '';
        const pullback = id === 'btc' ? btcPullbackMarkup(a) : '';
        const entryPrice = id === 'btc' && Number.isFinite(a.position_qty) && a.position_qty !== 0
          && Number.isFinite(a.entry_price) && a.entry_price > 0 ? `<div><dt>实际持仓均价</dt><dd>${money(a.entry_price)}</dd></div>` : '';
        return `<article class="panel parallel-card"><h2>${name}</h2><span class="parallel-state ${healthy ? '' : 'parallel-warning'}">${status}</span><div class="parallel-equity">${money(a.equity)}</div><small>USDT · 收益 ${a.return_pct == null ? '—' : Number(a.return_pct).toFixed(4)+'%'}</small><dl><div><dt>仓位</dt><dd>${a.position_qty == null ? '—' : a.position_qty === 0 ? '空仓' : escape(a.position_qty)}</dd></div><div><dt>模拟成交次数</dt><dd>${a.fill_count_total ?? '—'}</dd></div><div><dt>累计手续费</dt><dd>${money(a.fees_paid)}</dd></div><div><dt>累计资金费损益</dt><dd>${money(a.funding_pnl)}</dd></div><div><dt>最大回撤</dt><dd>${a.max_drawdown_pct == null ? '—' : Number(a.max_drawdown_pct).toFixed(4)+'%'}</dd></div><div><dt>标记价</dt><dd>${money(a.last_mark_price)}</dd></div>${entryPrice}</dl>${pullback}${profit}<details class="account-signal-details"><summary>策略与入场检查</summary>${signal}</details><p class="parallel-age">检查 ${localTime(a.checked_at_utc)}</p>${error ? `<p class="parallel-error">${error}</p>` : ''}</article>`;
      }).join('');
      if (group === 'btc' && $('#factor-freshness')) {
        $('#factor-freshness').innerHTML = factorFreshnessMarkup(btcFactorFreshness(result.accounts?.btc));
      }
      if (group === 'stocks' && $('#positions') && $('#trades')) {
        const results = await Promise.allSettled(ids.map(async id => {
          const response = await fetcher(`${dataRoot}/${id}/state.json?v=${Date.now()}`, {cache: 'no-store'});
          if (!response.ok) throw new Error('账本读取失败');
          const state = await response.json();
          if (state.mode !== 'simulation' || state.places_orders !== false || state.symbol !== stockSymbols[id]
              || !Array.isArray(state.fills) || !Object.hasOwn(state, 'position')) throw new Error('账本格式无效');
          return {id, state};
        }));
        const markup = stockTradingMarkup(results.map((result, i) => result.status === 'fulfilled'
          ? result.value : {id: ids[i], error: result.reason.message}));
        $('#positions').innerHTML = markup.positions;
        $('#trades').innerHTML = markup.trades;
        $('#trade-status').textContent = markup.status;
        $('#trade-status').classList.toggle('parallel-warning', markup.unavailable);
      }
      const comparison = group === 'stocks' ? result.profit_exit_comparison : null;
      if ($('#comparison')) $('#comparison').hidden = !comparison;
      if (comparison) {
        $('#comparison-start').textContent = `对照起点：北京时间 ${localTime(comparison.activated_at_utc)}。表内单位为USDT，净值变化包含浮盈；现金变化计入已实现损益、费用和资金费。`;
        $('#comparison-rows').innerHTML = ['mu','sndk','skhynix'].map(id => {
          const a = result.accounts?.[id], c = comparison.controls?.[id], b = comparison.baseline?.[id];
          const fresh = v => v?.status === 'healthy' && active && Date.now()-new Date(v.checked_at_utc).getTime() < 120000 && Date.now()-new Date(v.checked_at_utc).getTime() >= -5000;
          const delta = (v,key) => fresh(v) && v[key] != null && b?.[key] != null ? v[key]-b[key] : null;
          const next = delta(a,'equity'), old = delta(c,'equity');
          return `<tr><td>${names[id]}</td><td>${money(next)}</td><td>${money(old)}</td><td>${money(next != null && old != null ? next-old : null)}</td><td>${money(delta(a,'wallet_balance'))}</td><td>${money(delta(c,'wallet_balance'))}</td></tr>`;
        }).join('');
      }
    } catch (error) {
      $('#heartbeat').textContent = error.message;
      $('#heartbeat').classList.add('parallel-warning');
      $('#total').textContent = '当前权益暂不可用';
      $('#cards').textContent = '等待运行记录恢复';
      if (group === 'btc' && $('#factor-freshness')) {
        $('#factor-freshness').innerHTML = factorFreshnessMarkup(null);
      }
      if ($('#comparison')) $('#comparison').hidden = true;
      if ($('#positions')) $('#positions').innerHTML = '<tr><td colspan="7">持仓数据暂不可用，等待恢复</td></tr>';
      if ($('#trades')) $('#trades').innerHTML = '<tr><td colspan="7">成交数据暂不可用，等待恢复</td></tr>';
      if ($('#trade-status')) $('#trade-status').textContent = error.message;
    } finally {
      pending = false;
    }
  }
  const ready = refresh();
  const viewWindow = root.ownerDocument?.defaultView;
  const interval = viewWindow?.setInterval(refresh, 10_000);
  return {refresh, ready, destroy: () => viewWindow?.clearInterval(interval)};
}

if (typeof document !== "undefined") {
  document.querySelectorAll('[data-account-group]').forEach(root => createPaperAccounts(root));
}
