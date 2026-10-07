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
const blockerNames = {prior_5m_volume:'此前5分钟成交量为0或缺失', funding:'资金费超限',
  mark_index_basis:'标记价与指数价偏离超限', book_mark_basis:'盘口与标记价偏离超限',
  entry_gap:'追价距离超限', bid_ask_spread:'买卖价差超过入场上限'};
const dataRoot = '/data/parallel_simulation/btc_memory_stocks_latest_1000_each_20261007';
const stockSymbols = {mu: 'MUUSDT', sndk: 'SNDKUSDT', skhynix: 'SKHYNIXUSDT'};
const reasonName = reason => ({protective_stop: '保护止损', one_r_partial: '1R分批止盈',
  profit_target: '目标止盈', fresh_closed_5m_signal: '5分钟信号入场',
  fresh_closed_15m_signal: '15分钟信号入场', account_hard_stop: '账户回撤退出'}[reason] || reason || '—');

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
        const family = {breakout:'趋势突破', ema_transition:'均线交叉', trend_pullback:'趋势回调', range_reversion:'震荡回归'}[a.signal_family] || '策略更新中';
        const direction = {both:'双向', long:'只做多', short:'只做空'}[a.entry_direction] || '—';
        const blockers = (a.entry_blockers || []).map(k => blockerNames[k] || signalNames[k] || k).join('；');
        const volume = a.entry_checks ? `此前5分钟成交量 ${money(a.entry_checks.prior_5m_volume)} · 入场检查 ${localTime(a.entry_checks.checked_at_ms)}` : '';
        const spread = a.entry_checks?.max_spread_fraction != null ? `买卖价差 ${(a.entry_checks.spread_fraction * 100).toFixed(4)}% · 入场上限 ${(a.entry_checks.max_spread_fraction * 100).toFixed(4)}%` : '';
        const qualification = a.entry_qualification?.approved_for_forward_simulation === false ? a.entry_qualification.reason : '';
        const signal = `<p class="parallel-age">${id !== 'btc' ? escape(family)+' · '+escape(direction)+' · '+escape(a.signal_timeframe || '—')+'<br>' : ''}${escape(signalNames[a.signal_status] || '等待状态更新')}${blockers && !blockers.includes(signalNames[a.signal_status]) ? '<br>'+escape(blockers) : ''}${qualification ? '<br>'+escape(qualification) : ''}${volume ? '<br>'+escape(volume) : ''}${spread ? '<br>'+escape(spread) : ''}${id !== 'btc' ? '<br>下一根收盘 '+localTime(a.next_signal_time_ms) : ''}</p>`;
        const p = a.profit_exit_status;
        const profit = p ? `<p class="parallel-age">分批止盈：${p.split_skipped ? '数量不足以拆分 · 整仓保护' : p.stage_done ? '首段完成 · 余仓跟踪保护' : p.armed ? '首段部分成交 · 继续完成减仓' : p.trigger_observed_at_ms ? '已触发1R · 等待可成交数量' : '等待达到1R'}<br>首段已退出 ${money(p.filled_qty)} · 当前保护价 ${money(p.stop)}</p>` : '';
        return `<article class="panel parallel-card"><h2>${name}</h2><span class="parallel-state ${healthy ? '' : 'parallel-warning'}">${status}</span><div class="parallel-equity">${money(a.equity)}</div><small>USDT · 收益 ${a.return_pct == null ? '—' : Number(a.return_pct).toFixed(4)+'%'}</small><dl><div><dt>仓位</dt><dd>${a.position_qty == null ? '—' : a.position_qty === 0 ? '空仓' : escape(a.position_qty)}</dd></div><div><dt>模拟成交次数</dt><dd>${a.fill_count_total ?? '—'}</dd></div><div><dt>累计手续费</dt><dd>${money(a.fees_paid)}</dd></div><div><dt>累计资金费损益</dt><dd>${money(a.funding_pnl)}</dd></div><div><dt>最大回撤</dt><dd>${a.max_drawdown_pct == null ? '—' : Number(a.max_drawdown_pct).toFixed(4)+'%'}</dd></div><div><dt>标记价</dt><dd>${money(a.last_mark_price)}</dd></div></dl>${profit}<details class="account-signal-details"><summary>策略与入场检查</summary>${signal}</details><p class="parallel-age">检查 ${localTime(a.checked_at_utc)}</p>${error ? `<p class="parallel-error">${error}</p>` : ''}</article>`;
      }).join('');
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
