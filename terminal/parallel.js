const $ = s => document.querySelector(s);
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
async function refresh() {
  try {
    const response = await fetch('/data/parallel_simulation/btc_memory_stocks_1000_each_10x_20261005/status.json?v=' + Date.now(), {cache:'no-store'});
    if (!response.ok) throw new Error('四账户运行记录暂时无法读取');
    const result = await response.json();
    const age = (Date.now() - new Date(result.updated_at_utc).getTime()) / 1000;
    const active = result.running && age >= -5 && age < 120;
    $('#start').textContent = `共同起点：北京时间 ${localTime(result.started_at_utc)} · 每账户 1,000 USDT · 上限 10×`;
    $('#heartbeat').textContent = `${active ? '后台运行中' : '进程心跳停止或过期'} · 更新 ${localTime(result.updated_at_utc)}`;
    $('#heartbeat').classList.toggle('parallel-warning', !active);
    const views = Object.values(result.accounts || {});
    $('#total').textContent = '总权益 ' + (views.length === 4 && views.every(a => a.equity != null) ? money(views.reduce((sum,a)=>sum+a.equity,0)) : '—') + ' USDT / 初始 4,000';
    $('#cards').innerHTML = Object.entries(names).map(([id, name]) => {
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
      const qualification = a.entry_qualification?.reason || '';
      const signal = `<p class="parallel-age">${id !== 'btc' ? escape(family)+' · '+escape(direction)+' · '+escape(a.signal_timeframe || '—')+'<br>' : ''}${escape(signalNames[a.signal_status] || '等待状态更新')}${blockers && !blockers.includes(signalNames[a.signal_status]) ? '<br>'+escape(blockers) : ''}${qualification ? '<br>'+escape(qualification) : ''}${volume ? '<br>'+escape(volume) : ''}${spread ? '<br>'+escape(spread) : ''}${id !== 'btc' ? '<br>下一根收盘 '+localTime(a.next_signal_time_ms) : ''}</p>`;
      const p = a.profit_exit_status;
      const profit = p ? `<p class="parallel-age">分批止盈：${p.split_skipped ? '数量不足以拆分 · 整仓保护' : p.stage_done ? '首段完成 · 余仓跟踪保护' : p.armed ? '首段部分成交 · 继续完成减仓' : p.trigger_observed_at_ms ? '已触发1R · 等待可成交数量' : '等待达到1R'}<br>首段已退出 ${money(p.filled_qty)} · 当前保护价 ${money(p.stop)}</p>` : '';
      return `<article class="panel parallel-card"><h2>${name}</h2><span class="parallel-state ${healthy ? '' : 'parallel-warning'}">${status}</span><div class="parallel-equity">${money(a.equity)}</div><small>USDT · 收益 ${a.return_pct == null ? '—' : Number(a.return_pct).toFixed(4)+'%'}</small><dl><div><dt>仓位</dt><dd>${a.position_qty == null ? '—' : a.position_qty === 0 ? '空仓' : escape(a.position_qty)}</dd></div><div><dt>模拟成交次数</dt><dd>${a.fill_count_total ?? '—'}</dd></div><div><dt>累计手续费</dt><dd>${money(a.fees_paid)}</dd></div><div><dt>累计资金费损益</dt><dd>${money(a.funding_pnl)}</dd></div><div><dt>最大回撤</dt><dd>${a.max_drawdown_pct == null ? '—' : Number(a.max_drawdown_pct).toFixed(4)+'%'}</dd></div><div><dt>标记价</dt><dd>${money(a.last_mark_price)}</dd></div></dl>${profit}${signal}<p class="parallel-age">检查 ${localTime(a.checked_at_utc)}</p>${error ? `<p class="parallel-error">${error}</p>` : ''}</article>`;
    }).join('');
    const comparison = result.profit_exit_comparison;
    $('#comparison').hidden = !comparison;
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
    $('#comparison').hidden = true;
  }
}
refresh();
setInterval(refresh, 10_000);
