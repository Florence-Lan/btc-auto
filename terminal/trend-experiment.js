const names = {mu: '美光 · MU', sndk: '闪迪 · SNDK', skhynix: '海力士 · SKHYNIX'};
const ids = Object.keys(names);
const money = n => Number.isFinite(n) ? n.toLocaleString('en-US', {minimumFractionDigits: 4, maximumFractionDigits: 4}) : '—';
const escape = s => String(s ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[x]));
const localTime = t => t ? new Date(t).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai', hour12: false}) : '—';
const exitStatus = a => !a?.position ? '空仓' : a.position.pending_exit === 'trend_reversal_exit' ? '反转退出中'
  : a.trend_exit_status?.status === 'confirming' ? '反向趋势确认 1 / 2'
  : a.trend_exit_status?.status === 'unavailable' ? '等待完整趋势数据' : '持仓观察中';
const position = a => !a?.position ? '空仓' : `${a.position.direction > 0 ? '多' : '空'} ${money(a.position.qty)}`;

export function trendExperimentMarkup(result, now = Date.now()) {
  if (result.mode !== 'SIMULATION' || result.places_orders !== false) throw new Error('模拟实验状态格式无效');
  const freshTime = t => now - Date.parse(t) >= -5000 && now - Date.parse(t) < 120000;
  const active = result.running && freshTime(result.updated_at_utc);
  const fresh = a => active && a?.status === 'healthy' && freshTime(a.checked_at_utc);
  const delta = (a, baseline) => fresh(a) && Number.isFinite(a.equity) && Number.isFinite(baseline?.equity) ? a.equity - baseline.equity : null;
  const rows = ids.map(id => {
    const a = result.accounts?.[id]?.candidate, c = result.accounts?.[id]?.control;
    const next = delta(a, result.baseline?.[id]), old = delta(c, result.baseline?.[id]);
    return `<tr><td>${names[id]}</td><td>${money(next)}</td><td>${money(old)}</td><td>${money(next != null && old != null ? next - old : null)}</td><td>${fresh(a) ? position(a) : '数据待更新'}<small class="stock-table-symbol">${fresh(a) ? exitStatus(a) : ''}</small></td><td>${fresh(c) ? position(c) : '数据待更新'}</td><td>${a?.experiment_fill_count ?? '—'} / ${c?.experiment_fill_count ?? '—'}</td></tr>`;
  }).join('');
  const fills = ids.flatMap(id => ['candidate', 'control'].flatMap(arm =>
    (result.accounts?.[id]?.[arm]?.recent_fills || []).map(fill => ({id, arm, ...fill}))));
  fills.sort((a, b) => Number(b.time_ms) - Number(a.time_ms));
  const tradeRows = fills.slice(0, 30).map(f => `<tr><td>${localTime(f.time_utc || f.time_ms)}</td><td>${names[f.id]}</td><td>${f.arm === 'candidate' ? '反转退出' : '现行规则'}</td><td>${f.side === 'BUY' ? '买入' : '卖出'}</td><td>${money(f.price)}</td><td>${money(f.qty)}</td><td>${money(f.fee)}</td><td>${escape(({trend_reversal_exit: '趋势反转退出', protective_stop: '保护止损', one_r_partial: '1R分批止盈', fresh_closed_5m_signal: '5分钟信号入场', fresh_closed_15m_signal: '15分钟信号入场'})[f.reason] || f.reason)}</td></tr>`).join('');
  return {rows, tradeRows: tradeRows || '<tr><td colspan="8">实验已记录起点，等待新的模拟成交</td></tr>',
    active, status: `${active ? '实验运行中' : '实验心跳停止或过期'} · 起点 ${localTime(result.started_at_utc)} · 更新 ${localTime(result.updated_at_utc)}`};
}

export function createTrendExperiment(root, fetcher = globalThis.fetch) {
  const $ = name => root.querySelector(`[data-trend-field="${name}"]`);
  let pending = false;
  async function refresh() {
    if (pending) return;
    pending = true;
    try {
      const response = await fetcher(`/data/paper_trading/stock_trend_reversal_exit_20261006/status.json?v=${Date.now()}`, {cache: 'no-store'});
      if (!response.ok) throw new Error('实验状态暂不可用');
      const markup = trendExperimentMarkup(await response.json());
      $('status').textContent = markup.status;
      $('status').classList.toggle('parallel-warning', !markup.active);
      $('rows').innerHTML = markup.rows;
      $('trades').innerHTML = markup.tradeRows;
    } catch (error) {
      $('status').textContent = error.message;
      $('status').classList.add('parallel-warning');
      $('rows').innerHTML = '<tr><td colspan="7">等待实验数据恢复，原账户继续独立运行</td></tr>';
      $('trades').innerHTML = '<tr><td colspan="8">实验成交数据暂不可用</td></tr>';
    } finally { pending = false; }
  }
  const ready = refresh();
  const win = root.ownerDocument?.defaultView;
  const timer = win?.setInterval(refresh, 10_000);
  return {refresh, ready, destroy: () => win?.clearInterval(timer)};
}

if (typeof document !== 'undefined') document.querySelectorAll('[data-trend-experiment]').forEach(root => createTrendExperiment(root));
