const $ = s => document.querySelector(s);
const names = {btc: 'BTC', mu: '美光 · MU', sndk: '闪迪 · SNDK', skhynix: '海力士 · SKHYNIX'};
const money = n => n == null ? '—' : Number(n).toLocaleString('en-US', {minimumFractionDigits: 4, maximumFractionDigits: 4});
const escape = s => String(s ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const localTime = s => s ? new Date(s).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai', hour12: false}) : '—';
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
      return `<article class="panel parallel-card"><h2>${name}</h2><span class="parallel-state ${healthy ? '' : 'parallel-warning'}">${status}</span><div class="parallel-equity">${money(a.equity)}</div><small>USDT · 收益 ${a.return_pct == null ? '—' : Number(a.return_pct).toFixed(4)+'%'}</small><dl><div><dt>仓位</dt><dd>${a.position_qty == null ? '—' : a.position_qty === 0 ? '空仓' : escape(a.position_qty)}</dd></div><div><dt>模拟成交次数</dt><dd>${a.fill_count_total ?? '—'}</dd></div><div><dt>累计手续费</dt><dd>${money(a.fees_paid)}</dd></div><div><dt>累计资金费损益</dt><dd>${money(a.funding_pnl)}</dd></div><div><dt>最大回撤</dt><dd>${a.max_drawdown_pct == null ? '—' : Number(a.max_drawdown_pct).toFixed(4)+'%'}</dd></div><div><dt>标记价</dt><dd>${money(a.last_mark_price)}</dd></div></dl><p class="parallel-age">检查 ${localTime(a.checked_at_utc)}</p>${error ? `<p class="parallel-error">${error}</p>` : ''}</article>`;
    }).join('');
  } catch (error) {
    $('#heartbeat').textContent = error.message;
    $('#heartbeat').classList.add('parallel-warning');
    $('#total').textContent = '当前权益暂不可用';
    $('#cards').textContent = '等待运行记录恢复';
  }
}
refresh();
setInterval(refresh, 10_000);
