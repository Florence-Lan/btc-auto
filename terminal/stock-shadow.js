const $ = selector => document.querySelector(selector);
const money = value => value == null ? '—' : Number(value).toFixed(4);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const local = value => value ? new Date(value).toLocaleString('zh-CN', {timeZone:'Asia/Shanghai',hour12:false}) : '—';
const reasons = {nq_fetch_failed:'纳指行情获取失败',nq_not_yet_observed:'等待已收到的纳指行情',nq_fetch_stale:'纳指获取记录已过期',nq_stale:'纳指K线已过期',nq_insufficient_history:'纳指指标预热中',nq_trend_disagrees:'纳指方向尚未一致',no_pending_stock_signal:'等待启用后的新信号',forward_experiment_expired:'30天记录期结束，暂停新增'};
async function refresh() {
  try {
    const response = await fetch('/data/paper_trading/sndk_trend_volume_nq_net60_shadow_20261005_v2/status.json?v='+Date.now(), {cache:'no-store'});
    if (!response.ok) throw new Error('模拟状态暂不可用');
    const status = await response.json();
    const age = Date.now()-new Date(status.updated_at_utc).getTime();
    const active = status.running && age>=-5000 && age<120000;
    $('#heartbeat').textContent = `${active?'后台运行中':'心跳停止或过期'} · 北京时间 ${local(status.updated_at_utc)}`;
    $('#cards').innerHTML = Object.entries({normal_cost:'正常成本',double_cost:'双倍成本'}).map(([key,name])=> {
      const account=status.accounts?.[key]||{};
      const gate=account.external_entry_gate||{};
      const blocker=(gate.reasons||[]).map(reason=>reasons[reason]||reason).join('；');
      const nq=gate.factors?.nq;
      const signal=account.position_qty ? '持仓保护中' : gate.allowed ? '信号有效，检查成交与风险条件' : blocker || '等待行情';
      const errors=Object.values(account.errors||{}).map(escape).join('；');
      return `<article class="panel shadow-card"><h2>${name}</h2><p>${active&&account.status==='healthy'?'运行正常':'等待新鲜数据'}</p><div class="shadow-value">${money(account.equity)} USDT</div><dl><div><dt>净值相对初始资金</dt><dd>${account.equity==null?'—':money(account.equity-1000)} USDT</dd></div><div><dt>仓位数量</dt><dd>${escape(account.position_qty??'—')}</dd></div><div><dt>模拟成交次数</dt><dd>${escape(account.fill_count_total??'—')}</dd></div><div><dt>最大回撤</dt><dd>${money(account.max_drawdown_pct)}%</dd></div><div><dt>累计手续费</dt><dd>${money(account.fees_paid)}</dd></div><div><dt>资金费损益</dt><dd>${money(account.funding_pnl)}</dd></div></dl><p>${escape(signal)}</p><p>纳指获取时间 ${local(gate.fetch_first_seen_ms)}${nq?'<br>已用纳指K线 '+local(nq.available_ms)+' · '+money(nq.close):''}</p>${errors?'<p class="shadow-error">'+errors+'</p>':''}</article>`;
    }).join('');
  } catch(error) {
    $('#heartbeat').textContent=error.message;
    $('#cards').textContent='等待模拟记录恢复';
  }
}
refresh();
setInterval(refresh,10000);
