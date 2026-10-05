const $ = s => document.querySelector(s);
const escape = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money = n => n == null ? '—' : Number(n).toFixed(4);
const local = time => time ? new Date(time).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}) : '—';
const root = '/data/paper_trading/three_stocks_overnight_net60_20261005/';
const names = {MUUSDT:'美光 · MU',SNDKUSDT:'闪迪 · SNDK',SKHYNIXUSDT:'海力士 · SKHYNIX'};
const blockerNames = {no_pending_stock_signal:'等待启用后的新信号',nq_stale:'纳指K线已过期',nq_fetch_stale:'纳指获取记录已过期',nq_fetch_failed:'纳指获取失败',nq_trend_disagrees:'纳指方向不一致',nq_not_yet_observed:'等待纳指报价',forward_experiment_expired:'已到截止时间，停止新增',prior_5m_volume:'此前成交量不足',bid_ask_spread:'买卖价差超限',entry_gap:'追价距离超限'};
async function load(file) {
  const response = await fetch(root+file+'?v='+Date.now(),{cache:'no-store'});
  if(!response.ok) throw Error('模拟记录暂不可用');
  return response.json();
}
async function refresh() {
  try {
    const [status,result] = await Promise.all([load('status.json'),load('overnight_report.json')]);
    const age = Date.now()-new Date(status.updated_at_utc).getTime();
    const active = status.running && age>=-5000 && age<120000;
    const phase = status.phase==='completed'?'本轮已结算':status.phase==='closing'?'截止后结算中':active?'后台观察中':'心跳停止或过期';
    $('#heartbeat').textContent = `${phase} · 更新 ${local(status.updated_at_utc)} · 截止 ${local(status.end_utc)}`;
    $('#cards').innerHTML = Object.entries(names).map(([symbol,name])=> {
      let rows='';let details='';
      for(const [cost,label] of Object.entries({normal_cost:'正常成本',double_cost:'双倍成本'})) {
        const view=status.accounts?.[symbol]?.[cost]||{};
        const report=result.accounts?.[symbol]?.[cost]||{};
        const health=status.phase==='completed'?'记录已结束':active&&view.status==='healthy'?'健康':'数据或进程待检查';
        rows+=`<tr><td>${label}</td><td>${money(view.equity)}</td><td>${report.completed_positions??'—'}</td><td>${money(report.closed_position_net_pnl_usdt)}</td><td>${money(report.mean_closed_net_pnl_usdt)}</td></tr>`;
        const blockers=(view.entry_blockers||[]).map(x=>blockerNames[x]||x).join('；');
        const errors=Object.values(view.errors||{}).join('；');
        details+=`<p>${label}：${health} · 仓位 ${escape(view.position_qty??'—')}<br>回撤 ${money(view.max_drawdown_pct)}%${blockers?'<br>'+escape(blockers):''}${errors?'<br><span class="overnight-error">'+escape(errors)+'</span>':''}</p>`;
      }
      return `<article class="panel overnight-card"><h2>${name}</h2><table><thead><tr><th>成本</th><th>净值</th><th>平仓笔数</th><th>已平仓净收益</th><th>每笔净期望</th></tr></thead><tbody>${rows}</tbody></table>${details}<p>每个成本至少30笔才进入样本复核。零成交的净期望显示为“—”。</p></article>`;
    }).join('');
  }catch(error){
    $('#heartbeat').textContent=error.message;
    $('#cards').textContent='等待运行与报告记录恢复';
  }
}
refresh();
setInterval(refresh,10000);
