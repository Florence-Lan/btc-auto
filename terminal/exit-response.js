export const SYMBOLS = ['MUUSDT','SNDKUSDT','SKHYNIXUSDT'];
export const VARIANTS = {baseline:'原规则',breakout_failure:'突破失效退出',giveback:'浮盈回吐保护',combined:'组合退出'};
const names = {MUUSDT:'MU',SNDKUSDT:'SNDK',SKHYNIXUSDT:'SK Hynix'};
const labels = {healthy:'运行中',degraded:'数据降级',starting:'启动中',holding:'持仓中',entered:'新入场',no_signal:'暂无信号',waiting_for_new_closed_bar:'等待新收盘K线',waiting_for_first_forward_candle:'等待新起点后的信号',cooldown:'冷却中',armed:'保护已激活',inactive:'未激活',not_armed:'未激活',tracking:'跟踪中',unavailable:'数据不可用',triggered:'已触发',exit_pending:'退出待成交',flat:'空仓',disabled:'未启用',observing:'观察中',baseline:'原规则',confirming:'等待收盘确认',latched:'退出触发已锁定'};
const columns = ['股票 / 规则','权益 USDT','已平仓净收益','完整平仓','采样最大回撤','费用 / 资金费','持仓数量','相对原规则','配对跟踪','当前状态'];
let selectedCost = 'normal';
let lastStatus = null;
let refreshSequence = 0;
const STATUS_PATH = '/data/paper_trading/stock_exit_response_shadow_20261023/status.json';

export function timestampMs(value) {
  if(typeof value==='number')return Number.isSafeInteger(value)&&value>0?value:null;
  if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})$/.test(value))return null;
  const result=Date.parse(value);return Number.isSafeInteger(result)&&result>0?result:null;
}
export function timeText(value) {
  const ms=timestampMs(value);if(ms===null)return '—';
  return new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(ms);
}
export function numberText(value, digits=2) {
  return typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{minimumFractionDigits:digits,maximumFractionDigits:digits}):'—';
}
function text(id,value) {document.getElementById(id).textContent=String(value);}
function translated(value) {return labels[value]||value||'—';}
function heartbeat(value) {return timestampMs(value?.heartbeat_at_ms??value?.heartbeat_at_utc??value?.heartbeat);}
function freshness(value,now,maxAge) {const stamp=heartbeat(value);return stamp===null?'心跳缺失':stamp>now?'未来心跳无效':now-stamp>maxAge?'心跳过期':null;}
function count(value) {return Number.isSafeInteger(value)&&value>=0?String(value):'—';}
function addCell(tr,index,value) {const td=document.createElement('td');td.setAttribute('data-label',columns[index]);td.textContent=String(value);tr.append(td);}
function exitText(value) {
  if(typeof value==='string')return translated(value);
  if(!value||typeof value!=='object')return '退出观察未报告';
  const details=[translated(value.status),value.triggered?'已触发':null,...(Array.isArray(value.reasons)?value.reasons:[])].filter(Boolean);
  for(const key of ['breakout_failure','giveback'])if(value[key]&&typeof value[key]==='object')details.push(`${VARIANTS[key]}：${translated(value[key].status)}`);
  return details.join(' · ')||'退出观察未报告';
}
export function clearView(message='状态不可用') {
  lastStatus=null;
  document.getElementById('accounts').replaceChildren();
  for(const id of ['total-equity','inception','heartbeat','review-state'])text(id,'—');
  text('service-state',message);text('fingerprint','源码指纹待读取');
  text('review-detail','至少 14 天、各账户至少 10 笔完整平仓；每只股票、每种成本、每个候选规则至少 10 组已完成的提前退出配对跟踪');
}
export function setCost(value) {
  if(!['normal','double'].includes(value))throw new Error('未知成本口径');
  selectedCost=value;
  text('cost-note',value==='double'?'双倍成本：手续费与不利滑点假设分别加倍，资金费仍按发布值计入。':'正常成本：计入手续费、盘口价差、滑点及资金费。');
  if(lastStatus)render(lastStatus);
}
export function render(status, now=Date.now()) {
  if(!status||status.mode!=='SIMULATION'||status.places_orders!==false)throw new Error('独立模拟模式不匹配');
  const maxAge=Math.max(90,Number.isFinite(status.poll_seconds)&&status.poll_seconds>0?status.poll_seconds*3:90)*1000;
  const problem=freshness(status,now,maxAge)||(status.running!==true?'实验已停止':null);
  if(problem){clearView(problem);return false;}
  lastStatus=status;
  text('service-state',translated(status.status));
  text('inception',timeText(status.started_at_ms??status.started_at_utc));
  text('heartbeat',`心跳 ${timeText(heartbeat(status))} · 北京时间`);
  const body=document.getElementById('accounts');body.replaceChildren();
  let total=0,allReported=true,minimumClosed=Infinity,allArmsComplete=true;
  for(const symbol of SYMBOLS){
    for(const [variant,title] of Object.entries(VARIANTS)){
      const arm=status.by_symbol?.[symbol]?.[`${variant}_${selectedCost}`];
      const tr=document.createElement('tr');if(variant==='baseline')tr.setAttribute('data-symbol-start','true');
      const issue=arm?(arm.financial_integrity===false?'金融记录校验失败／账户已隔离':freshness(arm,now,maxAge)):'账户状态缺失';
      addCell(tr,0,`${names[symbol]} · ${title}`);
      if(issue){for(let i=1;i<9;i++)addCell(tr,i,'—');addCell(tr,9,issue);allReported=false;}
      else {
        if(typeof arm.equity==='number'&&Number.isFinite(arm.equity))total+=arm.equity;else allReported=false;
        const metric=arm.metrics||{},comp=status.comparisons?.[symbol]?.[selectedCost]?.[variant],comparisonValid=comp?.comparison_valid!==false,paired=comparisonValid?comp?.paired?.summary:null;
        addCell(tr,1,numberText(arm.equity));addCell(tr,2,numberText(metric.closed_net_pnl_usdt));
        addCell(tr,3,count(metric.closed_positions));addCell(tr,4,`${numberText(metric.max_sampled_drawdown_pct)}${Number.isFinite(metric.max_sampled_drawdown_pct)?'%':''}`);
        addCell(tr,5,`${numberText(arm.fees_paid)} / ${numberText(arm.funding_pnl)}`);addCell(tr,6,numberText(arm.position_qty,5));
        addCell(tr,7,variant==='baseline'?'本口径基准':!comparisonValid?'比较不可用':`权益 ${numberText(comp?.equity_delta_usdt)} / 平仓净 ${numberText(comp?.closed_net_delta_usdt)}`);
        addCell(tr,8,variant==='baseline'?'—':!comparisonValid?'比较不可用':paired?`匹配 ${count(paired.matched_count)} · 提前退出 ${count(paired.early_exit_count)} · 完成 ${count(paired.completed_followups)} · 截尾 ${count(paired.censored_followups)}`:'配对样本未报告');
        addCell(tr,9,`${translated(arm.status)} · ${translated(arm.signal_status)} · ${exitText(arm.exit_response_status)}`);
      }
      body.append(tr);
    }
    for(const variant of Object.keys(VARIANTS))for(const cost of ['normal','double']){
      const arm=status.by_symbol?.[symbol]?.[`${variant}_${cost}`],n=arm?.metrics?.closed_positions;
      if(!arm||arm.financial_integrity===false||freshness(arm,now,maxAge)||!Number.isSafeInteger(n)||n<0)allArmsComplete=false;
      else minimumClosed=Math.min(minimumClosed,n);
    }
  }
  text('total-equity',allReported?`${numberText(total)} USDT`:'账户数据不完整');
  text('capital-note',`本口径 12 个独立账户，本金 12,000 USDT；两种成本共 24,000 USDT`);
  const review=status.review||{},elapsed=review.elapsed_days,days=review.minimum_elapsed_days??14,needed=review.minimum_closed_positions_each??10;
  const neededPairs=Number.isSafeInteger(review.minimum_completed_early_exit_pairs)&&review.minimum_completed_early_exit_pairs>=10?review.minimum_completed_early_exit_pairs:10;
  let minimumPairs=Infinity,allPairsReported=true;
  for(const symbol of SYMBOLS)for(const cost of ['normal','double'])for(const variant of ['breakout_failure','giveback','combined']){
    const comparison=status.comparisons?.[symbol]?.[cost]?.[variant],n=comparison?.paired?.summary?.completed_followups;
    if(comparison?.comparison_valid===false||!Number.isSafeInteger(n)||n<0)allPairsReported=false;
    else minimumPairs=Math.min(minimumPairs,n);
  }
  const pairSufficient=review.completed_pair_sample_sufficient===true&&allPairsReported&&minimumPairs>=neededPairs;
  const ready=review.ready_for_first_review===true&&Number.isFinite(elapsed)&&elapsed>=days&&allArmsComplete&&minimumClosed>=needed&&pairSufficient;
  text('review-state',ready?'达到首次复核门槛':'尚未完成前瞻验证');
  text('review-detail',`${numberText(elapsed,1)} / ${numberText(days,0)} 天；全部 24 账户最少完整平仓 ${allArmsComplete?count(minimumClosed):'数据不完整'} / ${count(needed)}；全部 18 组股票／成本／候选规则最少已完成提前退出配对 ${allPairsReported?count(minimumPairs):'数据不完整'} / ${count(neededPairs)}。不自动替换主策略。`);
  const fp=status.fingerprint??status.source_fingerprint;
  const hashes=fp?.source_sha256??fp?.source_hashes??status.source_sha256;
  text('fingerprint',hashes&&typeof hashes==='object'?`冻结源码 ${Object.keys(hashes).length} 项；实验 ${String(status.experiment_id??'未报告')}。按 manifest 核验，页面不作为进程源码证明。`:typeof fp==='string'?`实验指纹 ${fp}`:'源码指纹请查实验 manifest；页面不作为进程源码证明。');
  return true;
}
export async function refresh() {
  const sequence=++refreshSequence,controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),10000);
  try{
    const response=await fetch(STATUS_PATH,{cache:'no-store',signal:controller.signal});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const status=await response.json();if(sequence!==refreshSequence)return;
    const valid=render(status);document.getElementById('load-error').hidden=valid;
    if(!valid)text('load-error','心跳不可核验，已清空旧金额；等待下一次有效状态。');
  }catch(error){if(sequence!==refreshSequence)return;clearView();document.getElementById('load-error').hidden=false;text('load-error',`无法读取实验状态（${error.message}），已清空旧金额。`);}
  finally{clearTimeout(timeout);}
}
if(typeof document!=='undefined'){
  for(const input of document.querySelectorAll('input[name="cost"]'))input.addEventListener('change',()=>{if(input.checked)setCost(input.value);});
  refresh();setInterval(refresh,30000);
}
