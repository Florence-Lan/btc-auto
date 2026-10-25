export const arms = {baseline:'原规则状态',event_only:'国际事件',market_only:'跨市场冲击',sector_budget:'同业预算',combined:'组合候选'};
const names = {nq_dec_2026:'纳指期货 NQ Dec26',wti_usd_per_barrel:'WTI 具名期货',brent_usd_per_barrel:'Brent 具名期货',usdjpy:'USD/JPY',us_2y_pct:'美国国债 2 年',us_10y_pct:'美国国债 10 年',us_10y_real_pct:'美国实际利率 10 年',jp_2y_pct:'日本国债 2 年',jp_10y_pct:'日本国债 10 年',jp_30y_pct:'日本国债 30 年',de_10y_pct:'德国国债 10 年',fr_10y_pct:'法国国债 10 年',it_10y_pct:'意大利国债 10 年',es_10y_pct:'西班牙国债 10 年',be_10y_pct:'比利时国债 10 年',uk_10y_pct:'英国国债 10 年',fr_de_10y_spread_bp:'法德利差',it_de_10y_spread_bp:'意德利差',es_de_10y_spread_bp:'西班牙／德国利差',be_de_10y_spread_bp:'比利时／德国利差'};
const reasons = {baseline_no_pending_entry:'原账户暂无待入场机会',baseline_origin_unavailable_or_future:'原信号时间缺失或无效',baseline_source_stale_or_unavailable:'原账户观察过期或缺失',baseline_existing_inventory_not_new_entry:'已有持仓，不属于新入场',baseline_original_signal_not_admitted:'原信号尚未准入',baseline_original_entry_checks_not_all_reported_passed:'原执行检查尚未全部通过',baseline_pending_expired_or_invalid:'原待入场机会已过期或无效',baseline_pullback_not_ready:'原回踩确认尚未完成',baseline_reported_unhealthy_or_unknown:'原数据状态降级或未知',forward_history_warming_up:'前瞻历史尚不足',observation_stale:'观察已过期',observation_missing_or_invalid:'缺少有效观察',sector_exposure_unavailable:'同业风险尚不可核算',candidate_direction_unknown:'当前无有效入场方向',international_calendar_blackout:'国际事件窗口',international_calendar_stale:'日历核验已过期',correlated_same_direction_position_limit:'同向存储股持仓达到研究上限',correlated_stop_risk_budget:'同向存储股止损风险超过研究预算',energy_and_technology_shock:'能源上涨与科技下跌冲击',yen_and_technology_shock:'日元升值与科技下跌冲击',technology_disorderly_move:'科技期货短时大幅波动'};
const labels = {observing:'观察中',collecting:'正在采集',degraded:'部分来源不可用',starting:'启动中',fresh:'有效观察',healthy:'有效观察',stale:'已过期',missing:'缺失',unavailable:'不可用',future:'未来时间无效',not_yet_available:'尚未可用',invalid_date:'观察日期无效',daily_current:'日频有效／不是盘中',date_only:'仅日频日期',cached:'缓存／原时间不变',ok:'抓取成功',error:'抓取失败',deadline_exceeded:'采集超时',contract_review_due:'合约待审核',market_closed:'参考时段休市',market_maintenance:'参考维护时段',session_unknown:'参考时段未知',clear:'当前无事件窗口',blackout:'事件窗口内',warming_up:'前瞻历史预热'};
const sessionStatuses = new Set(['market_closed','market_maintenance','session_unknown']);
const sessionPhases = {open:'参考交易时段内',closed:'参考时段休市',maintenance:'参考维护时段',unknown:'参考时段未知'};
const sessionReasons = {session_config_invalid:'参考时段配置不可用',session_asof_invalid:'参考时点无效',session_not_yet_verified:'参考时段尚未核验',session_outside_validity:'已超出参考覆盖期',session_holiday_excluded:'假期时段尚未核验',session_local_date_unreviewed:'参考日期尚未核验',session_timezone_unavailable:'参考时区不可用',session_next_open_outside_validity:'下次参考开市超出覆盖期',session_next_open_holiday_excluded:'下次参考时段涉及未核验假期',session_next_open_date_unreviewed:'下次参考开市日期尚未核验'};
export function timeText(value) {
  if (value === null || value === undefined || value === '') return '—';
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(date) : '无效时间';
}
export function numberText(value, unit='') {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toLocaleString('zh-CN',{maximumFractionDigits:4})}${unit==='percent'?'%':unit==='bp'?' bp':''}` : '—';
}
function translated(value) { return labels[value] || value || '—'; }
function dailyAgeDays(row, now) {
  if(typeof row.observed_date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(row.observed_date))return null;
  const observed=Date.parse(`${row.observed_date}T00:00:00Z`);
  if(!Number.isFinite(observed)||new Date(observed).toISOString().slice(0,10)!==row.observed_date)return null;
  try {
    const parts=new Intl.DateTimeFormat('en-CA',{timeZone:row.observation_timezone||'UTC',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(now);
    const date=Object.fromEntries(parts.map(part=>[part.type,part.value]));
    return (Date.UTC(Number(date.year),Number(date.month)-1,Number(date.day))-observed)/86400000;
  } catch { return null; }
}
function sessionText(session, primaryStatus) {
  if(!session||typeof session!=='object')return '';
  const parts=[];
  if(!sessionStatuses.has(primaryStatus))parts.push(sessionPhases[session.phase]||'参考时段未知');
  const basis={scheduled_exchange_reference:'公开交易时段参考',indicative_broker_reference:'经纪商指示时段参考'}[session.basis]||session.basis;
  if(basis)parts.push(`依据：${basis}`);
  if(Number.isFinite(session.next_open_at_ms))parts.push(`下次参考开市 ${timeText(session.next_open_at_ms)}（北京时间）`);
  else if(session.phase!=='open')parts.push('下次参考开市未确认');
  for(const reason of [session.reason_code,session.next_open_reason_code])if(reason&&reason!=='session_reference_schedule')parts.push(sessionReasons[reason]||reason);
  parts.push('参考时段，非交易所实时状态');
  return parts.join('；');
}
function contractReviewText(row, now) {
  const review=row.contract_review;
  const cutoff=review?.next_review_at_ms??row.contract_review_cutoff_at_ms;
  const status=Number.isFinite(cutoff)&&now>=cutoff?'review_due':review?.status;
  if(!status)return '';
  const label={review_due:'合约待审核',review_soon:'合约临近审查时间',reviewed:'合约已核对'}[status]||status;
  const parts=[label];
  if(Number.isFinite(cutoff))parts.push(`审查时间 ${timeText(cutoff)}（北京时间）`);
  if(review?.automatic_roll===false||row.instrument?.automatic_roll===false)parts.push('不自动换月');
  else if(review?.automatic_roll===true)parts.push('换月标记异常，需核查');
  return parts.join('；');
}
function fetchText(row) {
  const parts=[translated(row.fetch?.status||row.missing_reason)];
  const error=row.fetch?.error;
  const detail=typeof error==='string'?error:[error?.category,error?.error_type,error?.reason_code||error?.reason].filter(value=>typeof value==='string'||typeof value==='number').join('／');
  if(detail)parts.push(String(detail).slice(0,240));
  return parts.join('：');
}
function reasonText(value) { if(reasons[value])return reasons[value]; const split=String(value).indexOf(':');if(split>0){const key=value.slice(0,split),detail=value.slice(split+1);return `${names[key]||key}：${reasons[detail]||detail}`;}return value; }
function cell(tr, value) { const td=document.createElement('td'); td.textContent=String(value??'—'); tr.append(td); return td; }
function rows(id, records, render) { const root=document.getElementById(id); root.replaceChildren(); for (const record of records) { const tr=document.createElement('tr'); render(tr,record); root.append(tr); } }
export function clearView(message='状态不可用') {
  for (const id of ['observations','events','counts','decisions']) document.getElementById(id).replaceChildren();
  for (const id of ['inception','sample-count','exposure','heartbeat']) document.getElementById(id).textContent='—';
  document.getElementById('service-state').textContent=message;
  document.getElementById('exposure-detail').textContent='缺少可核验的完整状态';
  document.getElementById('event-state').textContent='日历状态不可用';
  document.getElementById('event-verification').textContent='';
}
export function render(status, snapshot, calendar) {
  const now=Date.now(),age=now-(status.heartbeat_at_ms||0);
  const stale=!Number.isFinite(age)||age<0||age>Math.max(90, (status.poll_seconds||30)*3)*1000 || status.running!==true;
  document.getElementById('service-state').textContent=age<0?'未来心跳无效':stale?'心跳过期／已停止':translated(status.status);
  document.getElementById('heartbeat').textContent=`${timeText(status.heartbeat_at_ms)} · 北京时间`;
  document.getElementById('inception').textContent=timeText(status.started_at_ms);
  document.getElementById('sample-count').textContent=`${status.post_inception_opportunity_count||0} / ${status.counts?.baseline?.reported_eligible_opportunities||0}`;
  const exposure=status.portfolio_exposure||{};
  document.getElementById('exposure').textContent=exposure.status==='available'?`${numberText(exposure.gross_notional_usdt)} USDT`:'尚不可计算';
  document.getElementById('exposure-detail').textContent=exposure.status==='available'?`多 ${exposure.same_direction?.['1']?.count||0} · 空 ${exposure.same_direction?.['-1']?.count||0}；仅事实持仓`:(exposure.reasons||[]).join('、');
  rows('observations',Object.entries(snapshot.observations||{}),(tr,[key,row])=>{
    const title=cell(tr,names[key]||key);
    try { const url=new URL(row.source_url); if(url.protocol==='https:'&&!url.username&&!url.password){const a=document.createElement('a');a.textContent=names[key]||key;a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';title.replaceChildren(a);} } catch {}
    cell(tr,numberText(row.value,row.unit));
    cell(tr,row.precision==='date'?`${row.observed_date||'—'}（日期口径）`:timeText(row.observed_at_ms));
    cell(tr,timeText(row.first_seen_at_ms));
    const primaryStatus=row.freshness?.status;
    const daily=row.frequency==='daily'||row.precision==='date';
    const ageDays=daily?dailyAgeDays(row,now):null;
    let quoteStatus=row.freshness?.quote_status||primaryStatus;
    if(row.first_seen_at_ms>now||row.available_at_ms>now||row.observed_at_ms>now)quoteStatus='future';
    else if(daily){
      if(ageDays===null)quoteStatus='invalid_date';
      else if(ageDays<0)quoteStatus='future';
      else if(Number.isFinite(row.freshness?.max_age_days)&&ageDays>row.freshness.max_age_days)quoteStatus='stale';
      else if(quoteStatus==='fresh'||quoteStatus==='healthy')quoteStatus='daily_current';
    } else if(row.precision==='timestamp'&&Number.isFinite(row.freshness?.max_age_ms)&&now-row.observed_at_ms>row.freshness.max_age_ms)quoteStatus='stale';
    let freshness=sessionStatuses.has(primaryStatus)?`${translated(primaryStatus)}；报价：${translated(quoteStatus)}`:translated(quoteStatus);
    if(daily)freshness+=ageDays!==null?`（距观察 ${ageDays} 天；非盘中报价）`:'（日频／非盘中报价）';
    else if(row.precision==='timestamp'&&Number.isFinite(row.observed_at_ms))freshness+=`（距观察 ${Math.max(0,(now-row.observed_at_ms)/60000).toFixed(1)} 分钟）`;
    const session=sessionText(row.market_session,primaryStatus),review=contractReviewText(row,now);
    if(session)freshness+=`；${session}`;
    if(review)freshness+=`；${review}`;
    cell(tr,stale?`${freshness}（服务心跳无效）`:freshness);
    cell(tr,fetchText(row));
  });
  const event=status.event||{};
  const calendarFuture=calendar.verified_at_ms>Date.now()||calendar.available_at_ms>Date.now();
  const calendarExpired=(Number.isFinite(calendar.valid_until_ms)&&Date.now()>calendar.valid_until_ms)||(Number.isFinite(calendar.coverage_end_ms)&&Date.now()>=calendar.coverage_end_ms);
  document.getElementById('event-state').textContent=stale?'服务心跳无效，日历当前判断不可用':calendarFuture?'日历未来时间无效':calendarExpired?'日历已过期':translated(event.status);
  document.getElementById('event-verification').textContent=`核验 ${timeText(calendar.verified_at_ms)}；覆盖至 ${timeText(calendar.coverage_end_ms)}。安排时间不等于实际发布时间。${status.calendar_refresh?.success===false?'最近复核失败，沿用原核验时间及其到期限制。':''}`;
  rows('events',calendar.events||[],(tr,row)=>{
    cell(tr,row.headline||row.event_id);
    cell(tr,row.scheduled_at_ms?timeText(row.scheduled_at_ms):`${timeText(row.bounded_start_ms??row.starts_at_ms)} — ${timeText(row.bounded_end_ms??row.ends_at_ms)}（保守区间）`);
    cell(tr,row.actual_at_ms?'已有实际时间记录':row.uncertainty?.note||row.uncertainty||'官方安排／实际时间未记录');
  });
  rows('counts',Object.entries(arms),(tr,[key,label])=>{const c=status.counts?.[key]||{};cell(tr,label);cell(tr,c.reported_eligible_opportunities||0);cell(tr,c.observed_allowed||0);cell(tr,`${c.observed_blocked||0}（缺数／预热 ${c.observed_blocked_unavailable_or_warmup||0}；事件 ${c.observed_blocked_event_blackout||0}；冲击 ${c.observed_blocked_market_shock||0}；同业 ${c.observed_blocked_sector_budget||0}）`);});
  rows('decisions',status.decisions||[],(tr,row)=>{cell(tr,String(row.account_id||'').toUpperCase());cell(tr,arms[row.arm]||row.arm);cell(tr,stale?'当前判断已失效':row.allowed?'观察通过':'观察阻止');cell(tr,(row.reasons||[]).map(reasonText).join('、')||'—');});
}
async function getJson(path) { const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json(); }
export async function refresh() {
  try {
    const [status,snapshot,calendar]=await Promise.all(['/data/cross_market_observer/status.json','/data/cross_market_observer/latest_observations.json','/data/cross_market_observer/calendar_current.json'].map(getJson));
    if(status.mode!=='OBSERVATION_ONLY'||status.places_orders!==false||status.no_pnl_simulated!==true)throw new Error('观察服务模式不匹配');
    render(status,snapshot,calendar);document.getElementById('load-error').hidden=true;
  } catch(error) {clearView();const node=document.getElementById('load-error');node.hidden=false;node.textContent=`暂时无法取得完整观察状态（${error.message}），请勿据此推断风险已解除。`;}
}
if (typeof document!=='undefined') { refresh();setInterval(refresh,30000); }
