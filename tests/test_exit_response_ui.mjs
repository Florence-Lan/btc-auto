import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {SYMBOLS,VARIANTS,render,refresh,setCost,timeText,timestampMs,numberText} from '../terminal/exit-response.js';

const NOW=Date.parse('2026-10-23T13:00:00Z');
const ids=['service-state','load-error','total-equity','inception','heartbeat','review-state','review-detail','capital-note','accounts','cost-note','fingerprint'];
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.ownText='';this.hidden=false;}
  set textContent(value){this.ownText=String(value);this.children=[];}
  get textContent(){return this.ownText+this.children.map(child=>child.textContent??String(child)).join('');}
  set innerHTML(_value){throw new Error('Unsafe HTML assignment');}
  get innerHTML(){throw new Error('Use safe DOM nodes');}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.ownText='';this.children=[...nodes];}
  setAttribute(key,value){this[key]=String(value);}
}
function environment(t){
  const previous={document:globalThis.document,fetch:globalThis.fetch,now:Date.now};
  const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
  globalThis.document={getElementById(id){assert.ok(elements[id],`Missing element ${id}`);return elements[id];},createElement(tag){return new Element(tag);}};
  Date.now=()=>NOW;setCost('normal');
  t.after(()=>{Date.now=previous.now;for(const name of ['document','fetch']){if(previous[name]===undefined)delete globalThis[name];else globalThis[name]=previous[name];}});
  return elements;
}
function fixture(){
  const status={mode:'SIMULATION',places_orders:false,running:true,status:'healthy',poll_seconds:30,heartbeat_at_ms:NOW,
    started_at_ms:NOW-86400000,experiment_id:'exit-response-test',by_symbol:{},comparisons:{},
    review:{elapsed_days:1,minimum_elapsed_days:14,minimum_closed_positions_each:10,minimum_completed_early_exit_pairs:10,completed_pair_sample_sufficient:false,ready_for_first_review:false},
    fingerprint:{source_sha256:{'scripts/example.py':'a'.repeat(64)}}};
  for(const symbol of SYMBOLS){status.by_symbol[symbol]={};status.comparisons[symbol]={};
    for(const cost of ['normal','double']){status.comparisons[symbol][cost]={};
      for(const variant of Object.keys(VARIANTS)){
        status.by_symbol[symbol][`${variant}_${cost}`]={heartbeat_at_ms:NOW,status:'healthy',signal_status:'holding',
          equity:cost==='normal'?1001:999,wallet_balance:1000,fees_paid:.2,funding_pnl:-.1,position_qty:-.01,
          exit_response_status:{status:'tracking'},metrics:{closed_positions:2,closed_net_pnl_usdt:1.1,max_sampled_drawdown_pct:.5}};
        if(variant!=='baseline')status.comparisons[symbol][cost][variant]={equity_delta_usdt:.25,closed_net_delta_usdt:.1,
          paired:{summary:{matched_count:2,early_exit_count:1,completed_followups:0,censored_followups:1}}};
      }
    }
  }
  return status;
}
function feed(value){globalThis.fetch=async(path,options)=>{assert.match(path,/stock_exit_response_shadow_20261023\/status\.json$/);assert.equal(options.cache,'no-store');return{ok:true,json:async()=>value};};}

test('dates require explicit zones and display Beijing time; invalid financial values stay missing',()=>{
  assert.match(timeText('2026-10-23T13:00:00Z'),/10\/23.*21:00:00/);
  for(const value of [null,true,false,-1,0,NaN,Infinity,'2026-10-23T13:00:00','garbage'])assert.equal(timestampMs(value),null);
  assert.equal(numberText(null),'—');assert.equal(numberText('10'),'—');assert.equal(numberText(NaN),'—');assert.equal(numberText(0),'0.00');
});

test('normal and double costs each show exactly three symbols times four rules with separate capital',t=>{
  const page=environment(t),value=fixture();assert.equal(render(value),true);
  assert.equal(page.accounts.children.length,12);assert.match(page['total-equity'].textContent,/12,012\.00/);
  assert.match(page['capital-note'].textContent,/12,000.*24,000/);
  assert.match(page.accounts.textContent,/MU.*原规则/);assert.match(page.accounts.textContent,/SNDK.*突破失效退出/);assert.match(page.accounts.textContent,/SK Hynix.*组合退出/);
  setCost('double');assert.equal(page.accounts.children.length,12);assert.match(page['total-equity'].textContent,/11,988\.00/);assert.match(page['cost-note'].textContent,/手续费.*滑点.*加倍/);
  assert.throws(()=>setCost('unknown'),/未知/);
});

test('paired diagnostics retain completed and censored followups without turning them into profits',t=>{
  const page=environment(t);render(fixture());
  const candidate=page.accounts.children[1];assert.match(candidate.children[7].textContent,/权益 0\.25 \/ 平仓净 0\.10/);
  assert.match(candidate.children[8].textContent,/匹配 2.*提前退出 1.*完成 0.*截尾 1/);
  assert.equal(page.accounts.children[0].children[7].textContent,'本口径基准');
  assert.match(page['review-state'].textContent,/尚未/);
});

test('future, expired, absent and stopped service heartbeats clear all earlier financial values',t=>{
  const page=environment(t);
  for(const kind of ['future','expired','missing','stopped']){
    const value=fixture();render(value);assert.equal(page.accounts.children.length,12);
    if(kind==='future')value.heartbeat_at_ms=NOW+1;
    if(kind==='expired')value.heartbeat_at_ms=NOW-90001;
    if(kind==='missing')delete value.heartbeat_at_ms;
    if(kind==='stopped')value.running=false;
    assert.equal(render(value),false);assert.equal(page.accounts.children.length,0);assert.equal(page['total-equity'].textContent,'—');assert.equal(page.heartbeat.textContent,'—');
    setCost('double');assert.equal(page.accounts.children.length,0);setCost('normal');
  }
});

test('one stale arm hides that arm and aggregate equity without erasing other current accounts',t=>{
  const page=environment(t),value=fixture();value.by_symbol.MUUSDT.giveback_normal.heartbeat_at_ms=NOW-100000;
  render(value);assert.equal(page.accounts.children.length,12);const stale=page.accounts.children[2];
  assert.equal(stale.children[1].textContent,'—');assert.match(stale.children[9].textContent,/过期/);
  assert.equal(page.accounts.children[3].children[1].textContent,'1,001.00');assert.match(page['total-equity'].textContent,/不完整/);
});

test('quarantined financial state hides all arm amounts and totals and blocks first review',t=>{
  const page=environment(t),value=fixture();value.review.elapsed_days=15;value.review.ready_for_first_review=true;value.review.completed_pair_sample_sufficient=true;
  for(const arms of Object.values(value.by_symbol))for(const arm of Object.values(arms))arm.metrics.closed_positions=10;
  for(const costs of Object.values(value.comparisons))for(const variants of Object.values(costs))for(const comparison of Object.values(variants))comparison.paired.summary.completed_followups=10;
  render(value);assert.match(page['review-state'].textContent,/达到首次复核/);
  value.by_symbol.MUUSDT.giveback_normal.financial_integrity=false;
  render(value);const quarantined=page.accounts.children[2];
  for(let i=1;i<9;i++)assert.equal(quarantined.children[i].textContent,'—');
  assert.match(quarantined.children[9].textContent,/校验失败.*隔离/);assert.match(page['total-equity'].textContent,/不完整/);assert.match(page['review-state'].textContent,/尚未/);
  value.by_symbol.MUUSDT.giveback_normal.financial_integrity=true;value.by_symbol.SNDKUSDT.combined_double.financial_integrity=false;
  render(value);assert.match(page['total-equity'].textContent,/12,012/);assert.match(page['review-state'].textContent,/尚未/);
});

test('invalid comparisons never display numeric deltas or paired samples and cannot satisfy review',t=>{
  const page=environment(t),value=fixture();value.review.elapsed_days=15;value.review.ready_for_first_review=true;value.review.completed_pair_sample_sufficient=true;
  for(const arms of Object.values(value.by_symbol))for(const arm of Object.values(arms))arm.metrics.closed_positions=10;
  for(const costs of Object.values(value.comparisons))for(const variants of Object.values(costs))for(const comparison of Object.values(variants))comparison.paired.summary.completed_followups=10;
  value.comparisons.MUUSDT.normal.breakout_failure.comparison_valid=false;
  render(value);const row=page.accounts.children[1];assert.equal(row.children[1].textContent,'1,001.00');assert.equal(row.children[7].textContent,'比较不可用');assert.equal(row.children[8].textContent,'比较不可用');assert.match(page['review-state'].textContent,/尚未/);assert.match(page['review-detail'].textContent,/配对 数据不完整/);
});

test('incomplete amounts and metrics do not become zero or allow an early review',t=>{
  const page=environment(t),value=fixture();value.review.ready_for_first_review=true;value.review.elapsed_days=20;
  delete value.by_symbol.MUUSDT.baseline_normal.equity;delete value.by_symbol.MUUSDT.baseline_normal.metrics;
  render(value);assert.equal(page.accounts.children[0].children[1].textContent,'—');assert.equal(page.accounts.children[0].children[2].textContent,'—');assert.match(page['total-equity'].textContent,/不完整/);assert.match(page['review-state'].textContent,/尚未/);
});

test('first review requires all twenty-four arms to have sufficient complete positions',t=>{
  const page=environment(t),value=fixture();value.review.elapsed_days=15;value.review.ready_for_first_review=true;value.review.completed_pair_sample_sufficient=true;
  for(const arms of Object.values(value.by_symbol))for(const arm of Object.values(arms))arm.metrics.closed_positions=10;
  for(const costs of Object.values(value.comparisons))for(const variants of Object.values(costs))for(const comparison of Object.values(variants))comparison.paired.summary.completed_followups=10;
  render(value);assert.match(page['review-state'].textContent,/达到首次复核/);assert.match(page['review-detail'].textContent,/不自动替换/);
  value.by_symbol.SNDKUSDT.combined_double.metrics.closed_positions=9;render(value);assert.match(page['review-state'].textContent,/尚未/);
});

test('fourteen days and ten closes cannot replace ten completed early-exit pairs in all eighteen groups',t=>{
  const page=environment(t),value=fixture();value.review.elapsed_days=14;value.review.ready_for_first_review=true;value.review.completed_pair_sample_sufficient=true;
  for(const arms of Object.values(value.by_symbol))for(const arm of Object.values(arms))arm.metrics.closed_positions=10;
  render(value);assert.match(page['review-state'].textContent,/尚未/);assert.match(page['review-detail'].textContent,/18 组.*0 \/ 10/);
  for(const costs of Object.values(value.comparisons))for(const variants of Object.values(costs))for(const comparison of Object.values(variants))comparison.paired.summary.completed_followups=10;
  render(value);assert.match(page['review-state'].textContent,/达到首次复核/);
  const pair=value.comparisons.SKHYNIXUSDT.double.combined.paired.summary;pair.completed_followups=9;pair.censored_followups=100;
  render(value);assert.match(page['review-state'].textContent,/尚未/);assert.match(page['review-detail'].textContent,/18 组.*9 \/ 10/);
  delete pair.completed_followups;render(value);assert.match(page['review-state'].textContent,/尚未/);assert.match(page['review-detail'].textContent,/配对 数据不完整/);
  pair.completed_followups=10;value.review.completed_pair_sample_sufficient=false;render(value);assert.match(page['review-state'].textContent,/尚未/);
  delete value.review.completed_pair_sample_sufficient;render(value);assert.match(page['review-state'].textContent,/尚未/);
});

test('baseline, confirmation and latched exit states have readable labels',t=>{
  const page=environment(t),value=fixture();
  value.by_symbol.MUUSDT.baseline_normal.exit_response_status='baseline';
  value.by_symbol.MUUSDT.breakout_failure_normal.exit_response_status={status:'confirming'};
  value.by_symbol.MUUSDT.giveback_normal.exit_response_status={status:'latched'};
  render(value);assert.match(page.accounts.children[0].children[9].textContent,/原规则/);assert.match(page.accounts.children[1].children[9].textContent,/等待收盘确认/);assert.match(page.accounts.children[2].children[9].textContent,/退出触发已锁定/);
});

test('untrusted status strings stay inert text and financial fields never coerce HTML',t=>{
  const page=environment(t),value=fixture(),attack='<img src=x onerror="alert(1)">';
  value.by_symbol.MUUSDT.combined_normal.exit_response_status={status:attack,reasons:[attack]};value.by_symbol.MUUSDT.combined_normal.equity=attack;
  value.experiment_id=attack;render(value);assert.ok(page.accounts.textContent.includes(attack));assert.ok(page.fingerprint.textContent.includes(attack));
  assert.equal(page.accounts.children[3].children[1].textContent,'—');
});

test('failed fetch and wrong mode erase previous results, then successful refresh recovers',async t=>{
  const page=environment(t),value=fixture();feed(value);await refresh();assert.equal(page.accounts.children.length,12);
  globalThis.fetch=async()=>{throw new Error('offline');};await refresh();assert.equal(page.accounts.children.length,0);assert.equal(page['load-error'].hidden,false);
  feed({...value,places_orders:true});await refresh();assert.equal(page.accounts.children.length,0);assert.match(page['load-error'].textContent,/模式不匹配/);
  feed(value);await refresh();assert.equal(page.accounts.children.length,12);assert.equal(page['load-error'].hidden,true);
});

test('an out-of-order request cannot restore superseded results',async t=>{
  const page=environment(t),value=fixture();let release;
  globalThis.fetch=()=>new Promise(resolve=>{release=resolve;});const earlier=refresh();
  globalThis.fetch=async()=>{throw new Error('newer request failed');};await refresh();
  release({ok:true,json:async()=>value});await earlier;assert.equal(page.accounts.children.length,0);assert.equal(page['load-error'].hidden,false);
});

test('page and task keep isolated semantics, safe recovery flags and mobile layout',async()=>{
  const [html,css,task]=await Promise.all(['terminal/exit-response.html','terminal/exit-response.css','scripts/install_stock_exit_response_task.ps1'].map(path=>readFile(new URL(`../${path}`,import.meta.url),'utf8')));
  for(const id of ids)assert.ok(html.includes(`id="${id}"`),`HTML missing ${id}`);
  assert.match(html,/未合入原主账户/);assert.match(html,/未证明盈利改善|尚未证明盈利改善/);assert.match(html,/截尾/);
  assert.match(html,/连续 2 根 1 小时收盘/);assert.match(html,/达到 1R 后启动/);assert.match(html,/回吐 0\.5R/);assert.match(html,/实际入场价到初始止损价的距离/);assert.match(html,/保留原硬止损，不自动反手/);assert.match(html,/每只股票、每种成本、每个候选规则至少 10 组/);
  assert.match(css,/@media\(max-width:680px\)/);assert.match(css,/content:attr\(data-label\)/);
  assert.match(task,/btc-auto-exit-response-/);assert.match(task,/'--check'/);assert.match(task,/'--require-existing'/);assert.match(task,/manifest\.json/);assert.match(task,/-WindowStyle Hidden/);
  assert.ok(task.indexOf("'--check'")<task.indexOf('Register-ScheduledTask'));
  assert.match(task,/if \(\$Start -and -not \$wasDisabled\)/);assert.doesNotMatch(task,/run_paper_services|Stop-Process|Stop-ScheduledTask/);
});
