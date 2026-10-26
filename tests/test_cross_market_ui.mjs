import test from 'node:test';
import assert from 'node:assert/strict';
import {render, refresh, timeText} from '../terminal/cross-market.js';

const NOW = Date.parse('2026-10-23T01:30:00Z');
const ids = ['service-state','heartbeat','inception','sample-count','exposure',
  'exposure-detail','observations','event-state','event-verification','events',
  'counts','decisions','load-error'];

class Element {
  constructor(tag='div') { this.tagName=tag;this.children=[];this.ownText='';this.hidden=false; }
  set textContent(value) { this.ownText=String(value);this.children=[]; }
  get textContent() { return this.ownText+this.children.map(child=>child.textContent??String(child)).join(''); }
  set innerHTML(_value) { throw new Error('Untrusted content must never use innerHTML'); }
  get innerHTML() { throw new Error('This view must use DOM text nodes'); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.ownText='';this.children=[...nodes]; }
  setAttribute(name,value) { this[name]=String(value); }
}

function environment(t) {
  const saved={document:globalThis.document,fetch:globalThis.fetch,now:Date.now};
  const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
  globalThis.document={getElementById(id) { assert.ok(elements[id],`Missing binding ${id}`);return elements[id]; },
    createElement(tag) { return new Element(tag); }};
  Date.now=()=>NOW;
  t.after(()=>{Date.now=saved.now;if(saved.document===undefined)delete globalThis.document;else globalThis.document=saved.document;
    if(saved.fetch===undefined)delete globalThis.fetch;else globalThis.fetch=saved.fetch;});
  return elements;
}

function fixtures() {
  const status={mode:'OBSERVATION_ONLY',places_orders:false,no_pnl_simulated:true,running:true,
    status:'observing',heartbeat_at_ms:NOW,started_at_ms:NOW-3600000,poll_seconds:30,
    post_inception_opportunity_count:2,counts:{baseline:{reported_eligible_opportunities:2,observed_allowed:2}},
    portfolio_exposure:{status:'available',gross_notional_usdt:101,same_direction:{'1':{count:0},'-1':{count:1}}},
    event:{status:'clear'},decisions:[{account_id:'mu',arm:'baseline',allowed:true,reasons:[]}]};
  const snapshot={observations:{nq_dec_2026:{value:29280,unit:'usd',precision:'timestamp',
    observed_at_ms:NOW-60000,first_seen_at_ms:NOW-30000,available_at_ms:NOW-30000,
    source_url:'https://example.com/official-market',freshness:{status:'fresh'},fetch:{status:'ok'}}}};
  const calendar={verified_at_ms:NOW-60000,valid_until_ms:NOW+3600000,coverage_end_ms:NOW+86400000,
    events:[{event_id:'cpi',headline:'Scheduled CPI',scheduled_at_ms:NOW+600000,
      actual_at_ms:null,uncertainty:'Scheduled only; actual has not been observed.'}]};
  return {status,snapshot,calendar};
}

function feed(value) {
  globalThis.fetch=async path=>({ok:true,json:async()=>path.endsWith('/status.json')?value.status:
    path.includes('latest_observations')?value.snapshot:value.calendar});
}

test('UTC timestamps explicitly render Beijing time and invalid values stay unavailable',()=>{
  assert.match(timeText('2026-10-23T01:30:00Z'),/10\/23.*09:30:00/);
  assert.equal(timeText(null),'—');
  assert.equal(timeText('not-a-date'),'无效时间');
});

test('date-only observations preserve their source date instead of inventing an intraday clock',t=>{
  const page=environment(t),value=fixtures();
  value.snapshot.observations={us_2y_pct:{value:4.81,unit:'percent',precision:'date',
    observed_date:'2026-10-21',observed_at_ms:Date.parse('2026-10-21T00:00:00Z'),
    first_seen_at_ms:NOW,freshness:{status:'date_only'},fetch:{status:'ok'}}};
  render(value.status,value.snapshot,value.calendar);
  assert.match(page.observations.textContent,/2026-10-21（日期口径）/);
  assert.match(page.observations.textContent,/仅日频日期/);
  assert.doesNotMatch(page.observations.children[0].children[2].textContent,/08:00:00/);
});

test('untrusted headlines and reasons use text nodes and unsafe source URLs never become anchors',t=>{
  const page=environment(t),value=fixtures();
  const attack='<img src=x onerror="alert(1)">';
  value.calendar.events[0].headline=attack;
  value.status.decisions[0].reasons=[attack];
  value.snapshot.observations.nq_dec_2026.source_url='javascript:alert(1)';
  render(value.status,value.snapshot,value.calendar);
  assert.ok(page.events.textContent.includes(attack));
  assert.ok(page.decisions.textContent.includes(attack));
  assert.equal(page.observations.children[0].children[0].children.length,0);
  value.snapshot.observations.nq_dec_2026.source_url='https://user:secret@example.com/path';
  render(value.status,value.snapshot,value.calendar);
  assert.equal(page.observations.children[0].children[0].children.length,0);
});

test('valid public source links keep noreferrer and noopener',t=>{
  const page=environment(t),value=fixtures();render(value.status,value.snapshot,value.calendar);
  const link=page.observations.children[0].children[0].children[0];
  assert.equal(link.tagName,'a');assert.equal(link.target,'_blank');
  assert.match(link.rel,/noopener/);assert.match(link.rel,/noreferrer/);
});

test('future and stale observations are visibly different from successful fetching',t=>{
  const page=environment(t),value=fixtures();
  value.snapshot.observations.nq_dec_2026.freshness.status='future';
  render(value.status,value.snapshot,value.calendar);
  assert.match(page.observations.textContent,/未来时间无效/);
  assert.match(page.observations.textContent,/抓取成功/);
  value.snapshot.observations.nq_dec_2026.freshness.status='stale';
  render(value.status,value.snapshot,value.calendar);
  assert.match(page.observations.textContent,/已过期/);
});

test('a future heartbeat never presents the service or event gate as currently healthy',t=>{
  const page=environment(t),value=fixtures();value.status.heartbeat_at_ms=NOW+60000;
  render(value.status,value.snapshot,value.calendar);
  assert.match(page['service-state'].textContent,/未来|无效|不可用/);
  assert.doesNotMatch(page['event-state'].textContent,/当前无事件窗口/);
});

test('expired heartbeat marks cached observations and does not claim event clearance',t=>{
  const page=environment(t),value=fixtures();value.status.heartbeat_at_ms=NOW-120000;
  render(value.status,value.snapshot,value.calendar);
  assert.match(page['service-state'].textContent,/过期|停止/);
  assert.match(page.observations.textContent,/心跳.*(?:过期|无效)/);
  assert.doesNotMatch(page['event-state'].textContent,/当前无事件窗口/);
});

test('calendar verification from the future or beyond expiry cannot display a clear gate',t=>{
  const page=environment(t),value=fixtures();
  value.calendar.verified_at_ms=NOW+1;
  render(value.status,value.snapshot,value.calendar);
  assert.doesNotMatch(page['event-state'].textContent,/当前无事件窗口/);
  assert.match(page['event-state'].textContent,/未来|无效|不可用/);
  value.calendar.verified_at_ms=NOW-3600000;value.calendar.valid_until_ms=NOW-1;
  render(value.status,value.snapshot,value.calendar);
  assert.doesNotMatch(page['event-state'].textContent,/当前无事件窗口/);
  assert.match(page['event-state'].textContent,/过期|不可用/);
});

test('unknown BoJ timing displays the bounded decision day without fabricating a point',t=>{
  const page=environment(t),value=fixtures();
  value.calendar.events=[{event_id:'boj',headline:'BoJ decision time unknown',scheduled_at_ms:null,
    bounded_start_ms:Date.parse('2026-10-30T00:00:00+09:00'),
    bounded_end_ms:Date.parse('2026-10-31T00:00:00+09:00'),
    actual_at_ms:null,uncertainty:'Full decision-day candidate assumption.'}];
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.events.children[0].children[1].textContent;
  assert.match(displayed,/10\/29.*23:00:00/);assert.match(displayed,/10\/30.*23:00:00/);
  assert.doesNotMatch(displayed,/— —/);
});

test('failed refresh clears earlier observations, decisions and exposure; recovery repopulates',async t=>{
  const page=environment(t),value=fixtures();feed(value);await refresh();
  assert.equal(page.observations.children.length,1);assert.equal(page.decisions.children.length,1);
  globalThis.fetch=async()=>{throw new Error('connection unavailable');};
  await refresh();
  assert.equal(page.observations.children.length,0);
  assert.equal(page.events.children.length,0);
  assert.equal(page.decisions.children.length,0);
  assert.doesNotMatch(page.exposure.textContent,/101/);
  assert.doesNotMatch(page['event-state'].textContent,/当前无事件窗口/);
  assert.equal(page['load-error'].hidden,false);
  feed(value);await refresh();
  assert.equal(page.observations.children.length,1);assert.equal(page['load-error'].hidden,true);
});

test('an endpoint mode mismatch also clears earlier observed passes',async t=>{
  const page=environment(t),value=fixtures();feed(value);await refresh();
  value.status.no_pnl_simulated=false;await refresh();
  assert.equal(page.decisions.children.length,0);
  assert.equal(page.observations.children.length,0);
  assert.match(page['load-error'].textContent,/模式不匹配/);
});

test('Spain and Belgium yields and German spreads have readable names and correct units',t=>{
  const page=environment(t),value=fixtures();
  value.snapshot.observations=Object.fromEntries([
    ['es_10y_pct',3.25,'percent'],['be_10y_pct',3.5,'percent'],
    ['es_de_10y_spread_bp',55,'bp'],['be_de_10y_spread_bp',80,'bp']
  ].map(([key,price,unit])=>[key,{value:price,unit,precision:'date',observed_date:'2026-10-22',
    first_seen_at_ms:NOW,freshness:{status:'date_only'},fetch:{status:'ok'}}]));
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children.map(row=>row.children.slice(0,2).map(cell=>cell.textContent));
  assert.deepEqual(displayed,[['西班牙国债 10 年','3.25%'],['比利时国债 10 年','3.5%'],
    ['西班牙／德国利差','55 bp'],['比利时／德国利差','80 bp']]);
});

for(const [primary,phase,label] of [
  ['market_closed','closed','参考时段休市'],
  ['market_maintenance','maintenance','参考维护时段'],
  ['session_unknown','unknown','参考时段未知']
])test(`${primary} remains separate from stale quote age and a failed fetch`,t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.observed_at_ms=NOW-2*86400000;
  row.freshness={status:primary,quote_status:'fresh',max_age_ms:20*60000};
  row.market_session={phase,basis:'scheduled_exchange_reference',next_open_at_ms:NOW+3600000,
    reason_code:'session_reference_schedule',is_actual_market_status:false};
  row.fetch={status:'error',error:{category:'timeout',error_type:'ReadTimeout'}};
  render(value.status,value.snapshot,value.calendar);
  const cells=page.observations.children[0].children,displayed=cells[4].textContent;
  assert.ok(displayed.startsWith(label));
  assert.match(displayed,/报价：已过期/);
  assert.match(displayed,/依据：公开交易时段参考/);
  assert.match(displayed,/下次参考开市 10\/23.*10:30:00（北京时间）/);
  assert.match(displayed,/参考时段，非交易所实时状态/);
  assert.match(cells[5].textContent,/抓取失败.*timeout.*ReadTimeout/);
  assert.doesNotMatch(displayed,/抓取失败|ReadTimeout/);
});

test('advancing the browser clock cannot infer an open market from an old closed snapshot',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.freshness={status:'market_closed',quote_status:'fresh',max_age_ms:20*60000};
  row.market_session={phase:'closed',basis:'scheduled_exchange_reference',next_open_at_ms:NOW+60000};
  Date.now=()=>NOW+3600000;
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.ok(displayed.startsWith('参考时段休市；报价：已过期'));
  assert.match(displayed,/服务心跳无效/);
  assert.doesNotMatch(displayed,/参考交易时段内/);
});

test('future quote clocks stay invalid without replacing the reference closed label',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.observed_at_ms=NOW+60000;
  row.freshness={status:'market_closed',quote_status:'fresh',max_age_ms:20*60000};
  row.market_session={phase:'closed',next_open_at_ms:null};
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.ok(displayed.startsWith('参考时段休市；报价：未来时间无效'));
  assert.match(displayed,/下次参考开市未确认/);
});

test('an open broker reference remains explicitly indicative and cannot make a stale quote fresh',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.observed_at_ms=NOW-3600000;
  row.freshness={status:'fresh',max_age_ms:20*60000};
  row.market_session={phase:'open',basis:'indicative_broker_reference',next_open_at_ms:null};
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.ok(displayed.startsWith('已过期'));
  assert.match(displayed,/参考交易时段内/);
  assert.match(displayed,/经纪商指示时段参考/);
  assert.match(displayed,/非交易所实时状态/);
});

test('unknown reference coverage explains why no next-open time is available',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.freshness={status:'session_unknown',quote_status:'stale'};
  row.market_session={phase:'unknown',basis:'scheduled_exchange_reference',next_open_at_ms:null,
    reason_code:'session_outside_validity',next_open_reason_code:'session_next_open_date_unreviewed'};
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/已超出参考覆盖期/);
  assert.match(displayed,/下次参考开市日期尚未核验/);
  assert.match(displayed,/下次参考开市未确认/);
});

test('contract review warnings retain the closed-session and quote labels independently',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.freshness={status:'market_closed',quote_status:'stale'};
  row.market_session={phase:'closed',basis:'scheduled_exchange_reference',next_open_at_ms:NOW+3600000};
  row.contract_review={status:'review_soon',next_review_at_ms:NOW+600000,automatic_roll:false};
  render(value.status,value.snapshot,value.calendar);
  let displayed=page.observations.children[0].children[4].textContent;
  assert.ok(displayed.startsWith('参考时段休市；报价：已过期'));
  assert.match(displayed,/合约临近审查时间/);
  assert.match(displayed,/审查时间 10\/23.*09:40:00（北京时间）/);
  assert.match(displayed,/不自动换月/);
  Date.now=()=>NOW+600001;
  render(value.status,value.snapshot,value.calendar);
  displayed=page.observations.children[0].children[4].textContent;
  assert.ok(displayed.startsWith('参考时段休市；报价：已过期'));
  assert.match(displayed,/合约待审核/);
  assert.doesNotMatch(displayed,/合约临近审查时间/);
});

test('collector contract_review_due retains the session reference and reviewed contracts show their review',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.freshness={status:'contract_review_due',quote_status:'stale'};
  row.market_session={phase:'closed',next_open_at_ms:null};
  row.contract_review={status:'review_due',next_review_at_ms:NOW-60000,automatic_roll:false};
  render(value.status,value.snapshot,value.calendar);
  let displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/合约待审核/);
  assert.match(displayed,/参考时段休市/);
  assert.match(displayed,/不自动换月/);
  row.freshness={status:'fresh'};delete row.market_session;
  row.contract_review={status:'reviewed',next_review_at_ms:NOW+10*86400000,automatic_roll:false};
  render(value.status,value.snapshot,value.calendar);
  displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/合约已核对/);
  assert.doesNotMatch(displayed,/合约待审核/);
});

test('legacy contract cutoffs remain visible while ordinary timestamp age is recalculated',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  row.observed_at_ms=NOW-3600000;
  row.freshness={status:'fresh',max_age_ms:20*60000};
  row.contract_review_cutoff_at_ms=NOW-60000;
  row.instrument={automatic_roll:false};
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/已过期/);
  assert.match(displayed,/合约待审核/);
  assert.match(displayed,/不自动换月/);
});

test('reference metadata and fetch errors remain bounded text rather than executable DOM',t=>{
  const page=environment(t),value=fixtures(),row=value.snapshot.observations.nq_dec_2026;
  const attack='<img src=x onerror="alert(1)">';
  row.freshness={status:'session_unknown',quote_status:'missing'};
  row.market_session={phase:'unknown',basis:attack,reason_code:attack,next_open_at_ms:attack};
  row.contract_review={status:attack,next_review_at_ms:attack,automatic_roll:false};
  row.fetch={status:'error',error:attack+'a'.repeat(300)};
  render(value.status,value.snapshot,value.calendar);
  const cells=page.observations.children[0].children;
  assert.ok(cells[4].textContent.includes(attack));
  assert.match(cells[4].textContent,/下次参考开市未确认/);
  assert.ok(cells[5].textContent.includes(attack));
  assert.ok(cells[5].textContent.length<=246);
  assert.equal(cells[4].children.length,0);
  assert.equal(cells[5].children.length,0);
});

function dailyBond(value) {
  const row=value.snapshot.observations.nq_dec_2026;
  Object.assign(row,{frequency:'daily',precision:'timestamp',observation_timezone:'UTC',
    observed_date:'2026-10-19',observed_at_ms:Date.parse('2026-10-19T15:05:00Z'),
    first_seen_at_ms:NOW-60000,available_at_ms:NOW-60000,
    freshness:{status:'daily_current',max_age_ms:1200000,max_age_days:4,age_days:4,eligible_intraday:false}});
  value.snapshot.observations={es_10y_pct:row};
  return row;
}

test('daily bond GMT timestamps use calendar-day freshness rather than the intraday TTL',t=>{
  const page=environment(t),value=fixtures();dailyBond(value);
  render(value.status,value.snapshot,value.calendar);
  const cells=page.observations.children[0].children,displayed=cells[4].textContent;
  assert.match(cells[2].textContent,/10\/19.*23:05:00/);
  assert.match(displayed,/日频有效／不是盘中/);
  assert.match(displayed,/距观察 4 天；非盘中报价/);
  assert.doesNotMatch(displayed,/已过期|有效观察|分钟/);
});

test('daily expiry advances at the UTC date boundary, including the max-age day itself',t=>{
  const page=environment(t),value=fixtures();dailyBond(value);
  Date.now=()=>Date.parse('2026-10-23T23:59:59Z');
  render(value.status,value.snapshot,value.calendar);
  assert.match(page.observations.children[0].children[4].textContent,/日频有效.*距观察 4 天/);
  Date.now=()=>Date.parse('2026-10-24T00:00:00Z');
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/已过期.*距观察 5 天；非盘中报价/);
  assert.doesNotMatch(displayed,/日频有效/);
});

for(const clock of ['first_seen_at_ms','available_at_ms','observed_at_ms'])test(`daily observations reject a future ${clock} even inside the day limit`,t=>{
  const page=environment(t),value=fixtures(),row=dailyBond(value);
  row[clock]=NOW+1;
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/未来时间无效/);
  assert.doesNotMatch(displayed,/日频有效/);
});

test('a future daily source date is invalid even when its timestamp is old',t=>{
  const page=environment(t),value=fixtures(),row=dailyBond(value);
  row.observed_date='2026-10-24';
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/未来时间无效/);
  assert.doesNotMatch(displayed,/日频有效/);
});

test('daily labeling never upgrades an existing stale quote or presents an intraday valid quote',t=>{
  const page=environment(t),value=fixtures(),row=dailyBond(value);
  row.freshness.status='stale';
  render(value.status,value.snapshot,value.calendar);
  assert.match(page.observations.children[0].children[4].textContent,/已过期.*非盘中报价/);
  row.freshness.status='fresh';
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/日频有效／不是盘中/);
  assert.doesNotMatch(displayed,/有效观察/);
});

test('invalid daily dates cannot be displayed as currently valid',t=>{
  const page=environment(t),value=fixtures(),row=dailyBond(value);
  row.observed_date='2026-02-30';
  render(value.status,value.snapshot,value.calendar);
  const displayed=page.observations.children[0].children[4].textContent;
  assert.match(displayed,/观察日期无效.*非盘中报价/);
  assert.doesNotMatch(displayed,/日频有效/);
});

test('a Brent roll replaces the displayed contract identity alongside its price and source link',t=>{
  const page=environment(t),value=fixtures();
  const row={...value.snapshot.observations.nq_dec_2026,value:115.2,unit:'USD/barrel',
    source_url:'https://query1.finance.yahoo.com/v8/finance/chart/BZZ26.NYM',
    instrument:{contract_month:'2026-12',symbol:'BZZ26.NYM',automatic_roll:false}};
  value.snapshot.observations={brent_usd_per_barrel:row};
  render(value.status,value.snapshot,value.calendar);
  assert.equal(page.observations.children[0].children[0].textContent,'Brent 具名期货 2026-12（BZZ26.NYM）');
  row.value=110.36;
  row.instrument={contract_month:'2027-01',symbol:'BZF27.NYM',automatic_roll:false};
  row.source_url='https://query2.finance.yahoo.com/v8/finance/chart/BZF27.NYM';
  render(value.status,value.snapshot,value.calendar);
  const cells=page.observations.children[0].children,link=cells[0].children[0];
  assert.equal(cells.length,6);
  assert.equal(link.textContent,'Brent 具名期货 2027-01（BZF27.NYM）');
  assert.equal(cells[1].textContent,'110.36');
  assert.equal(link.href,row.source_url);
  assert.match(link.rel,/noopener/);assert.match(link.rel,/noreferrer/);
  assert.doesNotMatch(page.observations.textContent,/BZZ26|2026-12/);
});

test('Nasdaq and WTI contract labels use row metadata rather than a hardcoded delivery month',t=>{
  const page=environment(t),value=fixtures(),base=value.snapshot.observations.nq_dec_2026;
  value.snapshot.observations={
    nq_dec_2026:{...base,instrument:{contract_month:'2026-12',symbol:'NQZ26.CME'}},
    wti_usd_per_barrel:{...base,instrument:{contract_month:'2027-02',symbol:'CLG27.NYM'}}
  };
  render(value.status,value.snapshot,value.calendar);
  assert.equal(page.observations.children[0].children[0].textContent,'纳指期货 2026-12（NQZ26.CME）');
  assert.equal(page.observations.children[1].children[0].textContent,'WTI 具名期货 2027-02（CLG27.NYM）');
  assert.doesNotMatch(page.observations.textContent,/NQ Dec26/);
});

test('absent or malformed named-contract metadata falls back safely to the generic label',t=>{
  const page=environment(t),value=fixtures();
  const row=value.snapshot.observations.nq_dec_2026;
  for(const instrument of [undefined,null,{},'NQZ26.CME',
    {contract_month:'2026-13',symbol:'NQZ26.CME'},
    {contract_month:'2026-12',symbol:'NQ=F'},
    {contract_month:'2026-12',symbol:'<img src=x onerror="alert(1)">'},
    {contract_month:'2026-12',symbol:'A'.repeat(100)}]){
    row.instrument=instrument;
    render(value.status,value.snapshot,value.calendar);
    const title=page.observations.children[0].children[0];
    assert.equal(title.textContent,'纳指期货');
    assert.equal(title.children[0].tagName,'a');
  }
  row.instrument={contract_month:'2026-12',symbol:'NQZ26.CME'};
  row.source_url='javascript:alert(1)';
  render(value.status,value.snapshot,value.calendar);
  const title=page.observations.children[0].children[0];
  assert.equal(title.textContent,'纳指期货 2026-12（NQZ26.CME）');
  assert.equal(title.children.length,0);
});

test('contract-like metadata on a currency or daily bond does not rename other observation types',t=>{
  const page=environment(t),value=fixtures(),base=value.snapshot.observations.nq_dec_2026;
  const instrument={contract_month:'2027-01',symbol:'BZF27.NYM'};
  value.snapshot.observations={usdjpy:{...base,instrument},de_10y_pct:{...base,instrument}};
  render(value.status,value.snapshot,value.calendar);
  assert.equal(page.observations.children[0].children[0].textContent,'USD/JPY');
  assert.equal(page.observations.children[1].children[0].textContent,'德国国债 10 年');
});
