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
