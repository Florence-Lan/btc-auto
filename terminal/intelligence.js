const $ = id => document.getElementById(id);
const fmt = (value, digits=2) => value == null ? '—' : Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits,minimumFractionDigits:digits});
const percent = (value, digits=2) => value == null ? '—' : `${fmt(value,digits)}%`;
const money = value => value == null ? '—' : fmt(value,0);
const time = value => value == null ? '—' : new Date(value).toLocaleString('zh-CN',{hour12:false});
function lines(id, rows) {
  const target=$(id); target.replaceChildren();
  for (const [label,value] of rows) {
    const line=document.createElement('div'); line.className='line';
    const key=document.createElement('span'), val=document.createElement('strong');
    key.textContent=label; val.textContent=value; line.append(key,val);target.append(line);
  }
}
function table(id, rows, empty) {
  const body=$(id);body.replaceChildren();
  for (const values of rows.length ? rows : [[empty]]) {
    const tr=document.createElement('tr');
    for (const value of values) {const td=document.createElement('td');td.textContent=value;tr.append(td);}
    body.append(tr);
  }
}
const names={spot_flow:'现货主动成交',futures_flow:'合约主动成交',open_interest:'BTC 持仓量',global_accounts:'全体账户多空比',top_accounts:'大户账户多空比',top_positions:'大户持仓多空比',spot_book:'现货盘口',futures_book:'合约盘口',funding_basis:'资金费率与基差',spot_tape:'现货成交样本',futures_tape:'合约成交样本',hyper_tape:'Hyperliquid 成交样本',hyper_context:'Hyperliquid BTC',hyper_positions:'链上持仓样本',bitcoin_chain:'Bitcoin 内存池'};
const states={buy_pressure:'买盘偏强',sell_pressure:'卖盘偏强',neutral:'方向不明显',conflicting:'现货与合约分歧',insufficient_data:'关键数据不足'};
function render(data) {
  const f=data.features||{}, d=data.direction||{}, age=(Date.now()-data.generated_at_ms)/1000, stale=age>180;
  $('freshness').textContent=stale ? `数据已过期 · ${Math.round(age/60)} 分钟前` : `采集中 · ${Math.max(0,Math.round(age))} 秒前`;
  $('freshness').classList.toggle('bad',stale);$('asof').textContent=time(data.generated_at_ms);
  $('direction').textContent=stale ? '等待新数据' : `${states[d.state]||'等待数据'}${d.score==null?'':` ${fmt(d.score,1)}`}`;
  lines('flows', [['现货主动净买卖占比',percent(f.spot_flow?.imbalance_1h==null?null:f.spot_flow.imbalance_1h*100)],['合约主动净买卖占比',percent(f.futures_flow?.imbalance_1h==null?null:f.futures_flow.imbalance_1h*100)],['BTC 1h 涨跌',percent(f.futures_flow?.price_change_1h_pct)]]);
  lines('leverage', [['合约持仓量 · BTC',fmt(f.open_interest?.oi_btc,1)],['持仓量 1h 变化',percent(f.open_interest?.change_1h_pct)],['全体账户 多/空',fmt(f.global_accounts?.long_short_ratio)],['大户账户 多/空',fmt(f.top_accounts?.long_short_ratio)],['大户持仓 多/空',fmt(f.top_positions?.long_short_ratio)]]);
  lines('books', [['合约基差 · 基点',fmt(f.funding_basis?.basis_bps)],['Binance 最近资金费率',percent(f.funding_basis?.last_funding_rate==null?null:f.funding_basis.last_funding_rate*100,4)],['现货买卖价差 · 基点',fmt(f.spot_book?.spread_bps)],['合约买卖价差 · 基点',fmt(f.futures_book?.spread_bps)],['合约买盘 / 卖盘金额',`${money(f.futures_book?.bid_10bps_notional)} / ${money(f.futures_book?.ask_10bps_notional)}`]]);
  const tapeNames={spot_tape:'Binance 现货',futures_tape:'Binance 合约',hyper_tape:'Hyperliquid'};
  const whales=[];const spans=[];
  for(const [key,name] of Object.entries(tapeNames)) {
    if(!f[key]){spans.push(`${name} 不可用`);continue;}
    spans.push(`${name} ${f[key].trades} 笔 / ${fmt(f[key].sample_span_seconds,1)} 秒${f[key].gap_since_previous_poll?'，与上轮之间存在缺口':''}`);
    for(const row of f[key].large_trades||[]) whales.push({name,currency:key==='hyper_tape'?'USD':'USDT',...row});
  }
  whales.sort((a,b)=>b.notional-a.notional);
  table('whales',whales.slice(0,20).map(r=>[r.name,time(r.time_ms),r.side==='buy'?'主动买入':'主动卖出',`${money(r.notional)} ${r.currency}`,fmt(r.qty_btc,5)]),'本轮已收集样本中未发现达到阈值的大单');
  $('tapeCoverage').textContent=`${spans.join('；')}。每个市场展示本轮最大 20 笔；样本不是完整清单，跨轮不要累加同一笔成交。`;
  lines('hyper',[['BTC 持仓金额 · USD',money(f.hyper_context?.oi_usd)],['1h 资金费率',percent(f.hyper_context?.funding_rate_1h==null?null:f.hyper_context.funding_rate_1h*100,5)],['24h 成交金额 · USD',money(f.hyper_context?.volume_24h_usd)]]);
  lines('positions',[['成功查询 / 选中地址',f.hyper_positions?`${f.hyper_positions.successful_addresses} / ${f.hyper_positions.queried_addresses}`:'—'],['样本多头金额 · USD',money(f.hyper_positions?.long_usd)],['样本空头金额 · USD',money(f.hyper_positions?.short_usd)],['样本多/空比',fmt(f.hyper_positions?.long_short_ratio)]]);
  lines('chain',[['本轮交易样本数',f.bitcoin_chain?.sample_size??'—'],['输出 ≥100 BTC 的笔数',f.bitcoin_chain?.large_output_transactions?.length??'—'],['可推断的买卖方向','不可推断']]);
  lines('context',[['已核实新闻风险分',fmt(data.news?.score)],['新闻数据状态',data.news?.feed_status||'不可用'],['宏观风险偏好分',fmt(data.macro?.score)],['宏观最早因子时间',time(data.macro?.asof_ms)]]);
  const flags={spot_futures_flow_disagree:'现货和合约主动资金方向相反',extreme_last_funding:'最近资金费率绝对值较高',global_accounts_crowded:'全体账户偏向拥挤',top_positions_crowded:'大户持仓偏向拥挤',rapid_deleveraging:'持仓量快速下降'};
  lines('risks',(data.risk_flags||[]).length ? data.risk_flags.map(x=>['观察',flags[x]||x]):[['本轮已覆盖规则','未触发阈值']]);
  table('health',Object.entries(data.health||{}).map(([k,v])=>[names[k]||k,v.status==='ok'?'可用':v.status,time(v.available_at_ms),v.error||'—']),'暂无数据源记录');
}
async function refresh(){
  try {const r=await fetch(`/data/market_intelligence/latest.json?_=${Date.now()}`,{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);render(await r.json());}
  catch(e){$('freshness').textContent='读取失败 · 请检查监控进程';$('freshness').classList.add('bad');$('direction').textContent='数据不可用';}
}
refresh();setInterval(refresh,15000);
