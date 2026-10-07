"""Development-only choice among fixed mechanisms, then frozen chronological audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import backtest_current_accounts as public
import backtest_stock_swing_120 as engine
from replay_stock_swing_per_symbol import after_close_summary

ROOT=Path(__file__).resolve().parents[1]


def choices():
    exits={'tactical':{'target_margin_return':.3,'max_holding_calendar_days':3,
                      'trail_activation_underlying_return':.02,'trail_locked_underlying_return':.005},
           'swing':{'target_margin_return':.6,'max_holding_calendar_days':7,
                    'trail_activation_underlying_return':.04,'trail_locked_underlying_return':.01}}
    return {f'{family}_{period}_{name}':{'signal_family':family,'signal_timeframe':period,
        'ema_fast':8,'ema_mid':24,'ema_slow':60,'warmup_bars':170 if family=='dual_momentum' else 60,
        'stop_atr':2,'trail_atr':2,'entry_direction':'both','cooldown_signal_bars':3,**exitcfg}
        for family in ('donchian20','dual_momentum','rsi_reclaim','shock_reclaim')
        for period in ('1h','4h') for name,exitcfg in exits.items()}


def fetch(symbol,start,end,out):
    destination=out/'inputs'/(symbol+'.json.gz')
    if destination.exists():
        return json.loads(gzip.decompress(destination.read_bytes()))
    import requests
    base='https://fapi.binance.com/fapi/v1' if symbol=='BTCUSDT' else 'https://fapi.asterdex.com/fapi/v3'
    session=requests.Session()
    rules=next(r for r in public.public_get(session,base,'/exchangeInfo')['symbols'] if r['symbol']==symbol)
    effective=max(start,((int(rules['onboardDate'])+engine.HOUR-1)//engine.HOUR)*engine.HOUR)
    source={'start_ms':effective,'rules_current':rules,'funding':[], 'gaps':{}}
    for label,endpoint in [('trade_1h','/klines'),('mark_1h','/markPriceKlines'),('index_1h','/indexPriceKlines')]:
        source[label],source['gaps'][label]=public.candles(session,base,endpoint,symbol,'1h',effective,end,engine.HOUR)
    source['funding']=public.funding(session,base,symbol,effective,end)
    value={'venue':base,'symbols':{symbol:source},'end_ms_exclusive':end,'execution_step_ms':engine.HOUR,
           'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode(),mtime=0))
    return value


def failures(s,minimum,pf):
    bad=[]
    if s['net_closed_pnl']<=0:bad.append('closed_net_nonpositive')
    if s['estimated_close_return_pct']<=0:bad.append('liquidation_net_nonpositive')
    if s['closed_trades']<minimum:bad.append('insufficient_closed_trades')
    if s['profit_factor'] is None or s['profit_factor']<pf:bad.append('profit_factor_below_threshold')
    if s['max_sampled_drawdown_pct']>6:bad.append('drawdown_over_6pct')
    if s['liquidation_stress_count']:bad.append('liquidation_stress')
    return bad


def select(experiments):
    eligible=[]
    for name,runs in experiments.items():
        if runs.get('normal_cost_run_skipped_after_strict_cost_failure'):
            continue
        if not any(failures(runs[f'development_cost{c}'],10,1.15) for c in (1,2)):
            eligible.append((min(runs[f'development_cost{c}']['net_closed_pnl'] for c in (1,2)),name))
    return max(eligible)[1] if eligible else None


def evaluate(symbol,snapshot,variants,base,out):
    source=snapshot['symbols'][symbol]
    if any(source['gaps'].values()):
        return {'unavailable':'source_hourly_gaps','gaps':source['gaps']}
    fee=.00045 if symbol=='BTCUSDT' else .001 if symbol=='SKHYNIXUSDT' else .000125
    common={**base,'symbols':[symbol],'initial_equity_usdt':1000,'max_positions':1,
        'risk_fraction_per_trade':.0025,'portfolio_stop_risk_fraction':.0025,
        'execution_timeframe':'1h','entry_signal_validity_minutes':0,
        'entry_max_previous_bar_participation_fraction':.1,'taker_fee_rate_assumption':fee,
        'min_stop_fraction':max(.005,8*(fee+.0002)),'max_stop_fraction':.05}
    end=snapshot['end_ms_exclusive'];split=engine.parse_time('2026-09-01T00:00:00Z')
    recent=end-30*engine.DAY
    experiments={};cache={}
    for name,overrides in variants.items():
        cfg={**common,**overrides,'symbol_profiles':{symbol:overrides}}
        key=(overrides['signal_family'],overrides['signal_timeframe'])
        if key not in cache:cache[key]=engine.prepare(snapshot,cfg)
        # Exit settings do not change causal entry signals.
        prepared={symbol:{**cache[key][symbol],'profile_config':cfg}}
        result=engine.simulate(snapshot,cfg,source['start_ms']+30*engine.DAY,split,2,prepared_data=prepared)
        runs={'development_cost2':after_close_summary(result)}
        # First screen at the stricter cost; do not inspect audits here.
        if not failures(runs['development_cost2'],10,1.15):
            runs['development_cost1']=after_close_summary(engine.simulate(snapshot,cfg,
                source['start_ms']+30*engine.DAY,split,1,prepared_data=prepared))
        else:
            runs['development_cost1']=None
            runs['normal_cost_run_skipped_after_strict_cost_failure']=True
        experiments[name]=runs
        print(symbol,name,'dev_cost2',round(runs['development_cost2']['net_closed_pnl'],3),
              'n',runs['development_cost2']['closed_trades'],flush=True)
    winner=select(experiments)
    record={'experiments':experiments,'development_selected':winner,'audit_passed':False}
    public.save(out/(symbol+'_development_choice.json'),record)  # Freeze before opening audit results.
    if winner:
        overrides=variants[winner];cfg={**common,**overrides,'symbol_profiles':{symbol:overrides}}
        prepared={symbol:{**cache[(overrides['signal_family'],overrides['signal_timeframe'])][symbol],'profile_config':cfg}}
        runs=experiments[winner]; reasons=[]
        for window,start,minimum in [('validation',split,5),('recent30d',recent,3),
                                     ('full',source['start_ms']+30*engine.DAY,10)]:
            for cost in (1,2):
                label=f'{window}_cost{cost}'
                result=engine.simulate(snapshot,cfg,start,end,cost,prepared_data=prepared)
                runs[label]=after_close_summary(result)
                engine.write_csv(out/(symbol+'_'+label+'_trades.csv'),result['trades'])
                reasons += [label+':'+reason for reason in failures(runs[label],minimum,1.1)]
                if window in ('validation','full') and result['trades']:
                    net=runs[label]['net_closed_pnl']-max(t['net_pnl'] for t in result['trades'])
                    runs[label]['net_without_best_winner']=net
                    if net<=0:reasons.append(label+':nonpositive_without_best_winner')
        stress=engine.simulate(snapshot,{**cfg,'adverse_slippage_fraction_assumption':.001},split,end,2,
                               prepared_data=prepared)
        record.update(selected_config=cfg,selected_runs=runs,stress=after_close_summary(stress),
                      audit_failures=reasons,audit_passed=not reasons)
    public.save(out/(symbol+'_results.json'),record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--download-only',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Preserve completed results')
    start=engine.parse_time('2026-02-01T00:00:00Z');end=engine.parse_time('2026-10-07T09:00:00Z')
    variants=choices()
    declaration={'declared_at_utc':datetime.now(timezone.utc).isoformat(),'variants':variants,
        'candidate_count_per_account':len(variants),'accounts':['BTCUSDT','MUUSDT','SNDKUSDT','SKHYNIXUSDT'],
        'development_end_exclusive':'2026-09-01T00:00:00Z', 'audit_end_exclusive':engine.iso(end),
        'selection':'Development only: >0 closed and estimated liquidation PnL at both costs, >=10 closed trades, PF>=1.15, DD<=6%, no liquidation. Rank worst-cost net PnL. Freeze before audit; no fallback to audit winner.',
        'audit':'Frozen winner: validation >=5/recent30d >=3/full>=10 trades, positive closed and estimated liquidation PnL at both costs, PF>=1.1, DD<=6%, no liquidation. Validation/full remain positive without best winner.',
        'known_history_previously_reviewed':True,'independent_holdout':False,'places_orders':False,
        'limitations':['Hourly OHLC screening uses conservative stop-before-target ordering; actual book and intrabar path unavailable.',
          'Entries capped by previous complete hourly volume, not historical visible depth. Selected candidates still need 5m production-account confirmation.',
          'Current fee assumptions applied to old history, historical fee changes unavailable.',
          'BTC candidates are independent price mechanisms, not the existing six-factor strategy.',
          '64 mechanism/period/exit comparisons can overfit. A retrospective pass is not forward profitability.'],
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}}
    public.save(out/'declaration.json',declaration)
    symbols=declaration['accounts']
    with ThreadPoolExecutor(max_workers=3) as pool:
        sources=dict(zip(symbols,pool.map(lambda s:fetch(s,start,end,out),symbols)))
    if args.download_only:return
    base=json.loads((ROOT/'config/stock_swing_120_candidate_20261004.json').read_text(encoding='utf-8'))
    results={'declaration':declaration,'accounts':{}}
    for symbol in symbols:
        results['accounts'][symbol]=evaluate(symbol,sources[symbol],variants,base,out)
        public.save(out/'progress.json',results)
        print('SELECTED',symbol,results['accounts'][symbol].get('development_selected'),
              'AUDIT',results['accounts'][symbol].get('audit_passed'),flush=True)
    results['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').glob('*.gz')}
    public.save(out/'results.json',results)


if __name__=='__main__':main()
