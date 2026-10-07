#!/usr/bin/env python3
"""Frozen six-ledger overnight experiment; public market reads, no real orders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from process_lock import exclusive_process_lock
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import threading

import run_parallel_simulation as paper
from run_stock_expectancy_shadow import ForwardNQ, GuardedSettings, ForwardSignalAdapter, report
from stock_profit_candidate import signal_at as fast_signal_at
from stock_swing_signals import compute_indicators
import backtest_stock_swing_120 as engine
from stock_overnight_report import make_report, markdown

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'config/stock_overnight_validation_20261005.json'
STOP=threading.Event()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FastAdapter(ForwardSignalAdapter):
    def __init__(self,seed,path,arm):
        super().__init__(seed,path)
        self.arm=arm

    def signal_at(self,bars,unused,index,config):
        end=bars[index].time_ms+engine.HOUR
        for bar in bars[:index+1]:
            self.rows[bar.time_ms]=[bar.time_ms,bar.open,bar.high,bar.low,bar.close,bar.volume,bar.time_ms+engine.HOUR-1]
        candles=[engine.candle(r) for t,r in sorted(self.rows.items()) if t+engine.HOUR<=end]
        if any(b.time_ms-a.time_ms!=engine.HOUR for a,b in zip(candles,candles[1:])):
            raise ValueError('Hourly warmup history has gaps; entry blocked')
        tmp=self.cache_path.with_suffix('.tmp.gz')
        with gzip.open(tmp,'wt') as handle:
            json.dump([self.rows[t] for t in sorted(self.rows)],handle,separators=(',',':'))
        tmp.replace(self.cache_path)
        archive=self.cache_path.parent/'signal_observations'/f'{end}_{digest(self.cache_path)[:16]}.json.gz'
        if not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True)
            archive.write_bytes(self.cache_path.read_bytes())
            paper.journal(self.cache_path.parent/'signal_observations.jsonl',{
                'observed_at_ms':paper.now_ms(),'signal_close_ms':end,
                'snapshot_path':str(archive.relative_to(ROOT)), 'snapshot_sha256':digest(archive)})
        candidate=fast_signal_at(candles,compute_indicators(candles,config),len(candles)-1,config,self.arm)
        config.side=candidate.direction if candidate else None
        return candidate


class SymbolDispatch:
    def __init__(self,adapters):
        self.adapters=adapters

    def signal_at(self,bars,indicators,index,config):
        symbol=config['symbols'][0]
        return self.adapters[symbol].signal_at(bars,indicators,index,config)


def request_end_exit(path,timestamp):
    state=json.loads(path.read_text())
    if state.get('position') and not state['position'].get('pending_exit'):
        state['position']['pending_exit']='overnight_validation_end'
        state['position']['trigger_observed_at_ms']=timestamp
        paper.write_json(path,state)


def frozen_manifest(profile,root):
    source_names=['run_stock_overnight_validation.py','stock_overnight_report.py',
        'run_stock_expectancy_shadow.py','run_parallel_simulation.py','stock_profit_candidate.py',
        'stock_swing_signals.py','stock_swing_profiles.py','stock_external_context.py',
        'stock_research_entry_policy.py','backtest_stock_swing_120.py','research_selective_stock_entries.py',
        'trading_execution.py','decision_runtime.py']
    paths=[PROFILE,ROOT/profile['seed_snapshot_path'],*[ROOT/'scripts'/n for n in source_names]]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    path=root/'manifest.json'
    end=engine.parse_time(profile['end_utc'])
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['source_and_seed_sha256']!=hashes:
            raise RuntimeError('Frozen source/profile changed: new experiment ID required')
        return manifest
    start=paper.now_ms()
    if end<=start:
        raise ValueError('New overnight experiment must end in the future')
    manifest={'experiment_id':profile['experiment_id'],'start_ms':start,
        'started_at_utc':engine.iso(start),'end_ms':end,'end_utc':profile['end_utc'],
        'source_and_seed_sha256':hashes,'profile':profile,'places_orders':False,
        'production_admission_approved':False,'forward_validated':False}
    paper.write_json(path,manifest)
    return manifest


def run(once=False,poll_seconds=30):
    profile=json.loads(PROFILE.read_text())
    if profile['places_orders'] is not False or profile['mode']!='overnight_paper_research':
        raise ValueError('Only public-data overnight simulation allowed')
    root=ROOT/'data/paper_trading'/profile['experiment_id']
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_process_lock(root / 'runner.lock'):
        _run_locked(profile, root, once, poll_seconds)


def _run_locked(profile, root, once, poll_seconds):
    manifest=frozen_manifest(profile,root)
    if (root/'completed.json').exists():
        return
    context=ForwardNQ(root)
    with gzip.open(ROOT/profile['seed_snapshot_path'],'rt') as handle:
        seeds=json.load(handle)['symbols']
    adapters={}
    workers={}
    for symbol,cfg in profile['symbol_profiles'].items():
        directory=root/symbol
        directory.mkdir(parents=True,exist_ok=True)
        seed=seeds[symbol]['trade_1h']
        if cfg['entry_arm']=='trend_volume_breakout_net60':
            adapters[symbol]=ForwardSignalAdapter(seed,directory/'signal_history.json.gz')
        else:
            adapters[symbol]=FastAdapter(seed,directory/'signal_history.json.gz',cfg['entry_arm'])
        workers[symbol]={}
        for label,multiple in (('normal_cost',1),('double_cost',2)):
            rule={**cfg,'taker_fee_rate_assumption':cfg['taker_fee_rate_assumption']*multiple,
                'adverse_slippage_fraction_assumption':cfg['adverse_slippage_fraction_assumption']*multiple}
            path=directory/label/'state.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if json.loads(path.read_text())['rule']!=rule:
                    raise RuntimeError('Existing overnight rule changed; refusing ledger mutation')
            else:
                state=paper.initial_stock(symbol,manifest['start_ms'],rule)
                state['signal_active_after_ms']=manifest['start_ms']
                paper.write_json(path,state)
            guarded=GuardedSettings(rule,context,manifest['end_ms'])
            workers[symbol][label]=paper.StockAccount(path,paper.PublicAster(),symbol,guarded)
    # Process-local dispatch: the original runner and original ledgers are untouched.
    paper.stock_swing_profiles=SymbolDispatch(adapters)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:STOP.set())
    views={}
    next_report=0
    ended=False
    def tick_symbol(symbol):
        result={}
        for label,worker in workers[symbol].items():
            try:
                now=paper.now_ms()
                if now>=manifest['end_ms']:
                    request_end_exit(worker.path,now)
                before=json.loads(worker.path.read_text())
                pending=before.get('pending_signal')
                worker.config.side=pending['signal']['direction'] if pending else None
                state=worker.step()
                gate=worker.config.last_decision
                state['external_entry_gate']=gate
                if not state.get('position') and gate and not gate['allowed']:
                    state['entry_blockers']=[x for x in state.get('entry_blockers',[]) if x!='strategy_not_qualified']+gate['reasons']
                    state['signal_status']='overnight_complete' if now>=manifest['end_ms'] else 'external_context_or_signal_blocked'
                paper.write_json(worker.path,state)
                paper.journal(worker.path.with_name('external_decisions.jsonl'),{
                    'observed_at_ms':paper.now_ms(),'decision':gate})
                result[label]=report(state)
            except Exception as exc:
                result[label]={**views.get(symbol,{}).get(label,{}),'status':'degraded',
                    'errors':{'worker':type(exc).__name__+': '+str(exc)[:180]}}
                paper.journal(root/'errors.jsonl',{'observed_at_ms':paper.now_ms(),'symbol':symbol,'cost':label,**result[label]})
        return symbol,result
    with ThreadPoolExecutor(max_workers=3) as pool:
        while not STOP.is_set():
            # Refresh failures only veto entries; liquidation/stop/end exits still run.
            context.refresh()
            for symbol,result in pool.map(tick_symbol,workers):
                views[symbol]=result
            now=paper.now_ms()
            deadline=now>=manifest['end_ms']
            all_flat=all(not json.loads(w.path.read_text()).get('position') for costs in workers.values() for w in costs.values())
            # Pending funding publication keeps protection/settlement polling active.
            all_healthy=all(v.get('status')=='healthy' for costs in views.values() for v in costs.values())
            ended=deadline and all_flat and all_healthy
            status={'experiment_id':profile['experiment_id'],'pid':os.getpid(),
                'running':not STOP.is_set() and not ended,'places_orders':False,
                'started_at_utc':manifest['started_at_utc'],'end_utc':profile['end_utc'],
                'updated_at_utc':engine.iso(now),'phase':'completed' if ended else 'closing' if deadline else 'observing',
                'accounts':views,'external_source_error':context.error,
                'forward_validated':False,'production_admission_approved':False}
            paper.write_json(root/'status.json',status)
            if now>=next_report or deadline or once:
                outcome=make_report(root,profile,now,ended)
                paper.write_json(root/'overnight_report.json',outcome)
                (root/'overnight_report.md').write_text(markdown(outcome))
                next_report=now+5*60_000
            if ended:
                paper.write_json(root/'completed.json',{'completed_at_utc':engine.iso(now),'places_orders':False})
                break
            if once:
                break
            STOP.wait(poll_seconds)
    status=json.loads((root/'status.json').read_text())
    status.update(running=False,updated_at_utc=engine.iso(paper.now_ms()))
    paper.write_json(root/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--poll-seconds',type=int,default=30)
    args=parser.parse_args()
    if args.poll_seconds<5:
        raise ValueError('Poll interval must be at least five seconds')
    run(args.once,args.poll_seconds)
