#!/usr/bin/env python3
"""Frozen, separate SNDK net60 forward experiment using public reads only.

The existing parallel accounts are never opened. Reuse their audited simulated
depth fills and protective exits through a process-local signal adapter; no
historical source file is edited and no exchange order endpoint is available.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading

import requests

import run_parallel_simulation as paper
import backtest_stock_swing_120 as engine
from research_selective_stock_entries import trend_side, volume_confirmed
from stock_external_context import MarketContext, HOUR, DELAY
from stock_swing_signals import compute_indicators, signal_at

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'config/stock_sndk_net60_shadow_20261005.json'
STOP = threading.Event()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ForwardNQ:
    def __init__(self, root):
        self.root = root
        self.context = None
        self.first_seen_ms = None
        self.next_refresh_ms = 0
        self.error = None

    def refresh(self):
        now = paper.now_ms()
        if now < self.next_refresh_ms:
            return
        self.next_refresh_ms = now + 60_000
        try:
            url = 'https://query1.finance.yahoo.com/v8/finance/chart/NQ%3DF'
            response = requests.get(url, params={'period1':(now-14*24*HOUR)//1000,
                'period2':now//1000,'interval':'1h'},headers={'User-Agent':'btc-auto-public-research/1.0'},timeout=12)
            observed = paper.now_ms()
            raw = response.content
            path = self.root / 'nq_observations' / f'{observed}.json.gz'
            path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(path,'wb') as handle:
                handle.write(raw)
            response.raise_for_status()
            chart = response.json()['chart']['result'][0]
            if chart['meta'].get('dataGranularity') != '1h':
                raise ValueError('NQ source is not hourly')
            rows = []
            closes = chart['indicators']['quote'][0]['close']
            for opening,close in zip(chart.get('timestamp',[]),closes):
                if close is None:
                    continue
                start = int(opening)*1000
                if not math.isfinite(close) or close<=0:
                    raise ValueError('Invalid NQ close')
                if start+HOUR+DELAY <= observed:
                    rows.append([start,start+HOUR+DELAY,float(close)])
            context = MarketContext({'series':{'nq':rows}})
            self.context,self.first_seen_ms,self.error = context,observed,None
            self.next_refresh_ms = observed+15*60_000
            paper.journal(self.root/'external_fetches.jsonl',{'observed_at_ms':observed,
                'source_url':url,'raw_path':str(path.relative_to(ROOT)),
                'raw_sha256':hashlib.sha256(raw).hexdigest(),'usable_closed_rows':len(rows)})
        except Exception as exc:
            self.error = type(exc).__name__+': '+str(exc)[:180]
            paper.journal(self.root/'external_fetches.jsonl',{'observed_at_ms':paper.now_ms(),'error':self.error})

    def decision(self, timestamp, side):
        reasons = []
        if self.error:
            reasons.append('nq_fetch_failed')
        if self.context is None or self.first_seen_ms is None or self.first_seen_ms > timestamp:
            reasons.append('nq_not_yet_observed')
        elif timestamp-self.first_seen_ms > 20*60_000:
            reasons.append('nq_fetch_stale')
        if side not in (-1,1):
            reasons.append('no_pending_stock_signal')
        if reasons:
            return {'allowed':False,'reasons':reasons,'decision_at_ms':timestamp,
                    'fetch_first_seen_ms':self.first_seen_ms}
        decision = self.context.decision(timestamp,side,'nq_confirmation')
        return {**decision,'fetch_first_seen_ms':self.first_seen_ms}


class GuardedSettings(dict):
    """Recheck the external source at each prospective entry, including retries."""
    def __init__(self, settings, context, expires):
        super().__init__(settings)
        self.context,self.expires,self.side,self.last_decision = context,expires,None,None

    def get(self,key,default=None):
        if key != 'entry_enabled':
            return super().get(key,default)
        now = paper.now_ms()
        self.last_decision = self.context.decision(now,self.side)
        if now>=self.expires:
            self.last_decision = {**self.last_decision,'allowed':False,
                'reasons':[*self.last_decision['reasons'],'forward_experiment_expired']}
        return super().get(key,default) and self.last_decision['allowed']


class ForwardSignalAdapter:
    """Preserve long EMA seeds, append only observed completed hourly candles."""
    def __init__(self, seed_rows, cache_path):
        self.cache_path = cache_path
        self.rows = {int(row[0]):row for row in seed_rows}
        if cache_path.exists():
            with gzip.open(cache_path,'rt') as handle:
                self.rows.update({int(row[0]):row for row in json.load(handle)})

    def signal_at(self, bars, unused_indicators, index, config):
        end = bars[index].time_ms+HOUR
        for bar in bars[:index+1]:
            self.rows[bar.time_ms] = [bar.time_ms,bar.open,bar.high,bar.low,bar.close,bar.volume,bar.time_ms+HOUR-1]
        rows = [engine.candle(r) for t,r in sorted(self.rows.items()) if t+HOUR<=end]
        if any(b.time_ms-a.time_ms != HOUR for a,b in zip(rows,rows[1:])):
            raise ValueError('Hourly signal history has gaps; new entries blocked')
        temporary = self.cache_path.with_suffix('.tmp.gz')
        with gzip.open(temporary,'wt') as handle:
            json.dump([self.rows[t] for t in sorted(self.rows)],handle,separators=(',',':'))
        temporary.replace(self.cache_path)
        archive=self.cache_path.parent/'signal_observations'/f'{end}_{sha(self.cache_path)[:16]}.json.gz'
        if not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True)
            archive.write_bytes(self.cache_path.read_bytes())
            paper.journal(self.cache_path.parent/'signal_observations.jsonl',{
                'observed_at_ms':paper.now_ms(),'signal_close_ms':end,
                'snapshot_path':str(archive.relative_to(ROOT)),
                'snapshot_sha256':sha(archive)})
        higher = engine.aggregate_4h(rows)
        indicators = compute_indicators(rows,config)
        higher_indicators = compute_indicators(higher,config)
        higher_side = trend_side(higher_indicators,len(higher)-1,config)
        candidate = signal_at(rows,indicators,len(rows)-1,config)
        if not candidate or candidate.direction!=higher_side or not volume_confirmed(rows,len(rows)-1):
            config.side = None
            return None
        config.side = candidate.direction
        return candidate


def report(state):
    # Wallet identity incorporates late funding settled against closed inventory.
    return {k:state.get(k) for k in ('status','equity','wallet_balance','realized_pnl','fees_paid',
        'funding_pnl','position_qty','fill_count_total','max_drawdown_pct','signal_status',
        'entry_blockers','entry_checks','external_entry_gate','observations','errors')}


def run(once=False,poll_seconds=30):
    profile=json.loads(PROFILE.read_text())
    if profile['places_orders'] is not False or profile['mode']!='separate_forward_research':
        raise ValueError('Only separate paper research is permitted')
    root=ROOT/'data/paper_trading'/profile['candidate_id']
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[PROFILE,Path(__file__),ROOT/'scripts/stock_external_context.py',
        ROOT/'scripts/research_selective_stock_entries.py',ROOT/'scripts/run_parallel_simulation.py',
        ROOT/'scripts/stock_swing_signals.py',ROOT/'scripts/backtest_stock_swing_120.py',
        ROOT/'scripts/stock_research_entry_policy.py']
    snapshot=ROOT/profile['seed_snapshot_path']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [*files,snapshot]}
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['source_and_seed_sha256']!=hashes:
            raise RuntimeError('Frozen shadow source/config changed; use a new experiment ID')
    else:
        start=paper.now_ms()
        manifest={'candidate_id':profile['candidate_id'],'start_ms':start,
            'started_at_utc':engine.iso(start),'new_entry_expiry_ms':start+30*24*HOUR,
            'source_and_seed_sha256':hashes,'places_orders':False,'forward_validated':False,
            'production_admission_approved':False,'profile':profile}
        paper.write_json(manifest_path,manifest)
    context=ForwardNQ(root)
    with gzip.open(snapshot,'rt') as handle:
        seed=json.load(handle)['symbols']['SNDKUSDT']['trade_1h']
    # This isolated process has no BTC or existing stock workers. Replace only
    # its signal dispatch reference, leaving source files and other processes intact.
    paper.stock_swing_profiles=ForwardSignalAdapter(seed,root/'signal_history.json.gz')
    workers={}
    for label,multiple in (('normal_cost',1),('double_cost',2)):
        cfg={**profile['settings'],
            'taker_fee_rate_assumption':profile['settings']['taker_fee_rate_assumption']*multiple,
            'adverse_slippage_fraction_assumption':profile['settings']['adverse_slippage_fraction_assumption']*multiple}
        path=root/label/'state.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            state=paper.initial_stock('SNDKUSDT',manifest['start_ms'],cfg)
            state['signal_active_after_ms']=manifest['start_ms']
            paper.write_json(path,state)
        elif json.loads(path.read_text())['rule']!=cfg:
            raise RuntimeError('Refusing to change a frozen existing shadow account')
        guarded=GuardedSettings(cfg,context,manifest['new_entry_expiry_ms'])
        workers[label]=paper.StockAccount(path,paper.PublicAster(),'SNDKUSDT',guarded)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:STOP.set())
    status={}
    while not STOP.is_set():
        context.refresh()
        for label,worker in workers.items():
            try:
                before=json.loads(worker.path.read_text())
                pending=before.get('pending_signal')
                worker.config.side=pending['signal']['direction'] if pending else None
                state=worker.step()
                decision=worker.config.last_decision
                state['external_entry_gate']=decision
                if not state.get('position') and decision and not decision['allowed']:
                    state['entry_blockers']=[*state.get('entry_blockers',[]),*decision['reasons']]
                    state['signal_status']='forward_experiment_expired' if 'forward_experiment_expired' in decision['reasons'] else 'external_context_or_signal_blocked'
                paper.write_json(worker.path,state)
                paper.journal(worker.path.with_name('external_decisions.jsonl'),{'observed_at_ms':paper.now_ms(),'decision':decision})
                status[label]=report(state)
            except Exception as exc:
                status[label]={**status.get(label,{}),'status':'degraded','errors':{'worker':type(exc).__name__+': '+str(exc)[:180]}}
                paper.journal(root/'errors.jsonl',{'time_ms':paper.now_ms(),'account':label,**status[label]})
        paper.write_json(root/'status.json',{'candidate_id':profile['candidate_id'],'pid':os.getpid(),
            'running':not STOP.is_set(),'updated_at_utc':engine.iso(paper.now_ms()),
            'places_orders':False,'forward_validated':False,'production_admission_approved':False,
            'accounts':status,'external_source_error':context.error})
        if once:
            break
        STOP.wait(poll_seconds)
    final_path=root/'status.json'
    final_status=json.loads(final_path.read_text())
    final_status.update(running=False,updated_at_utc=engine.iso(paper.now_ms()))
    paper.write_json(final_path,final_status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--poll-seconds',type=int,default=30)
    args=parser.parse_args()
    if args.poll_seconds<5:
        raise ValueError('Poll interval must be at least5seconds')
    run(args.once,args.poll_seconds)
