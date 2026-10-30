"""Isolated price-led short confirmation, using existing immutable public inputs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import FunctionType
import os
import time

import run_stock_technology_trial as feed
import stock_short_confirmation as short

ROOT = feed.ROOT
paper = feed.paper
shared = feed.shared
infrastructure = feed.infrastructure
timing_trial = feed.timing_trial
DEFAULT_CONFIG = ROOT/'config/stock_short_confirmation_trial_20261030.json'
SYMBOLS = timing_trial.SYMBOLS
COSTS = ('normal','double')
ARMS = tuple(variant+'_'+cost for cost in COSTS for variant in ('baseline','candidate'))
digest = feed.digest
source_batch = feed.source_batch


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = paper.read_json(path)
    fixed = {'schema_version':1,'mode':'SIMULATION','places_orders':False,
        'symbols':list(SYMBOLS),'variants':['baseline','candidate'],
        'cost_multipliers':{'normal':1,'double':2},'initial_balance_usdt_each':1000,
        'automatic_strategy_promotion':False,'historical_warmup_counts_as_forward_performance':False,
        'review_minimum_elapsed_days':14,'review_minimum_closed_positions_each':10,
        'candidate_only_added_rule':'short_confirmation_policy','poll_seconds':1,
        'source_trial':'config/stock_entry_timing_trial_20261029.json',
        'market_feed':'existing_entry_timing_trial_immutable_context'}
    if any(spec.get(k)!=value or type(spec.get(k)) is not type(value) for k,value in fixed.items()):
        raise ValueError('Invalid fixed short-confirmation experiment design')
    if type(spec.get('effective_from_ms')) is not int or spec['effective_from_ms']<=0:
        raise ValueError('Explicit prospective declaration required')
    if spec.get('short_confirmation_policy')!=short.POLICY:
        raise ValueError('Fixed short-confirmation policy required')
    output = infrastructure.contained(spec['output_dir'],'data/paper_trading')
    if output in (feed.SOURCE,ROOT/'data/paper_trading/stock_technology_trial_20261030'):
        raise ValueError('Independent new output required')
    old,old_rules,old_fp = timing_trial.load_design(ROOT/spec['source_trial'])
    timing_trial.check_source_rules(old,old_rules)
    manifest = paper.read_json(feed.SOURCE/'manifest.json')
    if manifest['fingerprint']!=old_fp or manifest['rules']!=old_rules:
        raise ValueError('Existing market feed source/rule identity changed')
    rules = {}
    for symbol in SYMBOLS:
        rules[symbol] = {}
        for arm in ARMS:
            variant,cost = arm.rsplit('_',1)
            cfg = deepcopy(old_rules[symbol]['baseline_'+cost])
            cfg['stock_event_risk_policy']['calendar_path'] = (output/'calendar_current.json').relative_to(ROOT).as_posix()
            cfg[timing_trial.macro.POLICY_KEY].update(
                snapshot_path=(output/'current_context/public_context.json').relative_to(ROOT).as_posix(),
                effective_from_ms=spec['effective_from_ms'])
            if variant=='candidate':
                cfg['short_confirmation_policy'] = deepcopy(short.POLICY)
                short.validate(cfg)
            paper.stock_entry_timing.validate(cfg)
            rules[symbol][arm] = cfg
    profiles = {**old_fp['profile_sha256'],spec['source_trial']:digest((ROOT/spec['source_trial']).read_bytes())}
    fp = {'experiment_config_sha256':digest(path.read_bytes()),'profile_sha256':profiles,
        'source_sha256':feed.clone(timing_trial.source_hashes,__file__=__file__)(),
        'rules_sha256':{s:{a:digest(c) for a,c in arms.items()} for s,arms in rules.items()}}
    return spec,rules,fp


def prepare(output,spec,rules,fingerprint,inception_ms,require_existing=False):
    return feed.clone(timing_trial.prepare,ARMS=ARMS)(output,spec,rules,fingerprint,inception_ms,require_existing)


class ShortProfiles:
    def __init__(self,inception_ms):
        self.inception_ms = inception_ms

    def signal_with_diagnostics(self,bars,indicators,index,config):
        if bars[index].time_ms+timing_trial.HOUR<=self.inception_ms:
            return timing_trial.TimingProfiles(self.inception_ms).signal_with_diagnostics(bars,indicators,index,config)
        return short.signal_with_diagnostics(bars,indicators,index,config)


def isolated_call(method,worker,*args):
    dependencies = {**method.__globals__,'stock_swing_profiles':ShortProfiles(worker.activated_at_ms),
                    'stock_entry_timing':paper.stock_entry_timing}
    original_journal = dependencies['journal']
    def journal(path,row):
        if Path(path).name in ('signals.jsonl','fills.jsonl','entry_timing_decisions.jsonl'):
            row.update(short_confirmation_variant=worker.variant,
                       short_confirmation_context_sha256=worker.shared_context.get(worker.symbol),
                       macro_context_sha256=worker.macro_context_sha256)
        if Path(path).name=='signals.jsonl':
            seen = worker.now_ms()
            row.update(first_seen_at_ms=seen,first_seen_lag_ms=seen-row['boundary_ms'])
        original_journal(path,row)
    dependencies['journal'] = journal
    cloned = FunctionType(method.__code__,dependencies,method.__name__,method.__defaults__,method.__closure__)
    return cloned(worker,*args)


class ShortAccount(timing_trial.macro_runner.MacroAccount):
    def __init__(self,*args,activated_at_ms,technology_arm,**kwargs):
        self.activated_at_ms = activated_at_ms
        self.variant = technology_arm.rsplit('_',1)[0]
        self.macro_context_sha256 = None
        self.shared_context = {}
        super().__init__(*args,**kwargs)

    def step(self,market_data=None):
        return isolated_call(paper.StockAccount.step,self,market_data)

    def fill(self,state,result,timestamp,direction,reason,trigger_ms=None):
        return isolated_call(paper.StockAccount.fill,self,state,result,timestamp,direction,reason,trigger_ms)


def tick_symbol(output,symbol,workers,venue,calendar_hash,context,responses):
    for worker in workers.values():
        worker.shared_context = context
    aliases = {('guarded_'+a.split('_',1)[1] if a.startswith('candidate_') else a):w for a,w in workers.items()}
    rows,comparison,clock = feed.tick_symbol(output,symbol,aliases,venue,calendar_hash,context,responses)
    renamed = {}
    for key,value in rows.items():
        key = 'candidate_'+key.split('_',1)[1] if key.startswith('guarded_') else key
        value.pop('technology_guard',None)
        value['short_confirmation'] = (value.get('signal_diagnostics') or {}).get('short_confirmation')
        renamed[key] = value
    return renamed,comparison,clock


class NoAdditionalMarketJournal:
    """The short hypothesis requires no NQ gate or additional market polling."""
    def __init__(self,output):
        self.rows = []
        self.input_error = None

    def refresh(self,output):
        return digest({'additional_market_gate':False})


def publish(output,manifest,accounts,comparisons,feeds,journal,*,running,cycles):
    now = int(time.time()*1000)
    flat = {s+'_'+a:v for s,arms in accounts.items() for a,v in arms.items()}
    comparisons = {s:{c:feed.metrics.compare_independent_arms(arms.get('baseline_'+c),arms.get('candidate_'+c),
        asof_ms=now,inception_ms=manifest['started_at_ms']) for c in COSTS} for s,arms in accounts.items()}
    healthy = len(flat)==12 and all(v.get('status')=='healthy' and v.get('financial_integrity') is True for v in flat.values())
    result = {'schema_version':1,'mode':'SIMULATION','places_orders':False,'experiment_id':manifest['experiment_id'],
        'pid':os.getpid(),'running':running,'status':'healthy' if healthy else 'degraded','cycles':cycles,
        'started_at_ms':manifest['started_at_ms'],'started_at_utc':manifest['started_at_utc'],
        'heartbeat_at_ms':now,'heartbeat_at_utc':paper.arithmetic.iso(now),'accounts':flat,'by_symbol':accounts,
        'feed_status':feeds,'comparisons':comparisons,'fingerprint':manifest['fingerprint'],
        'original_accounts_modified':False,'automatic_strategy_promotion':False,'forward_validated':False,
        'independent_initial_capital_usdt':12000,'original_account_capital_included':False,'paired_comparison':False,
        'review':feed.metrics.review_readiness(accounts,asof_ms=now,inception_ms=manifest['started_at_ms'])}
    paper.write_json(output/'status.json',result)
    return result


def main(argv=None):
    # Reuse the tested immutable-feed lifecycle without changing that module's
    # globals or invoking its long-entry policy/NQ journal in this process.
    return feed.clone(feed.main,DEFAULT_CONFIG=DEFAULT_CONFIG,load_design=load_design,prepare=prepare,
        ARMS=ARMS,TechnologyAccount=ShortAccount,QuoteJournal=NoAdditionalMarketJournal,
        tick_symbol=tick_symbol,publish=publish,__doc__=__doc__)(argv)


if __name__=='__main__':
    main()
