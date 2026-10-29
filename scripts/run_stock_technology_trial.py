"""Independent NQ new-long admission trial; reuse the existing public feed."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import signal
import threading
import time
from types import FunctionType

from process_lock import exclusive_process_lock
import run_stock_entry_timing_trial as timing_trial
import stock_technology_guard as guard

paper = timing_trial.paper
shared = timing_trial.shared
infrastructure = timing_trial.infrastructure
macro_runner = timing_trial.macro_runner
metrics = timing_trial.metrics
ROOT = paper.ROOT
DEFAULT_CONFIG = ROOT / 'config/stock_technology_trial_20261030.json'
SYMBOLS = timing_trial.SYMBOLS
COSTS = ('normal', 'double')
ARMS = tuple(v + '_' + c for c in COSTS for v in ('baseline', 'guarded'))
digest = infrastructure.digest
SOURCE = ROOT / 'data/paper_trading/stock_entry_timing_trial_20261029'


def clone(function, **overrides):
    return FunctionType(function.__code__, {**function.__globals__, **overrides},
                        function.__name__, function.__defaults__, function.__closure__)


def source_hashes():
    return clone(timing_trial.source_hashes, __file__=__file__)()


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = paper.read_json(path)
    fixed = {'schema_version': 1, 'mode': 'SIMULATION', 'places_orders': False,
        'symbols': list(SYMBOLS), 'variants': ['baseline', 'guarded'],
        'cost_multipliers': {'normal': 1, 'double': 2}, 'initial_balance_usdt_each': 1000,
        'automatic_strategy_promotion': False, 'historical_warmup_counts_as_forward_performance': False,
        'review_minimum_elapsed_days': 14, 'review_minimum_closed_positions_each': 10,
        'candidate_only_added_rule': 'technology_guard_policy', 'poll_seconds': 1,
        'market_feed': 'existing_entry_timing_trial_immutable_context',
        'source_trial': 'config/stock_entry_timing_trial_20261029.json'}
    if any(spec.get(k) != v or type(spec.get(k)) is not type(v) for k,v in fixed.items()):
        raise ValueError('Invalid fixed prospective technology trial design')
    if type(spec.get('effective_from_ms')) is not int or spec['effective_from_ms'] <= 0:
        raise ValueError('Explicit declaration time required')
    guard.validate({'technology_guard_policy': spec.get('technology_guard_policy')})
    output = infrastructure.contained(spec['output_dir'], 'data/paper_trading')
    if output == SOURCE:
        raise ValueError('Independent output required')
    original, old_rules, old_fp = timing_trial.load_design(ROOT/spec['source_trial'])
    timing_trial.check_source_rules(original, old_rules)
    source_manifest = paper.read_json(SOURCE/'manifest.json')
    if source_manifest['fingerprint'] != old_fp or source_manifest['rules'] != old_rules:
        raise ValueError('Existing feed trial rules or source identity changed')
    rules = {}
    for symbol in SYMBOLS:
        rules[symbol] = {}
        for arm in ARMS:
            variant, cost = arm.rsplit('_', 1)
            config = deepcopy(old_rules[symbol]['turn_pullback_'+cost])
            config['stock_event_risk_policy']['calendar_path'] = (output/'calendar_current.json').relative_to(ROOT).as_posix()
            config[timing_trial.macro.POLICY_KEY].update(
                snapshot_path=(output/'current_context/public_context.json').relative_to(ROOT).as_posix(),
                effective_from_ms=spec['effective_from_ms'])
            if variant == 'guarded':
                config['technology_guard_policy'] = deepcopy(guard.POLICY)
                guard.validate(config)
            rules[symbol][arm] = config
    fp = {'experiment_config_sha256': digest(path.read_bytes()),
        'profile_sha256': deepcopy(old_fp['profile_sha256']),
        'source_sha256': source_hashes(),
        'rules_sha256': {s: {a: digest(c) for a,c in arms.items()} for s,arms in rules.items()}}
    return spec, rules, fp


def prepare(output, spec, rules, fingerprint, inception_ms, require_existing=False):
    # The original strict financial/epoch checks apply to the new four arms.
    return clone(timing_trial.prepare, ARMS=ARMS)(
        output, spec, rules, fingerprint, inception_ms, require_existing)


def last_json_line(path):
    with Path(path).open('rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size-65536))
        raw = handle.read()
    lines = raw.split(b'\n')[:-1]  # Never read a writer's incomplete final row.
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict) and isinstance(row.get('shared_context_sha256'), str):
            return row
    raise ValueError('No complete shared observation available')


def read_block(directory, sha):
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Invalid immutable block identity')
    path = directory/(sha+'.json.gz')
    if path.stat().st_size > 8_000_000:
        raise ValueError('Oversized immutable block')
    with gzip.open(path, 'rb') as handle:
        raw = handle.read(8_000_001)
    if len(raw) > 8_000_000 or digest(raw) != sha:
        raise ValueError('Immutable shared context hash mismatch')
    return json.loads(raw)


def source_batch(symbol):
    """Consume current receipts only, preserving the original monotonic clock."""
    row = last_json_line(SOURCE/symbol/'baseline_normal/observations.jsonl')
    sha = row['shared_context_sha256']
    context = read_block(SOURCE/'context_archive'/symbol, sha)
    received = int(datetime.fromisoformat(context['received_at_utc']).timestamp()*1000)
    age = int(time.time()*1000)-received
    if context.get('schema_version') != 1 or context.get('symbol') != symbol or not 0 <= age <= 60_000:
        raise ValueError('Shared market receipt unavailable or older than sixty seconds')
    responses, labels = [], set()
    for item in context['responses']:
        label = item['label']
        if label not in {'clock','mark','book','funding','five','rules','signal'} or label in labels:
            raise ValueError('Invalid shared response labels')
        labels.add(label)
        payload = read_block(SOURCE/'context_blocks', item['payload_sha256']) if item['payload_sha256'] else None
        responses.append((label, payload, item['error']))
    if 'rules' not in labels and context.get('cached_contract_rules_sha256'):
        rules = read_block(SOURCE/'context_blocks', context['cached_contract_rules_sha256'])
        responses.append(('rules', {'symbols': [rules]}, None))
    if 'signal' not in labels and context.get('cached_signal_candles_sha256'):
        bars = read_block(SOURCE/'context_blocks', context['cached_signal_candles_sha256'])
        responses.append(('signal', bars, None))
    return sha, responses


class FundingOnlyVenue(shared.SharedPublicVenue):
    """No additional bulk polling; only an uncached settlement mark if needed."""
    def get(self, endpoint, params=None):
        if endpoint != 'markPriceKlines':
            raise RuntimeError('Technology trial bulk requests are disabled')
        return super().get(endpoint, params)


class QuoteJournal:
    """One immutable point per source observation; never renew duplicate age."""
    def __init__(self, output):
        self.path = output/'nq_observations.jsonl'
        self.rows = shared.read_rows(self.path) if self.path.exists() else []
        if self.rows:
            # Keep the complete on-disk journal. Runtime only needs the latest
            # conservatively separated provider date, including its conflicts.
            latest_date = max(row['observed_date'] for row in self.rows)
            self.rows = [row for row in self.rows if row['observed_date']==latest_date]
        self.keys = {}
        for row in self.rows:
            self.keys.setdefault(row.get('observed_at_ms'), row)
        self.last_snapshot = None
        self.input_error = 'nq_snapshot_not_read'

    @staticmethod
    def content(row):
        return {k:row.get(k) for k in ('value','provider_previous_close',
            'provider_reference_first_seen_at_ms','provider_reference_available_at_ms',
            'instrument','provider','source_id','source_config_id','observed_date')}

    def refresh(self, output):
        try:
            raw = (ROOT/'data/cross_market_observer/latest_observations.json').read_bytes()
            source = json.loads(raw)
            now = int(time.time()*1000)
            if not 0 <= now-source['generated_at_ms'] <= 20*60_000:
                raise ValueError('nq_observer_snapshot_stale')
            row = deepcopy(source['observations']['nq_dec_2026'])
            if row.get('eligible_intraday') is not True or row.get('market_session',{}).get('phase') != 'open':
                raise ValueError('nq_current_snapshot_not_eligible')
            sha = digest(raw)
            row.update(ingested_at_ms=now, source_snapshot_sha256=sha)
            # Validate transport/identity metadata without inventing old receipts.
            guard._point(row, asof_ms=now, policy=guard.validate({'technology_guard_policy':guard.POLICY}))
            if self.rows and row['observed_date'] > self.rows[-1]['observed_date']:
                self.rows = []
                self.keys = {}
            elif self.rows and row['observed_date'] < self.rows[-1]['observed_date']:
                raise ValueError('nq_observation_date_regressed')
            previous = self.keys.get(row['observed_at_ms'])
            if previous is not None and self.content(previous) != self.content(row):
                row.update(record_type='conflict', conflict_reason='same_observation_changed')
                key = digest(self.content(row))
                if not any(r.get('conflict_sha256') == key for r in self.rows):
                    row['conflict_sha256'] = key
                    paper.journal(self.path, row)
                    self.rows.append(row)
            elif previous is None:
                paper.journal(self.path, row)
                self.rows.append(row)
                self.keys[row['observed_at_ms']] = row
            # Raw source archive includes re-fetch metadata; it does not change
            # a previously journaled point or its first trial ingestion time.
            archive = output/'nq_source_archive'/(sha+'.json.gz')
            if not archive.exists():
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(gzip.compress(raw, mtime=0))
            self.input_error = None
            self.last_snapshot = sha
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            self.input_error = 'nq_snapshot_unavailable_or_invalid'
        payload = {'rows': self.rows, 'input_error': self.input_error}
        sha = digest(payload)
        target = output/'technology_context_archive'/(sha+'.json.gz')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(gzip.compress(infrastructure.canonical(payload),mtime=0))
        return sha


class TechnologyTiming:
    def __init__(self, worker):
        self.worker = worker

    def __getattr__(self, name):
        return getattr(timing_trial.timing, name)

    def evaluate(self, trigger, sig, config, **kwargs):
        result = timing_trial.timing.evaluate(trigger, sig, config, **kwargs)
        if 'technology_guard_policy' not in config:
            return result
        direction = sig.direction if hasattr(sig, 'direction') else sig['direction']
        decision = self.worker.technology_decision(direction)
        decision.update(original_timing_allowed=bool(result['allowed']),
                        blocked_by_added_guard=bool(result['allowed']) and not decision['allowed'],
                        executable_opportunity_proven=False)
        result['technology_guard'] = decision
        if not decision['allowed']:
            result['allowed'] = False
            result['reasons'] = list(dict.fromkeys(result['reasons'] + ['technology_'+r for r in decision['reasons']]))
        return result


def isolated_call(method, worker, *args):
    dependencies = {**method.__globals__, 'stock_swing_profiles': timing_trial.TimingProfiles(worker.activated_at_ms),
                    'stock_entry_timing': TechnologyTiming(worker)}
    original_journal = dependencies['journal']
    def journal(path, row):
        if Path(path).name in ('signals.jsonl','fills.jsonl','entry_timing_decisions.jsonl'):
            row.update(technology_arm=worker.technology_arm,
                       technology_context_sha256=worker.technology_context_sha256)
            if Path(path).name == 'signals.jsonl':
                seen = worker.now_ms()
                row.update(first_seen_at_ms=seen, first_seen_lag_ms=seen-row['boundary_ms'],
                           entry_timing_variant='turn_pullback', macro_context_sha256=worker.macro_context_sha256)
            if Path(path).name == 'fills.jsonl':
                state = args[0]
                row.update(signal_kind=((state.get('signal_diagnostics') or {}).get('entry_timing_candidate') or {}).get('signal_kind','donchian20'),
                           entry_timing_variant='turn_pullback', macro_context_sha256=worker.macro_context_sha256)
        original_journal(path, row)
    dependencies['journal'] = journal
    return FunctionType(method.__code__, dependencies, method.__name__, method.__defaults__, method.__closure__)(worker,*args)


class TechnologyAccount(timing_trial.TimingAccount):
    def __init__(self, *args, technology_arm, **kwargs):
        self.technology_arm = technology_arm
        self.technology_rows = []
        self.technology_input_error = 'nq_snapshot_not_read'
        self.technology_context_sha256 = None
        super().__init__(*args, **kwargs)

    def technology_decision(self, direction=1):
        decision = guard.evaluate(self.technology_rows, {'technology_guard_policy':guard.POLICY},
            asof_ms=self.now_ms(), inception_ms=self.activated_at_ms, direction=direction)
        if direction == 1 and self.technology_input_error:
            decision.update(allowed=False, status='unknown', weak=None, counts_as_weak_filter=False,
                            reasons=[self.technology_input_error])
        return decision

    def record_stock_event_risk(self, state, phase, reference_ms=None):
        result = super().record_stock_event_risk(state,phase,reference_ms)
        state['technology_guard'] = {**self.technology_decision(), 'phase':phase,
            'applied_to_entries': 'technology_guard_policy' in self.config,
            'context_sha256':self.technology_context_sha256}
        return result  # This observation never changes macro/earnings decisions.

    def step(self, market_data=None):
        self.technology_evaluations = []
        state = isolated_call(paper.StockAccount.step,self,market_data)
        for outcome in self.technology_evaluations:
            checks = (state.get('entry_checks') or {}).get('passed',{})
            other = {k:v for k,v in checks.items() if k!='entry_timing'}
            paper.journal(self.path.with_name('technology_candidates.jsonl'),{
                'symbol':self.symbol,'technology_arm':self.technology_arm,
                'signal_key':str(outcome['boundary_ms']),
                'repeated_pending_observations_are_not_independent_opportunities':True,
                'technology_context_sha256':self.technology_context_sha256,
                'other_pre_size_checks_pass':bool(other) and all(v is True for v in other.values()),
                'original_event_gate_allowed':(state.get('stock_event_risk') or {}).get('allowed'),
                'actual_fill_count_total':state['fill_count_total'],
                'would_have_filled_without_guard':None,**outcome})
        return state

    def record_entry_timing(self,state,pending,decision,phase):
        if decision.get('technology_guard') is not None:
            self.technology_evaluations.append({'checked_at_ms':self.now_ms(),
                'phase':phase,'boundary_ms':pending['boundary_ms'],
                'technology_guard':deepcopy(decision['technology_guard']),
                'combined_timing_allowed':decision['allowed']})
        return super().record_entry_timing(state,pending,decision,phase)

    def fill(self,state,result,timestamp,direction,reason,trigger_ms=None):
        return isolated_call(paper.StockAccount.fill,self,state,result,timestamp,direction,reason,trigger_ms)


def tick_symbol(output,symbol,workers,venue,calendar_hash,context,responses):
    responses, clock = shared.checked_snapshot(responses)
    context[symbol] = infrastructure.archive_context(output,symbol,responses,calendar_hash,
        workers['baseline_normal'].rules,workers['baseline_normal'].signal_bars)
    result = {}
    for arm,worker in workers.items():
        good = True
        try:
            if worker.path.with_name('quarantine.json').exists():
                raise infrastructure.LedgerIntegrityError('Arm quarantined')
            before = paper.read_json(worker.path)
            infrastructure.validate_ledger(before,worker.config,symbol,worker.activated_at_ms)
            infrastructure.validate_financial_journals(worker.path.parent,before)
            state = worker.step(deepcopy(responses))
            infrastructure.validate_ledger(state,worker.config,symbol,worker.activated_at_ms)
            infrastructure.validate_financial_journals(worker.path.parent,state)
        except Exception as exc:
            state = paper.read_json(worker.path)
            try:
                infrastructure.validate_ledger(state,worker.config,symbol,worker.activated_at_ms)
                infrastructure.validate_financial_journals(worker.path.parent,state)
            except (ValueError,TypeError,KeyError,OSError):
                good = False
            if isinstance(exc,infrastructure.LedgerIntegrityError) or not good:
                good = False
                paper.write_json(worker.path.with_name('quarantine.json'),{'reason':'Financial integrity failure; explicit repair required'})
            state = {**state,'status':'degraded','errors':{'worker':type(exc).__name__}}
        summary = {k:state.get(k) for k in ('equity','wallet_balance','return_pct','position_qty',
            'fill_count_total','fees_paid','funding_pnl','status','errors','signal_status','entry_blockers',
            'position','checked_at_utc','technology_guard','risk_monitor','signal_diagnostics')}
        summary.update(financial_integrity=good,start_ms=state['start_ms'],initial_balance=state['initial_balance'],
            heartbeat_at_ms=int(time.time()*1000),
            estimated_close=infrastructure.estimated_close(state,responses,worker.config,worker.now_ms(),worker.rules) if good else {'status':'unavailable','reason':'financial_integrity'},
            metrics=shared.closed_metrics(worker.path.parent,state) if good else None)
        result[arm] = summary
    comparisons = {c:metrics.compare_independent_arms(result['baseline_'+c],result['guarded_'+c],
        asof_ms=int(time.time()*1000),inception_ms=workers['baseline_normal'].activated_at_ms) for c in COSTS}
    return result,comparisons,clock


def publish(output,manifest,accounts,comparisons,feeds,journal,*,running,cycles):
    now = int(time.time()*1000)
    flat = {s+'_'+a:value for s,arms in accounts.items() for a,value in arms.items()}
    comparisons = {s:{c:metrics.compare_independent_arms(arms.get('baseline_'+c),arms.get('guarded_'+c),
        asof_ms=now,inception_ms=manifest['started_at_ms']) for c in COSTS} for s,arms in accounts.items()}
    healthy = len(flat)==12 and all(v.get('status')=='healthy' and v.get('financial_integrity') is True for v in flat.values())
    review_accounts = {s:{prefix+'_'+c:arms.get(('guarded' if prefix=='candidate' else 'baseline')+'_'+c)
        for c in COSTS for prefix in ('baseline','candidate')} for s,arms in accounts.items()}
    status = {'schema_version':1,'mode':'SIMULATION','places_orders':False,'experiment_id':manifest['experiment_id'],
        'pid':os.getpid(),'running':running,'status':'healthy' if healthy else 'degraded','cycles':cycles,
        'started_at_ms':manifest['started_at_ms'],'started_at_utc':manifest['started_at_utc'],
        'heartbeat_at_ms':now,'heartbeat_at_utc':paper.arithmetic.iso(now),'by_symbol':accounts,'accounts':flat,
        'comparisons':comparisons,'feed_status':feeds,'fingerprint':manifest['fingerprint'],
        'technology_context_sha256':digest({'rows':journal.rows,'input_error':journal.input_error}),
        'nq_input_error':journal.input_error,'nq_observation_count':len(journal.rows),
        'original_accounts_modified':False,'automatic_strategy_promotion':False,'forward_validated':False,
        'independent_initial_capital_usdt':12000,'original_account_capital_included':False,
        'paired_comparison':False,'review':metrics.review_readiness(review_accounts,asof_ms=now,inception_ms=manifest['started_at_ms'])}
    paper.write_json(output/'status.json',status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--require-existing',action='store_true')
    args = parser.parse_args(argv)
    spec,rules,fp = load_design(args.config)
    if args.check:
        print(json.dumps({'mode':'SIMULATION','places_orders':False,'ledger_count':12,
            'network_requests_performed':False,'source_count':len(fp['source_sha256']), 'fingerprint':fp}))
        return
    output = infrastructure.contained(spec['output_dir'],'data/paper_trading')
    output.mkdir(parents=True,exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    with exclusive_process_lock(output/'runner.lock'):
        manifest = prepare(output,spec,rules,fp,None,True) if args.require_existing or (output/'manifest.json').exists() else None
        deadline = time.monotonic()+60
        while manifest is None:
            for symbol in SYMBOLS:
                try:
                    _,responses = source_batch(symbol)
                    _,clock = shared.checked_snapshot(responses)
                    if clock['source']!='unavailable':
                        manifest = prepare(output,spec,rules,fp,paper.now_ms())
                        break
                except (OSError,ValueError,KeyError,TypeError):
                    pass
            if manifest is None:
                if time.monotonic()>=deadline:
                    raise ValueError('No fresh verified exchange receipt for new inception')
                stop.wait(1)
        venue = FundingOnlyVenue(paper.PublicAster())
        workers = {s:{a:TechnologyAccount(output/s/a/'state.json',venue,s,rules[s][a],
            activated_at_ms=manifest['started_at_ms'],technology_arm=a) for a in ARMS} for s in SYMBOLS}
        journal = QuoteJournal(output)
        accounts,comparisons,feeds,context,last = {},{},{},{},{}
        cycles,last_publish,last_quote = 0,0,0
        try:
            with infrastructure.compact_observations(output,context):
                while not stop.is_set():
                    now = time.monotonic()
                    if now-last_quote>=5:
                        nq_hash = journal.refresh(output)
                        last_quote = now
                    calendar = infrastructure.freeze_calendar(output,spec)
                    macro_hash = macro_runner.freeze_context(output)
                    updated = False
                    for symbol in SYMBOLS:
                        try:
                            sha,responses = source_batch(symbol)
                            if last.get(symbol)==sha:
                                if int(time.time()*1000)-feeds[symbol]['at_ms']>75_000:
                                    raise ValueError('Shared feed heartbeat stale')
                                continue
                            venue.begin_tick()
                            for worker in workers[symbol].values():
                                worker.macro_context_sha256 = macro_hash
                                worker.technology_rows = deepcopy(journal.rows)
                                worker.technology_input_error = journal.input_error
                                worker.technology_context_sha256 = nq_hash
                            accounts[symbol],comparisons[symbol],clock = tick_symbol(
                                output,symbol,workers[symbol],venue,calendar,context,responses)
                            feeds[symbol] = {'status':'received','source_context_sha256':sha,'clock':clock,'at_ms':int(time.time()*1000)}
                            last[symbol] = sha
                            updated = True
                        except (OSError,ValueError,KeyError,TypeError) as exc:
                            feeds[symbol] = {'status':'unavailable','reason':type(exc).__name__,'at_ms':int(time.time()*1000)}
                            for value in accounts.get(symbol,{}).values():
                                value.update(status='degraded',feed_unavailable=True)
                    if updated or now-last_publish>=30 or args.once:
                        cycles += 1
                        status = publish(output,manifest,accounts,comparisons,feeds,journal,running=not args.once,cycles=cycles)
                        print(json.dumps({k:status[k] for k in ('experiment_id','status','heartbeat_at_utc','cycles')}),flush=True)
                        last_publish = now
                    if args.once:
                        break
                    stop.wait(1)
        finally:
            publish(output,manifest,accounts,comparisons,feeds,journal,running=False,cycles=cycles)


if __name__=='__main__':
    main()
