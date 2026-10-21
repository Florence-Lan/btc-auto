"""Prospective isolated BTC original-entry versus same-trend-reentry paper arms."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from binance_terminal_client import BinanceTerminalClient
import btc_same_direction_reentry as reentry
import btc_pullback_entry
import decision_runtime
import llm_trade_gate
from process_lock import exclusive_process_lock
import trading_execution as execution

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'config/btc_reentry_shadow_20261020.json'
ARMS = ('baseline_normal','candidate_normal','baseline_double','candidate_double')
HOUR = 3600000
FIVE_MINUTES = 300000
ENVIRONMENT_KEYS = ('SIM_TAKER_FEE','SIM_SLIPPAGE_BPS','SIM_MAX_LEVERAGE',
    'SIM_MAX_NOTIONAL_USDT','MAX_SIGNAL_AGE_SECONDS','LLM_TRADE_GATE_ENABLED',
    'LLM_PROVIDER','LLM_MODEL','LLM_CODEX_MODEL','LLM_MIN_CONFIDENCE',
    'LLM_REJECTION_CACHE_SECONDS','LLM_TIMEOUT_SECONDS','LLM_CODEX_TIMEOUT_SECONDS',
    'LLM_CODEX_REASONING_EFFORT','CODEX_BIN')


def iso(ms):
    return datetime.fromtimestamp(ms/1000,timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    execution.write_json(Path(path),value)


def journal(path,value):
    with Path(path).open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def source_hashes():
    """AST dependency closure includes local imports inside functions as well."""
    pending, result = [Path(__file__).resolve(),ROOT/'scripts/run_parallel_btc_signals.py',
                       ROOT/'scripts/paper_trade_frozen_portfolio.py'],{}
    while pending:
        path = pending.pop()
        key = path.relative_to(ROOT).as_posix()
        if key in result:
            continue
        raw = path.read_bytes()
        result[key] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw,filename=str(path))):
            names = ([alias.name for alias in node.names] if isinstance(node,ast.Import)
                else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                local = ROOT/'scripts'/(name.split('.')[0]+'.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(result.items()))


def load_design(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = read(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'SIMULATION'
            or spec.get('places_orders') is not False or spec.get('symbol') != 'BTCUSDT'
            or spec.get('venue') != 'binance_usdm' or spec.get('initial_balance_usdt_each') != 1000
            or spec.get('cost_multipliers') != {'normal':1,'double':2}
            or spec.get('historical_warmup_counts_as_forward_performance') is not False
            or spec.get('automatic_strategy_promotion') is not False):
        raise ValueError('Invalid fixed BTC paper experiment design')
    reentry.validate_policy(spec['candidate_policy'])
    profile_path = (ROOT/spec['baseline_profile']).resolve()
    profile = read(profile_path)
    base_path = (ROOT/profile['base_manifest']).resolve()
    base = read(base_path)['config']
    if (profile.get('live_orders_allowed') is not False
            or profile.get('strategy_modes') != ['timeseries_trend']
            or profile.get('hourly_startup_enabled') is not False
            or profile.get('risk_limits',{}).get('portfolio_leverage_cap') != 10
            or profile.get('entry_timing_policy') != {
                'mode':'origin_signal_window_v1','max_initial_entry_age_seconds':3600}
            or base.get('timeseries_fast_ema') != 48 or base.get('timeseries_slow_ema') != 240
            or base.get('timeseries_min_ema_spread_pct') != .004
            or base.get('timeseries_target_vol') != .11):
        raise ValueError('BTC baseline must retain original hourly trend and risk policy')
    btc_pullback_entry.validate_policy(profile['btc_pullback_entry_policy'])
    fingerprint = {'experiment_config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'profile_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (profile_path,base_path)}, 'source_sha256':source_hashes(),
        'execution_environment':{key:os.getenv(key) for key in ENVIRONMENT_KEYS}}
    if spec.get('require_review_compatibility') is True:
        if not llm_trade_gate.env_enabled('LLM_TRADE_GATE_ENABLED'):
            raise ValueError('The corrected experiment requires the LLM review gate')
        fingerprint['review_provider_configuration'] = llm_trade_gate.review_provider_configuration()
        executable = Path(fingerprint['review_provider_configuration']['executable'])
        fingerprint['review_executable_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
    # Validate costs before any ledger initialization. Values are not credentials.
    fee = float(os.getenv('SIM_TAKER_FEE','0.00045') or .00045)
    slip = float(os.getenv('SIM_SLIPPAGE_BPS','1.0') or 1)
    if not math.isfinite(fee) or not 0 <= fee < .5 or not math.isfinite(slip) or not 0 <= slip < 5000:
        raise ValueError('Invalid frozen fee/slippage settings')
    return spec,profile,fingerprint


def prepare(root,spec,profile,fingerprint,inception_ms,require_existing=False):
    root = Path(root)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if (manifest.get('fingerprint') != fingerprint or manifest.get('design') != spec
                or manifest.get('profile') != profile or manifest.get('mode') != 'SIMULATION'
                or manifest.get('places_orders') is not False):
            raise ValueError('Frozen BTC experiment changed; incompatible restart refused')
        for arm in ARMS:
            path = root/arm/'state.json'
            state = read(path)
            if (state.get('mode') != 'simulation' or state.get('initial_balance') != 1000
                    or state.get('created_at_utc') != manifest['started_at_utc']
                    or state.get('funding_tracking_start_ms') != manifest['started_at_ms']
                    or state.get('experiment_arm') != arm
                    or state.get('experiment_id') != spec['experiment_id']):
                raise ValueError('Existing BTC ledger identity incompatible; no reset allowed')
            for name in ('performance.json','event_state.json'):
                if read(path.with_name(name)).get('started_at_ms') != manifest['started_at_ms']:
                    raise ValueError('Existing BTC experiment history incomplete')
        return manifest
    if require_existing:
        raise ValueError('Existing BTC experiment required; refusing fresh initialization')
    if any(p.name != 'runner.lock' for p in root.iterdir()):
        raise ValueError('Incomplete BTC experiment directory; refusing overwrite')
    if type(inception_ms) is not int or inception_ms <= 0:
        raise ValueError('Verified exchange inception is required')
    manifest = {'schema_version':1,'mode':'SIMULATION','places_orders':False,
        'experiment_id':spec['experiment_id'],'started_at_ms':inception_ms,
        'started_at_utc':iso(inception_ms),'initial_balance_each':1000,'arms':list(ARMS),
        'design':spec,'profile':profile,'fingerprint':fingerprint,
        'original_accounts_modified':False,'forward_validated':False,
        'automatic_strategy_promotion':False,
        'comparison_basis':'Shared public observation and frozen current first-seen inputs; independently sized ledgers.',
        'risk_model':'Original confirmed reversal; 8% soft and 15% hard account drawdown. No new per-trade ATR stop.'}
    # Manifest written last: a partial initialization cannot silently restart/reset.
    for arm in ARMS:
        path = root/arm/'state.json'
        factor = spec['cost_multipliers'][arm.rsplit('_',1)[1]]
        worker = execution.SimulationAccount(path,cost_multiplier=factor)
        state = worker.reset(1000,now_ms=inception_ms)
        state.update(experiment_arm=arm,experiment_id=spec['experiment_id'],places_orders=False)
        worker._save(state)
        write(path.with_name('event_state.json'),reentry.initial_state(inception_ms))
        write(path.with_name('performance.json'),{'started_at_ms':inception_ms,
            'peak_equity':1000.,'max_sampled_drawdown_pct':0.,'equity_samples':0})
    write(manifest_path,manifest)
    return manifest


class DeferredReview:
    """No long provider call runs on the risk-monitor/ledger thread."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2,thread_name_prefix='btc-shadow-review')
        self.futures = {}
        self.results = {}

    def begin_tick(self):
        # One immutable availability boundary for all four arms. A provider
        # finishing between arms cannot favor whichever arm executes later.
        for key,future in list(self.futures.items()):
            if future.done():
                _,self.results[key] = future.result()
                del self.futures[key]

    def gate(self,report,target,**kwargs):
        key = (str(target.get('position_id')),'long' if target['target_leverage']>0 else 'short')
        prior = kwargs.get('previous_decision') or {}
        if 'btc_shadow_review_pending' in str(prior.get('reason','')):
            prior = {}
        completed = self.results.get(key)
        if completed and str(completed.get('reviewed_at_utc',''))>=str(prior.get('reviewed_at_utc','')):
            prior = completed
        kwargs['previous_decision'] = prior or None

        def provider(context):
            future = self.futures.get(key)
            if future is None:
                # Bound queued work; an outage cannot create an unbounded queue.
                active = sum(not f.done() for f in self.futures.values())
                if active >= 4:
                    raise RuntimeError('btc_shadow_review_pending_capacity')
                future = self.executor.submit(llm_trade_gate.apply_llm_trade_gate,
                    deepcopy(report),deepcopy(target),**deepcopy(kwargs),
                    decision_provider=llm_trade_gate.request_llm_decision)
                self.futures[key] = future
            # A current original gate decides whether the completed review's
            # cache is still usable. Expired rejection/increased exposure calls
            # this provider again and schedules a new current-context review.
            raise RuntimeError('btc_shadow_review_pending')
        return llm_trade_gate.apply_llm_trade_gate(report,target,decision_provider=provider,**kwargs)

    def close(self):
        self.executor.shutdown(wait=False,cancel_futures=True)


@contextmanager
def deferred_review(review):
    original = execution.apply_llm_trade_gate
    execution.apply_llm_trade_gate = review.gate
    try:
        yield
    finally:
        execution.apply_llm_trade_gate = original


class MarketCollector:
    def __init__(self,client):
        self.client = client
        self.cached = {}

    def risk_observation(self):
        clock = decision_runtime.resolve_clock(self.client)
        observation = clock.get('mark_observation') or self.client.mark_price_observation('BTCUSDT')
        reference = clock.get('freshness_reference_ms',clock['time_ms'])
        decision_runtime.validate_mark(observation,reference)
        return {'clock':clock,'mark':observation,'observed_monotonic':time.monotonic()}

    def candles(self,interval,now_ms):
        step = HOUR if interval=='1h' else FIVE_MINUTES
        bucket = now_ms//step
        if (interval,bucket) not in self.cached:
            rows = self.client.public_get('/fapi/v1/klines',{
                'symbol':'BTCUSDT','interval':interval,'limit':500 if interval=='1h' else 50})
            compact = [[int(row[0]),*map(float,row[1:6]),float(row[7]),int(row[6])]
                for row in rows if int(row[6]) < now_ms]
            if not compact or compact[-1][7] != bucket*step-1:
                raise ValueError('Missing latest completed '+interval+' candle')
            if any(b[0]-a[0] != step for a,b in zip(compact,compact[1:])):
                raise ValueError('Nonconsecutive '+interval+' candles')
            self.cached = {key:value for key,value in self.cached.items() if key[0]!=interval}
            self.cached[(interval,bucket)] = compact
        return deepcopy(self.cached[(interval,bucket)])

    def complete(self,risk,states):
        if risk['clock']['source'] != 'exchange':
            raise ValueError('Direct exchange clock required for new exposure')
        now = risk['clock']['time_ms']
        hourly,five = self.candles('1h',now),self.candles('5m',now)
        rules = self.client.symbol_rules('BTCUSDT')
        starts = []
        for state in states.values():
            tracking = int(state['funding_tracking_start_ms'])
            start = max(tracking,int(state.get('funding_last_fetch_ms') or tracking)-86400000)
            if state.get('funding_pending_settlement_ms') is not None:
                start = max(tracking,min(start,int(state['funding_pending_settlement_ms'])))
            starts.append(start)
        start = min(starts)
        try:
            events = self.client.funding_history(start,now,'BTCUSDT')
            for row in events:
                if (not start <= int(row['fundingTime']) <= now
                        or not math.isfinite(float(row['fundingRate']))
                        or not math.isfinite(float(row['markPrice'])) or float(row['markPrice'])<=0):
                    raise ValueError('Invalid shared funding event')
            funding_available = True
        except (RuntimeError,ValueError,KeyError,TypeError,OSError):
            events,funding_available = [],False
        # Metadata/series requests precede the single final mark used by all arms.
        mark = self.client.mark_price_observation('BTCUSDT')
        clock = decision_runtime.resolve_clock(self.client,mode='live')
        decision_runtime.validate_mark(mark,clock['time_ms'])
        next_funding = execution._next_funding_boundary(risk['mark'])
        return {'clock':clock,'mark':mark,'rules':rules,'hourly':hourly,'five_minute':five,
            'funding_events':events,'funding_available':funding_available,
            'funding_fetch_through_ms':now if funding_available else None,
            'next_funding_time_ms':next_funding,'observed_monotonic':time.monotonic()}


def freeze_context(report,root):
    """Both cost scenarios see identical input bytes during this tick."""
    result = deepcopy(report)
    context = result.get('execution_entry_context')
    if not isinstance(context,dict) or not context.get('factor_profile') or not context.get('factor_snapshot'):
        raise ValueError('Missing original factor execution context')
    for key in ('factor_profile','factor_snapshot','event_snapshot','macro_snapshot'):
        if not context.get(key):
            continue
        source = Path(context[key])
        target = root/'current_context'/(key+''.join(source.suffixes))
        target.parent.mkdir(parents=True,exist_ok=True)
        payload = source.read_bytes()
        temporary = target.with_name(target.name+'.tmp')
        temporary.write_bytes(payload)
        temporary.replace(target)
        context[key] = str(target)
    return result


def validate_source_report(report,manifest):
    """An evolving producer report cannot silently change the frozen baseline."""
    expected_config = {'strategy_modes':['timeseries_trend'],'timeseries_fast_ema':48,
        'timeseries_slow_ema':240,'timeseries_min_ema_spread_pct':.004,
        'timeseries_target_vol':.11,'portfolio_leverage_cap':10}
    if any(report.get('config',{}).get(key)!=value for key,value in expected_config.items()):
        raise ValueError('Source BTC trend configuration differs from frozen experiment')
    expected_profile = hashlib.sha256(json.dumps(manifest['profile'],sort_keys=True).encode()).hexdigest()
    if report.get('execution_entry_context',{}).get('factor_profile_sha256')!=expected_profile:
        raise ValueError('Source BTC factor profile differs from frozen experiment')
    mapping = {'factor_engine_sha256':'multifactor.py','portfolio_risk_sha256':'portfolio_risk.py',
        'public_context_sha256':'public_context.py','event_risk_sha256':'event_risk.py',
        'hourly_execution_sha256':'timeseries_execution.py','execution_portfolio_sha256':'execution_portfolio.py'}
    shadow = report.get('shadow_profile') or {}
    for field,name in mapping.items():
        if shadow.get(field)!=manifest['fingerprint']['source_sha256'].get('scripts/'+name):
            raise ValueError('Source BTC producer dependency differs from frozen experiment: '+name)


def isolated_report(report,point,profile,start_ms,*,candidate=False):
    result = deepcopy(report)
    point = dict(point)
    # Candidate event origins are independently post-inception. Old source
    # inventory is never inherited by either initially flat account.
    if point.get('signed_qty') and int(point.get('origin_signal_time_ms') or 0)<start_ms:
        point.update(signed_qty=0,position_id='flat')
    result['execution_target'] = point
    result.setdefault('shadow_profile',{}).update({key:deepcopy(profile[key]) for key in (
        'entry_timing_policy','btc_pullback_entry_policy')})
    result['strategy_qualification'] = {'approved_for_forward_simulation':True,
        'authorization_basis':'user_authorized_isolated_paper_comparison',
        'historical_performance_required':False,'forward_validated':False}
    result['research_only'] = True
    result['btc_reentry_shadow'] = {'candidate':candidate,'started_at_ms':start_ms}
    return result


def closed_metrics(worker,root,mark):
    state = worker.load()
    path = worker.path.with_suffix('.fills.jsonl')
    fills = []
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    fills.append(json.loads(line))
    if [f.get('sequence') for f in fills] != list(range(1,int(state['fill_count_total'])+1)):
        raise ValueError('Incomplete fill archive; closed performance unavailable')
    groups,active,qty = [],None,0.
    for fill in fills:
        stamp = round(datetime.fromisoformat(fill['time_utc']).timestamp()*1000)
        delta = float(fill['quantity'])*(1 if fill['side']=='BUY' else -1)
        fee = float(fill['fee'])
        if not qty or qty*delta>0:
            if active is None:
                active = {'entry_ms':stamp,'closed_ms':None,'gross_pnl':0.,'fees':0.,'funding_pnl':0.}
                groups.append(active)
            active['fees'] += fee
        else:
            closing = min(abs(qty),abs(delta))
            exit_fee = fee*closing/abs(delta)
            active['fees'] += exit_fee
            active['gross_pnl'] += float(fill['realized_pnl'])
            new_qty = qty+delta
            if abs(new_qty)<1e-12 or new_qty*qty<0:
                active['closed_ms'] = stamp
                active = None
            if new_qty*qty<0:
                active = {'entry_ms':stamp,'closed_ms':None,'gross_pnl':0.,'fees':fee-exit_fee,'funding_pnl':0.}
                groups.append(active)
        qty += delta
        if abs(qty)<1e-12:
            qty = 0.
    for settlement in state.get('funding_settlements',[]):
        if not float(settlement['signed_qty']):
            continue
        matches = [g for g in groups if g['entry_ms']<int(settlement['time_ms'])
            and (g['closed_ms'] is None or int(settlement['time_ms'])<=g['closed_ms'])]
        if len(matches)!=1:
            raise ValueError('Funding cannot be uniquely allocated to an episode')
        matches[0]['funding_pnl'] += float(settlement['payment'])
    closed = [g for g in groups if g['closed_ms'] is not None]
    for group in groups:
        group['net_pnl'] = group['gross_pnl']-group['fees']+group['funding_pnl']
    net = sum(g['net_pnl'] for g in closed)
    wins = [g['net_pnl'] for g in closed if g['net_pnl']>0]
    performance_path = Path(root)/'performance.json'
    performance = read(performance_path)
    equity = worker.snapshot(mark)['account']['margin_balance']
    peak = max(performance['peak_equity'],equity,float(state.get('peak_equity') or equity))
    dd = max(0.,1-equity/peak)*100 if peak else 100.
    performance.update(peak_equity=peak,max_sampled_drawdown_pct=max(dd,
        performance['max_sampled_drawdown_pct'],state['max_drawdown_pct']),
        equity_samples=performance['equity_samples']+1)
    write(performance_path,performance)
    return {'closed_positions':len(closed),'net_wins':len(wins),
        'net_losses':sum(g['net_pnl']<0 for g in closed),'closed_net_pnl_usdt':net,
        'closed_net_excluding_best_winner_usdt':net-max(wins) if wins else net,
        'max_sampled_drawdown_pct':performance['max_sampled_drawdown_pct'],
        'grouped_positions':groups,'open_or_partial_positions_excluded':sum(g['closed_ms'] is None for g in groups)}


def source_reduction(worker,report,profile,start_ms,risk):
    """Fresh original flat/reversal exits need no entry history or new rules.

    A full close needs no unknown grid sizing. Previously observed rules are
    retained for audit; original SimulationAccount permits a complete reduction
    even below minimum quantity/notional. This path can never reverse/open.
    """
    now = risk['clock']['time_ms']+max(0,int((time.monotonic()-risk['observed_monotonic'])*1000))
    observation = risk['mark']
    decision_runtime.validate_mark(observation,now)
    original = execution.target_from_report(report,now_ms=now,
        max_age_seconds=float(os.getenv('MAX_SIGNAL_AGE_SECONDS','900') or 900),leverage_cap=10)
    state = worker.load()
    held = float(state['position_qty'])
    requested = float(original['target_leverage'])
    if not held or held*requested>0:
        return None
    point = report.get('execution_target') or report.get('summary',{}).get('last_equity_point') or {}
    flat = {**point,'signed_qty':0.,'position_id':'flat'}
    isolated = isolated_report(report,flat,profile,start_ms)
    target = execution.target_from_report(isolated,now_ms=now,
        max_age_seconds=float(os.getenv('MAX_SIGNAL_AGE_SECONDS','900') or 900),leverage_cap=10)
    rules = state.get('last_execution_rules') or execution.DEFAULT_SIMULATION_RULES
    result = worker.reconcile(target,observation['price'],rules,isolated,
        funding_events=(),funding_available=False,now_ms=now,mark_time_ms=observation['time_ms'],
        execution_clock=lambda:risk['clock']['time_ms']+max(0,int(
            (time.monotonic()-risk['observed_monotonic'])*1000)))
    if worker.load()['position_qty']:
        raise ValueError('Reduce-only original trend exit did not flatten inventory')
    journal(worker.path.with_name('source_exits.jsonl'),{'time_ms':now,
        'reason':'original_source_flat' if not requested else 'original_source_reversal_reduce_only',
        'source_target':original,'result':result,
        'entry_history_or_new_rules_required':False,'new_exposure_allowed':False})
    return result


def tick(root,manifest,workers,collector):
    accounts = {arm:{'status':'healthy','errors':{}} for arm in ARMS}
    risk = None
    try:
        risk = collector.risk_observation()
        risk['observed_monotonic'] = time.monotonic()
        mark,now = risk['mark'],risk['clock']['time_ms']
        # Hard stops precede slow funding, history and report/context reads.
        for arm,worker in workers.items():
            try:
                state = worker.load()
                previous = state.get('funding_next_settlement_ms')
                pending = previous if previous is not None and now>=int(previous) else None
                worker.observe(mark['price'],now,(),False,mark_time_ms=mark['time_ms'],
                    funding_pending_settlement_ms=pending,
                    next_funding_time_ms=execution._next_funding_boundary(mark))
            except Exception as exc:
                accounts[arm].update(status='degraded',errors={'risk':decision_runtime._error(exc)})
        report = read(ROOT/manifest['design']['source_report'])
        validate_source_report(report,manifest)
        # Confirmed trend exits are independent of the slower input collection
        # below. A 5m/1h/metadata/factor outage can only prevent new exposure.
        for arm,worker in workers.items():
            try:
                reduced = source_reduction(worker,report,manifest['profile'],manifest['started_at_ms'],risk)
                if reduced is not None:
                    accounts[arm]['source_exit'] = reduced
            except Exception as exc:
                accounts[arm]['status'] = 'degraded'
                accounts[arm]['errors']['source_exit'] = decision_runtime._error(exc)
        report = freeze_context(report,root)
        # Validate the unmodified source's freshness before any synthetic origin.
        execution.target_from_report(report,now_ms=risk['clock']['time_ms'],
            max_age_seconds=float(os.getenv('MAX_SIGNAL_AGE_SECONDS','900') or 900),leverage_cap=10)
        common = collector.complete(risk,{arm:worker.load() for arm,worker in workers.items()})
    except Exception as exc:
        for arm in ARMS:
            accounts[arm]['status'] = 'degraded'
            accounts[arm]['errors']['common_observation'] = decision_runtime._error(exc)
        common,report = None,None
    if common is not None:
        now,mark = common['clock']['time_ms'],common['mark']
        base_point = report.get('execution_target') or report.get('summary',{}).get('last_equity_point') or {}
        journal(root/'observations.jsonl',{'time_ms':now,'mark':mark,
            'source_point':base_point,'source_generated_at_utc':report.get('generated_at_utc'),
            'funding_available':common['funding_available'],
            'last_hour_close_ms':common['hourly'][-1][7],
            'shared_snapshot_sha256':hashlib.sha256(json.dumps({
                'mark':mark,'hourly':common['hourly'],'five':common['five_minute'],
                'funding':common['funding_events']},sort_keys=True).encode()).hexdigest()})
        for arm,worker in workers.items():
            try:
                # Every ledger settles the same actual funding publication set.
                worker.observe(mark['price'],now,common['funding_events'],common['funding_available'],
                    rules=common['rules'],funding_fetch_through_ms=common['funding_fetch_through_ms'],
                    next_funding_time_ms=common['next_funding_time_ms'],mark_time_ms=mark['time_ms'])
                state = worker.load()
                state['last_execution_rules'] = {key:str(value) for key,value in common['rules'].items()}
                worker._save(state)
                point = dict(base_point)
                if arm.startswith('candidate_'):
                    event_path = worker.path.with_name('event_state.json')
                    point,event_state,event_diagnostic = reentry.candidate_point(point,read(event_path),
                        worker.load(),common['hourly'],report.get('strategy_review_context',{}).get('timeseries_trend'),
                        now,manifest['profile'],manifest['design']['candidate_policy'])
                    write(event_path,event_state)  # Identity persisted before any fill.
                    accounts[arm]['reentry'] = event_diagnostic
                    opportunity = event_state.get('opportunity') or {}
                    state = worker.load()
                    if (opportunity.get('opportunity_id') == point.get('position_id')
                            and opportunity.get('origin_signal_time_ms',0)>manifest['started_at_ms']
                            and not state['position_qty'] and not state['fill_count_total']
                            and state.get('observed_flat_target') is False):
                        # These independently initialized accounts are verifiably
                        # flat. A newly created post-inception event must not be
                        # mistaken for inherited pre-start inventory by the
                        # original first-observation startup guard.
                        execution.apply_startup_entry_guard(state,{
                            'target_leverage':0.,'position_id':'flat'},0.)
                        state['candidate_event_startup_admission'] = {
                            'opportunity_id':opportunity['opportunity_id'],
                            'origin_signal_time_ms':opportunity['origin_signal_time_ms'],
                            'checked_at_ms':now,'basis':'verified_initial_flat_and_new_hourly_event'}
                        worker._save(state)
                isolated = isolated_report(report,point,manifest['profile'],manifest['started_at_ms'],
                    candidate=arm.startswith('candidate_'))
                current = isolated['execution_target']
                history_hours = max(manifest['profile']['btc_pullback_entry_policy']['lookback_hours'],
                    manifest['profile']['btc_pullback_entry_policy']['atr_period']+1,49)
                origin = current.get('origin_signal_time_ms') or 0
                # Match the exact original btc_pullback_market_data prefix; the
                # structural EMA is deliberately seeded from these 49 hours.
                original_prefix = [row for row in common['hourly'] if row[7]<origin][-history_hours:]
                isolated['btc_pullback_market_data'] = {'hourly':original_prefix,
                    'five_minute':common['five_minute'],'asof_ms':now,'complete':True}
                state = worker.load()
                pending = state.get('btc_pullback_pending') or {}
                retry = not state['position_qty'] and pending.get('status') in (
                    'waiting_retrace','waiting_confirmation','ready','data_unavailable')
                if state.get('last_signal_time_ms') != current.get('time_ms') or retry:
                    target = execution.target_from_report(isolated,now_ms=now,
                        max_age_seconds=float(os.getenv('MAX_SIGNAL_AGE_SECONDS','900') or 900),leverage_cap=10)
                    result = worker.reconcile(target,mark['price'],common['rules'],isolated,
                        funding_events=common['funding_events'],funding_available=common['funding_available'],
                        now_ms=now,mark_time_ms=mark['time_ms'],
                        funding_fetch_through_ms=common['funding_fetch_through_ms'],
                        next_funding_time_ms=common['next_funding_time_ms'],
                        execution_clock=lambda:now+max(0,int((time.monotonic()-common['observed_monotonic'])*1000)))
                    write(worker.path.with_name('last_report.json'),isolated)
                    journal(worker.path.with_name('execution.jsonl'),{'time_ms':now,'result':result})
                if not common['funding_available']:
                    accounts[arm]['status'] = 'degraded'
                    accounts[arm]['errors']['funding'] = 'Shared funding unavailable; new risk blocked'
            except Exception as exc:
                accounts[arm]['status'] = 'degraded'
                accounts[arm]['errors']['worker'] = decision_runtime._error(exc)
    for arm,worker in workers.items():
        state = worker.load()
        snapshot = worker.snapshot(state.get('last_mark_price'))['account']
        accounts[arm].update({key:state.get(key) for key in ('wallet_balance','position_qty',
            'entry_price','fill_count_total','fees_paid','funding_pnl','max_drawdown_pct',
            'last_mark_price','btc_entry_timing','btc_pullback_entry','llm_trade_gate','account_risk')})
        accounts[arm].update(equity=snapshot['margin_balance'],return_pct=snapshot['realized_return_pct'])
        try:
            accounts[arm]['metrics'] = closed_metrics(worker,worker.path.parent,state.get('last_mark_price'))
        except Exception as exc:
            accounts[arm]['status'] = 'degraded'
            accounts[arm]['errors']['metrics'] = decision_runtime._error(exc)
        stamp = int(time.time()*1000)
        accounts[arm].update(heartbeat_at_ms=stamp,heartbeat_at_utc=iso(stamp))
        write(worker.path.with_name('status.json'),accounts[arm])
    clock = (common or risk or {}).get('clock') or {'source':'unavailable'}
    return accounts,clock


def publish_status(root,manifest,accounts,clock,poll_seconds,running):
    now = int(time.time()*1000)
    comparisons = {}
    for cost in ('normal','double'):
        base,candidate = (accounts.get(prefix+'_'+cost,{}) for prefix in ('baseline','candidate'))
        if base.get('metrics') and candidate.get('metrics'):
            comparisons[cost] = {
                'candidate_minus_baseline_equity_usdt':candidate['equity']-base['equity'],
                'candidate_minus_baseline_closed_net_usdt':candidate['metrics']['closed_net_pnl_usdt']-base['metrics']['closed_net_pnl_usdt'],
                'candidate_minus_baseline_drawdown_percentage_points':candidate['metrics']['max_sampled_drawdown_pct']-base['metrics']['max_sampled_drawdown_pct']}
    status = {'mode':'SIMULATION','places_orders':False,'pid':os.getpid(),'running':running,
        'status':'healthy' if all(a.get('status')=='healthy' for a in accounts.values()) else 'degraded',
        'experiment_id':manifest['experiment_id'],'started_at_ms':manifest['started_at_ms'],
        'started_at_utc':manifest['started_at_utc'],'heartbeat_at_ms':now,'heartbeat_at_utc':iso(now),
        'updated_at_utc':iso(now),'poll_seconds':poll_seconds,'clock':clock,'accounts':accounts,
        'comparisons':comparisons,'automatic_strategy_promotion':False,'forward_validated':False,
        'original_accounts_modified':False,'review':{'elapsed_days':max(0,now-manifest['started_at_ms'])/86400000,
            'minimum_elapsed_days':manifest['design']['review_minimum_elapsed_days'],
            'minimum_closed_positions_each':manifest['design']['review_minimum_closed_positions_each'],
            'interpretation':'First review scale only; no automatic switch or profitability claim.'}}
    write(root/'status.json',status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--require-existing',action='store_true')
    parser.add_argument('--poll-seconds',type=int,default=30)
    args = parser.parse_args()
    if args.poll_seconds<5:
        raise ValueError('Polling must be at least five seconds')
    spec,profile,fingerprint = load_design(args.config)
    root = (args.output_dir or ROOT/spec['output_dir']).resolve()
    allowed = (ROOT/'data/paper_trading').resolve()
    if not root.is_relative_to(allowed) or root==allowed:
        raise ValueError('Use an isolated directory under data/paper_trading')
    root.mkdir(parents=True,exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    review = DeferredReview()
    try:
        with exclusive_process_lock(root/'runner.lock'),deferred_review(review):
            client = BinanceTerminalClient()
            # A restart can protect existing positions even while /time is down.
            # Only first initialization requires a direct authoritative timestamp.
            inception = (read(root/'manifest.json')['started_at_ms'] if (root/'manifest.json').exists()
                else decision_runtime.resolve_clock(client,mode='live')['time_ms'])
            manifest = prepare(root,spec,profile,fingerprint,inception,args.require_existing)
            workers = {arm:execution.SimulationAccount(root/arm/'state.json',
                cost_multiplier=spec['cost_multipliers'][arm.rsplit('_',1)[1]],
                profit_exit_policy=profile.get('profit_exit_policy')) for arm in ARMS}
            collector = MarketCollector(client)
            accounts = {arm:{'status':'starting','errors':{},'heartbeat_at_ms':int(time.time()*1000)} for arm in ARMS}
            clock = {'source':'starting'}
            publish_status(root,manifest,accounts,clock,args.poll_seconds,not args.once)
            try:
                while not stop.is_set():
                    review.begin_tick()
                    accounts,clock = tick(root,manifest,workers,collector)
                    status = publish_status(root,manifest,accounts,clock,args.poll_seconds,not args.once and not stop.is_set())
                    print(json.dumps({key:status[key] for key in ('experiment_id','pid','status','heartbeat_at_utc','running','comparisons')}),flush=True)
                    if args.once:
                        break
                    stop.wait(args.poll_seconds)
            finally:
                publish_status(root,manifest,accounts,clock,args.poll_seconds,False)
    finally:
        review.close()


if __name__=='__main__':
    main()
