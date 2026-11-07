"""Causal main-stock retest entries; no I/O, fills, or future swing labels.

An original admitted hourly breakout freezes the level/ATR/reference. Only
whole five-minute bars formed after first observation can touch and confirm.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

FIVE=300_000
HOUR=3_600_000
POLICY_KEY='main_trend_pullback_entry_policy'
POLICY={'mode':'frozen_breakout_retest_v1','maximum_setup_hours':3,
        'zone_half_width_atr':0.25,'invalidation_distance_atr':0.5,
        'minimum_price_improvement_atr':0.1,'confirmation_validity_minutes':10,
        'structural_stop_buffer_atr':0.1,'confirmation_timeframe':'5m'}
TERMINAL={'cancelled','expired','consumed'}

def enabled(config):return isinstance(config,Mapping) and POLICY_KEY in config

def validate(config):
    if not enabled(config):return
    p=config[POLICY_KEY]
    if (not isinstance(p,Mapping) or set(p)!=set(POLICY)
            or any(type(p[k]) is not type(v) or p[k]!=v for k,v in POLICY.items())):
        raise ValueError('pullback_policy_invalid')
    if config.get('signal_family')!='donchian20' or config.get('signal_timeframe')!='1h':
        raise ValueError('pullback_original_hourly_breakout_required')
    for key in ['main_stock_entry_trend_policy','main_stock_execution_guard_policy']:
        if key not in config:raise ValueError('pullback_existing_confirmation_and_execution_guards_required')

def number(v):
    if isinstance(v,bool):
        raise ValueError('pullback_invalid_price')
    try:result=float(v)
    except (ValueError,TypeError,OverflowError):raise ValueError('pullback_invalid_price') from None
    if not math.isfinite(result) or result<=0:raise ValueError('pullback_invalid_price')
    return result

def stamp(v):
    if type(v) is not int or v<0:raise ValueError('pullback_invalid_timestamp')
    return v

def seed(pending, *, timestamp, active_after):
    timestamp=stamp(timestamp);active_after=stamp(active_after)
    signal=deepcopy(pending['signal']);boundary=stamp(pending['boundary_ms'])
    origin=stamp(signal['time_ms']);side=signal['direction']
    if type(side) is not int or side not in (-1,1):raise ValueError('pullback_invalid_direction')
    if boundary%HOUR or origin+HOUR!=boundary or not active_after<boundary<=timestamp:
        raise ValueError('pullback_new_hourly_signal_required')
    trigger=pending['entry_trigger']
    if trigger.get('family')!='donchian20' or trigger.get('boundary_ms')!=boundary or trigger.get('direction')!=side:
        raise ValueError('pullback_original_breakout_provenance_required')
    level=number(trigger['breakout_level']);atr=number(signal['atr']);close=number(signal['close'])
    if not 0<side*(close-level)<=atr:raise ValueError('pullback_original_breakout_invalid')
    expiry=boundary+POLICY['maximum_setup_hours']*HOUR
    if timestamp>=expiry:raise ValueError('pullback_original_setup_expired')
    first=(timestamp//FIVE+1)*FIVE
    return {'version':1,'id':str(boundary)+':'+str(side),'status':'waiting_retest','signal':signal,
        'boundary_ms':boundary,'created_at_ms':timestamp,'expires_at_ms':expiry,
        'breakout_level':level,'atr':atr,'original_close':close,'direction':side,
        'zone_low':level-POLICY['zone_half_width_atr']*atr,
        'zone_high':level+POLICY['zone_half_width_atr']*atr,
        'setup_invalidation':level-side*POLICY['invalidation_distance_atr']*atr,
        'first_eligible_bar_open_ms':first,'last_processed_open_ms':first-FIVE,
        'reasons':['pullback_waiting_retest']}

def cancel(setup,reason,status='cancelled'):
    setup.update(status=status,reasons=[reason]);return setup

def advance(original,rows,*,timestamp):
    """Process each completed, contiguous observation at most once."""
    setup=deepcopy(original);timestamp=stamp(timestamp)
    if setup['status'] in TERMINAL:return setup
    if timestamp>=setup['expires_at_ms']:return cancel(setup,'pullback_setup_expired','expired')
    if setup['status']=='ready' and timestamp>=setup['confirmation_expires_at_ms']:
        return cancel(setup,'pullback_confirmation_expired','expired')
    side=setup['direction'];new=[];seen={}
    try:
        for row in rows:
            t=stamp(row[0]);end=stamp(row[6])
            if end>=timestamp:continue
            o,h,l,c=[number(row[i]) for i in (1,2,3,4)]
            if t%FIVE or end!=t+FIVE-1 or l>min(o,c) or h<max(o,c):
                raise ValueError('pullback_invalid_five_minute_bar')
            if end>=timestamp or t<setup['first_eligible_bar_open_ms']:continue
            bar={'open_ms':t,'close_ms':t+FIVE,'open':o,'high':h,'low':l,'close':c}
            if t in seen and bar!=seen[t]:raise ValueError('pullback_conflicting_completed_bar')
            seen[t]=bar
        previous=setup.get('last_bar')
        if previous and previous['open_ms'] in seen and previous!=seen[previous['open_ms']]:
            revision={'first_seen':previous,'later_published':seen[previous['open_ms']]}
            if revision!=setup.get('last_price_revision'):
                setup['last_price_revision']=revision
                setup['revised_price_observations']=setup.get('revised_price_observations',0)+1
                setup['revision_observed_at_ms']=timestamp
            # Keep the original decision prefix, as actually observed. A later
            # provider revision cannot move the frozen touch/confirmation/stop.
        new=[seen[t] for t in sorted(seen) if t>setup['last_processed_open_ms']]
        for bar in new:
            if bar['open_ms']!=setup['last_processed_open_ms']+FIVE:
                raise ValueError('pullback_five_minute_gap')
            if side*(bar['close']-setup['setup_invalidation'])<=0:
                return cancel(setup,'pullback_structure_invalidated')
            previous=setup.get('last_bar')
            if setup['status']=='waiting_retest' and bar['low']<=setup['zone_high'] and bar['high']>=setup['zone_low']:
                setup.update(status='waiting_confirmation',touch_close_ms=bar['close_ms'],
                    swing_low=bar['low'],swing_high=bar['high'],reasons=['pullback_waiting_confirmation'])
            elif setup['status']=='waiting_confirmation':
                setup['swing_low']=min(setup['swing_low'],bar['low'])
                setup['swing_high']=max(setup['swing_high'],bar['high'])
                level=previous['high'] if side==1 else previous['low']
                if (bar['open_ms']>=setup['touch_close_ms'] and side*(bar['close']-level)>0
                        and side*(bar['close']-bar['open'])>0):
                    setup.update(status='ready',confirmation_close_ms=bar['close_ms'],
                        confirmation_expires_at_ms=min(setup['expires_at_ms'],
                            bar['close_ms']+POLICY['confirmation_validity_minutes']*60_000),
                        confirmation_level=level,confirmation_price=bar['close'],
                        structural_invalidation=(setup['swing_low']-POLICY['structural_stop_buffer_atr']*setup['atr']
                            if side==1 else setup['swing_high']+POLICY['structural_stop_buffer_atr']*setup['atr']),
                        reasons=[])
            setup.update(last_bar=bar,last_processed_open_ms=bar['open_ms'])
    except (ValueError,TypeError,IndexError,KeyError) as exc:
        return cancel(setup,str(exc) if isinstance(exc,ValueError) else 'pullback_five_minute_data_invalid')
    setup['checked_at_ms']=timestamp
    return setup

def pending_entry(setup):
    if setup['status']!='ready':return None
    return {'signal':deepcopy(setup['signal']),'boundary_ms':setup['boundary_ms'],
        'expires_at_ms':setup['confirmation_expires_at_ms'],
        'entry_trigger':{'family':'trend_pullback','setup':deepcopy(setup)}}

def evaluate(trigger,signal,config,*,reference_price,mark_price,fill_price=None):
    """Original signal is retained. Every actual fill must improve its price."""
    result={'enabled':True,'allowed':False,'cancel_pending':False,'reasons':[]}
    try:
        validate(config)
        if not enabled(config) or trigger.get('family')!='trend_pullback':
            raise ValueError('pullback_trigger_required')
        s=trigger['setup'];side=s['direction']
        raw=signal if isinstance(signal,Mapping) else vars(signal)
        if (s['status']!='ready' or raw!=s['signal']):raise ValueError('pullback_signal_provenance_mismatch')
        prices=[('reference',number(reference_price)),('mark',number(mark_price))]
        if fill_price is not None:prices.append(('fill',number(fill_price)))
        improvements=[]
        for label,p in prices:
            if side*(p-s['structural_invalidation'])<=0:
                result['cancel_pending']=True;result['reasons'].append('pullback_structure_invalidated')
            if side*(p-s['confirmation_level'])<=0:
                result['cancel_pending']=True;result['reasons'].append('pullback_confirmation_lost')
            improvement=-side*(p-s['original_close'])/s['atr'];improvements.append(improvement)
            if improvement+1e-12<POLICY['minimum_price_improvement_atr']:
                result['reasons'].append('pullback_price_improvement_insufficient')
            if side*(p-s['confirmation_price'])>config.get('entry_gap_atr',.5)*s['atr']+1e-12:
                result['reasons'].append('pullback_confirmation_chase_exceeded')
        result.update(allowed=not result['reasons'],reasons=sorted(set(result['reasons'])),
            setup_id=s['id'],minimum_observed_improvement_atr=min(improvements),
            structural_invalidation=s['structural_invalidation'],confirmation_level=s['confirmation_level'])
    except (ValueError,KeyError,TypeError,AttributeError) as exc:
        result.update(cancel_pending=True,reasons=[str(exc) if isinstance(exc,ValueError) else 'pullback_trigger_invalid'])
    return result

def initial_stop(setup,entry,config):
    """Define structure BEFORE quantity. Retain the original min/max risk width."""
    if setup.get('status')!='ready':return None
    entry=number(entry);side=setup['direction'];line=number(setup['structural_invalidation'])
    if side*(entry-line)<=0:return None
    floor=entry*config.get('min_stop_fraction',.02)
    distance=max(side*(entry-line),floor)
    original=max(config.get('stop_atr',2.)*setup['atr'],floor)
    if distance>original+1e-12 or distance/entry>config.get('max_stop_fraction',.05)+1e-12:return None
    return entry-side*distance
