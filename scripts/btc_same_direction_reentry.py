"""Fixed causal hourly structural events for an isolated BTC paper candidate.

This module neither trades nor changes the original trend. A new event has its
own hour, identity and one-hour expiry; an old trend origin is never renewed.
The original structural module still requires subsequent closed-5m admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import btc_pullback_entry as pullback

HOUR = pullback.HOUR
MODE = 'closed_hour_structure_reentry_v1'
POLICY = {'mode': MODE, 'cooldown_hours': 3, 'maximum_attempts_per_trend': 3,
          'hourly_rejection_buffer_atr': .10}


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError('BTC reentry requires the fixed declared policy')
    return deepcopy(POLICY)


def initial_state(start_ms):
    if type(start_ms) is not int or start_ms <= 0:
        raise ValueError('Invalid reentry inception')
    return {'schema_version': 1, 'started_at_ms': start_ms, 'episode_key': None,
            'attempts': 0, 'last_processed_hour_ms': None, 'last_attempt_ms': None,
            'last_closed_ms': None, 'opportunity': None, 'events': []}


def episode_key(point):
    quantity = float(point.get('signed_qty') or 0)
    if not math.isfinite(quantity) or not quantity:
        return None
    origin = point.get('origin_signal_time_ms')
    identifier = point.get('position_id')
    if (type(origin) is not int or origin <= 0 or origin % HOUR
            or not isinstance(identifier, str) or not identifier.strip()):
        return None
    return f'{identifier}:{origin}:{"long" if quantity > 0 else "short"}'


def _prior_ema48(rows):
    value = float(rows[0][4])
    for row in rows[1:]:
        value += 2 / 49 * (float(row[4]) - value)
    return value


def create_event(point, hourly, now_ms, start_ms, profile, policy=POLICY):
    """Return a fresh structural event only on the latest complete hourly bar.

    Its event structure was knowable before that hour opened. The new post-hour
    entry structure is separately frozen by the existing pullback builder.
    """
    validate_policy(policy)
    key = episode_key(point)
    if key is None or type(now_ms) is not int:
        return None, 'invalid_trend_source'
    rows = [row for row in hourly if len(row) == 8 and int(row[7]) < now_ms]
    structure_policy = pullback.validate_policy(profile['btc_pullback_entry_policy'])
    history_hours = max(structure_policy['lookback_hours'],structure_policy['atr_period']+1,49)
    if len(rows) < history_hours+1:
        return None, 'hourly_history_insufficient'
    if any(int(b[0])-int(a[0]) != HOUR for a,b in zip(rows,rows[1:])):
        return None, 'hourly_gap'
    event = rows[-1]
    origin = int(event[7]) + 1
    if (int(event[0]) % HOUR or origin != int(event[0]) + HOUR
            or origin != now_ms // HOUR * HOUR or origin <= start_ms):
        return None, 'no_new_post_inception_closed_hour'
    quantity = float(point['signed_qty'])
    side = 1 if quantity > 0 else -1
    closing = float(event[4])
    identifier = 'btc-reentry-' + hashlib.sha256(f'{key}:{origin}'.encode()).hexdigest()[:24]
    target = {'target_leverage':side, 'position_id':identifier,
              'origin_signal_time_ms':origin, 'origin_entry_price':closing}
    pending = pullback.build_pending(target,rows[-history_hours:],structure_policy,now_ms)
    if pending.get('status') != 'waiting_retrace':
        return None, 'no_legal_post_hour_structure'
    selected = pending['selected_structure']
    if selected['kind'] == 'ema48':
        # The prior-hour EMA is known before this event candle begins.
        event_price = _prior_ema48(rows[-history_hours-1:-1])
        structure_available_ms = int(event[0])-1
    else:
        event_price = float(selected['price'])
        structure_available_ms = int(selected['available_at_ms'])
        if structure_available_ms >= int(event[0]):
            return None, 'structure_not_known_before_event_hour'
    width = pending['atr'] * policy['hourly_rejection_buffer_atr']
    low, high = event_price-width, event_price+width
    opening, bar_high, bar_low = map(float, event[1:4])
    touch = bar_low <= high and bar_high >= low
    reject = (closing > opening and closing > high if side > 0
              else closing < opening and closing < low)
    if not touch or not reject:
        return None, 'no_closed_hour_touch_and_rejection'
    return {'opportunity_id':identifier, 'episode_key':key, 'direction':side,
        'origin_signal_time_ms':origin, 'origin_entry_price':closing,
        'expires_at_ms':origin+HOUR, 'created_at_ms':now_ms,
        'event_hour_open_ms':int(event[0]), 'event_hour_close_ms':int(event[7]),
        'event_structure':{'kind':selected['kind'],'price':event_price,
                           'available_at_ms':structure_available_ms},
        'entry_structure':deepcopy(selected), 'original_source':{
            'position_id':point['position_id'],
            'origin_signal_time_ms':point['origin_signal_time_ms']},
        'status':'created'}, None


def candidate_point(point, state, account_state, hourly, context, now_ms, profile, policy=POLICY):
    """Advance persisted event admission and return an isolated execution point."""
    validate_policy(policy)
    updated = deepcopy(state)
    key = episode_key(point)
    held = float(account_state.get('position_qty') or 0)
    requested = float(point.get('signed_qty') or 0)
    opportunity = updated.get('opportunity')
    ledger_pending = account_state.get('btc_pullback_pending') or {}
    ledger_source = ledger_pending.get('source') or {}
    if (opportunity and not held and opportunity.get('status') in {'created','filled'}
            and (opportunity.get('status') == 'filled' or (
                ledger_pending.get('status') == 'filled'
                and ledger_source.get('position_id') == opportunity.get('opportunity_id')))):
        # A source-only exit may close inventory before this state machine runs
        # again. Preserve the post-close cooldown even on that immediate path.
        opportunity.update(status='closed',closed_at_ms=now_ms)
        updated['last_closed_ms'] = now_ms
    if key is not None and key != updated.get('episode_key'):
        if opportunity:
            opportunity.update(status='cancelled_source_changed',cancelled_at_ms=now_ms)
        updated.update(episode_key=key,attempts=0,opportunity=None)
        opportunity = None
    if not key:
        if opportunity and opportunity.get('status') == 'created':
            opportunity.update(status='cancelled_source_flat',cancelled_at_ms=now_ms)
        updated['opportunity'] = None
        return dict(point), updated, {'status':'source_flat_or_invalid'}
    # Candidate-held inventory follows the original source's sizing and exits,
    # while preserving the actual entry event identity across report refreshes.
    if held and held*requested > 0 and opportunity:
        opportunity['status'] = 'filled'
        return {**point,'position_id':opportunity['opportunity_id'],
            'origin_signal_time_ms':opportunity['origin_signal_time_ms'],
            'origin_entry_price':opportunity['origin_entry_price']}, updated, {'status':'holding_event'}
    if opportunity and opportunity.get('status') == 'filled' and not held:
        opportunity.update(status='closed',closed_at_ms=now_ms)
        updated['last_closed_ms'] = now_ms
    if opportunity and opportunity.get('status') == 'created' and now_ms < opportunity['expires_at_ms']:
        return {**point,'position_id':opportunity['opportunity_id'],
            'origin_signal_time_ms':opportunity['origin_signal_time_ms'],
            'origin_entry_price':opportunity['origin_entry_price']}, updated, {'status':'waiting_event_5m'}
    if opportunity and opportunity.get('status') == 'created':
        opportunity['status'] = 'expired'
    # Original fresh opportunities remain identical in both arms. Extras begin
    # only after their actual one-hour origin window has expired.
    source_origin = point['origin_signal_time_ms']
    if source_origin >= state['started_at_ms'] and now_ms-source_origin < HOUR:
        return dict(point), updated, {'status':'original_fresh_signal'}
    origin = now_ms // HOUR * HOUR
    if updated.get('last_processed_hour_ms') == origin:
        return dict(point), updated, {'status':'hour_already_checked'}
    if held:
        return dict(point), updated, {'status':'inventory_not_flat'}
    if updated['attempts'] >= policy['maximum_attempts_per_trend']:
        return dict(point), updated, {'status':'trend_attempt_limit'}
    cooldown_start = max(updated.get('last_attempt_ms') or 0, updated.get('last_closed_ms') or 0)
    if cooldown_start and now_ms-cooldown_start < policy['cooldown_hours']*HOUR:
        return dict(point), updated, {'status':'cooldown'}
    expected_side = 'long' if requested > 0 else 'short'
    last_bar = (context or {}).get('last_closed_bar') or {}
    if (not context or context.get('status') != 'available'
            or context.get('confirmed_side') != expected_side
            or context.get('target_side_valid_under_exit_rule') is not True
            or context.get('fast_ema_period_bars') != 48
            or context.get('slow_ema_period_bars') != 240
            or context.get('entry_min_ema_spread_pct') != .4
            or last_bar.get('close_time_ms') != origin-1):
        return dict(point), updated, {'status':'current_hour_trend_not_confirmed'}
    updated['last_processed_hour_ms'] = origin
    event, reason = create_event(point,hourly,now_ms,state['started_at_ms'],profile,policy)
    if event is None:
        return dict(point), updated, {'status':reason}
    updated.update(opportunity=event, attempts=updated['attempts']+1,last_attempt_ms=now_ms)
    updated['events'] = [*updated.get('events',[]),deepcopy(event)][-200:]
    return {**point,'position_id':event['opportunity_id'],
        'origin_signal_time_ms':event['origin_signal_time_ms'],
        'origin_entry_price':event['origin_entry_price']}, updated, {'status':'new_hourly_structural_event','event':event}
