"""Prospective, read-only risk hypotheses. No orders, prices or ledger mutation.

Intraday changes use observations first seen after the observer's inception.
They are not fitted trading signals. Daily bond series remain diagnostics until
their publication/availability and independent usefulness have been validated.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import math


DEFAULT_MARKET_POLICY = {
    'required_features': ['nq_dec_2026', 'brent_usd_per_barrel', 'usdjpy'],
    'lookback_minutes': 30,
    'reference_tolerance_minutes': 10,
    'maximum_quote_age_minutes': 20,
    'maximum_quote_skew_minutes': 15,
    'oil_rise_pct': 3.0,
    'nq_decline_pct': -1.5,
    'yen_appreciation_pct': 0.75,
    'nq_absolute_move_pct': 3.0,
}
DEFAULT_PORTFOLIO_POLICY = {
    'stock_accounts': ['mu', 'sndk', 'skhynix'],
    'maximum_state_age_seconds': 180,
    'maximum_same_direction_positions': 1,
    'same_direction_stop_risk_fraction_of_largest_account': 0.0025,
}
MINUTE = 60_000


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _stamp(value):
    if type(value) is not int or value < 0:
        raise ValueError('invalid_timestamp')
    return value


def _iso_ms(value):
    if not isinstance(value, str):
        raise ValueError('state_timestamp_missing')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('state_timestamp_naive')
    return int(parsed.timestamp() * 1000)


def _point(observation, asof_ms):
    if not isinstance(observation, Mapping) or not _number(observation.get('value')) or observation['value'] <= 0:
        raise ValueError('observation_missing_or_invalid')
    observed = _stamp(observation.get('observed_at_ms'))
    first_seen = _stamp(observation.get('first_seen_at_ms'))
    available = _stamp(observation.get('available_at_ms'))
    if not observed <= first_seen <= available <= asof_ms:
        raise ValueError('observation_not_causally_available')
    if observation.get('precision') == 'date' or observation.get('eligible_intraday') is False:
        raise ValueError('intraday_observation_required')
    cutoff = observation.get('contract_review_cutoff_at_ms')
    if cutoff is not None and asof_ms >= _stamp(cutoff):
        raise ValueError('contract_review_due')
    instrument = observation.get('instrument')
    if not instrument:
        raise ValueError('instrument_identity_missing')
    return {'value': float(observation['value']), 'observed_at_ms': observed,
            'first_seen_at_ms': first_seen, 'available_at_ms': available,
            'instrument': json.dumps(instrument, sort_keys=True, ensure_ascii=False)}


def _market_policy(policy):
    if policy is not None and (not isinstance(policy, Mapping) or set(policy) - set(DEFAULT_MARKET_POLICY)):
        raise ValueError('invalid_market_policy_keys')
    result = {**DEFAULT_MARKET_POLICY, **(policy or {})}
    if result['required_features'] != DEFAULT_MARKET_POLICY['required_features']:
        raise ValueError('fixed_research_features_changed')
    for key in ('lookback_minutes', 'reference_tolerance_minutes', 'maximum_quote_age_minutes',
                'maximum_quote_skew_minutes', 'oil_rise_pct', 'yen_appreciation_pct', 'nq_absolute_move_pct'):
        if not _number(result[key]) or result[key] <= 0:
            raise ValueError('invalid_market_policy:' + key)
    if not _number(result['nq_decline_pct']) or result['nq_decline_pct'] >= 0:
        raise ValueError('invalid_market_policy:nq_decline_pct')
    if result['reference_tolerance_minutes'] >= result['lookback_minutes']:
        raise ValueError('reference_tolerance_must_be_less_than_lookback')
    return result


def _portfolio_policy(policy):
    if policy is not None and (not isinstance(policy, Mapping) or set(policy) - set(DEFAULT_PORTFOLIO_POLICY)):
        raise ValueError('invalid_portfolio_policy_keys')
    result = {**DEFAULT_PORTFOLIO_POLICY, **(policy or {})}
    if result['stock_accounts'] != DEFAULT_PORTFOLIO_POLICY['stock_accounts']:
        raise ValueError('fixed_stock_sector_accounts_required')
    if type(result['maximum_state_age_seconds']) is not int or not 0 < result['maximum_state_age_seconds'] <= 3600:
        raise ValueError('invalid_portfolio_state_age')
    if type(result['maximum_same_direction_positions']) is not int or not 1 <= result['maximum_same_direction_positions'] <= 3:
        raise ValueError('invalid_sector_position_limit')
    fraction = result['same_direction_stop_risk_fraction_of_largest_account']
    if not _number(fraction) or not 0 < fraction <= 1:
        raise ValueError('invalid_sector_risk_fraction')
    return result


def update_forward_history(snapshot, history, *, asof_ms, inception_ms):
    """Append immutable observed identities; restarting cannot renew or backfill."""
    _stamp(asof_ms)
    _stamp(inception_ms)
    if inception_ms > asof_ms:
        raise ValueError('future_inception')
    observations = snapshot.get('observations', {}) if isinstance(snapshot, Mapping) else {}
    result = {}
    for feature in DEFAULT_MARKET_POLICY['required_features']:
        valid = []
        for row in (history or {}).get(feature, []):
            try:
                if (isinstance(row, Mapping) and _number(row.get('value')) and row['value'] > 0
                        and inception_ms <= row['observed_at_ms'] <= row['first_seen_at_ms']
                        <= row['available_at_ms'] <= asof_ms
                        and asof_ms - row['observed_at_ms'] <= 4 * 60 * MINUTE
                        and isinstance(row.get('instrument'), str)):
                    valid.append(dict(row))
            except (KeyError, TypeError):
                continue
        unique = {(row['instrument'], row['observed_at_ms']): row for row in valid}
        try:
            point = _point(observations.get(feature), asof_ms)
            if point['observed_at_ms'] >= inception_ms:
                unique.setdefault((point['instrument'], point['observed_at_ms']), point)
        except (ValueError, TypeError):
            pass
        result[feature] = sorted(unique.values(), key=lambda row: row['observed_at_ms'])[-600:]
    return result


def evaluate_cross_market_risk(snapshot, policy, *, asof_ms, direction=0, history=None):
    """Compare a fixed market-stress hypothesis, independently of price direction."""
    _stamp(asof_ms)
    policy = _market_policy(policy)
    result = {'enabled': True, 'allowed': False, 'status': 'unavailable', 'reasons': [],
              'checked_at_ms': asof_ms, 'scope': 'new_risk_only', 'protective_exits_allowed': True,
              'direction': direction, 'features': {}, 'lookback_minutes': policy['lookback_minutes'],
              'hypothesis_only': True, 'pnl_simulated': False,
              'daily_bonds_usage': 'observation_only_no_unvalidated_directional_weight'}
    observations = snapshot.get('observations', {}) if isinstance(snapshot, Mapping) else {}
    points = {}
    for feature in policy['required_features']:
        try:
            point = _point(observations.get(feature), asof_ms)
            if asof_ms - point['observed_at_ms'] > policy['maximum_quote_age_minutes'] * MINUTE:
                raise ValueError('observation_stale')
            same = [row for row in (history or {}).get(feature, [])
                    if row.get('instrument') == point['instrument']
                    and row.get('observed_at_ms') == point['observed_at_ms']]
            if same and same[0]['value'] != point['value']:
                raise ValueError('same_observation_revised')
            points[feature] = point
        except (ValueError, TypeError, KeyError) as exc:
            result['reasons'].append(feature + ':' + str(exc))
    if result['reasons']:
        return result
    stamps = [point['observed_at_ms'] for point in points.values()]
    if max(stamps) - min(stamps) > policy['maximum_quote_skew_minutes'] * MINUTE:
        result['reasons'] = ['cross_market_quote_times_not_comparable']
        return result
    result['valid_until_ms'] = min(stamps) + int(policy['maximum_quote_age_minutes'] * MINUTE)
    for feature, point in points.items():
        boundary = point['observed_at_ms'] - policy['lookback_minutes'] * MINUTE
        candidates = [row for row in (history or {}).get(feature, [])
                      if row.get('instrument') == point['instrument']
                      and _number(row.get('value')) and row['value'] > 0
                      and type(row.get('observed_at_ms')) is int
                      and boundary - policy['reference_tolerance_minutes'] * MINUTE <= row['observed_at_ms'] <= boundary
                      and type(row.get('available_at_ms')) is int and row['available_at_ms'] <= asof_ms
                      and type(row.get('first_seen_at_ms')) is int
                      and row['observed_at_ms'] <= row['first_seen_at_ms'] <= row['available_at_ms']]
        if not candidates:
            result['reasons'].append(feature + ':forward_history_warming_up')
            continue
        reference = max(candidates, key=lambda row: row['observed_at_ms'])
        result['features'][feature] = {'change_pct': 100 * (point['value'] / reference['value'] - 1),
            'current': point, 'reference': reference,
            'elapsed_minutes': (point['observed_at_ms'] - reference['observed_at_ms']) / MINUTE}
    if result['reasons']:
        result['status'] = 'warming_up'
        return result
    reference_stamps = [feature['reference']['observed_at_ms'] for feature in result['features'].values()]
    if max(reference_stamps) - min(reference_stamps) > policy['maximum_quote_skew_minutes'] * MINUTE:
        result.update(status='unavailable', reasons=['cross_market_reference_times_not_comparable'])
        return result
    changes = {name: feature['change_pct'] for name, feature in result['features'].items()}
    nq, oil, fx = (changes[name] for name in policy['required_features'])
    # USD/JPY falling means a stronger yen. Use the reciprocal return, not -FX.
    yen = 100 * (1 / (1 + fx / 100) - 1)
    result['yen_appreciation_pct'] = yen
    reasons = []
    if oil >= policy['oil_rise_pct'] and nq <= policy['nq_decline_pct']:
        reasons.append('energy_and_technology_shock')
    if yen >= policy['yen_appreciation_pct'] and nq <= policy['nq_decline_pct']:
        reasons.append('yen_and_technology_shock')
    if abs(nq) >= policy['nq_absolute_move_pct']:
        reasons.append('technology_disorderly_move')
    result.update(allowed=not reasons, status='shock' if reasons else 'clear', reasons=reasons)
    return result


def portfolio_exposure(states, *, asof_ms, policy=None):
    """Factual stock exposure and stop-distance estimates; not a synthetic book."""
    _stamp(asof_ms)
    policy = _portfolio_policy(policy)
    accounts = {}
    errors = []
    for account_id in policy['stock_accounts']:
        state = (states or {}).get(account_id)
        try:
            if not isinstance(state, Mapping):
                raise ValueError('state_missing')
            stamp = _iso_ms(state.get('checked_at_utc') or state.get('updated_at_utc'))
            if not 0 <= asof_ms - stamp <= policy['maximum_state_age_seconds'] * 1000:
                raise ValueError('state_stale_or_future')
            equity = state.get('equity', state.get('wallet_balance'))
            if not _number(equity) or equity <= 0:
                raise ValueError('equity_invalid')
            position = state.get('position')
            qty = state.get('position_qty', 0)
            if not _number(qty):
                raise ValueError('inventory_invalid')
            row = {'equity_usdt': equity, 'direction': 0, 'qty': 0.0,
                   'notional_usdt': 0.0, 'equity_to_stop_estimate_usdt': 0.0,
                   'entry_to_stop_loss_estimate_usdt': 0.0, 'checked_at_ms': stamp}
            if position:
                direction, amount = position.get('direction'), position.get('qty')
                mark, stop, entry = state.get('last_mark_price'), position.get('stop'), position.get('entry')
                if (direction not in (-1, 1) or type(direction) is bool
                        or any(not _number(v) or v <= 0 for v in (amount, mark, stop, entry))):
                    raise ValueError('position_or_stop_invalid')
                if abs(qty - direction * amount) > 1e-8:
                    raise ValueError('inventory_position_mismatch')
                if direction * (mark - stop) <= 0:
                    raise ValueError('stop_breached_execution_uncertain')
                rule = state.get('rule', {})
                fee, slip = rule.get('taker_fee_rate_assumption', 0), rule.get('adverse_slippage_fraction_assumption', 0)
                if any(not _number(v) or not 0 <= v < 0.5 for v in (fee, slip)):
                    raise ValueError('cost_assumption_invalid')
                closing_cost = amount * mark * (fee + slip)
                row.update(direction=direction, qty=amount, notional_usdt=amount * mark,
                    equity_to_stop_estimate_usdt=max(0.0, direction * (mark - stop)) * amount + closing_cost,
                    entry_to_stop_loss_estimate_usdt=max(0.0, direction * (entry - stop)) * amount + closing_cost)
            elif abs(qty) > 1e-10:
                raise ValueError('inventory_without_position')
            accounts[account_id] = row
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(account_id + ':' + str(exc))
    equity = sum(row['equity_usdt'] for row in accounts.values())
    groups = {str(side): {
        'accounts': [name for name, row in accounts.items() if row['direction'] == side],
        'count': sum(row['direction'] == side for row in accounts.values()),
        'gross_notional_usdt': sum(row['notional_usdt'] for row in accounts.values() if row['direction'] == side),
        'equity_to_stop_estimate_usdt': sum(row['equity_to_stop_estimate_usdt'] for row in accounts.values() if row['direction'] == side),
    } for side in (-1, 1)}
    return {'status': 'unavailable' if errors else 'available', 'reasons': errors,
            'checked_at_ms': asof_ms, 'accounts': accounts, 'combined_equity_usdt': equity,
            'largest_account_equity_usdt': max((r['equity_usdt'] for r in accounts.values()), default=0),
            'same_direction': groups, 'gross_notional_usdt': sum(r['notional_usdt'] for r in accounts.values()),
            'net_notional_usdt': sum(r['direction'] * r['notional_usdt'] for r in accounts.values()),
            'scope': 'factual_existing_positions_only', 'pnl_simulated': False,
            'estimate_note': 'Stop-distance plus assumed exit costs; gaps, funding and stop execution are not guaranteed.'}


def evaluate_sector_budget(exposure, *, account_id, direction, proposed_risk_usdt=0, policy=None, action='new_risk'):
    if action in ('reduce', 'close', 'protective_exit'):
        return {'enabled': True, 'allowed': True, 'status': 'protective_action', 'reasons': [],
                'scope': 'new_risk_only', 'protective_exits_allowed': True, 'pnl_simulated': False}
    if action != 'new_risk':
        raise ValueError('unknown_sector_action')
    policy = _portfolio_policy(policy)
    result = {'enabled': account_id in policy['stock_accounts'], 'allowed': True, 'status': 'not_applicable',
              'reasons': [], 'scope': 'new_risk_only', 'protective_exits_allowed': True,
              'hypothesis_only': True, 'pnl_simulated': False}
    if not result['enabled']:
        return result
    if exposure.get('status') != 'available' or not _number(proposed_risk_usdt) or proposed_risk_usdt < 0:
        return {**result, 'allowed': False, 'status': 'unavailable', 'reasons': ['sector_exposure_unavailable']}
    if direction not in (-1, 1) or isinstance(direction, bool):
        return {**result, 'allowed': False, 'status': 'unavailable', 'reasons': ['candidate_direction_unknown']}
    group = exposure['same_direction'][str(direction)]
    own = exposure['accounts'].get(account_id, {})
    if own.get('direction') in (-1, 1):
        return {**result, 'allowed': False, 'status': 'unavailable',
                'reasons': ['existing_inventory_outside_new_entry_observer']}
    cap = exposure['largest_account_equity_usdt'] * policy['same_direction_stop_risk_fraction_of_largest_account']
    reasons = []
    if group['count'] >= policy['maximum_same_direction_positions']:
        reasons.append('correlated_same_direction_position_limit')
    if group['equity_to_stop_estimate_usdt'] + proposed_risk_usdt > cap + 1e-9:
        reasons.append('correlated_stop_risk_budget')
    result.update(allowed=not reasons, status='blocked' if reasons else 'clear', reasons=reasons,
                  proposed_risk_usdt=proposed_risk_usdt, stop_risk_cap_usdt=cap,
                  current_same_direction_positions=group['count'],
                  note='Observe admission against factual positions; no counterfactual holdings or return simulation.')
    return result
