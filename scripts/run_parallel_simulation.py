#!/usr/bin/env python3
"""Four segregated forward paper accounts. Public GETs only; no order API."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_DOWN
from process_lock import exclusive_process_lock
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.parse import urlencode

import requests

import backtest_stock_swing_120 as arithmetic
from binance_terminal_client import BinanceTerminalClient
import decision_runtime
import simulation_risk_monitor
import stock_swing_profiles
import stock_swing_signals as signals
import stock_profit_exits as profit_exits
import stock_research_entry_policy as entry_policy
from trading_execution import SimulationAccount, execute_report, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/parallel_simulation_plan_20261005.json'
FOUR_HOURS = 14_400_000
FIVE_MINUTES = 300_000
SIGNAL_INTERVALS = {'5m': FIVE_MINUTES, '15m': 900_000, '30m': 1_800_000,
                    '1h': 3_600_000, '4h': FOUR_HOURS}
STOP = threading.Event()


def now_ms():
    return int(time.time() * 1000)


def journal(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def floor_qty(qty, step):
    unit = Decimal(str(step))
    return float((Decimal(str(qty)) / unit).to_integral_value(rounding=ROUND_DOWN) * unit)


def fresh(timestamp, reference, limit=60_000):
    if not -5_000 <= reference - int(timestamp) <= limit:
        raise ValueError('Stale or future market observation')


def book_fill(book, signed_qty, step, slip=0.0002, participation=0.1):
    """IOC paper fill using <=10% of each visible level within a 50bps band."""
    levels = book['asks' if signed_qty > 0 else 'bids']
    parsed = [(float(p), float(q)) for p, q in levels]
    if not parsed or any(not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q < 0 for p, q in parsed):
        raise ValueError('Invalid book levels')
    best = parsed[0][0]
    remaining = abs(signed_qty)
    qty = notional = 0.0
    for price, available in parsed:
        if abs(price / best - 1) > 0.005:
            break
        take = min(remaining, available * participation)
        qty += take
        notional += take * price
        remaining -= take
        if remaining <= 1e-12:
            break
    rounded = floor_qty(qty, step)
    if rounded <= 0:
        return {'qty': 0.0, 'price': None, 'requested_qty': abs(signed_qty), 'partial': True}
    return {'qty': rounded, 'price': notional / qty * (1 + (1 if signed_qty > 0 else -1) * slip),
            'requested_qty': abs(signed_qty), 'partial': rounded < abs(signed_qty) - step / 2,
            'model': 'visible_depth_10pct_ioc_plus_adverse_slippage',
            'book_time_ms': int(book.get('T') or book.get('E')),
            'last_update_id': book.get('lastUpdateId')}


class PublicAster:
    """Allowlisted credential-free reads, verified TLS on both transports."""
    allowed = {'time', 'exchangeInfo', 'premiumIndex', 'depth', 'klines', 'fundingRate', 'markPriceKlines'}

    def __init__(self):
        self.cooldown = 0
        self.lock = threading.Lock()

    def get(self, endpoint, params=None):
        if endpoint not in self.allowed:
            raise ValueError('Only public market endpoints allowed')
        if now_ms() < self.cooldown:
            raise RuntimeError('Aster public cooldown active')
        url = 'https://fapi.asterdex.com/fapi/v3/' + endpoint
        try:
            response = requests.get(url, params=params, timeout=12)
            status = response.status_code
            payload = response.json()
        except (requests.ConnectionError, requests.Timeout):
            query = '?' + urlencode(params) if params else ''
            result = subprocess.run(['/usr/bin/curl', '--silent', '--show-error', '--max-time', '15',
                                     '--write-out', '\n%{http_code}', url + query],
                                    capture_output=True, text=True, timeout=18)
            if result.returncode:
                raise RuntimeError('Public verified fallback transport failed')
            body, _, status = result.stdout.rpartition('\n')
            status, payload = int(status), json.loads(body)
        if status in (418, 429):
            with self.lock:
                self.cooldown = now_ms() + 120_000
        if status != 200 or isinstance(payload, dict) and int(payload.get('code', 0)) < 0:
            raise RuntimeError(f'Aster public {endpoint} HTTP {status}')
        return payload


def initial_stock(symbol, start, config):
    return {'version': 1, 'mode': 'simulation', 'places_orders': False, 'symbol': symbol,
            'created_at_utc': arithmetic.iso(start), 'start_ms': start, 'initial_balance': 1000.0,
            'wallet_balance': 1000.0, 'equity': 1000.0, 'realized_pnl': 0.0,
            'fees_paid': 0.0, 'funding_pnl': 0.0, 'max_leverage': 10,
            'peak_equity': 1000.0, 'max_drawdown_pct': 0.0, 'position': None,
            'position_history': [], 'settled_funding': [], 'fills': [], 'fill_count_total': 0,
            'last_signal_time_ms': None, 'cooldown_until_ms': 0, 'risk_halted': False,
            'observations': 0, 'rule': config, 'status': 'starting'}


class StockAccount:
    def __init__(self, path, venue, symbol, config):
        self.path, self.venue, self.symbol, self.config = path, venue, symbol, config
        self.rules = None
        self.rules_at_ms = 0
        self.signal_timeframe = config.get('signal_timeframe', '4h')
        self.signal_interval_ms = SIGNAL_INTERVALS[self.signal_timeframe]
        entry_policy.validate({**config, 'symbols': [symbol]}, FIVE_MINUTES)
        profit_exits.validate(config)
        self.signal_bars = []
        self.five = []
        self.candle_bucket = None

    def funding(self, state, events, timestamp):
        known = set(state['settled_funding'])
        for event in sorted(events, key=lambda x: int(x['fundingTime'])):
            event_time = int(event['fundingTime'])
            key = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
            if key in known or not state['start_ms'] <= event_time <= timestamp:
                continue
            qty = 0.0
            for point in state['position_history']:
                if point['time_ms'] >= event_time:
                    break  # An entry at the settlement time does not get that settlement.
                qty = point['signed_qty']
            if qty:
                price = event.get('markPrice')
                method = 'published_settlement_mark'
                if not price:
                    rows = self.venue.get('markPriceKlines', {'symbol': self.symbol, 'interval': '1m',
                        'startTime': event_time // 60_000 * 60_000, 'limit': 1})
                    if not rows or int(rows[0][0]) != event_time // 60_000 * 60_000:
                        raise ValueError('Funding mark proxy unavailable')
                    price = rows[0][1]
                    method = 'one_minute_mark_open_proxy'
                debit = qty * float(price) * float(event['fundingRate'])
                state['wallet_balance'] -= debit
                state['funding_pnl'] -= debit
                open_debit = 0.0
                if state['position'] and event_time > state['position']['entry_time']:
                    # A late event can refer to inventory larger than today's residual.
                    open_debit = debit * min(1.0, state['position']['qty'] / abs(qty))
                    state['position']['funding'] += open_debit
                journal(self.path.with_name('funding.jsonl'), {**event, 'debit': debit,
                    'inventory_qty': qty, 'settlement_price': float(price), 'price_method': method,
                    'open_funding_debit': open_debit, 'closed_funding_debit': debit - open_debit})
            state['settled_funding'].append(key)
            known.add(key)

    def fill(self, state, result, timestamp, direction, reason, trigger_ms=None):
        record = {**result, 'time_ms': timestamp, 'time_utc': arithmetic.iso(timestamp),
                  'symbol': self.symbol, 'side': 'BUY' if direction == 1 else 'SELL',
                  'reason': reason, 'mode': 'simulation', 'places_orders': False,
                  'sequence': state['fill_count_total'] + 1,
                  'trigger_observed_at_ms': trigger_ms,
                  'observed_trigger_to_fill_ms': timestamp - trigger_ms if trigger_ms else None}
        state['fees_paid'] += result['qty'] * result['price'] * self.config['taker_fee_rate_assumption']
        state['wallet_balance'] -= result['qty'] * result['price'] * self.config['taker_fee_rate_assumption']
        record['fee'] = result['qty'] * result['price'] * self.config['taker_fee_rate_assumption']
        state['fills'].append(record)
        state['fills'] = state['fills'][-100:]
        state['fill_count_total'] += 1
        journal(self.path.with_name('fills.jsonl'), record)
        return record

    def market_data(self, force_rules=False, force_signal=False):
        state = json.loads(self.path.read_text())  # Corrupt ledgers never reset silently.
        def fetch(label, endpoint, params):
            try:
                return label, self.venue.get(endpoint, params), None
            except Exception as exc:
                return label, None, type(exc).__name__ + ': ' + str(exc)[:160]
        requests_to_make = [('clock', 'time', {}), ('mark', 'premiumIndex', {'symbol': self.symbol}),
                            ('book', 'depth', {'symbol': self.symbol, 'limit': 100}),
                            ('funding', 'fundingRate', {'symbol': self.symbol,
                                'startTime': max(state['start_ms'], state.get('funding_through_ms', state['start_ms']) - 3_600_000),
                                'limit': 1000}),
                            # Closed volumes can be revised after their first publication.
                            # Refresh execution liquidity on every poll, independently of signals.
                            ('five', 'klines', {'symbol': self.symbol, 'interval': '5m', 'limit': 12})]
        if force_rules or self.rules is None or now_ms() - self.rules_at_ms >= 3_600_000:
            requests_to_make.append(('rules', 'exchangeInfo', {}))
        if force_signal or self.candle_bucket != now_ms() // FIVE_MINUTES:
            requests_to_make.append(('signal', 'klines', {'symbol': self.symbol, 'interval': self.signal_timeframe, 'limit': 500}))
        with ThreadPoolExecutor(max_workers=7) as pool:
            return list(pool.map(lambda args: fetch(*args), requests_to_make))

    def close(self, state, pos, book, requested, reason, trigger_ms, step, min_qty):
        """Book one observed exit leg and allocate costs to that leg exactly once."""
        result = book_fill(book, -pos['direction'] * requested, step,
                           self.config['adverse_slippage_fraction_assumption'])
        if result['qty'] < min_qty:
            return pos
        record = self.fill(state, result, now_ms(), -pos['direction'], reason, trigger_ms)
        gross = pos['direction'] * result['qty'] * (result['price'] - pos['entry'])
        state['wallet_balance'] += gross
        state['realized_pnl'] += gross
        fraction = result['qty'] / pos['qty']
        remainder = floor_qty(Decimal(str(pos['qty'])) - Decimal(str(result['qty'])), step)
        journal(self.path.with_name('closed_trades.jsonl'), {**record, 'gross_pnl': gross,
            'allocated_entry_fee': pos['entry_fee'] * fraction,
            'allocated_funding_debit': pos['funding'] * fraction,
            'net_pnl': gross - record['fee'] - pos['entry_fee'] * fraction - pos['funding'] * fraction,
            'entry_time_ms': pos['entry_time'], 'signal_time_ms': pos['signal_time'],
            'position_closed': remainder <= 0, 'remaining_qty': remainder})
        if reason == 'one_r_partial':
            profit_exits.record_partial(pos, result['qty'], step)
        pos['qty'] = remainder
        for key in ('margin', 'entry_fee', 'funding'):
            pos[key] *= 1 - fraction
        if remainder <= 0:
            state['position'] = None
            state['cooldown_until_ms'] = now_ms() + (
                self.config['cooldown_signal_bars'] * self.signal_interval_ms
                if 'cooldown_signal_bars' in self.config else self.config['cooldown_4h_bars'] * FOUR_HOURS)
        state['position_history'].append({'time_ms': record['time_ms'],
            'signed_qty': pos['direction'] * remainder})
        return pos if remainder > 0 else None

    def additional_exit_reason(self, state, position, data, errors, timestamp):
        """Optional paper-experiment extension after the existing full-exit guards."""
        return None

    def step(self, market_data=None):
        state = json.loads(self.path.read_text())  # Corrupt ledgers never reset silently.
        cfg = self.config
        errors = {}
        responses = self.market_data() if market_data is None else market_data
        data = {}
        for label, payload, error in responses:
            data[label] = payload
            if error:
                errors[label] = error
        clock = data.get('clock')
        timestamp = int(clock['serverTime']) if clock else now_ms()
        if clock:
            fresh(timestamp, now_ms(), 15_000)
        mark = data.get('mark')
        book = data.get('book')
        state['observations'] += 1
        state['checked_at_utc'] = arithmetic.iso(now_ms())
        state['errors'] = errors
        if data.get('rules'):
            self.rules = next(row for row in data['rules']['symbols'] if row['symbol'] == self.symbol)
            if self.rules['status'] != 'TRADING':
                errors['rules'] = 'Contract is not trading'
            self.rules_at_ms = now_ms()
            write_json(self.path.with_name('contract_rules.json'), self.rules)
        for key in ('signal', 'five'):
            if data.get(key):
                closed = [row for row in data[key] if int(row[6]) < timestamp]
                if any(int(b[0]) - int(a[0]) != (self.signal_interval_ms if key == 'signal' else FIVE_MINUTES)
                       for a, b in zip(closed, closed[1:])):
                    errors[key] = 'Nonconsecutive completed candles'
                elif key == 'signal':
                    self.signal_bars = closed
                else:
                    self.five = closed
            if key in data and not errors.get(key) and not data.get(key):
                errors[key] = 'Completed candle response is empty'
        if 'signal' in data and not errors.get('signal') and not errors.get('five'):
            self.candle_bucket = now_ms() // FIVE_MINUTES
        if not mark:
            state['status'] = 'degraded'
            write_json(self.path, state)
            journal(self.path.with_name('observations.jsonl'), {'checked_at_utc': state['checked_at_utc'], 'errors': errors})
            return state
        fresh(mark['time'], now_ms())
        price, index = float(mark['markPrice']), float(mark['indexPrice'])
        if not all(math.isfinite(p) and p > 0 for p in (price, index)):
            raise ValueError('Invalid mark/index price')
        fresh_book = False
        try:
            if not book:
                raise ValueError('No current depth')
            fresh(book.get('T') or book.get('E'), now_ms())
            bid, ask = float(book['bids'][0][0]), float(book['asks'][0][0])
            if not 0 < bid <= ask or not math.isfinite(ask):
                raise ValueError('Invalid crossed/empty book')
            fresh_book = True
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            errors['book'] = str(exc)
        if data.get('funding') is not None:
            try:
                if len(data['funding']) >= 1000:
                    raise ValueError('Funding history pagination required; new risk blocked')
                self.funding(state, data['funding'], timestamp)
                pending = state.get('next_funding_ms')
                if pending and timestamp >= pending and not any(pending <= int(e['fundingTime']) <= pending + 5000 for e in data['funding']):
                    errors['funding'] = 'Settlement publication pending'
                else:
                    state['funding_through_ms'] = timestamp
                    state['next_funding_ms'] = int(mark['nextFundingTime'])
            except Exception as exc:
                errors['funding'] = type(exc).__name__ + ': ' + str(exc)[:160]
        pos = state['position']
        equity = state['wallet_balance'] + (pos['direction'] * pos['qty'] * (price - pos['entry']) if pos else 0)
        state['peak_equity'] = max(state['peak_equity'], equity)
        dd = max(0, 1 - equity / state['peak_equity'])
        state['max_drawdown_pct'] = max(state['max_drawdown_pct'], dd * 100)
        state['risk_halted'] |= dd >= cfg['account_hard_drawdown_fraction'] or equity <= 0
        filters = {row['filterType']: row for row in (self.rules or {}).get('filters', [])}
        lot = filters.get('MARKET_LOT_SIZE') or filters.get('LOT_SIZE')
        if lot:
            step = float(lot['stepSize'])
            tick = float(filters['PRICE_FILTER']['tickSize'])
            minimum = float(filters.get('MIN_NOTIONAL', {}).get('notional', filters.get('MIN_NOTIONAL', {}).get('minNotional', 5)))
        if pos:
            side = pos['direction']
            executable = (bid if side == 1 else ask) if fresh_book else price
            managed = profit_exits.enabled(cfg)
            atr = atr_bar_ms = None
            if managed and fresh_book and lot:
                profit_exits.observe(pos, executable, cfg, timestamp, step,
                                     float(lot['minQty']), minimum)
                if self.signal_bars and not errors.get('signal'):
                    latest = self.signal_bars[-1]
                    if timestamp - self.signal_interval_ms - 5000 <= int(latest[6]) < timestamp:
                        try:
                            bars = [arithmetic.candle(row) for row in self.signal_bars]
                            atr = signals.compute_indicators(bars, cfg)['atr'][-1]
                            atr_bar_ms = int(latest[6])
                        except (ValueError, KeyError, IndexError, TypeError) as exc:
                            errors['profit_atr'] = type(exc).__name__ + ': ' + str(exc)[:160]
                profit_exits.protect(pos, cfg, tick, atr, atr_bar_ms)
            target = None if managed else signals.target_exit_price(pos['entry'], side, pos['funding'] / pos['qty'],
                cfg['taker_fee_rate_assumption'], cfg['target_margin_return'], cfg['leverage'],
                entry_fee_per_unit=pos['entry_fee'] / pos['qty'])
            reason = None
            if state['risk_halted']:
                reason = 'account_hard_stop'
            elif side * (price - arithmetic.liquidation_price(pos, cfg['maintenance_margin_fraction_assumption'])) <= 0:
                reason = 'liquidation_stress'
            elif side * (executable - pos['stop']) <= 0:
                reason = 'protective_stop'
            elif not managed and fresh_book and side * (executable * (1 - side * cfg['adverse_slippage_fraction_assumption']) - target) >= 0:
                reason = 'net120_target'
            elif timestamp - pos['entry_time'] >= cfg['max_holding_calendar_days'] * 86_400_000:
                reason = 'time_stop'
            elif pos['funding'] > pos['margin'] * cfg['max_funding_debit_fraction_initial_margin']:
                reason = 'funding_budget'
            else:
                reason = self.additional_exit_reason(state, pos, data, errors, timestamp)
            if reason and not pos.get('pending_exit'):
                pos['pending_exit'] = reason
                pos['trigger_observed_at_ms'] = now_ms()
            if pos.get('pending_exit') and fresh_book and lot:
                fresh(book.get('T') or book.get('E'), now_ms())
                fresh(mark['time'], now_ms())
                result = book_fill(book, -side * pos['qty'], step, cfg['adverse_slippage_fraction_assumption'])
                if result['qty'] >= float(lot['minQty']):
                    pos = self.close(state, pos, book, pos['qty'], pos['pending_exit'],
                                     pos['trigger_observed_at_ms'], step, float(lot['minQty']))
            elif managed and fresh_book and lot and not pos.get('pending_exit'):
                requested = profit_exits.requested_qty(pos, step)
                if requested >= float(lot['minQty']):
                    fresh(book.get('T') or book.get('E'), now_ms())
                    fresh(mark['time'], now_ms())
                    pos = self.close(state, pos, book, requested, 'one_r_partial',
                                     pos['profit_exit']['trigger_observed_at_ms'], step, float(lot['minQty']))
                    if pos:
                        profit_exits.protect(pos, cfg, tick, atr, atr_bar_ms)
            # Trailing update only from a newly closed signal bar and AFTER current-stop checks.
            if not managed and pos and self.signal_bars and not errors.get('signal'):
                latest = self.signal_bars[-1]
                if int(latest[6]) > pos.get('last_trail_bar_ms', 0):
                    bars = [arithmetic.candle(row) for row in self.signal_bars]
                    atr = signals.compute_indicators(bars, cfg)['atr'][-1]
                    gain = side * (bars[-1].close / pos['entry'] - 1)
                    pos['best_close_return'] = max(pos.get('best_close_return', 0), gain)
                    if atr and pos['best_close_return'] >= cfg['trail_activation_underlying_return']:
                        lock = pos['entry'] * (1 + side * cfg['trail_locked_underlying_return'])
                        trail = bars[-1].close - side * cfg['trail_atr'] * atr
                        candidate = max(lock, trail) if side == 1 else min(lock, trail)
                        if side * (candidate - pos['stop']) > 0:
                            pos['stop'] = arithmetic.round_tick(candidate, tick, side == 1)
                    pos['last_trail_bar_ms'] = int(latest[6])
        state['signal_timeframe'] = self.signal_timeframe
        state['signal_family'] = cfg.get('signal_family', 'breakout')
        state['entry_direction'] = cfg.get('entry_direction', 'both')
        state['next_signal_time_ms'] = (timestamp // self.signal_interval_ms + 1) * self.signal_interval_ms
        state['signal_status'] = 'holding' if pos else 'waiting_for_new_closed_bar'
        state['entry_checks'] = None
        state['entry_blockers'] = []
        pending = state.get('pending_signal')
        if pending and timestamp >= pending['expires_at_ms']:
            state['pending_signal'] = pending = None
            state['signal_status'] = 'signal_expired'
        if not pos and self.signal_bars and clock and not errors.get('signal') and not errors.get('profit_atr'):
            boundary = int(self.signal_bars[-1][0]) + self.signal_interval_ms
            active_after = state.get('signal_active_after_ms', state['start_ms'])
            if (boundary >= active_after and boundary == timestamp // self.signal_interval_ms * self.signal_interval_ms
                    and state['last_signal_time_ms'] != boundary):
                bars = [arithmetic.candle(row) for row in self.signal_bars]
                indicators = signals.compute_indicators(bars, cfg)
                sig = stock_swing_profiles.signal_at(bars, indicators, len(bars) - 1, cfg)
                rejected = entry_policy.rejection(cfg, self.symbol, timestamp, sig.direction) if sig else None
                raw_signal = asdict(sig) if sig else None
                if rejected:
                    sig = None
                state['last_signal_time_ms'] = boundary
                state['last_signal_result'] = rejected or ('signal' if sig else 'none')
                state['pending_signal'] = pending = ({'signal': asdict(sig), 'boundary_ms': boundary,
                    'expires_at_ms': boundary + self.signal_interval_ms} if sig else None)
                state['signal_status'] = 'direction_filtered' if rejected else 'no_signal' if sig is None else 'signal_observed'
                journal(self.path.with_name('signals.jsonl'), {'time_utc': arithmetic.iso(timestamp),
                    'signal_timeframe': self.signal_timeframe, 'boundary_ms': boundary,
                    'pending_signal': pending, 'raw_signal': raw_signal, 'rejection': rejected,
                    'rule_candidate_id': cfg.get('candidate_id')})
            elif boundary < active_after:
                state['signal_status'] = 'waiting_for_first_forward_candle'
            elif state['last_signal_time_ms'] == boundary and state.get('last_signal_result') == 'none':
                state['signal_status'] = 'no_signal'
            elif state['last_signal_time_ms'] == boundary and state.get('last_signal_result') in ('research_direction', 'research_session_clock'):
                state['signal_status'] = 'direction_filtered'
        if not pos and pending and cfg.get('entry_enabled', True):
            state['signal_status'] = 'entry_data_unavailable'
            state['entry_blockers'] = list(errors) or ['entry_data_unavailable']
            if self.five and lot and fresh_book and not errors and clock:
                sig = signals.Signal(**pending['signal'])
                boundary = pending['boundary_ms']
                state['signal_status'] = 'risk_halted' if state['risk_halted'] or dd >= cfg['account_soft_drawdown_fraction'] else 'cooldown'
                state['entry_blockers'] = [state['signal_status']]
                if not state['risk_halted'] and dd < cfg['account_soft_drawdown_fraction'] and timestamp >= state['cooldown_until_ms']:
                    expected_bar = timestamp // FIVE_MINUTES * FIVE_MINUTES - FIVE_MINUTES
                    volume = float(self.five[-1][5]) if int(self.five[-1][0]) == expected_bar else 0
                    reference = ask if sig.direction == 1 else bid
                    checks = {
                        'prior_5m_volume': math.isfinite(volume) and volume > 0,
                        'funding': sig.direction * float(mark['lastFundingRate']) <= cfg['entry_max_adverse_funding_rate'],
                        'mark_index_basis': abs(price / index - 1) <= cfg['max_mark_index_basis_fraction'],
                        'book_mark_basis': abs(reference / price - 1) <= cfg['max_contract_mark_basis_fraction'],
                        'entry_gap': signals.entry_gap_allowed(sig, reference, cfg),
                    }
                    spread_fraction = 2 * (ask - bid) / (ask + bid)
                    if cfg.get('max_entry_spread_fraction') is not None:
                        checks['bid_ask_spread'] = spread_fraction <= cfg['max_entry_spread_fraction']
                    state['entry_checks'] = {'passed': checks, 'volume_bar_time_ms': int(self.five[-1][0]),
                        'expected_volume_bar_time_ms': expected_bar, 'prior_5m_volume': volume,
                        'checked_at_ms': timestamp, 'reference_price': reference,
                        'adverse_gap': sig.direction * (reference - sig.close),
                        'max_adverse_gap': cfg['entry_gap_atr'] * sig.atr,
                        'spread_fraction': spread_fraction,
                        'max_spread_fraction': cfg.get('max_entry_spread_fraction')}
                    state['entry_blockers'] = [key for key, passed in checks.items() if not passed]
                    if not state['entry_blockers']:
                        entry_estimate = reference * (1 + sig.direction * cfg['adverse_slippage_fraction_assumption'])
                        stop = signals.initial_stop(entry_estimate, sig.direction, sig.atr, cfg)
                        if stop:
                            stop = arithmetic.round_tick(stop, tick, sig.direction == -1)
                            qty = signals.position_size(equity, entry_estimate, stop, step,
                                cfg['risk_fraction_per_trade'], cfg['taker_fee_rate_assumption'], cfg['adverse_slippage_fraction_assumption'])
                            qty = min(qty, equity * cfg['portfolio_gross_notional_fraction'] / entry_estimate,
                                      volume * 0.1, float(lot['maxQty']), equity * 10 / entry_estimate)
                            result = book_fill(book, sig.direction * floor_qty(qty, step), step, cfg['adverse_slippage_fraction_assumption'])
                            if result['qty'] >= float(lot['minQty']) and result['qty'] * result['price'] >= minimum:
                                stop = signals.initial_stop(result['price'], sig.direction, sig.atr, cfg)
                                if stop:
                                    stop = arithmetic.round_tick(stop, tick, sig.direction == -1)
                                    actual_risk = result['qty'] * signals.per_unit_stop_risk(result['price'], stop,
                                        cfg['taker_fee_rate_assumption'], cfg['adverse_slippage_fraction_assumption'])
                                    candidate = {'entry': result['price'], 'qty': result['qty'], 'direction': sig.direction,
                                        'entry_time': now_ms(), 'signal_time': boundary, 'stop': stop,
                                        'initial_stop': stop,
                                        'entry_fee': result['qty'] * result['price'] * cfg['taker_fee_rate_assumption'],
                                        'margin': result['qty'] * result['price'] / 10, 'funding': 0.0, 'atr': sig.atr}
                                    liq = arithmetic.liquidation_price(candidate, cfg['maintenance_margin_fraction_assumption'])
                                    if (now_ms() < pending['expires_at_ms']
                                            and actual_risk <= equity * cfg['risk_fraction_per_trade'] + 1e-9
                                            and abs(stop / result['price'] - 1) <= cfg['max_stop_fraction'] + 1e-12
                                            and sig.direction * (stop - liq) / result['price'] >= cfg['minimum_liquidation_distance_buffer']):
                                        fresh(book.get('T') or book.get('E'), now_ms())
                                        fresh(mark['time'], now_ms())
                                        record = self.fill(state, result, now_ms(), sig.direction,
                                            'fresh_closed_' + self.signal_timeframe + '_signal')
                                        state['position'] = pos = candidate
                                        state['pending_signal'] = None
                                        state['position_history'].append({'time_ms': record['time_ms'], 'signed_qty': sig.direction * result['qty']})
                                        state['signal_status'] = 'entered' if not result['partial'] else 'entered_partial_ioc_remainder_cancelled'
                                    else:
                                        state['signal_status'] = 'risk_or_liquidation_buffer_blocked'
                                        if now_ms() >= pending['expires_at_ms']:
                                            state['pending_signal'] = None
                                            state['signal_status'] = 'signal_expired'
                                else:
                                    state['signal_status'] = 'stop_distance_blocked'
                            else:
                                state['signal_status'] = 'insufficient_observed_liquidity_or_minimum_quantity'
                        else:
                            state['signal_status'] = 'stop_distance_blocked'
                    else:
                        state['signal_status'] = 'entry_data_gap_basis_funding_or_volume_blocked'
        state['entry_qualification'] = cfg.get('entry_qualification')
        if not pos and not cfg.get('entry_enabled', True):
            state['signal_status'] = 'strategy_not_qualified'
            state['entry_blockers'] = ['strategy_not_qualified']
        state['equity'] = state['wallet_balance'] + (pos['direction'] * pos['qty'] * (price - pos['entry']) if pos else 0)
        state['return_pct'] = (state['equity'] / 1000 - 1) * 100
        state['position_qty'] = pos['direction'] * pos['qty'] if pos else 0
        state['last_mark_price'] = price
        state['updated_at_utc'] = arithmetic.iso(now_ms())
        state['status'] = 'degraded' if errors else 'healthy'
        state['errors'] = errors
        # Exact independent cash identity, including late funding of already closed inventory.
        expected = 1000 + state['realized_pnl'] - state['fees_paid'] + state['funding_pnl']
        if abs(expected - state['wallet_balance']) > 1e-7:
            raise RuntimeError('Stock ledger identity failed')
        observation = {'time_utc': state['updated_at_utc'], 'symbol': self.symbol, 'mark': mark,
            'book': book, 'errors': errors, 'equity': state['equity'], 'position_qty': state['position_qty'],
            'signal_status': state['signal_status'], 'funding_events': data.get('funding')}
        observation['signal_timeframe'] = self.signal_timeframe
        observation['last_signal_time_ms'] = state['last_signal_time_ms']
        observation['pending_signal'] = state.get('pending_signal')
        observation['entry_checks'] = state['entry_checks']
        observation['entry_blockers'] = state['entry_blockers']
        state['profit_exit_status'] = ({**pos.get('profit_exit', {}), 'stop': pos['stop']}
            if pos and profit_exits.enabled(cfg) else None)
        observation['profit_exit_status'] = state['profit_exit_status']
        if 'signal' in data:
            observation['signal_candles'] = self.signal_bars
        if 'five' in data and not errors.get('five'):
            observation['preceding_candles_5m'] = self.five
        journal(self.path.with_name('observations.jsonl'), observation)
        write_json(self.path, state)
        return state


def stock_config(plan, symbol):
    profile = json.loads((ROOT / plan['stock_research_profile']).read_text())
    base = json.loads((ROOT / profile['base_config']).read_text())
    common = {key: profile[key] for key in ('signal_timeframe', 'execution_timeframe', 'cooldown_signal_bars',
                                          'max_entry_spread_fraction', 'profit_exit_policy') if key in profile}
    cfg = {**base, **common, **profile['symbol_profiles'][symbol],
        'candidate_id': profile['symbol_profiles'][symbol].get('candidate_id', profile['candidate_id']),
        'risk_fraction_per_trade': profile['risk_fraction_per_bucket_trade'],
        'initial_equity_usdt': 1000, 'leverage': 10}
    SIGNAL_INTERVALS[cfg['signal_timeframe']]  # Validate before changing any account.
    profit_exits.validate(cfg)
    return cfg


def synchronize_stock_rules(plan, root, manifest):
    """Apply an authorized paper-rule revision under the runner lock, without resetting cash."""
    configs = {a['account_id']: stock_config(plan, a['symbol']) for a in plan['accounts']
               if a['account_id'] != 'btc'}
    activated = now_ms()
    revisions = []
    for account in plan['accounts']:
        account_id = account['account_id']
        if account_id == 'btc':
            continue
        path = ROOT / account['state_path']
        state = json.loads(path.read_text())
        cfg = configs[account_id]
        if state['rule'] == cfg:
            continue
        exit_only = plan.get('stock_rule_revision_scope') == 'profit_exits_only'
        if exit_only:
            ignored = {'profit_exit_policy', 'candidate_id'}
            if ({k: v for k, v in state['rule'].items() if k not in ignored}
                    != {k: v for k, v in cfg.items() if k not in ignored}):
                raise ValueError('Exit-only migration cannot alter entry or risk settings')
        write_json(root / 'rule_revisions' / str(activated) / account_id / 'previous_state.json', state)
        revision = {'activated_at_ms': activated, 'activated_at_utc': arithmetic.iso(activated),
            'previous_rule': state['rule'], 'new_rule': cfg,
            'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reason': plan.get('stock_rule_revision_reason', 'User authorized independent per-account strategies and shorter stock signal cycles; closed-bar signals with 30s entry retries.')}
        state.setdefault('rule_history', []).append(revision)
        state.update(rule=cfg, signal_timeframe=cfg['signal_timeframe'])
        if not exit_only:
            state.update(signal_active_after_ms=activated, last_signal_time_ms=None,
                last_signal_result=None, pending_signal=None, signal_status='waiting_for_first_forward_candle')
        write_json(path, state)
        journal(path.with_name('rule_revisions.jsonl'), revision)
        revisions.append({'account_id': account_id, **revision})
    if revisions:
        manifest.setdefault('rule_revisions', []).extend(revisions)
        write_json(root / 'manifest.json', manifest)


def prepare_profit_comparison(plan, root):
    """Clone cash/inventory before exit migration; the old-rule control has its own ledger."""
    comparison = plan.get('profit_exit_comparison')
    if not comparison:
        return None
    path = root / comparison['directory']
    manifest_path = path / 'manifest.json'
    stocks = [a for a in plan['accounts'] if a['account_id'] != 'btc']
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for account in stocks:
            if not (path / account['account_id'] / 'state.json').exists():
                raise RuntimeError('Existing profit comparison missing a control ledger')
        return path, manifest
    if path.exists():
        raise RuntimeError('Incomplete profit comparison; refusing to reset controls')
    path.mkdir(parents=True)
    timestamp = now_ms()
    manifest = {'activated_at_ms': timestamp, 'activated_at_utc': arithmetic.iso(timestamp),
        'places_orders': False, 'baseline': {}, 'control_rules': {},
        'comparison_basis': 'equity/wallet changes from identical activation inventory; same public responses',
        'existing_positions': 'new exit extrema begin at first post-activation executable quote'}
    for account in stocks:
        state = read_json(ROOT / account['state_path'])
        if profit_exits.enabled(state['rule']):
            raise RuntimeError('Control must be captured before profit-rule activation')
        account_id = account['account_id']
        write_json(path / account_id / 'state.json', state)
        manifest['baseline'][account_id] = {k: state.get(k) for k in (
            'equity', 'wallet_balance', 'realized_pnl', 'fees_paid', 'funding_pnl', 'fill_count_total', 'position_qty')}
        manifest['control_rules'][account_id] = state['rule']
    write_json(manifest_path, manifest)
    return path, manifest


def bootstrap(plan):
    root = ROOT / 'data/parallel_simulation' / plan['plan_id']
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for account in plan['accounts']:
            if not (ROOT / account['state_path']).exists():
                raise RuntimeError('Existing generation is missing an account; refusing to reset')
        return root, manifest
    start = now_ms()
    manifest = {'plan_id': plan['plan_id'], 'start_ms': start, 'start_utc': arithmetic.iso(start),
        'places_orders': False, 'initial_balance_each': 1000, 'total_initial_balance': 4000,
        'max_leverage': 10, 'stock_rule': 'prior5m_per_symbol_existing_both_directions',
        'stock_selection_reason': 'User authorized starting prospective research now; use existing declared baseline without refitting.',
        'independent_holdout_after_start_only': True, 'forward_validated': False,
        'btc_signal_source': 'existing selected BTC supervisor, forward targets only',
        'btc_execution_model': 'existing verified mark_price_slippage_model',
        'stock_execution_model': 'observed_depth_10pct_ioc_plus_2bps; 30s protective polling',
        'funding_mark_fallback': 'published mark else 1-minute mark open proxy explicitly journaled',
        'source_hashes': {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
            for name in ('run_parallel_simulation.py', 'stock_swing_signals.py', 'stock_swing_profiles.py', 'trading_execution.py')}}
    for account in plan['accounts']:
        path = ROOT / account['state_path']
        if path.exists():
            raise RuntimeError('Refusing to overwrite an existing paper account')
        if account['account_id'] == 'btc':
            SimulationAccount(path).reset(1000, now_ms=start)
        else:
            cfg = stock_config(plan, account['symbol'])
            write_json(path, initial_stock(account['symbol'], start, cfg))
    write_json(manifest_path, manifest)
    return root, manifest


def btc_step(account, root, manifest, client):
    path = ROOT / account['state_path']
    execution = SimulationAccount(path)
    clock = decision_runtime.resolve_clock(client)
    health = simulation_risk_monitor.monitor(client, clock['time_ms'], account=execution,
        status_path=path.with_name('risk_monitor.json'), clock_source=clock['source'])
    profile = json.loads((ROOT / account['strategy_path']).read_text())
    report_path = ROOT / 'data/paper_trading' / (profile['candidate_id'] + '_report.json')
    report = json.loads(report_path.read_text())
    if 'entry_qualification' in account:
        report['strategy_qualification'] = account['entry_qualification']
    point = report.get('execution_target') or (report.get('summary') or {}).get('last_equity_point') or {}
    state = execution.load()
    status = 'waiting_for_first_forward_candle'
    if int(point.get('available_time_ms') or point.get('time_ms') or 0) >= manifest['start_ms']:
        if state.get('last_signal_time_ms') != point.get('time_ms'):
            # Do not inherit a pre-generation position even after observing a flat target.
            origin = point.get('origin_signal_time_ms')
            if point.get('signed_qty') and (origin is None or int(origin) < manifest['start_ms']):
                report = json.loads(json.dumps(report))
                report['execution_target'] = {**point, 'signed_qty': 0, 'position_id': 'flat'}
            write_json(path.with_name('last_report.json'), report)
            result = execute_report('simulation', report, client, account=execution)
            journal(path.with_name('execution.jsonl'), {'time_utc': arithmetic.iso(now_ms()), 'result': result,
                'report_generated_at_utc': report.get('generated_at_utc')})
        status = 'healthy'
    if int(point.get('time_ms') or 0) < clock['time_ms'] - 900_000:
        status = 'degraded'
    state = execution.load()
    snapshot = execution.snapshot(state.get('last_mark_price'))
    if (health or {}).get('monitor', {}).get('status') != 'healthy':
        status = 'degraded'
    return {'symbol': 'BTCUSDT', 'status': status, 'checked_at_utc': arithmetic.iso(now_ms()),
            'equity': snapshot['account']['margin_balance'], 'wallet_balance': state['wallet_balance'],
            'return_pct': snapshot['account']['realized_return_pct'], 'position_qty': state['position_qty'],
            'fill_count_total': state['fill_count_total'], 'fees_paid': state['fees_paid'],
            'funding_pnl': state['funding_pnl'], 'max_drawdown_pct': state['max_drawdown_pct'],
            'last_mark_price': state.get('last_mark_price'), 'signal_time_ms': point.get('time_ms'),
            'signal_status': ('strategy_not_qualified' if account.get('entry_qualification', {}).get('approved_for_forward_simulation') is False else 'no_signal' if not point.get('signed_qty') else 'signal_observed'),
            'entry_qualification': account.get('entry_qualification'),
            'entry_blockers': ['strategy_not_qualified'] if account.get('entry_qualification', {}).get('approved_for_forward_simulation') is False else [],
            'errors': (health or {}).get('monitor', {}).get('errors', {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError('Poll interval must be at least five seconds')
    plan = json.loads(PLAN.read_text())
    if plan['execution_mode'] != 'simulation' or plan['live_orders_allowed'] is not False:
        raise ValueError('Only simulation plans are accepted')
    root = ROOT / 'data/parallel_simulation' / plan['plan_id']
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(root / 'runner.lock'):
        _run_locked(plan, root, args)


def _run_locked(plan, root, args):
    root, manifest = bootstrap(plan)
    comparison = prepare_profit_comparison(plan, root)
    synchronize_stock_rules(plan, root, manifest)
    status_path = root / 'status.json'
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    venue = PublicAster()
    client = BinanceTerminalClient()
    workers = {}
    controls = {}
    control_shared = {}
    shared = {}
    mutex = threading.Lock()
    def publish():
        with mutex:
            write_json(status_path, {'plan_id': plan['plan_id'], 'mode': 'SIMULATION', 'places_orders': False,
                'pid': os.getpid(), 'running': not STOP.is_set(), 'started_at_utc': manifest['start_utc'],
                'updated_at_utc': arithmetic.iso(now_ms()), 'poll_seconds': args.poll_seconds,
                'accounts': dict(shared), 'forward_validated': False,
                'profit_exit_comparison': ({'activated_at_utc': comparison[1]['activated_at_utc'],
                    'baseline': comparison[1]['baseline'], 'controls': dict(control_shared)} if comparison else None)})
    for account in plan['accounts']:
        if account['account_id'] != 'btc':
            state = json.loads((ROOT / account['state_path']).read_text())
            workers[account['account_id']] = StockAccount(ROOT / account['state_path'], venue,
                account['symbol'], state['rule'])
            if comparison:
                path = comparison[0] / account['account_id'] / 'state.json'
                controls[account['account_id']] = StockAccount(path, venue, account['symbol'], read_json(path)['rule'])
    def run(account):
        account_id = account['account_id']
        while not STOP.is_set():
            try:
                if account_id == 'btc':
                    result = btc_step(account, root, manifest, client)
                elif account_id in controls:
                    responses = workers[account_id].market_data(
                        force_rules=controls[account_id].rules is None,
                        force_signal=controls[account_id].candle_bucket != now_ms() // FIVE_MINUTES)
                    # Read once for both arms, including contract rules and closed candles.
                    try:
                        control = controls[account_id].step(responses)
                        control_view = {k: control.get(k) for k in ('equity', 'wallet_balance',
                            'realized_pnl', 'fees_paid', 'funding_pnl', 'position_qty', 'fill_count_total',
                            'status', 'checked_at_utc', 'errors')}
                    except Exception as exc:
                        control_view = {'status': 'degraded', 'checked_at_utc': arithmetic.iso(now_ms()),
                            'errors': {'worker': type(exc).__name__ + ': ' + str(exc)[:180]}}
                        journal(controls[account_id].path.with_name('errors.jsonl'), control_view)
                    with mutex:
                        control_shared[account_id] = control_view
                    result = workers[account_id].step(responses)
                else:
                    result = workers[account_id].step()
                view = {k: result.get(k) for k in ('symbol', 'status', 'checked_at_utc', 'equity', 'wallet_balance',
                    'return_pct', 'position_qty', 'fill_count_total', 'fees_paid', 'funding_pnl', 'max_drawdown_pct',
                    'last_mark_price', 'signal_status', 'signal_timeframe', 'signal_family', 'entry_direction', 'next_signal_time_ms',
                    'last_signal_time_ms', 'signal_active_after_ms', 'entry_checks', 'entry_blockers', 'entry_qualification', 'errors', 'observations',
                    'profit_exit_status', 'realized_pnl')}
                with mutex:
                    shared[account_id] = view
            except Exception as exc:
                prior = dict(shared.get(account_id, {}))
                ledger = json.loads((ROOT / account['state_path']).read_text())
                if not prior:
                    prior = {k: ledger.get(k) for k in ('wallet_balance', 'fill_count_total', 'fees_paid',
                        'funding_pnl', 'max_drawdown_pct', 'position_qty', 'last_mark_price')}
                    prior['equity'] = ledger.get('equity')
                    if prior['equity'] is None and not ledger.get('position_qty'):
                        prior['equity'] = ledger['wallet_balance']
                    prior['return_pct'] = (prior['equity'] / 1000 - 1) * 100 if prior['equity'] is not None else None
                with mutex:
                    shared[account_id] = {**prior, 'symbol': account['symbol'], 'status': 'degraded',
                        'checked_at_utc': arithmetic.iso(now_ms()), 'errors': {'worker': type(exc).__name__ + ': ' + str(exc)[:180]}}
                journal(root / account_id / 'errors.jsonl', shared[account_id])
            publish()
            if args.once:
                break
            STOP.wait(args.poll_seconds)
    publish()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, account) for account in plan['accounts']]
        for future in futures:
            future.result()
    STOP.set()
    publish()


if __name__ == '__main__':
    main()
