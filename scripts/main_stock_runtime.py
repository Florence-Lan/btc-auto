"""Main-account-only signal dispatch and conservative shared-clock reuse.

No account migration, global monkeypatch, network client, or alternate execution
engine. All non-clock market reads and all execution/exit gates remain inherited.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from types import FunctionType
import zlib
from collections.abc import Mapping

import run_parallel_simulation as paper
import stock_short_confirmation as short
import stock_swing_profiles as profiles
import trade_profit_budget as profit_budget
import main_stock_trend as main_trend
import main_stock_entry_trend as entry_trend
from main_stock_sector import SectorContext
import main_stock_execution_guard as execution_guard
import main_trend_pullback_entry as pullback

SHARED_ROOT = paper.ROOT / 'data/paper_trading/stock_entry_timing_trial_20261029'
MAX_BLOCK_BYTES = 8_000_000
MAX_CLOCK_AGE_SECONDS = 15
HOUR = 3_600_000
_BASE_MARKET_DATA = paper.StockAccount.market_data
_BASE_STEP = paper.StockAccount.step
_BASE_WRITE = paper.write_json


def _block(directory, identity):
    if (not isinstance(identity, str) or len(identity) != 64
            or any(c not in '0123456789abcdef' for c in identity)):
        raise ValueError('main_shared_clock_identity_invalid')
    path = Path(directory) / (identity + '.json.gz')
    if path.stat().st_size > MAX_BLOCK_BYTES:
        raise ValueError('main_shared_clock_block_oversized')
    try:
        with gzip.open(path, 'rb') as handle:
            raw = handle.read(MAX_BLOCK_BYTES + 1)
    except (EOFError, zlib.error) as exc:
        raise ValueError('main_shared_clock_block_compression_invalid') from exc
    if len(raw) > MAX_BLOCK_BYTES or hashlib.sha256(raw).hexdigest() != identity:
        raise ValueError('main_shared_clock_block_invalid')
    return json.loads(raw)


def validate_shared_clock(payload, *, monotonic_now=None):
    """Require original receipt metadata; never certify a cache at read time."""
    current = time.monotonic() if monotonic_now is None else monotonic_now
    if not isinstance(payload, dict) or not isinstance(payload.get('_public_clock_timing'), dict):
        raise ValueError('main_shared_clock_original_timing_required')
    received = payload.get('_received_monotonic')
    timing = payload['_public_clock_timing']
    if (isinstance(received, bool) or not isinstance(received, (int, float))
            or not math.isfinite(received) or received != timing.get('received_monotonic')):
        raise ValueError('main_shared_clock_receipt_invalid')
    paper.verified_public_clock(payload, timing.get('started_monotonic'), current)
    if not 0 <= current - received <= MAX_CLOCK_AGE_SECONDS:
        raise ValueError('main_shared_clock_receipt_stale')
    return deepcopy(payload)


def load_shared_clock(symbol, *, source_root=None, wall_ms=None, monotonic_now=None):
    """Read only the clock of a bounded, hash-verified existing trial context."""
    if symbol not in ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT'):
        raise ValueError('main_shared_clock_symbol_invalid')
    root = SHARED_ROOT if source_root is None else Path(source_root)
    path = root / symbol / 'baseline_normal/observations.jsonl'
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        lines = handle.read(65536).split(b'\n')[:-1]
    row = None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get('shared_context_sha256'), str):
            row = value
            break
    if row is None:
        raise ValueError('main_shared_clock_context_missing')
    identity = row['shared_context_sha256']
    context = _block(root / 'context_archive' / symbol, identity)
    stamp = datetime.fromisoformat(context['received_at_utc'])
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('main_shared_clock_context_time_invalid')
    current_wall = int(time.time() * 1000) if wall_ms is None else wall_ms
    age = current_wall - int(stamp.timestamp() * 1000)
    if context.get('schema_version') != 1 or context.get('symbol') != symbol or not 0 <= age <= 60_000:
        raise ValueError('main_shared_clock_context_stale')
    labels = set()
    clock = None
    for item in context['responses']:
        label = item['label']
        if label not in {'clock', 'mark', 'book', 'funding', 'five', 'rules', 'signal'} or label in labels:
            raise ValueError('main_shared_clock_labels_invalid')
        labels.add(label)
        if label == 'clock':
            if item.get('error') or not item.get('payload_sha256'):
                raise ValueError('main_shared_clock_source_failed')
            clock = _block(root / 'context_blocks', item['payload_sha256'])
    clock = validate_shared_clock(clock, monotonic_now=monotonic_now)
    return clock, {'source': 'shared_verified', 'context_sha256': identity,
                   'received_monotonic': clock['_received_monotonic']}


class _SignalProfiles:
    def __init__(self, active_after_ms, main_active_after_ms=None, entry_active_after_ms=None):
        self.active_after_ms = active_after_ms
        self.main_active_after_ms = main_active_after_ms
        self.entry_active_after_ms = entry_active_after_ms

    def __getattr__(self, name):
        return getattr(profiles, name)

    def signal_with_diagnostics(self, candles, indicators, index, config=None):
        config = {} if config is None else config
        boundary = candles[index].time_ms + HOUR
        active = short.enabled(config) and boundary > self.active_after_ms
        main_active = (main_trend.enabled(config) and self.main_active_after_ms is not None
                       and boundary > self.main_active_after_ms)
        entry_active = (entry_trend.enabled(config) and self.entry_active_after_ms is not None
                        and boundary > self.entry_active_after_ms)
        function = (entry_trend.signal_with_diagnostics if entry_active else
                    main_trend.signal_with_diagnostics if main_active else
                    short.signal_with_diagnostics if active else profiles.signal_with_diagnostics)
        signal, diagnostic = function(candles, indicators, index, config)
        diagnostic = deepcopy(diagnostic)
        if main_trend.enabled(config) and not main_active:
            signal = None
            diagnostic['signal_after_confirmation'] = None
            diagnostic['signal_after_quality'] = None
            diagnostic['admission'].update(stage='main_trend_not_active',
                reasons=['main_trend_awaiting_new_signal'])
        diagnostic['main_short_confirmation_runtime'] = {
            'enabled': short.enabled(config), 'candidate_active_for_bar': active,
            'active_after_ms': self.active_after_ms, 'bar_close_ms': boundary,
            'mode': short.POLICY['mode'] if short.enabled(config) else None}
        if main_trend.enabled(config):
            diagnostic['main_trend_confirmation_runtime'] = {
                'enabled': True, 'candidate_active_for_bar': main_active,
                'active_after_ms': self.main_active_after_ms, 'bar_close_ms': boundary}
        if entry_trend.enabled(config):
            diagnostic['entry_trend_confirmation_runtime'] = {
                'enabled': True, 'candidate_active_for_bar': entry_active,
                'active_after_ms': self.entry_active_after_ms, 'bar_close_ms': boundary}
            if not entry_active:
                signal = None
                diagnostic['signal_after_confirmation'] = None
                diagnostic['signal_after_quality'] = None
                diagnostic['admission'].update(stage='entry_trend_not_active',
                    reasons=['entry_trend_awaiting_new_signal'])
        return signal, diagnostic

    def signal_at(self, *args, **kwargs):
        return self.signal_with_diagnostics(*args, **kwargs)[0]


class _ClockVenue:
    def __init__(self, account):
        self.account = account

    def _shared(self):
        payload, detail = self.account.shared_clock_loader(self.account.symbol)
        payload = validate_shared_clock(payload)
        self.account._main_clock_diagnostic = deepcopy(detail)
        return payload

    def get(self, endpoint, params=None):
        if endpoint != 'time' or params:
            return self.account.venue.get(endpoint, params)
        try:
            return self._shared()
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            pass
        started = time.monotonic()
        try:
            payload = self.account.venue.get(endpoint, params)
            paper.verified_public_clock(payload, started, time.monotonic())
        except Exception as original:
            try:
                payload = self._shared()
            except (OSError, ValueError, TypeError, KeyError, OverflowError):
                self.account._main_clock_diagnostic = {'source': 'unavailable', 'error_type': type(original).__name__}
                raise original
            self.account._main_clock_diagnostic['fallback_after_error_type'] = type(original).__name__
            return payload
        self.account._main_clock_diagnostic = {'source': 'venue_verified'}
        return payload


class _AccountView:
    def __init__(self, account):
        self._account = account
        self.venue = _ClockVenue(account)

    def __getattr__(self, name):
        return getattr(self._account, name)


class _EntryTiming:
    def __getattr__(self, name):
        return getattr(paper.stock_entry_timing, name)

    def evaluate(self, trigger, signal, config, **prices):
        if pullback.enabled(config):
            return pullback.evaluate(trigger, signal, config, **prices)
        return paper.stock_entry_timing.evaluate(trigger, signal, config, **prices)


class MainStockAccount(paper.StockAccount):
    """Use the existing main ledger with a private, post-activation dispatcher."""
    shared_clock_loader = staticmethod(load_shared_clock)

    def __init__(self, path, venue, symbol, config):
        short.validate(config)
        main_trend.validate(config)
        entry_trend.validate(config)
        execution_guard.validate(config)
        pullback.validate(config)
        super().__init__(path, venue, symbol, config)
        state = paper.read_json(self.path)
        self.pullback_entry_active_after_ms = state.get('pullback_entry_active_after_ms')
        if pullback.enabled(config) and (type(self.pullback_entry_active_after_ms) is not int
                or self.pullback_entry_active_after_ms <= state.get('execution_guard_active_after_ms', 0)):
            raise ValueError('pullback_activation_required')
        self.short_confirmation_active_after_ms = state.get('short_confirmation_active_after_ms')
        if short.enabled(config) and (type(self.short_confirmation_active_after_ms) is not int
                or self.short_confirmation_active_after_ms <= 0):
            raise ValueError('main_short_confirmation_activation_required')
        self._main_clock_diagnostic = {'source': 'not_observed'}
        self._declared_config = deepcopy(config)
        self._budget_decision = None
        self._guard_market = {}
        self._guard_decision = None
        self.execution_guard_active_after_ms = state.get('execution_guard_active_after_ms')
        if execution_guard.enabled(config) and (type(self.execution_guard_active_after_ms) is not int
                or self.execution_guard_active_after_ms <= state.get('entry_trend_active_after_ms', 0)):
            raise ValueError('main_execution_guard_activation_required')
        self._legacy_position_mode = False
        self.main_trend_active_after_ms = state.get('main_trend_active_after_ms')
        if main_trend.enabled(config) and (type(self.main_trend_active_after_ms) is not int
                or self.main_trend_active_after_ms <= 0):
            raise ValueError('main_trend_activation_required')
        self.entry_trend_active_after_ms = state.get('entry_trend_active_after_ms')
        if entry_trend.enabled(config) and (type(self.entry_trend_active_after_ms) is not int
                or self.entry_trend_active_after_ms <= (self.main_trend_active_after_ms or 0)):
            raise ValueError('entry_trend_activation_required')
        self._sector_context = None
        self._sector_blocked = None
        self._entry_trend_blocked = None
        self.trade_profit_budget_active_after_ms = state.get('trade_profit_budget_active_after_ms')
        if profit_budget.enabled(config):
            profit_budget.validate(config)
            if (type(self.trade_profit_budget_active_after_ms) is not int
                    or self.trade_profit_budget_active_after_ms <= 0):
                raise ValueError('main_profit_budget_activation_required')

    def sector_loader(self, timestamp, *, phase='observation'):
        if self._sector_context is None:
            self._sector_context = SectorContext(
                self.path.with_name('main_sector_quotes.jsonl'), self.main_trend_active_after_ms)
        return self._sector_context.decision(timestamp)

    def _record_pullback(self, state, setup, timestamp):
        previous = state.get('trend_pullback_setup') or {}
        if setup and (previous.get('id'), previous.get('status'), previous.get('last_processed_open_ms')) != (
                setup.get('id'), setup.get('status'), setup.get('last_processed_open_ms')):
            paper.runner_journal(self.path.with_name('trend_pullback_decisions.jsonl'),
                {'checked_at_ms': timestamp, 'active_after_ms': self.pullback_entry_active_after_ms,
                 'setup': deepcopy(setup), 'places_orders': False})
        state['trend_pullback_setup'] = setup

    def prepare_pending_entry(self, state, pending, data, errors, timestamp):
        if not pullback.enabled(self.config):
            return super().prepare_pending_entry(state, pending, data, errors, timestamp)
        setup = deepcopy(state.get('trend_pullback_setup'))
        try:
            if errors.get('five') or errors.get('signal') or errors.get('clock') or not data.get('clock'):
                state.update(signal_status='trend_pullback_data_unavailable',
                             entry_blockers=['pullback_completed_data_required'])
                return None
            if not self.five or int(self.five[-1][6])+1 != timestamp//pullback.FIVE*pullback.FIVE:
                state.update(signal_status='trend_pullback_data_unavailable',
                             entry_blockers=['pullback_latest_closed_five_required'])
                return None
            event = state.get('stock_event_risk') or {}
            if event.get('enabled') and not event.get('allowed'):
                if setup and setup['status'] not in pullback.TERMINAL:
                    setup = pullback.cancel(setup, 'pullback_event_gate_blocked')
                    self._record_pullback(state, setup, timestamp)
                return None
            if setup and timestamp >= setup['expires_at_ms'] and setup['status'] not in pullback.TERMINAL:
                setup = pullback.cancel(setup, 'pullback_setup_expired', 'expired')
            incoming = pending if (pending or {}).get('entry_trigger', {}).get('family') == 'donchian20' else None
            if incoming and (not setup or (incoming['boundary_ms'] > setup['boundary_ms'] and (
                    setup['status'] in pullback.TERMINAL or incoming['signal']['direction'] != setup['direction']))):
                if setup and setup['status'] not in pullback.TERMINAL:
                    self._record_pullback(state, pullback.cancel(setup, 'pullback_opposite_signal'), timestamp)
                setup = pullback.seed(incoming, timestamp=timestamp,
                                      active_after=self.pullback_entry_active_after_ms)
            if setup:
                setup = pullback.advance(setup, self.five, timestamp=timestamp)
                self._record_pullback(state, setup, timestamp)
                state['signal_status'] = 'trend_pullback_' + setup['status']
                state['entry_blockers'] = list(setup.get('reasons', []))
                return pullback.pending_entry(setup)
            state.update(signal_status='trend_pullback_waiting_signal', entry_blockers=['pullback_new_hourly_signal_required'])
            return None
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            if setup:
                self._record_pullback(state, pullback.cancel(setup,
                    str(exc) if isinstance(exc, ValueError) else 'pullback_data_invalid'), timestamp)
            state.update(signal_status='trend_pullback_data_unavailable', entry_blockers=['pullback_data_invalid'])
            return None

    def initial_entry_stop(self, state, signal, entry):
        if not pullback.enabled(self.config):
            return super().initial_entry_stop(state, signal, entry)
        setup = state.get('trend_pullback_setup') or {}
        try:
            if setup.get('signal') != vars(signal):
                return None
            return pullback.initial_stop(setup, entry, self.config)
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    def record_entry_timing(self, state, pending, decision, phase):
        result = super().record_entry_timing(state, pending, decision, phase)
        if (pullback.enabled(self.config) and decision.get('cancel_pending')
                and pending.get('entry_trigger', {}).get('family') == 'trend_pullback'):
            setup = deepcopy(state.get('trend_pullback_setup'))
            if setup:
                self._record_pullback(state, pullback.cancel(setup,
                    (decision.get('reasons') or ['pullback_cancelled'])[0]), self.now_ms())
        return result

    def fill(self, state, result, timestamp, direction, reason, trigger_ms=None):
        if pullback.enabled(self.config) and reason.startswith('fresh_closed_'):
            setup = state.get('trend_pullback_setup') or {}
            result = {**result, 'entry_method': pullback.POLICY['mode'],
                      'setup_id': setup.get('id'), 'origin_signal_boundary_ms': setup.get('boundary_ms')}
        record = super().fill(state, result, timestamp, direction, reason, trigger_ms)
        if pullback.enabled(self.config) and reason.startswith('fresh_closed_'):
            setup = deepcopy(state.get('trend_pullback_setup'))
            if setup:
                setup.update(status='consumed', filled_at_ms=timestamp, reasons=[])
                self._record_pullback(state, setup, timestamp)
        return record

    def _sector_decision(self, state, direction=None, *, phase='observation', timestamp=None):
        if not main_trend.enabled(self._declared_config):
            return {'enabled': False, 'allowed': True, 'reasons': []}
        timestamp = self.now_ms() if timestamp is None else timestamp
        try:
            outcome = deepcopy(self.sector_loader(timestamp, phase=phase))
            if not isinstance(outcome, Mapping):
                raise ValueError('main_sector_result_invalid')
            expected = {'bullish': 1, 'bearish': -1, 'neutral': 0}
            if (outcome.get('status') not in (*expected, 'unknown')
                    or (outcome.get('status') != 'unknown'
                        and (type(outcome.get('direction')) is not int
                             or outcome['direction'] != expected[outcome['status']]))) :
                raise ValueError('main_sector_result_invalid')
            if (outcome.get('status') != 'unknown' and outcome.get('valid_until_ms') is not None
                    and (type(outcome['valid_until_ms']) is not int or timestamp >= outcome['valid_until_ms'])):
                raise ValueError('main_sector_result_expired')
            if (outcome.get('reasons') is not None
                    and (not isinstance(outcome['reasons'], list)
                         or any(not isinstance(reason, str) for reason in outcome['reasons']))):
                raise ValueError('main_sector_reasons_invalid')
        except Exception:
            # Supplemental context must never interrupt protective accounting.
            outcome = {'status': 'unknown', 'direction': None,
                       'reasons': ['main_sector_data_unavailable']}
        reasons = list(outcome.get('reasons') or [])
        allowed = outcome['status'] != 'unknown'
        if not allowed:
            reasons = list(dict.fromkeys(['main_sector_unknown'] + reasons))
        elif direction in (-1, 1) and outcome['direction'] == -direction:
            allowed = False
            reasons = ['main_sector_opposes_entry']
        outcome.update(enabled=True, allowed=allowed, reasons=reasons,
                       candidate_direction=direction, phase=phase, checked_at_ms=timestamp,
                       protective_exits_allowed=True)
        state['main_sector_status'] = outcome
        if direction in (-1, 1) and not allowed:
            self._sector_blocked = deepcopy(outcome)
        return outcome

    def record_stock_event_risk(self, state, phase, reference_ms=None):
        original = super().record_stock_event_risk(state, phase, reference_ms)
        pending = state.get('pending_signal') or {}
        direction = (pending.get('signal') or {}).get('direction')
        if phase == 'fill' and direction in (-1, 1):
            entry = self._entry_trend_decision(state, pending.get('boundary_ms', 0), direction,
                                               phase=phase, timestamp=reference_ms)
            if not entry['allowed']:
                return {**original, 'allowed': False,
                        'reasons': list(dict.fromkeys(original.get('reasons', []) + entry['reasons']))}
        if phase == 'fill' and direction in (-1, 1) and main_trend.enabled(self._declared_config):
            if pending.get('boundary_ms', 0) <= self.main_trend_active_after_ms:
                return {**original, 'allowed': False, 'reasons': ['main_trend_awaiting_new_signal']}
            sector = self._sector_decision(state, direction, phase='fill', timestamp=reference_ms)
            if not sector['allowed']:
                return {**original, 'allowed': False,
                        'reasons': list(dict.fromkeys(original.get('reasons', []) + sector['reasons']))}
        return original

    def _entry_trend_decision(self, state, boundary, direction, *, phase, timestamp=None):
        if not entry_trend.enabled(self._declared_config):
            return {'enabled': False, 'allowed': True, 'reasons': []}
        timestamp = self.now_ms() if timestamp is None else timestamp
        if type(boundary) is not int or boundary <= self.entry_trend_active_after_ms:
            outcome = {'enabled': True, 'allowed': False,
                       'reasons': ['entry_trend_awaiting_new_signal']}
        else:
            try:
                bars = [paper.arithmetic.candle(row) for row in self.signal_bars]
                if not bars or bars[-1].time_ms + HOUR != timestamp // HOUR * HOUR:
                    raise ValueError('entry_trend_latest_closed_hour_missing')
                outcome = entry_trend.evaluate_confirmation(bars, len(bars) - 1,
                                                            self._declared_config, direction)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
                outcome = {'enabled': True, 'allowed': False,
                           'reasons': ['entry_trend_data_unavailable']}
        outcome.update(phase=phase, checked_at_ms=timestamp, signal_boundary_ms=boundary,
                       active_after_ms=self.entry_trend_active_after_ms,
                       protective_exits_allowed=True)
        state['entry_trend_execution'] = deepcopy(outcome)
        if not outcome['allowed']:
            self._entry_trend_blocked = deepcopy(outcome)
        return outcome

    def market_data(self, force_rules=False, force_signal=False):
        self._main_clock_diagnostic = {'source': 'not_observed'}
        responses = _BASE_MARKET_DATA(_AccountView(self), force_rules=force_rules, force_signal=force_signal)
        self._guard_market = {label: (value, error) for label, value, error in responses}
        return responses

    def _plan_budget(self, state, equity, sig, entry, stop, step, lot, volume, minimum, actual_qty=None):
        decision = profit_budget.stock_plan(self.config, equity=equity, entry=entry,
            stop=stop, step=step, direction=sig.direction,
            maximum_qty=float(lot['maxQty']), minimum_qty=float(lot['minQty']),
            minimum_notional=minimum, liquidity_qty=volume * .1, actual_qty=actual_qty)
        decision.update(checked_at_ms=self.now_ms(), signal_time_ms=sig.time_ms,
            signal_boundary_ms=sig.time_ms + self.signal_interval_ms)
        if execution_guard.enabled(self.config) and decision.get('allowed'):
            guard = self._entry_execution_guard(signal=sig, decision=decision, step=step)
            decision['execution_guard'] = guard
            if not guard['allowed']:
                decision.update(allowed=False, admitted_quantity=0., status='blocked',
                    reasons=list(dict.fromkeys(decision['reasons'] + guard['reasons'])))
        self._budget_decision = decision
        state['entry_profit_budget'] = deepcopy(decision)
        return decision

    def _record_execution_guard(self, state, decision, phase):
        result = {**decision, 'phase': phase, 'checked_at_ms': self.now_ms(),
                  'active_after_ms': self.execution_guard_active_after_ms}
        previous = (state or {}).get('execution_guard', {}).get('checked_at_ms')
        if type(previous) is int:
            result['interval_since_previous_guard_ms'] = result['checked_at_ms'] - previous
        self._guard_decision = result
        if state is not None:
            state['execution_guard'] = deepcopy(result)
        paper.runner_journal(self.path.with_name('execution_guard_decisions.jsonl'), result)
        return result

    def _entry_execution_guard(self, *, signal, decision, step):
        try:
            if signal.time_ms + self.signal_interval_ms <= self.execution_guard_active_after_ms:
                raise ValueError('main_execution_guard_awaiting_new_signal')
            mark, mark_error = self._guard_market['mark']
            book, book_error = self._guard_market['book']
            _, clock_error = self._guard_market['clock']
            if mark_error or book_error or clock_error:
                raise ValueError('main_execution_guard_market_unavailable')
            paper.fresh(mark['time'], self.now_ms())
            paper.fresh(book.get('T') or book.get('E'), self.now_ms())
            close = paper.book_fill(book, -signal.direction * decision['quantity'], step,
                                    self.config['adverse_slippage_fraction_assumption'])
            result = execution_guard.entry_decision(self.config, decision,
                direction=signal.direction, mark_price=float(mark['markPrice']), exit_fill=close)
            result.update(signal_boundary_ms=signal.time_ms+self.signal_interval_ms,
                          mark_time_ms=mark['time'], book_time_ms=book.get('T') or book.get('E'))
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            result = {'enabled': True, 'allowed': False, 'status': 'unavailable',
                      'reasons': [str(exc) if isinstance(exc, ValueError) else 'main_execution_guard_market_unavailable']}
        return self._record_execution_guard(None, result, 'entry')

    def entry_quantity(self, state, equity, signal, entry, stop, step, lot, volume, minimum):
        if self._legacy_position_mode:
            self._budget_decision = {'enabled': True, 'allowed': False,
                'reasons': ['legacy_position_closed_wait_for_next_observation']}
            return 0.
        if not self._entry_trend_decision(state, signal.time_ms + self.signal_interval_ms,
                                          signal.direction, phase='retry')['allowed']:
            return 0.
        if (main_trend.enabled(self._declared_config)
                and signal.time_ms + self.signal_interval_ms <= self.main_trend_active_after_ms):
            return 0.
        sector = self._sector_decision(state, signal.direction, phase='retry')
        if not sector['allowed']:
            return 0.
        if not profit_budget.enabled(self.config):
            return super().entry_quantity(state, equity, signal, entry, stop, step, lot, volume, minimum)
        if signal.time_ms + self.signal_interval_ms <= self.trade_profit_budget_active_after_ms:
            self._budget_decision = {'enabled': True, 'allowed': False,
                'reasons': ['profit_budget_pre_activation_signal']}
            return 0.
        decision = self._plan_budget(state, equity, signal, entry, stop, step, lot, volume, minimum)
        return decision['admitted_quantity']

    def entry_fill_admission(self, state, equity, signal, candidate, result, step, lot, volume, minimum):
        entry = self._entry_trend_decision(state, signal.time_ms + self.signal_interval_ms,
                                           signal.direction, phase='fill_admission')
        if not entry['allowed']:
            return entry
        if (main_trend.enabled(self._declared_config)
                and signal.time_ms + self.signal_interval_ms <= self.main_trend_active_after_ms):
            return {'allowed': False, 'reasons': ['main_trend_awaiting_new_signal']}
        sector = self._sector_decision(state, signal.direction, phase='fill_admission')
        if not sector['allowed']:
            return sector
        if not profit_budget.enabled(self.config):
            return super().entry_fill_admission(state, equity, signal, candidate, result, step, lot, volume, minimum)
        decision = self._plan_budget(state, equity, signal, result['price'], candidate['stop'],
            step, lot, volume, minimum, actual_qty=result['qty'])
        if decision['allowed']:
            candidate['trade_profit_budget'] = {**deepcopy(decision),
                'activated_at_ms': self.trade_profit_budget_active_after_ms,
                'entry_rule': deepcopy(self.config)}
            if pullback.enabled(self.config):
                candidate['trend_pullback_entry'] = deepcopy(state.get('trend_pullback_setup'))
        return decision

    def allow_profit_target(self, state, position, data, errors, timestamp):
        if not position.get('trade_profit_budget'):
            return super().allow_profit_target(state, position, data, errors, timestamp)
        due = state.get('next_funding_ms')
        return bool(data.get('clock') and data.get('funding') is not None
            and not errors.get('funding')
            and not (due and due <= self.now_ms() and state.get('funding_through_ms', 0) < due))

    def additional_exit_reason(self, state, position, data, errors, timestamp):
        original = super().additional_exit_reason(state, position, data, errors, timestamp)
        if original:
            return original
        guarded = bool(position.get('trade_profit_budget', {}).get('execution_guard', {}).get('allowed'))
        if guarded:
            reason = self._execution_guard_exit(state, position, data, errors)
            if reason:
                return reason
        if main_trend.enabled(self._declared_config):
            if data.get('clock') and not errors.get('clock') and not errors.get('signal'):
                bars = [paper.arithmetic.candle(row) for row in self.signal_bars]
                outcome = main_trend.evaluate_position_exit(position, bars, self.config,
                    timestamp, self.main_trend_active_after_ms)
            else:
                outcome = {'enabled': True, 'status': 'unavailable', 'triggered': False,
                           'reasons': ['main_trend_current_clock_or_candles_unavailable'],
                           'checked_at_ms': timestamp}
            state['main_trend_exit_status'] = outcome
            if outcome['triggered']:
                if not position.get('pending_exit'):
                    paper.journal(self.path.with_name('main_trend_decisions.jsonl'), outcome)
                return 'main_trend_invalidation'
            self._observe_profit_protection(state, position, data, errors, timestamp)
            if guarded:
                reason = self._execution_guard_exit(state, position, data, errors)
                if reason:
                    return reason
        if not position.get('trade_profit_budget'):
            return None
        try:
            side = position['direction']
            book = data.get('book') or {}
            price = float(book['bids' if side > 0 else 'asks'][0][0])
            paper.fresh(book.get('T') or book.get('E'), self.now_ms())
            exit_price = price * (1 - side * self.config['adverse_slippage_fraction_assumption'])
            net = side * position['qty'] * (exit_price - position['entry'])
            net -= (position['entry_fee'] + position['funding']
                    + position['qty'] * exit_price * self.config['taker_fee_rate_assumption'])
            cap = position['trade_profit_budget']['loss_budget_usdt']
            if net <= -cap:
                return 'trade_profit_budget_loss_limit'
        except (ValueError, KeyError, TypeError, IndexError):
            pass  # Original independent mark/stop protection still takes priority.
        return None

    def _execution_guard_exit(self, state, position, data, errors):
        """Use full observed liquidation depth; old stops survive unavailable data."""
        try:
            if errors.get('book') or errors.get('mark'):
                raise ValueError('main_execution_guard_market_unavailable')
            book = data['book']
            paper.fresh(book.get('T') or book.get('E'), self.now_ms())
            paper.fresh(data['mark']['time'], self.now_ms())
            filters = {item['filterType']: item for item in self.rules['filters']}
            lot = filters.get('MARKET_LOT_SIZE') or filters['LOT_SIZE']
            close = paper.book_fill(book, -position['direction'] * position['qty'], float(lot['stepSize']),
                                   self.config['adverse_slippage_fraction_assumption'])
            net = execution_guard.net_close(self.config, position, close)
            cap = position['trade_profit_budget']['execution_guard']['reserved_loss_budget_usdt']
            detail = position.get('main_trend_profit_protection') or {}
            reason = 'main_execution_loss_reserve' if net <= -cap else None
            floor = detail.get('net_floor_usdt') if detail.get('armed') else None
            if reason is None and floor is not None and net <= floor:
                reason = 'main_execution_profit_lock'
            self._record_execution_guard(state, {'enabled': True, 'status': 'evaluated',
                'entry_time_ms': position['entry_time'], 'net_full_close_usdt': net,
                'reserved_loss_budget_usdt': cap, 'net_floor_usdt': floor,
                'triggered': bool(reason), 'reason': reason,
                'book_time_ms': book.get('T') or book.get('E'), 'mark_time_ms': data['mark']['time'],
                'quantity': close['qty'], 'exit_price': close['price']}, 'holding')
            return reason
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            self._record_execution_guard(state, {'enabled': True, 'status': 'unavailable',
                'entry_time_ms': position['entry_time'], 'triggered': False,
                'reason': str(exc) if isinstance(exc, ValueError) else 'main_execution_guard_market_unavailable',
                'original_protective_exits_preserved': True}, 'holding')
            return None

    def _observe_profit_protection(self, state, position, data, errors, timestamp):
        """Tighten only from new executable observations, after old-stop checks."""
        if (not data.get('clock') or errors.get('clock') or errors.get('book')
                or timestamp < self.main_trend_active_after_ms or position.get('pending_exit')):
            return
        try:
            book = data['book']
            observed = int(book.get('T') or book.get('E'))
            if observed < self.main_trend_active_after_ms:
                return
            paper.fresh(observed, self.now_ms())
            paper.fresh(data['mark']['time'], self.now_ms())
            filters = {item['filterType']: item for item in self.rules['filters']}
            lot = filters.get('MARKET_LOT_SIZE') or filters['LOT_SIZE']
            side = position['direction']
            fill = paper.book_fill(book, -side * position['qty'], float(lot['stepSize']),
                                  self.config['adverse_slippage_fraction_assumption'])
            if fill['qty'] + 1e-12 < position['qty']:
                return
            gain = side * (fill['price'] / position['entry'] - 1)
            detail = position.setdefault('main_trend_profit_protection', {
                'active_after_ms': self.main_trend_active_after_ms, 'best_executable_return': 0.,
                'armed': False})
            detail['best_executable_return'] = max(detail['best_executable_return'], gain)
            detail.update(checked_at_ms=self.now_ms(), observed_at_ms=observed,
                          executable_close_price=fill['price'])
            if detail['best_executable_return'] >= self.config['trail_activation_underlying_return']:
                tick = float(filters['PRICE_FILTER']['tickSize'])
                lock = position['entry'] * (1 + side * self.config['trail_locked_underlying_return'])
                guarded = bool(position.get('trade_profit_budget', {}).get('execution_guard', {}).get('allowed'))
                if guarded:
                    lock_plan = execution_guard.lock_quote(self.config, position)
                    lock = lock_plan['quote']
                    detail['net_floor_usdt'] = lock_plan['net_floor_usdt']
                    detail['known_costs_included'] = True
                candidate = paper.arithmetic.round_tick(lock, tick, side == 1)
                previous_stop = position['stop']
                if side * (candidate - position['stop']) > 0:
                    position['stop'] = candidate
                if not detail['armed'] or position['stop'] != previous_stop:
                    paper.runner_journal(self.path.with_name('main_trend_decisions.jsonl'), {
                        'record_type': 'executable_profit_lock', 'checked_at_ms': self.now_ms(),
                        'observed_at_ms': observed, 'active_after_ms': self.main_trend_active_after_ms,
                        'entry_time_ms': position['entry_time'], 'previous_stop': previous_stop,
                        'new_stop': position['stop'], 'executable_close_price': fill['price'],
                        'gain_fraction': gain})
                detail.update(armed=True, locked_stop=position['stop'])
            state['main_trend_profit_protection'] = deepcopy(detail)
        except (ValueError, KeyError, TypeError, IndexError, OverflowError):
            return  # Independent mark/stop exits remain available.

    def _current_diagnostics(self, state):
        timestamp = self.now_ms()
        risk = state.get('risk_monitor') or {}
        errors = state.get('errors') or {}
        verified_clock = (risk.get('clock_source') == 'exchange'
                          and risk.get('assessment') == 'evaluated' and not errors.get('clock'))
        if verified_clock and not errors.get('signal'):
            bars = [paper.arithmetic.candle(row) for row in self.signal_bars]
            function = (entry_trend.trend_diagnostics if entry_trend.enabled(self._declared_config)
                        else main_trend.trend_diagnostics)
            current = function(bars, self._declared_config, timestamp)
        else:
            current = {'enabled': True, 'status': 'unavailable', 'checked_at_ms': timestamp,
                       'reasons': ['main_trend_current_clock_or_candles_unavailable'],
                       'hourly': None, 'four_hour': None}
        if verified_clock:
            self._sector_decision(state, phase='observation', timestamp=timestamp)
        else:
            state['main_sector_status'] = {'enabled': True, 'allowed': False, 'status': 'unknown',
                'direction': None, 'reasons': ['main_sector_verified_clock_unavailable'],
                'checked_at_ms': timestamp, 'protective_exits_allowed': True}
        state['current_trend'] = current
        if entry_trend.enabled(self._declared_config):
            state['entry_trend_confirmation'] = deepcopy(
                (state.get('signal_diagnostics') or {}).get('entry_trend_confirmation'))
            state['entry_trend_runtime'] = {'enabled': True, 'mode': entry_trend.POLICY['mode'],
                'active_after_ms': self.entry_trend_active_after_ms,
                'confirmation_timeframes': list(entry_trend.POLICY['confirmation_timeframes'])}
        state['long_confirmation'] = deepcopy((state.get('signal_diagnostics') or {}).get('long_confirmation'))
        if state.get('position'):
            four = current.get('four_hour') or {}
            direction = four.get('direction', 0)
            state['swing_confirmation'] = {'enabled': True, 'allowed': bool(direction),
                **four, 'reasons': current.get('reasons', []), 'direction': direction,
                'confirmation_timeframe': '4h', 'checked_at_ms': timestamp,
                'confirmation_close_ms': current.get('confirmation_close_ms'),
                'decision_scope': 'current_context_not_entry_permission'}
            if state['position'].get('pending_exit') == 'main_trend_invalidation':
                state['main_trend_exit_status'] = {**state.get('main_trend_exit_status', {}),
                    'status': 'exit_pending', 'triggered': True}
        else:
            state['main_trend_profit_protection'] = None
            state['main_trend_exit_status'] = {'enabled': True, 'status': 'flat', 'triggered': False,
                                              'checked_at_ms': timestamp}

    def step(self, market_data=None):
        state = paper.read_json(self.path)
        if pullback.enabled(self._declared_config) and state.get('pullback_entry_active_after_ms') != self.pullback_entry_active_after_ms:
            raise ValueError('pullback_activation_changed')
        self._guard_decision = None
        if market_data is not None:
            self._guard_market = {label: (value, error) for label, value, error in market_data}
        if execution_guard.enabled(self._declared_config) and state.get('execution_guard_active_after_ms') != self.execution_guard_active_after_ms:
            raise ValueError('main_execution_guard_activation_changed')
        if short.enabled(self.config) and state.get('short_confirmation_active_after_ms') != self.short_confirmation_active_after_ms:
            raise ValueError('main_short_confirmation_activation_changed')
        if (profit_budget.enabled(self._declared_config)
                and state.get('trade_profit_budget_active_after_ms') != self.trade_profit_budget_active_after_ms):
            raise ValueError('main_profit_budget_activation_changed')
        if (main_trend.enabled(self._declared_config)
                and state.get('main_trend_active_after_ms') != self.main_trend_active_after_ms):
            raise ValueError('main_trend_activation_changed')
        if (entry_trend.enabled(self._declared_config)
                and state.get('entry_trend_active_after_ms') != self.entry_trend_active_after_ms):
            raise ValueError('entry_trend_activation_changed')
        self._budget_decision = None
        self._sector_blocked = None
        self._entry_trend_blocked = None
        self._legacy_position_mode = bool(state.get('position')
            and not state['position'].get('trade_profit_budget') and profit_budget.enabled(self._declared_config))
        if self._legacy_position_mode:
            legacy = state.get('trade_profit_budget_legacy_position_rule')
            if not isinstance(legacy, dict) or profit_budget.enabled(legacy):
                raise ValueError('Grandfathered position requires its original rule snapshot')
            self.config = deepcopy(legacy)
            if main_trend.enabled(self._declared_config):
                self.config[main_trend.POLICY_KEY] = deepcopy(self._declared_config[main_trend.POLICY_KEY])
        else:
            self.config = self._declared_config
        adapter = _SignalProfiles(self.short_confirmation_active_after_ms or 0,
                                  self.main_trend_active_after_ms, self.entry_trend_active_after_ms)
        def write(path, value):
            if Path(path) == Path(self.path) and isinstance(value, dict):
                value['main_stock_runtime'] = {'mode': 'price_led_short_v1',
                    'short_confirmation_active_after_ms': self.short_confirmation_active_after_ms,
                    'clock': deepcopy(self._main_clock_diagnostic)}
                value['short_confirmation'] = deepcopy((value.get('signal_diagnostics') or {}).get('short_confirmation'))
                if main_trend.enabled(self._declared_config):
                    try:
                        self._current_diagnostics(value)
                    except Exception:
                        # A completed protective fill must still reach its state
                        # file if an auxiliary diagnostic snapshot is malformed.
                        value['current_trend'] = {'enabled': True, 'status': 'unavailable',
                            'checked_at_ms': self.now_ms(), 'reasons': ['main_trend_diagnostics_unavailable'],
                            'hourly': None, 'four_hour': None}
                        value['main_sector_status'] = {'enabled': True, 'allowed': False,
                            'status': 'unknown', 'direction': None,
                            'reasons': ['main_sector_data_unavailable'], 'protective_exits_allowed': True}
                    value['main_stock_runtime']['main_trend'] = {
                        'mode': main_trend.POLICY['mode'], 'active_after_ms': self.main_trend_active_after_ms,
                        'grandfathered_exit_overlay': self._legacy_position_mode}
                    if entry_trend.enabled(self._declared_config):
                        value['main_stock_runtime']['entry_trend'] = {
                            'enabled': True, 'mode': entry_trend.POLICY['mode'],
                            'active_after_ms': self.entry_trend_active_after_ms,
                            'confirmation_timeframes': list(entry_trend.POLICY['confirmation_timeframes'])}
                    if self._entry_trend_blocked and not value.get('position') and value.get('pending_signal'):
                        value['signal_status'] = 'swing_confirmation_blocked'
                        value['entry_blockers'] = list(dict.fromkeys(value.get('entry_blockers', [])
                            + self._entry_trend_blocked['reasons']))
                    if self._sector_blocked and not value.get('position') and value.get('pending_signal'):
                        value['main_sector_status'] = deepcopy(self._sector_blocked)
                        value['signal_status'] = 'main_sector_blocked'
                        value['entry_blockers'] = list(dict.fromkeys(value.get('entry_blockers', [])
                            + self._sector_blocked['reasons']))
                if self._budget_decision is not None:
                    value['entry_profit_budget'] = deepcopy(self._budget_decision)
                    if not self._budget_decision.get('allowed') and not value.get('position'):
                        value['signal_status'] = 'trade_profit_budget_blocked'
                        value['entry_blockers'] = list(dict.fromkeys(value.get('entry_blockers', [])
                            + self._budget_decision.get('reasons', [])))
                value['main_stock_runtime']['trade_profit_budget'] = {
                    'enabled': profit_budget.enabled(self._declared_config),
                    'active_after_ms': self.trade_profit_budget_active_after_ms,
                    'grandfathered_existing_position': self._legacy_position_mode}
                if execution_guard.enabled(self._declared_config):
                    value['main_stock_runtime']['execution_guard'] = {
                        'enabled': True, 'active_after_ms': self.execution_guard_active_after_ms,
                        'planned_budget_fraction': execution_guard.POLICY['planned_budget_fraction']}
                    if self._guard_decision is not None:
                        value['execution_guard'] = deepcopy(self._guard_decision)
                if pullback.enabled(self._declared_config):
                    value['main_stock_runtime']['trend_pullback_entry'] = {
                        'enabled': True, 'mode': pullback.POLICY['mode'],
                        'active_after_ms': self.pullback_entry_active_after_ms}
            return _BASE_WRITE(path, value)
        function = FunctionType(_BASE_STEP.__code__, {**_BASE_STEP.__globals__,
            'stock_swing_profiles': adapter, 'stock_entry_timing': _EntryTiming(),
            'write_json': write}, _BASE_STEP.__name__,
            _BASE_STEP.__defaults__, _BASE_STEP.__closure__)
        function.__kwdefaults__ = _BASE_STEP.__kwdefaults__
        try:
            return function(self, market_data)
        finally:
            self.config = self._declared_config
