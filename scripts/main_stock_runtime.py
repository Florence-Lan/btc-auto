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

import run_parallel_simulation as paper
import stock_short_confirmation as short
import stock_swing_profiles as profiles
import trade_profit_budget as profit_budget

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
    def __init__(self, active_after_ms):
        self.active_after_ms = active_after_ms

    def __getattr__(self, name):
        return getattr(profiles, name)

    def signal_with_diagnostics(self, candles, indicators, index, config=None):
        config = {} if config is None else config
        boundary = candles[index].time_ms + HOUR
        active = short.enabled(config) and boundary > self.active_after_ms
        function = short.signal_with_diagnostics if active else profiles.signal_with_diagnostics
        signal, diagnostic = function(candles, indicators, index, config)
        diagnostic = deepcopy(diagnostic)
        diagnostic['main_short_confirmation_runtime'] = {
            'enabled': short.enabled(config), 'candidate_active_for_bar': active,
            'active_after_ms': self.active_after_ms, 'bar_close_ms': boundary,
            'mode': short.POLICY['mode'] if short.enabled(config) else None}
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


class MainStockAccount(paper.StockAccount):
    """Use the existing main ledger with a private, post-activation dispatcher."""
    shared_clock_loader = staticmethod(load_shared_clock)

    def __init__(self, path, venue, symbol, config):
        short.validate(config)
        super().__init__(path, venue, symbol, config)
        state = paper.read_json(self.path)
        self.short_confirmation_active_after_ms = state.get('short_confirmation_active_after_ms')
        if short.enabled(config) and (type(self.short_confirmation_active_after_ms) is not int
                or self.short_confirmation_active_after_ms <= 0):
            raise ValueError('main_short_confirmation_activation_required')
        self._main_clock_diagnostic = {'source': 'not_observed'}
        self._declared_config = deepcopy(config)
        self._budget_decision = None
        self._legacy_position_mode = False
        self.trade_profit_budget_active_after_ms = state.get('trade_profit_budget_active_after_ms')
        if profit_budget.enabled(config):
            profit_budget.validate(config)
            if (type(self.trade_profit_budget_active_after_ms) is not int
                    or self.trade_profit_budget_active_after_ms <= 0):
                raise ValueError('main_profit_budget_activation_required')

    def market_data(self, force_rules=False, force_signal=False):
        self._main_clock_diagnostic = {'source': 'not_observed'}
        return _BASE_MARKET_DATA(_AccountView(self), force_rules=force_rules, force_signal=force_signal)

    def _plan_budget(self, state, equity, sig, entry, stop, step, lot, volume, minimum, actual_qty=None):
        decision = profit_budget.stock_plan(self.config, equity=equity, entry=entry,
            stop=stop, step=step, direction=sig.direction,
            maximum_qty=float(lot['maxQty']), minimum_qty=float(lot['minQty']),
            minimum_notional=minimum, liquidity_qty=volume * .1, actual_qty=actual_qty)
        decision.update(checked_at_ms=self.now_ms(), signal_time_ms=sig.time_ms,
            signal_boundary_ms=sig.time_ms + self.signal_interval_ms)
        self._budget_decision = decision
        state['entry_profit_budget'] = deepcopy(decision)
        return decision

    def entry_quantity(self, state, equity, signal, entry, stop, step, lot, volume, minimum):
        if self._legacy_position_mode:
            self._budget_decision = {'enabled': True, 'allowed': False,
                'reasons': ['legacy_position_closed_wait_for_next_observation']}
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
        if not profit_budget.enabled(self.config):
            return super().entry_fill_admission(state, equity, signal, candidate, result, step, lot, volume, minimum)
        decision = self._plan_budget(state, equity, signal, result['price'], candidate['stop'],
            step, lot, volume, minimum, actual_qty=result['qty'])
        if decision['allowed']:
            candidate['trade_profit_budget'] = {**deepcopy(decision),
                'activated_at_ms': self.trade_profit_budget_active_after_ms,
                'entry_rule': deepcopy(self.config)}
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
        if original or not position.get('trade_profit_budget'):
            return original
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

    def step(self, market_data=None):
        state = paper.read_json(self.path)
        if short.enabled(self.config) and state.get('short_confirmation_active_after_ms') != self.short_confirmation_active_after_ms:
            raise ValueError('main_short_confirmation_activation_changed')
        if (profit_budget.enabled(self._declared_config)
                and state.get('trade_profit_budget_active_after_ms') != self.trade_profit_budget_active_after_ms):
            raise ValueError('main_profit_budget_activation_changed')
        self._budget_decision = None
        self._legacy_position_mode = bool(state.get('position')
            and not state['position'].get('trade_profit_budget') and profit_budget.enabled(self._declared_config))
        if self._legacy_position_mode:
            legacy = state.get('trade_profit_budget_legacy_position_rule')
            if not isinstance(legacy, dict) or profit_budget.enabled(legacy):
                raise ValueError('Grandfathered position requires its original rule snapshot')
            self.config = legacy
        else:
            self.config = self._declared_config
        adapter = _SignalProfiles(self.short_confirmation_active_after_ms or 0)
        def write(path, value):
            if Path(path) == Path(self.path) and isinstance(value, dict):
                value['main_stock_runtime'] = {'mode': 'price_led_short_v1',
                    'short_confirmation_active_after_ms': self.short_confirmation_active_after_ms,
                    'clock': deepcopy(self._main_clock_diagnostic)}
                value['short_confirmation'] = deepcopy((value.get('signal_diagnostics') or {}).get('short_confirmation'))
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
            return _BASE_WRITE(path, value)
        function = FunctionType(_BASE_STEP.__code__, {**_BASE_STEP.__globals__,
            'stock_swing_profiles': adapter, 'write_json': write}, _BASE_STEP.__name__,
            _BASE_STEP.__defaults__, _BASE_STEP.__closure__)
        function.__kwdefaults__ = _BASE_STEP.__kwdefaults__
        try:
            return function(self, market_data)
        finally:
            self.config = self._declared_config
