"""Offline user-specified historical entries; never rewrites forward accounts.

The entry is a hindsight hypothesis at a recorded signal-capture time, including
an explicit override of original signal-admission gates. Historical market depth,
costs, sizing and subsequent original protective exits are used without networking.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import FunctionType

import run_parallel_simulation as paper

ROOT = paper.ROOT
RESEARCH = ROOT / 'data/research/retrospective_catchup_20261031'
MAIN = ROOT / 'data/parallel_simulation/btc_memory_stocks_latest_1000_each_20261007'
LABEL = 'RETROSPECTIVE_HYPOTHETICAL_ONLY'
MAX_LOOKBACK_BYTES = 256 * 1024 * 1024


def milliseconds(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


def row_time(row):
    return milliseconds(row.get('time_utc') or row['checked_at_utc'])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def retrospective_path(path):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to((ROOT/'data/retrospective_simulation').resolve())
    except ValueError:
        raise ValueError('Writes must stay inside data/retrospective_simulation') from None
    return resolved


def persist_json(path, value):
    """Retry only the atomic write when Windows briefly shares a replay file."""
    path = retrospective_path(path)
    for attempt in range(7):
        try:
            return paper.write_json(path, value)
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.005 * 2 ** attempt)


def historical_window(path, start_ms, cutoff_ms):
    """Read complete original rows backwards; never consume post-cutoff data."""
    found, remainder = [], b''
    with Path(path).open('rb') as handle:
        handle.seek(0, 2)
        end = cursor = handle.tell()
        floor = max(0, end - MAX_LOOKBACK_BYTES)
        stopped = False
        first = True
        while cursor > floor and not stopped:
            lower = max(floor, cursor - 1024 * 1024)
            handle.seek(lower)
            data = handle.read(cursor - lower) + remainder
            pieces = data.split(b'\n')
            remainder = pieces[0] if lower else b''
            complete = pieces[1:] if lower else pieces
            if first:
                complete = complete[:-1]  # Last fragment might still be written.
                first = False
            for raw in reversed(complete):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                at = row_time(row)
                if at > cutoff_ms:
                    continue
                if at < start_ms - 2 * paper.FIVE_MINUTES:
                    stopped = True
                    break
                found.append({'at_ms': at, 'row_sha256': sha(raw), 'row': row})
            cursor = lower
        if not stopped and floor:
            raise ValueError('Historical window exceeds the bounded journal read')
    found.reverse()
    if not found or any(a['at_ms'] > b['at_ms'] for a, b in zip(found, found[1:])):
        raise ValueError('Missing or nonchronological original observations')
    return found, {'source_path': str(Path(path).relative_to(ROOT)),
                   'source_size_at_read': end, 'bounded_bytes_read': end - cursor,
                   'rows': len(found), 'first_ms': found[0]['at_ms'],
                   'last_ms': found[-1]['at_ms']}


def recorded_responses(row):
    """Use recorded clock admission, not a newly certified network receipt.

    Main observations persist the exchange-clock validation result but not the
    raw time endpoint body. This explicit replay coordinate uses that recorded
    result. It cannot be supplied to a live runner and never refreshes old data.
    """
    at = row_time(row)
    errors = row.get('errors') or {}
    risk = row.get('risk_monitor') or {}
    clock_ok = risk.get('clock_source') == 'exchange' and not errors.get('clock')
    result = [('clock', {'serverTime': at} if clock_ok else None,
               None if clock_ok else errors.get('clock', 'Original exchange clock unavailable'))]
    for label, key in (('mark', 'mark'), ('book', 'book'), ('funding', 'funding_events'),
                       ('five', 'preceding_candles_5m'), ('signal', 'signal_candles')):
        if key in row or label in errors:
            result.append((label, deepcopy(row.get(key)), errors.get(label)))
    return result


class HistoricalVenue:
    """Only a documented historical settlement proxy; no network capability."""
    def __init__(self, evidence=None, supplements=None):
        self.evidence = evidence or []
        self.supplements = supplements or []
        self.at_ms = 0

    def get(self, endpoint, params=None):
        params = params or {}
        if endpoint == 'markPriceKlines':
            for point in self.evidence:
                if (point['symbol'] == params.get('symbol')
                        and point['minute_ms'] == params.get('startTime')
                        and self.at_ms >= point['available_by_ms']):
                    # Only the archived open is known; never invent high/low/close.
                    return [[point['minute_ms'], point['open']]]
        raise ValueError('Historical endpoint evidence unavailable; network disabled')


class ReplayStockAccount(paper.StockAccount):
    def __init__(self, path, config, symbol, rules, evidence=None, supplements=None):
        path = retrospective_path(path)
        super().__init__(path, HistoricalVenue(evidence, supplements), symbol, config)
        self.rules = deepcopy(rules)
        self.replay_at_ms = 0
        self.original_row = {}

    def now_ms(self):
        return self.replay_at_ms

    def tagged_journal(self, path, row):
        paper.journal(path, {**row, 'retrospective_only': True,
            'performance_class': LABEL, 'not_forward_performance': True})

    def fill(self, *args, **kwargs):
        method = paper.StockAccount.fill
        function = FunctionType(method.__code__, {**method.__globals__,
            'journal': self.tagged_journal}, method.__name__, method.__defaults__)
        result = function(self, *args, **kwargs)
        result.update(retrospective_only=True, performance_class=LABEL,
                      not_forward_performance=True)
        return result

    def close(self, *args, **kwargs):
        method = paper.StockAccount.close
        function = FunctionType(method.__code__, {**method.__globals__,
            'journal': self.tagged_journal}, method.__name__, method.__defaults__)
        return function(self, *args, **kwargs)

    def record_stock_event_risk(self, state, phase, reference_ms=None):
        # Initial entry is explicitly user-specified. Subsequent steps cannot
        # create entries; exits do not require a future calendar permission.
        result = deepcopy(self.original_row.get('stock_event_risk') or {
            'enabled': True, 'allowed': False, 'status': 'historical_calendar_unavailable',
            'reasons': ['historical_calendar_unavailable']})
        result['phase'] = phase
        state['stock_event_risk'] = result
        return result

    def funding(self, state, events, timestamp):
        # Supplemental and primary receipts can encode the same settlement
        # differently. Its economic identity remains symbol + settlement time.
        supplemental_ids = {(point['event']['symbol'], int(point['event']['fundingTime']))
                            for point in self.venue.supplements}
        known = state.setdefault('retrospective_settled_funding', {})
        method = paper.StockAccount.funding
        function = FunctionType(method.__code__, {**method.__globals__,
            'journal': self.tagged_journal}, method.__name__, method.__defaults__)
        for event in sorted(events, key=lambda item: int(item['fundingTime'])):
            identity = (event.get('symbol', self.symbol), int(event['fundingTime']))
            key = identity[0] + ':' + str(identity[1])
            if identity in supplemental_ids and key in known:
                if float(event['fundingRate']) != known[key]:
                    raise ValueError('Conflicting historical funding settlement rate')
                continue
            function(self, state, [event], timestamp)
            event_hash = sha(json.dumps(event, sort_keys=True).encode())
            if identity in supplemental_ids and event_hash in state['settled_funding']:
                known[key] = float(event['fundingRate'])

    def step_row(self, row):
        self.replay_at_ms = row_time(row)
        self.venue.at_ms = self.replay_at_ms
        self.original_row = row
        method = paper.StockAccount.step
        function = FunctionType(method.__code__, {**method.__globals__,
            'now_ms': self.now_ms, 'journal': self.tagged_journal,
            'write_json': persist_json},
            method.__name__, method.__defaults__)
        responses = recorded_responses(row)
        supplements = [point['event'] for point in self.venue.supplements
            if point['event']['symbol'] == self.symbol
            and self.replay_at_ms >= point['event_available_by_ms']]
        if supplements:
            previous = next((item for item in responses if item[0] == 'funding'), None)
            events = deepcopy(previous[1] or []) if previous else []
            for event in supplements:
                existing = [item for item in events if int(item['fundingTime']) == int(event['fundingTime'])]
                if existing and any(float(item['fundingRate']) != float(event['fundingRate']) for item in existing):
                    raise ValueError('Conflicting historical funding evidence')
                if not existing:
                    events.append(deepcopy(event))
            responses = [item for item in responses if item[0] != 'funding']
            responses.append(('funding', events, None))
        return function(self, responses)


def hypothetical_entry(account, state, candidate, row, capture_ms):
    """Book one requested historical short, preserving real depth/risk/costs."""
    if state.get('position'):
        raise ValueError('Historical seed must be flat')
    if ((row.get('risk_monitor') or {}).get('clock_source') != 'exchange'
            or (row.get('errors') or {}).get('clock')):
        raise ValueError('Original snapshot has no recorded exchange-clock validation')
    cfg = account.config
    signal = candidate['original_signal']
    if signal['direction'] != -1 or signal['time_ms'] + paper.SIGNAL_INTERVALS['1h'] != candidate['boundary_ms']:
        raise ValueError('Invalid original short signal identity')
    book, mark = row['book'], row['mark']
    paper.fresh(book.get('T') or book.get('E'), capture_ms)
    paper.fresh(mark['time'], capture_ms)
    bid, ask = float(book['bids'][0][0]), float(book['asks'][0][0])
    if not 0 < bid <= ask or not math.isfinite(ask):
        raise ValueError('Invalid historical book')
    for levels in (book['bids'], book['asks']):
        if any(not math.isfinite(float(p)) or float(p) <= 0
               or not math.isfinite(float(q)) or float(q) < 0 for p, q in levels):
            raise ValueError('Invalid historical book levels')
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in (
            mark['markPrice'], mark['indexPrice'], signal['atr'], signal['close'])):
        raise ValueError('Invalid historical mark, index or signal')
    five = row.get('preceding_candles_5m') or []
    expected = capture_ms // paper.FIVE_MINUTES * paper.FIVE_MINUTES - paper.FIVE_MINUTES
    volume = float(five[-1][5]) if five and int(five[-1][0]) == expected else 0
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError('Historical preceding five-minute volume unavailable')
    filters = {r['filterType']: r for r in account.rules['filters']}
    lot = filters.get('MARKET_LOT_SIZE') or filters['LOT_SIZE']
    step, tick = float(lot['stepSize']), float(filters['PRICE_FILTER']['tickSize'])
    minimum = float(filters.get('MIN_NOTIONAL', {}).get('notional',
                        filters.get('MIN_NOTIONAL', {}).get('minNotional', 5)))
    estimate = bid * (1 - cfg['adverse_slippage_fraction_assumption'])
    stop = paper.signals.initial_stop(estimate, -1, signal['atr'], cfg)
    if not stop:
        raise ValueError('Original stop-distance model rejects historical entry')
    stop = paper.arithmetic.round_tick(stop, tick, True)
    qty = paper.signals.position_size(state['wallet_balance'], estimate, stop, step,
        cfg['risk_fraction_per_trade'], cfg['taker_fee_rate_assumption'],
        cfg['adverse_slippage_fraction_assumption'])
    qty = min(qty, state['wallet_balance'] * cfg['portfolio_gross_notional_fraction'] / estimate,
              volume * .1, float(lot['maxQty']), state['wallet_balance'] * 10 / estimate)
    result = paper.book_fill(book, -paper.floor_qty(qty, step), step,
                             cfg['adverse_slippage_fraction_assumption'])
    if result['qty'] < float(lot['minQty']) or result['qty'] * result['price'] < minimum:
        raise ValueError('Historical observed depth/minimum quantity cannot support entry')
    stop = paper.signals.initial_stop(result['price'], -1, signal['atr'], cfg)
    if not stop:
        raise ValueError('Observed fill violates original stop-distance bounds')
    stop = paper.arithmetic.round_tick(stop, tick, True)
    risk = result['qty'] * paper.signals.per_unit_stop_risk(result['price'], stop,
        cfg['taker_fee_rate_assumption'], cfg['adverse_slippage_fraction_assumption'])
    if risk > state['wallet_balance'] * cfg['risk_fraction_per_trade'] + 1e-9:
        raise ValueError('Historical entry exceeds original risk budget')
    position = {'entry': result['price'], 'qty': result['qty'], 'direction': -1,
        'entry_time': capture_ms, 'signal_time': candidate['boundary_ms'], 'stop': stop,
        'initial_stop': stop, 'entry_fee': result['qty'] * result['price'] * cfg['taker_fee_rate_assumption'],
        'margin': result['qty'] * result['price'] / cfg['leverage'], 'funding': 0.,
        'atr': signal['atr'], 'entry_origin': LABEL}
    liquidation = paper.arithmetic.liquidation_price(position, cfg['maintenance_margin_fraction_assumption'])
    if (-(stop - liquidation) / result['price'] < cfg['minimum_liquidation_distance_buffer']
            or abs(stop / result['price'] - 1) > cfg['max_stop_fraction'] + 1e-12):
        raise ValueError('Historical entry violates liquidation or stop buffer')
    account.replay_at_ms = capture_ms
    fill = account.fill(state, result, capture_ms, -1, 'retrospective_user_requested_signal_capture')
    state['position'] = position
    state['position_qty'] = -result['qty']
    state['position_history'].append({'time_ms': capture_ms, 'signed_qty': -result['qty']})
    state['last_mark_price'] = float(mark['markPrice'])
    state['next_funding_ms'] = int(mark['nextFundingTime'])
    state['equity'] = state['wallet_balance'] - result['qty'] * (float(mark['markPrice']) - result['price'])
    state['retrospective_entry'] = {'original_signal': signal, 'actual_capture_time_ms': capture_ms,
        'original_admission': candidate.get('original_admission'),
        'original_expiry_ms': candidate.get('original_expiry_ms'),
        'original_trigger': candidate.get('trigger'), 'risk_budget_usdt':
            (state['wallet_balance'] + fill['fee']) * cfg['risk_fraction_per_trade'],
        'planned_cost_aware_stop_risk_usdt': risk,
        'initial_stop_price': stop,
        'historical_snapshot_recorded_at_ms': row_time(row),
        'historical_clock_basis': 'original_main_recorded_exchange_validation',
        'entry_is_user_specified_not_original_strategy_admission': True,
        'entry_admission_overrides': ['original_confirmation', 'original_breakout_validity',
            'original_chase_and_entry_funding_filters'],
        'historical_mark_time_ms': mark['time'], 'historical_book_time_ms': book.get('T') or book.get('E')}
    persist_json(account.path, state)
    return fill


def replay_one(name, candidate, seed, rule, source, output, cutoff_ms, funding=None, supplements=None):
    output = retrospective_path(output)
    if name not in ('mu', 'sndk', 'skhynix'):
        raise ValueError('Only the three declared retrospective stock accounts are supported')
    rows, source_info = historical_window(source, seed['start_ms'], cutoff_ms)
    capture_ms = seed['start_ms']
    first = next(item for item in rows if item['at_ms'] >= capture_ms
                 and item['row'].get('mark') and item['row'].get('book'))
    if first['at_ms'] - capture_ms > 30_000:
        raise ValueError('No original main snapshot sufficiently near signal capture')
    path = output / name / 'state.json'
    path.parent.mkdir(parents=True, exist_ok=False)
    state = deepcopy(seed)
    state.update(pending_signal=None, signal_active_after_ms=cutoff_ms + 3_600_000,
        retrospective_only=True, performance_class=LABEL, forward_validated=False)
    persist_json(path, state)
    account = ReplayStockAccount(path, state['rule'], state['symbol'], rule, funding, supplements)
    for item in rows:
        if item['at_ms'] <= capture_ms:
            if item['row'].get('signal_candles'):
                account.signal_bars = deepcopy(item['row']['signal_candles'])
    if first['row'].get('signal_candles'):
        account.signal_bars = deepcopy(first['row']['signal_candles'])
    if first['row'].get('preceding_candles_5m'):
        account.five = deepcopy(first['row']['preceding_candles_5m'])
    initial_wallet = state['wallet_balance']
    fill = hypothetical_entry(account, state, candidate, first['row'], capture_ms)
    errors, gaps, last = [], [], None
    previous = capture_ms
    for item in rows:
        if item['at_ms'] <= first['at_ms']:
            continue
        if item['at_ms'] - previous > 120_000:
            gaps.append({'from_ms': previous, 'to_ms': item['at_ms'],
                         'milliseconds': item['at_ms'] - previous})
        previous = item['at_ms']
        last = item['row']
        try:
            account.step_row(last)
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            errors.append({'at_ms': item['at_ms'], 'error': str(exc)[:180]})
            account.tagged_journal(path.with_name('replay_errors.jsonl'), errors[-1])
    state = read_json(path)
    closed = [json.loads(line) for line in path.with_name('closed_trades.jsonl').read_text(encoding='utf-8').splitlines()] if path.with_name('closed_trades.jsonl').exists() else []
    pos = state['position']
    pending_funding = state.get('next_funding_ms')
    outstanding_inventory = 0.
    if pending_funding and pending_funding <= cutoff_ms:
        for point in state['position_history']:
            if point['time_ms'] >= pending_funding:
                break
            outstanding_inventory = point['signed_qty']
    funding_complete = not (outstanding_inventory and
        int(state.get('funding_through_ms') or 0) < pending_funding)
    summary = {'account': name, 'symbol': state['symbol'], 'performance_class': LABEL,
        'entry': fill, 'initial_stop': state['retrospective_entry']['initial_stop_price'],
        'entry_assumptions': state['retrospective_entry'],
        'seed_wallet': initial_wallet, 'cutoff_ms': cutoff_ms, 'last_observation_ms': previous,
        'last_risk_success_ms': (state.get('risk_monitor') or {}).get('last_success_at_ms'),
        'cash_pnl_usdt': state['wallet_balance'] - initial_wallet,
        'net_pnl_marked_usdt': state.get('equity', state['wallet_balance']) - initial_wallet,
        'position': pos, 'closed_trades': closed, 'status': state.get('status'),
        'errors': state.get('errors'), 'replay_errors': errors, 'coverage_gaps_over_120s': gaps,
        'funding_complete_for_replay_inventory': funding_complete,
        'net_result_complete': funding_complete,
        'supplemental_historical_funding_sources': deepcopy(supplements or []),
        'source': source_info, 'entry_snapshot_row_sha256': first['row_sha256'],
        'main_forward_accounts_modified': False, 'no_network_requests': True}
    if not funding_complete:
        summary['net_pnl_marked_usdt'] = None
        summary['unsettled_funding_time_ms'] = pending_funding
    if pos and not funding_complete:
        summary['estimated_exit_unavailable'] = 'Unsettled funding; complete net result unavailable'
    elif pos and last and last.get('book') and last.get('mark'):
        try:
            paper.fresh(last['book'].get('T') or last['book'].get('E'), cutoff_ms)
            paper.fresh(last['mark']['time'], cutoff_ms)
            filters = {r['filterType']: r for r in rule['filters']}
            lot = filters.get('MARKET_LOT_SIZE') or filters['LOT_SIZE']
            exit_fill = paper.book_fill(last['book'], -pos['direction'] * pos['qty'],
                float(lot['stepSize']), state['rule']['adverse_slippage_fraction_assumption'])
            if exit_fill['qty'] + 1e-12 < pos['qty']:
                raise ValueError('Observed depth cannot support complete estimated exit')
            summary['estimated_net_if_closed_at_cutoff'] = (state['wallet_balance'] - initial_wallet
                + pos['direction'] * pos['qty'] * (exit_fill['price'] - pos['entry'])
                - exit_fill['qty'] * exit_fill['price'] * state['rule']['taker_fee_rate_assumption'])
            summary['estimated_exit_not_booked'] = exit_fill
        except (ValueError, TypeError, KeyError) as exc:
            summary['estimated_exit_unavailable'] = str(exc)
    persist_json(output / name / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, default=RESEARCH)
    parser.add_argument('--candidates', type=Path, default=ROOT/'data/research/main_catchup_20261030/candidates.json')
    parser.add_argument('--cutoff-utc', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = retrospective_path(args.output_dir)
    if output.exists():
        raise ValueError('Retrospective output exists; refusing to rewrite any ledger')
    cutoff = milliseconds(args.cutoff_utc)
    seeds = read_json(args.research/'seeds.json')['accounts']
    candidates = read_json(args.candidates)['candidates']
    rules = read_json(args.research/'rules_by_account.json')['rules']
    funding_path = args.research/'funding_evidence.json'
    funding = []
    if funding_path.exists():
        evidence = read_json(funding_path)
        source = evidence['journal_source']
        with (ROOT/source['path']).open('rb') as handle:
            if sha(handle.read(source['prefix_size'])) != source['prefix_sha256']:
                raise ValueError('Historical funding journal evidence changed')
        funding = [{'symbol': evidence['symbol'],
            'minute_ms': evidence['funding_purpose_projection'][0],
            'open': evidence['funding_purpose_projection'][1],
            'available_by_ms': evidence['conservative_evidence_available_by_ms']}]
    supplement_path = args.research/'funding_supplement_evidence.json'
    supplements = read_json(supplement_path)['supplemental_events'] if supplement_path.exists() else []
    for point in supplements:
        funding.append({'symbol': point['event']['symbol'],
            'minute_ms': point['proxy_minute_ms'], 'open': point['proxy_open'],
            'available_by_ms': point['proxy_available_by_ms']})
    output.mkdir(parents=True)
    manifest = {'mode': LABEL, 'complete': False, 'places_orders': False, 'main_returns_modified': False,
        'cutoff_ms': cutoff, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'entry_basis': 'User-specified hypothetical entry at the original main signal-capture time, original observed depth, normal historical sizing and costs; original admission gates are explicitly overridden.',
        'exit_basis': 'Original protective exits replayed chronologically; no selection of best later exit.',
        'limitations': ['Observed snapshots cannot prove exchange fills or unobserved intrapoll price paths.',
            'Original recorded exchange-clock validation is used as a replay coordinate; raw main clock responses were not archived.',
            'This hindsight manual-entry hypothesis is not original strategy forward profit.',
            'Missing main funding events may be supplemented only from separately archived historical public receipts at their recorded availability times.'],
        'inputs_sha256': {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in
            (args.candidates, args.research/'seeds.json', args.research/'rules_by_account.json',
             *([funding_path] if funding_path.exists() else []),
             *([supplement_path] if supplement_path.exists() else []))},
        'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__).read_bytes()),
            'scripts/run_parallel_simulation.py': sha(Path(paper.__file__).read_bytes())},
        'accounts': {}}
    persist_json(output/'manifest.json', manifest)
    for name in ('mu', 'sndk', 'skhynix'):
        seed = read_json(ROOT/seeds[name]['replay_state_path'])
        manifest['accounts'][name] = replay_one(name, candidates[name], seed,
            rules[name]['raw_contract_rules'], MAIN/name/'observations.jsonl', output, cutoff, funding, supplements)
        persist_json(output/'manifest.json', manifest)
    manifest['complete'] = True
    persist_json(output/'manifest.json', manifest)
    print(json.dumps({name: {key: value for key, value in row.items() if key in (
        'entry', 'cash_pnl_usdt', 'net_pnl_marked_usdt', 'position', 'closed_trades',
        'estimated_net_if_closed_at_cutoff', 'errors', 'coverage_gaps_over_120s')}
        for name, row in manifest['accounts'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
