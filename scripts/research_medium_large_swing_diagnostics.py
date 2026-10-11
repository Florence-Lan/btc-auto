#!/usr/bin/env python3
"""Fixed recent-30-day hourly behavior diagnostics, with no accounts or orders.

The production paper execution uses 5m volume and observed books. These inputs
contain only 1h OHLC, so this deliberately records its execution degradation and
does not claim to reproduce the running account. No parameter search is made.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import backtest_stock_swing_120 as engine
import stock_swing_profiles as profiles
from replay_stock_swing_per_symbol import after_close_summary, combine_buckets


COMMON_FIELDS = (
    'signal_timeframe', 'execution_timeframe', 'cooldown_signal_bars',
    'max_entry_spread_fraction', 'profit_exit_policy', 'trend_exit_policy',
    'entry_quality_policy', 'swing_confirmation_policy',
)
SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_source(snapshot, symbol):
    if snapshot['execution_step_ms'] != engine.HOUR:
        raise ValueError('This diagnostic requires explicitly hourly inputs')
    source = snapshot['symbols'][symbol]
    expected = list(range(source['start_ms'], snapshot['end_ms_exclusive'], engine.HOUR))
    audit = {}
    for name in ('trade_1h', 'mark_1h', 'index_1h'):
        rows = source[name]
        if [int(row[0]) for row in rows] != expected:
            raise ValueError(f'Missing, misordered or duplicate {symbol} {name}')
        for row in rows:
            engine.candle(row)  # Candle validates positive, finite and consistent OHLC.
            if int(row[6]) >= snapshot['end_ms_exclusive']:
                raise ValueError(f'Uncompleted bar in {symbol} {name}')
        audit[name] = {'rows': len(rows), 'zero_volume_bars': sum(float(r[5]) == 0 for r in rows)}
    funding_times = [int(item['fundingTime']) for item in source['funding']]
    if funding_times != sorted(set(funding_times)):
        raise ValueError(f'Invalid funding event order for {symbol}')
    audit.update(start_utc=engine.iso(source['start_ms']),
                 end_utc_exclusive=engine.iso(snapshot['end_ms_exclusive']),
                 funding_events=len(funding_times), exact_funding_marks_available=False)
    return audit


def production_rule(profile, base, symbol):
    common = {key: profile[key] for key in COMMON_FIELDS if key in profile}
    return {**base, **common, **profile['symbol_profiles'][symbol],
            'candidate_id': profile['symbol_profiles'][symbol].get('candidate_id', profile['candidate_id']),
            'risk_fraction_per_trade': profile['risk_fraction_per_bucket_trade'],
            'initial_equity_usdt': 1000, 'leverage': 10}


def effective_hourly_rule(rule, symbol):
    return {**copy.deepcopy(rule), 'symbols': [symbol], 'max_positions': 1,
            'execution_timeframe': '1h', 'entry_signal_validity_minutes': 0,
            'entry_max_previous_bar_participation_fraction': .1}


def prepare_pair(snapshot, candidate, symbol, start, end):
    """Use the official profile dispatch on every window signal opportunity.

    First prepare the identical ungated rule over all available history. The 4h
    gate cannot create a primary signal, so re-evaluating only those opportunities
    gives precisely the candidate signal set, retaining the original EMA seed.
    Calls for non-signal bars and opportunities outside this window cannot affect
    its simulation and are avoided. No replacement gate formula is implemented.
    """
    ungated = copy.deepcopy(candidate)
    ungated.pop('swing_confirmation_policy', None)
    ungated['symbol_profiles'] = {symbol: {}}
    prepared = engine.prepare(snapshot, ungated)
    candidate['symbol_profiles'] = {symbol: {}}
    candidate_prepared = {symbol: {**prepared[symbol], 'profile_config': candidate, 'signals': {}}}
    candles = [engine.candle(row) for row in snapshot['symbols'][symbol]['trade_1h']]
    indicators = engine.compute_indicators(candles, candidate)
    index_by_time = {bar.time_ms: index for index, bar in enumerate(candles)}
    counts = Counter()
    accepted_side = Counter()
    primary_side = Counter()
    rejection_reasons = Counter()
    gate_events = []
    for arrival, raw in prepared[symbol]['signals'].items():
        if not start <= arrival < end:
            continue
        side = 'long' if raw.direction == 1 else 'short'
        counts['primary_1h_signal_opportunities'] += 1
        primary_side[side] += 1
        index = index_by_time[raw.time_ms]
        signal, diagnostics = profiles.signal_with_diagnostics(candles, indicators, index, candidate)
        confirmation = diagnostics['swing_confirmation']
        if signal is not None:
            if signal != raw:
                raise AssertionError('4h gate altered the primary signal')
            candidate_prepared[symbol]['signals'][arrival] = signal
            counts['admitted_by_4h_gate'] += 1
            accepted_side[side] += 1
        else:
            counts['rejected_by_4h_gate'] += 1
            if confirmation['allowed']:
                raise AssertionError('Different primary or entry-quality rules in the control')
            rejection_reasons.update(confirmation['reasons'])
        gate_events.append({'signal_close_utc': engine.iso(arrival), 'direction': side,
                            'admitted': signal is not None, **confirmation})
    return ungated, prepared, candidate_prepared, {
        **dict(counts), 'primary_by_direction': dict(primary_side),
        'admitted_by_direction': dict(accepted_side),
        'rejection_reason_counts': dict(rejection_reasons),
        'reason_counts_can_overlap': True,
    }, gate_events


def describe(result, prepared, symbol, end):
    summary = after_close_summary(result)
    trades = result['trades']
    durations = [float(t['hours_held']) for t in trades]
    for trade in trades:
        reconstructed = trade['gross_pnl'] - trade['entry_fee'] - trade['exit_fee'] - trade['funding_debit']
        if not math.isclose(reconstructed, trade['net_pnl'], abs_tol=1e-8):
            raise AssertionError('Closed ledger does not reconcile')
    inventory = []
    for item in summary['open_positions']:
        arrival = engine.parse_time(item['entry_utc'])
        signal = prepared[symbol]['signals'][arrival]
        inventory.append({**item, 'direction': 'long' if signal.direction == 1 else 'short',
                          'hours_held_through_window_end': (end-arrival)/engine.HOUR})
    summary['open_positions'] = inventory
    summary['entries_total'] = len(trades) + len(inventory)
    summary['closed_net_pnl_usdt'] = summary['net_closed_pnl']
    summary['estimated_close_total_net_pnl_usdt'] = summary['estimated_close_equity'] - summary['initial_equity']
    summary['closed_trade_duration_hours'] = {
        'median': statistics.median(durations) if durations else None,
        'mean': statistics.mean(durations) if durations else None,
        'minimum': min(durations) if durations else None, 'maximum': max(durations) if durations else None,
        'less_than_24h_count': sum(h < 24 for h in durations),
        'at_least_24h_count': sum(h >= 24 for h in durations),
        'at_least_72h_count': sum(h >= 72 for h in durations),
        'warning': 'OHLC exit timestamp is hour-open; unknown actual duration may add up to one hour.',
    }
    summary['direction_diagnostics'] = {
        side: {'closed_trades': len(subset),
               'closed_net_pnl_usdt': sum(t['net_pnl'] for t in subset),
               'median_closed_duration_hours': statistics.median([t['hours_held'] for t in subset]) if subset else None,
               'open_positions': sum(t['direction'] == side for t in inventory)}
        for side in ('long', 'short')
        for subset in ([t for t in trades if t['direction'] == side],)
    }
    summary['win_loss_counts'] = {'wins': sum(t['net_pnl'] > 0 for t in trades),
                                  'losses': sum(t['net_pnl'] < 0 for t in trades)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('data/research/broad_mechanisms_20261007/inputs'))
    parser.add_argument('--config', type=Path, default=Path('config/stock_medium_large_swing_20261011.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new directory to preserve research evidence')
    config_bytes = args.config.read_bytes()
    profile = json.loads(config_bytes)
    base_path = Path(profile['base_config'])
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    if set(profile['symbol_profiles']) != set(SYMBOLS) or profile['signal_timeframe'] != '1h':
        raise ValueError('Expected the fixed three-stock hourly candidate')
    source_names = (Path(__file__).name, 'backtest_stock_swing_120.py', 'stock_swing_profiles.py',
                    'stock_swing_confirmation.py', 'stock_swing_signals.py', 'stock_mechanism_signals.py',
                    'stock_entry_quality.py', 'stock_research_entry_policy.py', 'replay_stock_swing_per_symbol.py')
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in source_names}
    snapshots = {}
    input_hashes = {}
    audits = {}
    for symbol in SYMBOLS:
        path = args.inputs / f'{symbol}.json.gz'
        input_hashes[symbol] = digest(path)
        snapshot = json.loads(gzip.decompress(path.read_bytes()))
        audits[symbol] = audit_source(snapshot, symbol)
        snapshots[symbol] = snapshot
    ends = {s['end_ms_exclusive'] for s in snapshots.values()}
    if len(ends) != 1:
        raise ValueError('Fixed common end required')
    end = ends.pop()
    start = end - 30*engine.DAY
    if any(s['symbols'][symbol]['start_ms'] > start for symbol, s in snapshots.items()):
        raise ValueError('Every stock needs the full fixed 30-day window')
    limitations = [
        'Behavior and risk regression only; known history was already reviewed. No unseen validation, search, selection, performance gate or profitability proof.',
        'Hourly trade/mark/index OHLC cannot reproduce production 5m entry-volume guards, observed bid/ask books, spread, 30s retries or partial fills. Previous completed 1h volume capped at 10% is only a common capacity proxy.',
        'Production 60-minute pending entry validity is degraded to one attempt at the next 1h open, explicitly set to zero minutes for this hourly engine.',
        'Both variants share identical 1h entry, exit and risk rules; the only strategy contrast is removal of the fixed completed-4h confirmation. The old 15m strategy is not compared because these inputs have no 15m history.',
        'All available earlier history seeds EMA/ATR, whereas production downloads a rolling prefix of about 500 hourly candles; seeding may differ near admission boundaries. Balances start flat at 1000 USDT each at the fixed recent-30-day boundary. Existing production positions and cash are not replayed.',
        'Fee and adverse slippage assumptions are applied to entry and exit; double cost doubles fees and slippage, not historical funding rates. Actual historical funding timestamps/rates are included, with hour-open mark proxies where settlement marks are absent.',
        'Unknown hourly intrabar order assumes liquidation stress before stop before target; adverse within-hour funding debits are charged before exit and ambiguous credits may be withheld. Gaps can exceed planned 0.25% stop risk.',
        'Drawdown is sampled at hourly opens/closes and can miss intrahour extremes. Exit-hour timestamps give duration lower bounds up to one hour short.',
        'Current quantity/tick/maintenance assumptions are used historically; no historical rule tiers, earnings or halt calendar, or real order-book execution is reconstructed.',
        'Estimated final close includes fees and slippage for open inventory; it is not a realized fill. These research outputs never update account state or place orders.',
    ]
    declaration = {'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'fixed_behavior_and_risk_regression_for_user_requested_prospective_paper_sampling',
        'signal_parameters_refit': False, 'parameter_search': False, 'performance_required_for_paper_sampling': False,
        'places_orders': False, 'window_start_utc': engine.iso(start), 'window_end_utc_exclusive': engine.iso(end),
        'initial_equity_usdt_each': 1000, 'cost_multipliers': [1, 2],
        'variants': ['fixed_candidate_4h_gate', 'same_1h_rules_without_4h_gate'],
        'candidate_config': profile, 'base_config': base,
        'input_sha256': input_hashes, 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'base_config_sha256': hashlib.sha256(base_bytes).hexdigest(), 'source_sha256': source_hashes,
        'input_audit': audits, 'limitations': limitations}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'declaration.json').write_text(json.dumps(declaration, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    outputs = {}
    combined = defaultdict(dict)
    effective_rules = {}
    signals = {}
    for symbol in SYMBOLS:
        snapshot = snapshots[symbol]
        production = production_rule(profile, base, symbol)
        candidate = effective_hourly_rule(production, symbol)
        if candidate.get('profit_exit_policy') is not None or candidate.get('trend_exit_policy') is not None:
            raise ValueError('Hourly baseline engine cannot reconstruct a separate exit policy')
        ungated, plain_data, candidate_data, counts, events = prepare_pair(snapshot, candidate, symbol, start, end)
        effective_rules[symbol] = {'production_rule_reference': production, 'hourly_candidate': candidate, 'hourly_ungated': ungated}
        signals[symbol] = counts
        (args.output_dir/f'{symbol}_4h_confirmation_events.json').write_text(json.dumps(events, indent=2)+'\n', encoding='utf-8')
        runs = {}
        for variant, cfg, data in (('fixed_candidate_4h_gate', candidate, candidate_data),
                                   ('same_1h_rules_without_4h_gate', ungated, plain_data)):
            for cost in (1, 2):
                name = f'{variant}_cost{cost}'
                result = engine.simulate(snapshot, cfg, start, end, cost, prepared_data=data)
                summary = describe(result, data, symbol, end)
                runs[name] = summary
                combined[name][symbol] = result
                engine.write_csv(args.output_dir/f'{symbol}_{name}_trades.csv', result['trades'])
                engine.write_csv(args.output_dir/f'{symbol}_{name}_equity.csv', result['equity_path'])
                print(symbol, name, json.dumps({key: summary[key] for key in (
                    'entries_total', 'closed_trades', 'closed_net_pnl_usdt',
                    'estimated_close_total_net_pnl_usdt', 'max_sampled_drawdown_pct',
                    'closed_trade_duration_hours', 'direction_diagnostics')}), flush=True)
        outputs[symbol] = runs
    current_sources = {name: digest(Path(__file__).with_name(name)) for name in source_names}
    if source_hashes != current_sources:
        raise RuntimeError('Research source changed during execution; rerun into a new directory')
    artifact = {**declaration, 'stock_runs': outputs, 'effective_rules': effective_rules,
                'signal_admission_diagnostics': signals,
                'combined_equal_capital_buckets': {name: combine_buckets(results, 3000) for name, results in combined.items()}}
    (args.output_dir/'results.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('SIGNAL_ADMISSION', json.dumps(signals), flush=True)
    print('COMBINED', json.dumps(artifact['combined_equal_capital_buckets']), flush=True)


if __name__ == '__main__':
    main()
