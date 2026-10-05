#!/usr/bin/env python3
"""Preregistered NQ/sector/macro entry ablation; no account writes or orders."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import backtest_stock_swing_120 as engine
import research_selective_stock_entries as selective
from replay_stock_swing_per_symbol import after_close_summary
from research_stock_mechanisms import validate_ledger, volume_diagnostics
from research_stock_swing_robustness import audit_snapshot, trade_diagnostics
from stock_external_context import MarketContext, OVERLAYS, MAX_AGE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(snapshot_path, context_path, prior_path, output):
    if output.exists():
        raise FileExistsError('Preserve evidence: choose a new output directory')
    prior = json.loads(prior_path.read_text())
    common = prior['declaration']['common_config']
    sources = ['research_stock_external_context.py', 'stock_external_context.py',
        *prior['declaration']['source_sha256']]
    for name, expected in prior['declaration']['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == expected, name
    assert digest(snapshot_path) == prior['declaration']['snapshot_sha256']
    output.mkdir(parents=True)
    declaration = {'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'places_orders': False, 'active_strategy_changed': False,
        'stock_snapshot_path': str(snapshot_path), 'stock_snapshot_sha256': digest(snapshot_path),
        'context_snapshot_path': str(context_path), 'context_snapshot_sha256': digest(context_path),
        'prior_result_path': str(prior_path), 'prior_result_sha256': digest(prior_path),
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in sources},
        'base_modes': list(selective.MODES), 'overlays': list(OVERLAYS),
        'common_config': common, 'fee_assumptions': prior['declaration']['fees'],
        'stock_entry_rules': prior['declaration']['rules'],
        'external_rules': {'price_only': 'Exact prior control, no external admission.',
            'nq_confirmation': 'Most recent available NQ hourly EMA20>EMA50 with rising EMA50 over6 observations and positive6-observation return for longs; mirror for shorts.',
            'nq_sector_risk_confirmation': 'Require NQ, ES and SOX agreeing with stock signal. VIX<=25 and latest3-observation rise<=10%. Longs vetoed if dollar6-observation return>0.2% AND yield10 relative6-observation rise>0.5%; both trade directions retained.'},
        'availability': 'Every five-minute ENTRY RETRY checks external data at its own timestamp, using only rows with provider open +1h +20min <= retry time. Not signal-close-only filtering.',
        'maximum_age_ms': MAX_AGE, 'missing_or_stale_policy': 'Block affected external overlay; no proxy substitution or neutral imputation.',
        'evaluation_policy': prior['declaration']['window_policy'],
        'qualification_gate': prior['declaration']['gate'],
        'known_stock_history_previously_reviewed': True,
        'historical_external_first_seen_available': False, 'independent_unseen_holdout': False,
        'limitations': ['Newest historical external vintage and assumed20min publication delay are not exact historical first-seen records.',
            'NQ/ES front-contract rolls and source revisions are not reconstructed; no NQ trade-return claim.',
            'Six external factors can overlap; a same-direction vote is not an independent probability estimate.',
            'Original stock execution/funding/index proxies and zero-volume fill limitations retained.',
            'Every arm and time window is declared; no tuning or selecting a later-window winner.'],
        'deployment': 'No retrospective result changes active admission. Prospective unseen validation is still required.'}
    (output / 'declaration.json').write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + '\n')
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    context_payload = json.loads(gzip.decompress(context_path.read_bytes()))
    context = MarketContext(context_payload)
    data_audit = audit_snapshot(snapshot)
    end = snapshot['end_ms_exclusive']
    split = engine.parse_time('2026-09-01T00:00:00Z')
    recent = engine.parse_time('2026-09-20T00:00:00Z')
    results = {}
    decision_cache = {}
    for symbol, source in snapshot['symbols'].items():
        if symbol not in declaration['fee_assumptions']:
            continue
        one = {**snapshot, 'symbols': {symbol: source}}
        cfg = {**common, 'symbols': [symbol], 'taker_fee_rate_assumption': declaration['fee_assumptions'][symbol]}
        prepared = engine.prepare(one, {**cfg, 'symbol_profiles': {symbol: cfg}})
        hourly = [engine.candle(r) for r in source['trade_1h']]
        higher = engine.aggregate_4h(hourly)
        results[symbol] = {}
        windows = {'development': (source['start_ms'], split), 'validation': (split, recent),
                   'recent': (recent, end), 'full': (source['start_ms'], end)}
        for base_mode in selective.MODES:
            signals, checkpoints = selective.build_signals(hourly, higher, cfg, base_mode)
            results[symbol][base_mode] = {}
            for overlay in OVERLAYS:
                accepted, reasons, checks = {}, Counter(), []
                for timestamp, signal in signals.items():
                    if timestamp >= end:
                        continue
                    key = (timestamp, signal.direction, overlay)
                    if key not in decision_cache:
                        decision_cache[key] = context.decision(*key)
                    decision = decision_cache[key]
                    reasons.update(decision['reasons'])
                    if decision['allowed']:
                        accepted[timestamp] = signal
                        checks.append({'entry_retry_ms': timestamp, 'signal_bar_open_ms': signal.time_ms,
                            'direction': signal.direction,
                            'factor_available_ms': {name: point['available_ms'] for name, point in decision['factors'].items()}})
                name = f'{symbol}_{base_mode}_{overlay}'
                (output / (name + '_accepted_retries.json')).write_text(json.dumps(checks, indent=2) + '\n')
                modified = {symbol: {**prepared[symbol], 'signals': accepted,
                    'first_signal_time': min(accepted, default=None)}}
                runs = {}
                for window, (start, finish) in windows.items():
                    for cost in (1, 2):
                        result = engine.simulate(one, cfg, start, finish, cost, prepared_data=modified)
                        validate_ledger(result, cfg, source, cost)
                        summary = after_close_summary(result)
                        summary['user_outcomes'] = selective.outcome_metrics(result['trades'])
                        summary['trade_diagnostics'] = trade_diagnostics(result['trades'], 1000, samples=500)
                        volumes = volume_diagnostics(result, source)
                        summary['execution_volume_diagnostics'] = {k: v for k, v in volumes.items() if k != 'closed_trade_bar_checks'}
                        summary['failure_reasons'] = selective.qualification_failures(summary, selective.MINIMUMS[window])
                        scenario = f'{window}_cost{cost}'
                        runs[scenario] = summary
                        path = output / f'{name}_{scenario}_trades.csv'
                        engine.write_csv(path, result['trades'])
                        if not result['trades']:
                            path.write_text('net_pnl,net_return_initial_margin_pct\n')
                        if overlay == 'price_only':
                            control = prior['results'][symbol][base_mode]['runs'][scenario]
                            assert summary['user_outcomes'] == control['user_outcomes']
                            for field in ('net_closed_pnl', 'estimated_close_return_pct', 'max_sampled_drawdown_pct'):
                                assert math.isclose(summary[field], control[field], abs_tol=1e-7)
                        print(symbol, base_mode, overlay, scenario,
                            'trades', summary['closed_trades'], 'win', summary['user_outcomes']['net_win_rate_pct'],
                            'net60', summary['user_outcomes']['net60_success_rate_pct'], 'net', round(summary['net_closed_pnl'], 4), flush=True)
                results[symbol][base_mode][overlay] = {'runs': runs,
                    'base_closed_signal_count': len(checkpoints), 'accepted_entry_retries': len(accepted),
                    'blocked_entry_retry_reasons': dict(reasons),
                    'passes_retrospective_joint_screen': not any(r['failure_reasons'] for r in runs.values()),
                    'forward_validated': False, 'approved_for_new_simulated_entries': False}
                artifact = {'declaration': declaration, 'external_data_coverage': context_payload['coverage'],
                    'data_audit': data_audit, 'results': results, 'places_orders': False,
                    'active_strategy_changed': False}
                (output / 'results.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    return artifact


def verify(directory):
    artifact = json.loads((directory / 'results.json').read_text())
    declaration = json.loads((directory / 'declaration.json').read_text())
    assert artifact['declaration'] == declaration
    for name, expected in declaration['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for kind in ('stock_snapshot', 'context_snapshot', 'prior_result'):
        assert digest(declaration[kind + '_path']) == declaration[kind + '_sha256']
    payload = json.loads(gzip.decompress(Path(declaration['context_snapshot_path']).read_bytes()))
    context = MarketContext(payload)
    scenarios = ledger_records = entry_retries = 0
    for symbol, bases in artifact['results'].items():
        assert set(bases) == set(selective.MODES)
        for base, overlays in bases.items():
            assert set(overlays) == set(OVERLAYS)
            for overlay, result in overlays.items():
                name = f'{symbol}_{base}_{overlay}'
                retries = json.loads((directory / (name + '_accepted_retries.json')).read_text())
                accepted = {r['entry_retry_ms']: r for r in retries}
                assert len(accepted) == len(retries) == result['accepted_entry_retries']
                for retry in retries:
                    timestamp = retry['entry_retry_ms']
                    decision = context.decision(timestamp, retry['direction'], overlay)
                    assert decision['allowed']
                    assert retry['signal_bar_open_ms'] + engine.HOUR <= timestamp
                    assert {k: p['available_ms'] for k, p in decision['factors'].items()} == retry['factor_available_ms']
                    for name_, point in decision['factors'].items():
                        assert point['provider_open_ms'] + engine.HOUR + 20 * 60_000 == point['available_ms'] <= timestamp
                        assert timestamp - point['available_ms'] <= MAX_AGE[name_]
                for scenario, summary in result['runs'].items():
                    with (directory / f'{name}_{scenario}_trades.csv').open(encoding='utf-8-sig', newline='') as handle:
                        trades = list(csv.DictReader(handle))
                    normalized = []
                    for trade in trades:
                        net = float(trade['net_pnl'])
                        assert math.isclose(net, float(trade['gross_pnl']) - float(trade['entry_fee']) - float(trade['exit_fee']) - float(trade['funding_debit']), abs_tol=1e-7)
                        assert math.isclose(float(trade['net_return_initial_margin_pct']), net / float(trade['initial_margin']) * 100, abs_tol=1e-7)
                        timestamp = engine.parse_time(trade['entry_utc'])
                        side = 1 if trade['direction'] == 'long' else -1
                        assert timestamp in accepted and accepted[timestamp]['direction'] == side
                        assert context.decision(timestamp, side, overlay)['allowed']
                        normalized.append({'net_pnl': net, 'net_return_initial_margin_pct': float(trade['net_return_initial_margin_pct'])})
                    assert selective.outcome_metrics(normalized) == summary['user_outcomes']
                    assert len(trades) == summary['closed_trades']
                    assert math.isclose(sum(t['net_pnl'] for t in normalized), summary['net_closed_pnl'], abs_tol=1e-7)
                    window = scenario.rsplit('_cost', 1)[0]
                    assert selective.qualification_failures(summary, selective.MINIMUMS[window]) == summary['failure_reasons']
                    ledger_records += len(trades)
                    scenarios += 1
                assert result['passes_retrospective_joint_screen'] == (not any(s['failure_reasons'] for s in result['runs'].values()))
                assert result['approved_for_new_simulated_entries'] is False
                entry_retries += len(retries)
    assert scenarios == 216
    verification = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'csv_scenarios': scenarios,
        'overlapping_ledger_records': ledger_records, 'accepted_entry_retries_rechecked': entry_retries,
        'all_closed_entries_pass_asof_external_gate': True, 'source_and_data_hashes_match': True,
        'prior_price_only_controls_reproduced': True, 'net_accounting_and_user_metrics_match': True,
        'records_are_not_independent_trades': True, 'places_orders': False}
    (directory / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz'))
    parser.add_argument('--context', type=Path, default=Path('data/research/stock_external_context_20261005/market_context.json.gz'))
    parser.add_argument('--prior', type=Path, default=Path('data/research/stock_selective_entries_20261005/results.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output_dir)))
    else:
        run(args.snapshot, args.context, args.prior, args.output_dir)
