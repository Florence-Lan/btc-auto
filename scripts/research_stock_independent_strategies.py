#!/usr/bin/env python3
"""Declared, per-stock short-cycle comparison. No account client or order API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import backtest_stock_swing_120 as engine
from replay_stock_swing_per_symbol import after_close_summary
from research_stock_mechanisms import validate_ledger
from research_stock_swing_robustness import audit_snapshot

FAMILIES = {
    'breakout_8_24_60': {'signal_family': 'breakout', 'ema_fast': 8, 'ema_mid': 24,
        'ema_slow': 60, 'warmup_bars': 60, 'breakout_bars': 6, 'stop_atr': 2.0},
    'cross_8_24_60': {'signal_family': 'ema_transition', 'ema_fast': 8, 'ema_mid': 24,
        'ema_slow': 60, 'warmup_bars': 60, 'require_price_slow': True, 'stop_atr': 2.0},
    'cross_12_36_120': {'signal_family': 'ema_transition', 'ema_fast': 12, 'ema_mid': 36,
        'ema_slow': 120, 'warmup_bars': 120, 'require_price_slow': False, 'stop_atr': 2.0},
}
EXITS = {
    'swing_net120': {'target_margin_return': 1.2, 'max_holding_calendar_days': 15,
        'trail_activation_underlying_return': .08, 'trail_locked_underlying_return': .03, 'trail_atr': 2.0},
    'tactical_net30': {'target_margin_return': .3, 'max_holding_calendar_days': 3,
        'trail_activation_underlying_return': .02, 'trail_locked_underlying_return': .005, 'trail_atr': 2.0},
}


def variants():
    return {f'{family}_{period}_{direction}_{exit_name}': {
        **settings, **exits, 'signal_timeframe': period, 'entry_direction': direction,
        'entry_signal_validity_minutes': 5 if period == '5m' else 15, 'cooldown_signal_bars': 3,
        'research_variant': f'{family}_{period}_{direction}_{exit_name}'}
        for family, settings in FAMILIES.items() for period in ('5m', '15m')
        for direction in ('both', 'long') for exit_name, exits in EXITS.items()}


def closed_return(summary):
    return summary['net_closed_pnl'] / summary['initial_equity'] * 100


def screen(runs, prefix, minimum):
    failures = []
    for cost in (1, 2):
        s = runs[f'{prefix}_cost{cost}']
        if closed_return(s) <= 0: failures.append(f'{prefix}_cost{cost}: nonpositive closed net return')
        if s['closed_trades'] < minimum: failures.append(f'{prefix}_cost{cost}: too few closed trades')
        if s['max_sampled_drawdown_pct'] > 8: failures.append(f'{prefix}_cost{cost}: drawdown exceeds 8%')
        if s['liquidation_stress_count']: failures.append(f'{prefix}_cost{cost}: liquidation stress')
    return failures


def select_on_development(experiments):
    eligible = [(min(closed_return(v['runs'][f'development_cost{c}']) for c in (1, 2)), key)
        for key, v in experiments.items() if not screen(v['runs'], 'development', 5)]
    return max(eligible)[1] if eligible else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path,
        default=Path('data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError('Use a new output directory to preserve evidence')
    plan = json.loads(Path('config/parallel_simulation_plan_20261005.json').read_text())
    profile = json.loads(Path(plan['stock_research_profile']).read_text())
    base = json.loads(Path(profile['base_config']).read_text())
    choices = variants()
    args.output_dir.mkdir(parents=True)
    declaration = {'declared_at_utc': datetime.now(timezone.utc).isoformat(), 'places_orders': False,
        'variants': choices, 'variant_count_each': len(choices), 'initial_equity_each': 1000,
        'objective': 'Independent closed net profit after fees, slippage and funding; no fixed net120 acceptance requirement.',
        'selection': 'Development only: >=5 closed trades, positive closed net PnL at both costs, max drawdown <=8%, no liquidation. Rank worst-cost development closed net return. Freeze winner before considering audit windows; do not substitute a validation winner.',
        'audit': 'Selected winner needs >=3 closed validation trades, positive validation and recent30d closed net PnL at both costs, <=8% drawdown, no liquidation. No minimum 120% margin-target count.',
        'independent_unseen_holdout': False, 'known_history_previously_reviewed': True,
        'snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('research_stock_independent_strategies.py', 'backtest_stock_swing_120.py',
                         'stock_swing_signals.py', 'stock_swing_profiles.py')},
        'limitations': ['Known retrospective history; multiple comparisons may overfit.',
            'Five-minute trade/mark paths and hourly index openings; no historical bid/ask depth or partial fills.',
            'Intrabar liquidation/stop/target ordering conservative; funding settlement marks may use five-minute open proxy.',
            'Previous completed 5m volume caps entry; no proof of real execution. A 5m signal replay has one entry attempt per signal, whereas paper observes every 30 seconds.',
            'No result guarantees profitability. Keep independent forward rule revisions and ledgers.']}
    (args.output_dir/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    snapshot = json.loads(gzip.decompress(args.snapshot.read_bytes()))
    audit_snapshot(snapshot)
    end = snapshot['end_ms_exclusive']
    split = engine.parse_time(base['evaluation_split_utc'])
    all_results = {}
    for symbol in base['symbols']:
        source = snapshot['symbols'][symbol]
        one_snapshot = {**snapshot, 'symbols': {symbol: source}}
        common = {**base, 'symbols': [symbol], 'initial_equity_usdt': 1000, 'max_positions': 1,
            'risk_fraction_per_trade': .005, 'execution_timeframe': '5m',
            'entry_max_previous_bar_participation_fraction': .1}
        experiments = {}
        cache = {}
        for name, overrides in choices.items():
            cfg = {**common, **overrides, 'symbol_profiles': {symbol: overrides}}
            cache_key = (overrides['signal_family'], overrides['ema_fast'], overrides['signal_timeframe'])
            if cache_key not in cache: cache[cache_key] = engine.prepare(one_snapshot, cfg)
            prepared = {symbol: {**cache[cache_key][symbol], 'profile_config': cfg}}
            runs = {}
            for window, start, finish in (('development', source['start_ms'], split), ('validation', split, end)):
                for cost in (1, 2):
                    label = f'{window}_cost{cost}'
                    replay = engine.simulate(one_snapshot, cfg, start, finish, cost, prepared_data=prepared)
                    validate_ledger(replay, cfg, source, cost)
                    runs[label] = after_close_summary(replay)
                    engine.write_csv(args.output_dir/f'{symbol}_{name}_{label}_trades.csv', replay['trades'])
            experiments[name] = {'settings': overrides, 'runs': runs}
            print(symbol, name, 'dev_cost2', round(closed_return(runs['development_cost2']), 4),
                  'audit_cost2', round(closed_return(runs['validation_cost2']), 4), flush=True)
        selected = select_on_development(experiments)
        audit_failures = []
        if selected:
            settings = choices[selected]
            cfg = {**common, **settings, 'symbol_profiles': {symbol: settings}}
            prepared = {symbol: {**cache[(settings['signal_family'], settings['ema_fast'], settings['signal_timeframe'])][symbol], 'profile_config': cfg}}
            runs = experiments[selected]['runs']
            for window, start in (('full', source['start_ms']), ('recent30d', max(source['start_ms'], end-30*engine.DAY))):
                for cost in (1, 2):
                    label = f'{window}_cost{cost}'
                    replay = engine.simulate(one_snapshot, cfg, start, end, cost, prepared_data=prepared)
                    validate_ledger(replay, cfg, source, cost)
                    runs[label] = after_close_summary(replay)
                    engine.write_csv(args.output_dir/f'{symbol}_{selected}_{label}_trades.csv', replay['trades'])
            audit_failures = screen(runs, 'validation', 3)+screen(runs, 'recent30d', 1)
        all_results[symbol] = {'experiments': experiments, 'development_selected': selected,
            'selected_passes_retrospective_audit': bool(selected) and not audit_failures,
            'audit_failure_reasons': audit_failures, 'forward_validated': False}
        artifact = {'declaration': declaration, 'data_end_utc_exclusive': engine.iso(end),
                    'stock_results': all_results, 'places_orders': False, 'forward_validated': False}
        (args.output_dir/'results.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n')
        print('SELECTED', symbol, selected, 'audit_failures', audit_failures, flush=True)


if __name__ == '__main__':
    main()
