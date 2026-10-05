#!/usr/bin/env python3
"""Three preregistered entry hypotheses; isolated archived replay, no orders."""
from __future__ import annotations

import argparse
import bisect
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import backtest_stock_swing_120 as engine
from replay_stock_swing_per_symbol import after_close_summary
from research_stock_mechanisms import validate_ledger, volume_diagnostics
from research_stock_swing_robustness import audit_snapshot, trade_diagnostics
from stock_swing_signals import compute_indicators, signal_at, Signal, DEFAULT_CONFIG

MODES = ('breakout_1h_net60', 'trend_volume_breakout_net60', 'trend_volume_retest_net60')
MINIMUMS = {'development': 30, 'validation': 10, 'recent': 10, 'full': 50}


def trend_side(indicators, index, config):
    slope = config['ema_slope_bars']
    if index < max(config['ema_slow'] - 1, slope):
        return 0
    values = [indicators[k][index] for k in ('ema_fast', 'ema_mid', 'ema_slow')]
    values += [indicators[k][index - slope] for k in ('ema_mid', 'ema_slow')]
    if any(v is None for v in values):
        return 0
    fast, mid, slow, old_mid, old_slow = values
    if fast > mid > slow and mid > old_mid and slow > old_slow:
        return 1
    if fast < mid < slow and mid < old_mid and slow < old_slow:
        return -1
    return 0


def volume_confirmed(bars, index):
    """Known closed hour >=1.25x prior active-hour median; 4/6 hours active."""
    if index < 20:
        return False
    active = [b.volume for b in bars[index - 20:index] if b.volume > 0]
    return (len(active) >= 5 and sum(b.volume > 0 for b in bars[index - 5:index + 1]) >= 4
            and bars[index].volume > 0 and bars[index].volume >= 1.25 * statistics.median(active))


def completed_higher_index(closing_times, signal_close):
    return bisect.bisect_right(closing_times, signal_close) - 1


def retest_confirmed(bars, index, setup, direction, boundary, atr):
    """A later bar retests the boundary and closes beyond the preceding bar."""
    if not 1 <= index - setup <= 6:
        return False
    bar, previous = bars[index], bars[index - 1]
    wick = bar.low if direction == 1 else bar.high
    previous_extreme = previous.high if direction == 1 else previous.low
    distance = direction * (bar.close - boundary)
    return (direction * (wick - boundary) <= .25 * atr
            and direction * (wick - boundary) >= -.5 * atr
            and 0 < distance <= atr
            and direction * (bar.close - previous_extreme) > 0)


def build_signals(hourly, higher, config, mode):
    if mode not in MODES:
        raise ValueError('Unknown declared hypothesis')
    config = {**DEFAULT_CONFIG, **config}
    indicators = compute_indicators(hourly, config)
    higher_indicators = compute_indicators(higher, config)
    higher_ends = [b.time_ms + engine.FOUR_HOURS for b in higher]
    seeds = {i: s for i in range(len(hourly))
             if (s := signal_at(hourly, indicators, i, config)) is not None}
    signals, checkpoints, used = {}, [], set()
    for index, bar in enumerate(hourly):
        end = bar.time_ms + engine.HOUR
        hi = completed_higher_index(higher_ends, end)
        side = trend_side(higher_indicators, hi, config)
        seed_index = index
        signal = seeds.get(index)
        if mode != MODES[0]:
            if not volume_confirmed(hourly, index) or not side:
                continue
            if mode == MODES[1]:
                if signal is None or signal.direction != side:
                    continue
            else:
                signal = None
                if trend_side(indicators, index, config) != side:
                    continue
                for setup in range(index - 1, max(-1, index - 7), -1):
                    seed = seeds.get(setup)
                    if seed is None or setup in used or seed.direction != side:
                        continue
                    setup_hi = completed_higher_index(higher_ends, hourly[setup].time_ms + engine.HOUR)
                    if trend_side(higher_indicators, setup_hi, config) != side or not volume_confirmed(hourly, setup):
                        continue
                    prior = hourly[setup - config['breakout_bars']:setup]
                    boundary = max(b.high for b in prior) if side == 1 else min(b.low for b in prior)
                    atr = indicators['atr'][index]
                    if atr is None or atr <= 0 or atr / bar.close > config['max_atr_fraction']:
                        continue
                    if retest_confirmed(hourly, index, setup, side, boundary, seed.atr):
                        signal = Signal(side, atr, bar.close, abs(bar.close - boundary) / atr, bar.time_ms)
                        seed_index = setup
                        used.add(setup)
                        break
        if signal is None:
            continue
        # Every retry preserves the same closed signal; next-hour future data is invisible.
        for attempt in range(end, end + engine.HOUR, 300_000):
            signals[attempt] = signal
        checkpoints.append({'signal_close_ms': end, 'signal_bar_open_ms': bar.time_ms,
            'setup_close_ms': hourly[seed_index].time_ms + engine.HOUR,
            'higher_close_ms': higher_ends[hi] if hi >= 0 else None,
            'direction': signal.direction, 'volume_confirmed': volume_confirmed(hourly, index),
            'higher_direction': side})
    return signals, checkpoints


def wilson_interval(successes, total):
    """Descriptive binomial interval; serially dependent trades violate IID."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius) * 100, min(1, center + radius) * 100]


def outcome_metrics(trades):
    count = len(trades)
    wins = sum(t['net_pnl'] > 0 for t in trades)
    hits = sum(t['net_return_initial_margin_pct'] >= 60 - 1e-7 for t in trades)
    return {'closed_trades': count, 'net_winners': wins, 'net60_trades': hits,
        'net_win_rate_pct': wins / count * 100 if count else None,
        'net60_success_rate_pct': hits / count * 100 if count else None,
        'profitable_trades_below_net60': wins - hits,
        'net_win_rate_wilson95_pct': wilson_interval(wins, count),
        'net60_rate_wilson95_pct': wilson_interval(hits, count),
        'mean_net_return_initial_margin_pct': statistics.mean(t['net_return_initial_margin_pct'] for t in trades) if count else None}


def qualification_failures(summary, minimum):
    m = summary['user_outcomes']
    errors = []
    if m['closed_trades'] < minimum:
        errors.append('insufficient_closed_trade_sample')
    if m['net_win_rate_pct'] is None or m['net_win_rate_pct'] < 70:
        errors.append('net_win_rate_below_70pct')
    if m['net60_success_rate_pct'] is None or m['net60_success_rate_pct'] < 70:
        errors.append('net60_success_rate_below_70pct')
    if m['profitable_trades_below_net60']:
        errors.append('some_winning_trades_below_net60')
    if summary['net_closed_pnl'] <= 0 or summary['estimated_close_return_pct'] <= 0:
        errors.append('nonpositive_net_profit')
    if summary['max_sampled_drawdown_pct'] > 6:
        errors.append('drawdown_above_6pct')
    if summary['liquidation_stress_count']:
        errors.append('liquidation_stress')
    return errors


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(snapshot_path, output):
    if output.exists():
        raise FileExistsError('Preserve evidence: use a new output directory')
    base_path = Path('config/stock_swing_120_candidate_20261004.json')
    base = json.loads(base_path.read_text())
    common = {**base, 'ema_fast': 20, 'ema_mid': 50, 'ema_slow': 200,
        'warmup_bars': 200, 'signal_timeframe': '1h', 'execution_timeframe': '5m',
        'entry_signal_validity_minutes': 60, 'entry_direction': 'both',
        'min_stop_fraction': .01, 'target_margin_return': .6, 'max_holding_calendar_days': 7,
        'risk_fraction_per_trade': .0025, 'initial_equity_usdt': 1000,
        'max_positions': 1, 'entry_max_previous_bar_participation_fraction': .1,
        'cooldown_signal_bars': 3}
    snapshot_hash = digest(snapshot_path)
    output.mkdir(parents=True)
    sources = ['research_selective_stock_entries.py', 'backtest_stock_swing_120.py',
        'stock_swing_signals.py', 'stock_swing_profiles.py', 'stock_research_entry_policy.py',
        'research_stock_mechanisms.py', 'research_stock_swing_robustness.py', 'replay_stock_swing_per_symbol.py']
    declaration = {'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'places_orders': False, 'snapshot_path': str(snapshot_path), 'snapshot_sha256': snapshot_hash,
        'source_sha256': {s: digest(Path(__file__).with_name(s)) for s in sources},
        'base_config_path': str(base_path), 'base_config_sha256': digest(base_path),
        'common_config': common, 'modes': list(MODES), 'minimum_closed_trades': MINIMUMS,
        'rules': {'control': '1h EMA20/50/200 breakout of prior20-hour extrema, rising/falling mid EMA; 60% net initial-margin target.',
            'trend_volume': 'Same breakout plus most recent CLOSED 4h EMA20/50/200 ordering, mid/slow slopes over6 bars; closed hour volume >=1.25x median prior20 active hours (>=5); >=4 of last6 hours active.',
            'retest': 'Same trend/volume at setup and confirmation; within next1..6 closed hours retest boundary wick in [-0.5,+0.25] setup ATR directional distance, close on breakout side within1ATR and beyond prior hour extreme. Current 1h trend also agrees; consume each setup once.'},
        'fees': {'MUUSDT': .000125, 'SNDKUSDT': .000125, 'SKHYNIXUSDT': .001},
        'fee_source': 'https://docs.asterdex.com/trading/perpetuals/fees-and-specs/fees',
        'costs': 'Normal fees plus2bp adverse slippage/side and historical funding; cost2 doubles fees/slippage, preserves funding. Current fees on old paths are scenarios, not historical fee reconstruction.',
        'window_policy': 'Development before Sep1; validation Sep1..Sep20; recent Sep20..snapshot cutoff; full is overlapping diagnostic. All THREE declared arms evaluated, no tuning, ranking or later-window fallback. Require EACH chronological window and EACH cost to pass.',
        'gate': 'Minimum samples 30 development/10 validation/10 recent/50 full; net win rate>=70%, actual net60 success rate>=70%, no winning trade below net60, positive closed and estimated net liquidation PnL, DD<=6%, no liquidation. 70% actual net60 success is a stronger joint-outcome criterion, reported separately from ordinary win rate.',
        'deployment': 'No retrospective replay automatically enables entries. No unseen or forward-validation claim. Report zero-volume fills, both directions, outlier removal and open inventory.',
        'known_history_previously_reviewed': True, 'independent_unseen_holdout': False,
        'limitations': ['5m OHLC cannot reconstruct historical depth, fill latency, partial fills or exact intrabar order; stop-first convention retained.',
            'Index uses already observable hour-open proxy; funding settlement marks use recorded engine approximations.',
            'Wilson intervals assume IID and are descriptive; serial dependence, selection and prior history review invalidate predictive interpretation.',
            'Windows start with fresh cash and no inventory, retain causal indicator warmup; full continuous replay separately reports inventory crossing boundaries.',
            'Declared filters may leave too few observations to estimate success reliably.']}
    (output / 'declaration.json').write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + '\n')
    # The declaration is persisted BEFORE loading or evaluating candle outcomes.
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    data_audit = audit_snapshot(snapshot)
    end = snapshot['end_ms_exclusive']
    split = engine.parse_time('2026-09-01T00:00:00Z')
    later = engine.parse_time('2026-09-20T00:00:00Z')
    if not split < later < end:
        raise ValueError('Snapshot must cover every declared chronological window')
    results = {}
    for symbol, source in snapshot['symbols'].items():
        if symbol not in declaration['fees']:
            continue
        one = {**snapshot, 'symbols': {symbol: source}}
        cfg = {**common, 'symbols': [symbol], 'taker_fee_rate_assumption': declaration['fees'][symbol]}
        # Reuse audited market/funding preparation; replace only the signal map.
        prepared = engine.prepare(one, {**cfg, 'symbol_profiles': {symbol: cfg}})
        hourly = [engine.candle(r) for r in source['trade_1h']]
        higher = engine.aggregate_4h(hourly)
        results[symbol] = {}
        for mode in MODES:
            signals, checkpoints = build_signals(hourly, higher, cfg, mode)
            modified = {symbol: {**prepared[symbol], 'signals': signals}}
            (output / f'{symbol}_{mode}_signal_checkpoints.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
            runs = {}
            windows = {'development': (source['start_ms'], split), 'validation': (split, later),
                       'recent': (later, end), 'full': (source['start_ms'], end)}
            for window, (start, finish) in windows.items():
                for cost in (1, 2):
                    result = engine.simulate(one, cfg, start, finish, cost, prepared_data=modified)
                    validate_ledger(result, cfg, source, cost)
                    summary = after_close_summary(result)
                    summary['user_outcomes'] = outcome_metrics(result['trades'])
                    summary['trade_diagnostics'] = trade_diagnostics(result['trades'], 1000, samples=500)
                    volume = volume_diagnostics(result, source)
                    summary['execution_volume_diagnostics'] = {k: v for k, v in volume.items() if k != 'closed_trade_bar_checks'}
                    summary['failure_reasons'] = qualification_failures(summary, MINIMUMS[window])
                    runs[f'{window}_cost{cost}'] = summary
                    ledger_path = output / f'{symbol}_{mode}_{window}_cost{cost}_trades.csv'
                    engine.write_csv(ledger_path, result['trades'])
                    if not result['trades']:
                        ledger_path.write_text('net_pnl,net_return_initial_margin_pct\n')
                    print(symbol, mode, window, cost, summary['user_outcomes'],
                          'net', round(summary['net_closed_pnl'], 4), flush=True)
            results[symbol][mode] = {'runs': runs, 'unique_closed_signal_count': len(checkpoints),
                'passes_retrospective_joint_screen': not any(r['failure_reasons'] for r in runs.values()),
                'forward_validated': False, 'approved_for_new_simulated_entries': False}
            artifact = {'declaration': declaration, 'data_audit': data_audit, 'end_utc_exclusive': engine.iso(end),
                'results': results, 'places_orders': False, 'active_strategy_changed': False}
            (output / 'results.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    assert digest(snapshot_path) == snapshot_hash
    return artifact


def verify(output):
    artifact = json.loads((output / 'results.json').read_text())
    declaration = json.loads((output / 'declaration.json').read_text())
    assert artifact['declaration'] == declaration
    for name, expected in declaration['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == expected, name
    assert digest(declaration['snapshot_path']) == declaration['snapshot_sha256']
    assert digest(declaration['base_config_path']) == declaration['base_config_sha256']
    scenarios = count = signals = 0
    for symbol, modes in artifact['results'].items():
        assert set(modes) == set(MODES)
        for mode, result in modes.items():
            checkpoints = json.loads((output / f'{symbol}_{mode}_signal_checkpoints.json').read_text())
            used = set()
            for point in checkpoints:
                end = point['signal_close_ms']
                assert point['signal_bar_open_ms'] + engine.HOUR == end
                assert point['setup_close_ms'] <= end
                if mode != MODES[0]:
                    assert point['higher_close_ms'] <= end
                    assert point['higher_direction'] == point['direction']
                    assert point['volume_confirmed']
                if mode == MODES[2]:
                    assert engine.HOUR <= end - point['setup_close_ms'] <= 6 * engine.HOUR
                    assert point['setup_close_ms'] not in used
                    used.add(point['setup_close_ms'])
                signals += 1
            for scenario, summary in result['runs'].items():
                with (output / f'{symbol}_{mode}_{scenario}_trades.csv').open(encoding='utf-8-sig', newline='') as handle:
                    trades = list(csv.DictReader(handle))
                normalized = []
                for trade in trades:
                    net = float(trade['net_pnl'])
                    assert math.isclose(net, float(trade['gross_pnl']) - float(trade['entry_fee']) - float(trade['exit_fee']) - float(trade['funding_debit']), abs_tol=1e-7)
                    assert math.isclose(float(trade['net_return_initial_margin_pct']), net / float(trade['initial_margin']) * 100, abs_tol=1e-7)
                    assert engine.parse_time(trade['signal_utc']) <= engine.parse_time(trade['entry_utc'])
                    normalized.append({'net_pnl': net, 'net_return_initial_margin_pct': float(trade['net_return_initial_margin_pct'])})
                metrics = outcome_metrics(normalized)
                assert metrics == summary['user_outcomes']
                assert len(trades) == summary['closed_trades']
                assert math.isclose(sum(t['net_pnl'] for t in normalized), summary['net_closed_pnl'], abs_tol=1e-7)
                window = scenario.rsplit('_cost', 1)[0]
                assert summary['failure_reasons'] == qualification_failures(summary, MINIMUMS[window])
                scenarios += 1
                count += len(trades)
            assert result['passes_retrospective_joint_screen'] == (not any(s['failure_reasons'] for s in result['runs'].values()))
            assert result['approved_for_new_simulated_entries'] is False
    assert scenarios == 72
    verification = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'csv_scenarios': scenarios,
        'overlapping_ledger_records': count, 'unique_closed_signal_checkpoints': signals,
        'source_and_snapshot_hashes_match': True, 'net_accounting_and_user_metrics_match': True,
        'multi_timeframe_closure_and_one_retest_per_setup_checked': True,
        'records_are_not_independent_trades': True, 'places_orders': False}
    (output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output_dir)))
    else:
        run(args.snapshot, args.output_dir)
