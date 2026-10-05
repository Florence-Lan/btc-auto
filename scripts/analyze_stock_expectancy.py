#!/usr/bin/env python3
"""Reconcile net expectancy from the fixed external-context CSV experiments."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from research_selective_stock_entries import MINIMUMS


def summarize(trades, initial, bootstrap):
    values = [float(t['net_pnl']) for t in trades]
    margins = [float(t['initial_margin']) for t in trades]
    returns = [float(t['net_return_initial_margin_pct']) for t in trades]
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    total = len(values)
    mean_win = statistics.mean(wins) if wins else None
    mean_loss = -statistics.mean(losses) if losses else None
    expectancy = statistics.mean(values) if values else None
    decomposition = (len(wins) / total * (mean_win or 0) - len(losses) / total * (mean_loss or 0)) if total else None
    if total:
        assert math.isclose(expectancy, decomposition, abs_tol=1e-9)
    interval = ([bootstrap['mean_trade_pnl_pct_initial_capital_' + suffix] * initial / 100
                 for suffix in ('p05', 'p95')] if bootstrap else None)
    return {'closed_trades': total, 'net_winners': len(wins), 'net_losers': len(losses),
        'flat_trades': total - len(wins) - len(losses),
        'net_win_rate_pct': len(wins) / total * 100 if total else None,
        'mean_winning_net_pnl_usdt': mean_win, 'mean_losing_net_loss_usdt': mean_loss,
        'mean_closed_net_pnl_usdt': expectancy, 'probability_weighted_net_pnl_usdt': decomposition,
        'net_pnl_usdt': sum(values),
        'mean_after_removing_best_winner_usdt': ((sum(values) - max(wins)) / (total - 1)
                                                if wins and total > 1 else None),
        'nonflat_break_even_win_rate_pct': mean_loss / (mean_win + mean_loss) * 100
                                          if mean_win and mean_loss else None,
        'mean_net_return_initial_margin_pct': statistics.mean(returns) if total else None,
        'margin_weighted_net_return_pct': sum(values) / sum(margins) * 100 if margins else None,
        'conditional_trade_block_resample_mean_p05_p95_usdt': interval,
        'future_expectancy_proven_positive': False}


def revised_failures(summary, metrics, window):
    reasons = []
    if metrics['closed_trades'] < MINIMUMS[window]:
        reasons.append('insufficient_closed_trade_sample')
    if metrics['mean_closed_net_pnl_usdt'] is None or metrics['mean_closed_net_pnl_usdt'] <= 0:
        reasons.append('nonpositive_sample_net_expectancy')
    if summary['estimated_close_return_pct'] <= 0:
        reasons.append('nonpositive_net_with_estimated_open_close')
    if summary['max_sampled_drawdown_pct'] > 6:
        reasons.append('drawdown_above_6pct')
    if summary['liquidation_stress_count']:
        reasons.append('liquidation_stress')
    if window == 'full':
        after_best = metrics['mean_after_removing_best_winner_usdt']
        if after_best is None or after_best <= 0:
            reasons.append('nonpositive_or_unassessable_without_best_winner')
    return reasons


def run(source, output):
    if output.exists():
        raise FileExistsError('Preserve previous reviews')
    original_path = source / 'results.json'
    artifact = json.loads(original_path.read_text())
    results, csv_hashes = {}, {}
    for symbol, bases in artifact['results'].items():
        results[symbol] = {}
        for base, overlays in bases.items():
            results[symbol][base] = {}
            for overlay, result in overlays.items():
                reviewed = {}
                for scenario, summary in result['runs'].items():
                    path = source / f'{symbol}_{base}_{overlay}_{scenario}_trades.csv'
                    csv_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with path.open(encoding='utf-8-sig', newline='') as handle:
                        trades = list(csv.DictReader(handle))
                    for t in trades:
                        net = float(t['net_pnl'])
                        assert math.isclose(net, float(t['gross_pnl']) - float(t['entry_fee']) - float(t['exit_fee']) - float(t['funding_debit']), abs_tol=1e-7)
                        assert math.isclose(float(t['net_return_initial_margin_pct']), net / float(t['initial_margin']) * 100, abs_tol=1e-7)
                    metrics = summarize(trades, summary['initial_equity'],
                        summary['trade_diagnostics']['historical_trade_block_bootstrap'])
                    assert len(trades) == summary['closed_trades']
                    assert math.isclose(metrics['net_pnl_usdt'], summary['net_closed_pnl'], abs_tol=1e-7)
                    reviewed[scenario] = {**metrics, 'estimated_open_close_net_pnl_usdt': summary['estimated_open_close_net_pnl'],
                        'failure_reasons_without_win_rate_gate': revised_failures(summary, metrics, scenario.rsplit('_cost', 1)[0])}
                results[symbol][base][overlay] = {'runs': reviewed,
                    'passes_revised_retrospective_gate': not any(s['failure_reasons_without_win_rate_gate'] for s in reviewed.values()),
                    'approved_for_new_entries': False}
    report = {'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'places_orders': False,
        'source_results_path': str(original_path), 'source_results_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'source_csv_sha256': csv_hashes,
        'criteria': {'minimum_closed_net_win_rate': None, 'minimum_net60_success_rate': None,
            'minimum_target_net_return_on_initial_margin': .6, 'minimum_sample_mean_closed_net_pnl_usdt_exclusive': 0,
            'minimum_closed_trades_by_window': MINIMUMS, 'both_normal_and_double_cost_required': True,
            'positive_net_with_estimated_open_close_required': True, 'positive_full_net_without_best_winner_required': True,
            'maximum_sampled_drawdown_pct': 6, 'no_modeled_liquidation': True},
        'results': results,
        'limitations': ['Post-hoc reassessment of already reviewed fixed candidates after user waives70% win-rate requirement; not a newly preregistered or unseen experiment.',
            'Sample net expectancy is a historical estimate, not a proven future expected return.',
            'Dollar expectancy and unweighted margin-return expectancy differ when position sizes vary; do not use the average margin to convert between them.',
            'Resample percentiles are conditional trade-block diagnostics, not a calibrated confidence interval or a forecast.',
            'Open positions are reported separately, never counted as winning closed trades. Full and chronological windows overlap.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('data/research/stock_external_confirmation_20261005'))
    parser.add_argument('--output', type=Path, default=Path('data/research/stock_expectancy_review_20261005/results.json'))
    args = parser.parse_args()
    result = run(args.source_dir, args.output)
    print(json.dumps({'reviewed_scenarios': len(result['source_csv_sha256']),
        'passing_stock_base_overlay_combinations': sum(v['passes_revised_retrospective_gate'] for b in result['results'].values() for o in b.values() for v in o.values()),
        'places_orders': False}))
