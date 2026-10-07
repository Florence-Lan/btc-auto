"""Audit one predeclared exit candidate against the captured 30-day control.

Known, previously reviewed history; chronological subwindows are diagnostics,
not independent holdouts. Never tune parameters or modify account ledgers.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import backtest_current_accounts as replay


def windows(report, split):
    early = [p for p in report['equity_curve'] if p['time_ms'] < split]
    equity = early[-1]['equity'] if early else 1000.
    return {'first20d_net_pnl': equity - 1000,
            'last10d_net_pnl_with_carried_inventory': report['summary']['final_equity'] - equity}


def assess(control, candidate):
    """Require improvement at both costs without increasing drawdown by >0.25pp."""
    failures = []
    for cost in ('cost1', 'cost2'):
        old, new = control[cost], candidate[cost]
        if new['total_return_pct'] <= old['total_return_pct']:
            failures.append(cost + ':net_return_not_improved')
        if new['max_drawdown_pct'] > old['max_drawdown_pct'] + .25:
            failures.append(cost + ':drawdown_increased_over_0.25_percentage_points')
        if new['closed_positions'] < 10:
            failures.append(cost + ':fewer_than_10_closed_positions')
    return {'paper_exit_improvement_passed': not failures, 'failures': failures,
            'positive_at_both_costs': all(candidate[c]['total_return_pct'] > 0 for c in ('cost1', 'cost2')),
            'future_profitability_proven': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=replay.ROOT/'data/validation/current_four_accounts_30d_20261007')
    parser.add_argument('--candidate-profile', type=Path, default=replay.ROOT/'config/stock_confirmed_trend_exit_candidate_20261007.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out/'results.json').exists():
        raise FileExistsError('Use a new output directory; preserve completed validation')
    prior = json.loads((args.source_dir/'results.json').read_text(encoding='utf-8'))
    plan = json.loads(replay.paper.PLAN.read_text(encoding='utf-8'))
    plan['stock_research_profile'] = str(args.candidate_profile.resolve().relative_to(replay.ROOT)).replace('\\', '/')
    candidate = json.loads(args.candidate_profile.read_text(encoding='utf-8'))
    start = replay.paper.arithmetic.parse_time(prior['declaration']['start_utc'])
    end = replay.paper.arithmetic.parse_time(prior['declaration']['end_utc_exclusive'])
    declaration = {'candidate': candidate, 'start_utc': prior['declaration']['start_utc'],
        'end_utc_exclusive': prior['declaration']['end_utc_exclusive'],
        'known_history_previously_reviewed': True, 'independent_holdout': False,
        'parameter_sweep': False, 'scope': candidate.get('validation_scope', 'one fixed exit policy; entries, sizing and costs unchanged'),
        'criteria': {'net_return_improves_at_both_costs': True, 'max_drawdown_increase_percentage_points': .25,
                     'minimum_closed_positions_each_cost': 10},
        'places_orders': False, 'live_orders_allowed': False,
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source_dir/'inputs').glob('*.gz')},
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (replay.ROOT/'scripts').glob('*.py')}}
    replay.save(out/'declaration.json', declaration)
    results = {'declaration': declaration, 'accounts': {}, 'btc': 'Unchanged: no complete historical first-seen archive; cannot qualify a replacement.'}
    for account in plan['accounts']:
        key, symbol = account['account_id'], account['symbol']
        if key == 'btc':
            continue
        source = json.loads(gzip.decompress((args.source_dir/'inputs'/(symbol+'.json.gz')).read_bytes()))
        cfg = replay.paper.stock_config(plan, symbol)
        runs, control, audit = {}, {}, {}
        for cost in (1, 2):
            label = 'cost' + str(cost)
            report = replay.stock_replay(source, cfg, start, end, cost, out)
            if report['status'] != 'price_proxy_diagnostic':
                raise ValueError('Candidate replay unavailable: ' + symbol)
            full = json.loads((out/(symbol+'_'+label+'.json')).read_text(encoding='utf-8'))
            runs[label] = report['summary']
            control[label] = prior['accounts'][key][label]['summary']
            old_full = json.loads((args.source_dir/(symbol+'_'+label+'.json')).read_text(encoding='utf-8'))
            audit[label] = {'candidate': windows(full, start+20*replay.DAY),
                           'control': windows(old_full, start+20*replay.DAY),
                           'exit_reasons': sorted({t['reason'] for t in full['closed_trades']})}
        results['accounts'][key] = {'symbol': symbol, 'control': control, 'candidate': runs,
                                    'chronological_audit': audit, 'assessment': assess(control, runs)}
        replay.save(out/'progress.json', results)
        print(symbol, json.dumps(results['accounts'][key]['assessment']), flush=True)
    replay.save(out/'results.json', results)
    for path in (out.parent/'funding_mark_cache').glob('*.json'):
        replay.save(out/'inputs'/('funding_marks_1m_'+path.name), json.loads(path.read_text(encoding='utf-8')))
    results['supplemental_input_sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').glob('*.json')}
    replay.save(out/'results.json', results)
    print('Validation complete', out, flush=True)


if __name__ == '__main__':
    main()
