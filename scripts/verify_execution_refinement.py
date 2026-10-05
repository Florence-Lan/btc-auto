#!/usr/bin/env python3
"""Independently reconcile the declared refinement artifacts with CSV ledgers."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from research_bidirectional_regimes import refinement_failures


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError(f'Ledger mismatch: {actual} != {expected}')


def verify(directory):
    artifact = json.loads((directory / 'results.json').read_text())
    declaration = json.loads((directory / 'declaration.json').read_text())
    assert artifact['declaration'] == declaration
    assert declaration['experiment'] == 'stock_execution_refinement'
    for name, expected in declaration['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256(Path(declaration['control_profile_path']).read_bytes()).hexdigest() == declaration['control_profile_sha256']
    assert hashlib.sha256(Path(declaration['control_profile']['base_config']).read_bytes()).hexdigest() == declaration['base_config_sha256']
    prior = json.loads(Path('data/research/bidirectional_regimes_current_fees_20261005/results.json').read_text())
    scenarios = records = 0
    for symbol, item in artifact['results'].items():
        experiments = item['experiments']
        for name, experiment in experiments.items():
            for scenario, summary in experiment['runs'].items():
                path = directory / f'{symbol}_{name}_{scenario}_trades.csv'
                trades = list(csv.DictReader(path.open(encoding='utf-8-sig', newline=''))) if path.exists() else []
                assert len(trades) == summary['closed_trades'], path
                pnls = []
                side_pnls = {'long': [], 'short': []}
                for trade in trades:
                    net = float(trade['net_pnl'])
                    close(net, float(trade['gross_pnl']) - float(trade['entry_fee'])
                          - float(trade['exit_fee']) - float(trade['funding_debit']))
                    close(float(trade['net_return_initial_margin_pct']), net / float(trade['initial_margin']) * 100)
                    pnls.append(net)
                    side_pnls[trade['direction']].append(net)
                close(sum(pnls), summary['net_closed_pnl'])
                for side, values in side_pnls.items():
                    diag = summary['trade_diagnostics']['by_side'][side]
                    assert diag['closed_trades'] == len(values)
                    close(sum(values), diag['net_pnl'])
                close((sum(pnls) - max([0, *pnls])) / 10,
                      summary['trade_diagnostics']['closed_return_without_largest_winner_pct'])
                scenarios += 1
                records += len(trades)
            for cost in (1, 2):
                key = f'development_cost{cost}'
                expected = (['control_not_replacement'] if name == 'current_control' else
                            refinement_failures(experiment['runs'][key], 20, 1.15, 5,
                                                experiments['current_control']['runs'][key]))
                assert experiment['development_failures'][str(cost)] == expected
        for cost in (1, 2):
            key = f'development_cost{cost}'
            previous = prior['results'][symbol]['experiments']['current_control']['runs'][key]
            current = experiments['current_control']['runs'][key]
            assert current['closed_trades'] == previous['closed_trades']
            for field in ('net_closed_pnl', 'estimated_close_return_pct', 'max_sampled_drawdown_pct'):
                close(current[field], previous[field])
        eligible = [(min(e['runs'][f'development_cost{c}']['net_closed_pnl'] for c in (1, 2)), name)
                    for name, e in experiments.items() if not any(e['development_failures'].values())]
        expected_choice = max(eligible)[1] if eligible else None
        assert item['development_selected'] == expected_choice
        frozen = json.loads((directory / f'{symbol}_frozen_choice.json').read_text())
        assert frozen['selected'] == expected_choice
        assert item['approved_for_forward_simulation'] == (
            bool(expected_choice) and not item['audit_failures'] and not item['deployment_failures'])
    return {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'csv_scenarios': scenarios, 'cross_scenario_ledger_records': records,
            'records_are_not_independent_trades': True, 'source_hashes_match': True,
            'original_control_reproduced': True, 'net_accounting_and_direction_totals_match': True,
            'development_only_selection_matches': True, 'places_orders': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    (args.directory / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
