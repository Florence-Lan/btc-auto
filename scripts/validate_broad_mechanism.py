"""Confirm only a frozen, audit-passing hourly research winner in StockAccount."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import backtest_current_accounts as replay
import backtest_stock_swing_120 as hourly


def closed_statistics(full):
    """Aggregate IOC legs; use reconciled cash for completed-position net PnL."""
    grouped = {}
    for row in full['closed_trades']:
        item = grouped.setdefault(row['entry_time_ms'], {'net':0., 'closed':False})
        item['net'] += row['net_pnl']
        item['closed'] |= row['position_closed']
    closed = [v['net'] for v in grouped.values() if v['closed']]
    state = full['final_state']
    position = state['position']
    net = state['wallet_balance'] - 1000
    if position:
        net += position['entry_fee'] + position['funding']
        # Already closed legs of the unfinished position are not complete trades.
        net -= grouped.get(position['entry_time'], {}).get('net', 0.)
    # Late settlements need not be present in the close-time trade journal.
    # Subtract all funding credits again for a conservative best-winner test.
    credit = sum(max(0., -r['debit']) for r in full['funding_settlements'])
    return {'closed_net_pnl':net, 'close_time_journal_net_pnl':sum(closed),
            'net_without_best_winner':net-max(closed, default=0)-credit,
            'best_winner_test_extra_funding_credit_deduction':credit}


def assess(control, runs):
    reasons = []
    for cost in (1, 2):
        label = f'cost{cost}'
        old, new = control[label], runs['recent30d'][label]
        if new['total_return_pct'] <= max(0, old['total_return_pct']):
            reasons.append(label + ':no_positive_improvement')
        if new['estimated_liquidated_return_pct'] <= 0:
            reasons.append(label + ':nonpositive_estimated_close')
        if new['max_drawdown_pct'] > old['max_drawdown_pct'] + .25:
            reasons.append(label + ':drawdown_increase_over_0.25pp')
        if new['closed_positions'] < 10:
            reasons.append(label + ':fewer_than_10_closed_positions')
        for window in ('validation', 'full'):
            if window not in runs:
                continue
            summary = runs[window][label]
            if summary['estimated_liquidated_return_pct'] <= 0 or summary['closed_net_pnl'] <= 0:
                reasons.append(window + ':' + label + ':nonpositive_net')
            if summary['closed_positions'] < (10 if window == 'full' else 5):
                reasons.append(window + ':' + label + ':insufficient_trades')
            if summary['net_without_best_winner'] <= 0:
                reasons.append(window + ':' + label + ':nonpositive_without_best_winner')
            if summary['max_drawdown_pct'] > 6:
                reasons.append(window + ':' + label + ':drawdown_over_6pct')
    return {'production_replay_passed': not reasons, 'failures': reasons,
            'forward_validated': False, 'full_history_confirmed': 'full' in runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-dir', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, default=replay.ROOT/'data/validation/current_four_accounts_30d_20261007')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--symbol', default='SNDKUSDT')
    parser.add_argument('--include-long', action='store_true')
    parser.add_argument('--full-start', help='Explicit continuous-data start; never selected by PnL')
    parser.add_argument('--source-cache-dir', type=Path)
    parser.add_argument('--retain-current-exits', action='store_true',
                        help='Separate declared follow-up: change only the frozen winner entry family/cadence')
    parser.add_argument('--require-slow-alignment', action='store_true',
                        help='Separate declared follow-up: reject breakouts against the completed EMA60')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out/'results.json').exists():
        raise FileExistsError('Preserve completed results')
    research = json.loads((args.research_dir/'results.json').read_text(encoding='utf-8'))
    selected = research['accounts'][args.symbol]
    if not selected.get('audit_passed'):
        raise ValueError('Only the frozen audit-passing development winner may be confirmed')
    cfg = deepcopy(selected['selected_config'])
    if args.retain_current_exits:
        plan = json.loads(replay.paper.PLAN.read_text(encoding='utf-8'))
        cfg = replay.paper.stock_config(plan, args.symbol)
        cfg.update({k:selected['selected_config'][k] for k in
                    ('signal_family', 'signal_timeframe', 'warmup_bars', 'cooldown_signal_bars')})
    cfg.update(candidate_id=args.symbol.lower()+'_broad_mechanism_paper_20261007',
               status='retrospective_candidate_pending_production_confirmation', execution_timeframe='5m',
               entry_enabled=True, entry_signal_validity_minutes=0)
    # Keep the declared net-margin exits: no inherited 1R policy or optional filters.
    discarded = ('trend_exit_policy', 'entry_quality_policy', 'notes')
    if not args.retain_current_exits:
        discarded += ('profit_exit_policy',)
    for key in discarded:
        cfg.pop(key, None)
    if args.require_slow_alignment:
        cfg['require_price_slow'] = True
    source_hourly = json.loads(gzip.decompress((args.research_dir/'inputs'/(args.symbol+'.json.gz')).read_bytes()))
    hourly_source = source_hourly['symbols'][args.symbol]
    end = source_hourly['end_ms_exclusive']
    recent = end - 30*replay.DAY
    validation = hourly.parse_time('2026-09-01T00:00:00Z')
    windows = {'recent30d':recent, 'validation':validation}
    if args.include_long:
        windows['full'] = (hourly.parse_time(args.full_start) if args.full_start
                           else hourly_source['start_ms'] + 30*replay.DAY)
    declaration = {'selected_name':selected['development_selected'], 'config':cfg,
        'windows':windows, 'end_ms_exclusive':end, 'independent_holdout':False,
        'places_orders':False, 'criteria': 'Positive improvement at both costs in recent30d, >=10 closed positions, drawdown increase <=0.25pp. Other windows positive, >=5 validation/10 full trades, DD<=6%, positive without best winner.',
        'limitations':['5m opening quotes and previous completed 5m volume proxy historical depth; no intrabar stop observations.',
                       'Funding settlement marks use archived actual 1m mark opens when venue omits markPrice.',
                       'Hourly signal ATR uses the production rolling 500-bar prefix. Known history is not forward validation.'],
        'research_sha256':hashlib.sha256((args.research_dir/'results.json').read_bytes()).hexdigest()}
    declaration['full_start_override'] = args.full_start
    declaration['retain_current_exits_followup'] = args.retain_current_exits
    declaration['require_slow_alignment_followup'] = args.require_slow_alignment
    if args.retain_current_exits:
        declaration['followup_reason'] = ('Net-margin winner failed production best-winner removal. Keep current '
            '1R half/ATR exits and stop/risk settings; only change causal entry family and cadence. '
            'Declare before running; do not refit any thresholds. Previously inspected history remains retrospective.')
    if args.require_slow_alignment:
        declaration['slow_alignment_reason'] = ('Unfiltered hour breakouts include countertrend entries. Reuse the '
            'incumbent EMA60 directional price filter, without choosing its length or changing sizing/exits. '
            'One fixed follow-up, no fallback to another development winner; known history is retrospective.')
    replay.save(out/'declaration.json', declaration)
    account = {'symbol':args.symbol, 'account_id':args.symbol.lower()}
    if args.source_cache_dir:
        cache = args.source_cache_dir/'inputs'/(args.symbol+'.json.gz')
        source = json.loads(gzip.decompress(cache.read_bytes()))
        if source['start_ms'] > min(windows.values()) or source['end_ms'] != end:
            raise ValueError('Cached source does not cover declared boundaries')
        (out/'inputs').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache, out/'inputs'/cache.name)
    else:
        source = replay.download(account, min(windows.values()), end, out, cfg)
    if cfg['signal_timeframe'] != '1h':
        raise ValueError('This confirmer currently supports 1h frozen winners only')
    # Long immutable hourly history supplies exactly the completed signal prefix.
    source['signal'] = hourly_source['trade_1h']
    source['gaps']['signal'] = hourly_source['gaps']['trade_1h']
    replay.save(out/'inputs'/'signal_source_reference.json', {'source':str(args.research_dir),
        'sha256':hashlib.sha256((args.research_dir/'inputs'/(args.symbol+'.json.gz')).read_bytes()).hexdigest()})
    prior = json.loads((args.control_dir/'results.json').read_text(encoding='utf-8'))
    control = next(r for r in prior['accounts'].values() if r.get('symbol') == args.symbol) if any(
        r.get('symbol') == args.symbol for r in prior['accounts'].values()) else prior['accounts'][args.symbol.replace('USDT','').lower()]
    controls = {f'cost{c}':control[f'cost{c}']['summary'] for c in (1,2)}
    runs = {}
    for window, start in windows.items():
        runs[window] = {}
        for cost in (1,2):
            destination = out/window
            scoped = {**source, 'gaps':deepcopy(source['gaps'])}
            # Mark and index history is used only at evaluation timestamps,
            # unlike trade liquidity and signal warmup. Preserve raw gap evidence.
            for label in ('mark', 'index'):
                scoped['gaps'][label] = [t for t in source['gaps'][label] if start <= t < end]
            report = replay.stock_replay(scoped, cfg, start, end, cost, destination)
            if report['status'] != 'price_proxy_diagnostic':
                raise ValueError('Incomplete production replay')
            full = json.loads((destination/f'{args.symbol}_cost{cost}.json').read_text(encoding='utf-8'))
            summary = report['summary']
            summary.update(closed_statistics(full),
                           exit_reasons=sorted({t['reason'] for t in full['closed_trades']}))
            runs[window][f'cost{cost}'] = summary
            replay.save(out/'progress.json', {'declaration':declaration, 'runs':runs})
    result = {'declaration':declaration, 'control':controls, 'runs':runs, 'assessment':assess(controls, runs)}
    result['raw_source_gaps'] = source['gaps']
    result['input_sha256'] = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (out/'inputs').glob('*')}
    result['funding_mark_sha256'] = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (out/'funding_mark_cache').glob('*.json')}
    replay.save(out/'results.json', result)
    print(json.dumps(result['assessment']), flush=True)


if __name__ == '__main__':
    main()
