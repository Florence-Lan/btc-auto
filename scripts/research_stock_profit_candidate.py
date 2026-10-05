#!/usr/bin/env python3
"""Fixed three-arm net60 study and chronological monthly admission, no orders."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import backtest_stock_swing_120 as engine
from analyze_stock_expectancy import summarize, revised_failures
from replay_stock_swing_per_symbol import after_close_summary
from research_stock_mechanisms import validate_ledger, volume_diagnostics
from stock_external_context import MarketContext
from stock_profit_candidate import ARMS, FAST_CONFIG, build_signals, training_eligible


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(snapshot_path, context_path, output):
    if output.exists():
        raise FileExistsError('Preserve previous experiments')
    prior = json.loads(Path('data/research/stock_selective_entries_20261005/declaration.json').read_text())
    common = {**prior['common_config'], 'stop_atr': 1.25,
              'min_stop_fraction': .0075, 'max_stop_fraction': .03}
    source_names = ['stock_profit_candidate.py', 'research_stock_profit_candidate.py',
                    'stock_external_context.py', 'analyze_stock_expectancy.py', *prior['source_sha256']]
    output.mkdir(parents=True)
    declaration = {'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'places_orders': False, 'active_strategy_changed': False,
        'snapshot_path': str(snapshot_path), 'snapshot_sha256': digest(snapshot_path),
        'context_path': str(context_path), 'context_sha256': digest(context_path),
        'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in source_names},
        'common_config': common, 'fast_config': FAST_CONFIG, 'arms': ARMS,
        'fees': prior['fees'], 'external_overlay': 'nq_confirmation',
        'entry_rules': {'tight_breakout': 'Prior EMA20/50/200 1h breakout + closed4h trend + volume confirmation; stop now1.25ATR, min0.75%, cap3%.',
            'trend_pullback': '1h EMA8>24>60 or mirror, mid/slow6-bar slopes agree, 4/6 active hours. Prior3 hours touch contemporaneousEMA8. Current directional candle closes beyond prior high/low, at most2ATR fromEMA24.',
            'compression_breakout': 'Same fast trend/activity; close breaks prior12-hour high/low by at most1ATR; prior6-hour range<=3currentATR.'},
        'monthly_policy': 'Each arm evaluated separately, no winner ranking. Jul/Aug/Sep/Oct eligibility uses only replay ending at that month start, stressed costs, >=20closed, positive net expectancy including open-close, positive removingbest, DD<=6%, no liquidation. If eligible, admit that arm through that month; continuous replay keeps exits during blocked months. This is retrospective, history already reviewed.',
        'screen_policy': 'Keep previous expectancy admission thresholds; no loosening samples or costs to make a candidate pass. Normal+doublecost; development beforeSep1, validationSep1..20,recentSep20..cutoff,full. Full overlaps segments.',
        'limitations': ['Known reviewed history; no unseen holdout or proven future profits.',
            'Public hourly NQ latest historical vintage, reconstructed1h+20min availability, max90min age. Actual first-seen and rolls unavailable.',
            '5m OHLC fills and historical liquidity remain approximations; zero-volume eventualfillbars reported, never futurefiltered.',
            'Current fees on earlier data are cost scenarios, not exact historical fee reconstruction.'],
        'deployment': 'Never enable existing accounts from historical results. Freeze any candidate for a separate forward simulation.'}
    (output / 'declaration.json').write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + '\n')
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    context = MarketContext(json.loads(gzip.decompress(context_path.read_bytes())))
    end = snapshot['end_ms_exclusive']
    split, recent = [engine.parse_time(t) for t in ('2026-09-01T00:00:00Z','2026-09-20T00:00:00Z')]
    months = [engine.parse_time(f'2026-{m:02d}-01T00:00:00Z') for m in (7, 8, 9, 10)]
    results, hashes, checks, ledger_count = {}, {}, 0, 0
    for symbol, source in snapshot['symbols'].items():
        if symbol not in prior['fees']:
            continue
        one = {**snapshot, 'symbols': {symbol: source}}
        results[symbol] = {}
        for arm in ARMS:
            cfg = {**common, **(FAST_CONFIG if arm != ARMS[0] else {}),
                   'symbols': [symbol], 'taker_fee_rate_assumption': prior['fees'][symbol]}
            prepared = engine.prepare(one, cfg)
            hourly = [engine.candle(row) for row in source['trade_1h']]
            signals = build_signals(hourly, engine.aggregate_4h(hourly), cfg, arm)
            accepted = {t:s for t,s in signals.items() if t < end and context.decision(t,s.direction,'nq_confirmation')['allowed']}
            modified = {symbol: {**prepared[symbol], 'signals': accepted, 'first_signal_time': min(accepted,default=None)}}
            windows = {'development': (source['start_ms'],split),'validation': (split,recent),
                       'recent': (recent,end),'full': (source['start_ms'],end)}
            runs = {}
            def replay(label, start, finish, cost, data):
                nonlocal checks, ledger_count
                replayed = engine.simulate(one,cfg,start,finish,cost,prepared_data=data)
                validate_ledger(replayed,cfg,source,cost)
                summary = after_close_summary(replayed)
                metrics = summarize(replayed['trades'],1000,None)
                volumes = volume_diagnostics(replayed,source)
                summary['execution_volume_diagnostics'] = {k:v for k,v in volumes.items() if k != 'closed_trade_bar_checks'}
                summary['expectancy'] = metrics
                for trade in replayed['trades']:
                    timestamp = engine.parse_time(trade['entry_utc'])
                    assert timestamp in data[symbol]['signals']
                    assert context.decision(timestamp,1 if trade['direction']=='long' else -1,'nq_confirmation')['allowed']
                    assert math.isclose(trade['net_pnl'],trade['gross_pnl']-trade['entry_fee']-trade['exit_fee']-trade['funding_debit'],abs_tol=1e-7)
                path = output / f'{symbol}_{arm}_{label}_trades.csv'
                engine.write_csv(path,replayed['trades'])
                if not replayed['trades']:
                    path.write_text('net_pnl,net_return_initial_margin_pct\n')
                with path.open(encoding='utf-8-sig',newline='') as handle:
                    rows = list(csv.DictReader(handle))
                assert len(rows) == metrics['closed_trades']
                assert math.isclose(sum(float(r['net_pnl']) for r in rows),metrics['net_pnl_usdt'],abs_tol=1e-7)
                hashes[path.name] = digest(path)
                checks += 1
                ledger_count += len(rows)
                print(symbol,arm,label,'n',len(rows),'net',round(metrics['net_pnl_usdt'],3),'mean',round(metrics['mean_closed_net_pnl_usdt'] or 0,3),flush=True)
                return summary
            for window,(start,finish) in windows.items():
                for cost in (1,2):
                    name = f'{window}_cost{cost}'
                    summary = replay(name,start,finish,cost,modified)
                    summary['failure_reasons'] = revised_failures(summary,summary['expectancy'],window)
                    runs[name] = summary
            monthly, causal_signals = [], {}
            for i,start in enumerate(months):
                finish = min(months[i+1] if i+1<len(months) else end,end)
                training = replay(f'train_{engine.iso(start)[:7]}',source['start_ms'],start,2,modified)
                eligible = training_eligible(training,training['expectancy'])
                monthly.append({'month_start_ms':start,'month_end_ms':finish,'training_end_ms_exclusive':start,
                                'training':training,'eligible':eligible})
                if eligible:
                    causal_signals.update({t:s for t,s in accepted.items() if start<=t<finish})
            causal_data = {symbol:{**modified[symbol],'signals':causal_signals,'first_signal_time':min(causal_signals,default=None)}}
            for cost in (1,2):
                runs[f'monthly_admission_cost{cost}'] = replay(f'monthly_admission_cost{cost}',months[0],end,cost,causal_data)
            results[symbol][arm] = {'config':cfg,'accepted_retries':len(accepted),'runs':runs,'monthly_decisions':monthly,
                'passes_original_expectancy_screen':not any(runs[f'{w}_cost{c}']['failure_reasons'] for w in windows for c in (1,2)),
                'future_expectancy_proven_positive':False,'approved_for_existing_entries':False}
            artifact = {'declaration':declaration,'results':results,'places_orders':False,'active_strategy_changed':False}
            (output/'results.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    verification = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),'csv_scenarios':checks,
        'overlapping_ledger_records':ledger_count,'csv_sha256':hashes,'all_closed_entries_pass_nq_asof':True,
        'net_accounting_reconciled':True,'monthly_eligibility_uses_only_prior_history':True,'places_orders':False}
    assert checks == 126
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return artifact


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=Path('data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz'))
    parser.add_argument('--context',type=Path,default=Path('data/research/stock_external_context_20261005/market_context.json.gz'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.snapshot,args.context,args.output_dir)
