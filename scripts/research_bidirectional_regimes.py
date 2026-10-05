#!/usr/bin/env python3
"""Small declared bidirectional comparison; public archived data, no orders."""
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
from research_stock_swing_robustness import audit_snapshot, trade_diagnostics


def candidates(current):
    common = {'ema_fast': 20, 'ema_mid': 50, 'ema_slow': 200, 'warmup_bars': 200,
        'ema_slope_bars': 6, 'breakout_bars': 20, 'stop_atr': 2.0,
        'min_stop_fraction': .01, 'max_stop_fraction': .05,
        'entry_direction': 'both', 'target_margin_return': .3,
        'max_holding_calendar_days': 3, 'trail_activation_underlying_return': .02,
        'trail_locked_underlying_return': .005, 'trail_atr': 2.0, 'cooldown_signal_bars': 3}
    result = {f'{family}_{period}': {**common, 'signal_family': family, 'signal_timeframe': period,
        'entry_signal_validity_minutes': 15 if period == '15m' else 60}
        for family in ('trend_pullback', 'breakout', 'range_reversion') for period in ('15m', '1h')}
    # A named control is eligible under the same development-only gate.
    if current is not None:
        result['current_control'] = current
    return result


def failures(summary, minimum, min_pf):
    reasons = []
    if summary['net_closed_pnl'] <= 0: reasons.append('closed_net_pnl_nonpositive')
    if summary['estimated_close_return_pct'] <= 0: reasons.append('net_liquidation_equity_nonpositive')
    if summary['closed_trades'] < minimum: reasons.append('insufficient_closed_trades')
    if summary['profit_factor'] is not None and summary['profit_factor'] < min_pf: reasons.append('profit_factor_below_gate')
    if summary['max_sampled_drawdown_pct'] > 6: reasons.append('drawdown_above_6pct')
    if summary['liquidation_stress_count']: reasons.append('liquidation_stress')
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('data/research/stock_swing_liquidity_20261004/effective_snapshot.json.gz'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=Path('config/stock_bidirectional_forward_20261005.json'),
        help='Frozen pre-research control profile; independent of the currently active profile')
    parser.add_argument('--current-fee-scenario', action='store_true',
        help='Apply documented current fee assumptions to old price paths, not realized historical returns')
    parser.add_argument('--btc-snapshot', type=Path,
        help='Evaluate BTC separately with archived trade prices as mark/index proxies; no historical basis claim')
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError('Preserve prior evidence: use a new output directory')
    profile = json.loads(args.profile.read_text())
    base = json.loads(Path(profile['base_config']).read_text())
    all_choices = {s: candidates({**{k:profile[k] for k in ('signal_timeframe','cooldown_signal_bars')},
                                  **profile['symbol_profiles'][s]}) for s in base['symbols']}
    if args.btc_snapshot:
        all_choices={'BTCUSDT':candidates(None)}
    args.output_dir.mkdir(parents=True)
    declaration = {'declared_at_utc':datetime.now(timezone.utc).isoformat(), 'places_orders':False,
        'known_history_previously_reviewed':True, 'independent_unseen_holdout':False,
        'snapshot_sha256':hashlib.sha256((args.btc_snapshot or args.snapshot).read_bytes()).hexdigest(),
        'control_profile_path':str(args.profile), 'control_profile_sha256':hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('stock_swing_profiles.py'),Path(__file__).with_name('backtest_stock_swing_120.py')]},
        'candidates':all_choices, 'risk_fraction_per_trade':.0025, 'leverage_cap':10,
        'fee_model':'current_documented_rates_on_historical_prices' if args.current_fee_scenario else 'conservative_uniform_0.10pct_each_side',
        'fee_assumptions':{'MUUSDT':.000125,'SNDKUSDT':.000125,'SKHYNIXUSDT':.001} if args.current_fee_scenario else {s:.001 for s in base['symbols']},
        'fee_source':'https://docs.asterdex.com/trading/perpetuals/fees-and-specs/fees',
        'selection':'Development before 2026-09-01 only. Both costs: positive closed net and estimated liquidation equity, >=20 closed trades, profit factor >=1.15, drawdown <=6%, no liquidation. Rank worst-cost closed return, freeze before audit.',
        'audit':'Frozen winner only: validation >=5 closed trades, recent30d >=3, both costs positive closed and liquidation return, PF >=1.1, DD <=6%, no liquidation. Do not select an audit winner if frozen choice fails.',
        'limitations':['Previously reviewed retrospective history; not independent profitability evidence.',
            '5m OHLC execution and hourly opening index proxy; historical depth, actual fills and exact price path unavailable.',
            '0.10% fee/side, 2bp adverse slippage, observed historical funding. Double fees/slippage at cost2.',
            'Entry capped at 10% of previous completed 5m volume; missing and zero volumes block entry.',
            'Forward runtime has 30s observations; these replays have 5m observations.']}
    if args.btc_snapshot:
        declaration.update(fee_model='existing_project_binance_assumptions',fee_assumptions={'BTCUSDT':.00045},
            fee_source=None,price_model='archived_trade_prices_as_mark_and_hourly_index_proxies')
        declaration['limitations'].append('BTC mark/index historical prices are unavailable in this snapshot. Trade-price proxies disable realistic historical basis checks; this is an exploratory price-only replay, not full execution validation or a reproduction of the six-factor BTC control.')
    (args.output_dir/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    if args.btc_snapshot:
        old=json.loads(gzip.decompress(args.btc_snapshot.read_bytes()))
        def rows(period): return [[*r[:6],r[7]] for r in old['intervals'][period]]
        five=rows('5m');hourly=rows('1h')
        source={'start_ms':int(five[0][0]),'trade_5m':five,'mark_5m':five,
            'trade_1h':hourly,'mark_1h':hourly,'index_1h':hourly,
            'funding':[{'fundingTime':r[0],'fundingRate':r[1]} for r in old['funding_rates']],
            'rules_current':{'filters':[{'filterType':'LOT_SIZE','stepSize':'0.001','minQty':'0.001','maxQty':'100'},
                {'filterType':'PRICE_FILTER','tickSize':'0.1'},{'filterType':'MIN_NOTIONAL','notional':'50'}]}}
        snapshot={'execution_step_ms':300_000,'end_ms_exclusive':int(five[-1][6])+1,'symbols':{'BTCUSDT':source}}
    else:
        snapshot=json.loads(gzip.decompress(args.snapshot.read_bytes()))
    audit_snapshot(snapshot)
    split=engine.parse_time('2026-09-01T00:00:00Z');end=snapshot['end_ms_exclusive']
    results={}
    for symbol,choices in all_choices.items():
        source=snapshot['symbols'][symbol];one={**snapshot,'symbols':{symbol:source}}
        common={**base,'symbols':[symbol],'initial_equity_usdt':1000,'risk_fraction_per_trade':.0025,
            'max_positions':1,'execution_timeframe':'5m','entry_max_previous_bar_participation_fraction':.1}
        if args.current_fee_scenario or args.btc_snapshot:
            common['taker_fee_rate_assumption']=declaration['fee_assumptions'][symbol]
        if args.btc_snapshot:
            common.update(adverse_slippage_fraction_assumption=.0001,maintenance_margin_fraction_assumption=.004)
        experiments={};prepared={};configs={}
        for name,settings in choices.items():
            cfg={**common,**settings,'symbol_profiles':{symbol:settings}}
            configs[name]=cfg;prepared[name]=engine.prepare(one,cfg)
            runs={}
            for cost in (1,2):
                result=engine.simulate(one,cfg,source['start_ms'],split,cost,prepared_data=prepared[name])
                validate_ledger(result,cfg,source,cost)
                runs[f'development_cost{cost}']=after_close_summary(result)
                engine.write_csv(args.output_dir/f'{symbol}_{name}_development_cost{cost}_trades.csv',result['trades'])
            rejected={str(c):failures(runs[f'development_cost{c}'],20,1.15) for c in (1,2)}
            experiments[name]={'settings':settings,'runs':runs,'development_failures':rejected}
            print(symbol,name,'development',[(c,round(runs[f'development_cost{c}']['net_closed_pnl'],4),runs[f'development_cost{c}']['closed_trades']) for c in (1,2)],flush=True)
        eligible=[(min(v['runs'][f'development_cost{c}']['net_closed_pnl'] for c in (1,2)),name)
            for name,v in experiments.items() if not any(v['development_failures'].values())]
        selected=max(eligible)[1] if eligible else None
        (args.output_dir/f'{symbol}_frozen_choice.json').write_text(json.dumps({'selected':selected,'development':experiments},indent=2)+'\n')
        audit_errors=[]
        if selected:
            cfg=configs[selected];runs=experiments[selected]['runs']
            for window,start,finish in [('validation',split,end),('recent30d',end-30*engine.DAY,end),('full',source['start_ms'],end)]:
                for cost in (1,2):
                    result=engine.simulate(one,cfg,start,finish,cost,prepared_data=prepared[selected])
                    validate_ledger(result,cfg,source,cost)
                    summary=after_close_summary(result)
                    summary['trade_diagnostics']=trade_diagnostics(result['trades'],1000,samples=500)
                    runs[f'{window}_cost{cost}']=summary
                    engine.write_csv(args.output_dir/f'{symbol}_{selected}_{window}_cost{cost}_trades.csv',result['trades'])
                    if window!='full':audit_errors += [f'{window}_cost{cost}: {v}' for v in failures(summary,5 if window=='validation' else 3,1.1)]
        results[symbol]={'experiments':experiments,'development_selected':selected,'selected_passes_audit':bool(selected) and not audit_errors,'audit_failures':audit_errors,'forward_validated':False}
        artifact={'declaration':declaration,'end_utc_exclusive':engine.iso(end),'results':results}
        (args.output_dir/'results.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
        print('FROZEN',symbol,selected,'audit',audit_errors,flush=True)


if __name__=='__main__': main()
