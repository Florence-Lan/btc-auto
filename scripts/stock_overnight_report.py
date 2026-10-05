"""Read-only overnight accounting; completed positions, not exit-fill counts."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completed_positions(fills, funding):
    rounds, current = [], None
    for fill in sorted(fills, key=lambda row: row['sequence']):
        if fill['reason'].startswith('fresh_closed_'):
            if current is not None:
                raise ValueError('New entry before prior position is fully closed')
            current = {'entry_time_ms': fill['time_ms'], 'qty': float(fill['qty']),
                       'remaining': float(fill['qty']), 'gross_pnl': 0.,
                       'fees': float(fill['fee']), 'exit_time_ms': None}
        else:
            if current is None:
                raise ValueError('Exit without a matching entry')
            current['remaining'] -= float(fill['qty'])
            current['fees'] += float(fill['fee'])
            current['gross_pnl'] += float(fill['gross_pnl'])
            if current['remaining'] < -1e-8:
                raise ValueError('Exit quantity exceeds entry')
            if current['remaining'] <= 1e-8:
                current['exit_time_ms'] = fill['time_ms']
                rounds.append(current)
                current = None
    all_rounds = rounds + ([current] if current else [])
    allocated = 0.
    for position in all_rounds:
        events = [e for e in funding if position['entry_time_ms'] < int(e['fundingTime'])
                  and (position['exit_time_ms'] is None or int(e['fundingTime']) <= position['exit_time_ms'])]
        position['funding_debit'] = sum(float(e['debit']) for e in events)
        allocated += position['funding_debit']
        position['net_pnl'] = position['gross_pnl'] - position['fees'] - position['funding_debit']
    if not math.isclose(allocated, sum(float(e['debit']) for e in funding), abs_tol=1e-7):
        raise ValueError('Funding could not be reconciled to position intervals')
    return rounds, current


def account_report(path, now_ms):
    state = json.loads((path / 'state.json').read_text())
    fills = read_rows(path / 'fills.jsonl')
    exits = {row['sequence']: row for row in read_rows(path / 'closed_trades.jsonl')}
    enriched = []
    for fill in fills:
        if not fill['reason'].startswith('fresh_closed_'):
            if fill['sequence'] not in exits:
                raise ValueError('Missing exit accounting record')
            fill = {**fill, 'gross_pnl': exits[fill['sequence']]['gross_pnl']}
        enriched.append(fill)
    funding = read_rows(path / 'funding.jsonl')
    rounds, partial = completed_positions(enriched, funding)
    cash_net = sum(r['net_pnl'] for r in rounds) + (partial['net_pnl'] if partial else 0)
    assert math.isclose(cash_net, state['wallet_balance']-state['initial_balance'], abs_tol=1e-7)
    assert math.isclose(sum(float(f['fee']) for f in fills), state['fees_paid'], abs_tol=1e-7)
    assert math.isclose(-sum(float(e['debit']) for e in funding), state['funding_pnl'], abs_tol=1e-7)
    net = [r['net_pnl'] for r in rounds]
    wins, losses = [p for p in net if p>0], [p for p in net if p<0]
    checked = datetime.fromisoformat(state['checked_at_utc']).timestamp()*1000 if state.get('checked_at_utc') else 0
    fresh = -5000 <= now_ms-checked < 120_000
    estimate = state['equity']-state['initial_balance']
    pos = state.get('position')
    if pos:
        mark = state['last_mark_price']
        fill = mark*(1-pos['direction']*state['rule']['adverse_slippage_fraction_assumption'])
        estimate += pos['direction']*pos['qty']*(fill-mark) - pos['qty']*fill*state['rule']['taker_fee_rate_assumption']
    reasons = []
    if len(rounds) < 30:
        reasons.append('insufficient_completed_positions')
    if not net or statistics.mean(net)<=0:
        reasons.append('nonpositive_or_unavailable_sample_expectancy')
    after_best = (sum(net)-max(wins))/(len(net)-1) if wins and len(net)>1 else None
    if after_best is None or after_best<=0:
        reasons.append('nonpositive_or_unavailable_without_best_winner')
    if estimate<=0:
        reasons.append('nonpositive_net_with_estimated_open_close')
    if state['max_drawdown_pct']>6:
        reasons.append('drawdown_above_6pct')
    if any(f['reason']=='liquidation_stress' for f in fills):
        reasons.append('modeled_liquidation')
    if not fresh or state.get('errors') or state.get('status')!='healthy':
        reasons.append('data_or_process_health_unverified')
    if pos:
        reasons.append('remaining_open_inventory')
    event_times = [int(e['fundingTime']) for e in funding]
    if len(event_times)!=len(set(event_times)):
        reasons.append('duplicate_funding_timestamp_requires_review')
    return {'symbol': state['symbol'], 'status': state['status'], 'checked_at_utc': state.get('checked_at_utc'),
        'initial_equity_usdt': state['initial_balance'], 'equity_usdt': state['equity'],
        'wallet_net_pnl_usdt': state['wallet_balance']-state['initial_balance'],
        'marked_account_net_pnl_usdt': state['equity']-state['initial_balance'],
        'estimated_flat_net_pnl_usdt': estimate,
        'open_close_estimate_method': 'actual_flat_wallet' if not pos else 'last_mark_less_slippage_and_exit_fee_not_depth',
        'completed_positions': len(rounds), 'exit_fills': len(exits), 'net_winners': len(wins),
        'closed_position_net_pnl_usdt': sum(net), 'mean_closed_net_pnl_usdt': statistics.mean(net) if net else None,
        'net_win_rate_pct': len(wins)/len(net)*100 if net else None,
        'mean_winning_net_pnl_usdt': statistics.mean(wins) if wins else None,
        'mean_losing_net_loss_usdt': -statistics.mean(losses) if losses else None,
        'mean_without_best_winner_usdt': after_best,
        'fees_paid_usdt': state['fees_paid'], 'funding_pnl_usdt': state['funding_pnl'],
        'max_drawdown_pct': state['max_drawdown_pct'], 'position_qty': state.get('position_qty',0),
        'partial_exit_round_counted_as_closed': False, 'health_fresh': fresh,
        'retrospective_sample_gate_passed': not reasons, 'failure_reasons': reasons,
        'future_expectancy_proven_positive': False, 'accounting_reconciled': True,
        'signal_status': state.get('signal_status'), 'entry_blockers': state.get('entry_blockers',[])}


def make_report(root, profile, now_ms, ended=False):
    manifest = json.loads((root / 'manifest.json').read_text())
    results = {}
    for symbol in profile['symbol_profiles']:
        results[symbol] = {}
        for cost in ('normal_cost','double_cost'):
            try:
                results[symbol][cost] = account_report(root/symbol/cost,now_ms)
            except Exception as exc:
                results[symbol][cost] = {'symbol':symbol,'accounting_reconciled':False,
                    'retrospective_sample_gate_passed':False,'future_expectancy_proven_positive':False,
                    'error':type(exc).__name__+': '+str(exc)[:240]}
    return {'experiment_id': profile['experiment_id'], 'generated_at_utc': datetime.fromtimestamp(now_ms/1000,timezone.utc).isoformat(),
        'started_at_utc': manifest['started_at_utc'], 'scheduled_end_utc': profile['end_utc'],
        'deadline_reached': now_ms>=manifest['end_ms'], 'run_ended':ended,
        'places_orders':False, 'future_expectancy_proven_positive':False,
        'one_night_is_not_sufficient_proof_of_profitability':True,
        'automatic_promotion':False, 'accounts':results}


def markdown(report):
    lines = ['# 三股票隔夜模拟报告','',f"开始：{report['started_at_utc']}；计划截止：{report['scheduled_end_utc']}；更新：{report['generated_at_utc']}。",'',
        '本报告记录实际新行情上的模拟结果。一晚和少量交易不足以证明未来盈利；零成交不计为成功，未平仓浮盈不计为已实现盈利。','',
        '| 股票 | 成本 | 完整平仓笔数 | 已平仓净收益 USDT | 账户净值盈亏 USDT | 每笔净期望 USDT | 最大回撤 |',
        '|---|---|---:|---:|---:|---:|---:|']
    def number(v):
        return '—' if v is None else f'{v:.4f}'
    for symbol,costs in report['accounts'].items():
        for cost,account in costs.items():
            label='正常' if cost=='normal_cost' else '双倍'
            lines.append(f"| {symbol} | {label} | {account.get('completed_positions','—')} | {number(account.get('closed_position_net_pnl_usdt'))} | {number(account.get('marked_account_net_pnl_usdt'))} | {number(account.get('mean_closed_net_pnl_usdt'))} | {number(account.get('max_drawdown_pct'))}% |")
    lines.extend(['','计划截止后暂停新增并按新鲜盘口退出剩余模拟仓位；未能完全退出时仍运行保护，报告保留未完成状态。',
        '手续费、滑点包含在成交价和费用中，资金费按结算时实际持仓分配，包括平仓后才公布的结算。部分退出不算一笔完整平仓。',
        '每个成本账户至少30笔、净期望为正、去掉最大赢家仍为正、含持仓估算退出净收益为正、回撤不超过6%、无强平且行情健康，才达到预定样本筛查。满足筛查仍须复核，不会自动启用原账户。',''])
    return '\n'.join(lines)
