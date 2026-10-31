"""Offline hindsight replay uses historical evidence and isolated ledgers only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_missed_stock_entries as replay
import run_parallel_simulation as paper
import stock_short_confirmation as short
from test_main_stock_runtime import main_worker

HOUR = 3_600_000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Existing main fixture supplies the real engine's config, contract filters,
    # complete candles and depth. All venue calls are already offline mocks.
    source, clock, _ = main_worker(tmp_path / 'source_fixture', monkeypatch)
    responses = source.market_data(force_rules=True, force_signal=True)
    data = {label: body for label, body, error in responses if not error}
    at = clock['now']
    bars = [paper.arithmetic.candle(row) for row in data['signal'] if int(row[6]) < at]
    signal, diagnostics = short.signal_with_diagnostics(
        bars, paper.signals.compute_indicators(bars, source.config), len(bars) - 1, source.config)
    assert signal is not None and signal.direction == -1
    candidate = {'original_signal': asdict(signal), 'boundary_ms': signal.time_ms + HOUR,
        'original_expiry_ms': signal.time_ms + 2 * HOUR,
        'original_admission': diagnostics['admission'],
        'trigger': paper.stock_entry_timing.build_trigger(
            bars, signal, source.config, signal.time_ms + HOUR, observed_at_ms=at)}
    row = {'time_utc': paper.arithmetic.iso(at), 'symbol': source.symbol,
        'mark': deepcopy(data['mark']), 'book': deepcopy(data['book']),
        'funding_events': [], 'preceding_candles_5m': deepcopy(
            [item for item in data['five'] if int(item[6]) < at]),
        'signal_candles': deepcopy(data['signal']), 'errors': {},
        'risk_monitor': {'clock_source': 'exchange'},
        'stock_event_risk': {'enabled': False, 'allowed': True, 'reasons': []}}
    rules = next(item for item in data['rules']['symbols'] if item['symbol'] == source.symbol)
    seed = paper.initial_stock(source.symbol, at, deepcopy(source.config))
    seed.update(signal_active_after_ms=at + 10 * HOUR, pending_signal=None,
                retrospective_only=True, performance_class=replay.LABEL)
    monkeypatch.setattr(replay, 'ROOT', tmp_path)
    output = tmp_path / 'data/retrospective_simulation/offline_fixture'
    path = output / 'mu/state.json'
    paper.write_json(path, seed)
    account = replay.ReplayStockAccount(path, seed['rule'], seed['symbol'], rules)
    return {'account': account, 'seed': seed, 'candidate': candidate, 'row': row,
            'capture': at, 'output': output, 'rules': rules, 'root': tmp_path}


def moved_row(f, *, seconds=30, price=None, clock_ok=True, hourly=True):
    row = deepcopy(f['row'])
    at = f['capture'] + seconds * 1000
    row['time_utc'] = paper.arithmetic.iso(at)
    row['mark']['time'] = at
    row['mark']['nextFundingTime'] = at + HOUR
    row['book']['T'] = at
    row['book'].pop('E', None)
    if price is not None:
        row['mark'].update(markPrice=str(price), indexPrice=str(price))
        row['book']['bids'] = [[str(price - .0001), '1000']]
        row['book']['asks'] = [[str(price + .0001), '1000']]
    if not clock_ok:
        row['errors']['clock'] = 'Original clock timeout'
    if not hourly:
        row.pop('signal_candles', None)
        row['errors']['signal'] = 'Original hourly outage'
    return row


def enter(f, row=None):
    row = deepcopy(row or f['row'])
    state = deepcopy(f['seed'])
    fill = replay.hypothetical_entry(
        f['account'], state, deepcopy(f['candidate']), row, f['capture'])
    return state, fill


def test_historical_manual_entry_has_exact_capture_time_real_depth_fees_and_risk(fixture):
    f = fixture
    before = deepcopy((f['candidate'], f['row']))
    state, fill = enter(f)
    cfg, pos = state['rule'], state['position']
    assert fill['time_ms'] == pos['entry_time'] == f['capture']
    assert fill['time_utc'] == paper.arithmetic.iso(f['capture'])
    assert fill['reason'] == 'retrospective_user_requested_signal_capture'
    assert fill['performance_class'] == replay.LABEL and fill['retrospective_only'] is True
    assert fill['not_forward_performance'] is True
    assert fill['places_orders'] is False and fill['mode'] == 'simulation'
    assert fill['price'] < float(f['row']['book']['bids'][0][0])
    assert fill['fee'] == pytest.approx(fill['qty'] * fill['price'] * cfg['taker_fee_rate_assumption'])
    assert state['wallet_balance'] == pytest.approx(1000 - fill['fee'])
    risk = fill['qty'] * paper.signals.per_unit_stop_risk(
        fill['price'], pos['stop'], cfg['taker_fee_rate_assumption'], cfg['adverse_slippage_fraction_assumption'])
    assert risk <= 1000 * .0025 + 1e-9
    assert pos['initial_stop'] == pos['stop'] > pos['entry']
    assert pos['qty'] * pos['entry'] <= 1000 + 1e-9
    assert pos['margin'] == pytest.approx(pos['qty'] * pos['entry'] / 10)
    audit = state['retrospective_entry']
    assert audit['entry_is_user_specified_not_original_strategy_admission']
    assert audit['original_signal'] == f['candidate']['original_signal']
    assert set(audit['entry_admission_overrides']) == {
        'original_confirmation', 'original_breakout_validity', 'original_chase_and_entry_funding_filters'}
    assert before == (f['candidate'], f['row'])
    rows = [json.loads(line) for line in f['account'].path.with_name('fills.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows == state['fills'] == [fill]


def test_hindsight_entry_labels_overrides_instead_of_pretending_auto_admission(fixture):
    f = fixture
    candidate = deepcopy(f['candidate'])
    candidate['original_admission'] = {'stage': 'confirmation_rejected', 'reasons': ['old_4h_ordering']}
    row = moved_row(f, seconds=0, price=candidate['trigger']['breakout_level'] + .5)
    state = deepcopy(f['seed'])
    fill = replay.hypothetical_entry(f['account'], state, candidate, row, f['capture'])
    assert fill['performance_class'] == replay.LABEL
    assert state['retrospective_entry']['original_admission']['stage'] == 'confirmation_rejected'
    assert state['retrospective_entry']['entry_is_user_specified_not_original_strategy_admission']


def test_historical_ioc_remains_limited_to_observed_ten_percent_depth(fixture):
    row = deepcopy(fixture['row'])
    row['book']['bids'][0][1] = '1'
    state, fill = enter(fixture, row)
    assert fill['qty'] <= .1 + 1e-12
    assert fill['partial'] is True
    assert state['position']['qty'] == fill['qty']


@pytest.mark.parametrize('part,change', [
    ('book', 'stale'), ('book', 'future'), ('mark', 'stale'), ('mark', 'future'),
    ('five', 'missing'), ('five', 'nan'), ('mark_price', 'nan'),
])
def test_bad_market_evidence_never_books_a_historical_entry(fixture, part, change):
    f, row = fixture, deepcopy(fixture['row'])
    if part == 'book':
        row['book']['T'] = f['capture'] + (6000 if change == 'future' else -60001)
    elif part == 'mark':
        row['mark']['time'] = f['capture'] + (6000 if change == 'future' else -60001)
    elif part == 'five':
        row['preceding_candles_5m'] = [] if change == 'missing' else deepcopy(row['preceding_candles_5m'])
        if change == 'nan':
            row['preceding_candles_5m'][-1][5] = 'NaN'
    else:
        row['mark']['markPrice'] = 'NaN'
    before = f['account'].path.read_bytes()
    with pytest.raises(ValueError):
        enter(f, row)
    assert f['account'].path.read_bytes() == before
    assert not f['account'].path.with_name('fills.jsonl').exists()


@pytest.mark.parametrize('clock_source,error,allowed', [
    ('exchange', None, True), ('exchange', 'Recorded timeout', False),
    ('mark_endpoint', None, False), (None, None, False),
])
def test_recorded_clock_coordinate_preserves_original_failure_without_new_receipt(clock_source,error,allowed):
    row = {'time_utc': '2026-10-30T12:01:50.315+00:00', 'errors': {},
           'risk_monitor': {'clock_source': clock_source}, 'mark': {'time': 123}}
    if error:
        row['errors']['clock'] = error
    original = deepcopy(row)
    result = {label: (body, failure) for label, body, failure in replay.recorded_responses(row)}
    assert bool(result['clock'][0]) is allowed
    if allowed:
        assert result['clock'][0] == {'serverTime': replay.row_time(row)}
        assert '_received_monotonic' not in result['clock'][0]
        assert '_public_clock_timing' not in result['clock'][0]
    else:
        assert result['clock'][0] is None and result['clock'][1]
    result['mark'][0]['time'] = 999
    assert row == original


def test_historical_settlement_proxy_is_unavailable_until_original_available_time():
    point = {'symbol': 'MUUSDT', 'minute_ms': 600000, 'open': '100.2', 'available_by_ms': 700000}
    venue = replay.HistoricalVenue([point])
    venue.at_ms = 699999
    with pytest.raises(ValueError, match='network disabled'):
        venue.get('markPriceKlines', {'symbol': 'MUUSDT', 'startTime': 600000})
    venue.at_ms = 700000
    assert venue.get('markPriceKlines', {'symbol': 'MUUSDT', 'startTime': 600000}) == [[600000, '100.2']]
    for endpoint, params in [('depth', {}), ('time', {}), ('markPriceKlines', {'symbol': 'SNDKUSDT','startTime':600000})]:
        with pytest.raises(ValueError, match='network disabled'):
            venue.get(endpoint, params)


def test_missing_funding_proxy_is_deferred_then_settled_once_when_available(fixture):
    f = fixture
    state, fill = enter(f)
    event_at = f['capture'] + 120000
    point = {'symbol': f['seed']['symbol'], 'minute_ms': event_at // 60000 * 60000,
             'open': str(fill['price']), 'available_by_ms': f['capture'] + 240000}
    f['account'].venue.evidence = [point]
    event = {'symbol': f['seed']['symbol'], 'fundingTime': event_at, 'fundingRate': '.0001'}
    first = moved_row(f, seconds=180)
    first['funding_events'] = [event]
    deferred = f['account'].step_row(first)
    assert 'funding' in deferred['errors'] and not deferred['settled_funding']
    assert deferred['funding_pnl'] == 0
    second = moved_row(f, seconds=250)
    second['funding_events'] = [event]
    settled = f['account'].step_row(second)
    expected = fill['qty'] * fill['price'] * .0001
    assert settled['funding_pnl'] == pytest.approx(expected)
    assert settled['wallet_balance'] == pytest.approx(1000 - fill['fee'] + expected)
    assert len(settled['settled_funding']) == 1
    repeated = f['account'].step_row({**deepcopy(second), 'time_utc': paper.arithmetic.iso(f['capture']+260000)})
    assert repeated['funding_pnl'] == settled['funding_pnl']
    assert len(repeated['settled_funding']) == 1


def test_real_stop_uses_historical_time_even_without_original_clock_or_hourly_data(fixture, monkeypatch):
    f = fixture
    state, entry = enter(f)
    stop = state['position']['stop']
    # Host time is deliberately unrelated to the original recorded period.
    monkeypatch.setattr(paper, 'now_ms', lambda: 9_000_000_000_000)
    row = moved_row(f, seconds=30, price=stop + .1, clock_ok=False, hourly=False)
    stopped = f['account'].step_row(row)
    assert stopped['position'] is None and stopped['fill_count_total'] == 2
    assert stopped['fills'][-1]['reason'] == 'protective_stop'
    assert stopped['fills'][-1]['time_ms'] == f['capture'] + 30000
    assert stopped['fills'][-1]['performance_class'] == replay.LABEL
    assert stopped['risk_monitor']['assessment_time_ms'] == f['capture'] + 30000
    assert stopped['wallet_balance'] == pytest.approx(
        1000 + stopped['realized_pnl'] - stopped['fees_paid'] + stopped['funding_pnl'])
    closed = [json.loads(line) for line in f['account'].path.with_name('closed_trades.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(closed) == 1 and closed[0]['position_closed'] is True
    assert closed[0]['performance_class'] == replay.LABEL
    assert closed[0]['net_pnl'] == pytest.approx(stopped['wallet_balance'] - 1000)
    before = deepcopy(stopped['fills'])
    again = f['account'].step_row(moved_row(f, seconds=60))
    assert again['position'] is None and again['fills'] == before


def source_journal(path, rows, partial=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b''.join(json.dumps(row, separators=(',', ':')).encode()+b'\n' for row in rows)
    path.write_bytes(raw + (partial or b''))


def test_bounded_reader_keeps_first_complete_row_and_discards_partial_future(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,'ROOT',tmp_path)
    at = 1_000_000_000
    rows = [{'time_utc':paper.arithmetic.iso(at+i*30000),'marker':i} for i in range(3)]
    path=tmp_path/'source/observations.jsonl'
    source_journal(path,rows,partial=b'{"time_utc":"unfinished')
    before=path.read_bytes()
    result,info=replay.historical_window(path,at,at+30000)
    assert [item['row']['marker'] for item in result]==[0,1]
    assert info['first_ms']==at and info['last_ms']==at+30000
    assert path.read_bytes()==before


def test_bounded_reader_accepts_a_single_complete_original_capture(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,'ROOT',tmp_path)
    at=1_000_000_000;path=tmp_path/'source/observations.jsonl'
    source_journal(path,[{'time_utc':paper.arithmetic.iso(at),'marker':'first'}])
    result,_=replay.historical_window(path,at,at)
    assert len(result)==1 and result[0]['row']['marker']=='first'


def test_replay_one_uses_original_journal_without_mutation_or_extra_entries(fixture):
    f=fixture
    first=deepcopy(f['row'])
    expected_stop=paper.signals.initial_stop(float(first['book']['bids'][0][0]),-1,
        f['candidate']['original_signal']['atr'],f['seed']['rule'])
    stop_row=moved_row(f,seconds=30,price=expected_stop+1)
    later=moved_row(f,seconds=60)
    future=moved_row(f,seconds=120)
    source=f['root']/'source/observations.jsonl'
    source_journal(source,[first,stop_row,later,future])
    before=digest(source)
    output=f['root']/'data/retrospective_simulation/real_replay'
    summary=replay.replay_one('mu',deepcopy(f['candidate']),deepcopy(f['seed']),
        f['rules'],source,output,f['capture']+60000)
    state=replay.read_json(output/'mu/state.json')
    assert digest(source)==before
    assert state['position'] is None and state['fill_count_total']==2
    assert summary['entry']['time_ms']==f['capture']
    assert summary['main_forward_accounts_modified'] is False
    assert summary['no_network_requests'] is True
    assert summary['performance_class']==replay.LABEL
    assert summary['last_observation_ms']==f['capture']+60000
    assert len(summary['closed_trades'])==1
    assert summary['cash_pnl_usdt']==pytest.approx(state['wallet_balance']-1000)


def test_direct_account_constructor_rejects_forward_account_output(fixture):
    f=fixture
    target=f['root']/'data/parallel_simulation/original/mu/state.json'
    paper.write_json(target,{'untouched':True})
    before=target.read_bytes()
    with pytest.raises(ValueError):
        replay.ReplayStockAccount(target,f['seed']['rule'],f['seed']['symbol'],f['rules'])
    assert target.read_bytes()==before


def test_replay_api_cannot_escape_dedicated_output_tree(fixture):
    f=fixture
    source=f['root']/'source/observations.jsonl';source_journal(source,[f['row']])
    output=f['root']/'data/parallel_simulation/do_not_write'
    with pytest.raises(ValueError):
        replay.replay_one('mu',f['candidate'],f['seed'],f['rules'],source,output,f['capture'])
    assert not output.exists()


def test_initial_snapshot_is_not_reprocessed_after_new_inventory_was_seeded(fixture):
    f=fixture
    first=deepcopy(f['row'])
    first['mark']['time']=f['capture']-1000
    first['book']['T']=f['capture']-1000
    later=moved_row(f,seconds=30)
    source=f['root']/'source/observations.jsonl';source_journal(source,[first,later])
    summary=replay.replay_one('mu',f['candidate'],f['seed'],f['rules'],source,
        f['root']/'data/retrospective_simulation/initial_once',f['capture']+30000)
    assert summary['replay_errors']==[]
    assert summary['entry']['time_ms']==f['capture']
    assert summary['position'] and len(summary['closed_trades'])==0


def test_missing_first_post_entry_settlement_cannot_be_called_complete(fixture):
    f=fixture
    first=deepcopy(f['row']);event_at=f['capture']+120000
    first['mark']['nextFundingTime']=event_at
    later=moved_row(f,seconds=180)
    later['funding_events']=[{'symbol':f['seed']['symbol'],'fundingTime':event_at,'fundingRate':'.0001'}]
    source=f['root']/'source/observations.jsonl';source_journal(source,[first,later])
    summary=replay.replay_one('mu',f['candidate'],f['seed'],f['rules'],source,
        f['root']/'data/retrospective_simulation/missing_first_settlement',f['capture']+180000)
    assert 'funding' in summary['errors']
    assert summary['funding_complete_for_replay_inventory'] is False
    assert summary['net_result_complete'] is False
    assert summary['net_pnl_marked_usdt'] is None
    assert summary.get('estimated_net_if_closed_at_cutoff') is None


def test_known_pending_settlement_keeps_close_estimate_incomplete(fixture):
    f=fixture
    first=deepcopy(f['row']);event_at=f['capture']+120000
    first['mark']['nextFundingTime']=event_at
    early=moved_row(f,seconds=30);early['mark']['nextFundingTime']=event_at
    later=moved_row(f,seconds=180)
    later['funding_events']=[{'symbol':f['seed']['symbol'],'fundingTime':event_at,'fundingRate':'.0001'}]
    source=f['root']/'source/observations.jsonl';source_journal(source,[first,early,later])
    summary=replay.replay_one('mu',f['candidate'],f['seed'],f['rules'],source,
        f['root']/'data/retrospective_simulation/known_pending',f['capture']+180000)
    assert summary['net_result_complete'] is False
    assert summary.get('estimated_net_if_closed_at_cutoff') is None


def test_account_name_path_escape_rejected_before_creating_directories(fixture):
    f=fixture
    source=f['root']/'source/observations.jsonl';source_journal(source,[f['row']])
    output=f['root']/'data/retrospective_simulation/names'
    outside=f['root']/'data/parallel_simulation/escaped'
    with pytest.raises(ValueError):
        replay.replay_one('../../parallel_simulation/escaped',f['candidate'],f['seed'],
            f['rules'],source,output,f['capture'])
    assert not outside.exists()


def test_transient_write_retries_do_not_repeat_financial_step_and_check_containment_first(fixture,monkeypatch):
    f=fixture
    entered,_=enter(f)
    original_writer=paper.write_json
    attempts,sleeps=[],[]
    def sharing_writer(path,value):
        if Path(path)==f['account'].path:
            attempts.append((id(value),deepcopy(value)))
            if len(attempts)<=2:
                raise PermissionError('Simulated Windows shared-reader conflict')
        return original_writer(path,value)
    monkeypatch.setattr(paper,'write_json',sharing_writer)
    monkeypatch.setattr(replay.time,'sleep',sleeps.append)
    stopped=f['account'].step_row(moved_row(f,seconds=30,price=entered['position']['stop']+1))
    assert len(attempts)==3 and len({identity for identity,_ in attempts})==1
    assert attempts[0][1]==attempts[1][1]==attempts[2][1]
    assert sleeps==[.005,.01]
    assert stopped['position'] is None and stopped['fill_count_total']==2
    assert stopped['wallet_balance']==pytest.approx(
        1000+stopped['realized_pnl']-stopped['fees_paid']+stopped['funding_pnl'])
    fills=[json.loads(line) for line in f['account'].path.with_name('fills.jsonl').read_text(encoding='utf-8').splitlines()]
    closed=[json.loads(line) for line in f['account'].path.with_name('closed_trades.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(fills)==2 and len(closed)==1
    assert fills[-1]['reason']=='protective_stop'
    calls_before=len(attempts)
    outside=f['root']/'data/parallel_simulation/never_write/state.json'
    with pytest.raises(ValueError,match='Writes must stay'):
        replay.persist_json(outside,{'sentinel':'must not reach writer'})
    assert len(attempts)==calls_before and sleeps==[.005,.01]
    assert not outside.parent.exists()


@pytest.mark.parametrize('error,expected_attempts,expected_sleeps',[
    (PermissionError,7,[.005,.01,.02,.04,.08,.16]),
    (OSError,1,[]),
])
def test_persistent_write_failure_is_bounded_without_any_state_or_journal_retry(
        fixture,monkeypatch,error,expected_attempts,expected_sleeps):
    f=fixture
    before=f['account'].path.read_bytes()
    attempts,sleeps=[],[]
    value={'sentinel':'unchanged payload'}
    def unavailable(path,payload):
        attempts.append((Path(path),id(payload),deepcopy(payload)))
        raise error('Simulated persistent local write failure')
    monkeypatch.setattr(paper,'write_json',unavailable)
    monkeypatch.setattr(replay.time,'sleep',sleeps.append)
    with pytest.raises(error):
        replay.persist_json(f['account'].path,value)
    assert len(attempts)==expected_attempts
    assert sleeps==expected_sleeps
    assert all(path==f['account'].path and identity==id(value) and payload==value
               for path,identity,payload in attempts)
    assert f['account'].path.read_bytes()==before
    assert not f['account'].path.with_name('fills.jsonl').exists()


def test_supplement_waits_for_availability_filters_symbol_and_preserves_original_clock_failure(fixture):
    f=fixture
    event_at=f['capture']+60000
    f['row']['mark']['nextFundingTime']=event_at
    _,fill=enter(f)
    event={'symbol':f['seed']['symbol'],'fundingTime':event_at,'fundingRate':'.0001'}
    available=f['capture']+120000
    points=[{'event':deepcopy(event),'event_available_by_ms':available},
        {'event':{**event,'symbol':'SNDKUSDT','fundingRate':'.02'},'event_available_by_ms':f['capture']}]
    f['account'].venue.supplements=deepcopy(points)
    f['account'].venue.evidence=[{'symbol':f['seed']['symbol'],
        'minute_ms':event_at//60000*60000,'open':str(fill['price']),'available_by_ms':available}]
    early=moved_row(f,seconds=90)
    early_before=deepcopy(early)
    pending=f['account'].step_row(early)
    assert pending['funding_pnl']==0 and not pending['settled_funding']
    assert early==early_before
    failed_clock=moved_row(f,seconds=120,clock_ok=False)
    original=deepcopy(failed_clock)
    deferred=f['account'].step_row(failed_clock)
    assert deferred['funding_pnl']==0 and not deferred['settled_funding']
    assert deferred['errors']['clock']=='Original clock timeout'
    assert failed_clock==original
    settled=f['account'].step_row(moved_row(f,seconds=130))
    assert settled['funding_pnl']==pytest.approx(fill['qty']*fill['price']*.0001)
    assert len(settled['settled_funding'])==1
    assert f['account'].venue.supplements==points


def test_supplement_and_later_equivalent_primary_payload_settle_one_event_once(fixture):
    f=fixture
    _,fill=enter(f)
    event_at=f['capture']+60000
    event={'symbol':f['seed']['symbol'],'fundingTime':event_at,'fundingRate':'0.00010000'}
    point={'event':deepcopy(event),'event_available_by_ms':f['capture']+90000}
    f['account'].venue.supplements=[deepcopy(point),deepcopy(point)]
    f['account'].venue.evidence=[{'symbol':f['seed']['symbol'],
        'minute_ms':event_at//60000*60000,'open':str(fill['price']),
        'available_by_ms':f['capture']+90000}]
    first=f['account'].step_row(moved_row(f,seconds=120))
    first_pnl=first['funding_pnl']
    assert first_pnl==pytest.approx(fill['qty']*fill['price']*.0001)
    primary=moved_row(f,seconds=150)
    primary['funding_events']=[{**event,'fundingRate':'.0001','markPrice':str(fill['price'])}]
    original=deepcopy(primary)
    repeated=f['account'].step_row(primary)
    assert repeated['funding_pnl']==pytest.approx(first_pnl)
    assert len(repeated['settled_funding'])==1
    rows=f['account'].path.with_name('funding.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(rows)==1
    assert primary==original


def test_conflicting_available_funding_evidence_fails_without_mutating_ledger_or_primary_row(fixture):
    f=fixture
    enter(f)
    event={'symbol':f['seed']['symbol'],'fundingTime':f['capture']+60000,'fundingRate':'.0001'}
    points=[{'event':deepcopy(event),'event_available_by_ms':f['capture']+90000}]
    f['account'].venue.supplements=deepcopy(points)
    row=moved_row(f,seconds=120)
    row['funding_events']=[{**event,'fundingRate':'.0002'}]
    original=deepcopy(row)
    before={path.name:path.read_bytes() for path in f['account'].path.parent.iterdir() if path.is_file()}
    with pytest.raises(ValueError,match='Conflicting historical funding evidence'):
        f['account'].step_row(row)
    assert {path.name:path.read_bytes() for path in f['account'].path.parent.iterdir() if path.is_file()}==before
    assert row==original and f['account'].venue.supplements==points
