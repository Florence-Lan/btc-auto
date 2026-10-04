"""Default entrypoints must follow the terminal's selected rules and execution gates."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import active_strategy
import backtest_execution as backtest
import frozen_strategy
import run_multifactor_shadow as shadow
import simulate_range_swing as sim
from test_hourly_execution_timing import inputs, stream, cfg


@pytest.fixture(autouse=True)
def execution_environment(monkeypatch):
    for key,value in {'SIM_TAKER_FEE':'0.00045','SIM_SLIPPAGE_BPS':'1',
                      'SIM_MAX_LEVERAGE':'2','SIM_MAX_NOTIONAL_USDT':'0'}.items():
        monkeypatch.setenv(key,value)


def test_defaults_follow_selection_changes_without_a_dated_fallback(tmp_path,monkeypatch):
    config_dir=tmp_path/'config'
    config_dir.mkdir()
    selected=config_dir/'selection.json'
    monkeypatch.setattr(sim,'repo_root',lambda:tmp_path)
    for name in ('chosen_a.json','chosen_b.json'):
        candidate=config_dir/name
        candidate.write_text('{}')
        payload={'candidate_path':'config/'+name,'execution_mode':'simulation'}
        selected.write_text(json.dumps(payload))
        (config_dir/'active_simulation_candidate.json').write_text(json.dumps(payload))
        assert active_strategy.candidate_path(selected)==candidate
        assert shadow.parse_args(['--once']).profile==candidate
    selected.write_text(json.dumps({'candidate_path':'../elsewhere.json','execution_mode':'simulation'}))
    with pytest.raises(ValueError,match='in config'):
        active_strategy.candidate_path(selected)


def test_replay_rechecks_calendar_for_an_existing_target(tmp_path):
    hours,base=inputs([100]*6+[107,110,112,114])
    _,sleeve=stream(hours,base)
    event=tmp_path/'events.json'
    hour=3_600_000
    event.write_text(json.dumps({'schema_version':1,'events':[{
        'event_id':'release','published_at_utc':sim.iso_utc_from_ms(0),
        'starts_at_utc':sim.iso_utc_from_ms(7*hour),
        'ends_at_utc':sim.iso_utc_from_ms(8*hour-1),'severity':1,'block_entries':True}]}))
    report=backtest.replay(base,[sleeve],cfg(),4*hour,[],initial=10000,
        entry_report={'execution_entry_context':{'event_snapshot':str(event)}})
    assert report['fills'][0]['time_utc']==sim.iso_utc_from_ms(8*hour+3000)
    assert report['entry_gate_diagnostics']['reason_counts']['current_event_blocks_entries']==12
    assert report['execution_model']=='closed_signal_known_open_v1'


def selected_inputs(tmp_path):
    hour=3_600_000
    start=45*sim.MS_PER_DAY
    end=start+12*hour
    values=[100]*(start//hour)+[100,100,101,101.5,102,102.5,103,103.5,104,104.5,105,105.5]
    hours,base=inputs(values)
    settings=replace(cfg(),timeseries_min_ema_spread_pct=.0005)
    config=asdict(settings)
    engine=Path(sim.__file__)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'freeze_id':'selected_test','config':config,
        'config_sha256':frozen_strategy.canonical_config_hash(config),
        'engine_path':str(engine),'engine_sha256':frozen_strategy.sha256_file(engine)}))
    public_path=tmp_path/'public.json'
    import public_context
    public={'schema_version':1,'events':[{'event_id':'release',
        'published_at_utc':sim.iso_utc_from_ms(start),'starts_at_utc':sim.iso_utc_from_ms(start+6*hour),
        'ends_at_utc':sim.iso_utc_from_ms(start+7*hour-1),'severity':1,'block_entries':True}],
        'coverage_checks':[{'available_at_utc':sim.iso_utc_from_ms(t),
            'sources':{k:{'ok':True} for k in public_context.REQUIRED}}
            for t in range(start,end+hour,hour)]}
    public_path.write_text(json.dumps(public))
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps({'candidate_id':'selected_test','live_orders_allowed':False,
        'base_manifest':str(manifest),'groups':{'btc_momentum':1},'minimum_group_coverage':1,
        'minimum_risk_multiplier':.35,'block_alignment_below':-.35,'block_stress_at':.9,
        'availability_mode':'first_seen','public_context_enabled':True,'policy_expectations_enabled':True,
        'event_snapshot':str(public_path),'strategy_modes':['trend','timeseries_trend'],
        'risk_limits':{'starting_equity_usdt':1000,'soft_drawdown_start_pct':8,
                       'hard_drawdown_stop_pct':15,'portfolio_leverage_cap':2}}))
    factors=tmp_path/'factors.json'
    factors.write_text(json.dumps({'schema_version':1,'series':{'btc_close':[
        [t,t,100,t] for t in range(start-8*sim.MS_PER_DAY,end+hour,hour)]}}))
    market=tmp_path/'market.json.gz'
    sim.save_market_snapshot(market,'BTCUSDT',{'5m':base,'1h':hours},sim.FundingHistory([],[]),0,end)
    funding=tmp_path/'funding.json'
    funding.write_text(json.dumps({'start_utc':sim.iso_utc_from_ms(start),
        'end_utc':sim.iso_utc_from_ms(end),'events':[]}))
    return backtest.parse_args(['--profile',str(profile),'--factor-snapshot',str(factors),
        '--market-snapshot',str(market),'--funding-snapshot',str(funding),
        '--start-utc',sim.iso_utc_from_ms(start),'--end-utc',sim.iso_utc_from_ms(end),
        '--output',str(tmp_path/'result.json')])


def test_selected_pipeline_captures_inputs_uses_account_size_and_current_gates(tmp_path):
    args=selected_inputs(tmp_path)
    report=backtest.run_selected_strategy(args)
    assert report['candidate_id']=='selected_test'
    assert report['summary']['initial_equity']==1000
    assert report['summary']['fills']>0
    assert report['entry_gate_diagnostics']['complete_factor_coverage_pct']==100
    assert report['entry_gate_diagnostics']['public_health_coverage_pct']==100
    assert report['entry_gate_diagnostics']['reason_counts']['current_event_blocks_entries']==12
    assert report['execution_model']=='closed_signal_known_open_v1'
    for name,path in report['captured_inputs'].items():
        assert Path(path).parent!=tmp_path
        assert frozen_strategy.sha256_file(Path(path))==report['input_sha256'][path]
    assert not (tmp_path/'unused_execution_replay.json').exists()
    assert report['places_orders'] is False


def test_selected_pipeline_refuses_to_invent_missing_context_history(tmp_path):
    args=selected_inputs(tmp_path)
    public=Path(json.loads(args.profile.read_text())['event_snapshot'])
    payload=json.loads(public.read_text())
    payload['coverage_checks']=payload['coverage_checks'][1:]
    public.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='cannot be backfilled'):
        backtest.run_selected_strategy(args)
    assert not args.output.exists()


def test_selected_pipeline_uses_tenfold_profile_and_replay_report_caps(tmp_path, monkeypatch):
    args = selected_inputs(tmp_path)
    profile = json.loads(args.profile.read_text())
    profile['risk_limits']['portfolio_leverage_cap'] = 10
    args.profile.write_text(json.dumps(profile))
    monkeypatch.delenv('SIM_MAX_LEVERAGE', raising=False)
    original = backtest.SimulationAccount.reconcile
    seen = []

    def reconcile(account, target, mark, rules, report, **kwargs):
        seen.append(report['config']['portfolio_leverage_cap'])
        return original(account, target, mark, rules, report, **kwargs)

    monkeypatch.setattr(backtest.SimulationAccount, 'reconcile', reconcile)
    report = backtest.run_selected_strategy(args)
    assert seen and set(seen) == {10}
    assert report['config']['leverage'] == 10
    assert report['config']['timeseries_max_leverage'] == 10
    assert report['config']['portfolio_leverage_cap'] == 10
    assert report['runtime_risk_limits']['simulation_leverage_cap'] == 10
    assert report['effective_config_sha256'] == frozen_strategy.canonical_config_hash(report['config'])
    assert report['summary']['fills'] > 0


def test_independent_observation_does_not_overwrite_terminal_state(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    args=selected_inputs(tmp_path)
    terminal_state=tmp_path/'data/paper_trading/selected_test_state.json'
    terminal_state.parent.mkdir(parents=True)
    terminal_state.write_text('original terminal state')
    run=Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(sim,'repo_root',lambda:tmp_path)
    monkeypatch.setattr(shadow.subprocess,'run',run)
    monkeypatch.setattr(shadow.sys,'argv',['shadow','--once','--profile',str(args.profile),
                                        '--snapshot',str(args.factor_snapshot)])
    assert shadow.main()==0
    assert run.call_count==1
    command=run.call_args.args[0]
    assert command[command.index('--state-path')+1]==str(tmp_path/'data/paper_trading/selected_test_shadow_state.json')
    assert terminal_state.read_text()=='original terminal state'
