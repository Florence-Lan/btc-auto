"""Producer source maintenance must preserve the existing exact-profile guard.

These existing-state fixtures are local only. A narrowly approved metadata
repair is represented explicitly; the production loader never renews a pin.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
with patch('dotenv.load_dotenv', return_value=False):
    import paper_trade_frozen_portfolio as paper

OLD_CONTEXT = '1' * 64
APPROVED_CONTEXT = '2' * 64


@pytest.fixture(autouse=True)
def no_network_or_existing_state_save(monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, 'request',
        Mock(side_effect=AssertionError('Profile identity tests forbid network')))
    monkeypatch.setattr(paper, 'save_state',
        Mock(side_effect=AssertionError('Existing-state loading must not save or renew clocks')))


@pytest.fixture
def producer_files(tmp_path):
    manifest = {'freeze_id': 'local_frozen_profile_fixture', 'config_sha256': 'a' * 64}
    profile = {
        'strategy_modes': ['trend', 'range', 'timeseries_trend'],
        'macro_snapshot': None, 'macro_min_multiplier': .35,
        'macro_block_score': -.8, 'macro_factors': ['equities', 'vix', 'dollar'],
        'tiered_drawdown': True, 'soft_drawdown_start_pct': 8.,
        'hard_drawdown_stop_pct': 15., 'drawdown_min_multiplier': .35,
        'event_snapshot': str(tmp_path / 'context.json'),
        'factor_profile_sha256': 'b' * 64,
        'entry_timing_policy': {'mode': 'local_entry_fixture', 'entry_validity_minutes': 60},
        'btc_pullback_entry_policy': {'mode': 'local_pullback_fixture', 'expiry_minutes': 60},
        'factor_snapshot': str(tmp_path / 'factors.json.gz'),
        'factor_engine_sha256': 'c' * 64, 'portfolio_risk_sha256': 'd' * 64,
        'public_context_sha256': OLD_CONTEXT, 'public_text_transport_sha256': 'e' * 64,
        'event_risk_sha256': 'f' * 64, 'hourly_execution_model': 'local_closed_hour_fixture',
        'hourly_execution_sha256': '3' * 64, 'execution_portfolio_sha256': '4' * 64,
    }
    assert len(profile) == 22
    state = {'version': 1, 'mode': 'frozen_portfolio_shadow', 'places_orders': False,
        'symbol': 'BTCUSDT', **manifest, 'created_at_utc': '2026-10-07T13:50:45+00:00',
        'updated_at_utc': '2026-10-28T17:25:27+00:00', 'observations': 3528,
        'summary': {'label': '保留原始观察结果', 'net_pnl': 12.5, 'closed_trades': 19,
                    'nested': {'history': [1, 2, 3]}},
        'profile': profile,
        'policy_revisions': [{'revision_id': 'previous_revision', 'effective_at_ms': 1791766900000}],
        'strategy_effective_at_utc': '2026-10-11T00:00:00+00:00',
        'btc_pullback_entry_effective_at_utc': '2026-10-20T00:00:00+00:00'}
    path = tmp_path / 'producer_state.json'
    raw = (json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    path.write_bytes(raw)
    report = tmp_path / 'report.json'
    report.write_bytes(b'{"generated_at_utc":"2026-10-28T17:25:27Z","historical":true}\n')
    main_state = tmp_path / 'main_state.json'
    main_state.write_bytes(b'{"wallet_balance":997.78,"position_qty":0}\n')
    fills = tmp_path / 'main_fills.jsonl'
    fills.write_bytes(b'{"sequence":1,"qty":0.002,"fee":0.07}\n')
    return path, state, manifest, {p: p.read_bytes() for p in (path, report, main_state, fills)}


def approved_profile(state):
    expected = deepcopy(state['profile'])
    expected['public_context_sha256'] = APPROVED_CONTEXT
    return expected


def assert_files_equal(snapshot):
    assert all(path.read_bytes() == raw for path, raw in snapshot.items())


def test_old_context_pin_is_rejected_without_resetting_state_or_publishing_report(producer_files):
    path, state, manifest, before = producer_files
    with pytest.raises(RuntimeError, match='different shadow profile'):
        paper.load_or_create_state(path, manifest, 'BTCUSDT', approved_profile(state))
    assert_files_equal(before)
    assert json.loads(path.read_bytes()) == state


def test_explicit_single_pin_repair_allows_read_only_idempotent_loading(producer_files):
    path, state, manifest, before = producer_files
    expected = approved_profile(state)
    raw = before[path]
    assert raw.count(OLD_CONTEXT.encode('ascii')) == 1
    migrated = raw.replace(OLD_CONTEXT.encode('ascii'), APPROVED_CONTEXT.encode('ascii'))
    wanted = deepcopy(state)
    wanted['profile']['public_context_sha256'] = APPROVED_CONTEXT
    assert json.loads(migrated) == wanted
    assert migrated.replace(APPROVED_CONTEXT.encode('ascii'), OLD_CONTEXT.encode('ascii')) == raw
    # This is the explicitly declared fixture migration, not behavior supplied
    # by load_or_create_state or a bypass of its exact comparison.
    path.write_bytes(migrated)
    after_migration = {**before, path: migrated}
    for _ in range(2):
        assert paper.load_or_create_state(path, manifest, 'BTCUSDT', expected) == wanted
        assert_files_equal(after_migration)
    assert {k: v for k, v in wanted.items() if k != 'profile'} == {
        k: v for k, v in state.items() if k != 'profile'}
    assert {k: v for k, v in wanted['profile'].items() if k != 'public_context_sha256'} == {
        k: v for k, v in state['profile'].items() if k != 'public_context_sha256'}


@pytest.mark.parametrize('field,value', [
    ('soft_drawdown_start_pct', 9.), ('hard_drawdown_stop_pct', 16.),
    ('drawdown_min_multiplier', .5), ('tiered_drawdown', False),
    ('strategy_modes', ['timeseries_trend']), ('factor_profile_sha256', '5' * 64),
    ('factor_engine_sha256', '6' * 64), ('portfolio_risk_sha256', '7' * 64),
    ('public_text_transport_sha256', '8' * 64), ('event_risk_sha256', '9' * 64),
    ('hourly_execution_sha256', '0' * 64), ('execution_portfolio_sha256', '5' * 64),
    ('hourly_execution_model', 'unapproved_model'),
    ('entry_timing_policy', {'mode': 'unapproved'}),
    ('btc_pullback_entry_policy', {'mode': 'unapproved'}),
])
def test_approved_context_does_not_allow_another_risk_rule_or_source_change(producer_files, field, value):
    path, state, manifest, before = producer_files
    expected = approved_profile(state)
    state = deepcopy(state)
    state['profile'] = deepcopy(expected)
    path.write_bytes((json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    unchanged = {**before, path: path.read_bytes()}
    unapproved = deepcopy(expected)
    unapproved[field] = value
    with pytest.raises(RuntimeError, match='different shadow profile'):
        paper.load_or_create_state(path, manifest, 'BTCUSDT', unapproved)
    assert_files_equal(unchanged)


@pytest.mark.parametrize('mutation', ['missing_field', 'extra_field', 'old_context'])
def test_complete_profile_identity_is_still_required_after_pin_maintenance(producer_files, mutation):
    path, state, manifest, before = producer_files
    expected = approved_profile(state)
    state['profile'] = deepcopy(expected)
    path.write_bytes(json.dumps(state).encode('utf-8'))
    unchanged = {**before, path: path.read_bytes()}
    wrong = deepcopy(expected)
    if mutation == 'missing_field': wrong.pop('portfolio_risk_sha256')
    elif mutation == 'extra_field': wrong['unapproved_policy'] = True
    else: wrong['public_context_sha256'] = OLD_CONTEXT
    with pytest.raises(RuntimeError, match='different shadow profile'):
        paper.load_or_create_state(path, manifest, 'BTCUSDT', wrong)
    assert_files_equal(unchanged)


@pytest.mark.parametrize('field,message', [('freeze_id', 'different frozen strategy'),
                                         ('config_sha256', 'config hash mismatch')])
def test_source_pin_maintenance_cannot_bypass_frozen_base_identity(producer_files, field, message):
    path, state, manifest, before = producer_files
    state['profile'] = approved_profile(state)
    path.write_bytes(json.dumps(state).encode('utf-8'))
    unchanged = {**before, path: path.read_bytes()}
    wrong_manifest = {**manifest, field: 'unapproved'}
    with pytest.raises(RuntimeError, match=message):
        paper.load_or_create_state(path, wrong_manifest, 'BTCUSDT', state['profile'])
    assert_files_equal(unchanged)
