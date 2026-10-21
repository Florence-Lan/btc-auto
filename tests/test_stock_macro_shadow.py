"""Exercise admission at real paper retries/fills and preserve protected exits."""
from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_stock_macro_shadow as shadow
import run_parallel_simulation as paper
from test_stock_swing_confirmation import worker_fixture


def macro_permit(reference, boundary=None):
    allowed = boundary is None or reference < boundary
    return {'enabled': True, 'allowed': allowed, 'status': 'clear' if allowed else 'blackout',
            'reasons': [] if allowed else ['stock_macro_event_blackout'],
            'checked_at_ms': reference, 'event_ids': [] if allowed else ['fomc'],
            'valid_until_ms': reference + 3_600_000, 'coverage_end_ms': reference + 86_400_000,
            'next_event': {'starts_at_ms': boundary} if allowed and boundary is not None else None,
            'scope': 'new_risk_only', 'protective_exits_allowed': True}


def worker(tmp_path, monkeypatch, side=1, boundary=None):
    original, path, clock = worker_fixture(tmp_path, monkeypatch, side)
    config = deepcopy(original.config)
    config[shadow.macro.POLICY_KEY] = {**shadow.macro.POLICY,
        'snapshot_path': 'data/calendar.json', 'effective_from_ms': 1}
    result = shadow.MacroAccount(path, original.venue, 'MUUSDT', config)
    paper.write_json(path, {**paper.read_json(path), 'rule': config})
    def evaluate(config, symbol, reference, **kwargs):
        return macro_permit(reference, boundary)
    monkeypatch.setattr(shadow.macro, 'evaluate', evaluate)
    return result, path, clock


@pytest.mark.parametrize('side', [1, -1])
def test_macro_blackout_blocks_stock_entry_both_directions(tmp_path, monkeypatch, side):
    account, path, _ = worker(tmp_path, monkeypatch, side, boundary=0)
    result = account.step()
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['pending_signal'] is not None
    assert result['entry_blockers'] == ['stock_macro_event_blackout']
    assert result['stock_event_risk']['status'] == 'blackout'
    assert result['stock_event_risk']['macro']['event_ids'] == ['fomc']
    assert result['stock_macro_event_risk']['scope'] == 'new_risk_only'
    assert result['wallet_balance'] == 1000 and result['fees_paid'] == 0


@pytest.mark.parametrize('side', [1, -1])
def test_final_fill_rechecks_macro_window_after_depth_calculation(tmp_path, monkeypatch, side):
    account, _, clock = worker(tmp_path, monkeypatch, side)
    threshold = clock['now'] + 1000
    monkeypatch.setattr(shadow.macro, 'evaluate', lambda cfg, symbol, now, **kwargs:
                        macro_permit(now, threshold))
    original = paper.book_fill
    def delayed(*args, **kwargs):
        fill = original(*args, **kwargs)
        clock['now'] += 2000
        return fill
    monkeypatch.setattr(paper, 'book_fill', delayed)
    result = account.step()
    assert result['position'] is None and result['fill_count_total'] == 0
    assert result['stock_macro_event_risk']['phase'] == 'fill'
    assert result['wallet_balance'] == 1000 and result['fees_paid'] == 0


def test_pending_signal_is_not_renewed_when_retry_enters_macro_window(tmp_path, monkeypatch):
    account, _, clock = worker(tmp_path, monkeypatch)
    threshold = clock['now'] + 1000
    monkeypatch.setattr(shadow.macro, 'evaluate', lambda cfg, symbol, now, **kwargs:
                        macro_permit(now, threshold))
    clock['volume'] = 0
    before = account.step()
    clock.update(now=clock['now'] + 300000, volume=10000)
    after = account.step()
    assert after['pending_signal'] == before['pending_signal']
    assert after['fill_count_total'] == 0
    assert after['entry_blockers'] == ['stock_macro_event_blackout']


def test_macro_failure_never_prevents_protective_stop(tmp_path, monkeypatch):
    account, path, clock = worker(tmp_path, monkeypatch, boundary=0)
    state = paper.read_json(path)
    state['position'] = {'entry': 100, 'qty': 1, 'direction': 1,
        'entry_time': clock['now'] - 5000, 'signal_time': clock['now'] - 5000,
        'initial_stop': 98, 'stop': 111, 'entry_fee': 0, 'margin': 10, 'funding': 0}
    state['position_history'] = [{'time_ms': clock['now'] - 5000, 'signed_qty': 1}]
    paper.write_json(path, state)
    result = account.step()
    assert result['position'] is None and result['fill_count_total'] == 1
    assert result['fills'][-1]['reason'] == 'protective_stop'
    assert result['wallet_balance'] == pytest.approx(1000 + result['realized_pnl'] - result['fees_paid'] + result['funding_pnl'])


def test_design_keeps_only_declared_gate_difference_and_stresses_costs():
    spec, rules, _ = shadow.load_design()
    assert spec['places_orders'] is False
    for symbol in shadow.SYMBOLS:
        normal = rules[symbol]['baseline_normal']
        candidate = deepcopy(rules[symbol]['candidate_normal'])
        assert candidate.pop(shadow.macro.POLICY_KEY)
        assert candidate == normal
        doubled = rules[symbol]['candidate_double']
        assert doubled['taker_fee_rate_assumption'] == normal['taker_fee_rate_assumption'] * 2
        assert doubled['adverse_slippage_fraction_assumption'] == normal['adverse_slippage_fraction_assumption'] * 2


def test_restart_conserves_all_twelve_ledgers_and_rejects_fingerprint_drift(tmp_path):
    spec, rules, fingerprint = shadow.load_design()
    inception = spec['effective_from_ms'] + 1
    manifest = shadow.prepare(tmp_path, spec, rules, fingerprint, inception)
    path = tmp_path / 'MUUSDT/candidate_normal/state.json'
    state = paper.read_json(path)
    state.update(wallet_balance=997., realized_pnl=-2., fees_paid=1., fill_count_total=2)
    paper.write_json(path, state)
    frozen = path.read_bytes()
    assert shadow.prepare(tmp_path, spec, rules, fingerprint, inception + 10, True) == manifest
    assert path.read_bytes() == frozen
    with pytest.raises(ValueError, match='Frozen'):
        shadow.prepare(tmp_path, spec, rules, {**fingerprint, 'changed': True}, inception, True)
    path.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        shadow.prepare(tmp_path, spec, rules, fingerprint, inception, True)
    assert not path.exists()


def test_missing_current_context_replaces_prior_permission(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    source = root / 'data/snapshots/public_context_latest.json'
    source.parent.mkdir(parents=True)
    source.write_text('{"valid":true}')
    output = tmp_path / 'experiment'
    monkeypatch.setattr(shadow, 'ROOT', root)
    shadow.freeze_context(output)
    source.unlink()
    shadow.freeze_context(output)
    assert json.loads((output / 'current_context/public_context.json').read_text()) == {}


def combined_fixture(tmp_path, monkeypatch):
    """Both real gate formats, independent files, and an execution clock."""
    from test_stock_event_risk import calendar_data
    from test_stock_macro_event_risk import NOW, ms, snapshot
    now = ms(NOW)
    hour = 3_600_000
    calendar = calendar_data(anchor=now + 26 * hour)
    calendar.update(first_seen_at_ms=now - hour, available_at_ms=now - hour,
                    verified_at_ms=now - hour, valid_until_ms=now + 48 * hour,
                    coverage_start_ms=now - hour, coverage_end_ms=now + 72 * hour)
    calendar['symbol_reviews']['MUUSDT']['verified_at_ms'] = now - hour
    calendar['events'][0].update(first_seen_at_ms=now - hour, available_at_ms=now - hour,
                                 verified_at_ms=now - hour)
    company_path = tmp_path / 'company.json'
    macro_path = tmp_path / 'public_context.json'
    company_path.write_text(json.dumps(calendar), encoding='utf-8')
    public = snapshot(at=now)
    macro_path.write_text(json.dumps(public), encoding='utf-8')
    config = {paper.stock_event_risk.POLICY_KEY: {**paper.stock_event_risk.POLICY,
        'calendar_path': str(company_path), 'effective_from_ms': now - hour},
        shadow.macro.POLICY_KEY: {**shadow.macro.POLICY, 'snapshot_path': str(macro_path),
                                  'effective_from_ms': now - hour}}
    account = shadow.MacroAccount.__new__(shadow.MacroAccount)
    account.config, account.symbol = config, 'MUUSDT'
    account.signal_bars, account.signal_timeframe = [], '1h'
    clock = {'now': now}
    account.now_ms = lambda: clock['now']
    return account, clock, calendar, public, company_path, macro_path


@pytest.mark.parametrize('fault', ['blackout', 'expiry'])
def test_slow_macro_read_crossing_real_company_boundary_cannot_authorize_fill(tmp_path, monkeypatch, fault):
    account, clock, calendar, _, company_path, macro_path = combined_fixture(tmp_path, monkeypatch)
    initial = clock['now']
    if fault == 'blackout':
        calendar['events'][0]['anchor_time_ms'] = initial + 24 * 3_600_000 + 1000
        reason = 'stock_event_blackout'
    else:
        calendar['valid_until_ms'] = initial + 1000
        reason = 'stock_event_calendar_expired'
    company_path.write_text(json.dumps(calendar), encoding='utf-8')
    original = Path.read_bytes
    reads = {company_path: 0, macro_path: 0}
    def slow_read(path):
        payload = original(path)
        if path in reads:
            reads[path] += 1
        if path == macro_path:
            clock['now'] += 2000
        return payload
    monkeypatch.setattr(Path, 'read_bytes', slow_read)
    outcome = account.record_stock_event_risk({}, 'fill', initial)
    assert not outcome['allowed'] and reason in outcome['reasons']
    assert outcome['status'] == ('blackout' if fault == 'blackout' else 'unavailable')
    assert outcome['checked_at_ms'] == initial + 2000
    assert outcome['protective_exits_allowed']
    assert reads == {company_path: 1, macro_path: 1}


@pytest.mark.parametrize('fault', ['blackout', 'expiry'])
def test_final_clock_bounds_real_company_permit_after_its_evaluation(tmp_path, monkeypatch, fault):
    account, clock, calendar, _, company_path, _ = combined_fixture(tmp_path, monkeypatch)
    initial = clock['now']
    if fault == 'blackout':
        calendar['events'][0]['anchor_time_ms'] = initial + 24 * 3_600_000 + 1000
        reason = 'stock_event_blackout'
    else:
        calendar['valid_until_ms'] = initial + 1000
        reason = 'stock_event_calendar_expired'
    company_path.write_text(json.dumps(calendar), encoding='utf-8')
    original = paper.stock_event_risk.evaluate
    def delayed_return(*args, **kwargs):
        outcome = original(*args, **kwargs)
        assert outcome['allowed']
        clock['now'] += 2000
        return outcome
    monkeypatch.setattr(paper.stock_event_risk, 'evaluate', delayed_return)
    outcome = account.record_stock_event_risk({}, 'fill', initial)
    assert not outcome['allowed'] and reason in outcome['reasons']
    assert outcome['status'] == ('blackout' if fault == 'blackout' else 'unavailable')
    assert outcome['checked_at_ms'] == initial + 2000


@pytest.mark.parametrize('fault,delay,allowed', [
    ('blackout', 999, True), ('blackout', 1000, False), ('blackout', 2000, False),
    ('expiry', 1000, True), ('expiry', 1001, False), ('expiry', 2000, False)])
def test_slow_company_read_cannot_outlive_macro_permit(tmp_path, monkeypatch, fault, delay, allowed):
    from test_stock_macro_event_risk import iso
    account, clock, _, public, company_path, macro_path = combined_fixture(tmp_path, monkeypatch)
    initial = clock['now']
    if fault == 'blackout':
        public['calendars'][0]['scheduled_at_utc'] = iso(initial + 30 * 60_000 + 1000)
        public['events'] = shadow.macro.public_context.make_events([], public['calendars'])
        reason = 'stock_macro_event_blackout'
    else:
        for state in public['coverage_checks'][0]['sources'].values():
            state['last_success_at_utc'] = iso(initial - 3_600_000 + 1000)
        reason = 'stock_macro_event_source_expired_during_evaluation'
    macro_path.write_text(json.dumps(public), encoding='utf-8')
    original = Path.read_bytes
    reads = {company_path: 0, macro_path: 0}
    def slow_read(path):
        payload = original(path)
        if path in reads:
            reads[path] += 1
        if path == company_path:
            clock['now'] += delay
        return payload
    monkeypatch.setattr(Path, 'read_bytes', slow_read)
    outcome = account.record_stock_event_risk({}, 'fill', initial)
    assert outcome['allowed'] is allowed
    if not allowed:
        assert reason in outcome['reasons']
    assert outcome['status'] == ('clear' if allowed else 'blackout' if fault == 'blackout' else 'unavailable')
    assert outcome['checked_at_ms'] == initial + delay
    assert outcome['macro']['checked_at_ms'] == initial
    assert reads == {company_path: 1, macro_path: 1}


@pytest.mark.parametrize('missing', ['valid_until_ms', 'coverage_end_ms', 'next_event',
                                   'status', 'checked_at_ms'])
def test_unbounded_macro_permit_fails_closed(tmp_path, monkeypatch, missing):
    account, clock, *_ = combined_fixture(tmp_path, monkeypatch)
    outcome = macro_permit(clock['now'])
    outcome.pop(missing)
    monkeypatch.setattr(shadow.macro, 'evaluate', lambda *args, **kwargs: outcome)
    result = account.record_stock_event_risk({}, 'fill', clock['now'])
    assert not result['allowed']
    assert result['status'] == 'unavailable'
    assert result['reasons'] == ['stock_macro_combined_permit_metadata_invalid:stock_macro_event']
    assert result['protective_exits_allowed']


@pytest.mark.parametrize('missing', ['valid_until_ms', 'coverage_end_ms', 'next_event'])
def test_unbounded_company_permit_fails_closed(tmp_path, monkeypatch, missing):
    account, clock, *_ = combined_fixture(tmp_path, monkeypatch)
    original = paper.stock_event_risk.evaluate
    def incomplete(*args, **kwargs):
        outcome = original(*args, **kwargs)
        outcome.pop(missing)
        return outcome
    monkeypatch.setattr(paper.stock_event_risk, 'evaluate', incomplete)
    result = account.record_stock_event_risk({}, 'fill', clock['now'])
    assert not result['allowed']
    assert result['reasons'] == ['stock_macro_combined_permit_metadata_invalid:stock_event']


def test_not_active_macro_permission_cannot_survive_activation_during_company_read(tmp_path, monkeypatch):
    account, clock, _, _, company_path, _ = combined_fixture(tmp_path, monkeypatch)
    initial = clock['now']
    account.config[shadow.macro.POLICY_KEY]['effective_from_ms'] = initial + 1000
    original = Path.read_bytes
    def slow_read(path):
        payload = original(path)
        if path == company_path:
            clock['now'] += 2000
        return payload
    monkeypatch.setattr(Path, 'read_bytes', slow_read)
    result = account.record_stock_event_risk({}, 'fill', initial)
    assert not result['allowed']
    assert result['reasons'] == ['stock_macro_event_activation_recheck_required']


def test_company_only_baseline_keeps_original_path_without_macro_evaluation(tmp_path, monkeypatch):
    account, clock, *_ = combined_fixture(tmp_path, monkeypatch)
    account.config.pop(shadow.macro.POLICY_KEY)
    def forbidden(*args, **kwargs):
        raise AssertionError('Baseline must not read or combine macro permission')
    monkeypatch.setattr(shadow.macro, 'evaluate', forbidden)
    state = {}
    outcome = account.record_stock_event_risk(state, 'fill', clock['now'])
    assert outcome['allowed'] and outcome['status'] == 'clear'
    assert 'macro' not in outcome and 'permit_deadlines' not in outcome
    assert state['stock_macro_event_risk'] is None
