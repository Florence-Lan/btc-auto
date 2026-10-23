from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_cross_market_observer as observer
import refresh_international_event_calendar as refresh
from international_event_calendar import evaluate_international_event_calendar


@pytest.fixture
def project(tmp_path, monkeypatch):
    calendar = json.loads((ROOT / 'config/international_event_calendar_20261023.json').read_text(encoding='utf-8'))
    template = tmp_path / 'config/calendar.json'
    observer.write(template, calendar)
    output = tmp_path / 'data/observer'
    output.mkdir(parents=True)
    monkeypatch.setattr(observer, 'ROOT', tmp_path)
    spec = {'international_calendar': 'config/calendar.json', 'calendar_refresh_enabled': True}
    return template, output, spec, calendar


def compatible(template, calendar, verified):
    return {**deepcopy(calendar), 'template_sha256': observer.digest(template.read_bytes()),
            'verified_at_ms': verified, 'valid_until_ms': verified + 86_400_000}


def test_calendar_refresh_is_scheduled_and_exact_chosen_bytes_are_archivable(project):
    template, output, spec, calendar = project
    now = calendar['verified_at_ms'] + 1000
    calls = []
    def renew(path, directory, *, asof_ms):
        calls.append(asof_ms)
        observer.write(directory / 'calendar_verified.json', compatible(path, calendar, asof_ms))
        status = {'success': True, 'last_attempt_at_ms': asof_ms, 'next_attempt_after_ms': asof_ms + 21600000}
        observer.write(directory / 'calendar_refresh_status.json', status)
        return status
    chosen, raw, status = observer.maintained_calendar(output, spec, asof_ms=now, refresher=renew)
    assert status['success'] and chosen['verified_at_ms'] == now
    assert json.loads(raw) == chosen
    assert raw == (output / 'calendar_verified.json').read_bytes()
    observer.maintained_calendar(output, spec, asof_ms=now + 1000,
        refresher=lambda *a, **k: pytest.fail('six-hour refresh is not due'))
    assert calls == [now]


def test_failed_refresh_never_renews_compatible_calendar_or_clears_its_expiry(project):
    template, output, spec, calendar = project
    old = calendar['verified_at_ms']
    observer.write(output / 'calendar_verified.json', compatible(template, calendar, old))
    raw_before = (output / 'calendar_verified.json').read_bytes()
    def fail(*args, **kwargs):
        raise TimeoutError('optional public request unavailable')
    now = old + 25 * 3600000
    chosen, raw, status = observer.maintained_calendar(output, spec, asof_ms=now, refresher=fail)
    assert not status['success'] and chosen['verified_at_ms'] == old
    assert raw == raw_before and (output / 'calendar_verified.json').read_bytes() == raw_before
    assert not evaluate_international_event_calendar(chosen, asof_ms=now)['allowed']
    observer.maintained_calendar(output, spec, asof_ms=now + 1000,
        refresher=lambda *a, **k: pytest.fail('failed refresh backs off one hour'))


def test_changed_template_does_not_inherit_old_runtime_verification(project):
    template, output, spec, calendar = project
    now = calendar['verified_at_ms'] + 1000
    observer.write(output / 'calendar_verified.json', compatible(template, calendar, now))
    observer.write(output / 'calendar_refresh_status.json', {'success': True,
        'last_attempt_at_ms': now, 'next_attempt_after_ms': now + 21600000})
    updated = deepcopy(calendar)
    updated['calendar_id'] += '_new'
    observer.write(template, updated)
    chosen, raw, _ = observer.maintained_calendar(output, spec, asof_ms=now)
    assert chosen['calendar_id'] == updated['calendar_id']
    assert chosen['verified_at_ms'] == calendar['verified_at_ms']
    assert raw == template.read_bytes()
