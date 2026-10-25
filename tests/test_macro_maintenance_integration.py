"""Offline collector integration: daily provenance, bounded reads and sessions.

Portable fixtures preserve the real Radiocor report shape. When the locally
archived original HTML is present, one extra test replays those exact bytes.
All public responses and clocks are injected; no test sends network requests.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cross_market_observations as co
import europe_bond_observations as eb

ARTICLE_URL = ('https://www.borsaitaliana.it/borsa/notizie/radiocor/finance/dettaglio/'
               'euro-zone-benchmark-10year-sovereign-bond-yields-nRC_23102026_1734_551141600.html?lang=en')
RESEARCH = ROOT / 'data/research/macro_maintenance_20261025'


def ms(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


OBSERVED = ms('2026-10-23T15:05:00Z')
RECEIVED = ms('2026-10-25T13:46:01.512Z')


def brief(*, france='5.04'):
    rows = [('Germany', '3.35', 'up 0.01 percentage points'),
            ('France', france, 'down 0.04 percentage points'),
            ('Belgium', '4.44', 'down 0.01 percentage points'),
            ('Italy', '4.59', 'down 0.01 percentage points'),
            ('Spain', '4.05', 'unchanged')]
    return ('<html><h1>Euro zone benchmark 10-year sovereign bond yields</h1>'
            '<span class="t-text -block -line-1-3em">'
            '<p>(Il Sole 24 Ore Radiocor) - Milano, 23 ott - Euro zone benchmark '
            '10-year sovereign bond yields at 1505 GMT.</p>'
            + ''.join(f'<p>{country} - yield {value}%; {change} from the previous close.</p>'
                      for country, value, change in rows)
            + '<p>(RADIOCOR) 23-10-26 17:34:49 (0551) 5 NNNN</p></span></html>')


def archive(*, candidate=True, next_page=None):
    link = (f'<a href="{ARTICLE_URL}">{eb.TITLE}</a>' if candidate else
            '<a href="/borsa/notizie/other.html">Other news</a>')
    return (f'<ul id="itemListArchive"><li>{link}</li></ul>'
            + (f'<a title="Next" href="?lang=en&amp;page={next_page}">Next</a>'
               if next_page else ''))


def euro_source():
    return {'id': 'radiocor_europe', 'parser': 'europe_bond_brief',
            'provider': eb.PROVIDER, 'url': eb.INDEX_URL, 'timezone': 'UTC',
            'poll_seconds': 30, 'timeout_seconds': .5, 'maximum_index_pages': 3,
            'instrument': {'kind': 'provider_generic_sovereign_benchmark_yield', 'tenor': '10Y'},
            'fields': {f'{country}_10y_pct': {'unit': 'percent', 'max_age_days': 4,
                       'max_age_seconds': 3600, 'required_intraday': True}
                       for country in ('de', 'fr', 'it', 'es', 'be')}}


def conf(*sources, spreads=False):
    result = {'schema_version': 1, 'mode': 'OBSERVATION_ONLY',
              'collection_timeout_seconds': 1, 'sources': list(sources)}
    if spreads:
        result['derived_spreads'] = {f'{country}_de_10y_spread_bp': {
            'left': f'{country}_10y_pct', 'right': 'de_10y_pct',
            'same_snapshot_required': True, 'max_age_days': 4}
            for country in ('fr', 'it', 'es', 'be')}
    return result


def euro_feed(body=None, *, index=None, calls=None):
    def fetch(source, timeout):
        if calls is not None:
            calls.append(source['url'])
        if source['url'] == eb.INDEX_URL:
            return archive() if index is None else index
        assert source['url'] == ARTICLE_URL
        return brief() if body is None else body
    return fetch


def calendar():
    return {'schema_version': 1, 'session_id': 'nq', 'schedule_kind': 'cme_nymex_23x5',
            'timezone': 'America/New_York', 'basis': 'scheduled_exchange_reference',
            'source_urls': ['https://www.cmegroup.com/trading-hours.html'],
            'source_published_date': None,
            'first_seen_at_ms': ms('2026-10-24T00:00:00Z'),
            'verified_at_ms': ms('2026-10-24T00:00:00Z'),
            'valid_from_ms': ms('2026-10-24T04:00:00Z'),
            'valid_until_ms': ms('2026-11-02T05:00:00Z'),
            'reviewed_local_dates': [f'2026-10-{d}' for d in range(24, 32)] + ['2026-11-01'],
            'holiday_exclusions': []}


def quote_source():
    return {'id': 'nq_named', 'parser': 'yahoo_chart', 'provider': 'Yahoo Finance',
            'url': 'https://query1.finance.yahoo.com/v8/finance/chart/NQZ26.CME',
            'timezone': 'America/New_York', 'expected_symbol': 'NQZ26.CME',
            'expected_type': 'FUTURE', 'expected_currency': 'USD',
            'contract_review_cutoff_utc': '2026-11-20T00:00:00Z',
            'poll_seconds': 30, 'timeout_seconds': .5,
            'instrument': {'name': 'NQ December 2026', 'contract_month': '2026-12',
                           'automatic_roll': False},
            'session_calendar': calendar(),
            'fields': {'nq_dec_2026': {'unit': 'index_points', 'max_age_seconds': 1200,
                                      'required_intraday': True}}}


def quote(source, at, *, value=30000):
    return {'chart': {'result': [{'meta': {
        'symbol': source['expected_symbol'], 'instrumentType': source['expected_type'],
        'currency': source['expected_currency'], 'exchangeTimezoneName': source['timezone'],
        'regularMarketTime': at // 1000, 'regularMarketPrice': value,
        'previousClose': 29700, 'longName': 'NQ December 2026'}}], 'error': None}}


@pytest.fixture(autouse=True)
def isolated_deadlines():
    co._DEADLINES.clear()
    yield
    co._DEADLINES.clear()


def test_portable_real_shape_cannot_be_promoted_to_intraday_even_when_time_is_fresh():
    # Injected receipt is on the report date; its precise GMT time is within the
    # deliberately generous 1-hour timestamp-age limit. Daily still wins.
    at = OBSERVED + 31 * 60_000
    calls = []
    snapshot = co.collect_observations(conf(euro_source(), spreads=True), now_ms=at,
                                       fetcher=euro_feed(calls=calls))
    assert calls == [eb.INDEX_URL, ARTICLE_URL]
    assert snapshot['status'] == 'degraded'
    assert set(snapshot['required_intraday_missing_or_stale']) == set(euro_source()['fields'])
    for row in snapshot['observations'].values():
        assert not row['eligible_intraday'] and not row['freshness']['eligible_intraday']
        assert row['freshness']['status'] == 'daily_current'
        assert row['freshness']['eligible_daily']
    assert snapshot['observations']['fr_de_10y_spread_bp']['value'] == 169
    assert snapshot['observations']['de_10y_pct']['observed_at_ms'] == OBSERVED
    assert snapshot['observations']['de_10y_pct']['available_at_ms'] == at


def test_locally_archived_original_html_replays_through_collector_without_network():
    capture = RESEARCH / 'europe_source_capture.json'
    if not capture.exists():
        pytest.skip('Optional original HTML capture remains local; portable fixture is tested separately.')
    rows = json.loads(capture.read_text(encoding='utf-8'))
    bodies = {}
    for row in rows:
        raw = (ROOT / row['path']).read_bytes()
        assert len(raw) == row['size']
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        bodies[row['source_url']] = raw
    calls = []
    def fetch(source, timeout):
        calls.append(source['url'])
        return bodies[source['url']]
    receipt = ms(rows[1]['received_at_utc'])
    snapshot = co.collect_observations(conf(euro_source(), spreads=True), now_ms=receipt,
                                       fetcher=fetch)
    assert calls == [eb.INDEX_URL, ARTICLE_URL]
    assert {k: snapshot['observations'][k]['value'] for k in euro_source()['fields']} == {
        'de_10y_pct': 3.35, 'fr_10y_pct': 5.04, 'it_10y_pct': 4.59,
        'es_10y_pct': 4.05, 'be_10y_pct': 4.44}
    row = snapshot['observations']['fr_10y_pct']
    assert row['first_seen_at_ms'] == receipt and row['observed_at_ms'] == OBSERVED
    assert row['provenance']['published_at_ms'] is None
    assert not row['eligible_intraday'] and row['freshness']['eligible_daily']


def test_refetch_correction_and_expiry_preserve_causal_report_identity():
    config = conf(euro_source(), spreads=True)
    first = co.collect_observations(config, now_ms=RECEIVED, fetcher=euro_feed())
    again = co.collect_observations(config, previous=first, now_ms=RECEIVED + 31_000,
                                    fetcher=euro_feed('<nav>Changed navigation</nav>' + brief()))
    for key in euro_source()['fields']:
        assert again['observations'][key]['available_at_ms'] == RECEIVED
        assert again['observations'][key]['observed_at_ms'] == OBSERVED
    corrected_at = RECEIVED + 62_000
    corrected = co.collect_observations(config, previous=again, now_ms=corrected_at,
                                        fetcher=euro_feed(brief(france='5.05')))
    for key in euro_source()['fields']:
        assert corrected['observations'][key]['available_at_ms'] == corrected_at
        assert corrected['observations'][key]['snapshot_id'] != first['observations'][key]['snapshot_id']
    assert corrected['observations']['fr_de_10y_spread_bp']['value'] == 170
    expired_at = ms('2026-10-28T00:00:00Z')
    expired = co.collect_observations(config, previous=corrected, now_ms=expired_at,
                                      fetcher=euro_feed(brief(france='5.05')))
    assert expired['observations']['de_10y_pct']['available_at_ms'] == corrected_at
    assert expired['observations']['de_10y_pct']['freshness']['status'] == 'stale'
    assert expired['observations']['fr_de_10y_spread_bp']['value'] is None
    assert not expired['observations']['de_10y_pct']['eligible_intraday']


def test_clock_rollback_cannot_make_first_received_daily_report_available_in_the_past():
    config = conf(euro_source(), spreads=True)
    first = co.collect_observations(config, now_ms=RECEIVED, fetcher=euro_feed())
    result = co.collect_observations(config, previous=first, now_ms=RECEIVED - 1,
                                     fetcher=lambda *_: pytest.fail('Clock rollback must not refetch.'))
    assert result['sources']['radiocor_europe']['status'] == 'clock_rollback'
    row = result['observations']['fr_10y_pct']
    assert row['freshness']['status'] == 'not_yet_available'
    assert not row['freshness']['eligible_daily'] and not row['eligible_intraday']
    assert result['observations']['fr_de_10y_spread_bp']['value'] is None


@pytest.mark.parametrize('field,value', [
    ('snapshot_id', 'different-report'), ('source_id', 'other-provider'),
    ('source_url', ARTICLE_URL + '&different=true'), ('observed_at_ms', OBSERVED + 60_000),
    ('observed_date', '2026-10-22'), ('frequency', 'intraday'), ('unit', 'basis_points'),
])
def test_collector_refuses_spreads_from_mixed_cached_components(field, value):
    config = conf(euro_source(), spreads=True)
    first = co.collect_observations(config, now_ms=RECEIVED, fetcher=euro_feed())
    mixed = deepcopy(first)
    mixed['observations']['fr_10y_pct'][field] = value
    result = co.collect_observations(config, previous=mixed, now_ms=RECEIVED + 1,
                                     fetcher=lambda *_: pytest.fail('Valid poll cache should be used.'))
    spread = result['observations']['fr_de_10y_spread_bp']
    assert spread['value'] is None and not spread['eligible_intraday']
    assert spread['missing_reason'] == 'missing_stale_or_asynchronous_percent_components'


@pytest.mark.parametrize('at,status', [
    ('2026-10-25T13:00:00Z', 'market_closed'),
    ('2026-10-26T21:30:00Z', 'market_maintenance'),
])
def test_fresh_quote_in_closed_or_maintenance_session_is_rejected_without_erasing_quote_status(at, status):
    source = quote_source()
    now = ms(at)
    result = co.collect_observations(conf(source), now_ms=now,
                                     fetcher=lambda s, _: quote(s, now - 10_000))
    row = result['observations']['nq_dec_2026']
    assert row['freshness']['status'] == status
    assert row['freshness']['quote_status'] == 'fresh'
    assert row['fetch']['status'] == 'ok' and row['value'] == 30000
    assert not row['eligible_intraday'] and not row['market_session']['freshness_override']
    assert result['required_intraday_missing_or_stale'] == ['nq_dec_2026']


@pytest.mark.parametrize('kind', ['empty', 'none', 'expired', 'unreviewed', 'excluded', 'future_verification'])
def test_unknown_session_cannot_be_bypassed_by_fresh_quote(kind):
    source = quote_source()
    now = ms('2026-10-25T13:00:00Z')
    if kind in {'empty', 'none'}:
        source['session_calendar'] = {} if kind == 'empty' else None
    elif kind == 'expired':
        source['session_calendar']['valid_until_ms'] = now
    elif kind == 'unreviewed':
        source['session_calendar']['reviewed_local_dates'].remove('2026-10-25')
    elif kind == 'excluded':
        source['session_calendar']['holiday_exclusions'] = ['2026-10-25']
    else:
        source['session_calendar']['verified_at_ms'] = now + 1
    result = co.collect_observations(conf(source), now_ms=now,
                                     fetcher=lambda s, _: quote(s, now - 10_000))
    row = result['observations']['nq_dec_2026']
    assert row['market_session']['phase'] == 'unknown'
    assert row['freshness']['status'] == 'session_unknown'
    assert row['freshness']['quote_status'] == 'fresh' and not row['eligible_intraday']


def test_open_session_does_not_refresh_a_stale_quote():
    source = quote_source()
    now = ms('2026-10-26T14:00:00Z')
    result = co.collect_observations(conf(source), now_ms=now,
                                     fetcher=lambda s, _: quote(s, now - 1_201_000))
    row = result['observations']['nq_dec_2026']
    assert row['market_session']['phase'] == 'open'
    assert row['freshness']['status'] == 'stale' and not row['eligible_intraday']


def test_closed_session_retains_fetch_failure_and_original_quote_clock():
    source = quote_source()
    now = ms('2026-10-25T13:00:00Z')
    first = co.collect_observations(conf(source), now_ms=now,
                                    fetcher=lambda s, _: quote(s, now - 10_000))
    secret = 'DO_NOT_LEAK_PROXY_PASSWORD'
    def fail(*_):
        raise requests.exceptions.ProxyError('https://user:' + secret + '@proxy.invalid')
    second = co.collect_observations(conf(source), previous=first, now_ms=now + 31_000,
                                     fetcher=fail)
    row = second['observations']['nq_dec_2026']
    assert row['fetch']['status'] == 'error' and row['fetch']['error']['category'] == 'proxy'
    assert row['freshness']['status'] == 'market_closed'
    assert row['freshness']['quote_status'] == 'fresh'
    assert row['available_at_ms'] == now and row['observed_at_ms'] == now - 10_000
    assert not row['eligible_intraday'] and secret not in json.dumps(second)


def test_identity_mismatch_rejects_cached_fresh_value_and_does_not_echo_payload():
    source = quote_source()
    now = ms('2026-10-26T14:00:00Z')
    config = conf(source)
    first = co.collect_observations(config, now_ms=now,
                                    fetcher=lambda s, _: quote(s, now - 10_000))
    assert first['observations']['nq_dec_2026']['eligible_intraday']
    secret = 'DO_NOT_LEAK_PROVIDER_TOKEN'
    payload = quote(source, now + 30_000, value=1)
    payload['chart']['result'][0]['meta'].update(
        symbol='https://user:' + secret + '@evil.invalid', currency={'secret': secret},
        longName=secret)
    second = co.collect_observations(config, previous=first, now_ms=now + 31_000,
                                     fetcher=lambda *_: payload)
    row = second['observations']['nq_dec_2026']
    assert row['value'] == 30000 and row['available_at_ms'] == now
    assert row['freshness']['status'] == 'instrument_identity_mismatch'
    assert row['fetch']['error']['reason'] == 'instrument_identity_mismatch:symbol,currency'
    assert not row['eligible_intraday'] and secret not in json.dumps(second)


def test_archive_requests_use_decreasing_one_budget_and_no_more_than_three_pages():
    clock = SimpleNamespace(value=0.0)
    clock.monotonic = lambda: clock.value
    calls = []
    def fetch(source, remaining):
        calls.append((source['url'], remaining))
        clock.value += .2
        page = len(calls)
        if page <= 2:
            return archive(candidate=False, next_page=page + 1)
        return archive() if page == 3 else brief()
    original_time = co.time
    try:
        co.time = clock
        result = co._europe_fetch(euro_source(), fetch, lambda: RECEIVED, 1)
    finally:
        co.time = original_time
    assert len(calls) == 4 and calls[-1][0] == ARTICLE_URL
    assert [remaining for _, remaining in calls] == pytest.approx([1, .8, .6, .4])
    assert result['de_10y_pct']['value'] == 3.35


@pytest.mark.parametrize('candidate', [False, True])
def test_expired_archive_request_cannot_start_a_later_page_or_article(candidate):
    clock = SimpleNamespace(value=0.0)
    clock.monotonic = lambda: clock.value
    calls = []
    def fetch(source, remaining):
        calls.append(source['url'])
        clock.value = 1.001
        return archive(candidate=candidate, next_page=2)
    original_time = co.time
    try:
        co.time = clock
        with pytest.raises(TimeoutError):
            co._europe_fetch(euro_source(), fetch, lambda: RECEIVED, 1)
    finally:
        co.time = original_time
    assert calls == [eb.INDEX_URL]


@pytest.mark.parametrize('stall_at', ['index', 'article'])
def test_timed_out_real_worker_never_promotes_late_payload_or_starts_a_later_get(monkeypatch, stall_at):
    entered, release, finished = (threading.Event() for _ in range(3))
    calls = []
    original = co._europe_fetch
    def tracked(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(co, '_europe_fetch', tracked)
    def fetch(source, timeout):
        calls.append(source['url'])
        if stall_at == 'article' and source['url'] == eb.INDEX_URL:
            return archive()
        entered.set()
        assert release.wait(2), 'Test must release the fake stalled response.'
        return archive() if source['url'] == eb.INDEX_URL else brief()
    source = euro_source()
    source['timeout_seconds'] = .05
    try:
        result = co.collect_observations(conf(source), now_ms=RECEIVED, fetcher=fetch)
        assert entered.is_set()
        assert result['sources'][source['id']]['error']['category'] == 'timeout'
        assert result['observations']['de_10y_pct']['value'] is None
        before = deepcopy(result)
    finally:
        release.set()
        assert finished.wait(2)
    assert calls == ([eb.INDEX_URL] if stall_at == 'index' else [eb.INDEX_URL, ARTICLE_URL])
    assert result == before


def test_missing_brief_stops_after_three_real_archive_pages_without_guessing_article_url():
    calls = []
    def fetch(source, timeout):
        calls.append(source['url'])
        return archive(candidate=False, next_page=len(calls) + 1)
    result = co.collect_observations(conf(euro_source()), now_ms=RECEIVED, fetcher=fetch)
    assert calls == [eb.INDEX_URL, eb.INDEX_URL + '&page=2', eb.INDEX_URL + '&page=3']
    error = result['sources']['radiocor_europe']['error']
    assert error['reason'] == 'europe_daily_brief_not_found_in_bounded_archive'
    assert result['observations']['de_10y_pct']['value'] is None


@pytest.mark.parametrize('status', [403, 429])
def test_public_denial_stops_discovery_and_keeps_bounded_cooldown(status):
    source, calls = euro_source(), []
    def fail(source, timeout):
        calls.append(source['url'])
        raise co.PublicHttpError(status)
    first = co.collect_observations(conf(source), now_ms=RECEIVED, fetcher=fail)
    result = co.collect_observations(conf(source), previous=first, now_ms=RECEIVED + 60_000,
                                     fetcher=lambda *_: pytest.fail('Denial cooldown forbids another request.'))
    assert calls == [eb.INDEX_URL]
    assert result['sources'][source['id']]['error']['http_status'] == status
    assert result['observations']['de_10y_pct']['value'] is None
