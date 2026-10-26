"""Named Brent manual-roll invariants; all quotes are offline fixtures."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cross_market_observations as co
import cross_market_risk as risk

MINUTE = 60_000
NOW = 1_793_022_583_000
INCEPTION = 1_792_718_906_206
OLD_CUTOFF = '2026-10-28T00:00:00Z'
NEW_CUTOFF = '2026-11-25T00:00:00Z'


@pytest.fixture(autouse=True)
def isolated_deadlines():
    co._DEADLINES.clear()
    yield
    co._DEADLINES.clear()


def source(january=True):
    symbol, month = ('BZF27.NYM', '2027-01') if january else ('BZZ26.NYM', '2026-12')
    return {'id': 'yahoo_brent_jan_2027' if january else 'yahoo_brent_dec_2026',
            'parser': 'yahoo_chart', 'provider': 'Offline named Brent fixture',
            'url': 'https://query2.finance.yahoo.com/v8/finance/chart/' + symbol,
            'expected_symbol': symbol, 'expected_type': 'FUTURE', 'expected_currency': 'USD',
            'timezone': 'America/New_York', 'poll_seconds': 60, 'timeout_seconds': .5,
            'contract_review_cutoff_utc': NEW_CUTOFF if january else OLD_CUTOFF,
            'instrument': {'symbol': symbol, 'contract_month': month, 'exchange': 'NYMEX',
                           'currency': 'USD', 'automatic_roll': False, 'continuous_contract': False},
            'fields': {'brent_usd_per_barrel': {'unit': 'USD/barrel', 'max_age_seconds': 1200,
                                             'required_intraday': True}}}


def config(s):
    return {'schema_version': 1, 'mode': 'OBSERVATION_ONLY', 'places_orders': False,
            'collection_timeout_seconds': 1, 'sources': [s]}


def quote(s, observed=NOW - 10 * MINUTE, value=110.36):
    return {'chart': {'result': [{'meta': {'symbol': s['expected_symbol'],
            'instrumentType': 'FUTURE', 'currency': 'USD', 'exchangeName': 'NYM',
            'exchangeTimezoneName': 'America/New_York', 'regularMarketTime': observed // 1000,
            'regularMarketPrice': value, 'previousClose': 111.7}}], 'error': None}}


def test_january_quote_uses_its_own_named_contract_denominator():
    s = source()
    row = co.parse_yahoo_chart(quote(s), s, NOW)['brent_usd_per_barrel']
    assert row['observed_at_ms'] == NOW - 10 * MINUTE
    assert row['provider_previous_close'] == 111.7
    assert row['return_basis'] == 'same_named_instrument_provider_previousClose'
    assert row['return_pct'] == pytest.approx((110.36 / 111.7 - 1) * 100)
    assert s['instrument']['automatic_roll'] is False


@pytest.mark.parametrize('field,value', [('symbol', 'BZZ26.NYM'), ('symbol', 'BZ=F'),
                                       ('instrumentType', 'EQUITY'), ('currency', 'EUR')])
def test_new_source_rejects_old_continuous_or_wrong_identity(field, value):
    s = source()
    body = quote(s)
    body['chart']['result'][0]['meta'][field] = value
    with pytest.raises(co.ObservationError, match='instrument_identity_mismatch'):
        co.parse_yahoo_chart(body, s, NOW)


def test_local_december_review_cutoff_is_enforced_before_official_termination():
    old = source(False)
    cutoff = co.parse_timestamp_ms(OLD_CUTOFF)
    assert cutoff < co.parse_timestamp_ms('2026-10-30T00:00:00Z')
    body = quote(old, observed=cutoff - 10 * MINUTE)
    with pytest.raises(co.ObservationError, match='contract_review_due'):
        co.parse_yahoo_chart(body, old, cutoff)
    # The explicit January identity has its own separately reviewed cutoff.
    new = source()
    assert co.parse_yahoo_chart(quote(new, observed=cutoff), new, cutoff)['brent_usd_per_barrel']['value'] == 110.36
    assert old['contract_review_cutoff_utc'] == OLD_CUTOFF


def test_future_quote_time_is_rejected_for_a_valid_january_identity():
    s = source()
    with pytest.raises(co.ObservationError, match='future_observation'):
        co.parse_yahoo_chart(quote(s, observed=NOW + 1000), s, NOW)


def test_contract_change_gets_new_first_receipt_without_reusing_or_destroying_old_snapshot():
    old, new = source(False), source()
    old_snapshot = co.collect_observations(config(old), now_ms=NOW - MINUTE,
                                          fetcher=lambda s, t: quote(s, value=115.2))
    before = deepcopy(old_snapshot)
    snapshot = co.collect_observations(config(new), previous=old_snapshot, now_ms=NOW,
                                      fetcher=lambda s, t: quote(s))
    old_row = old_snapshot['observations']['brent_usd_per_barrel']
    row = snapshot['observations']['brent_usd_per_barrel']
    assert old_snapshot == before
    assert row['observed_at_ms'] == old_row['observed_at_ms']
    assert row['first_seen_at_ms'] == row['available_at_ms'] == NOW
    assert old_row['first_seen_at_ms'] == NOW - MINUTE
    assert row['source_config_id'] != old_row['source_config_id']
    assert row['instrument']['contract_month'] == '2027-01'
    assert row['freshness']['status'] == 'fresh'
    assert snapshot['required_intraday_missing_or_stale'] == []
    again = co.collect_observations(config(new), previous=snapshot, now_ms=NOW + MINUTE,
                                   fetcher=lambda s, t: quote(s))
    assert again['observations']['brent_usd_per_barrel']['first_seen_at_ms'] == NOW
    assert again['observations']['brent_usd_per_barrel']['observed_at_ms'] == row['observed_at_ms']


def test_refetch_of_identical_january_print_cannot_extend_its_twenty_minute_freshness():
    s = source()
    before = co.collect_observations(config(s), now_ms=NOW, fetcher=lambda s, t: quote(s))
    after = co.collect_observations(config(s), previous=before, now_ms=NOW + 11 * MINUTE,
                                   fetcher=lambda s, t: quote(s))
    row = after['observations']['brent_usd_per_barrel']
    assert row['first_seen_at_ms'] == NOW
    assert row['freshness']['age_ms'] == 21 * MINUTE
    assert row['freshness']['status'] == 'stale'
    assert not row['eligible_intraday']


def risk_snapshot(at, brent_source, oil=110.36):
    values = {'nq_dec_2026': 30_000, 'brent_usd_per_barrel': oil, 'usdjpy': 157}
    return {'observations': {key: {'value': value, 'observed_at_ms': at,
        'first_seen_at_ms': at + 1, 'available_at_ms': at + 1, 'precision': 'timestamp',
        'instrument': brent_source['instrument'] if key == 'brent_usd_per_barrel' else key}
        for key, value in values.items()}}


def test_roll_preserves_epoch_and_old_history_but_requires_a_new_same_identity_reference():
    old = risk_snapshot(NOW - 30 * MINUTE, source(False), oil=115.2)
    history = risk.update_forward_history(old, {}, asof_ms=NOW - 30 * MINUTE + 1,
                                          inception_ms=INCEPTION)
    first_january = risk_snapshot(NOW, source())
    history = risk.update_forward_history(first_january, history, asof_ms=NOW + 1,
                                          inception_ms=INCEPTION)
    warm = risk.evaluate_cross_market_risk(first_january, {}, asof_ms=NOW + 1, history=history)
    assert warm['status'] == 'warming_up'
    assert warm['reasons'] == ['brent_usd_per_barrel:forward_history_warming_up']
    assert 'brent_usd_per_barrel' not in warm['features']
    assert len(history['brent_usd_per_barrel']) == 2  # no history reset or splice
    later = risk_snapshot(NOW + 30 * MINUTE, source(), oil=111)
    history = risk.update_forward_history(later, history, asof_ms=NOW + 30 * MINUTE + 1,
                                          inception_ms=INCEPTION)
    decision = risk.evaluate_cross_market_risk(later, {}, asof_ms=NOW + 30 * MINUTE + 1,
                                               history=history)
    assert decision['status'] == 'clear' and decision['allowed']
    oil = decision['features']['brent_usd_per_barrel']
    assert oil['reference']['value'] == 110.36
    assert oil['reference']['instrument'] == oil['current']['instrument']
    assert oil['change_pct'] == pytest.approx((111 / 110.36 - 1) * 100)
    assert len(history['brent_usd_per_barrel']) == 3
