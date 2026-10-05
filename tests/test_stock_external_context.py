"""External context must reject unavailable/stale data without looking ahead."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stock_external_context as external


def payload(side=1):
    series = {}
    for name in external.SYMBOLS:
        values = [100 + side * i * .1 for i in range(70)] if name in ('nq', 'es', 'sox') else [20 if name == 'vix' else 100 if name == 'dollar' else 5] * 70
        series[name] = [[i * external.HOUR, (i + 1) * external.HOUR + external.DELAY, value]
                        for i, value in enumerate(values)]
    return {'series': series}


@pytest.mark.parametrize('side', [1, -1])
def test_consensus_retains_both_directions_and_rejects_countertrend(side):
    context = external.MarketContext(payload(side))
    now = 61 * external.HOUR + external.DELAY
    assert context.decision(now, side, external.OVERLAYS[2])['allowed']
    opposite = context.decision(now, -side, external.OVERLAYS[2])
    assert not opposite['allowed']
    assert 'nq_trend_disagrees' in opposite['reasons']


def test_missing_sector_blocks_full_context_without_blocking_nq_only():
    data = payload()
    del data['series']['sox']
    context = external.MarketContext(data)
    now = 61 * external.HOUR + external.DELAY
    assert context.decision(now, 1, external.OVERLAYS[1])['allowed']
    result = context.decision(now, 1, external.OVERLAYS[2])
    assert not result['allowed'] and 'sox_missing' in result['reasons']


def test_no_external_information_can_become_usable_before_delay():
    context = external.MarketContext(payload())
    end = 61 * external.HOUR
    before, _ = context.at('nq', end + external.DELAY - 1)
    after, _ = context.at('nq', end + external.DELAY)
    assert before['provider_open_ms'] == 59 * external.HOUR
    assert after['provider_open_ms'] == 60 * external.HOUR


def test_future_data_cannot_change_past_admission():
    data = payload()
    now = 61 * external.HOUR + external.DELAY
    before = external.MarketContext(data).decision(now, 1, external.OVERLAYS[2])
    changed = copy.deepcopy(data)
    for rows in changed['series'].values():
        for row in rows[61:]:
            row[2] = 10000
    after = external.MarketContext(changed).decision(now, 1, external.OVERLAYS[2])
    assert before == after


def test_stale_futures_block_entry_instead_of_reusing_old_direction():
    data = payload()
    context = external.MarketContext(data)
    last = data['series']['nq'][-1][1]
    assert context.decision(last + external.MAX_AGE['nq'], 1, external.OVERLAYS[1])['allowed']
    result = context.decision(last + external.MAX_AGE['nq'] + 1, 1, external.OVERLAYS[1])
    assert not result['allowed'] and result['reasons'] == ['nq_stale']


def test_volatility_spike_and_joint_dollar_yield_rise_veto_are_explicit():
    data = payload()
    now = 61 * external.HOUR + external.DELAY
    data['series']['vix'][60][2] = 30
    result = external.MarketContext(data).decision(now, 1, external.OVERLAYS[2])
    assert not result['allowed'] and 'vix_risk_spike' in result['reasons']
    data = payload()
    data['series']['dollar'][60][2] = 101
    data['series']['yield10'][60][2] = 5.1
    result = external.MarketContext(data).decision(now, 1, external.OVERLAYS[2])
    assert not result['allowed'] and 'dollar_and_yield_rise_veto_long' in result['reasons']


def test_price_control_does_not_require_external_data_and_early_close_is_invalid():
    assert external.MarketContext({'series': {}}).decision(0, 1, external.OVERLAYS[0])['allowed']
    data = payload()
    data['series']['nq'][0][1] -= external.DELAY
    with pytest.raises(ValueError, match='declared delay'):
        external.MarketContext(data)
