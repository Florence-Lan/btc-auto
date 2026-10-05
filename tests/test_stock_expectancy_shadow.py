"""Forward quote availability and external gating preserve protective exits."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_stock_expectancy_shadow as shadow
from stock_external_context import HOUR, DELAY, MarketContext


def market(tmp_path):
    source=shadow.ForwardNQ(tmp_path)
    rows=[[i*HOUR,i*HOUR+HOUR+DELAY,100+i] for i in range(80)]
    source.context=MarketContext({'series':{'nq':rows}})
    source.first_seen_ms=rows[-1][1]
    return source,source.first_seen_ms


def test_actual_first_seen_and_source_freshness_required(tmp_path):
    source,now=market(tmp_path)
    assert source.decision(now,1)['allowed']
    assert not source.decision(now-1,1)['allowed']
    assert not source.decision(now,-1)['allowed']
    assert not source.decision(now+20*60_000+1,1)['allowed']
    source.error='fetchfailed'
    assert not source.decision(now,1)['allowed']


def test_every_retry_rechecks_direction_and_expiry_without_disabling_exits(tmp_path,monkeypatch):
    source,now=market(tmp_path)
    settings=shadow.GuardedSettings({'entry_enabled':True,'target_margin_return':.6,
        'max_holding_calendar_days':7},source,now+1000)
    monkeypatch.setattr(shadow.paper,'now_ms',lambda:now)
    settings.side=1
    assert settings.get('entry_enabled')
    settings.side=-1
    assert not settings.get('entry_enabled')
    # Exit parameters remain accessible when external admission rejects entry.
    assert settings['target_margin_return']==.6
    assert settings.get('max_holding_calendar_days')==7
    settings.side=1
    monkeypatch.setattr(shadow.paper,'now_ms',lambda:now+1000)
    assert not settings.get('entry_enabled')
    assert 'forward_experiment_expired' in settings.last_decision['reasons']


def test_no_pending_signal_cannot_create_macro_only_trade(tmp_path):
    source,now=market(tmp_path)
    result=source.decision(now,None)
    assert not result['allowed']
    assert result['reasons']==['no_pending_stock_signal']
