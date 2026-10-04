"""Accounting and source-gap checks for fixed-profile research."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extend_stock_swing_snapshot_5m as download
import research_stock_swing_robustness as research


def row(timestamp):
    return [timestamp, "100", "101", "99", "100", "1", timestamp + download.STEP - 1]


def test_refetch_internal_gaps_even_when_cache_has_latest_bar(monkeypatch):
    step = download.STEP
    cached = {t: row(t) for t in (0, step, 4 * step, 6 * step)}
    requests = []

    def fetch(session, base, prefix, endpoint, params):
        requests.append(params)
        return [row(t) for t in range(params["startTime"], params["endTime"] + 1, step)]

    monkeypatch.setattr(download, "fetch_with_retry", fetch)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    assert download.repair_gaps(None, "base", "prefix", "/mark", "MU", 0, 7 * step, cached) == 3
    assert sorted(cached) == list(range(0, 7 * step, step))
    assert [(p["startTime"], p["endTime"]) for p in requests] == [(2 * step, 4 * step - 1), (5 * step, 6 * step - 1)]


def test_unavailable_gap_never_gets_interpolated(monkeypatch):
    cached = {0: row(0), 2 * download.STEP: row(2 * download.STEP)}
    monkeypatch.setattr(download, "fetch_with_retry", lambda *args: [])
    with pytest.raises(ValueError, match="cannot repair"):
        download.repair_gaps(None, "base", "prefix", "/mark", "MU", 0, 3 * download.STEP, cached)
    assert download.STEP not in cached


def test_monthly_continuous_inventory_and_midnight_accounting():
    result = {"summary": {"initial_equity": 100}, "equity_path": [
        {"time_utc": "2026-08-31T23:55:00Z", "equity": 110},
        {"time_utc": "2026-09-01T00:00:00Z", "equity": 120},
        {"time_utc": "2026-09-01T00:05:00Z", "equity": 115},
    ], "trades": [{"exit_utc": "2026-09-01T00:00:00Z", "net_return_initial_margin_pct": 120}]}
    months = research.monthly_contributions(result)
    assert [m["month_utc"] for m in months] == ["2026-08", "2026-09"]
    assert [m["marked_pnl"] for m in months] == [20, -5]
    assert sum(m["marked_pnl"] for m in months) == 115 - 100
    assert months[1]["start_mark_equity"] == 120
    assert months[1]["closed_trades"] == months[1]["net120_targets"] == 1


def test_winner_dependency_excludes_terminal_open_estimates():
    trades = [{"net_pnl": p, "direction": side, "net_return_initial_margin_pct": r}
              for p, side, r in [(30, "long", 120), (-20, "short", -40), (5, "long", 20)]]
    result = research.trade_diagnostics(trades, 100, samples=100)
    assert result["closed_return_pct_initial_capital"] == 15
    assert result["closed_return_without_largest_winner_pct"] == -15
    assert result["largest_winner_share_gross_profit_pct"] == pytest.approx(30 / 35 * 100)
    assert result["by_side"]["long"]["net120_targets"] == 1
    assert research.trade_diagnostics([], 100)["historical_trade_block_bootstrap"] is None


def test_block_bootstrap_constant_sample_has_exact_mean():
    trades = [{"net_pnl": 5, "direction": "long", "net_return_initial_margin_pct": 10} for _ in range(4)]
    result = research.trade_diagnostics(trades, 100, samples=100)["historical_trade_block_bootstrap"]
    for quantile in ("p05", "p50", "p95"):
        assert result["mean_trade_pnl_pct_initial_capital_" + quantile] == 5


def test_snapshot_audit_rejects_gap_and_unfinished_bar():
    step = download.STEP
    source = {"start_ms": 0, "trade_1h": [row(0)], "mark_1h": [row(0)],
              "trade_5m": [row(t) for t in range(0, 12 * step, step)],
              "mark_5m": [row(t) for t in range(0, 12 * step, step)]}
    snapshot = {"execution_step_ms": step, "end_ms_exclusive": 12 * step, "symbols": {"MU": source}}
    assert research.audit_snapshot(snapshot)["MU"]["mark"]["max_hourly_ohlc_difference"] == 0
    source["mark_5m"][3][6] = 12 * step
    with pytest.raises(ValueError, match="Invalid completed"):
        research.audit_snapshot(snapshot)
    source["mark_5m"].pop(3)
    with pytest.raises(ValueError, match="Missing, duplicate"):
        research.audit_snapshot(snapshot)


def test_complete_suffix_discards_gap_and_prior_warmup_for_every_symbol():
    step = download.STEP
    end = 2 * research.engine.FOUR_HOURS
    executions = [row(t) for t in range(0, end, step)]
    hourly = [row(t) for t in range(0, end, research.engine.HOUR)]
    sources = {symbol: {"start_ms": 0, "trade_1h": hourly, "mark_1h": hourly,
                        "index_1h": hourly, "trade_5m": executions,
                        "mark_5m": [r for r in executions if symbol != "MU" or r[0] != 3 * step]}
               for symbol in ("MU", "SNDK")}
    original = {"execution_step_ms": step, "end_ms_exclusive": end, "symbols": sources}
    trimmed, provenance = research.complete_suffix_snapshot(original)
    assert provenance["missing"] == {"MU:mark_5m": [3 * step]}
    for source in trimmed["symbols"].values():
        assert source["start_ms"] == research.engine.FOUR_HOURS
        assert source["trade_1h"][0][0] == research.engine.FOUR_HOURS
        assert source["mark_5m"][0][0] == research.engine.FOUR_HOURS
    assert original["symbols"]["MU"]["start_ms"] == 0
    assert len(original["symbols"]["SNDK"]["trade_5m"]) == end // step
    assert set(research.audit_snapshot(trimmed)) == {"MU", "SNDK"}


def test_gap_at_end_cannot_be_hidden_by_suffix_trim():
    snapshot = {"end_ms_exclusive": 4 * download.STEP, "symbols": {
        "MU": {"start_ms": 0, "trade_5m": [row(0)], "mark_5m": [row(0)]}
    }}
    with pytest.raises(ValueError, match="No complete suffix"):
        research.complete_suffix_snapshot(snapshot)


def test_hourly_signals_use_exact_complete_execution_prices():
    rows = [row(t) for t in range(0, research.engine.HOUR, download.STEP)]
    rows[0][1] = "100.5"
    rows[3][2] = "110"
    rows[7][3] = "90"
    rows[-1][4] = "99.5"
    hourly = research.hourly_from_five_minute(rows)
    assert hourly == [[0, 100.5, 110.0, 90.0, 99.5, 12.0, research.engine.HOUR - 1]]
    with pytest.raises(ValueError, match="twelve complete"):
        research.hourly_from_five_minute(rows[:-1])
