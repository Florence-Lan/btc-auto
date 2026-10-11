"""Cost-aware BTC episode accounting and full-ledger fail-closed behavior."""
import copy
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import btc_profit_target as target


EPOCH = "2026-10-01T00:00:00+00:00"
POLICY = {"mode": "net_margin_target_v1", "target_margin_return": 0.60,
          "margin_leverage": 10}


def ms(value):
    return round(datetime.fromisoformat(value).timestamp() * 1000)


def fill(sequence, side="BUY", quantity=1, price=100, fee=0.045, pnl=0, **metadata):
    return {"sequence": sequence, "time_utc": f"2026-10-01T00:{sequence:02d}:00+00:00",
            "side": side, "quantity": quantity, "price": price, "fee": fee,
            "realized_pnl": pnl, "signal_time_ms": sequence * 1000,
            "symbol": "BTCUSDT", "mode": "simulation", **metadata}


def account(rows=(), funding=(), *, entry=100):
    quantity = Decimal(0)
    history = []
    for row in rows:
        quantity += Decimal(str(row["quantity"])) * (1 if row["side"] == "BUY" else -1)
        history.append({"time_ms": ms(row["time_utc"]), "signed_qty": float(quantity)})
    fees = sum(row["fee"] for row in rows)
    realized = sum(row["realized_pnl"] for row in rows)
    funded = sum(row["payment"] for row in funding)
    return {"mode": "simulation", "symbol": "BTCUSDT", "created_at_utc": EPOCH,
            "fill_count_total": len(rows), "fills": list(rows)[-200:],
            "position_history": history, "position_qty": float(quantity),
            "entry_price": entry if quantity else 0.0, "initial_balance": 1000,
            "fees_paid": fees, "realized_pnl": realized, "funding_pnl": funded,
            "wallet_balance": 1000 + realized - fees + funded,
            "funding_settlements": list(funding)}


def funding(time_ms, quantity, payment, mark=100):
    return {"time_ms": time_ms, "signed_qty": quantity, "payment": payment,
            "mark_price": mark, "rate": -payment / (quantity * mark) if quantity else 0}


@pytest.mark.parametrize("bad", [
    False, {}, {**POLICY, "mode": "unknown"},
    {**POLICY, "target_margin_return": True},
    {**POLICY, "target_margin_return": float("nan")},
    {**POLICY, "target_margin_return": float("inf")},
    {**POLICY, "target_margin_return": 0.599},
    {**POLICY, "margin_leverage": True},
    {**POLICY, "margin_leverage": 0},
    {**POLICY, "margin_leverage": 10.01},
])
def test_policy_rejects_invalid_values_and_targets_below_sixty_percent(bad):
    with pytest.raises(ValueError):
        target.validate_policy(bad)


def test_policy_normalization_and_flat_account():
    assert target.validate_policy(None) is None
    assert target.validate_policy(POLICY) == POLICY
    result = target.evaluate_profit_target(account(), 100)
    assert result["status"] == "flat" and not result["should_exit"]
    assert result["ledger"]["complete"]


@pytest.mark.parametrize("side, raw_sixty_mark, beyond_target", [
    ("BUY", 106, 106.2), ("SELL", 94, 93.8),
])
def test_both_directions_require_sixty_percent_after_entry_and_exit_costs(
    side, raw_sixty_mark, beyond_target,
):
    state = account([fill(1, side)])
    before = copy.deepcopy(state)
    at_gross_sixty = target.evaluate_profit_target(state, raw_sixty_mark)
    assert at_gross_sixty["status"] == "below_target"
    assert at_gross_sixty["projected_net_roi_pct"] < 60
    after = target.evaluate_profit_target(state, beyond_target)
    assert after["status"] == "target_reached" and after["should_exit"]
    assert after["projected_net_roi_pct"] >= 60
    assert after["initial_margin_usdt"] == 10
    assert after["target_net_pnl_usdt"] == 6
    assert state == before  # A decision does not mutate account or order state.
    solved = target.evaluate_profit_target(state, after["target_mark_price"])
    assert solved["projected_net_pnl_usdt"] == pytest.approx(6)


@pytest.mark.parametrize("payment", [-0.5, 0.5])
def test_real_funding_debits_and_credits_change_target_price(payment):
    row = fill(1)
    state = account([row], [funding(ms(row["time_utc"]) + 1000, 1, payment)])
    result = target.evaluate_profit_target(state, 106.2)
    base = target.evaluate_profit_target(account([row]), 106.2)
    assert result["projected_net_pnl_usdt"] == pytest.approx(base["projected_net_pnl_usdt"] + payment)
    assert result["target_mark_price"] > base["target_mark_price"] if payment < 0 else (
        result["target_mark_price"] < base["target_mark_price"])
    assert result["funding_pnl_usdt"] == payment


def test_addition_partial_exit_preserves_full_margin_and_previous_profit_and_costs():
    rows = [fill(1, price=100, origin_signal_time_ms=777, position_id="first"),
            fill(2, price=120, fee=0.054),
            fill(3, "SELL", price=140, fee=0.063, pnl=30)]
    state = account(rows, entry=110)
    result = target.evaluate_profit_target(state, 110, slippage_bps=0)
    assert result["ledger"]["complete"] and result["should_exit"]
    assert result["initial_margin_usdt"] == 22  # 100/10 + 120/10, unchanged by reduction.
    assert result["position_qty"] == 1
    assert result["entry_price"] == 110
    assert result["realized_pnl_usdt"] == 30
    assert result["fees_paid_usdt"] == pytest.approx(0.162)
    assert result["projected_net_pnl_usdt"] == pytest.approx(30 - 0.162 - 0.0495)
    assert result["origin_signal_time_ms"] == 777 and result["position_id"] == "first"
    assert result["entry_sequence"] == 1 and result["last_fill_sequence"] == 3
    assert result["source_signal_times_ms"] == [1000, 2000]
    assert result["last_entry_signal_time_ms"] == 2000


def test_reversal_splits_fee_into_closed_and_new_episodes_and_funding_before_fill():
    rows = [fill(1, quantity=2, fee=0.09),
            fill(2, "SELL", quantity=3, price=110, fee=0.1485, pnl=20,
                 origin_signal_time_ms=2000, position_id="new-short")]
    settled = funding(ms(rows[1]["time_utc"]), 2, -0.3, mark=110)
    state = account(rows, [settled], entry=110)
    episodes = target.reconstruct_episodes(state, rows, margin_leverage=10)
    assert len(episodes) == 2
    assert episodes[0]["closed"] and episodes[0]["initial_margin_usdt"] == 20
    assert episodes[0]["fees_paid_usdt"] == pytest.approx(0.09 + 0.099)
    assert episodes[0]["funding_pnl_usdt"] == -0.3
    assert episodes[1]["fees_paid_usdt"] == pytest.approx(0.0495)
    assert episodes[1]["funding_pnl_usdt"] == 0
    result = target.evaluate_profit_target(state, 103, slippage_bps=0)
    assert result["side"] == "short" and result["entry_sequence"] == 2
    assert result["initial_margin_usdt"] == 11
    assert result["origin_signal_time_ms"] == 2000 and result["position_id"] == "new-short"
    assert result["projected_net_pnl_usdt"] == pytest.approx(7 - 0.0495 - 0.04635)
    assert result["should_exit"]


def test_same_timestamp_close_and_reopen_use_sequence_and_keep_episodes_separate():
    rows = [fill(1), fill(2, "SELL", price=110, fee=0.0495, pnl=10),
            fill(3, price=110, fee=0.0495, time_utc="2026-10-01T00:02:00+00:00")]
    result = target.evaluate_profit_target(account(rows, entry=110), 111, slippage_bps=0)
    assert result["entry_sequence"] == 3
    assert result["realized_pnl_usdt"] == 0
    assert result["initial_margin_usdt"] == 11
    assert not result["should_exit"]  # Previous closed profit cannot fund this target.


def test_full_epoch_journal_recovers_truncated_tail_and_ignores_foreign_and_future_rows(tmp_path):
    rows = [fill(1, quantity=1), fill(2, quantity=1)]
    state = account(rows)
    state["fills"] = rows[-1:]
    assert target.evaluate_profit_target(state, 107)["status"] == "ledger_unavailable"
    journal = tmp_path / "account.fills.jsonl"
    records = [{**row, "account_epoch": EPOCH} for row in rows]
    records.extend([{**rows[0], "fee": 9, "account_epoch": "old-epoch"},
                    {**fill(3), "account_epoch": EPOCH}])
    journal.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    result = target.evaluate_profit_target(state, 107, journal_path=journal)
    assert result["should_exit"] and result["ledger"]["complete"]
    assert result["ledger"]["verified_rows"] == 2
    assert result["ledger"]["other_or_unidentified_epoch_rows_ignored"] == 1
    assert result["initial_margin_usdt"] == 20


def test_conflicting_sequence_and_malformed_journal_disable_only_target(tmp_path):
    row = fill(1)
    state = account([row])
    conflicting = {**row, "fee": 1}
    result = target.evaluate_profit_target(state, 1000, fills=[conflicting])
    assert result["status"] == "ledger_unavailable" and not result["should_exit"]
    assert "conflicting_fill_sequence" in result["ledger"]["issues"]
    journal = tmp_path / "account.fills.jsonl"
    journal.write_text('{"partial":', encoding="utf-8")
    result = target.evaluate_profit_target(state, 1000, journal_path=journal)
    assert not result["should_exit"]
    assert "malformed_journal_record" in result["ledger"]["issues"]


@pytest.mark.parametrize("key, bad", [
    ("position_qty", 2), ("entry_price", 101), ("wallet_balance", 1000),
    ("fees_paid", 0), ("realized_pnl", 1), ("funding_pnl", 1),
    ("position_history", []), ("mode", "live"),
])
def test_inconsistent_account_totals_never_trigger(key, bad):
    state = account([fill(1)])
    state[key] = bad
    result = target.evaluate_profit_target(state, 1000)
    assert result["status"] == "ledger_unavailable" and not result["should_exit"]


def test_fill_claimed_pnl_and_funding_inventory_require_actual_price_path():
    rows = [fill(1), fill(2, "SELL", price=110, pnl=100)]
    state = account(rows)
    result = target.evaluate_profit_target(state, 1000)
    assert result["status"] == "ledger_unavailable"
    row = fill(1)
    state = account([row], [funding(ms(row["time_utc"]) + 1000, 2, -1)])
    result = target.evaluate_profit_target(state, 1000)
    assert result["status"] == "ledger_unavailable" and not result["should_exit"]


@pytest.mark.parametrize("kwargs", [
    {"mark_price": True}, {"mark_price": 0}, {"fee_rate": -1},
    {"fee_rate": float("nan")}, {"slippage_bps": 10_000},
    {"slippage_bps": True}, {"target_roi": 0.5},
])
def test_bad_market_cost_inputs_are_rejected(kwargs):
    values = {"mark_price": 100, **kwargs}
    with pytest.raises(ValueError):
        target.evaluate_profit_target(account([fill(1)]), **values)
