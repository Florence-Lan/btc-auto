"""Auditable net-margin profit targets for the local BTC simulation account.

An episode starts when flat inventory opens, and ends when it becomes flat or
reverses. Its margin basis includes every same-direction entry at its actual
execution price; a reduction does not reset the basis or discard prior costs.
This module reads accounting evidence and proposes an exit. The caller retains
control of funding freshness, risk exits, execution, and entry-signal locks.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


POLICY_MODE = "net_margin_target_v1"
QUANTITY_EPSILON = 1e-12


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Invalid {name}")
    return result


def _time_ms(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError("Invalid ledger timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Ledger timestamp requires a timezone")
        return round(parsed.timestamp() * 1000)
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        raise ValueError("Invalid ledger timestamp") from exc


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def validate_policy(value: Any) -> dict[str, Any] | None:
    """Normalize the simulation-only policy; None explicitly disables it."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("mode") != POLICY_MODE:
        raise ValueError("BTC profit-exit policy mode must be net_margin_target_v1")
    roi = _number(value.get("target_margin_return"), "target margin return")
    leverage = _number(value.get("margin_leverage"), "margin leverage")
    if roi < 0.60:
        raise ValueError("BTC target margin return must be at least 0.60")
    if not 1 <= leverage <= 10:
        raise ValueError("BTC margin leverage must be between 1 and 10")
    return {"mode": POLICY_MODE, "target_margin_return": roi,
            "margin_leverage": leverage}


def complete_fills(
    state: Mapping[str, Any], *, journal_path: Path | str | None = None,
    fills: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge the account tail and epoch-matched journal without guessing gaps.

Records beyond the captured account counter belong to a later snapshot and are
ignored. A supplied ``fills`` list is additional complete-history evidence; its
contents still undergo sequence, duplicate, and account-epoch checks.
"""
    issues: list[str] = []
    ignored = 0
    rows: dict[int, dict[str, Any]] = {}
    try:
        epoch = state["created_at_utc"]
        epoch_ms = _time_ms(epoch)
        expected = _integer(state["fill_count_total"], "fill count")
        records = list(state.get("fills", []))
        if fills is not None:
            records.extend(fills)
        if journal_path is not None:
            path = Path(journal_path)
            if path.exists():
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("Journal record is not an object")
                        if row.get("account_epoch") != epoch:
                            ignored += 1
                            continue
                        records.append(row)
                    except (ValueError, TypeError):
                        issues.append("malformed_journal_record")
        for raw in records:
            if not isinstance(raw, Mapping):
                issues.append("invalid_fill_record")
                continue
            if "account_epoch" in raw and raw["account_epoch"] != epoch:
                ignored += 1
                continue
            row = {key: value for key, value in raw.items() if key != "account_epoch"}
            try:
                sequence = _integer(row.get("sequence"), "fill sequence", minimum=1)
            except ValueError:
                issues.append("fill_sequence_missing_or_invalid")
                continue
            if sequence > expected:
                continue
            if sequence in rows and rows[sequence] != row:
                issues.append("conflicting_fill_sequence")
            rows[sequence] = row
        if sorted(rows) != list(range(1, expected + 1)):
            issues.append("full_account_epoch_fill_history_missing")
        ordered = [rows[key] for key in sorted(rows)]
        timestamps = [_time_ms(row["time_utc"]) for row in ordered]
        if any(timestamp < epoch_ms for timestamp in timestamps):
            issues.append("fill_before_account_epoch")
        if any(left > right for left, right in zip(timestamps, timestamps[1:])):
            issues.append("fill_time_not_monotonic")
    except (KeyError, TypeError, ValueError, OSError, UnicodeError) as exc:
        expected = state.get("fill_count_total")
        ordered = [rows[key] for key in sorted(rows)]
        issues.append(f"invalid_account_epoch_ledger: {exc}")
    return ordered, {
        "complete": not issues, "issues": sorted(set(issues)),
        "account_counter": expected, "verified_rows": len(ordered),
        "other_or_unidentified_epoch_rows_ignored": ignored,
    }


def _matches(actual: float, expected: float, name: str, *, quantity: bool = False) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9,
                        abs_tol=1e-10 if quantity else 1e-6):
        raise ValueError(f"Ledger reconciliation mismatch: {name}")


def reconstruct_episodes(
    state: Mapping[str, Any], fills: list[dict[str, Any]], *, margin_leverage: float,
) -> list[dict[str, Any]]:
    """Rebuild average-cost inventory, proportional reversal fees, and funding.

Settlement precedes all orders at the same millisecond, matching the simulator.
Both the stored totals and each fill's claimed realization are checked against
the execution path, including the entry basis used for the remaining position.
"""
    leverage = _number(margin_leverage, "margin leverage")
    if not 1 <= leverage <= 10:
        raise ValueError("BTC margin leverage must be between 1 and 10")
    epoch = state["created_at_utc"]
    epoch_ms = _time_ms(epoch)
    events = [(_time_ms(row["time_utc"]), 1, row["sequence"], row) for row in fills]
    for index, row in enumerate(state.get("funding_settlements", [])):
        events.append((_integer(row["time_ms"], "funding timestamp"), 0, index, row))
    qty = Decimal(0)
    entry = fees = realized = funding = 0.0
    active = None
    episodes = []
    settlements: set[int] = set()
    positions = []
    for timestamp, kind, _, row in sorted(events, key=lambda item: item[:3]):
        if timestamp < epoch_ms:
            raise ValueError("Ledger event precedes account epoch")
        if kind == 0:
            if timestamp in settlements:
                raise ValueError("Duplicate funding settlement")
            settlements.add(timestamp)
            payment = _number(row["payment"], "funding payment")
            settled_qty = _number(row["signed_qty"], "funding quantity")
            _matches(float(qty), settled_qty, "funding inventory", quantity=True)
            rate = _number(row["rate"], "funding rate")
            mark = _number(row["mark_price"], "funding mark price")
            if mark <= 0:
                raise ValueError("Funding mark price must be positive")
            _matches(payment, -settled_qty * mark * rate, "funding payment")
            if active is None and abs(payment) > 1e-6:
                raise ValueError("Funding charged without open inventory")
            if active is not None:
                active["funding_pnl_usdt"] += payment
            funding += payment
            continue
        if row.get("mode", "simulation") != "simulation":
            raise ValueError("Profit target accepts only simulation fills")
        if state.get("symbol") and row.get("symbol") and state["symbol"] != row["symbol"]:
            raise ValueError("Fill symbol does not match account")
        if row["side"] not in {"BUY", "SELL"}:
            raise ValueError("Invalid fill side")
        amount = _number(row["quantity"], "fill quantity")
        price = _number(row["price"], "fill price")
        fee = _number(row["fee"], "fill fee")
        gross = _number(row["realized_pnl"], "fill realized PnL")
        if amount <= 0 or price <= 0 or fee < 0:
            raise ValueError("Invalid fill quantity, price, or fee")
        delta = Decimal(str(row["quantity"])) * (1 if row["side"] == "BUY" else -1)
        closing = min(abs(qty), abs(delta)) if qty * delta < 0 else Decimal(0)
        expected_gross = ((price - entry) * float(closing)
                          * (1 if qty > 0 else -1)) if closing else 0.0
        _matches(gross, expected_gross, f"fill {row['sequence']} realized PnL")
        new_qty = qty + delta
        if abs(new_qty) < Decimal(str(QUANTITY_EPSILON)):
            new_qty = Decimal(0)
        fee_close = fee * float(closing / abs(delta))
        entering = abs(delta) - closing
        if active is not None:
            active["realized_pnl_usdt"] += gross
            active["fees_paid_usdt"] += fee if qty * new_qty > 0 else fee_close
        if active is not None and qty * new_qty <= 0:
            active.update(exit_ms=timestamp, closed=True, position_qty=0.0,
                          last_fill_sequence=row["sequence"])
            episodes.append(active)
            active = None
        if entering:
            if active is None:
                active = {
                    "episode_id": f"{epoch}:{row['sequence']}:{'long' if new_qty > 0 else 'short'}",
                    "entry_sequence": row["sequence"], "entry_ms": timestamp,
                    "entry_signal_time_ms": row.get("signal_time_ms"),
                    "last_entry_signal_time_ms": row.get("signal_time_ms"),
                    "origin_signal_time_ms": row.get("origin_signal_time_ms"),
                    "position_id": row.get("position_id"), "source_signal_times_ms": [],
                    "side": "long" if new_qty > 0 else "short", "closed": False,
                    "initial_margin_usdt": 0.0, "entered_quantity": 0.0,
                    "realized_pnl_usdt": 0.0, "fees_paid_usdt": fee - fee_close,
                    "funding_pnl_usdt": 0.0,
                }
            active["initial_margin_usdt"] += float(entering) * price / leverage
            active["entered_quantity"] += float(entering)
            active["last_entry_signal_time_ms"] = row.get("signal_time_ms")
            signal = row.get("signal_time_ms")
            if signal is not None and signal not in active["source_signal_times_ms"]:
                active["source_signal_times_ms"].append(signal)
        if qty == 0 or qty * delta > 0:
            new_entry = (float(abs(qty)) * entry + float(abs(delta)) * price) / float(abs(new_qty))
        else:
            new_entry = entry if qty * new_qty > 0 else (price if new_qty else 0.0)
        qty, entry = new_qty, new_entry
        if active is not None:
            active.update(position_qty=float(qty), entry_price=entry,
                          last_fill_sequence=row["sequence"])
        positions.append((timestamp, float(qty)))
        fees += fee
        realized += gross
    if active is not None:
        episodes.append(active)
    for episode in episodes:
        episode["net_realized_pnl_usdt"] = (episode["realized_pnl_usdt"]
                                             - episode["fees_paid_usdt"]
                                             + episode["funding_pnl_usdt"])
    expected = {
        "position_qty": float(qty), "entry_price": entry,
        "realized_pnl": realized, "fees_paid": fees, "funding_pnl": funding,
        "wallet_balance": _number(state["initial_balance"], "initial balance") + realized - fees + funding,
    }
    for name, actual in expected.items():
        _matches(_number(state[name], name), actual, name, quantity=name == "position_qty")
    history = state.get("position_history", [])
    if len(history) != len(positions):
        raise ValueError("Position history does not cover the complete fill path")
    for row, (timestamp, quantity) in zip(history, positions):
        if _integer(row["time_ms"], "position timestamp") != timestamp:
            raise ValueError("Position history timestamp mismatch")
        _matches(_number(row["signed_qty"], "position-history quantity"), quantity,
                 "position history", quantity=True)
    return episodes


def evaluate_profit_target(
    state: Mapping[str, Any], mark_price: float, *, target_roi: float = 0.60,
    margin_leverage: float = 10.0, fee_rate: float = 0.00045,
    slippage_bps: float = 1.0, journal_path: Path | str | None = None,
    fills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate closing the whole current episode after fee and adverse slippage.

Invalid policy/market inputs raise ValueError. Missing or inconsistent accounting
returns ``ledger_unavailable`` with ``should_exit=False``; risk management must
remain available independently. Target ROI is a ratio, so 0.60 means 60%.
"""
    policy = validate_policy({"mode": POLICY_MODE, "target_margin_return": target_roi,
                              "margin_leverage": margin_leverage})
    mark = _number(mark_price, "mark price")
    fee = _number(fee_rate, "exit fee rate")
    slip = _number(slippage_bps, "exit slippage")
    if mark <= 0 or not 0 <= fee < 1 or not 0 <= slip < 10_000:
        raise ValueError("Invalid mark price, exit fee, or exit slippage")
    result: dict[str, Any] = {
        "mode": POLICY_MODE, "target_margin_return_pct": policy["target_margin_return"] * 100,
        "margin_leverage": policy["margin_leverage"], "should_exit": False,
    }
    rows, integrity = complete_fills(state, journal_path=journal_path, fills=fills)
    result["ledger"] = integrity
    if state.get("mode", "simulation") != "simulation":
        integrity["issues"].append("account_is_not_simulation")
        integrity["complete"] = False
    if not integrity["complete"]:
        return {**result, "status": "ledger_unavailable"}
    try:
        episodes = reconstruct_episodes(state, rows, margin_leverage=policy["margin_leverage"])
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        integrity["complete"] = False
        integrity["issues"].append(str(exc))
        return {**result, "status": "ledger_unavailable"}
    if not episodes or episodes[-1]["closed"]:
        return {**result, "status": "flat"}
    episode = episodes[-1]
    quantity = episode["position_qty"]
    exit_price = mark * (1 - math.copysign(slip / 10_000, quantity))
    projected_exit_fee = abs(quantity) * exit_price * fee
    projected_exit_gross = quantity * (exit_price - episode["entry_price"])
    projected_net = episode["net_realized_pnl_usdt"] + projected_exit_gross - projected_exit_fee
    margin = episode["initial_margin_usdt"]
    target_net = margin * policy["target_margin_return"]
    net_roi = projected_net / margin
    # Solve the same net-PnL equation for the mark that would hit the target.
    target_exit = (target_net - episode["net_realized_pnl_usdt"]
                   + quantity * episode["entry_price"]) / (quantity - abs(quantity) * fee)
    target_mark = target_exit / (1 - math.copysign(slip / 10_000, quantity))
    reached = projected_net >= target_net
    return {
        **result, **episode, "status": "target_reached" if reached else "below_target",
        "should_exit": reached, "mark_price": mark,
        "projected_exit_price": exit_price, "projected_exit_fee_usdt": projected_exit_fee,
        "projected_exit_gross_pnl_usdt": projected_exit_gross,
        "projected_net_pnl_usdt": projected_net, "projected_net_roi_pct": net_roi * 100,
        "target_net_pnl_usdt": target_net,
        "target_mark_price": target_mark if target_mark > 0 else None,
    }
