from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Mapping

from binance_terminal_client import BinanceTerminalClient
from llm_trade_gate import apply_llm_trade_gate
from account_risk import constrain_target, block_increases
from active_strategy import simulation_leverage_cap
import execution_entry_gate
import timeseries_execution
import btc_entry_timing
import btc_profit_target
import btc_pullback_entry


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_STATE_PATH = ROOT / "data/runtime/simulation_account_20260917.json"
LIVE_STATE_PATH = ROOT / "data/runtime/live_execution.json"
SYMBOL = "BTCUSDT"
DEFAULT_SIMULATION_RULES = {
    "step_size": Decimal("0.001"),
    "min_qty": Decimal("0.001"),
    "max_qty": Decimal("120"),
    "min_notional": Decimal("50"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _funding_coverage_available(available, now_ms, fetched_through_ms, next_funding_time_ms):
    """A successful fetch covers its cutoff and a known settlement-free gap."""
    if not available:
        return False
    if fetched_through_ms is None:
        return True  # Historical/direct callers retain their explicit contract.
    if now_ms <= fetched_through_ms:
        return True
    return (next_funding_time_ms is not None
            and fetched_through_ms < next_funding_time_ms
            and now_ms < next_funding_time_ms)


def _funding_timestamp(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Invalid funding coverage timestamp")
    try:
        timestamp = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid funding coverage timestamp") from exc
    if timestamp <= 0 or (isinstance(value, float) and value != timestamp):
        raise ValueError("Invalid funding coverage timestamp")
    return timestamp


def _next_funding_boundary(observation):
    try:
        boundary = _funding_timestamp(observation.get("next_funding_time_ms"))
        observed_at = _funding_timestamp(observation.get("time_ms"))
    except ValueError:
        return None
    return boundary if boundary is not None and observed_at is not None and boundary > observed_at else None


def _funding_boundary_observed(state, events, boundary, through_ms):
    if boundary is None:
        return True
    timestamps = [int(row["time_ms"]) for row in state.get("funding_settlements", [])]
    timestamps.extend(int(row["fundingTime"]) for row in events)
    # Binance records may be a few milliseconds after the advertised boundary.
    # A later, unrelated settlement must not clear an unpublished earlier one.
    return any(boundary <= timestamp <= min(through_ms, boundary + 5_000)
               for timestamp in timestamps)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def target_from_report(
    report: dict[str, Any],
    *,
    now_ms: int | None = None,
    max_age_seconds: float = 900.0,
    leverage_cap: float = 2.0,
) -> dict[str, Any]:
    summary = report.get("summary") or {}
    point = report.get("execution_target") or summary.get("last_equity_point") or {}
    signal_time_ms = int(point.get("time_ms") or 0)
    strategy_equity = float(point.get("equity") or 0)
    signal_price = float(point.get("price") or 0)
    signed_qty = float(point.get("signed_qty") or 0)
    if signal_time_ms <= 0 or strategy_equity <= 0 or signal_price <= 0:
        raise RuntimeError("Strategy report does not contain a valid target position")
    current_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    available_time_ms = int(point.get("available_time_ms") or signal_time_ms)
    if signal_time_ms > current_ms or available_time_ms > current_ms:
        raise RuntimeError("Strategy target is from the future; execution blocked")
    age_seconds = max(0.0, (current_ms - signal_time_ms) / 1000)
    if age_seconds > max_age_seconds:
        raise RuntimeError(
            f"Strategy target is stale ({age_seconds:.0f}s > {max_age_seconds:.0f}s); execution blocked"
        )
    target_leverage = signed_qty * signal_price / strategy_equity
    target_leverage = max(-leverage_cap, min(leverage_cap, target_leverage))
    return {
        "signal_time_ms": signal_time_ms,
        "signal_price": signal_price,
        "target_leverage": target_leverage,
        "age_seconds": age_seconds,
        "position_id": point.get("position_id"),
        "origin_signal_time_ms": point.get("origin_signal_time_ms"),
        "origin_entry_price": point.get("origin_entry_price"),
    }


def quantize_signed_quantity(
    quantity: float,
    rules: Mapping[str, Decimal] | None = None,
) -> float:
    active_rules = rules or DEFAULT_SIMULATION_RULES
    value = Decimal(str(abs(quantity)))
    step = Decimal(str(active_rules["step_size"]))
    quantized = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
    if quantized < Decimal(str(active_rules["min_qty"])):
        return 0.0
    signed = float(quantized)
    return -signed if quantity < 0 else signed


def apply_startup_entry_guard(
    state: dict[str, Any],
    target: dict[str, Any],
    current_qty: float,
    *,
    allow_fresh_signal: bool = False,
    now_ms: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    guarded = dict(target)
    position_id = str(target.get("position_id") or "")
    if not position_id:
        return guarded, None
    desired_leverage = float(target.get("target_leverage") or 0.0)
    if abs(desired_leverage) <= 1e-12:
        state["observed_flat_target"] = True
        state["blocked_target_id"] = None
        return guarded, {"status": "flat", "position_id": position_id}
    if abs(current_qty) > 1e-12:
        state["observed_flat_target"] = True
        state["blocked_target_id"] = None
        return guarded, {"status": "tracking", "position_id": position_id}
    blocked_id = str(state.get("blocked_target_id") or "")
    if blocked_id and blocked_id != position_id:
        state["blocked_target_id"] = None
        state["observed_flat_target"] = True
        return guarded, {"status": "new_signal_allowed", "position_id": position_id}
    first_observation = (
        state.get("last_signal_time_ms") is None
        and state.get("signal_time_ms") is None
        and not bool(state.get("observed_flat_target"))
    )
    if first_observation and allow_fresh_signal:
        origin = target.get("origin_signal_time_ms")
        try:
            inception_ms = int(datetime.fromisoformat(state["created_at_utc"].replace("Z", "+00:00")).timestamp() * 1000)
        except (KeyError, ValueError, TypeError):
            inception_ms = None
        if (isinstance(origin, int) and not isinstance(origin, bool) and inception_ms is not None
                and now_ms is not None and inception_ms <= origin <= now_ms):
            state["observed_flat_target"] = True
            state["blocked_target_id"] = None
            return guarded, {"status": "fresh_signal_allowed", "position_id": position_id,
                             "origin_signal_time_ms": origin}
    if blocked_id == position_id or first_observation:
        state["blocked_target_id"] = position_id
        guarded["target_leverage"] = 0.0
        return guarded, {
            "status": "waiting_for_next_signal",
            "position_id": position_id,
            "origin_signal_time_ms": target.get("origin_signal_time_ms"),
        }
    return guarded, {"status": "allowed", "position_id": position_id}


def profit_reentry_blocked(state, target):
    if state.get("position_qty") or not target.get("target_leverage", target.get("signed_qty")):
        return False
    origin, identifier = target.get("origin_signal_time_ms"), target.get("position_id")
    return any((origin is not None and origin == item.get("origin_signal_time_ms"))
               or (identifier and identifier == item.get("position_id"))
               for item in state.get("profit_target_consumed_signals", []))


class SimulationAccount:
    def __init__(self, path: Path = SIMULATION_STATE_PATH, *, persist: bool = True,
                 cost_multiplier: float = 1.0, allow_llm: bool = True,
                 profit_exit_policy=None) -> None:
        self.path = path
        self.persist = persist
        self._state: dict[str, Any] | None = None
        if not math.isfinite(cost_multiplier) or cost_multiplier <= 0:
            raise ValueError("Execution cost multiplier must be positive")
        self.cost_multiplier = cost_multiplier
        self.allow_llm = allow_llm
        # Invalid optional exit settings must never prevent account hard stops.
        self.profit_exit_policy = profit_exit_policy

    def _profit_policy(self, report=None):
        value = self.profit_exit_policy
        if value is None:
            value = ((report or {}).get("shadow_profile") or {}).get("profit_exit_policy")
        return btc_profit_target.validate_policy(value)

    @staticmethod
    def profit_reentry_blocked(state, target):
        return profit_reentry_blocked(state, target)

    def _profit_decision(self, state, mark_price, now_ms, funding_covered, report=None, target=None):
        try:
            policy = self._profit_policy(report)
        except ValueError as exc:
            decision = {"enabled": True, "should_exit": False, "status": "invalid_policy",
                        "reason": str(exc), "checked_at_ms": now_ms}
            state["profit_exit_status"] = decision
            return decision
        if policy is None:
            return {"enabled": False, "should_exit": False, "status": "disabled"}
        fee = float(os.getenv("SIM_TAKER_FEE", "0.00045") or 0.00045) * self.cost_multiplier
        slip = float(os.getenv("SIM_SLIPPAGE_BPS", "1.0") or 1.0) * self.cost_multiplier
        decision = btc_profit_target.evaluate_profit_target(
            state, mark_price, target_roi=policy["target_margin_return"],
            margin_leverage=policy["margin_leverage"], fee_rate=fee, slippage_bps=slip,
            journal_path=self.path.with_suffix(".fills.jsonl") if self.persist else None)
        decision.update(enabled=True, target_margin_return=policy["target_margin_return"],
                        margin_leverage=policy["margin_leverage"], checked_at_ms=now_ms)
        if state.get("position_qty") and decision.get("entry_sequence") is not None:
            source = state.get("profit_target_origin") or {}
            if source.get("entry_sequence") != decision["entry_sequence"]:
                source = {"entry_sequence": decision["entry_sequence"],
                          "origin_signal_time_ms": decision.get("origin_signal_time_ms"),
                          "position_id": decision.get("position_id")}
            if not source.get("origin_signal_time_ms") or not source.get("position_id"):
                point = target
                if point is None and self.persist:
                    previous = read_json(self.path.with_name("last_report.json"), {}) or {}
                    point = previous.get("execution_target") or {}
                point = point or {}
                requested = point.get("target_leverage", point.get("signed_qty", 0)) or 0
                origin = point.get("origin_signal_time_ms")
                if (float(requested) * float(state["position_qty"]) > 0
                        and isinstance(origin, int) and not isinstance(origin, bool)
                        and 0 < origin <= decision["entry_ms"]):
                    source.update(origin_signal_time_ms=origin, position_id=point.get("position_id"))
                state["profit_target_origin"] = source
            decision["source_origin"] = dict(source)
        if not funding_covered and decision["status"] not in {"flat", "ledger_unavailable"}:
            decision.update(status="funding_unavailable", should_exit=False)
        state["profit_exit_status"] = decision
        return decision

    @staticmethod
    def _record_profit_exit(state, decision, fill):
        fill["reason"] = "net_margin_target"
        source = decision.get("source_origin") or {}
        consumed = {**source, "episode_id": decision.get("episode_id"),
                    "closed_at_utc": fill["time_utc"], "exit_sequence": fill["sequence"]}
        state.setdefault("profit_target_consumed_signals", []).append(consumed)
        state["profit_target_consumed_signals"] = state["profit_target_consumed_signals"][-200:]
        state["last_profit_target_exit"] = {**decision, "fill": dict(fill)}
        state["profit_exit_status"] = {**decision, "status": "closed_target", "should_exit": False,
                                      "exit_sequence": fill["sequence"]}

    @staticmethod
    def _consume_pullback_source(state, pending):
        source = (pending or {}).get("source_key")
        if source:
            consumed = state.setdefault("btc_pullback_consumed_sources", [])
            if source not in consumed:
                consumed.append(source)
            state["btc_pullback_consumed_sources"] = consumed[-200:]

    def _pullback_decision(self, state, target, report, mark_price, now_ms, *, risk_halted=False):
        """Hold only the new direction; reductions and inventory management bypass."""
        shadow = (report or {}).get("shadow_profile") or {}
        raw_policy = shadow.get("btc_pullback_entry_policy")
        requested = float(target.get("target_leverage", 0) or 0)
        current = float(state.get("position_qty", 0) or 0)
        new_direction = requested != 0 and (abs(current) <= 1e-12 or requested * current < 0)
        side = "long" if requested > 0 else "short" if requested < 0 else None
        decision = {"enabled": raw_policy is not None, "allowed": True,
                    "status": "disabled", "side": side, "reasons": [],
                    "checked_at_ms": now_ms}
        pending = state.get("btc_pullback_pending") or {}
        if raw_policy is None:
            state["btc_pullback_entry"] = decision
            return decision
        if not new_direction or risk_halted:
            # Once a direction is actually held, sizing is the original policy.
            # A flat or changed source cancels an unfilled opportunity forever.
            source = f"{target.get('position_id')}:{target.get('origin_signal_time_ms')}:{side}"
            if pending and pending.get("status") not in btc_pullback_entry.TERMINAL_STATUSES | {"filled"}:
                if requested == 0 or pending.get("source_key") != source:
                    pending.update(status="cancelled", cancel_reason="signal_changed_or_flat",
                                   cancelled_at_ms=now_ms, reasons=["btc_pullback_target_flat" if requested == 0
                                                                  else "btc_pullback_source_changed"])
                    self._consume_pullback_source(state, pending)
            decision.update(status="risk_exit" if risk_halted else
                            "existing_direction" if requested else "exit_or_flat")
            state["btc_pullback_entry"] = decision
            return decision
        try:
            policy = btc_pullback_entry.validate_policy(raw_policy)
        except (ValueError, TypeError, OverflowError) as exc:
            decision.update(allowed=False, status="invalid_policy",
                            reasons=["btc_pullback_invalid_policy"], detail=str(exc))
            state["btc_pullback_entry"] = decision
            return decision
        source = f"{target.get('position_id')}:{target.get('origin_signal_time_ms')}:{side}"
        if pending and pending.get("source_key") != source:
            self._consume_pullback_source(state, pending)
            pending = {}
        if (source in state.get("btc_pullback_consumed_sources", [])
                and (not pending or pending.get("status") == "filled")):
            pending = {**pending, "version": 1, "mode": btc_pullback_entry.MODE, "policy": policy,
                       "source_key": source, "status": "cancelled", "side": side,
                       "signal_origin": target.get("origin_signal_time_ms"),
                       "reasons": ["btc_pullback_source_consumed"]}
        market = (report or {}).get("btc_pullback_market_data") or {}
        if not isinstance(market, Mapping):
            market = {}
        try:
            if not pending or pending.get("status") == "data_unavailable":
                pending = btc_pullback_entry.build_pending(
                    target, market.get("hourly", []), policy, now_ms)
            # Expiry is independent of market data. The runner deliberately
            # omits this feed outside the entry window; advance_pending still
            # validates the frozen record before terminating/consuming it.
            expiry = pending.get("expires_at_ms")
            expired = type(expiry) is int and now_ms >= expiry
            if pending.get("status") not in btc_pullback_entry.TERMINAL_STATUSES and not expired:
                cutoff = market.get("asof_ms")
                if (market.get("complete") is not True or isinstance(cutoff, bool)
                        or not isinstance(cutoff, int) or cutoff > now_ms):
                    decision.update(allowed=False, status="data_unavailable",
                                    reasons=["btc_pullback_data_unavailable"])
                    state["btc_pullback_pending"] = pending
                    state["btc_pullback_entry"] = decision
                    return decision
            slip = float(os.getenv("SIM_SLIPPAGE_BPS", "1.0") or 1.0) * self.cost_multiplier
            pending, decision = btc_pullback_entry.advance_pending(
                pending, target, market.get("five_minute", []), mark_price, now_ms, slip, policy)
        except (ValueError, TypeError, OverflowError, KeyError, IndexError) as exc:
            decision.update(allowed=False, status="data_unavailable",
                            reasons=["btc_pullback_data_unavailable"], detail=str(exc))
        state["btc_pullback_pending"] = pending
        if pending.get("status") in btc_pullback_entry.TERMINAL_STATUSES:
            self._consume_pullback_source(state, pending)
        state["btc_pullback_entry"] = decision
        return decision

    def _save(self, state: dict[str, Any]) -> None:
        if self.persist:
            write_json(self.path, state)
        else:
            self._state = state

    def _record_fills(self, state, records):
        if not self.persist or not any(records):
            return
        journal = self.path.with_suffix(".fills.jsonl")
        with journal.open("a", encoding="utf-8") as handle:
            for fill in records:
                if fill:
                    handle.write(json.dumps({**fill, "account_epoch": state["created_at_utc"]},
                                            ensure_ascii=False) + "\n")

    def _new_state(self) -> dict[str, Any]:
        initial = float(os.getenv("SIM_INITIAL_BALANCE_USDT", "100") or 100)
        if not math.isfinite(initial) or initial <= 0:
            raise ValueError("SIM_INITIAL_BALANCE_USDT must be greater than zero")
        now = utc_now()
        return {
            "version": 3,
            "mode": "simulation",
            "symbol": SYMBOL,
            "created_at_utc": now,
            "updated_at_utc": now,
            "initial_balance": initial,
            "wallet_balance": initial,
            "position_qty": 0.0,
            "entry_price": 0.0,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "peak_equity": initial,
            "max_drawdown_pct": 0.0,
            "fills": [],
            "fill_count_total": 0,
            "equity_curve": [],
            "last_signal_time_ms": None,
            "last_mark_price": None,
            "observed_flat_target": False,
            "blocked_target_id": None,
            "entry_guard": None,
            "funding_pnl": 0.0,
            "funding_tracking_start_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "funding_settlements": [],
            "position_history": [],
            "risk_halt_at_utc": None,
        }

    def load(self) -> dict[str, Any]:
        state = read_json(self.path) if self.persist else self._state
        if not isinstance(state, dict):
            return self._new_state()
        if "fills" not in state:
            legacy = list(state.get("trades") or [])
            state["fills"] = legacy
            state["fill_count_total"] = max(
                int(state.get("fill_count_total") or 0),
                len(legacy),
            )
        state.setdefault("observed_flat_target", False)
        state.setdefault("blocked_target_id", None)
        state.setdefault("entry_guard", None)
        if "funding_tracking_start_ms" not in state:
            # Old ledgers lack an auditable inventory path. Never invent past charges.
            tracking_start = int(datetime.now(timezone.utc).timestamp() * 1000)
            state["funding_tracking_start_ms"] = tracking_start
            state["position_history"] = [{"time_ms": tracking_start,
                                          "signed_qty": float(state.get("position_qty", 0))}]
        state.setdefault("funding_pnl", 0.0)
        state.setdefault("funding_settlements", [])
        state.setdefault("position_history", [])
        state["version"] = 3
        return state

    def reset(self, initial_balance: float, *, now_ms: int | None = None) -> dict[str, Any]:
        amount = float(initial_balance)
        if not math.isfinite(amount) or amount < 1 or amount > 1_000_000_000:
            raise ValueError("模拟盘初始金额必须在 1 至 1,000,000,000 USDT 之间")
        state = self._new_state()
        state["initial_balance"] = amount
        state["wallet_balance"] = amount
        state["peak_equity"] = amount
        state["updated_at_utc"] = utc_now()
        if now_ms is not None:
            state["created_at_utc"] = datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat()
            state["funding_tracking_start_ms"] = now_ms
        self._save(state)
        return state

    @staticmethod
    def settle_funding(state: dict[str, Any], events, now_ms: int) -> None:
        if not events:
            return
        settled = {int(row["time_ms"]) for row in state["funding_settlements"]}
        history = state["position_history"]
        for event in sorted(events, key=lambda row: int(row["fundingTime"])):
            timestamp = int(event["fundingTime"])
            if timestamp in settled or timestamp < state["funding_tracking_start_ms"] or timestamp > now_ms:
                continue
            rate, mark = float(event["fundingRate"]), float(event["markPrice"])
            if not math.isfinite(rate) or not math.isfinite(mark) or mark <= 0:
                raise ValueError("Invalid funding settlement rate or mark price")
            # Settlement precedes an order recorded at exactly the same timestamp.
            qty = next((float(row["signed_qty"]) for row in reversed(history)
                        if int(row["time_ms"]) < timestamp), 0.0)
            payment = -qty * mark * rate
            state["wallet_balance"] += payment
            state["funding_pnl"] += payment
            state["funding_settlements"].append({"time_ms": timestamp, "signed_qty": qty,
                                                "mark_price": mark, "rate": rate, "payment": payment})
            settled.add(timestamp)

    def _fill_to(self, state, target_qty, mark_price, now_ms, signal_time_ms, rules):
        current_qty = float(state.get("position_qty", 0))
        delta = float(Decimal(str(target_qty)) - Decimal(str(current_qty)))
        full_close = target_qty == 0 and abs(current_qty) > 1e-12
        reduction = current_qty * delta < 0 and abs(delta) <= abs(current_qty) + 1e-12
        if not (full_close or (abs(delta) >= float(rules["min_qty"]) and
                              (reduction or abs(delta * mark_price) >= float(rules["min_notional"])))):
            return None
        slippage = float(os.getenv("SIM_SLIPPAGE_BPS", "1.0") or 1.0) * self.cost_multiplier
        fee_rate = float(os.getenv("SIM_TAKER_FEE", "0.00045") or 0.00045) * self.cost_multiplier
        if not math.isfinite(slippage) or not 0 <= slippage < 10000 or not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError("Invalid execution fee or slippage setting")
        fill_price = mark_price * (1 + math.copysign(slippage / 10_000, delta))
        entry = float(state.get("entry_price", 0))
        realized = 0.0
        new_qty = target_qty
        if current_qty == 0 or current_qty * delta > 0:
            new_entry = ((abs(current_qty) * entry + abs(delta) * fill_price) / abs(new_qty))
        else:
            realized = (fill_price - entry) * min(abs(current_qty), abs(delta)) * math.copysign(1, current_qty)
            new_entry = entry if current_qty * new_qty > 0 else fill_price
        if abs(new_qty) < 1e-12:
            new_qty, new_entry = 0.0, 0.0
        fee = abs(delta * fill_price) * fee_rate
        state["wallet_balance"] += realized - fee
        state["position_qty"], state["entry_price"] = new_qty, new_entry
        state["realized_pnl"] += realized
        state["fees_paid"] += fee
        state["position_history"].append({"time_ms": now_ms, "signed_qty": new_qty})
        fill = {"time_utc": datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat(),
                "signal_time_ms": signal_time_ms, "symbol": SYMBOL,
                "side": "BUY" if delta > 0 else "SELL", "quantity": abs(delta),
                "price": fill_price, "realized_pnl": realized, "fee": fee, "mode": "simulation"}
        state.setdefault("fills", []).append(fill)
        state["fills"] = state["fills"][-200:]
        state["fill_count_total"] = int(state.get("fill_count_total") or 0) + 1
        fill["sequence"] = state["fill_count_total"]
        return fill

    @staticmethod
    def _mark(state: dict[str, Any], mark_price: float) -> dict[str, float]:
        qty = float(state.get("position_qty", 0))
        entry = float(state.get("entry_price", 0))
        unrealized = (mark_price - entry) * qty if qty else 0.0
        wallet = float(state["wallet_balance"])
        equity = wallet + unrealized
        return {"unrealized_pnl": unrealized, "equity": equity}

    def snapshot(self, mark_price: float | None = None) -> dict[str, Any]:
        state = self.load()
        price = float(mark_price or state.get("last_mark_price") or 0)
        marked = self._mark(state, price) if price > 0 else {
            "unrealized_pnl": 0.0,
            "equity": float(state["wallet_balance"]),
        }
        qty = float(state.get("position_qty", 0))
        stored_cap = float(state.get("simulation_leverage_cap", 2.0))
        # A zero execution bound disables exposure; it is not a zero strategy
        # limit. Still validate the current environment when reporting margin.
        leverage_cap = simulation_leverage_cap(strategy_cap=stored_cap if stored_cap != 0 else 2.0)
        if stored_cap == 0:
            leverage_cap = 0.0
        reserved_margin = abs(qty * price) / leverage_cap if leverage_cap > 0 else (
            marked["equity"] if abs(qty) > 1e-12 else 0.0)
        positions = []
        if abs(qty) > 1e-12:
            positions.append({
                "symbol": SYMBOL,
                "side": "LONG" if qty > 0 else "SHORT",
                "quantity": abs(qty),
                "signed_quantity": qty,
                "entry_price": float(state.get("entry_price", 0)),
                "mark_price": price,
                "unrealized_pnl": marked["unrealized_pnl"],
                "leverage": abs(qty * price) / max(marked["equity"], 1e-12),
                "source": "Local simulation · Binance mainnet price",
            })
        initial = float(state["initial_balance"])
        return {
            "state": state,
            "account": {
                "wallet_balance": float(state["wallet_balance"]),
                "available_balance": max(marked["equity"] - reserved_margin, 0.0),
                "simulation_leverage_cap": leverage_cap,
                "unrealized_pnl": marked["unrealized_pnl"],
                "margin_balance": marked["equity"],
                "margin_ratio_pct": 0.0,
                "realized_return_pct": (marked["equity"] / initial - 1) * 100,
                "source": "simulation",
            },
            "positions": positions,
            "open_orders": [],
            "recent_trades": list(reversed(state.get("fills", [])[-20:])),
            "equity_curve": state.get("equity_curve", []),
            "max_drawdown_pct": float(state.get("max_drawdown_pct", 0)),
        }

    def reconcile(
        self,
        target: dict[str, Any],
        mark_price: float,
        rules: Mapping[str, Decimal] | None = None,
        report: Mapping[str, Any] | None = None,
        *, funding_events=(), funding_available: bool = True, now_ms: int | None = None,
        execution_clock=None, mark_time_ms=None,
        funding_fetch_through_ms=None, next_funding_time_ms=None,
        funding_pending_settlement_ms=None,
    ) -> dict[str, Any]:
        state = self.load()
        now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        initial_clock_ms, initial_monotonic = now_ms, time.monotonic()
        funding_fetch_through_ms = _funding_timestamp(funding_fetch_through_ms)
        next_funding_time_ms = _funding_timestamp(next_funding_time_ms)
        pending_boundaries = [_funding_timestamp(state.get("funding_pending_settlement_ms")),
                              _funding_timestamp(funding_pending_settlement_ms)]
        pending_funding = min((value for value in pending_boundaries if value is not None), default=None)

        def funding_covered_at(current_ms):
            nonlocal pending_funding
            if next_funding_time_ms is not None and current_ms >= next_funding_time_ms:
                pending_funding = min(pending_funding or next_funding_time_ms, next_funding_time_ms)
            event_cutoff = min(current_ms, funding_fetch_through_ms) if funding_fetch_through_ms is not None else current_ms
            if _funding_boundary_observed(state, funding_events, pending_funding, event_cutoff):
                pending_funding = None
            if pending_funding is not None or "funding_pending_settlement_ms" in state:
                state["funding_pending_settlement_ms"] = pending_funding
            return (pending_funding is None and _funding_coverage_available(
                funding_available, current_ms, funding_fetch_through_ms, next_funding_time_ms))
        if not math.isfinite(mark_price) or mark_price <= 0 or not math.isfinite(float(target["target_leverage"])):
            raise ValueError("Execution requires a positive price and finite target")
        if mark_time_ms is not None:
            from decision_runtime import validate_mark
            validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
        if state["position_history"] and now_ms < state["position_history"][-1]["time_ms"]:
            raise ValueError("Execution clock cannot move backwards")
        self.settle_funding(state, funding_events, now_ms)
        if funding_fetch_through_ms is not None:
            state["funding_last_fetch_ms"] = funding_fetch_through_ms
        elif funding_available:
            state["funding_last_fetch_ms"] = now_ms
        marked = self._mark(state, mark_price)
        leverage_cap = simulation_leverage_cap(report)
        configured_cap = float(os.getenv("SIM_MAX_NOTIONAL_USDT", "0") or 0)
        if not math.isfinite(leverage_cap) or leverage_cap < 0 or not math.isfinite(configured_cap) or configured_cap < 0:
            raise ValueError("Invalid simulation leverage or notional cap")
        state["simulation_leverage_cap"] = leverage_cap
        equity = max(marked["equity"], 0.0)
        natural_cap = equity * leverage_cap
        max_notional = min(natural_cap, configured_cap) if configured_cap > 0 else natural_cap
        current_qty = float(state.get("position_qty", 0))
        effective_target, entry_guard = apply_startup_entry_guard(
            state,
            target,
            current_qty,
            allow_fresh_signal=(report or {}).get("execution_model") == timeseries_execution.STARTUP_MODEL,
            now_ms=now_ms,
        )
        state["entry_guard"] = entry_guard
        effective_target, risk = constrain_target(state, effective_target, equity, current_qty,
                                                  mark_price, now_ms, report)
        timing = btc_entry_timing.decision_at(report, target, current_qty=current_qty, now_ms=now_ms)
        if not timing["allowed"]:
            effective_target["target_leverage"] = 0.0
        state["btc_entry_timing"] = timing
        pullback = self._pullback_decision(state, target, report, mark_price, now_ms,
                                          risk_halted=risk["status"] == "halted")
        if not pullback["allowed"]:
            effective_target = block_increases(effective_target, current_qty, equity, mark_price)
        funding_covered = funding_covered_at(now_ms)
        profit_decision = self._profit_decision(state, mark_price, now_ms, funding_covered,
                                                report, target)
        if self.profit_reentry_blocked(state, target):
            effective_target["target_leverage"] = 0.0
            state["entry_guard"] = entry_guard = {
                "status": "profit_target_signal_consumed", "position_id": target.get("position_id")}
        if profit_decision.get("status") in {"ledger_unavailable", "invalid_policy"}:
            effective_target = block_increases(effective_target, current_qty, equity, mark_price)
        if not funding_covered:
            effective_target = block_increases(effective_target, current_qty, equity, mark_price)
        state["funding_status"] = "ok" if funding_covered else "unavailable_new_risk_blocked"
        if not pullback["allowed"]:
            llm_trade_gate = {"enabled": False, "status": "waiting_for_pullback"}
        elif profit_decision.get("should_exit"):
            llm_trade_gate = {"enabled": False, "status": "profit_target_exit"}
        elif self.allow_llm:
            effective_target, llm_trade_gate = apply_llm_trade_gate(
                report or {}, effective_target, current_qty=current_qty, equity=equity,
                mark_price=mark_price, mode="simulation", previous_decision=state.get("llm_trade_gate"))
        else:
            llm_trade_gate = {"enabled": False, "status": "disabled_historical_replay"}
        state["llm_trade_gate"] = llm_trade_gate
        # The LLM/provider may have crossed a calendar boundary. Production
        # injects the exchange clock; historical replays keep their explicit clock.
        clock_failed = False
        if execution_clock is not None:
            try:
                resolved_ms = int(execution_clock())
                elapsed_ms = max(0, int(round((time.monotonic() - initial_monotonic) * 1000)))
                now_ms = max(initial_clock_ms + elapsed_ms, resolved_ms)
            except (RuntimeError, OSError, ValueError, TypeError):
                # Losing the fresh exchange clock must not become an exit blocker.
                now_ms = max(now_ms, int(datetime.now(timezone.utc).timestamp() * 1000))
                clock_failed = True
        if mark_time_ms is not None or execution_clock is not None:
            elapsed_ms = max(0, int(round((time.monotonic() - initial_monotonic) * 1000)))
            now_ms = max(now_ms, initial_clock_ms + elapsed_ms)
        clock_checked_monotonic = time.monotonic()
        if mark_time_ms is not None:
            # Funding/rule lookups or the LLM may have delayed this reconciliation.
            # Even a reduction cannot create a simulated fill at an expired price.
            from decision_runtime import validate_mark
            validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
        entry_gate = execution_entry_gate.decision_at(
            report, now_ms, "long" if effective_target["target_leverage"] >= 0 else "short")
        if clock_failed:
            entry_gate.update(allowed=False, status="blocked")
            entry_gate["reasons"].append("execution_clock_unavailable")
        state["execution_entry_gate"] = entry_gate
        if mark_time_ms is not None:
            # Current source health can require large snapshot reads. Include
            # that elapsed time in the last check immediately before a fill.
            now_ms += max(0, int(round((time.monotonic() - clock_checked_monotonic) * 1000)))
            validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
        if state["position_history"] and now_ms < int(state["position_history"][-1]["time_ms"]):
            raise ValueError("Execution clock cannot move backwards")
        funding_covered = funding_covered_at(now_ms)
        if not funding_covered:
            effective_target = block_increases(effective_target, current_qty, equity, mark_price)
        state["funding_status"] = "ok" if funding_covered else "unavailable_new_risk_blocked"
        target_cap = max_notional / max(equity, 1e-12)
        effective_target["target_leverage"] = max(-target_cap, min(
            target_cap, float(effective_target["target_leverage"])))
        target_notional = float(effective_target["target_leverage"]) * equity
        target_notional = max(-max_notional, min(max_notional, target_notional))
        target_qty = target_notional / mark_price if mark_price > 0 else 0.0
        target_qty = execution_entry_gate.constrain_quantity(target_qty, current_qty, entry_gate)
        target_qty = quantize_signed_quantity(target_qty, rules)
        if not entry_gate["allowed"]:
            effective_target["target_leverage"] = target_qty * mark_price / max(equity, 1e-12)
        active_rules = rules or DEFAULT_SIMULATION_RULES
        # Source age can expire during an LLM review or current-source checks.
        # Recheck the original direction: a cached approval or earlier clamp
        # must never turn a refreshed 5m target into a new hourly opportunity.
        if timing["enabled"]:
            now_ms = max(now_ms, initial_clock_ms + max(0, int(round(
                (time.monotonic() - initial_monotonic) * 1000))))
            if mark_time_ms is not None:
                validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
        timing = btc_entry_timing.decision_at(report, target, current_qty=current_qty, now_ms=now_ms)
        if not timing["allowed"]:
            effective_target["target_leverage"] = 0.0
            target_qty = 0.0
        state["btc_entry_timing"] = timing
        profit_decision = self._profit_decision(state, mark_price, now_ms, funding_covered,
                                                report, target)
        # Full-journal validation is I/O. Include it in the final execution cutoff.
        if mark_time_ms is not None or execution_clock is not None:
            now_ms = max(now_ms, initial_clock_ms + max(0, int(round(
                (time.monotonic() - initial_monotonic) * 1000))))
            if mark_time_ms is not None:
                validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
            funding_covered = funding_covered_at(now_ms)
            state["funding_status"] = "ok" if funding_covered else "unavailable_new_risk_blocked"
            if not funding_covered and profit_decision.get("enabled"):
                profit_decision.update(status="funding_unavailable", should_exit=False)
                target_qty = execution_entry_gate.constrain_quantity(target_qty, current_qty, {"allowed": False})
            timing = btc_entry_timing.decision_at(report, target, current_qty=current_qty, now_ms=now_ms)
            state["btc_entry_timing"] = timing
            if not timing["allowed"]:
                target_qty = 0.0
                effective_target["target_leverage"] = 0.0
        profit_close = profit_decision.get("should_exit") and risk["status"] != "halted"
        if profit_close or self.profit_reentry_blocked(state, target):
            effective_target["target_leverage"] = 0.0
            target_qty = 0.0
        # Recheck the frozen entry plan after every potentially slow review/read.
        pullback = self._pullback_decision(state, target, report, mark_price, now_ms,
                                          risk_halted=risk["status"] == "halted")
        if (pullback.get("enabled") and pullback.get("status") not in
                {"existing_direction", "exit_or_flat", "risk_exit"}
                and (mark_time_ms is not None or execution_clock is not None)):
            now_ms = max(now_ms, initial_clock_ms + max(0, int(round(
                (time.monotonic() - initial_monotonic) * 1000))))
            if mark_time_ms is not None:
                validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
            funding_covered = funding_covered_at(now_ms)
            if not funding_covered:
                target_qty = execution_entry_gate.constrain_quantity(target_qty, current_qty, {"allowed": False})
            state["funding_status"] = "ok" if funding_covered else "unavailable_new_risk_blocked"
            pullback = self._pullback_decision(state, target, report, mark_price, now_ms,
                                              risk_halted=risk["status"] == "halted")
            # The final candle validation itself is work, so include its latency.
            now_ms = max(now_ms, initial_clock_ms + max(0, int(round(
                (time.monotonic() - initial_monotonic) * 1000))))
            if mark_time_ms is not None:
                validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
            funding_covered = funding_covered_at(now_ms)
            if not funding_covered:
                target_qty = execution_entry_gate.constrain_quantity(target_qty, current_qty, {"allowed": False})
            state["funding_status"] = "ok" if funding_covered else "unavailable_new_risk_blocked"
            pending = state.get("btc_pullback_pending") or {}
            expiry = pending.get("expires_at_ms")
            if (pullback.get("enabled") and pullback.get("allowed") and isinstance(expiry, int)
                    and now_ms >= expiry):
                pending.update(status="expired", reasons=["btc_pullback_origin_expired"])
                self._consume_pullback_source(state, pending)
                pullback.update(allowed=False, status="expired", reasons=pending["reasons"],
                                checked_at_ms=now_ms)
                state["btc_pullback_entry"] = pullback
            if (pullback.get("enabled") and pullback.get("allowed")
                    and pending.get("status") == "ready"
                    and pending.get("last_processed_open_ms") !=
                    ((now_ms + 1) // btc_pullback_entry.FIVE_MINUTES - 1) * btc_pullback_entry.FIVE_MINUTES):
                pullback.update(allowed=False, status="data_unavailable",
                                reasons=["btc_pullback_5m_gap"], checked_at_ms=now_ms)
                state["btc_pullback_entry"] = pullback
        if not pullback["allowed"]:
            target_qty = execution_entry_gate.constrain_quantity(target_qty, current_qty, {"allowed": False})
            effective_target["target_leverage"] = target_qty * mark_price / max(equity, 1e-12)
        fill = self._fill_to(state, target_qty, mark_price, now_ms, int(target["signal_time_ms"]), active_rules)
        if fill:
            fill.update(position_id=target.get("position_id"),
                        origin_signal_time_ms=target.get("origin_signal_time_ms"))
            if profit_close:
                self._record_profit_exit(state, profit_decision, fill)
            elif risk["status"] == "halted":
                fill["reason"] = "account_hard_stop"
            if (pullback.get("enabled") and pullback.get("allowed")
                    and current_qty * float(state.get("position_qty", 0)) <= 0
                    and abs(float(state.get("position_qty", 0))) > 1e-12):
                pending = state.get("btc_pullback_pending") or {}
                pending.update(status="filled", filled_at_ms=now_ms, fill_sequence=fill["sequence"])
                self._consume_pullback_source(state, pending)
                fill["btc_pullback_entry"] = dict(pullback)
                state["btc_pullback_entry"] = {**pullback, "status": "filled", "allowed": False,
                                                "fill_sequence": fill["sequence"]}
        state["last_signal_time_ms"] = int(target["signal_time_ms"])
        state["last_mark_price"] = mark_price
        after = self._mark(state, mark_price)
        _, risk = constrain_target(state, effective_target, after["equity"], state["position_qty"],
                                   mark_price, now_ms, report)
        risk_fill = None
        if risk["status"] == "halted" and state["position_qty"]:
            risk_fill = self._fill_to(state, 0.0, mark_price, now_ms, int(target["signal_time_ms"]), active_rules)
            after = self._mark(state, mark_price)
            _, risk = constrain_target(state, effective_target, after["equity"], 0.0, mark_price, now_ms)
            target_qty = 0.0
            effective_target["target_leverage"] = 0.0
        drawdown = risk["drawdown_pct"]
        state.setdefault("equity_curve", []).append({
            "time_ms": now_ms,
            "equity": after["equity"],
            "drawdown_pct": drawdown,
            "signed_qty": float(state.get("position_qty", 0)),
            "price": mark_price,
        })
        state["equity_curve"] = state["equity_curve"][-1000:]
        state["updated_at_utc"] = utc_now()
        self._save(state)
        self._record_fills(state, [fill, risk_fill])
        return {
            "mode": "simulation",
            "target_leverage": effective_target["target_leverage"],
            "desired_target_leverage": target["target_leverage"],
            "simulation_leverage_cap": leverage_cap,
            "target_qty": target_qty,
            "fill": fill,
            "risk_fill": risk_fill,
            "account_risk": risk,
            "funding_pnl": state["funding_pnl"],
            "funding_fetch_through_ms": state.get("funding_last_fetch_ms"),
            "funding_pending_settlement_ms": pending_funding,
            "entry_guard": entry_guard,
            "btc_entry_timing": timing,
            "btc_pullback_entry": state.get("btc_pullback_entry"),
            "llm_trade_gate": llm_trade_gate,
            "execution_entry_gate": entry_gate,
            "profit_exit_status": state.get("profit_exit_status"),
            "account": self.snapshot(mark_price)["account"],
        }

    def observe(self, mark_price, now_ms, funding_events=(), funding_available=True, rules=None, *,
                funding_fetch_through_ms=None, next_funding_time_ms=None,
                funding_pending_settlement_ms=None, mark_time_ms=None):
        """Monitor equity and hard stops without repeatedly applying soft reductions."""
        state = self.load()
        started_monotonic = time.monotonic()
        initial_ms = now_ms
        self.settle_funding(state, funding_events, now_ms)
        if funding_available:
            state["funding_last_fetch_ms"] = funding_fetch_through_ms or now_ms
        pending_values = [_funding_timestamp(state.get("funding_pending_settlement_ms")),
                          _funding_timestamp(funding_pending_settlement_ms)]
        pending_funding = min((value for value in pending_values if value is not None), default=None)
        if pending_funding is not None:
            state["funding_pending_settlement_ms"] = pending_funding
        if pending_funding is not None and _funding_boundary_observed(
                state, funding_events, pending_funding, now_ms):
            state["funding_pending_settlement_ms"] = None
            pending_funding = None
        next_funding_time_ms = _funding_timestamp(next_funding_time_ms)
        if next_funding_time_ms is not None:
            state["funding_next_settlement_ms"] = next_funding_time_ms
            if now_ms >= next_funding_time_ms and not _funding_boundary_observed(
                    state, funding_events, next_funding_time_ms, now_ms):
                pending_funding = min(pending_funding or next_funding_time_ms, next_funding_time_ms)
                state["funding_pending_settlement_ms"] = pending_funding
        funding_covered = pending_funding is None and _funding_coverage_available(
            funding_available, now_ms, _funding_timestamp(funding_fetch_through_ms), next_funding_time_ms)
        equity = self._mark(state, mark_price)["equity"]
        target = {"target_leverage": 0.0, "signal_time_ms": now_ms}
        _, risk = constrain_target(state, target, equity, state["position_qty"], mark_price, now_ms)
        fill = None
        if risk["status"] == "halted" and state["position_qty"]:
            fill = self._fill_to(state, 0.0, mark_price, now_ms, now_ms, rules or DEFAULT_SIMULATION_RULES)
            if fill:
                fill["reason"] = "account_hard_stop"
            equity = self._mark(state, mark_price)["equity"]
            _, risk = constrain_target(state, target, equity, 0.0, mark_price, now_ms)
        decision = self._profit_decision(state, mark_price, now_ms, funding_covered)
        if mark_time_ms is not None and fill is None:
            now_ms = initial_ms + max(0, int(round((time.monotonic()-started_monotonic)*1000)))
            from decision_runtime import validate_mark
            validate_mark({"price": mark_price, "time_ms": mark_time_ms}, now_ms)
            if next_funding_time_ms is not None and now_ms >= next_funding_time_ms and not _funding_boundary_observed(
                    state, funding_events, next_funding_time_ms, funding_fetch_through_ms or initial_ms):
                pending_funding = min(pending_funding or next_funding_time_ms, next_funding_time_ms)
                state["funding_pending_settlement_ms"] = pending_funding
            funding_covered = pending_funding is None and _funding_coverage_available(
                funding_available, now_ms, _funding_timestamp(funding_fetch_through_ms), next_funding_time_ms)
            if not funding_covered and decision.get("enabled"):
                decision.update(status="funding_unavailable", should_exit=False)
        if fill is None and risk["status"] != "halted" and decision.get("should_exit"):
            fill = self._fill_to(state, 0.0, mark_price, now_ms, now_ms, rules or DEFAULT_SIMULATION_RULES)
            if fill:
                self._record_profit_exit(state, decision, fill)
                equity = self._mark(state, mark_price)["equity"]
                _, risk = constrain_target(state, target, equity, 0.0, mark_price, now_ms)
        state["last_mark_price"] = mark_price
        state["funding_status"] = "ok" if funding_covered else "unavailable_new_risk_blocked"
        state["updated_at_utc"] = datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat()
        self._save(state)
        self._record_fills(state, [fill])
        return {"account_risk": risk, "fill": fill, "funding_pnl": state["funding_pnl"],
                "profit_exit_status": state.get("profit_exit_status")}


def client_order_id(signal_time_ms: int, target_qty: float, phase: str) -> str:
    payload = f"{signal_time_ms}:{target_qty:.8f}:{phase}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    return f"btcauto-{signal_time_ms}-{phase}-{digest}"[:36]


class LiveExecutor:
    def __init__(self, client: BinanceTerminalClient, path: Path = LIVE_STATE_PATH) -> None:
        self.client = client
        self.path = path

    def snapshot(self) -> dict[str, Any]:
        return self.client.account_snapshot(SYMBOL)

    def reconcile(
        self,
        target: dict[str, Any],
        mark_price: float,
        report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ready = self.client.validate_live_ready(SYMBOL)
        leverage = int(ready["leverage"])
        self.client.set_leverage(leverage, SYMBOL)
        snapshot = self.client.account_snapshot(SYMBOL)
        wallet = float(snapshot["account"]["wallet_balance"])
        equity = float(snapshot["account"]["margin_balance"])
        max_notional = min(float(ready["max_notional_usdt"]), max(equity, 0.0) * leverage)
        position = next(iter(snapshot["positions"]), None)
        current_qty = float(position["signed_quantity"]) if position else 0.0
        history = read_json(self.path, {}) or {}
        effective_target, entry_guard = apply_startup_entry_guard(
            history,
            target,
            current_qty,
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        effective_target, risk = constrain_target(history, effective_target, equity, current_qty,
                                                  mark_price, now_ms, report)
        # Persist the hard-stop latch even if a subsequent exchange request fails.
        write_json(self.path, history)
        effective_target, llm_trade_gate = apply_llm_trade_gate(
            report or {},
            effective_target,
            current_qty=current_qty,
            equity=max(equity, 0.0),
            mark_price=mark_price,
            mode="live",
            previous_decision=history.get("llm_trade_gate"),
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        entry_side = "long" if effective_target["target_leverage"] >= 0 else "short"
        entry_gate = execution_entry_gate.decision_at(report, now_ms, entry_side)
        desired_notional = float(effective_target["target_leverage"]) * max(equity, 0.0)
        desired_notional = max(-max_notional, min(max_notional, desired_notional))
        desired_qty = execution_entry_gate.constrain_quantity(desired_notional / mark_price, current_qty, entry_gate)
        target_qty = self.client.quantize_quantity(desired_qty, SYMBOL)
        if desired_qty < 0:
            target_qty = -target_qty
        orders: list[Any] = []
        signal_time = int(target["signal_time_ms"])

        if current_qty and target_qty and current_qty * target_qty < 0:
            close_id = client_order_id(signal_time, 0.0, "close")
            orders.append(self.client.market_order(
                "SELL" if current_qty > 0 else "BUY",
                abs(current_qty),
                close_id,
                symbol=SYMBOL,
                reduce_only=True,
            ))
            current_qty = 0.0

        # Closing a reversal can take time. Recheck immediately before its new
        # exposure rather than reuse the approval from before the close request.
        entry_gate = execution_entry_gate.decision_at(report,
            int(datetime.now(timezone.utc).timestamp() * 1000), entry_side)
        target_qty = execution_entry_gate.constrain_quantity(target_qty, current_qty, entry_gate)
        effective_target["target_leverage"] = target_qty * mark_price / max(equity, 1e-12)

        delta = target_qty - current_qty
        delta_qty = self.client.quantize_quantity(delta, SYMBOL)
        if delta < 0:
            delta_qty = -delta_qty
        if abs(delta_qty) > 0:
            reducing = bool(
                current_qty
                and current_qty * delta_qty < 0
                and abs(target_qty) < abs(current_qty)
            ) or target_qty == 0
            phase = "reduce" if reducing else "open"
            order_id = client_order_id(signal_time, target_qty, phase)
            orders.append(self.client.market_order(
                "BUY" if delta_qty > 0 else "SELL",
                abs(delta_qty),
                order_id,
                symbol=SYMBOL,
                reduce_only=reducing,
            ))

        post_snapshot = self.client.account_snapshot(SYMBOL)
        margin_balance = float(post_snapshot["account"]["margin_balance"])
        _, risk = constrain_target(history, effective_target, margin_balance, target_qty,
                                   mark_price, now_ms, report)
        actual_position = next(iter(post_snapshot["positions"]), None)
        if risk["status"] == "halted" and actual_position:
            actual_qty = float(actual_position["signed_quantity"])
            orders.append(self.client.market_order(
                "SELL" if actual_qty > 0 else "BUY", abs(actual_qty),
                client_order_id(signal_time, 0.0, "risk"), symbol=SYMBOL, reduce_only=True))
            post_snapshot = self.client.account_snapshot(SYMBOL)
            margin_balance = float(post_snapshot["account"]["margin_balance"])
            target_qty = 0.0
            effective_target["target_leverage"] = 0.0
        actual_position = next(iter(post_snapshot["positions"]), None)
        initial_equity = float(history.get("initial_equity") or margin_balance)
        peak_equity = max(float(history.get("peak_equity") or margin_balance), margin_balance)
        drawdown = (
            (peak_equity - margin_balance) / peak_equity * 100 if peak_equity > 0 else 0.0
        )
        equity_curve = list(history.get("equity_curve") or [])
        equity_curve.append({
            "time_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "equity": margin_balance,
            "drawdown_pct": drawdown,
            "signed_qty": float(actual_position["signed_quantity"]) if actual_position else 0.0,
            "price": mark_price,
        })
        result = {
            "mode": "live",
            "updated_at_utc": utc_now(),
            "signal_time_ms": signal_time,
            "target_leverage": effective_target["target_leverage"],
            "desired_target_leverage": target["target_leverage"],
            "target_qty": target_qty,
            "previous_qty": current_qty,
            "max_notional_usdt": max_notional,
            "orders": orders,
            "initial_equity": initial_equity,
            "peak_equity": peak_equity,
            "max_drawdown_pct": max(float(history.get("max_drawdown_pct") or 0), drawdown),
            "equity_curve": equity_curve[-1000:],
            "observed_flat_target": history.get("observed_flat_target", False),
            "blocked_target_id": history.get("blocked_target_id"),
            "entry_guard": entry_guard,
            "llm_trade_gate": llm_trade_gate,
            "execution_entry_gate": entry_gate,
            "account_risk": risk,
            "risk_halt_at_utc": history.get("risk_halt_at_utc"),
            "risk_halt_reason": history.get("risk_halt_reason"),
        }
        write_json(self.path, result)
        return result


def simulation_funding(client, account, now_ms):
    state = account.load()
    start = max(int(state["funding_tracking_start_ms"]),
                int(state.get("funding_last_fetch_ms", state["funding_tracking_start_ms"])) - 86_400_000)
    pending = state.get("funding_pending_settlement_ms")
    if pending is not None:
        start = max(int(state["funding_tracking_start_ms"]), min(start, int(pending)))
    try:
        events = client.funding_history(start, now_ms, SYMBOL)
        if not isinstance(events, list):
            raise ValueError("Invalid funding history response")
        # Validate before any fills. Publication delay is covered by a one-day overlap.
        for row in events:
            int(row["fundingTime"])
            rate, mark = float(row["fundingRate"]), float(row["markPrice"])
            if not math.isfinite(rate) or not math.isfinite(mark) or mark <= 0:
                raise ValueError("Invalid funding history values")
        return events, True
    except (RuntimeError, ValueError, KeyError, TypeError, OSError):
        return (), False


def monitor_simulation_account(client, now_ms, *, clock_available=True, clock_error=None,
                               clock_source=None):
    import simulation_risk_monitor
    return simulation_risk_monitor.monitor(
        client, now_ms, clock_available=clock_available, clock_error=clock_error,
        clock_source=clock_source)


def execute_report(
    mode: str,
    report: dict[str, Any],
    client: BinanceTerminalClient,
    *, account: SimulationAccount | None = None,
) -> dict[str, Any]:
    if mode not in {"simulation", "live"}:
        raise ValueError("Execution mode must be simulation or live")
    if mode == "live" and report.get("research_only"):
        raise ValueError("Research factor candidates are simulation-only")
    if mode == "live" and report.get("freeze_id") == "btc_trend_filter_research_20260917":
        raise ValueError("The September 17 research candidate is simulation-only")
    max_age = float(os.getenv("MAX_SIGNAL_AGE_SECONDS", "900") or 900)
    if mode == "simulation":
        import decision_runtime
        account = account if account is not None else SimulationAccount()
        clock = decision_runtime.resolve_clock(client)
        target = target_from_report(report, now_ms=clock["time_ms"], max_age_seconds=max_age,
                                    leverage_cap=simulation_leverage_cap(report))
        rules = client.symbol_rules(SYMBOL)
        # The next settlement comes from the exchange, never a guessed fixed
        # funding interval. Keep it from BEFORE the fetch to detect crossings.
        before_funding = client.mark_price_observation(SYMBOL)
        clock = decision_runtime.resolve_clock(client, mark_observation=before_funding)
        decision_runtime.validate_mark(before_funding, clock["time_ms"])
        next_funding = _next_funding_boundary(before_funding)
        pending_funding = _funding_timestamp(account.load().get("funding_pending_settlement_ms"))
        funding_cutoff = clock["time_ms"]
        events, available = simulation_funding(client, account, funding_cutoff)
        fetched_through = funding_cutoff if available else None
        # Price comes after the potentially slow metadata and funding lookups.
        observation = client.mark_price_observation(SYMBOL)
        clock = decision_runtime.resolve_clock(client, mark_observation=observation)
        observed = decision_runtime.validate_mark(observation, clock["time_ms"])
        if next_funding is not None and clock["time_ms"] >= next_funding:
            pending_funding = min(pending_funding or next_funding, next_funding)
        if available and not _funding_coverage_available(
                available, clock["time_ms"], fetched_through, next_funding):
            # Fetch once more if a settlement boundary was crossed. A second
            # outage still permits exits with the known cash events, but never
            # grants additional exposure against incomplete funding coverage.
            next_funding = _next_funding_boundary(observation)
            refresh_cutoff = clock["time_ms"]
            refreshed_events, refreshed_available = simulation_funding(client, account, refresh_cutoff)
            if refreshed_available:
                events, fetched_through = refreshed_events, refresh_cutoff
            available = refreshed_available
            observation = client.mark_price_observation(SYMBOL)
            clock = decision_runtime.resolve_clock(client, mark_observation=observation)
            observed = decision_runtime.validate_mark(observation, clock["time_ms"])
        target = target_from_report(report, now_ms=clock["time_ms"], max_age_seconds=max_age,
                                    leverage_cap=simulation_leverage_cap(report))
        client.simulation_clock = clock
        return account.reconcile(
            target,
            observed["price"],
            rules,
            report,
            funding_events=events, funding_available=available, now_ms=clock["time_ms"],
            mark_time_ms=observed["time_ms"],
            funding_fetch_through_ms=fetched_through, next_funding_time_ms=next_funding,
            funding_pending_settlement_ms=pending_funding,
            execution_clock=lambda: decision_runtime.resolve_clock(
                client, mark_observation=observation)["time_ms"],
        )
    target = target_from_report(report, max_age_seconds=max_age)
    mark_price = client.mark_price(SYMBOL)
    return LiveExecutor(client).reconcile(target, mark_price, report)
