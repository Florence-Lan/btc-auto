from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Mapping

from binance_terminal_client import BinanceTerminalClient
from llm_trade_gate import apply_llm_trade_gate


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_STATE_PATH = ROOT / "data/runtime/simulation_account.json"
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
    current_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
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
    if blocked_id == position_id or first_observation:
        state["blocked_target_id"] = position_id
        guarded["target_leverage"] = 0.0
        return guarded, {
            "status": "waiting_for_next_signal",
            "position_id": position_id,
            "origin_signal_time_ms": target.get("origin_signal_time_ms"),
        }
    return guarded, {"status": "allowed", "position_id": position_id}


class SimulationAccount:
    def __init__(self, path: Path = SIMULATION_STATE_PATH) -> None:
        self.path = path

    def _new_state(self) -> dict[str, Any]:
        initial = float(os.getenv("SIM_INITIAL_BALANCE_USDT", "100") or 100)
        if initial <= 0:
            raise ValueError("SIM_INITIAL_BALANCE_USDT must be greater than zero")
        now = utc_now()
        return {
            "version": 2,
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
        }

    def load(self) -> dict[str, Any]:
        state = read_json(self.path)
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
        state["version"] = 2
        return state

    def reset(self, initial_balance: float) -> dict[str, Any]:
        amount = float(initial_balance)
        if not math.isfinite(amount) or amount < 1 or amount > 1_000_000_000:
            raise ValueError("模拟盘初始金额必须在 1 至 1,000,000,000 USDT 之间")
        state = self._new_state()
        state["initial_balance"] = amount
        state["wallet_balance"] = amount
        state["peak_equity"] = amount
        state["updated_at_utc"] = utc_now()
        write_json(self.path, state)
        return state

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
                "available_balance": max(marked["equity"] - abs(qty * price) / 2, 0.0),
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
    ) -> dict[str, Any]:
        state = self.load()
        marked = self._mark(state, mark_price)
        leverage_cap = min(float(os.getenv("SIM_MAX_LEVERAGE", "2") or 2), 2.0)
        configured_cap = float(os.getenv("SIM_MAX_NOTIONAL_USDT", "0") or 0)
        equity = max(marked["equity"], 0.0)
        natural_cap = equity * leverage_cap
        max_notional = min(natural_cap, configured_cap) if configured_cap > 0 else natural_cap
        current_qty = float(state.get("position_qty", 0))
        effective_target, entry_guard = apply_startup_entry_guard(
            state,
            target,
            current_qty,
        )
        state["entry_guard"] = entry_guard
        effective_target, llm_trade_gate = apply_llm_trade_gate(
            report or {},
            effective_target,
            current_qty=current_qty,
            equity=equity,
            mark_price=mark_price,
            mode="simulation",
            previous_decision=state.get("llm_trade_gate"),
        )
        state["llm_trade_gate"] = llm_trade_gate
        target_notional = float(effective_target["target_leverage"]) * equity
        target_notional = max(-max_notional, min(max_notional, target_notional))
        target_qty = target_notional / mark_price if mark_price > 0 else 0.0
        target_qty = quantize_signed_quantity(target_qty, rules)
        delta = target_qty - current_qty
        fill = None
        active_rules = rules or DEFAULT_SIMULATION_RULES
        min_qty = float(active_rules["min_qty"])
        min_notional = float(active_rules["min_notional"])
        delta_notional = abs(delta * mark_price)
        full_close = target_qty == 0 and abs(current_qty) > 1e-12
        if full_close or (
            abs(delta) >= min_qty and delta_notional >= min_notional
        ):
            slippage_bps = float(os.getenv("SIM_SLIPPAGE_BPS", "1.0") or 1.0)
            fee_rate = float(os.getenv("SIM_TAKER_FEE", "0.00045") or 0.00045)
            fill_price = mark_price * (1 + math.copysign(slippage_bps / 10_000, delta))
            entry = float(state.get("entry_price", 0))
            realized = 0.0
            if current_qty == 0 or current_qty * delta > 0:
                new_qty = current_qty + delta
                if current_qty == 0:
                    new_entry = fill_price
                else:
                    new_entry = (
                        abs(current_qty) * entry + abs(delta) * fill_price
                    ) / abs(new_qty)
            else:
                closing_qty = min(abs(current_qty), abs(delta))
                realized = (fill_price - entry) * closing_qty * math.copysign(1, current_qty)
                new_qty = current_qty + delta
                if abs(new_qty) < 1e-12:
                    new_qty = 0.0
                    new_entry = 0.0
                elif current_qty * new_qty > 0:
                    new_entry = entry
                else:
                    new_entry = fill_price
            fee = abs(delta * fill_price) * fee_rate
            state["wallet_balance"] = float(state["wallet_balance"]) + realized - fee
            state["position_qty"] = new_qty
            state["entry_price"] = new_entry
            state["realized_pnl"] = float(state.get("realized_pnl", 0)) + realized
            state["fees_paid"] = float(state.get("fees_paid", 0)) + fee
            fill = {
                "time_utc": utc_now(),
                "signal_time_ms": int(target["signal_time_ms"]),
                "symbol": SYMBOL,
                "side": "BUY" if delta > 0 else "SELL",
                "quantity": abs(delta),
                "price": fill_price,
                "realized_pnl": realized,
                "fee": fee,
                "mode": "simulation",
            }
            state.setdefault("fills", []).append(fill)
            state["fills"] = state["fills"][-200:]
            state["fill_count_total"] = int(state.get("fill_count_total") or 0) + 1
        state["last_signal_time_ms"] = int(target["signal_time_ms"])
        state["last_mark_price"] = mark_price
        after = self._mark(state, mark_price)
        state["peak_equity"] = max(float(state.get("peak_equity", after["equity"])), after["equity"])
        drawdown = (
            (state["peak_equity"] - after["equity"]) / state["peak_equity"] * 100
            if state["peak_equity"] > 0 else 0.0
        )
        state["max_drawdown_pct"] = max(float(state.get("max_drawdown_pct", 0)), drawdown)
        state.setdefault("equity_curve", []).append({
            "time_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "equity": after["equity"],
            "drawdown_pct": drawdown,
            "signed_qty": float(state.get("position_qty", 0)),
            "price": mark_price,
        })
        state["equity_curve"] = state["equity_curve"][-1000:]
        state["updated_at_utc"] = utc_now()
        write_json(self.path, state)
        return {
            "mode": "simulation",
            "target_leverage": effective_target["target_leverage"],
            "desired_target_leverage": target["target_leverage"],
            "target_qty": target_qty,
            "fill": fill,
            "entry_guard": entry_guard,
            "llm_trade_gate": llm_trade_gate,
            "account": self.snapshot(mark_price)["account"],
        }


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
        max_notional = min(float(ready["max_notional_usdt"]), wallet * leverage)
        position = next(iter(snapshot["positions"]), None)
        current_qty = float(position["signed_quantity"]) if position else 0.0
        history = read_json(self.path, {}) or {}
        effective_target, entry_guard = apply_startup_entry_guard(
            history,
            target,
            current_qty,
        )
        effective_target, llm_trade_gate = apply_llm_trade_gate(
            report or {},
            effective_target,
            current_qty=current_qty,
            equity=wallet,
            mark_price=mark_price,
            mode="live",
            previous_decision=history.get("llm_trade_gate"),
        )
        desired_notional = float(effective_target["target_leverage"]) * wallet
        desired_notional = max(-max_notional, min(max_notional, desired_notional))
        target_qty = self.client.quantize_quantity(desired_notional / mark_price, SYMBOL)
        if desired_notional < 0:
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
            "signed_qty": target_qty,
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
        }
        write_json(self.path, result)
        return result


def execute_report(
    mode: str,
    report: dict[str, Any],
    client: BinanceTerminalClient,
) -> dict[str, Any]:
    if mode not in {"simulation", "live"}:
        raise ValueError("Execution mode must be simulation or live")
    max_age = float(os.getenv("MAX_SIGNAL_AGE_SECONDS", "900") or 900)
    target = target_from_report(report, max_age_seconds=max_age)
    mark_price = client.mark_price(SYMBOL)
    if mode == "simulation":
        return SimulationAccount().reconcile(
            target,
            mark_price,
            client.symbol_rules(SYMBOL),
            report,
        )
    return LiveExecutor(client).reconcile(target, mark_price, report)
