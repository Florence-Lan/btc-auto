from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MAINNET_FUTURES_URL = "https://fapi.binance.com"


def sign_query(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def datetime_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class BinanceApiError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class BinanceTerminalClient:
    """Binance USD-M client with public market data and guarded live execution."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "BINANCE_FUTURES_BASE_URL",
            MAINNET_FUTURES_URL,
        ).rstrip("/")
        self.api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        self.session = requests.Session()
        self.lock = threading.RLock()
        self.cached_at = 0.0
        self.cached: dict[str, Any] | None = None
        self.rules_cache: dict[str, dict[str, Decimal]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def environment(self) -> str:
        return "testnet" if "demo-fapi" in self.base_url or "testnet" in self.base_url else "live"

    @property
    def live_trading_enabled(self) -> bool:
        return env_flag("LIVE_TRADING_ENABLED")

    def _decode_response(self, response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.ok:
            return payload
        code = payload.get("code") if isinstance(payload, dict) else None
        message = payload.get("msg") if isinstance(payload, dict) else response.text
        raise BinanceApiError(
            f"Binance API {response.status_code}: {message or 'request failed'}",
            code=int(code) if code is not None else None,
            status=response.status_code,
        )

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=15)
        return self._decode_response(response)

    def signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if not self.configured:
            raise RuntimeError("Binance API credentials are not configured")
        server_time = self.public_get("/fapi/v1/time")["serverTime"]
        payload = {
            **(params or {}),
            "recvWindow": 5000,
            "timestamp": int(server_time),
        }
        query = urlencode(payload)
        signature = sign_query(self.api_secret, query)
        response = self.session.request(
            method.upper(),
            f"{self.base_url}{path}?{query}&signature={signature}",
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=20,
        )
        return self._decode_response(response)

    def signed_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.signed_request("GET", path, params)

    def mark_price(self, symbol: str = "BTCUSDT") -> float:
        payload = self.public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(payload["markPrice"])

    def server_time_ms(self) -> int:
        return int(self.public_get("/fapi/v1/time")["serverTime"])

    def symbol_rules(self, symbol: str = "BTCUSDT") -> dict[str, Decimal]:
        if symbol in self.rules_cache:
            return self.rules_cache[symbol]
        payload = self.public_get("/fapi/v1/exchangeInfo")
        item = next(entry for entry in payload["symbols"] if entry["symbol"] == symbol)
        filters = {entry["filterType"]: entry for entry in item["filters"]}
        lot = filters["LOT_SIZE"]
        market_lot = filters.get("MARKET_LOT_SIZE", lot)
        min_notional = filters.get("MIN_NOTIONAL", {})
        rules = {
            "step_size": Decimal(str(market_lot.get("stepSize") or lot["stepSize"])),
            "min_qty": Decimal(str(market_lot.get("minQty") or lot["minQty"])),
            "max_qty": Decimal(str(market_lot.get("maxQty") or lot["maxQty"])),
            "min_notional": Decimal(str(min_notional.get("notional", "0"))),
        }
        self.rules_cache[symbol] = rules
        return rules

    def quantize_quantity(self, quantity: float, symbol: str = "BTCUSDT") -> float:
        rules = self.symbol_rules(symbol)
        value = Decimal(str(abs(quantity)))
        step = rules["step_size"]
        quantized = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
        if quantized < rules["min_qty"]:
            return 0.0
        return float(quantized)

    def account_snapshot(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        account = self.signed_get("/fapi/v2/account")
        raw_positions = self.signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
        raw_orders = self.signed_get("/fapi/v1/openOrders", {"symbol": symbol})
        asset = next(
            (item for item in account.get("assets", []) if item.get("asset") == "USDT"),
            {},
        )
        positions = [
            {
                "symbol": item.get("symbol"),
                "side": "LONG" if float(item.get("positionAmt", 0)) > 0 else "SHORT",
                "quantity": abs(float(item.get("positionAmt", 0))),
                "signed_quantity": float(item.get("positionAmt", 0)),
                "entry_price": float(item.get("entryPrice", 0)),
                "mark_price": float(item.get("markPrice", 0)),
                "unrealized_pnl": float(item.get("unRealizedProfit", 0)),
                "leverage": float(item.get("leverage", 0)),
                "source": "Binance live",
            }
            for item in raw_positions
            if abs(float(item.get("positionAmt", 0))) > 1e-12
        ]
        orders = [
            {
                "symbol": item.get("symbol"),
                "type": item.get("type"),
                "side": item.get("side"),
                "price": float(item.get("price", 0)),
                "quantity": float(item.get("origQty", 0)),
                "status": item.get("status"),
                "reduce_only": bool(item.get("reduceOnly", False)),
                "order_id": item.get("orderId"),
            }
            for item in raw_orders
        ]
        return {
            "account": {
                "wallet_balance": float(asset.get("walletBalance", 0)),
                "available_balance": float(asset.get("availableBalance", 0)),
                "unrealized_pnl": float(asset.get("unrealizedProfit", 0)),
                "margin_balance": float(asset.get("marginBalance", 0)),
                "margin_ratio_pct": (
                    float(account.get("totalMaintMargin", 0))
                    / max(float(account.get("totalMarginBalance", 0)), 1e-12)
                    * 100
                ),
            },
            "positions": positions,
            "open_orders": orders,
        }

    def recent_trades(self, symbol: str = "BTCUSDT", limit: int = 20) -> list[dict[str, Any]]:
        rows = self.signed_get(
            "/fapi/v1/userTrades",
            {"symbol": symbol, "limit": max(1, min(int(limit), 100))},
        )
        return [
            {
                "time_utc": datetime_from_ms(int(item.get("time", 0))),
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "price": float(item.get("price", 0)),
                "quantity": float(item.get("qty", 0)),
                "realized_pnl": float(item.get("realizedPnl", 0)),
                "fee": float(item.get("commission", 0)),
                "mode": "live",
                "order_id": item.get("orderId"),
            }
            for item in reversed(rows)
        ]

    def snapshot(self, symbol: str = "BTCUSDT", cache_seconds: float = 8.0) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "connected": False,
                "environment": self.environment,
                "read_only": not self.live_trading_enabled,
                "live_trading_enabled": self.live_trading_enabled,
                "error": "API credentials not configured",
            }
        with self.lock:
            now = time.time()
            if self.cached is not None and now - self.cached_at < cache_seconds:
                return self.cached
            try:
                data = self.account_snapshot(symbol)
                result = {
                    "configured": True,
                    "connected": True,
                    "environment": self.environment,
                    "read_only": not self.live_trading_enabled,
                    "live_trading_enabled": self.live_trading_enabled,
                    **data,
                    "error": None,
                }
            except (requests.RequestException, KeyError, TypeError, ValueError, RuntimeError) as exc:
                result = {
                    "configured": True,
                    "connected": False,
                    "environment": self.environment,
                    "read_only": not self.live_trading_enabled,
                    "live_trading_enabled": self.live_trading_enabled,
                    "error": str(exc),
                }
            self.cached = result
            self.cached_at = now
            return result

    def validate_live_ready(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        if not self.live_trading_enabled:
            raise RuntimeError("Set LIVE_TRADING_ENABLED=true before enabling live mode")
        if self.environment != "live" or self.base_url != MAINNET_FUTURES_URL:
            raise RuntimeError("Live mode requires BINANCE_FUTURES_BASE_URL=https://fapi.binance.com")
        max_notional = float(os.getenv("LIVE_MAX_NOTIONAL_USDT", "0") or 0)
        leverage = int(os.getenv("LIVE_LEVERAGE", "1") or 1)
        if max_notional <= 0:
            raise RuntimeError("LIVE_MAX_NOTIONAL_USDT must be configured and greater than zero")
        if leverage < 1 or leverage > 2:
            raise RuntimeError("LIVE_LEVERAGE must be between 1 and 2")
        position_mode = self.signed_get("/fapi/v1/positionSide/dual")
        if bool(position_mode.get("dualSidePosition")):
            raise RuntimeError("Live execution requires Binance one-way position mode")
        snapshot = self.account_snapshot(symbol)
        rules = self.symbol_rules(symbol)
        mark_price = self.mark_price(symbol)
        minimum_order_notional = max(
            float(rules["min_notional"]),
            float(rules["min_qty"]) * mark_price,
        )
        if max_notional < minimum_order_notional:
            raise RuntimeError(
                f"LIVE_MAX_NOTIONAL_USDT must be at least {minimum_order_notional:.2f} "
                "for the current BTCUSDT minimum quantity"
            )
        return {
            "max_notional_usdt": max_notional,
            "leverage": leverage,
            "wallet_balance": snapshot["account"]["wallet_balance"],
            "minimum_order_notional": minimum_order_notional,
        }

    def set_leverage(self, leverage: int, symbol: str = "BTCUSDT") -> Any:
        return self.signed_request(
            "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}
        )

    def query_order(self, client_order_id: str, symbol: str = "BTCUSDT") -> Any:
        return self.signed_get(
            "/fapi/v1/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
        )

    def market_order(
        self,
        side: str,
        quantity: float,
        client_order_id: str,
        *,
        symbol: str = "BTCUSDT",
        reduce_only: bool = False,
    ) -> Any:
        qty = self.quantize_quantity(quantity, symbol)
        if qty <= 0:
            raise ValueError("Order quantity is below the Binance minimum")
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": format(qty, ".8f").rstrip("0").rstrip("."),
            "newClientOrderId": client_order_id[:36],
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        try:
            result = self.signed_request("POST", "/fapi/v1/order", params)
        except BinanceApiError as exc:
            if exc.code in {-2010, -4111}:
                return self.query_order(client_order_id[:36], symbol)
            raise
        self.cached_at = 0.0
        return result

    def cancel_all_orders(self, symbol: str = "BTCUSDT") -> Any:
        result = self.signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})
        self.cached_at = 0.0
        return result

    def flatten_position(self, symbol: str = "BTCUSDT", client_order_id: str = "btcauto-emergency") -> Any | None:
        snapshot = self.account_snapshot(symbol)
        position = next(iter(snapshot["positions"]), None)
        if not position:
            return None
        signed_qty = float(position["signed_quantity"])
        return self.market_order(
            "SELL" if signed_qty > 0 else "BUY",
            abs(signed_qty),
            client_order_id,
            symbol=symbol,
            reduce_only=True,
        )
