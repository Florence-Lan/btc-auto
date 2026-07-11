from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def sign_query(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class BinanceTerminalClient:
    """Read-only USD-M futures account adapter for the local terminal."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "BINANCE_FUTURES_BASE_URL",
            "https://demo-fapi.binance.com",
        ).rstrip("/")
        self.api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.cached_at = 0.0
        self.cached: dict[str, Any] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def environment(self) -> str:
        return "testnet" if "demo-fapi" in self.base_url or "testnet" in self.base_url else "live"

    def signed_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.configured:
            raise RuntimeError("Binance API key is not configured")
        server_time = self.session.get(
            f"{self.base_url}/fapi/v1/time",
            timeout=10,
        ).json()["serverTime"]
        payload = {
            **(params or {}),
            "recvWindow": 5000,
            "timestamp": int(server_time),
        }
        query = urlencode(payload)
        signature = sign_query(self.api_secret, query)
        response = self.session.get(
            f"{self.base_url}{path}?{query}&signature={signature}",
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def snapshot(self, symbol: str = "BTCUSDT", cache_seconds: float = 8.0) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "connected": False,
                "environment": self.environment,
                "read_only": True,
                "error": "API credentials not configured",
            }
        with self.lock:
            now = time.time()
            if self.cached is not None and now - self.cached_at < cache_seconds:
                return self.cached
            try:
                account = self.signed_get("/fapi/v2/account")
                raw_positions = self.signed_get(
                    "/fapi/v2/positionRisk",
                    {"symbol": symbol},
                )
                raw_orders = self.signed_get(
                    "/fapi/v1/openOrders",
                    {"symbol": symbol},
                )
                asset = next(
                    (item for item in account.get("assets", []) if item.get("asset") == "USDT"),
                    {},
                )
                positions = [
                    {
                        "symbol": item.get("symbol"),
                        "side": "LONG" if float(item.get("positionAmt", 0)) > 0 else "SHORT",
                        "quantity": abs(float(item.get("positionAmt", 0))),
                        "entry_price": float(item.get("entryPrice", 0)),
                        "mark_price": float(item.get("markPrice", 0)),
                        "unrealized_pnl": float(item.get("unRealizedProfit", 0)),
                        "liquidation_price": float(item.get("liquidationPrice", 0)),
                        "leverage": float(item.get("leverage", 0)),
                        "source": f"Binance {self.environment}",
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
                result = {
                    "configured": True,
                    "connected": True,
                    "environment": self.environment,
                    "read_only": True,
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
                    "error": None,
                }
            except (requests.RequestException, KeyError, TypeError, ValueError, RuntimeError) as exc:
                result = {
                    "configured": True,
                    "connected": False,
                    "environment": self.environment,
                    "read_only": True,
                    "error": str(exc),
                }
            self.cached = result
            self.cached_at = now
            return result
