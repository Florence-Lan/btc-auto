from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MAINNET_FUTURES_URL = "https://fapi.binance.com"
COOLDOWN_PATH = ROOT / "data/runtime/binance_api_cooldown.json"
RULES_CACHE_PATH = ROOT / "data/runtime/binance_symbol_rules.json"
RULES_CACHE_TTL_SECONDS = 6 * 3600
PUBLIC_TRANSPORT_PREFERENCE_SECONDS = 300
PUBLIC_PATHS = {
    "/fapi/v1/ping", "/fapi/v1/time", "/fapi/v1/exchangeInfo",
    "/fapi/v1/premiumIndex", "/fapi/v1/klines", "/fapi/v1/fundingRate",
    "/fapi/v1/depth", "/fapi/v1/ticker/price", "/fapi/v1/ticker/bookTicker",
}
PRIVATE_QUERY_FIELDS = {
    "signature", "api_key", "apikey", "api_secret", "apisecret",
    "x-mbx-apikey", "authorization", "secret",
}


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
    def __init__(self, message: str, code: int | None = None, status: int | None = None,
                 retry_at_ms: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_at_ms = retry_at_ms


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
        self.rules_cached_at_ms: dict[str, int] = {}
        self.last_public_transport: dict[str, Any] = {}
        self._curl_preferred_until = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def environment(self) -> str:
        return "testnet" if "demo-fapi" in self.base_url or "testnet" in self.base_url else "live"

    @property
    def live_trading_enabled(self) -> bool:
        return env_flag("LIVE_TRADING_ENABLED")

    def cooldown_until_ms(self) -> int:
        if not COOLDOWN_PATH.exists():
            return 0
        payload = json.loads(COOLDOWN_PATH.read_text(encoding="utf-8"))
        if payload.get("base_url") != self.base_url:
            return 0
        until = int(payload.get("retry_at_ms") or 0)
        return until if until > int(time.time() * 1000) else 0

    def record_cooldown(self, message: str, retry_after: str | None = None) -> int:
        now_ms = int(time.time() * 1000)
        match = re.search(r"banned until\s+(\d{13})", message, re.IGNORECASE)
        until = int(match.group(1)) + 5000 if match else now_ms + 60_000
        if retry_after:
            try:
                until = max(until, now_ms + int(float(retry_after) * 1000) + 5000)
            except ValueError:
                pass
        until = max(until, self.cooldown_until_ms(), now_ms + 5000)
        payload = {"base_url": self.base_url, "retry_at_ms": until,
                   "reason": "Binance HTTP rate limit; requests suspended"}
        COOLDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = COOLDOWN_PATH.with_name(f"{COOLDOWN_PATH.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(COOLDOWN_PATH)
        return until

    def _decode_response(self, response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.ok:
            return payload
        code = payload.get("code") if isinstance(payload, dict) else None
        message = payload.get("msg") if isinstance(payload, dict) else response.text
        retry_at_ms = None
        if response.status_code in {418, 429}:
            retry_at_ms = self.record_cooldown(str(message or ""), response.headers.get("Retry-After"))
        raise BinanceApiError(
            f"Binance API {response.status_code}: {message or 'request failed'}",
            code=int(code) if code is not None else None,
            status=response.status_code,
            retry_at_ms=retry_at_ms,
        )

    def _check_public_cooldown(self) -> None:
        retry_at_ms = self.cooldown_until_ms()
        if retry_at_ms:
            raise BinanceApiError(
                f"Binance requests paused until {datetime_from_ms(retry_at_ms)}",
                status=418, retry_at_ms=retry_at_ms,
            )

    def _curl_public_get(self, path: str, params: dict[str, Any] | None) -> requests.Response:
        """Alternate TLS stack for public reads; never reuse authenticated headers."""
        origin = urlsplit(self.base_url)
        if (origin.scheme != "https" or not origin.hostname or origin.username
                or origin.password or origin.path or origin.query or origin.fragment):
            raise requests.ConnectionError("Public fallback requires an HTTPS origin without credentials")
        executable = shutil.which("curl")
        if not executable:
            raise requests.ConnectionError("Public fallback transport is unavailable")
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urlencode(params, doseq=True)
        with tempfile.TemporaryDirectory(prefix="binance-public-") as directory:
            headers_path = Path(directory) / "headers.txt"
            # -q must be first: local curl configuration cannot add credentials,
            # redirects or insecure TLS options to these public requests.
            command = [
                executable, "-q", "--silent", "--show-error", "--request", "GET",
                "--proto", "=https", "--proto-redir", "=https",
                "--connect-timeout", "5", "--max-time", "15",
                "--dump-header", str(headers_path), "--write-out", "\n%{http_code}",
                "--url", url,
            ]
            public_environment = {
                key: value for key, value in os.environ.items()
                if key in {"PATH", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                           "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy", "SSL_CERT_FILE",
                           "SSL_CERT_DIR", "CURL_CA_BUNDLE"}
            }
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=17,
                                        check=False, env=public_environment)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise requests.ConnectionError("Public fallback transport failed") from exc
            if result.returncode:
                # Do not propagate tool output: it can contain local proxy details.
                raise requests.ConnectionError(f"Public fallback transport failed (curl {result.returncode})")
            body, separator, status = result.stdout.rpartition("\n")
            if not separator or not status.isdigit() or not 100 <= int(status) <= 599:
                raise requests.ConnectionError("Invalid public fallback HTTP response")
            response = requests.Response()
            response.status_code = int(status)
            response._content = body.encode("utf-8")
            response.encoding = "utf-8"
            response.url = url
            try:
                raw_headers = headers_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise requests.ConnectionError("Missing public fallback HTTP headers") from exc
            blocks = [block for block in re.split(r"\r?\n\r?\n", raw_headers) if block.strip()]
            for line in blocks[-1].splitlines() if blocks else []:
                name, separator, value = line.partition(":")
                if separator:
                    response.headers[name.strip()] = value.strip()
            return response

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if path not in PUBLIC_PATHS and not re.fullmatch(r"/futures/data/[A-Za-z_]+", path):
            raise ValueError("Unsupported public Binance endpoint")
        if params and any(str(key).lower() in PRIVATE_QUERY_FIELDS for key in params):
            raise ValueError("Credentials and signatures are not allowed in public queries")
        transports = ("curl", "requests") if time.monotonic() < self._curl_preferred_until else ("requests", "curl")
        first_error: requests.RequestException | None = None
        for index, transport in enumerate(transports):
            # A transport retry must respect a ban recorded by another worker.
            self._check_public_cooldown()
            self.last_public_transport = {
                "transport": transport, "endpoint": path,
                "observed_at_ms": int(time.time() * 1000), "ok": False,
                "fallback": index > 0,
                "primary_error_type": type(first_error).__name__ if first_error else None,
            }
            try:
                response = (self._curl_public_get(path, params) if transport == "curl" else
                            self.session.get(f"{self.base_url}{path}", params=params, timeout=15))
                payload = self._decode_response(response)
                if payload is None:
                    raise ValueError("Invalid public Binance JSON response")
            except (requests.ConnectionError, requests.Timeout) as exc:
                self.last_public_transport["error_type"] = type(exc).__name__
                if transport == "curl":
                    self._curl_preferred_until = 0.0
                if index:
                    raise requests.ConnectionError("Both Binance public transports failed") from exc
                first_error = exc
                continue
            self.last_public_transport.update(ok=True, observed_at_ms=int(time.time() * 1000))
            if transport == "curl":
                self._curl_preferred_until = time.monotonic() + PUBLIC_TRANSPORT_PREFERENCE_SECONDS
            return payload
        raise requests.ConnectionError("No Binance public transport available")

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

    def mark_price_observation(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Return the exchange timestamp too, so risk monitoring can reject stale marks."""
        payload = self.public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
        if not isinstance(payload, dict) or payload.get("symbol") != symbol:
            raise ValueError("Invalid mark-price observation")
        observation = {"price": float(payload["markPrice"]), "time_ms": int(payload["time"])}
        if payload.get("nextFundingTime") is not None:
            observation["next_funding_time_ms"] = int(payload["nextFundingTime"])
        return observation

    def server_time_ms(self) -> int:
        return int(self.public_get("/fapi/v1/time")["serverTime"])

    def funding_history(self, start_ms: int, end_ms: int, symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
        """Fetch actual settlement rates and their associated mark prices, with pagination."""
        rows = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self.public_get("/fapi/v1/fundingRate", {
                "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
            })
            if not isinstance(batch, list):
                raise ValueError("Invalid funding history response")
            rows.extend(row for row in batch if start_ms <= int(row["fundingTime"]) <= end_ms)
            if not batch or len(batch) < 1000:
                break
            next_cursor = int(batch[-1]["fundingTime"]) + 1
            if next_cursor <= cursor:
                raise ValueError("Funding history pagination did not advance")
            cursor = next_cursor
        return sorted(rows, key=lambda row: int(row["fundingTime"]))

    def symbol_rules(self, symbol: str = "BTCUSDT") -> dict[str, Decimal]:
        with self.lock:
            now_ms = int(time.time() * 1000)
            cached_at = self.rules_cached_at_ms.get(symbol, 0)
            if symbol in self.rules_cache and 0 <= now_ms - cached_at < RULES_CACHE_TTL_SECONDS * 1000:
                return self.rules_cache[symbol]
            cached = self._load_symbol_rules(symbol, now_ms)
            if cached is not None:
                rules, fetched_at_ms = cached
                self.rules_cache[symbol] = rules
                self.rules_cached_at_ms[symbol] = fetched_at_ms
                return rules
            payload = self.public_get("/fapi/v1/exchangeInfo")
            item = next(entry for entry in payload["symbols"] if entry["symbol"] == symbol)
            filters = {entry["filterType"]: entry for entry in item["filters"]}
            lot = filters["LOT_SIZE"]
            market_lot = filters.get("MARKET_LOT_SIZE", lot)
            min_notional = filters.get("MIN_NOTIONAL", {})
            rules = self._validate_symbol_rules({
                "step_size": market_lot.get("stepSize") or lot["stepSize"],
                "min_qty": market_lot.get("minQty") or lot["minQty"],
                "max_qty": market_lot.get("maxQty") or lot["maxQty"],
                "min_notional": min_notional.get("notional", "0"),
            })
            fetched_at_ms = int(time.time() * 1000)
            self.rules_cache[symbol] = rules
            self.rules_cached_at_ms[symbol] = fetched_at_ms
            self._save_symbol_rules(symbol, rules, fetched_at_ms)
            return rules

    @staticmethod
    def _validate_symbol_rules(raw: dict[str, Any]) -> dict[str, Decimal]:
        rules = {name: Decimal(str(raw[name])) for name in ("step_size", "min_qty", "max_qty", "min_notional")}
        if (any(not value.is_finite() for value in rules.values())
                or rules["step_size"] <= 0 or rules["min_qty"] <= 0
                or rules["max_qty"] < rules["min_qty"] or rules["min_notional"] < 0):
            raise ValueError("Invalid Binance symbol rules")
        return rules

    def _load_symbol_rules(self, symbol: str, now_ms: int) -> tuple[dict[str, Decimal], int] | None:
        try:
            payload = json.loads(RULES_CACHE_PATH.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                    or payload.get("base_url") != self.base_url):
                return None
            cached = payload["symbols"][symbol]
            fetched_at_ms = int(cached["fetched_at_ms"])
            if not 0 <= now_ms - fetched_at_ms < RULES_CACHE_TTL_SECONDS * 1000:
                return None
            return self._validate_symbol_rules(cached["rules"]), fetched_at_ms
        except (OSError, KeyError, TypeError, ValueError, InvalidOperation):
            return None

    def _save_symbol_rules(self, symbol: str, rules: dict[str, Decimal], fetched_at_ms: int) -> None:
        temporary: Path | None = None
        try:
            try:
                payload = json.loads(RULES_CACHE_PATH.read_text(encoding="utf-8"))
                if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                        or payload.get("base_url") != self.base_url or not isinstance(payload.get("symbols"), dict)):
                    payload = {}
            except (OSError, TypeError, ValueError):
                payload = {}
            payload.update(schema_version=1, base_url=self.base_url)
            payload.setdefault("symbols", {})[symbol] = {
                "fetched_at_ms": fetched_at_ms, "rules": {key: str(value) for key, value in rules.items()},
            }
            RULES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=RULES_CACHE_PATH.parent,
                                             prefix=RULES_CACHE_PATH.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            temporary.replace(RULES_CACHE_PATH)
        except OSError:
            # A cache-write failure cannot invalidate freshly fetched exchange rules.
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

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
