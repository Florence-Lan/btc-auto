"""Incremental, validated public market history for forward paper decisions.

Only complete bars enter strategy input. A forming bar contributes its immutable
opening price alone; its unfinished high, low, close and volume are never parsed.
All requests use the terminal client, including its shared rate-limit cooldown.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import simulate_range_swing as sim
from timeseries_execution import Opening


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "data/runtime/market_cache"
CACHE_VERSION = 1
FUNDING_OVERLAP_MS = sim.MS_PER_DAY


class MarketDataUnavailable(RuntimeError):
    """Required current history is incomplete; stale inputs cannot be evaluated."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class MarketDataResult:
    candles: dict[str, list[sim.Candle]]
    funding: sim.FundingHistory
    openings: dict[str, dict[int, Opening]]
    diagnostics: dict[str, Any]
    asof_ms: int

    def opening(self, interval: str, timestamp: int) -> Opening | None:
        """Resolve a known hourly open from closed 5m history or an open tick."""
        if timestamp > self.asof_ms:
            return None
        for source in dict.fromkeys((interval, "5m")):
            for bar in reversed(self.candles.get(source, [])):
                if bar.open_time_ms == timestamp:
                    return Opening(timestamp, bar.open, bar.close_time_ms)
                if bar.open_time_ms < timestamp:
                    break
            tick = self.openings.get(source, {}).get(timestamp)
            if tick is not None and tick.observed_at_ms <= self.asof_ms:
                return tick
        return None


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"Invalid {name}")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"Invalid {name}")
    return parsed


def _price(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Invalid {name}")
    return parsed


def _venue(client: Any) -> str:
    value = getattr(client, "environment", "live")
    # Public-only adapters and test clients may omit the trading-environment
    # property. An actual terminal client always returns live or testnet.
    return value if isinstance(value, str) else "live"


def _validate_candle(bar: sim.Candle, step: int) -> None:
    if (bar.open_time_ms < 0 or bar.open_time_ms % step
            or bar.close_time_ms != bar.open_time_ms + step - 1):
        raise ValueError("Invalid candle timestamps")
    prices = (bar.open, bar.high, bar.low, bar.close)
    if any(not math.isfinite(p) or p <= 0 for p in prices):
        raise ValueError("Invalid candle price")
    if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
        raise ValueError("Invalid candle OHLC range")
    if any(not math.isfinite(v) or v < 0 for v in (bar.volume, bar.quote_volume)):
        raise ValueError("Invalid candle volume")


def _read(path: Path, symbol: str, kind: str, venue: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("version") != CACHE_VERSION or payload.get("symbol") != symbol
            or payload.get("kind") != kind or payload.get("venue", "live") != venue):
        raise ValueError("Market cache identity does not match request")
    return payload


def _write(path: Path, symbol: str, kind: str, venue: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CACHE_VERSION, "symbol": symbol, "kind": kind, "venue": venue, **payload}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _required_bounds(start_ms: int, asof_ms: int, step: int) -> tuple[int, int]:
    first = (start_ms + step - 1) // step * step
    last = (asof_ms + 1) // step * step - step
    if last < first:
        raise ValueError("Market request does not contain a closed warmup candle")
    return first, last


def _missing_spans(rows: dict[int, sim.Candle], first: int, last: int, step: int) -> list[tuple[int, int]]:
    spans = []
    start = None
    for timestamp in range(first, last + 1, step):
        if timestamp not in rows:
            if start is None:
                start = timestamp
        elif start is not None:
            spans.append((start, timestamp - step))
            start = None
    if start is not None:
        spans.append((start, last))
    return spans


def _load_interval(client: Any, symbol: str, interval: str, start_ms: int,
                   asof_ms: int, cache_dir: Path) -> tuple[list[sim.Candle], dict[int, Opening], dict[str, Any]]:
    step = sim.interval_to_ms(interval)
    first, last = _required_bounds(start_ms, asof_ms, step)
    venue = _venue(client)
    path = cache_dir / f"{'testnet_' if venue == 'testnet' else ''}{symbol}_{interval}.json"
    rows: dict[int, sim.Candle] = {}
    openings: dict[int, Opening] = {}
    diagnostics: dict[str, Any] = {"required_first_open_ms": first,
                                   "required_last_close_ms": last + step - 1}
    try:
        payload = _read(path, symbol, interval, venue)
        for raw in payload.get("candles", []):
            if len(raw) != 8:
                raise ValueError("Invalid compact candle")
            _integer(raw[0], "candle open timestamp")
            _integer(raw[7], "candle close timestamp")
            bar = sim.candle_from_compact(raw)
            _validate_candle(bar, step)
            if bar.open_time_ms in rows:
                raise ValueError("Duplicate candle in market cache")
            rows[bar.open_time_ms] = bar
        for raw in payload.get("openings", []):
            timestamp = _integer(raw[0], "opening timestamp")
            observed = _integer(raw[2], "opening observation timestamp")
            if timestamp % step or observed < timestamp:
                raise ValueError("Invalid cached opening timestamp")
            openings[timestamp] = Opening(timestamp, _price(raw[1], "opening price"), observed)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        # A damaged cache can be rebuilt from public data, never silently trusted.
        diagnostics["cache_error"] = str(exc)
        rows, openings = {}, {}

    spans = _missing_spans(rows, first, last, step)
    forming_time = asof_ms // step * step
    needs_opening = (forming_time > last and (forming_time not in openings
                     or openings[forming_time].observed_at_ms > asof_ms))
    if needs_opening:
        if spans and spans[-1][1] == last:
            spans[-1] = (spans[-1][0], forming_time)
        else:
            spans.append((forming_time, forming_time))
    requests_made = 0
    error = None
    try:
        for span_start, span_last in spans:
            cursor = span_start
            end = min(asof_ms, span_last + step - 1)
            while cursor <= end:
                requests_made += 1
                batch = client.public_get("/fapi/v1/klines", {
                    "symbol": symbol, "interval": interval, "startTime": cursor,
                    "endTime": end, "limit": 1500,
                })
                if not isinstance(batch, list):
                    raise ValueError("Invalid kline response")
                if not batch:
                    break
                previous = cursor - step
                staged_rows = {}
                staged_openings = {}
                for raw in batch:
                    timestamp = _integer(raw[0], "kline timestamp")
                    if timestamp % step or timestamp <= previous or not cursor <= timestamp <= end:
                        raise ValueError("Unordered or out-of-range kline response")
                    previous = timestamp
                    if timestamp + step - 1 <= asof_ms:
                        # Only closed bars are parsed as full OHLCV.
                        _integer(raw[6], "kline close timestamp")
                        bar = sim.candle_from_kline(raw)
                        _validate_candle(bar, step)
                        staged_rows[timestamp] = bar
                    else:
                        price = _price(raw[1], "opening price")
                        old = openings.get(timestamp)
                        if old is not None and old.price != price:
                            raise ValueError("Opening price changed after observation")
                        staged_openings[timestamp] = Opening(timestamp, price,
                            min(old.observed_at_ms, asof_ms) if old else asof_ms)
                # Commit a page only after its complete shape and chronology
                # pass validation, so a malformed response cannot seed a cache.
                rows.update(staged_rows)
                openings.update(staged_openings)
                next_cursor = previous + step
                if next_cursor <= cursor:
                    raise ValueError("Kline pagination did not advance")
                cursor = next_cursor
                if len(batch) < 1500:
                    break
    except Exception as exc:
        # The caller still gets a usable input if independently validated cached
        # data already contains every candle required for this exact decision.
        error = f"{type(exc).__name__}: {exc}"

    if requests_made and (rows or openings):
        _write(path, symbol, interval, venue, {
            "candles": [sim.candle_to_compact(rows[t]) for t in sorted(rows)],
            "openings": [[t, tick.price, tick.observed_at_ms] for t, tick in sorted(openings.items())],
        })
    missing = _missing_spans(rows, first, last, step)
    diagnostics.update({"source": "cache_fallback" if error else "live" if requests_made else "cache",
                        "requests": requests_made, "network_error": error,
                        "cached_last_close_ms": max((b.close_time_ms for b in rows.values()
                                                     if b.close_time_ms <= asof_ms), default=None),
                        "complete": not missing})
    if missing:
        diagnostics["missing_spans"] = missing
        raise MarketDataUnavailable(f"{interval} closed history is incomplete or stale", diagnostics)
    closed = [rows[t] for t in range(first, last + 1, step)]
    ticks = {t: tick for t, tick in openings.items()
             if t <= asof_ms and tick.observed_at_ms <= asof_ms}
    return closed, ticks, diagnostics


def _load_funding(client: Any, symbol: str, start_ms: int, asof_ms: int,
                  required_until_ms: int, cache_dir: Path) -> tuple[sim.FundingHistory, dict[str, Any]]:
    venue = _venue(client)
    path = cache_dir / f"{'testnet_' if venue == 'testnet' else ''}{symbol}_funding.json"
    rows: dict[int, float] = {}
    coverage_start = None
    coverage_end = -1
    diagnostics: dict[str, Any] = {"required_until_ms": required_until_ms}
    try:
        payload = _read(path, symbol, "funding", venue)
        if payload:
            coverage_start = _integer(payload["coverage_start_ms"], "funding coverage start")
            coverage_end = _integer(payload["coverage_end_ms"], "funding coverage end")
            if coverage_end < coverage_start:
                raise ValueError("Invalid funding coverage")
        for raw in payload.get("rates", []):
            timestamp = _integer(raw[0], "funding timestamp")
            rate = float(raw[1])
            if not math.isfinite(rate) or timestamp in rows or not coverage_start <= timestamp <= coverage_end:
                raise ValueError("Invalid cached funding rate")
            rows[timestamp] = rate
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        diagnostics["cache_error"] = str(exc)
        rows, coverage_start, coverage_end = {}, None, -1

    # Historical replay may precede the cache watermark. Such rows are filtered
    # by asof below; no future settlement is ever passed to the strategy.
    spans = []
    if coverage_start is None:
        spans.append((start_ms, asof_ms))
    else:
        if start_ms < coverage_start:
            spans.append((start_ms, coverage_start - 1))
        if asof_ms > coverage_end:
            # Binance can publish a settlement after its fundingTime. A query
            # that was empty just after a boundary must not permanently skip
            # the later-published event. Recheck a day of prior coverage only
            # when the decision cutoff advances; an identical cutoff remains
            # a zero-request cache read.
            spans.append((max(start_ms, coverage_start,
                              coverage_end - FUNDING_OVERLAP_MS), asof_ms))
    requests_made = 0
    error = None
    try:
        for span_start, span_end in spans:
            staged = {}
            cursor = span_start
            while cursor <= span_end:
                requests_made += 1
                batch = client.public_get("/fapi/v1/fundingRate", {
                    "symbol": symbol, "startTime": cursor, "endTime": span_end, "limit": 1000,
                })
                if not isinstance(batch, list):
                    raise ValueError("Invalid funding response")
                previous = cursor - 1
                for raw in batch:
                    timestamp = _integer(raw["fundingTime"], "funding timestamp")
                    rate = float(raw["fundingRate"])
                    if (timestamp <= previous or not cursor <= timestamp <= span_end
                            or not math.isfinite(rate) or raw.get("symbol", symbol) != symbol):
                        raise ValueError("Invalid funding response row")
                    staged[timestamp] = rate
                    previous = timestamp
                if not batch or len(batch) < 1000:
                    break
                cursor = previous + 1
            # A partial failed page must never advance the coverage watermark.
            rows.update(staged)
            coverage_start = min(coverage_start, span_start) if coverage_start is not None else span_start
            coverage_end = max(coverage_end, span_end)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if requests_made and coverage_start is not None:
        _write(path, symbol, "funding", venue, {"coverage_start_ms": coverage_start,
            "coverage_end_ms": coverage_end, "rates": [[t, rate] for t, rate in sorted(rows.items())]})
    complete = coverage_start is not None and coverage_start <= start_ms and coverage_end >= required_until_ms
    diagnostics.update({"source": "cache_fallback" if error else "live" if requests_made else "cache",
                        "requests": requests_made, "network_error": error,
                        "coverage_start_ms": coverage_start, "coverage_end_ms": coverage_end,
                        "complete": complete})
    if not complete:
        raise MarketDataUnavailable("Funding history coverage is incomplete or stale", diagnostics)
    timestamps = sorted(t for t in rows if start_ms <= t <= asof_ms)
    return sim.FundingHistory(timestamps, [rows[t] for t in timestamps]), diagnostics


def load_market_data(symbol: str, start_ms: int, asof_ms: int, *, client: Any = None,
                     cache_dir: Path | None = None,
                     intervals: tuple[str, ...] = ("5m", "1h")) -> MarketDataResult:
    """Fetch missing history, or reuse complete cached input at the same cutoff.

    ``start_ms`` includes the strategy warmup. The first candle opens at the
    first interval boundary at or after start, matching the frozen downloader.
    Cache failures and transport failures are reported independently from a
    neutral strategy decision; incomplete input always raises explicitly.
    """
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{3,30}", symbol):
        raise ValueError("Invalid market symbol")
    start_ms, asof_ms = _integer(start_ms, "start timestamp"), _integer(asof_ms, "asof timestamp")
    if start_ms >= asof_ms or not intervals or len(set(intervals)) != len(intervals):
        raise ValueError("Invalid market history range or intervals")
    for interval in intervals:
        if not re.fullmatch(r"[1-9][0-9]*[mhdw]", interval):
            raise ValueError("Invalid market interval")
        _required_bounds(start_ms, asof_ms, sim.interval_to_ms(interval))
    if client is None:
        from binance_terminal_client import BinanceTerminalClient
        client = BinanceTerminalClient()
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    candles = {}
    openings = {}
    diagnostics: dict[str, Any] = {"asof_ms": asof_ms, "start_ms": start_ms,
                                   "cache_dir": str(cache_dir), "intervals": {}}
    try:
        for interval in intervals:
            try:
                candles[interval], openings[interval], diagnostics["intervals"][interval] = _load_interval(
                    client, symbol, interval, start_ms, asof_ms, cache_dir)
            except MarketDataUnavailable as exc:
                diagnostics["intervals"][interval] = exc.diagnostics
                raise
        required_until = max(bars[-1].close_time_ms for bars in candles.values())
        try:
            funding, diagnostics["funding"] = _load_funding(client, symbol, start_ms,
                asof_ms, required_until, cache_dir)
        except MarketDataUnavailable as exc:
            diagnostics["funding"] = exc.diagnostics
            raise
    except MarketDataUnavailable as exc:
        raise MarketDataUnavailable(str(exc), diagnostics) from exc
    diagnostics["complete"] = True
    diagnostics["degraded"] = any(item.get("network_error") or item.get("cache_error")
        for item in [*diagnostics["intervals"].values(), diagnostics["funding"]])
    return MarketDataResult(candles, funding, openings, diagnostics, asof_ms)
