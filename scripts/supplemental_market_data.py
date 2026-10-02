"""Public options, depth, exchange flows, ETF flows and optional consensus archive.

These observations are visible in the simulation terminal. They are not fitted
directional signals. Receipt times and revisions are retained for future tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

from binance_terminal_client import BinanceTerminalClient
from trading_execution import read_json, write_json

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data/snapshots/supplemental_market_latest.json"
ETF_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
FOREX_CALENDAR_URL = "https://www.forexfactory.com/calendar"
TTL = {"options": 300, "orderbook": 300, "exchange_flows": 3600,
       "etf_flows": 21600, "economic_consensus": 3600}
MAX_AGE = {"options": 900, "orderbook": 900, "exchange_flows": 4 * 86400,
           "etf_flows": 5 * 86400, "economic_consensus": 86400}


def now_ms():
    return int(datetime.now(UTC).timestamp() * 1000)


def utc_ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def get(url, params=None):
    response = requests.get(url, params=params, timeout=20,
                            headers={"User-Agent": "btc-auto-public-data/1.0"})
    # Avoid putting credential-bearing URLs into logs or terminal status.
    if not response.ok:
        raise ValueError(f"Provider HTTP {response.status_code}")
    return response


def parse_options(rows, timestamp):
    legs = []
    for row in rows:
        try:
            _, date, strike, kind = row["instrument_name"].split("-")
            expiry = datetime.strptime(date, "%d%b%y").replace(hour=8, tzinfo=UTC)
            days = (expiry.timestamp() * 1000 - timestamp) / 86400000
            iv, forward, strike = float(row["mark_iv"]), float(row["underlying_price"]), float(strike)
            bid, ask = float(row["bid_price"] or 0), float(row["ask_price"] or 0)
            if not (7 <= days <= 60 and 0 < iv < 1000 and forward > 0 and strike > 0 and 0 < bid <= ask):
                continue
            sigma_t = iv / 100 * math.sqrt(days / 365)
            d1 = math.log(forward / strike) / sigma_t + sigma_t / 2
            delta = .5 * (1 + math.erf(d1 / math.sqrt(2))) - (1 if kind == "P" else 0)
            legs.append({"instrument": row["instrument_name"], "expiry": expiry.isoformat(),
                         "days": days, "iv": iv, "model_delta": delta,
                         "moneyness": abs(math.log(strike / forward)), "kind": kind})
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
    if not legs:
        raise ValueError("No usable two-sided options quotes")
    expiry = min(legs, key=lambda r: abs(r["days"] - 30))["expiry"]
    chain = [r for r in legs if r["expiry"] == expiry]
    puts, calls = [r for r in chain if r["kind"] == "P"], [r for r in chain if r["kind"] == "C"]
    if not puts or not calls:
        raise ValueError("Missing put/call quotes at chosen expiry")
    put = min(puts, key=lambda r: abs(r["model_delta"] + .25))
    call = min(calls, key=lambda r: abs(r["model_delta"] - .25))
    atm = min(chain, key=lambda r: r["moneyness"])
    return {"observed_at_ms": timestamp, "atm_iv_pct": atm["iv"],
            "approx_25delta_put_minus_call_iv_pp": put["iv"] - call["iv"],
            "selected_legs": [atm, put, call], "source": "Deribit public API",
            "method": "Nearest 30d expiry; approximate forward Black-Scholes delta, not interpolated 30d/25delta"}


def options():
    url = "https://www.deribit.com/api/v2/public/"
    rows = get(url + "get_book_summary_by_currency", {"currency": "BTC", "kind": "option"}).json()["result"]
    timestamp = now_ms()
    value = parse_options(rows, timestamp)
    dvol = get(url + "get_volatility_index_data", {"currency": "BTC", "resolution": "3600",
               "start_timestamp": timestamp - 86400000, "end_timestamp": timestamp}).json()["result"]["data"]
    closed = [r for r in dvol if int(r[0]) + 3600000 <= timestamp]
    if closed:
        value["dvol_pct"] = float(closed[-1][4])
        value["dvol_closed_at_ms"] = int(closed[-1][0]) + 3600000
    return value


def parse_depth(book, received):
    bids = [(float(p), float(q)) for p, q in book["bids"]]
    asks = [(float(p), float(q)) for p, q in book["asks"]]
    if not bids or not asks or any(not math.isfinite(p + q) or p <= 0 or q < 0 for p, q in bids + asks):
        raise ValueError("Invalid orderbook")
    if bids[0][0] >= asks[0][0]:
        raise ValueError("Crossed orderbook")
    mid = (bids[0][0] + asks[0][0]) / 2
    bid = sum(p * q for p, q in bids if p >= mid * .999)
    ask = sum(p * q for p, q in asks if p <= mid * 1.001)
    observed = int(book.get("T") or book.get("E") or received)
    if abs(received - observed) > 60000:
        raise ValueError("Orderbook timestamp outside freshness limit")
    return {"observed_at_ms": observed, "spread_bps": (asks[0][0] - bids[0][0]) / mid * 10000,
            "bid_depth_10bps_usdt": bid, "ask_depth_10bps_usdt": ask,
            "depth_imbalance": (bid - ask) / (bid + ask) if bid + ask else 0,
            "last_update_id": book["lastUpdateId"], "source": "Binance USD-M depth REST, top 100 levels",
            "limitation": "Sampled depth, not a full streaming book; 10bps depth can be truncated by 100-level limit"}


def orderbook():
    book = BinanceTerminalClient().public_get("/fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 100})
    return parse_depth(book, now_ms())


def parse_flows(rows):
    output = []
    for row in rows:
        if row.get("FlowInExUSD") is None or row.get("FlowOutExUSD") is None:
            continue
        incoming, outgoing = float(row["FlowInExUSD"]), float(row["FlowOutExUSD"])
        if not all(math.isfinite(v) and v >= 0 for v in (incoming, outgoing)):
            continue
        output.append({"observed_at_ms": utc_ms(row["time"]), "inflow_usd": incoming,
                       "outflow_usd": outgoing, "net_inflow_usd": incoming - outgoing,
                       "inflow_status": row.get("FlowInExUSD-status"),
                       "outflow_status": row.get("FlowOutExUSD-status")})
    if not output:
        raise ValueError("No exchange flow observations")
    output.sort(key=lambda r: r["observed_at_ms"])
    return {**output[-1], "daily_rows": output, "source": "Coin Metrics Community API",
            "limitation": "Provider-attributed exchange addresses; flash observations may be revised"}


def exchange_flows():
    start = (datetime.now(UTC) - timedelta(days=60)).date().isoformat()
    response = get("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
                   {"assets": "btc", "metrics": "FlowInExUSD,FlowOutExUSD", "frequency": "1d",
                    "page_size": 100, "start_time": start}).json()
    return parse_flows(response["data"])


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.cells = []; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.cells = []
        if tag in {"td", "th"}: self.cell = ""
    def handle_data(self, data):
        if self.cell is not None: self.cell += data
    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.cells.append(self.cell.strip()); self.cell = None
        if tag == "tr" and self.cells: self.rows.append(self.cells)


def parse_etf(rows):
    daily = []
    for row in rows:
        try:
            timestamp = int(datetime.strptime(row[0], "%d %b %Y").replace(tzinfo=UTC).timestamp() * 1000)
        except (ValueError, IndexError):
            continue
        if len(row) < 12 or row[-1] in {"", "-", "—"}:
            continue
        value = row[-1].replace(",", "")
        total = -float(value[1:-1]) if value.startswith("(") else float(value)
        if not math.isfinite(total): continue
        daily.append({"observed_at_ms": timestamp, "net_flow_usd": total * 1000000,
                      "complete": all(c not in {"", "-", "—"} for c in row[1:-1]),
                      "fund_cells": row[1:-1]})
    daily.sort(key=lambda r: r["observed_at_ms"])
    if not daily: raise ValueError("No ETF flow table")
    complete = [r for r in daily if r["complete"]]
    return {**daily[-1], "daily_rows": daily, "latest_complete": complete[-1] if complete else None,
            "source": ETF_URL, "limitation": "Dash cells remain missing; partial totals are not final totals"}


def etf_flows():
    parser = TableParser(); parser.feed(get(ETF_URL).text)
    return parse_etf(parser.rows)


def parse_forex_calendar(body):
    match = re.search(r"(?m)^days:\s*", body)
    if not match:
        raise ValueError("Forex Factory calendar schema not found")
    days, _ = json.JSONDecoder().raw_decode(body[match.end():])
    records = []
    for day in days:
        for row in day.get("events", []):
            if row.get("currency") != "USD" or not row.get("forecast") or row.get("timeMasked"):
                continue
            timestamp = int(row["dateline"]) * 1000
            records.append({"CalendarId": str(row["id"]), "Date": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
                            "Event": row["name"], "Actual": row.get("actual") or None,
                            "Forecast": row["forecast"], "Previous": row.get("previous"),
                            "impact": row.get("impactName"), "scheduled_at_ms": timestamp,
                            "source_url": FOREX_CALENDAR_URL + "#detail=" + str(row["id"])})
    if not records:
        raise ValueError("No USD forecast records in calendar")
    return {"observed_at_ms": now_ms(), "records": records, "source": "Forex Factory published forecasts and actuals",
            "limitation": "Publisher forecast, not an independently verified consensus survey; approximate schedule; hourly sampling. Surprise requires a forecast archived before release."}


def economic_consensus():
    key = os.getenv("TRADING_ECONOMICS_API_KEY", "").strip()
    if not key:
        return parse_forex_calendar(get(FOREX_CALENDAR_URL).text)
    rows = get("https://api.tradingeconomics.com/calendar/country/united%20states",
               {"c": key, "f": "json"}).json()
    if not isinstance(rows, list) or not rows:
        raise ValueError("No verified calendar response")
    # Archive actual provider consensus, not the provider's own TEForecast.
    forecasts = [{k: r.get(k) for k in ("CalendarId", "Date", "Event", "Actual", "Forecast", "Previous", "Unit")}
                 for r in rows if r.get("Forecast") not in {None, ""}]
    if not forecasts: raise ValueError("Calendar response has no consensus Forecast")
    return {"observed_at_ms": now_ms(), "records": forecasts, "source": "Trading Economics licensed API",
            "limitation": "Forecast must be archived before release before computing a point-in-time surprise"}


def numeric_value(value):
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([%KMB]?)", str(value).replace(",", "").strip())
    if not match: return None
    amount, suffix = float(match[1]), match[2]
    return amount * {"": 1, "%": 1, "K": 1000, "M": 1000000, "B": 1000000000}[suffix], (
        "percentage_points" if suffix == "%" else "numeric_units")


def economic_surprises(samples, timestamp):
    # Use the last forecast received BEFORE the release, and the first actual
    # received AFTER release. Never silently replace it with a revised actual.
    forecasts, surprises = {}, {}
    for sample in samples:
        received, value = sample["received_at_ms"], sample["value"]
        if received > timestamp: continue
        for row in value.get("records", []):
            scheduled = utc_ms(row["Date"])
            identity = (value["source"], row["CalendarId"], scheduled)
            if received < scheduled and row.get("Actual") in {None, ""}:
                forecasts[identity] = (row, received)
            elif received >= scheduled and identity in forecasts and identity not in surprises:
                actual, forecast = numeric_value(row.get("Actual")), numeric_value(forecasts[identity][0]["Forecast"])
                if actual and forecast and actual[1] == forecast[1]:
                    surprises[identity] = {"event": row["Event"], "calendar_id": row["CalendarId"],
                        "scheduled_at_ms": scheduled, "forecast_received_at_ms": forecasts[identity][1],
                        "actual_received_at_ms": received, "actual": actual[0], "forecast": forecast[0],
                        "surprise": actual[0] - forecast[0], "unit": actual[1], "source": value["source"]}
    return list(surprises.values())


def source_view(payload, timestamp=None):
    timestamp = timestamp or now_ms()
    views = {}
    for name in TTL:
        samples = [s for s in payload.get("samples", {}).get(name, []) if s["received_at_ms"] <= timestamp]
        sample = samples[-1] if samples else None
        observed = sample["value"]["observed_at_ms"] if sample else None
        stale = sample is None or timestamp - observed > MAX_AGE[name] * 1000
        views[name] = {**payload.get("status", {}).get(name, {}), "stale": stale,
                       "age_seconds": max(0, (timestamp - observed) / 1000) if observed else None,
                       "received_at_ms": sample["received_at_ms"] if sample else None,
                       "value": sample["value"] if sample else None, "usage": "observation_only"}
        if name == "economic_consensus":
            views[name]["surprises"] = economic_surprises(samples, timestamp)
    return views


def collect(output=DEFAULT_PATH, force=False):
    previous = read_json(output, {}) or {}
    samples, status = previous.get("samples", {}), previous.get("status", {})
    functions = {"options": options, "orderbook": orderbook, "exchange_flows": exchange_flows,
                 "etf_flows": etf_flows, "economic_consensus": economic_consensus}
    due = [n for n in TTL if force or now_ms() - status.get(n, {}).get("checked_at_ms", 0) >= TTL[n] * 1000]
    def run(name):
        try: return name, functions[name](), None
        except Exception as exc:
            # Do not expose request URLs/API credentials from exception strings.
            message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            return name, None, message[:200]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, value, error in pool.map(run, due):
            received = now_ms()
            status[name] = {"ok": error is None, "error": error, "checked_at_ms": received}
            if value is not None:
                samples.setdefault(name, []).append({"received_at_ms": received, "value": value})
            print(f"{name}: {error or 'ok'}", flush=True)
    payload = {"schema_version": 1, "generated_at_utc": datetime.now(UTC).isoformat(),
               "samples": samples, "status": status, "refresh_seconds": TTL,
               "usage": "observation_only_no_directional_weights"}
    write_json(output, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); collect(args.output, args.force)
