"""Read-only BTC market observations. Scores are hypotheses, never order instructions."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

VERSION = "market-intelligence-v1"
FIVE_MIN = 300_000


def number(value, minimum=None):
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError("Invalid market number")
    return result


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def imbalance(buy, sell):
    return (buy - sell) / (buy + sell) if buy + sell > 0 else None


def closed_flow(rows, received_ms):
    # A mutable current candle is never allowed into the direction signal.
    rows = sorted({int(r[0]): r for r in rows if int(r[6]) < received_ms}.values(), key=lambda r: int(r[0]))
    if len(rows) < 12:
        raise ValueError("Need twelve completed five-minute bars")
    selected = rows[-12:]
    if any(int(b[0]) - int(a[0]) != FIVE_MIN for a, b in zip(selected, selected[1:])):
        raise ValueError("Gapped flow window")
    if received_ms - int(selected[-1][6]) > 2 * FIVE_MIN:
        raise ValueError("Stale flow window")
    if any(number(r[10], 0) > number(r[7], 0) * (1 + 1e-8) for r in selected):
        raise ValueError("Taker buy volume exceeds candle volume")
    quote = sum(number(r[7], 0) for r in selected)
    buy = sum(number(r[10], 0) for r in selected)
    if buy > quote * (1 + 1e-8):
        raise ValueError("Taker buy volume exceeds total volume")
    return {"buy_quote": buy, "sell_quote": max(0, quote - buy), "imbalance_1h": imbalance(buy, max(0, quote - buy)),
            "price_change_1h_pct": (number(selected[-1][4], 1e-12) / number(selected[0][1], 1e-12) - 1) * 100,
            "close": number(selected[-1][4], 1e-12), "observed_ms": int(selected[-1][6]), "bars": 12}


def oi_features(rows, received_ms):
    # Conservative extra period lag also excludes an in-progress statistics bucket.
    rows = sorted({int(r["timestamp"]): r for r in rows
                   if int(r["timestamp"]) + FIVE_MIN <= received_ms}.values(), key=lambda r: int(r["timestamp"]))
    if len(rows) < 13:
        raise ValueError("Insufficient OI history")
    selected = rows[-13:]
    if any(int(b["timestamp"]) - int(a["timestamp"]) != FIVE_MIN for a, b in zip(selected, selected[1:])):
        raise ValueError("Gapped OI history")
    if received_ms - int(selected[-1]["timestamp"]) > 3 * FIVE_MIN:
        raise ValueError("Stale OI")
    first, last = selected[0], selected[-1]
    quantity = number(last["sumOpenInterest"], 0)
    return {"oi_btc": quantity, "oi_usd": number(last["sumOpenInterestValue"], 0),
            "change_1h_pct": (quantity / number(first["sumOpenInterest"], 1e-12) - 1) * 100,
            "observed_ms": int(last["timestamp"])}


def ratio_features(rows, received_ms):
    rows = [r for r in rows if int(r["timestamp"]) + FIVE_MIN <= received_ms]
    if not rows:
        raise ValueError("No completed ratio bucket")
    row = max(rows, key=lambda r: int(r["timestamp"]))
    if received_ms - int(row["timestamp"]) > 3 * FIVE_MIN:
        raise ValueError("Stale ratio")
    return {"long_short_ratio": number(row["longShortRatio"], 0),
            "observed_ms": int(row["timestamp"])}


def tape_features(rows, received_ms, venue, threshold=250_000):
    clean = {}
    for r in rows:
        if venue == "hyperliquid":
            ts = int(r["time"])
            key = f"{ts}:{r['tid']}"
            if r["side"] not in ("B", "A"):
                raise ValueError("Unknown aggressor side")
            side = "buy" if r["side"] == "B" else "sell"
            price, qty = number(r["px"], 1e-12), number(r["sz"], 0)
        else:
            ts, key = int(r["T"]), str(r["a"])
            if type(r["m"]) is not bool:
                raise ValueError("Maker side must be boolean")
            side = "sell" if r["m"] else "buy"
            price, qty = number(r["p"], 1e-12), number(r["q"], 0)
        if ts > received_ms + 5000:
            raise ValueError("Future trade timestamp")
        clean[key] = {"id": key, "time_ms": ts, "side": side, "price": price,
                      "qty_btc": qty, "notional": price * qty}
    tape = sorted(clean.values(), key=lambda r: r["time_ms"])
    if not tape or received_ms - tape[-1]["time_ms"] > FIVE_MIN:
        raise ValueError("Empty or stale trade tape")
    buy = sum(r["notional"] for r in tape if r["side"] == "buy")
    sell = sum(r["notional"] for r in tape if r["side"] == "sell")
    large = [r for r in tape if r["notional"] >= threshold]
    return {"trades": len(tape), "sample_start_ms": tape[0]["time_ms"], "observed_ms": tape[-1]["time_ms"],
            "sample_span_seconds": (tape[-1]["time_ms"] - tape[0]["time_ms"]) / 1000,
            "sample_only": True, "imbalance": imbalance(buy, sell),
            "buy_notional": buy, "sell_notional": sell,
            "large_trade_threshold": threshold, "large_trades": sorted(large, key=lambda r: -r["notional"])[:20],
            "large_buy_notional": sum(r["notional"] for r in large if r["side"] == "buy"),
            "large_sell_notional": sum(r["notional"] for r in large if r["side"] == "sell"),
            "min_id": min(int(k) for k in clean) if venue != "hyperliquid" else None,
            "max_id": max(int(k) for k in clean) if venue != "hyperliquid" else None}


def book_features(data, received_ms):
    bids = [(number(p, 1e-12), number(q, 0)) for p, q in data["bids"]]
    asks = [(number(p, 1e-12), number(q, 0)) for p, q in data["asks"]]
    if not bids or not asks or bids[0][0] >= asks[0][0]:
        raise ValueError("Invalid or crossed order book")
    mid = (bids[0][0] + asks[0][0]) / 2
    bid = sum(p * q for p, q in bids if p >= mid * .999)
    ask = sum(p * q for p, q in asks if p <= mid * 1.001)
    timestamp = int(data.get("E", received_ms))
    if abs(received_ms - timestamp) > FIVE_MIN:
        raise ValueError("Stale book")
    return {"spread_bps": (asks[0][0] - bids[0][0]) / mid * 10000,
            "bid_10bps_notional": bid, "ask_10bps_notional": ask,
            "imbalance": imbalance(bid, ask), "observed_ms": timestamp,
            "full_10bps_band": bids[-1][0] <= mid * .999 and asks[-1][0] >= mid * 1.001,
            "warning": "Displayed limit orders may be cancelled; this is not executed demand."}


def funding_features(data, received_ms):
    mark, index = number(data["markPrice"], 1e-12), number(data["indexPrice"], 1e-12)
    timestamp = int(data["time"])
    if abs(received_ms - timestamp) > FIVE_MIN:
        raise ValueError("Stale funding context")
    return {"mark_price": mark, "index_price": index,
            "last_funding_rate": number(data["lastFundingRate"]),
            "basis_bps": (mark / index - 1) * 10000,
            "next_funding_ms": int(data["nextFundingTime"]), "observed_ms": timestamp}


def hyper_context(data):
    universe, contexts = data[0]["universe"], data[1]
    index = next(i for i, item in enumerate(universe) if item["name"] == "BTC")
    row = contexts[index]
    mark = number(row["markPx"], 1e-12)
    return {"oi_btc": number(row["openInterest"], 0), "oi_usd": number(row["openInterest"], 0) * mark,
            "funding_rate_1h": number(row["funding"]), "mark_price": mark,
            "premium": number(row["premium"]) if row.get("premium") is not None else None,
            "volume_24h_usd": number(row["dayNtlVlm"], 0)}


def position_sample(records):
    if not records or not any(r["status"] == "ok" for r in records):
        raise ValueError("No available position observations")
    positions = []
    for record in records:
        if record["status"] != "ok":
            continue
        for item in record["data"]["assetPositions"]:
            p = item["position"]
            if p["coin"] != "BTC":
                continue
            size = number(p["szi"])
            value = number(p["positionValue"], 0)
            positions.append({"address": record["address"], "size_btc": size, "notional_usd": value,
                              "side": "long" if size > 0 else "short" if size < 0 else "flat",
                              "liquidation_price": number(p["liquidationPx"], 0) if p.get("liquidationPx") else None})
    long = sum(p["notional_usd"] for p in positions if p["size_btc"] > 0)
    short = sum(p["notional_usd"] for p in positions if p["size_btc"] < 0)
    return {"selection": "Up to six participants in largest recent sampled BTC trades; not market-wide or representative.",
            "partial": any(r["status"] != "ok" for r in records),
            "queried_addresses": len(records), "successful_addresses": sum(r["status"] == "ok" for r in records),
            "long_usd": long, "short_usd": short, "net_usd": long - short,
            "long_short_ratio": long / short if short > 0 else None,
            "positions": positions, "sample_only": True}


def chain_features(rows, threshold_btc=100):
    # `value` is total transaction outputs, including change, not economic transfer.
    unique = {r["txid"]: r for r in rows}
    large = [{"txid": r["txid"], "output_btc": number(r["value"], 0) / 1e8}
             for r in unique.values() if number(r["value"], 0) / 1e8 >= threshold_btc]
    return {"sample_size": len(unique), "large_output_transactions": large,
            "threshold_btc": threshold_btc, "direction": None, "sample_only": True,
            "warning": "Recent mempool sample only; unconfirmed, includes change and self-transfers; no exchange labels."}


def build_report(records, now_ms, previous=None):
    features, health = {}, {}
    parsers = {
        "spot_flow": closed_flow, "futures_flow": closed_flow, "open_interest": oi_features,
        "global_accounts": ratio_features, "top_accounts": ratio_features, "top_positions": ratio_features,
        "spot_book": book_features, "futures_book": book_features, "funding_basis": funding_features,
        "spot_tape": lambda d, t: tape_features(d, t, "spot"),
        "futures_tape": lambda d, t: tape_features(d, t, "futures"),
        "hyper_tape": lambda d, t: tape_features(d, t, "hyperliquid"),
        "hyper_context": lambda d, t: hyper_context(d), "bitcoin_chain": lambda d, t: chain_features(d),
        "hyper_positions": lambda d, t: position_sample(d),
    }
    for name, record in records.items():
        state = {"status": record["status"], "available_at_ms": record["received_ms"],
                 "source": record.get("url"), "error": record.get("error")}
        health[name] = state
        if record["status"] != "ok":
            continue
        try:
            if now_ms - record["received_ms"] > FIVE_MIN or record["received_ms"] > now_ms:
                raise ValueError("Stale or future receipt")
            features[name] = parsers[name](record["data"], record["received_ms"])
        except (KeyError, ValueError, TypeError, IndexError, StopIteration) as exc:
            state.update(status="invalid", error=type(exc).__name__)
    if previous is not None:
        for name in ("spot_tape", "futures_tape"):
            new, old = features.get(name), previous.get("features", {}).get(name)
            if new and old:
                new["gap_since_previous_poll"] = new["min_id"] > old["max_id"] + 1
    required = ("spot_flow", "futures_flow", "open_interest")
    contributions = {}
    for name in ("spot_flow", "futures_flow"):
        if name in features and features[name]["imbalance_1h"] is not None:
            contributions[name] = max(-1, min(1, features[name]["imbalance_1h"] / .15))
    if "open_interest" in features and "futures_flow" in features:
        change = features["open_interest"]["change_1h_pct"]
        price = features["futures_flow"]["price_change_1h_pct"]
        # OI measures outstanding contracts, not net bullish money. Rising OI
        # can only confirm a direction already present in price, weakly.
        contributions["open_interest"] = max(-1, min(1, price / .5)) * max(0, min(1, change / 2))
    weights = {"spot_flow": .45, "futures_flow": .35, "open_interest": .20}
    enough = all(name in contributions for name in required)
    direction = 100 * sum(weights[k] * v for k, v in contributions.items()) if enough else None
    disagreement = (contributions.get("spot_flow", 0) * contributions.get("futures_flow", 0) < 0)
    warnings = []
    if disagreement:
        warnings.append("spot_futures_flow_disagree")
    if "funding_basis" in features and abs(features["funding_basis"]["last_funding_rate"]) > .001:
        warnings.append("extreme_last_funding")
    for name in ("global_accounts", "top_positions"):
        if name in features:
            ratio = features[name]["long_short_ratio"]
            if ratio > 3 or ratio < 1 / 3:
                warnings.append(name + "_crowded")
    if features.get("open_interest", {}).get("change_1h_pct", 0) < -3:
        warnings.append("rapid_deleveraging")
    # No full-market on-chain long/short or exchange-netflow number is invented.
    return {"schema_version": 1, "rule_version": VERSION, "generated_at_ms": now_ms,
            "generated_at_utc": iso(now_ms), "mode": "observation_only", "places_orders": False,
            "features": features, "health": health,
            "direction": {"score": direction, "scale": "-100 to +100; uncalibrated, not a probability",
                          "state": "insufficient_data" if not enough else "conflicting" if disagreement else
                                   "buy_pressure" if direction > 20 else "sell_pressure" if direction < -20 else "neutral",
                          "contributions": contributions},
            "risk_flags": warnings,
            "unavailable": {"exchange_netflow": "No verified exchange address labels or entity-adjusted provider configured.",
                            "market_wide_onchain_long_short": "Sampled participant positions do not represent all wallets.",
                            "liquidation_tape": "Not collected; falling OI alone does not identify liquidations.",
                            "etf_flows": "No timestamped daily source configured."}}


def open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=20)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS payloads (digest TEXT PRIMARY KEY, content BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY, source TEXT NOT NULL,
          received_ms INTEGER NOT NULL, status TEXT NOT NULL, digest TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY, received_ms INTEGER NOT NULL,
          rule_version TEXT NOT NULL, content TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS reports_time ON reports(received_ms);
        CREATE INDEX IF NOT EXISTS readings_time ON readings(received_ms);
    """)
    return db


def persist(db, records, report):
    with db:
        for source, row in records.items():
            raw = json.dumps(row.get("data"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            digest = hashlib.sha256(raw).hexdigest()
            db.execute("INSERT OR IGNORE INTO payloads VALUES (?,?)", (digest, zlib.compress(raw)))
            db.execute("INSERT INTO readings(source, received_ms, status, digest, error) VALUES (?,?,?,?,?)",
                       (source, row["received_ms"], row["status"], digest, row.get("error")))
        db.execute("INSERT INTO reports(received_ms,rule_version,content) VALUES (?,?,?)",
                   (report["generated_at_ms"], VERSION, json.dumps(report, ensure_ascii=False, allow_nan=False)))


def at_time(db, timestamp_ms, max_age_ms=FIVE_MIN):
    row = db.execute("SELECT received_ms,content FROM reports WHERE received_ms<=? ORDER BY received_ms DESC,id DESC LIMIT 1",
                     (timestamp_ms,)).fetchone()
    if row is None or timestamp_ms - row[0] > max_age_ms:
        return None
    return json.loads(row[1])


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def entry_multiplier(report, side):
    """Fixed research hypothesis; missing/conflicting evidence has no action."""
    if side not in ("long", "short"):
        raise ValueError("Unknown position side")
    if report is None or report.get("rule_version") != VERSION:
        return None
    direction = report.get("direction", {})
    score = direction.get("score")
    if score is None:
        return None
    sign = 1 if side == "long" else -1
    features = report.get("features", {})
    flows = [features.get(name, {}).get("imbalance_1h") for name in ("spot_flow", "futures_flow")]
    if any(value is None for value in flows):
        return None
    return .5 if score * sign <= -35 and all(value * sign <= -.05 for value in flows) else 1.0


def apply_shadow_overlay(sleeves, database_path):
    # A read-only connection must not silently create an empty database.
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    adjusted, decisions, covered, throttled = [], 0, 0, 0
    with closing(sqlite3.connect(uri, uri=True)) as db:
        for sleeve in sleeves:
            from execution_ledger import attach_ledger
            sleeve = attach_ledger(sleeve)
            trades = []
            for raw in sleeve.get("trades", []):
                parsed = datetime.fromisoformat(raw["entry_time_utc"].replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("Entry timestamp requires timezone")
                observation = at_time(db, int(parsed.timestamp() * 1000))
                multiplier = entry_multiplier(observation, raw["side"])
                decisions += 1
                covered += multiplier is not None
                trade = dict(raw)
                # Absence of archived evidence cannot retrospectively block a trade.
                prior = min(float(raw.get("macro_world_risk_multiplier", raw.get("macro_risk_multiplier", 1.0))),
                            float(raw.get("multifactor_risk_multiplier", 1.0)))
                if not math.isfinite(prior) or not 0 < prior <= 1:
                    raise ValueError("Invalid upstream multiplier")
                final = min(prior, multiplier) if multiplier is not None else prior
                ratio = final / prior
                if ratio < 1:
                    throttled += 1
                    for field in ("initial_qty", "pnl", "fees", "net_pnl", "funding_pnl", "slippage_cost"):
                        if field in trade:
                            trade[field] = float(trade[field]) * ratio
                trade["intelligence_covered"] = multiplier is not None
                trade["intelligence_multiplier"] = multiplier
                trade["combined_information_multiplier"] = final
                trades.append(trade)
            adjusted.append({**sleeve, "trades": trades})
    return adjusted, {"rule_version": VERSION, "research_only": True,
                      "decisions": decisions, "covered": covered, "throttled": throttled,
                      "coverage_pct": covered / decisions * 100 if decisions else None,
                      "profitability_validated": False}
