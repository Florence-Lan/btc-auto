"""Side-aware research overlay. Frozen price-signal engine remains unchanged."""
from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import event_risk

DAY = 86_400_000
HOUR = 3_600_000
GROUPS = {"btc_momentum", "positioning", "fed", "treasury", "fx", "global_risk"}


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("live_orders_allowed") is not False:
        raise ValueError("Multifactor candidate must be simulation-only")
    weights = profile["groups"]
    if not weights or set(weights) - GROUPS or any(
        not math.isfinite(v) or v <= 0 for v in weights.values()
    ):
        raise ValueError("Invalid factor weights")
    for key in ("minimum_group_coverage", "minimum_risk_multiplier", "block_stress_at"):
        if not 0 < profile[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if not -1 <= profile["block_alignment_below"] <= 0:
        raise ValueError("Invalid alignment threshold")
    if profile["availability_mode"] not in {"first_seen", "reconstructed"}:
        raise ValueError("Invalid availability mode")
    return profile


def profile_hash(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


class Snapshot:
    """Rows: observation time, estimated availability, value, first collection time.

    first_seen prevents backfilled/revised values entering past forward decisions.
    reconstructed is explicitly retrospective research, never an OOS claim.
    """

    def __init__(self, payload: Mapping[str, Any], mode: str = "first_seen"):
        if payload.get("schema_version") != 1 or mode not in {"first_seen", "reconstructed"}:
            raise ValueError("Invalid multifactor snapshot schema or mode")
        self.metadata = payload.get("metadata", {})
        self.mode = mode
        self.series = {}
        for name, raw in payload.get("series", {}).items():
            rows = []
            seen = set()
            for row in raw:
                if len(row) != 4:
                    raise ValueError(f"Invalid observation: {name}")
                observed, available, value, first_seen = row
                if (not all(math.isfinite(float(v)) for v in row)
                        or available < observed or first_seen < observed or observed in seen):
                    raise ValueError(f"Invalid/duplicate observation: {name}")
                seen.add(observed)
                usable = max(available, first_seen) if mode == "first_seen" else available
                rows.append((int(usable), int(observed), float(value)))
            rows.sort()
            self.series[name] = (tuple(r[0] for r in rows), tuple(rows))

    def window(self, name: str, timestamp: int, age: int, lookback: int = 0):
        times, rows = self.series.get(name, ((), ()))
        index = bisect.bisect_right(times, timestamp) - 1
        if index < 0:
            return None
        # Normally append-only observations; select newest observed among available vintages.
        available_rows = rows[:index + 1]
        current = max(available_rows, key=lambda row: row[1])
        if timestamp - current[1] > age:
            return None
        if not lookback:
            return current[2]
        target = current[1] - lookback
        earlier = [row for row in available_rows if row[1] <= target]
        if not earlier:
            return None
        previous = max(earlier, key=lambda row: row[1])
        # Weekly Fed balance sheets need a full weekly tolerance.
        if target - previous[1] > min(age, 8 * DAY):
            return None
        return current[2], previous[2]

    def change(self, name: str, timestamp: int, age: int, lookback: int, *, relative=True):
        values = self.window(name, timestamp, age, lookback)
        if values is None:
            return None
        new, old = values
        if relative:
            return new / old - 1 if new > 0 and old > 0 else None
        return new - old


def load_snapshot(path: Path, mode="first_seen") -> Snapshot:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return Snapshot(json.load(handle), mode)


def clamp(v, low=-1.0, high=1.0):
    return max(low, min(v, high))


@dataclass(frozen=True)
class Decision:
    allowed: bool
    risk_multiplier: float
    directional_score: float
    side_alignment: float
    stress: float
    coverage: float
    contributions: dict[str, float]
    missing_groups: tuple[str, ...]
    reasons: tuple[str, ...]
    features: dict[str, float | None]


def decision_at(snapshot: Snapshot, timestamp: int, side: str, profile: Mapping[str, Any], public=None) -> Decision:
    if side not in {"long", "short"}:
        raise ValueError("Expected long or short")
    f: dict[str, float | None] = {}

    def change(name, lookback, age, relative=True):
        value = snapshot.change(name, timestamp, age, lookback, relative=relative)
        f[name + "_change"] = value
        return value

    def level(name, age):
        value = snapshot.window(name, timestamp, age)
        f[name] = value
        return value

    scores = {}
    stress_components = {}
    btc = change("btc_close", DAY, 3 * HOUR)
    btc_week = snapshot.change("btc_close", timestamp, 3 * HOUR, 7 * DAY)
    f["btc_7d_change"] = btc_week
    if btc is not None and btc_week is not None:
        scores["btc_momentum"] = .5 * clamp(btc / .04) + .5 * clamp(btc_week / .10)
        stress_components["btc_momentum"] = clamp((abs(btc) - .05) / .10, 0, 1)

    ratio = level("global_long_short", 3 * HOUR)
    top = level("top_position_long_short", 3 * HOUR)
    taker = level("taker_buy_sell", 3 * HOUR)
    oi = change("open_interest", DAY, 3 * HOUR)
    funding = level("funding", 12 * HOUR)
    if all(v is not None for v in (ratio, top, taker, oi, funding, btc)):
        if min(ratio, top, taker) > 0:
            # Account counts are crowding, not a measure of capital or a direct buy signal.
            crowd = .5 * clamp(math.log(ratio) / math.log(3)) + .5 * clamp(math.log(top) / math.log(3))
            flow = clamp(math.log(taker) / math.log(1.5))
            buildup = clamp(oi / .10, 0, 1)
            momentum = clamp(btc / .04)
            scores["positioning"] = clamp(.5 * flow + .2 * momentum * buildup
                                                   - .15 * crowd - .15 * clamp(funding / .0005))
            stress_components["positioning"] = clamp(abs(funding) / .001, 0, 1) * buildup * abs(crowd)

    policy = change("fed_target", 30 * DAY, 7 * DAY, False)
    assets = change("fed_assets", 28 * DAY, 16 * DAY)
    if policy is not None and assets is not None:
        scores["fed"] = -.6 * clamp(policy / .5) + .4 * clamp(assets / .03)
    if profile.get("policy_expectations_enabled"):
        import public_context
        expectation = public_context.expectation_at(public or {}, timestamp)
        f["fed_expected_change_bps"] = expectation["expected_change_bps"] if expectation else None
        if expectation and "fed" in scores:
            scores["fed"] = .65 * scores["fed"] - .35 * clamp(expectation["expected_change_bps"] / 50)
        else:
            scores.pop("fed", None)
        surprises = [r for r in (public or {}).get("policy_surprises", [])
                     if 0 <= timestamp - event_risk._utc_ms(r["available_at_utc"]) <= 3 * DAY]
        surprise = max(surprises, key=lambda r: r["available_at_utc"]) if surprises else None
        f["fed_policy_surprise_bps"] = surprise["surprise_bps"] if surprise else None
        if surprise and "fed" in scores:
            scores["fed"] = .8 * scores["fed"] - .2 * clamp(surprise["surprise_bps"] / 25)

    y2 = change("ust_2y", 7 * DAY, 7 * DAY, False)
    y10 = change("ust_10y", 7 * DAY, 7 * DAY, False)
    real = change("ust_real_10y", 7 * DAY, 7 * DAY, False)
    if all(v is not None for v in (y2, y10, real)):
        # Yield changes are percentage POINTS; works with negative real yields.
        scores["treasury"] = -.4 * clamp(y2 / .25) - .25 * clamp(y10 / .25) - .35 * clamp(real / .25)
        f["yield_curve_change_pp"] = y10 - y2

    fx = []
    # Quote conventions: EUR/GBP are USD per currency, others currency per USD.
    for name, sign in (("eurusd", 1), ("gbpusd", 1), ("usdjpy", -1), ("usdcny", -1), ("usdchf", -1)):
        value = change(name, 7 * DAY, 16 * DAY)
        if value is not None:
            fx.append(sign * clamp(value / .02))
    dollar = change("broad_dollar", 7 * DAY, 16 * DAY)
    if len(fx) == 5 and dollar is not None:
        # One combined FX budget avoids counting USD exposure six separate times.
        scores["fx"] = .5 * sum(fx) / len(fx) - .5 * clamp(dollar / .02)

    stocks = change("sp500", 7 * DAY, 7 * DAY)
    nasdaq = change("nasdaq", 7 * DAY, 7 * DAY)
    vix = level("vix", 7 * DAY)
    oil = change("oil", 7 * DAY, 7 * DAY)
    gold = change("gold", 7 * DAY, 7 * DAY)
    if all(v is not None for v in (stocks, nasdaq, vix, oil)):
        scores["global_risk"] = .5 * clamp(stocks / .05) + .5 * clamp(nasdaq / .06)
        # Liquidity/geopolitical stress reduces BOTH sides, never rewards short leverage.
        stress_components["global_risk"] = max(clamp((vix - 20) / 25, 0, 1),
                     clamp((oil - .08) / .15, 0, 1),
                     min(clamp((gold or 0) / .05, 0, 1), clamp(-stocks / .05, 0, 1)))

    weights = profile["groups"]
    stress = max((v for k, v in stress_components.items() if k in weights), default=0.0)
    scores = {k: v for k, v in scores.items() if k in weights}
    missing = tuple(sorted(set(weights) - set(scores)))
    total = sum(weights.values())
    covered = sum(weights[k] for k in scores)
    coverage = covered / total
    directional = sum(weights[k] * scores[k] for k in scores) / covered if covered else 0.0
    alignment = directional * (1 if side == "long" else -1)
    reasons = []
    if coverage + 1e-12 < profile["minimum_group_coverage"]:
        reasons.append("missing_or_stale_factors")
    if alignment < profile["block_alignment_below"]:
        reasons.append("direction_conflict")
    if stress >= profile["block_stress_at"]:
        reasons.append("market_stress")
    floor = profile["minimum_risk_multiplier"]
    multiplier = (floor + (1 - floor) * (alignment + 1) / 2) * (1 - .65 * stress)
    return Decision(not reasons, clamp(multiplier, 0, 1) if not reasons else 0.0,
                    directional, alignment, stress, coverage, scores, missing, tuple(reasons), f)


def apply_overlay(sleeves, snapshot, profile, public=None):
    adjusted, rows = [], []
    for sleeve in sleeves:
        from execution_ledger import attach_ledger
        sleeve = attach_ledger(sleeve)
        trades = []
        for raw in sleeve.get("trades", []):
            timestamp = event_risk._utc_ms(str(raw["entry_time_utc"]))
            decision = decision_at(snapshot, timestamp, raw["side"], profile, public)
            rows.append({"entry_time_utc": raw["entry_time_utc"], "side": raw["side"], **asdict(decision)})
            if not decision.allowed:
                continue
            trade = dict(raw)
            for field in ("initial_qty", "pnl", "fees", "net_pnl", "funding_pnl", "slippage_cost"):
                if field in trade:
                    trade[field] = float(trade[field]) * decision.risk_multiplier
            trade["signal_reason"] = (str(trade.get("signal_reason", "")) +
                f" multifactor_alignment={decision.side_alignment:.3f} x={decision.risk_multiplier:.3f}")
            trades.append(trade)
        adjusted.append({**sleeve, "trades": trades})
    return adjusted, {
        "candidate_id": profile["candidate_id"], "profile_sha256": profile_hash(profile),
        "availability_mode": snapshot.mode, "decisions": len(rows),
        "blocked": sum(not r["allowed"] for r in rows),
        "group_coverage_pct": {k: 100 * sum(k in r["contributions"] for r in rows) / len(rows)
                               if rows else 0 for k in profile["groups"]},
        "decision_log": rows,
        "limitation": "Entry overlay on frozen trades; no refit, dynamic exit model or performance guarantee.",
    }
