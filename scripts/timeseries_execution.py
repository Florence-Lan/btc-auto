"""Frozen hourly signals with a known opening tick, without unfinished OHLC lookahead."""
from dataclasses import asdict, dataclass, replace
import math

import simulate_range_swing as sim
from execution_ledger import attach_ledger

MODEL = "closed_signal_known_open_v1"


@dataclass(frozen=True)
class Opening:
    time_ms: int
    price: float
    observed_at_ms: int


def opening_from_base(base, timestamp, asof_ms):
    for bar in reversed(base):
        if bar.open_time_ms == timestamp:
            return Opening(timestamp, bar.open, timestamp) if timestamp <= asof_ms else None
        if bar.open_time_ms < timestamp:
            break
    return None


def fetch_opening(symbol, interval, timestamp, asof_ms, client=None):
    from binance_terminal_client import BinanceTerminalClient
    if timestamp > asof_ms:
        raise ValueError("Opening is not yet observable")
    client = client or BinanceTerminalClient()
    rows = client.public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                            "startTime": timestamp, "endTime": timestamp, "limit": 1})
    if not rows or int(rows[0][0]) != timestamp:
        raise ValueError("Expected opening tick unavailable")
    # Deliberately never parse the forming candle's high, low, close or volume.
    price = float(rows[0][1])
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid opening price")
    return Opening(timestamp, price, asof_ms)


def build_sleeve(closed, cfg, start, funding=None, *, opening=None, asof_ms=None):
    step = sim.interval_to_ms(cfg.timeseries_timeframe)
    if not closed:
        raise ValueError("Hourly signal input must contain closed candles")
    asof_ms = asof_ms if asof_ms is not None else closed[-1].close_time_ms
    if any(bar.close_time_ms > asof_ms for bar in closed):
        raise ValueError("Hourly signal input must contain only closed candles")
    if any(b.open_time_ms-a.open_time_ms != step for a,b in zip(closed, closed[1:])):
        raise ValueError("Hourly signal input has a gap")
    # Entry/reversal impact at an hourly open can only use previous closed
    # liquidity. Using that hour's eventual volume changes earlier order sizing.
    candles = [replace(bar, volume=closed[max(0,i-1)].volume,
                       quote_volume=closed[max(0,i-1)].quote_volume) for i,bar in enumerate(closed)]
    if opening is not None:
        if (opening.time_ms != closed[-1].close_time_ms + 1 or opening.time_ms % step
                or opening.time_ms > asof_ms or opening.observed_at_ms > asof_ms
                or opening.observed_at_ms < opening.time_ms
                or not math.isfinite(opening.price) or opening.price <= 0):
            raise ValueError("Opening does not follow the last closed hour")
        p = opening.price
        candles.append(sim.Candle(opening.time_ms, sim.iso_utc_from_ms(opening.time_ms),
            p,p,p,p,closed[-1].volume,closed[-1].quote_volume,opening.time_ms))
        if funding is not None:
            # The frozen shadow ledger observes funding with the CLOSED hour.
            # An opening-only observation must not move that cash event earlier
            # than full replay. The execution account settles actual funding
            # independently against its inventory at the settlement timestamp.
            rows = [(t,r) for t,r in zip(funding.times,funding.rates) if t <= closed[-1].close_time_ms]
            funding = sim.FundingHistory([t for t,r in rows], [r for t,r in rows])
    result = sim.simulate_timeseries_trend(candles, cfg, start, funding)
    if opening is not None and result["equity_curve"]:
        point = result["equity_curve"][-1]
        if int(point["time_ms"]) == opening.time_ms:
            point["available_time_ms"] = opening.time_ms
            result = attach_ledger(result)
            for trade in result["trades"]:
                if trade["exit_reason"] == "end":
                    trade["_ledger_exit_ms"] = opening.time_ms
                    trade["_ledger"][-1]["time_ms"] = opening.time_ms
    result["execution_timing"] = {"model": MODEL, "opening": asdict(opening) if opening else None,
                                  "liquidity": "previous_closed_hour"}
    return result


def decision_open_prices(base, sleeves=(), opening=None):
    prices = {bar.open_time_ms: bar.open for bar in base}
    for sleeve in sleeves:
        value = sleeve.get("execution_timing", {}).get("opening")
        if value:
            prices[int(value["time_ms"])] = float(value["price"])
    if opening is not None:
        prices[opening.time_ms] = opening.price
    return prices
