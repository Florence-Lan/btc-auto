"""Causal entry restrictions for retrospective stock experiments only."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def validate(config: dict, step: int) -> None:
    spread = config.get('max_entry_spread_fraction')
    if spread is not None and (isinstance(spread, bool) or not isinstance(spread, (int, float))
                              or not math.isfinite(spread) or not 0 < spread <= .01):
        raise ValueError('Maximum entry spread must be a finite fraction in (0, 0.01]')
    validity = config.get("entry_signal_validity_minutes", 0)
    if isinstance(validity, bool) or not isinstance(validity, int) or not 0 <= validity <= 240:
        raise ValueError("Signal validity must be an integer in [0, 240] minutes")
    if validity and (step != 300_000 or validity % 5):
        raise ValueError("Deferred signals require five-minute execution and aligned expiry")
    lookback = config.get("entry_volume_lookback_bars", 1)
    if isinstance(lookback, bool) or not isinstance(lookback, int) or not 1 <= lookback <= 12:
        raise ValueError("Entry volume lookback must be an integer in [1, 12]")
    if lookback != 1 and config.get("entry_max_previous_bar_participation_fraction") is None:
        raise ValueError("Volume lookback requires a participation cap")
    if config.get("entry_direction", "both") not in ("both", "long", "short"):
        raise ValueError("Unknown entry direction")
    if config.get("entry_session", "all") not in ("all", "weekday_regular_clock"):
        raise ValueError("Unknown stock entry session")
    if config.get("entry_session", "all") != "all":
        if step != 300_000:
            raise ValueError("Session experiments require five-minute execution")
        if not set(config["symbols"]).issubset({"MUUSDT", "SNDKUSDT", "SKHYNIXUSDT"}):
            raise ValueError("No declared session clock for symbol")


def regular_clock(symbol: str, timestamp: int) -> bool:
    """Weekday clock hypothesis; holidays and early closes are NOT modeled."""
    if symbol in ("MUUSDT", "SNDKUSDT"):
        zone, opening, closing = "America/New_York", 9 * 60 + 30, 16 * 60
    elif symbol == "SKHYNIXUSDT":
        zone, opening, closing = "Asia/Seoul", 9 * 60, 15 * 60 + 30
    else:
        raise ValueError("No declared session clock for symbol")
    local = datetime.fromtimestamp(timestamp / 1000, timezone.utc).astimezone(ZoneInfo(zone))
    minute = local.hour * 60 + local.minute
    return local.weekday() < 5 and opening <= minute < closing


def rejection(config: dict, symbol: str, timestamp: int, direction: int) -> str | None:
    allowed = config.get("entry_direction", "both")
    if (allowed == "long" and direction != 1) or (allowed == "short" and direction != -1):
        return "research_direction"
    if config.get("entry_session", "all") == "weekday_regular_clock" and not regular_clock(symbol, timestamp):
        return "research_session_clock"
    return None


def preceding_volume(source: dict, timestamp: int, step: int, lookback: int) -> float | None:
    """Minimum volume of contiguous CLOSED bars, never the entry bar."""
    volumes = []
    for offset in range(1, lookback + 1):
        bar = source["trade"].get(timestamp - offset * step)
        if bar is None or not math.isfinite(bar.volume) or bar.volume <= 0:
            return None
        volumes.append(bar.volume)
    return min(volumes)
