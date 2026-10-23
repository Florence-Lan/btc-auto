"""Isolated, public, point-in-time cross-market observations; never places orders.

An HTTP success is not a new market observation. Original timestamps and receipt
times survive cache hits and unchanged source rows. Date-only official statistics
remain date-only and cannot satisfy an intraday risk gate. Futures are explicitly
named; this observer never rolls a continuous symbol into a new contract.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import requests

try:
    from public_request_deadline import PublicRequestDeadline
    from public_transport import safe_transport_error
except ModuleNotFoundError:  # Support importing as scripts.cross_market_observations.
    from scripts.public_request_deadline import PublicRequestDeadline
    from scripts.public_transport import safe_transport_error

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/cross_market_observation_20261023.json"
DEFAULT_OUTPUT = ROOT / "data/cross_market_observer"
MAX_BODY_BYTES = 2 * 1024 * 1024
_DEADLINES: dict[float, PublicRequestDeadline] = {}
_DEADLINE_LOCK = threading.Lock()


class ObservationError(ValueError):
    """An observation cannot be used with the claimed identity or time basis."""


class PublicHttpError(requests.HTTPError):
    def __init__(self, status: int, retry_after_seconds: int | None = None):
        self.status = status
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Public observer HTTP {status}")


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ObservationError("non_numeric_value")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ObservationError("non_numeric_value") from None
    if not math.isfinite(number):
        raise ObservationError("non_finite_value")
    return number


def parse_timestamp_ms(value: Any) -> int:
    """Accept explicit Unix milliseconds or an ISO timestamp WITH a timezone."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if not isinstance(value, str):
        raise ObservationError("timestamp_required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ObservationError("invalid_timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ObservationError("timestamp_timezone_required")
    return int(parsed.timestamp() * 1000)


def _zone(name: Any) -> ZoneInfo:
    if not isinstance(name, str) or not name:
        raise ObservationError("observation_timezone_required")
    try:
        return ZoneInfo(name)
    except (KeyError, ValueError):
        raise ObservationError("invalid_observation_timezone") from None


def _public_url(url: str) -> str:
    p = urlsplit(url)
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.fragment:
        raise ObservationError("public_https_url_required")
    return url


def _config_id(source: dict) -> str:
    # Poll/freshness policy changes do not change the identity of a market print.
    identity = {k: source.get(k) for k in (
        "parser", "provider", "url", "expected_symbol", "expected_type",
        "expected_currency", "instrument", "date_column", "date_format", "timezone")}
    identity["fields"] = {key: {name: spec.get(name) for name in ("unit", "column")}
                          for key, spec in source.get("fields", {}).items()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("mode") != "OBSERVATION_ONLY":
        raise ValueError("Observer configuration must be schema 1 and OBSERVATION_ONLY")
    sources = config.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 16:
        raise ValueError("Observer requires 1 to 16 sources")
    ids, fields = set(), set()
    budget = _finite(config.get("collection_timeout_seconds", 12))
    if not 0 < budget <= 30:
        raise ValueError("Collection deadline must be at most 30 seconds")
    for source in sources:
        sid = source["id"]
        if sid in ids or not re.fullmatch(r"[a-z0-9_]+", sid):
            raise ValueError("Unique safe source ids required")
        ids.add(sid)
        if source.get("parser") not in {"yahoo_chart", "dated_csv", "unavailable"}:
            raise ValueError("Unsupported public observer parser")
        if source.get("parser") != "unavailable":
            _public_url(source["url"])
        _zone(source.get("timezone"))
        if not 30 <= _finite(source.get("poll_seconds", 60)) <= 86400:
            raise ValueError("Public observer poll interval must be 30s to one day")
        if not 0 < _finite(source.get("timeout_seconds", 7)) <= budget:
            raise ValueError("Source deadline exceeds collection deadline")
        if not source.get("fields"):
            raise ValueError("Source fields required")
        for key, spec in source["fields"].items():
            if key in fields:
                raise ValueError("Observation keys must be unique")
            fields.add(key)
            if spec.get("unit") not in {"USD/barrel", "index_points", "JPY/USD", "percent"}:
                raise ValueError("Explicit supported unit required")
        if source.get("parser") == "yahoo_chart":
            symbol = source.get("expected_symbol", "")
            if "=F" in symbol or not symbol:
                raise ValueError("A named futures contract or explicit currency pair is required")
            if source.get("expected_type") == "FUTURE":
                cutoff = source.get("contract_review_cutoff_utc")
                if not cutoff or not source.get("instrument", {}).get("contract_month"):
                    raise ValueError("Named futures require contract month and review cutoff")
                parse_timestamp_ms(cutoff)


def fetch_public(source: dict, timeout_seconds: float) -> bytes:
    """One bounded public request, no credential use, retry, or denial bypass.

    The outer PublicRequestDeadline bounds DNS/trickling reads. requests retains
    the user's proxy and CA configuration. 403/429 are returned as errors and
    are never retried via another host, client, or transport.
    """
    url = _public_url(source["url"])
    initial_host = urlsplit(url).hostname
    for _ in range(3):
        response = requests.get(url, headers={"User-Agent": "btc-auto-public-observer/1.0"},
                                timeout=(min(3, timeout_seconds), timeout_seconds),
                                stream=True, allow_redirects=False)
        try:
            if response.status_code in {301, 302, 303, 307, 308}:
                target = _public_url(urljoin(url, response.headers.get("Location", "")))
                if urlsplit(target).hostname != initial_host:
                    raise ObservationError("cross_origin_redirect_rejected")
                url = target
                continue
            if response.status_code != 200:
                retry = response.headers.get("Retry-After", "")
                raise PublicHttpError(response.status_code, int(retry) if retry.isdigit() else None)
            body, size = [], 0
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > MAX_BODY_BYTES:
                    raise ObservationError("response_body_limit_exceeded")
                body.append(chunk)
            if not size:
                raise ObservationError("empty_response")
            return b"".join(body)
        finally:
            response.close()
    raise ObservationError("redirect_limit_exceeded")


def _text(payload: bytes | str) -> str:
    if isinstance(payload, str):
        return payload
    # MOF's CSV has a Shift-JIS footer. Numeric rows are ASCII; decoding it must
    # not turn an encoding mismatch into an invented observation.
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload.decode("shift_jis")


def parse_yahoo_chart(payload: Any, source: dict, received_at_ms: int) -> dict:
    data = payload if isinstance(payload, dict) else json.loads(_text(payload))
    result = data.get("chart", {}).get("result")
    if not isinstance(result, list) or len(result) != 1:
        raise ObservationError("missing_quote_result")
    meta = result[0].get("meta", {})
    if (meta.get("symbol") != source["expected_symbol"]
            or meta.get("instrumentType") != source["expected_type"]
            or meta.get("currency") != source["expected_currency"]):
        raise ObservationError("instrument_identity_mismatch")
    tz = meta.get("exchangeTimezoneName")
    _zone(tz)
    raw_time = meta.get("regularMarketTime")
    if not isinstance(raw_time, int) or isinstance(raw_time, bool) or raw_time <= 0:
        raise ObservationError("quote_time_required")
    observed = raw_time * 1000
    if observed > received_at_ms:
        raise ObservationError("future_observation")
    cutoff = source.get("contract_review_cutoff_utc")
    if cutoff and received_at_ms >= parse_timestamp_ms(cutoff):
        raise ObservationError("contract_review_due")
    value = _finite(meta.get("regularMarketPrice"))
    if value <= 0:
        raise ObservationError("nonpositive_quote")
    row = {"value": value, "observed_at_ms": observed, "precision": "timestamp",
           "observed_date": datetime.fromtimestamp(observed / 1000, _zone(tz)).date().isoformat(),
           "observation_timezone": tz, "time_basis": "provider_regularMarketTime_unix_seconds",
           "provider_reported_delay_raw": meta.get("exchangeDataDelayedBy"),
           "provider_reported_delay_unit": "provider_metadata_unspecified_use_observation_age",
           "reported_name": meta.get("longName") or meta.get("shortName")}
    previous = meta.get("previousClose")
    if previous is not None:
        previous = _finite(previous)
        if previous > 0:
            row.update(provider_previous_close=previous,
                       return_pct=(value / previous - 1) * 100,
                       return_basis="same_named_instrument_provider_previousClose",
                       previous_close_observed_at_ms=None)
    if len(source["fields"]) != 1:
        raise ObservationError("one_quote_field_required")
    return {next(iter(source["fields"])): row}


def parse_dated_csv(payload: Any, source: dict, received_at_ms: int) -> dict:
    text = _text(payload).replace("\r\r\n", "\n")
    date_column = source.get("date_column", "Date")
    rows = list(csv.reader(io.StringIO(text)))
    header_at = next((i for i, row in enumerate(rows)
                      if date_column in [cell.strip().lstrip("\ufeff") for cell in row]), None)
    if header_at is None:
        raise ObservationError("csv_date_header_missing")
    header = [cell.strip().lstrip("\ufeff") for cell in rows[header_at]]
    local_today = datetime.fromtimestamp(received_at_ms / 1000, _zone(source["timezone"])).date()
    parsed = []
    for cells in rows[header_at + 1:]:
        if len(cells) != len(header):
            continue
        row = dict(zip(header, (cell.strip() for cell in cells)))
        raw_date = row.get(date_column, "")
        try:
            observed_date = datetime.strptime(raw_date, source.get("date_format", "%Y-%m-%d")).date()
        except ValueError:
            continue  # Titles/footer/non-numeric remarks are never market rows.
        if observed_date > local_today:
            raise ObservationError("future_observation_date")
        parsed.append((observed_date, row))
    if not parsed:
        raise ObservationError("no_dated_rows")
    latest_date, latest = max(parsed, key=lambda pair: pair[0])
    # Use one latest source date across maturities. Never fill a missing latest
    # maturity from an older day and pretend the curve is synchronized.
    result = {}
    for key, spec in source["fields"].items():
        raw = latest.get(spec["column"])
        if raw in {None, "", "N/A", "NA", ".", "-"}:
            continue
        value = _finite(raw)
        if not -20 <= value <= 100:
            raise ObservationError("implausible_percent_unit")
        result[key] = {"value": value, "observed_at_ms": None,
                       "observed_date": latest_date.isoformat(), "precision": "date",
                       "observation_timezone": source["timezone"],
                       "time_basis": "provider_date_only_no_intraday_timestamp"}
    if not result:
        raise ObservationError("latest_row_has_no_requested_values")
    return result


def _identity(row: dict) -> tuple:
    return tuple(json.dumps(row.get(key), sort_keys=True) for key in (
        "source_config_id", "instrument", "value", "observed_at_ms", "observed_date", "precision"))


def _base_row(source: dict, key: str) -> dict:
    return {"key": key, "value": None, "unit": source["fields"][key]["unit"],
            "provider": source.get("provider", source["id"]), "source_id": source["id"],
            "source_url": source.get("url"), "source_config_id": _config_id(source),
            "instrument": copy.deepcopy(source.get("instrument", {})),
            "observed_at_ms": None, "observed_date": None, "precision": None,
            "observation_timezone": source["timezone"], "first_seen_at_ms": None,
            "available_at_ms": None, "received_at_ms": None,
            "contract_review_cutoff_at_ms": parse_timestamp_ms(source["contract_review_cutoff_utc"])
                if source.get("contract_review_cutoff_utc") else None}


def freshness_at(row: dict, at_ms: int, spec: dict) -> dict:
    result = {"status": "missing", "age_ms": None, "age_days": None,
              "max_age_ms": int(spec.get("max_age_seconds", 1200) * 1000),
              "max_age_days": spec.get("max_age_days", 4),
              "eligible_intraday": False, "eligible_daily": False}
    if row.get("value") is None:
        return result
    first, available = row.get("first_seen_at_ms"), row.get("available_at_ms")
    if (not isinstance(first, int) or not isinstance(available, int)
            or available < first or first > at_ms or available > at_ms):
        result["status"] = "not_yet_available"
        return result
    observed = row.get("observed_at_ms")
    if row.get("precision") == "timestamp" and isinstance(observed, int):
        age = at_ms - observed
        result["age_ms"] = age
        if observed > first:
            result["status"] = "observation_after_first_receipt"
        elif age < 0:
            result["status"] = "future_observation"
        else:
            result["status"] = "fresh" if age <= result["max_age_ms"] else "stale"
            result["eligible_intraday"] = result["status"] == "fresh"
    elif row.get("precision") == "date":
        try:
            obs_date = date.fromisoformat(row["observed_date"])
            today = datetime.fromtimestamp(at_ms / 1000, _zone(row["observation_timezone"])).date()
        except (ValueError, KeyError, TypeError):
            result["status"] = "invalid_date"
            return result
        age_days = (today - obs_date).days
        result["age_days"] = age_days
        if age_days < 0:
            result["status"] = "future_observation_date"
        else:
            result["status"] = "daily_current" if age_days <= result["max_age_days"] else "stale"
            result["eligible_daily"] = result["status"] == "daily_current"
    return result


def _stamp_row(candidate: dict, old: dict | None, received: int) -> dict:
    candidate.update(first_seen_at_ms=received, available_at_ms=received, received_at_ms=received)
    if candidate.get("provider_previous_close") is not None:
        # The reference is optional metadata, with its own causal receipt time.
        # A corrected previous close must not be backdated into an old quote.
        unchanged_reference = (old and old.get("provider_previous_close") == candidate["provider_previous_close"]
                               and old.get("instrument") == candidate.get("instrument"))
        candidate["provider_reference_first_seen_at_ms"] = (
            old.get("provider_reference_first_seen_at_ms", received) if unchanged_reference else received)
        candidate["provider_reference_available_at_ms"] = (
            old.get("provider_reference_available_at_ms", received) if unchanged_reference else received)
    if old and _identity(old) == _identity(candidate):
        # A provider returning the same print cannot renew either causal time.
        candidate["first_seen_at_ms"] = old.get("first_seen_at_ms")
        candidate["available_at_ms"] = old.get("available_at_ms")
        candidate["received_at_ms"] = old.get("received_at_ms")
    return candidate


def _source_fetch(source: dict, fetcher: Callable, now_fn: Callable) -> tuple:
    timeout = float(source.get("timeout_seconds", 7))
    with _DEADLINE_LOCK:
        deadline = _DEADLINES.setdefault(timeout, PublicRequestDeadline(timeout, max_pending=16))
    started = time.monotonic()
    try:
        payload = deadline.run(("cross_market", source["id"], source["url"]),
                               lambda: fetcher(source, timeout))
        received = now_fn()
        parser = parse_yahoo_chart if source["parser"] == "yahoo_chart" else parse_dated_csv
        parsed = parser(payload, source, received)
        return parsed, received, int((time.monotonic() - started) * 1000), None
    except Exception as exc:
        error = safe_transport_error(exc)
        if isinstance(exc, ObservationError):
            error["reason"] = str(exc)  # Only our closed, nonsecret error vocabulary.
        if isinstance(exc, PublicHttpError):
            error.update(http_status=exc.status, retry_after_seconds=exc.retry_after_seconds)
        return {}, now_fn(), int((time.monotonic() - started) * 1000), error


def _derive_spread(key: str, definition: dict, observations: dict, old: dict | None,
                   at_ms: int) -> dict:
    left, right = (observations.get(definition[name], {}) for name in ("left", "right"))
    row = {"key": key, "value": None, "unit": "bp", "provider": "derived",
           "source_id": "derived", "source_url": None, "instrument": {
               "left": definition["left"], "right": definition["right"],
               "formula": "100 * (left_percent - right_percent)"},
           "source_config_id": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
           "observed_at_ms": None, "observed_date": None, "precision": None,
           "observation_timezone": left.get("observation_timezone", "UTC"),
           "first_seen_at_ms": None, "available_at_ms": None, "received_at_ms": None,
           "component_available_at_ms": [left.get("available_at_ms"), right.get("available_at_ms")]}
    good = all(r.get("value") is not None and r.get("unit") == "percent"
               and r.get("freshness", {}).get("eligible_daily") for r in (left, right))
    if (good and left.get("precision") == right.get("precision") == "date"
            and left.get("observed_date") == right.get("observed_date")
            and left.get("observation_timezone") == right.get("observation_timezone")):
        row.update(value=round(100 * (left["value"] - right["value"]), 10),
                   observed_date=left["observed_date"], precision="date",
                   time_basis="same_source_calendar_date_derived_spread")
        _stamp_row(row, old, at_ms)
        row["available_at_ms"] = max(row["available_at_ms"], *row["component_available_at_ms"])
    else:
        row["missing_reason"] = "missing_stale_or_asynchronous_percent_components"
    row["freshness"] = freshness_at(row, at_ms, definition)
    row["eligible_intraday"] = False
    return row


def collect_observations(config: dict, previous: dict | None = None,
                         now_ms: int | None = None, fetcher: Callable | None = None) -> dict:
    """Collect within one total budget; injectable fetcher(source, timeout)->body.

    previous is the entire prior snapshot. Daily sources have their own poll
    clocks, so fast quote polling does not repeatedly download daily statistics.
    No source error or missing value is replaced by a fabricated number.
    """
    validate_config(config)
    previous = previous or {}
    now_fn = _clock_ms if now_ms is None else lambda: int(now_ms)
    started_at = now_fn()
    old_rows = previous.get("observations", {})
    old_sources = previous.get("sources", {})
    sources, observations, pending = {}, {}, {}
    responses: queue.Queue = queue.Queue()
    end = time.monotonic() + float(config.get("collection_timeout_seconds", 12))
    fetcher = fetcher or fetch_public
    for source in config["sources"]:
        sid, config_id = source["id"], _config_id(source)
        old_state = old_sources.get(sid, {})
        compatible = old_state.get("source_config_id") == config_id
        state = copy.deepcopy(old_state) if compatible else {}
        state.update(source_config_id=config_id, provider=source.get("provider", sid),
                     poll_seconds=source.get("poll_seconds", 60))
        sources[sid] = state
        for key in source["fields"]:
            old = old_rows.get(key, {})
            observations[key] = (copy.deepcopy(old) if compatible
                                 and old.get("source_config_id") == config_id else _base_row(source, key))
        if source["parser"] == "unavailable":
            state.update(status="missing", reason=source.get("reason", "no_verified_public_source"),
                         next_due_at_ms=None)
            continue
        # A clock rollback must not turn observations received in the future
        # into available history, nor cause a retry storm.
        if state.get("last_attempt_at_ms", 0) > started_at:
            state.update(status="clock_rollback", next_due_at_ms=state["last_attempt_at_ms"])
            continue
        if state.get("next_due_at_ms", 0) > started_at:
            state["status"] = "cached"
            continue
        state.update(status="pending", last_attempt_at_ms=started_at)
        pending[sid] = source

        def work(s=source):
            responses.put((s["id"], _source_fetch(s, fetcher, now_fn)))

        threading.Thread(target=work, name=f"cross-market-{sid}", daemon=True).start()
    while pending and time.monotonic() < end:
        try:
            sid, (parsed, received, elapsed, error) = responses.get(timeout=max(.001, end - time.monotonic()))
        except queue.Empty:
            break
        source = pending.pop(sid)
        state = sources[sid]
        state["request_elapsed_ms"] = elapsed
        interval = int(source.get("poll_seconds", 60) * 1000)
        if error:
            state.update(status="error", error=error)
            if error.get("http_status") in {403, 429}:
                interval = max(interval, 3_600_000,
                               min(int(error.get("retry_after_seconds") or 0), 86400) * 1000)
        else:
            state.update(status="ok", last_success_at_ms=received)
            state.pop("error", None)
            state["observations_returned"] = len(parsed)
            for key in source["fields"]:
                if key in parsed:
                    candidate = {**_base_row(source, key), **parsed[key]}
                    old = observations[key]
                    # Regressing provider data cannot overwrite a newer print.
                    new_sort = candidate.get("observed_at_ms") or candidate.get("observed_date") or ""
                    old_sort = old.get("observed_at_ms") or old.get("observed_date") or ""
                    if old.get("value") is not None and type(new_sort) is type(old_sort) and new_sort < old_sort:
                        state["status"] = "regressed_observation"
                        continue
                    observations[key] = _stamp_row(candidate, old, received)
                else:
                    observations[key]["missing_reason"] = "latest_source_row_missing_field"
        state["next_due_at_ms"] = received + interval
    for sid, source in pending.items():
        sources[sid].update(status="deadline_exceeded", error={"category": "timeout"},
                            next_due_at_ms=started_at + int(source.get("poll_seconds", 60) * 1000))
    completed_at = now_fn()
    required_missing = []
    for source in config["sources"]:
        state = sources[source["id"]]
        for key, spec in source["fields"].items():
            row = observations[key]
            row["fetch"] = copy.deepcopy(state)
            row["freshness"] = freshness_at(row, completed_at, spec)
            if source.get("contract_review_cutoff_utc") and completed_at >= parse_timestamp_ms(source["contract_review_cutoff_utc"]):
                row["freshness"].update(status="contract_review_due", eligible_intraday=False)
            row["eligible_intraday"] = row["freshness"]["eligible_intraday"]
            if state.get("status") == "missing":
                row["missing_reason"] = state.get("reason")
            if spec.get("required_intraday") and not row["freshness"]["eligible_intraday"]:
                required_missing.append(key)
    for key, definition in config.get("derived_spreads", {}).items():
        observations[key] = _derive_spread(key, definition, observations, old_rows.get(key), completed_at)
    return {"schema_version": 1, "mode": "OBSERVATION_ONLY", "places_orders": False,
            "automatic_strategy_promotion": False, "original_accounts_modified": False,
            "started_at_ms": started_at, "generated_at_ms": completed_at,
            "status": "degraded" if required_missing else "healthy",
            "required_intraday_missing_or_stale": required_missing,
            "observations": observations, "sources": sources,
            "notes": ["Date-only statistics cannot satisfy intraday eligibility.",
                      "Provider receipt/HTTP success never refreshes an unchanged observation.",
                      "Named futures are not automatically rolled; inspect contract review cutoffs."]}


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--once", action="store_true", help="One bounded collection (default)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    protected = (ROOT / "data/paper_trading").resolve()
    if output_dir == protected or protected in output_dir.parents:
        raise ValueError("Observer output must not overwrite paper-trading snapshots")
    latest = output_dir / "latest.json"
    previous = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None
    result = collect_observations(config, previous=previous)
    archive = output_dir / "snapshots" / f"{result['generated_at_ms']}.json"
    if archive.exists():
        raise FileExistsError("Observer archive already exists; do not overwrite causal history")
    _atomic_json(archive, result)
    _atomic_json(latest, result)
    print(json.dumps({"status": result["status"], "output": str(latest),
                      "required_intraday_missing_or_stale": result["required_intraday_missing_or_stale"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
