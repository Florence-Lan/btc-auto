"""Pure prospective NQ admission hypothesis for NEW stock longs only.

No file or network access. The caller supplies its own immutable journal rows:
native cross_market_observations NQ rows plus actual ingested_at_ms and
source_snapshot_sha256. This function neither grants a trade nor evaluates an
existing position's exits. Native observed_date is a conservative date boundary,
not a claim about the actual CME trading session or official settlement.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import json
import math
import re
from zoneinfo import ZoneInfo

MINUTE = 60_000
_FIXED = {
    "mode": "named_nq_daily_and_30m_long_guard_v1",
    "source_id": "yahoo_nq_dec_2026",
    "source_config_id": "c7f9a7e67f5e9b08a3296c5d41752edcdf3862c6f1fe080e4f55d6e96593c622",
    "provider": "Yahoo Finance public chart (CME quote)",
    "source_url": "https://query1.finance.yahoo.com/v8/finance/chart/NQZ26.CME?interval=1m&range=1d",
    "symbol": "NQZ26.CME",
    "contract_month": "2026-12",
    "exchange": "CME",
    "currency": "USD",
    "unit": "index_points",
    "observation_timezone": "America/New_York",
    "contract_review_cutoff_at_ms": 1796947200000,
    "maximum_quote_age_minutes": 20,
    "lookback_minutes": 30,
    "reference_earlier_tolerance_minutes": 5,
    "minimum_trial_elapsed_minutes": 30,
    "daily_return_weak_below_pct": 0,
    "intraday_return_weak_below_pct": 0,
    "unknown_new_long_allowed": False,
    "automatic_roll": False,
}
_FIXED_JSON = json.dumps(_FIXED, sort_keys=True, allow_nan=False)
POLICY = deepcopy(_FIXED)


def validate(config: Mapping) -> dict:
    """Require the exact declared policy; no parameter search or silent defaults."""
    if not isinstance(config, Mapping):
        raise ValueError("technology_guard_configuration_required")
    policy = config.get("technology_guard_policy")
    try:
        equal = json.dumps(policy, sort_keys=True, allow_nan=False) == _FIXED_JSON
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise ValueError("fixed_technology_guard_policy_required")
    return deepcopy(_FIXED)


def _stamp(value, reason):
    if type(value) is not int or value <= 0:
        raise ValueError(reason)
    return value


def _positive(value, reason):
    if type(value) not in (int, float):
        raise ValueError(reason)
    try:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(reason)
        return float(value)
    except (OverflowError, TypeError):
        raise ValueError(reason) from None


def _point(row: Mapping, *, asof_ms: int, policy: dict) -> dict:
    if not isinstance(row, Mapping):
        raise ValueError("journal_row_object_required")
    stamps = {name: _stamp(row.get(name), "missing_or_invalid_" + name) for name in (
        "observed_at_ms", "first_seen_at_ms", "available_at_ms", "received_at_ms", "ingested_at_ms")}
    observed, first, available, received, ingested = (stamps[name] for name in (
        "observed_at_ms", "first_seen_at_ms", "available_at_ms", "received_at_ms", "ingested_at_ms"))
    if not observed <= first <= available <= received <= ingested <= asof_ms:
        raise ValueError("observation_not_causally_received")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_snapshot_sha256", ""))):
        raise ValueError("source_snapshot_sha256_required")
    for name in ("source_id", "source_config_id", "provider", "source_url", "unit", "observation_timezone"):
        if row.get(name) != policy[name]:
            raise ValueError("nq_identity_mismatch:" + name)
    instrument = row.get("instrument")
    if not isinstance(instrument, Mapping):
        raise ValueError("instrument_identity_missing")
    for name in ("symbol", "contract_month", "exchange", "currency"):
        if instrument.get(name) != policy[name]:
            raise ValueError("nq_identity_mismatch:" + name)
    if instrument.get("automatic_roll") is not False or instrument.get("continuous_contract") is not False:
        raise ValueError("named_contract_without_automatic_roll_required")
    if row.get("precision") != "timestamp" or row.get("frequency") == "daily":
        raise ValueError("intraday_timestamp_required")
    if row.get("eligible_intraday") is not True:
        raise ValueError("intraday_eligibility_unavailable")
    cutoff = _stamp(row.get("contract_review_cutoff_at_ms"), "contract_review_cutoff_required")
    if cutoff != policy["contract_review_cutoff_at_ms"]:
        raise ValueError("contract_review_configuration_changed")
    if asof_ms >= cutoff:
        raise ValueError("contract_review_due")
    session = row.get("market_session")
    if not isinstance(session, Mapping) or session.get("phase") != "open":
        raise ValueError("session_reference_not_open")
    if session.get("is_actual_market_status") is not False:
        raise ValueError("scheduled_session_reference_required")
    session_asof = _stamp(session.get("asof_ms"), "session_reference_clock_required")
    valid_until = _stamp(session.get("valid_until_ms"), "session_reference_expiry_required")
    if session_asof > ingested or asof_ms >= valid_until:
        raise ValueError("session_reference_not_current")
    day = datetime.fromtimestamp(observed / 1000, ZoneInfo(policy["observation_timezone"])).date().isoformat()
    if row.get("observed_date") != day:
        raise ValueError("provider_observed_date_mismatch")
    value = _positive(row.get("value"), "positive_quote_required")
    previous = _positive(row.get("provider_previous_close"), "provider_previous_close_required")
    if row.get("return_basis") != "same_named_instrument_provider_previousClose":
        raise ValueError("same_contract_provider_previous_close_basis_required")
    ref_first = _stamp(row.get("provider_reference_first_seen_at_ms"), "provider_reference_first_seen_required")
    ref_available = _stamp(row.get("provider_reference_available_at_ms"), "provider_reference_availability_required")
    if not ref_first <= ref_available <= ingested:
        raise ValueError("provider_reference_not_causally_available")
    # The upstream source has no prior-close observation date. Its version is
    # the retained first-receipt identity, not an invented official settle date.
    base_identity = (row["provider"], row["source_id"], row["source_config_id"],
                     json.dumps(dict(instrument), sort_keys=True, allow_nan=False))
    return {**stamps, "value": value, "provider_previous_close": previous,
            "observed_date": day, "base_identity": base_identity,
            "session_identity": (base_identity, day),
            "reference_identity": (previous, ref_first, ref_available),
            "provider_reference_first_seen_at_ms": ref_first,
            "provider_reference_available_at_ms": ref_available,
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "instrument": deepcopy(dict(instrument)), "unit": row["unit"],
            "contract_review_cutoff_at_ms": cutoff, "session_reference_valid_until_ms": valid_until,
            "record_type": row.get("record_type", "observation")}


def _public_point(point):
    return {key: deepcopy(value) for key, value in point.items()
            if key not in {"base_identity", "session_identity", "reference_identity", "record_type"}}


def evaluate(rows, config, *, asof_ms, inception_ms, direction=1, new_risk=True) -> dict:
    """Evaluate one causal journal prefix; never mutate rows or persist history.

    ``allowed`` means this added guard does not object; all original admission
    rules remain required. A short or any existing-position exit bypasses this
    guard even when the supplied market rows are missing. Conflict journal
    markers use the native flat row shape plus record_type='conflict'.
    """
    result = {"enabled": True, "scope": "new_long_only", "allowed": True,
              "status": "not_applicable", "reasons": [], "checked_at_ms": asof_ms,
              "direction": direction, "protective_exits_allowed": True,
              "original_admission_still_required": True, "automatic_strategy_promotion": False,
              "weak": None, "counts_as_weak_filter": False,
              "daily_return_pct": None, "return_30m_pct": None,
              "latest": None, "reference": None, "valid_until_ms": None,
              "reference_basis": "same_contract_provider_previousClose_not_official_settlement",
              "session_basis": "provider_local_date_and_reference_version_not_actual_CME_session"}
    if new_risk is False:
        result["reasons"] = ["existing_position_exit_not_filtered"]
        return result
    if type(direction) is int and direction == -1:
        result["reasons"] = ["short_original_admission_unchanged"]
        return result
    policy = validate(config)
    _stamp(asof_ms, "decision_timestamp_required")
    _stamp(inception_ms, "trial_inception_required")
    result.update(allowed=False, status="unknown", inception_ms=inception_ms,
                  lookback_minutes=30, reference_elapsed_minutes_range=[30, 35])

    def unknown(reason):
        result["reasons"] = [reason]
        return result

    if new_risk is not True or type(direction) is not int or direction != 1:
        return unknown("new_long_direction_required")
    if asof_ms < inception_ms:
        return unknown("decision_before_trial_inception")
    if asof_ms - inception_ms < policy["minimum_trial_elapsed_minutes"] * MINUTE:
        return unknown("trial_history_warming_up")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return unknown("immutable_journal_rows_required")
    points, markers = [], []
    for row in rows:
        if not isinstance(row, Mapping):
            return unknown("journal_row_object_required")
        ingested = row.get("ingested_at_ms")
        if type(ingested) is int and ingested > asof_ms:
            continue  # Future journal entries cannot contaminate historical decisions.
        try:
            point = _point(row, asof_ms=asof_ms, policy=policy)
        except ValueError as exc:
            return unknown(str(exc))
        except (TypeError, OverflowError, OSError):
            return unknown("invalid_journal_observation")
        if point["observed_at_ms"] < inception_ms:
            continue
        if point["record_type"] == "conflict":
            markers.append(point)
        elif point["record_type"] == "observation":
            points.append(point)
        else:
            return unknown("unknown_journal_record_type")
    if not points:
        return unknown("post_inception_observation_missing")
    latest = max(points, key=lambda item: item["observed_at_ms"])
    session = latest["session_identity"]
    if any(point["session_identity"] == session for point in markers):
        return unknown("session_observation_conflict")
    # A revision is visible only once received by this trial. It cannot replace
    # the original point or make an older asof evaluation see the later value.
    seen, unique = {}, {}
    for point in points:
        key = (point["base_identity"], point["observed_at_ms"])
        content = (point["value"], point["reference_identity"])
        if key in seen and seen[key] != content and point["session_identity"] == session:
            return unknown("session_observation_conflict")
        seen[key] = content
        prior = unique.get(key)
        if prior is None or point["ingested_at_ms"] < prior["ingested_at_ms"]:
            unique[key] = point
    latest = unique[(latest["base_identity"], latest["observed_at_ms"])]
    result["latest"] = _public_point(latest)
    if asof_ms - latest["observed_at_ms"] > policy["maximum_quote_age_minutes"] * MINUTE:
        return unknown("latest_observation_stale")
    target = latest["observed_at_ms"] - policy["lookback_minutes"] * MINUTE
    earliest = target - policy["reference_earlier_tolerance_minutes"] * MINUTE
    references = [point for point in unique.values()
                  if earliest <= point["observed_at_ms"] <= target
                  and point["session_identity"] == session
                  and point["reference_identity"] == latest["reference_identity"]]
    if not references:
        return unknown("same_reference_30m_history_warming_up")
    reference = max(references, key=lambda item: item["observed_at_ms"])
    daily = 100 * (latest["value"] / latest["provider_previous_close"] - 1)
    intraday = 100 * (latest["value"] / reference["value"] - 1)
    weak = latest["value"] < latest["provider_previous_close"] and latest["value"] < reference["value"]
    result.update(allowed=not weak, status="weak" if weak else "clear", weak=weak,
                  counts_as_weak_filter=weak,
                  reasons=["nq_below_provider_previous_close_and_30m_reference"] if weak else [],
                  daily_return_pct=daily, return_30m_pct=intraday,
                  reference=_public_point(reference),
                  elapsed_observation_minutes=(latest["observed_at_ms"] - reference["observed_at_ms"]) / MINUTE,
                  valid_until_ms=min(latest["observed_at_ms"] + policy["maximum_quote_age_minutes"] * MINUTE,
                                     latest["contract_review_cutoff_at_ms"],
                                     latest["session_reference_valid_until_ms"]))
    return result
