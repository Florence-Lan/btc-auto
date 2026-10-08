import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multifactor as mf

NOW = 100 * mf.DAY


def profile(optional=True, effective=NOW):
    result = {"candidate_id": "optional_oil_test", "live_orders_allowed": False,
        "groups": {"global_risk": 1}, "minimum_group_coverage": 1,
        "minimum_risk_multiplier": .35, "block_alignment_below": -.35,
        "block_stress_at": .9, "availability_mode": "first_seen"}
    if optional:
        result.update(optional_features=["oil"], optional_features_effective_at_ms=effective)
    return result


def row(timestamp, value, first_seen=None):
    return [timestamp, timestamp, value, timestamp if first_seen is None else first_seen]


def market(*, oil="missing", missing=(), vix=15):
    current = NOW - mf.HOUR
    series = {name: [row(current - 7 * mf.DAY, 100), row(current, 100)]
              for name in ("sp500", "nasdaq")}
    series["vix"] = [row(current, vix)]
    if oil == "stale":
        series["oil"] = [row(NOW - 15 * mf.DAY, 100), row(NOW - 8 * mf.DAY, 140)]
    elif oil == "stress":
        series["oil"] = [row(current - 7 * mf.DAY, 100), row(current, 140)]
    elif oil == "calm":
        series["oil"] = [row(current - 7 * mf.DAY, 100), row(current, 100)]
    for name in missing:
        series.pop(name, None)
    return mf.Snapshot({"schema_version": 1, "series": series})


@pytest.mark.parametrize("oil", ["missing", "stale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_optional_unavailable_oil_does_not_block_or_count_as_zero(oil, side):
    decision = mf.decision_at(market(oil=oil), NOW, side, profile())
    assert decision.allowed and decision.coverage == 1
    assert decision.missing_groups == () and decision.reasons == ()
    assert decision.ignored_features == ("oil",)
    assert decision.features["oil_change"] is None
    assert decision.stress == 0
    assert asdict(decision)["ignored_features"] == ("oil",)


def test_optional_oil_rule_does_not_rewrite_decisions_before_activation():
    snapshot = market(oil="stale")
    configured = profile(effective=NOW)
    before = mf.decision_at(snapshot, NOW - 1, "long", configured)
    original = mf.decision_at(snapshot, NOW - 1, "long", profile(optional=False))
    assert before == original
    assert before.reasons == ("missing_or_stale_factors",)
    assert before.ignored_features == ()
    assert mf.decision_at(snapshot, NOW, "long", configured).allowed
    assert mf.decision_at(snapshot, NOW + 1, "short", configured).allowed


@pytest.mark.parametrize("oil", ["missing", "stale"])
def test_unconfigured_profiles_keep_oil_required(oil):
    decision = mf.decision_at(market(oil=oil), NOW, "long", profile(optional=False))
    assert not decision.allowed and decision.coverage == 0
    assert decision.missing_groups == ("global_risk",)
    assert decision.ignored_features == ()


@pytest.mark.parametrize("side", ["long", "short"])
def test_new_fresh_oil_automatically_restores_original_stress(side):
    configured = profile()
    assert mf.decision_at(market(oil="stale"), NOW, side, configured).allowed
    stressed = mf.decision_at(market(oil="stress"), NOW, side, configured)
    original = mf.decision_at(market(oil="stress"), NOW, side, profile(optional=False))
    assert stressed == original
    assert not stressed.allowed and stressed.stress == 1
    assert stressed.reasons == ("market_stress",)
    assert stressed.features["oil_change"] == pytest.approx(.4)
    assert stressed.ignored_features == () and stressed.risk_multiplier == 0


@pytest.mark.parametrize("missing", ["sp500", "nasdaq", "vix"])
def test_other_global_risk_inputs_are_still_required(missing):
    decision = mf.decision_at(market(missing=[missing]), NOW, "long", profile())
    assert not decision.allowed and decision.coverage == 0
    assert decision.missing_groups == ("global_risk",)
    assert decision.reasons == ("missing_or_stale_factors",)


def test_optional_oil_does_not_lower_coverage_requirement_for_other_groups():
    configured = profile()
    configured["groups"] = {"global_risk": .5, "btc_momentum": .5}
    decision = mf.decision_at(market(), NOW, "long", configured)
    assert not decision.allowed and decision.coverage == .5
    assert decision.missing_groups == ("btc_momentum",)
    assert decision.reasons == ("missing_or_stale_factors",)
    assert configured["minimum_group_coverage"] == 1


@pytest.mark.parametrize("side", ["long", "short"])
def test_high_vix_still_blocks_both_sides_without_oil(side):
    decision = mf.decision_at(market(vix=50), NOW, side, profile())
    assert not decision.allowed and decision.stress == 1 and decision.coverage == 1
    assert decision.reasons == ("market_stress",)
    assert decision.ignored_features == ("oil",)


def test_future_first_seen_oil_cannot_apply_until_received():
    current = NOW - mf.HOUR
    payload = {"schema_version": 1, "series": {
        name: [row(current - 7 * mf.DAY, 100), row(current, 100)]
        for name in ("sp500", "nasdaq")}}
    payload["series"]["vix"] = [row(current, 15)]
    payload["series"]["oil"] = [row(current - 7 * mf.DAY, 100), row(current, 140, NOW + 1)]
    snapshot = mf.Snapshot(payload)
    before = mf.decision_at(snapshot, NOW, "long", profile())
    assert before.allowed and before.features["oil_change"] is None
    assert before.ignored_features == ("oil",)
    after = mf.decision_at(snapshot, NOW + 1, "long", profile())
    assert not after.allowed and after.features["oil_change"] == pytest.approx(.4)
    assert after.reasons == ("market_stress",) and after.ignored_features == ()


def test_valid_optional_profile_roundtrips_and_changes_profile_hash(tmp_path):
    path = tmp_path / "profile.json"
    configured = profile()
    path.write_text(json.dumps(configured), encoding="utf-8")
    assert mf.load_profile(path) == configured
    assert mf.profile_hash(configured) != mf.profile_hash(profile(optional=False))


@pytest.mark.parametrize("features,effective", [
    (["vix"], NOW), (["oil", "vix"], NOW), ("oil", NOW), (None, NOW),
    (["oil", "oil"], NOW), (["oil"], None), (["oil"], 0), (["oil"], -1),
    (["oil"], True), (["oil"], 1.5), (["oil"], str(NOW)),
    (["oil"], 253_402_300_800_000), ([], NOW),
])
def test_invalid_optional_feature_policy_is_rejected(tmp_path, features, effective):
    configured = profile()
    configured["optional_features"] = features
    configured["optional_features_effective_at_ms"] = effective
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(configured), encoding="utf-8")
    with pytest.raises(ValueError):
        mf.load_profile(path)
