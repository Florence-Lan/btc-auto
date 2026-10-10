from __future__ import annotations

import sys
import json
import copy
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import llm_trade_gate


def target(leverage: float, position_id: str = "signal-1") -> dict[str, object]:
    return {
        "signal_time_ms": 1_000,
        "signal_price": 50_000.0,
        "target_leverage": leverage,
        "position_id": position_id,
        "origin_signal_time_ms": 900,
    }


class LLMTradeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = datetime(2026, 10, 9, tzinfo=timezone.utc)
        self.environment = mock.patch.dict(
            "os.environ",
            {
                "LLM_TRADE_GATE_ENABLED": "true",
                "LLM_MIN_CONFIDENCE": "0.70",
                "LLM_REJECTION_CACHE_SECONDS": "900",
            },
            clear=False,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        clock_patch = mock.patch.object(
            llm_trade_gate, "utc_now", side_effect=lambda: self.clock.isoformat()
        )
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        provider_patch = mock.patch.object(
            llm_trade_gate,
            "request_llm_decision",
            side_effect=AssertionError("tests must not invoke a real decision provider"),
        )
        provider_patch.start()
        self.addCleanup(provider_patch.stop)

    def advance(self, seconds: float) -> None:
        self.clock += timedelta(seconds=seconds)

    @staticmethod
    def response(allow: bool) -> dict[str, object]:
        return {
            "allow": allow,
            "confidence": 0.95,
            "reason": "Point-in-time risk review.",
            "risk_flags": [] if allow else ["risk"],
        }

    def apply(
        self,
        requested: float,
        *,
        current_qty: float = 0.0,
        provider=None,
        previous=None,
        position_id: str = "signal-1",
    ):
        with mock.patch.dict(
            "os.environ",
            {"LLM_TRADE_GATE_ENABLED": "true", "LLM_MIN_CONFIDENCE": "0.70"},
            clear=False,
        ):
            return llm_trade_gate.apply_llm_trade_gate(
                {"equity_curve": []},
                target(requested, position_id),
                current_qty=current_qty,
                equity=100.0,
                mark_price=50_000.0,
                mode="simulation",
                previous_decision=previous,
                decision_provider=provider,
            )

    def test_disabled_gate_leaves_target_unchanged(self) -> None:
        with mock.patch.dict("os.environ", {"LLM_TRADE_GATE_ENABLED": "false"}, clear=False):
            gated, decision = llm_trade_gate.apply_llm_trade_gate(
                {}, target(1.0), current_qty=0, equity=100, mark_price=50_000, mode="simulation"
            )
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(decision["status"], "disabled")

    def test_approved_entry_keeps_strategy_target(self) -> None:
        gated, decision = self.apply(
            1.0,
            provider=lambda _: {
                "allow": True,
                "confidence": 0.88,
                "reason": "Trend and risk context align.",
                "risk_flags": [],
            },
        )
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(decision["status"], "approved")

    def test_low_confidence_entry_is_rejected(self) -> None:
        gated, decision = self.apply(
            -1.0,
            provider=lambda _: {
                "allow": True,
                "confidence": 0.60,
                "reason": "Evidence is weak.",
                "risk_flags": ["weak_signal"],
            },
        )
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertEqual(decision["status"], "rejected")
        self.assertFalse(decision["allow"])

    def test_reduction_bypasses_llm(self) -> None:
        provider = mock.Mock(side_effect=AssertionError("provider must not be called"))
        gated, decision = self.apply(0.5, current_qty=0.002, provider=provider)
        self.assertEqual(gated["target_leverage"], 0.5)
        self.assertEqual(decision["status"], "bypassed_non_increasing")
        provider.assert_not_called()

    def test_rejected_reversal_can_flatten_existing_position(self) -> None:
        gated, decision = self.apply(
            -1.0,
            current_qty=0.002,
            provider=lambda _: {
                "allow": False,
                "confidence": 0.90,
                "reason": "Reversal is not confirmed.",
                "risk_flags": ["reversal"],
            },
        )
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertEqual(decision["status"], "rejected")

    def test_provider_error_blocks_new_exposure(self) -> None:
        gated, decision = self.apply(
            1.0,
            provider=mock.Mock(side_effect=TimeoutError("timed out")),
        )
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertEqual(decision["status"], "error_blocked")

    def test_rejected_position_id_uses_cached_decision(self) -> None:
        first_target, first = self.apply(
            1.0,
            provider=lambda _: {
                "allow": False,
                "confidence": 0.95,
                "reason": "Risk is elevated.",
                "risk_flags": ["risk"],
            },
        )
        self.advance(300)
        provider = mock.Mock(side_effect=AssertionError("cached decision expected"))
        second_target, second = self.apply(1.0, provider=provider, previous=first)
        self.assertEqual(first_target["target_leverage"], 0.0)
        self.assertEqual(second_target["target_leverage"], 0.0)
        self.assertEqual(second["status"], "cached_rejected")
        self.assertEqual(second["cache_version"], llm_trade_gate.DECISION_CACHE_VERSION)
        self.assertEqual(second["reviewed_at_utc"], first["reviewed_at_utc"])
        self.assertEqual(second["decided_at_utc"], first["decided_at_utc"])
        self.assertEqual(second["checked_at_utc"], self.clock.isoformat())
        provider.assert_not_called()

    def test_review_timestamps_record_provider_completion(self) -> None:
        for allow in (False, True):
            with self.subTest(allow=allow):
                started = self.clock.isoformat()

                def provider(_):
                    self.advance(47)
                    return self.response(allow)

                _, decision = self.apply(1.0, provider=provider)
                self.assertEqual(decision["cache_version"], llm_trade_gate.DECISION_CACHE_VERSION)
                self.assertEqual(decision["reviewed_at_utc"], self.clock.isoformat())
                self.assertEqual(decision["decided_at_utc"], self.clock.isoformat())
                self.assertNotEqual(decision["reviewed_at_utc"], started)

    def test_provider_failure_also_records_completion_time(self) -> None:
        def provider(_):
            self.advance(12)
            raise TimeoutError("timed out")

        gated, decision = self.apply(1.0, provider=provider)
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertEqual(decision["status"], "error_blocked")
        self.assertEqual(decision["cache_version"], llm_trade_gate.DECISION_CACHE_VERSION)
        self.assertEqual(decision["reviewed_at_utc"], self.clock.isoformat())
        self.assertEqual(decision["decided_at_utc"], self.clock.isoformat())

    def test_default_rejection_ttl_and_serialized_hits_do_not_extend_review(self) -> None:
        with mock.patch.dict("os.environ", clear=False):
            llm_trade_gate.os.environ.pop("LLM_REJECTION_CACHE_SECONDS", None)
            _, original = self.apply(1.0, provider=lambda _: self.response(False))
            forbidden = mock.Mock(side_effect=AssertionError("unexpired cache must be used"))
            previous = original
            for elapsed in (300, 599.999):
                self.advance(elapsed)
                previous = json.loads(json.dumps(previous))
                gated, previous = self.apply(1.0, provider=forbidden, previous=previous)
                self.assertEqual(gated["target_leverage"], 0.0)
                self.assertEqual(previous["reviewed_at_utc"], original["reviewed_at_utc"])
                self.assertEqual(previous["decided_at_utc"], original["decided_at_utc"])
                self.assertEqual(previous["checked_at_utc"], self.clock.isoformat())
            forbidden.assert_not_called()
            self.advance(0.001)
            provider = mock.Mock(return_value=self.response(True))
            gated, reviewed = self.apply(1.0, provider=provider, previous=previous)
        provider.assert_called_once()
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(reviewed["status"], "approved")
        self.assertNotEqual(reviewed["reviewed_at_utc"], original["reviewed_at_utc"])

    def test_fractional_ttl_and_maximum_ttl_expire_at_boundary(self) -> None:
        for ttl in (0.5, 3600):
            with self.subTest(ttl=ttl), mock.patch.dict(
                "os.environ", {"LLM_REJECTION_CACHE_SECONDS": str(ttl)}, clear=False
            ):
                _, first = self.apply(1.0, provider=lambda _: self.response(False))
                self.advance(ttl - 0.001)
                cached_provider = mock.Mock()
                blocked, cached = self.apply(1.0, provider=cached_provider, previous=first)
                cached_provider.assert_not_called()
                self.assertEqual(blocked["target_leverage"], 0.0)
                self.advance(0.001)
                provider = mock.Mock(return_value=self.response(True))
                gated, decision = self.apply(1.0, provider=provider, previous=cached)
                provider.assert_called_once()
                self.assertEqual(gated["target_leverage"], 1.0)
                self.assertEqual(decision["status"], "approved")

    def test_low_confidence_rejection_can_be_approved_after_ttl(self) -> None:
        gated, first = self.apply(
            -1.0,
            provider=lambda _: {**self.response(True), "confidence": 0.60},
        )
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertTrue(first["raw_allow"])
        self.assertFalse(first["allow"])
        self.advance(900)
        provider = mock.Mock(return_value=self.response(True))
        gated, approved = self.apply(-1.0, provider=provider, previous=first)
        provider.assert_called_once()
        self.assertEqual(gated["target_leverage"], -1.0)
        self.assertEqual(approved["status"], "approved")

    def test_invalid_rejection_cache_configuration_blocks_without_provider(self) -> None:
        for invalid in ("", " ", "0", "-1", "NaN", "inf", "-inf", "not-a-number", "3601", "3600.00001"):
            with self.subTest(value=invalid), mock.patch.dict(
                "os.environ", {"LLM_REJECTION_CACHE_SECONDS": invalid}, clear=False
            ):
                provider = mock.Mock(side_effect=AssertionError("invalid config must not invoke provider"))
                gated, decision = self.apply(1.0, provider=provider)
                provider.assert_not_called()
                self.assertEqual(gated["target_leverage"], 0.0)
                self.assertEqual(decision["status"], "error_blocked")
                self.assertFalse(decision["allow"])

    def test_invalid_cache_configuration_cannot_reuse_approval(self) -> None:
        _, approved = self.apply(1.0, provider=lambda _: self.response(True))
        with mock.patch.dict("os.environ", {"LLM_REJECTION_CACHE_SECONDS": "0"}, clear=False):
            provider = mock.Mock(side_effect=AssertionError("invalid config must not invoke provider"))
            gated, decision = self.apply(1.0, provider=provider, previous=approved)
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertEqual(decision["status"], "error_blocked")

    def test_reduction_bypasses_even_invalid_rejection_cache_configuration(self) -> None:
        with mock.patch.dict("os.environ", {"LLM_REJECTION_CACHE_SECONDS": "NaN"}, clear=False):
            provider = mock.Mock(side_effect=AssertionError("reduction must not invoke provider"))
            gated, decision = self.apply(0.5, current_qty=0.002, provider=provider)
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 0.5)
        self.assertEqual(decision["status"], "bypassed_non_increasing")

    def test_legacy_and_untrustworthy_rejection_review_times_are_reexamined(self) -> None:
        _, rejection = self.apply(1.0, provider=lambda _: self.response(False))
        cases = [
            {key: value for key, value in rejection.items() if key != "cache_version"},
            {**rejection, "cache_version": 0},
            {key: value for key, value in rejection.items() if key != "reviewed_at_utc"},
            {**rejection, "reviewed_at_utc": None},
            {**rejection, "reviewed_at_utc": ""},
            {**rejection, "reviewed_at_utc": "not-a-timestamp"},
            {**rejection, "reviewed_at_utc": self.clock.replace(tzinfo=None).isoformat()},
            {**rejection, "reviewed_at_utc": (self.clock + timedelta(seconds=1)).isoformat()},
        ]
        for previous in cases:
            with self.subTest(previous=previous):
                provider = mock.Mock(return_value=self.response(True))
                gated, decision = self.apply(1.0, provider=provider, previous=previous)
                provider.assert_called_once()
                self.assertEqual(gated["target_leverage"], 1.0)
                self.assertEqual(decision["status"], "approved")
                self.assertEqual(decision["cache_version"], llm_trade_gate.DECISION_CACHE_VERSION)
                self.assertEqual(decision["reviewed_at_utc"], self.clock.isoformat())

    def test_aware_review_time_is_compared_as_an_instant(self) -> None:
        _, rejection = self.apply(1.0, provider=lambda _: self.response(False))
        rejection["reviewed_at_utc"] = self.clock.astimezone(timezone(timedelta(hours=8))).isoformat()
        self.advance(899)
        provider = mock.Mock()
        gated, cached = self.apply(1.0, provider=provider, previous=rejection)
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 0.0)
        self.advance(1)
        provider = mock.Mock(return_value=self.response(True))
        gated, _ = self.apply(1.0, provider=provider, previous=cached)
        provider.assert_called_once()
        self.assertEqual(gated["target_leverage"], 1.0)

    def test_cached_approval_keeps_original_review_times_after_rejection_ttl(self) -> None:
        _, first = self.apply(1.0, provider=lambda _: self.response(True))
        self.advance(86_400)
        provider = mock.Mock(side_effect=AssertionError("approval is not subject to rejection TTL"))
        gated, cached = self.apply(1.0, provider=provider, previous=json.loads(json.dumps(first)))
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(cached["status"], "cached_approved")
        self.assertEqual(cached["reviewed_at_utc"], first["reviewed_at_utc"])
        self.assertEqual(cached["decided_at_utc"], first["decided_at_utc"])
        self.assertEqual(cached["checked_at_utc"], self.clock.isoformat())

    def test_increase_above_approved_limit_requires_new_review(self) -> None:
        _, approved = self.apply(1.0, provider=lambda _: self.response(True))
        self.advance(10)
        provider = mock.Mock(return_value=self.response(False))
        gated, decision = self.apply(2.0, current_qty=0.002, provider=provider, previous=approved)
        provider.assert_called_once()
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["reviewed_at_utc"], self.clock.isoformat())

    def test_legacy_and_future_approval_cache_require_a_new_provider_review(self) -> None:
        _, approval = self.apply(1.0, provider=lambda _: self.response(True))
        cases = [
            {key: value for key, value in approval.items() if key != "cache_version"},
            {**approval, "cache_version": 0},
            {key: value for key, value in approval.items() if key != "reviewed_at_utc"},
            {**approval, "reviewed_at_utc": (self.clock + timedelta(seconds=1)).isoformat()},
        ]
        for previous in cases:
            with self.subTest(previous=previous):
                provider = mock.Mock(return_value=self.response(False))
                gated, decision = self.apply(1.0, provider=provider, previous=previous)
                provider.assert_called_once()
                self.assertEqual(gated["target_leverage"], 0.0)
                self.assertEqual(decision["status"], "rejected")
                self.assertEqual(decision["review_trigger"], "untrusted_approval_cache")
                self.assertFalse(decision["allow"])
                self.assertEqual(decision["reviewed_at_utc"], self.clock.isoformat())

    def test_approval_can_cover_a_smaller_target_but_not_a_different_signal(self) -> None:
        _, approved = self.apply(1.0, provider=lambda _: self.response(True))
        self.advance(10)
        forbidden = mock.Mock(side_effect=AssertionError("target within approved limit should reuse review"))
        gated, cached = self.apply(0.8, provider=forbidden, previous=approved)
        forbidden.assert_not_called()
        self.assertEqual(gated["target_leverage"], 0.8)
        self.assertEqual(cached["status"], "cached_approved")
        self.assertEqual(cached["approved_target_leverage"], 1.0)
        provider = mock.Mock(return_value=self.response(False))
        gated, decision = self.apply(0.8, provider=provider, previous=cached, position_id="signal-2")
        provider.assert_called_once()
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertEqual(decision["status"], "rejected")
        self.assertNotEqual(decision["decision_key"], cached["decision_key"])

    def test_rejected_increase_preserves_current_exposure_during_cached_rejection(self) -> None:
        gated, rejected = self.apply(2.0, current_qty=0.002, provider=lambda _: self.response(False))
        self.assertEqual(gated["target_leverage"], 1.0)
        self.advance(100)
        provider = mock.Mock()
        gated, cached = self.apply(2.0, current_qty=0.002, provider=provider, previous=rejected)
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertFalse(cached["allow"])

    def test_cached_rejected_reversal_still_allows_flattening(self) -> None:
        _, rejected = self.apply(-1.0, current_qty=0.002, provider=lambda _: self.response(False))
        self.advance(100)
        provider = mock.Mock()
        gated, cached = self.apply(-1.0, current_qty=0.002, provider=provider, previous=rejected)
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 0.0)
        self.assertFalse(cached["allow"])

    def test_error_cache_expires_and_failed_rereview_remains_safe_then_retries(self) -> None:
        first_provider = mock.Mock(side_effect=TimeoutError("first timeout"))
        gated, failed = self.apply(2.0, current_qty=0.002, provider=first_provider)
        self.assertEqual(gated["target_leverage"], 1.0)
        self.advance(899)
        provider = mock.Mock()
        gated, cached = self.apply(2.0, current_qty=0.002, provider=provider, previous=failed)
        provider.assert_not_called()
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(cached["reviewed_at_utc"], failed["reviewed_at_utc"])
        self.advance(1)
        retry = mock.Mock(side_effect=TimeoutError("second timeout"))
        gated, failed_again = self.apply(2.0, current_qty=0.002, provider=retry, previous=cached)
        retry.assert_called_once()
        self.assertEqual(gated["target_leverage"], 1.0)
        self.assertEqual(failed_again["status"], "error_blocked")
        self.assertEqual(failed_again["reviewed_at_utc"], self.clock.isoformat())
        self.advance(899)
        provider = mock.Mock()
        _, cached_error = self.apply(2.0, current_qty=0.002, provider=provider, previous=failed_again)
        provider.assert_not_called()
        self.assertEqual(cached_error["reviewed_at_utc"], failed_again["reviewed_at_utc"])
        self.advance(1)
        recovered = mock.Mock(return_value=self.response(True))
        gated, approved = self.apply(2.0, current_qty=0.002, provider=recovered, previous=cached_error)
        recovered.assert_called_once()
        self.assertEqual(gated["target_leverage"], 2.0)
        self.assertEqual(approved["status"], "approved")

    def test_codex_provider_uses_read_only_structured_exec(self) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["input"] = kwargs["input"]
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps({
                "allow": True,
                "confidence": 0.91,
                "reason": "Aligned point-in-time evidence.",
                "risk_flags": [],
            }), encoding="utf-8")
            return mock.Mock(returncode=0, stderr="", stdout="")

        with (
            mock.patch.object(llm_trade_gate.shutil, "which", return_value="codex.exe"),
            mock.patch.object(llm_trade_gate.subprocess, "run", side_effect=fake_run),
        ):
            decision = llm_trade_gate.request_codex_decision({"symbol": "BTCUSDT"})
        self.assertEqual(decision["provider"], "codex")
        self.assertIn("read-only", captured["command"])
        self.assertIn('history.persistence="none"', captured["command"])
        self.assertIn("--output-schema", captured["command"])
        self.assertIn("Point-in-time trade context", captured["input"])

    def review_report(self):
        cutoff = int(self.clock.timestamp() * 1000)
        origin = self.clock - timedelta(days=2)
        opened = cutoff - llm_trade_gate.BASE_BAR_MS
        component = {"strategy": "timeseries_trend_6h", "side": "short",
                     "entry_time_utc": origin.isoformat(), "entry_price": 100,
                     "signal_reason": "confirmed slow trend short"}
        source = {
            "decision_asof_ms": cutoff,
            "generated_at_utc": (self.clock + timedelta(seconds=30)).isoformat(),
            "execution_target": {"time_ms": opened, "available_time_ms": cutoff - 1,
                                 "origin_signal_time_ms": int(origin.timestamp() * 1000),
                                 "origin_entry_price": 100, "components": [component]},
            "strategy_qualification": {"approved_for_forward_simulation": True,
                                       "historical_performance_required": False},
            "strategy_review_context": {"timeseries_trend": {
                "schema_version": 1, "strategy": "timeseries_trend", "status": "available",
                "timeframe": "1h", "decision_asof_ms": cutoff,
                "fast_ema_period_bars": 48, "slow_ema_period_bars": 240,
                "ema_spread_pct": -0.8, "entry_min_ema_spread_pct": 0.4,
                "spread_unit": "percent", "last_confirmed_side": "short",
                "current_target_side": "short", "target_side_valid_under_exit_rule": True,
                "trend_side_under_exit_rule": "short",
                "last_closed_bar": {"open_time_ms": cutoff - 3_600_000,
                                    "close_time_ms": cutoff - 1, "available_at_ms": cutoff - 1,
                                    "close_price": 104.8},
            }},
            "equity_curve": [{"time_ms": opened - (48 - i) * llm_trade_gate.BASE_BAR_MS,
                              "available_time_ms": cutoff - (48 - i) * llm_trade_gate.BASE_BAR_MS - 1,
                              "price": 100 + i * .1} for i in range(49)],
            "event_overlay": {"rate_expectation": {
                "available_at_utc": (self.clock - timedelta(seconds=5)).isoformat(),
                "quote_at_utc": (self.clock - timedelta(hours=1)).isoformat(),
                "meeting_at_utc": (self.clock + timedelta(days=20)).isoformat(),
                "expected_change_bps": 4.5,
            }},
            "multifactor_overlay": {"current": {"short": {"side_alignment": -0.005}},
                                    "data_metadata": {
                                        "generated_at_utc": (self.clock - timedelta(minutes=1)).isoformat(),
                                        "source_status": {"btc_close": {
                                            "latest_observed_at_ms": cutoff - 3_600_000,
                                            "last_success_ms": cutoff - 30_000}},
                                    }},
        }
        proposed = {"signal_time_ms": opened, "signal_price": 104.8,
                    "target_leverage": -.247, "position_id": "old-short",
                    "origin_signal_time_ms": int(origin.timestamp() * 1000)}
        return source, proposed

    def test_rate_available_after_signal_open_is_valid_before_current_decision(self) -> None:
        report, proposed = self.review_report()
        before = copy.deepcopy(report)
        context = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0,
                                                        mode="simulation")
        cutoff = context["decision_clock"]["decision_asof_ms"]
        rate = context["event_overlay"]["rate_expectation"]
        available = llm_trade_gate._timestamp_ms(rate["available_at_utc"])
        self.assertGreater(available, context["decision_clock"]["signal_candle_open_ms"])
        self.assertLess(available, cutoff)
        self.assertEqual(rate["expected_change_bps"], 4.5)
        self.assertGreater(llm_trade_gate._timestamp_ms(rate["meeting_at_utc"]), cutoff)
        self.assertEqual(context["input_integrity"]["excluded_inputs"], [])
        self.assertEqual(context["components"][0]["strategy_context"]["timeframe"], "1h")
        self.assertEqual(context["components"][0]["strategy_context"]["slow_ema_period_bars"], 240)
        self.assertEqual(context["delayed_entry"]["origin_signal_age_seconds"], 2 * 86400)
        self.assertAlmostEqual(context["delayed_entry"]["adverse_move_from_origin_pct"], 4.8)
        self.assertEqual(report, before)

    def test_post_cutoff_rate_and_price_cannot_leak_when_wall_clock_is_later(self) -> None:
        report, proposed = self.review_report()
        baseline = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0,
                                                        mode="simulation")
        cutoff = report["decision_asof_ms"]
        report["event_overlay"]["rate_expectation"]["available_at_utc"] = (
            self.clock + timedelta(seconds=5)).isoformat()
        report["event_overlay"]["rate_expectation"]["expected_change_bps"] = -999
        report["equity_curve"].append({"time_ms": cutoff, "available_time_ms": cutoff + 300000,
                                      "price": 1_000_000})
        self.advance(3600)
        context = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0,
                                                        mode="simulation")
        self.assertNotIn("rate_expectation", context["event_overlay"])
        self.assertEqual(context["decision_clock"]["decision_asof_ms"], cutoff)
        self.assertEqual(context["recent_market"]["returns_pct"], baseline["recent_market"]["returns_pct"])
        self.assertEqual(context["recent_market"]["excluded_price_observations"], 1)
        self.assertTrue(any(item["path"] == "event_overlay.rate_expectation" and
                            item["reason"] == "after_decision_cutoff"
                            for item in context["input_integrity"]["excluded_inputs"]))

    def test_price_windows_are_sorted_closed_and_do_not_turn_gaps_into_shorter_returns(self) -> None:
        report, _ = self.review_report()
        baseline = llm_trade_gate._recent_price_features(report)
        report["equity_curve"].reverse()
        self.assertEqual(llm_trade_gate._recent_price_features(report), baseline)
        last_open = max(p["time_ms"] for p in report["equity_curve"])
        report["equity_curve"] = [p for p in report["equity_curve"]
                                  if p["time_ms"] != last_open - llm_trade_gate.BASE_BAR_MS]
        report["equity_curve"].append({"time_ms": last_open + llm_trade_gate.BASE_BAR_MS,
                                      "available_time_ms": report["decision_asof_ms"] + 1,
                                      "price": 0.1})
        features = llm_trade_gate._recent_price_features(report)
        self.assertIsNone(features["returns_pct"]["5m"])
        self.assertEqual(features["returns_pct"]["15m"], baseline["returns_pct"]["15m"])
        self.assertEqual(features["excluded_price_observations"], 1)

    def test_missing_cutoff_never_uses_current_wall_clock(self) -> None:
        report, proposed = self.review_report()
        report.pop("decision_asof_ms")
        report.pop("generated_at_utc")
        self.advance(86400)
        context = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0,
                                                        mode="simulation")
        self.assertTrue(context["decision_clock"]["cutoff_missing"])
        self.assertIsNone(context["decision_clock"]["decision_asof_ms"])
        self.assertEqual(context["recent_market"]["observations"], 0)
        self.assertNotIn("rate_expectation", context["event_overlay"])
        report["generated_at_utc"] = (self.clock - timedelta(days=1)).isoformat()
        fallback = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0,
                                                         mode="simulation")
        self.assertEqual(fallback["decision_clock"]["cutoff_source"], "report.generated_at_utc_fallback")
        self.assertEqual(fallback["recent_market"]["observations"], 49)

    def test_future_indicator_or_factor_snapshot_does_not_keep_derived_values(self) -> None:
        report, proposed = self.review_report()
        report["strategy_review_context"]["timeseries_trend"]["last_closed_bar"]["available_at_ms"] += 10
        report["multifactor_overlay"]["data_metadata"]["generated_at_utc"] = (
            self.clock + timedelta(seconds=1)).isoformat()
        context = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0,
                                                        mode="simulation")
        self.assertEqual(context["strategy_review_context"], {})
        self.assertEqual(context["components"][0]["strategy_context"]["status"], "unavailable")
        self.assertEqual(context["multifactor_overlay"], {})

    def test_forward_policy_retains_opposing_returns_but_does_not_force_approval(self) -> None:
        report, proposed = self.review_report()
        captured = []

        def reject_current_risk(context):
            captured.append(context)
            return {**self.response(False), "reason": "Current target invalidated by opposite confirmed trend."}

        gated, decision = llm_trade_gate.apply_llm_trade_gate(
            report, proposed, current_qty=0, equity=1000, mark_price=104.8, mode="simulation",
            decision_provider=reject_current_risk)
        context = captured[0]
        self.assertGreater(context["recent_market"]["returns_pct"]["5m"], 0)
        self.assertLess(context["multifactor_overlay"]["current"]["short"]["side_alignment"], 0)
        self.assertFalse(context["admission_policy"]["weak_short_horizon_opposition_alone_is_veto"])
        self.assertFalse(context["admission_policy"]["near_neutral_factor_score_alone_is_veto"])
        self.assertFalse(context["delayed_entry"]["age_alone_is_veto"])
        self.assertIn("not entry vetoes", llm_trade_gate.SYSTEM_INSTRUCTIONS)
        self.assertIn("supplied value and units", llm_trade_gate.SYSTEM_INSTRUCTIONS)
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(gated["target_leverage"], 0)
        self.assertEqual(decision["minimum_confidence"], .70)
        live = llm_trade_gate.build_decision_context(report, proposed, current_leverage=0, mode="live")
        self.assertTrue(live["admission_policy"]["historical_performance_required"])
        self.assertEqual(live["admission_policy"]["review_standard"], "conservative_historical_and_current_evidence")

    def test_prior_policy_caches_are_reevaluated_and_audit_actual_context(self) -> None:
        report, proposed = self.review_report()
        for allow in (False, True):
            with self.subTest(allow=allow):
                prior = {"cache_version": 2, "allow": allow, "decision_key": "old-short:short",
                         "approved_target_leverage": -.247, "reviewed_at_utc": self.clock.isoformat()}
                provider = mock.Mock(return_value=self.response(True))
                gated, decision = llm_trade_gate.apply_llm_trade_gate(
                    report, proposed, current_qty=0, equity=1000, mark_price=104.8, mode="simulation",
                    previous_decision=prior, decision_provider=provider)
                provider.assert_called_once()
                self.assertEqual(gated["target_leverage"], -.247)
                self.assertEqual(decision["cache_version"], llm_trade_gate.DECISION_CACHE_VERSION)
                self.assertEqual(decision["review_prompt_sha256"], llm_trade_gate.REVIEW_PROMPT_SHA256)
                self.assertEqual(decision["review_decision_clock"]["decision_asof_ms"], report["decision_asof_ms"])
                self.assertEqual(len(decision["review_context_sha256"]), 64)

    def test_prompt_or_admission_changes_cannot_reuse_an_approval(self) -> None:
        report, proposed = self.review_report()
        _, approved = llm_trade_gate.apply_llm_trade_gate(
            report, proposed, current_qty=0, equity=1000, mark_price=104.8, mode="simulation",
            decision_provider=lambda _: self.response(True))
        cases = [
            ({**approved, "review_prompt_sha256": "old-prompt"}, report, "simulation"),
            ({**approved, "review_context_version": 0}, report, "simulation"),
            ({**approved, "admission_policy_version": 0}, report, "simulation"),
            (approved, report, "live"),
            (approved, {**report, "strategy_qualification": {}}, "simulation"),
        ]
        for prior, source, mode in cases:
            with self.subTest(mode=mode, historical=source.get("strategy_qualification")):
                provider = mock.Mock(return_value=self.response(False))
                gated, decision = llm_trade_gate.apply_llm_trade_gate(
                    source, proposed, current_qty=0, equity=1000, mark_price=104.8, mode=mode,
                    previous_decision=prior, decision_provider=provider)
                provider.assert_called_once()
                self.assertEqual(gated["target_leverage"], 0)
                self.assertEqual(decision["status"], "rejected")
                self.assertEqual(decision["review_execution_mode"], mode)


if __name__ == "__main__":
    unittest.main()
