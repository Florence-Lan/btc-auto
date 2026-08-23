from __future__ import annotations

import sys
import json
import unittest
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
    def apply(
        self,
        requested: float,
        *,
        current_qty: float = 0.0,
        provider=None,
        previous=None,
    ):
        with mock.patch.dict(
            "os.environ",
            {"LLM_TRADE_GATE_ENABLED": "true", "LLM_MIN_CONFIDENCE": "0.70"},
            clear=False,
        ):
            return llm_trade_gate.apply_llm_trade_gate(
                {"equity_curve": []},
                target(requested),
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
        provider = mock.Mock(side_effect=AssertionError("cached decision expected"))
        second_target, second = self.apply(1.0, provider=provider, previous=first)
        self.assertEqual(first_target["target_leverage"], 0.0)
        self.assertEqual(second_target["target_leverage"], 0.0)
        self.assertEqual(second["status"], "cached_rejected")
        provider.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
