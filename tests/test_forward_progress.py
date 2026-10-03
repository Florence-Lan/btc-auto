"""Audit accounting, archive replay and strict separation of account/model samples."""
import gzip
import json

import pytest

import multifactor
import report_forward_progress as audit


EPOCH = "2026-10-01T00:00:00+00:00"
START = "2026-10-02T00:00:00+00:00"
CHANGE = "2026-10-03T00:00:00+00:00"
END = "2026-10-04T00:00:00+00:00"


def fill(sequence, time, side="BUY", quantity=1, fee=1, pnl=0):
    return {"sequence": sequence, "time_utc": time, "side": side, "quantity": quantity,
            "price": 100, "fee": fee, "realized_pnl": pnl, "mode": "simulation"}


def account(fills=(), funding=()):
    qty = sum(row["quantity"] * (1 if row["side"] == "BUY" else -1) for row in fills)
    realized, fees = sum(row["realized_pnl"] for row in fills), sum(row["fee"] for row in fills)
    funding_pnl = sum(row["payment"] for row in funding)
    return {"created_at_utc": EPOCH, "updated_at_utc": END, "initial_balance": 1000,
            "wallet_balance": 1000 + realized - fees + funding_pnl, "position_qty": qty,
            "entry_price": 100 if qty else 0, "last_mark_price": 100,
            "realized_pnl": realized, "fees_paid": fees, "funding_pnl": funding_pnl,
            "fill_count_total": len(fills), "fills": list(fills)[-200:],
            "funding_settlements": list(funding), "position_history": [], "equity_curve": [],
            "max_drawdown_pct": 0}


def packed(account_state=None):
    profile = {"candidate_id": "trial", "live_orders_allowed": False,
               "groups": {"btc_momentum": .2, "positioning": .2, "fed": .15,
                          "treasury": .15, "fx": .15, "global_risk": .15},
               "minimum_group_coverage": 1, "minimum_risk_multiplier": .35,
               "block_alignment_below": -.35, "block_stress_at": .9,
               "availability_mode": "first_seen", "policy_expectations_enabled": False,
               "event_snapshot": "data/snapshots/public_context_latest.json"}
    digest = multifactor.profile_hash(profile)
    state = {"created_at_utc": START, "observations": 1, "profile": {"factor_profile_sha256": digest},
             "execution_model_changes": [{"at_utc": CHANGE, "model": "corrected", "previous_profile": {}}]}
    report = {"paper_inception_utc": START, "generated_at_utc": END, "execution_model": "corrected",
              "summary": {"initial_equity": 100}, "multifactor_overlay": {"candidate_id": "trial"},
              "execution_entry_context": {"factor_profile_sha256": digest}}
    payloads = {"profile": profile, "account": account_state or account(), "paper_state": state,
                "report": report, "public": {"schema_version": 1, "events": [], "coverage_checks": []}}
    inputs = {key: json.dumps(value).encode() for key, value in payloads.items()}
    inputs.update(factors=gzip.compress(json.dumps({"schema_version": 1, "series": {}, "metadata": {}}).encode()),
                  journal=None, scheduler_log=None)
    return {"asof_utc": END}, inputs


def test_complete_journal_recovers_tail_and_ignores_other_epochs():
    rows = [fill(i, f"2026-10-01T00:{(i - 1) // 60:02d}:{(i - 1) % 60:02d}+00:00") for i in range(1, 206)]
    state = account(rows)
    foreign = {**fill(1, EPOCH), "account_epoch": "old"}
    journal = "\n".join(json.dumps({**row, "account_epoch": EPOCH}) for row in rows)
    journal += "\n" + json.dumps(foreign)
    actual, integrity = audit.complete_fills(state, journal.encode())
    assert len(actual) == 205 and integrity["complete"]
    assert integrity["other_or_unidentified_epoch_journal_rows_ignored"] == 1
    assert not audit.complete_fills(state, None)[1]["complete"]


def test_conflicting_sequence_and_missing_prefix_disable_metrics():
    state = account([fill(1, START), fill(2, CHANGE, "SELL")])
    journal = json.dumps({**fill(1, START, fee=7), "account_epoch": EPOCH}).encode()
    _, integrity = audit.complete_fills(state, journal)
    assert "conflicting_fill_sequence" in integrity["issues"]
    state["fills"] = state["fills"][1:]
    assert not audit.complete_fills(state, None)[1]["complete"]
    result = audit.build_report(*packed(state))
    assert result["candidate_metrics"]["fills"] is None
    assert result["candidate_metrics"]["closed_episode_profit_factor"] is None


def test_reversal_partial_exit_funding_and_same_timestamp_risk_close():
    rows = [fill(1, START, quantity=2, fee=2),
            fill(2, "2026-10-02T12:00:00+00:00", "SELL", 1, 1, 10),
            fill(3, CHANGE, "SELL", 2, 2, -5),
            fill(4, CHANGE, "BUY", 1, 1, 2)]
    settlement = {"time_ms": audit.ms(CHANGE), "signed_qty": 1, "payment": -3}
    state = account(rows, [settlement])
    state["position_history"] = [{"time_ms": audit.ms(row["time_utc"]), "signed_qty": qty}
                                 for row, qty in zip(rows, [2, 1, -1, 0])]
    episodes = audit.reconstruct(state, rows)
    assert len(episodes) == 2 and all(row["closed"] for row in episodes)
    assert episodes[0]["fees"] == 4 and episodes[0]["funding_pnl"] == -3
    assert episodes[0]["net_realized_pnl"] == -2
    assert episodes[1]["fees"] == 2 and episodes[1]["net_realized_pnl"] == 0
    assert sum(row["net_realized_pnl"] for row in episodes) == state["wallet_balance"] - 1000


def test_boundary_crossing_episode_does_not_count_as_new_model_sample():
    rows = [fill(1, START), fill(2, "2026-10-03T01:00:00+00:00", "SELL", pnl=20)]
    state = account(rows)
    result = audit.build_report(*packed(state))
    assert result["candidate_metrics"]["closed_net_position_episodes"] == 1
    latest = result["execution_model_segments"][-1]["metrics"]
    assert latest["fills"] == 1 and latest["closed_net_position_episodes"] == 0
    assert latest["boundary_crossing_episodes_excluded"] == 1
    assert latest["closed_episode_profit_factor"] is None
    assert result["evaluation"]["minimum_gate_status"] == "not_met"


def test_zero_account_and_three_clocks_exclude_shadow_equity_and_validation():
    result = audit.build_report(*packed())
    assert result["account_epoch_elapsed_calendar_days"] == 3
    assert result["candidate_elapsed_calendar_days"] == 2
    assert result["execution_model_segments"][-1]["metrics"]["elapsed_calendar_days"] == 1
    assert result["account"]["equity_usdt_at_last_mark"] == 1000
    assert result["production_report"]["shadow_initial_equity_excluded"] == 100
    assert result["candidate_metrics"]["closed_net_position_episodes"] == 0
    assert not result["evaluation"]["profitability_validated"]
    assert not result["evaluation"]["gates"]["closed_episode_profit_factor_at_least_1_3"]


def test_funding_inventory_mismatch_is_reported_not_invented():
    state = account([fill(1, START)], [{"time_ms": audit.ms(CHANGE), "signed_qty": 0, "payment": 0}])
    result = audit.build_report(*packed(state))
    assert not result["ledger_integrity"]["complete"]
    assert "Funding inventory" in result["ledger_integrity"]["issues"][0]
    assert result["candidate_metrics"]["fees_usdt"] is None


def test_archive_replay_verifies_hashes_and_uses_original_asof(tmp_path):
    _, inputs = packed()
    directory = tmp_path / "archive"
    (directory / "inputs").mkdir(parents=True)
    records = {}
    for key, raw in inputs.items():
        path = f"inputs/{key}.bin"
        if raw is not None:
            (directory / path).write_bytes(raw)
        records[key] = {"present": raw is not None, "archive_file": path,
                        "sha256": audit.hashlib.sha256(raw).hexdigest() if raw is not None else None}
    (directory / "captured_inputs.json").write_text(json.dumps({"asof_utc": END, "inputs": records}))
    manifest, restored = audit.read_archive(directory)
    assert audit.build_report(manifest, restored) == audit.build_report(manifest, inputs)
    (directory / "inputs/account.bin").write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.read_archive(directory)


def test_asof_and_mismatched_profile_fail_closed():
    manifest, inputs = packed()
    manifest["asof_utc"] = CHANGE
    with pytest.raises(ValueError, match="As-of"):
        audit.build_report(manifest, inputs)
    manifest["asof_utc"] = END
    profile = json.loads(inputs["profile"])
    profile["minimum_group_coverage"] = .8
    inputs["profile"] = json.dumps(profile).encode()
    with pytest.raises(ValueError, match="do not match"):
        audit.build_report(manifest, inputs)


def test_monitor_only_counts_archived_checks_for_current_epoch():
    rows = [{"account_epoch": EPOCH, "checked_at_ms": audit.ms(CHANGE), "status": "healthy", "assessment": "evaluated"},
            {"account_epoch": EPOCH, "checked_at_ms": audit.ms(CHANGE) + 30_000, "status": "unavailable", "assessment": "not_evaluated", "errors": ["SSLError"]},
            {"account_epoch": "old", "checked_at_ms": audit.ms(CHANGE), "status": "healthy"}]
    raw = "\n".join(json.dumps(row) for row in rows).encode()
    result = audit.monitor_history(raw, EPOCH, audit.ms(START), audit.ms(END))
    assert result["recorded_checks"] == 2 and result["other_account_epoch_records_ignored"] == 1
    assert result["status_counts"] == {"healthy": 1, "unavailable": 1}
    assert len(result["failed_checks"]) == 1
    assert result["first_archived_check_utc"] == CHANGE


def test_capture_only_writes_new_audit_directory_and_replays_exact_bytes(tmp_path):
    _, inputs = packed()
    root = tmp_path / "repo"
    paths = {"profile": "config/trial.json", "report": "data/paper_trading/trial_report.json",
             "paper_state": "data/paper_trading/trial_state.json",
             "account": "data/runtime/simulation_account_20260917.json",
             "factors": "data/snapshots/multifactor_latest.json.gz",
             "public": "data/snapshots/public_context_latest.json"}
    for key, relative in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(inputs[key])
    selection = root / "config/active_simulation_candidate.json"
    selection.write_text(json.dumps({"candidate_path": "config/trial.json", "execution_mode": "simulation"}))
    original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    output = root / "data/validation/forward"
    manifest, captured = audit.capture(root, output, END)
    result = audit.build_report(manifest, captured)
    restored = audit.read_archive(output)
    assert audit.build_report(*restored) == result
    assert all(path.read_bytes() == raw for path, raw in original.items())
    assert (output / "code/report_forward_progress.py").exists()
    with pytest.raises(ValueError, match="already exists"):
        audit.capture(root, output, END)
