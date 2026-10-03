"""Read-only forward-trial audit; capture immutable inputs without refreshing or executing.

python scripts/report_forward_progress.py
python scripts/report_forward_progress.py --from-archive <previous output directory>
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import event_risk
import multifactor
import public_context

ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000
STEP_MS = 300_000


def ms(value):
    return event_risk._utc_ms(value)


def iso(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite account record")
    return result


def load_payload(raw, compressed=False):
    return json.loads(gzip.decompress(raw) if compressed else raw)


def capture(root, output, asof_utc=None):
    """Capture byte contents once. Writes are restricted to a new audit directory."""
    if output.exists():
        raise ValueError("Output directory already exists; choose a new audit directory")
    selected_path = root / "config/active_simulation_candidate.json"
    selected_raw = selected_path.read_bytes()
    selected = load_payload(selected_raw)
    profile_path = root / selected["candidate_path"]
    profile_raw = profile_path.read_bytes()
    profile = load_payload(profile_raw)
    if selected.get("execution_mode") != "simulation" or profile.get("live_orders_allowed") is not False:
        raise ValueError("This report supports simulation-only candidates")
    candidate = profile["candidate_id"]
    report_path = root / f"data/paper_trading/{candidate}_report.json"
    report_raw = report_path.read_bytes()
    report = load_payload(report_raw)
    context = report.get("execution_entry_context") or {}
    sources = {
        "selection": (selected_path, selected_raw),
        "profile": (profile_path, profile_raw),
        "report": (report_path, report_raw),
        "paper_state": (root / f"data/paper_trading/{candidate}_state.json", None),
        "account": (root / "data/runtime/simulation_account_20260917.json", None),
        "journal": (root / "data/runtime/simulation_account_20260917.fills.jsonl", None),
        "factors": (Path(context.get("factor_snapshot") or root / "data/snapshots/multifactor_latest.json.gz"), None),
        "public": (Path(context.get("event_snapshot") or root / profile["event_snapshot"]), None),
        "cooldown": (root / "data/runtime/binance_api_cooldown.json", None),
        "scheduler_log": (root / "data/runtime/execution_supervisor_stdout.log", None),
        "risk_monitor": (root / "data/runtime/simulation_risk_monitor.json", None),
        "risk_monitor_history": (root / "data/runtime/simulation_risk_monitor.jsonl", None),
    }
    inputs, records = {}, {}
    for key, (path, raw) in sources.items():
        if not path.is_absolute():
            path = root / path
        if raw is None:
            raw = path.read_bytes() if path.exists() else None
        inputs[key] = raw
        records[key] = {"source_path": str(path), "present": raw is not None,
                        "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                        "archive_file": f"inputs/{key}{'.json.gz' if key == 'factors' else '.bin'}"}
    asof = asof_utc or datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for key, raw in inputs.items():
        if raw is not None:
            (output / records[key]["archive_file"]).write_bytes(raw)
    manifest = {"schema_version": 1, "asof_utc": asof, "inputs": records,
                "capture_note": "Sequential read-only capture, not an atomic production checkpoint; sequence counters exclude journal records newer than the captured account."}
    (output / "code").mkdir()
    manifest["code_resources"] = {}
    for path in (Path(__file__), Path(multifactor.__file__), Path(event_risk.__file__), Path(public_context.__file__)):
        raw = path.read_bytes()
        archive_file = "code/" + path.name
        (output / archive_file).write_bytes(raw)
        manifest["code_resources"][path.name] = {"archive_file": archive_file, "sha256": hashlib.sha256(raw).hexdigest()}
    (output / "captured_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, inputs


def read_archive(directory):
    manifest = json.loads((directory / "captured_inputs.json").read_text())
    inputs = {}
    for key, row in manifest["inputs"].items():
        raw = (directory / row["archive_file"]).read_bytes() if row["present"] else None
        if raw is not None and hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Archived input hash mismatch: {key}")
        inputs[key] = raw
    for name, row in manifest.get("code_resources", {}).items():
        if hashlib.sha256((directory / row["archive_file"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Archived code hash mismatch: {name}")
    return manifest, inputs


def complete_fills(account, journal_raw):
    epoch, expected = account["created_at_utc"], int(account["fill_count_total"])
    rows, issues, foreign = {}, [], 0
    records = list(account.get("fills", []))
    if journal_raw:
        for line in journal_raw.decode().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("account_epoch") != epoch:
                    foreign += 1
                    continue
                records.append(row)
            except (ValueError, TypeError):
                issues.append("malformed_journal_record")
    for raw in records:
        row = {k: v for k, v in raw.items() if k != "account_epoch"}
        sequence = row.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            issues.append("fill_sequence_missing_or_invalid")
            continue
        if sequence > expected:
            continue  # Journal can advance after the captured account read.
        if sequence in rows and rows[sequence] != row:
            issues.append("conflicting_fill_sequence")
        rows[sequence] = row
    ordered = [rows[key] for key in sorted(rows)]
    if expected < 0 or list(sorted(rows)) != list(range(1, expected + 1)):
        issues.append("full_account_epoch_fill_history_missing")
    if any(ms(row["time_utc"]) < ms(epoch) for row in ordered):
        issues.append("fill_before_account_epoch")
    if any(ms(a["time_utc"]) > ms(b["time_utc"]) for a, b in zip(ordered, ordered[1:])):
        issues.append("fill_time_not_monotonic")
    return ordered, {"complete": not issues, "issues": sorted(set(issues)),
                     "account_counter": expected, "verified_rows": len(ordered),
                     "other_or_unidentified_epoch_journal_rows_ignored": foreign,
                     "source": "account tail plus sequence/epoch-matched append-only journal"}


def reconstruct(account, fills):
    """Same-direction net episodes; split reversal fees and settle funding before fills."""
    events = [(ms(row["time_utc"]), 1, row["sequence"], row) for row in fills]
    events += [(int(row["time_ms"]), 0, i, row) for i, row in enumerate(account.get("funding_settlements", []))]
    qty, active, episodes = Decimal(0), None, []
    fees = realized = funding = 0.0
    last_funding = set()
    positions = []
    for timestamp, kind, _, row in sorted(events, key=lambda item: item[:3]):
        if kind == 0:
            if timestamp in last_funding:
                raise ValueError("Duplicate funding settlement")
            last_funding.add(timestamp)
            payment = finite(row["payment"])
            if abs(float(qty) - finite(row["signed_qty"])) > 1e-9:
                raise ValueError("Funding inventory does not match complete fill path")
            if active:
                active["funding_pnl"] += payment
            elif abs(payment) > 1e-8:
                raise ValueError("Nonzero funding without a reconstructed position")
            funding += payment
            continue
        if row["side"] not in {"BUY", "SELL"}:
            raise ValueError("Invalid fill side")
        amount = finite(row["quantity"])
        if amount <= 0 or finite(row["price"]) <= 0:
            raise ValueError("Invalid fill quantity or price")
        delta = Decimal(str(row["quantity"])) * (1 if row["side"] == "BUY" else -1)
        new_qty = qty + delta
        fee, gross = finite(row["fee"]), finite(row["realized_pnl"])
        if fee < 0:
            raise ValueError("Negative fee")
        closing = min(abs(qty), abs(delta)) if qty * delta < 0 else Decimal(0)
        fee_close = fee * float(closing / abs(delta))
        if active:
            active["realized_pnl"] += gross
            active["fees"] += fee if qty * new_qty > 0 else fee_close
        elif abs(gross) > 1e-8:
            raise ValueError("Realized PnL without prior inventory")
        if active and qty * new_qty <= 0:
            active.update(exit_ms=timestamp, closed=True)
            episodes.append(active)
            active = None
        if new_qty and active is None:
            active = {"entry_ms": timestamp, "side": "long" if new_qty > 0 else "short",
                      "realized_pnl": 0.0, "fees": fee - fee_close, "funding_pnl": 0.0, "closed": False}
        qty = new_qty
        positions.append((timestamp, float(qty)))
        fees += fee
        realized += gross
    if active:
        episodes.append(active)
    for row in episodes:
        row["net_realized_pnl"] = row["realized_pnl"] - row["fees"] + row["funding_pnl"]
    expected = {"position_qty": float(qty), "fees_paid": fees, "realized_pnl": realized,
                "funding_pnl": funding,
                "wallet_balance": finite(account["initial_balance"]) + realized - fees + funding}
    for key, actual in expected.items():
        if abs(actual - finite(account[key])) > 1e-6:
            raise ValueError(f"Account reconciliation mismatch: {key}")
    history = account.get("position_history", [])
    if history and len(history) != len(positions):
        raise ValueError("Position history does not cover the complete fill path")
    for row, (timestamp, quantity) in zip(history, positions):
        if int(row["time_ms"]) != timestamp or abs(quantity - finite(row["signed_qty"])) > 1e-9:
            raise ValueError("Position history does not match complete fill path")
    return episodes


def period_metrics(account, fills, episodes, start, end, ledger_complete):
    selected = [row for row in fills if start <= ms(row["time_utc"]) < end]
    closed = [row for row in episodes if row["closed"] and start <= row["entry_ms"] and row["exit_ms"] < end]
    crossings = [row for row in episodes if row["entry_ms"] < start and row.get("exit_ms", end) >= start]
    points = [row for row in account.get("equity_curve", []) if start <= int(row["time_ms"]) < end]
    peak, dd = None, 0.0
    for row in points:
        equity = finite(row["equity"])
        peak = max(peak or equity, equity)
        if peak > 0:
            dd = max(dd, 100 * (1 - equity / peak))
    wins = sum(max(0.0, row["net_realized_pnl"]) for row in closed)
    losses = sum(max(0.0, -row["net_realized_pnl"]) for row in closed)
    return {"elapsed_calendar_days": max(0, end - start) / DAY_MS,
            "fills": len(selected) if ledger_complete else None,
            "closed_net_position_episodes": len(closed) if ledger_complete else None,
            "boundary_crossing_episodes_excluded": len(crossings) if ledger_complete else None,
            "closed_episode_profit_factor": wins / losses if ledger_complete and losses else None,
            "closed_net_episode_pnl_usdt": sum(row["net_realized_pnl"] for row in closed) if ledger_complete else None,
            "fees_usdt": sum(finite(row["fee"]) for row in selected) if ledger_complete else None,
            "funding_pnl_usdt": sum(finite(row["payment"]) for row in account.get("funding_settlements", [])
                                    if start <= int(row["time_ms"]) < end) if ledger_complete else None,
            "retained_execution_marks": len(points),
            "first_retained_mark_utc": iso(int(points[0]["time_ms"])) if points else None,
            "last_retained_mark_utc": iso(int(points[-1]["time_ms"])) if points else None,
            "retained_marks_drawdown_pct_lower_bound": dd if points else None,
            "drawdown_note": "At most 1,000 strategy execution marks are retained. This lower bound omits earlier marks, boundary equity and prices during disconnections; it is not a complete model-segment drawdown."}


def segments(state, report, inception, end):
    changes = sorted((row for row in state.get("execution_model_changes", []) if inception < ms(row["at_utc"]) < end), key=lambda row: ms(row["at_utc"]))
    previous = (changes[0].get("previous_profile", {}).get("hourly_execution_model") or "unrecorded_previous_model") if changes else report.get("execution_model", "unrecorded_model")
    output, start, model = [], inception, previous
    for row in changes:
        boundary = ms(row["at_utc"])
        output.append({"model": model, "start_ms": start, "end_ms": boundary})
        start, model = boundary, row["model"]
    output.append({"model": model, "start_ms": start, "end_ms": end})
    return output


def coverage(factors, public, profile, start, end):
    events = event_risk.events_from_payload(public)
    counts, groups, failures, gap_runs = Counter(), Counter(), Counter(), []
    active_gap = None
    first = ((start + STEP_MS - 1) // STEP_MS) * STEP_MS
    for timestamp in range(first, end, STEP_MS):
        decision = multifactor.decision_at(factors, timestamp, "long", profile, public)
        healthy, missing = public_context.health_at(public, timestamp)
        event = event_risk.event_decision_at(events, timestamp)
        counts["grid_points"] += 1
        counts["all_factor_groups_available"] += not decision.missing_groups
        counts["public_sources_healthy"] += healthy
        side_allowed = decision.allowed or (decision.coverage + 1e-12 >= profile["minimum_group_coverage"]
                           and decision.stress < profile["block_stress_at"]
                           and -decision.directional_score >= profile["block_alignment_below"])
        counts["at_least_one_side_permitted"] += healthy and event.allowed and side_allowed
        groups.update(decision.contributions.keys())
        reasons = tuple(sorted(set(decision.missing_groups) | set(missing)))
        failures.update(reasons)
        if reasons != (active_gap or {}).get("reasons"):
            if active_gap:
                active_gap["ends_at_utc"] = iso(timestamp)
                gap_runs.append(active_gap)
            active_gap = {"starts_at_utc": iso(timestamp), "reasons": reasons} if reasons else None
    if active_gap:
        active_gap["ends_at_utc"] = iso(end)
        gap_runs.append(active_gap)
    total = counts["grid_points"]
    checks = [row for row in public.get("coverage_checks", []) if start <= ms(row["available_at_utc"]) < end]
    bad = [{"available_at_utc": row["available_at_utc"],
            "failed_sources": {key: value for key, value in row["sources"].items() if not value.get("ok")}}
           for row in checks if any(not row["sources"].get(key, {}).get("ok") for key in public_context.REQUIRED)]
    return {"sampling_seconds": STEP_MS // 1000, "counts": dict(counts),
            "pct": {key: 100 * value / total if total else None for key, value in counts.items() if key != "grid_points"},
            "group_available_pct": {key: 100 * groups[key] / total if total else None for key in profile["groups"]},
            "missing_reason_grid_counts": dict(failures), "availability_gap_runs": gap_runs,
            "archived_public_health_checks": len(checks), "public_source_failures": bad,
            "definition": "Five-minute first-seen availability reconstruction from captured archives; not actual execution observations, uptime, trade win rate or filter alpha. Historical factor download failures are not independently archived."}


def monitor_history(raw, epoch, start, end):
    """Summarize only archived monitor checks; never infer older success or uptime."""
    rows, malformed, foreign = [], 0, 0
    for line in (raw or b"").decode(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("account_epoch") != epoch:
                foreign += 1
                continue
            timestamp = int(row["checked_at_ms"])
            if start <= timestamp < end:
                rows.append(row)
        except (ValueError, TypeError, KeyError):
            malformed += 1
    rows.sort(key=lambda row: int(row["checked_at_ms"]))
    return {"recorded_checks": len(rows), "status_counts": dict(Counter(row.get("status", "unrecorded") for row in rows)),
            "assessment_counts": dict(Counter(row.get("assessment", "unrecorded") for row in rows)),
            "first_archived_check_utc": iso(int(rows[0]["checked_at_ms"])) if rows else None,
            "last_archived_check_utc": iso(int(rows[-1]["checked_at_ms"])) if rows else None,
            "malformed_records": malformed, "other_account_epoch_records_ignored": foreign,
            "failed_checks": [row for row in rows if row.get("status") != "healthy"],
            "note": "This is the existing risk-monitor health journal only. Earlier unarchived monitoring and intervals between checks remain unknown; healthy checks do not establish continuous uptime."}


def build_report(manifest, inputs):
    for path in (Path(__file__), Path(multifactor.__file__), Path(event_risk.__file__), Path(public_context.__file__)):
        stored = manifest.get("code_resources", {}).get(path.name)
        if stored and hashlib.sha256(path.read_bytes()).hexdigest() != stored["sha256"]:
            raise ValueError("Archived audit code differs; run the captured code/report_forward_progress.py with --from-archive")
    data = {key: load_payload(raw, key == "factors") for key, raw in inputs.items()
            if raw is not None and key not in {"journal", "scheduler_log", "risk_monitor_history"}}
    profile, account, state, source_report = (data[key] for key in ("profile", "account", "paper_state", "report"))
    digest = multifactor.profile_hash(profile)
    if source_report.get("multifactor_overlay", {}).get("candidate_id") != profile["candidate_id"] or source_report.get("execution_entry_context", {}).get("factor_profile_sha256") != digest or state.get("profile", {}).get("factor_profile_sha256") != digest:
        raise ValueError("Selected candidate and captured production profile do not match")
    if state["created_at_utc"] != source_report["paper_inception_utc"]:
        raise ValueError("Paper state and report inception differ")
    end, candidate_start, epoch = ms(manifest["asof_utc"]), ms(state["created_at_utc"]), ms(account["created_at_utc"])
    if end < max(candidate_start, epoch, ms(account["updated_at_utc"]), ms(source_report["generated_at_utc"])):
        raise ValueError("As-of precedes captured production records; use their original capture time")
    start = max(candidate_start, epoch)
    fills, integrity = complete_fills(account, inputs.get("journal"))
    episodes = []
    if integrity["complete"]:
        try:
            episodes = reconstruct(account, fills)
        except (ValueError, KeyError, TypeError) as exc:
            integrity.update(complete=False, issues=integrity["issues"] + [str(exc)])
    phases = segments(state, source_report, start, end)
    for row in phases:
        row["start_utc"], row["end_utc"] = iso(row["start_ms"]), iso(row["end_ms"])
        row["metrics"] = period_metrics(account, fills, episodes, row["start_ms"], row["end_ms"], integrity["complete"])
    snapshot = multifactor.Snapshot(data["factors"], "first_seen")
    health, missing = public_context.health_at(data["public"], end)
    current = {side: multifactor.asdict(multifactor.decision_at(snapshot, end, side, profile, data["public"])) for side in ("long", "short")}
    target = source_report.get("execution_target") or {}
    metrics = phases[-1]["metrics"]
    dd = finite(account["max_drawdown_pct"])
    gates = {"calendar_days_at_least_90": metrics["elapsed_calendar_days"] >= 90,
             "closed_net_episodes_at_least_30": metrics["closed_net_position_episodes"] is not None and metrics["closed_net_position_episodes"] >= 30,
             "closed_episode_profit_factor_at_least_1_3": metrics["closed_episode_profit_factor"] is not None and metrics["closed_episode_profit_factor"] >= 1.3,
             "tracked_account_epoch_max_drawdown_at_most_15_pct": dd <= 15,
             "complete_epoch_ledger": integrity["complete"]}
    curve = sorted(account.get("equity_curve", []), key=lambda row: int(row["time_ms"]))
    observed = [int(row["time_ms"]) for row in curve if start <= int(row["time_ms"]) <= end]
    gaps = [{"from_utc": iso(a), "to_utc": iso(b), "gap_minutes": (b - a) / 60000}
            for a, b in zip([start] + observed, observed + [end]) if b - a > 2 * STEP_MS]
    qty, price, entry = (finite(account.get(key) or 0) for key in ("position_qty", "last_mark_price", "entry_price"))
    equity = finite(account["wallet_balance"]) + (price - entry) * qty if not qty or price > 0 else None
    pause_rows = re.findall(r"market_data_paused retry_at_utc=([^\s]+)", (inputs.get("scheduler_log") or b"").decode(errors="replace"))
    pauses = sorted(set(value for value in pause_rows if start <= ms(value) <= end))
    return {"schema_version": 1, "asof_utc": manifest["asof_utc"], "candidate_id": profile["candidate_id"],
            "read_only": True, "places_orders": False, "profile_sha256": digest,
            "candidate_inception_utc": state["created_at_utc"], "account_epoch_utc": account["created_at_utc"],
            "effective_sample_start_utc": iso(start), "account_reset_after_candidate_inception": epoch > candidate_start,
            "candidate_elapsed_calendar_days": (end - candidate_start) / DAY_MS,
            "account_epoch_elapsed_calendar_days": (end - epoch) / DAY_MS,
            "account": {"updated_at_utc": account["updated_at_utc"], "age_seconds": (end - ms(account["updated_at_utc"])) / 1000,
                        "initial_balance_usdt": account["initial_balance"], "wallet_balance_usdt": account["wallet_balance"],
                        "equity_usdt_at_last_mark": equity, "position_qty": qty, "account_epoch_fill_counter": account["fill_count_total"],
                        "account_epoch_fees_usdt": account["fees_paid"], "account_epoch_funding_pnl_usdt": account["funding_pnl"],
                        "account_epoch_max_drawdown_pct": dd},
            "production_report": {"generated_at_utc": source_report["generated_at_utc"], "shadow_initial_equity_excluded": source_report.get("summary", {}).get("initial_equity"),
                                  "paper_observation_counter": state.get("observations"), "execution_target": target},
            "ledger_integrity": integrity, "candidate_metrics": period_metrics(account, fills, episodes, start, end, integrity["complete"]),
            "execution_model_segments": phases,
            "evaluation": {"basis": "Latest recorded execution-model segment only; prior execution models are not stitched into its 90-day or closed-episode sample.",
                           "requirements": {"minimum_days": 90, "minimum_closed_net_episodes": 30, "minimum_profit_factor": 1.3, "maximum_drawdown_pct": 15},
                           "gates": gates, "minimum_gate_status": "met_observed_minima_only" if all(gates.values()) else "not_met",
                           "profitability_validated": False, "live_promotion_authorized": False,
                           "note": "These minima alone do not validate profitability; real first-seen coverage, doubled execution cost and independent sample quality still require review. Zero/no-loss samples have undefined PF. Drawdown uses the conservative tracked account-epoch latch, not a complete segment price path."},
            "current_data_health": {"factor_snapshot_generated_at_utc": snapshot.metadata.get("generated_at_utc"), "factor_refresh_errors": snapshot.metadata.get("errors", {}),
                                    "factor_source_status": snapshot.metadata.get("source_status", {}),
                                    "factors": current, "public_healthy": health, "public_missing": missing,
                                    "event": multifactor.asdict(event_risk.event_decision_at(event_risk.events_from_payload(data["public"]), end)),
                                    "latest_account_entry_gate": account.get("execution_entry_gate"), "binance_cooldown": data.get("cooldown")},
            "risk_monitor": {"latest": data.get("risk_monitor"),
                             "latest_matches_account_epoch": bool(data.get("risk_monitor") and data["risk_monitor"].get("account_epoch") == account["created_at_utc"]),
                             "history": monitor_history(inputs.get("risk_monitor_history"), account["created_at_utc"], start, end)},
            "historical_availability": coverage(snapshot, data["public"], profile, start, end),
            "execution_observation_gaps": {"over_10_minutes": gaps, "recorded_binance_retry_times_utc": pauses,
                                          "note": "Retained equity marks record strategy reconciliations, not every risk-monitor check. Gaps cannot identify process uptime or the exact failure cause. Retry timestamps are schedules, not proof of recovery; log times outside this sample are excluded."},
            "input_manifest": "captured_inputs.json", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": ["Inputs are read sequentially while production may advance; exact captured bytes and hashes permit replay.",
                            "Candidate and account epoch are different; shadow 100 USDT is excluded from real account 1,000 USDT accounting.",
                            "Only fully closed continuous same-direction net positions beginning inside a segment count; boundary-crossing and open positions are excluded from PF.",
                            "Equity marks retain at most 1,000 points, fill tail at most 200; missing journal or inventory reconciliation disables sample metrics rather than filling gaps.",
                            "Elapsed days are calendar duration, not proof of continuous operation or healthy data coverage.",
                            "Sample metrics stop at the captured account update; no market prices, funding or fills are invented beyond it."]}


def markdown(result):
    account, latest = result["account"], result["execution_model_segments"][-1]
    value = latest["metrics"]
    return f"""# 当前策略前瞻进度（只读）

截至 {result['asof_utc']}（UTC），当前候选为 `{result['candidate_id']}`。
候选起点：{result['candidate_inception_utc']}；真实账户起点：{result['account_epoch_utc']}。
候选已历时 {result['candidate_elapsed_calendar_days']:.3f} 天；当前执行模型 `{latest['model']}` 自 {latest['start_utc']} 起历时 {value['elapsed_calendar_days']:.3f} 天。这是日历时间，不能当作持续健康运行时长。

真实账户钱包 {account['wallet_balance_usdt']:.2f} USDT，净仓位 {account['position_qty']} BTC，账户累计成交计数 {account['account_epoch_fill_counter']}。影子策略初始资金不计入账户绩效。
当前模型段成交数：{value['fills']}；完整已结束净持仓段：{value['closed_net_position_episodes']}；利润因子：{value['closed_episode_profit_factor']}（None 表示缺失或无有效亏损分母）。
当前模型段手续费：{value['fees_usdt']} USDT；资金费损益：{value['funding_pnl_usdt']} USDT。

90 天、30 个完整已结束净持仓段、PF ≥ 1.3、已跟踪账户回撤 ≤ 15% 的最低门槛状态：**{result['evaluation']['minimum_gate_status']}**。旧执行模型不拼接进入当前模型样本。零成交、零回撤不代表盈利验收通过；不会自动转实盘。

当前公共来源健康：{result['current_data_health']['public_healthy']}；多头许可：{result['current_data_health']['factors']['long']['allowed']}；空头许可：{result['current_data_health']['factors']['short']['allowed']}。
缺失因子：{result['current_data_health']['factors']['long']['missing_groups']}。

完整分段、首见数据覆盖诊断、实际留存执行观测空缺及账本校验见 `report.json`。覆盖率来自五分钟网格重建，与实际执行周期数、胜率不同。留存净值最多 1,000 点，不能声称覆盖完整段回撤；断线期间价格路径没有补造。
本报告固定输入和 SHA-256 保存在 `captured_inputs.json` 与 `inputs/`，审计脚本及计算依赖保存在 `code/`，可离线复现。未刷新数据、未写账户、未操作进程、未下单。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-archive", type=Path)
    parser.add_argument("--asof-utc", help="Must not precede captured records; archived replay uses its original as-of")
    args = parser.parse_args()
    if args.from_archive:
        if args.asof_utc:
            parser.error("Archived replay preserves its original as-of")
        manifest, inputs = read_archive(args.from_archive)
        result = build_report(manifest, inputs)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    output = args.output_dir or ROOT / "data/validation/forward_progress" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for protected in (ROOT / "data/runtime", ROOT / "data/paper_trading", ROOT / "data/snapshots", ROOT / "config"):
        if output.resolve() == protected.resolve() or protected.resolve() in output.resolve().parents:
            parser.error("Output must be independent of production state/config/snapshot directories")
    manifest, inputs = capture(ROOT, output, args.asof_utc)
    result = build_report(manifest, inputs)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(markdown(result), encoding="utf-8")
    print(output)
    print(f"current_model_days={result['execution_model_segments'][-1]['metrics']['elapsed_calendar_days']:.3f} "
          f"gate_status={result['evaluation']['minimum_gate_status']} ledger_complete={result['ledger_integrity']['complete']}")


if __name__ == "__main__":
    main()
