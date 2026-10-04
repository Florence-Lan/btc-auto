#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import run_macro_candidate_shadow as strategy_supervisor
import simulate_range_swing as sim
import decision_runtime
from active_strategy import simulation_leverage_cap
from binance_terminal_client import BinanceApiError, BinanceTerminalClient, datetime_from_ms
from trading_execution import (
    LIVE_STATE_PATH,
    SIMULATION_STATE_PATH,
    execute_report,
    monitor_simulation_account,
    read_json,
    target_from_report,
)


BAR_INTERVAL_MS = 5 * 60 * 1000
DEFAULT_POLL_SECONDS = 30
BAR_SETTLE_DELAY_SECONDS = 3


def parse_args() -> argparse.Namespace:
    root = sim.repo_root()
    parser = argparse.ArgumentParser(
        description="Run the BTC strategy against simulation or guarded Binance live execution.",
    )
    parser.add_argument("--mode", choices=("simulation", "live"), required=True)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--bar-settle-delay-seconds",
        type=int,
        default=BAR_SETTLE_DELAY_SECONDS,
    )
    parser.add_argument("--refresh-hours", type=float, default=12.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--factor-profile", type=Path)
    parser.add_argument("--factor-snapshot", type=Path,
                        default=root / "data/snapshots/multifactor_latest.json.gz")
    parser.add_argument("--supplemental-snapshot", type=Path,
                        default=root / "data/snapshots/supplemental_market_latest.json")
    parser.add_argument(
        "--macro-snapshot",
        type=Path,
        default=root / "data/snapshots/macro_shadow_latest.json.gz",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_20260917_state.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_20260917_report.json",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=root / "data/paper_trading/macro_candidate_20260917_trades.csv",
    )
    return parser.parse_args()


def strategy_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        poll_seconds=args.poll_seconds,
        refresh_hours=args.refresh_hours,
        once=True,
        macro_snapshot=args.macro_snapshot,
        state_path=args.state_path,
        report_path=args.report_path,
        trades_path=args.trades_path,
    )


def run_cycle(args: argparse.Namespace, client: BinanceTerminalClient, *,
              asof_ms=None, required_bar=None) -> dict[str, object]:
    if args.mode != "simulation":
        raise ValueError("The September 17 research candidate is simulation-only")
    report = read_json(args.report_path, {}) or {}
    paper_epoch = (read_json(args.state_path, {}) or {}).get("created_at_utc")
    cached_point = report.get("execution_target") or {}
    reuse = (required_bar is not None and int(cached_point.get("time_ms") or 0) == required_bar
             and paper_epoch is not None and report.get("paper_inception_utc") == paper_epoch
             and report.get("decision_asof_ms") is not None
             and int(report["decision_asof_ms"]) >= required_bar + BAR_INTERVAL_MS - 1
             and (asof_ms is None or int(report["decision_asof_ms"]) <= asof_ms))
    if reuse and getattr(args, "factor_profile", None):
        import multifactor
        context = report.get("execution_entry_context") or {}
        reuse = (context.get("factor_profile") == str(args.factor_profile.resolve())
                 and context.get("factor_profile_sha256") == multifactor.profile_hash(
                     multifactor.load_profile(args.factor_profile)))
    if not reuse:
        if getattr(args, "factor_profile", None):
            import information_runtime
            args.signal_asof_ms = asof_ms
            strategy_supervisor.run_checked(information_runtime.paper_command(args))
        else:
            shadow_args = strategy_args(args)
            strategy_supervisor.refresh_macro_if_needed(shadow_args)
            strategy_supervisor.run_shadow_once(shadow_args)
        report = json.loads(args.report_path.read_text(encoding="utf-8"))
    point = report.get("execution_target") or (report.get("summary") or {}).get("last_equity_point") or {}
    if asof_ms is not None and int(point.get("available_time_ms") or point.get("time_ms") or 0) > asof_ms:
        raise RuntimeError("Strategy target is ahead of the verified decision clock")
    if required_bar is not None and int(point.get("time_ms") or 0) != required_bar:
        raise RuntimeError("Strategy report does not match the due closed candle")
    target = target_from_report(report, now_ms=asof_ms,
                                leverage_cap=simulation_leverage_cap(report))
    judgment = {
        "mode": args.mode, "places_orders": False,
        "report_path": str(args.report_path.resolve()), "account_epoch": (read_json(SIMULATION_STATE_PATH, {}) or {}).get("created_at_utc"),
        "decision_status": "evaluated", "signal_time_ms": int(point.get("time_ms") or 0),
        "last_judged_at_utc": report.get("generated_at_utc"), "target_leverage": target["target_leverage"],
        "position_id": point.get("position_id"), "market_data": report.get("market_data"),
        "clock": getattr(client, "simulation_clock", None), "execution_status": "pending",
        "reused_report": reuse,
    }
    decision_runtime.write_judgment(judgment)
    try:
        result = execute_report(args.mode, report, client)
    except Exception as exc:
        # A recorded current decision survives a transient execution-data failure.
        # The execution account cursor is unchanged, so a later check can retry
        # this report with fresh permissions/price without recomputing history.
        judgment.update(execution_status="deferred", execution_error=f"{type(exc).__name__}: {exc}"[:240])
        decision_runtime.write_judgment(judgment)
        print(f"strategy_judgment_complete={datetime.now(timezone.utc).isoformat()} "
              f"mode={args.mode} execution=deferred signal_time_ms={judgment['signal_time_ms']}", flush=True)
        return {"mode": args.mode, "target_leverage": target["target_leverage"],
                "signal_time_ms": judgment["signal_time_ms"], "execution_status": "deferred"}
    judgment.update(execution_status="completed", execution_error=None,
                    effective_target_leverage=result["target_leverage"],
                    execution_entry_gate=result.get("execution_entry_gate"),
                    entry_guard=result.get("entry_guard"))
    decision_runtime.write_judgment(judgment)
    result["signal_time_ms"] = int(point.get("time_ms") or 0)
    print(
        f"execution_cycle_complete={datetime.now(timezone.utc).isoformat()} "
        f"mode={args.mode} target_leverage={float(result['target_leverage']):.6f} "
        f"llm_gate={(result.get('llm_trade_gate') or {}).get('status', 'unknown')}",
        flush=True,
    )
    return result


def last_processed_signal_ms(mode: str) -> int:
    path = SIMULATION_STATE_PATH if mode == "simulation" else LIVE_STATE_PATH
    state = read_json(path, {}) or {}
    key = "last_signal_time_ms" if mode == "simulation" else "signal_time_ms"
    return int(state.get(key) or 0)


def due_closed_bar_open_ms(
    server_time_ms: int,
    last_processed_ms: int,
    settle_delay_seconds: int = BAR_SETTLE_DELAY_SECONDS,
) -> int | None:
    current_bar_open_ms = server_time_ms // BAR_INTERVAL_MS * BAR_INTERVAL_MS
    phase_ms = server_time_ms - current_bar_open_ms
    if phase_ms < settle_delay_seconds * 1000:
        return None
    latest_closed_bar_open_ms = current_bar_open_ms - BAR_INTERVAL_MS
    return (
        latest_closed_bar_open_ms
        if latest_closed_bar_open_ms > last_processed_ms
        else None
    )


def seconds_until_next_check(
    server_time_ms: int,
    poll_seconds: int,
    settle_delay_seconds: int = BAR_SETTLE_DELAY_SECONDS,
) -> float:
    current_bar_open_ms = server_time_ms // BAR_INTERVAL_MS * BAR_INTERVAL_MS
    phase_ms = server_time_ms - current_bar_open_ms
    if phase_ms < settle_delay_seconds * 1000:
        next_due_ms = current_bar_open_ms + settle_delay_seconds * 1000
    else:
        next_due_ms = current_bar_open_ms + BAR_INTERVAL_MS + settle_delay_seconds * 1000
    until_due = max(0.25, (next_due_ms - server_time_ms) / 1000)
    return min(float(poll_seconds), until_due)


def monitored_clock(mode, client):
    """A failed scheduler clock must not skip the independent risk observation."""
    server_time_ms = int(time.time() * 1000)
    clock_error = None
    clock = None
    try:
        clock = decision_runtime.resolve_clock(client, mode)
        server_time_ms = clock["time_ms"]
        client.simulation_clock = clock
    except Exception as exc:
        clock_error = exc
    if mode == "simulation":
        monitor_simulation_account(
            client, server_time_ms, clock_available=clock_error is None,
            clock_error=clock_error or (clock or {}).get("primary_error"),
            clock_source=(clock or {}).get("source"))
    if clock_error is not None:
        # No verified clock or bounded anchor remains; this is unavailable
        # input, not a neutral/flat strategy assessment.
        raise clock_error
    return server_time_ms


def check_once(args, client):
    server_time_ms = monitored_clock(args.mode, client)
    last_processed = last_processed_signal_ms(args.mode)
    due_bar = due_closed_bar_open_ms(server_time_ms, last_processed, args.bar_settle_delay_seconds)
    if due_bar is not None:
        result = run_cycle(args, client, asof_ms=server_time_ms, required_bar=due_bar)
        actual_signal = int(result.get("signal_time_ms") or 0)
        if actual_signal < due_bar:
            raise RuntimeError(f"Strategy report signal {actual_signal} is older than due bar {due_bar}")
    else:
        previous = read_json(decision_runtime.STATUS_PATH, {}) or {}
        account_epoch = (read_json(SIMULATION_STATE_PATH, {}) or {}).get("created_at_utc")
        if (previous.get("account_epoch") != account_epoch
                or previous.get("report_path") != str(args.report_path.resolve())):
            previous = {}
        previous.update(mode=args.mode, places_orders=False, report_path=str(args.report_path.resolve()),
                        account_epoch=account_epoch, clock=getattr(client, "simulation_clock", None), decision_error=None)
        latest_closed = server_time_ms // BAR_INTERVAL_MS * BAR_INTERVAL_MS - BAR_INTERVAL_MS
        if int(previous.get("signal_time_ms") or 0) == latest_closed and latest_closed <= last_processed:
            previous.update(decision_status="evaluated", execution_status="completed", execution_error=None)
        else:
            previous.update(decision_status="waiting_for_bar")
        decision_runtime.write_judgment(previous)
    return server_time_ms


def record_unavailable(args, error):
    prior = read_json(decision_runtime.STATUS_PATH, {}) or {}
    account_epoch = (read_json(SIMULATION_STATE_PATH, {}) or {}).get("created_at_utc")
    if (prior.get("account_epoch") != account_epoch
            or prior.get("report_path") != str(args.report_path.resolve())):
        prior = {}
    prior.update(mode=args.mode, places_orders=False,
                 report_path=str(args.report_path.resolve()),
                 account_epoch=account_epoch,
                 decision_status="unavailable", decision_error=f"{type(error).__name__}: {error}"[:240],
                 execution_status="deferred")
    decision_runtime.write_judgment(prior)


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be >= 5")
    if args.bar_settle_delay_seconds < 1 or args.bar_settle_delay_seconds > 30:
        raise ValueError("--bar-settle-delay-seconds must be between 1 and 30")
    if args.refresh_hours <= 0:
        raise ValueError("--refresh-hours must be > 0")
    client = BinanceTerminalClient()
    if args.mode == "live":
        client.validate_live_ready()
    if args.once:
        server_time_ms = monitored_clock(args.mode, client)
        if args.factor_profile:
            import information_runtime
            import multifactor
            profile = multifactor.load_profile(args.factor_profile)
            information_runtime.refresh_if_needed(args.factor_snapshot,
                sim.repo_root() / profile["event_snapshot"], args.supplemental_snapshot)
        run_cycle(args, client, asof_ms=server_time_ms)
        return 0
    if args.factor_profile:
        import information_runtime
        import multifactor
        profile = multifactor.load_profile(args.factor_profile)
        information_runtime.start_worker(args.factor_snapshot,
            sim.repo_root() / profile["event_snapshot"], args.supplemental_snapshot)
    print(
        f"scheduler_started mode={args.mode} check_seconds={args.poll_seconds} "
        f"bar_seconds={BAR_INTERVAL_MS // 1000} settle_delay_seconds={args.bar_settle_delay_seconds}",
        flush=True,
    )
    announced_retry_ms = None
    while True:
        loop_started = time.time()
        server_time_ms = int(loop_started * 1000)
        try:
            server_time_ms = check_once(args, client)
        except BinanceApiError as exc:
            if args.mode != "simulation":
                raise
            record_unavailable(args, exc)
            if not exc.retry_at_ms:
                traceback.print_exc()
            elif exc.retry_at_ms != announced_retry_ms:
                print(f"market_data_paused retry_at_utc={datetime_from_ms(exc.retry_at_ms)} "
                      "mode=simulation reason=binance_rate_limit", flush=True)
                announced_retry_ms = exc.retry_at_ms
        except Exception as exc:
            record_unavailable(args, exc)
            traceback.print_exc()
            if args.mode == "live":
                raise
        estimated_server_time_ms = server_time_ms + int((time.time() - loop_started) * 1000)
        time.sleep(seconds_until_next_check(
            estimated_server_time_ms,
            args.poll_seconds,
            args.bar_settle_delay_seconds,
        ))


if __name__ == "__main__":
    raise SystemExit(main())
