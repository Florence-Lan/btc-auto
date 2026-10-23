"""Isolated integration checks: real stock hook/book/fees, no public requests."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_parallel_simulation as paper
import run_stock_exit_response_shadow as shadow
import stock_entry_timing
import stock_exit_response as exits
from test_stock_swing_confirmation import worker_fixture, hourly_bars

HOUR = shadow.HOUR


def cash_identity(state):
    assert state["wallet_balance"] == pytest.approx(
        state["initial_balance"] + state["realized_pnl"] - state["fees_paid"] + state["funding_pnl"], abs=1e-9)
    assert state["fill_count_total"] >= len(state["fills"])


def design_for_fixture():
    spec, rules, fingerprint = shadow.load_design()
    # Deterministic exchange-time fixtures run at 600h; only this isolated test
    # declaration's epoch is translated. Current original files are read only.
    spec = {**spec, "effective_from_ms": 600 * HOUR - 20000}
    fingerprint = {**fingerprint, "experiment_config_sha256": shadow.digest(spec)}
    return spec, rules, fingerprint


def initialize(tmp_path):
    spec, rules, fingerprint = design_for_fixture()
    output = tmp_path / "isolated"
    output.mkdir()
    start = 600 * HOUR - 10000
    manifest = shadow.prepare(output, spec, rules, fingerprint, start)
    return output, spec, rules, fingerprint, manifest


def widen_fixture_candles(venue, side=1):
    """Keep the trend/breakout fixture, with ATR that permits real fill costs."""
    source_get = venue.get.side_effect
    def get(endpoint, params=None):
        rows = source_get(endpoint, params)
        if endpoint == "klines" and params["interval"] == "1h":
            rows = deepcopy(rows)
            for row in rows:
                if side == 1:
                    row[3] = float(row[4]) - .2
                else:
                    row[2] = float(row[4]) + .2
        return rows
    venue.get.side_effect = get


def hour_entry_worker(tmp_path, monkeypatch, side=1, variant="combined", *, path=None, config=None):
    original, original_path, clock = worker_fixture(tmp_path / "source_fixture", monkeypatch, side=side)
    clock["volume"] = 10000
    widen_fixture_candles(original.venue, side)
    cfg = deepcopy(config or original.config)
    cfg["entry_timing_policy"] = stock_entry_timing.POLICY.copy()
    if variant != "baseline":
        cfg["exit_response_policy"] = {"mode": exits.MODE, "variant": variant}
    else:
        cfg.pop("exit_response_policy", None)
    target = path or tmp_path / "candidate" / "state.json"
    start = 600 * HOUR - 10000
    if not target.exists():
        paper.write_json(target, paper.initial_stock("MUUSDT", start, cfg))
    # Calendar semantics are tested independently; fixture quotes represent a
    # clear event window and must never read/refresh a real public calendar.
    monkeypatch.setattr(paper.stock_event_risk, "evaluate", lambda *args, **kwargs: {
        "enabled": True, "allowed": True, "reasons": [], "status": "clear"})
    candidate = shadow.ExitResponseAccount(target, original.venue, "MUUSDT", cfg, activated_at_ms=start)
    return candidate, clock, original.venue


def held_pair(tmp_path, monkeypatch, *, side=1, variant="breakout_failure", partial=False):
    original, _, clock = worker_fixture(tmp_path / "source_fixture", monkeypatch, side=side)
    clock.update(volume=10000, price=100 + side * .5, depth_qty=2 if partial else 1000,
                 signal_error=False)
    cfg = deepcopy(original.config)
    cfg["entry_enabled"] = False  # Test only existing-position exit behavior.
    cfg["entry_timing_policy"] = stock_entry_timing.POLICY.copy()
    source_get = original.venue.get.side_effect
    closed = hourly_bars(side=side, length=500, start=100)
    values = (100, 100 - side, 100 - side) if variant == "breakout_failure" else (100, 100 + 2 * side, 100 + side)
    for idx, value in zip((-3, -2, -1), values):
        bar = closed[idx]
        closed[idx] = paper.signals.Candle(bar.time_ms, value, value + .005, value - .005, value, 10000)
    def get(endpoint, params=None):
        if endpoint == "klines" and params["interval"] == "1h":
            if clock["signal_error"]:
                raise RuntimeError("Fixture hourly outage")
            return [[b.time_ms, b.open, b.high, b.low, b.close, b.volume, b.time_ms + HOUR - 1]
                    for b in closed]
        if endpoint == "premiumIndex":
            return {"time": clock["now"], "markPrice": str(clock["price"]),
                    "indexPrice": str(clock["price"]), "lastFundingRate": "0",
                    "nextFundingTime": clock["now"] + HOUR}
        if endpoint == "depth":
            return {"T": clock["now"], "bids": [[str(clock["price"] - .01), str(clock["depth_qty"])]],
                    "asks": [[str(clock["price"] + .01), str(clock["depth_qty"])]]}
        return source_get(endpoint, params)
    original.venue.get.side_effect = get
    start, entered = 597 * HOUR, 597 * HOUR + 60000
    base = paper.initial_stock("MUUSDT", start, cfg)
    base.update(wallet_balance=999.88, fees_paid=.1, funding_pnl=-.02, equity=999.88,
                position_qty=float(side), signal_active_after_ms=start)
    base["position"] = {"entry": 100., "initial_stop": 100. - 2 * side,
        "stop": 100. - 2 * side, "qty": 1., "direction": side,
        "entry_time": entered, "signal_time": 597 * HOUR,
        "entry_fee": .1, "margin": 10., "funding": .02,
        "entry_trigger": {"breakout_level": 100 - side, "direction": side, "boundary_ms": 597 * HOUR}}
    base["position_history"] = [{"time_ms": entered, "signed_qty": float(side)}]
    workers = {}
    for name in ("baseline", variant):
        state = deepcopy(base)
        rule = deepcopy(cfg)
        if name != "baseline":
            rule["exit_response_policy"] = {"mode": exits.MODE, "variant": variant}
        state["rule"] = rule
        path = tmp_path / name / "state.json"
        paper.write_json(path, state)
        workers[name] = shadow.ExitResponseAccount(path, original.venue, "MUUSDT", rule,
                                                   activated_at_ms=start)
    return workers, clock, original.venue, closed


def test_twenty_four_ledgers_only_add_exit_policy_with_cost_matched_rules():
    spec, rules, fingerprint = shadow.load_design()
    assert len(rules) == 3 and sum(len(v) for v in rules.values()) == 24
    assert spec["places_orders"] is False and spec["initial_balance_usdt_each"] == 1000
    shadow.validate_fingerprint(fingerprint, rules)
    for symbol, arms in rules.items():
        normal = arms["baseline_normal"]
        assert normal["signal_timeframe"] == "1h" and normal["signal_family"] == "donchian20"
        assert normal["risk_fraction_per_trade"] == .0025 and normal["cooldown_signal_bars"] == 3
        assert "profit_exit_policy" not in normal and "trend_exit_policy" not in normal
        for cost in shadow.COSTS:
            baseline = arms["baseline_" + cost]
            for variant in shadow.VARIANTS[1:]:
                candidate = deepcopy(arms[variant + "_" + cost])
                assert candidate.pop("exit_response_policy") == {"mode": exits.MODE, "variant": variant}
                assert candidate == baseline
        doubled = deepcopy(arms["baseline_double"])
        for key in ("taker_fee_rate_assumption", "adverse_slippage_fraction_assumption"):
            assert doubled[key] == 2 * normal[key]
            doubled[key] /= 2
        assert doubled == normal


def test_initialization_makes_independent_flat_epoch_and_restore_preserves_bytes(tmp_path):
    output, spec, rules, fingerprint, manifest = initialize(tmp_path)
    before = {}
    for symbol in shadow.SYMBOLS:
        for arm in shadow.ARMS:
            path = output / symbol / arm / "state.json"
            state = paper.read_json(path)
            assert state["wallet_balance"] == state["initial_balance"] == 1000
            assert state["position"] is None and state["fill_count_total"] == 0
            assert state["signal_active_after_ms"] == manifest["started_at_ms"]
            cash_identity(state)
            before[path] = path.read_bytes()
    resumed = shadow.prepare(output, spec, rules, fingerprint, manifest["started_at_ms"] + HOUR, True)
    assert resumed == manifest
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert manifest["original_accounts_modified"] is False and manifest["places_orders"] is False


@pytest.mark.parametrize("bad", [None, {}, {"experiment_config_sha256": "f" * 64}])
def test_initialization_refuses_missing_fingerprint_without_creating_ledgers(tmp_path, bad):
    spec, rules, _ = design_for_fixture()
    with pytest.raises(ValueError, match="fingerprint"):
        shadow.prepare(tmp_path, spec, rules, bad, 600 * HOUR - 10000)
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "MUUSDT").exists()


def test_restart_rejects_missing_manifest_fingerprint_and_changed_rules(tmp_path):
    output, spec, rules, fingerprint, manifest = initialize(tmp_path)
    manifest.pop("fingerprint")
    paper.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed"):
        shadow.prepare(output, spec, rules, fingerprint, None, True)
    changed = deepcopy(rules)
    changed["MUUSDT"]["giveback_normal"]["risk_fraction_per_trade"] = .005
    with pytest.raises(ValueError, match="fingerprint"):
        shadow.prepare(output, spec, changed, fingerprint, None, True)


def test_missing_existing_state_cannot_be_reinitialized(tmp_path):
    output, spec, rules, fingerprint, _ = initialize(tmp_path)
    missing = output / "MUUSDT/giveback_normal/state.json"
    missing.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        shadow.prepare(output, spec, rules, fingerprint, 600 * HOUR, True)
    assert not missing.exists()


@pytest.mark.parametrize("side", [1, -1])
def test_original_pending_breakout_trigger_is_attached_on_real_entry(side, tmp_path, monkeypatch):
    worker, clock, venue = hour_entry_worker(tmp_path, monkeypatch, side=side)
    result = worker.step()
    assert result["position"] is not None
    pos = result["position"]
    signals = shadow.shared.read_rows(worker.path.with_name("signals.jsonl"))
    original = next(row["pending_signal"]["entry_trigger"] for row in signals
                    if row["boundary_ms"] == pos["signal_time"])
    assert pos["entry_trigger"] == original and original["direction"] == side
    assert result["fills"][-1]["reason"] == "fresh_closed_1h_signal"
    entries = shadow.shared.read_rows(worker.path.with_name("entries.jsonl"))
    assert entries[0]["entry_trigger"] == original and entries[0]["initial_qty"] == pos["qty"]
    cash_identity(result)


def test_restart_restores_exact_signal_trigger_and_original_quantity_before_immediate_stop(tmp_path, monkeypatch):
    worker, clock, venue = hour_entry_worker(tmp_path, monkeypatch)
    entered = worker.step()
    pos = entered["position"]
    original_trigger, original_qty = deepcopy(pos["entry_trigger"]), pos["qty"]
    # Remove only redundant metadata to exercise deterministic journal recovery.
    pos.pop("entry_trigger")
    paper.write_json(worker.path, entered)
    worker.path.with_name("entries.jsonl").write_text("", encoding="utf-8")
    paper.journal(worker.path.with_name("signals.jsonl"), {
        "boundary_ms": pos["signal_time"] + HOUR,
        "pending_signal": {"entry_trigger": {**original_trigger,
            "boundary_ms": pos["signal_time"] + HOUR, "breakout_level": 999}}})
    original_get = venue.get.side_effect
    def stop_book(endpoint, params=None):
        if endpoint == "depth":
            return {"T": clock["now"], "bids": [[str(pos["stop"] - .01), "1000"]],
                    "asks": [[str(pos["stop"] + .01), "1000"]]}
        return original_get(endpoint, params)
    venue.get.side_effect = stop_book
    clock["now"] += 30000
    restarted = shadow.ExitResponseAccount(worker.path, venue, "MUUSDT", worker.config,
                                           activated_at_ms=worker.activated_at_ms)
    result = restarted.step()
    assert result["position"] is None and result["fills"][-1]["reason"] == "protective_stop"
    entries = shadow.shared.read_rows(worker.path.with_name("entries.jsonl"))
    assert len(entries) == 1 and entries[0]["entry_trigger"] == original_trigger
    assert entries[0]["initial_qty"] == original_qty
    cash_identity(result)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("variant", ["breakout_failure", "giveback"])
def test_candidate_exits_both_directions_at_current_book_while_baseline_holds(tmp_path, monkeypatch, side, variant):
    workers, clock, venue, closed = held_pair(tmp_path, monkeypatch, side=side, variant=variant)
    responses = workers["baseline"].market_data()
    baseline = workers["baseline"].step(deepcopy(responses))
    candidate = workers[variant].step(deepcopy(responses))
    assert baseline["position"] is not None and baseline["fill_count_total"] == 0
    assert candidate["position"] is None and candidate["fill_count_total"] == 1
    fill = candidate["fills"][-1]
    assert fill["reason"] == exits.EXIT_REASONS[variant]
    expected = (clock["price"] - side * .01) * (1 - side * workers[variant].config["adverse_slippage_fraction_assumption"])
    assert fill["price"] == pytest.approx(expected)
    assert fill["price"] != closed[-1].close  # Historical close is not an executable fill.
    assert fill["time_ms"] == clock["now"]
    assert candidate["cooldown_until_ms"] == clock["now"] + 3 * HOUR
    assert not candidate["pending_signal"]
    assert candidate["exit_response_status"]["status"] == "flat"
    for state in (baseline, candidate):
        cash_identity(state)


def test_existing_hard_stop_overrides_the_optional_exit_hook(tmp_path, monkeypatch):
    workers, clock, venue, _ = held_pair(tmp_path, monkeypatch)
    worker = workers["breakout_failure"]
    state = paper.read_json(worker.path)
    state["risk_halted"] = True
    paper.write_json(worker.path, state)
    hook = Mock(side_effect=AssertionError("Optional exit must not precede a hard guard"))
    monkeypatch.setattr(shadow.exits, "evaluate", hook)
    result = worker.step()
    assert result["position"] is None and result["fills"][-1]["reason"] == "account_hard_stop"
    hook.assert_not_called()
    cash_identity(result)


def test_protective_stop_has_priority_over_simultaneously_satisfied_breakout_exit(tmp_path, monkeypatch):
    workers, clock, venue, _ = held_pair(tmp_path, monkeypatch)
    worker = workers["breakout_failure"]
    state = paper.read_json(worker.path)
    state["position"]["stop"] = clock["price"] + 1
    paper.write_json(worker.path, state)
    result = worker.step()
    assert result["position"] is None and result["fills"][-1]["reason"] == "protective_stop"
    assert not worker.path.with_name("exit_decisions.jsonl").exists()
    cash_identity(result)


@pytest.mark.parametrize("side", [1, -1])
def test_partial_exit_latch_retries_residual_after_restart_and_hourly_outage(tmp_path, monkeypatch, side):
    workers, clock, venue, _ = held_pair(tmp_path, monkeypatch, side=side, partial=True)
    candidate = workers["breakout_failure"]
    first = candidate.step()
    assert first["position"]["qty"] == pytest.approx(.8)
    assert first["position"]["pending_exit"] == exits.EXIT_REASONS["breakout_failure"]
    assert first["position"]["exit_response_state"]["latched"]
    decisions = shadow.shared.read_rows(candidate.path.with_name("exit_decisions.jsonl"))
    assert len(decisions) == 1
    clock.update(now=clock["now"] + 30000, depth_qty=1000, signal_error=True)
    restarted = shadow.ExitResponseAccount(candidate.path, venue, "MUUSDT", candidate.config,
                                           activated_at_ms=candidate.activated_at_ms)
    second = restarted.step()
    assert second["position"] is None and second["fill_count_total"] == 2
    assert second["fills"][-1]["qty"] == pytest.approx(.8)
    assert all(fill["reason"] == exits.EXIT_REASONS["breakout_failure"] for fill in second["fills"])
    closed = shadow.shared.read_rows(candidate.path.with_name("closed_trades.jsonl"))
    assert [row["position_closed"] for row in closed] == [False, True]
    assert second["cooldown_until_ms"] == clock["now"] + 3 * HOUR
    assert len(shadow.shared.read_rows(candidate.path.with_name("exit_decisions.jsonl"))) == 1
    third = restarted.step()
    assert third["position"] is None and third["fill_count_total"] == 2
    for state in (first, second, third):
        cash_identity(state)


def test_actual_entry_then_exit_history_survives_prepare_with_noninitial_cash(tmp_path, monkeypatch):
    output, spec, rules, fingerprint, manifest = initialize(tmp_path)
    path = output / "MUUSDT/combined_normal/state.json"
    worker, clock, venue = hour_entry_worker(tmp_path, monkeypatch, path=path,
                                            config=rules["MUUSDT"]["combined_normal"])
    entry = worker.step()
    assert entry["position"] is not None
    state = paper.read_json(path)
    state["risk_halted"] = True
    paper.write_json(path, state)
    clock["now"] += 30000
    stopped = worker.step()
    assert stopped["wallet_balance"] != 1000 and stopped["fill_count_total"] == 2
    cash_identity(stopped)
    before = {p: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    resumed = shadow.prepare(output, spec, rules, fingerprint, None, True)
    assert resumed == manifest
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_symbol_batch_shares_one_source_read_and_compacts_context_for_all_eight_arms(tmp_path, monkeypatch):
    output, spec, rules, fingerprint, manifest = initialize(tmp_path)
    original, _, clock = worker_fixture(tmp_path / "source_fixture", monkeypatch)
    clock["volume"] = 10000
    widen_fixture_candles(original.venue)
    monkeypatch.setattr(paper.stock_event_risk, "evaluate", lambda *args, **kwargs: {
        "enabled": True, "allowed": True, "reasons": [], "status": "clear"})
    venue = shadow.shared.SharedPublicVenue(original.venue)
    workers = {arm: shadow.ExitResponseAccount(output / "MUUSDT" / arm / "state.json", venue,
        "MUUSDT", cfg, activated_at_ms=manifest["started_at_ms"])
        for arm, cfg in rules["MUUSDT"].items()}
    for arm, worker in workers.items():
        if arm != "baseline_normal":
            worker.market_data = Mock(side_effect=AssertionError("No per-arm market requests"))
    original.venue.get.reset_mock()
    context = {}
    with shadow.compact_observations(output, context):
        results, comparisons, clock_info = shadow.tick_symbol(output, "MUUSDT", workers, venue,
                                                               "calendar_fixture_sha256", context)
    assert len(results) == 8 and set(comparisons) == {"normal", "double"}
    endpoints = [call.args[0] for call in original.venue.get.call_args_list]
    assert endpoints.count("time") == endpoints.count("premiumIndex") == endpoints.count("depth") == 1
    assert endpoints.count("exchangeInfo") == endpoints.count("fundingRate") == 1
    assert endpoints.count("klines") == 2
    assert len(endpoints) == 7  # Shared batch, not 8 times this count.
    for arm, worker in workers.items():
        state = paper.read_json(worker.path)
        cash_identity(state)
        assert state["fill_count_total"] == 1 and state["position"]
        assert "worker" not in results[arm].get("errors", {})
        rows = shadow.shared.read_rows(worker.path.with_name("observations.jsonl"))
        assert rows[-1]["shared_context_sha256"] == context["MUUSDT"]
        assert "responses" not in rows[-1] and "book" not in rows[-1]
    archives = list((output / "context_archive/MUUSDT").glob("*.json.gz"))
    assert len(archives) == 1
    payload = json.loads(gzip.decompress(archives[0].read_bytes()))
    assert payload["calendar_sha256"] == "calendar_fixture_sha256"
    assert len(payload["responses"]) == 7


def test_publish_never_confuses_independent_capital_with_original_book_or_promotes(tmp_path, monkeypatch):
    output, spec, rules, fingerprint, manifest = initialize(tmp_path)
    now = manifest["started_at_ms"] + 14 * 86400000
    monkeypatch.setattr(shadow.time, "time", lambda: now / 1000)
    accounts = {symbol: {arm: {"status": "healthy", "financial_integrity": True,
                              "metrics": {"closed_positions": 10}}
                         for arm in shadow.ARMS} for symbol in shadow.SYMBOLS}
    comparisons = {symbol: {cost: {variant: {"paired": {"summary": {"completed_followups": 10}}}
        for variant in shadow.VARIANTS[1:]} for cost in shadow.COSTS} for symbol in shadow.SYMBOLS}
    status = shadow.publish(output, manifest, accounts, comparisons, {}, running=True,
                             cycles=1, calendar_hash="fixture")
    assert status["independent_initial_capital_usdt"] == 24000
    assert status["original_account_capital_included"] is False
    assert status["original_accounts_modified"] is False and status["places_orders"] is False
    assert status["review"]["ready_for_first_review"] is True
    assert status["review"]["readiness_is_pass"] is False
    assert status["forward_validated"] is False and status["automatic_strategy_promotion"] is False
    comparisons["MUUSDT"]["normal"]["giveback"]["paired"]["summary"]["completed_followups"] = 9
    too_few_pairs = shadow.publish(output, manifest, accounts, comparisons, {}, running=True,
                                    cycles=2, calendar_hash="fixture")
    assert too_few_pairs["review"]["ready_for_first_review"] is False
