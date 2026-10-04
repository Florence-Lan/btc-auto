"""Read-only terminal view of stock research; no account or execution imports."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOL_NAMES = {"MUUSDT": "美光", "SNDKUSDT": "闪迪", "SKHYNIXUSDT": "海力士"}
METRICS = (
    "start_utc", "end_utc_exclusive", "initial_equity", "estimated_close_return_pct",
    "max_sampled_drawdown_pct", "closed_trades", "target_trades_net_at_least_120pct_margin",
    "net_closed_pnl", "estimated_open_close_net_pnl", "profit_factor", "liquidation_stress_count",
)


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a research object")
    return data


def _path(root: Path, value: str, folder: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to((root / folder).resolve()):
        raise ValueError("Research path is outside its designated directory")
    return path


def _optional(path: Path) -> dict:
    try:
        return _load(path)
    except (OSError, ValueError):
        return {}


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid numeric research metric")
    return value


def _percent(value):
    value = _number(value)
    return value * 100 if value is not None else None


def _metrics(summary: dict | None) -> dict | None:
    if not summary:
        return None
    result = {key: summary.get(key) for key in METRICS}
    for key in METRICS:
        if key not in {"start_utc", "end_utc_exclusive"}:
            result[key] = _number(result[key])
    initial = result.get("initial_equity")
    closed = result.get("net_closed_pnl")
    result["closed_return_pct"] = _number(closed / initial * 100) if initial and closed is not None else None
    diagnostics = summary.get("execution_volume_diagnostics") or {}
    result["zero_volume_entries"] = _number(diagnostics.get("entry_bars_zero_reported_volume"))
    result["return_without_best_trade_pct"] = _number((summary.get("trade_diagnostics") or {}).get(
        "closed_return_without_largest_winner_pct"))
    return result


def research_status(root: Path = ROOT) -> dict:
    """Missing artifacts remain unavailable; historical inventory is never live."""
    base = {"status": "unavailable", "places_orders": False, "execution_status": "research_only", "stocks": []}
    try:
        selection = _load(root / "config/stock_research_dashboard.json")
        result = _load(_path(root, selection["results_path"], "data/research"))
        verification = _optional(_path(root, selection["verification_path"], "data/research"))
        plan = _optional(_path(root, selection["forward_plan_path"], "data/research"))
        report = _path(root, selection["report_path"], "docs")
        profile = result["profile"]
        if profile.get("status") != "research_only" or result.get("places_orders") is not False:
            raise ValueError("Expected research-only results")
        stocks = []
        for symbol, name in SYMBOL_NAMES.items():
            runs = result["runs"].get(symbol, {})
            scenarios = {}
            for family, suffix in (("baseline", ""), ("liquidity", "_prior5m_volume10pct")):
                scenarios[family] = {
                    str(cost): {window: _metrics(runs.get(f"{window}{suffix}_cost{cost}"))
                                for window in ("full", "recent30d", "recent60d")}
                    for cost in (1, 2)
                }
            stocks.append({"symbol": symbol, "name": name,
                           "profile": profile["symbol_profiles"].get(symbol, {}),
                           "capital_weight": profile["capital_weights"].get(symbol),
                           "scenarios": scenarios,
                           "slippage_stress": _metrics(runs.get("full_slippage50bps"))})
        first = next((stock["scenarios"]["liquidity"]["1"]["full"] for stock in stocks
                      if stock["scenarios"]["liquidity"]["1"]["full"]), None)
        if not first:
            raise ValueError("No liquidity experiment results")
        config = result["base_config"]
        return {
            **base, "status": "available", "stocks": stocks,
            "candidate_id": profile["candidate_id"], "data_end_utc_exclusive": first["end_utc_exclusive"],
            "reviewed_at_utc": verification.get("verified_at_utc"),
            "forward_status": plan.get("status", "not_observed"),
            "forward_start_utc": plan.get("paper_start_utc"),
            "forward_validated": result.get("forward_validated") is True,
            "coverage": result.get("coverage_choice", {}),
            "assumptions": {"leverage": config.get("leverage"),
                            "target_margin_return_pct": _percent(config.get("target_margin_return")),
                            "risk_per_bucket_trade_pct": _percent(profile.get("risk_fraction_per_bucket_trade")),
                            "taker_fee_pct": _percent(config.get("taker_fee_rate_assumption")),
                            "slippage_pct": _percent(config.get("adverse_slippage_fraction_assumption"))},
            "liquidity_rule": (result.get("declaration", {}).get("liquidity_experiment") or {}).get(
                "entry_max_previous_bar_participation_fraction"),
            "report_url": "/" + report.relative_to(root.resolve()).as_posix() if report.is_file() else None,
            "notes": ["回顾性回测；每只标的独立资金，收益含期末估计平仓损益。",
                      "成交量约束使用此前已收盘五分钟线；当前盘口及真实成交尚未验证。",
                      "保证金盈利目标针对单笔初始保证金，不是整个账户收益。"],
        }
    except FileNotFoundError:
        return {**base, "message": "股票研究结果尚未生成"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {**base, "message": "股票研究数据暂时无法读取"}
