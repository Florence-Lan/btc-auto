"""Apply the existing portfolio ledger to decision marks including a known next open."""
from dataclasses import replace
import math

import portfolio_risk


def combine(base, sleeves, cfg, policy, start=None, *, include_execution_target=False,
            decision_open_prices=None):
    if decision_open_prices is None:
        return portfolio_risk.combine_sleeves_with_drawdown_policy(base,sleeves,cfg,policy,start,
            include_execution_target=include_execution_target)
    decisions = []
    for bar in base:
        boundary = bar.close_time_ms + 1
        price = decision_open_prices.get(boundary)
        if price is None:
            decisions.append(bar)
        else:
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Invalid known opening price")
            # Only the closing decision mark changes: the previous bar has
            # closed and this opening tick is now known. No next-bar HLC or
            # volume enters portfolio sizing. The signal bar identity remains.
            decisions.append(replace(bar, close=price, close_time_ms=boundary,
                                     high=max(bar.high,price), low=min(bar.low,price)))
    result = portfolio_risk.combine_sleeves_with_drawdown_policy(decisions,sleeves,cfg,policy,start,
        include_execution_target=include_execution_target)
    available = {bar.open_time_ms:bar.close_time_ms for bar in decisions}
    for point in result["equity_curve"]:
        point["available_time_ms"] = available[int(point["time_ms"])]
    result["risk_diagnostics"]["accounting_version"] = "bar_close_known_open_v2"
    return result
