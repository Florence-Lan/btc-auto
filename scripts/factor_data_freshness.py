"""Read-only observation/collection diagnostics; never alter factor decisions.

Keep this outside multifactor.py: its whole-file fingerprint belongs to the
frozen forward strategy and must not change for collection diagnostics.
"""

from datetime import datetime, timezone
import math

OIL_MAX_AGE_MS = 7 * 86_400_000

DAY = 86_400_000
HOUR = 3_600_000
# Mirrors decision_at's existing tolerances, not a new freshness policy.
# The feed's cadence and publication lag are distinct from these tolerances.
FACTOR_SPECS = {
    'btc_close': ('BTC 小时收盘', 'btc_momentum', 3 * HOUR, 'hourly', (DAY, 7 * DAY)),
    'global_long_short': ('全市场多空人数比', 'positioning', 3 * HOUR, 'hourly', ()),
    'top_position_long_short': ('大户持仓多空比', 'positioning', 3 * HOUR, 'hourly', ()),
    'taker_buy_sell': ('主动买卖比', 'positioning', 3 * HOUR, 'hourly', ()),
    'open_interest': ('未平仓量', 'positioning', 3 * HOUR, 'hourly', (DAY,)),
    'funding': ('已结算资金费率', 'positioning', 12 * HOUR, 'settlement', ()),
    'fed_target': ('联邦基金目标上限', 'fed', 7 * DAY, 'daily', (30 * DAY,)),
    'fed_assets': ('美联储资产负债表', 'fed', 16 * DAY, 'weekly', (28 * DAY,)),
    'fed_effective': ('有效联邦基金利率', None, None, 'daily', ()),
    'ust_2y': ('美国两年期国债收益率', 'treasury', 7 * DAY, 'daily', (7 * DAY,)),
    'ust_10y': ('美国十年期国债收益率', 'treasury', 7 * DAY, 'daily', (7 * DAY,)),
    'ust_real_10y': ('美国十年期实际收益率', 'treasury', 7 * DAY, 'daily', (7 * DAY,)),
    'eurusd': ('欧元兑美元', 'fx', 16 * DAY, 'daily', (7 * DAY,)),
    'gbpusd': ('英镑兑美元', 'fx', 16 * DAY, 'daily', (7 * DAY,)),
    'usdjpy': ('美元兑日元', 'fx', 16 * DAY, 'daily', (7 * DAY,)),
    'usdcny': ('美元兑人民币', 'fx', 16 * DAY, 'daily', (7 * DAY,)),
    'usdchf': ('美元兑瑞郎', 'fx', 16 * DAY, 'daily', (7 * DAY,)),
    'broad_dollar': ('广义美元指数', 'fx', 16 * DAY, 'daily', (7 * DAY,)),
    'sp500': ('标普 500 日收盘', 'global_risk', 7 * DAY, 'daily', (7 * DAY,)),
    'nasdaq': ('纳斯达克日收盘', 'global_risk', 7 * DAY, 'daily', (7 * DAY,)),
    'vix': ('VIX 日收盘', 'global_risk', 7 * DAY, 'daily', ()),
    'oil': ('WTI 原油日观测', 'global_risk', OIL_MAX_AGE_MS, 'daily', (7 * DAY,)),
    'gold': ('黄金日观测', 'global_risk', 7 * DAY, 'daily', (7 * DAY,)),
}


def _timestamp(value, asof):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value <= 0 or value > asof):
        return None
    return int(value)


def _utc(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat() if value is not None else None


def _describe_factors(snapshot, timestamp, profile=None, decision=None):
    """Describe dates from the very observations Snapshot.window can see.

    Download metadata cannot replace observation time. First-seen/future rows
    remain excluded exactly as in Snapshot.window. This result is display-only.
    """
    profile = profile or {}
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    sources = metadata.get('source_status') or {}
    sources = sources if isinstance(sources, dict) else {}
    optional_oil = ('oil' in profile.get('optional_features', ())
        and timestamp >= profile.get('optional_features_effective_at_ms', float('inf')))
    ignored = set(getattr(decision, 'ignored_features', ()))
    factors = {}
    for name in dict.fromkeys([*FACTOR_SPECS, *snapshot.series]):
        label, group, limit, cadence, lookbacks = FACTOR_SPECS.get(
            name, (name, None, None, 'unknown', ()))
        rows = snapshot.series.get(name, ((), ()))[1]
        available = [row for row in rows if row[0] <= timestamp]
        current = max(available, key=lambda row: row[1], default=None)
        observed = current[1] if current else None
        age = timestamp - observed if observed is not None else None
        status = ('missing' if current is None else 'not_used' if limit is None
                  else 'stale' if age > limit else 'fresh')
        raw = sources.get(name) or {}
        raw = raw if isinstance(raw, dict) else {}
        attempted = _timestamp(raw.get('last_attempt_ms'), timestamp)
        explicit_success = _timestamp(raw.get('last_fetch_success_ms'), timestamp)
        usable_success = _timestamp(raw.get('last_success_ms'), timestamp)
        successful = explicit_success or usable_success
        # The old `ok` conflates collection health with oil freshness. Do not
        # infer transport success for stale/legacy data without a clear record.
        fetch_ok = raw.get('fetch_ok') if attempted is not None else None
        if type(fetch_ok) is not bool:
            fetch_ok = (True if attempted is not None and raw.get('ok') is True
                        else False if attempted is not None and raw.get('status') == 'error' else None)
        optional = name == 'gold' or name == 'oil' and (optional_oil or name in ignored)
        values_available = (all(snapshot.window(name, timestamp, limit, lookback) is not None
                                for lookback in lookbacks or (0,)) if limit is not None else None)
        group_enabled = group in profile.get('groups', {}) if 'groups' in profile else None
        notes = []
        if cadence in {'daily', 'weekly'}:
            notes.append('按日/周发布，含发布延迟；允许滞后不代表实时行情')
        if optional:
            notes.append('当前为可选输入；缺失时不单独阻断该因子组')
        if limit is None:
            notes.append('当前模型未直接使用；没有设定交易新鲜度阈值')
        elif status == 'fresh' and not values_available:
            notes.append('最新观测在时限内，但变化计算所需历史不足')
        if group_enabled is False:
            notes.append('当前配置未启用此因子组')
        factors[name] = {
            'name': label, 'group': group, 'group_enabled': group_enabled,
            'latest_observed_at_ms': observed, 'latest_observed_at_utc': _utc(observed),
            'usable_at_ms': current[0] if current else None, 'age_ms': age,
            'max_age_ms': limit, 'status': status, 'stale': status == 'stale',
            'cadence': cadence, 'slow_frequency': cadence in {'daily', 'weekly'},
            'optional': optional, 'ignored': name in ignored,
            'required_for_group': group is not None and not optional,
            'values_available': values_available,
            'fetch_ok': fetch_ok, 'source_status': raw.get('status') if attempted is not None else None,
            'last_fetch_attempt_ms': attempted, 'last_fetch_attempt_utc': _utc(attempted),
            'last_fetch_success_ms': successful, 'last_fetch_success_utc': _utc(successful),
            'fetch_success_time_kind': ('transport_record' if explicit_success is not None
                                        else 'legacy_usable_collection_record' if usable_success is not None else None),
            'notes': notes,
        }
    return {'checked_at_ms': timestamp, 'checked_at_utc': _utc(timestamp),
            'availability_mode': snapshot.mode, 'display_only': True,
            'note': '来源抓取成功与观测新鲜度分别展示；日期和现行容忍期不改变模型权重或入场门禁。',
            'factors': factors}


def describe_factors(snapshot, timestamp, profile=None, decision=None):
    """A diagnostic failure must never add a restriction to a trading decision."""
    try:
        return _describe_factors(snapshot, timestamp, profile, decision)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, OSError) as exc:
        return {'checked_at_ms': timestamp, 'display_only': True, 'factors': {},
                'status': 'diagnostics_unavailable', 'error_type': type(exc).__name__}
