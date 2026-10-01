"""Archive actual first observation and pin each forward entry's macro decision."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from execution_ledger import attach_ledger
import macro_regime
import simulate_range_swing as sim
from trading_execution import read_json, write_json


def archive_snapshot(path: Path, *, now_ms=None):
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    folder = path.parent/"first_seen_macro"
    folder.mkdir(parents=True,exist_ok=True)
    archive = folder/f"{digest}{''.join(path.suffixes)}"
    if not archive.exists():
        with archive.open('xb') as handle:
            handle.write(payload)
    elif hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError("Archived macro snapshot hash mismatch")
    receipt = folder/f"{digest}.receipt.json"
    if not receipt.exists():
        now = datetime.now(timezone.utc).isoformat() if now_ms is None else sim.iso_utc_from_ms(now_ms)
        with receipt.open('x',encoding='utf-8') as handle:
            json.dump({"sha256":digest,"first_seen_at_utc":now,"archive":str(archive),
                       "policy":"First observation is not backdated to provider history."},handle,indent=2)
    return digest


def apply_overlay(sleeves, snapshot_path, cache_path, now_ms, *, factors, min_multiplier=.35, block_score=-.8):
    digest = archive_snapshot(snapshot_path,now_ms=now_ms)
    snapshot = macro_regime.load_macro_snapshot(snapshot_path)
    cached = read_json(cache_path,None) if cache_path.exists() else {"version":1,"decisions":{}}
    if not isinstance(cached,dict) or not isinstance(cached.get('decisions'),dict):
        raise ValueError("Invalid forward macro decision archive")
    adjusted, decisions = [], []
    for sleeve in sleeves:
        rows = []
        for raw in attach_ledger(sleeve)['trades']:
            key = '|'.join((str(raw['strategy']),str(raw['side']),str(raw['entry_time_utc'])))
            if key not in cached['decisions']:
                decision = macro_regime.macro_decision_at(snapshot,sim._utc_ms(raw['entry_time_utc']),
                                                         enabled_factors=factors,min_multiplier=min_multiplier,block_score=block_score)
                cached['decisions'][key] = {"decision":asdict(decision),"snapshot_sha256":digest,
                                           "first_seen_at_utc":sim.iso_utc_from_ms(now_ms),
                                           "historical_entry_utc":raw['entry_time_utc']}
            record = cached['decisions'][key]
            decision = macro_regime.MacroDecision(**record['decision'])
            decisions.append(decision)
            if decision.allowed:
                rows.append(macro_regime._scaled_trade(raw,decision.risk_multiplier,decision))
        adjusted.append({**sleeve,'trades':rows})
    write_json(cache_path,cached)
    multipliers = [d.risk_multiplier for d in decisions if d.allowed]
    return adjusted, {"decisions":len(decisions),"blocked":sum(not d.allowed for d in decisions),
                      "average_risk_multiplier":sum(multipliers)/len(multipliers) if multipliers else 0,
                      "min_risk_multiplier":min(multipliers) if multipliers else None,
                      "max_risk_multiplier":max(multipliers) if multipliers else None,
                      "factor_coverage_pct":{name:sum(name in d.available_factors for d in decisions)/len(decisions)*100
                                             if decisions else 0 for name in factors},
                      "decision_archive":str(cache_path),"current_snapshot_sha256":digest,
                      "availability_policy":"Pin decisions at first observation; pre-existing entries remain reconstructed history."}
