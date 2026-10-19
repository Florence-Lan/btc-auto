"""Validated optional paper services and an unchanged historical runtime.

Every path is local to the workspace. Existing experiment fingerprints remain
authoritative; this module never initializes or edits their ledgers.
"""
from pathlib import Path
import hashlib
import json

CONFIG = 'config/paper_service_extensions_20261020.json'
REENTRY_CONFIG = 'config/btc_reentry_shadow_20261020.json'
REENTRY_OUTPUT = 'data/paper_trading/btc_reentry_shadow_20261020'
ARMS = ('baseline_normal', 'candidate_normal', 'baseline_double', 'candidate_double')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root, relative, prefix):
    path = (root / relative).resolve()
    path.relative_to((root / prefix).resolve())
    return path


def load(root, verify_frozen=True):
    root = Path(root).resolve()
    path = root / CONFIG
    if not path.exists():
        return {}
    spec = read(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'SIMULATION'
            or spec.get('places_orders') is not False
            or set(spec) - {'schema_version', 'mode', 'places_orders',
                             'sndk_frozen_runtime', 'btc_reentry_shadow'}):
        raise ValueError('Invalid paper service extensions')
    result = {}
    frozen = spec.get('sndk_frozen_runtime')
    if frozen is not None:
        if not isinstance(frozen, dict) or set(frozen) != {'runtime_root'}:
            raise ValueError('Invalid frozen runtime binding')
        runtime = contained(root, frozen['runtime_root'], 'data/runtime')
        if runtime == (root / 'data/runtime').resolve():
            raise ValueError('Use an isolated frozen runtime directory')
        record = read(runtime / 'runtime_manifest.json')
        experiment = read(root / 'data/paper_trading/sndk_4h_gate_shadow_20261012/manifest.json')
        if (record.get('schema_version') != 1 or record.get('service') != 'sndk_shadow'
                or record.get('runtime_root') != runtime.relative_to(root).as_posix()
                or record.get('experiment_root') != 'data/paper_trading/sndk_4h_gate_shadow_20261012'
                or record.get('fingerprint') != experiment.get('fingerprint')):
            raise ValueError('Frozen runtime does not match the existing SNDK experiment')
        fingerprint = record['fingerprint']
        files = {**fingerprint['source_sha256'], **fingerprint['profile_sha256'],
                 'config/sndk_4h_gate_shadow_20261012.json': fingerprint['experiment_config_sha256']}
        if record.get('files') != files or (runtime / 'data').resolve() != (root / 'data').resolve():
            raise ValueError('Invalid frozen runtime sources or shared data binding')
        for relative, digest in files.items():
            source = contained(runtime, relative, '.')
            if (not source.is_file() or source == runtime
                    or (verify_frozen and hashlib.sha256(source.read_bytes()).hexdigest() != digest)):
                raise ValueError('Frozen runtime source changed: ' + relative)
        result['sndk_script'] = runtime / 'scripts/run_sndk_4h_gate_shadow.py'
    reentry = spec.get('btc_reentry_shadow')
    if reentry is not None:
        if (not isinstance(reentry, dict) or set(reentry) != {'enabled', 'config_path', 'output_dir'}
                or type(reentry['enabled']) is not bool
                or reentry['config_path'] != REENTRY_CONFIG or reentry['output_dir'] != REENTRY_OUTPUT):
            raise ValueError('Invalid BTC reentry service binding')
        if reentry['enabled']:
            result['reentry_config'] = contained(root, REENTRY_CONFIG, 'config')
            result['reentry_output'] = contained(root, REENTRY_OUTPUT, 'data/paper_trading')
    return result


def require_reentry_ledgers(root, config, output):
    spec = read(config)
    if (spec.get('mode') != 'SIMULATION' or spec.get('places_orders') is not False
            or spec.get('automatic_strategy_promotion') is not False):
        raise ValueError('Only an isolated order-disabled BTC experiment is accepted')
    if not (output / 'manifest.json').is_file() or any(
            not (output / arm / 'state.json').is_file() for arm in ARMS):
        raise ValueError('Existing BTC reentry manifest and all four ledgers required; bootstrap separately')
