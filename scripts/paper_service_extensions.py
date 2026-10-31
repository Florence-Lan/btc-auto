"""Validated optional paper services and an unchanged historical runtime.

Every path is local to the workspace. Existing experiment fingerprints remain
authoritative; this module never initializes or edits their ledgers.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import json

CONFIG = 'config/paper_service_extensions_20261020.json'
REENTRY_CONFIG = 'config/btc_reentry_shadow_20261020.json'
REENTRY_OUTPUT = 'data/paper_trading/btc_reentry_shadow_20261020'
REENTRY_BINDINGS = {REENTRY_CONFIG: REENTRY_OUTPUT,
    'config/btc_reentry_compat_shadow_20261020.json': 'data/paper_trading/btc_reentry_compat_shadow_20261020'}
MACRO_CONFIG = 'config/stock_macro_shadow_20261021.json'
MACRO_OUTPUT = 'data/paper_trading/stock_macro_shadow_20261021'
MACRO_SYMBOLS = ('MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT')
ARMS = ('baseline_normal', 'candidate_normal', 'baseline_double', 'candidate_double')
REVIEW_CONFIG = 'config/paper_llm_review_20261020.json'


@lru_cache(maxsize=4)
def _executable_digest(path, modified_ns, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root, relative, prefix):
    path = (root / relative).resolve()
    path.relative_to((root / prefix).resolve())
    return path


def review_environment(root):
    """A paper-service-only override; never edit global Codex settings or dotenv."""
    path = Path(root) / REVIEW_CONFIG
    if not path.exists():
        return {}
    config = read(path)
    if (set(config) != {'schema_version', 'mode', 'places_orders', 'provider', 'model',
                       'reasoning_effort', 'codex_bin', 'codex_sha256', 'cli_version'}
            or config['schema_version'] != 1 or config['mode'] != 'SIMULATION'
            or config['places_orders'] is not False or config['provider'] != 'codex'
            or config['model'] != 'gpt-6.1-sol' or config['reasoning_effort'] not in ('minimal', 'low', 'medium', 'high')):
        raise ValueError('Invalid independent paper LLM review configuration')
    executable = Path(config['codex_bin'])
    if not executable.is_absolute() or not executable.is_file() or executable.suffix.lower() != '.exe':
        raise ValueError('Pinned paper review executable missing or changed; run a new diagnostic first')
    identity = executable.stat()
    if _executable_digest(str(executable), identity.st_mtime_ns, identity.st_size) != config['codex_sha256']:
        raise ValueError('Pinned paper review executable missing or changed; run a new diagnostic first')
    return {'LLM_PROVIDER': config['provider'], 'LLM_CODEX_MODEL': config['model'],
            'LLM_CODEX_REASONING_EFFORT': config['reasoning_effort'], 'CODEX_BIN': str(executable),
            'LLM_TRADE_GATE_ENABLED': 'true'}


def load(root, verify_frozen=True):
    root = Path(root).resolve()
    path = root / CONFIG
    if not path.exists():
        return {}
    spec = read(path)
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'SIMULATION'
            or spec.get('places_orders') is not False
            or set(spec) - {'schema_version', 'mode', 'places_orders',
                             'sndk_shadow', 'sndk_frozen_runtime', 'btc_reentry_shadow', 'stock_macro_shadow'}):
        raise ValueError('Invalid paper service extensions')
    result = {}
    if 'sndk_shadow' in spec:
        service = spec['sndk_shadow']
        if (not isinstance(service, dict) or set(service) != {'enabled'}
                or type(service['enabled']) is not bool):
            raise ValueError('Invalid SNDK shadow service switch')
        result['sndk_enabled'] = service['enabled']
    frozen = spec.get('sndk_frozen_runtime')
    # An explicitly disabled comparison must not prevent the two primary
    # services from starting because its historical archive is unavailable.
    # Preserve the binding as configuration only; validate it if re-enabled.
    if frozen is not None and result.get('sndk_enabled', True):
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
                or REENTRY_BINDINGS.get(reentry['config_path']) != reentry['output_dir']):
            raise ValueError('Invalid BTC reentry service binding')
        if reentry['enabled']:
            result['reentry_config'] = contained(root, reentry['config_path'], 'config')
            result['reentry_output'] = contained(root, reentry['output_dir'], 'data/paper_trading')
    macro = spec.get('stock_macro_shadow')
    if macro is not None:
        if (not isinstance(macro, dict) or set(macro) != {'enabled', 'config_path', 'output_dir'}
                or type(macro['enabled']) is not bool or macro['config_path'] != MACRO_CONFIG
                or macro['output_dir'] != MACRO_OUTPUT):
            raise ValueError('Invalid stock macro service binding')
        if macro['enabled']:
            result['macro_config'] = contained(root, MACRO_CONFIG, 'config')
            result['macro_output'] = contained(root, MACRO_OUTPUT, 'data/paper_trading')
    return result


def require_reentry_ledgers(root, config, output):
    spec = read(config)
    if (spec.get('mode') != 'SIMULATION' or spec.get('places_orders') is not False
            or spec.get('automatic_strategy_promotion') is not False):
        raise ValueError('Only an isolated order-disabled BTC experiment is accepted')
    if not (output / 'manifest.json').is_file() or any(
            not (output / arm / 'state.json').is_file() for arm in ARMS):
        raise ValueError('Existing BTC reentry manifest and all four ledgers required; bootstrap separately')


def require_macro_ledgers(root, config, output):
    spec = read(config)
    if (spec.get('mode') != 'SIMULATION' or spec.get('places_orders') is not False
            or spec.get('automatic_strategy_promotion') is not False
            or spec.get('symbols') != list(MACRO_SYMBOLS)):
        raise ValueError('Only an isolated order-disabled stock macro experiment is accepted')
    if not (output / 'manifest.json').is_file() or any(
            not (output / symbol / arm / 'state.json').is_file()
            for symbol in MACRO_SYMBOLS for arm in ARMS):
        raise ValueError('Existing stock macro manifest and all twelve ledgers required; bootstrap separately')
