"""Standard-library-only, explicit references; no 'latest' or lens fallback."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT = Path(__file__).with_name('current.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_path(name):
    path = (PROJECT / name).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError('workbench path escapes project: ' + name)
    return path


def load_registry(path=DEFAULT):
    config = read(path)
    if config.get('schema') != 'ballplate-current-workbench-1':
        raise ValueError('unsupported workbench registry')
    paths = {}
    for name, record in config['artifacts'].items():
        bound = source_path(record['path'])
        if digest(bound) != record['sha256']:
            raise ValueError('artifact changed: ' + name)
        paths[name] = bound
    catalog = read(paths['real_input_catalog'])
    head = read(paths['real_linear_head'])
    protocol = read(paths['recorded_protocol'])
    for key, role in (('contract', 'hardware_contract'), ('native_weights', 'native_readout8'),
                      ('SLM_RAW_GRAY_NO_LUT', 'slm_raw_gray')):
        if Path(catalog[key]).resolve() != paths[role]:
            raise ValueError('catalog/registry mismatch: ' + key)
    if (Path(head['bundle']).resolve() != paths['real_input_catalog']
            or head['bundle_sha256'] != digest(paths['real_input_catalog'])
            or head['phase_sha256'] != digest(paths['slm_raw_gray'])
            or head['native_weights_sha256'] != digest(paths['native_readout8'])
            or Path(protocol['head']).resolve() != paths['real_linear_head']
            or protocol['head_sha256'] != digest(paths['real_linear_head'])
            or protocol['bundle_sha256'] != digest(paths['real_input_catalog'])):
        raise ValueError('head/catalog/protocol identities differ')
    return config, paths


def runtime_source_check(config):
    """Read file hashes only, without touching the running optical owner."""
    manifest = read(source_path(config['runtime']['loaded_source_manifest']))
    mismatches = [name for name, sha in manifest['sources'].items() if digest(name) != sha]
    if mismatches:
        raise ValueError('running source changed on disk: ' + repr(mismatches))
    return dict(pid=manifest['pid'], checked_source_files=len(manifest['sources']), all_unchanged=True)
