"""Validate preserved controls and original fault; never authorize projection."""
import hashlib
import math
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(snapshot):
    if snapshot['schema'] != 'v19-fault-restart-controls-v1' or snapshot['restore_registration'] is not False:
        raise ValueError('explicit controls-only restart contract required')
    for group in ('evidence_sha256', 'old_sources', 'old_source_configs'):
        for path, sha in snapshot[group].items():
            if digest(path) != sha: raise ValueError('old restart evidence changed: ' + path)
    h = snapshot['original_heartbeat']
    if (h['busy'] or h['live_camera'] or h['projection_armed'] or snapshot['settings'] != h['manual_settings']
            or snapshot['optical_config'] != h['optical_reference'] or snapshot['roi'] != h['actual_roi']
            or snapshot['fault'] != h['hardware_fault']): raise ValueError('saved values differ from stopped GUI')
    if not all(c['all_RGB_zero'] for c in snapshot['actual_black_checks'].values()):
        raise ValueError('old black output evidence required')
    if snapshot['settings']['dmd'] == 'file' or snapshot['settings']['slm'] == 'file':
        raise ValueError('cannot silently restore file caches')
    exposure = snapshot['requested_exposure_us']
    if not math.isfinite(exposure) or not 50 <= exposure <= 4000: raise ValueError('requested exposure out of bounds')
    if snapshot['original_fault_before']['DeviceSerialNumber'] != '15328133': raise ValueError('wrong camera rollback target')
    return snapshot
