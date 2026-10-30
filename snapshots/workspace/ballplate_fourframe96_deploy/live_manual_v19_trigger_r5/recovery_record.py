"""Validate explicit old baseline after manual process termination; no writes."""
import hashlib
import json
from pathlib import Path


def validate_recovery(record):
    if record['schema']!='manual-alignment-recovery-v1':raise ValueError('recovery schema')
    path=Path(record['baseline_source'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['baseline_source_sha256']:
        raise ValueError('original baseline source changed')
    old=json.loads(path.read_text())
    expected=old.get('camera_before') or old.get('original_fault_before')
    if expected!=record['camera_baseline'] or expected['DeviceSerialNumber']!='15328133':
        raise ValueError('original baseline differs')
    if not record['manual_stop_requires_recheck']:raise ValueError('recheck cannot be bypassed')
    return record
