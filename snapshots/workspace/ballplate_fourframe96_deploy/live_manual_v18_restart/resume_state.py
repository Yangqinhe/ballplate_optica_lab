"""Explicit, hash-bound transfer of alignment evidence; no hardware imports."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]/'live_manual_v17'
sys.path.insert(0, str(BACKEND))
from diffraction_model import OpticalReference, reference_signature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(contract):
    if contract['schema'] != 'v18-retained-alignment-v1':
        raise ValueError('unknown resume contract')
    for path, sha in contract['evidence_sha256'].items():
        if digest(path) != sha:
            raise ValueError('resume evidence changed: '+path)
    runtime = json.loads(Path(contract['runtime_path']).read_text())
    for group in ('sources', 'source_configs'):
        for path, sha in runtime[group].items():
            if digest(path) != sha:
                raise ValueError('original loaded source/config changed: '+path)
    folder = Path(contract['reference_folder'])
    for name in ('frozen_geometry.json', 'capture.json', 'model_reference.json'):
        if str(folder/name) not in contract['evidence_sha256']:
            raise ValueError('missing bound reference evidence')
    registration = json.loads((folder/'frozen_geometry.json').read_text())
    capture = json.loads((folder/'capture.json').read_text())
    config = json.loads((folder/'model_reference.json').read_text())
    if (registration != contract['registration'] or capture['active'] != contract['expected_active']
            or config != contract['optical_config']):
        raise ValueError('resume values differ from their original evidence')
    c = OpticalReference(**config); c.validate()
    if (not registration['frozen'] or registration['signature'] != reference_signature(c, capture['active'])
            or registration['reference_folder'] != str(folder)):
        raise ValueError('original reference signature invalid')
    if contract['roi'] != {key: int(capture['active'][key]) for key in ('OffsetX', 'OffsetY', 'Width', 'Height')}:
        raise ValueError('resume ROI not the reference ROI')
    if not 50 <= contract['requested_exposure_us'] <= 4000:
        raise ValueError('exposure out of diagnostic bound')
    if abs(contract['requested_exposure_us']-float(capture['active']['ExposureTime'])) > 50:
        raise ValueError('requested exposure not consistent with reference readback')
    if capture['active']['TriggerMode'] != 'Off' or capture['active']['DeviceSerialNumber'] != '15328133':
        raise ValueError('wrong camera/trigger contract')
    if contract['settings']['dmd'] == 'file' or contract['settings']['slm'] == 'file':
        raise ValueError('cannot silently transfer in-memory file caches')
    return contract


def build(session, reference_folder, verification, exposure):
    session, folder, verification = [Path(p).resolve() for p in (session, reference_folder, verification)]
    h = json.loads((session/'heartbeat.json').read_text())
    v = json.loads(verification.read_text())
    if (not 0 <= time.time_ns()-h['unix_ns'] < 10_000_000_000 or h['busy'] or h['live_camera']
            or h['projection_armed'] or h['hardware_fault'] or not h['geometry_frozen']):
        raise ValueError('original owner not freshly parked with frozen geometry')
    if not v['parked_verification_passed'] or v['pid'] != h['pid']:
        raise ValueError('original parked verification required')
    registration = json.loads((folder/'frozen_geometry.json').read_text())
    capture = json.loads((folder/'capture.json').read_text())
    config = json.loads((folder/'model_reference.json').read_text())
    if h['optical_reference'] != config:
        raise ValueError('current optical controls differ from reference')
    evidence = [folder/n for n in ('frozen_geometry.json', 'capture.json', 'model_reference.json')]
    evidence += [session/'runtime_sources.json', verification]
    result = dict(schema='v18-retained-alignment-v1', created_unix_ns=time.time_ns(),
        original_pid=h['pid'], original_session=str(session), reference_folder=str(folder),
        runtime_path=str(session/'runtime_sources.json'),
        evidence_sha256={str(p): digest(p) for p in evidence},
        settings=h['manual_settings'], roi=h['actual_roi'], requested_exposure_us=float(exposure),
        optical_config=config, expected_active=capture['active'], registration=registration,
        display_roles=v['roles'], automatic_capture=False, automatic_projection=False,
        requires_matching_new_camera_readback=True, release_approved=False)
    return validate(result)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--reference-folder', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--exposure-us', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = build(a.session, a.reference_folder, a.verification, a.exposure_us)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('Saved explicit pending resume contract; no camera or output commands:', a.output)
