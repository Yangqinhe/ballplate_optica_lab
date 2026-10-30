"""Evidence-backed SIMULATION SLM axes, separate from CCD and hardware flips."""
import hashlib
import json
from pathlib import Path

PROFILE=Path(__file__).with_name('slm_phase_coordinates.json')


def load_slm_hint(path=PROFILE):
    path=Path(path).resolve();raw=path.read_bytes();hint=json.loads(raw)
    if (hint.get('format_version')!=1 or type(hint.get('slm_to_model_flip_y')) is not bool
            or hint.get('simulation_only') is not True
            or hint.get('hardware_output_transform_changed') is not False
            or hint.get('CCD_orientation_changed') is not False
            or hint.get('physical_phase_LUT_calibrated') is not False
            or hint.get('release_approved') is not False
            or any(k in hint for k in ('matrix_sensor','ROI','LUT','signature'))):
        raise ValueError('SLM model axes must not change hardware/CCD or impersonate calibration release')
    evidence=(path.parent/hint['source_reference']).resolve()
    if hashlib.sha256(evidence.read_bytes()).hexdigest()!=hint['source_sha256']:
        raise ValueError('SLM coordinate evidence hash mismatch')
    return dict(hint,path=str(path),sha256=hashlib.sha256(raw).hexdigest())
