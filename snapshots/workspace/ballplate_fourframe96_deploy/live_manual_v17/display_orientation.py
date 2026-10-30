"""Display-coordinate hints only; never changes raw CCD, masks or calibration gates."""
import hashlib
import json
from pathlib import Path

PROFILE=Path(__file__).with_name('display_orientation.json')


def checked_code(code):
    if type(code) is not int or not 0<=code<8:raise ValueError('orientation must be an explicit D4 code 0..7')
    return code


def load_hint(path=PROFILE):
    raw=Path(path).read_bytes();hint=json.loads(raw)
    checked_code(hint['simulation_to_native_CCD_orientation'])
    if (hint.get('format_version')!=1 or hint.get('display_only') is not True
            or hint.get('calibrated_geometry') is not False or hint.get('release_approved') is not False
            or any(k in hint for k in ('matrix_sensor','ROI','LUT','signature'))):
        raise ValueError('display direction must NOT impersonate a calibrated geometry / deployment profile')
    return dict(hint,path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest())


def orientation_name(code):
    code=checked_code(code)
    return '逆时针%d°'%((code%4)*90)+('→左右镜像' if code>=4 else '')


def edited_orientation(code,action):
    """Compose in displayed-camera coordinates: H**mirror @ R**turns."""
    code=checked_code(code);turns=code%4;mirror=code//4
    if action=='horizontal':mirror=1-mirror
    elif action=='vertical':turns=(turns+2)%4;mirror=1-mirror
    elif action=='rotate_ccw':turns=(turns+(-1 if mirror else 1))%4
    else:raise ValueError('unknown preview direction action')
    return mirror*4+turns
