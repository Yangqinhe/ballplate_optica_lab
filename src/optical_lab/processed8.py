"""Explicit processed-code eight-detector research domain, never RAW8.

Offline import/construction only reads the explicitly named weights file. This
does not replace camera freshness/saturation/trigger gates or open a control
loop. A fitted processed-domain Linear8x2 has 18 coefficients; legacy RAW8 heads
and their deployment eligibility are deliberately not accepted or inherited.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from . import readout
from .storage import sha256


CONTRACT_SCHEMA = 'optical-lab-processed8-contract-1'
FRAME_SCHEMA = 'optical-lab-processed8-single-frame-1'
HEAD_SCHEMA = 'optical-lab-processed8-linear18-research-1'


def fixed_config():
    return dict(schema=readout.SCHEMA, mode='requested_clip', gain=1.,
                black_ADU=12.5, gamma=.3, output='display_uint8')


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('explicit lowercase SHA256 required')
    return value


def _encoding(value):
    if value not in ('R0', 'R2'):
        raise ValueError('explicit R0 original or R2 dilated DMD encoding required')
    return dict(name=value, radius_logical_px=0 if value == 'R0' else 2,
        frame_shape=[96,96], history_ages_s=[.18,.12,.06,0.],
        chronological_order='TL/TR/BL/BR oldest_to_newest', scale_repeat=3,
        black_border_pixels=12, canvas_shape=[600,600])


class Readout:
    """Frozen current-platform context + LUT, with no camera/model ownership."""
    def __init__(self, weights_path, weights_sha256, *, roi, camera_active,
                 slm_state, slm_raw_gray_sha256, dmd_encoding, geometry_frozen):
        path = Path(weights_path)
        if not path.is_absolute(): raise ValueError('explicit absolute weights path required')
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != _sha(weights_sha256):
            raise ValueError('native eight-region file SHA mismatch')
        if (set(roi) != {'OffsetX','OffsetY','Width','Height'} or
                any(type(v) is not int or v < (1 if k in ('Width','Height') else 0)
                    for k,v in roi.items()) or type(geometry_frozen) is not bool):
            raise ValueError('explicit native ROI and factual geometry status required')
        if any(int(camera_active.get(k, -1)) != v for k,v in roi.items()):
            raise ValueError('actual camera ROI does not match native regions')
        expected = dict(PixelFormat='Mono8', TriggerMode='On', TriggerSource='Line0',
            TriggerActivation='FallingEdge', ExposureAuto='Off', GainAuto='Off')
        if any(camera_active.get(k) != v for k,v in expected.items()):
            raise ValueError('frozen Mono8/Line0/manual photometry context required')
        if not isinstance(slm_state,dict) or not slm_state:
            raise ValueError('explicit current SLM mode/position configuration required')
        if (not np.isfinite(float(camera_active['ExposureTime'])) or
                not np.isfinite(float(camera_active['Gain']))):
            raise ValueError('finite actual exposure and Gain required')
        weights = np.load(io.BytesIO(blob), allow_pickle=False)
        shape = (roi['Height'], roi['Width'])
        readout.integrate_pixels(np.zeros(shape,np.uint8), weights)
        if np.any(weights.sum((1,2),dtype=np.float64) <= 0):
            raise ValueError('all eight explicit detector regions must be nonempty')
        self.weights = np.array(weights,copy=True); self.weights.setflags(write=False)
        coverage = np.count_nonzero(np.any(weights > 0,axis=0))
        self._contract = dict(schema=CONTRACT_SCHEMA, preprocessing=fixed_config(),
            weights_file=str(path), weights_file_sha256=weights_sha256,
            roi=deepcopy(roi), expected_active=deepcopy(camera_active),
            slm_state=deepcopy(slm_state), slm_raw_gray_sha256=_sha(slm_raw_gray_sha256),
            dmd_encoding=_encoding(dmd_encoding), geometry_frozen=geometry_frozen,
            region_coverage_fraction=float(coverage/np.prod(shape)),
            full_ROI_coverage=bool(coverage == np.prod(shape)),
            channel_weight_sums=weights.sum((1,2),dtype=np.float64).tolist(),
            source_sha256={str(Path(__file__).resolve()):sha256(__file__),
                str(Path(readout.__file__).resolve()):sha256(readout.__file__)},
            upstream_raw_headroom_exclusive_ADU=240, upstream_checked_capture_required=True,
            CCD_history=0, legacy_RAW8_compatible=False, real_actuators_enabled=False,
            production_qualified=False, release_approved=False)
        self.contract_sha256 = _digest(self._contract)
        self._lut = readout.preprocess_intensity(np.arange(256,dtype=np.uint8).reshape(1,256),
                                                 fixed_config())[0]
        self._lut.setflags(write=False)

    @property
    def contract(self): return deepcopy(self._contract)

    def verify(self):
        """Read-only asset/source audit at research job boundaries, no SDK."""
        bindings = dict(self._contract['source_sha256'])
        bindings[self._contract['weights_file']] = self._contract['weights_file_sha256']
        if any(sha256(path) != expected for path,expected in bindings.items()):
            raise ValueError('processed8 bound asset/source changed')
        return True

    def process(self, raw, *, camera_active, dmd_encoding):
        """After original checked_capture, transform THIS raw frame, then sum.

        No model/mean/image history or GUI controls are used. The original
        raw headroom rule is checked before LUT mapping; software darkness
        cannot hide ADC saturation. Freshness/complete-frame/display-association
        checks still belong to the upstream unchanged capture owner.
        """
        a = np.asarray(raw)
        shape = (self._contract['roi']['Height'],self._contract['roi']['Width'])
        if a.dtype != np.uint8 or a.shape != shape or int(a.max()) >= 240:
            raise ValueError('one native unsaturated raw Mono8 frame required before processing')
        if camera_active != self._contract['expected_active']:
            raise ValueError('camera photometry/ROI/trigger contract changed; no stale head reuse')
        if _encoding(dmd_encoding) != self._contract['dmd_encoding']:
            raise ValueError('R0/R2 encoding mismatch; static large-ball is not runtime R0')
        processed = self._lut[a]
        return dict(schema=FRAME_SCHEMA, feature_contract_sha256=self.contract_sha256,
            raw=np.array(a,copy=True), processed_uint8=processed,
            raw_pixels_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),
            processed_pixels_sha256=hashlib.sha256(processed.tobytes(order='C')).hexdigest(),
            raw_integrals8_diagnostic=readout.integrate_pixels(a,self.weights),
            processed_integrals8=readout.integrate_pixels(processed,self.weights),
            raw_lower_than_threshold_fraction=float(np.mean(a <= 12.5)),
            control_release_approved=False, upstream_checked_capture_still_required=True)


def research_head(weight, bias, *, feature_contract, training_evidence):
    """Label NEW fitted coefficients; never convert/requalify an old RAW head.

    This stores provenance but grants no pilot eligibility. Dataset/held-out
    validation and separate opt-in experimental control remain external.
    """
    if (feature_contract.get('schema') != CONTRACT_SCHEMA or
            readout.validate_config(feature_contract.get('preprocessing')) != fixed_config() or
            not isinstance(training_evidence,dict) or not training_evidence):
        raise ValueError('explicit processed8 contract and separate fitting evidence required')
    w,b = np.asarray(weight,np.float64), np.asarray(bias,np.float64)
    if w.shape != (2,8) or b.shape != (2,) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('finite Linear8x2 weights and two biases, exactly18 coefficients required')
    return dict(schema=HEAD_SCHEMA, feature_contract_sha256=_digest(feature_contract),
        preprocessing=fixed_config(), weight_processed_code_sum_to_rad=w.tolist(),
        bias_rad=b.tolist(), parameter_count=18, MACs=16, CCD_history=0,
        training_evidence=deepcopy(training_evidence), legacy_RAW8_compatible=False,
        short_MuJoCo_pilot_eligible=False, control_release_approved=False,
        real_actuators_enabled=False, production_qualified=False, release_approved=False)


def predict(head, frame):
    """Pure arithmetic only; this is not a deployment or actuator entrypoint."""
    if (head.get('schema') != HEAD_SCHEMA or frame.get('schema') != FRAME_SCHEMA
            or head.get('feature_contract_sha256') != frame.get('feature_contract_sha256')
            or 'weight_raw_ADU_to_rad' in head or head.get('parameter_count') != 18
            or readout.validate_config(head.get('preprocessing')) != fixed_config()):
        raise ValueError('distinct processed8 head/frame schemas and identical contract required')
    w=np.asarray(head['weight_processed_code_sum_to_rad'],np.float64)
    b=np.asarray(head['bias_rad'],np.float64); x=np.asarray(frame['processed_integrals8'],np.float64)
    if (w.shape!=(2,8) or b.shape!=(2,) or x.shape!=(8,) or
            not np.isfinite(w).all() or not np.isfinite(b).all() or not np.isfinite(x).all()):
        raise ValueError('finite processed8 and exactly18 coefficients required')
    return w@x+b
