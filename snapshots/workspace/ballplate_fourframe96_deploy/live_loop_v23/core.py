"""Pure actor validation, bounded perturbations and frame-order accounting."""
import json
import math
from pathlib import Path
import threading
import uuid

import numpy as np
from live_adapt_v22.core import Bundle, audit_capture, digest, predict

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_HEAD = PROJECT/'outputs/live_calibration/MANUAL_RESTART_20261020_231007_257706/GUI/single_frame_20261021_174421_310000/HEAD.json'
DEFAULT_SHA = '08013176018d95bd8845343f38887a1c6460efd6853179861e5ca604b313b29c'


def load_head(path, sha=None, audit=False):
    path = Path(path).resolve()
    if sha and digest(path) != sha:
        raise ValueError('head SHA256 changed')
    head = json.loads(path.read_text()); bundle = Bundle(head['bundle'])
    if (head['bundle_sha256'] != bundle.sha or head['parameter_count'] != 18
            or head['CCD_history'] != 0 or head['phase_mode'] != 'learned'
            or not head.get('real_hardware_features') or not head['short_MuJoCo_pilot_eligible']
            or not all(head['test_criteria'].values()) or bundle.data.get('input_frame_stride') != 3
            or head['phase_sha256'] != digest(bundle.data['SLM_RAW_GRAY_NO_LUT'])
            or head['native_weights_sha256'] != digest(bundle.data['native_weights'])):
        raise ValueError('requires validated real single-frame Linear18 / four RAW stride3 head')
    for source, bound in head['source_sha256'].items():
        if digest(source) != bound: raise ValueError('head fitting source changed')
    predict(head, np.zeros(8))  # Validate finite weight/bias shape.
    if audit:
        audit_capture(bundle, path.parent)
        if (digest(path.parent/'capture.json') != head['capture_sha256']
                or digest(path.parent/'PROSPECTIVE_PROTOCOL.json') != head['prospective_protocol_sha256']):
            raise ValueError('head capture/prospective protocol changed')
    return head, bundle


def limited_action(head, values, previous):
    old = np.asarray(previous, float)
    if old.shape != (2,) or not np.isfinite(old).all() or np.abs(old).max() > .800001:
        raise ValueError('bad prior command')
    raw = predict(head, values)
    if raw.shape != (2,): raise ValueError('one raw8 vector per action')
    return raw, np.clip(np.clip(raw, -.8, .8), old-.08, old+.08)


def validate_push(event):
    if not isinstance(event.get('id'), str) or not 1 <= len(event['id']) <= 64:
        raise ValueError('unique disturbance id required')
    v = np.asarray(event.get('delta_v_m_s'), float)
    if v.shape != (2,) or not np.isfinite(v).all() or not 0 < np.linalg.norm(v) <= .1500001:
        raise ValueError('MuJoCo-only velocity impulse must be in (0,.15] m/s')
    return v


class PushQueue:
    """At most eight pending events; take once at an optical reply boundary."""
    def __init__(self):
        self.lock = threading.Lock(); self.pending = []; self.enabled = False

    def begin(self):
        with self.lock: self.pending = []; self.enabled = True

    def end(self):
        with self.lock:
            self.enabled = False; abandoned = self.pending; self.pending = []
        return abandoned

    def close(self):
        with self.lock: self.enabled = False

    def add(self, axis, sign, speed):
        if axis not in (0, 1) or sign not in (-1, 1) or not math.isfinite(speed):
            raise ValueError('bad disturbance control')
        v = [0., 0.]; v[axis] = sign*speed
        event = dict(id=uuid.uuid4().hex, delta_v_m_s=v)
        validate_push(event)
        with self.lock:
            if not self.enabled or len(self.pending) >= 8: raise RuntimeError('not running or queue full')
            self.pending.append(event)
        return event

    def take(self):
        with self.lock: events = self.pending; self.pending = []
        return events


class FrameOrder:
    """Keep raw SDK FrameID; account ONLY for a narrow observed 16-bit wrap.

    FrameID is not forged into a monotone hardware id. Logical sequence is
    separate. Timestamp reversal, duplicate and arbitrary counter reset fail.
    Device reconnect/reconfiguration is a NEW acquisition epoch.
    """
    def __init__(self):
        self.previous = None; self.timestamp = None; self.sequence = 0; self.wraps = 0

    def accept(self, raw_id, timestamp):
        if type(raw_id) is not int or type(timestamp) is not int or raw_id < 0 or timestamp < 0:
            raise ValueError('integer SDK counters required')
        wrapped = False
        if self.previous is not None:
            if timestamp <= self.timestamp: raise RuntimeError('camera timestamp did not advance')
            if raw_id <= self.previous:
                delta = (raw_id-self.previous) % 65536
                seconds = (timestamp-self.timestamp)/125000000.
                if not (65000 <= self.previous <= 65535 and 0 <= raw_id <= 255
                        and 0 < delta <= 120 and .001 <= seconds <= 3.):
                    raise RuntimeError('duplicate/out-of-order/reset camera frame')
                wrapped = True
        self.previous, self.timestamp = raw_id, timestamp
        self.sequence += 1; self.wraps += int(wrapped)
        return dict(logical_sequence=self.sequence, FrameID_raw=raw_id,
                    observed_16bit_wraps=self.wraps, frame_id_wrap=wrapped)
