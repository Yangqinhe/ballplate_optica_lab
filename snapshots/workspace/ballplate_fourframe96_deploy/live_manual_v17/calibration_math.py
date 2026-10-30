"""Hardware-free calibration helpers. Original calibration sources stay read-only."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ARCHIVE = Path(__file__).resolve().parents[2] / 'ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015'
sys.path.insert(0, str(ARCHIVE / 'src'))
from sdon.calibration.roi import detect_intensity_square, roi_around_square
from sdon.calibration.patterns import build_alignment_pattern, generate_lens_phase
from sdon.calibration.intensity import build_linearization_lut
from sdon.calibration.alignment import (calibrate_simulation_geometry, optical_structure_metrics,
    affine_geometry_metrics, intensity_centroid, spatial_modulation_response, phase_registration_shift)
from sdon.calibration.slm_check import build_slm_check_patterns, analyze_slm_check


@dataclass(frozen=True)
class Settings:
    exposure_us: float = 200.0
    wavelength_m: float = 532e-9
    pixel_pitch_m: float = 8e-6
    f1_m: float = .15
    f2_m: float = .15
    lens_f_m: float = .4
    padding: int = 3
    offset_x: int = 0  # physical screen offset AFTER the single horizontal flip
    offset_y: int = 0
    lens_x: int = 0
    lens_y: int = 0
    illumination_note: str = '现场低功率/ND条件，尚未填写档位'
    binary_diagnostic: bool = False


def source_hashes():
    names = ('sdon/calibration/roi.py', 'sdon/calibration/patterns.py',
             'sdon/calibration/intensity.py', 'sdon/calibration/alignment.py',
             'sdon/calibration/slm_check.py', 'sdon/simulation.py', 'sdon/hardware/modulator.py')
    result={name: hashlib.sha256((ARCHIVE / 'src' / name).read_bytes()).hexdigest() for name in names}
    for name in ('calibration_math.py','calibration_camera.py','calibration_worker.py','calibration_gui.py',
                 'reference_runner.py','bench.py','spin_c.py','display_linux.py','single_window_ui.py',
                 'camera_recheck.py','link_diagnostics.py','black_pixels.py'):
        result[Path(__file__).parent.name+'/'+name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return result


def signature(video, camera, settings):
    keys = ('DeviceModelName', 'DeviceSerialNumber', 'PixelFormat', 'OffsetX', 'OffsetY', 'Width', 'Height',
            'ExposureMode', 'ExposureAuto', 'ExposureTime', 'GainAuto', 'Gain', 'BlackLevel',
            'SensorShutterMode', 'TriggerMode', 'TriggerSelector', 'TriggerSource', 'TriggerActivation',
            'TriggerDelayEnabled', 'TriggerDelay', 'AcquisitionMode', 'GevSCPSPacketSize', 'GevSCPD')
    payload = dict(video=video, camera={k: camera.get(k) for k in keys},
                   illumination_note=settings.illumination_note, DMD_canvas=[600, 600],
                   SLM_canvas=[600, 600], SLM_horizontal_flip_count=1)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def embed_phase(phase, offset=(0, 0)):
    """Canonical full panel; caller flips ONCE. Offsets remain physical-screen XY."""
    if np.asarray(phase).shape != (600, 600):
        raise ValueError('a native 600x600 phase patch is required')
    ox, oy = map(int, offset)
    if abs(ox) > 660 or abs(oy) > 240:
        raise ValueError('600x600 SLM window must fit completely; no silent clipping')
    panel = np.zeros((1080, 1920), np.uint8)
    x, y = 660 - ox, 240 + oy
    panel[y:y + 600, x:x + 600] = phase
    return panel


def lens(settings, size=600):
    return generate_lens_phase((size, size), settings.lens_f_m,
                              (settings.lens_x, settings.lens_y), settings.wavelength_m, settings.pixel_pitch_m)


def simulate(dmd, phase, settings):
    """NumPy port of archival sdon.simulation's padded two-lens ASM reference.

    OFFLINE TEST/analysis helper only. Live scoring calls the unmodified
    original Torch simulator in reference_runner.py, NOT this NumPy port.
    This is its historical equal-sampling model, NOT a newly identified real
    device model or the RL policy forward. Its /255 phase convention is kept
    explicitly for reference compatibility, not used as a physical SLM LUT.
    """
    size = phase.shape[0]
    if dmd.shape != phase.shape or phase.shape != (size, size):
        raise ValueError('matching square reference planes required')
    n = size * settings.padding
    left = (n - size) // 2
    right = n - size - left
    field = np.pad(np.sqrt(dmd.astype(np.float64) / 255), ((left, right), (left, right))).astype(np.complex128)
    freq = np.fft.fftshift(np.fft.fftfreq(n, settings.pixel_pitch_m))
    fy, fx = np.meshgrid(freq, freq, indexing='ij')
    k = 2 * np.pi / settings.wavelength_m
    kz = np.sqrt(np.maximum(k*k - (2*np.pi*fx)**2 - (2*np.pi*fy)**2, 0))
    axis = (np.arange(n) - n // 2) * settings.pixel_pitch_m
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    def propagate(u, f):
        h = np.fft.fftshift(np.exp(1j * f * kz))
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(u)) * h))
    def glass(u, f):
        return u * np.exp(-1j * k * (xx*xx + yy*yy) / (2*f))
    field = propagate(glass(propagate(field, settings.f1_m), settings.f1_m), settings.f1_m)
    radians = np.pad(phase.astype(float) / 255 * 2*np.pi, ((left, right), (left, right)))
    field *= np.exp(1j * radians)
    field = propagate(glass(propagate(field, settings.f2_m), settings.f2_m), settings.f2_m)
    return (np.abs(field[left:left+size, left:left+size])**2 * 255).astype(np.float32)


def orient(image, code):
    turns, flip = code
    a = np.rot90(image, turns)
    return np.ascontiguousarray(np.fliplr(a) if flip else a)


def fit_registration(off, sim_off):
    """Choose orientation from OFF ONLY, then fit one affine on its full ROI."""
    small_sim = cv2.resize(sim_off, (192, 192))
    best = None
    for turns in range(4):
        for flip in (False, True):
            measured = cv2.resize(orient(off, (turns, flip)), (192, 192))
            warped, _ = calibrate_simulation_geometry(measured, small_sim, small_sim)
            corr = optical_structure_metrics(measured, warped)[0]
            if np.isfinite(corr) and (best is None or corr > best[0]):
                best = (corr, (turns, flip))
    if best is None:
        raise RuntimeError('OFF baseline cannot be registered')
    code = best[1]
    warped, transform = calibrate_simulation_geometry(orient(off, code), sim_off, sim_off)
    return dict(orientation=list(code), transform=transform.tolist(),
                correlation_off=float(optical_structure_metrics(orient(off, code), warped)[0]),
                **affine_geometry_metrics(transform))


def score_registered(on, off, sim_on, sim_off, registration):
    """No fitting to ON: freeze OFF geometry throughout the candidate scan."""
    measured_on = orient(on, registration['orientation'])
    measured_off = orient(off, registration['orientation'])
    transform = np.asarray(registration['transform'], np.float32)
    shape = (measured_off.shape[1], measured_off.shape[0])
    sim = cv2.warpAffine(sim_on, transform, shape)
    zero = cv2.warpAffine(sim_off, transform, shape)
    corr, ssim, edge, error = optical_structure_metrics(measured_on, sim)
    difference_corr = optical_structure_metrics(measured_on.astype(float)-measured_off, sim-zero)[0]
    _, _, shift, _ = phase_registration_shift(measured_on, measured_off)
    modulation = spatial_modulation_response(measured_on, measured_off)
    score = .5 * (corr if np.isfinite(corr) else -1) + .3 * ssim + .2 * edge - min(.4, .02*error)
    feasible = bool(np.isfinite(score) and registration['correlation_off'] >= .5
                    and abs(registration['geometry_shear_cosine']) <= .15
                    and np.isfinite(shift) and shift <= 15 and modulation >= .02)
    return dict(score=float(score), correlation_on=float(corr), structure_ssim=float(ssim),
                edge_f1=float(edge), target_centroid_error_px=float(error),
                correlation_difference=float(difference_corr), registration_shift_px=float(shift),
                spatial_modulation_response=float(modulation), preliminary_feasible=feasible,
                center_modulation_verified=False, **registration), sim


def unsafe_exposure(frame):
    values = np.asarray(frame)
    hot = (values >= 250).astype(np.uint8)
    coherent = cv2.boxFilter(hot, cv2.CV_32F, (3, 3), normalize=False).max() >= 5
    return bool(coherent or np.mean(values == 255) > .001)


def analyze_modulation(frames):
    """Rotation-invariant ±axis geometry, preserving original lens/centroid test.

    Camera axes can be rotated ~90 degrees. Require two opposite, separated,
    near-orthogonal vectors; never call a tiny noise displacement a pass.
    """
    original = analyze_slm_check(frames)
    points = {k: np.asarray(v) for k, v in original['centroids'].items()}
    zero = points['zero']
    vectors = []
    pairs = {}
    for axis in ('x', 'y'):
        a, b = points['plus_'+axis]-zero, points['minus_'+axis]-zero
        lengths = [float(np.linalg.norm(a)), float(np.linalg.norm(b))]
        cosine = float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
        separation = float(np.linalg.norm(a-b))
        pairs[axis] = dict(plus_vector=a.tolist(), minus_vector=b.tolist(),
                           shift_lengths_px=lengths, opposite_cosine=cosine, separation_px=separation,
                           passed=bool(min(lengths) >= 3 and separation >= 6 and cosine <= -.7))
        vectors.append(a-b)
    axis_cos = float(abs(np.dot(*vectors)) / max(np.linalg.norm(vectors[0])*np.linalg.norm(vectors[1]), 1e-12))
    passed = bool(original['lens']['passed'] and all(p['passed'] for p in pairs.values())
                  and axis_cos <= .3 and not any(unsafe_exposure(f) for f in frames.values()))
    return dict(passed=passed, lens=original['lens'], pairs=pairs, axis_abs_cosine=axis_cos,
                centroids=original['centroids'], original_axis_specific_analysis=original,
                coordinate_rule='raw camera vectors; NOT assumed camera-x == logical-x',
                thresholds=dict(min_shift_px=3, min_separation_px=6, max_opposite_cosine=-.7,
                                max_axis_abs_cosine=.3, min_lens_change=.02))


def feature_coverage(frames, narrow_roi):
    """Exploratory bright-feature coverage in native full-sensor coordinates.

    This is not a first-order classifier or proof that all diffracted energy
    is inside the camera. Raw images and geometric modulation tests remain
    separate; component boxes never redefine the saved hardware ROI.
    """
    results = {}
    rx, ry = narrow_roi['OffsetX'], narrow_roi['OffsetY']
    right, bottom = rx+narrow_roi['Width'], ry+narrow_roi['Height']
    for name, image in frames.items():
        values = np.asarray(image, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('finite full-sensor 2-D frame required')
        blurred = cv2.GaussianBlur(values, (0,0), 2.0)
        signal = np.maximum(blurred-float(np.percentile(blurred,20)), 0)
        threshold = max(3., float(signal.max())*.35)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((signal>=threshold).astype(np.uint8),8)
        boxes = []
        for i in range(1,count):
            x,y,w,h,area = map(int,stats[i])
            if area<9:continue
            boxes.append(dict(x=x,y=y,width=w,height=h,area=area,
                outside_narrow_roi=bool(x<rx or y<ry or x+w>right or y+h>bottom),
                near_sensor_edge=bool(x<5 or y<5 or x+w>values.shape[1]-5 or y+h>values.shape[0]-5)))
        results[name]=dict(threshold_ADU=threshold,components=boxes,
            outside_narrow_roi=any(b['outside_narrow_roi'] for b in boxes),
            near_sensor_edge=any(b['near_sensor_edge'] for b in boxes),
            no_bright_component=not bool(boxes))
    return dict(full_sensor_analysis=True,narrow_roi_reference=narrow_roi,frames=results,
        any_bright_feature_outside_narrow_roi=any(r['outside_narrow_roi'] for r in results.values()),
        any_bright_feature_near_sensor_edge=any(r['near_sensor_edge'] for r in results.values()),
        diagnostic_only=True,hardware_roi_redefined=False,
        all_optical_energy_captured_proven=False)


class Workflow:
    """No autoload and no deployment release token, even after diagnostic passes."""
    order = ('roi_candidate', 'roi', 'intensity', 'alignment', 'modulation')
    def __init__(self):
        self.data = {}
        self.links = {}

    def invalidate(self, stage):
        if stage == 'roi_candidate': self.links.clear()
        elif stage == 'alignment': self.links.pop('slm', None)
        start = self.order.index(stage)
        for key in self.order[start:]:
            self.data.pop(key, None)

    def accept(self, stage, result):
        self.invalidate(stage)
        self.data[stage] = result

    def require(self, stage, binary=False):
        if stage == 'link_dmd': return
        if stage == 'link_slm':
            dmd=self.links.get('dmd', {})
            if not dmd.get('response_observed',dmd.get('passed',False)):
                raise RuntimeError('先完成本条件下的DMD黑白／图案链路检查')
            return
        if stage in ('roi_commit',) and 'roi_candidate' not in self.data:
            raise RuntimeError('先测量完整白方块，检查ROI候选')
        if stage in ('intensity', 'align', 'coarse', 'fine', 'check') and 'roi' not in self.data:
            raise RuntimeError('先确认并写入/读回当前ROI')
        if stage in ('align', 'coarse', 'fine', 'check') and not binary and 'intensity' not in self.data:
            raise RuntimeError('先完成DMD全256级扫描，或显式选择“仅二值诊断”')
        if stage == 'check' and 'alignment' not in self.data:
            raise RuntimeError('先测量一个SLM透镜对齐候选')

    def report(self):
        return dict(stages=self.data, link_checks=self.links,
                    calibration_scope='diagnostic candidates; NOT formal HardwareManager qualification',
                    release_approved=False, hardware_trigger_accepted=False, phase_LUT_qualified=False,
                    static_stability_qualified=False, control_20ms_accepted=False)
