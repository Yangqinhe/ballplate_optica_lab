"""Pure NumPy Linear18 adaptation. No Qt, SDK, simulator or optical predictor."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


RULES = dict(single_frame_MAE_rad_max=.08, single_frame_p95_L2_rad_max=.20,
             improvement_over_train_constant_min=.20, within_input_action_RMS_rad_max=.04,
             independent_test_trajectories_min=4, maximum_abs_action_rad=.8,
             maximum_step_action_change_rad=.08, short_loop_simulated_seconds=3.,
             minimum_software_settle_seconds=1.5, realtime_20ms_qualified=False)


def validate_samples(x, y, groups, split):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    groups = np.asarray(groups); split = np.asarray(split)
    if (x.ndim != 3 or x.shape[-1] != 8 or x.shape[1] < 4 or y.shape != (len(x), 2)
            or groups.shape != (len(x),) or split.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.abs(y).max() > .800001):
        raise ValueError('N inputs x repeated SINGLE-FRAME 8D vectors, bounded Nx2 labels required')
    if set(split.tolist()) != {'train', 'select', 'test'}:
        raise ValueError('predeclared train/select/test trajectories required')
    cohorts = {s: set(groups[split == s].tolist()) for s in ('train', 'select', 'test')}
    if any(cohorts[a] & cohorts[b] for a, b in (('train', 'select'), ('train', 'test'), ('select', 'test'))):
        raise ValueError('trajectory leakage across splits')
    return x, y, groups, split


def predict(head, raw8):
    x = np.asarray(raw8, np.float64)
    w = np.asarray(head['weight_raw_ADU_to_rad'], np.float64)
    b = np.asarray(head['bias_rad'], np.float64)
    if (x.shape[-1:] != (8,) or w.shape != (2, 8) or b.shape != (2,)
            or not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isfinite(b).all()):
        raise ValueError('finite raw 8D single-frame feature and Linear18 required')
    return x @ w.T + b


def metrics(predictions, labels):
    error = np.asarray(predictions) - np.asarray(labels)[:, None, :]
    jitter = predictions - predictions.mean(1, keepdims=True)
    return dict(MAE_rad=float(np.abs(error).mean()), MSE_rad2=float((error**2).mean()),
                p95_L2_rad=float(np.quantile(np.linalg.norm(error, axis=-1), .95)),
                within_input_action_RMS_rad=float(np.sqrt((jitter**2).mean())),
                unclipped_over_limit_fraction=float((np.abs(predictions) > .8).mean()))


def fit_head(x, y, groups, split, rules=None):
    """Every raw observation is a training example. No temporal mean at inference.

    Training centering/scaling is folded EXACTLY into 16 weights + 2 biases.
    Selection uses only select trajectories. Test is evaluated once afterwards.
    """
    rules = dict(RULES if rules is None else rules)
    x, y, groups, split = validate_samples(x, y, groups, split)
    tr = split == 'train'; se = split == 'select'; te = split == 'test'
    center = x[tr].mean((0, 1)); scale = x[tr].reshape(-1, 8).std(0)
    if np.any(scale < 1e-8):
        raise ValueError('constant detector channel; no silently invented feature scale')
    z = (x[tr].reshape(-1, 8)-center)/scale
    design = np.column_stack((z, np.ones(len(z))))
    target = np.repeat(y[tr], x.shape[1], axis=0)
    candidates = []
    for ridge in (.001, .01, .1, 1., 10.):
        penalty = np.diag([ridge]*8+[0.])
        beta = np.linalg.solve(design.T@design/len(z)+penalty, design.T@target/len(z))
        weight = beta[:8].T/scale
        head = dict(weight_raw_ADU_to_rad=weight.tolist(),
                    bias_rad=(beta[8]-weight@center).tolist())
        candidates.append((metrics(predict(head, x[se]), y[se])['MSE_rad2'], ridge, head))
    loss, ridge, head = min(candidates, key=lambda row: row[0])
    # From this point onward parameters are frozen; no test-based refit/retry.
    result = dict(schema='real-single-frame-linear18-v22', **head, selected_ridge=ridge,
                  parameter_count=18, MACs=16, CCD_history=0, SLM_frozen=True,
                  training_center_scale_folded_into_head=True, background_folded_into_bias=True,
                  inference_uses_raw_signed_unsubtracted_integrals=True,
                  selection_MSE_rad2=loss, rules=rules, release_approved=False)
    result['metrics'] = {s: metrics(predict(result, x[split == s]), y[split == s])
                         for s in ('train', 'select', 'test')}
    constant = np.broadcast_to(y[tr].mean(0), (int(te.sum()), x.shape[1], 2))
    baseline = metrics(constant, y[te])
    test = result['metrics']['test']
    improvement = 1-test['MSE_rad2']/max(baseline['MSE_rad2'], 1e-20)
    tests = dict(MAE=test['MAE_rad'] <= rules['single_frame_MAE_rad_max'],
                 p95=test['p95_L2_rad'] <= rules['single_frame_p95_L2_rad_max'],
                 improvement=improvement >= rules['improvement_over_train_constant_min'],
                 jitter=test['within_input_action_RMS_rad'] <= rules['within_input_action_RMS_rad_max'],
                 independent_groups=len(set(groups[te].tolist())) >= rules['independent_test_trajectories_min'])
    result.update(test_vs_train_constant_MSE_improvement=improvement, constant_baseline_test=baseline,
                  test_criteria=tests, short_MuJoCo_pilot_eligible=all(tests.values()),
                  NOT_a_control_stability_certificate=True,
                  split_input_counts={s: int((split == s).sum()) for s in ('train', 'select', 'test')},
                  selected_once_without_test_feedback=True)
    return result


def audit_capture(bundle, folder):
    """Recompute features from HASHED complete raws; never trust cached vectors."""
    folder = Path(folder); report_path = folder/'capture.json'
    report = json.loads(report_path.read_text())
    plan = json.loads((folder/'PROSPECTIVE_PROTOCOL.json').read_text())
    if (not report['completed'] or report['error'] or report['rejected_incomplete_frames']
            or report.get('park_error') or not report.get('actual_park_verified')
            or report['camera_node_writes_during_test'] != 0
            or plan['bundle_sha256'] != bundle.sha):
        raise ValueError('capture incomplete/changed/unsafe; cannot fit a head')
    bundle.verify(); expected = bundle.contract['expected_active']
    for filename, sha in plan.get('sources', {}).items():
        if digest(filename) != sha:
            raise ValueError('capture worker source changed')
    if report['camera_active_at_end'] != expected:
        raise ValueError('capture active camera contract changed')
    if len(report['inputs']) != len(bundle.cases):
        raise ValueError('missing/extra input records')
    vectors = []; seen_ids = set(); last_id = -1; last_ts = -1
    for row, case in zip(report['inputs'], bundle.cases):
        if row['id'] != case['id'] or row['id'] in seen_ids or len(row['frames']) != plan['repeats']:
            raise ValueError('input order/count mismatch')
        seen_ids.add(row['id']); current = []
        verification = row['display_verification']
        if not verification.get('actual_framebuffer_checked'):
            raise ValueError('actual output readback missing')
        for record in row['frames']:
            path = folder/record['raw_relative']; meta = record['frame']
            if (digest(path) != record['raw_sha256'] or record['camera_active'] != expected
                    or meta['FrameID'] <= last_id or meta['TimeStamp'] <= last_ts
                    or meta['host_get_started_unix_ns'] < verification['verified_unix_ns']+int(plan['settle_seconds']*1e9)):
                raise ValueError('raw hash/contract/freshness mismatch')
            raw = np.load(path, allow_pickle=False)
            if raw.dtype != np.uint8 or raw.shape != bundle.weights.shape[1:] or raw.max() >= 240:
                raise ValueError('complete unsaturated native Mono8 required')
            current.append(np.einsum('khw,hw->k', bundle.weights, raw.astype(np.float64)))
            last_id, last_ts = meta['FrameID'], meta['TimeStamp']
        vectors.append(current)
    return np.asarray(vectors), report, digest(report_path)


class Bundle:
    def __init__(self, path):
        from live_deploy_v20.contract import validate
        self.path = Path(path).resolve(); self.sha = digest(self.path)
        self.data = json.loads(self.path.read_text()); self.verify()
        d = self.data
        if d['schema'] != 'real-single-frame-adaptation-inputs-v22' or d['release_approved'] is not False:
            raise ValueError('research-only v22 input bundle required')
        self.contract = validate(json.loads(Path(d['contract']).read_text()))
        self.weights = np.load(d['native_weights'], allow_pickle=False)
        shape = (self.contract['roi']['Height'], self.contract['roi']['Width'])
        if (self.weights.shape != (8, *shape) or not np.isfinite(self.weights).all()
                or (self.weights < 0).any() or (self.weights > 1).any()
                or self.weights.sum(0).max() > 1.00001 or (self.weights.sum((1, 2)) < 16).any()):
            raise ValueError('eight fixed disjoint native detector regions required')
        self.phase = np.load(d['SLM_RAW_GRAY_NO_LUT'], allow_pickle=False)
        if (self.phase.shape != (600, 600) or self.phase.dtype != np.uint8
                or not np.array_equal(self.phase, self.phase[::4, ::4].repeat(4, 0).repeat(4, 1))):
            raise ValueError('fixed 4x4-macro-pixel SLM mask required')
        self.cases = d['cases']
        if not 40 <= len(self.cases) <= 96 or len({c['id'] for c in self.cases}) != len(self.cases):
            raise ValueError('bounded diverse input set required')
        validate_samples(np.zeros((len(self.cases), 4, 8)), [c['teacher_action_rad'] for c in self.cases],
                         [c['trajectory'] for c in self.cases], [c['split'] for c in self.cases])
        from live_deploy_v20.protocol import pack_dmd
        for c in self.cases:
            if not np.array_equal(pack_dmd(np.load(c['raw96'], allow_pickle=False)),
                                  np.load(c['dmd'], allow_pickle=False)):
                raise ValueError('four chronological RAW96 packing changed')

    def verify(self):
        if digest(self.path) != self.sha:
            raise ValueError('adaptation catalog changed')
        for path, sha in self.data['bound_sha256'].items():
            if digest(path) != sha:
                raise ValueError('adaptation bound asset/source changed: '+path)


def adapt_capture(bundle, folder):
    folder = Path(folder)
    if (folder/'HEAD.json').exists():
        raise ValueError('already fitted; cannot silently retune/re-evaluate held-out data')
    x, capture, capture_sha = audit_capture(bundle, folder)
    result = fit_head(x, [c['teacher_action_rad'] for c in bundle.cases],
                      [c['trajectory'] for c in bundle.cases], [c['split'] for c in bundle.cases],
                      bundle.data['rules'])
    result.update(bundle=str(bundle.path), bundle_sha256=bundle.sha,
                  capture_sha256=capture_sha, source_sha256={str(Path(__file__).resolve()): digest(__file__)},
                  prospective_protocol_sha256=digest(folder/'PROSPECTIVE_PROTOCOL.json'),
                  phase_mode=capture['phase_mode'],
                  phase_sha256=digest(bundle.data['SLM_RAW_GRAY_NO_LUT']),
                  native_weights_sha256=digest(bundle.data['native_weights']),
                  real_hardware_features=True, old_image_gate_NOT_reclassified=True)
    write_new(folder/'HEAD.json', result)
    return result
