"""CCD display and explicit manual camera controls in one preserved workbench.

Import is offline: no Qt, old workbench or SDK import. Hardware construction
is an explicit main action requiring shielding, a restart contract and no old
owner. Manual-preview exposure warnings are a scoped worker adapter; the
camera/outputs, quantitative exposure gates and actor are not replaced. Manual
Gain commands affect real raw values, only after an explicit apply action.
"""
import argparse
import hashlib
import io
import json
import numbers
import os
from pathlib import Path
import sys
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .filtering import spatial_lowpass
from .display import map_display_pair
from .storage import PROJECT, write_json_new, sha256


FOURBALL_PRESETS = {
    'r0': dict(label='运行原始四球（R0）', radius=0,
        path=str(PROJECT/'experiments/E005_observability_bank_20261023_1/R0_REF_DMD.npy'),
        sha256='96b866ed8dbf875676fecad195cf063a5c3348de94dcd35ee82986cc39766efe',
        raw_path=str(PROJECT/'experiments/E005_observability_bank_20261023_1/R0_REF_RAW96.npy'),
        raw_sha256='886106e8206bf0a80187a0f0f36e4baaf63fde95e069341a8d5976a435508523'),
    'r2': dict(label='大球调试（R2，非原始编码）', radius=2,
        path=str(PROJECT/'experiments/E012_direction_assets_20261024_1/REF_DMD.npy'),
        sha256='02b09e9e12e102c65bd549cd555f3195d72d7a1fd5a1ea253b0dc5b4177eedca',
        raw_path=str(PROJECT/'experiments/E012_direction_assets_20261024_1/REF_RAW96.npy'),
        raw_sha256='9b894c670345154dc2bae6b84fdfe6f7a215c33b38dd9eb6515ba4482d47bce6'),
}


def load_fourball_preset(key):
    """SHA-bound static render in the actual four-history packing, no devices.

    R0 matches the runtime encoding, not an ongoing MuJoCo rollout. R2 is
    explicitly a different, dilated encoding, never a new learned policy.
    """
    if key not in FOURBALL_PRESETS:
        raise ValueError('unknown explicit four-ball preset')
    preset = dict(FOURBALL_PRESETS[key])
    arrays = {}
    for path_key, hash_key in (('path','sha256'), ('raw_path','raw_sha256')):
        path = Path(preset[path_key])
        if not path.is_absolute():
            raise ValueError('explicit absolute four-ball source required')
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != preset[hash_key]:
            raise ValueError('four-ball preset source SHA differs: '+path_key)
        arrays[path_key] = np.load(io.BytesIO(blob), allow_pickle=False)
    image, raw = arrays['path'], arrays['raw_path']
    from .study import _input
    _input(raw, image)  # Same chronological 2x2 / repeat3 / black pad12.
    metadata = dict(preset, preset=key, static_demo=True, realtime=False,
        history_ages_s=[.18,.12,.06,0.], current_position_m=[.06,-.04], velocity_m_s=[0.,0.],
        canvas_shape=[600,600], frame_shape=[96,96], chronological_order='TL/TR/BL/BR oldest_to_newest',
        scale_repeat=3, black_border_pixels=12, live_MuJoCo_control=False,
        logical_ball_dilation_radius=preset['radius'])
    return dict(image=np.array(image, copy=True), metadata=metadata)


def attach_fourball_controls(window):
    """One preset row; use the old file cache/render path, never auto-arm."""
    from PyQt5 import QtCore, QtWidgets
    from stable_views import StatusLabel
    if window.worker is not None:
        raise RuntimeError('four-ball controls require a fresh unstarted GUI')
    body = window.dmd_mode.parentWidget().layout()
    row = QtWidgets.QHBoxLayout()
    presets = QtWidgets.QComboBox(); presets.setObjectName('fourball_dmd_preset')
    presets.setAccessibleName('四球输入编码')
    for key, spec in FOURBALL_PRESETS.items(): presets.addItem(spec['label'], key)
    row.addWidget(presets, 1)
    button = QtWidgets.QPushButton('载入四球图'); button.setObjectName('load_fourball_dmd')
    button.setAccessibleName('载入四球图'); row.addWidget(button)
    body.addLayout(row)
    status = StatusLabel('四幅96²历史→2×2→3倍→600²；静态示例，非实时闭环。', 2)
    body.addWidget(status)

    def blocked():
        return any(bool(getattr(window, k, False)) for k in
            ('override','roi_busy','reference_busy','deployment_busy','adaptation_active','loop_active','fault'))

    def load(key):
        try:
            if blocked(): raise RuntimeError('配置／任务处理中，暂不切换DMD输入')
            data = load_fourball_preset(key)  # Complete validation BEFORE changing cached state.
            index = window.dmd_mode.findData('file')
            if index < 0: raise RuntimeError('original DMD file mode missing')
            window.dmd_file = data['image']
            prior = window.dmd_mode.blockSignals(True)
            try: window.dmd_mode.setCurrentIndex(index)
            finally: window.dmd_mode.blockSignals(prior)
            window.changed()  # Original render/simulation/output checks, no new output windows.
            if window.hardware:
                window.record(dict(state='manual_file_loaded', path=data['metadata']['path'],
                    sha256=data['metadata']['sha256'], SLM=False, fourball_preset=data['metadata']))
            window.note('已载入'+data['metadata']['label']+'；沿用现有投图开关，不自动允许投图。')
            window.fourball_last_loaded = data['metadata']
            return True
        except Exception as error:
            window.note('四球图未载入：'+str(error))
            return False

    def refresh():
        enabled = not blocked()
        button.setEnabled(enabled); presets.setEnabled(enabled)

    button.clicked.connect(lambda: load(presets.currentData()))
    timer = QtCore.QTimer(window); timer.setInterval(200); timer.timeout.connect(refresh); timer.start()
    window.fourball_presets = presets; window.fourball_load_button = button
    window.fourball_status = status; window.fourball_timer = timer
    window.load_fourball_preset = load
    refresh()
    return status


def make_preview(raw, *, enabled=False, sigma=1., gain=16., black=10., gamma=.5, mode='fixed'):
    """Same single Mono8 frame -> two equally mapped display buffers only."""
    a = np.asarray(raw)
    if (a.ndim != 2 or not a.size or a.size > 1048576 or a.dtype != np.uint8):
        raise ValueError("nonempty native Mono8 plane, at most1048576 pixels, required")
    if type(enabled) is not bool:
        raise ValueError("enabled must be bool")
    for value, low, high in ((sigma, 0, 2), (gain, 1, 128), (black, 0, 255), (gamma, .3, 3)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError("finite bounded scalar preview controls required")
    effective = float(sigma) if enabled else 0.
    filtered = None if effective == 0 else spatial_lowpass(a, effective)
    result = map_display_pair(a,filtered,mode=mode,gain=gain,black=black,gamma=gamma)
    return dict(**result, raw_peak_ADU=int(a.max()), raw_mean_ADU=float(a.mean()),
                display_mode=mode,
                effective_sigma_pixels=effective, display_only=True, temporal_filtering=False)


def _load_legacy(root):
    """Read-only reuse of the explicitly bound successful hardware project."""
    from .references import verify
    expected = (PROJECT.parent/"ballplate_fourframe96_deploy").resolve()
    if Path(root).resolve() != expected or verify("hardware")["verified"] is not True:
        raise ValueError("preserved hardware reference identity required")
    sys.path[:0] = [str(expected), str(expected/"live_manual_v17")]
    from live_loop_v23.direct_alignment_gui import LoopWindow
    return LoopWindow


def _assert_no_other_owner(root):
    names = {str(p.resolve()) for p in Path(root).glob("live_*/direct_alignment_gui.py")}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try:
            if entry.stat().st_uid != os.getuid(): continue
            args = [s.decode() for s in (entry/"cmdline").read_bytes().split(b"\0") if s]
            if "--hardware" in args and (names.intersection(args) or "optical_lab.preview" in args):
                raise RuntimeError("old optical owner still running; no new device windows created: " + entry.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def manual_geometry_matches(prior, active, roi, config, orientation):
    """An old spatial matrix is a display prior, never photometric validation."""
    from .preview_restart import _manual_photometry_changes
    try:
        _manual_photometry_changes(prior['expected_active'], active)
        return (roi == prior['roi'] and config == prior['optical_config']
                and orientation == prior['registration']['orientation'])
    except (KeyError, TypeError, ValueError):
        return False


def attach_manual_geometry_prior(window, prior):
    """Inject only a comparison request's matrix, NEVER window.registration.

    Legacy quantitative matching therefore cannot mistake this exposure/gain-
    changed, alignment-only prior for a newly measured frozen reference.
    """
    from PyQt5 import QtWidgets
    state = dict(enabled=True, used=False, reason='waiting for matching camera frame',
                 manual_alignment_only=True, photometry_calibrated=False)
    original_request = window.reference_request_for
    original_describe = window.describe_comparison
    pending = '保留上次倍率矩阵，仅用于手动对图；等待CCD。曝光/Gain已变，非新标定。'
    active_text = '保留上次倍率矩阵，仅用于手动对图；曝光/Gain已变，未重新做光度标定，不用于部署认证。'
    window.mapping_status.setText(pending)

    def invalidate(reason='explicit mapping/model change'):
        if not state['enabled']:
            return
        state.update(enabled=False, used=False, reason=str(reason))
        window.mapping_status.setText('保留对图矩阵已撤销；可重新采零相位参考。')
        window.comparison_changed()
        if window.hardware:
            window.record(dict(state='manual_geometry_prior_disabled', reason=str(reason),
                               photometry_calibrated=False, registration_not_assigned=True))

    def request(packet):
        result = original_request(packet)
        state['used'] = False
        if window.registration is not None:
            return result  # A genuine NEW reference always takes precedence.
        if state['enabled']:
            if manual_geometry_matches(prior, packet.get('active'), window.actual_roi,
                                       asdict(window.optical_config()), window.orientation.currentData()):
                result['mapping'] = dict(result['mapping'], registration=deepcopy(prior['registration']))
                result['manual_alignment_prior_only'] = True
                state.update(used=True, reason='same ROI/model; manual photometry only')
                window.mapping_status.setText(active_text)
            else:
                invalidate('camera/ROI/model/direction no longer matches alignment prior')
        return result

    def describe(now):
        answer = original_describe(now)
        if state['enabled'] and state['used'] and window.registration is None:
            window.mapping_status.setText(active_text)
        return answer

    window.reference_request_for = request
    window.describe_comparison = describe
    window.manual_geometry_prior_state = state
    window.manual_geometry_prior_disable = invalidate
    for control in window.optics.values():
        control.valueChanged.connect(lambda *_: invalidate('optical model edited'))
    window.slm_flip_y.toggled.connect(lambda *_: invalidate('SLM model direction edited'))
    window.orientation.currentIndexChanged.connect(lambda *_: invalidate('CCD direction edited'))
    for button in window.findChildren(QtWidgets.QPushButton):
        if button.text() == '清除冻结几何 / 复位显示映射':
            button.clicked.connect(lambda *_: invalidate('operator cleared mapping'))
    if window.hardware:
        window.record(dict(state='manual_geometry_prior_pending',
            source_reference_folder=prior['source_reference_folder'],
            manual_alignment_only=True, photometry_calibrated=False,
            registration_not_assigned=True, auto_capture=False, auto_projection=False))
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hardware", action="store_true")
    p.add_argument("--light-blocked-confirmed", action="store_true")
    p.add_argument("--restart-state", type=Path); p.add_argument("--recovery-record", type=Path)
    p.add_argument("--adaptation-catalog", type=Path); p.add_argument("--loop-head", type=Path)
    p.add_argument("--smoke-test", action="store_true"); p.add_argument("--preview-screenshot", type=Path)
    p.add_argument("--preview-frame", type=Path); p.add_argument("--preview-frame-sha256")
    p.add_argument("--display-state", type=Path); p.add_argument("--display-state-sha256")
    p.add_argument("--manual-geometry-state", type=Path); p.add_argument("--manual-geometry-sha256")
    args = p.parse_args(argv)
    if args.hardware and (not args.light_blocked_confirmed or args.smoke_test or args.preview_screenshot or args.preview_frame):
        p.error("new device windows require fresh physical shielding; no offline fixture in hardware mode")
    if args.hardware and (not args.restart_state or not args.recovery_record):
        p.error("explicit retained controls and original-baseline recovery record required")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("NEW GUI session required; no overwrites")
    if args.preview_screenshot and args.preview_screenshot.exists():
        raise FileExistsError("NEW offline QA screenshot required")
    fixture = None
    initial = None
    manual_geometry = None
    if bool(args.manual_geometry_state) != bool(args.manual_geometry_sha256):
        raise ValueError('explicit alignment-only geometry path and SHA must be supplied together')
    if args.manual_geometry_state:
        from .preview_restart import validate_manual_geometry_sidecar
        manual_geometry = validate_manual_geometry_sidecar(args.manual_geometry_state, args.manual_geometry_sha256)
    if bool(args.display_state) != bool(args.display_state_sha256):
        raise ValueError('explicit retained display state and SHA must be supplied together')
    if args.display_state:
        blob=args.display_state.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=args.display_state_sha256:
            raise ValueError('retained display-state SHA differs')
        state=json.loads(blob)
        if state.get('schema')!='optical-lab-manual-display-state-1':
            raise ValueError('explicit manual display-state schema required')
        from .display import validate_display_parameters
        initial=validate_display_parameters(state['manual'])
    if args.preview_frame:
        if not args.preview_frame_sha256 or sha256(args.preview_frame) != args.preview_frame_sha256:
            raise ValueError("explicit offline frame SHA required")
        fixture = np.load(args.preview_frame, allow_pickle=False)
        make_preview(fixture)
    if args.hardware: _assert_no_other_owner(args.legacy_root)
    Window = _load_legacy(args.legacy_root)
    from PyQt5 import QtCore, QtWidgets
    import cv2
    cv2.setNumThreads(2)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(["ballplate-realtime-manual"])
    if args.hardware and app.platformName() != "xcb": raise RuntimeError("local X11 required")
    load = lambda path: None if path is None else json.loads(path.read_text())
    window = Window(args.output, args.hardware, load(args.restart_state), load(args.recovery_record),
                    adaptation_catalog=args.adaptation_catalog, loop_head=args.loop_head)
    from .exposure import attach_exposure_warnings, attach_camera_gain_controls
    attach_exposure_warnings(window)
    attach_camera_gain_controls(window)
    attach_fourball_controls(window)
    from .display import attach_manual_display, attach_adaptation_display
    panel = attach_manual_display(window, initial)
    attach_adaptation_display(window)
    if manual_geometry is not None:
        # Restore ONLY the direction selector; the global registration remains
        # absent. No synthetic signature, capture, deployment evidence or fit.
        window.orientation.blockSignals(True)
        window.orientation.setCurrentIndex(manual_geometry['registration']['orientation'])
        window.orientation.blockSignals(False)
        attach_manual_geometry_prior(window, manual_geometry)
    window.tabs.setCurrentIndex(0)
    window.setWindowTitle(window.windowTitle()+" · 相机Gain／显示增益分离")
    if fixture is not None:
        panel.offline = dict(image=fixture, meta=dict(FrameID="历史样图"), received_monotonic=-1e30)
    if args.hardware:
        write_json_new(args.output/"spatial_preview_extension.json", dict(
            single_original_window_composition=True, no_old_source_modified=True,
            display_only=False, display_mapping_only=True, actor_modified=False,
            camera_gain_controls_manual_apply_only=True, camera_gain_changes_real_raw=True,
            fourball_presets_manual_load_only=True,
            fourball_presets=FOURBALL_PRESETS,
            default_filter_enabled=False,
            spatial_preview_page_removed=True, spatial_lowpass_in_live_display=False,
            manual_display_controls_integrated=True,
            initial_manual_display=panel.controls.parameters(window.enhance.isChecked()),
            retained_display_state=None if args.display_state is None else dict(
                path=str(args.display_state.resolve()), sha256=args.display_state_sha256),
            default_display_mode='follow_manual_alignment',
            fixed_display_preset=dict(gain=4.,black_ADU=10.,gamma=1.),
            adaptation_controls_independent_of_deployment_page=True,
            manual_preview_overexposure="warning_only_including_startup_zero_input",
            quantitative_overexposure="original_strict_gates_retained",
            other_camera_and_display_faults="original_fatal_gates_retained",
            automatic_capture=False, automatic_projection=False,
            manual_geometry_prior=None if args.manual_geometry_state is None else dict(
                path=str(args.manual_geometry_state.resolve()), sha256=args.manual_geometry_sha256,
                manual_alignment_only=True, photometry_calibrated=False, registration_not_assigned=True),
            new_source_sha256={str(path): sha256(path) for path in (Path(__file__),
                Path(__file__).with_name("filtering.py"), Path(__file__).with_name("exposure.py"),
                Path(__file__).with_name("display.py"),Path(__file__).with_name("preview_restart.py"))}))
    window.show(); app.processEvents(); panel.refresh()
    if args.smoke_test or args.preview_screenshot:
        assert not window.windows and window.worker is None and not args.output.exists()
        if args.preview_screenshot and not window.grab().save(str(args.preview_screenshot)):
            raise RuntimeError("offline screenshot save failed")
        QtCore.QTimer.singleShot(350, app.quit)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
