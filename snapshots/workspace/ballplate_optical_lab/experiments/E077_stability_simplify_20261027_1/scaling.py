"""One manual DMD presentation control; import has no device/Qt actions.

600-square source files remain immutable. A nearest-neighbour enlarged square
is centred in a native1920x1080 black frame. SLM and camera are untouched.
Installed only in a fresh process, never into the currently running owner.
"""
from dataclasses import asdict
import hashlib
import math
import time

import numpy as np


def side_for_scale(scale):
    scale = float(scale)
    if not math.isfinite(scale) or not 1.0 <= scale <= 1.8:
        raise ValueError('DMD scale must be between1.00 and1.80')
    return int(round(600 * scale))


def resized(image, side):
    image = np.asarray(image)
    if (image.shape != (600, 600) or image.dtype != np.uint8
            or not np.isin(image, (0, 255)).all()):
        raise ValueError('binary uint8 canonical600 required')
    if type(side) is not int or not 600 <= side <= 1080:
        raise ValueError('integer DMD side600..1080 required')
    if side == 600:
        return image.copy()
    import cv2
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_NEAREST_EXACT)


def presentation(image, side):
    square = resized(image, side)
    x, y = (1920-side)//2, (1080-side)//2
    panel = np.zeros((1080, 1920), np.uint8)
    panel[y:y+side, x:x+side] = square
    metadata = dict(source_shape=[600, 600], native_size=[1920, 1080],
                    content_rect=[x, y, side, side], scale=side/600.,
                    interpolation='nearest_exact', binary=True,
                    source_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                    physical_panel_sha256=hashlib.sha256(panel.tobytes()).hexdigest(),
                    control_qualified=False)
    return panel, metadata


def show_outputs(self, dmd, slm):
    if not self.hardware:
        return
    from PyQt5 import QtCore
    if QtCore.QThread.currentThread() != self.dmd_output.thread():
        raise RuntimeError('DMD presentation must stay on owner GUI thread')
    panel, meta = presentation(dmd, getattr(self, '_dmd_side', 600))
    if slm.shape != (1080, 1920) or slm.dtype != np.uint8:
        raise ValueError('native SLM uint8 required')
    physical = np.ascontiguousarray(np.fliplr(slm))
    # Expand the EXISTING child once; never rebuild output windows or scale a
    # native full frame through the old600-square child a second time.
    output = self.dmd_output
    if output.modulated_size != (1920, 1080):
        output.modulated_size = (1920, 1080)
        output.label.setFixedSize(1920, 1080)
        output._center_label()
    output.update_image(panel)
    self.slm_output.update_image(physical)
    self.expected = dict(dmd=panel, slm=physical)
    self.last_apply = time.monotonic()
    self.revision += 1
    self.mismatches = 0
    self.last_verified_revision = None
    self.last_arrays = dict(dmd=dmd.copy(), slm=slm.copy(), dmd_physical=panel.copy())
    self.record(dict(state='manual_display_requested', revision=self.revision,
                     settings=asdict(self.setting()), dmd_presentation=meta,
                     dmd_sha256=meta['source_sha256'],
                     slm_canonical_sha256=hashlib.sha256(slm.tobytes()).hexdigest(),
                     dmd_nonzero=bool(panel.any()), slm_nonzero=bool(slm.any()),
                     blocked_reason=self.projection_block_reason(),
                     SLM_horizontal_flip_count=1,
                     frame_pattern_association_verified=False))


def busy(window):
    if any(getattr(window, k, False) for k in
           ('override', 'roi_busy', 'reference_busy', 'spatial_busy')):
        return True
    flow = getattr(window, 'stability_workflow', None)
    if flow and (flow.active or flow.future is not None or flow.current_core() is not None):
        return True
    sampler = getattr(window, 'state_sampling_controller', None)
    return bool(sampler and sampler.current_core() is not None)


def install_reference(worker_class):
    """Resize DMD BEFORE propagation, never resize the predicted CCD image.

    Keep the existing fixed numerical grid (1920x1800 at padding3), lenses,
    pitch, SLM and phase. All enlarged squares fit this same grid. This avoids
    increasing the old padding with square size and exceeding the memory bound.
    The existing ASM approximations remain uncalibrated, not a new physical fit.
    """
    original = worker_class.target

    def target(self, request):
        side = request.get('dmd_side', 600)
        if type(side) is not int or not 600 <= side <= 1080:
            raise ValueError('bad simulation DMD side')
        if getattr(self, '_dmd_side', None) != side:
            self.cache_key = None
            self.engine.key = None
            self._dmd_side = side
        if not hasattr(self.engine, '_unscaled_prepare'):
            self.engine._unscaled_prepare = self.engine._prepare

            def prepare(dmd, panel_shape, shape, config):
                scaled = resized(dmd, self._dmd_side)
                return self.engine._unscaled_prepare(scaled, panel_shape, shape, config)
            self.engine._prepare = prepare
        result = original(self, request)
        return result
    worker_class.target = target


def install_window(cls):
    """Fresh class composition: no additional GUI inheritance hierarchy."""
    original_ensure = cls.ensure_diffraction_worker
    original_received = cls.comparison_received
    original_render = cls.render_patterns
    original_refresh = cls.refresh

    def ensure(self):
        original_ensure(self)
        worker = self.diffraction
        if not getattr(worker, '_scale_submit_bound', False):
            submit = worker.submit

            def scaled_submit(request):
                return submit(dict(request, dmd_side=getattr(self, '_dmd_side', 600)))
            worker.submit = scaled_submit
            worker._scale_submit_bound = True

    def received(self, result):
        # A user can change size after a reference capture but before the CPU
        # fit finishes. Do not accept that late old-size registration.
        if result.get('dmd_side', 600) != getattr(self, '_dmd_side', 600):
            return
        return original_received(self, result)

    def render(self):
        original_render(self)
        if hasattr(self, 'dmd_scale') and hasattr(self, 'desired_dmd'):
            panel, _ = presentation(self.desired_dmd, self._dmd_side)
            self.paint_preview(self.dmd_preview, panel)

    def refresh(self):
        original_refresh(self)
        if hasattr(self, 'dmd_scale'):
            self.dmd_scale.setEnabled(not busy(self))

    cls.ensure_diffraction_worker = ensure
    cls.comparison_received = received
    cls.render_patterns = render
    cls.refresh = refresh


def attach(window):
    from PyQt5 import QtWidgets
    if hasattr(window, 'dmd_scale'):
        return
    window._dmd_side = 600
    group = window.dmd_mode.parentWidget()
    if not isinstance(group, QtWidgets.QGroupBox):
        raise RuntimeError('DMD control group not found; do not attach elsewhere')
    group.setTitle('DMD · 1920×1080输出 / 方图等比缩放')
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel('DMD等比缩放'))
    spin = QtWidgets.QDoubleSpinBox()
    spin.setObjectName('dmd_isotropic_scale')
    spin.setRange(1., 1.8)
    spin.setDecimals(2)
    spin.setSingleStep(.05)
    spin.setValue(1.)
    spin.setSuffix(' ×')
    spin.setKeyboardTracking(False)
    row.addWidget(spin)
    group.layout().addLayout(row)
    label = QtWidgets.QLabel('600×600；居中；二值最近邻；SLM／CCD不变')
    label.setWordWrap(True)
    group.layout().addWidget(label)
    window.dmd_scale, window.dmd_scale_status = spin, label

    def changed(value):
        side = side_for_scale(value)
        if busy(window):
            spin.blockSignals(True)
            spin.setValue(window._dmd_side/600.)
            spin.blockSignals(False)
            window.note('采集／标定中保持DMD尺寸固定；本次缩放未采用。')
            return
        if side == window._dmd_side:
            return
        before = window._dmd_side
        window._dmd_side = side
        # Clear only the invalid registration/score, never write CCD ROI or
        # SLM coordinates, or reinterpret old measurements at the new size.
        window.clear_mapping()
        if hasattr(window, 'manual_geometry_prior_disable'):
            window.manual_geometry_prior_disable('DMD等比缩放已改变；旧倍率参考失效')
        window.sim_view.setText('DMD尺寸已改变，等待新仿真')
        window.diff_view.setText('DMD尺寸已改变，等待新对照')
        window.changed()
        label.setText(f'{side}×{side}；居中；二值最近邻；SLM／CCD不变')
        window.dmd_presentation_state = dict(state='dmd_scale_changed', previous_side=before,
            side=side, scale=side/600., camera_changed=False,
            slm_changed=False, previous_geometry_invalidated=True, control_qualified=False)
        if window.hardware:
            window.record(window.dmd_presentation_state)

    spin.valueChanged.connect(changed)


def tag_reference_results(worker_class):
    original = worker_class.process
    def process(self, request):
        result = original(self, request)
        result['dmd_side'] = request.get('dmd_side', 600)
        result['dmd_scale'] = result['dmd_side']/600.
        if result.get('kind') == 'zero':
            result['registration']['dmd_side'] = result['dmd_side']
        # The source canonical image is useful for provenance; the physical
        # square, rather than the native padded panel, is used by the ASM.
        if 'target_dmd' in result:
            result['target_dmd'] = resized(result['target_dmd'], result['dmd_side'])
        return result
    worker_class.process = process
