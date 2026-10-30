"""Manual CCD preview warns on exposure; quantitative jobs remain strict.

Default import is offline. The explicit preview launcher installs one worker
adapter into its own GUI module, not into the preserved worker classes/files.
The optional manual Gain control changes only the existing camera owner's Gain
under its hold-black transaction. Display gain remains a separate operation.
"""
import ctypes as C
from functools import wraps
import math
import threading
import time

import numpy as np

from .storage import write_json_new


STRICT_METHODS = ("collect_roi", "collect_zero_reference", "collect_spatial",
                  "collect_static", "run_adaptation", "run_pilot", "run_loop", "checked_capture")
# Manual preview reports what the SDK actually delivers, without stopping on
# numerical Gain disagreement. None is explicit metadata, not infinite safety
# tolerances. Quantitative jobs still freshly enforce the original Gain0 rule.
GAIN_REQUEST_TOLERANCE_DB = None
GAIN_READBACK_TOLERANCE_DB = None
GAIN_POLICY = 'manual_actual_readback_no_deviation_stop'


def exposure_status(raw, *, startup=False):
    """Raw Mono8 statistics, including the original local-saturation rule."""
    a = np.asarray(raw)
    if a.ndim != 2 or not a.size or a.dtype != np.uint8:
        raise ValueError("native nonempty 2D Mono8 required")
    maximum = int(a.max())
    hot_count = int(np.count_nonzero(a >= 250)) if maximum >= 250 else 0
    clipped_count = int(np.count_nonzero(a == 255)) if maximum == 255 else 0
    local = False
    if hot_count:
        # OpenCV's default BORDER_REFLECT_101, including singleton axes.
        hot = np.pad(a >= 250, 1, mode="reflect")
        counts = np.zeros(a.shape, np.uint8)
        for y in range(3):
            for x in range(3):
                np.add(counts, hot[y:y+a.shape[0], x:x+a.shape[1]], out=counts)
        local = bool(counts.max() >= 5)
    clipped = clipped_count / a.size
    saturated = bool(local or clipped > .001)
    warning = maximum >= 240
    return dict(max_ADU=maximum, mean_ADU=float(a.mean()),
                near_saturation_count=hot_count, near_saturation_fraction=hot_count/a.size,
                clipped_count=clipped_count, clipped_fraction=clipped,
                local_saturation=local, saturated=saturated, warning=warning,
                quantitative_headroom_passed=not warning, quantitative_valid=False,
                startup_zero_input=bool(startup),
                policy="manual_preview_warning_only", raw_values_modified=False,
                physical_power_safety_NOT_assessed=True)


def _strict(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        self._assert_quantitative_gain()
        camera = getattr(self, 'camera', None)
        manual = camera is not None and hasattr(camera, 'manual_gain_applied')
        prior = getattr(camera, 'manual_gain_quantitative', False)
        outer = self._preview_strict_depth == 0
        self._preview_strict_depth += 1
        try:
            if manual:
                camera.manual_gain_quantitative = True
                if outer:
                    # Worker-side only: do not qualify a job from cached Gain.
                    # Subsequent camera.revalidate calls in this whole strict
                    # scope also traverse the unchanged original validator.
                    camera.revalidate(self.exposure, self.requested_roi)
            return method(self, *args, **kwargs)
        finally:
            if manual: camera.manual_gain_quantitative = prior
            self._preview_strict_depth -= 1
    return call


def gain_capabilities(camera):
    """Read GenApi limits/unit through the existing owner's bound C library.

    This function must only run in that owner's worker thread. It never opens
    an SDK, writes a node or assumes a model-specific maximum.
    """
    handle, boolean = C.c_void_p, C.c_uint8
    for name in ('spinFloatGetMin', 'spinFloatGetMax'):
        function = getattr(camera.lib, name)
        function.argtypes = [handle, C.POINTER(C.c_double)]
        function.restype = C.c_int
    camera.lib.spinFloatGetUnit.argtypes = [handle, C.c_char_p, C.POINTER(C.c_size_t)]
    camera.lib.spinFloatGetUnit.restype = C.c_int
    with camera.node('Gain') as node:
        readable = boolean()
        camera.call('spinNodeIsReadable', node, C.byref(readable))
        if not readable.value:
            raise RuntimeError('Gain节点不可读')
        low, high = C.c_double(), C.c_double()
        camera.call('spinFloatGetMin', node, C.byref(low))
        camera.call('spinFloatGetMax', node, C.byref(high))
        unit = C.create_string_buffer(128); length = C.c_size_t(len(unit))
        camera.call('spinFloatGetUnit', node, unit, C.byref(length))
    unit = unit.value.decode('ascii', errors='strict').strip()
    if unit.lower() != 'db' or not all(map(math.isfinite, (low.value, high.value))) or high.value < low.value:
        raise RuntimeError('Gain范围／单位未经确认，不猜测dB或器件上限')
    return dict(available=True, minimum_db=low.value, maximum_db=high.value,
                unit=unit, request_tolerance_db=GAIN_REQUEST_TOLERANCE_DB,
                readback_tolerance_db=GAIN_READBACK_TOLERANCE_DB,
                gain_policy=GAIN_POLICY)


def make_manual_gain_camera(base):
    """A local manual-only camera adapter; original Gain0 contract is intact.

    A nonzero preview is compared to a real, strictly validated Gain0 startup
    baseline. The actual metadata is never rewritten to fool that validator.
    """
    class ManualGainCamera(base):
        manual_gain_requested_db = 0.
        manual_gain_actual_db = 0.
        manual_gain_contract = None
        manual_gain_applied = False
        manual_gain_quantitative = False

        def set_manual_gain_stopped(self, value):
            if self.acquiring:
                raise RuntimeError('Gain写入前必须停止采集')
            value = float(value)
            limits = gain_capabilities(self)
            if not math.isfinite(value) or not limits['minimum_db'] <= value <= limits['maximum_db']:
                raise ValueError('Gain请求超出SDK实际范围')
            self.set_temporary('GainAuto', 'Off')
            previous = self.read('Gain')
            if previous is None:
                raise RuntimeError('无法保存Gain回退值')
            # Register before a potentially partial write, retaining the
            # original undo chain and exact original session baseline.
            if float(previous) != value:
                self.undo.append((self.map, 'Gain', previous))
                self._write('Gain', str(value), self.map)
            actual = float(self.read('Gain'))
            if not math.isfinite(actual):
                raise RuntimeError('Gain实际回读不是有效有限数值：'+str(actual))
            if self.read('GainAuto') != 'Off':
                raise RuntimeError('GainAuto未保持Off')
            self.manual_gain_requested_db = value
            self.manual_gain_actual_db = actual
            self.manual_gain_applied = True
            return actual

        def start_source(self, exposure_us, source):
            requested = self.manual_gain_requested_db
            # This real startup still traverses the unchanged Gain0 contract.
            self.manual_gain_applied = False
            super().start_source(exposure_us, source)
            self.manual_gain_contract = dict(self.active)
            self.manual_gain_requested_db = self.manual_gain_actual_db = 0.
            if requested != 0:
                self.call('spinCameraEndAcquisition', self.camera); self.acquiring = False
                self.set_manual_gain_stopped(requested)
                self.guard.check(force=True)
                self.call('spinCameraBeginAcquisition', self.camera); self.acquiring = True
                self.last_frame_id = None
                self.revalidate(exposure_us, self.roi())

        def revalidate(self, exposure_us, roi):
            if not self.manual_gain_applied or self.manual_gain_quantitative:
                return super().revalidate(exposure_us, roi)
            active = self.parameters()
            if not self.manual_gain_contract:
                raise RuntimeError('缺少真实Gain0严格启动基线')
            changed = {key for key in active.keys() | self.manual_gain_contract.keys()
                       if key != 'Gain' and active.get(key) != self.manual_gain_contract.get(key)}
            if changed:
                raise RuntimeError('手动Gain预览中其他相机参数改变：'+','.join(sorted(changed)))
            actual = float(active['Gain'])
            if (not math.isfinite(actual)
                    or active.get('GainAuto') != 'Off'
                    or abs(float(active['ExposureTime'])-exposure_us) > 50
                    or any(int(active[key]) != value for key, value in roi.items())
                    or active.get('BlackLevel') != self.before.get('BlackLevel')):
                raise RuntimeError('Gain读回无效或其他相机参数不匹配')
            self.manual_gain_actual_db = actual
            self.active = active
            return active

    return ManualGainCamera


def make_warning_worker(base, *, display_previews, incomplete_type,
                        manual_camera_class=None, guard_factory=None):
    """One local adapter; the base and all other worker classes stay intact."""
    class WarningPreviewWorker(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._preview_strict_depth = 0
            self._preview_warning_frames = 0
            self._preview_warning_active = False
            self._preview_warning_saved = False
            self._preview_last_note = -float('inf')
            self._manual_gain_pending = None
            self._manual_gain_busy = False
            self._manual_gain_serial = 0
            self._manual_gain_barrier = threading.Lock()
            self.manual_gain_state = dict(available=False, actual_db=None, busy=False,
                                          reason='等待唯一CCD线程读取SDK范围')
            if manual_camera_class is not None:
                self.camera_factory = lambda: manual_camera_class(guard_factory(self.check_stop))

        def _assert_quantitative_gain(self):
            active = getattr(getattr(self, 'camera', None), 'active', {})
            if (self._manual_gain_pending is not None or self._manual_gain_busy
                    or float(active.get('Gain', 0)) != 0):
                raise RuntimeError('非零／待处理相机Gain仅供手动预览；恢复0 dB并读回后才能定量标定或控制')

        def _gain_task_busy(self):
            return any(getattr(self, name, None) for name in
                ('_manual_exposure_busy', 'configuration', 'reference_requested', 'spatial_requested', 'static_request', 'static_busy',
                 'adaptation_request', 'adaptation_busy', 'loop_request', 'loop_busy')) or bool(
                     getattr(self, 'auto_requested', None) and self.auto_requested.is_set()) or bool(
                         self._preview_strict_depth)

        def request_manual_gain(self, value):
            """GUI-thread entry: queue a number only; no SDK calls here."""
            value = float(value)
            with self._manual_gain_barrier, self.command_lock:
                state = self.manual_gain_state
                if (not state.get('available') or self._manual_gain_pending is not None
                        or self._manual_gain_busy or self._gain_task_busy()
                        or not self.camera or not self.camera.acquiring or self.startup_dark_required):
                    raise RuntimeError('CCD未就绪或另有配置／定量任务，不能更改Gain')
                if not math.isfinite(value) or not state['minimum_db'] <= value <= state['maximum_db']:
                    raise ValueError('Gain请求超出SDK实际范围')
                self._manual_gain_pending = value
                self.manual_gain_state = dict(state, busy=True, requested_db=value, error=None)

        def _publish_manual_gain_state(self):
            if not hasattr(self.camera, 'set_manual_gain_stopped'):
                return
            limits = None
            if not self.manual_gain_state.get('limits_read'):
                try:
                    limits = dict(gain_capabilities(self.camera), limits_read=True)
                except Exception as error:
                    limits = dict(available=False, limits_read=True, reason=str(error))
            with self._manual_gain_barrier:
                old = self.manual_gain_state if limits is None else dict(self.manual_gain_state, **limits)
                pending = self._manual_gain_pending
                self.manual_gain_state = dict(old, busy=self._manual_gain_busy or pending is not None,
                    actual_db=float(self.camera.active['Gain']), gain_auto=self.camera.active.get('GainAuto'),
                    requested_db=(self.camera.manual_gain_requested_db if pending is None else pending))

        def _apply_manual_gain(self, requested):
            self._manual_gain_serial += 1
            folder = self.directory/('manual_gain_%04d' % self._manual_gain_serial)
            folder.mkdir(exist_ok=False)
            report = dict(requested_db=requested, completed=False, frames=[],
                before=dict(self.camera.active), single_SDK_owner=True,
                request_tolerance_db=GAIN_REQUEST_TOLERANCE_DB,
                drift_tolerance_db=GAIN_READBACK_TOLERANCE_DB,
                gain_policy=GAIN_POLICY,
                changes=['GainAuto=Off', 'Gain'], no_automatic_exposure_or_power_change=True,
                quantitative_valid=False, original_session_rollback_preserved=True)
            try:
                self.show('hold_black')
                self.check_stop()
                self.camera.call('spinCameraEndAcquisition', self.camera.camera)
                self.camera.acquiring = False
                self.camera.set_manual_gain_stopped(requested)
                self.camera.guard.check(force=True)
                self.camera.call('spinCameraBeginAcquisition', self.camera.camera)
                self.camera.acquiring = True; self.camera.last_frame_id = None
                # Camera timestamp checks remain global; frame order is scoped
                # to the new acquisition epoch, as for original reconfigure.
                self.camera.frame_order = type(self.camera.frame_order)()
                self.camera.revalidate(self.exposure, self.requested_roi)
                self.epoch += 1; self.times.clear(); self.next_contract_check = 0.
                self.startup_dark_required = True
                for index in range(3):
                    raw, meta = self.capture_one(dark=True)
                    path = folder/('transition_%d.npy' % index)
                    with path.open('xb') as stream:
                        np.save(stream, raw, allow_pickle=False)
                    report['frames'].append(dict(raw_path=str(path), frame=meta,
                                                 active=dict(self.camera.active)))
                self.startup_dark_required = False
                report.update(completed=True, after=dict(self.camera.revalidate(self.exposure, self.requested_roi)),
                              actual_db=float(self.camera.active['Gain']),
                              actual_minus_requested_db=float(self.camera.active['Gain'])-requested)
                self._publish_manual_gain_state()
                self.message.emit('相机Gain实际 %.4f dB（请求 %.4f）；3张转场原图已保存，仅手动预览。' %
                                  (report['actual_db'], requested))
                # Existing resume never checks/arms the projection checkbox.
                self.show('resume')
            except BaseException as error:
                report['error'] = str(error)
                try: report['failed_readback'] = self.camera.parameters()
                except Exception as read_error: report['failed_readback_error'] = str(read_error)
                self.manual_gain_state = dict(self.manual_gain_state, error=str(error), busy=False,
                    actual_db=(report.get('failed_readback') or {}).get('Gain'))
                self.black.emit()
                raise
            finally:
                write_json_new(folder/'transaction.json', report)

        def next_frame(self):
            with self._manual_gain_barrier, self.command_lock:
                pending = self._manual_gain_pending; self._manual_gain_pending = None
                self._manual_gain_busy = pending is not None
            if pending is not None:
                try:
                    self._apply_manual_gain(pending)
                finally:
                    self._manual_gain_busy = False
                    self.manual_gain_state = dict(self.manual_gain_state, busy=False)
            result = super().next_frame()
            self._publish_manual_gain_state()
            return result

        def reconfigure(self, roi, exposure, initial=False, resume=True):
            from .manual_exposure import qualifies, apply
            if qualifies(self, roi, initial=initial):
                apply(self, exposure, resume=resume)
                return  # Rejection must NOT fall back to End/Begin.
            return super().reconfigure(roi, exposure, initial=initial, resume=resume)

        def capture_one(self, dark=False):
            strict = self._preview_strict_depth or any(getattr(self, name, False)
                for name in ("static_busy", "adaptation_busy", "loop_busy"))
            if strict:
                return super().capture_one(dark=dark)
            self.check_stop()
            started = time.monotonic()
            try:
                image, meta = self.camera.capture(timeout_ms=1000)
            except incomplete_type as error:
                self.bad.append(error.record)
                raise
            received = time.monotonic()
            if image.dtype != np.uint8 or image.shape != (self.requested_roi['Height'], self.requested_roi['Width']):
                self.save_rejected_raw(image, meta, 'wrong raw dtype/ROI shape')
                raise RuntimeError('硬触发原图格式／ROI不匹配')
            if self.last_timestamp is not None and meta['TimeStamp'] <= self.last_timestamp:
                self.save_rejected_raw(image, meta, 'camera timestamp did not advance across epochs')
                raise RuntimeError('硬触发时间戳未递增，拒绝旧帧')
            self.last_timestamp = meta['TimeStamp']
            status = exposure_status(image, startup=dark)
            if status['warning']:
                self._preview_warning_frames += 1
            status['warning_frames_in_this_worker'] = self._preview_warning_frames
            # Original snapshot() saves this exact FrameID-bound metadata.
            meta = dict(meta, exposure_warning=status, manual_preview_only=True,
                        quantitative_valid=status['quantitative_valid'])
            if status['warning'] and not self._preview_warning_saved:
                folder = self.directory/'preview_exposure_warning_first'
                folder.mkdir(exist_ok=False)
                with (folder/'CCD_raw.npy').open('xb') as stream:
                    np.save(stream, image, allow_pickle=False)
                write_json_new(folder/'metadata.json', dict(frame=meta, active=dict(self.camera.active),
                    preview_only=True, quantitative_valid=False, no_automatic_light_adjustment=True))
                self._preview_warning_saved = True
            if status['warning'] != self._preview_warning_active:
                # A threshold-straddling signal must not flood the Qt log.
                # Per-frame metadata and the latest-frame banner stay exact.
                if received-self._preview_last_note >= 1.:
                    self.message.emit('过曝／接近饱和：仅提示，手动预览与投图继续；该帧不得用于定量标定或控制。'
                                      if status['warning'] else '过曝提示已解除；手动预览继续。')
                    self._preview_last_note = received
                self._preview_warning_active = status['warning']
            if dark:
                folder = self.directory/('startup_zero_input_%04d'%self.epoch)
                folder.mkdir(exist_ok=True)
                with (folder/('frame_%06d.npy'%meta['FrameID'])).open('xb') as stream:
                    np.save(stream, image, allow_pickle=False)
                baseline = dict(epoch=self.epoch, frame_id=meta['FrameID'], max_ADU=status['max_ADU'],
                    mean_ADU=status['mean_ADU'], high_background_warning=bool(status['mean_ADU']>30),
                    physical_dark_confirmed=False, quantitative_dark_qualified=False,
                    exposure_warning=status)
                self.zero_input_baselines.append(baseline)
                write_json_new(folder/('frame_%06d.json'%meta['FrameID']), dict(frame=meta,
                    **baseline, accepted_for_manual_preview=True,
                    first_returned_frame_NOT_necessarily_first_sensor_exposure=True))
            raw, enhanced = display_previews(image)
            timing = dict(capture_ms=(received-started)*1000,
                          processing_ms=(time.monotonic()-received)*1000)
            self.processing.append(timing); self.count += 1; self.times.append(received)
            fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0.
            self.mailbox.put(dict(image=image, meta=meta, active=dict(self.camera.active),
                preview_raw=raw, preview_enhanced=enhanced, timing=timing, received_monotonic=received,
                frames=self.count, bad=len(self.bad), fps=fps, max_ADU=status['max_ADU'],
                exposure_warning=status, camera_hardware_trigger_enabled=True, hardware_synchronized=False,
                SLM_hardware_synchronized=False, frame_pattern_association_verified=False,
                zero_input_background_warning_mean_ADU=max((r['mean_ADU'] for r in self.zero_input_baselines
                    if r['epoch']==self.epoch and r['high_background_warning']), default=None),
                acquisition_epoch=self.epoch))
            return image, meta

    for name in STRICT_METHODS:
        if hasattr(base, name):
            setattr(WarningPreviewWorker, name, _strict(getattr(base, name)))
    def guarded_request(method, quantitative):
        @wraps(method)
        def call(self, *args, **kwargs):
            with self._manual_gain_barrier:
                if (self._manual_gain_pending is not None or self._manual_gain_busy
                        or getattr(self, '_manual_exposure_busy', False)):
                    raise RuntimeError('相机参数事务进行中，请等待实际回读')
                if quantitative:
                    self._assert_quantitative_gain()
                return method(self, *args, **kwargs)
        return call
    for name in ('configure', 'auto_roi', 'request_zero_reference', 'request_spatial',
                 'request_static', 'request_adaptation', 'request_loop'):
        if hasattr(base, name):
            setattr(WarningPreviewWorker, name, guarded_request(getattr(base, name), name != 'configure'))
    return WarningPreviewWorker


def attach_exposure_warnings(window):
    """Explicit GUI composition before acquisition; never touches an old process."""
    from PyQt5 import QtCore
    from stable_views import StatusLabel
    from realtime_model import display_previews
    from spin_c import IncompleteFrame
    import live_loop_v23.direct_alignment_gui as gui
    from live_loop_v23.worker import LoopWorker
    from live_loop_v23.camera import StreamingCamera
    from live_manual_v19_trigger.camera import WorkerGuard

    if window.worker is not None or gui.LoopWorker is not LoopWorker:
        raise RuntimeError('exposure preview adapter requires a fresh, unstarted GUI')
    gui.LoopWorker = make_warning_worker(LoopWorker, display_previews=display_previews,
        incomplete_type=IncompleteFrame, manual_camera_class=make_manual_gain_camera(StreamingCamera),
        guard_factory=WorkerGuard)
    label = StatusLabel('手动预览：过曝只提示，不中断；定量标定／部署／闭环仍拒绝过曝数据。', 2)
    label.setObjectName('manual_preview_exposure_warning')
    window.centralWidget().layout().insertWidget(2, label)
    last = [None]

    def update():
        packet = window.last_packet
        status = (packet or {}).get('meta', {}).get('exposure_warning')
        fresh = bool(window.running and packet and time.monotonic()-packet['received_monotonic'] <= 1.5)
        background_text = window.recovery_status.text()
        if '饱和保护仍启用' in background_text:
            window.recovery_status.setText(background_text.replace('饱和保护仍启用',
                '手动过曝只提示；定量标定／闭环仍严格拒绝'))
        if fresh and status and status['warning']:
            text = ('过曝／接近饱和：仅提示，画面与投图继续；本帧不用于定量或控制。\n'
                    f"原始峰值 {status['max_ADU']}/255 · 饱和255占比 {status['clipped_fraction']:.3%} · "
                    f"≥250占比 {status['near_saturation_fraction']:.3%} · FrameID {packet['meta']['FrameID']}")
            label.setStyleSheet('color:#b02020;font-weight:bold;background:#fff0e8;')
            if not window.fault and not any(getattr(window, name, False)
                    for name in ('deployment_busy', 'adaptation_active', 'loop_active')):
                window.recovery_status.setText('手动预览过曝仅提示；原图保留，该帧不用于定量标定或控制。')
        else:
            text = '手动预览：过曝只提示，不中断；定量标定／部署／闭环仍拒绝过曝数据。'
            if not fresh: text += '\nCCD未运行／无新鲜帧；采集或触发故障不由过曝提示屏蔽。'
            label.setStyleSheet('')
        label.setText(text)
        state = None if not fresh or status is None else bool(status['warning'])
        if state != last[0] and window.hardware:
            window.record(dict(state='manual_exposure_warning_transition', warning=state,
                               exposure_status=status, camera_not_restarted=True))
        last[0] = state

    timer = QtCore.QTimer(window); timer.setInterval(200); timer.timeout.connect(update); timer.start()
    window.exposure_warning_label = label; window.exposure_warning_timer = timer
    window.exposure_warning_refresh = update
    update()
    return label


def attach_camera_gain_controls(window):
    """Add explicit manual Gain requests; Qt reads only worker-published state."""
    from PyQt5 import QtCore, QtWidgets
    from stable_views import StatusLabel

    if window.worker is not None:
        raise RuntimeError('Gain controls require a fresh, unstarted GUI')
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel('相机Gain'))
    spin = QtWidgets.QDoubleSpinBox(); spin.setObjectName('manual_camera_gain_db')
    spin.setAccessibleName('相机Gain请求值dB')
    spin.setDecimals(2); spin.setSuffix(' dB'); spin.setRange(0, 0)
    spin.setMaximumWidth(100)
    spin.setKeyboardTracking(False); spin.setEnabled(False)
    apply = QtWidgets.QPushButton('应用'); apply.setObjectName('apply_manual_camera_gain')
    apply.setAccessibleName('应用相机Gain')
    zero = QtWidgets.QPushButton('归零'); zero.setObjectName('zero_manual_camera_gain')
    zero.setAccessibleName('相机Gain恢复0dB')
    for widget in (spin, apply, zero): row.addWidget(widget)
    body = window.exposure.parentWidget().layout()
    body.addLayout(row)
    label = StatusLabel('相机Gain尚未读回；显示增益仅影响画面，不改变原始CCD或积分。', 3)
    label.setObjectName('camera_gain_actual'); body.addWidget(label)
    seen_worker = [None]

    def request(value):
        worker = window.worker
        try:
            if worker is None: raise RuntimeError('请先启动唯一CCD')
            worker.request_manual_gain(value)
            window.record(dict(state='manual_camera_gain_requested', requested_db=value,
                               quantitative_valid=False, SDK_write_in_GUI_thread=False))
        except Exception as error:
            window.note('相机Gain未应用：'+str(error))
        update()

    def update():
        worker = window.worker
        state = dict(getattr(worker, 'manual_gain_state', {}) or {})
        busy = bool(state.get('busy') or window.override or window.roi_busy or window.reference_busy
                    or any(getattr(window, name, False) for name in
                           ('deployment_busy', 'adaptation_active', 'loop_active')))
        available = bool(window.running and window.live_ready and not window.fault
                         and state.get('available') and not busy)
        if state.get('available'):
            spin.setRange(state['minimum_db'], state['maximum_db'])
            if seen_worker[0] is not worker:
                spin.setValue(float(state['actual_db'])); seen_worker[0] = worker
            actual = state.get('actual_db')
            text = ('实际Gain %s dB · SDK范围 %.3f～%.3f dB · GainAuto=%s\n'
                    '以实际值为准，Gain差值不停采；非零Gain仍禁止定量／部署。' %
                    (actual, state['minimum_db'], state['maximum_db'], state.get('gain_auto')))
            if busy: text += '\n配置／任务处理中，等待实际回读。'
            if state.get('error'): text += '\n应用失败：'+state['error']
        else:
            text = '相机Gain不可调：'+state.get('reason', '等待唯一CCD线程读回SDK范围')
        spin.setEnabled(available); apply.setEnabled(available)
        zero.setEnabled(available and state['minimum_db'] <= 0 <= state['maximum_db'])
        label.setText(text)

    apply.clicked.connect(lambda: request(spin.value()))
    zero.clicked.connect(lambda: (spin.setValue(0), request(0.)))
    timer = QtCore.QTimer(window); timer.setInterval(200); timer.timeout.connect(update); timer.start()
    window.camera_gain = spin; window.camera_gain_apply = apply; window.camera_gain_zero = zero
    window.camera_gain_status = label; window.camera_gain_timer = timer
    window.camera_gain_refresh = update
    update()
    return label
