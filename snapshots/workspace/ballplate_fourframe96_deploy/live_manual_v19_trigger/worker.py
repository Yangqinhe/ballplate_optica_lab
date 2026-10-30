"""One persistent Line0 owner; fail on first incomplete frame or timeout."""
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PyQt5 import QtCore

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'live_manual_v17'))
from realtime_worker import LiveCameraWorker, VideoGuard, write_json
from realtime_model import display_previews
from calibration_math import unsafe_exposure
from spin_c import IncompleteFrame
from live_manual_v19_trigger.camera import HardwareTriggerCamera, WorkerGuard, UnexpectedNoCommandImage


class HardwareCameraWorker(LiveCameraWorker):
    def __init__(self, video, directory, exposure=4000., roi=None, camera_factory=None):
        super().__init__(video, directory, exposure, roi,
                         camera_factory=camera_factory or (lambda: HardwareTriggerCamera(WorkerGuard(self.check_stop))))
        self.startup_dark_required = True
        self.epoch = 0
        self.last_timestamp = None
        self.next_contract_check = 0.

    def save_rejected_raw(self, image, meta, reason):
        folder = self.directory/('rejected_%04d'%self.epoch)
        folder.mkdir(exist_ok=True)
        # One fatal image per epoch, never overwrite an earlier rejection.
        image_path = folder/'CCD_raw.npy'
        if image_path.exists():
            raise RuntimeError('拒绝覆盖先前异常原图')
        np.save(image_path, image, allow_pickle=False)
        write_json(folder/'metadata.json', dict(frame=meta, reason=reason,
            active=self.camera.active, max_ADU=int(image.max()), mean_ADU=float(image.mean()),
            incomplete_or_rejected_frame_NOT_accepted=True))

    def capture_one(self, dark=False):
        self.check_stop()
        started = time.monotonic()
        try:
            image, meta = self.camera.capture(timeout_ms=1000)
        except IncompleteFrame as error:
            self.bad.append(error.record)
            raise  # No retries, old-frame filling, or free-run fallback.
        received = time.monotonic()
        if image.dtype != np.uint8 or image.shape != (self.requested_roi['Height'], self.requested_roi['Width']):
            self.save_rejected_raw(image, meta, 'wrong raw dtype/ROI shape')
            raise RuntimeError('硬触发原图格式／ROI不匹配')
        if self.last_timestamp is not None and meta['TimeStamp'] <= self.last_timestamp:
            self.save_rejected_raw(image, meta, 'camera timestamp did not advance across epochs')
            raise RuntimeError('硬触发时间戳未递增，拒绝旧帧')
        self.last_timestamp = meta['TimeStamp']
        maximum = int(image.max())
        if dark:
            folder = self.directory/('startup_dark_%04d'%self.epoch)
            folder.mkdir(exist_ok=True)
            np.save(folder/('frame_%06d.npy'%meta['FrameID']), image, allow_pickle=False)
            if maximum >= 240 or float(image.mean()) > 30:
                self.save_rejected_raw(image, meta, 'startup dark gate: max>=240 or mean>30')
                raise RuntimeError('硬触发启动黑场异常；原图已保存，不丢帧重试')
        elif maximum >= 250 and unsafe_exposure(image):
            self.save_rejected_raw(image, meta, 'original saturation protection')
            raise RuntimeError('CCD连片接近饱和；停止投图与采集')
        raw, enhanced = display_previews(image)
        now = time.monotonic()
        timing = dict(capture_ms=(received-started)*1000, processing_ms=(now-received)*1000)
        self.processing.append(timing)
        self.count += 1
        self.times.append(received)
        fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0.
        self.mailbox.put(dict(image=image, meta=meta, active=dict(self.camera.active),
            preview_raw=raw, preview_enhanced=enhanced, timing=timing, received_monotonic=received,
            frames=self.count, bad=len(self.bad), fps=fps, max_ADU=maximum,
            camera_hardware_trigger_enabled=True, hardware_synchronized=False,
            SLM_hardware_synchronized=False, frame_pattern_association_verified=False,
            acquisition_epoch=self.epoch))
        if dark:
            write_json(folder/('frame_%06d.json'%meta['FrameID']), dict(frame=meta,
                max_ADU=maximum, mean_ADU=float(image.mean()), accepted=True,
                first_returned_frame_NOT_necessarily_first_sensor_exposure=True))
        return image, meta

    def next_frame(self):
        if time.monotonic() >= self.next_contract_check:
            self.camera.revalidate(self.exposure, self.requested_roi)
            self.next_contract_check = time.monotonic()+3
        if self.startup_dark_required:
            # All three are retained and tested, not discarded until a good frame appears.
            for _ in range(3):
                pair = self.capture_one(dark=True)
            self.startup_dark_required = False
            return pair
        return self.capture_one()

    def reconfigure(self, roi, exposure, initial=False, resume=True):
        self.epoch += 1
        self.startup_dark_required = True
        self.next_contract_check = 0.
        # Parent keeps a persistent SDK, holds black, stops, sets ROI, calls
        # our Line0 start_calibration and resets per-epoch FrameID checks.
        super().reconfigure(roi, exposure, initial=initial, resume=False)
        if not initial:
            self.next_frame()  # Three bounded black frames BEFORE resuming any nonzero output.
            if resume:
                self.show('resume')

    def run(self):
        report = dict(error=None, camera_settings_restored=False, hardware_trigger_changed=True,
            camera_trigger='FrameStart/Line0/FallingEdge', no_free_run_fallback=True,
            single_SDK_connection=True, SLM_hardware_synchronized=False,
            control_20ms_accepted=False, release_approved=False,
            mode='Line0 continuous manual preview; NOT quantitative policy deployment')
        started = time.monotonic()
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            if self.video:
                self.video_guard = VideoGuard(self.video, self.video_failed)
                self.video_guard.start()
                while not self.video_guard.initial.wait(.05):
                    self.check_stop()
                self.check_stop()
            self.camera = self.camera_factory()
            report['camera_before'] = self.camera.before
            # Match the accepted single-connection qualification: a no-command
            # negative control FIRST, never a silent software-trigger fallback.
            actual = self.camera.set_roi(self.requested_roi)
            self.requested_roi = dict(actual)
            self.message.emit('全0下检查Software源不发命令应超时，随后才切Line0；不是软件触发采集。')
            try:
                report['startup_negative_control'] = self.camera.qualify_no_command(self.exposure)
            except UnexpectedNoCommandImage as error:
                self.save_rejected_raw(error.image, error.meta, str(error))
                raise
            write_json(self.directory/'startup_negative_control.json', report['startup_negative_control'])
            if not report['startup_negative_control']['passed']:
                raise RuntimeError('硬触发负对照未通过')
            self.reconfigure(self.requested_roi, self.exposure, initial=True)
            self.next_frame()
            self.ready.emit(dict(active=dict(self.camera.active), actual_roi=self.camera.roi()))
            while not self.stopped.is_set():
                with self.command_lock:
                    command = self.configuration; self.configuration = None
                    reference = self.reference_requested; self.reference_requested = False
                    spatial = self.spatial_requested; self.spatial_requested = None
                if spatial:
                    raise RuntimeError('本版不启用点阵或扫描')
                if command:
                    self.reconfigure(*command)
                if self.auto_requested.is_set():
                    self.auto_requested.clear(); self.collect_roi()
                if reference:
                    self.collect_zero_reference()
                self.next_frame()
            self.check_stop()
        except InterruptedError:
            report['operator_stopped'] = True
        except BaseException as error:
            report.update(error=str(error), traceback=traceback.format_exc())
        finally:
            self.black.emit()
            if self.video_guard:
                self.video_guard.stop(); self.video_guard.join(timeout=4)
                report['display_guard'] = dict(checks=self.video_guard.checks, error=self.video_guard.error,
                    stopped=not self.video_guard.is_alive(), probe_seconds=self.video_guard.probe_seconds)
                if self.video_guard.is_alive():
                    report['error'] = report['error'] or '显示核验线程未退出'
            if self.camera:
                report['exposure_startup_transaction'] = getattr(self.camera, 'exposure_startup_transaction', None)
                try:
                    report['stream_statistics'] = self.camera.stream_statistics()
                except Exception as error:
                    report['stream_statistics_error'] = str(error)
                try:
                    self.camera.close()
                    report['camera_settings_restored'] = True
                except Exception as error:
                    report['restoration_error'] = str(error)
                report['camera_after'] = getattr(self.camera, 'after', None)
                report['trigger_write_trace'] = getattr(self.camera, 'write_trace', [])
                before, after = report.get('camera_before', {}), report['camera_after']
                if isinstance(after, dict):
                    report['restore_differences'] = {key: dict(before=before.get(key), after=after.get(key))
                        for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
            else:
                report.update(camera_settings_restored=True, no_camera_parameters_written=True)
            report.update(valid_frames=self.count, rejected_incomplete_frames=self.bad,
                acquisition_epochs=self.epoch, wall_seconds=time.monotonic()-started,
                last_frame_processing_timings_ms=list(self.processing))
            if self.directory.exists():
                write_json(self.directory/'report.json', report)
            self.result.emit(report)
