"""Manual preview startup: high zero-input background warns, not dark-calibrates.

Retains all first three raws. Peak >=240 still rejects startup; live coherent
saturation, malformed/incomplete/stale frames and timeouts remain fatal.
Independent quantitative protocols keep their existing mean<=30 dark gate.
"""
import time
import traceback
import numpy as np
from realtime_worker import VideoGuard, write_json
from realtime_model import display_previews
from calibration_math import unsafe_exposure
from spin_c import IncompleteFrame
from live_manual_v19_trigger_r4.worker import DirectLine0Worker


class ManualPreviewWorker(DirectLine0Worker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.zero_input_baselines=[]

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
            folder = self.directory/('startup_zero_input_%04d'%self.epoch)
            folder.mkdir(exist_ok=True)
            np.save(folder/('frame_%06d.npy'%meta['FrameID']), image, allow_pickle=False)
            baseline = dict(epoch=self.epoch, frame_id=meta['FrameID'], max_ADU=maximum,
                mean_ADU=float(image.mean()), high_background_warning=bool(image.mean()>30),
                physical_dark_confirmed=False, quantitative_dark_qualified=False)
            self.zero_input_baselines.append(baseline)
            if maximum >= 240:
                self.save_rejected_raw(image, meta, 'startup zero-input headroom gate: max>=240')
                raise RuntimeError('启动零输入接近饱和（峰值≥240）；原图已保存，不重试')
            if baseline['high_background_warning']:
                self.message.emit('零输入背景偏亮（均值 %.1f，峰值 %d）；仅供手动预览，不是暗场标定通过。' %
                    (baseline['mean_ADU'], maximum))
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
            zero_input_background_warning_mean_ADU=max((r['mean_ADU'] for r in self.zero_input_baselines
                if r['epoch']==self.epoch and r['high_background_warning']),default=None),
            acquisition_epoch=self.epoch))
        if dark:
            write_json(folder/('frame_%06d.json'%meta['FrameID']), dict(frame=meta,
                max_ADU=maximum, mean_ADU=float(image.mean()), accepted_for_manual_preview=True,
                quantitative_dark_qualified=False, high_background_warning=bool(image.mean()>30),
                first_returned_frame_NOT_necessarily_first_sensor_exposure=True))
        return image, meta


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
            # Direct Line0 startup: do not switch to Software during normal
            # operator startup. The earlier failure is retained; this path has
            # a separately recorded three-connection held-black diagnostic.
            actual = self.camera.set_roi(self.requested_roi)
            self.requested_roi = dict(actual)
            report['startup_negative_control'] = dict(performed=False,
                reason='normal startup is direct Line0, not Software-source qualification',
                TriggerSoftware_commands=0)
            report['startup_sequence'] = 'direct_Line0_no_Software_source'
            self.message.emit('直接启动Line0下降沿硬触发；验证并保留3张零输入帧，不切Software源。')
            self.reconfigure(self.requested_roi, self.exposure, initial=True)
            self.next_frame()
            self.ready.emit(dict(active=dict(self.camera.active), actual_roi=self.camera.roi(),
                zero_input_baselines=list(self.zero_input_baselines), quantitative_dark_qualified=False))
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
            report['startup_profile']='manual_preview_zero_input_not_physical_dark'
            report['zero_input_baselines']=list(self.zero_input_baselines)
            report['quantitative_dark_qualified']=False
            report.update(valid_frames=self.count, rejected_incomplete_frames=self.bad,
                acquisition_epochs=self.epoch, wall_seconds=time.monotonic()-started,
                last_frame_processing_timings_ms=list(self.processing))
            if self.directory.exists():
                write_json(self.directory/'report.json', report)
            self.result.emit(report)
