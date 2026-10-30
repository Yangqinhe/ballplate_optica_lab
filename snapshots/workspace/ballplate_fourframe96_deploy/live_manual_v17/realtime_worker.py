"""One persistent SDK owner; manual display changes never reopen the camera."""
from collections import deque
import json
from pathlib import Path
import threading
import time
import traceback
import hashlib
import shutil

import numpy as np
from PyQt5 import QtCore

from calibration_camera import CalibrationCamera
from calibration_math import unsafe_exposure, detect_intensity_square, roi_around_square
from display_linux import read_roles, same_signals
from link_diagnostics import signal_evidence, signal_repeat_noise
from spin_c import IncompleteFrame
from realtime_model import LatestFrame,display_previews
from realtime_model import LivePattern,patterns
from spatial_calibration import SpatialPlan,point_pattern,detect_point,fit_geometry,geometry_overlay


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


class VideoGuard(threading.Thread):
    """Read-only display probe, never on the capture or Qt event-loop thread."""
    def __init__(self,expected,on_failure,probe=read_roles,interval=3.):
        super().__init__(name='optical-video-identity',daemon=True)
        self.expected=expected;self.on_failure=on_failure;self.probe=probe;self.interval=interval
        self.stopped=threading.Event();self.initial=threading.Event()
        self.error=None;self.checks=0;self.probe_seconds=[]

    def stop(self):self.stopped.set()

    def run(self):
        while not self.stopped.is_set():
            started=time.monotonic()
            try:
                same_signals(self.expected,self.probe());self.checks+=1
            except BaseException as error:
                self.error=str(error);self.on_failure(self.error);self.initial.set();return
            finally:
                self.probe_seconds.append(time.monotonic()-started)
                self.probe_seconds=self.probe_seconds[-120:]
            self.initial.set()
            if self.stopped.wait(self.interval):return


class LiveCameraWorker(QtCore.QThread):
    ready=QtCore.pyqtSignal(object)
    configured=QtCore.pyqtSignal(object)
    roi_found=QtCore.pyqtSignal(object)
    zero_reference=QtCore.pyqtSignal(object)
    spatial_result=QtCore.pyqtSignal(object)
    display=QtCore.pyqtSignal(object)
    message=QtCore.pyqtSignal(str)
    black=QtCore.pyqtSignal()
    result=QtCore.pyqtSignal(object)

    def __init__(self,video,directory,exposure=1000.,roi=None,camera_factory=CalibrationCamera):
        super().__init__()
        self.video=video;self.directory=Path(directory);self.exposure=exposure
        self.requested_roi=roi or dict(OffsetX=0,OffsetY=0,Width=0,Height=0)
        self.camera_factory=camera_factory;self.camera=None;self.mailbox=LatestFrame()
        self.stopped=threading.Event();self.auto_requested=threading.Event()
        self.command_lock=threading.Lock();self.configuration=None;self.reference_requested=False
        self.count=0;self.bad=[];self.times=deque(maxlen=120);self.serial=0
        self.video_guard=None;self.video_error=None;self.processing=deque(maxlen=120)
        self.spatial_requested=None

    def stop(self):self.stopped.set()
    def check_stop(self):
        if self.video_error:raise RuntimeError('显示身份核验失败：'+self.video_error)
        if self.stopped.is_set():raise InterruptedError('已停止实时采集')
    def video_failed(self,error):
        self.video_error=error;self.black.emit();self.stopped.set()
    def configure(self,roi,exposure):
        with self.command_lock:self.configuration=(dict(roi),float(exposure))
    def auto_roi(self):self.auto_requested.set()
    def request_zero_reference(self):
        with self.command_lock:self.reference_requested=True

    def request_spatial(self,plan):
        if not isinstance(plan,SpatialPlan):raise ValueError('declared spatial plan required')
        plan.validate()
        with self.command_lock:
            if self.spatial_requested or self.reference_requested or self.configuration or self.auto_requested.is_set():
                raise RuntimeError('another calibration/configuration command is pending')
            self.spatial_requested=plan

    def show(self,kind,dmd=None,slm=None):
        self.check_stop()
        request=dict(kind=kind,dmd=dmd,slm=slm,done=threading.Event(),error=None)
        self.display.emit(request)
        deadline=time.monotonic()+5
        while not request['done'].wait(.05):
            self.check_stop()
            if time.monotonic()>deadline:raise RuntimeError('GUI投图响应超时')
        if request['error']:raise RuntimeError(request['error'])
        return request

    def next_frame(self):
        for _ in range(4):
            self.check_stop()
            try:
                started=time.monotonic()
                image,meta=self.camera.capture(timeout_ms=1000)
                received=time.monotonic();maximum=int(image.max())
                # Exact fast path: no pixel >=250 implies neither original
                # saturation condition can be true. Never downsample safety.
                if maximum>=250 and unsafe_exposure(image):raise RuntimeError('CCD连片接近饱和；已请求全0，请增加ND或缩短曝光')
                raw,enhanced=display_previews(image)
                now=time.monotonic();timing=dict(capture_ms=(received-started)*1000,processing_ms=(now-received)*1000)
                self.processing.append(timing);self.count+=1;self.times.append(received)
                fps=(len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0.
                self.mailbox.put(dict(image=image,meta=meta,active=dict(self.camera.active),
                    preview_raw=raw,preview_enhanced=enhanced,timing=timing,
                    received_monotonic=received,frames=self.count,bad=len(self.bad),fps=fps,
                    max_ADU=maximum,hardware_synchronized=False))
                return image,meta
            except IncompleteFrame as error:
                self.bad.append(error.record);self.message.emit('丢弃不完整帧：'+str(error.record))
        raise RuntimeError('连续4张不完整帧，停止实时采集；不补旧帧')

    def reconfigure(self,roi,exposure,initial=False,resume=True):
        if not initial:self.show('hold_black')
        if self.camera.acquiring:
            self.camera.call('spinCameraEndAcquisition',self.camera.camera);self.camera.acquiring=False
            self.stopped.wait(.35);self.check_stop()
        actual=self.camera.set_roi(roi)
        # FrameIDs may restart on AcquisitionStart. Preserve monotonic checks
        # within each explicit epoch, not across a stopped/restarted stream.
        self.camera.last_frame_id=None
        self.camera.start_calibration(exposure)
        self.exposure=exposure;self.requested_roi=dict(actual);self.times.clear()
        payload=dict(actual_roi=actual,active=dict(self.camera.active),requested_roi=roi)
        self.serial+=1;write_json(self.directory/('configuration_%04d.json'%self.serial),payload)
        self.configured.emit(payload)
        if not initial and resume:self.show('resume')

    def collect_roi(self):
        self.serial+=1;folder=self.directory/('roi_measure_%04d'%self.serial);folder.mkdir()
        zero=np.zeros((600,600),np.uint8);phase=np.zeros((1080,1920),np.uint8)
        stacks={};records={};resume_allowed=False
        try:
            # Detect on MAXIMUM sensor, never an old/cropped region.
            self.reconfigure(dict(OffsetX=0,OffsetY=0,Width=0,Height=0),self.exposure,resume=False)
            for name,dmd in (('dark',zero),('white',np.full_like(zero,255))):
                self.show('roi_override',dmd,phase)
                self.stopped.wait(.5);self.check_stop()
                discarded=[self.next_frame()[1] for _ in range(2)]
                pairs=[self.next_frame() for _ in range(3)]
                stacks[name]=np.stack([p[0] for p in pairs])
                records[name]=dict(discarded=discarded,frames=[p[1] for p in pairs])
            dark=np.median(stacks['dark'],axis=0).astype(np.uint8)
            white=np.median(stacks['white'],axis=0).astype(np.uint8)
            noise=signal_repeat_noise(stacks['white'],dark,0.)
            evidence,_=signal_evidence(white,dark,noise)
            try:
                if not evidence['observed']:raise ValueError('无足够方块信号；请检查光路，也可鼠标框选ROI')
                bbox,metrics,_=detect_intensity_square(white,dark)
                if metrics['aspect_ratio']<.6 or metrics['fill_ratio']<.65:
                    raise ValueError('白方块不完整/形状不合理，请人工框选或调整光路')
                proposed=roi_around_square(self.camera.roi(),bbox,white.shape,.2)
            except (ValueError,RuntimeError) as error:
                self.message.emit('ROI未采用：'+str(error))
                write_json(folder/'error.json',dict(error=str(error)))
                resume_allowed=True
                return
            result=dict(requested_roi=proposed,bbox=list(bbox),detection=metrics,signal=evidence,
                        active=dict(self.camera.active),records=records,adopted=False,
                        requires_operator_confirm=True,release_approved=False)
            write_json(folder/'candidate.json',result);self.roi_found.emit(result)
            resume_allowed=True
        finally:
            if stacks:np.savez_compressed(folder/'raw_frames.npz',**stacks)
            if resume_allowed and not self.stopped.is_set():self.show('resume')

    def collect_zero_reference(self):
        """Explicit two-pattern capture, same ROI/exposure/SDK; no scan or fitting."""
        self.serial+=1;folder=self.directory/('zero_reference_%04d'%self.serial);folder.mkdir()
        fiducial,phase=patterns(LivePattern(dmd='fiducial',slm='zero'))
        stacks={};records={};resume_allowed=False
        try:
            for name,dmd in (('dark',np.zeros_like(fiducial)),('fiducial',fiducial)):
                self.show('reference_override',dmd,phase)
                self.stopped.wait(.7);self.check_stop()
                discarded=[self.next_frame()[1] for _ in range(2)]
                pairs=[self.next_frame() for _ in range(3)]
                stacks[name]=np.stack([p[0] for p in pairs])
                records[name]=dict(discarded=discarded,frames=[p[1] for p in pairs])
            dark=np.median(stacks['dark'],axis=0).astype(np.uint8)
            light=np.median(stacks['fiducial'],axis=0).astype(np.uint8)
            noise=signal_repeat_noise(stacks['fiducial'],dark,0.)
            evidence,_=signal_evidence(light,dark,noise)
            payload=dict(active=dict(self.camera.active),records=records,signal=evidence,
                folder=str(folder),hardware_synchronized=False,release_approved=False,
                sequence='DMD dark -> asymmetric fiducial; SLM zero throughout',
                camera_configuration_unchanged=True)
            write_json(folder/'capture.json',payload)
            if not evidence['observed']:
                self.message.emit('零相位参考信号不足，未采用映射；恢复手动图案。')
            else:
                payload['measured']=light.astype(np.float32)-dark
                self.zero_reference.emit(payload)
            resume_allowed=True
        finally:
            if stacks:np.savez_compressed(folder/'raw_frames.npz',**stacks)
            if resume_allowed and not self.stopped.is_set():self.show('resume')

    def collect_spatial(self,plan):
        """Bounded native-frame acquisition in the EXISTING SDK owner thread."""
        plan.validate();self.serial+=1
        folder=self.directory/f'spatial_{self.serial:04d}_{plan.mode}';folder.mkdir()
        catalog=plan.catalog();write_json(folder/'plan.json',catalog)
        report=dict(plan_sha256=catalog['sha256'],plan=catalog['plan'],requested_exposure_us=self.exposure,folder=str(folder),completed=False,
                    error=None,groups=[],release_approved=False,
                    hardware_synchronized=False,geometry_installed_in_model=False)
        deadline=time.monotonic()+plan.timeout_s;points=[];dark=None;dark_end=None
        phase=np.zeros((1080,1920),np.uint8)
        try:
            if shutil.disk_usage(folder).free<2_000_000_000:
                raise RuntimeError('原始帧剩余空间不足2 GB；未开始投点阵')
            self.reconfigure(dict(OffsetX=0,OffsetY=0,Width=0,Height=0),self.exposure,resume=False)
            active=dict(self.camera.active);report['active']=active
            # At most 45 groups, fixed before measurement. No retries or model
            # selection on the held-out validation set.
            for i,group in enumerate(catalog['groups']):
                self.check_stop()
                if time.monotonic()>deadline:raise RuntimeError('点阵测量超出预定总时限')
                dmd=point_pattern(group['xy'],plan.radius)
                request=self.show('spatial_override',dmd,phase)
                verified=request.get('video_verification') or {}
                if (verified.get('matches') is not True or verified.get('actual_framebuffer_checked') is not True
                        or verified.get('DMD_sha256')!=hashlib.sha256(dmd.tobytes()).hexdigest()
                        or verified.get('SLM_canonical_sha256')!=hashlib.sha256(phase.tobytes()).hexdigest()):
                    raise RuntimeError('点阵实际显示未核验，不采用CCD帧')
                self.stopped.wait(plan.settle_s);self.check_stop()
                discarded=[self.next_frame()[1] for _ in range(plan.discard)]
                pairs=[self.next_frame() for _ in range(plan.frames)]
                stack=np.stack([p[0] for p in pairs]);mean=stack.mean(0,dtype=np.float32)
                row=dict(**group,discarded=discarded,frames=[p[1] for p in pairs],
                         video_verification=request.get('video_verification'),
                         DMD_sha256=hashlib.sha256(dmd.tobytes()).hexdigest(),
                         SLM_canonical_sha256=hashlib.sha256(phase.tobytes()).hexdigest())
                raw=folder/(group['name']+'.npz')
                np.savez_compressed(raw,CCD_raw=stack,DMD_canonical=dmd)
                row['raw_file']=raw.name;row['raw_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
                report['groups'].append(row)
                # Persist raw evidence BEFORE detection can fail.
                write_json(folder/'capture.json',report)
                if group['name']=='dark_start':dark=mean
                elif group['name']=='dark_end':dark_end=mean
                else:
                    row['detection']=detect_point(mean,dark,active,plan)
                    points.append(dict(name=group['name'],split=group['split'],xy=group['xy'],detection=row['detection']))
                self.message.emit(f'空间标定 {i+1}/{len(catalog["groups"])}：{group["name"]}；原始帧已保存')
                if time.monotonic()>deadline:raise RuntimeError('点阵测量超出预定总时限')
            result=fit_geometry(points,plan)
            dark_rms=float(np.sqrt(np.mean((dark_end-dark)**2)))
            result.update(dark_return_rms_ADU=dark_rms,dark_return_passed=bool(dark_rms<=plan.min_signal_ADU))
            result['spatial_geometry_validated'] &= result['dark_return_passed']
            result['pilot_validated']=bool(result['return_passed'] and result['dark_return_passed'])
            report.update(completed=True,analysis=result)
            from PIL import Image
            Image.fromarray(geometry_overlay(result,active)).save(folder/'POINT_GEOMETRY.png')
        except BaseException as error:
            report.update(error=str(error),traceback=traceback.format_exc())
            raise
        finally:
            write_json(folder/'capture.json',report)
            self.spatial_result.emit(report)
            # No automatic return to a previously nonzero manual pattern.
            self.black.emit()
            if not self.stopped.is_set():self.show('spatial_park')

    def run(self):
        report=dict(error=None,camera_settings_restored=False,hardware_trigger_changed=False,
                    release_approved=False,mode='manual continuous preview; NOT quantitative phase comparison')
        started=time.monotonic()
        try:
            self.directory.mkdir(parents=True,exist_ok=False)
            if self.video:
                self.video_guard=VideoGuard(self.video,self.video_failed);self.video_guard.start()
                while not self.video_guard.initial.wait(.05):self.check_stop()
                self.check_stop()
            self.camera=self.camera_factory();report['camera_before']=self.camera.before
            self.reconfigure(self.requested_roi,self.exposure,initial=True)
            self.next_frame()  # First complete safe dark frame before enabling projection.
            self.ready.emit(dict(active=dict(self.camera.active),actual_roi=self.camera.roi()))
            while not self.stopped.is_set():
                with self.command_lock:
                    command=self.configuration;self.configuration=None
                    reference=self.reference_requested;self.reference_requested=False
                    spatial=self.spatial_requested;self.spatial_requested=None
                if command:self.reconfigure(*command)
                if self.auto_requested.is_set():
                    self.auto_requested.clear();self.collect_roi()
                if reference:self.collect_zero_reference()
                if spatial:self.collect_spatial(spatial)
                self.next_frame()
            self.check_stop()
        except InterruptedError:report['operator_stopped']=True
        except BaseException as error:report.update(error=str(error),traceback=traceback.format_exc())
        finally:
            self.black.emit()
            if self.video_guard:
                self.video_guard.stop();self.video_guard.join(timeout=4)
                report['display_guard']=dict(checks=self.video_guard.checks,error=self.video_guard.error,
                    probe_seconds=self.video_guard.probe_seconds,independent_of_capture_thread=True,
                    query='xrandr --current --verbose; server state, no hardware polling',
                    stopped=not self.video_guard.is_alive())
                if self.video_guard.is_alive():report['error']=report.get('error') or '显示核验线程未按时退出'
            if self.camera:
                report['exposure_startup_transaction']=getattr(self.camera,'exposure_startup_transaction',None)
                try:report['stream_statistics']=self.camera.stream_statistics()
                except Exception as error:report['stream_statistics_error']=str(error)
                try:self.camera.close();report['camera_settings_restored']=True
                except Exception as error:report['restoration_error']=str(error)
                report['camera_after']=getattr(self.camera,'after',None)
                before,after=report.get('camera_before',{}),report['camera_after']
                if isinstance(after,dict):report['restore_differences']={k:dict(before=before.get(k),after=after.get(k))
                    for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
            else:
                # Opening failed before any temporary node writes. Keep the
                # error, but do not demand an unknowable recovery target.
                report.update(camera_settings_restored=True,no_camera_parameters_written=True)
            report.update(valid_frames=self.count,rejected_incomplete_frames=self.bad,wall_seconds=time.monotonic()-started)
            report['last_frame_processing_timings_ms']=list(self.processing)
            if self.directory.exists():write_json(self.directory/'report.json',report)
            self.result.emit(report)
