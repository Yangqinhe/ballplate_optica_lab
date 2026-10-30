"""Shared finite RAW-window filtering for native CCD preview and analysis.

No Qt/SDK on import. Independent non-overlapping windows; no rolling history.
The camera thread is the only producer; UI reads a bounded latest result.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
from functools import wraps
import json
import math
import threading
import time
import numpy as np


@dataclass(frozen=True)
class FilterSpec:
    mode: str = 'mean'
    frames: int = 10
    alpha: float = .5

    def validate(self):
        if self.mode not in ('off','mean','ema'):raise ValueError('滤波方式必须为off/mean/ema')
        if type(self.frames) is not int or not 1<=self.frames<=120:raise ValueError('RAW帧数须为1–120整数')
        if self.mode=='off' and self.frames!=1 or self.mode!='off' and self.frames<2:
            raise ValueError('关闭滤波用1帧；均值/EMA独立窗用2–120帧')
        if isinstance(self.alpha,bool) or not isinstance(self.alpha,(int,float)) or not math.isfinite(self.alpha) or not 0<self.alpha<=1:
            raise ValueError('α须在(0,1]')
        return self


def from_controls(values):
    return FilterSpec(mode=values['mode'],frames=1 if values['mode']=='off' else int(values['frames']),
                      alpha=float(values['alpha'])).validate()


class EffectiveWindow:
    """O(pixels) accumulator; push returns exactly one complete native image."""
    def __init__(self,spec):
        self.spec=spec.validate();self.reset()

    def reset(self):self.count=0;self.value=None;self.shape=None

    def push(self,raw):
        a=np.asarray(raw)
        if a.dtype!=np.uint8 or a.ndim!=2 or not a.size or a.size>2592*1944:
            raise ValueError('需要完整Mono8原始帧，不使用GUI缩略图')
        if self.shape is not None and a.shape!=self.shape:raise ValueError('独立窗内ROI发生变化')
        if self.count==0:
            self.shape=a.shape
            self.value=a.astype(np.uint32 if self.spec.mode=='mean' else np.float32)
        elif self.spec.mode=='mean':self.value+=a
        elif self.spec.mode=='ema':
            self.value*=np.float32(1-self.spec.alpha)
            self.value+=a.astype(np.float32)*np.float32(self.spec.alpha)
        self.count+=1
        if self.count<self.spec.frames:return None
        result=(self.value.astype(np.float32)/np.float32(self.spec.frames)
                if self.spec.mode=='mean' else self.value.copy())
        self.reset();return result


class NativeFeed:
    """GUI publishes settings, worker feeds native captures; no Qt in worker."""
    def __init__(self):
        self.lock=threading.Lock();self.token=None;self.spec=None;self.ready_at=0.;self.seen_gui=0.
        self.acc=None;self.last_stamp=None;self.first_meta=None;self.latest=None;self.serial=0
        self.progress=0;self.last_received=None;self.error=None

    def configure(self,token,spec,now,settle=1.5):
        spec.validate()
        with self.lock:
            if token!=self.token or spec!=self.spec:
                self.token=token;self.spec=spec;self.acc=EffectiveWindow(spec)
                self.ready_at=now+settle;self.latest=None;self.progress=0
                self.last_stamp=None;self.last_received=None;self.first_meta=None;self.error=None
            self.seen_gui=now

    def ticket(self):
        with self.lock:return self.token

    def invalidate(self,reason=None):
        with self.lock:
            self.token=None;self.latest=None;self.acc=None;self.progress=0;self.error=reason

    def feed(self,raw,meta,token,active,received):
        with self.lock:
            if token is None or token!=self.token or received<self.ready_at:return
            if received-self.seen_gui>.75:
                self.acc.reset();self.first_meta=None;self.latest=None;self.progress=0
                return
            stamp=meta.get('TimeStamp')
            if type(stamp) is not int or self.last_stamp is not None and stamp<=self.last_stamp:return
            if self.last_received is not None and received-self.last_received>2:
                self.acc.reset();self.first_meta=None;self.latest=None
            self.last_stamp=stamp;self.last_received=received
            if self.acc.count==0:self.first_meta=deepcopy(meta)
            image=self.acc.push(raw);self.progress=self.acc.count
            if image is not None:
                self.serial+=1;image.setflags(write=False);last=np.array(raw,copy=True);last.setflags(write=False)
                self.latest=dict(image=image,last_raw=last,filter=asdict(self.spec),token=token,
                    first_meta=self.first_meta,last_meta=deepcopy(meta),active=deepcopy(active),
                    completed_monotonic=received,serial=self.serial,source='native_camera_window')
                self.first_meta=None

    def snapshot(self):
        with self.lock:return dict(result=self.latest,progress=self.progress,error=self.error,ready_at=self.ready_at)


def capture_adapter(original,bridge):
    @wraps(original)
    def capture(worker,*args,**kwargs):
        dark=kwargs.get('dark',args[0] if args else False)
        ticket=bridge.feed.ticket()
        try:result=original(worker,*args,**kwargs)
        except BaseException:
            bridge.feed.invalidate('采集失败，滤波历史已清空');raise
        if not dark and ticket is not None:
            try:
                raw,meta=result
                active=dict(worker.camera.active)
                # Context lease binds the worker, acquisition epoch and camera profile.
                if (ticket[0]==id(worker) and ticket[1]==getattr(worker,'epoch',None)
                        and ticket[3]==json.dumps(active,sort_keys=True)):
                    bridge.feed.feed(raw,meta,ticket,active,time.monotonic())
            except Exception as error:
                # Derived preview failure must not stop/replace a genuine RAW capture.
                bridge.feed.invalidate('有效帧预览失败：'+str(error))
        return result
    return capture


class Bridge:
    """One fresh-window attachment; sampled previews never enter this filter."""
    def __init__(self,window,controls):
        self.window=window;self.controls=controls;self.feed=NativeFeed();self.test_result=None
        self.generation=0;self.test_active=False;self.was_running=False

    def reset(self):
        self.generation+=1;self.feed.invalidate();self.test_result=None

    def sync(self):
        w=self.window;worker=w.worker
        if getattr(w,'fault',None) or not w.running or worker is None:
            self.feed.invalidate();self.test_result=None;self.was_running=False;return
        active=bool(getattr(w,'stability_workflow',None) and w.stability_workflow.active)
        if active:
            self.feed.invalidate();self.test_active=True;return
        if self.test_active:self.reset();self.test_active=False
        packet=w.last_packet or {}
        if (time.monotonic()-packet.get('received_monotonic',-1e30)>1.5
                or w.last_verified_revision!=w.revision
                or any(getattr(w,k,False) for k in ('override','roi_busy','reference_busy','spatial_busy'))
                or getattr(worker,'manual_gain_state',{}).get('busy')):
            self.feed.invalidate();return
        spec=from_controls(self.controls.filter_parameters())
        token=(id(worker),getattr(worker,'epoch',None),w.revision,
               json.dumps(packet.get('active',{}),sort_keys=True),self.generation,
               self.controls.filter_generation)
        self.feed.configure(token,spec,time.monotonic())

    def publish_test(self,image,last_raw,row,active):
        image.setflags(write=False);last_raw.setflags(write=False)
        self.test_result=dict(image=image,last_raw=last_raw,filter=row['effective_filter'],
            first_meta={'FrameID':row['first_frame_id']},last_meta={'FrameID':row['last_frame_id']},
            active=deepcopy(active),completed_monotonic=time.monotonic(),serial=row['visit'],
            source='archived_test_window',input_id=row['input_id'],readout=row['raw8'])


def attach(window,controls):
    if window.worker is not None:raise ValueError('只能在未启动CCD的新窗口安装有效帧接口')
    import live_loop_v23.direct_alignment_gui as gui
    from PyQt5 import QtCore
    bridge=Bridge(window,controls)
    gui.LoopWorker.capture_one=capture_adapter(gui.LoopWorker.capture_one,bridge)
    timer=QtCore.QTimer(window);timer.setInterval(50);timer.timeout.connect(bridge.sync);timer.start()
    bridge.timer=timer;window.effective_frames=bridge
    return bridge
