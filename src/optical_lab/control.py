"""Current E107 controller math extracted unchanged; no device IO."""
from collections import deque
from copy import deepcopy
import json
import threading
import numpy as np

def integrate(image, layout):
    a = np.asarray(image)
    if a.shape != tuple(layout['shape']) or not np.isfinite(a).all():
        raise ValueError('CCD尺寸或数值无效，不能使用旧帧替代')
    regions = layout['regions']
    if len(regions) != 64: raise ValueError('此电子头只接受64区')
    return np.array([a[y0:y1, x0:x1].sum(dtype=np.float64)
                     for x0,y0,x1,y1 in (r['bounds'] for r in regions)])

def action(sums, head):
    x = np.asarray(sums, dtype=float); w = np.asarray(head['weights_sum']); b = np.asarray(head['bias'])
    if x.shape != (64,) or w.shape != (64,2) or b.shape != (2,): raise ValueError('64→2 shape mismatch')
    y = x @ w + b
    if not np.isfinite(y).all(): raise ValueError('nonfinite control')
    return y

class History:
    """Ten 20ms samples; four optical frames at ages 180/120/60/0ms."""
    def __init__(self): self.frames = deque(maxlen=10)
    def push(self, image):
        a = np.asarray(image)
        if a.shape != (1,96,96) or a.dtype != np.uint8 or not np.isin(a,(0,255)).all():
            raise ValueError('binary rendered 96 frame required')
        if not self.frames: self.frames.extend(a.copy() for _ in range(9))
        self.frames.append(a.copy())
        return np.stack([self.frames[-1-i] for i in (9,6,3,0)])

def random_xy(rng, radius_cm):
    if not 0 <= radius_cm <= 20: raise ValueError('reset radius must be 0–20 cm')
    radius = radius_cm / 100 * np.sqrt(rng.random()); angle = rng.uniform(0,2*np.pi)
    return radius*np.array([np.cos(angle),np.sin(angle)])

def impulse_velocity(j_ns, angle_deg, mass):
    if not np.isfinite([j_ns,angle_deg,mass]).all() or j_ns < 0 or mass <= 0:
        raise ValueError('invalid impulse')
    theta = np.deg2rad(angle_deg)
    return j_ns/mass*np.array([np.cos(theta),np.sin(theta)])

class Burst:
    """One bounded RAW window on the existing camera owner; no Qt or SDK.

    A ticket is taken BEFORE GetNextImage. GUI arms only after display readback
    and settling; a frame already in flight therefore cannot enter the burst.
    RAW is copied to a preallocated N-frame buffer, never written on SDK thread.
    """
    def __init__(self): self.lock=threading.Lock(); self.generation=0; self.cancel()
    def cancel(self):
        with self.lock:
            self.generation+=1; self.token=None; self.raw=None; self.result=None
            self.count=0; self.error=None; self.meta=[]; self.last_stamp=None
    def arm(self, owner, epoch, active, spec, shape):
        from optical_lab.effective_frame import EffectiveWindow
        self.cancel()
        with self.lock:
            self.token=(self.generation,owner,epoch,json.dumps(active,sort_keys=True))
            self.acc=EffectiveWindow(spec); self.raw=np.empty((spec.frames,*shape),np.uint8)
            return self.token
    def ticket(self):
        with self.lock: return self.token
    def feed(self,ticket,raw,meta,owner,epoch,active):
        with self.lock:
            if ticket is None or ticket != self.token: return
            try:
                if ticket[1:] != (owner,epoch,json.dumps(active,sort_keys=True)):
                    raise ValueError('采集期间相机配置改变')
                stamp=meta.get('TimeStamp')
                if type(stamp) is not int or self.last_stamp is not None and stamp<=self.last_stamp:
                    raise ValueError('重复或倒序CCD时间戳')
                a=np.asarray(raw)
                if a.shape != self.raw.shape[1:] or a.dtype != np.uint8: raise ValueError('RAW shape/dtype changed')
                self.last_stamp=stamp; self.raw[self.count]=a; self.meta.append(deepcopy(meta)); self.count+=1
                image=self.acc.push(a)
                if image is not None:
                    self.result=dict(raw=self.raw,image=image,meta=self.meta,token=self.token)
                    self.token=None
            except Exception as e: self.error=str(e); self.token=None
    def snapshot(self):
        with self.lock: return self.result,self.count,self.error
