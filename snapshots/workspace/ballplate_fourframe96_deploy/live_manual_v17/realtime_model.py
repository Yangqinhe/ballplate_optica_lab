"""Pure image generation and latest-frame mailbox for manual live alignment."""
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image

from calibration_math import build_alignment_pattern, embed_phase, generate_lens_phase
from sdon.calibration.patterns import generate_blazed_grating


@dataclass(frozen=True)
class LivePattern:
    dmd: str = 'fiducial'
    square: int = 128
    slm: str = 'lens'
    full_panel: bool = True
    x: int = 0
    y: int = 0
    focal_mm: float = 400.
    period: int = 32


def load_image(path, phase=False):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        array=np.load(path,allow_pickle=False)
        if array.dtype!=np.uint8 or array.ndim!=2:raise ValueError('NPY必须为二维uint8，禁止猜测相位单位')
    else:
        with Image.open(path) as image:array=np.asarray(image.convert('L')).copy()
    if phase:
        if array.shape not in ((600,600),(1080,1920)):
            raise ValueError('SLM图必须为600×600或1920×1080，不自动缩放')
        return np.ascontiguousarray(array)
    h,w=array.shape
    scale=min(600/w,600/h)
    size=(max(1,round(w*scale)),max(1,round(h*scale)))
    small=cv2.resize(array,size,interpolation=cv2.INTER_NEAREST)
    canvas=np.zeros((600,600),np.uint8)
    y,x=(600-size[1])//2,(600-size[0])//2
    canvas[y:y+size[1],x:x+size[0]]=np.where(small>=128,255,0)
    return canvas


def patterns(s, dmd_file=None, slm_file=None):
    if not 16<=s.square<=600 or not 50<=s.focal_mm<=3000 or not 4<=s.period<=512:
        raise ValueError('图案参数超出手动诊断范围')
    limits=(960,540) if s.full_panel else (660,240)
    if abs(s.x)>limits[0] or abs(s.y)>limits[1]:raise ValueError('SLM坐标超出范围')
    dmd=np.zeros((600,600),np.uint8)
    if s.dmd=='fiducial':dmd=build_alignment_pattern(size=600)
    elif s.dmd=='square':
        start=(600-s.square)//2;dmd[start:start+s.square,start:start+s.square]=255
    elif s.dmd=='white':dmd.fill(255)
    elif s.dmd=='file':
        if dmd_file is None:raise ValueError('尚未加载DMD二值图')
        dmd=np.asarray(dmd_file).copy()
    elif s.dmd!='black':raise ValueError('unknown DMD mode')
    if dmd.shape!=(600,600) or dmd.dtype!=np.uint8 or not np.isin(dmd,(0,255)).all():
        raise ValueError('DMD仅允许600×600二值0/255')
    shape=(1080,1920) if s.full_panel else (600,600)
    if s.slm=='zero':phase=np.zeros(shape,np.uint8)
    elif s.slm=='lens':
        # GUI XY is physical SCREEN lens centre. The final horizontal flip is
        # applied only by the output owner. -1 compensates even-width pixels,
        # so physical centre is exactly (960+X,540+Y), not one pixel left.
        center=(-1-s.x,s.y) if s.full_panel else (-1,0)
        phase=generate_lens_phase(shape,s.focal_mm/1000,center,532e-9,8e-6)
    elif s.slm in ('+x','-x','+y','-y'):
        phase=generate_blazed_grating(shape,s.period,s.slm[1],1 if s.slm[0]=='+' else -1)
    elif s.slm=='file':
        if slm_file is None:raise ValueError('尚未加载SLM规范灰度图')
        phase=np.asarray(slm_file).copy()
        if phase.dtype!=np.uint8 or phase.shape not in ((600,600),(1080,1920)):
            raise ValueError('SLM规范图形状/类型不符')
    else:raise ValueError('unknown SLM mode')
    if phase.shape==(600,600):
        if abs(s.x)>660 or abs(s.y)>240:raise ValueError('600²窗口坐标超界，请选择窗口模式')
        phase=embed_phase(phase,(s.x,s.y))
    return np.ascontiguousarray(dmd),np.ascontiguousarray(phase)


class LatestFrame:
    """One frame only: no unbounded Qt queue of 5MP camera images."""
    def __init__(self):self.lock=threading.Lock();self.sequence=0;self.value=None
    def put(self,value):
        with self.lock:self.sequence+=1;self.value=value
    def get(self):
        with self.lock:return self.sequence,self.value


def display_previews(frame,maximum=(1024,768)):
    """Bounded display-only processing; raw sensor pixels remain untouched.

    Compute contrast from a small uint8 histogram, not a 5MP float percentile
    and float image conversion in the Qt event loop.
    """
    frame=np.asarray(frame)
    if frame.ndim!=2 or frame.dtype!=np.uint8:raise ValueError('Mono8 preview required')
    h,w=frame.shape;scale=min(1.,maximum[0]/w,maximum[1]/h)
    if scale<1:
        raw=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    else:raw=frame.copy()
    histogram=np.bincount(raw.ravel(),minlength=256);cumulative=histogram.cumsum()
    low=int(np.searchsorted(cumulative,max(1,int(np.ceil(raw.size*.05)))))
    high=int(np.searchsorted(cumulative,max(1,int(np.ceil(raw.size*.999)))))
    if high-low<3:return raw,raw
    lut=np.clip((np.arange(256,dtype=np.int32)-low)*255//(high-low),0,255).astype(np.uint8)
    return raw,cv2.LUT(raw,lut)


def drag_roi(start,end,display_rect,active_roi):
    """Widget coordinates -> native CCD sensor coordinates (no rotation)."""
    x,y,w,h=display_rect
    if w<=0 or h<=0:raise ValueError('没有可框选的CCD画面')
    def point(p):
        u=np.clip((p[0]-x)/w,0,1);v=np.clip((p[1]-y)/h,0,1)
        return (min(active_roi['Width']-1,int(u*active_roi['Width'])),
                min(active_roi['Height']-1,int(v*active_roi['Height'])))
    a,b=point(start),point(end)
    left,top=min(a[0],b[0]),min(a[1],b[1])
    width,height=abs(a[0]-b[0])+1,abs(a[1]-b[1])+1
    if min(width,height)<16:raise ValueError('ROI至少16×16相机像素')
    return dict(OffsetX=active_roi['OffsetX']+left,OffsetY=active_roi['OffsetY']+top,
                Width=width,Height=height)
