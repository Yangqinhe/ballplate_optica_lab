"""Bounded manual display and spatial-calibration quality recovery.

No SDK/Qt/display operations on import. Installed only at fresh GUI startup.
Reject clipped calibration data without labelling it a hardware trigger fault.
Transport errors, stale frames and restoration errors remain fatal.
"""
from functools import wraps
import time
import numpy as np

SATURATION_ERRORS = frozenset({
    'CCD连片接近饱和；停止投图与采集',
    'CCD连片接近饱和；已请求全0，请增加ND或缩短曝光',
})

def preview_sample(raw, maximum=(1024,768)):
    """Display-only bounded resize. Native array is never modified."""
    a=np.asarray(raw)
    if a.dtype!=np.uint8 or a.ndim!=2 or not a.size or a.size>2592*1944:
        raise ValueError('native Mono8 up to2592x1944 required')
    w,h=maximum
    if not isinstance(w,int) or not isinstance(h,int) or min(w,h)<1 or w*h>1048576:
        raise ValueError('bounded preview dimensions required')
    scale=min(1.,w/a.shape[1],h/a.shape[0])
    if scale==1.:return a
    import cv2
    return cv2.resize(a,(max(1,round(a.shape[1]*scale)),max(1,round(a.shape[0]*scale))),interpolation=cv2.INTER_AREA)

def reference_crop(measured,active):
    """Crop a dark-subtracted fiducial for fitting, preserving SENSOR coordinates.

    This is an analysis crop, never a camera ROI write or an optical correction.
    Small full-sensor patterns otherwise lose most detail in a256-square fit.
    """
    import cv2
    image=np.asarray(measured,np.float32)
    if image.ndim!=2 or image.shape!=(int(active['Height']),int(active['Width'])) or not np.isfinite(image).all():
        raise ValueError('native finite reference and matching active ROI required')
    median=float(np.median(image));mad=float(np.median(np.abs(image-median)))
    peak=float(image.max());threshold=median+max(3.,6*1.4826*mad,.02*(peak-median))
    if peak<=threshold:raise ValueError('零相位参考信号不足，不能从底噪拟合倍率')
    mask=np.uint8(image>threshold)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    accepted=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=9]
    if not accepted:raise ValueError('零相位参考无连续有效结构')
    x0=min(int(stats[i,cv2.CC_STAT_LEFT]) for i in accepted)
    y0=min(int(stats[i,cv2.CC_STAT_TOP]) for i in accepted)
    x1=max(int(stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]) for i in accepted)
    y1=max(int(stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]) for i in accepted)
    h,w=image.shape
    if x0<2 or y0<2 or x1>w-2 or y1>h-2:
        raise ValueError('零相位参考触及当前ROI边缘；请扩大/重选ROI后再锁倍率')
    margin=max(16,round(.15*max(x1-x0,y1-y0)))
    left=max(0,x0-margin);top=max(0,y0-margin)
    right=min(w,x1+margin);bottom=min(h,y1+margin)
    cropped=image[top:bottom,left:right].copy()
    sensor=dict(active,OffsetX=str(int(active['OffsetX'])+left),OffsetY=str(int(active['OffsetY'])+top),
        Width=str(right-left),Height=str(bottom-top))
    evidence=dict(native_analysis_crop=[left,top,right-left,bottom-top],signal_bbox=[x0,y0,x1-x0,y1-y0],
        threshold_ADU=threshold,camera_ROI_changed=False,original_active=dict(active))
    return cropped,sensor,evidence

def support_aware_fit(original):
    def fit(measured,simulated,active,dmd_size=600):
        crop,sensor,evidence=reference_crop(measured,active)
        result=original(crop,simulated,sensor,dmd_size=dmd_size)
        result['fit_preprocessing']=evidence
        return result
    return fit

def recover_quality_failure(method, label):
    """Only known raw saturation is a recoverable calibration rejection.

    The original strict method still saves the rejected native frame and does
    not emit/fit a calibration result. No data or gate is relaxed.
    """
    @wraps(method)
    def call(worker,*args,**kwargs):
        roi=dict(worker.requested_roi);exposure=worker.exposure
        worker._spatial_attempt_counter=getattr(worker,'_spatial_attempt_counter',0)+1
        worker._spatial_quality_attempt=worker._spatial_attempt_counter
        try:return method(worker,*args,**kwargs)
        except RuntimeError as error:
            if str(error) not in SATURATION_ERRORS:raise
            worker.check_stop()
            worker.show('hold_black')
            if worker.requested_roi!=roi or worker.exposure!=exposure:
                worker.reconfigure(roi,exposure,resume=False)
            worker.camera.revalidate(exposure,roi)
            worker.show('calibration_park')
            worker.message.emit(label+'未采用：RAW图案局部饱和，拒绝拟合；CCD继续、两路全0。请降低标定曝光后重测。这不是触发线故障。')
            worker.calibration_quality_failure=dict(stage=label,reason=str(error),
                roi_restored=roi,requested_exposure_us=exposure,raw_rejection_retained=True,
                calibration_adopted=False,trigger_failure=False)
            from pathlib import Path
            from optical_lab.storage import write_json_new
            worker.serial+=1
            write_json_new(Path(worker.directory)/('calibration_rejected_%04d.json'%worker.serial),
                           worker.calibration_quality_failure)
        finally:worker._spatial_quality_attempt=None
    return call

def install_worker_factory(exposure_module):
    """Decorate the one fresh worker factory; no new inheritance hierarchy."""
    original=exposure_module.make_warning_worker
    def factory(*args,**kwargs):
        cls=original(*args,**kwargs)
        original_rejected=cls.save_rejected_raw
        def rejected(self,image,meta,reason):
            attempt=getattr(self,'_spatial_quality_attempt',None)
            if attempt is None:return original_rejected(self,image,meta,reason)
            from pathlib import Path
            from optical_lab.storage import write_json_new
            folder=Path(self.directory)/('spatial_rejected_attempt_%04d'%attempt)
            folder.mkdir(exist_ok=False)
            with (folder/'CCD_raw.npy').open('xb') as stream:np.save(stream,image,allow_pickle=False)
            write_json_new(folder/'metadata.json',dict(frame=meta,reason=reason,active=dict(self.camera.active),
                max_ADU=int(image.max()),mean_ADU=float(image.mean()),rejected_NOT_accepted=True))
        cls.save_rejected_raw=rejected
        for name,label in (('collect_roi','白方块定位'),('collect_zero_reference','倍率/位置参考')):
            setattr(cls,name,recover_quality_failure(getattr(cls,name),label))
        return cls
    exposure_module.make_warning_worker=factory
    return original

def install_display_transactions(cls):
    """ROI/reference/hold-black wait for actual framebuffer ACK, not Qt enqueue.

    Existing signal/thread ownership and native pixel checks remain unchanged.
    """
    original_request=cls.display_request;original_tick=cls.tick
    def request(self,r):
        kind=r['kind']
        if kind not in ('roi_override','reference_override','hold_black','calibration_park'):
            return original_request(self,r)
        try:
            if getattr(self,'calibration_display_pending',None) is not None:
                raise RuntimeError('another spatial display ACK pending')
            if kind in ('roi_override','reference_override'):
                if not self.can_project():raise RuntimeError('当前不允许标定投图')
                self.override=True;self.show_outputs(r['dmd'],r['slm'])
            elif kind=='hold_black':self.override=True;self.black_outputs()
            else:
                self.override=False;self.roi_busy=False;self.reference_busy=False
                self.armed.setChecked(False);self.black_outputs();self.dirty=True
            r['requested_revision']=self.revision
            r['requested_monotonic']=time.monotonic()
            self.calibration_display_pending=r
        except Exception as error:
            r['error']=str(error);self.black_outputs();r['done'].set()
        self.refresh()
    def tick(self):
        original_tick(self)
        r=getattr(self,'calibration_display_pending',None)
        if r is None:return
        if self.fault:
            r['error']='标定投图核验故障：'+str(self.fault)
        elif self.revision!=r['requested_revision']:
            # A dirty refresh may redundantly enqueue the SAME zero image.
            # Only accept a newer revision if the full requested arrays match.
            expected_black=r['kind'] in ('hold_black','calibration_park')
            if expected_black and self.expected and all(not a.any() for a in self.expected.values()):
                r['requested_revision']=self.revision
            else:r['error']='标定期间输出被其他操作改变'
        if r.get('error') or self.last_verified_revision==r['requested_revision']:
            if not r.get('error'):
                r['video_verification']=dict(actual_framebuffer_checked=True,revision=self.revision,
                    acknowledged_monotonic=time.monotonic(),hardware_exposure_association_verified=False)
            self.calibration_display_pending=None;r['done'].set()
    cls.display_request=request;cls.tick=tick
    return original_request,original_tick
