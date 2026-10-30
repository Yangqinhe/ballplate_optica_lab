"""CPU reference optics and display-only comparison; never owns hardware.

Extends the archived two-lens ASM to the actual rectangular SLM array without
resizing the DMD or the phase. Equal sampling, ideal phase depth and distances
are reference assumptions, NOT identified device parameters or a phase LUT.
"""
from dataclasses import dataclass,asdict
import hashlib

import cv2
import numpy as np
from scipy import fft


@dataclass(frozen=True)
class OpticalReference:
    wavelength_m: float=532e-9
    pitch_m: float=8e-6
    f1_m: float=.15
    f2_m: float=.10  # paper Sec. 5.1: 150 / 100 mm, NOT old config 150 / 150
    padding: int=3
    phase_depth_rad: float=2*np.pi
    panel_aperture: bool=True
    slm_flip_y: bool=False  # model-coordinate adapter, NEVER a hardware-output transform

    def validate(self):
        if not (350e-9<=self.wavelength_m<=1100e-9 and 2e-6<=self.pitch_m<=30e-6
                and .05<=self.f1_m<=1. and .05<=self.f2_m<=1.
                and self.padding in (2,3) and 0<self.phase_depth_rad<=4*np.pi
                and type(self.slm_flip_y) is bool):
            raise ValueError('参考光学参数越界；不修改实际器件')


def centred(array,shape):
    if any(a>b for a,b in zip(array.shape,shape)):raise ValueError('plane does not fit simulation grid')
    out=np.zeros(shape,array.dtype);y,x=[(b-a)//2 for a,b in zip(array.shape,shape)]
    out[y:y+array.shape[0],x:x+array.shape[1]]=array
    return out


class CachedASM:
    """Two FFT workers, one cached DMD->SLM field, bounded last-value cache."""
    def __init__(self):self.key=None;self.source_field=None;self.config=None

    def simulate(self,dmd,phase,config):
        config.validate();dmd=np.asarray(dmd);phase=np.asarray(phase)
        if dmd.ndim!=2 or phase.ndim!=2 or dmd.dtype!=np.uint8 or phase.dtype!=np.uint8:
            raise ValueError('DMD/SLM reference requires 2-D uint8, no inferred phase units')
        shape=tuple(fft.next_fast_len(max(a*config.padding,b)) for a,b in zip(dmd.shape,phase.shape))
        if np.prod(shape)>8_000_000:raise ValueError('reference simulation grid exceeds CPU memory bound')
        key=(hashlib.sha256(dmd.tobytes()).hexdigest(),dmd.shape,phase.shape,shape,config)
        if key!=self.key:self._prepare(dmd,phase.shape,shape,config);self.key=key
        # Raw canonical patterns still go unchanged to the output owner's
        # single horizontal flip. Only the simulation SLM plane has this
        # independently measured coordinate reflection, around array centre.
        model_phase=np.flipud(phase) if config.slm_flip_y else phase
        phase_field=centred(model_phase.astype(np.float32)*(config.phase_depth_rad/255.),shape)
        modulation=np.exp(1j*phase_field).astype(np.complex64)
        if config.panel_aperture:modulation*=self.aperture
        u=self.source_field*modulation
        u=self._propagate(u,self.h2);u*=self.l2;u=self._propagate(u,self.h2)
        intensity=(u.real*u.real+u.imag*u.imag)*255.
        return np.asarray(intensity,np.float32)

    def _prepare(self,dmd,panel_shape,shape,config):
        ny,nx=shape;k=2*np.pi/config.wavelength_m
        fy=fft.fftshift(fft.fftfreq(ny,config.pitch_m)).astype(np.float64)
        fx=fft.fftshift(fft.fftfreq(nx,config.pitch_m)).astype(np.float64)
        kz=np.sqrt(np.maximum(k*k-(2*np.pi*fy[:,None])**2-(2*np.pi*fx[None,:])**2,0.))
        # Removing the common carrier only removes a global phase. Compute
        # transfer/lens phase in float64 before casting complex fields to 64.
        def transfer(distance):return fft.fftshift(np.exp(1j*distance*(kz-k))).astype(np.complex64)
        y=(np.arange(ny,dtype=np.float64)-ny//2)*config.pitch_m
        x=(np.arange(nx,dtype=np.float64)-nx//2)*config.pitch_m
        radius=y[:,None]**2+x[None,:]**2
        self.h1=transfer(config.f1_m);self.h2=self.h1 if config.f2_m==config.f1_m else transfer(config.f2_m)
        self.l1=np.exp(-1j*k*radius/(2*config.f1_m)).astype(np.complex64)
        self.l2=self.l1 if config.f2_m==config.f1_m else np.exp(-1j*k*radius/(2*config.f2_m)).astype(np.complex64)
        self.aperture=centred(np.ones(panel_shape,np.float32),shape)
        u=centred(np.sqrt(dmd.astype(np.float32)/255.).astype(np.complex64),shape)
        u=self._propagate(u,self.h1);u*=self.l1;self.source_field=self._propagate(u,self.h1)
        self.config=config

    @staticmethod
    def _propagate(u,transfer):
        spectrum=fft.fft2(fft.fftshift(u),workers=2)
        return fft.ifftshift(fft.ifft2(spectrum*transfer,workers=2))


def oriented(image,code):
    result=np.rot90(image,int(code)%4)
    return np.fliplr(result) if int(code)>=4 else result


def default_mapping(model_shape,active,dmd_size=600):
    """Uncalibrated display guess, explicitly NOT an accepted registration."""
    h,w=model_shape;sx=int(active['Width'])/dmd_size;sy=int(active['Height'])/dmd_size
    cx=int(active['OffsetX'])+int(active['Width'])/2;cy=int(active['OffsetY'])+int(active['Height'])/2
    return np.array([[sx,0,cx-sx*(w//2)],[0,sy,cy-sy*(h//2)]],np.float64)


def warp_to_camera(model,active,orientation=0,scale=1.,dx=0.,dy=0.,registration=None,size=256):
    values=oriented(model,orientation if registration is None else registration['orientation'])
    matrix=default_mapping(values.shape,active) if registration is None else np.asarray(registration['matrix_sensor'],np.float64).copy()
    cx=int(active['OffsetX'])+int(active['Width'])/2;cy=int(active['OffsetY'])+int(active['Height'])/2
    matrix[:,:2]*=scale;matrix[:,2]=(matrix[:,2]-[cx,cy])*scale+[cx+dx,cy+dy]
    native=matrix.copy();matrix[:,2]-=[int(active['OffsetX']),int(active['OffsetY'])]
    matrix[0]*=size/int(active['Width']);matrix[1]*=size/int(active['Height'])
    return cv2.warpAffine(np.asarray(values,np.float32),matrix.astype(np.float32),(size,size)),native


def normalize_shape(image):
    a=np.asarray(image,np.float32)
    low,high=np.percentile(a,(10,99.7))
    if not np.isfinite(low+high) or high-low<=1e-8:return np.zeros_like(a),0.
    return np.clip((a-low)/(high-low),0,1),float(high-low)


def compare_shape(measured,simulated,active):
    m=cv2.resize(np.asarray(measured,np.float32),(256,256),interpolation=cv2.INTER_AREA)
    s=cv2.resize(np.asarray(simulated,np.float32),(256,256),interpolation=cv2.INTER_AREA)
    m,contrast=normalize_shape(m);s,span=normalize_shape(s)
    d=m-s;v=np.clip(d,-1,1)
    rgb=np.stack((1-np.maximum(-v,0),1-np.abs(v),1-np.maximum(v,0)),axis=-1)
    overlay=np.stack((m,s,np.zeros_like(m)),axis=-1)
    a=m-m.mean();b=s-s.mean();denominator=float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
    valid=bool(contrast>=3 and span>1e-8 and denominator>1e-7)
    metrics=dict(valid=valid,measured_contrast_ADU=contrast,normalized_rmse=float(np.sqrt(np.mean(d*d))),
        correlation=None if not valid else float(np.sum(a*b)/denominator),
        units='independently normalized shape; NOT ADU/phase error',
        physical_phase_calibration=False)
    return dict(sim_gray=np.round(s*255).astype(np.uint8),difference_rgb=np.round(rgb*255).astype(np.uint8),
                overlay_rgb=np.round(overlay*255).astype(np.uint8),metrics=metrics)


def fit_zero_mapping(measured,simulated,active,dmd_size=600):
    """Fit ONCE from an explicit asymmetric zero-phase reference, never ON."""
    from calibration_math import calibrate_simulation_geometry,optical_structure_metrics
    measured=np.asarray(measured,np.float32)
    small=cv2.resize(measured,(256,256),interpolation=cv2.INTER_AREA)
    best=None
    for code in range(8):
        values=oriented(simulated,code);h,w=values.shape;side=min(dmd_size,h,w)
        y,x=(h-side)//2,(w-side)//2;cropped=values[y:y+side,x:x+side]
        reference=cv2.resize(np.asarray(cropped,np.float32),(256,256),interpolation=cv2.INTER_AREA)
        warped,matrix=calibrate_simulation_geometry(small,reference,reference)
        score=optical_structure_metrics(small,warped)[0]
        if np.isfinite(score) and (best is None or score>best['correlation_zero']):
            # oriented full model -> cropped small -> measured small -> native sensor.
            transform=matrix.astype(np.float64);transform[:,:2]*=256/side
            transform[:,2]-=transform[:,:2]@np.array([x,y])
            transform[0]*=int(active['Width'])/256;transform[1]*=int(active['Height'])/256
            transform[:,2]+=[int(active['OffsetX']),int(active['OffsetY'])]
            best=dict(orientation=code,matrix_sensor=transform.tolist(),correlation_zero=float(score),
                      fit_source='explicit zero-phase asymmetric reference ONLY',frozen=True,release_approved=False)
    if best is None or best['correlation_zero']<.5:raise ValueError('零相位参考相关不足0.5；不采用映射，请检查ROI/方向/模型参数')
    return best


def reference_signature(config,active):
    keys=('DeviceModelName','DeviceSerialNumber','OffsetX','OffsetY','Width','Height',
          'ExposureTime','ExposureAuto','Gain','GainAuto','BlackLevel','PixelFormat',
          'TriggerMode','SensorShutterMode')
    import json
    payload=dict(model=asdict(config),camera={k:active.get(k) for k in keys})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
