"""Four RAW96 tiles -> one historical-grid ASM -> CURRENT CCD8 -> Linear18.

The 8 um pitch is the archive's REGISTERED-GRID approximation, not proof of
the current optical magnification or raw CCD pixel pitch. All exports fail
closed for live deployment until new calibration and optical timing pass.
"""
import runtime96 as rt
from collections import deque
from dataclasses import dataclass,asdict
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mnist_sdon.config import OpticalConfig
from mnist_sdon.optical import DifferentiableOpticalLayer


@dataclass(frozen=True)
class ModelConfig:
    frame_size:int=96
    canvas:int=600
    dmd_repeat:int=3
    slm_block:int=8
    simulation_size:int=1200
    padding:int=3
    wavelength_m:float=532e-9
    registered_pitch_m:float=8e-6
    f1_m:float=.15
    f2_m:float=.15
    adu_per_intensity:float=32.
    feature_reference:float=100000.
    def __post_init__(self):
        if self.frame_size!=96 or self.canvas!=600 or self.dmd_repeat!=3:raise ValueError("v1 geometry is fixed and audited")
        if self.slm_block not in (4,8) or self.canvas%self.slm_block:raise ValueError("SLM needs exact 4x4 or 8x8 blocks")
        if self.simulation_size%self.canvas or self.simulation_size<600:raise ValueError("whole samples per archive canvas pixel required")
        if self.padding<3 or self.padding%2!=1:raise ValueError("use odd padding >=3")
        if min(self.adu_per_intensity,self.feature_reference)<=0:raise ValueError("positive fixed detector scale required")


def resize_binary(images):
    """Full field, area resampling, fixed threshold; no image moments or crop."""
    a=np.asarray(images)
    if a.dtype!=np.uint8 or a.shape[-2:] not in ((128,128),(96,96)) or not np.all((a==0)|(a==255)):
        raise ValueError("binary uint8 RAW 128 or RAW96 required")
    if a.shape[-2:]==(96,96):return a.copy()
    flat=a.reshape(-1,128,128)
    out=np.stack([np.where(cv2.resize(x,(96,96),interpolation=cv2.INTER_AREA)>=127.5,255,0).astype(np.uint8) for x in flat])
    return out.reshape(*a.shape[:-2],96,96)


def pack(raw):
    if raw.dtype!=torch.uint8 or raw.ndim!=5 or tuple(raw.shape[1:])!=(4,1,96,96):raise ValueError("exactly four RAW96 inputs required")
    if bool(((raw!=0)&(raw!=255)).any()):raise ValueError("nonbinary DMD input")
    return torch.cat((torch.cat((raw[:,0],raw[:,1]),-1),torch.cat((raw[:,2],raw[:,3]),-1)),-2)


def dmd_canvas(raw):
    image=pack(raw).repeat_interleave(3,-2).repeat_interleave(3,-1)
    return F.pad(image,(12,12,12,12))


class FIFO96:
    def __init__(self):self.history=deque(maxlen=4)
    def reset(self):self.history.clear()
    def push(self,image):
        image=resize_binary(image)
        if image.shape!=(1,96,96):raise ValueError("one camera RAW image expected")
        if not self.history:self.history.extend(image.copy() for _ in range(3))
        self.history.append(image.copy());return np.stack(self.history)


def calibration_tables():
    lut=np.asarray(np.load(rt.PHASE/"slm_phase_lut.npy",allow_pickle=False))
    response=np.asarray(np.load(rt.PHASE/"slm_phase_response_rad.npy",allow_pickle=False),np.float32)
    if lut.shape!=(256,) or response.shape!=(256,) or not np.isfinite(response).all():raise ValueError("invalid historical phase tables")
    if np.any(lut<0) or np.any(lut>255):raise ValueError("invalid LUT range")
    lut=np.rint(lut).astype(np.uint8)
    return lut,response,response[lut]


def quantize(value):return value+(value.round()-value).detach()


class ArchiveOptics(DifferentiableOpticalLayer):
    def __init__(self,cfg):
        config=OpticalConfig(logical_size=192,readout_size=600,simulation_size=cfg.simulation_size,
            physical_size=600,dmd_active_size=576,slm_active_size=600,ccd_active_size=600,
            physical_pixel_pitch_m=cfg.registered_pitch_m,wavelength_m=cfg.wavelength_m,
            focal_length_1_m=cfg.f1_m,focal_length_2_m=cfg.f2_m,padding_factor=cfg.padding,
            phase_control_size=600//cfg.slm_block,phase_initialization="zero",phase_residual_std=.5,
            phase_modulation_depth_rad=float(calibration_tables()[2].max()),phase_command_scale=1.)
        super().__init__(config);self.cfg96=cfg
        self.register_buffer("phase_command_response",torch.from_numpy(calibration_tables()[2]),persistent=False)
        # Build lens coordinates/phases in FP64; the archive's FP32 kernels are
        # NOT silently reused as numerically converged physical ground truth.
        n=self.padded_size;pitch=config.effective_pixel_pitch_m
        axis=(torch.arange(n,dtype=torch.float64)-n//2)*pitch;y,x=torch.meshgrid(axis,axis,indexing="ij")
        for index,focal in ((1,cfg.f1_m),(2,cfg.f2_m)):
            lens=torch.exp(-1j*math.pi*(x*x+y*y)/(cfg.wavelength_m*focal)).to(torch.complex64)
            setattr(self,f"lens_{index}",lens)
    @staticmethod
    def _build_transfer(size,pitch,wavelength,distance):
        f=torch.fft.fftfreq(size,d=pitch,dtype=torch.float64);fy,fx=torch.meshgrid(f,f,indexing="ij")
        k=2*math.pi/wavelength;kz=torch.sqrt((k*k-(2*math.pi*fx).square()-(2*math.pi*fy).square()).clamp_min(0))
        # Remove a spatially uniform carrier before converting to complex64.
        return torch.exp(1j*distance*(kz-k)).to(torch.complex64)
    def command_controls(self):return quantize(self.phase_raw.sigmoid()*255)
    def command_native(self):
        return self.command_controls().repeat_interleave(self.cfg96.slm_block,0).repeat_interleave(self.cfg96.slm_block,1)
    @property
    def applied_phase(self):
        command=self.command_native();indices=command.detach().round().long().clamp(0,255)
        measured=self.phase_command_response[indices]
        # Exact tabulated FORWARD, explicitly approximate STE BACKWARD.
        surrogate=command*(float(self.phase_command_response.max())/255)
        phase=surrogate+(measured-surrogate).detach()
        return F.interpolate(phase[None,None],size=(self.size,self.size),mode="nearest")[0,0]
    def phase_uint8(self):return self.command_native().detach().round().clamp(0,255).to(torch.uint8)


class Policy96(nn.Module):
    architecture_id="four_RAW96_DMD576_SLM600_block_LUT_CCD8_linear18_v1"
    def __init__(self,cfg=None):
        super().__init__();self.cfg=cfg or ModelConfig();self.optical=ArchiveOptics(self.cfg)
        self.actor_head=nn.Linear(8,2);self.critic=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,1))
        self.log_std=nn.Parameter(torch.full((2,),math.log(.02)))
        nn.init.zeros_(self.actor_head.weight);nn.init.zeros_(self.actor_head.bias)
    def ccd_counts(self,raw):
        intensity=self.optical._intensity(pack(raw).float()/255)[:,0]
        return quantize((intensity*self.cfg.adu_per_intensity).clamp(0,255))
    def features(self,counts):
        if counts.shape[-2:]!=(600,600):raise ValueError("registered 600x600 CCD coordinate domain required")
        return counts.reshape(-1,2,300,4,150).sum((2,4)).flatten(1)/self.cfg.feature_reference
    def encode_frames(self,raw):return self.features(self.ccd_counts(raw))
    def forward(self,raw):return self.actor_head(self.encode_frames(raw))
    def geometry(self):
        return dict(frame_size=96,input_times_s=[-.06,-.04,-.02,0.],RAW_FIFO=4,CCD_history=0,
            logical_mosaic=[192,192],DMD_canvas=[600,600],DMD_active=[576,576],DMD_margin=12,
            DMD_video_repeat=[3,3],DMD_native_mirror_mapping_verified=False,
            SLM_canvas=[600,600],SLM_block=[self.cfg.slm_block]*2,SLM_controls=[600//self.cfg.slm_block]*2,
            SLM_expansion="nearest_exact_blocks",phase_LUT_applied_once=True,extra_phase_scale=1.,
            phase_reference_depth_rad=float(self.optical.phase_command_response.max()),
            current_registered_CCD=[600,600],CCD_readout_values=8,actor_parameters=18,actor_macs=16,
            propagations_per_action=1,auxiliary_heads_at_inference=False,
            optical_model="historical_registered_grid_surrogate_FP64_kernel",calibrated_current_hardware=False,
            raw_sensor_ROI_different_from_registered_canvas=True)


def save(path,model,metadata):
    path=Path(path);temporary=path.with_name(path.name+".tmp")
    torch.save(dict(architecture=model.architecture_id,config=asdict(model.cfg),
        phase_raw=model.optical.phase_raw.detach().cpu(),actor_head=model.actor_head.state_dict(),
        metadata=metadata,source_sha256=rt.sources(),hardware_enabled=False,release_approved=False),temporary)
    temporary.replace(path)


def load(path,device="cpu",simulation_size=None):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload["architecture"]!=Policy96.architecture_id:raise ValueError("wrong architecture")
    for filename,expected in payload["source_sha256"].items():
        if rt.digest(filename)!=expected:raise RuntimeError(f"bound source changed: {filename}")
    cfg=dict(payload["config"])
    if simulation_size is not None:cfg["simulation_size"]=simulation_size
    model=Policy96(ModelConfig(**cfg)).to(device)
    with torch.no_grad():model.optical.phase_raw.copy_(payload["phase_raw"].to(device))
    model.actor_head.load_state_dict(payload["actor_head"])
    return model,payload
