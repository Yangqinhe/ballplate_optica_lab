"""Strictly one/two RAW inputs -> ONE SLM field -> CURRENT CCD8 -> Linear(8,2)."""
import fewframe_runtime as rt
from collections import deque
import math
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Independent,Normal
from optical.encoder import OpticalFrameEncoder
from photonic_model import RegionDetector
from fourframe_model import detector_regions,default_config as four_config
import canonical_region
from dataclasses import replace


def default_config():
    return replace(four_config(),configuration_id="minimal_frame_shared_slm_uncalibrated_v1",
                   grid_size=6144,samples_per_mirror=4,phase_controls=128)


def validate_mode(frames,gap):
    if frames not in (1,2) or (frames==1 and gap!=0) or (frames==2 and gap not in (1,3)):
        raise ValueError("supported: single/current; pair with 20ms or 60ms gap")


def select_from_four(raw,frames,gap):
    validate_mode(frames,gap)
    if raw.ndim!=5 or tuple(raw.shape[1:])!=(4,1,128,128):raise ValueError("four-time TRAINING fixture expected")
    return raw[:,[-1] if frames==1 else [-1-gap,-1]]


def pack_frames(raw,frames):
    if raw.ndim!=5 or tuple(raw.shape[1:])!=(frames,1,128,128):
        raise ValueError("actor must receive EXACTLY its declared one/two RAW frames")
    if raw.dtype!=torch.uint8 or bool(((raw!=0)&(raw!=255)).any()):raise ValueError("binary uint8 RAW images required")
    return raw[:,0] if frames==1 else torch.cat((raw[:,0],raw[:,1]),-1)


class RawSelector:
    def __init__(self,frames=2,gap=1):
        validate_mode(frames,gap);self.frames,self.gap=frames,gap
        self.history=deque(maxlen=1 if frames==1 else gap+1)
    def reset(self):self.history.clear()
    def push(self,image):
        image=np.asarray(image)
        if image.dtype!=np.uint8 or image.shape!=(1,128,128) or not np.all((image==0)|(image==255)):
            raise ValueError("binary [1,128,128] raw input required")
        if not self.history:self.history.extend(image.copy() for _ in range(self.history.maxlen-1))
        self.history.append(image.copy())
        return np.stack([self.history[-1]] if self.frames==1 else [self.history[0],self.history[-1]])


class FewFrameEncoder(OpticalFrameEncoder):
    def __init__(self,cfg,frames=2,reference=100000.):
        super().__init__(cfg);self.frames=frames
        self.regions=RegionDetector(self.detector,detector_regions(cfg.ccd_roi_pixels,"tiles8"),reference)
        self.regions.forward=types.MethodType(canonical_region.forward,self.regions)
    def input_field(self,image):
        shape=(1,128,128*self.frames)
        if image.ndim!=4 or tuple(image.shape[1:])!=shape:raise ValueError("wrong physical DMD composite")
        if image.dtype!=torch.uint8 or bool(((image!=0)&(image!=255)).any()):raise ValueError("nonbinary DMD")
        factor=self.cfg.superpixel*self.cfg.samples_per_mirror
        if 128*self.cfg.superpixel>1080 or 128*self.frames*self.cfg.superpixel>1920:
            raise ValueError("DMD composite exceeds native panel")
        amplitude=(image[:,0].float()/255).repeat_interleave(factor,-2).repeat_interleave(factor,-1)
        amplitude=amplitude+(1-amplitude)*self.cfg.input_off_amplitude
        dh,dw=self.cfg.grid_size-amplitude.shape[-2],self.cfg.grid_size-amplitude.shape[-1]
        if min(dh,dw)<0 or dh%2 or dw%2:raise ValueError("invalid propagation grid padding")
        return F.pad(amplitude,(dw//2,dw//2,dh//2,dh//2)).to(torch.complex64)
    def forward(self,image,seeds=None):
        field=self.field(image)
        if self.cfg.noise_enabled:
            from dashboard.acquisition import native_counts
            return self.regions.full_reference(native_counts(self.detector,field,seeds))
        return self.regions(field,seeds)


class FewFramePolicy(nn.Module):
    architecture_id="one_or_two_RAW_frames_single_SLM_CURRENT_CCD8_linear18_v1"
    def __init__(self,cfg=None,frames=2,gap=1,reference=100000.):
        super().__init__();validate_mode(frames,gap)
        self.options=dict(frames=frames,gap=gap,reference=reference)
        self.encoder=FewFrameEncoder(cfg or default_config(),frames,reference)
        self.actor_head=nn.Linear(8,2);self.critic=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,1))
        self.log_std=nn.Parameter(torch.full((2,),math.log(.02)))
        with torch.no_grad():self.actor_head.weight.zero_();self.actor_head.bias.zero_()
        self.electric_history=False;self.feature_dim=8
    def encode_frames(self,raw):return self.encoder(pack_frames(raw,self.options["frames"]))
    def actor_mean(self,q):
        if q.ndim!=2 or q.shape[-1]!=8:raise ValueError("only CURRENT CCD8 enters the actor")
        return self.actor_head(q)
    def forward(self,raw):return self.actor_mean(self.encode_frames(raw))
    def value(self,q):return self.critic(q.detach()).squeeze(-1)
    def distribution(self,q):
        mean=self.actor_mean(q)
        return Independent(Normal(mean,self.log_std.exp().clamp(.005,.04).expand_as(mean)),1)
    def actor_parameters(self):return sum(p.numel() for p in self.actor_head.parameters())
    def geometry(self):
        n,gap=self.options["frames"],self.options["gap"];cfg=self.encoder.cfg
        return dict(optical_raw_frames=n,input_ages_s=[0.] if n==1 else [gap*.02,0.],
            input_RAW_FIFO_capacity=1 if n==1 else gap+1,
            logical_DMD_shape=[128,128*n],native_DMD_patch_shape=[128*cfg.superpixel,128*n*cfg.superpixel],
            actor_parameters=18,actor_macs=16,CCD_history_length=0,CCD_readout_values=8,
            native_CCD_integration_pixels=cfg.ccd_roi_pixels**2,optical_acquisitions_per_action=1,
            laser_intensity_and_exposure_not_increased=True,raw_state_or_centroid_at_inference=False)
