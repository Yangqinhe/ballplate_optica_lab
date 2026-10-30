"""Independent paraxial 4f research solver, not a measured hardware twin.

Exact rectangular DMD-pixel Fourier envelope; midpoint SLM/CCD quadrature.
Full SLM panel is a rectangular aperture, with an UNVERIFIED manual origin.
No camera, GUI or output-window dependencies. Do not apply a LUT at export.
"""
from dataclasses import dataclass, asdict, replace
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from route1_150_100.model import Config as PriorConfig, Policy as PriorPolicy, source_manifest as prior_sources
from model96 import FIFO96, pack, resize_binary, quantize, calibration_tables
import runtime96 as rt
try:
    from .transform import FourierAxis, transform2
except ImportError:
    from transform import FourierAxis, transform2


@dataclass(frozen=True)
class Config(PriorConfig):
    simulation_size: int = 1800
    slm_panel_width: int = 1920
    slm_panel_height: int = 1080
    slm_origin_x: int = 546
    slm_origin_y: int = 493

    def __post_init__(self):
        super().__post_init__()
        if not (self.slm_panel_width == 1920 and self.slm_panel_height == 1080
                and 300 <= self.slm_origin_x <= self.slm_panel_width-300
                and 300 <= self.slm_origin_y <= self.slm_panel_height-300):
            raise ValueError('explicit full SLM panel and contained 600px patch required')
        if self.simulation_size//600 > 8:
            raise ValueError('bounded quadrature required')


class FourierOptics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg96 = cfg
        self.samples = cfg.simulation_size//600
        self.phase_raw = nn.Parameter(torch.zeros(600//cfg.slm_block, 600//cfg.slm_block))
        curve = np.asarray(np.load(rt.PHASE/'slm_phase_response_rad.npy', allow_pickle=False), np.float32)
        self.register_buffer('phase_command_response', torch.from_numpy(curve), persistent=False)
        self.phase_scale = 1.
        s, pitch = self.samples, cfg.registered_pitch_m
        self.sw, self.sh = cfg.slm_panel_width*s, cfg.slm_panel_height*s
        ds = pitch/s
        # The model's Y reflection is applied to the WHOLE panel, including its aperture origin.
        self.ox = cfg.slm_origin_x
        self.oy = cfg.slm_panel_height-cfg.slm_origin_y if cfg.slm_to_model_flip_y else cfg.slm_origin_y
        x0s, y0s = (.5/s-self.ox)*pitch, (.5/s-self.oy)*pitch
        dmd_pitch = cfg.dmd_repeat*pitch
        x0d = (.5-192/2)*dmd_pitch
        wave_f1, wave_f2 = cfg.wavelength_m*cfg.f1_m, cfg.wavelength_m*cfg.f2_m
        self.first_x = FourierAxis(192, self.sw, x0d, dmd_pitch, x0s/wave_f1, ds/wave_f1)
        self.first_y = FourierAxis(192, self.sh, x0d, dmd_pitch, y0s/wave_f1, ds/wave_f1)
        xs = x0s+torch.arange(self.sw, dtype=torch.float64)*ds
        ys = y0s+torch.arange(self.sh, dtype=torch.float64)*ds
        self.register_buffer('sinc_x', torch.sinc(dmd_pitch*xs/wave_f1).float())
        self.register_buffer('sinc_y', torch.sinc(dmd_pitch*ys/wave_f1).float())
        nout, dc = 600*s, pitch/s
        x0c = (.5/s-300)*pitch
        self.second_x = FourierAxis(self.sw, nout, x0s, ds, x0c/wave_f2, dc/wave_f2)
        self.second_y = FourierAxis(self.sh, nout, y0s, ds, x0c/wave_f2, dc/wave_f2)
        self.scale_first = dmd_pitch**2/wave_f1
        self.scale_second = ds**2/wave_f2

    def command_controls(self):
        return quantize(self.phase_raw.sigmoid()*255.)

    def command_native(self):
        return self.command_controls().repeat_interleave(self.cfg96.slm_block, 0).repeat_interleave(self.cfg96.slm_block, 1)

    def phase_uint8(self):
        return self.command_native().detach().round().clamp(0, 255).to(torch.uint8)

    def transmittance(self):
        command = self.command_native()
        raw = command.detach().round().long().clamp(0, 255)
        measured = self.phase_command_response[raw]
        surrogate = command*(float(self.phase_command_response.max())/255.)
        phase = (surrogate+(measured-surrogate).detach())*self.phase_scale
        if self.cfg96.slm_to_model_flip_y:
            phase = phase.flip(0)
        top, left = self.oy-300, self.ox-300
        outside = float(self.phase_command_response[0])*self.phase_scale
        full = F.pad(phase, (left, self.cfg96.slm_panel_width-600-left,
                            top, self.cfg96.slm_panel_height-600-top), value=outside)
        return torch.exp(1j*full.repeat_interleave(self.samples, 0).repeat_interleave(self.samples, 1))

    def _intensity(self, mosaic):
        if mosaic.shape[-3:] != (1, 192, 192):
            raise ValueError('one 192x192 packed DMD amplitude required')
        amplitude = mosaic[:, 0].float().sqrt()
        field = transform2(amplitude, self.first_y, self.first_x)*self.scale_first
        field = field*self.sinc_y[:, None]*self.sinc_x[None, :]
        field = field*self.transmittance()
        output = transform2(field, self.second_y, self.second_x)*self.scale_second
        intensity = output.real.square()+output.imag.square()
        return F.avg_pool2d(intensity[:, None], self.samples)


class Policy(PriorPolicy):
    architecture_id = 'route1_FOURIER4F_RAW96x4_FULLSLM_RAWGRAY_CCD8_LINEAR18_v3'

    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.cfg = cfg or Config()
        self.optical = FourierOptics(self.cfg)
        self.actor_head = nn.Linear(8, 2)
        self.critic = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_std = nn.Parameter(torch.full((2,), math.log(.02)))
        self.randomize = True
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)

    def geometry(self):
        return dict(super().geometry(), solver='paraxial_4f_CZT_midpoint',
            SLM_full_aperture=[self.cfg.slm_panel_height, self.cfg.slm_panel_width],
            canonical_manual_origin_UNVERIFIED=[self.cfg.slm_origin_x, self.cfg.slm_origin_y],
            real_lens_apertures_and_defocus_not_identified=True,
            exact_rectangular_DMD_pixel_envelope=True, midpoint_samples=self.optical.samples,
            not_numerically_identical_to_legacy_ASM=True)


def source_manifest():
    return dict(prior_sources(), **{str(p): rt.digest(p) for p in (ROOT/'model.py', ROOT/'transform.py')})


def save(path, model, metadata):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    torch.save(dict(architecture=model.architecture_id, config=asdict(model.cfg),
        phase_raw=model.optical.phase_raw.detach().cpu(),
        actor_head={k: v.detach().cpu() for k, v in model.actor_head.state_dict().items()},
        metadata=metadata, source_sha256=source_manifest(), hardware_enabled=False, release_approved=False), temp)
    temp.replace(path)


def load(path, device='cpu', samples=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['architecture'] != Policy.architecture_id:
        raise ValueError('wrong Fourier research architecture')
    for name, expected in payload['source_sha256'].items():
        if rt.digest(name) != expected:
            raise RuntimeError('bound source changed: '+name)
    cfg = Config(**payload['config'])
    if samples is not None:
        cfg = replace(cfg, simulation_size=600*samples)
    model = Policy(cfg).to(device)
    with torch.no_grad():
        model.optical.phase_raw.copy_(payload['phase_raw'].to(device))
        model.actor_head.load_state_dict(payload['actor_head'])
    return model, payload


@torch.no_grad()
def migrate_prior(model, path, copy_head=False):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['architecture'] != PriorPolicy.architecture_id:
        raise ValueError('only explicit route-1 raw-gray prior accepted')
    for name, expected in payload['source_sha256'].items():
        if rt.digest(name) != expected:
            raise RuntimeError('bound prior source changed: '+name)
    if payload['phase_raw'].shape != model.optical.phase_raw.shape:
        raise ValueError('cannot implicitly resize phase controls')
    model.optical.phase_raw.copy_(payload['phase_raw'].to(model.optical.phase_raw))
    if copy_head:
        model.actor_head.load_state_dict(payload['actor_head'])
    return dict(source_checkpoint=str(Path(path).resolve()), source_checkpoint_sha256=rt.digest(path),
        phase_only=not copy_head, old_head_copied=copy_head, old_propagation_buffers_copied=False,
        new_solver='paraxial_4f_CZT', not_a_resume_of_identical_physics=True)
