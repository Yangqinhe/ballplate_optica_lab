"""Independent 150/100 mm research actor. No GUI, SDK, or hardware imports.

The output mask is RAW GRAY, not a target-phase command: do not apply a LUT
again. The historical raw-gray response is an explicitly unverified prior.
"""
from dataclasses import dataclass, asdict, replace
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
import runtime96 as rt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model96 import ModelConfig as LegacyConfig, ArchiveOptics, FIFO96, pack, resize_binary, quantize, calibration_tables


@dataclass(frozen=True)
class Config(LegacyConfig):
    f2_m: float = .10
    slm_block: int = 4
    slm_to_model_flip_y: bool = True
    phase_model: str = 'historical_raw_gray_response_UNVERIFIED'
    gain_jitter: float = .08
    phase_depth_jitter: float = .04
    read_noise_ADU: float = .15
    detector_shift_pixels: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.f1_m != .15 or self.f2_m != .10:
            raise ValueError('this branch is explicitly 150/100 mm')
        if self.phase_model != 'historical_raw_gray_response_UNVERIFIED':
            raise ValueError('raw-gray phase prior must be explicit, never silently use a LUT')
        if not (0 <= self.gain_jitter <= .3 and 0 <= self.phase_depth_jitter <= .2
                and 0 <= self.read_noise_ADU <= 2 and self.detector_shift_pixels in (0, 1, 2)):
            raise ValueError('bounded research perturbations required')


def source_manifest():
    paths = [ROOT/'model.py', rt.PHASE/'slm_phase_response_rad.npy']
    return {**rt.sources(), **{str(p): rt.digest(p) for p in paths}}


class RawGrayOptics(ArchiveOptics):
    def __init__(self, cfg):
        super().__init__(cfg)
        raw_response = np.asarray(np.load(rt.PHASE/'slm_phase_response_rad.npy', allow_pickle=False), np.float32)
        if raw_response.shape != (256,) or not np.isfinite(raw_response).all():
            raise ValueError('historical raw-gray response invalid')
        # Parent buffer normally stores response[LUT[command]]. Here the actor
        # itself emits final raw grays, so use response[raw_gray] exactly once.
        self.phase_command_response.copy_(torch.from_numpy(raw_response))
        self.phase_scale = 1.

    @property
    def applied_phase(self):
        command = self.command_native()
        index = command.detach().round().long().clamp(0, 255)
        measured = self.phase_command_response[index]
        surrogate = command * (float(self.phase_command_response.max()) / 255.)
        phase = (surrogate + (measured - surrogate).detach()) * self.phase_scale
        if self.cfg96.slm_to_model_flip_y:
            phase = phase.flip(0)
        return F.interpolate(phase[None, None], size=(self.size, self.size), mode='nearest')[0, 0]


class Policy(nn.Module):
    architecture_id = 'route1_RAW96x4_150_100_RAWGRAY_CCD8_LINEAR18_v1'

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.optical = RawGrayOptics(self.cfg)
        self.actor_head = nn.Linear(8, 2)
        self.critic = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_std = nn.Parameter(torch.full((2,), math.log(.02)))
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)
        self.randomize = True

    def ccd_counts(self, raw):
        stochastic = self.training and self.randomize
        self.optical.phase_scale = (1 + (2*torch.rand(()).item()-1)*self.cfg.phase_depth_jitter) if stochastic else 1.
        signal = self.optical._intensity(pack(raw).float()/255)[:, 0] * self.cfg.adu_per_intensity
        if stochastic:
            signal = signal * (1 + (2*torch.rand((), device=signal.device)-1)*self.cfg.gain_jitter)
            margin = self.cfg.detector_shift_pixels
            if margin:
                dy, dx = torch.randint(-margin, margin+1, (2,)).tolist()
                signal = F.pad(signal, (margin, margin, margin, margin))[
                    ..., margin+dy:margin+dy+600, margin+dx:margin+dx+600]
            if self.cfg.read_noise_ADU:
                signal = signal + torch.randn_like(signal)*self.cfg.read_noise_ADU
        return quantize(signal.clamp(0, 255))

    def features(self, counts):
        if counts.shape[-2:] != (600, 600):
            raise ValueError('CCD8 requires registered 600x600 counts, NOT a raw camera ROI')
        return counts.reshape(-1, 2, 300, 4, 150).sum((2, 4)).flatten(1)/self.cfg.feature_reference

    def encode_frames(self, raw):
        return self.features(self.ccd_counts(raw))

    def forward(self, raw):
        return self.actor_head(self.encode_frames(raw))

    def geometry(self):
        return dict(RAW_FIFO=4, CCD_history=0, input_frames=[4, 1, 96, 96],
                    mosaic=[192, 192], DMD_canvas=[600, 600], DMD_active=[576, 576],
                    SLM_raw_gray_patch=[600, 600], SLM_block=self.cfg.slm_block,
                    CCD_registered=[600, 600], CCD_readouts=8, actor_parameters=18,
                    f1_f2_mm=[150, 100], slm_to_model_flip_y=self.cfg.slm_to_model_flip_y,
                    phase_prior=self.cfg.phase_model, apply_LUT_at_deployment=False,
                    hardware_horizontal_flip_count=1, calibrated_current_hardware=False,
                    common_8um_grid_is_a_surrogate=True, release_approved=False)


@torch.no_grad()
def initialize(model, mode='migrate', checkpoint=None):
    if mode == 'flat':
        model.optical.phase_raw.fill_(-30.)
        return dict(mode=mode, old_propagation_buffers_copied=False)
    if mode == 'migrate':
        if checkpoint is None:
            raise ValueError('explicit old checkpoint required')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['architecture'] != 'four_RAW96_DMD576_SLM600_block_LUT_CCD8_linear18_v1':
            raise ValueError('not the known legacy actor')
        for path, digest in payload['source_sha256'].items():
            if rt.digest(path) != digest:
                raise RuntimeError('legacy bound source changed: '+path)
        if payload['phase_raw'].shape != model.optical.phase_raw.shape:
            raise ValueError('SLM block mismatch')
        old_command = (payload['phase_raw'].sigmoid()*255).round().clamp(0, 255).long()
        lut = torch.from_numpy(calibration_tables()[0].astype(np.int64))
        raw_gray = lut[old_command]
        model.optical.phase_raw.copy_(torch.logit((raw_gray.float()/255).clamp(1e-6, 1-1e-6)).to(model.optical.phase_raw))
        return dict(mode=mode, source_checkpoint=str(Path(checkpoint).resolve()),
                    source_checkpoint_sha256=rt.digest(checkpoint),
                    command_to_raw_gray='historical_LUT[legacy_command] ONCE for migration',
                    old_head_copied=False, old_propagation_buffers_copied=False,
                    new_focal_lengths_mm=[150, 100], current_phase_response_verified=False)
    raise ValueError('unsupported initialization')


def save(path, model, metadata):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    torch.save(dict(architecture=model.architecture_id, config=asdict(model.cfg),
                    phase_raw=model.optical.phase_raw.detach().cpu(),
                    actor_head={k: v.detach().cpu() for k, v in model.actor_head.state_dict().items()},
                    metadata=metadata, source_sha256=source_manifest(),
                    hardware_enabled=False, release_approved=False), temp)
    temp.replace(path)


def load(path, device='cpu', samples=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['architecture'] != Policy.architecture_id:
        raise ValueError('wrong actor architecture')
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
