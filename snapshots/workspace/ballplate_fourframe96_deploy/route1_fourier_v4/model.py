"""Higher-order numerical pixel integration of the SAME ideal 4f architecture.

Richardson extrapolation is applied before CCD clipping/quantization. The two
computational grids are NOT two real optical passes or an electronic layer.
The actual optics are still uncalibrated; original ASM checkpoints are kept.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from route1_fourier_v3.model import Config as BaseConfig, Policy as BasePolicy, FourierOptics, source_manifest as base_sources
from route1_fourier_v3.model import migrate_prior, FIFO96, pack, resize_binary, quantize, calibration_tables, rt


@dataclass(frozen=True)
class Config(BaseConfig):
    refinement_samples: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not self.simulation_size//600 < self.refinement_samples <= 8:
            raise ValueError('strictly finer, bounded integration grid required')


def extrapolate(low, high, low_samples, high_samples):
    if low.shape != high.shape or not 0 < low_samples < high_samples:
        raise ValueError('same physical pixels and increasing quadrature required')
    lo2, hi2 = low_samples**2, high_samples**2
    return (hi2*high-lo2*low)/(hi2-lo2)


class IntegratedOptics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg96 = cfg
        self.samples = cfg.simulation_size//600
        self.low = FourierOptics(cfg)
        # BaseConfig does not impose the parent's low/high-pair constraint.
        fine_cfg = BaseConfig(**{key: value for key, value in asdict(cfg).items() if key != 'refinement_samples'})
        fine_cfg = replace(fine_cfg, simulation_size=600*cfg.refinement_samples)
        self.high = FourierOptics(fine_cfg)
        self.phase_raw = self.low.phase_raw
        self.high.phase_raw = self.phase_raw
        self.phase_scale = 1.

    @property
    def phase_command_response(self):
        return self.low.phase_command_response

    def command_native(self):
        return self.low.command_native()

    def phase_uint8(self):
        return self.low.phase_uint8()

    def _intensity(self, mosaic):
        self.low.phase_scale = self.high.phase_scale = self.phase_scale
        low = self.low._intensity(mosaic)
        high = self.high._intensity(mosaic)
        return extrapolate(low, high, self.samples, self.cfg96.refinement_samples).clamp_min(0)


class Policy(BasePolicy):
    architecture_id = 'route1_FOURIER4F_RAW96x4_FULLSLM_RAWGRAY_CCD8_LINEAR18_v4'

    def __init__(self, cfg=None):
        cfg = cfg or Config()
        super().__init__(cfg)
        self.optical = IntegratedOptics(cfg)

    def geometry(self):
        return dict(super().geometry(), solver='paraxial_4f_CZT_Richardson_pixel_integration',
            quadrature_pair=[self.optical.samples, self.cfg.refinement_samples],
            extrapolation_before_CCD_quantization=True, real_optical_passes_per_action=1,
            computational_grids_are_NOT_electronic_layers=True)


def source_manifest():
    return dict(base_sources(), **{str(ROOT/'model.py'): rt.digest(ROOT/'model.py')})


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
        raise ValueError('wrong integrated Fourier architecture')
    for name, expected in payload['source_sha256'].items():
        if rt.digest(name) != expected:
            raise RuntimeError('bound source changed: '+name)
    cfg = Config(**payload['config'])
    if samples is not None:
        fine = {2: 3, 3: 4, 4: 6, 6: 8}
        if samples not in fine:
            raise ValueError('explicit integration ladder required')
        cfg = replace(cfg, simulation_size=600*samples, refinement_samples=fine[samples])
    model = Policy(cfg).to(device)
    with torch.no_grad():
        model.optical.phase_raw.copy_(payload['phase_raw'].to(device))
        model.actor_head.load_state_dict(payload['actor_head'])
    return model, payload
