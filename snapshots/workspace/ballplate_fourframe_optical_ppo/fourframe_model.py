"""Spatially multiplex four RAW times, not four independently filtered channels.

All four tiles share ONE Fourier-plane SLM. The entire coherent field propagates
once; no per-tile encoder calls, separate kernels, or cached CCD features exist.
"""
import fourframe_runtime as rt
from collections import deque
import math
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Independent, Normal
from optical.config import OpticalConfig
from optical.encoder import OpticalFrameEncoder
from photonic_model import RegionDetector
import canonical_region


def default_config():
    return OpticalConfig(configuration_id="fourframe_2x2_shared_slm_uncalibrated_v1",
                         propagation="asm", superpixel=2, grid_size=4608,
                         samples_per_mirror=3, phase_controls=64)


def pack_four(frames):
    """[B,4,1,128,128], oldest first -> [B,1,256,256], row-major."""
    if frames.ndim != 5 or tuple(frames.shape[1:]) != (4, 1, 128, 128):
        raise ValueError("four ordered RAW 128x128 input images required")
    if frames.dtype != torch.uint8 or bool(((frames != 0) & (frames != 255)).any()):
        raise ValueError("binary uint8 0/255 images required")
    return torch.cat((torch.cat((frames[:, 0], frames[:, 1]), -1),
                      torch.cat((frames[:, 2], frames[:, 3]), -1)), -2)


class RawFrameFIFO:
    """Input-side memory only. Startup repeats the first observation, never future data."""
    def __init__(self):
        self._frames = deque(maxlen=4)

    def reset(self):
        self._frames.clear()

    def push(self, image):
        image = np.asarray(image)
        if image.shape != (1, 128, 128) or image.dtype != np.uint8:
            raise ValueError("raw camera input must be uint8 [1,128,128]")
        if not np.all((image == 0) | (image == 255)):
            raise ValueError("raw camera input must be binary")
        if not self._frames:
            self._frames.extend(image.copy() for _ in range(3))
        self._frames.append(image.copy())
        return np.stack(tuple(self._frames))


def dmd_bitmap(frames, superpixel=2):
    """A display/SDK-independent 1920x1080 binary canvas, NOT a hardware write."""
    if superpixel < 2:
        raise ValueError("at least 2x2 native mirrors per logical bit required")
    tiles = pack_four(frames)
    patch = tiles.repeat_interleave(superpixel, -2).repeat_interleave(superpixel, -1)
    h, w = patch.shape[-2:]
    if h > 1080 or w > 1920:
        raise ValueError("four-frame patch exceeds physical DMD canvas")
    top, left = (1080 - h) // 2, (1920 - w) // 2
    return F.pad(patch, (left, 1920-w-left, top, 1080-h-top))


def detector_regions(pixels=1280, readout="tiles8"):
    grids = {"tiles8": (2, 4), "diagnostic16": (4, 4), "quad4": (2, 2)}
    if readout not in grids:
        raise ValueError("unknown readout")
    rows, cols = grids[readout]
    # Full CCD coverage first. Do NOT assume light has already been routed into
    # tiny windows. These sums still cost pixels; they are not camera-side binning.
    return [[r*pixels//rows, (r+1)*pixels//rows,
             c*pixels//cols, (c+1)*pixels//cols]
            for r in range(rows) for c in range(cols)]


def geometry(cfg):
    native = 256 * cfg.superpixel
    extent = native * cfg.dmd_pitch_m * cfg.magnification / cfg.ccd_pitch_m
    return dict(raw_input_shape=[4, 1, 128, 128], logical_dmd_shape=[256, 256],
                native_patch_shape=[native, native], dmd_canvas_shape=[1080, 1920],
                ideal_geometric_ccd_extent_pixels=extent,
                geometric_margin_each_side_pixels=(cfg.ccd_roi_pixels-extent)/2,
                geometric_image_fits_assumed_ccd=extent <= cfg.ccd_roi_pixels,
                does_not_bound_diffracted_or_aberrated_field=True,
                dt_s=.02, input_ages_s=[.06, .04, .02, 0.],
                input_fifo_uint8_bytes=4*128*128,
                shared_slm_planes=1, optical_acquisitions_per_action=1,
                electronic_ccd_history_length=0, calibrated=False)


class FourFrameEncoder(OpticalFrameEncoder):
    def __init__(self, cfg, readout="tiles8", reference=100000.):
        super().__init__(cfg)
        self.regions = RegionDetector(self.detector, detector_regions(cfg.ccd_roi_pixels, readout), reference)
        self.regions.forward = types.MethodType(canonical_region.forward, self.regions)

    def input_field(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 256, 256):
            raise ValueError("one 256x256 composite, not a per-tile propagation, is required")
        if images.dtype != torch.uint8 or bool(((images != 0) & (images != 255)).any()):
            raise ValueError("DMD composite must be binary uint8")
        factor = self.cfg.superpixel * self.cfg.samples_per_mirror
        side = 256 * factor
        margin = self.cfg.grid_size - side
        if 256*self.cfg.superpixel > 1080 or margin < 0 or margin % 2:
            raise ValueError("invalid DMD/propagation grid geometry")
        amplitude = images[:, 0].float()/255
        amplitude = amplitude.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
        amplitude = amplitude + (1-amplitude)*self.cfg.input_off_amplitude
        return F.pad(amplitude, (margin//2,)*4).to(torch.complex64)

    def forward(self, image, seeds=None):
        field = self.field(image)
        if self.cfg.noise_enabled:
            from dashboard.acquisition import native_counts
            return self.regions.full_reference(native_counts(self.detector, field, seeds))
        return self.regions(field, seeds)


class FourFramePolicy(nn.Module):
    architecture_id = "four_raw_time_tiles_one_slm_current_flux_linear_v1"

    def __init__(self, cfg=None, readout="tiles8", reference=100000.):
        super().__init__()
        self.options = dict(readout=readout, reference=reference)
        self.encoder = FourFrameEncoder(cfg or default_config(), **self.options)
        self.feature_dim = len(self.encoder.regions.regions)
        self.actor_head = nn.Linear(self.feature_dim, 2)
        self.critic = nn.Sequential(nn.Linear(self.feature_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_std = nn.Parameter(torch.full((2,), math.log(.02)))
        with torch.no_grad():
            self.actor_head.weight.zero_()
            self.actor_head.bias.zero_()
        self.electric_history = False

    def encode_frames(self, raw_frames, seeds=None):
        return self.encoder(pack_four(raw_frames), seeds)

    def actor_mean(self, current_flux):
        if current_flux.ndim != 2 or current_flux.shape[-1] != self.feature_dim:
            raise ValueError("only the CURRENT CCD vector may enter the actor")
        return self.actor_head(current_flux)

    def value(self, current_flux):
        return self.critic(current_flux.detach()).squeeze(-1)

    def distribution(self, current_flux):
        mean = self.actor_mean(current_flux)
        return Independent(Normal(mean, self.log_std.exp().clamp(.005, .04).expand_as(mean)), 1)

    def forward(self, raw_frames, seeds=None):
        return self.actor_mean(self.encode_frames(raw_frames, seeds))

    def actor_parameters(self):
        return sum(p.numel() for p in self.actor_head.parameters())


def save_candidate(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(architecture=model.architecture_id, options=model.options,
                    optical_config=model.encoder.cfg.to_dict(), model=model.state_dict(),
                    source_sha256=rt.sources(), physical_manifest=rt.reused.physical_manifest(),
                    metadata=metadata, calibrated=False, hardware_enabled=False), temporary)
    temporary.replace(path)
