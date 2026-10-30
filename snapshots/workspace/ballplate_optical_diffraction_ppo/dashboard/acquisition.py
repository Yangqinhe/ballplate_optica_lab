"""Passive sensor observer; policy always uses the ORIGINAL encoder output.

The training detector only exposes 64 features. This observer reconstructs its
native counts from the same optical field and random seed, then checks that the
displayed counts reproduce the exact policy features. It does not recompute
diffraction, change the model, or edit versioned optical source files.
"""
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from optical.devices import quantize_ste
from dashboard.interfaces import CCDFrame

# Fail closed if the original detector changes without updating this observer.
DETECTOR_SOURCE_SHA256 = "5dd446cb1edb8b6c60fb5e6384c3ad22da8198525f745fc5f4947db05cf5bf17"


def native_counts(detector, field, noise_seeds):
    """Same operation order as the pinned detector, without its feature pooling."""
    cfg = detector.cfg
    components = torch.stack((field.real, field.imag), dim=1)
    grid = detector.sample_grid.expand(len(field), -1, -1, -1)
    count = cfg.ccd_area_samples
    intensity = None
    for row in range(count):
        for col in range(count):
            offset = field.real.new_tensor([(col+.5)/count-.5, (row+.5)/count-.5])
            offset = -offset * cfg.ccd_pitch_m / detector.plane_pitch * (2/(cfg.grid_size-1))
            sampled = F.grid_sample(components, grid + offset, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
            value = sampled.square().sum(dim=1) / count**2
            intensity = value if intensity is None else intensity + value
    electrons = intensity * cfg.electrons_per_intensity * (
        cfg.ccd_exposure_s / cfg.detector_reference_exposure_s)
    if cfg.noise_enabled:
        if noise_seeds is None or len(noise_seeds) != len(field):
            raise ValueError("one fixed noise seed per frame is required")
        noisy = []
        for index, seed in enumerate(noise_seeds):
            generator = torch.Generator(device=field.device).manual_seed(int(seed))
            mean = electrons[index]
            if cfg.shot_noise_model == "poisson":
                if mean.requires_grad:
                    raise RuntimeError("Poisson is not an exact differentiable sampler")
                sample = torch.poisson(mean.clamp_min(0), generator=generator)
            else:
                sample = mean + mean.clamp_min(1e-6).sqrt() * torch.randn(
                    mean.shape, generator=generator, device=field.device)
            sample = sample + cfg.read_noise_electrons * torch.randn(
                mean.shape, generator=generator, device=field.device)
            noisy.append(sample)
        electrons = torch.stack(noisy)
    electrons = electrons.clamp(0, cfg.full_well_electrons)
    counts = electrons * (cfg.mono8_counts_per_intensity / cfg.electrons_per_intensity)
    counts = counts.clamp(0, 255)
    return counts if detector.diagnostic_continuous else quantize_ste(counts)


def features_from_raw(detector, raw_counts):
    # Training's detector already samples the fixed 180-degree-corrected image.
    # The dashboard shows native sensor orientation, so undo that for pooling.
    counts = raw_counts.flip((-2, -1))
    roi = counts[:, None, detector.crop_start:detector.crop_stop,
                 detector.crop_start:detector.crop_stop]
    spatial = F.adaptive_avg_pool2d(roi, (32, 32))
    spatial = spatial * (detector.cfg.magnification**2 / detector.cfg.mono8_counts_per_intensity)
    return F.adaptive_avg_pool2d(spatial, (8, 8)).flatten(1)


def readonly(array):
    value = np.array(array, copy=True)
    value.setflags(write=False)
    return value


class SimulationAcquisition:
    def __init__(self, model):
        path = Path(__file__).resolve().parents[1]/"optical/devices.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != DETECTOR_SOURCE_SHA256:
            raise RuntimeError("Detector source changed; update and revalidate the passive CCD observer")
        self.model = model
        self._capture = None
        self.handle = model.encoder.detector.register_forward_hook(self.observe)

    @torch.no_grad()
    def observe(self, detector, inputs, output):
        field, seeds = inputs
        if len(field) != 1:
            raise ValueError("desktop acquisition requires a single current frame")
        corrected = native_counts(detector, field, seeds)
        raw = corrected.flip((-2, -1))
        raw_bytes = raw.to(torch.uint8)
        rebuilt = features_from_raw(detector, raw_bytes.float())
        error = float((rebuilt-output).abs().max())
        if not torch.equal(rebuilt, output):
            raise RuntimeError(f"CCD display and policy feature mismatch: {error}")
        self._capture = (raw_bytes[0].cpu().numpy().copy(), error)
        # A forward hook returning None does not modify the policy output.
        return None

    @torch.no_grad()
    def acquire(self, binary_image, frame_id, episode, simulated_time_s, noise_seed):
        self._capture = None
        device = next(self.model.parameters()).device
        started = time.perf_counter()
        image = torch.from_numpy(np.array(binary_image, copy=True)).unsqueeze(0).to(device)
        q = self.model.encoder(image, [noise_seed])
        if self._capture is None:
            raise RuntimeError("original detector did not emit a matched CCD capture")
        raw, error = self._capture
        frame = CCDFrame(frame_id, episode, simulated_time_s, time.time(), noise_seed,
                         readonly(raw), readonly(q[0].cpu().numpy()), error,
                         time.perf_counter()-started)
        self._capture = None
        return q, frame

    def close(self):
        self.handle.remove()
        self._capture = None
