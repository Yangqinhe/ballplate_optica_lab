"""Fourier-optics reference simulation used by SLM alignment scoring."""

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sdon.config import OpticsConfig


def prepare_simulation_source(
    dmd_image: np.ndarray,
    canvas_size: int,
    active_size: Optional[int] = None,
) -> np.ndarray:
    """Place the DMD field on the simulation canvas without hidden stretching.

    The historical path resized the DMD pattern to the SLM phase size.  That is
    only physically valid while both modulators use the same active canvas.
    With a 512-pixel DMD and an 800-pixel SLM, the DMD must instead remain 512
    pixels wide and be centred inside the larger propagation canvas.
    """
    values = np.asarray(dmd_image)
    canvas = int(canvas_size)
    if values.ndim != 2 or canvas < 1:
        raise ValueError("DMD image must be 2-D and canvas_size must be positive")
    if active_size is None:
        return cv2.resize(values, (canvas, canvas), interpolation=cv2.INTER_NEAREST)
    active = int(active_size)
    if active < 1 or active > canvas:
        raise ValueError("DMD active_size must fit inside the simulation canvas")
    resized = cv2.resize(values, (active, active), interpolation=cv2.INTER_NEAREST)
    output = np.zeros((canvas, canvas), dtype=resized.dtype)
    start = (canvas - active) // 2
    output[start : start + active, start : start + active] = resized
    return output


def initialize_torch_runtime():
    """Load Torch before PySpin to avoid a Windows DLL collision.

    The Spinnaker binding bundled on the experiment PC loads a dependency that
    prevents Torch 1.13 from subsequently loading torch/lib/shm.dll. Loading
    Torch first is sufficient, and the imported module remains cached for the
    calibration worker thread.
    """
    import torch

    return torch


def simulate_optical_image(
    dmd_image: np.ndarray,
    phase_uint8: np.ndarray,
    config: OpticsConfig,
    dmd_active_size: Optional[int] = None,
) -> np.ndarray:
    """Simulate the two-lens path used in the source experiment.

    Torch is imported here so configuration, LUT, and metric code can be used
    without installing a GPU stack. CPU and CUDA devices are both supported.
    """
    import torch.nn.functional as functional

    torch = initialize_torch_runtime()
    cuda_interlocked = Path(
        "outputs/mnist_sdon/CUDA_TRAINING_DISABLED.json"
    ).is_file()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cuda_interlocked else "cpu"
    )
    size = int(phase_uint8.shape[0])
    if phase_uint8.shape != (size, size):
        raise ValueError("SLM phase must be square")
    source = prepare_simulation_source(
        dmd_image, size, active_size=dmd_active_size
    ).astype(np.float32)
    field = torch.sqrt(torch.as_tensor(source / 255.0, dtype=torch.float32, device=device))[None]
    field = torch.complex(field, torch.zeros_like(field))

    padding_factor = int(config.padding_factor)
    padded_size = size * padding_factor
    total_padding = padded_size - size
    left = total_padding // 2
    right = total_padding - left
    field = functional.pad(field, (left, right, left, right))

    def propagate(current: "torch.Tensor", distance: float) -> "torch.Tensor":
        height, width = current.shape[-2:]
        wave_number = 2.0 * math.pi / config.wavelength_m
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, d=config.pixel_pitch_m, device=device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, d=config.pixel_pitch_m, device=device))
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radicand = torch.clamp(
            wave_number ** 2 - (2.0 * math.pi * xx) ** 2 - (2.0 * math.pi * yy) ** 2, min=0.0,
        )
        transfer = torch.fft.fftshift(torch.exp(1j * distance * torch.sqrt(radicand)))
        spectrum = torch.fft.fft2(torch.fft.fftshift(current))
        return torch.fft.ifftshift(torch.fft.ifft2(spectrum * transfer))

    def apply_lens(current: "torch.Tensor", focal_length: float) -> "torch.Tensor":
        height, width = current.shape[-2:]
        x = (torch.arange(width, device=device) - width // 2) * config.pixel_pitch_m
        y = (torch.arange(height, device=device) - height // 2) * config.pixel_pitch_m
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        phase = torch.exp(
            -1j * (2.0 * math.pi / config.wavelength_m) * (xx ** 2 + yy ** 2) / (2.0 * focal_length)
        )
        return current * phase

    with torch.no_grad():
        field = propagate(field, config.focal_length_1_m)
        field = apply_lens(field, config.focal_length_1_m)
        field = propagate(field, config.focal_length_1_m)
        phase = torch.as_tensor(
            np.asarray(phase_uint8, dtype=np.float32) / 255.0 * 2.0 * math.pi, device=device,
        )
        phase = functional.pad(phase[None], (left, right, left, right))
        field = field * torch.exp(1j * phase)
        field = propagate(field, config.focal_length_2_m)
        field = apply_lens(field, config.focal_length_2_m)
        field = propagate(field, config.focal_length_2_m)
        field = field[:, left : left + size, left : left + size]
        intensity = torch.real(field * torch.conj(field)) * 255.0
    return intensity.squeeze().cpu().numpy().astype(np.float32)
