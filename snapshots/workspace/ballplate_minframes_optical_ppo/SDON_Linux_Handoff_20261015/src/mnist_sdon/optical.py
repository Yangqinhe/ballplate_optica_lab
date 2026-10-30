"""Differentiable two-lens optical propagation for phase-one SDON training."""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import OpticalConfig


def _center_padding(size: int, factor: int) -> Tuple[int, int, int]:
    padded_size = size * factor
    total = padded_size - size
    left = total // 2
    return padded_size, left, total - left


def _gaussian_kernel_2d(sigma_x: float, sigma_y: float) -> torch.Tensor:
    """Return one normalized anisotropic Gaussian kernel."""
    sigma_x, sigma_y = float(sigma_x), float(sigma_y)
    radius_x = max(0, int(math.ceil(3.0 * sigma_x)))
    radius_y = max(0, int(math.ceil(3.0 * sigma_y)))
    xx = torch.arange(-radius_x, radius_x + 1, dtype=torch.float32)
    yy = torch.arange(-radius_y, radius_y + 1, dtype=torch.float32)
    kernel_x = (
        torch.ones_like(xx)
        if sigma_x <= 0.0
        else torch.exp(-0.5 * (xx / sigma_x).square())
    )
    kernel_y = (
        torch.ones_like(yy)
        if sigma_y <= 0.0
        else torch.exp(-0.5 * (yy / sigma_y).square())
    )
    kernel = kernel_y[:, None] * kernel_x[None, :]
    return kernel / kernel.sum().clamp_min(1.0e-12)


def _blur_2d(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a fixed single-channel kernel while keeping image dimensions."""
    if kernel.numel() == 1:
        return values
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("measured forward-model blur expects [N,1,H,W]")
    pad_x = int(kernel.shape[-1]) // 2
    pad_y = int(kernel.shape[-2]) // 2
    padded = functional.pad(values, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    return functional.conv2d(padded, kernel[None, None].to(values))


class DifferentiableOpticalLayer(nn.Module):
    """Train a coarse SLM phase while preserving the physical aperture."""

    def __init__(self, config: OpticalConfig) -> None:
        super().__init__()
        self.config = config
        size = int(config.simulation_size)
        padded_size, left, right = _center_padding(size, int(config.padding_factor))
        self.size = size
        self.padded_size = padded_size
        self.left_padding = left
        self.right_padding = right
        pitch = float(config.effective_pixel_pitch_m)

        coordinate = (torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0) * pitch
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        initialization = str(config.phase_initialization).lower()
        if initialization == "lens":
            base_phase = (
                -math.pi
                * (xx.square() + yy.square())
                / (float(config.wavelength_m) * float(config.phase_focal_length_m))
            )
        elif initialization == "zero":
            base_phase = torch.zeros((size, size), dtype=torch.float32)
        elif initialization == "random":
            base_phase = torch.rand((size, size), dtype=torch.float32) * (2.0 * math.pi)
        else:
            raise ValueError("phase_initialization must be lens, zero, or random")
        carrier_period = int(config.phase_carrier_period_px)
        if carrier_period:
            carrier_coordinate = torch.arange(size, dtype=torch.float32)
            carrier_line = torch.remainder(
                float(config.phase_carrier_direction)
                * (2.0 * math.pi)
                * carrier_coordinate
                / float(carrier_period),
                2.0 * math.pi,
            )
            carrier = (
                carrier_line[None, :].expand(size, size)
                if config.phase_carrier_axis == "x"
                else carrier_line[:, None].expand(size, size)
            )
            base_phase = base_phase + carrier
        self.register_buffer("base_phase", base_phase)
        control_size = int(config.resolved_phase_control_size)
        self.control_size = control_size
        phase_raw = torch.empty((control_size, control_size), dtype=torch.float32)
        nn.init.normal_(phase_raw, mean=0.0, std=float(config.phase_residual_std))
        self.phase_raw = nn.Parameter(phase_raw)
        self._phase_override: Optional[torch.Tensor] = None
        self.register_buffer("reference_scale", torch.tensor(1.0, dtype=torch.float32))
        gain_map_path = str(config.input_intensity_gain_map_path).strip()
        if gain_map_path:
            gain_map = np.asarray(np.load(Path(gain_map_path)), dtype=np.float32)
            expected_gain_shape = (int(config.logical_size), int(config.logical_size))
            if gain_map.shape != expected_gain_shape:
                raise ValueError(
                    "input intensity gain map shape %s does not match %s"
                    % (gain_map.shape, expected_gain_shape)
                )
            if not np.all(np.isfinite(gain_map)) or float(gain_map.min()) <= 0.0:
                raise ValueError("input intensity gain map must be finite and positive")
            input_gain = torch.from_numpy(gain_map)
        else:
            input_gain = torch.ones(
                (int(config.logical_size), int(config.logical_size)),
                dtype=torch.float32,
            )
        self.register_buffer("input_intensity_gain", input_gain, persistent=False)
        self.register_buffer(
            "readout_blur_kernel",
            _gaussian_kernel_2d(
                float(config.readout_blur_sigma_x_px),
                float(config.readout_blur_sigma_y_px),
            ),
            persistent=False,
        )
        readout_coordinate = torch.arange(
            int(config.readout_size), dtype=torch.float32
        ) - (int(config.readout_size) - 1) / 2.0
        readout_y, readout_x = torch.meshgrid(
            readout_coordinate, readout_coordinate, indexing="ij"
        )
        readout_gain = torch.exp(
            float(config.readout_log_gain_slope_x_per_px) * readout_x
            + float(config.readout_log_gain_slope_y_per_px) * readout_y
        )
        self.register_buffer("readout_intensity_gain", readout_gain, persistent=False)
        self.register_buffer(
            "phase_crosstalk_kernel",
            _gaussian_kernel_2d(
                float(config.phase_crosstalk_sigma_simulation_px),
                float(config.phase_crosstalk_sigma_simulation_px),
            ),
            persistent=False,
        )

        padded_coordinate = (
            torch.arange(padded_size, dtype=torch.float32) - padded_size // 2
        ) * pitch
        pyy, pxx = torch.meshgrid(padded_coordinate, padded_coordinate, indexing="ij")
        wave_number = 2.0 * math.pi / float(config.wavelength_m)
        lens_1 = torch.exp(
            -1j
            * wave_number
            * (pxx.square() + pyy.square())
            / (2.0 * float(config.focal_length_1_m))
        ).to(torch.complex64)
        lens_2 = torch.exp(
            -1j
            * wave_number
            * (pxx.square() + pyy.square())
            / (2.0 * float(config.focal_length_2_m))
        ).to(torch.complex64)
        self.register_buffer("lens_1", lens_1)
        self.register_buffer("lens_2", lens_2)
        self.register_buffer(
            "transfer_1",
            self._build_transfer(
                padded_size, pitch, float(config.wavelength_m), float(config.focal_length_1_m)
            ),
        )
        self.register_buffer(
            "transfer_2",
            self._build_transfer(
                padded_size, pitch, float(config.wavelength_m), float(config.focal_length_2_m)
            ),
        )

    @staticmethod
    def _build_transfer(
        size: int, pitch: float, wavelength: float, distance: float
    ) -> torch.Tensor:
        frequency = torch.fft.fftshift(torch.fft.fftfreq(size, d=pitch))
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        wave_number = 2.0 * math.pi / wavelength
        radicand = torch.clamp(
            wave_number ** 2 - (2.0 * math.pi * fx).square() - (2.0 * math.pi * fy).square(),
            min=0.0,
        )
        transfer = torch.fft.fftshift(torch.exp(1j * distance * torch.sqrt(radicand)))
        return transfer.to(torch.complex64)

    @property
    def control_phase(self) -> torch.Tensor:
        """Return the trainable phase on its native low-resolution grid."""
        if self.config.phase_parameterization == "unbounded":
            return self.phase_raw
        if self.config.phase_parameterization == "positive_bounded":
            return (
                float(self.config.phase_modulation_depth_rad)
                * float(self.config.phase_residual_scale)
                * torch.sigmoid(self.phase_raw)
            )
        return (
            math.pi
            * float(self.config.phase_residual_scale)
            * torch.tanh(self.phase_raw)
        )

    def expand_control_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Replicate each control value into one constant SLM superpixel block."""
        if tuple(phase.shape) != (self.control_size, self.control_size):
            raise ValueError("control phase must match phase_control_size")
        if self.control_size == self.size:
            return phase
        return functional.interpolate(
            phase[None, None], size=(self.size, self.size), mode="nearest"
        )[0, 0]

    @property
    def continuous_phase(self) -> torch.Tensor:
        if self._phase_override is not None:
            return self._phase_override
        return self.base_phase + self.trainable_residual_phase

    @property
    def trainable_residual_phase(self) -> torch.Tensor:
        """Return only the optimized phase, excluding fixed lens/carrier terms.

        Regularizers must operate on this tensor.  Applying smoothness to the
        complete phase would create gradients that cancel a fixed blaze carrier.
        """
        return self.expand_control_phase(self.control_phase)

    def set_phase_override(self, phase: Optional[torch.Tensor]) -> None:
        if phase is not None and tuple(phase.shape) != (self.size, self.size):
            raise ValueError("phase override must match the simulation grid")
        self._phase_override = phase

    def command_phase(self, design_phase: torch.Tensor) -> torch.Tensor:
        """Map a designed phase to the calibrated physical SLM command."""
        wrapped = torch.remainder(design_phase, 2.0 * math.pi)
        return torch.remainder(
            wrapped * float(self.config.phase_command_scale),
            2.0 * math.pi,
        )

    def phase_for_export(self) -> torch.Tensor:
        phase = torch.remainder(self.continuous_phase, 2.0 * math.pi)
        # Overrides represent an already exported/physical phase (for example
        # an 8-bit mask reloaded for quantized validation), so do not scale it
        # a second time. Preserve the CLOSED command interval [0, 2*pi]:
        # exported uint8 255 maps to LUT[255], not LUT[0]. When the calibrated
        # depth is below 2*pi these endpoints have different physical phases.
        # Unwrapped diagnostic overrides outside that interval retain the
        # historical periodic behavior; continuous training below is unchanged.
        if self._phase_override is not None:
            command = self._phase_override
            return torch.where((command >= 0) & (command <= 2.0 * math.pi), command, phase)
        return self.command_phase(phase)

    @property
    def applied_phase(self) -> torch.Tensor:
        """Approximate the physical phase after the calibrated 8-bit LUT."""
        return torch.clamp(
            self.phase_for_export(),
            max=float(self.config.phase_modulation_depth_rad),
        )

    def phase_uint8(self) -> torch.Tensor:
        phase = self.phase_for_export()
        return torch.round(phase / (2.0 * math.pi) * 255.0).clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _propagate(field: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(torch.fft.fftshift(field, dim=(-2, -1)))
        propagated = torch.fft.ifft2(spectrum * transfer)
        return torch.fft.ifftshift(propagated, dim=(-2, -1))

    def _intensity(self, logical_spikes: torch.Tensor) -> torch.Tensor:
        if logical_spikes.ndim != 4 or tuple(logical_spikes.shape[1:]) != (
            1,
            self.config.logical_size,
            self.config.logical_size,
        ):
            raise ValueError("logical_spikes must be [N,1,32,32]")
        if self.config.propagation_mode == "direct":
            return functional.adaptive_avg_pool2d(
                logical_spikes.float(), (self.config.readout_size, self.config.readout_size)
            )
        dmd_size = (
            self.size
            * int(self.config.resolved_dmd_active_size)
            // int(self.config.physical_size)
        )
        calibrated_input = logical_spikes.float() * self.input_intensity_gain[
            None, None
        ]
        amplitude = functional.interpolate(
            calibrated_input, size=(dmd_size, dmd_size), mode="nearest"
        ).sqrt()
        if dmd_size != self.size:
            dmd_total = self.size - dmd_size
            dmd_left = dmd_total // 2
            amplitude = functional.pad(
                amplitude,
                (dmd_left, dmd_total - dmd_left, dmd_left, dmd_total - dmd_left),
            )
        field = torch.complex(amplitude, torch.zeros_like(amplitude))
        field = functional.pad(
            field, (self.left_padding, self.right_padding, self.left_padding, self.right_padding,),
        )
        field = self._propagate(field, self.transfer_1)
        field = field * self.lens_1
        field = self._propagate(field, self.transfer_1)
        phase_values = (
            torch.zeros_like(self.applied_phase)
            if self.config.propagation_mode == "zero"
            else self.applied_phase
        )
        phase = functional.pad(
            phase_values[None, None],
            (self.left_padding, self.right_padding, self.left_padding, self.right_padding,),
        )
        field_at_phase_plane = field

        def output_intensity(active_field: torch.Tensor) -> torch.Tensor:
            propagated = self._propagate(active_field, self.transfer_2)
            propagated = propagated * self.lens_2
            propagated = self._propagate(propagated, self.transfer_2)
            start = self.left_padding
            propagated = propagated[
                ..., start : start + self.size, start : start + self.size
            ]
            values = propagated.real.square() + propagated.imag.square()
            ccd_size = (
                self.size
                * int(self.config.resolved_ccd_active_size)
                // int(self.config.physical_size)
            )
            if ccd_size != self.size:
                ccd_start = (self.size - ccd_size) // 2
                values = values[
                    ...,
                    ccd_start : ccd_start + ccd_size,
                    ccd_start : ccd_start + ccd_size,
                ]
            return values

        phase_transmittance = torch.exp(1j * phase)
        if self.phase_crosstalk_kernel.numel() > 1:
            real = _blur_2d(phase_transmittance.real, self.phase_crosstalk_kernel)
            imaginary = _blur_2d(
                phase_transmittance.imag, self.phase_crosstalk_kernel
            )
            phase_transmittance = torch.complex(real, imaginary)
        modulated_intensity = output_intensity(
            field_at_phase_plane * phase_transmittance
        )
        modulated_fraction = float(self.config.phase_modulated_fraction)
        if modulated_fraction >= 1.0:
            intensity = modulated_intensity
        elif modulated_fraction <= 0.0:
            intensity = output_intensity(field_at_phase_plane.detach())
        else:
            # The measured PSF is an intensity mixture, not a stable coherent
            # field sum.  Detach the fixed leakage branch so gradients describe
            # only the physically controllable fraction of the light.
            leakage_intensity = output_intensity(field_at_phase_plane.detach())
            intensity = (
                modulated_fraction * modulated_intensity
                + (1.0 - modulated_fraction) * leakage_intensity
            )
        readout = functional.adaptive_avg_pool2d(
            intensity, (self.config.readout_size, self.config.readout_size)
        )
        readout = _blur_2d(readout, self.readout_blur_kernel)
        return readout * self.readout_intensity_gain[None, None]

    @torch.no_grad()
    def calibrate_reference(self) -> float:
        """Freeze one sample-independent white scale at the initial phase."""
        white = torch.ones(
            (1, 1, self.config.logical_size, self.config.logical_size),
            dtype=self.phase_raw.dtype,
            device=self.phase_raw.device,
        )
        intensity = self._intensity(white)
        scale = torch.quantile(intensity.flatten(), 0.99).clamp_min(1.0e-8)
        self.reference_scale.copy_(scale)
        return float(scale.item())

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Map ``[B,T,1,32,32]`` spikes to normalized optical currents."""
        if spikes.ndim != 5:
            raise ValueError("spikes must have shape [B,T,1,32,32]")
        batch, time_steps = spikes.shape[:2]
        intensity = self._intensity(spikes.reshape(batch * time_steps, *spikes.shape[2:]))
        normalized = intensity / self.reference_scale.clamp_min(1.0e-8)
        return normalized.reshape(batch, time_steps, -1)
