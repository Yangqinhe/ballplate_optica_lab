"""Electronic-weight-free MNIST classifier with fixed optical detector regions."""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .config import OpticalClassifierConfig, OpticalConfig
from .optical import DifferentiableOpticalLayer


def build_detector_masks(
    readout_size: int, config: OpticalClassifierConfig
) -> torch.Tensor:
    """Place equal non-overlapping class regions on a grid or circular ring."""
    class_count = len(tuple(config.class_digits))
    if config.detector_layout == "ring":
        masks = torch.zeros((class_count, readout_size, readout_size), dtype=torch.float32)
        center_x_base = (readout_size - 1) / 2.0 + float(config.center_offset_x)
        center_y_base = (readout_size - 1) / 2.0 + float(config.center_offset_y)
        for label in range(class_count):
            angle = -math.pi / 2.0 + label * 2.0 * math.pi / float(class_count)
            center_x = center_x_base + float(config.ring_radius) * math.cos(angle)
            center_y = center_y_base + float(config.ring_radius) * math.sin(angle)
            x0 = int(round(center_x - (config.region_width - 1) / 2.0))
            y0 = int(round(center_y - (config.region_height - 1) / 2.0))
            x1 = x0 + config.region_width
            y1 = y0 + config.region_height
            if x0 < 0 or y0 < 0 or x1 > readout_size or y1 > readout_size:
                raise ValueError("ring detector regions do not fit inside the readout grid")
            if torch.any(masks[:, y0:y1, x0:x1]):
                raise ValueError("ring detector regions overlap")
            masks[label, y0:y1, x0:x1] = 1.0
        return masks.reshape(class_count, -1)
    if config.detector_layout != "grid":
        raise ValueError("detector_layout must be grid or ring")
    used_height = (
        config.detector_rows * config.region_height
        + (config.detector_rows - 1) * config.region_gap_y
    )
    used_width = (
        config.detector_columns * config.region_width
        + (config.detector_columns - 1) * config.region_gap_x
    )
    if used_height > readout_size or used_width > readout_size:
        raise ValueError("detector regions do not fit inside the readout grid")
    top = (readout_size - used_height) // 2 + int(round(config.center_offset_y))
    left = (readout_size - used_width) // 2 + int(round(config.center_offset_x))
    if top < 0 or left < 0 or top + used_height > readout_size or left + used_width > readout_size:
        raise ValueError("offset detector regions do not fit inside the readout grid")
    masks = torch.zeros(
        (config.detector_rows * config.detector_columns, readout_size, readout_size),
        dtype=torch.float32,
    )
    label = 0
    for row in range(config.detector_rows):
        for column in range(config.detector_columns):
            y0 = top + row * (config.region_height + config.region_gap_y)
            x0 = left + column * (config.region_width + config.region_gap_x)
            masks[label, y0 : y0 + config.region_height, x0 : x0 + config.region_width] = 1.0
            label += 1
    return masks.reshape(class_count, -1)


def load_simulation_detector_masks(
    path: Path, readout_size: int, class_count: int
) -> torch.Tensor:
    """Load fixed FIT-calibrated masks for differentiable optical training."""
    values = np.asarray(np.load(Path(path).resolve(), allow_pickle=False))
    expected_image_shape = (int(class_count), int(readout_size), int(readout_size))
    expected_flat_shape = (int(class_count), int(readout_size) * int(readout_size))
    if values.shape == expected_image_shape:
        values = values.reshape(expected_flat_shape)
    elif values.shape != expected_flat_shape:
        raise ValueError(
            "simulation detector masks must have shape %s or %s, got %s"
            % (expected_image_shape, expected_flat_shape, values.shape)
        )
    values = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("simulation detector masks must be finite and non-negative")
    if np.any(values.sum(axis=1) <= 0.0):
        raise ValueError("every simulation detector mask must contain positive area")
    if np.any(values.sum(axis=0) > 1.0 + 1.0e-6):
        raise ValueError("simulation detector masks must not overlap")
    return torch.from_numpy(values.copy())


class OpticalRegionClassifier(nn.Module):
    """Train only SLM phase; fixed detector energies are the class scores."""

    def __init__(self, optical: OpticalConfig, classifier: OpticalClassifierConfig) -> None:
        super().__init__()
        if optical.propagation_mode != "diffractive":
            raise ValueError('optical-only classifier requires propagation_mode="diffractive"')
        self.optical = DifferentiableOpticalLayer(optical)
        self.classifier_config = classifier
        detector_path = str(classifier.simulation_detector_regions_path).strip()
        detector_masks = (
            load_simulation_detector_masks(
                Path(detector_path),
                int(optical.readout_size),
                len(tuple(classifier.class_digits)),
            )
            if detector_path
            else build_detector_masks(optical.readout_size, classifier)
        )
        self.register_buffer(
            "detector_masks", detector_masks
        )

    def forward(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        optical_current = self.optical(spikes)
        region_trace = torch.einsum("btp,kp->btk", optical_current, self.detector_masks)
        region_energy = region_trace.sum(dim=1).clamp_min(1.0e-12)
        region_fraction = region_energy / region_energy.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        scores = torch.log(region_fraction.clamp_min(1.0e-12)) / float(
            self.classifier_config.temperature
        )
        total_energy = optical_current.sum(dim=(1, 2)).clamp_min(1.0e-12)
        captured_fraction = region_energy.sum(dim=1) / total_energy
        return {
            "optical_current": optical_current,
            "region_energy_trace": region_trace,
            "region_energy": region_energy,
            "region_fraction": region_fraction,
            "captured_fraction": captured_fraction,
            "scores": scores,
        }
