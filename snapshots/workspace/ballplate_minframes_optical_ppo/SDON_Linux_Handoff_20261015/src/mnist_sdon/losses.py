"""Joint classification, firing-rate, and phase regularization losses."""

from typing import Dict

import torch
import torch.nn.functional as functional

from .config import TrainingConfig
from .model import OpticalSNN


def _rate_penalty(rate: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    below = functional.relu(torch.as_tensor(minimum, device=rate.device) - rate)
    above = functional.relu(rate - torch.as_tensor(maximum, device=rate.device))
    return below.square() + above.square()


def phase_total_variation(phase: torch.Tensor) -> torch.Tensor:
    horizontal = (phase[:, 1:] - phase[:, :-1]).abs().mean()
    vertical = (phase[1:, :] - phase[:-1, :]).abs().mean()
    return horizontal + vertical


def joint_loss(
    model: OpticalSNN,
    result: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    training: TrainingConfig,
    auxiliary_weight: float,
) -> Dict[str, torch.Tensor]:
    """Return the differentiable total and detached-friendly components."""
    classification = functional.cross_entropy(result["scores"], labels)
    auxiliary = functional.cross_entropy(result["output_current"].mean(dim=1), labels)
    hidden_rate = result["hidden_spikes"].mean()
    output_rate = result["output_spikes"].mean()
    hidden_penalty = _rate_penalty(hidden_rate, training.hidden_rate_min, training.hidden_rate_max)
    output_penalty = _rate_penalty(output_rate, training.output_rate_min, training.output_rate_max)
    phase_tv = phase_total_variation(model.optical.trainable_residual_phase)
    total = (
        classification
        + float(auxiliary_weight) * auxiliary
        + training.hidden_rate_weight * hidden_penalty
        + training.output_rate_weight * output_penalty
        + training.phase_tv_weight * phase_tv
    )
    return {
        "total": total,
        "classification": classification,
        "auxiliary": auxiliary,
        "hidden_rate": hidden_rate,
        "output_rate": output_rate,
        "hidden_rate_penalty": hidden_penalty,
        "output_rate_penalty": output_penalty,
        "phase_tv": phase_tv,
    }
