"""Surrogate-gradient spikes and bounded leaky integrate-and-fire neurons."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1.0e-6)))


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, distance: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(distance)  # type: ignore[attr-defined]
        ctx.slope = float(slope)  # type: ignore[attr-defined]
        return (distance >= 0.0).to(distance.dtype)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (distance,) = ctx.saved_tensors  # type: ignore[attr-defined]
        slope = ctx.slope  # type: ignore[attr-defined]
        surrogate = 1.0 / (1.0 + slope * distance.abs()).square()
        return gradient * surrogate, None


def surrogate_spike(distance: torch.Tensor, slope: float) -> torch.Tensor:
    """Use hard 0/1 spikes forward and a fast-sigmoid derivative backward."""
    return _SurrogateSpike.apply(distance, float(slope))


class BoundedLIF(nn.Module):
    """One LIF layer with trainable shared leak and threshold scalars."""

    def __init__(
        self,
        leak: float,
        threshold: float,
        surrogate_slope: float,
        threshold_minimum: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < leak < 1.0:
            raise ValueError("leak must be in (0,1)")
        if threshold <= threshold_minimum:
            raise ValueError("threshold must be greater than threshold_minimum")
        self.leak_raw = nn.Parameter(torch.tensor(math.log(leak / (1.0 - leak))))
        self.threshold_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(threshold - threshold_minimum))
        )
        self.surrogate_slope = float(surrogate_slope)
        self.threshold_minimum = float(threshold_minimum)

    @property
    def leak(self) -> torch.Tensor:
        return torch.sigmoid(self.leak_raw)

    @property
    def threshold(self) -> torch.Tensor:
        return self.threshold_minimum + functional.softplus(self.threshold_raw)

    def step(
        self, current: torch.Tensor, membrane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        before_reset = self.leak * membrane + current
        spikes = surrogate_spike(before_reset - self.threshold, self.surrogate_slope)
        after_reset = before_reset - spikes * self.threshold
        return spikes, after_reset
