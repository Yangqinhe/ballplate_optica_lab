"""Scheme-B Optical-SNN: optical phase, hidden LIF, FC, and output LIF."""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import NeuronConfig, OpticalConfig
from .neurons import BoundedLIF, _inverse_softplus
from .optical import DifferentiableOpticalLayer


class OpticalSNN(nn.Module):
    """End-to-end differentiable phase-one network."""

    def __init__(self, optical: OpticalConfig, neurons: NeuronConfig) -> None:
        super().__init__()
        self.optical = DifferentiableOpticalLayer(optical)
        self.hidden_lif = BoundedLIF(
            neurons.hidden_leak, neurons.hidden_threshold, neurons.surrogate_slope
        )
        self.output_lif = BoundedLIF(
            neurons.output_leak, neurons.output_threshold, neurons.surrogate_slope
        )
        self.hidden_gain_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(neurons.hidden_gain), dtype=torch.float32)
        )
        self.fc = nn.Linear(optical.readout_size * optical.readout_size, 10)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.25)
        nn.init.zeros_(self.fc.bias)
        self.membrane_weight = float(neurons.membrane_weight)

    @property
    def hidden_gain(self) -> torch.Tensor:
        return functional.softplus(self.hidden_gain_raw)

    def forward(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        optical_current = self.optical(spikes)
        batch, time_steps, hidden_size = optical_current.shape
        hidden_membrane = torch.zeros(
            (batch, hidden_size), dtype=optical_current.dtype, device=optical_current.device
        )
        output_membrane = torch.zeros(
            (batch, 10), dtype=optical_current.dtype, device=optical_current.device
        )
        hidden_spikes: List[torch.Tensor] = []
        hidden_membranes: List[torch.Tensor] = []
        output_currents: List[torch.Tensor] = []
        output_spikes: List[torch.Tensor] = []
        output_membranes: List[torch.Tensor] = []
        for time_index in range(time_steps):
            hidden_current = self.hidden_gain * optical_current[:, time_index]
            hidden_spike, hidden_membrane = self.hidden_lif.step(hidden_current, hidden_membrane)
            output_current = self.fc(hidden_spike)
            output_spike, output_membrane = self.output_lif.step(output_current, output_membrane)
            hidden_spikes.append(hidden_spike)
            hidden_membranes.append(hidden_membrane)
            output_currents.append(output_current)
            output_spikes.append(output_spike)
            output_membranes.append(output_membrane)
        hidden_spike_trace = torch.stack(hidden_spikes, dim=1)
        output_spike_trace = torch.stack(output_spikes, dim=1)
        output_membrane_trace = torch.stack(output_membranes, dim=1)
        output_count = output_spike_trace.sum(dim=1)
        normalized_final_membrane = output_membrane_trace[
            :, -1
        ] / self.output_lif.threshold.clamp_min(1.0e-8)
        scores = output_count + self.membrane_weight * normalized_final_membrane
        return {
            "optical_current": optical_current,
            "hidden_spikes": hidden_spike_trace,
            "hidden_membrane": torch.stack(hidden_membranes, dim=1),
            "output_current": torch.stack(output_currents, dim=1),
            "output_spikes": output_spike_trace,
            "output_membrane": output_membrane_trace,
            "scores": scores,
        }
