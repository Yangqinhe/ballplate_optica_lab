"""Deterministic Bernoulli rate encoding for simulation and hardware parity."""

from typing import Optional

import torch


def _sample_seed(base_seed: int, dataset_index: int, split_offset: int, epoch: int) -> int:
    value = (
        int(base_seed) * 1_000_003
        + int(dataset_index) * 10_007
        + int(split_offset) * 1_009
        + int(epoch) * 97_003
    )
    return value % (2 ** 63 - 1)


def encode_rate_batch(
    images: torch.Tensor,
    dataset_indices: torch.Tensor,
    time_steps: int,
    base_seed: int,
    split_offset: int,
    epoch: Optional[int],
    minimum_probability: float = 1.0e-6,
) -> torch.Tensor:
    """Encode ``[B,1,32,32]`` images as reproducible ``[B,T,1,32,32]`` spikes."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 32, 32):
        raise ValueError("images must have shape [B,1,32,32]")
    if dataset_indices.ndim != 1 or len(dataset_indices) != len(images):
        raise ValueError("dataset_indices must contain one value per image")
    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    probabilities = images.detach().cpu().float().clamp(0.0, 1.0)
    probabilities = torch.where(
        probabilities >= float(minimum_probability), probabilities, torch.zeros_like(probabilities)
    )
    encoded = torch.empty(
        (len(images), int(time_steps), 1, 32, 32), dtype=torch.float32, device="cpu"
    )
    epoch_value = 0 if epoch is None else int(epoch) + 1
    for position, dataset_index in enumerate(dataset_indices.detach().cpu().tolist()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _sample_seed(base_seed, int(dataset_index), split_offset, epoch_value)
        )
        random_values = torch.rand(
            (time_steps, 1, 32, 32), generator=generator, dtype=torch.float32
        )
        encoded[position] = (random_values < probabilities[position]).float()
    return encoded


def encode_input_batch(
    images: torch.Tensor,
    dataset_indices: torch.Tensor,
    time_steps: int,
    base_seed: int,
    split_offset: int,
    epoch: Optional[int],
    minimum_probability: float = 1.0e-6,
    input_mode: str = "rate",
    binary_threshold: float = 0.5,
) -> torch.Tensor:
    """Encode rate spikes or one deterministic binary ONN input frame."""
    mode = str(input_mode).lower()
    if mode == "rate":
        return encode_rate_batch(
            images,
            dataset_indices,
            time_steps,
            base_seed,
            split_offset,
            epoch,
            minimum_probability,
        )
    if mode != "binary":
        raise ValueError("input_mode must be rate or binary")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 32, 32):
        raise ValueError("images must have shape [B,1,32,32]")
    if dataset_indices.ndim != 1 or len(dataset_indices) != len(images):
        raise ValueError("dataset_indices must contain one value per image")
    threshold = float(binary_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("binary_threshold must be in [0,1]")
    return (images.detach().cpu().float() >= threshold).float().unsqueeze(1)
