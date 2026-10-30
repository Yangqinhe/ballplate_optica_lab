"""Warm-up, joint surrogate-BPTT training, validation, and checkpoints."""

import random
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from .config import Phase1Config
from .data import DatasetSplit, Phase1Data, load_phase1_data
from .encoding import encode_rate_batch
from .losses import joint_loss
from .model import OpticalSNN
from .reporting import (
    create_run_directory,
    save_figures,
    save_phase_arrays,
    write_config_snapshot,
    write_history,
    write_summary,
)


def _cuda_runtime_interlocked() -> bool:
    path = Path("outputs/mnist_sdon/CUDA_TRAINING_DISABLED.json")
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("recovery_override_blocked", False):
            return True
    except (OSError, TypeError, ValueError, AttributeError):
        return True
    override = os.environ.get("SDON_CUDA_REPAIR_OVERRIDE", "").strip().lower()
    override_enabled = override in {"1", "true", "yes", "fan_repair_verified"}
    return not override_enabled


@dataclass
class EpochResult:
    metrics: Dict[str, float]
    labels: np.ndarray
    predictions: np.ndarray


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(
            "cuda"
            if not _cuda_runtime_interlocked() and torch.cuda.is_available()
            else "cpu"
        )
    device = torch.device(value)
    if device.type == "cuda" and _cuda_runtime_interlocked():
        return torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not _cuda_runtime_interlocked():
        torch.cuda.manual_seed_all(seed)


def _batches(
    split: DatasetSplit, batch_size: int, shuffle: bool, seed: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    order = (
        torch.randperm(len(split), generator=generator)
        if shuffle
        else torch.arange(len(split), dtype=torch.long)
    )
    for start in range(0, len(split), batch_size):
        positions = order[start : start + batch_size]
        yield split.images[positions], split.labels[positions], split.indices[positions]


def _gradient_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())


def _auxiliary_weight(config: Phase1Config, epoch: int, total_epochs: int) -> float:
    decay_epochs = max(1, int(round(total_epochs * 0.30)))
    fraction = min(max(epoch - 1, 0) / decay_epochs, 1.0)
    return float(config.training.auxiliary_start_weight) * (1.0 - fraction)


def _make_optimizer(model: OpticalSNN, config: Phase1Config) -> torch.optim.Optimizer:
    lif_parameters = _electrical_lif_parameters(model)
    return torch.optim.AdamW(
        [
            {
                "params": [model.optical.phase_raw],
                "lr": config.training.phase_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.fc.parameters()),
                "lr": config.training.fc_learning_rate,
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": lif_parameters,
                "lr": config.training.lif_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )


def _electrical_lif_parameters(model: OpticalSNN) -> List[torch.nn.Parameter]:
    return [
        model.hidden_gain_raw,
        model.hidden_lif.leak_raw,
        model.hidden_lif.threshold_raw,
        model.output_lif.leak_raw,
        model.output_lif.threshold_raw,
    ]


def _set_electrical_trainable(model: OpticalSNN, enabled: bool) -> None:
    for parameter in list(model.fc.parameters()) + _electrical_lif_parameters(model):
        parameter.requires_grad_(enabled)


def run_epoch(
    model: OpticalSNN,
    split: DatasetSplit,
    config: Phase1Config,
    device: torch.device,
    epoch: int,
    auxiliary_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    split_offset: int,
    resample_spikes: bool = True,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    accumulation = config.training.effective_batch_size // config.training.micro_batch_size
    if training:
        optimizer.zero_grad(set_to_none=True)
    totals: Dict[str, float] = {}
    all_labels: List[torch.Tensor] = []
    all_predictions: List[torch.Tensor] = []
    sample_count = 0
    no_spike_count = 0
    update_count = 0
    phase_gradient_sum = 0.0
    batches = list(
        _batches(
            split,
            config.training.micro_batch_size,
            training,
            config.experiment.seed + epoch * 17 + split_offset,
        )
    )
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, (images, labels, indices) in enumerate(batches):
            spikes = encode_rate_batch(
                images,
                indices,
                config.encoding.time_steps,
                config.experiment.seed,
                split_offset,
                epoch if training and resample_spikes else None,
                config.encoding.minimum_probability,
            ).to(device)
            labels = labels.to(device)
            result = model(spikes)
            losses = joint_loss(model, result, labels, config.training, auxiliary_weight)
            batch_size = int(labels.shape[0])
            sample_count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_size
            predictions = result["scores"].argmax(dim=1)
            all_labels.append(labels.detach().cpu())
            all_predictions.append(predictions.detach().cpu())
            no_spike_count += int((result["output_spikes"].sum(dim=(1, 2)) == 0).sum().item())
            if training:
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, len(batches) - group_start)
                (losses["total"] / group_size).backward()
                last_batch = batch_index + 1 == len(batches)
                if (batch_index + 1) % accumulation == 0 or last_batch:
                    phase_gradient_sum += _gradient_norm(model.optical.phase_raw)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.gradient_clip_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_count += 1
    label_array = torch.cat(all_labels).numpy()
    prediction_array = torch.cat(all_predictions).numpy()
    metrics = {key: value / max(sample_count, 1) for key, value in totals.items()}
    metrics.update(
        accuracy=float(np.mean(label_array == prediction_array)),
        no_output_spike_ratio=float(no_spike_count / max(sample_count, 1)),
        phase_gradient_norm=float(phase_gradient_sum / max(update_count, 1)),
    )
    return EpochResult(metrics, label_array, prediction_array)


def gradient_connectivity_check(
    config: Phase1Config, data: Optional[Phase1Data] = None
) -> Dict[str, float]:
    """Prove that classification gradients reach phase, FC, and both LIF layers."""
    seed_everything(config.experiment.seed)
    data = load_phase1_data(config) if data is None else data
    device = resolve_device(config.experiment.device)
    model = OpticalSNN(config.optical, config.neurons).to(device)
    reference_scale = model.optical.calibrate_reference()
    split = data.overfit
    count = min(4, len(split))
    spikes = encode_rate_batch(
        split.images[:count],
        split.indices[:count],
        config.encoding.time_steps,
        config.experiment.seed,
        77,
        0,
        config.encoding.minimum_probability,
    ).to(device)
    labels = split.labels[:count].to(device)
    result = model(spikes)
    losses = joint_loss(model, result, labels, config.training, 1.0)
    losses["total"].backward()
    values = {
        "loss": float(losses["total"].item()),
        "reference_scale": reference_scale,
        "phase_gradient_norm": _gradient_norm(model.optical.phase_raw),
        "fc_gradient_norm": _gradient_norm(model.fc.weight),
        "hidden_threshold_gradient": _gradient_norm(model.hidden_lif.threshold_raw),
        "output_threshold_gradient": _gradient_norm(model.output_lif.threshold_raw),
        "hidden_rate": float(result["hidden_spikes"].mean().item()),
        "output_rate": float(result["output_spikes"].mean().item()),
    }
    required = (
        "phase_gradient_norm",
        "fc_gradient_norm",
        "hidden_threshold_gradient",
        "output_threshold_gradient",
    )
    if not all(np.isfinite(values[key]) and values[key] > 0.0 for key in required):
        raise RuntimeError("gradient connectivity check failed: %s" % values)
    return values


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save_checkpoint(
    path: Path,
    model: OpticalSNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: List[Dict[str, float]],
    best_accuracy: float,
    best_loss: float,
    initial_phase: torch.Tensor,
    phase_training: bool,
) -> None:
    payload = {
        "format_version": 2,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_accuracy": best_accuracy,
        "best_loss": best_loss,
        "initial_phase": initial_phase.detach().cpu(),
        "phase_training": bool(phase_training),
        "rng_state": _rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # A forced process stop can otherwise leave a newly renamed but unreadable
    # archive on Windows. Validate the temporary checkpoint before replacement.
    verified = torch.load(temporary, map_location="cpu")
    if int(verified.get("epoch", -1)) != int(epoch):
        raise RuntimeError("checkpoint verification failed for %s" % temporary)
    temporary.replace(path)


def train_phase1(
    config: Phase1Config,
    mode: str,
    epoch_limit: Optional[int] = None,
    phase_training: bool = True,
    resume_path: Optional[Path] = None,
) -> Path:
    """Run the overfit gate or the 1000/200 phase-one training curriculum."""
    if mode not in {"overfit", "train"}:
        raise ValueError("mode must be overfit or train")
    if phase_training and config.optical.propagation_mode != "diffractive":
        raise ValueError('phase training requires optical.propagation_mode="diffractive"')
    seed_everything(config.experiment.seed)
    data = load_phase1_data(config)
    fit = data.overfit if mode == "overfit" else data.fit
    validation = data.overfit if mode == "overfit" else data.validation
    device = resolve_device(config.experiment.device)
    model = OpticalSNN(config.optical, config.neurons).to(device)
    reference_scale = model.optical.calibrate_reference()
    initial_phase = model.optical.continuous_phase.detach().clone()
    optimizer = _make_optimizer(model, config)
    run_label = mode + ("_trainable_phase" if phase_training else "_frozen_phase")
    start_epoch = 1
    history: List[Dict[str, float]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    if resume_path is None:
        run_dir = create_run_directory(config, run_label)
        write_config_snapshot(config, run_dir)
    else:
        resume_path = Path(resume_path).resolve()
        checkpoint = torch.load(resume_path, map_location=device)
        if bool(checkpoint.get("phase_training", phase_training)) != bool(phase_training):
            raise ValueError("resume checkpoint phase mode does not match --phase-mode")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        initial_phase = checkpoint.get("initial_phase", initial_phase.cpu()).to(device)
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng_state"])
        run_dir = resume_path.parent
    total_epochs = (
        config.training.warmup_epochs
        + config.training.phase_only_epochs
        + config.training.joint_epochs
    )
    if epoch_limit is not None:
        total_epochs = min(total_epochs, int(epoch_limit))
    for epoch in range(start_epoch, total_epochs + 1):
        warmup = phase_training and epoch <= min(config.training.warmup_epochs, total_epochs)
        phase_only = (
            phase_training
            and not warmup
            and epoch <= config.training.warmup_epochs + config.training.phase_only_epochs
        )
        model.optical.phase_raw.requires_grad_(phase_training and not warmup)
        _set_electrical_trainable(model, not phase_only)
        auxiliary = _auxiliary_weight(config, epoch, total_epochs)
        fit_offset = 77 if mode == "overfit" else 11
        validation_offset = 77 if mode == "overfit" else 29
        fit_result = run_epoch(
            model,
            fit,
            config,
            device,
            epoch,
            auxiliary,
            optimizer,
            split_offset=fit_offset,
            resample_spikes=mode != "overfit",
        )
        validation_result = run_epoch(
            model, validation, config, device, epoch, 0.0, None, split_offset=validation_offset,
        )
        phase_change = float(
            (model.optical.continuous_phase.detach() - initial_phase).square().mean().sqrt().item()
        )
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "phase_trainable": float(phase_training and not warmup),
            "electrical_trainable": float(not phase_only),
            "phase_only_stage": float(phase_only),
            "auxiliary_weight": auxiliary,
            "phase_rms_change": phase_change,
        }
        row.update({"fit_" + key: value for key, value in fit_result.metrics.items()})
        row.update({"validation_" + key: value for key, value in validation_result.metrics.items()})
        history.append(row)
        validation_accuracy = validation_result.metrics["accuracy"]
        validation_loss = validation_result.metrics["total"]
        eligible = not warmup or total_epochs <= config.training.warmup_epochs
        improved = eligible and (
            validation_accuracy > best_accuracy
            or (validation_accuracy == best_accuracy and validation_loss < best_loss)
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            _save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                history,
                best_accuracy,
                best_loss,
                initial_phase,
                phase_training,
            )
        _save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            history,
            best_accuracy,
            best_loss,
            initial_phase,
            phase_training,
        )
        write_history(history, run_dir / "training_history.csv")
        print(
            "epoch %d/%d [%s] | fit=%.3f val=%.3f | hidden=%.3f output=%.3f | phase=%.5f"
            % (
                epoch,
                total_epochs,
                "warmup" if warmup else ("phase-only" if phase_only else "joint"),
                fit_result.metrics["accuracy"],
                validation_accuracy,
                fit_result.metrics["hidden_rate"],
                fit_result.metrics["output_rate"],
                phase_change,
            )
        )

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.optical.phase_raw.requires_grad_(False)
    validation_offset = 77 if mode == "overfit" else 29
    best_result = run_epoch(
        model,
        validation,
        config,
        device,
        int(checkpoint["epoch"]),
        0.0,
        None,
        split_offset=validation_offset,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.experiment.seed + 404)
    trained_phase = model.optical.continuous_phase.detach()
    model.optical.set_phase_override(initial_phase)
    initial_phase_result = run_epoch(
        model,
        validation,
        config,
        device,
        int(checkpoint["epoch"]),
        0.0,
        None,
        split_offset=validation_offset,
    )
    permutation = torch.randperm(trained_phase.numel(), generator=generator).to(device)
    shuffled_phase = trained_phase.flatten()[permutation].reshape_as(trained_phase)
    model.optical.set_phase_override(shuffled_phase)
    shuffled_result = run_epoch(
        model,
        validation,
        config,
        device,
        int(checkpoint["epoch"]),
        0.0,
        None,
        split_offset=validation_offset,
    )
    model.optical.set_phase_override(None)
    save_phase_arrays(model, run_dir)
    save_figures(history, model, (best_result.labels, best_result.predictions), run_dir)
    write_summary(
        {
            "mode": mode,
            "propagation_mode": config.optical.propagation_mode,
            "phase_training": phase_training,
            "device": str(device),
            "reference_scale": reference_scale,
            "epochs": total_epochs,
            "best_validation_accuracy": best_accuracy,
            "best_validation_loss": best_loss,
            "initial_phase_validation_accuracy": initial_phase_result.metrics["accuracy"],
            "shuffled_phase_validation_accuracy": shuffled_result.metrics["accuracy"],
            "phase_rms_change": float(
                (model.optical.continuous_phase.detach() - initial_phase)
                .square()
                .mean()
                .sqrt()
                .item()
            ),
            "run_dir": run_dir,
        },
        run_dir / "summary.json",
    )
    return run_dir


def run_phase1_ablation(config: Phase1Config, epoch_limit: Optional[int] = None) -> Dict[str, Any]:
    """Train fair frozen/trainable controls and report the phase contribution."""
    frozen_dir = train_phase1(config, "train", epoch_limit=epoch_limit, phase_training=False)
    trainable_dir = train_phase1(config, "train", epoch_limit=epoch_limit, phase_training=True)
    with (frozen_dir / "summary.json").open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    with (trainable_dir / "summary.json").open("r", encoding="utf-8") as handle:
        trainable = json.load(handle)
    comparison = {
        "frozen_run": str(frozen_dir),
        "trainable_run": str(trainable_dir),
        "frozen_validation_accuracy": frozen["best_validation_accuracy"],
        "trainable_validation_accuracy": trainable["best_validation_accuracy"],
        "trainable_model_with_initial_phase_accuracy": trainable[
            "initial_phase_validation_accuracy"
        ],
        "shuffled_phase_validation_accuracy": trainable["shuffled_phase_validation_accuracy"],
        "trainable_minus_frozen": (
            trainable["best_validation_accuracy"] - frozen["best_validation_accuracy"]
        ),
    }
    write_summary(comparison, trainable_dir / "ablation_summary.json")
    return comparison
