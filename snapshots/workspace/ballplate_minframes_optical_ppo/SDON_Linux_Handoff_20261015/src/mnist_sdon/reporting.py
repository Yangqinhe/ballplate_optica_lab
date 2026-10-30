"""Phase-one run directories, numeric reports, and compact diagnostic figures."""

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .config import Phase1Config
from .model import OpticalSNN


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def create_run_directory(config: Phase1Config, mode: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = config.outputs.root / "runs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("phase1_%s_%s" % (mode, timestamp))
    suffix = 1
    while path.exists():
        path = root / ("phase1_%s_%s_%02d" % (mode, timestamp, suffix))
        suffix += 1
    path.mkdir(parents=True)
    return path


def write_config_snapshot(config: Phase1Config, run_dir: Path) -> None:
    values = asdict(config)
    with (run_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(values), handle, ensure_ascii=False, indent=2)


def write_history(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    # Resumed runs may combine checkpoints produced by an older or
    # crash-isolated trainer with newly computed metrics.  Preserve the union
    # of fields instead of assuming every epoch has the first row's schema.
    fieldnames = list(history[0])
    for row in history[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, class_count: int = 10
) -> np.ndarray:
    matrix = np.zeros((int(class_count), int(class_count)), dtype=np.int64)
    for label, prediction in zip(labels.astype(int), predictions.astype(int)):
        matrix[label, prediction] += 1
    return matrix


def save_phase_arrays(model: OpticalSNN, run_dir: Path) -> None:
    control_phase = model.optical.control_phase.detach().cpu().numpy().astype(np.float32)
    phase = model.optical.continuous_phase.detach().cpu().numpy().astype(np.float32)
    phase_export = model.optical.phase_for_export().detach().cpu().numpy().astype(np.float32)
    phase_uint8 = model.optical.phase_uint8().detach().cpu().numpy()
    np.save(run_dir / "phase_control.npy", control_phase)
    np.save(run_dir / "phase_continuous.npy", phase)
    np.save(run_dir / "phase_wrapped.npy", phase_export)
    np.save(run_dir / "phase_uint8.npy", phase_uint8)


def save_figures(
    history: List[Dict[str, float]],
    model: OpticalSNN,
    validation: Tuple[np.ndarray, np.ndarray],
    run_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if history:
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, [row["fit_total"] for row in history], label="fit")
        axes[0].plot(epochs, [row["validation_total"] for row in history], label="validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(epochs, [row["fit_accuracy"] for row in history], label="fit")
        axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="validation")
        axes[1].set_title("LIF accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(figure_dir / "01_training_curves.png", dpi=160)
        plt.close(figure)

    phase = model.optical.phase_for_export().detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    axis.set_title("Trained phase [0, 2pi)")
    axis.axis("off")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(figure_dir / "02_trained_phase.png", dpi=160)
    plt.close(figure)

    labels, predictions = validation
    matrix = confusion_matrix(labels, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(figure_dir / "03_validation_confusion.png", dpi=160)
    plt.close(figure)


def write_summary(values: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(values), handle, ensure_ascii=False, indent=2)
