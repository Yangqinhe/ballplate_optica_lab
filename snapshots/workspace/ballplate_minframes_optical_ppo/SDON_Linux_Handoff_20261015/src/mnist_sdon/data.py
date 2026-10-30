"""MNIST loading and class-balanced phase-one splits."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as functional

from .config import Phase1Config


@dataclass
class DatasetSplit:
    images: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class Phase1Data:
    fit: DatasetSplit
    validation: DatasetSplit
    overfit: DatasetSplit


class _RawMnistDataset:
    """Minimal tensor-backed reader for the official MNIST IDX files.

    The hardware process does not need torchvision's image codec extension.
    Reading the local IDX cache directly also avoids loading ``image.pyd`` in
    the same process as the FLIR camera SDK, which is more stable on the
    laboratory Windows environment.
    """

    def __init__(self, images: torch.Tensor, targets: torch.Tensor) -> None:
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.images[int(index)].to(dtype=torch.float32).div(255.0)
        return image.unsqueeze(0), int(self.targets[int(index)].item())


def _read_mnist_idx(root: Path, train: bool) -> _RawMnistDataset:
    """Read an existing torchvision-style MNIST raw cache without torchvision."""
    raw_dir = Path(root).resolve() / "MNIST" / "raw"
    stem = "train" if train else "t10k"
    image_path = raw_dir / (stem + "-images-idx3-ubyte")
    label_path = raw_dir / (stem + "-labels-idx1-ubyte")
    if not image_path.is_file() or not label_path.is_file():
        missing = image_path if not image_path.is_file() else label_path
        raise FileNotFoundError(missing)

    with image_path.open("rb") as handle:
        header = np.frombuffer(handle.read(16), dtype=">i4")
        if header.shape != (4,) or int(header[0]) != 2051:
            raise ValueError("invalid MNIST image IDX header: %s" % image_path)
        count, rows, columns = (int(value) for value in header[1:])
        image_values = np.frombuffer(handle.read(), dtype=np.uint8)
    if image_values.size != count * rows * columns:
        raise ValueError("truncated MNIST image IDX file: %s" % image_path)

    with label_path.open("rb") as handle:
        header = np.frombuffer(handle.read(8), dtype=">i4")
        if header.shape != (2,) or int(header[0]) != 2049:
            raise ValueError("invalid MNIST label IDX header: %s" % label_path)
        label_count = int(header[1])
        label_values = np.frombuffer(handle.read(), dtype=np.uint8)
    if label_values.size != label_count or label_count != count:
        raise ValueError("MNIST image/label count mismatch in %s" % raw_dir)

    # Copy the arrays so Torch owns writable memory after the file handles close.
    images = torch.from_numpy(
        image_values.reshape(count, rows, columns).copy()
    )
    targets = torch.from_numpy(label_values.astype(np.int64, copy=True))
    return _RawMnistDataset(images, targets)


def load_fixed_validation_manifest(
    path: Path, config: Phase1Config
) -> Dict[str, Any]:
    """Load and validate the explicit validation set shared by GUI pages 10/11."""
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("fixed validation manifest must contain a JSON object")
    if payload.get("kind") not in {
        "mnist_onn_fixed_validation_set",
        "mnist_onn4_fixed_validation_set",  # Legacy four-class manifests.
    }:
        raise ValueError("unsupported fixed validation manifest kind")
    class_digits = tuple(int(value) for value in payload.get("class_digits", ()))
    expected_digits = tuple(
        int(value) for value in config.optical_classifier.class_digits
    )
    if class_digits != expected_digits:
        raise ValueError(
            "fixed validation digits %s do not match config %s"
            % (class_digits, expected_digits)
        )
    indices = [int(value) for value in payload.get("dataset_indices", ())]
    labels = [int(value) for value in payload.get("mapped_labels", ())]
    if not indices or len(indices) != len(labels):
        raise ValueError("fixed validation indices and labels must be non-empty and matched")
    if len(set(indices)) != len(indices):
        raise ValueError("fixed validation dataset indices must be unique")
    if int(payload.get("sample_count", -1)) != len(indices):
        raise ValueError("fixed validation sample_count does not match its indices")
    if any(label < 0 or label >= len(class_digits) for label in labels):
        raise ValueError("fixed validation mapped label is outside the class range")
    expected_sha = str(payload.get("test_set_sha256", "")).lower()
    canonical = dict(payload)
    canonical.pop("test_set_sha256", None)
    actual_sha = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError("fixed validation manifest SHA256 does not match its contents")
    return {
        "path": manifest_path,
        "test_set_id": actual_sha,
        "dataset_indices": indices,
        "mapped_labels": labels,
        "sample_count": len(indices),
    }


def create_fixed_validation_manifest(
    path: Path,
    config: Phase1Config,
    sample_count: int,
    sample_offset: int = 0,
) -> Path:
    """Freeze an ordered validation subset and its labels for repeatable hardware runs."""
    start = max(0, int(sample_offset))
    count = max(1, int(sample_count))
    validation = load_phase1_data(config).validation
    stop = start + count
    if stop > len(validation):
        raise ValueError(
            "fixed validation selection [%d:%d] exceeds %d samples"
            % (start, stop, len(validation))
        )
    selected_indices = validation.indices[start:stop].tolist()
    selected_labels = validation.labels[start:stop].tolist()
    payload = {
        "format_version": 1,
        "kind": "mnist_onn_fixed_validation_set",
        "split_name": "validation",
        "class_digits": [
            int(value) for value in config.optical_classifier.class_digits
        ],
        "sample_offset": start,
        "sample_count": count,
        "dataset_indices": [int(value) for value in selected_indices],
        "mapped_labels": [int(value) for value in selected_labels],
    }
    payload["test_set_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def select_fixed_validation_split(
    split: DatasetSplit, manifest: Mapping[str, Any]
) -> DatasetSplit:
    """Select an ordered DatasetSplit by original MNIST indices and verify labels."""
    available = {
        int(dataset_index): position
        for position, dataset_index in enumerate(split.indices.tolist())
    }
    requested = [int(value) for value in manifest["dataset_indices"]]
    missing = [value for value in requested if value not in available]
    if missing:
        raise ValueError(
            "fixed validation set is not contained in the configured validation split; "
            "first missing MNIST index=%d" % missing[0]
        )
    positions = torch.tensor(
        [available[value] for value in requested], dtype=torch.long
    )
    selected = DatasetSplit(
        images=split.images[positions].clone(),
        labels=split.labels[positions].clone(),
        indices=split.indices[positions].clone(),
    )
    expected_labels = torch.tensor(
        [int(value) for value in manifest["mapped_labels"]], dtype=torch.long
    )
    if not torch.equal(selected.labels, expected_labels):
        raise ValueError("fixed validation labels do not match the local MNIST cache")
    return selected


def prepare_mnist_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a torchvision MNIST image to one centered 32x32 channel."""
    image = image.float()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if tuple(image.shape) == (1, 28, 28):
        image = functional.pad(image, (2, 2, 2, 2))
    elif tuple(image.shape) != (1, 32, 32):
        image = functional.interpolate(
            image.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze(0)
    return image.clamp(0.0, 1.0)


def _class_buckets(
    targets: torch.Tensor, seed: int, class_labels: Sequence[int]
) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    values = targets.detach().cpu().numpy()
    buckets: Dict[int, List[int]] = {}
    for label in class_labels:
        indices = np.flatnonzero(values == label)
        rng.shuffle(indices)
        buckets[label] = [int(value) for value in indices]
    return buckets


def _take_balanced(
    buckets: Dict[int, List[int]], positions: Dict[int, int], count: int
) -> List[int]:
    selected: List[int] = []
    while len(selected) < count:
        progressed = False
        for label in buckets:
            if len(selected) >= count:
                break
            position = positions[label]
            if position < len(buckets[label]):
                selected.append(buckets[label][position])
                positions[label] += 1
                progressed = True
        if not progressed:
            raise ValueError("MNIST train split does not contain enough samples")
    return selected


def _materialize(
    dataset: object, indices: List[int], label_map: Mapping[int, int]
) -> DatasetSplit:
    images = torch.empty((len(indices), 1, 32, 32), dtype=torch.float32)
    labels = torch.empty((len(indices),), dtype=torch.long)
    for position, dataset_index in enumerate(indices):
        image, label = dataset[dataset_index]  # type: ignore[index]
        images[position] = prepare_mnist_image(image)
        labels[position] = int(label_map[int(label)])
    return DatasetSplit(images, labels, torch.tensor(indices, dtype=torch.long))


def load_phase1_data(config: Phase1Config) -> Phase1Data:
    """Load balanced MNIST fit/validation subsets without cross-split leakage."""
    try:
        fit_dataset = _read_mnist_idx(config.data.root, train=True)
        validation_dataset = (
            _read_mnist_idx(config.data.root, train=False)
            if config.data.validation_source == "test"
            else fit_dataset
        )
    except FileNotFoundError as raw_error:
        # Keep the original download capability for a new workstation. The lab
        # hardware path normally takes the direct-IDX branch above and therefore
        # never loads torchvision's optional native image decoder.
        if not config.data.download:
            raise RuntimeError(
                "MNIST raw IDX cache is incomplete and data.download is false"
            ) from raw_error
        try:
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "MNIST download requires matching torch and torchvision"
            ) from exc
        fit_dataset = MNIST(
            root=str(config.data.root), train=True, transform=ToTensor(), download=True,
        )
        validation_dataset = (
            MNIST(
                root=str(config.data.root),
                train=False,
                transform=ToTensor(),
                download=True,
            )
            if config.data.validation_source == "test"
            else fit_dataset
        )
    class_digits = tuple(int(value) for value in config.optical_classifier.class_digits)
    label_map = {digit: index for index, digit in enumerate(class_digits)}
    fit_buckets = _class_buckets(
        fit_dataset.targets, config.experiment.seed, class_digits
    )
    fit_positions = {label: 0 for label in class_digits}
    fit_indices = _take_balanced(
        fit_buckets, fit_positions, config.data.fit_samples
    )
    fit = _materialize(fit_dataset, fit_indices, label_map)

    if config.data.validation_source == "test":
        validation_buckets = _class_buckets(
            validation_dataset.targets, config.experiment.seed + 1, class_digits
        )
        validation_positions = {label: 0 for label in class_digits}
    else:
        validation_dataset = fit_dataset
        validation_buckets = fit_buckets
        validation_positions = fit_positions
    validation_indices = _take_balanced(
        validation_buckets,
        validation_positions,
        config.data.validation_samples,
    )
    validation = _materialize(validation_dataset, validation_indices, label_map)

    overfit_count = min(config.data.overfit_samples, len(fit_indices))
    remapped_labels = tuple(range(len(class_digits)))
    overfit_positions = _take_balanced(
        _class_buckets(fit.labels, config.experiment.seed + 101, remapped_labels),
        {label: 0 for label in remapped_labels},
        overfit_count,
    )
    overfit = DatasetSplit(
        fit.images[overfit_positions].clone(),
        fit.labels[overfit_positions].clone(),
        fit.indices[overfit_positions].clone(),
    )
    return Phase1Data(fit=fit, validation=validation, overfit=overfit)
