"""Deploy one trained optical-only phase to DMD/SLM/CCD hardware."""

import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sdon.calibration.intensity import capture_median
from sdon.config import load_config

from .config import Phase1Config, load_phase1_config
from .data import (
    load_fixed_validation_manifest,
    load_phase1_data,
    select_fixed_validation_split,
)
from .encoding import encode_input_batch
from .optical_classifier import OpticalRegionClassifier, build_detector_masks
from .reporting import write_summary
from .training import resolve_device, seed_everything


def prepare_modulator_aperture(
    image: np.ndarray,
    active_size: int,
    canvas_size: Optional[int] = None,
    effective_size: Optional[int] = None,
) -> np.ndarray:
    """Resize one logical pattern into a centered physical modulator aperture.

    Passing a small array directly to the Qt display stretches it over the
    complete 512-pixel canvas.  DMD and SLM call sites therefore select their
    own physical support explicitly; the two planes need not share one size.
    """
    values = np.asarray(image)
    active = int(active_size)
    canvas = max(512, active) if canvas_size is None else int(canvas_size)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("modulator image must be one square 2-D array")
    if active <= 0 or canvas <= 0 or active > canvas:
        raise ValueError("active_size must be in [1, canvas_size]")
    if effective_size is not None:
        effective = int(effective_size)
        if effective < 1 or active % effective:
            raise ValueError("effective_size must divide active_size")
        if values.shape != (effective, effective):
            values = cv2.resize(
                values, (effective, effective), interpolation=cv2.INTER_NEAREST
            )
        reuse = active // effective
        values = np.repeat(np.repeat(values, reuse, axis=0), reuse, axis=1)
    elif values.shape != (active, active):
        values = cv2.resize(values, (active, active), interpolation=cv2.INTER_NEAREST)
    output = np.zeros((canvas, canvas), dtype=values.dtype)
    start = (canvas - active) // 2
    output[start : start + active, start : start + active] = values
    return output


def crop_registered_aperture(
    registered_frames: np.ndarray, active_size: Optional[int]
) -> np.ndarray:
    """Crop registered CCD data to the centered physical experiment aperture."""
    frames = np.asarray(registered_frames)
    if frames.ndim < 2:
        raise ValueError("registered frames must have at least two dimensions")
    height, width = frames.shape[-2:]
    if height != width:
        raise ValueError("registered frames must be square")
    if active_size is None:
        return frames
    active = int(active_size)
    if active <= 0 or active > height:
        raise ValueError("active aperture must fit inside registered frames")
    start = (height - active) // 2
    return frames[..., start : start + active, start : start + active]


def pool_registered_readout(
    registered_frames: np.ndarray,
    readout_size: int,
    active_aperture_px: Optional[int] = None,
    correction_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the exact registered CCD image seen by the detector masks.

    The hardware preview and the power calculation must share this operation:
    central active-aperture crop first, then area pooling to the simulated
    readout grid.  Showing the larger registration canvas with readout-grid
    boxes makes the two coordinate systems look inconsistent even when the
    numerical integration is correct.
    """
    frames = np.asarray(registered_frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("registered_frames must be [T,H,W]")
    size = int(readout_size)
    if size < 1:
        raise ValueError("readout_size must be positive")
    active = crop_registered_aperture(frames, active_aperture_px)
    pooled = np.stack(
        [
            cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            for frame in active
        ],
        axis=0,
    ).astype(np.float32)
    if correction_matrix is None:
        return pooled
    matrix = np.asarray(correction_matrix, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("readout correction matrix must be one finite 2x3 affine")
    return np.stack(
        [
            cv2.warpAffine(
                frame,
                matrix,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            for frame in pooled
        ],
        axis=0,
    ).astype(np.float32)


def load_readout_correction(path: Optional[Path]) -> Optional[np.ndarray]:
    """Load one label-free ONN readout correction JSON."""
    if path is None or not str(path).strip():
        return None
    source = Path(path).resolve()
    payload = _load_json(source)
    if payload.get("kind") != "onn_label_free_readout_affine":
        raise ValueError("unsupported ONN readout correction kind: %s" % source)
    if not bool(payload.get("valid", False)):
        raise ValueError("ONN readout correction is not marked valid: %s" % source)
    matrix = np.asarray(payload.get("matrix_2x3"), dtype=np.float32)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("ONN readout correction has no finite 2x3 matrix")
    return matrix


def load_detector_masks(
    path: Optional[Path],
    readout_size: int,
    class_count: int,
    require_valid: bool = True,
) -> Optional[np.ndarray]:
    """Load fixed hardware-only detector rectangles.

    Production callers keep ``require_valid=True``.  A diagnostic caller may
    explicitly inspect a provisional FIT result without making that result
    eligible for deployment or changing its persisted ``valid`` flag.
    """
    if path is None or not str(path).strip():
        return None
    source = Path(path).resolve()
    payload = _load_json(source)
    if payload.get("kind") != "onn_fixed_detector_region_calibration":
        raise ValueError("unsupported ONN detector calibration kind: %s" % source)
    if bool(require_valid) and not bool(payload.get("valid", False)):
        raise ValueError("ONN detector calibration is not marked valid: %s" % source)
    rectangles = payload.get("calibrated_rectangles_xyxy")
    if not isinstance(rectangles, list) or len(rectangles) != int(class_count):
        raise ValueError("ONN detector calibration class count does not match")
    size = int(readout_size)
    masks = np.zeros((int(class_count), size, size), dtype=np.float32)
    occupied = np.zeros((size, size), dtype=bool)
    for index, rectangle in enumerate(rectangles):
        if not isinstance(rectangle, list) or len(rectangle) != 4:
            raise ValueError("ONN detector rectangle must be [x0,y0,x1,y1]")
        x0, y0, x1, y1 = map(int, rectangle)
        if x0 < 0 or y0 < 0 or x1 > size or y1 > size or x0 >= x1 or y0 >= y1:
            raise ValueError("ONN detector rectangle lies outside readout")
        if np.any(occupied[y0:y1, x0:x1]):
            raise ValueError("ONN detector rectangles overlap")
        masks[index, y0:y1, x0:x1] = 1.0
        occupied[y0:y1, x0:x1] = True
    return masks.reshape(int(class_count), -1)


def register_ccd_to_simulation(
    measured: np.ndarray, simulation_to_ccd: np.ndarray, output_size: int = 512
) -> np.ndarray:
    """Resample one measured CCD frame onto the saved simulation coordinate system."""
    transform = np.asarray(simulation_to_ccd, dtype=np.float32)
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("simulation_to_ccd must be one finite 2x3 affine transform")
    return cv2.warpAffine(
        np.asarray(measured, dtype=np.float32),
        transform,
        (int(output_size), int(output_size)),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rebase_centered_affine(
    simulation_to_ccd: np.ndarray, source_canvas_size: int, output_canvas_size: int
) -> np.ndarray:
    """Express a centered affine registration on a different simulation canvas."""
    transform = np.asarray(simulation_to_ccd, dtype=np.float32).copy()
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("simulation_to_ccd must be one finite 2x3 affine transform")
    source = int(source_canvas_size)
    output = int(output_canvas_size)
    if source < 1 or output < 1:
        raise ValueError("canvas sizes must be positive")
    shift = np.asarray([(output - source) / 2.0] * 2, dtype=np.float32)
    transform[:, 2] -= transform[:, :2] @ shift
    return transform


def detector_region_trace(
    registered_frames: np.ndarray,
    detector_masks: np.ndarray,
    active_aperture_px: Optional[int] = None,
    correction_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert registered CCD frames into per-time-step region energy and fractions."""
    frames = np.asarray(registered_frames, dtype=np.float32)
    masks = np.asarray(detector_masks, dtype=np.float32)
    if frames.ndim != 3 or masks.ndim != 2:
        raise ValueError("frames must be [T,H,W] and masks must be [classes,P]")
    readout_size = int(round(np.sqrt(masks.shape[1])))
    if readout_size * readout_size != masks.shape[1]:
        raise ValueError("detector mask width must be a square readout grid")
    pooled = pool_registered_readout(
        frames,
        readout_size,
        active_aperture_px=active_aperture_px,
        correction_matrix=correction_matrix,
    )
    energy = pooled.reshape(len(pooled), -1) @ masks.T
    fractions = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1.0e-12)
    return energy.astype(np.float32), fractions.astype(np.float32)


def detector_energy_utilization_trace(
    registered_frames: np.ndarray,
    detector_masks: np.ndarray,
    active_aperture_px: Optional[int] = None,
    correction_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return detector-box energy divided by the full pooled CCD energy.

    Region fractions always sum to one and therefore hide how much light lies
    outside every class box.  This independent accounting value makes a tiny
    or misplaced detector layout immediately visible in saved results and in
    the GUI.
    """
    frames = np.asarray(registered_frames, dtype=np.float32)
    masks = np.asarray(detector_masks, dtype=np.float32)
    if frames.ndim != 3 or masks.ndim != 2:
        raise ValueError("frames must be [T,H,W] and masks must be [classes,P]")
    readout_size = int(round(np.sqrt(masks.shape[1])))
    if readout_size * readout_size != masks.shape[1]:
        raise ValueError("detector mask width must be a square readout grid")
    pooled = pool_registered_readout(
        frames,
        readout_size,
        active_aperture_px=active_aperture_px,
        correction_matrix=correction_matrix,
    )
    flattened = pooled.reshape(len(pooled), -1)
    boxed = (flattened @ masks.T).sum(axis=1)
    full = flattened.sum(axis=1)
    return (boxed / np.maximum(full, 1.0e-12)).astype(np.float32)


def fixed_lif_readout(
    region_fraction_trace: np.ndarray,
    leak: float = 0.875,
    threshold: float = 1.0,
    gain: float = 3.0,
    membrane_weight: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Apply ten fixed, identically configured LIF neurons without trainable weights."""
    if not 0.0 < float(leak) < 1.0:
        raise ValueError("LIF leak must be in (0,1)")
    if float(threshold) <= 0.0 or float(gain) <= 0.0:
        raise ValueError("LIF threshold and gain must be positive")
    fractions = np.asarray(region_fraction_trace, dtype=np.float32)
    if fractions.ndim != 2 or fractions.shape[1] != 10:
        raise ValueError("region_fraction_trace must have shape [T,10]")
    membrane = np.zeros(10, dtype=np.float32)
    spikes: List[np.ndarray] = []
    membranes: List[np.ndarray] = []
    for fraction in fractions:
        before_reset = float(leak) * membrane + float(gain) * fraction
        spike = (before_reset >= float(threshold)).astype(np.float32)
        membrane = before_reset - spike * float(threshold)
        spikes.append(spike)
        membranes.append(membrane.copy())
    spike_trace = np.stack(spikes, axis=0)
    membrane_trace = np.stack(membranes, axis=0)
    scores = spike_trace.sum(axis=0) + float(membrane_weight) * membrane / float(threshold)
    return {
        "spikes": spike_trace,
        "membrane": membrane_trace,
        "scores": scores.astype(np.float32),
    }


def radiometric_region_gain(
    simulation_white_fraction: np.ndarray,
    measured_white_fraction: np.ndarray,
    minimum_gain: float = 0.1,
    maximum_gain: float = 10.0,
) -> np.ndarray:
    """Match label-free measured white-field region response to the simulator."""
    simulation = np.asarray(simulation_white_fraction, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_white_fraction, dtype=np.float64).reshape(-1)
    if simulation.shape != (10,) or measured.shape != (10,):
        raise ValueError("white-field fractions must each contain ten regions")
    simulation = simulation / max(float(simulation.sum()), 1.0e-12)
    measured = measured / max(float(measured.sum()), 1.0e-12)
    gain = simulation / np.maximum(measured, 1.0e-12)
    gain = np.clip(gain, float(minimum_gain), float(maximum_gain))
    return gain.astype(np.float32)


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return value


def simulate_optical_chain(
    config: Phase1Config,
    phase_uint8: np.ndarray,
    sample_count: int,
    sample_offset: int = 0,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    test_set_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return matched DMD, SLM and simulated CCD values for fixed-phase inference."""
    seed_everything(config.experiment.seed)
    data = load_phase1_data(config)
    split = data.validation
    manifest = None
    if test_set_manifest is not None:
        manifest = load_fixed_validation_manifest(Path(test_set_manifest), config)
        split = select_fixed_validation_split(split, manifest)
    start = max(0, int(sample_offset))
    count = min(int(sample_count), max(0, len(split) - start))
    if count < 1:
        raise ValueError("simulation sample range is outside the validation split")
    stop = start + count
    device = resolve_device(config.experiment.device)
    model = OpticalRegionClassifier(config.optical, config.optical_classifier).to(device)
    phase = torch.from_numpy(np.asarray(phase_uint8, dtype=np.float32)).to(device)
    phase = phase / 255.0 * (2.0 * torch.pi)
    model.optical.set_phase_override(phase)
    model.optical.calibrate_reference()
    step = count if batch_size is None else max(1, min(int(batch_size), count))
    all_predictions = []
    all_fractions = []
    all_spikes = []
    all_ccd = []
    with torch.no_grad():
        white_result = model(
            torch.ones(
                (
                    1,
                    1,
                    1,
                    config.optical.logical_size,
                    config.optical.logical_size,
                ),
                dtype=torch.float32,
                device=device,
            )
        )
        for batch_start in range(start, stop, step):
            batch_stop = min(batch_start + step, stop)
            spikes = encode_input_batch(
                split.images[batch_start:batch_stop],
                split.indices[batch_start:batch_stop],
                config.encoding.time_steps,
                config.experiment.seed,
                29,
                None,
                config.encoding.minimum_probability,
                config.encoding.input_mode,
                config.encoding.binary_threshold,
            )
            result = model(spikes.to(device))
            batch_count = batch_stop - batch_start
            all_predictions.append(result["scores"].argmax(dim=1).cpu().numpy())
            all_fractions.append(result["region_fraction"].cpu().numpy())
            all_spikes.append(spikes.numpy())
            all_ccd.append(
                result["optical_current"]
                .reshape(
                    batch_count,
                    config.encoding.time_steps,
                    config.optical.readout_size,
                    config.optical.readout_size,
                )
                .cpu()
                .numpy()
            )
            if progress_callback is not None:
                progress_callback(batch_stop - start, count)
    predictions = np.concatenate(all_predictions, axis=0)
    fractions = np.concatenate(all_fractions, axis=0)
    white_fraction = white_result["region_fraction"][0].cpu().numpy()
    return {
        "predictions": predictions,
        "fractions": fractions,
        "white_fraction": white_fraction,
        "spikes": np.concatenate(all_spikes, axis=0),
        "ccd": np.concatenate(all_ccd, axis=0),
        "labels": split.labels[start:stop].numpy(),
        "dataset_indices": split.indices[start:stop].numpy(),
        "split": split,
        "sample_offset": start,
        "test_set_manifest": None if manifest is None else manifest["path"],
        "test_set_id": None if manifest is None else manifest["test_set_id"],
    }


def _simulation_predictions(
    config: Phase1Config,
    phase_uint8: np.ndarray,
    sample_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Compatibility wrapper for existing ten-class hardware workflows."""
    result = simulate_optical_chain(config, phase_uint8, sample_count)
    return (
        result["predictions"],
        result["fractions"],
        result["white_fraction"],
        result["spikes"],
        result["split"],
    )


def _create_hardware_run_directory(root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / "hardware_runs" / ("optical_only_%s" % timestamp)
    suffix = 1
    while path.exists():
        path = root / "hardware_runs" / ("optical_only_%s_%02d" % (timestamp, suffix))
        suffix += 1
    path.mkdir(parents=True)
    return path


def _write_prediction_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_hardware_optical_experiment(
    hardware_config_path: Path,
    experiment_config_path: Path,
    phase_path: Path,
    alignment_path: Path,
    dmd_lut_path: Path,
    sample_count: int = 20,
    frames_per_pattern: int = 3,
    settle_ms: int = 100,
    dry_run: bool = False,
    apply_white_field_gain: bool = False,
    lif_leak: float = 0.875,
    lif_threshold: float = 1.0,
    lif_gain: float = 3.0,
) -> Path:
    """Collect matched simulated/real predictions using one fixed trained phase."""
    hardware_config_path = Path(hardware_config_path).resolve()
    experiment_config_path = Path(experiment_config_path).resolve()
    phase_path = Path(phase_path).resolve()
    alignment_path = Path(alignment_path).resolve()
    dmd_lut_path = Path(dmd_lut_path).resolve()
    hardware_config = load_config(hardware_config_path)
    experiment_config = load_phase1_config(experiment_config_path)
    experiment_config = replace(
        experiment_config,
        optical=replace(
            experiment_config.optical,
            phase_initialization="zero",
            phase_residual_std=0.0,
            phase_parameterization="unbounded",
            propagation_mode="diffractive",
        ),
    )
    dmd_active_px = int(experiment_config.optical.resolved_dmd_active_size)
    slm_active_px = int(experiment_config.optical.resolved_slm_active_size)
    ccd_active_px = int(experiment_config.optical.resolved_ccd_active_size)
    phase_uint8 = np.load(phase_path)
    expected_shape = (
        experiment_config.optical.simulation_size,
        experiment_config.optical.simulation_size,
    )
    if phase_uint8.shape != expected_shape:
        raise ValueError("phase shape %s does not match %s" % (phase_uint8.shape, expected_shape))
    phase_uint8 = np.asarray(phase_uint8, dtype=np.uint8)
    alignment = _load_json(alignment_path)
    if not bool((alignment.get("quality") or {}).get("constraint_satisfied", False)):
        raise RuntimeError("saved alignment did not pass its quality constraints")
    transform = np.asarray(alignment.get("geometry_transform"), dtype=np.float32)
    if transform.shape != (2, 3):
        raise ValueError("alignment configuration has no valid geometry transform")
    dmd_lut = np.load(dmd_lut_path)
    (
        simulation_predictions,
        simulation_fractions,
        simulation_white_fraction,
        spikes,
        split,
    ) = _simulation_predictions(experiment_config, phase_uint8, sample_count)
    count = len(simulation_predictions)
    run_dir = _create_hardware_run_directory(experiment_config.outputs.root)
    write_summary(
        {
            "status": "planned" if not dry_run else "dry_run_complete",
            "hardware_config": hardware_config_path,
            "experiment_config": experiment_config_path,
            "phase_path": phase_path,
            "alignment_path": alignment_path,
            "dmd_lut_path": dmd_lut_path,
            "sample_count": count,
            "simulation_accuracy": float(
                np.mean(simulation_predictions == split.labels[:count].numpy())
            ),
        },
        run_dir / "summary.json",
    )
    if dry_run:
        return run_dir

    from PyQt5 import QtWidgets

    from sdon.hardware import HardwareManager

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["mnist-sdon-deploy"])
    screens = app.screens()
    indexes = (hardware_config.display.dmd_screen, hardware_config.display.slm_screen)
    if min(indexes) < 0 or max(indexes) >= len(screens):
        raise RuntimeError("configured DMD/SLM screen indexes are unavailable")
    manager: Optional[HardwareManager] = None
    rows: List[Dict[str, Any]] = []
    hardware_predictions: List[int] = []
    hardware_raw_predictions: List[int] = []
    lif_predictions: List[int] = []
    saturation_values: List[float] = []
    raw_dir = run_dir / "raw_frames"
    raw_dir.mkdir()
    try:
        manager = HardwareManager(
            hardware_config, screens[indexes[0]], screens[indexes[1]]
        )
        manager.assert_camera_roi_matches(alignment.get("camera_roi") or {})
        manager.set_ccd_orientation(str(alignment.get("ccd_orientation", "identity")))
        slm_offset = alignment.get("slm_offset")
        if not isinstance(slm_offset, list) or len(slm_offset) != 2:
            raise ValueError("alignment configuration has no valid slm_offset")
        manager.update_slm_offset(int(slm_offset[0]), int(slm_offset[1]))
        manager.set_dmd_linearization_lut(dmd_lut)
        manager.update_slm_image(
            prepare_modulator_aperture(phase_uint8, slm_active_px)
        )
        manager.set_slm_phase_enabled(True)
        app.processEvents()
        if not manager.wait_for_slm_update(2.0):
            raise RuntimeError("SLM update timed out")
        time.sleep(max(10, int(settle_ms)) / 1000.0)

        black = np.zeros((32, 32), dtype=np.uint8)
        manager.update_dmd_image(black)
        app.processEvents()
        if not manager.wait_for_dmd_update(2.0):
            raise RuntimeError("DMD dark-frame update timed out")
        time.sleep(max(10, int(settle_ms)) / 1000.0)
        background = capture_median(manager, max(3, int(frames_per_pattern)), (512, 512))
        if background is None:
            raise RuntimeError("CCD returned no dark frame")
        np.save(run_dir / "background.npy", background)

        masks = build_detector_masks(
            experiment_config.optical.readout_size,
            experiment_config.optical_classifier,
        ).numpy()
        white = np.full((32, 32), 255, dtype=np.uint8)
        dmd_effective_px = (
            experiment_config.optical.simulation_size
            * dmd_active_px
            // experiment_config.optical.physical_size
        )
        manager.update_dmd_image(
            prepare_modulator_aperture(
                white,
                dmd_active_px,
                effective_size=dmd_effective_px,
            )
        )
        app.processEvents()
        if not manager.wait_for_dmd_update(2.0):
            raise RuntimeError("DMD white-field update timed out")
        time.sleep(max(10, int(settle_ms)) / 1000.0)
        white_frame = capture_median(
            manager, max(3, int(frames_per_pattern)), (512, 512)
        )
        if white_frame is None:
            raise RuntimeError("CCD returned no white-field frame")
        white_corrected = np.clip(
            white_frame.astype(np.float32) - background.astype(np.float32), 0.0, None
        )
        white_registered = register_ccd_to_simulation(white_corrected, transform, 512)
        white_energy, measured_white_trace = detector_region_trace(
            white_registered[None],
            masks,
            active_aperture_px=ccd_active_px,
        )
        measured_white_fraction = measured_white_trace[0]
        proposed_region_gain = radiometric_region_gain(
            simulation_white_fraction, measured_white_fraction
        )
        region_gain = (
            proposed_region_gain
            if apply_white_field_gain
            else np.ones_like(proposed_region_gain)
        )
        np.savez_compressed(
            run_dir / "white_field_calibration.npz",
            raw=white_frame,
            registered=white_registered,
            measured_energy=white_energy[0],
            measured_fraction=measured_white_fraction,
            simulation_fraction=simulation_white_fraction,
            proposed_region_gain=proposed_region_gain,
            region_gain=region_gain,
        )
        for position in range(count):
            raw_frames: List[np.ndarray] = []
            registered_frames: List[np.ndarray] = []
            sample_saturation: List[float] = []
            for time_index in range(experiment_config.encoding.time_steps):
                dmd_image = np.asarray(spikes[position, time_index, 0] * 255.0, dtype=np.uint8)
                manager.update_dmd_image(
                    prepare_modulator_aperture(
                        dmd_image,
                        dmd_active_px,
                        effective_size=dmd_effective_px,
                    )
                )
                app.processEvents()
                if not manager.wait_for_dmd_update(2.0):
                    raise RuntimeError("DMD update timed out at sample %d" % position)
                time.sleep(max(10, int(settle_ms)) / 1000.0)
                frame = capture_median(
                    manager, max(1, int(frames_per_pattern)), (512, 512)
                )
                if frame is None:
                    raise RuntimeError("CCD returned no frame at sample %d" % position)
                raw = np.asarray(frame)
                maximum = (
                    float(np.iinfo(raw.dtype).max)
                    if np.issubdtype(raw.dtype, np.integer)
                    else 1.0
                )
                sample_saturation.append(float(np.mean(raw >= 0.98 * maximum)))
                corrected = np.clip(
                    raw.astype(np.float32) - background.astype(np.float32), 0.0, None
                )
                registered = register_ccd_to_simulation(corrected, transform, 512)
                raw_frames.append(raw.copy())
                registered_frames.append(registered)
            energy_trace, fraction_trace = detector_region_trace(
                np.stack(registered_frames),
                masks,
                active_aperture_px=ccd_active_px,
            )
            raw_total_energy = energy_trace.sum(axis=0)
            raw_fractions = raw_total_energy / max(float(raw_total_energy.sum()), 1.0e-12)
            calibrated_energy_trace = energy_trace * region_gain[None, :]
            calibrated_fraction_trace = calibrated_energy_trace / np.maximum(
                calibrated_energy_trace.sum(axis=1, keepdims=True), 1.0e-12
            )
            total_energy = calibrated_energy_trace.sum(axis=0)
            fractions = total_energy / max(float(total_energy.sum()), 1.0e-12)
            raw_optical_prediction = int(np.argmax(raw_fractions))
            optical_prediction = int(np.argmax(fractions))
            lif = fixed_lif_readout(
                calibrated_fraction_trace,
                leak=lif_leak,
                threshold=lif_threshold,
                gain=lif_gain,
            )
            lif_prediction = int(np.argmax(lif["scores"]))
            label = int(split.labels[position].item())
            hardware_predictions.append(optical_prediction)
            hardware_raw_predictions.append(raw_optical_prediction)
            lif_predictions.append(lif_prediction)
            saturation = float(max(sample_saturation))
            saturation_values.append(saturation)
            row: Dict[str, Any] = {
                "position": position,
                "dataset_index": int(split.indices[position].item()),
                "label": label,
                "simulation_prediction": int(simulation_predictions[position]),
                "hardware_raw_prediction": raw_optical_prediction,
                "hardware_prediction": optical_prediction,
                "lif_prediction": lif_prediction,
                "hardware_correct": int(optical_prediction == label),
                "lif_correct": int(lif_prediction == label),
                "saturation_fraction": saturation,
            }
            for label_index in range(10):
                row["energy_fraction_%d" % label_index] = float(fractions[label_index])
                row["simulation_fraction_%d" % label_index] = float(
                    simulation_fractions[position, label_index]
                )
                row["lif_score_%d" % label_index] = float(lif["scores"][label_index])
            rows.append(row)
            np.savez_compressed(
                raw_dir / ("sample_%04d.npz" % position),
                raw=np.stack(raw_frames),
                registered=np.stack(registered_frames),
                dmd_spikes=spikes[position],
                region_energy_trace=energy_trace,
                region_fraction_trace=fraction_trace,
                calibrated_region_energy_trace=calibrated_energy_trace,
                calibrated_region_fraction_trace=calibrated_fraction_trace,
            )
            _write_prediction_csv(rows, run_dir / "predictions.csv")
            print(
                "sample %d/%d label=%d sim=%d raw=%d ccd=%d lif=%d saturation=%.6f"
                % (
                    position + 1,
                    count,
                    label,
                    int(simulation_predictions[position]),
                    raw_optical_prediction,
                    optical_prediction,
                    lif_prediction,
                    saturation,
                )
            )
    except Exception as exc:
        write_summary(
            {
                "status": "failed",
                "error": "%s: %s" % (type(exc).__name__, exc),
                "completed_samples": len(rows),
                "camera_roi_expected": alignment.get("camera_roi"),
            },
            run_dir / "failure.json",
        )
        raise
    finally:
        if manager is not None:
            try:
                manager.update_dmd_image(np.zeros((32, 32), dtype=np.uint8), apply_lut=False)
                manager.update_slm_image(np.zeros_like(phase_uint8))
                app.processEvents()
                manager.wait_for_dmd_update(1.0)
                manager.wait_for_slm_update(1.0)
                time.sleep(0.05)
            finally:
                manager.close()
                manager.dmd.close()
                manager.slm.close()
                app.processEvents()

    labels = split.labels[:count].numpy()
    hardware_array = np.asarray(hardware_predictions)
    hardware_raw_array = np.asarray(hardware_raw_predictions)
    lif_array = np.asarray(lif_predictions)
    write_summary(
        {
            "status": "complete",
            "hardware_config": hardware_config_path,
            "experiment_config": experiment_config_path,
            "phase_path": phase_path,
            "alignment_path": alignment_path,
            "dmd_lut_path": dmd_lut_path,
            "camera_roi": alignment.get("camera_roi"),
            "slm_offset": alignment.get("slm_offset"),
            "ccd_orientation": alignment.get("ccd_orientation"),
            "geometry_transform": alignment.get("geometry_transform"),
            "sample_count": count,
            "time_steps": experiment_config.encoding.time_steps,
            "frames_per_pattern": int(frames_per_pattern),
            "settle_ms": int(settle_ms),
            "simulation_accuracy": float(np.mean(simulation_predictions == labels)),
            "hardware_raw_optical_accuracy": float(
                np.mean(hardware_raw_array == labels)
            ),
            "hardware_optical_accuracy": float(np.mean(hardware_array == labels)),
            "fixed_lif_accuracy": float(np.mean(lif_array == labels)),
            "simulation_hardware_agreement": float(
                np.mean(simulation_predictions == hardware_array)
            ),
            "maximum_saturation_fraction": float(max(saturation_values or [0.0])),
            "white_field": {
                "gain_applied": bool(apply_white_field_gain),
                "simulation_fraction": simulation_white_fraction,
                "measured_fraction": measured_white_fraction,
                "proposed_region_gain": proposed_region_gain,
                "region_gain": region_gain,
            },
            "lif": {
                "trainable_parameters": 0,
                "leak": float(lif_leak),
                "threshold": float(lif_threshold),
                "gain": float(lif_gain),
            },
            "run_dir": run_dir,
        },
        run_dir / "summary.json",
    )
    return run_dir
