"""DMD-SLM alignment metrics, evaluation, and coarse-to-fine search."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sdon.simulation import simulate_optical_image
from sdon.hardware.flir_camera import validate_saved_camera_measurement
from sdon.hardware.display_timing import validate_saved_display_mode, capture_display_mode

from .intensity import capture_median
from .patterns import build_alignment_pattern

ORIENTATIONS = ("identity", "flip_horizontal", "flip_vertical", "rotate_180")


def validate_alignment_canvas(manager: Any, configuration: Dict[str, Any]) -> None:
    """Reject a legacy registration or one made on different display canvases."""
    validate_saved_camera_measurement(manager, configuration)
    validate_saved_display_mode(manager, configuration)
    if int(configuration.get("format_version", 0)) < 3 or configuration.get(
        "geometry_model"
    ) != "full_affine_ecc":
        raise ValueError(
            "alignment geometry is legacy (scale/translation only); "
            "run full-affine alignment again"
        )
    display = getattr(getattr(manager, "config", None), "display", None)
    if display is None:
        return
    current_dmd = int(getattr(display, "dmd_modulated_size", display.modulated_size)[0])
    current_slm = int(getattr(display, "slm_modulated_size", display.modulated_size)[0])
    # Alignment files created before canvas metadata was added were all made
    # on the historical 512-pixel workbench.  Never reinterpret an unlabelled
    # legacy file as whatever canvas happens to be active today.
    saved_simulation = int(configuration.get("simulation_canvas_size", 512))
    saved_dmd = int(configuration.get("dmd_canvas_size", 512))
    saved_slm = int(configuration.get("slm_canvas_size", saved_simulation))
    if (saved_dmd, saved_slm, saved_simulation) != (
        current_dmd,
        current_slm,
        current_slm,
    ):
        raise ValueError(
            "alignment canvas changed: saved DMD/SLM/SIM=%d/%d/%d, current=%d/%d/%d; "
            "run alignment again"
            % (
                saved_dmd,
                saved_slm,
                saved_simulation,
                current_dmd,
                current_slm,
                current_slm,
            )
        )


def _structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compute mean SSIM with the standard Gaussian-window formulation."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    mu_first_sq, mu_second_sq = mu_first ** 2, mu_second ** 2
    mu_both = mu_first * mu_second
    sigma_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first_sq
    sigma_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second_sq
    sigma_both = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_both
    numerator = (2.0 * mu_both + c1) * (2.0 * sigma_both + c2)
    denominator = (mu_first_sq + mu_second_sq + c1) * (sigma_first + sigma_second + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def apply_orientation(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "flip_horizontal":
        return cv2.flip(image, 1)
    if mode == "flip_vertical":
        return cv2.flip(image, 0)
    if mode == "rotate_180":
        return cv2.flip(image, -1)
    return image


def normalize_optical_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, [20.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def optical_foreground_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
    normalized = normalize_optical_image(image)
    ys, xs = np.nonzero(normalized >= 0.12)
    if len(xs) < 16:
        return 0, 0, normalized.shape[1] - 1, normalized.shape[0] - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def calibrate_simulation_geometry(
    measured_zero: np.ndarray, simulated_zero: np.ndarray, simulated_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a robust full affine simulation-to-CCD registration.

    A foreground bounding box supplies a stable scale/translation initial
    estimate. ECC then refines rotation, shear and anisotropic scale. The
    refinement is accepted only when it is finite, physically plausible and
    improves normalized optical correlation; otherwise the initial model is
    retained as a safe fallback.
    """
    measured = np.asarray(measured_zero)
    reference_zero = np.asarray(simulated_zero, dtype=np.float32)
    mx0, my0, mx1, my1 = optical_foreground_bbox(measured)
    sx0, sy0, sx1, sy1 = optical_foreground_bbox(reference_zero)
    scale_x = max(1.0, mx1 - mx0) / max(1.0, sx1 - sx0)
    scale_y = max(1.0, my1 - my0) / max(1.0, sy1 - sy0)
    initial_transform = np.array(
        [[scale_x, 0.0, mx0 - scale_x * sx0], [0.0, scale_y, my0 - scale_y * sy0]],
        dtype=np.float32,
    )
    initial_zero = cv2.warpAffine(
        reference_zero,
        initial_transform,
        (measured.shape[1], measured.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    transform = initial_transform
    measured_normalized = normalize_optical_image(measured)
    simulated_normalized = normalize_optical_image(reference_zero)
    initial_correlation = optical_structure_metrics(
        measured_normalized, normalize_optical_image(initial_zero)
    )[0]
    try:
        # findTransformECC returns a template-to-input map intended for use
        # with WARP_INVERSE_MAP. Invert it so saved transforms always follow
        # the explicit simulation-to-CCD convention used by this project.
        template_to_input = cv2.invertAffineTransform(initial_transform)
        _, template_to_input = cv2.findTransformECC(
            cv2.GaussianBlur(measured_normalized, (0, 0), 1.5),
            cv2.GaussianBlur(simulated_normalized, (0, 0), 1.5),
            template_to_input.astype(np.float32),
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1.0e-6),
            None,
            5,
        )
        refined_transform = cv2.invertAffineTransform(template_to_input)
        linear = refined_transform[:, :2].astype(np.float64)
        initial_linear = initial_transform[:, :2].astype(np.float64)
        singular = np.linalg.svd(linear, compute_uv=False)
        initial_singular = np.linalg.svd(initial_linear, compute_uv=False)
        relative_scale = singular / np.maximum(initial_singular, 1.0e-12)
        determinant = float(np.linalg.det(linear))
        refined_zero = cv2.warpAffine(
            reference_zero,
            refined_transform,
            (measured.shape[1], measured.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        refined_correlation = optical_structure_metrics(
            measured_normalized, normalize_optical_image(refined_zero)
        )[0]
        plausible = bool(
            np.all(np.isfinite(refined_transform))
            and determinant > 0.0
            and np.all((relative_scale >= 0.75) & (relative_scale <= 1.25))
        )
        improved = bool(
            np.isfinite(refined_correlation)
            and (
                not np.isfinite(initial_correlation)
                or refined_correlation >= initial_correlation + 1.0e-4
            )
        )
        if plausible and improved:
            transform = refined_transform.astype(np.float32)
    except cv2.error:
        transform = initial_transform

    warped = cv2.warpAffine(
        np.asarray(simulated_image, dtype=np.float32),
        transform,
        (measured.shape[1], measured.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, transform


def affine_geometry_metrics(transform: Any) -> Dict[str, float]:
    """Return interpretable rotation, scale and shear values for one affine map."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return {
            "geometry_rotation_deg": float("nan"),
            "geometry_scale_x": float("nan"),
            "geometry_scale_y": float("nan"),
            "geometry_shear_cosine": float("nan"),
        }
    linear = matrix[:, :2]
    column_x, column_y = linear[:, 0], linear[:, 1]
    scale_x = float(np.linalg.norm(column_x))
    scale_y = float(np.linalg.norm(column_y))
    shear = float(
        np.dot(column_x, column_y) / max(scale_x * scale_y, 1.0e-12)
    )
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return {
        "geometry_rotation_deg": angle,
        "geometry_scale_x": scale_x,
        "geometry_scale_y": scale_y,
        "geometry_shear_cosine": shear,
    }


def intensity_centroid(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    weights = np.clip(array - float(np.percentile(array, 20.0)), 0.0, None)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.array([np.nan, np.nan], dtype=np.float64)
    y_grid, x_grid = np.indices(weights.shape)
    return np.array(
        [float((weights * x_grid).sum() / total), float((weights * y_grid).sum() / total)]
    )


def optical_structure_metrics(
    measured: np.ndarray, reference: np.ndarray
) -> Tuple[float, float, float, float]:
    measured_n = normalize_optical_image(measured)
    reference_n = normalize_optical_image(reference)
    mask = cv2.dilate((reference_n >= 0.08).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(
        bool
    )
    if int(mask.sum()) < 16:
        return float("nan"), float("nan"), 0.0, float("nan")

    measured_values, reference_values = measured_n[mask], reference_n[mask]
    measured_zero = measured_values - float(measured_values.mean())
    reference_zero = reference_values - float(reference_values.mean())
    denominator = float(np.linalg.norm(measured_zero) * np.linalg.norm(reference_zero))
    correlation = (
        float(np.dot(measured_zero, reference_zero) / denominator)
        if denominator > 1e-8
        else float("nan")
    )

    x0, y0, x1, y1 = optical_foreground_bbox(reference_n)
    margin_x, margin_y = max(8, (x1 - x0 + 1) // 5), max(8, (y1 - y0 + 1) // 5)
    x0, x1 = max(0, x0 - margin_x), min(reference_n.shape[1] - 1, x1 + margin_x)
    y0, y1 = max(0, y0 - margin_y), min(reference_n.shape[0] - 1, y1 + margin_y)
    measured_crop = measured_n[y0 : y1 + 1, x0 : x1 + 1]
    reference_crop = reference_n[y0 : y1 + 1, x0 : x1 + 1]
    ssim_score = _structural_similarity(measured_crop, reference_crop)

    measured_edges = cv2.Canny((measured_n * 255).astype(np.uint8), 40, 100) > 0
    reference_edges = cv2.Canny((reference_n * 255).astype(np.uint8), 40, 100) > 0
    kernel = np.ones((5, 5), np.uint8)
    measured_dilated = cv2.dilate(measured_edges.astype(np.uint8), kernel).astype(bool)
    reference_dilated = cv2.dilate(reference_edges.astype(np.uint8), kernel).astype(bool)
    precision = float(np.mean(reference_dilated[measured_edges])) if measured_edges.any() else 0.0
    recall = float(np.mean(measured_dilated[reference_edges])) if reference_edges.any() else 0.0
    edge_f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    centroid_error = float(
        np.linalg.norm(intensity_centroid(measured_n) - intensity_centroid(reference_n))
    )
    return correlation, ssim_score, edge_f1, centroid_error


def publish_sim_ccd_comparison(
    manager: Any,
    ccd_image: np.ndarray,
    simulated_image: np.ndarray,
    orientation_mode: str,
    selected_orientation: Optional[str] = None,
    geometry_transform: Optional[Any] = None,
    auto_calibrate_geometry: bool = True,
    message_prefix: str = "SIM_VS_CCD",
) -> Dict[str, Any]:
    """Build and publish Qt-safe CCD, calibrated SIM, and absolute error previews."""
    if selected_orientation is not None:
        candidates = (selected_orientation,)
    elif orientation_mode == "auto":
        candidates = ORIENTATIONS
    else:
        candidates = (orientation_mode,)

    evaluated = []
    for orientation in candidates:
        measured = (
            apply_orientation(ccd_image, orientation)
            if orientation_mode == "auto"
            else np.asarray(ccd_image)
        )
        if geometry_transform is None and auto_calibrate_geometry:
            calibrated, transform = calibrate_simulation_geometry(
                measured, simulated_image, simulated_image
            )
        elif geometry_transform is None:
            transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            calibrated = cv2.resize(
                np.asarray(simulated_image, dtype=np.float32),
                (measured.shape[1], measured.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            transform = np.asarray(geometry_transform, dtype=np.float32)
            calibrated = cv2.warpAffine(
                np.asarray(simulated_image, dtype=np.float32),
                transform,
                (measured.shape[1], measured.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        correlation = optical_structure_metrics(measured, calibrated)[0]
        evaluated.append((correlation, orientation, measured, calibrated, transform))

    finite = [item for item in evaluated if np.isfinite(item[0])]
    correlation, orientation, measured, calibrated, transform = max(
        finite or evaluated, key=lambda item: item[0] if np.isfinite(item[0]) else -1e9
    )
    measured_preview = (normalize_optical_image(measured) * 255).astype(np.uint8)
    simulated_preview = (normalize_optical_image(calibrated) * 255).astype(np.uint8)
    difference_preview = cv2.absdiff(simulated_preview, measured_preview)
    mae = float(np.mean(difference_preview) / 255.0)
    manager.publish_preview(
        ccd=measured_preview,
        sim=simulated_preview,
        difference=difference_preview,
        message=(
            "%s | normalized MAE=%.5f | correlation=%.5f | orientation=%s"
            % (message_prefix, mae, correlation, orientation)
        ),
    )
    return {
        "ccd": measured_preview,
        "sim": simulated_preview,
        "difference": difference_preview,
        "mae": mae,
        "correlation": float(correlation),
        "orientation": orientation,
        "geometry_transform": np.asarray(transform).tolist(),
    }


def phase_registration_shift(
    image_on: np.ndarray, image_off: np.ndarray
) -> Tuple[float, float, float, float]:
    prepared = []
    for image in (image_off, image_on):
        array = np.asarray(image, dtype=np.float32)
        low, high = np.percentile(array, [1.0, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return float("nan"), float("nan"), float("nan"), 0.0
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
        gradient = cv2.magnitude(
            cv2.Sobel(array, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(array, cv2.CV_32F, 0, 1, ksize=3),
        )
        prepared.append(gradient - float(gradient.mean()))
    window = cv2.createHanningWindow((prepared[0].shape[1], prepared[0].shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(prepared[0], prepared[1], window)
    return float(dx), float(dy), float(np.hypot(dx, dy)), float(response)


def spatial_modulation_response(image_on: np.ndarray, image_off: np.ndarray) -> float:
    distributions = []
    for image in (image_on, image_off):
        array = np.asarray(image, dtype=np.float64)
        array = np.clip(array - float(np.percentile(array, 20.0)), 0.0, None)
        total = float(array.sum())
        if total <= 1e-12:
            return 0.0
        distributions.append(array / total)
    return float(0.5 * np.abs(distributions[0] - distributions[1]).sum())


def phase_toggle_metrics(
    ccd_on: np.ndarray,
    ccd_off: np.ndarray,
    sim_on: np.ndarray,
    sim_off: np.ndarray,
    orientation_mode: str,
) -> Dict[str, Any]:
    """Score measured ON/OFF frames against the simulated lens structure."""
    candidates = ORIENTATIONS if orientation_mode == "auto" else (orientation_mode,)
    evaluated = []
    for orientation in candidates:
        measured_on = (
            apply_orientation(ccd_on, orientation) if orientation_mode == "auto" else ccd_on
        )
        measured_off = (
            apply_orientation(ccd_off, orientation) if orientation_mode == "auto" else ccd_off
        )
        calibrated_off, transform = calibrate_simulation_geometry(measured_off, sim_off, sim_off)
        calibrated_on = cv2.warpAffine(
            np.asarray(sim_on, dtype=np.float32),
            transform,
            (measured_off.shape[1], measured_off.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        baseline = optical_structure_metrics(measured_off, calibrated_off)[0]
        evaluated.append(
            (
                baseline,
                orientation,
                measured_on,
                measured_off,
                calibrated_on,
                calibrated_off,
                transform,
            )
        )
    finite = [item for item in evaluated if np.isfinite(item[0])]
    selected = max(finite or evaluated, key=lambda item: item[0] if np.isfinite(item[0]) else -1e9)
    (
        corr_off,
        orientation,
        measured_on,
        measured_off,
        calibrated_on,
        calibrated_off,
        transform,
    ) = selected

    corr_on, ssim_score, edge_f1, target_error = optical_structure_metrics(
        measured_on, calibrated_on
    )
    measured_difference = measured_on.astype(np.float32) - measured_off.astype(np.float32)
    simulated_difference = calibrated_on.astype(np.float32) - calibrated_off.astype(np.float32)
    corr_difference = optical_structure_metrics(measured_difference, simulated_difference)[0]
    if not np.isfinite(corr_difference):
        corr_difference = 0.0

    measured_vector = intensity_centroid(measured_on) - intensity_centroid(measured_off)
    simulated_vector = intensity_centroid(calibrated_on) - intensity_centroid(calibrated_off)
    measured_shift = (
        float(np.linalg.norm(measured_vector))
        if np.all(np.isfinite(measured_vector))
        else float("nan")
    )
    simulated_shift = (
        float(np.linalg.norm(simulated_vector))
        if np.all(np.isfinite(simulated_vector))
        else float("nan")
    )
    shift_error = (
        float(np.linalg.norm(measured_vector - simulated_vector))
        if np.isfinite(measured_shift) and np.isfinite(simulated_shift)
        else float("nan")
    )
    modulation_contrast = float(
        np.linalg.norm(measured_difference)
        / max(float(np.linalg.norm(measured_off.astype(np.float32))), 1e-8)
    )
    spatial_response = spatial_modulation_response(measured_on, measured_off)
    (
        registration_dx,
        registration_dy,
        registration_shift,
        registration_response,
    ) = phase_registration_shift(measured_on, measured_off)
    sensor_max = (
        float(np.iinfo(measured_on.dtype).max)
        if np.issubdtype(measured_on.dtype, np.integer)
        else float(np.max(measured_on))
    )
    p999 = float(np.percentile(measured_on, 99.9))
    saturation = float(np.mean(measured_on >= sensor_max)) if sensor_max > 0 else 0.0
    # A handful of clipped pixels is expected when Mono8 uses its full range.
    # Reject material clipping, not 10-12 hot pixels in a 600x600 image.
    valid_exposure = p999 < sensor_max * 0.995 and saturation <= 1e-3
    finite_corr = corr_on if np.isfinite(corr_on) else -1.0
    finite_ssim = ssim_score if np.isfinite(ssim_score) else 0.0
    centroid_penalty = 0.4 if not np.isfinite(target_error) else min(0.4, 0.02 * target_error)
    score = 0.50 * finite_corr + 0.30 * finite_ssim + 0.20 * edge_f1 - centroid_penalty
    return {
        "score": float(score),
        "simulation_score": float(score),
        "correlation_on": float(corr_on),
        "correlation_off": float(corr_off),
        "correlation_difference": float(corr_difference),
        "structure_ssim": float(ssim_score),
        "edge_f1": float(edge_f1),
        "target_centroid_error_px": float(target_error),
        "measured_shift_px": measured_shift,
        "simulated_shift_px": simulated_shift,
        "shift_error_px": shift_error,
        "modulation_contrast": modulation_contrast,
        "spatial_modulation_response": spatial_response,
        "registration_dx_px": registration_dx,
        "registration_dy_px": registration_dy,
        "registration_shift_px": registration_shift,
        "registration_response": registration_response,
        "p99_9": p999,
        "saturation_fraction": saturation,
        "valid_exposure": bool(valid_exposure),
        "simulation_canvas_size": int(sim_on.shape[0]),
        "orientation": orientation,
        "geometry_transform": transform.tolist(),
        **affine_geometry_metrics(transform),
    }


def _capture_on_off(
    manager: Any, settle_seconds: float, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not manager.wait_for_slm_update(1.0):
        raise RuntimeError("等待 SLM 显示更新超时")
    time.sleep(settle_seconds)
    size = manager.config.display.modulated_size
    on = capture_median(manager, samples, size)
    manager.set_slm_phase_enabled(False)
    try:
        if not manager.wait_for_slm_update(1.0):
            raise RuntimeError("等待 SLM 空白相位更新超时")
        time.sleep(settle_seconds)
        off = capture_median(manager, samples, size)
    finally:
        manager.set_slm_phase_enabled(True)
        manager.wait_for_slm_update(1.0)
        time.sleep(settle_seconds)
    if on is None or off is None:
        raise RuntimeError("CCD 未返回有效图像")
    return on, off


def _is_feasible(item: Dict[str, Any], request: Dict[str, Any]) -> bool:
    # A small ON/OFF displacement alone says nothing about whether the target
    # is complete or the simulation-to-camera geometry is credible. In
    # particular, a clipped target can be fitted by a wrong mirror plus shear.
    baseline = float(item.get("correlation_off", float("nan")))
    shear = float(item.get("geometry_shear_cosine", float("nan")))
    return bool(
        np.isfinite(item["score"])
        and item["valid_exposure"]
        and np.isfinite(baseline)
        and baseline >= float(request.get("minimum_baseline_correlation", 0.5))
        and np.isfinite(shear)
        and abs(shear) <= float(request.get("maximum_geometry_shear", 0.15))
        and np.isfinite(item["registration_shift_px"])
        and item["registration_shift_px"] <= request["max_registration_shift"]
        and item["spatial_modulation_response"] >= request["minimum_modulation"]
        and ("center_modulation" not in item or bool(item["center_modulation"].get("passed", False)))
    )


def _verify_center_modulation(manager: Any, request: Dict[str, Any],
                              phase: np.ndarray, pattern: np.ndarray,
                              orientation: str) -> Dict[str, Any]:
    """Reject a stationary unmodulated centre before promoting an alignment.

    Run only for the final candidate, not every scan point. A small phase
    patch overlapping a corner can otherwise pass full-image registration.
    """
    from .slm_check import build_slm_check_patterns, analyze_slm_check
    frames: Dict[str, np.ndarray] = {}
    previous_orientation = manager.get_ccd_orientation()
    previous_enabled = manager.is_slm_phase_enabled()
    config = manager.config
    try:
        manager.update_alignment_status(message="最终复核：中心方块 ±X/±Y 光栅调制")
        manager.set_ccd_orientation(orientation)
        manager.set_slm_phase_enabled(True)
        aperture = min(128, int(pattern.shape[0]))
        manager.update_dmd_image(build_alignment_pattern(
            "center_rectangle", size=int(pattern.shape[0]), rectangle_size=aperture), apply_lut=False)
        if not manager.wait_for_dmd_update(2.0):
            raise RuntimeError("centre modulation DMD update timed out")
        phases = build_slm_check_patterns(phase.shape, 16,
            config.optics.alignment_lens_focal_length_m,
            config.optics.wavelength_m, config.optics.pixel_pitch_m)
        for name, value in phases.items():
            if manager.is_alignment_cancelled():
                raise InterruptedError("alignment cancelled")
            manager.update_slm_image(value, apply_lut=False)
            if not manager.wait_for_slm_update(2.0):
                raise RuntimeError("centre modulation SLM update timed out")
            time.sleep(max(.2, request["settle_ms"] / 1000.0))
            frame = capture_median(manager, config.timing.samples, config.display.modulated_size)
            if frame is None:
                raise RuntimeError("centre modulation capture failed")
            frames[name] = frame
        analysis = analyze_slm_check(frames)
        folder = config.paths.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        evidence = folder / ("alignment_center_check_" + time.strftime("%Y%m%d_%H%M%S") + ".npz")
        np.savez_compressed(evidence, **frames)
        return {**analysis, "evidence_path": str(evidence.resolve()),
                "aperture_px": aperture, "period_px": 16}
    except InterruptedError:
        raise
    except Exception as error:
        return {"passed": False, "error": str(error)}
    finally:
        manager.set_ccd_orientation(previous_orientation)
        manager.update_dmd_image(pattern)
        manager.update_slm_image(phase, apply_lut=False)
        manager.set_slm_phase_enabled(previous_enabled)
        manager.wait_for_dmd_update(2.0)
        manager.wait_for_slm_update(2.0)


def _status_values(item: Dict[str, Any], constraint_satisfied: bool) -> Dict[str, Any]:
    return {
        "best_score": item["score"],
        "center_modulation": item.get("center_modulation"),
        "best_offset": (item["x"], item["y"]),
        "best_orientation": item["orientation"],
        "best_shift_error": item["shift_error_px"],
        "best_modulation_contrast": item["modulation_contrast"],
        "best_spatial_response": item["spatial_modulation_response"],
        "best_structure_ssim": item["structure_ssim"],
        "best_correlation_on": item["correlation_on"],
        "best_correlation_off": item["correlation_off"],
        "best_edge_f1": item["edge_f1"],
        "best_target_centroid_error": item["target_centroid_error_px"],
        "best_geometry_transform": item["geometry_transform"],
        "best_geometry_rotation_deg": item.get("geometry_rotation_deg"),
        "best_geometry_scale_x": item.get("geometry_scale_x"),
        "best_geometry_scale_y": item.get("geometry_scale_y"),
        "best_geometry_shear_cosine": item.get("geometry_shear_cosine"),
        "simulation_canvas_size": item.get("simulation_canvas_size"),
        "best_position_shift": item["measured_shift_px"],
        "best_registration_shift": item["registration_shift_px"],
        "best_registration_response": item["registration_response"],
        "best_valid_exposure": item["valid_exposure"],
        "best_p99_9": item["p99_9"],
        "best_saturation_fraction": item["saturation_fraction"],
        "constraint_satisfied": constraint_satisfied,
    }


def _save_evaluation(
    output_dir: Path,
    item: Dict[str, Any],
    on: np.ndarray,
    off: np.ndarray,
    phase: np.ndarray,
    auto_orientation: bool,
    preview: Dict[str, Any],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = output_dir / ("slm_alignment_score_" + timestamp)
    if auto_orientation:
        on, off = (
            apply_orientation(on, item["orientation"]),
            apply_orientation(off, item["orientation"]),
        )
    cv2.imwrite(str(prefix) + "_on.png", on)
    cv2.imwrite(str(prefix) + "_off.png", off)
    cv2.imwrite(str(prefix) + "_lens_phase.png", phase)
    cv2.imwrite(str(prefix) + "_sim_calibrated.png", preview["sim"])
    cv2.imwrite(str(prefix) + "_sim_vs_ccd.png", preview["difference"])
    csv_path = str(prefix) + ".csv"
    _write_results(Path(csv_path), [item])
    return csv_path


RESULT_FIELDS = (
    "stage",
    "x",
    "y",
    "score",
    "simulation_score",
    "correlation_on",
    "correlation_off",
    "correlation_difference",
    "measured_shift_px",
    "simulated_shift_px",
    "shift_error_px",
    "structure_ssim",
    "edge_f1",
    "target_centroid_error_px",
    "registration_dx_px",
    "registration_dy_px",
    "registration_shift_px",
    "registration_response",
    "modulation_contrast",
    "spatial_modulation_response",
    "p99_9",
    "saturation_fraction",
    "valid_exposure",
    "constraint_satisfied",
    "orientation",
    "geometry_rotation_deg",
    "geometry_scale_x",
    "geometry_scale_y",
    "geometry_shear_cosine",
    "sim_ccd_mae",
)


def _write_results(path: Path, results: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            writer.writerow(item)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def save_alignment_configuration(
    manager: Any,
    item: Dict[str, Any],
    request: Dict[str, Any],
    operation: str,
    result_path: Optional[Any] = None,
) -> str:
    """Save a reusable alignment baseline and refresh the stable latest file."""
    output_dir = manager.config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pattern_mode, rectangle_size = manager.get_alignment_pattern()
    pattern_configuration = {
        "mode": pattern_mode,
        "rectangle_size": int(rectangle_size),
    }
    display = getattr(getattr(manager, "config", None), "display", None)
    legacy_size = int(display.modulated_size[0]) if display is not None else 512
    dmd_canvas_size = (
        int(getattr(display, "dmd_modulated_size", display.modulated_size)[0])
        if display is not None
        else legacy_size
    )
    slm_canvas_size = (
        int(getattr(display, "slm_modulated_size", display.modulated_size)[0])
        if display is not None
        else legacy_size
    )
    simulation_canvas_size = int(
        item.get("simulation_canvas_size", slm_canvas_size)
    )
    configuration = {
        "format_version": 3,
        "geometry_model": "full_affine_ecc",
        "timestamp": timestamp,
        "operation": operation,
        "dmd_canvas_size": dmd_canvas_size,
        "slm_canvas_size": slm_canvas_size,
        "simulation_canvas_size": simulation_canvas_size,
        "camera_roi": manager.get_camera_roi(),
        "display_mode": capture_display_mode(manager),
        "camera": manager.camera.get_camera_parameters() if hasattr(manager,"camera") else {},
        "slm_offset": [int(item["x"]), int(item["y"])],
        "ccd_orientation": item.get("orientation", "identity"),
        "geometry_transform": item.get("geometry_transform"),
        "dmd_pattern": pattern_configuration,
        "quality": {
            "score": item.get("score"),
            "correlation_off": item.get("correlation_off"),
            "center_modulation": item.get("center_modulation"),
            "sim_ccd_mae": item.get("sim_ccd_mae"),
            "structure_ssim": item.get("structure_ssim"),
            "edge_f1": item.get("edge_f1"),
            "valid_exposure": item.get("valid_exposure"),
            "p99_9": item.get("p99_9"),
            "saturation_fraction": item.get("saturation_fraction"),
            "registration_shift_px": item.get("registration_shift_px"),
            "centroid_shift_px": item.get("measured_shift_px"),
            "spatial_modulation_response": item.get("spatial_modulation_response"),
            "geometry_rotation_deg": item.get("geometry_rotation_deg"),
            "geometry_scale_x": item.get("geometry_scale_x"),
            "geometry_scale_y": item.get("geometry_scale_y"),
            "geometry_shear_cosine": item.get("geometry_shear_cosine"),
            "constraint_satisfied": bool(item.get("constraint_satisfied", False)),
        },
        "criteria": {
            "max_registration_shift_px": float(
                request.get("max_registration_shift", 2.0)
            ),
            "minimum_spatial_modulation_response": float(
                request.get("minimum_modulation", 0.02)
            ),
            "maximum_saturation_fraction": 1e-3,
            "maximum_p99_9_fraction": 0.995,
            "minimum_baseline_correlation": float(request.get("minimum_baseline_correlation", 0.5)),
            "maximum_geometry_shear": float(request.get("maximum_geometry_shear", 0.15)),
            "requires_center_blaze_verification": True,
        },
        "scan": {
            "fine_radius": int(request.get("fine_radius", 5)),
            "fine_step": int(request.get("fine_step", 1)),
            "settle_ms": int(request.get("settle_ms", 100)),
        },
        "result_path": None if result_path is None else str(result_path),
    }
    path = output_dir / ("alignment_config_%s_%s.json" % (operation, timestamp))
    latest_path = output_dir / "alignment_config_latest.json"
    targets = [path]
    if configuration["quality"]["constraint_satisfied"]:
        targets.append(latest_path)
    for target in targets:
        with target.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(configuration), handle, ensure_ascii=False, indent=2, allow_nan=False
            )
    return str(path)


def _activate_saved_alignment(manager: Any, config_path: str) -> None:
    """Promote a passing attempt to the stable alignment used by the GUI gates."""
    if not hasattr(manager, "set_active_alignment_configuration"):
        return
    with Path(config_path).open("r", encoding="utf-8") as handle:
        configuration = json.load(handle)
    manager.set_active_alignment_configuration(config_path, configuration)


def _restore_active_alignment(manager: Any, fallback: Tuple[int, int]) -> bool:
    """Restore a prior valid configuration, or at least the operation start offset."""
    restored = False
    if hasattr(manager, "restore_active_alignment"):
        restored = bool(manager.restore_active_alignment())
    if not restored:
        manager.update_slm_offset(*fallback)
    return restored


def run_alignment_evaluation(
    manager: Any, request: Dict[str, Any], phase_uint8: np.ndarray
) -> None:
    """Evaluate the current SLM offset without moving it."""
    x_value, y_value = request["center_x"], request["center_y"]
    try:
        mode, rectangle_size = manager.get_alignment_pattern()
        dmd_size = int(manager.config.display.dmd_modulated_size[0])
        pattern = build_alignment_pattern(
            mode, size=dmd_size, rectangle_size=rectangle_size
        )
        manager.update_slm_offset(x_value, y_value)
        manager.update_dmd_image(pattern)
        manager.update_slm_image(phase_uint8, apply_lut=False)
        manager.wait_for_dmd_update(1.0)
        sim_on = simulate_optical_image(
            pattern,
            phase_uint8,
            manager.config.optics,
            dmd_active_size=dmd_size,
        )
        sim_off = simulate_optical_image(
            pattern,
            np.zeros_like(phase_uint8),
            manager.config.optics,
            dmd_active_size=dmd_size,
        )
        on, off = _capture_on_off(
            manager, request["settle_ms"] / 1000.0, manager.config.timing.samples
        )
        metrics = phase_toggle_metrics(on, off, sim_on, sim_off, request["orientation_mode"])
        preview = publish_sim_ccd_comparison(
            manager,
            on,
            sim_on,
            request["orientation_mode"],
            selected_orientation=metrics["orientation"],
            geometry_transform=metrics["geometry_transform"],
            message_prefix="当前位置 SIM_VS_CCD",
        )
        item = {"stage": "evaluate", "x": x_value, "y": y_value, **metrics}
        item["sim_ccd_mae"] = preview["mae"]
        feasible = _is_feasible(item, request)
        if feasible:
            item["center_modulation"] = _verify_center_modulation(
                manager, request, phase_uint8, pattern, metrics["orientation"])
            feasible = _is_feasible(item, request)
        item["constraint_satisfied"] = feasible
        if request["orientation_mode"] == "auto":
            manager.set_ccd_orientation(metrics["orientation"])
        path = _save_evaluation(
            manager.config.paths.output_dir,
            item,
            on,
            off,
            phase_uint8,
            request["orientation_mode"] == "auto",
            preview,
        )
        config_path = save_alignment_configuration(
            manager, item, request, "evaluate", result_path=path
        )
        active_restored = False
        if feasible:
            _activate_saved_alignment(manager, config_path)
        else:
            active_restored = _restore_active_alignment(manager, (x_value, y_value))
        manager.update_alignment_status(
            state="complete",
            operation="evaluate",
            current=1,
            total=1,
            message=("当前位置评分完成；结果: %s；配置: %s" % (path, config_path))
                    + ("；中心光栅调制未通过，不能解锁" if item.get("center_modulation", {}).get("passed") is False else ""),
            config_path=config_path,
            active_restored=active_restored,
            **_status_values(item, feasible),
        )
    except Exception as exc:
        active_restored = _restore_active_alignment(manager, (x_value, y_value))
        manager.update_alignment_status(
            state="error", operation="evaluate", message="当前位置评分失败: %s" % exc
        )
    finally:
        manager.update_slm_image(phase_uint8, apply_lut=False)


def _scan_axis(center: int, radius: int, step: int, lower: int, upper: int) -> List[int]:
    start, stop = max(lower, center - radius), min(upper, center + radius)
    values = list(range(start, stop + 1, max(1, step)))
    values.extend([max(lower, min(upper, center)), stop])
    return sorted(set(values))


def _run_alignment_search(
    manager: Any, request: Dict[str, Any], phase_uint8: np.ndarray, fine_only: bool,
) -> None:
    """Run an independent fine scan or the complete coarse-to-fine search."""
    start_offset = (request["center_x"], request["center_y"])
    results: List[Dict[str, Any]] = []
    best_frames: Dict[str, Optional[np.ndarray]] = {"on": None, "off": None}
    try:
        mode, rectangle_size = manager.get_alignment_pattern()
        dmd_size = int(manager.config.display.dmd_modulated_size[0])
        pattern = build_alignment_pattern(
            mode, size=dmd_size, rectangle_size=rectangle_size
        )
        manager.update_slm_image(phase_uint8, apply_lut=False)
        manager.update_dmd_image(pattern)
        manager.wait_for_dmd_update(1.0)
        time.sleep(max(manager.config.timing.slm_min_interval_s, request["settle_ms"] / 1000.0))
        sim_on = simulate_optical_image(
            pattern,
            phase_uint8,
            manager.config.optics,
            dmd_active_size=dmd_size,
        )
        sim_off = simulate_optical_image(
            pattern,
            np.zeros_like(phase_uint8),
            manager.config.optics,
            dmd_active_size=dmd_size,
        )
        max_x, max_y = manager.slm.offset_limits()

        def select_best(allow_fallback: bool = True) -> Tuple[Optional[Dict[str, Any]], bool]:
            feasible = [item for item in results if _is_feasible(item, request)]
            if feasible:
                return max(feasible, key=lambda item: item["score"]), True
            if not allow_fallback:
                return None, False
            fallback = [
                item
                for item in results
                if np.isfinite(item["score"])
                and item["valid_exposure"]
                and np.isfinite(item["registration_shift_px"])
                and item["spatial_modulation_response"] >= request["minimum_modulation"]
            ]
            if not fallback:
                return None, False
            return (
                min(
                    fallback,
                    key=lambda item: (item["registration_shift_px"], -item["score"]),
                ),
                False,
            )

        def scan(center_x: int, center_y: int, radius: int, step: int, stage: str) -> bool:
            xs = _scan_axis(center_x, radius, step, -max_x, max_x)
            ys = _scan_axis(center_y, radius, step, -max_y, max_y)
            points = [
                (x, y) for row, y in enumerate(ys) for x in (xs if row % 2 == 0 else reversed(xs))
            ]
            manager.update_alignment_status(total=len(results) + len(points), message=stage + " 扫描")
            for x, y in points:
                if manager.is_alignment_cancelled():
                    return False
                manager.update_slm_offset(x, y)
                on, off = _capture_on_off(
                    manager, request["settle_ms"] / 1000.0, manager.config.timing.samples
                )
                metrics = phase_toggle_metrics(
                    on, off, sim_on, sim_off, request["orientation_mode"]
                )
                preview = publish_sim_ccd_comparison(
                    manager,
                    on,
                    sim_on,
                    request["orientation_mode"],
                    selected_orientation=metrics["orientation"],
                    geometry_transform=metrics["geometry_transform"],
                    message_prefix="%s (%d,%d) SIM_VS_CCD" % (stage, x, y),
                )
                item = {"stage": stage, "x": x, "y": y, **metrics}
                item["sim_ccd_mae"] = preview["mae"]
                item["constraint_satisfied"] = _is_feasible(item, request)
                results.append(item)
                best, satisfied = select_best()
                if best is item:
                    best_frames["on"], best_frames["off"] = on.copy(), off.copy()
                values = {} if best is None else _status_values(best, satisfied)
                manager.update_alignment_status(
                    current=len(results), message="%s 扫描 (%d,%d)" % (stage, x, y), **values
                )
            return True

        if fine_only:
            fine_center = {"x": start_offset[0], "y": start_offset[1]}
        else:
            if not scan(*start_offset, request["radius"], request["coarse_step"], "coarse"):
                raise InterruptedError("扫描已停止")
            fine_center, _ = select_best()
            if fine_center is None:
                raise RuntimeError("粗扫没有获得有效 CCD 相关性")
        if not scan(
            fine_center["x"],
            fine_center["y"],
            request["fine_radius"],
            request["fine_step"],
            "fine",
        ):
            raise InterruptedError("扫描已停止")

        best, satisfied = select_best(allow_fallback=False)
        if best is None:
            best, _ = select_best(allow_fallback=True)
        if best is None:
            raise RuntimeError("扫描没有得到可用候选点")

        manager.update_slm_offset(best["x"], best["y"])
        confirm_on, confirm_off = _capture_on_off(
            manager, request["settle_ms"] / 1000.0, manager.config.timing.samples
        )
        confirmed = {
            "stage": "confirm",
            "x": best["x"],
            "y": best["y"],
            **phase_toggle_metrics(
                confirm_on, confirm_off, sim_on, sim_off, request["orientation_mode"]
            ),
        }
        confirmed_preview = publish_sim_ccd_comparison(
            manager,
            confirm_on,
            sim_on,
            request["orientation_mode"],
            selected_orientation=confirmed["orientation"],
            geometry_transform=confirmed["geometry_transform"],
            message_prefix="复测 SIM_VS_CCD",
        )
        confirmed["sim_ccd_mae"] = confirmed_preview["mae"]
        satisfied = _is_feasible(confirmed, request)
        if satisfied:
            confirmed["center_modulation"] = _verify_center_modulation(
                manager, request, phase_uint8, pattern, confirmed["orientation"])
            satisfied = _is_feasible(confirmed, request)
        confirmed["constraint_satisfied"] = satisfied
        results.append(confirmed)
        best_frames["on"], best_frames["off"] = confirm_on.copy(), confirm_off.copy()

        if satisfied:
            manager.update_slm_offset(confirmed["x"], confirmed["y"])
            if request["orientation_mode"] == "auto":
                manager.set_ccd_orientation(confirmed["orientation"])
            active_restored = False
        else:
            active_restored = _restore_active_alignment(manager, start_offset)

        output_dir = manager.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_prefix = "slm_fine_alignment" if fine_only else "slm_auto_alignment"
        csv_path = output_dir / (output_prefix + "_" + timestamp + ".csv")
        _write_results(csv_path, results)
        cv2.imwrite(str(output_dir / (output_prefix + "_pattern_" + timestamp + ".png")), pattern)
        cv2.imwrite(
            str(output_dir / (output_prefix + "_lens_phase_" + timestamp + ".png")), phase_uint8
        )
        if best_frames["on"] is not None:
            on, off = best_frames["on"], best_frames["off"]
            if request["orientation_mode"] == "auto":
                on = apply_orientation(on, confirmed["orientation"])
                off = apply_orientation(off, confirmed["orientation"])
            cv2.imwrite(str(output_dir / (output_prefix + "_best_on_" + timestamp + ".png")), on)
            cv2.imwrite(str(output_dir / (output_prefix + "_best_off_" + timestamp + ".png")), off)
            cv2.imwrite(
                str(output_dir / (output_prefix + "_sim_calibrated_" + timestamp + ".png")),
                confirmed_preview["sim"],
            )
            cv2.imwrite(
                str(output_dir / (output_prefix + "_sim_vs_ccd_" + timestamp + ".png")),
                confirmed_preview["difference"],
            )

        operation = "fine" if fine_only else "auto"
        config_path = save_alignment_configuration(
            manager, confirmed, request, operation, result_path=csv_path
        )
        if satisfied:
            _activate_saved_alignment(manager, config_path)
        display_name = "独立细扫" if fine_only else "自动对齐"
        message = "%s完成；结果: %s；配置: %s" % (display_name, csv_path, config_path)
        if not satisfied:
            message += "；复测未通过约束，已恢复扫描起点"
            if confirmed.get("center_modulation", {}).get("passed") is False:
                message += "；中心光栅调制未通过，不能解锁"
        manager.update_alignment_status(
            state="complete" if satisfied else "error",
            current=len(results),
            total=len(results),
            message=message,
            config_path=config_path,
            active_restored=active_restored,
            **_status_values(confirmed, satisfied),
        )
    except InterruptedError as exc:
        active_restored = _restore_active_alignment(manager, start_offset)
        manager.update_alignment_status(state="cancelled", message=str(exc))
    except Exception as exc:
        active_restored = _restore_active_alignment(manager, start_offset)
        manager.update_alignment_status(
            state="error", message="%s失败: %s" % ("独立细扫" if fine_only else "自动对齐", exc),
        )
    finally:
        manager.update_slm_image(phase_uint8, apply_lut=False)


def run_fine_alignment(manager: Any, request: Dict[str, Any], phase_uint8: np.ndarray) -> None:
    """Fine scan only, centered on the currently applied or loaded offset."""
    _run_alignment_search(manager, request, phase_uint8, fine_only=True)


def run_automatic_alignment(manager: Any, request: Dict[str, Any], phase_uint8: np.ndarray) -> None:
    """Full automatic alignment: coarse scan followed by configured fine scan."""
    _run_alignment_search(manager, request, phase_uint8, fine_only=False)
