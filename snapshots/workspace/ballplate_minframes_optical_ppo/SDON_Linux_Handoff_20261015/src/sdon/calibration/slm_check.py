"""Standalone SLM zero/lens/blazed-grating hardware self-check."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import cv2
import numpy as np

from .alignment import validate_alignment_canvas
from .intensity import capture_median
from .patterns import build_alignment_pattern, generate_blazed_grating, generate_lens_phase


PATTERN_LABELS = {
    "zero": "Zero phase",
    "lens": "Lens phase",
    "plus_x": "+X blaze",
    "minus_x": "-X blaze",
    "plus_y": "+Y blaze",
    "minus_y": "-Y blaze",
}


def build_slm_check_patterns(
    shape: Tuple[int, int],
    period_px: int,
    focal_length: float,
    wavelength: float,
    pixel_pitch: float,
) -> Dict[str, np.ndarray]:
    """Build the six phase maps used by the quick check in capture order."""
    return {
        "zero": np.zeros(shape, dtype=np.uint8),
        "lens": generate_lens_phase(
            shape,
            focal_length=focal_length,
            wavelength=wavelength,
            pixel_pitch=pixel_pitch,
        ),
        "plus_x": generate_blazed_grating(shape, period_px, "x", 1),
        "minus_x": generate_blazed_grating(shape, period_px, "x", -1),
        "plus_y": generate_blazed_grating(shape, period_px, "y", 1),
        "minus_y": generate_blazed_grating(shape, period_px, "y", -1),
    }


def bright_feature_centroid(image: np.ndarray) -> Tuple[float, float]:
    """Locate the bright diffracted feature while suppressing a broad background."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("CCD frame must be a finite 2-D image")
    blurred = cv2.GaussianBlur(values, (0, 0), 2.0)
    signal = np.maximum(blurred - float(np.percentile(blurred, 20.0)), 0.0)
    peak = float(signal.max())
    if peak <= 0.0:
        raise RuntimeError("CCD frame has no measurable bright feature")
    # The observed feature can be either a compact diffraction spot or a
    # translated image of the illuminated DMD aperture.  Keeping the brightest
    # five percent still isolates a compact spot through the 35%-of-peak floor,
    # while averaging enough pixels of a broad aperture to reject speckle.
    threshold = max(float(np.percentile(signal, 95.0)), peak * 0.35)
    weights = np.where(signal >= threshold, signal, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        weights, total = signal, float(signal.sum())
    yy, xx = np.indices(values.shape, dtype=np.float32)
    return float((weights * xx).sum() / total), float((weights * yy).sum() / total)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left -= np.percentile(left, 20.0)
    right -= np.percentile(right, 20.0)
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 1.0


def _axis_result(
    plus: Tuple[float, float],
    minus: Tuple[float, float],
    zero: Tuple[float, float],
    axis: str,
    min_shift_px: float,
    min_axis_ratio: float,
) -> Dict[str, Any]:
    pair_dx, pair_dy = plus[0] - minus[0], plus[1] - minus[1]
    primary = pair_dx if axis == "x" else pair_dy
    cross = pair_dy if axis == "x" else pair_dx
    plus_primary = plus[0] - zero[0] if axis == "x" else plus[1] - zero[1]
    minus_primary = minus[0] - zero[0] if axis == "x" else minus[1] - zero[1]
    separation = float(np.hypot(pair_dx, pair_dy))
    axis_ratio = abs(primary) / max(abs(cross), 1e-6)
    opposite_about_zero = plus_primary * minus_primary < 0.0
    passed = (
        separation >= 2.0 * min_shift_px
        and axis_ratio >= min_axis_ratio
        and abs(plus_primary) >= min_shift_px
        and abs(minus_primary) >= min_shift_px
        and opposite_about_zero
    )
    positive_direction = (
        ("右" if plus_primary > 0 else "左")
        if axis == "x"
        else ("下" if plus_primary > 0 else "上")
    )
    negative_direction = (
        ("右" if minus_primary > 0 else "左")
        if axis == "x"
        else ("下" if minus_primary > 0 else "上")
    )
    return {
        "pair_dx_px": float(pair_dx),
        "pair_dy_px": float(pair_dy),
        "separation_px": separation,
        "axis_ratio": float(axis_ratio),
        "plus_from_zero_px": float(plus_primary),
        "minus_from_zero_px": float(minus_primary),
        "opposite_about_zero": bool(opposite_about_zero),
        "plus_observed_direction": positive_direction,
        "minus_observed_direction": negative_direction,
        "passed": bool(passed),
    }


def analyze_slm_check(
    frames: Mapping[str, np.ndarray],
    min_shift_px: float = 3.0,
    min_axis_ratio: float = 1.5,
) -> Dict[str, Any]:
    """Measure lens response and opposite diffraction of the ±X/±Y pairs."""
    missing = [name for name in PATTERN_LABELS if name not in frames]
    if missing:
        raise ValueError("missing SLM check frames: %s" % ", ".join(missing))
    shapes = {np.asarray(frames[name]).shape for name in PATTERN_LABELS}
    if len(shapes) != 1:
        raise ValueError("all SLM check frames must have the same shape")

    centroids = {name: bright_feature_centroid(frames[name]) for name in PATTERN_LABELS}
    lens_correlation = _normalized_correlation(frames["zero"], frames["lens"])
    lens_change = max(0.0, 1.0 - lens_correlation)
    x_result = _axis_result(
        centroids["plus_x"],
        centroids["minus_x"],
        centroids["zero"],
        "x",
        float(min_shift_px),
        float(min_axis_ratio),
    )
    y_result = _axis_result(
        centroids["plus_y"],
        centroids["minus_y"],
        centroids["zero"],
        "y",
        float(min_shift_px),
        float(min_axis_ratio),
    )
    saturation = {}
    for name, frame in frames.items():
        array = np.asarray(frame)
        maximum = (
            float(np.iinfo(array.dtype).max)
            if np.issubdtype(array.dtype, np.integer)
            else 1.0
        )
        saturation[name] = float(np.mean(array >= maximum))
    exposure_ok = max(saturation.values()) <= 0.001
    lens_passed = lens_change >= 0.02
    return {
        "centroids": {
            name: [float(point[0]), float(point[1])] for name, point in centroids.items()
        },
        "lens": {
            "zero_lens_correlation": float(lens_correlation),
            "normalized_change": float(lens_change),
            "passed": bool(lens_passed),
        },
        "x_grating": x_result,
        "y_grating": y_result,
        "saturation_fraction": saturation,
        "valid_exposure": bool(exposure_ok),
        "passed": bool(lens_passed and x_result["passed"] and y_result["passed"] and exposure_ok),
        "thresholds": {
            "min_shift_px": float(min_shift_px),
            "min_axis_ratio": float(min_axis_ratio),
            "min_lens_change": 0.02,
            "max_saturation_fraction": 0.001,
        },
    }


def save_slm_check_result(
    output_dir: Path,
    phases: Mapping[str, np.ndarray],
    frames: Mapping[str, np.ndarray],
    report: Dict[str, Any],
) -> Path:
    """Save every physical input/output pair and a machine-readable report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in PATTERN_LABELS:
        cv2.imwrite(str(output_dir / (name + "_phase.png")), np.asarray(phases[name]))
        cv2.imwrite(str(output_dir / (name + "_ccd.png")), np.asarray(frames[name]))
        np.save(output_dir / (name + "_ccd.npy"), np.asarray(frames[name]))
    report["output_dir"] = str(output_dir.resolve())
    report_path = output_dir / "slm_check_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report_path


def run_slm_check(manager: Any, request: Dict[str, Any], restore_phase: np.ndarray) -> None:
    """Run the six-pattern check inside the calibration worker's hardware queue."""
    config = manager.config
    slm_size = getattr(
        config.display, "slm_modulated_size", config.display.modulated_size
    )
    shape = (slm_size[1], slm_size[0])
    dmd_size = int(
        getattr(config.display, "dmd_modulated_size", config.display.modulated_size)[0]
    )
    phases = build_slm_check_patterns(
        shape,
        int(request["period_px"]),
        config.optics.alignment_lens_focal_length_m,
        config.optics.wavelength_m,
        config.optics.pixel_pitch_m,
    )
    frames: Dict[str, np.ndarray] = {}
    original_mode, original_rectangle = manager.get_alignment_pattern()
    manager.update_slm_check_status(state="running", message="正在加载历史 ROI 与对齐位置")
    try:
        alignment_path = Path(request["alignment_path"]).resolve()
        with alignment_path.open("r", encoding="utf-8") as handle:
            alignment = json.load(handle)
        manager.assert_camera_roi_matches(alignment["camera_roi"])
        validate_alignment_canvas(manager, alignment)
        offset = alignment["slm_offset"]
        manager.update_slm_offset(int(offset[0]), int(offset[1]))
        orientation = str(alignment.get("ccd_orientation", "identity"))
        manager.set_ccd_orientation(orientation)
        dmd_pattern = build_alignment_pattern(
            "center_rectangle",
            size=dmd_size,
            rectangle_size=int(request["aperture_px"]),
        )
        manager.update_dmd_image(dmd_pattern, apply_lut=False)
        if not manager.wait_for_dmd_update(1.0):
            raise RuntimeError("等待 DMD 固定照明图案超时")

        settle_s = max(
            config.timing.slm_min_interval_s,
            float(request["settle_ms"]) / 1000.0,
        )
        for index, (name, phase) in enumerate(phases.items(), start=1):
            if manager.is_slm_check_cancelled():
                raise InterruptedError("用户停止了 SLM 自检")
            manager.update_slm_check_status(
                current=index - 1,
                message="正在加载 %s；采集期间相位保持固定" % PATTERN_LABELS[name],
                phase=phase,
                current_pattern=name,
                frames=frames,
            )
            manager.update_slm_image(phase, apply_lut=False)
            if not manager.wait_for_slm_update(1.0):
                raise RuntimeError("等待 SLM 更新 %s 超时" % PATTERN_LABELS[name])
            time.sleep(settle_s)
            frame = capture_median(
                manager,
                int(request["samples"]),
                config.display.modulated_size,
            )
            if frame is None:
                raise RuntimeError("%s 未收到 CCD 图像" % PATTERN_LABELS[name])
            frames[name] = frame
            manager.update_slm_check_status(
                current=index,
                message="已采集 %s (%d/%d)" % (PATTERN_LABELS[name], index, len(phases)),
                frames=frames,
            )

        analysis = analyze_slm_check(frames)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = config.paths.output_dir / ("slm_check_" + timestamp)
        report = {
            "format_version": 1,
            "timestamp": timestamp,
            "purpose": "qualitative SLM zero/lens/positive-negative blaze verification",
            "alignment_path": str(alignment_path),
            "camera_roi": manager.get_camera_roi(),
            "slm_offset": [int(offset[0]), int(offset[1])],
            "ccd_orientation": orientation,
            "settings": {
                "period_px": int(request["period_px"]),
                "aperture_px": int(request["aperture_px"]),
                "settle_ms": int(request["settle_ms"]),
                "samples": int(request["samples"]),
            },
            "analysis": analysis,
        }
        report_path = save_slm_check_result(output_dir, phases, frames, report)
        manager.update_slm_check_status(
            state="complete",
            message="SLM 快速自检完成：%s" % ("通过" if analysis["passed"] else "需人工复核"),
            analysis=analysis,
            report_path=str(report_path),
            frames=frames,
        )
    except InterruptedError as exc:
        manager.update_slm_check_status(state="cancelled", message=str(exc), frames=frames)
    except Exception as exc:
        manager.update_slm_check_status(
            state="error", message="SLM 快速自检失败：%s" % exc, frames=frames
        )
    finally:
        manager.update_slm_image(restore_phase, apply_lut=False)
        manager.update_dmd_image(
            build_alignment_pattern(
                original_mode,
                size=dmd_size,
                rectangle_size=original_rectangle,
            )
        )
        manager.update_slm_check_status(phase=restore_phase, current_pattern="lens")
