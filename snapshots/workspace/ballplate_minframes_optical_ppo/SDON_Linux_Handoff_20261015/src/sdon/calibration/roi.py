"""Automatic camera ROI calibration from the DMD intensity reference square."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from sdon.config import write_camera_roi

from .patterns import build_alignment_pattern


def _raw_median(camera: Any, samples: int) -> np.ndarray:
    frames = []
    for _ in range(max(1, int(samples))):
        frame = camera.get_frame()
        if frame is not None and np.asarray(frame).size:
            frames.append(np.asarray(frame))
    if not frames:
        raise RuntimeError("CCD did not return a valid raw frame")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise RuntimeError("camera frame shape changed during ROI calibration")
    return np.median(np.stack(frames, axis=0), axis=0).astype(frames[0].dtype)


def detect_intensity_square(
    bright: np.ndarray, dark: Optional[np.ndarray] = None
) -> Tuple[Tuple[int, int, int, int], Dict[str, float], np.ndarray]:
    """Locate the largest square-like bright component in a dark-subtracted frame."""
    bright_array = np.asarray(bright)
    if bright_array.ndim != 2 or min(bright_array.shape) < 32:
        raise ValueError("ROI calibration requires a 2-D camera frame of at least 32x32")
    if dark is None:
        signal = bright_array.astype(np.float32)
    else:
        dark_array = np.asarray(dark)
        if dark_array.shape != bright_array.shape:
            raise ValueError("dark and bright ROI frames must have the same shape")
        signal = bright_array.astype(np.float32) - dark_array.astype(np.float32)
    signal = np.maximum(signal, 0.0)
    low, high = np.percentile(signal, [10.0, 99.7])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError("reference square has no detectable intensity contrast")
    normalized = np.clip((signal - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    blur_size = max(5, int(round(min(normalized.shape) / 80.0)) | 1)
    blurred = cv2.GaussianBlur(normalized, (blur_size, blur_size), 0)
    otsu, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = max(3, int(round(min(normalized.shape) / 100.0)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    frame_area = float(normalized.size)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < max(64.0, frame_area * 0.002) or width < 8 or height < 8:
            continue
        aspect = min(width, height) / float(max(width, height))
        fill = area / float(width * height)
        candidates.append((area * (0.5 + 0.5 * aspect) * fill, label, x, y, width, height, area))
    if not candidates:
        raise RuntimeError("no complete bright reference square was found")
    _, label, x, y, width, height, area = max(candidates)
    rows, cols = normalized.shape
    edge_guard = max(2, int(round(min(rows, cols) * 0.005)))
    touches_edge = (
        x <= edge_guard
        or y <= edge_guard
        or x + width >= cols - edge_guard
        or y + height >= rows - edge_guard
    )
    if touches_edge:
        raise RuntimeError(
            "reference square touches the camera edge even at the detection ROI; "
            "reposition the optical field inside the sensor before retrying"
        )

    component = labels == label
    inside = signal[component]
    outside = signal[~component]
    inside_level = float(np.median(inside))
    outside_level = float(np.median(outside)) if outside.size else 0.0
    contrast = inside_level - outside_level
    if contrast <= max(1.0e-6, 0.03 * float(high - low)):
        raise RuntimeError("reference square contrast is too low for a safe ROI update")
    metrics = {
        "threshold_8bit": float(otsu),
        "component_area_px": float(area),
        "fill_ratio": float(area / float(width * height)),
        "aspect_ratio": float(min(width, height) / float(max(width, height))),
        "inside_median": inside_level,
        "outside_median": outside_level,
        "contrast": contrast,
    }
    return (x, y, width, height), metrics, normalized


def roi_around_square(
    current_roi: Dict[str, int],
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    margin_ratio: float = 0.20,
) -> Dict[str, int]:
    """Expand the detected box inside the hardware ROI used for detection."""
    rows, cols = (int(frame_shape[0]), int(frame_shape[1]))
    x, y, width, height = (int(value) for value in bbox)
    if cols != int(current_roi["Width"]) or rows != int(current_roi["Height"]):
        raise RuntimeError(
            "raw frame size %dx%d does not match camera ROI %dx%d"
            % (cols, rows, current_roi["Width"], current_roi["Height"])
        )
    margin_x = max(12, int(round(width * float(margin_ratio))))
    margin_y = max(12, int(round(height * float(margin_ratio))))
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(cols, x + width + margin_x)
    bottom = min(rows, y + height + margin_y)
    return {
        "OffsetX": int(current_roi["OffsetX"]) + left,
        "OffsetY": int(current_roi["OffsetY"]) + top,
        "Width": right - left,
        "Height": bottom - top,
    }


def run_camera_roi_calibration(
    manager: Any, request: Dict[str, Any], restore_phase: np.ndarray
) -> None:
    """Measure, apply, persist, and document a hardware camera ROI."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = manager.config.paths.output_dir / ("camera_roi_" + timestamp)
    original_roi = manager.get_camera_roi()
    full_roi: Optional[Dict[str, int]] = None
    manager_roi_applied = False
    was_phase_enabled = manager.is_slm_phase_enabled()
    size = manager.config.display.modulated_size
    try:
        manager.update_roi_status(
            state="running", message="正在临时恢复相机最大 ROI，确保完整看到参考方块"
        )
        # Do not call manager.set_camera_roi here: the full frame is temporary,
        # and invalidating all calibrations before detection succeeds would make
        # a recoverable failure destructive. FlirCamera.set_roi uses 0/0 size
        # as the maximum sensor extent and updates only the camera object.
        full_roi = manager.camera.set_roi(0, 0, 0, 0)
        manager.update_roi_status(
            message="已切换最大 ROI %dx%d，正在采集黑场参考"
            % (int(full_roi["Width"]), int(full_roi["Height"]))
        )
        manager.set_slm_phase_enabled(False)
        manager.update_dmd_image(np.zeros((size[1], size[0]), dtype=np.uint8), apply_lut=False)
        if not manager.wait_for_dmd_update(1.0):
            raise RuntimeError("waiting for the black DMD frame timed out")
        time.sleep(request["settle_ms"] / 1000.0)
        dark = _raw_median(manager.camera, request["samples"])

        manager.update_roi_status(message="正在采集强度标定白色方块")
        manager.update_dmd_image(np.full((size[1], size[0]), 255, dtype=np.uint8), apply_lut=False)
        if not manager.wait_for_dmd_update(1.0):
            raise RuntimeError("waiting for the white DMD square timed out")
        time.sleep(request["settle_ms"] / 1000.0)
        bright = _raw_median(manager.camera, request["samples"])
        manager.publish_preview(ccd=bright, message="已采集强度标定白色方块")
        bbox, metrics, normalized = detect_intensity_square(bright, dark)
        proposed = roi_around_square(
            full_roi, bbox, bright.shape, margin_ratio=request["margin_ratio"]
        )
        manager.update_roi_status(message="已找到方块，正在写入相机硬件 ROI")
        actual = manager.set_camera_roi(proposed)
        manager_roi_applied = True
        config_path = Path(request["config_path"])
        write_camera_roi(config_path, actual)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "dark_raw.npy", dark)
        np.save(output_dir / "bright_raw.npy", bright)
        overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        x, y, width, height = bbox
        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 3)
        cv2.imwrite(str(output_dir / "detected_square.png"), overlay)
        from sdon.hardware.display_timing import capture_display_mode
        summary = {
            "timestamp": timestamp,
            "camera": manager.camera.get_camera_parameters(),
            "display_mode": capture_display_mode(manager),
            "camera_roi_before": original_roi,
            "temporary_full_sensor_roi": full_roi,
            "detected_square_in_full_sensor_roi": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "detection": metrics,
            "margin_ratio_each_side": request["margin_ratio"],
            "camera_roi_requested": proposed,
            "camera_roi_applied": actual,
            "config_path": str(config_path.resolve()),
        }
        with (output_dir / "roi_calibration_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        post = _raw_median(manager.camera, 1)
        manager.publish_preview(ccd=post, message="自动 ROI 标定完成")
        manager.update_roi_status(
            state="complete",
            message=(
                "自动 ROI 已应用并保存；旧 ROI 下的光强、对齐和相位标定已失效，请从第 2 页重做"
            ),
            roi=actual,
            bbox=bbox,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        rollback_error = None
        try:
            if manager_roi_applied:
                restored = manager.set_camera_roi(original_roi)
                write_camera_roi(Path(request["config_path"]), restored)
            elif full_roi is not None:
                manager.camera.set_roi(
                    original_roi["OffsetX"],
                    original_roi["OffsetY"],
                    original_roi["Width"],
                    original_roi["Height"],
                )
        except Exception as restore_exc:
            rollback_error = restore_exc
        message = "自动 ROI 标定失败：%s" % exc
        if rollback_error is None:
            message += "；已恢复原 ROI"
        else:
            message += "；恢复原 ROI 失败：%s" % rollback_error
        manager.update_roi_status(state="error", message=message)
    finally:
        manager.update_slm_image(restore_phase, apply_lut=False)
        manager.set_slm_phase_enabled(was_phase_enabled)
        mode, rectangle_size = manager.get_alignment_pattern()
        manager.update_dmd_image(
            build_alignment_pattern(mode, size=size[0], rectangle_size=rectangle_size)
        )
