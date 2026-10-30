"""DMD-to-CCD intensity response measurement and linearization."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sdon.hardware.flir_camera import validate_saved_camera_measurement
from sdon.hardware.display_timing import validate_saved_display_mode, capture_display_mode


def validate_intensity_canvas(manager: Any, summary: Dict[str, Any]) -> None:
    """Reject a DMD response measured on another physical output aperture."""
    validate_saved_camera_measurement(manager, summary)
    validate_saved_display_mode(manager, summary)
    display = getattr(getattr(manager, "config", None), "display", None)
    if display is None:
        return
    current = int(getattr(display, "dmd_modulated_size", display.modulated_size)[0])
    saved = int(summary.get("dmd_canvas_size", 512))
    if saved != current:
        raise ValueError(
            "DMD intensity LUT canvas changed: saved=%d, current=%d; recalibrate intensity"
            % (saved, current)
        )


def normalize_response_to_uint8(response: np.ndarray) -> np.ndarray:
    """Map the measured dark-to-peak response to a display range of 0..255."""
    values = np.asarray(response, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("intensity response must contain finite values")
    dynamic = float(values[-1] - values[0])
    if dynamic <= 0.0:
        raise RuntimeError("measured response has no usable dynamic range")
    return np.clip((values - values[0]) / dynamic * 255.0, 0.0, 255.0)


def build_linearization_lut(response: np.ndarray) -> np.ndarray:
    """Map desired linear gray levels to physical DMD input values.

    Small response reversals caused by noise are removed with a cumulative
    maximum before inversion. Flat response regions are handled by keeping the
    first DMD gray level at which that response was reached.
    """
    values = np.asarray(response, dtype=np.float64).reshape(-1)
    if values.size != 256 or not np.all(np.isfinite(values)):
        raise ValueError("intensity response must contain 256 finite values")
    monotonic = np.maximum.accumulate(values)
    dark, peak = float(monotonic[0]), float(monotonic[-1])
    if peak <= dark:
        raise RuntimeError("DMD-CCD response has no usable dynamic range")
    unique_response, unique_indices = np.unique(monotonic, return_index=True)
    physical_gray = np.arange(256, dtype=np.float64)[unique_indices]
    target_response = np.linspace(dark, peak, 256)
    return np.interp(target_response, unique_response, physical_gray).astype(np.float32)


def capture_median(manager: Any, samples: int, size: tuple = (512, 512)) -> Optional[np.ndarray]:
    frames: List[np.ndarray] = []
    for _ in range(max(1, int(samples))):
        frame = manager.capture_ccd_image(size)
        if frame is not None and np.asarray(frame).size:
            frames.append(np.asarray(frame))
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(frames[0].dtype)


def _response_metrics(response: np.ndarray) -> Dict[str, float]:
    dark, peak = float(response[0]), float(response[-1])
    dynamic_range = peak - dark
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        raise RuntimeError("measured response has no usable dynamic range")
    differences = np.diff(response)
    tolerance = dynamic_range * 0.001
    early_slope = float(np.median(differences[16:96]))
    late_slope = float(np.median(differences[224:255]))
    return {
        "dark": dark,
        "peak": peak,
        "dynamic_range": dynamic_range,
        "monotonic_violations": int(np.count_nonzero(differences < -tolerance)),
        "early_slope": early_slope,
        "late_slope": late_slope,
        "high_low_slope_ratio": late_slope / early_slope if early_slope > 0 else float("nan"),
    }


def _save_results(
    output_dir: Path,
    timestamp: str,
    response: np.ndarray,
    lut: np.ndarray,
    frame_records: List[Dict[str, float]],
    camera_parameters: Dict[str, Any],
    camera_roi: Dict[str, int],
    dmd_canvas_size: int,
    display_mode: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _response_metrics(response)
    normalized_8bit = normalize_response_to_uint8(response)
    response_path = output_dir / "ccd_mean_map.npy"
    lut_path = output_dir / "dmd_linearization_lut.npy"
    csv_path = output_dir / "ccd_mean_map.csv"
    np.save(response_path, response)
    np.save(lut_path, lut)
    np.savetxt(
        csv_path,
        np.column_stack((np.arange(256), response, normalized_8bit, lut)),
        delimiter=",",
        comments="",
        header="dmd_gray,ccd_mean,normalized_response_8bit,linearized_dmd_gray",
        fmt=["%d", "%.9g", "%.9g", "%.6f"],
    )
    summary = {
        "timestamp": timestamp,
        "camera": camera_parameters,
        "camera_roi": camera_roi,
        "display_mode": display_mode,
        "dmd_canvas_size": int(dmd_canvas_size),
        "normalized_display_range": [0, 255],
        "maximum_p99_9": float(max(item["p99_9"] for item in frame_records)),
        "maximum_saturation_fraction": float(
            max(item["saturation_fraction"] for item in frame_records)
        ),
        **metrics,
        "frames": frame_records,
    }
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(256), normalized_8bit, label="measured response (normalized)")
    axis.plot(lut, np.arange(256), "--", label="linearized target")
    axis.set(
        xlabel="DMD gray",
        ylabel="Normalized CCD intensity (0-255)",
        ylim=(-5, 260),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    curve_path = output_dir / "ccd_response_curve.png"
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    return {
        **metrics,
        "normalized_display_range": [0, 255],
        "maximum_p99_9": float(max(item["p99_9"] for item in frame_records)),
        "maximum_saturation_fraction": float(
            max(item["saturation_fraction"] for item in frame_records)
        ),
        "response_path": str(response_path),
        "lut_path": str(lut_path),
        "csv_path": str(csv_path),
        "curve_path": str(curve_path),
    }


def run_intensity_calibration(
    manager: Any, request: Dict[str, Any], restore_phase: np.ndarray
) -> None:
    """Scan all DMD gray levels inside the sole hardware worker thread."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    requested_dir = request.get("output_dir")
    output_dir = (
        Path(requested_dir)
        if requested_dir
        else (manager.config.paths.output_dir / ("intensity_calibration_" + timestamp))
    )
    response = np.full(256, np.nan, dtype=np.float64)
    records: List[Dict[str, float]] = []
    was_phase_enabled = manager.is_slm_phase_enabled()
    manager.set_slm_phase_enabled(False)
    try:
        for gray in range(256):
            if manager.is_intensity_cancelled():
                manager.update_intensity_status(state="cancelled", message="光强标定已停止")
                return
            width, height = manager.config.display.modulated_size
            manager.update_dmd_image(
                np.full((height, width), gray, dtype=np.uint8), apply_lut=False
            )
            if not manager.wait_for_dmd_update(1.0):
                raise RuntimeError("等待 DMD 灰度更新超时")
            time.sleep(request["settle_ms"] / 1000.0)
            image = capture_median(
                manager, request["samples"], manager.config.display.modulated_size
            )
            if image is None:
                raise RuntimeError("CCD 未返回有效图像 (gray=%d)" % gray)
            sensor_max = (
                float(np.iinfo(image.dtype).max)
                if np.issubdtype(image.dtype, np.integer)
                else float(np.max(image))
            )
            mean = float(np.mean(image))
            p999 = float(np.percentile(image, 99.9))
            saturation = float(np.mean(image >= sensor_max)) if sensor_max > 0 else 0.0
            response[gray] = mean
            records.append(
                {
                    "dmd_gray": gray,
                    "ccd_mean": mean,
                    "p99_9": p999,
                    "saturation_fraction": saturation,
                }
            )
            manager.update_intensity_status(
                current=gray + 1,
                latest_gray=gray,
                latest_mean=mean,
                latest_p99_9=p999,
                latest_saturation=saturation,
                message="扫描 DMD gray=%d/255" % gray,
            )

        lut = build_linearization_lut(response)
        saved = _save_results(
            output_dir,
            timestamp,
            response,
            lut,
            records,
            manager.camera.get_camera_parameters(),
            manager.get_camera_roi(),
            int(
                getattr(
                    manager.config.display,
                    "dmd_modulated_size",
                    manager.config.display.modulated_size,
                )[0]
            ),
            capture_display_mode(manager),
        )
        manager.set_dmd_linearization_lut(lut)
        manager.update_intensity_status(
            state="complete",
            current=256,
            total=256,
            message="光强标定完成",
            output_dir=str(output_dir),
            **saved
        )
    except Exception as exc:
        manager.update_intensity_status(state="error", message="光强标定失败: %s" % exc)
    finally:
        manager.update_slm_image(restore_phase, apply_lut=False)
        manager.set_slm_phase_enabled(was_phase_enabled)
        mode, rectangle_size = manager.get_alignment_pattern()
        from .patterns import build_alignment_pattern

        dmd_size = int(
            getattr(
                manager.config.display,
                "dmd_modulated_size",
                manager.config.display.modulated_size,
            )[0]
        )
        manager.update_dmd_image(
            build_alignment_pattern(
                mode, size=dmd_size, rectangle_size=rectangle_size
            )
        )
