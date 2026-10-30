"""No-training validation of the DMD -> SLM phase -> CCD hardware link."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from mnist_sdon.config import load_phase1_config

from .alignment import validate_alignment_canvas
from .intensity import capture_median
from .patterns import generate_blazed_grating
from .slm_check import bright_feature_centroid


DEFAULT_PROBE_POSITIONS = (
    (8, 8),
    (16, 8),
    (24, 8),
    (8, 16),
    (16, 16),
    (24, 16),
    (8, 24),
    (16, 24),
    (24, 24),
)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left -= float(np.percentile(left, 20.0))
    right -= float(np.percentile(right, 20.0))
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def _centered_nearest(
    image: np.ndarray,
    active_size: int,
    canvas_size: int,
    effective_size: Optional[int] = None,
) -> np.ndarray:
    values = np.asarray(image)
    active, canvas = int(active_size), int(canvas_size)
    if values.ndim != 2 or active < 1 or canvas < active:
        raise ValueError("invalid square modulator aperture")
    if effective_size is not None:
        effective = int(effective_size)
        if effective < 1 or active % effective:
            raise ValueError("effective_size must divide active_size")
        resized = cv2.resize(
            values, (effective, effective), interpolation=cv2.INTER_NEAREST
        )
        reuse = active // effective
        resized = np.repeat(np.repeat(resized, reuse, axis=0), reuse, axis=1)
    else:
        resized = cv2.resize(values, (active, active), interpolation=cv2.INTER_NEAREST)
    output = np.zeros((canvas, canvas), dtype=values.dtype)
    start = (canvas - active) // 2
    output[start : start + active, start : start + active] = resized
    return output


def build_dmd_probe(
    logical_size: int,
    position_xy: Tuple[int, int],
    width: int = 2,
) -> np.ndarray:
    """Return one small binary probe on the logical DMD grid."""
    size = int(logical_size)
    x, y = map(int, position_xy)
    patch = max(1, int(width))
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError("DMD probe position is outside the logical grid")
    half = patch // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(size, x0 + patch), min(size, y0 + patch)
    pattern = np.zeros((size, size), dtype=np.uint8)
    pattern[y0:y1, x0:x1] = 255
    return pattern


def build_weight_link_phases(
    simulation_size: int,
    periods: Sequence[int],
    control_sizes: Sequence[int],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Build zero, signed gratings, and deterministic block phase masks."""
    size = int(simulation_size)
    if size < 1:
        raise ValueError("simulation_size must be positive")
    phases: Dict[str, np.ndarray] = {
        "zero": np.zeros((size, size), dtype=np.uint8)
    }
    for period in periods:
        period = int(period)
        if period < 4:
            raise ValueError("grating periods must be at least 4 pixels")
        for axis in ("x", "y"):
            for direction, sign in ((1, "plus"), (-1, "minus")):
                name = "p%d_%s_%s" % (period, sign, axis)
                phases[name] = generate_blazed_grating(
                    (size, size), period, axis, direction
                )
    for control_size in control_sizes:
        control = int(control_size)
        if control < 1 or size % control:
            raise ValueError("every control size must divide simulation_size")
        rng = np.random.RandomState(int(seed) + control)
        # Binary 0/pi blocks are easy to see and avoid an accidental smooth
        # interpolation path.  Repeating with nearest-neighbour is the exact
        # block definition used by the hardware phase loader.
        block = rng.randint(0, 2, size=(control, control)).astype(np.uint8) * 128
        scale = size // control
        phases["c%d_blocks" % control] = np.repeat(
            np.repeat(block, scale, axis=0), scale, axis=1
        )
    return phases


def _register_frame(
    frame: np.ndarray,
    background: np.ndarray,
    orientation: str,
    transform: np.ndarray,
    output_size: int,
) -> np.ndarray:
    # HardwareManager.capture_ccd_image already applies the selected saved
    # orientation.  Applying it again here would silently flip/rotate the link
    # data twice and invalidate the saved simulation-to-CCD transform.
    if orientation not in {"identity", "flip_horizontal", "flip_vertical", "rotate_180"}:
        raise ValueError("invalid confirmed CCD orientation")
    corrected = np.clip(
        np.asarray(frame, dtype=np.float32) - np.asarray(background, dtype=np.float32),
        0.0,
        None,
    )
    return cv2.warpAffine(
        corrected,
        np.asarray(transform, dtype=np.float32),
        (int(output_size), int(output_size)),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _fit_probe_mapping(
    zero_frames: Mapping[Tuple[int, int], np.ndarray]
) -> Dict[str, Any]:
    positions = np.asarray(list(zero_frames), dtype=np.float64)
    centroids = np.asarray(
        [bright_feature_centroid(zero_frames[tuple(point)]) for point in positions],
        dtype=np.float64,
    )
    design = np.column_stack([positions, np.ones(len(positions))])
    coefficients, _, _, _ = np.linalg.lstsq(design, centroids, rcond=None)
    predicted = design @ coefficients
    residual = centroids - predicted
    ss_residual = float(np.sum(residual**2))
    ss_total = float(np.sum((centroids - centroids.mean(axis=0)) ** 2))
    r_squared = 1.0 - ss_residual / max(ss_total, 1.0e-12)
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    affine = coefficients.T
    x_vector, y_vector = affine[:, 0], affine[:, 1]
    x_span = float(np.linalg.norm(x_vector) * 16.0)
    y_span = float(np.linalg.norm(y_vector) * 16.0)
    determinant = float(np.linalg.det(affine[:, :2]))
    passed = r_squared >= 0.95 and rmse <= 3.0 and min(x_span, y_span) >= 5.0
    return {
        "r_squared": float(r_squared),
        "rmse_px": rmse,
        "x_span_for_16_logical_px": x_span,
        "y_span_for_16_logical_px": y_span,
        "orientation_determinant": determinant,
        "affine_logical_dmd_to_registered_ccd": affine.tolist(),
        "centroids": {
            "%d,%d" % tuple(map(int, position)): centroids[index].tolist()
            for index, position in enumerate(positions)
        },
        "passed": bool(passed),
    }


def _analyze_grating_pair(
    zero_frames: Mapping[Tuple[int, int], np.ndarray],
    plus_frames: Mapping[Tuple[int, int], np.ndarray],
    minus_frames: Mapping[Tuple[int, int], np.ndarray],
) -> Dict[str, Any]:
    pair_vectors: List[np.ndarray] = []
    shift_magnitudes: List[float] = []
    opposite: List[bool] = []
    per_position: Dict[str, Any] = {}
    for position in zero_frames:
        zero = np.asarray(bright_feature_centroid(zero_frames[position]))
        plus = np.asarray(bright_feature_centroid(plus_frames[position]))
        minus = np.asarray(bright_feature_centroid(minus_frames[position]))
        plus_delta, minus_delta = plus - zero, minus - zero
        pair = plus - minus
        magnitude = float(np.linalg.norm(pair) / 2.0)
        is_opposite = bool(float(np.dot(plus_delta, minus_delta)) < 0.0)
        pair_vectors.append(pair)
        shift_magnitudes.append(magnitude)
        opposite.append(is_opposite)
        per_position["%d,%d" % position] = {
            "plus_delta_xy_px": plus_delta.tolist(),
            "minus_delta_xy_px": minus_delta.tolist(),
            "half_pair_shift_px": magnitude,
            "opposite_about_zero": is_opposite,
        }
    vectors = np.asarray(pair_vectors)
    mean_vector = vectors.mean(axis=0)
    mean_shift = float(np.mean(shift_magnitudes))
    shift_cv = float(np.std(shift_magnitudes) / max(mean_shift, 1.0e-9))
    opposite_fraction = float(np.mean(opposite))
    passed = mean_shift >= 3.0 and shift_cv <= 0.25 and opposite_fraction >= 0.8
    return {
        "mean_pair_vector_xy_px": mean_vector.tolist(),
        "mean_single_order_shift_px": mean_shift,
        "shift_cv_across_field": shift_cv,
        "opposite_fraction": opposite_fraction,
        "per_position": per_position,
        "passed": bool(passed),
    }


def analyze_weight_link(
    probe_frames: Mapping[str, Mapping[Tuple[int, int], np.ndarray]],
    block_repeats: Mapping[int, Sequence[np.ndarray]],
    periods: Sequence[int],
    maximum_saturation_fraction: float = 0.0,
) -> Dict[str, Any]:
    """Analyze measurements without using labels or a trained phase mask."""
    zero_frames = probe_frames["zero"]
    mapping = _fit_probe_mapping(zero_frames)
    gratings: Dict[str, Any] = {}
    mean_vectors: Dict[Tuple[int, str], np.ndarray] = {}
    for period in periods:
        for axis in ("x", "y"):
            key = "p%d_%s" % (int(period), axis)
            result = _analyze_grating_pair(
                zero_frames,
                probe_frames["p%d_plus_%s" % (int(period), axis)],
                probe_frames["p%d_minus_%s" % (int(period), axis)],
            )
            gratings[key] = result
            mean_vectors[(int(period), axis)] = np.asarray(
                result["mean_pair_vector_xy_px"], dtype=np.float64
            )
    period_ratios: Dict[str, float] = {}
    periods = tuple(map(int, periods))
    ratio_passed = True
    if len(periods) >= 2:
        fine, coarse = min(periods), max(periods)
        expected = coarse / float(fine)
        for axis in ("x", "y"):
            measured = gratings["p%d_%s" % (fine, axis)][
                "mean_single_order_shift_px"
            ] / max(
                gratings["p%d_%s" % (coarse, axis)][
                    "mean_single_order_shift_px"
                ],
                1.0e-9,
            )
            period_ratios[axis] = float(measured)
            ratio_passed = ratio_passed and abs(measured - expected) <= 0.3 * expected
    orthogonality: Dict[str, float] = {}
    orthogonal_passed = True
    for period in periods:
        x_vector, y_vector = mean_vectors[(period, "x")], mean_vectors[(period, "y")]
        cosine = abs(float(np.dot(x_vector, y_vector))) / max(
            float(np.linalg.norm(x_vector) * np.linalg.norm(y_vector)), 1.0e-9
        )
        orthogonality[str(period)] = cosine
        orthogonal_passed = orthogonal_passed and cosine <= 0.30
    grating_passed = bool(
        all(item["passed"] for item in gratings.values()) and ratio_passed and orthogonal_passed
    )

    zero_center = zero_frames[(16, 16)]
    bandwidth: Dict[str, Any] = {}
    reliable: List[int] = []
    for control_size, repeats in sorted(block_repeats.items()):
        if len(repeats) != 2:
            raise ValueError("every block mask requires exactly two captures")
        repeatability = _normalized_correlation(repeats[0], repeats[1])
        average = (np.asarray(repeats[0]) + np.asarray(repeats[1])) * 0.5
        effect = 1.0 - _normalized_correlation(zero_center, average)
        passed = repeatability >= 0.95 and effect >= 0.03
        if passed:
            reliable.append(int(control_size))
        bandwidth["C%d" % int(control_size)] = {
            "repeatability_correlation": float(repeatability),
            "effect_from_zero": float(effect),
            "passed": bool(passed),
        }
    exposure_passed = float(maximum_saturation_fraction) <= 0.001
    return {
        "probe_mapping": mapping,
        "gratings": gratings,
        "grating_period_shift_ratios": period_ratios,
        "grating_axis_absolute_cosine": orthogonality,
        "gratings_passed": grating_passed,
        "block_bandwidth": bandwidth,
        "maximum_repeatable_control_size": max(reliable) if reliable else None,
        "maximum_saturation_fraction": float(maximum_saturation_fraction),
        "exposure_passed": exposure_passed,
        "passed": bool(
            mapping["passed"] and grating_passed and bool(reliable) and exposure_passed
        ),
        "thresholds": {
            "probe_mapping_r_squared_min": 0.95,
            "probe_mapping_rmse_px_max": 3.0,
            "grating_shift_px_min": 3.0,
            "grating_shift_cv_max": 0.25,
            "block_repeatability_min": 0.95,
            "block_effect_from_zero_min": 0.03,
            "maximum_saturation_fraction_max": 0.001,
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_report(run_dir: Path, summary: Dict[str, Any]) -> Path:
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    analysis = summary["analysis"]
    mapping = analysis["probe_mapping"]
    maximum_control = max(int(value) for value in summary["settings"]["control_sizes"])
    lines = [
        "# DMD–SLM 权重链路验证报告",
        "",
        "本测试不训练网络，也不使用 MNIST 标签。它只检查 DMD 输入、SLM 相位和 CCD 输出是否形成可重复的物理链路。",
        "",
        "- 总结：%s" % ("通过" if analysis["passed"] else "未通过/需复核"),
        "- DMD 九点映射：R²=%.4f，RMSE=%.2f px，%s"
        % (
            mapping["r_squared"],
            mapping["rmse_px"],
            "通过" if mapping["passed"] else "未通过",
        ),
        "- 正负光栅：%s" % ("通过" if analysis["gratings_passed"] else "未通过"),
        "- 最大饱和像素比例：%.4f%%，%s"
        % (
            analysis["maximum_saturation_fraction"] * 100.0,
            "通过" if analysis["exposure_passed"] else "未通过",
        ),
        "- 最大可重复块状控制网格：%s"
        % (
            "C%d" % analysis["maximum_repeatable_control_size"]
            if analysis["maximum_repeatable_control_size"] is not None
            else "无",
        ),
        "",
        "## 像素定义",
        "",
        "- DMD：%d×%d 逻辑输入映射到 %d×%d 有效像素；每个逻辑像素约为 %.1f×%.1f 个 DMD 像素。"
        % (
            summary["geometry"]["logical_size"],
            summary["geometry"]["logical_size"],
            summary["geometry"]["dmd_active_size"],
            summary["geometry"]["dmd_active_size"],
            summary["geometry"]["dmd_pixels_per_logical_pixel"],
            summary["geometry"]["dmd_pixels_per_logical_pixel"],
        ),
        "- SLM：相位计算网格为 %d×%d，有效输出为 %d×%d；C%d 时每个权重为 %.1f×%.1f 个 SLM 像素。"
        % (
            summary["geometry"]["simulation_size"],
            summary["geometry"]["simulation_size"],
            summary["geometry"]["slm_active_size"],
            summary["geometry"]["slm_active_size"],
            maximum_control,
            summary["geometry"]["slm_active_size"] / float(maximum_control),
            summary["geometry"]["slm_active_size"] / float(maximum_control),
        ),
        "",
        "## 解释边界",
        "",
        (
            "通过表示相位图已被正确显示、正负方向和周期关系可测、块状权重能产生可重复变化。"
            "它不是分类准确率，也不证明仿真传播模型已经准确；通过后才适合继续做仿真—实机 "
            "PSF 对照和 ONN 训练。"
        ),
    ]
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_weight_link_validation(
    manager: Any, request: Dict[str, Any], restore_phase: np.ndarray
) -> None:
    """Execute the validation in the single shared hardware worker."""
    original_mode, original_rectangle = manager.get_alignment_pattern()
    original_phase_enabled = manager.is_slm_phase_enabled()
    probe_frames: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    block_repeats: Dict[int, List[np.ndarray]] = {}
    run_dir: Optional[Path] = None
    try:
        experiment_path = Path(request["experiment_config"]).resolve()
        alignment_path = Path(request["alignment_path"]).resolve()
        if not experiment_path.is_file() or not alignment_path.is_file():
            raise FileNotFoundError("实验配置或对齐配置不存在")
        experiment = load_phase1_config(experiment_path)
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        manager.assert_camera_roi_matches(alignment.get("camera_roi") or {})
        validate_alignment_canvas(manager, alignment)
        if not bool((alignment.get("quality") or {}).get("constraint_satisfied", False)):
            raise RuntimeError("历史 DMD–SLM 对齐未通过质量约束")
        if not manager.has_slm_phase_lut():
            raise RuntimeError("尚未加载有效的 SLM 相位 LUT")

        optical = experiment.optical
        logical_size = int(optical.logical_size)
        simulation_size = int(optical.simulation_size)
        dmd_active = int(optical.resolved_dmd_active_size)
        slm_active = int(optical.resolved_slm_active_size)
        dmd_effective = simulation_size * dmd_active // int(optical.physical_size)
        dmd_canvas = int(manager.dmd.modulated_size[0])
        slm_canvas = int(manager.slm.modulated_size[0])
        registration_canvas = int(
            alignment.get("simulation_canvas_size", slm_canvas)
        )
        if tuple(manager.dmd.modulated_size) != (dmd_canvas, dmd_canvas):
            raise RuntimeError("DMD 输出画布必须为正方形")
        if tuple(manager.slm.modulated_size) != (slm_canvas, slm_canvas):
            raise RuntimeError("SLM 输出画布必须为正方形")
        if dmd_canvas != dmd_active or slm_canvas != slm_active:
            raise RuntimeError(
                "输出画布与实验定义不一致：当前 DMD/SLM=%d/%d，实验要求=%d/%d。"
                "请使用匹配的硬件配置启动 GUI（当前统一配置使用 config.toml），"
                "不要让 Qt 把相位图缩放后继续实验。"
                % (dmd_canvas, slm_canvas, dmd_active, slm_active)
            )
        if registration_canvas != slm_canvas:
            raise RuntimeError(
                "对齐注册画布与当前 SLM 不一致：对齐=%d，SLM=%d；请重新对齐"
                % (registration_canvas, slm_canvas)
            )
        transform = np.asarray(alignment.get("geometry_transform"), dtype=np.float32)
        if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
            raise RuntimeError("历史对齐缺少有效的 CCD 几何变换")
        orientation = str(alignment.get("ccd_orientation", "identity"))
        offset = alignment.get("slm_offset")
        if not isinstance(offset, list) or len(offset) != 2:
            raise RuntimeError("历史对齐缺少 SLM offset")
        manager.update_slm_offset(int(offset[0]), int(offset[1]))
        manager.set_ccd_orientation(orientation)
        manager.set_slm_phase_enabled(True)

        periods = tuple(int(value) for value in request["periods"])
        controls = tuple(int(value) for value in request["control_sizes"])
        phases = build_weight_link_phases(simulation_size, periods, controls)
        positions = tuple(DEFAULT_PROBE_POSITIONS)
        grating_names = [
            name for name in phases if name == "zero" or name.startswith("p")
        ]
        total = len(grating_names) * len(positions) + 2 * len(controls)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(manager.config.paths.output_dir) / (
            "dmd_slm_weight_link_" + timestamp
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "phases").mkdir()
        (run_dir / "registered_ccd").mkdir()
        for name, phase in phases.items():
            cv2.imwrite(str(run_dir / "phases" / (name + ".png")), phase)

        settle_s = max(
            float(manager.config.timing.slm_min_interval_s),
            float(request["settle_ms"]) / 1000.0,
        )
        samples = int(request["samples"])
        current = 0
        saturation_values: List[float] = []

        def cancelled() -> bool:
            return bool(manager.is_weight_link_cancelled())

        def load_phase(name: str) -> np.ndarray:
            phase = phases[name]
            display = _centered_nearest(
                phase,
                slm_active,
                slm_canvas,
                effective_size=simulation_size,
            )
            manager.update_weight_link_status(
                state="running",
                current=current,
                total=total,
                current_pattern=name,
                phase=phase,
                message="正在加载 SLM 相位：%s；本相位采集期间保持固定" % name,
            )
            manager.update_slm_image(display, apply_lut=True)
            if not manager.wait_for_slm_update(2.0):
                raise RuntimeError("等待 SLM 更新超时：%s" % name)
            time.sleep(settle_s)
            return phase

        def capture_background() -> np.ndarray:
            black = np.zeros((logical_size, logical_size), dtype=np.uint8)
            dmd = _centered_nearest(
                black,
                dmd_active,
                dmd_canvas,
                effective_size=dmd_effective,
            )
            manager.update_dmd_image(dmd, apply_lut=False)
            if not manager.wait_for_dmd_update(2.0):
                raise RuntimeError("等待 DMD 黑场更新超时")
            time.sleep(settle_s)
            frame = capture_median(manager, samples, manager.config.display.modulated_size)
            if frame is None:
                raise RuntimeError("CCD 未返回相位对应黑场")
            return np.asarray(frame)

        def capture_probe(
            phase_name: str,
            position: Tuple[int, int],
            background: np.ndarray,
            repeat: int = 0,
        ) -> np.ndarray:
            nonlocal current
            if cancelled():
                raise InterruptedError("用户停止了 DMD–SLM 权重链路验证")
            logical = build_dmd_probe(
                logical_size, position, int(request["probe_width"])
            )
            dmd = _centered_nearest(
                logical,
                dmd_active,
                dmd_canvas,
                effective_size=dmd_effective,
            )
            manager.update_dmd_image(dmd, apply_lut=True)
            if not manager.wait_for_dmd_update(2.0):
                raise RuntimeError("等待 DMD 探针更新超时")
            time.sleep(settle_s)
            raw = capture_median(manager, samples, manager.config.display.modulated_size)
            if raw is None:
                raise RuntimeError("CCD 未返回 DMD 探针图像")
            raw_array = np.asarray(raw)
            maximum = (
                float(np.iinfo(raw_array.dtype).max)
                if np.issubdtype(raw_array.dtype, np.integer)
                else 1.0
            )
            saturation_values.append(float(np.mean(raw_array >= 0.98 * maximum)))
            registered = _register_frame(
                raw,
                background,
                orientation,
                transform,
                registration_canvas,
            )
            current += 1
            manager.update_weight_link_status(
                state="running",
                current=current,
                total=total,
                current_pattern=phase_name,
                dmd=logical,
                phase=phases[phase_name],
                ccd=raw_array,
                registered=registered,
                message=(
                    "采集 %d/%d：SLM=%s，DMD=(%d,%d)，重复=%d"
                    % (current, total, phase_name, position[0], position[1], repeat + 1)
                ),
            )
            filename = "%s_x%02d_y%02d_r%d.npy" % (
                phase_name,
                position[0],
                position[1],
                repeat + 1,
            )
            np.save(run_dir / "registered_ccd" / filename, registered)
            return registered

        for phase_name in grating_names:
            load_phase(phase_name)
            background = capture_background()
            probe_frames[phase_name] = {}
            for position in positions:
                probe_frames[phase_name][position] = capture_probe(
                    phase_name, position, background
                )

        for control in controls:
            phase_name = "c%d_blocks" % control
            load_phase(phase_name)
            background = capture_background()
            block_repeats[control] = [
                capture_probe(phase_name, (16, 16), background, repeat)
                for repeat in range(2)
            ]

        analysis = analyze_weight_link(
            probe_frames,
            block_repeats,
            periods,
            maximum_saturation_fraction=max(saturation_values or [0.0]),
        )
        geometry = {
            "logical_size": logical_size,
            "simulation_size": simulation_size,
            "dmd_active_size": dmd_active,
            "slm_active_size": slm_active,
            "dmd_output_canvas": dmd_canvas,
            "slm_output_canvas": slm_canvas,
            "registered_output_canvas": registration_canvas,
            "dmd_pixels_per_logical_pixel": dmd_active / float(logical_size),
            "slm_pixels_per_simulation_pixel": slm_active / float(simulation_size),
            "slm_pixels_per_control": {
                "C%d" % value: slm_active / float(value) for value in controls
            },
            "same_normalized_dmd_cell_in_slm_pixels": slm_active / float(logical_size),
        }
        summary = {
            "format_version": 1,
            "timestamp": timestamp,
            "purpose": "no-training DMD-to-SLM phase-weight link validation",
            "experiment_config": str(experiment_path),
            "alignment_path": str(alignment_path),
            "slm_phase_lut": manager.get_slm_phase_lut_source_path(),
            "camera_roi": manager.get_camera_roi(),
            "slm_offset": [int(offset[0]), int(offset[1])],
            "ccd_orientation": orientation,
            "settings": dict(request),
            "geometry": geometry,
            "analysis": analysis,
            "display_status": manager.get_display_status(),
            "run_dir": str(run_dir.resolve()),
        }
        report_path = _write_report(run_dir, summary)
        manager.update_weight_link_status(
            state="complete",
            current=total,
            total=total,
            message="权重链路验证完成：%s" % ("通过" if analysis["passed"] else "未通过/需复核"),
            analysis=analysis,
            geometry=geometry,
            report_path=str(report_path.resolve()),
            result_rows=_result_rows(analysis),
        )
    except InterruptedError as exc:
        manager.update_weight_link_status(state="cancelled", message=str(exc))
    except Exception as exc:
        manager.update_weight_link_status(
            state="error", message="权重链路验证失败：%s" % exc
        )
    finally:
        manager.update_slm_image(restore_phase, apply_lut=False)
        manager.set_slm_phase_enabled(original_phase_enabled)
        size = int(
            getattr(
                manager.config.display,
                "dmd_modulated_size",
                manager.config.display.modulated_size,
            )[0]
        )
        from .patterns import build_alignment_pattern

        manager.update_dmd_image(
            build_alignment_pattern(
                original_mode, size=size, rectangle_size=original_rectangle
            )
        )


def _result_rows(analysis: Mapping[str, Any]) -> List[Dict[str, Any]]:
    mapping = analysis["probe_mapping"]
    rows = [
        {
            "item": "DMD 九点映射",
            "metric": "R²=%.3f, RMSE=%.2f px"
            % (mapping["r_squared"], mapping["rmse_px"]),
            "passed": mapping["passed"],
        }
    ]
    rows.append(
        {
            "item": "CCD 曝光",
            "metric": "最大饱和像素=%.4f%%"
            % (analysis["maximum_saturation_fraction"] * 100.0),
            "passed": analysis["exposure_passed"],
        }
    )
    for name, item in analysis["gratings"].items():
        rows.append(
            {
                "item": name,
                "metric": "位移=%.2f px, 场内CV=%.3f, 反向比例=%.2f"
                % (
                    item["mean_single_order_shift_px"],
                    item["shift_cv_across_field"],
                    item["opposite_fraction"],
                ),
                "passed": item["passed"],
            }
        )
    for name, item in analysis["block_bandwidth"].items():
        rows.append(
            {
                "item": name + " 块状权重",
                "metric": "重复相关=%.3f, 相对零相位变化=%.3f"
                % (item["repeatability_correlation"], item["effect_from_zero"]),
                "passed": item["passed"],
            }
        )
    return rows
