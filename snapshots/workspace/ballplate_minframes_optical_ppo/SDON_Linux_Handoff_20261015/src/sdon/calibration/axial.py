"""Manual-z SLM axial calibration using repeatable two-point interference."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .alignment import validate_alignment_canvas
from .intensity import capture_median
from .patterns import build_alignment_pattern
from .weight_link import _centered_nearest, _register_frame, build_weight_link_phases


DEFAULT_PAIRS: Mapping[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "横向间隔2": ((15, 16), (17, 16)),
    "横向间隔4": ((14, 16), (18, 16)),
    "横向间隔8": ((12, 16), (20, 16)),
    "纵向间隔8": ((16, 12), (16, 20)),
    "对角间隔8": ((12, 12), (20, 20)),
}


def apply_radial_defocus_correction(
    phase_uint8: np.ndarray, edge_cycles: float
) -> np.ndarray:
    """Add a wrapped quadratic phase whose mid-edge value is ``edge_cycles``.

    The input and output are ideal physical phase codes (0..255 -> 0..2pi),
    before the measured SLM gray LUT is applied.  A zero coefficient is an
    exact no-op so enabling the feature cannot alter existing experiments.
    """
    phase = np.asarray(phase_uint8)
    if phase.ndim != 2:
        raise ValueError("SLM defocus correction requires a 2-D phase image")
    coefficient = float(edge_cycles)
    if not np.isfinite(coefficient) or abs(coefficient) > 16.0:
        raise ValueError("SLM defocus edge cycles must be finite and within +/-16")
    source = np.asarray(phase, dtype=np.uint8)
    if abs(coefficient) <= 1.0e-12:
        return source.copy()
    height, width = source.shape
    x_radius = max((width - 1) / 2.0, 1.0)
    y_radius = max((height - 1) / 2.0, 1.0)
    yy, xx = np.indices((height, width), dtype=np.float64)
    rho_squared = (
        ((xx - (width - 1) / 2.0) / x_radius) ** 2
        + ((yy - (height - 1) / 2.0) / y_radius) ** 2
    )
    combined_turns = np.mod(
        source.astype(np.float64) / 255.0 + coefficient * rho_squared,
        1.0,
    )
    return np.clip(np.rint(combined_turns * 255.0), 0, 255).astype(np.uint8)


def save_active_defocus_correction(
    manager: Any,
    measurement: Mapping[str, Any],
    session_path: Path,
) -> Path:
    """Persist one explicitly accepted software-defocus result and activate it."""
    if str(measurement.get("measurement_mode", "")) != "software_defocus":
        raise ValueError("only a software-defocus scan result can be activated")
    if not bool(measurement.get("stable", False)):
        raise ValueError("the selected defocus result did not pass repeatability")
    coefficient = float(measurement["defocus_edge_cycles"])
    if not np.isfinite(coefficient):
        raise ValueError("the selected defocus coefficient is not finite")
    output_dir = Path(manager.config.paths.output_dir).resolve()
    path = output_dir / "slm_wavefront_correction_latest.json"
    dmd_size = int(manager.dmd.modulated_size[0])
    slm_size = int(manager.slm.modulated_size[0])
    alignment_getter = getattr(manager, "get_active_alignment_status", None)
    alignment = alignment_getter() if callable(alignment_getter) else {}
    payload = {
        "format_version": 1,
        "valid": True,
        "kind": "radial_quadratic_defocus",
        "defocus_edge_cycles": coefficient,
        "camera_roi": manager.get_camera_roi(),
        "dmd_canvas_size": dmd_size,
        "slm_canvas_size": slm_size,
        "alignment_path": str(alignment.get("config_path") or ""),
        "slm_phase_lut_path": str(manager.get_slm_phase_lut_source_path() or ""),
        "source_session": str(Path(session_path).resolve()),
        "source_measurement": str(measurement.get("measurement_path", "")),
        "score": measurement.get("score"),
        "mean_repeat_correlation": measurement.get("mean_repeat_correlation"),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    manager.set_slm_wavefront_correction(coefficient, str(path))
    return path


def build_two_point_patterns(
    logical_size: int = 32,
    pairs: Mapping[str, Tuple[Tuple[int, int], Tuple[int, int]]] = DEFAULT_PAIRS,
) -> Tuple[List[str], np.ndarray, Dict[Tuple[int, int], int], Dict[str, int]]:
    """Build unique singles followed by one two-point pattern for each pair."""
    names: List[str] = []
    patterns: List[np.ndarray] = []
    single_indexes: Dict[Tuple[int, int], int] = {}
    pair_indexes: Dict[str, int] = {}
    for pair_name, (first, second) in pairs.items():
        for point in (first, second):
            if point in single_indexes:
                continue
            x, y = point
            if not (0 <= x < logical_size and 0 <= y < logical_size):
                raise ValueError("two-point probe lies outside the logical DMD")
            pattern = np.zeros((logical_size, logical_size), dtype=np.float32)
            pattern[y, x] = 1.0
            single_indexes[point] = len(patterns)
            names.append("single_%02d_%02d" % (x, y))
            patterns.append(pattern)
        combined = np.zeros((logical_size, logical_size), dtype=np.float32)
        for x, y in (first, second):
            combined[y, x] = 1.0
        pair_indexes[pair_name] = len(patterns)
        names.append("pair_" + pair_name)
        patterns.append(combined)
    return names, np.stack(patterns), single_indexes, pair_indexes


def _normalized_fields(fields: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(fields, dtype=np.float64), 0.0, None)
    totals = values.reshape(len(values), -1).sum(axis=1)
    return values / np.maximum(totals[:, None, None], 1.0e-12)


def _correlation(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if float(left.std()) <= 1.0e-12 or float(right.std()) <= 1.0e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyze_two_point_interference(
    real_fields: np.ndarray,
    simulated_fields: np.ndarray,
    repeat_fields: Sequence[np.ndarray],
    single_indexes: Mapping[Tuple[int, int], int],
    pair_indexes: Mapping[str, int],
    pairs: Mapping[str, Tuple[Tuple[int, int], Tuple[int, int]]] = DEFAULT_PAIRS,
) -> Dict[str, Any]:
    """Compare coherent A+B with the measured/simulated A plus B reference."""
    real = _normalized_fields(real_fields)
    simulated = _normalized_fields(simulated_fields)
    repeats = [_normalized_fields(values) for values in repeat_fields]
    rows: List[Dict[str, Any]] = []
    for pair_name, (first, second) in pairs.items():
        first_index = int(single_indexes[first])
        second_index = int(single_indexes[second])
        pair_index = int(pair_indexes[pair_name])
        real_sum = _normalized_fields(
            (real_fields[first_index] + real_fields[second_index])[None]
        )[0]
        simulated_sum = _normalized_fields(
            (simulated_fields[first_index] + simulated_fields[second_index])[None]
        )[0]
        real_interference = real[pair_index] - real_sum
        simulated_interference = simulated[pair_index] - simulated_sum
        real_strength = float(0.5 * np.abs(real_interference).sum())
        simulated_strength = float(0.5 * np.abs(simulated_interference).sum())
        rows.append(
            {
                "pair": pair_name,
                "points": [list(first), list(second)],
                "repeat_correlation": _correlation(
                    repeats[0][pair_index], repeats[1][pair_index]
                ),
                "coherent_field_correlation": _correlation(
                    real[pair_index], simulated[pair_index]
                ),
                "interference_correlation": _correlation(
                    real_interference, simulated_interference
                ),
                "real_interference_strength": real_strength,
                "simulation_interference_strength": simulated_strength,
                "interference_strength_ratio": real_strength
                / max(simulated_strength, 1.0e-12),
            }
        )

    def finite_mean(key: str) -> Optional[float]:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and np.isfinite(row[key])
        ]
        return float(np.mean(values)) if values else None

    repeat = finite_mean("repeat_correlation")
    interference = finite_mean("interference_correlation")
    return {
        "pairs": rows,
        "mean_repeat_correlation": repeat,
        "mean_coherent_field_correlation": finite_mean(
            "coherent_field_correlation"
        ),
        "mean_interference_correlation": interference,
        "mean_interference_strength_ratio": finite_mean(
            "interference_strength_ratio"
        ),
        "stable": bool(repeat is not None and repeat >= 0.98),
        "score": interference,
    }


def _simulate_patterns(experiment: Any, phase: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    import torch

    from mnist_sdon.optical_classifier import OpticalRegionClassifier
    from mnist_sdon.training import resolve_device

    device = resolve_device(experiment.experiment.device)
    model = OpticalRegionClassifier(
        experiment.optical, experiment.optical_classifier
    ).to(device)
    phase_rad = torch.as_tensor(
        np.asarray(phase, dtype=np.float32) / 255.0 * (2.0 * math.pi),
        device=device,
    )
    model.optical.set_phase_override(phase_rad)
    spikes = torch.as_tensor(
        np.asarray(patterns, dtype=np.float32)[:, None], device=device
    )
    with torch.no_grad():
        intensity = model.optical._intensity(spikes)
    values = intensity.detach().cpu().numpy().astype(np.float32)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    return values


def _new_session(output_dir: Path) -> Path:
    base = "slm_axial_calibration_" + time.strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / base
    suffix = 1
    while candidate.exists():
        candidate = output_dir / (base + "_%02d" % suffix)
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _load_history(session_dir: Path) -> List[Dict[str, Any]]:
    path = session_dir / "summary.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("purpose") != "manual SLM axial two-point interference calibration":
        raise ValueError("selected directory is not an SLM axial calibration session")
    return [dict(row) for row in value.get("measurements", [])]


def _write_curve(session_dir: Path, history: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [
        row
        for row in history
        if row.get("score") is not None and np.isfinite(float(row["score"]))
    ]
    curve_path = session_dir / "axial_score_curve.png"
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    if valid:
        software_scan = all(
            str(row.get("measurement_mode", "manual_z")) == "software_defocus"
            for row in valid
        )
        x_key = "defocus_edge_cycles" if software_scan else "axial_position_mm"
        ordered = sorted(valid, key=lambda row: float(row[x_key]))
        positions = [float(row[x_key]) for row in ordered]
        scores = [float(row["score"]) for row in ordered]
        axis.plot(positions, scores, "o-", color="#1565c0", label="interference corr")
        best = max(ordered, key=lambda row: float(row["score"]))
        axis.scatter(
            [float(best[x_key])],
            [float(best["score"])],
            color="#d32f2f",
            s=70,
            zorder=3,
            label="best",
        )
    axis.axhline(0.0, color="#777", linewidth=0.8)
    axis.set_xlabel(
        "software defocus at pupil edge (phase cycles)"
        if valid and software_scan
        else "SLM relative z (mm; manual label)"
    )
    axis.set_ylabel("mean interference correlation")
    axis.set_ylim(-1.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    return curve_path


def _write_session(
    session_dir: Path,
    history: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> Tuple[Path, Path, Optional[Mapping[str, Any]]]:
    valid = [row for row in history if row.get("score") is not None]
    best = max(valid, key=lambda row: float(row["score"])) if valid else None
    summary = {
        "format_version": 2,
        "purpose": "manual SLM axial two-point interference calibration",
        "explanation": (
            "Manual z values label physical positions; software-defocus values are "
            "wrapped radial quadratic phase cycles at the SLM mid-edge."
        ),
        "settings": dict(settings),
        "measurements": list(history),
        "best_measurement": best,
        "session_path": str(session_dir.resolve()),
    }
    summary_path = session_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = session_dir / "axial_measurements.csv"
    fields = [
        "timestamp",
        "measurement_mode",
        "axial_position_mm",
        "defocus_edge_cycles",
        "score",
        "mean_repeat_correlation",
        "mean_coherent_field_correlation",
        "mean_interference_strength_ratio",
        "measurement_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    curve_path = _write_curve(session_dir, history)
    best_description = "暂无有效测量"
    if best is not None:
        if str(best.get("measurement_mode", "manual_z")) == "software_defocus":
            best_description = "补偿=%+.3f 圈，score=%.4f" % (
                float(best["defocus_edge_cycles"]),
                float(best["score"]),
            )
        else:
            best_description = "z=%.3f mm，score=%.4f" % (
                float(best["axial_position_mm"]),
                float(best["score"]),
            )
    lines = [
        "# SLM 轴向两点干涉标定",
        "",
        "本页既支持手动记录物理 z，也支持保持 SLM 不动、扫描径向二次相位补偿。",
        "评分使用五组 DMD 点对的真实/仿真干涉项平均相关性，越接近 1 越好。",
        "",
        "- 测量次数：%d" % len(history),
        "- 当前最佳：%s" % best_description,
        "- 曲线：%s" % curve_path.name,
        "- 数据：%s" % csv_path.name,
        "",
        "轴向位置确定后，需要重新执行 DMD–SLM X/Y 细扫和 CCD 几何注册。",
    ]
    report_path = session_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, curve_path, best


def run_axial_calibration_measurement(
    manager: Any, request: Dict[str, Any], restore_phase: np.ndarray
) -> None:
    """Capture a manual-z point or an automatic software-defocus coefficient scan."""
    original_mode, original_rectangle = manager.get_alignment_pattern()
    original_phase_enabled = manager.is_slm_phase_enabled()
    correction_getter = getattr(manager, "get_slm_wavefront_correction", None)
    previous_correction = (
        correction_getter()
        if callable(correction_getter)
        else {
            "defocus_edge_cycles": 0.0,
            "source_path": None,
        }
    )
    session_dir: Optional[Path] = None
    try:
        from mnist_sdon.config import load_phase1_config
        from mnist_sdon.hardware_optical import (
            crop_registered_aperture,
            rebase_centered_affine,
        )

        experiment_path = Path(request["experiment_config"]).resolve()
        alignment_path = Path(request["alignment_path"]).resolve()
        if not experiment_path.is_file() or not alignment_path.is_file():
            raise FileNotFoundError("实验配置或对齐配置不存在")
        experiment = load_phase1_config(experiment_path)
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        manager.assert_camera_roi_matches(alignment.get("camera_roi") or {})
        validate_alignment_canvas(manager, alignment)
        if not bool((alignment.get("quality") or {}).get("constraint_satisfied", False)):
            raise RuntimeError("当前 DMD–SLM 对齐配置没有通过质量约束")
        if not manager.has_slm_phase_lut():
            raise RuntimeError("尚未加载有效的 SLM 相位 LUT")

        optical = experiment.optical
        logical_size = int(optical.logical_size)
        simulation_size = int(optical.simulation_size)
        physical_size = int(optical.physical_size)
        dmd_active = int(optical.resolved_dmd_active_size)
        slm_active = int(optical.resolved_slm_active_size)
        ccd_active = int(optical.resolved_ccd_active_size)
        dmd_canvas = int(manager.dmd.modulated_size[0])
        slm_canvas = int(manager.slm.modulated_size[0])
        if (dmd_canvas, slm_canvas) != (dmd_active, slm_active):
            raise RuntimeError(
                "当前 DMD/SLM 画布为 %d/%d，实验要求 %d/%d"
                % (dmd_canvas, slm_canvas, dmd_active, slm_active)
            )
        registration_canvas = physical_size
        transform = rebase_centered_affine(
            np.asarray(alignment["geometry_transform"], dtype=np.float32),
            int(alignment.get("simulation_canvas_size", 512)),
            registration_canvas,
        )
        offset = alignment.get("slm_offset")
        if not isinstance(offset, list) or len(offset) != 2:
            raise RuntimeError("对齐配置缺少 SLM offset")
        manager.set_ccd_orientation(str(alignment.get("ccd_orientation", "identity")))
        manager.update_slm_offset(int(offset[0]), int(offset[1]))
        manager.set_slm_phase_enabled(True)

        correction_setter = getattr(manager, "set_slm_wavefront_correction", None)
        if callable(correction_setter):
            # Every scan coefficient is absolute.  Temporarily disable the
            # accepted global correction so a previous result is not added a
            # second time by the common hardware phase loader.
            correction_setter(0.0, None, reapply=False)

        control_size = int(request["control_size"])
        if simulation_size % control_size:
            raise ValueError("C%d 必须整除 %d 相位网格" % (control_size, simulation_size))
        target_phase = build_weight_link_phases(
            simulation_size, (16,), (control_size,), seed=42
        )["c%d_blocks" % control_size]
        target_phase_display = _centered_nearest(
            target_phase, slm_active, slm_canvas, effective_size=simulation_size
        )
        names, patterns, single_indexes, pair_indexes = build_two_point_patterns(
            logical_size
        )
        per_coefficient = 2 * len(patterns)
        operation = str(request.get("operation", "manual_z"))
        if operation == "software_defocus_scan":
            coefficients = tuple(float(value) for value in request["defocus_coefficients"])
            measurement_mode = "software_defocus"
        elif operation == "manual_z":
            coefficients = (
                float(previous_correction.get("defocus_edge_cycles", 0.0)),
            )
            measurement_mode = "manual_z"
        else:
            raise ValueError("unknown SLM axial calibration operation: %s" % operation)
        if not coefficients:
            raise ValueError("software defocus scan has no coefficients")
        total = per_coefficient * len(coefficients)

        raw_session = str(request.get("session_path", "")).strip()
        session_dir = (
            Path(raw_session).resolve()
            if raw_session
            else _new_session(Path(manager.config.paths.output_dir).resolve())
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        history = _load_history(session_dir)
        manager.update_axial_calibration_status(
            state="running",
            current=0,
            total=total,
            session_path=str(session_dir.resolve()),
            history_rows=history,
            phase=target_phase_display,
            message=(
                "准备扫描 %d 个软件离焦系数" % len(coefficients)
                if measurement_mode == "software_defocus"
                else "正在加载固定 C%d 块状相位" % control_size
            ),
        )

        settle_s = max(
            float(manager.config.timing.slm_min_interval_s),
            float(request["settle_ms"]) / 1000.0,
        )
        samples = int(request["samples"])
        dmd_effective = simulation_size * dmd_active // physical_size
        black = _centered_nearest(
            np.zeros((logical_size, logical_size), dtype=np.uint8),
            dmd_active,
            dmd_canvas,
            effective_size=dmd_effective,
        )
        simulated = _simulate_patterns(experiment, target_phase, patterns)
        current = 0
        last_raw: Optional[np.ndarray] = None
        last_registered: Optional[np.ndarray] = None
        latest_measurement: Optional[Dict[str, Any]] = None
        report_path: Optional[Path] = None
        curve_path: Optional[Path] = None
        best: Optional[Mapping[str, Any]] = None
        for coefficient_index, coefficient in enumerate(coefficients, start=1):
            if manager.is_axial_calibration_cancelled():
                raise InterruptedError("用户停止了 SLM 轴向标定")
            applied_phase_display = apply_radial_defocus_correction(
                target_phase_display, coefficient
            )
            manager.update_slm_image(applied_phase_display, apply_lut=True)
            if not manager.wait_for_slm_update(2.0):
                raise RuntimeError("等待 SLM 相位更新超时")
            time.sleep(settle_s)

            manager.update_dmd_image(black, apply_lut=False)
            if not manager.wait_for_dmd_update(2.0):
                raise RuntimeError("等待 DMD 黑场更新超时")
            time.sleep(settle_s)
            background = capture_median(
                manager, max(3, samples), (registration_canvas, registration_canvas)
            )
            if background is None:
                raise RuntimeError("CCD 没有返回相位对应黑场")
            background = np.asarray(background)

            repetitions: List[np.ndarray] = []
            for repeat, order in enumerate(
                (np.arange(len(patterns)), np.arange(len(patterns) - 1, -1, -1)),
                start=1,
            ):
                captured: List[np.ndarray] = []
                for pattern_index in order:
                    if manager.is_axial_calibration_cancelled():
                        raise InterruptedError("用户停止了 SLM 轴向标定")
                    logical = np.asarray(
                        patterns[int(pattern_index)] * 255.0, dtype=np.uint8
                    )
                    display = _centered_nearest(
                        logical,
                        dmd_active,
                        dmd_canvas,
                        effective_size=dmd_effective,
                    )
                    manager.update_dmd_image(display, apply_lut=True)
                    if not manager.wait_for_dmd_update(2.0):
                        raise RuntimeError("等待 DMD 点对更新超时")
                    time.sleep(settle_s)
                    raw = capture_median(
                        manager, samples, (registration_canvas, registration_canvas)
                    )
                    if raw is None:
                        raise RuntimeError("CCD 没有返回两点干涉图像")
                    corrected = np.clip(
                        np.asarray(raw, dtype=np.float32)
                        - np.asarray(background, dtype=np.float32),
                        0.0,
                        None,
                    )
                    registered = _register_frame(
                        corrected,
                        np.zeros_like(corrected),
                        str(alignment.get("ccd_orientation", "identity")),
                        transform,
                        registration_canvas,
                    )
                    active = crop_registered_aperture(registered, ccd_active)
                    captured.append(np.asarray(active, dtype=np.float32))
                    last_raw = np.asarray(raw).copy()
                    last_registered = np.asarray(registered).copy()
                    current += 1
                    description = (
                        "软件补偿 %+.3f 圈" % coefficient
                        if measurement_mode == "software_defocus"
                        else "位置 %.3f mm" % float(request["axial_position_mm"])
                    )
                    manager.update_axial_calibration_status(
                        state="running",
                        current=current,
                        total=total,
                        dmd=logical,
                        phase=applied_phase_display,
                        ccd=last_raw,
                        registered=last_registered,
                        message="%s（%d/%d）：采集 %d/%d，%s，第 %d 轮"
                        % (
                            description,
                            coefficient_index,
                            len(coefficients),
                            current,
                            total,
                            names[int(pattern_index)],
                            repeat,
                        ),
                    )
                values = np.stack(captured)
                repetitions.append(values if repeat == 1 else values[::-1])

            averaged = np.mean(repetitions, axis=0).astype(np.float32)
            real_32 = np.stack(
                [
                    cv2.resize(field, (32, 32), interpolation=cv2.INTER_AREA)
                    for field in averaged
                ]
            )
            analysis = analyze_two_point_interference(
                real_32,
                simulated,
                [
                    np.stack(
                        [
                            cv2.resize(field, (32, 32), interpolation=cv2.INTER_AREA)
                            for field in values
                        ]
                    )
                    for values in repetitions
                ],
                single_indexes,
                pair_indexes,
            )

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dirname = (
                "defocus_%+.3fturn_%s" % (coefficient, timestamp)
                if measurement_mode == "software_defocus"
                else "z_%+.3fmm_%s"
                % (float(request["axial_position_mm"]), timestamp)
            )
            measurement_dir = session_dir / dirname
            measurement_dir.mkdir()
            np.save(measurement_dir / "registered_fields.npy", averaged)
            np.save(measurement_dir / "registered_fields_32.npy", real_32)
            np.save(measurement_dir / "simulation_fields_32.npy", simulated)
            np.save(measurement_dir / "target_phase_uint8.npy", target_phase)
            np.save(
                measurement_dir / "applied_phase_uint8.npy", applied_phase_display
            )
            measurement = {
                "timestamp": timestamp,
                "measurement_mode": measurement_mode,
                "axial_position_mm": float(request["axial_position_mm"]),
                "defocus_edge_cycles": float(coefficient),
                "score": analysis["score"],
                "mean_repeat_correlation": analysis["mean_repeat_correlation"],
                "mean_coherent_field_correlation": analysis[
                    "mean_coherent_field_correlation"
                ],
                "mean_interference_strength_ratio": analysis[
                    "mean_interference_strength_ratio"
                ],
                "stable": analysis["stable"],
                "pair_results": analysis["pairs"],
                "measurement_path": str(measurement_dir.resolve()),
            }
            (measurement_dir / "summary.json").write_text(
                json.dumps(measurement, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            history.append(measurement)
            latest_measurement = measurement
            report_path, curve_path, best = _write_session(
                session_dir,
                history,
                {
                    **dict(request),
                    "experiment_config": str(experiment_path),
                    "alignment_path": str(alignment_path),
                    "phase_kind": "deterministic C%d binary 0/pi blocks"
                    % control_size,
                    "defocus_definition": (
                        "wrapped radial quadratic phase; coefficient is cycles at "
                        "the horizontal/vertical SLM mid-edge"
                    ),
                    "pairs": DEFAULT_PAIRS,
                },
            )
            manager.update_axial_calibration_status(
                state="running",
                current=current,
                total=total,
                message="系数 %+.3f 圈完成：干涉相关性=%s"
                % (
                    coefficient,
                    "—" if analysis["score"] is None else "%.4f" % analysis["score"],
                ),
                session_path=str(session_dir.resolve()),
                report_path=str(report_path.resolve()),
                curve_path=str(curve_path.resolve()),
                history_rows=history,
                latest_measurement=measurement,
                best_measurement=best,
            )

        best_text = (
            (
                "当前最佳补偿=%+.3f 圈，干涉相关性=%.4f"
                % (float(best["defocus_edge_cycles"]), float(best["score"]))
                if str(best.get("measurement_mode", "manual_z"))
                == "software_defocus"
                else "当前最佳 z=%.3f mm，干涉相关性=%.4f"
                % (float(best["axial_position_mm"]), float(best["score"]))
            )
            if best is not None
            else "暂无有效最佳点"
        )
        manager.update_axial_calibration_status(
            state="complete",
            current=total,
            total=total,
            message=(
                "软件离焦扫描完成；%s。确认后再点击“设最佳为当前补偿”。" % best_text
                if measurement_mode == "software_defocus"
                else "当前位置采集完成；%s" % best_text
            ),
            session_path=str(session_dir.resolve()),
            report_path=str(report_path.resolve()) if report_path else "",
            curve_path=str(curve_path.resolve()) if curve_path else "",
            history_rows=history,
            latest_measurement=latest_measurement,
            best_measurement=best,
            dmd=np.asarray(patterns[-1] * 255.0, dtype=np.uint8),
            phase=(
                apply_radial_defocus_correction(
                    target_phase_display,
                    float(latest_measurement["defocus_edge_cycles"]),
                )
                if latest_measurement is not None
                else target_phase_display
            ),
            ccd=last_raw,
            registered=last_registered,
        )
    except InterruptedError as exc:
        manager.update_axial_calibration_status(state="cancelled", message=str(exc))
    except Exception as exc:
        manager.update_axial_calibration_status(
            state="error", message="SLM 轴向标定失败：%s" % exc
        )
    finally:
        correction_setter = getattr(manager, "set_slm_wavefront_correction", None)
        if callable(correction_setter):
            correction_setter(
                float(previous_correction.get("defocus_edge_cycles", 0.0)),
                previous_correction.get("source_path"),
                reapply=False,
            )
        manager.update_slm_image(restore_phase, apply_lut=False)
        manager.set_slm_phase_enabled(original_phase_enabled)
        dmd_size = int(manager.config.display.dmd_modulated_size[0])
        manager.update_dmd_image(
            build_alignment_pattern(
                original_mode, size=dmd_size, rectangle_size=original_rectangle
            )
        )
