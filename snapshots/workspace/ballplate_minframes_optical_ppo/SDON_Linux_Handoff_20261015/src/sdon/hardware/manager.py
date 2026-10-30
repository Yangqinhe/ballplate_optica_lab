"""Shared, thread-safe ownership of DMD, SLM and CCD devices."""

import threading
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from sdon.config import AppConfig

from .flir_camera import FlirCamera
from .modulator import OpticalModulator
from .display_timing import read_display_timing, display_mode_signature, assert_display_mode_matches


class HardwareManager:
    def __init__(
        self,
        config: AppConfig,
        dmd_screen: QtGui.QScreen,
        slm_screen: QtGui.QScreen,
        camera: Optional[FlirCamera] = None,
    ) -> None:
        self.config = config
        self._display_role_sources={'dmd':dmd_screen.name(),'slm':slm_screen.name()}
        self._session_display_mode=self.get_display_mode_signature()
        self.dmd = OpticalModulator(
            dmd_screen,
            config.display.dmd_modulated_size,
            "DMD",
            config.display.dmd_offset,
        )
        self.slm = OpticalModulator(
            slm_screen,
            config.display.slm_modulated_size,
            "SLM",
            config.display.slm_offset,
        )
        self.dmd.present()
        self.slm.present()
        self._flush_qt_events_if_owner()
        if camera is None:
            # Torch 1.13 and the installed PySpin build conflict on Windows when
            # PySpin is imported first. Preload Torch before opening the camera.
            from sdon.simulation import initialize_torch_runtime

            initialize_torch_runtime()
            self.camera = FlirCamera(config.camera)
        else:
            self.camera = camera

        self._pattern_lock = threading.Lock()
        self._pattern_mode = "multiscale_fiducial"
        self._rectangle_size = 32
        self._orientation_lock = threading.Lock()
        self._orientation = "auto"
        self._phase_lock = threading.Lock()
        self._phase_enabled = True
        self._last_phase: Optional[np.ndarray] = None
        self._last_phase_input: Optional[np.ndarray] = None
        self._last_phase_apply_lut = True
        self._slm_phase_lut: Optional[np.ndarray] = None
        self._slm_phase_lut_path: Optional[str] = None
        self._slm_wavefront_defocus_edge_cycles = 0.0
        self._slm_wavefront_correction_path: Optional[str] = None
        self._lut_lock = threading.Lock()
        self._dmd_lut: Optional[np.ndarray] = None
        self._preview_lock = threading.Lock()
        self._preview_sequence = 0
        self._preview_frames: Dict[str, Any] = {
            "ccd": None,
            "sim": None,
            "difference": None,
            "message": "等待 CCD 图像",
        }
        self._live_preview_lock = threading.Lock()
        self._live_preview_enabled = False
        self._live_preview_mode = "off"

        self._controller_lock = threading.Lock()
        self._controller_check_requested = False
        self._controller_status = {"state": "idle", "message": "DMD USB 未检查（不是通过）"}

        self._roi_lock = threading.Lock()
        self._roi_request: Optional[Dict[str, Any]] = None
        self._roi_status: Dict[str, Any] = {
            "state": "idle",
            "message": "尚未进行自动 ROI 标定",
        }

        self._intensity_lock = threading.Lock()
        self._intensity_request: Optional[Dict[str, Any]] = None
        self._intensity_cancelled = False
        self._intensity_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 256,
            "message": "未开始光强标定",
        }
        self._alignment_lock = threading.Lock()
        self._alignment_request: Optional[Dict[str, Any]] = None
        self._alignment_cancelled = False
        self._alignment_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 0,
            "message": "未开始对齐",
        }

        # A scan/evaluation is only an attempt. Keep the last known-good
        # configuration separately so a failed re-check cannot lock every
        # downstream page or erase a reusable calibration.
        self._active_alignment_status: Dict[str, Any] = {
            "constraint_satisfied": False,
            "config_path": None,
            "loaded_configuration": None,
            "message": "no active alignment configuration",
        }

        self._slm_check_lock = threading.Lock()
        self._slm_check_request: Optional[Dict[str, Any]] = None
        self._slm_check_cancelled = False
        self._slm_check_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 6,
            "message": "尚未开始 SLM 快速自检",
            "frames": {},
        }

        self._phase_calibration_lock = threading.Lock()
        self._phase_calibration_request: Optional[Dict[str, Any]] = None
        self._phase_calibration_cancelled = False
        self._phase_calibration_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 262,
            "message": "尚未开始 SLM 灰度—物理相位标定",
        }

        self._weight_link_lock = threading.Lock()
        self._weight_link_request: Optional[Dict[str, Any]] = None
        self._weight_link_cancelled = False
        self._weight_link_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 0,
            "message": "尚未开始 DMD–SLM 权重链路验证",
            "result_rows": [],
        }

        self._axial_lock = threading.Lock()
        self._axial_request: Optional[Dict[str, Any]] = None
        self._axial_cancelled = False
        self._axial_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 30,
            "message": "尚未开始 SLM 轴向两点干涉标定",
            "history_rows": [],
        }

        self._sdon_lock = threading.Lock()
        self._sdon_request: Optional[Dict[str, Any]] = None
        self._sdon_cancelled = False
        self._sdon_status: Dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 0,
            "message": "尚未开始纯光 SDON 验证",
        }

    @staticmethod
    def _flush_qt_events_if_owner() -> bool:
        """Synchronize queued paints when a CLI owns the Qt GUI thread.

        The normal workbench has a continuously running Qt event loop. Hardware
        command-line tools do not, so without this flush Windows can keep showing
        the desktop or an old frame even though QLabel.setPixmap() has returned.
        Worker threads deliberately skip the flush; the workbench main thread
        services their queued signals through its normal event loop.
        """
        app = QtWidgets.QApplication.instance()
        if app is None or QtCore.QThread.currentThread() is not app.thread():
            return False
        QtCore.QCoreApplication.sendPostedEvents(None, 0)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        QtGui.QGuiApplication.sync()
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        return True

    def update_dmd_image(self, image: np.ndarray, apply_lut: bool = True) -> None:
        prepared = np.asarray(image)
        if apply_lut:
            with self._lut_lock:
                lut = None if self._dmd_lut is None else self._dmd_lut.copy()
            if lut is not None:
                prepared = lut[np.clip(np.rint(prepared), 0, 255).astype(np.uint8)]
        self.dmd.update_image(np.asarray(prepared, dtype=np.uint8))
        self._flush_qt_events_if_owner()

    def set_dmd_linearization_lut(self, lut: np.ndarray) -> None:
        values = np.asarray(lut, dtype=np.float32).reshape(-1)
        if values.size != 256 or not np.all(np.isfinite(values)):
            raise ValueError("DMD linearization LUT must contain 256 finite values")
        with self._lut_lock:
            self._dmd_lut = np.clip(np.rint(values), 0, 255).astype(np.uint8)

    def update_slm_image(self, image: np.ndarray, apply_lut: bool = True) -> None:
        source = np.asarray(image, dtype=np.uint8)
        with self._phase_lock:
            correction = (
                float(self._slm_wavefront_defocus_edge_cycles) if apply_lut else 0.0
            )
            if abs(correction) > 1.0e-12:
                from sdon.calibration.axial import apply_radial_defocus_correction

                source_for_lut = apply_radial_defocus_correction(source, correction)
            else:
                source_for_lut = source
            physical = (
                self._slm_phase_lut[source_for_lut]
                if apply_lut and self._slm_phase_lut is not None
                else source_for_lut
            )
            prepared = np.ascontiguousarray(np.flip(physical, 1), dtype=np.uint8)
            self._last_phase_input = source.copy()
            self._last_phase_apply_lut = bool(apply_lut)
            self._last_phase = prepared.copy()
            output = prepared if self._phase_enabled else np.zeros_like(prepared)
        self.slm.update_image(output)
        self._flush_qt_events_if_owner()

    def set_slm_wavefront_correction(
        self,
        defocus_edge_cycles: float,
        source_path: Optional[str] = None,
        reapply: bool = True,
    ) -> None:
        """Set the accepted radial correction applied before the SLM phase LUT."""
        coefficient = float(defocus_edge_cycles)
        if not np.isfinite(coefficient) or abs(coefficient) > 16.0:
            raise ValueError("SLM defocus edge cycles must be finite and within +/-16")
        with self._phase_lock:
            self._slm_wavefront_defocus_edge_cycles = coefficient
            self._slm_wavefront_correction_path = (
                None if source_path in {None, ""} else str(source_path)
            )
            source = None if self._last_phase_input is None else self._last_phase_input.copy()
            should_reapply = bool(reapply and self._last_phase_apply_lut)
        if source is not None and should_reapply:
            self.update_slm_image(source, apply_lut=True)

    def get_slm_wavefront_correction(self) -> Dict[str, Any]:
        with self._phase_lock:
            return {
                "kind": "radial_quadratic_defocus",
                "defocus_edge_cycles": float(
                    self._slm_wavefront_defocus_edge_cycles
                ),
                "source_path": self._slm_wavefront_correction_path,
            }

    def clear_slm_wavefront_correction(self, reapply: bool = True) -> None:
        self.set_slm_wavefront_correction(0.0, None, reapply=reapply)

    def set_slm_phase_lut(self, lut: np.ndarray) -> None:
        values = np.asarray(lut, dtype=np.float64).reshape(-1)
        if values.size != 256 or not np.all(np.isfinite(values)):
            raise ValueError("SLM phase LUT must contain 256 finite values")
        physical_lut = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        if np.any(np.diff(physical_lut.astype(np.int16)) < 0):
            raise ValueError("SLM phase LUT must be monotonic")
        with self._phase_lock:
            self._slm_phase_lut = physical_lut
            source = None if self._last_phase_input is None else self._last_phase_input.copy()
            reapply = self._last_phase_apply_lut
        if source is not None and reapply:
            self.update_slm_image(source, apply_lut=True)

    def set_slm_phase_lut_source_path(self, path: Optional[str]) -> None:
        """Remember which validated LUT is active, independently of GUI job status."""
        with self._phase_lock:
            self._slm_phase_lut_path = None if path in {None, ""} else str(path)

    def get_slm_phase_lut_source_path(self) -> Optional[str]:
        with self._phase_lock:
            return self._slm_phase_lut_path

    def has_slm_phase_lut(self) -> bool:
        with self._phase_lock:
            return self._slm_phase_lut is not None

    def get_slm_phase_lut(self) -> Optional[np.ndarray]:
        with self._phase_lock:
            return None if self._slm_phase_lut is None else self._slm_phase_lut.copy()

    def get_last_slm_input(self) -> Optional[Tuple[np.ndarray, bool]]:
        """Return the logical SLM image so a temporary test can restore it."""
        with self._phase_lock:
            if self._last_phase_input is None:
                return None
            return self._last_phase_input.copy(), bool(self._last_phase_apply_lut)

    def clear_slm_phase_lut(self) -> None:
        with self._phase_lock:
            self._slm_phase_lut = None
            self._slm_phase_lut_path = None

    def set_slm_phase_enabled(self, enabled: bool) -> None:
        with self._phase_lock:
            self._phase_enabled = bool(enabled)
            if self._phase_enabled and self._last_phase is not None:
                output = self._last_phase.copy()
            else:
                width, height = self.slm.modulated_size
                output = np.zeros((height, width), dtype=np.uint8)
        self.slm.update_image(output)
        self._flush_qt_events_if_owner()

    def is_slm_phase_enabled(self) -> bool:
        with self._phase_lock:
            return self._phase_enabled

    def update_slm_offset(self, x: int, y: int) -> None:
        self.slm.update_offset(x, y)
        self._flush_qt_events_if_owner()

    def wait_for_dmd_update(self, timeout: float = 1.0) -> bool:
        self._flush_qt_events_if_owner()
        applied = self.dmd.wait_until_applied(timeout)
        self._flush_qt_events_if_owner()
        return applied

    def wait_for_slm_update(self, timeout: float = 1.0) -> bool:
        self._flush_qt_events_if_owner()
        applied = self.slm.wait_until_applied(timeout)
        self._flush_qt_events_if_owner()
        return applied

    def get_display_status(self) -> Dict[str, Any]:
        """Return auditable evidence for the latest DMD and SLM frames."""
        return {
            "refresh_policy": "fullscreen_qt_flush_sequence_sha256_v1",
            "dmd": self.dmd.display_status(),
            "slm": self.slm.display_status(),
        }

    def set_alignment_pattern(self, mode: str, rectangle_size: int = 32) -> None:
        from sdon.calibration.patterns import PATTERN_MODES, build_alignment_pattern

        if mode not in PATTERN_MODES:
            raise ValueError("unknown alignment pattern: %s" % mode)
        with self._pattern_lock:
            self._pattern_mode = mode
            self._rectangle_size = max(1, int(rectangle_size))
        # Pattern selection is a device control, so apply it immediately. The
        # live preview worker may refresh the same frame later, but is not
        # required for switching DMD content.
        size = getattr(
            self.config.display,
            "dmd_modulated_size",
            self.config.display.modulated_size,
        )[0]
        self.update_dmd_image(
            build_alignment_pattern(mode, size=size, rectangle_size=rectangle_size)
        )

    def get_alignment_pattern(self) -> Tuple[str, int]:
        with self._pattern_lock:
            return self._pattern_mode, self._rectangle_size

    def set_ccd_orientation(self, mode: str) -> None:
        if mode not in {"identity", "flip_horizontal", "flip_vertical", "rotate_180", "auto"}:
            raise ValueError("unknown CCD orientation: %s" % mode)
        with self._orientation_lock:
            self._orientation = mode

    def get_ccd_orientation(self) -> str:
        with self._orientation_lock:
            return self._orientation

    def get_camera_roi(self) -> Dict[str, int]:
        parameters = self.camera.get_camera_parameters()
        names = ("OffsetX", "OffsetY", "Width", "Height")
        missing = [name for name in names if parameters.get(name) is None]
        if missing:
            raise RuntimeError("camera ROI readback is unavailable: %s" % ", ".join(missing))
        return {name: int(parameters[name]) for name in names}

    def assert_camera_roi_matches(self, expected: Dict[str, Any]) -> None:
        current = self.get_camera_roi()
        normalized = {name: int(expected[name]) for name in current if name in expected}
        if set(normalized) != set(current):
            raise ValueError("saved calibration does not contain a complete camera ROI")
        if normalized != current:
            raise ValueError("camera ROI changed: saved=%s, current=%s" % (normalized, current))

    def set_camera_roi(self, requested: Dict[str, Any]) -> Dict[str, int]:
        """Apply a runtime ROI and invalidate calibrations tied to the old one."""
        before = self.get_camera_roi()
        actual = self.camera.set_roi(
            int(requested["OffsetX"]),
            int(requested["OffsetY"]),
            int(requested["Width"]),
            int(requested["Height"]),
        )
        self.config = replace(
            self.config,
            camera=replace(
                self.config.camera,
                roi_offset_x=actual["OffsetX"],
                roi_offset_y=actual["OffsetY"],
                roi_width=actual["Width"],
                roi_height=actual["Height"],
            ),
        )
        if actual != before:
            self._invalidate_roi_dependent_calibrations()
        return actual

    def _invalidate_roi_dependent_calibrations(self) -> None:
        with self._lut_lock:
            self._dmd_lut = None
        self.clear_slm_phase_lut()
        self.clear_slm_wavefront_correction(reapply=False)
        with self._intensity_lock:
            self._intensity_status = {
                "state": "idle",
                "current": 0,
                "total": 256,
                "message": "相机 ROI 已更新，请重新进行光强标定",
            }
        with self._alignment_lock:
            self._alignment_status = {
                "state": "idle",
                "current": 0,
                "total": 0,
                "message": "相机 ROI 已更新，请重新进行 DMD–SLM 对齐",
            }
        with self._alignment_lock:
            self._active_alignment_status = {
                "constraint_satisfied": False,
                "config_path": None,
                "loaded_configuration": None,
                "message": "camera ROI changed; alignment is invalid",
            }
        with self._slm_check_lock:
            self._slm_check_status = {
                "state": "idle",
                "current": 0,
                "total": 6,
                "message": "相机 ROI 已更新，旧 SLM 自检结果已失效",
                "frames": {},
            }
        with self._phase_calibration_lock:
            self._phase_calibration_status = {
                "state": "idle",
                "current": 0,
                "total": 262,
                "message": "相机 ROI 已更新，旧 SLM 相位 LUT 已失效",
            }
        if hasattr(self, "_weight_link_lock"):
            with self._weight_link_lock:
                self._weight_link_status = {
                    "state": "idle",
                    "current": 0,
                    "total": 0,
                    "message": "相机 ROI 已更新，旧权重链路验证结果已失效",
                    "result_rows": [],
                }
        if hasattr(self, "_axial_lock"):
            with self._axial_lock:
                self._axial_status = {
                    "state": "idle",
                    "current": 0,
                    "total": 30,
                    "message": "相机 ROI 已更新，旧轴向干涉结果已失效",
                    "history_rows": [],
                }

    def request_camera_roi_calibration(
        self, settle_ms: int = 180, samples: int = 3, margin_ratio: float = 0.20
    ) -> bool:
        self.assert_dmd_controller_ready()
        config_path = getattr(self, "config_path", None)
        if config_path is None:
            raise RuntimeError("camera ROI config path is unavailable")
        request = {
            "settle_ms": max(20, min(2000, int(settle_ms))),
            "samples": max(1, min(9, int(samples))),
            "margin_ratio": max(0.05, min(0.50, float(margin_ratio))),
            "config_path": str(config_path),
        }
        with self._roi_lock:
            if self._roi_status["state"] in {"queued", "running"}:
                return False
            self._roi_request = request
            self._roi_status = {
                "state": "queued",
                "message": "等待切换最大相机 ROI，再显示强度方块并自动定位",
            }
        return True

    def take_camera_roi_request(self) -> Optional[Dict[str, Any]]:
        with self._roi_lock:
            request, self._roi_request = self._roi_request, None
            if request is not None:
                self._roi_status.update(
                    state="running", message="正在准备切换相机最大 ROI"
                )
            return request

    def update_roi_status(self, **values: Any) -> None:
        with self._roi_lock:
            self._roi_status.update(values)

    def get_roi_status(self) -> Dict[str, Any]:
        with self._roi_lock:
            return dict(self._roi_status)

    @staticmethod
    def apply_ccd_orientation(image: np.ndarray, mode: str) -> np.ndarray:
        if mode == "flip_horizontal":
            return cv2.flip(image, 1)
        if mode == "flip_vertical":
            return cv2.flip(image, 0)
        if mode == "rotate_180":
            return cv2.flip(image, -1)
        return image

    def capture_ccd_image(self, output_size: Tuple[int, int]) -> Optional[np.ndarray]:
        self.assert_dmd_controller_ready()
        if getattr(self,'_session_display_mode',None) is not None:
            assert_display_mode_matches(self._session_display_mode,self.get_display_mode_signature())
        image = self.camera.get_frame()
        if image is None or np.asarray(image).size == 0:
            return None
        image = np.asarray(image)
        rows, cols = image.shape[:2]
        rotation = cv2.getRotationMatrix2D(
            (cols / 2.0, rows / 2.0), self.config.camera.rotation_deg, 1.0
        )
        image = cv2.warpAffine(image, rotation, (cols, rows))
        c = self.config.camera
        bottom = rows - c.crop_bottom if c.crop_bottom else rows
        right = cols - c.crop_right if c.crop_right else cols
        image = image[c.crop_top : bottom, c.crop_left : right]
        if image.size == 0:
            raise RuntimeError("camera crop settings produced an empty image")
        orientation = self.get_ccd_orientation()
        if orientation != "auto":
            image = self.apply_ccd_orientation(image, orientation)

        side = min(image.shape[:2])
        x0 = (image.shape[1] - side) // 2
        y0 = (image.shape[0] - side) // 2
        roi = image[y0 : y0 + side, x0 : x0 + side]
        resized = cv2.resize(roi, output_size, interpolation=cv2.INTER_AREA)
        if c.show_preview:
            self.publish_preview(ccd=resized, message="CCD 已更新")
        return resized

    def get_dmd_controller_status(self) -> Dict[str, Any]:
        if not hasattr(self, '_controller_lock'):
            return {'state': 'idle', 'message': 'DMD USB 未检查（不是通过）'}
        with self._controller_lock:
            return deepcopy(self._controller_status)

    def assert_dmd_controller_ready(self) -> None:
        status = self.get_dmd_controller_status()
        if status['state'] in {'queued', 'running', 'failed', 'error'}:
            raise RuntimeError(status['message'] + '；请在首页重新完成 DMD 控制器只读检查')
        if status['state'] == 'complete':
            assert_display_mode_matches(status['display_mode'], self.get_display_mode_signature())

    def request_dmd_controller_check(self) -> bool:
        # The single worker serializes this with all camera jobs. Reject an
        # already queued/running job rather than interrupting a measurement.
        for getter in (self.get_roi_status, self.get_intensity_status, self.get_alignment_status,
                       self.get_slm_check_status, self.get_phase_calibration_status,
                       self.get_weight_link_status, self.get_axial_calibration_status,
                       self.get_sdon_status):
            if getter().get('state') in {'queued', 'running'}:
                return False
        with self._controller_lock:
            if self._controller_status['state'] in {'queued', 'running'}:
                return False
            self._controller_check_requested = True
            self._controller_status = {'state': 'queued', 'message': '等待串行工作线程读取 DMD USB 状态'}
        return True

    def take_dmd_controller_check(self) -> bool:
        with self._controller_lock:
            requested, self._controller_check_requested = self._controller_check_requested, False
            if requested:
                self._controller_status = {'state': 'running', 'message': '正在只读检查 DMD（不采相机）'}
            return requested

    def update_dmd_controller_status(self, **values: Any) -> None:
        with self._controller_lock:
            self._controller_status.update(values)

    def get_display_mode_signature(self) -> Dict[str, Any]:
        current=display_mode_signature(read_display_timing(),self._display_role_sources)
        reference=getattr(self,'_session_display_mode',None)
        if reference is not None:assert_display_mode_matches(reference,current)
        return current

    def publish_preview(self, **values: Any) -> None:
        """Publish immutable preview frames for polling from the Qt thread."""
        with self._preview_lock:
            for key, value in values.items():
                if key in {"ccd", "sim", "difference"} and value is not None:
                    self._preview_frames[key] = np.asarray(value).copy()
                else:
                    self._preview_frames[key] = value
            self._preview_sequence += 1

    def get_preview_snapshot(self) -> Dict[str, Any]:
        with self._preview_lock:
            snapshot = dict(self._preview_frames)
            for key in ("ccd", "sim", "difference"):
                if snapshot.get(key) is not None:
                    snapshot[key] = snapshot[key].copy()
            snapshot["sequence"] = self._preview_sequence
            return snapshot

    def set_live_comparison_enabled(self, enabled: bool) -> None:
        self.set_live_preview_mode("comparison", enabled)

    def set_live_ccd_enabled(self, enabled: bool) -> None:
        """Enable the lightweight page-1 CCD stream without running simulation."""
        self.set_live_preview_mode("ccd", enabled)

    def set_axial_live_preview_enabled(self, enabled: bool) -> None:
        """Enable the fixed two-point/C50 preview used while moving the SLM."""
        self.set_live_preview_mode("axial", enabled)

    def set_live_preview_mode(self, mode: str, enabled: bool) -> None:
        if mode not in {"ccd", "comparison", "axial"}:
            raise ValueError("unknown live preview mode: %s" % mode)
        with self._live_preview_lock:
            requested = bool(enabled) and self.config.camera.show_preview
            if requested:
                self._live_preview_mode = mode
            elif self._live_preview_mode == mode:
                self._live_preview_mode = "off"
            # Keep the original flag for lightweight test managers and older
            # callers, but make its meaning specific to SIM_VS_CCD.
            self._live_preview_enabled = self._live_preview_mode == "comparison"

    def get_live_preview_mode(self) -> str:
        with self._live_preview_lock:
            fallback = (
                "comparison"
                if bool(getattr(self, "_live_preview_enabled", False))
                else "off"
            )
            return str(getattr(self, "_live_preview_mode", fallback))

    def is_live_comparison_enabled(self) -> bool:
        return self.get_live_preview_mode() == "comparison"

    def request_intensity_calibration(
        self, settle_ms: int, samples: int, output_dir: Optional[str] = None
    ) -> bool:
        self.assert_dmd_controller_ready()
        request = {
            "settle_ms": max(10, int(settle_ms)),
            "samples": max(1, min(9, int(samples))),
            "output_dir": output_dir,
        }
        with self._intensity_lock:
            if self._intensity_status["state"] in {"queued", "running"}:
                return False
            self._intensity_request = request
            self._intensity_cancelled = False
            self._intensity_status = {
                "state": "queued",
                "current": 0,
                "total": 256,
                "message": "等待开始光强标定",
            }
        return True

    def take_intensity_request(self) -> Optional[Dict[str, Any]]:
        with self._intensity_lock:
            request, self._intensity_request = self._intensity_request, None
            if request is not None:
                self._intensity_status.update(state="running", message="正在扫描 DMD 灰度")
            return request

    def cancel_intensity_calibration(self) -> None:
        with self._intensity_lock:
            self._intensity_cancelled = True

    def is_intensity_cancelled(self) -> bool:
        with self._intensity_lock:
            return self._intensity_cancelled

    def update_intensity_status(self, **values: Any) -> None:
        with self._intensity_lock:
            self._intensity_status.update(values)

    def get_intensity_status(self) -> Dict[str, Any]:
        with self._intensity_lock:
            return dict(self._intensity_status)

    def request_alignment(self, **request: Any) -> bool:
        self.assert_dmd_controller_ready()
        normalized = dict(request)
        scan_mode = str(request.get("scan_mode", "auto"))
        if scan_mode not in {"auto", "fine"}:
            raise ValueError("unknown alignment scan mode: %s" % scan_mode)
        normalized.update(
            center_x=int(request["center_x"]),
            center_y=int(request["center_y"]),
            radius=max(1, int(request["radius"])),
            coarse_step=max(1, int(request["coarse_step"])),
            fine_radius=max(1, int(request.get("fine_radius", 5))),
            fine_step=max(1, int(request.get("fine_step", 1))),
            settle_ms=max(10, int(request["settle_ms"])),
            max_registration_shift=max(0.0, float(request["max_registration_shift"])),
            minimum_modulation=max(0.0, float(request["minimum_modulation"])),
            evaluate_only=bool(request.get("evaluate_only", False)),
            scan_mode=scan_mode,
        )
        with self._alignment_lock:
            if self._alignment_status["state"] in {"queued", "running"}:
                return False
            self._alignment_request = normalized
            self._alignment_cancelled = False
            self._alignment_status = {
                "state": "queued",
                "current": 0,
                "total": 0,
                "operation": "evaluate" if normalized["evaluate_only"] else scan_mode,
                "message": "等待开始对齐",
            }
        return True

    def take_alignment_request(self) -> Optional[Dict[str, Any]]:
        with self._alignment_lock:
            request, self._alignment_request = self._alignment_request, None
            if request is not None:
                self._alignment_status.update(state="running", message="正在准备仿真参考图")
            return request

    def cancel_alignment(self) -> None:
        with self._alignment_lock:
            self._alignment_cancelled = True

    def is_alignment_cancelled(self) -> bool:
        with self._alignment_lock:
            return self._alignment_cancelled

    def update_alignment_status(self, **values: Any) -> None:
        with self._alignment_lock:
            self._alignment_status.update(values)

    def get_alignment_status(self) -> Dict[str, Any]:
        with self._alignment_lock:
            return dict(self._alignment_status)

    def set_active_alignment_configuration(
        self, config_path: str, configuration: Dict[str, Any]
    ) -> None:
        """Record a validated alignment independently from the latest attempt."""
        quality = configuration.get("quality") or {}
        if not bool(quality.get("constraint_satisfied", False)):
            raise ValueError("cannot activate an alignment that failed its constraints")
        offset = configuration.get("slm_offset")
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("active alignment has no valid slm_offset")
        orientation = str(configuration.get("ccd_orientation", "identity"))
        with self._alignment_lock:
            self._active_alignment_status = {
                "constraint_satisfied": True,
                "config_path": str(config_path),
                "loaded_configuration": deepcopy(configuration),
                "best_score": quality.get("score"),
                "best_offset": (int(offset[0]), int(offset[1])),
                "best_orientation": orientation,
                "best_geometry_transform": configuration.get("geometry_transform"),
                "best_geometry_rotation_deg": quality.get("geometry_rotation_deg"),
                "best_geometry_scale_x": quality.get("geometry_scale_x"),
                "best_geometry_scale_y": quality.get("geometry_scale_y"),
                "best_geometry_shear_cosine": quality.get(
                    "geometry_shear_cosine"
                ),
                "simulation_canvas_size": configuration.get("simulation_canvas_size"),
                "message": "active alignment is valid",
            }

    def get_active_alignment_status(self) -> Dict[str, Any]:
        """Return the last known-good alignment used by page gates/experiments."""
        with self._alignment_lock:
            status = getattr(self, "_active_alignment_status", None)
            if status is None:
                # Compatibility for small dry-run managers created via __new__.
                status = {
                    key: self._alignment_status.get(key)
                    for key in (
                        "constraint_satisfied",
                        "config_path",
                        "loaded_configuration",
                        "best_score",
                        "best_offset",
                        "best_orientation",
                        "best_geometry_transform",
                        "best_geometry_rotation_deg",
                        "best_geometry_scale_x",
                        "best_geometry_scale_y",
                        "best_geometry_shear_cosine",
                        "simulation_canvas_size",
                    )
                }
            return deepcopy(status)

    def restore_active_alignment(self) -> bool:
        """Re-apply the known-good offset/orientation/pattern after a failed scan."""
        status = self.get_active_alignment_status()
        configuration = status.get("loaded_configuration") or {}
        if not bool(status.get("constraint_satisfied", False)) or not configuration:
            return False
        offset = configuration.get("slm_offset")
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            return False
        self.update_slm_offset(int(offset[0]), int(offset[1]))
        self.set_ccd_orientation(str(configuration.get("ccd_orientation", "identity")))
        pattern = configuration.get("dmd_pattern") or {}
        self.set_alignment_pattern(
            str(pattern.get("mode", "multiscale_fiducial")),
            int(pattern.get("rectangle_size", 32)),
        )
        return True

    def request_slm_check(self, **request: Any) -> bool:
        self.assert_dmd_controller_ready()
        normalized = {
            "alignment_path": str(request["alignment_path"]),
            "period_px": max(4, min(256, int(request.get("period_px", 32)))),
            "aperture_px": max(32, min(512, int(request.get("aperture_px", 128)))),
            "settle_ms": max(20, min(2000, int(request.get("settle_ms", 120)))),
            "samples": max(1, min(9, int(request.get("samples", 3)))),
        }
        with self._slm_check_lock:
            if self._slm_check_status["state"] in {"queued", "running"}:
                return False
            self._slm_check_request = normalized
            self._slm_check_cancelled = False
            self._slm_check_status = {
                "state": "queued",
                "current": 0,
                "total": 6,
                "message": "等待开始 SLM 快速自检",
                "frames": {},
            }
        return True

    def take_slm_check_request(self) -> Optional[Dict[str, Any]]:
        with self._slm_check_lock:
            request, self._slm_check_request = self._slm_check_request, None
            if request is not None:
                self._slm_check_status.update(state="running", message="正在准备 SLM 自检")
            return request

    def cancel_slm_check(self) -> None:
        with self._slm_check_lock:
            self._slm_check_cancelled = True

    def is_slm_check_cancelled(self) -> bool:
        with self._slm_check_lock:
            return self._slm_check_cancelled

    def update_slm_check_status(self, **values: Any) -> None:
        with self._slm_check_lock:
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    self._slm_check_status[key] = value.copy()
                elif key == "frames" and isinstance(value, dict):
                    self._slm_check_status[key] = {
                        name: np.asarray(frame).copy() for name, frame in value.items()
                    }
                else:
                    self._slm_check_status[key] = value

    def get_slm_check_status(self) -> Dict[str, Any]:
        with self._slm_check_lock:
            snapshot = dict(self._slm_check_status)
            snapshot["frames"] = {
                name: np.asarray(frame).copy()
                for name, frame in self._slm_check_status.get("frames", {}).items()
            }
            if isinstance(snapshot.get("phase"), np.ndarray):
                snapshot["phase"] = snapshot["phase"].copy()
            return snapshot

    def request_phase_calibration(self, **request: Any) -> bool:
        self.assert_dmd_controller_ready()
        operation = str(request.get("operation", "calibration"))
        if operation not in {"calibration", "spatial_check"}:
            raise ValueError("unknown SLM phase operation: %s" % operation)
        period = max(4, min(256, int(request.get("period_px", 32))))
        if period % 2:
            period += 1
        normalized = {
            "operation": operation,
            "alignment_path": str(request["alignment_path"]),
            "period_px": period,
            "aperture_px": max(32, min(512, int(request.get("aperture_px", 128)))),
            "settle_ms": max(20, min(2000, int(request.get("settle_ms", 80)))),
            "samples": max(1, min(9, int(request.get("samples", 3)))),
        }
        if operation == "spatial_check":
            phase_summary_path = request.get("phase_summary_path")
            if not phase_summary_path:
                raise ValueError("spatial phase check requires phase_summary_path")
            normalized["phase_summary_path"] = str(phase_summary_path)
        with self._phase_calibration_lock:
            if self._phase_calibration_status["state"] in {"queued", "running"}:
                return False
            self._phase_calibration_request = normalized
            self._phase_calibration_cancelled = False
            self._phase_calibration_status = {
                "state": "queued",
                "operation": operation,
                "current": 0,
                "total": 45 if operation == "spatial_check" else 262,
                "message": (
                    "等待开始 SLM 全口径一致性检查"
                    if operation == "spatial_check"
                    else "等待开始 SLM 相位标定"
                ),
            }
        return True

    def take_phase_calibration_request(self) -> Optional[Dict[str, Any]]:
        with self._phase_calibration_lock:
            request, self._phase_calibration_request = self._phase_calibration_request, None
            if request is not None:
                self._phase_calibration_status.update(
                    state="running",
                    message=(
                        "正在准备 SLM 全口径一致性检查"
                        if request.get("operation") == "spatial_check"
                        else "正在准备 SLM 相位标定"
                    ),
                )
            return request

    def cancel_phase_calibration(self) -> None:
        with self._phase_calibration_lock:
            self._phase_calibration_cancelled = True

    def is_phase_calibration_cancelled(self) -> bool:
        with self._phase_calibration_lock:
            return self._phase_calibration_cancelled

    def update_phase_calibration_status(self, **values: Any) -> None:
        with self._phase_calibration_lock:
            for key, value in values.items():
                self._phase_calibration_status[key] = (
                    value.copy() if isinstance(value, np.ndarray) else value
                )

    def get_phase_calibration_status(self) -> Dict[str, Any]:
        with self._phase_calibration_lock:
            snapshot = dict(self._phase_calibration_status)
            for key, value in tuple(snapshot.items()):
                if isinstance(value, np.ndarray):
                    snapshot[key] = value.copy()
            return snapshot

    def request_weight_link_validation(self, **request: Any) -> bool:
        self.assert_dmd_controller_ready()
        periods = tuple(sorted({int(value) for value in request.get("periods", (8, 16))}))
        controls = tuple(
            sorted({int(value) for value in request.get("control_sizes", (50, 100, 150, 300))})
        )
        if not periods or any(value < 4 or value > 256 for value in periods):
            raise ValueError("grating periods must lie in [4, 256]")
        if not controls or any(value < 1 or value > 800 for value in controls):
            raise ValueError("control sizes must lie in [1, 800]")
        normalized = {
            "experiment_config": str(request["experiment_config"]),
            "alignment_path": str(request["alignment_path"]),
            "periods": periods,
            "control_sizes": controls,
            "probe_width": max(1, min(8, int(request.get("probe_width", 2)))),
            "settle_ms": max(20, min(2000, int(request.get("settle_ms", 120)))),
            "samples": max(1, min(9, int(request.get("samples", 3)))),
        }
        total = (1 + 4 * len(periods)) * 9 + 2 * len(controls)
        with self._weight_link_lock:
            if self._weight_link_status["state"] in {"queued", "running"}:
                return False
            self._weight_link_request = normalized
            self._weight_link_cancelled = False
            self._weight_link_status = {
                "state": "queued",
                "current": 0,
                "total": total,
                "message": "等待开始 DMD–SLM 权重链路验证",
                "result_rows": [],
            }
        return True

    def take_weight_link_request(self) -> Optional[Dict[str, Any]]:
        with self._weight_link_lock:
            request, self._weight_link_request = self._weight_link_request, None
            if request is not None:
                self._weight_link_status.update(
                    state="running", message="正在核对硬件画布、ROI、对齐和相位 LUT"
                )
            return request

    def cancel_weight_link_validation(self) -> None:
        with self._weight_link_lock:
            self._weight_link_cancelled = True

    def is_weight_link_cancelled(self) -> bool:
        with self._weight_link_lock:
            return self._weight_link_cancelled

    def update_weight_link_status(self, **values: Any) -> None:
        with self._weight_link_lock:
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    self._weight_link_status[key] = value.copy()
                elif key == "result_rows" and isinstance(value, list):
                    self._weight_link_status[key] = [dict(row) for row in value]
                else:
                    self._weight_link_status[key] = value

    def get_weight_link_status(self) -> Dict[str, Any]:
        with self._weight_link_lock:
            snapshot = dict(self._weight_link_status)
            for key, value in tuple(snapshot.items()):
                if isinstance(value, np.ndarray):
                    snapshot[key] = value.copy()
            snapshot["result_rows"] = [
                dict(row) for row in self._weight_link_status.get("result_rows", [])
            ]
            return snapshot

    def request_axial_calibration(self, **request: Any) -> bool:
        self.assert_dmd_controller_ready()
        operation = str(request.get("operation", "manual_z"))
        if operation not in {"manual_z", "software_defocus_scan"}:
            raise ValueError("axial operation must be manual_z or software_defocus_scan")
        raw_coefficients = request.get("defocus_coefficients", ())
        coefficients = tuple(float(value) for value in raw_coefficients)
        if operation == "software_defocus_scan":
            if not coefficients or len(coefficients) > 41:
                raise ValueError("software defocus scan requires 1..41 coefficients")
            if any(not np.isfinite(value) or abs(value) > 16.0 for value in coefficients):
                raise ValueError("software defocus coefficients must be within +/-16")
        normalized = {
            "operation": operation,
            "experiment_config": str(request["experiment_config"]),
            "alignment_path": str(request["alignment_path"]),
            "axial_position_mm": float(request.get("axial_position_mm", 0.0)),
            "session_path": str(request.get("session_path", "") or ""),
            "control_size": max(4, min(300, int(request.get("control_size", 50)))),
            "settle_ms": max(20, min(2000, int(request.get("settle_ms", 120)))),
            "samples": max(1, min(9, int(request.get("samples", 3)))),
            "defocus_coefficients": coefficients,
        }
        with self._axial_lock:
            if self._axial_status["state"] in {"queued", "running"}:
                return False
            self._axial_request = normalized
            self._axial_cancelled = False
            previous_history = [
                dict(row) for row in self._axial_status.get("history_rows", [])
            ]
            self._axial_status = {
                "state": "queued",
                "current": 0,
                "total": 30 * (len(coefficients) if coefficients else 1),
                "message": (
                    "等待自动软件离焦扫描"
                    if operation == "software_defocus_scan"
                    else "等待采集当前 SLM 轴向位置"
                ),
                "history_rows": previous_history,
                "session_path": normalized["session_path"],
            }
        return True

    def take_axial_calibration_request(self) -> Optional[Dict[str, Any]]:
        with self._axial_lock:
            request, self._axial_request = self._axial_request, None
            if request is not None:
                self._axial_status.update(
                    state="running",
                    message="正在检查画布、ROI、对齐和相位 LUT",
                )
            return request

    def cancel_axial_calibration(self) -> None:
        with self._axial_lock:
            self._axial_cancelled = True

    def is_axial_calibration_cancelled(self) -> bool:
        with self._axial_lock:
            return self._axial_cancelled

    def update_axial_calibration_status(self, **values: Any) -> None:
        with self._axial_lock:
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    self._axial_status[key] = value.copy()
                elif key == "history_rows" and isinstance(value, list):
                    self._axial_status[key] = [dict(row) for row in value]
                else:
                    self._axial_status[key] = value

    def get_axial_calibration_status(self) -> Dict[str, Any]:
        with self._axial_lock:
            snapshot = dict(self._axial_status)
            for key, value in tuple(snapshot.items()):
                if isinstance(value, np.ndarray):
                    snapshot[key] = value.copy()
            snapshot["history_rows"] = [
                dict(row) for row in self._axial_status.get("history_rows", [])
            ]
            return snapshot

    def request_sdon_validation(self, **request: Any) -> bool:
        self.assert_dmd_controller_ready()
        input_mode = str(request.get("input_mode", "rate"))
        if input_mode not in {"rate", "binary"}:
            raise ValueError("SDON input_mode must be rate or binary")
        normalized = dict(request)
        normalized.update(
            operation="validation",
            input_mode=input_mode,
            binary_threshold=float(request.get("binary_threshold", 0.5)),
            phase_mode=str(request.get("phase_mode", "trained")),
            phase_scale=float(request.get("phase_scale", 1.0)),
            experiment_config=str(request["experiment_config"]),
            phase_path=str(request["phase_path"]),
            alignment_path=str(request["alignment_path"]),
            dmd_lut_path=str(request["dmd_lut_path"]),
            hardware_detector_regions_path=str(
                request.get("hardware_detector_regions_path", "")
            ),
            sample_count=max(1, int(request.get("sample_count", 20))),
            frames_per_pattern=max(1, min(9, int(request.get("frames_per_pattern", 3)))),
            settle_ms=max(10, int(request.get("settle_ms", 100))),
            slm_initial_settle_ms=max(
                10,
                min(
                    120000,
                    int(
                        request.get(
                            "slm_initial_settle_ms",
                            request.get("settle_ms", 100),
                        )
                    ),
                ),
            ),
            lif_leak=float(request.get("lif_leak", 0.875)),
            lif_threshold=float(request.get("lif_threshold", 1.0)),
            lif_gain=float(request.get("lif_gain", 3.0)),
        )
        if not 0.0 < normalized["lif_leak"] < 1.0:
            raise ValueError("LIF leak must be in (0,1)")
        if not 0.0 <= normalized["binary_threshold"] <= 1.0:
            raise ValueError("binary_threshold must be in [0,1]")
        if normalized["phase_mode"] not in {"trained", "zero"}:
            raise ValueError("phase_mode must be trained or zero")
        if normalized["phase_mode"] == "zero":
            normalized["phase_scale"] = 0.0
        elif not 0.0 < normalized["phase_scale"] <= 2.0:
            raise ValueError("phase_scale must be in (0, 2]")
        if normalized["lif_threshold"] <= 0.0 or normalized["lif_gain"] <= 0.0:
            raise ValueError("LIF threshold and gain must be positive")
        with self._sdon_lock:
            if self._sdon_status["state"] in {"queued", "running"}:
                return False
            self._sdon_request = normalized
            self._sdon_cancelled = False
            self._sdon_status = {
                "state": "queued",
                "operation": "validation",
                "experiment_variant": str(
                    normalized.get("experiment_variant", "standard")
                ),
                "input_mode": input_mode,
                "current": 0,
                "total": normalized["sample_count"],
                "message": "等待开始 %s 固定相位验证（倍率 %.3f）"
                % (normalized["phase_mode"], normalized["phase_scale"]),
            }
        return True

    def request_sdon_simulation_preview(self, **request: Any) -> bool:
        """Queue a simulation-only export of the complete four-class split."""
        normalized = dict(request)
        normalized.update(
            operation="four_class_simulation",
            experiment_variant="onn_four_class_simulation",
            input_mode="binary",
            experiment_config=str(request["experiment_config"]),
            phase_path=str(request["phase_path"]),
            sample_count=max(1, int(request.get("sample_count", 200))),
            batch_size=max(1, int(request.get("batch_size", 8))),
            binary_threshold=float(request.get("binary_threshold", 0.5)),
            phase_scale=float(request.get("phase_scale", 1.0)),
        )
        if not 0.0 <= normalized["binary_threshold"] <= 1.0:
            raise ValueError("binary_threshold must be in [0,1]")
        if not 0.0 < normalized["phase_scale"] <= 2.0:
            raise ValueError("phase_scale must be in (0,2]")
        with self._sdon_lock:
            if self._sdon_status["state"] in {"queued", "running"}:
                return False
            self._sdon_request = normalized
            self._sdon_cancelled = False
            self._sdon_status = {
                "state": "queued",
                "operation": "four_class_simulation",
                "experiment_variant": "onn_four_class_simulation",
                "input_mode": "binary",
                "class_count": 4,
                "current": 0,
                "total": normalized["sample_count"],
                "message": "等待四分类完整 DMD → SLM → CCD 仿真",
            }
        return True

    def request_onn_deployment(self, **request: Any) -> bool:
        """Queue one FIT-only ONN deployment calibration step."""
        task = str(request.get("deployment_task", ""))
        if task != 'fit_regions':
            self.assert_dmd_controller_ready()
        if task not in {"offset_scan", "fit_session", "fit_regions", "holdout"}:
            raise ValueError("unknown ONN deployment task: %s" % task)
        normalized = dict(request)
        deployment_variant = str(request.get("experiment_variant", "onn_deployment"))
        if deployment_variant not in {
            "onn_deployment",
            "onn4_deployment",
            "onn10_deployment",
        }:
            raise ValueError("unknown ONN deployment variant: %s" % deployment_variant)
        normalized.update(
            operation="onn_deployment",
            experiment_variant=deployment_variant,
            deployment_task=task,
            draft_dir=str(request["draft_dir"]),
            sample_count=max(4, min(1000, int(request.get("sample_count", 200)))),
            sample_offset=max(0, int(request.get("sample_offset", 0))),
            frames_per_pattern=max(
                1, min(9, int(request.get("frames_per_pattern", 1)))
            ),
            settle_ms=max(10, min(2000, int(request.get("settle_ms", 200)))),
            slm_initial_settle_ms=max(
                10,
                min(
                    120000,
                    int(
                        request.get(
                            "slm_initial_settle_ms",
                            30000 if deployment_variant == "onn10_deployment" else 5000,
                        )
                    ),
                ),
            ),
            radius_x=max(0, min(20, int(request.get("radius_x", 2)))),
            radius_y=max(0, min(20, int(request.get("radius_y", 1)))),
            region_size=max(2, min(8, int(request.get("region_size", 4)))),
            search_radius=max(1, min(15, int(request.get("search_radius", 8)))),
        )
        total = (
            (2 * normalized["radius_x"] + 1)
            * (2 * normalized["radius_y"] + 1)
            if task == "offset_scan"
            else normalized["sample_count"] * (2 if task == "holdout" else 1)
            if task in {"fit_session", "holdout"}
            else 1
        )
        with self._sdon_lock:
            if self._sdon_status["state"] in {"queued", "running"}:
                return False
            self._sdon_request = normalized
            self._sdon_cancelled = False
            self._sdon_status = {
                "state": "queued",
                "operation": "onn_deployment_" + task,
                "experiment_variant": deployment_variant,
                "deployment_task": task,
                "current": 0,
                "total": total,
                "message": "等待ONN部署标定步骤：%s" % task,
            }
        return True

    def request_carrier_shift_test(self, **request: Any) -> bool:
        """Queue one temporary carrier or forward-model calibration probe."""
        self.assert_dmd_controller_ready()
        test_mode = str(request.get("test_mode", "single"))
        if test_mode not in {"single", "carrier_sweep", "field_scan", "bandwidth"}:
            raise ValueError("unknown temporary optical test mode")
        raw_periods = request.get("carrier_periods", (8, 10, 12, 14, 16))
        carrier_periods = tuple(
            sorted(
                {
                    max(2, min(64, int(value)))
                    for value in raw_periods
                }
            )
        )
        if not carrier_periods:
            raise ValueError("carrier sweep requires at least one period")
        field_coordinates = tuple(
            int(value) for value in request.get("field_coordinates", (8, 16, 24))
        )
        if len(field_coordinates) != 3 or any(
            value < 3 or value > 28 for value in field_coordinates
        ):
            raise ValueError("field coordinates must contain three values in [3,28]")
        bandwidth_blocks = tuple(
            sorted(
                {
                    int(value)
                    for value in request.get(
                        "bandwidth_blocks_physical", (4, 8, 16, 32)
                    )
                }
            )
        )
        if not bandwidth_blocks or any(
            value not in {4, 8, 16, 32, 64} for value in bandwidth_blocks
        ):
            raise ValueError("bandwidth blocks must use 4/8/16/32/64 physical pixels")
        total = (
            3
            if test_mode == "single"
            else 1 + 2 * len(carrier_periods)
            if test_mode == "carrier_sweep"
            else 18
            if test_mode == "field_scan"
            else 1 + 3 * len(bandwidth_blocks)
        )
        normalized = dict(request)
        normalized.update(
            operation="carrier_shift_test",
            experiment_variant="carrier_shift_test",
            test_mode=test_mode,
            experiment_config=str(request["experiment_config"]),
            alignment_path=str(request["alignment_path"]),
            period_px=max(2, min(64, int(request.get("period_px", 8)))),
            carrier_periods=carrier_periods,
            field_coordinates=field_coordinates,
            bandwidth_blocks_physical=bandwidth_blocks,
            dmd_square_px=max(4, min(24, int(request.get("dmd_square_px", 5)))),
            settle_ms=max(20, min(2000, int(request.get("settle_ms", 100)))),
            samples=max(1, min(9, int(request.get("samples", 3)))),
            total=total,
        )
        with self._sdon_lock:
            if self._sdon_status["state"] in {"queued", "running"}:
                return False
            self._sdon_request = normalized
            self._sdon_cancelled = False
            self._sdon_status = {
                "state": "queued",
                "operation": "carrier_shift_test",
                "experiment_variant": "carrier_shift_test",
                "test_mode": test_mode,
                "current": 0,
                "total": total,
                "message": "等待临时光路测试：%s" % test_mode,
            }
        return True

    def request_sdon_finetune(self, **request: Any) -> bool:
        """Queue hardware-in-the-loop SPSA or orthogonal Hadamard training."""
        self.assert_dmd_controller_ready()
        training_kind = str(request.get("training_kind", "finetune"))
        if training_kind not in {"finetune", "online"}:
            raise ValueError("SPSA training_kind must be finetune or online")
        online = training_kind == "online"
        training_method = str(request.get("training_method", "spsa"))
        if training_method not in {"spsa", "hadamard", "ppo", "gmfo"}:
            raise ValueError("training_method must be spsa, hadamard, ppo, or gmfo")
        allowed_control_sizes = {4, 8, 16, 32, 64}
        requested_control_size = int(request.get("control_size", 16 if online else 8))
        if online and requested_control_size not in allowed_control_sizes:
            raise ValueError("online SPSA control_size must be C8, C16, C32, or C64")
        initial_mode = str(request.get("initial_mode", "file"))
        if initial_mode not in {"file", "zero", "lens"}:
            raise ValueError("online SPSA initial_mode must be file, zero, or lens")
        input_mode = str(request.get("input_mode", "rate"))
        if input_mode not in {"rate", "binary"}:
            raise ValueError("SPSA input_mode must be rate or binary")
        if training_method == "hadamard" and (
            not online or input_mode != "binary" or requested_control_size not in {4, 8, 16}
        ):
            raise ValueError("Hadamard training requires binary ONN with C4, C8, or C16")
        if training_method == "ppo" and (
            not online or input_mode != "binary" or requested_control_size != 16
        ):
            raise ValueError("PPO training currently requires binary ONN with C16")
        if training_method == "gmfo" and (
            not online or input_mode != "binary" or requested_control_size not in {8, 16}
        ):
            raise ValueError("G-MFO training requires binary ONN with C8 or C16")
        directions_per_epoch = int(request.get("directions_per_epoch", 32))
        if training_method == "hadamard" and directions_per_epoch not in {16, 32, 64, 128, 256}:
            raise ValueError("Hadamard directions_per_epoch must be 16/32/64/128/256")
        if training_method == "hadamard" and directions_per_epoch > requested_control_size**2:
            raise ValueError("Hadamard directions_per_epoch exceeds the selected control grid")
        normalized = dict(request)
        normalized.update(
            operation="spsa_finetune",
            training_kind=training_kind,
            training_method=training_method,
            initial_mode=initial_mode,
            input_mode=input_mode,
            binary_threshold=float(request.get("binary_threshold", 0.5)),
            experiment_config=str(request["experiment_config"]),
            phase_path=str(request.get("phase_path", "")),
            alignment_path=str(request["alignment_path"]),
            dmd_lut_path=str(request["dmd_lut_path"]),
            hardware_detector_regions_path=str(
                request.get("hardware_detector_regions_path", "")
            ),
            rounds=max(
                1,
                min(
                    1000,
                    int(
                        request.get(
                            "rounds",
                            8
                            if training_method in {"hadamard", "ppo", "gmfo"}
                            else (200 if online else 12),
                        )
                    ),
                ),
            ),
            batch_size=(
                max(10, min(50, int(request.get("batch_size", 20))))
                if training_method in {"ppo", "gmfo"}
                else max(10, min(50, int(request.get("batch_size", 20))))
                if training_method == "hadamard"
                else max(1, min(10, int(request.get("batch_size", 2))))
            ),
            training_pool_size=max(
                10,
                min(500, int(request.get("training_pool_size", 100 if online else 20))),
            ),
            monitor_sample_count=max(
                10,
                min(
                    200,
                    int(
                        request.get(
                            "monitor_sample_count",
                            50 if training_method in {"hadamard", "ppo", "gmfo"} else 10,
                        )
                    ),
                ),
            ),
            monitor_interval=max(1, min(100, int(request.get("monitor_interval", 20)))),
            control_size=requested_control_size,
            perturbation_rad=float(
                request.get(
                    "perturbation_rad",
                    0.35
                    if training_method == "hadamard"
                    else (0.40 if online else 0.20),
                )
            ),
            learning_rate_rad=float(request.get("learning_rate_rad", 0.08)),
            minimum_update_rad=float(2.0 * np.pi / 255.0 if online else 0.0),
            optimizer_mode=(
                "hadamard_adam"
                if training_method == "hadamard"
                else ("quantized_sign" if online else "gradient")
            ),
            correction_limit_rad=float(np.pi if online else 0.75),
            directions_per_epoch=directions_per_epoch,
            hadamard_objective=str(request.get("hadamard_objective", "margin")),
            direction_repeats=2,
            diagnostic_directions=min(16, directions_per_epoch),
            diagnostic_repeats=3,
            trainability_snr_threshold=1.5,
            direction_repeat_correlation_threshold=0.50,
            direction_sign_agreement_threshold=0.75,
            ppo_phase_samples=max(
                8, min(128, int(request.get("ppo_phase_samples", 32)))
            ),
            ppo_sigma=float(request.get("ppo_sigma", 0.04)),
            ppo_clip_epsilon=float(request.get("ppo_clip_epsilon", 0.02)),
            ppo_update_steps=max(1, min(16, int(request.get("ppo_update_steps", 8)))),
            ppo_learning_rate=float(request.get("ppo_learning_rate", 0.0001)),
            ppo_target_kl=float(request.get("ppo_target_kl", 0.02)),
            ppo_reward_nll_weight=float(request.get("ppo_reward_nll_weight", 0.05)),
            policy_reward_mode=str(request.get("policy_reward_mode", "nll")),
            gmfo_step_cycles=float(request.get("gmfo_step_cycles", 0.01)),
            gmfo_reference_repeats=max(
                2, min(16, int(request.get("gmfo_reference_repeats", 4)))
            ),
            gmfo_min_effect_snr=float(request.get("gmfo_min_effect_snr", 1.0)),
            gmfo_persistent_updates=bool(
                request.get("gmfo_persistent_updates", False)
            ),
            gmfo_abba_pairs=bool(request.get("gmfo_abba_pairs", False)),
            phase_boundary_mode=str(request.get("phase_boundary_mode", "clip")),
            project_base_to_control=bool(request.get("project_base_to_control", False)),
            classification_margin=0.20,
            frames_per_pattern=max(1, min(9, int(request.get("frames_per_pattern", 3)))),
            settle_ms=max(10, int(request.get("settle_ms", 100))),
        )
        if normalized["batch_size"] > normalized["training_pool_size"]:
            raise ValueError("SPSA batch_size cannot exceed training_pool_size")
        if normalized["hadamard_objective"] not in {
            "margin",
            "linear_margin",
            "nll",
        }:
            raise ValueError(
                "Hadamard objective must be margin, linear_margin, or nll"
            )
        if training_method in {"ppo", "gmfo", "hadamard"} and normalized["batch_size"] % 10:
            raise ValueError("Hadamard/PPO batch_size must be a multiple of 10")
        if not 0.0 <= normalized["binary_threshold"] <= 1.0:
            raise ValueError("binary_threshold must be in [0,1]")
        if not 0.0 < normalized["ppo_sigma"] <= 0.25:
            raise ValueError("PPO sigma must be in (0, 0.25]")
        if not 0.0 < normalized["ppo_clip_epsilon"] < 1.0:
            raise ValueError("PPO clip epsilon must be in (0,1)")
        if normalized["ppo_learning_rate"] <= 0.0:
            raise ValueError("PPO learning rate must be positive")
        if normalized["ppo_target_kl"] <= 0.0:
            raise ValueError("PPO target KL must be positive")
        if not 0.0 <= normalized["ppo_reward_nll_weight"] <= 1.0:
            raise ValueError("PPO reward NLL weight must be in [0,1]")
        if normalized["policy_reward_mode"] not in {
            "nll",
            "linear_margin",
            "accuracy_nll",
        }:
            raise ValueError("policy reward mode is invalid")
        if normalized["gmfo_step_cycles"] <= 0.0:
            raise ValueError("G-MFO step must be positive")
        if normalized["gmfo_min_effect_snr"] < 0.0:
            raise ValueError("G-MFO minimum effect SNR must be non-negative")
        if normalized["phase_boundary_mode"] not in {"clip", "wrap"}:
            raise ValueError("phase boundary mode must be clip or wrap")
        with self._sdon_lock:
            if self._sdon_status["state"] in {"queued", "running"}:
                return False
            self._sdon_request = normalized
            self._sdon_cancelled = False
            self._sdon_status = {
                "state": "queued",
                "operation": "spsa_finetune",
                "training_kind": training_kind,
                "training_method": training_method,
                "input_mode": input_mode,
                "current": 0,
                "total": normalized["rounds"],
                "message": (
                    "等待开始 C16 PPO 实机在线训练"
                    if training_method == "ppo"
                    else "等待开始 C16 Hadamard 实机在线训练"
                    if training_method == "hadamard"
                    else "等待开始 C%d SPSA 实机在线训练" % requested_control_size
                    if online
                    else "等待开始 8x8 SPSA 实机相位细调"
                ),
            }
        return True

    def take_sdon_request(self) -> Optional[Dict[str, Any]]:
        with self._sdon_lock:
            request, self._sdon_request = self._sdon_request, None
            if request is not None:
                self._sdon_status.update(state="running", message="正在准备 MNIST 与仿真基线")
                if request.get("operation") == "four_class_simulation":
                    self._sdon_status.update(
                        operation="four_class_simulation",
                        experiment_variant="onn_four_class_simulation",
                        input_mode="binary",
                        class_count=4,
                        message="正在计算四分类完整 DMD → SLM → CCD 仿真",
                    )
                elif request.get("operation") == "carrier_shift_test":
                    self._sdon_status.update(
                        operation="carrier_shift_test",
                        experiment_variant="carrier_shift_test",
                        test_mode=str(request.get("test_mode", "single")),
                        current=0,
                        total=int(request.get("total", 3)),
                        message="正在准备临时光路测试",
                    )
                elif request.get("operation") == "onn_deployment":
                    task = str(request.get("deployment_task", ""))
                    self._sdon_status.update(
                        operation="onn_deployment_" + task,
                        experiment_variant=str(
                            request.get("experiment_variant", "onn_deployment")
                        ),
                        deployment_task=task,
                        message="正在准备ONN部署标定：%s" % task,
                    )
                elif request.get("operation") == "spsa_finetune":
                    self._sdon_status.update(
                        operation="spsa_finetune",
                        message=(
                            "正在准备 C16 PPO 实机在线训练"
                            if request.get("training_method") == "ppo"
                            else "正在准备 C16 Hadamard 实机在线训练"
                            if request.get("training_method") == "hadamard"
                            else "正在准备 C%d SPSA 实机在线训练"
                            % int(request.get("control_size", 16))
                            if request.get("training_kind") == "online"
                            else "正在准备 8x8 SPSA 实机相位细调"
                        ),
                    )
            return request

    def cancel_sdon_validation(self) -> None:
        with self._sdon_lock:
            self._sdon_cancelled = True

    def is_sdon_cancelled(self) -> bool:
        with self._sdon_lock:
            return self._sdon_cancelled

    def update_sdon_status(self, **values: Any) -> None:
        with self._sdon_lock:
            for key, value in values.items():
                self._sdon_status[key] = (
                    np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                )

    def get_sdon_status(self) -> Dict[str, Any]:
        with self._sdon_lock:
            snapshot = dict(self._sdon_status)
            for key, value in tuple(snapshot.items()):
                if isinstance(value, np.ndarray):
                    snapshot[key] = value.copy()
            return snapshot

    def close(self) -> None:
        self.camera.close()
