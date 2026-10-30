"""Small FLIR/Point Grey Spinnaker adapter.

PySpin is loaded lazily because it is distributed with the vendor SDK rather
than PyPI. Importing the rest of the project therefore remains possible on a
development machine without a connected camera.
"""

import ctypes
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sdon.config import CameraConfig


_RETAINED_WINDOWS_TRANSPORT_LIBRARIES = []

_MEASUREMENT_CAMERA_KEYS = (
    "DeviceModelName", "SensorShutterMode", "PixelFormat", "ExposureMode",
    "ExposureAuto", "GainAuto", "TriggerMode", "TriggerSource", "TriggerSelector",
    "TriggerActivation", "TriggerDelayEnabled", "ExposureTime_us", "Gain_dB", "BlackLevel",
)


def camera_measurement_signature(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """The small acquisition contract needed by this platform's optical calibrations."""
    result = {name: parameters.get(name) for name in _MEASUREMENT_CAMERA_KEYS}
    # An inactive delay value is not an acquisition difference.
    if str(parameters.get("TriggerDelayEnabled", "0")).lower() not in {"0", "false", "n/a", "none"}:
        result["TriggerDelay_us"] = parameters.get("TriggerDelay_us")
    return result


def assert_camera_measurement_matches(saved: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Reject mixing shutter/exposure/trigger conditions, allowing SDK quantization."""
    if not saved or saved.get("SensorShutterMode") in {None, "", "N/A"}:
        raise ValueError("calibration has no sensor shutter binding; recalibrate in the current camera mode")
    expected = camera_measurement_signature(saved)
    actual = camera_measurement_signature(current)
    mismatches = []
    for name in sorted(set(expected) | set(actual)):
        left,right = expected.get(name),actual.get(name)
        if name in {"ExposureTime_us", "Gain_dB", "BlackLevel", "TriggerDelay_us"}:
            try:
                # ROI-dependent exposure quantization is <1% on this BFLY.
                same = math.isclose(float(left),float(right),rel_tol=.01 if name=='ExposureTime_us' else 1e-6,abs_tol=1e-3)
            except (TypeError,ValueError):
                same = False
        else:
            same = left == right
        if not same:mismatches.append(name)
    if mismatches:
        raise ValueError("camera acquisition changed (%s); use calibrations from the same shutter/exposure/trigger mode" % ', '.join(mismatches))


def validate_saved_camera_measurement(manager: Any, calibration: Dict[str, Any]) -> None:
    """New calibrations always bind camera state; explicit modes reject legacy files."""
    saved = calibration.get("camera") or calibration.get("camera_acquisition")
    config = getattr(getattr(manager,"config",None),"camera",None)
    strict = bool(getattr(config,"sensor_shutter_mode",""))
    if not strict and (not saved or saved.get("SensorShutterMode") in {None,"","N/A"}):return
    getter = getattr(getattr(manager,"camera",None),"get_camera_parameters",None)
    if not callable(getter):
        raise ValueError("camera acquisition readback unavailable")
    assert_camera_measurement_matches(saved or {},getter())


def retain_windows_transport_libraries() -> None:
    """Keep legacy Spinnaker transport dependencies loaded for this process.

    Spinnaker 3.0 on the lab PC intermittently leaves a transport callback
    alive after releasing ``NetSetupEngine.dll`` (and, less frequently, the
    NVIDIA DirectGMA helper).  Windows then reports an access violation in a
    module whose name ends in ``_unloaded``.  Holding one explicit loader
    reference until interpreter shutdown prevents that use-after-unload while
    leaving all camera parameters and acquisition semantics unchanged.
    """
    if os.name != "nt" or _RETAINED_WINDOWS_TRANSPORT_LIBRARIES:
        return
    candidates = [("NetSetupEngine", Path("NetSetupEngine.dll"))]
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    driver_store = system_root / "System32" / "DriverStore" / "FileRepository"
    try:
        direct_gma = sorted(driver_store.glob("nv_disp*.inf_amd64_*/*nvdxgdmal64.dll"))
    except OSError:
        direct_gma = []
    if direct_gma:
        candidates.append(("NVIDIA DirectGMA", direct_gma[-1]))
    for _label, path in candidates:
        try:
            _RETAINED_WINDOWS_TRANSPORT_LIBRARIES.append(ctypes.WinDLL(str(path)))
        except OSError:
            # DirectGMA is optional and driver-version specific.  PySpin can
            # still use its ordinary host-memory path when it is unavailable.
            continue


def validate_camera_contract(parameters: Dict[str, Any], config: CameraConfig) -> None:
    """Verify that the camera actually accepted the requested acquisition mode."""
    pixel_format = str(parameters.get("PixelFormat", ""))
    if pixel_format != "Mono8":
        raise RuntimeError(
            "camera PixelFormat readback is %s, expected Mono8 (0..255)"
            % (pixel_format or "unavailable")
        )
    requested_shutter = config.sensor_shutter_mode
    if requested_shutter and parameters.get("SensorShutterMode") != requested_shutter:
        raise RuntimeError(
            "camera shutter readback mismatch: %s, expected %s"
            % (parameters.get("SensorShutterMode", "unavailable"), requested_shutter)
        )
    if not config.use_trigger:
        return
    expected_source = "Software" if config.software_trigger else config.trigger_source
    actual_mode = str(parameters.get("TriggerMode", ""))
    actual_source = str(parameters.get("TriggerSource", ""))
    if actual_mode != "On" or actual_source != expected_source:
        raise RuntimeError(
            "camera trigger readback mismatch: mode/source=%s/%s, expected=On/%s"
            % (actual_mode, actual_source, expected_source)
        )


class FlirCamera:
    def __init__(self, config: CameraConfig) -> None:
        retain_windows_transport_libraries()
        try:
            import PySpin  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PySpin is unavailable. Install the FLIR Spinnaker SDK and its matching "
                "Python binding before starting hardware calibration."
            ) from exc
        self.pyspin = PySpin
        self.config = config
        self.system = PySpin.System.GetInstance()
        self.camera_list = None
        self.camera = None
        self.nodemap = None
        self.acquiring = False
        try:
            self._open()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _open(self) -> None:
        cameras = self.system.GetCameras()
        self.camera_list = cameras
        count = cameras.GetSize()
        if not 0 <= self.config.index < count:
            cameras.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError(
                "camera index %d is unavailable; detected %d camera(s)" % (self.config.index, count)
            )
        self.camera = cameras[self.config.index]
        self.camera.Init()
        self.nodemap = self.camera.GetNodeMap()
        self._set_enum("TriggerMode", "Off")
        # Force an auditable 8-bit sensor output.  Exposure controls how much
        # of this range is used; software must not rescale saturated raw data.
        self._set_enum("PixelFormat", "Mono8")
        self._configure_roi()
        self._set_enum("ExposureAuto", "Off")
        self._set_enum("ExposureMode", "Timed", required=False)
        self._set_float("ExposureTime", self.config.exposure_ms * 1000.0)
        self._set_enum("GainAuto", "Off")
        self._set_float("Gain", self.config.gain_db)
        self._set_float("BlackLevel", self.config.black_level)
        self._set_enum("AcquisitionMode", "Continuous")
        self._set_stream_newest_only()

        if self.config.use_trigger:
            self._set_enum("TriggerSelector", "FrameStart")
            source = "Software" if self.config.software_trigger else self.config.trigger_source
            self._set_enum("TriggerSource", source)
            self._set_enum("TriggerMode", "On")

        # BFLY-PGE-50A2 can change this mode when TriggerMode is enabled.
        # Applying it earlier can silently restore GlobalReset and row-dependent
        # exposure despite an apparently correct ExposureTime readback.
        self._configure_sensor_shutter()
        validate_camera_contract(self.get_camera_parameters(), self.config)

        self.camera.BeginAcquisition()
        self.acquiring = True
        time.sleep(1.0)
        if self.config.sensor_shutter_mode:
            self.nodemap.InvalidateNodes()
            validate_camera_contract(self.get_camera_parameters(), self.config)

    def _configure_sensor_shutter(self) -> None:
        if self.config.sensor_shutter_mode:
            self._set_enum("SensorShutterMode", self.config.sensor_shutter_mode)

    def _set_enum(self, node_name: str, entry_name: str, required: bool = True) -> None:
        pyspin = self.pyspin
        node = pyspin.CEnumerationPtr(self.nodemap.GetNode(node_name))
        entry = node.GetEntryByName(entry_name)
        if not (pyspin.IsWritable(node) and pyspin.IsReadable(entry)):
            if required:
                raise RuntimeError("camera node %s=%s is unavailable" % (node_name, entry_name))
            return
        node.SetIntValue(entry.GetValue())

    def _set_float(self, node_name: str, value: float) -> None:
        pyspin = self.pyspin
        node = pyspin.CFloatPtr(self.nodemap.GetNode(node_name))
        if not pyspin.IsWritable(node):
            raise RuntimeError("camera node %s is not writable" % node_name)
        node.SetValue(max(node.GetMin(), min(node.GetMax(), float(value))))

    def _integer_node(self, node_name: str):
        return self.pyspin.CIntegerPtr(self.nodemap.GetNode(node_name))

    def _read_integer(self, node_name: str) -> Optional[int]:
        node = self._integer_node(node_name)
        return int(node.GetValue()) if self.pyspin.IsReadable(node) else None

    def _set_integer(self, node_name: str, value: int) -> int:
        node = self._integer_node(node_name)
        if not self.pyspin.IsWritable(node):
            raise RuntimeError("camera node %s is not writable" % node_name)
        minimum = int(node.GetMin())
        maximum = int(node.GetMax())
        increment = max(1, int(node.GetInc()))
        clipped = max(minimum, min(maximum, int(value)))
        quantized = minimum + ((clipped - minimum) // increment) * increment
        node.SetValue(quantized)
        return int(node.GetValue())

    def _configure_roi_axis(
        self, size_name: str, offset_name: str, requested_size: int, requested_offset: int
    ) -> None:
        # Offsets must be reset before enlarging Width/Height on most Spinnaker cameras.
        self._set_integer(offset_name, 0)
        size_node = self._integer_node(size_name)
        if not self.pyspin.IsWritable(size_node):
            raise RuntimeError("camera node %s is not writable" % size_name)
        sensor_extent = int(size_node.GetMax())
        available = max(int(size_node.GetMin()), sensor_extent - int(requested_offset))
        target_size = available if requested_size == 0 else min(int(requested_size), available)
        self._set_integer(size_name, target_size)
        self._set_integer(offset_name, requested_offset)

    def _configure_roi(self) -> None:
        self._configure_roi_axis(
            "Width", "OffsetX", self.config.roi_width, self.config.roi_offset_x
        )
        self._configure_roi_axis(
            "Height", "OffsetY", self.config.roi_height, self.config.roi_offset_y
        )

    def set_roi(
        self, offset_x: int, offset_y: int, width: int, height: int
    ) -> Dict[str, int]:
        """Change the hardware ROI safely and return the quantized camera readback."""
        requested = tuple(int(value) for value in (offset_x, offset_y, width, height))
        if min(requested) < 0:
            raise ValueError(
                "runtime camera ROI requires non-negative size and offset values"
            )
        # Width/Height=0 deliberately reuse the startup convention: reset the
        # offset, then request the largest sensor extent available on that axis.
        if (width == 0 and offset_x != 0) or (height == 0 and offset_y != 0):
            raise ValueError("full-frame width/height requires a zero offset")
        previous = self.config
        was_acquiring = self.acquiring
        if was_acquiring:
            self.camera.EndAcquisition()
            self.acquiring = False
        try:
            self._configure_roi_axis("Width", "OffsetX", width, offset_x)
            self._configure_roi_axis("Height", "OffsetY", height, offset_y)
            parameters = self.get_camera_parameters()
            roi = {
                name: int(parameters[name])
                for name in ("OffsetX", "OffsetY", "Width", "Height")
            }
            self.config = replace(
                previous,
                roi_offset_x=roi["OffsetX"],
                roi_offset_y=roi["OffsetY"],
                roi_width=roi["Width"],
                roi_height=roi["Height"],
            )
        except Exception:
            # Best-effort rollback keeps a failed automatic calibration from
            # silently leaving the camera in a half-updated ROI.
            self.config = previous
            self._configure_roi()
            raise
        finally:
            if was_acquiring:
                self._configure_sensor_shutter()
                self.camera.BeginAcquisition()
                self.acquiring = True
                time.sleep(0.25)
                if self.config.sensor_shutter_mode:
                    self.nodemap.InvalidateNodes()
                    validate_camera_contract(self.get_camera_parameters(), self.config)
        return roi

    def _set_stream_newest_only(self) -> None:
        pyspin = self.pyspin
        stream_map = self.camera.GetTLStreamNodeMap()
        node = pyspin.CEnumerationPtr(stream_map.GetNode("StreamBufferHandlingMode"))
        entry = node.GetEntryByName("NewestOnly")
        if pyspin.IsWritable(node) and pyspin.IsReadable(entry):
            node.SetIntValue(entry.GetValue())

    def _read_enum(self, name: str) -> str:
        node = self.pyspin.CEnumerationPtr(self.nodemap.GetNode(name))
        if not self.pyspin.IsReadable(node):
            return "N/A"
        entry = node.GetCurrentEntry()
        return entry.GetSymbolic() if self.pyspin.IsReadable(entry) else str(node.GetIntValue())

    def _read_float(self, name: str) -> Optional[float]:
        node = self.pyspin.CFloatPtr(self.nodemap.GetNode(name))
        return float(node.GetValue()) if self.pyspin.IsReadable(node) else None

    def _read_optional_value(self, name: str) -> str:
        try:
            node = self.pyspin.CValuePtr(self.nodemap.GetNode(name))
            return str(node.ToString()) if self.pyspin.IsReadable(node) else "N/A"
        except Exception:
            return "N/A"

    def get_camera_parameters(self) -> Dict[str, Any]:
        if self.camera is None:
            return {}
        parameters = {
            "DeviceModelName": self._read_optional_value("DeviceModelName"),
            "SensorShutterMode": self._read_optional_value("SensorShutterMode"),
            "TriggerActivation": self._read_optional_value("TriggerActivation"),
            "TriggerDelay_us": self._read_optional_value("TriggerDelay"),
            "TriggerDelayEnabled": self._read_optional_value("TriggerDelayEnabled"),
            "ExposureAuto": self._read_enum("ExposureAuto"),
            "ExposureMode": self._read_enum("ExposureMode"),
            "ExposureTime_us": self._read_float("ExposureTime"),
            "GainAuto": self._read_enum("GainAuto"),
            "Gain_dB": self._read_float("Gain"),
            "BlackLevel": self._read_float("BlackLevel"),
            "TriggerMode": self._read_enum("TriggerMode"),
            "TriggerSelector": self._read_enum("TriggerSelector"),
            "TriggerSource": self._read_enum("TriggerSource"),
            "AcquisitionMode": self._read_enum("AcquisitionMode"),
            "PixelFormat": self._read_enum("PixelFormat"),
            "OffsetX": self._read_integer("OffsetX"),
            "OffsetY": self._read_integer("OffsetY"),
            "Width": self._read_integer("Width"),
            "Height": self._read_integer("Height"),
        }
        pixel_format = str(parameters.get("PixelFormat", ""))
        parameters["IntensityRange"] = [0, 255] if pixel_format == "Mono8" else None
        trigger_source = str(parameters.get("TriggerSource", ""))
        parameters["TriggerKind"] = (
            "Software"
            if trigger_source == "Software"
            else "Hardware"
            if parameters.get("TriggerMode") == "On"
            else "Off"
        )
        return parameters

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.acquiring:
            return None
        pyspin = self.pyspin
        last_error: Optional[str] = None
        for attempt in range(5):
            try:
                if self.config.use_trigger and self.config.software_trigger:
                    trigger = pyspin.CCommandPtr(self.nodemap.GetNode("TriggerSoftware"))
                    if not pyspin.IsWritable(trigger):
                        raise RuntimeError("software trigger is not writable")
                    trigger.Execute()
                exposure_us = (
                    self._read_float("ExposureTime")
                    or self.config.exposure_ms * 1000.0
                )
                result = self.camera.GetNextImage(int(exposure_us / 1000.0 + 1000))
                try:
                    if result.IsIncomplete():
                        return None
                    return result.GetNDArray().copy()
                finally:
                    result.Release()
            except pyspin.SpinnakerException as exc:
                # A SWIG traceback retains CamPtr references. Keeping the
                # exception as the eventual cause can prevent SDK shutdown
                # after a timeout (ReleaseInstance: -1004).
                last_error = str(exc)
                if attempt < 4:
                    # Merely restarting acquisition can leave the BFLY in a
                    # stale trigger state after reconnect. Re-arm the same
                    # configured source, then reapply the shutter AFTER On.
                    try:
                        if self.acquiring:
                            self.camera.EndAcquisition()
                            self.acquiring = False
                        time.sleep(0.05)
                        if self.config.use_trigger:
                            self._set_enum('TriggerMode', 'Off')
                            self._set_enum('TriggerSelector', 'FrameStart')
                            self._set_enum('TriggerSource', 'Software' if self.config.software_trigger else self.config.trigger_source)
                            self._set_enum('TriggerMode', 'On')
                        self._configure_sensor_shutter()
                        self.camera.BeginAcquisition()
                        self.acquiring = True
                        time.sleep(0.20)
                    except pyspin.SpinnakerException:
                        time.sleep(0.10)
        raise RuntimeError(
            "camera capture failed after 5 attempts: %s" % last_error
        ) from None

    def close(self) -> None:
        if self.camera is not None:
            if self.acquiring:
                self.camera.EndAcquisition()
                self.acquiring = False
            if self.config.use_trigger:
                self._set_enum("TriggerMode", "Off", required=False)
            self.camera.DeInit()
            self.camera = None
            self.nodemap = None
        if self.camera_list is not None:
            self.camera_list.Clear()
            self.camera_list = None
        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None

    def __enter__(self) -> "FlirCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
