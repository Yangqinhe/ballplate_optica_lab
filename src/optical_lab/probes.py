"""Fixed E002 *offline* SLM pilot assets; never a display or device command.

The existing protocol stores its pilot under ``repetition.pilot``.  This module
supports that exact 600/512, macro16, P128-X, raw0/64 pilot only.  In particular,
canonical600 is NOT a physical full-panel frame: no DMD input, panel placement,
flip, LUT, camera contract, collection, or control release is inferred here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zlib

import numpy as np


SCHEMA = "optical-lab-offline-pilot-assets-1"
PROTOCOL_SCHEMA = "BALLPLATE_OPTICAL_ASSESSMENT_PROTOCOL_1"
PILOT_ID = "E002_OFFLINE_PILOT_ASSETS_1"
ASSET_NAMES = ("Z", "A", "B")
SNAPSHOT = "protocol_snapshot.json"
MANIFEST = "MANIFEST.json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FILES = frozenset([SNAPSHOT, *[f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT.npy" for name in ASSET_NAMES],
                   *[f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT.png" for name in ASSET_NAMES]])
_SPEC = {
    "canvas_shape_yx": [600, 600],
    "interior_shape_yx": [512, 512],
    "interior_origin_xy": [44, 44],
    "outside_gray": 0,
    "macro_block_side_physical_px": 16,
    "period_physical_px": 128,
    "period_macro_samples": 8,
    "axis": "x",
    "gray_pair": [0, 64],
    "first_half_period_gray_A": 0,
    "A_B_complement_scope": "512_interior_only_outside_both_zero",
    "gray_domain": "raw_uint8_not_calibrated_phase",
    "canonical_coordinates": "x_right_y_down_no_flip_no_LUT",
    "PNG_limits_adu": [0, 255],
}
_UNRELEASED = {
    "captured_frames": 0,
    "projected_patterns": 0,
    "projection_executed": False,
    "hardware_connected": False,
    "hardware_actions_enabled": False,
    "hardware_release_approved": False,
    "projection_release_approved": False,
    "control_release_approved": False,
    "production_qualified": False,
    "physical_panel_frame_generated": False,
    "DMD_input_selected": False,
    "DMD_asset_generated": False,
    "field_contract_selected": False,
    "SLM_phase_calibration_available": False,
    "LUT_applied": False,
    "flip_applied": False,
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(data: bytes):
    def reject_constant(value):
        raise ValueError("non-finite JSON constant rejected: " + value)
    return json.loads(data.decode("utf-8"), parse_constant=reject_constant)


def _exact(actual, expected, message):
    # JSON identity distinguishes False/0, 16/16.0, and unsupported extra fields.
    if _json_bytes(actual) != _json_bytes(expected):
        raise ValueError(message)


def _pilot_spec(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported assessment protocol schema")
    if protocol.get("id") != "ASSESSMENT_RAWGRAY_MACROBLOCK_1":
        raise ValueError("explicit fixed assessment protocol identity required")
    if protocol.get("hardware_actions_enabled") is not False:
        raise ValueError("hardware actions must remain disabled")
    geometry = protocol["geometry"]
    for key, expected in (
        ("canvas_shape_px", [600, 600]),
        ("assessment_interior_shape_px", [512, 512]),
        ("assessment_interior_top_left_px", [44, 44]),
        ("outside_gray", 0),
        ("resampling", "integer_nearest_repeat_only_no_bilinear_resize"),
        ("old_training_window_is_changed", False),
    ):
        _exact(geometry[key], expected, "fixed pilot geometry differs: " + key)
    pilot = protocol["repetition"]["pilot"]
    _exact(pilot, {"input_count": 1, "macro_block_side_physical_px": 16,
                   "period_physical_px": 128, "axis": "x", "raw_gray_pair": [0, 64]},
           "only the declared macro16/P128/X/raw0-64 pilot is supported")
    design = protocol["probe_design"]
    if design["primary_pattern"] != "binary_50_percent_duty_grating":
        raise ValueError("fixed pilot requires binary 50 percent duty")
    if design["raw_gray_pi_is_assumed"] is not False or design["phase_lut_measured_in_this_project"] is not False:
        raise ValueError("fixed raw-gray pilot does not claim or use a phase calibration")
    if [0, 64] not in design["primary_raw_gray_pairs"]:
        raise ValueError("raw0/64 pair absent from protocol")
    for key in ("collection_enabled", "connect_sdk_usb_or_display_owner", "restart_current_window"):
        if protocol["acquisition"][key] is not False:
            raise ValueError("offline pilot requires disabled " + key)
    return _read_json(_json_bytes(_SPEC))


def _patterns(spec):
    _exact(spec, _SPEC, "unsupported pilot specification")
    row = np.where((np.arange(512) % 128) < 64, 0, 64).astype(np.uint8)
    interior = np.broadcast_to(row[None, :], (512, 512))
    zero = np.zeros((600, 600), dtype=np.uint8)
    a = zero.copy()
    b = zero.copy()
    a[44:556, 44:556] = interior
    b[44:556, 44:556] = 64 - interior
    return {"Z": zero, "A": a, "B": b}


def _validate_arrays(arrays, spec):
    expected = _patterns(spec)
    for name in ASSET_NAMES:
        value = arrays[name]
        if value.dtype != np.uint8 or value.shape != (600, 600):
            raise ValueError("canonical600 uint8 shape required: " + name)
        if not np.array_equal(value, expected[name]):
            raise ValueError("exact pilot shape/support/gray/period differs: " + name)
    a = arrays["A"][44:556, 44:556]
    b = arrays["B"][44:556, 44:556]
    if not np.array_equal(a.astype(np.int16) + b, np.full((512, 512), 64)):
        raise ValueError("interior A/B complement failed")
    if not np.array_equal(np.bincount(a.ravel(), minlength=256), np.bincount(b.ravel(), minlength=256)):
        raise ValueError("A/B gray histogram mismatch")
    for value in (a, b):
        macro = value.reshape(32, 16, 32, 16)
        if not np.all(macro == macro[:, :1, :, :1]):
            raise ValueError("macro16 blocks are not constant")
        row = value[0].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(row - row.mean()))
        if not spectrum[4] > 0 or int(np.argmax(spectrum[1:])) + 1 != 4:
            raise ValueError("expected nonzero P128 fundamental absent")


def _npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _png_bytes(value):
    """Exact 8-bit grayscale PNG (0..255), with no palette or display autogain."""
    if value.dtype != np.uint8 or value.ndim != 2:
        raise ValueError("PNG requires a uint8 grayscale array")
    height, width = value.shape
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    raw = b"".join(b"\x00" + value[y].tobytes() for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def prepare_pilot(protocol_path, output):
    """Create one NEW offline asset directory; return its manifest JSON object.

    Existing output paths are never reused or overwritten.  The directory's
    parent must exist.  On a preparation failure, partial files remain and a
    FAILED.json is added if possible; they are not valid pilot assets.
    """
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol = _read_json(protocol_bytes)
    spec = _pilot_spec(protocol)
    source_path = Path(__file__).resolve()
    source_sha = _file_digest(source_path)
    arrays = _patterns(spec)
    _validate_arrays(arrays, spec)
    root = Path(output)
    # mkdir before resolve prevents an existing output symlink being accepted.
    root.mkdir(exist_ok=False)
    root = root.resolve(strict=True)
    try:
        files = {SNAPSHOT: protocol_bytes}
        assets = {}
        for name in ASSET_NAMES:
            npy = f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT.npy"
            png = f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT.png"
            files[npy] = _npy_bytes(arrays[name])
            files[png] = _png_bytes(arrays[name])
            assets[name] = {"npy": npy, "png": png, "shape_yx": [600, 600], "dtype": "uint8",
                            "pixel_sha256": _digest(arrays[name].tobytes(order="C")),
                            "meaning": "canonical600_zero_NOT_physical_panel_frame" if name == "Z" else "canonical600_finite_support_raw_gray_probe"}
        for name, data in files.items():
            _write_new(root / name, data)
        if _file_digest(protocol_path) != _digest(protocol_bytes) or _file_digest(source_path) != source_sha:
            raise ValueError("protocol or generator source changed during preparation")
        for name, data in files.items():
            if _file_digest(root / name) != _digest(data):
                raise ValueError("new asset changed during preparation: " + name)
        manifest = {
            "schema": SCHEMA, "id": PILOT_ID, "asset_generation_complete": True,
            "status": "PREPARED_NOT_PROJECTED_NOT_CAPTURED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_directory": str(root),
            "pilot_spec": spec, "assets": assets,
            "protocol": {"id": protocol["id"], "schema": protocol["schema"], "snapshot": SNAPSHOT,
                         "sha256": _digest(protocol_bytes), "original_path_information_only": str(protocol_path)},
            "implementation": {"module": "optical_lab.probes", "source_sha256": source_sha},
            "file_sha256": {name: _digest(data) for name, data in files.items()},
            "manifest_excludes_itself": True,
            "unreleased": dict(_UNRELEASED),
            "not_selected": {"DMD_input": None, "physical_panel_mapping": None,
                             "field_hardware_contract": None, "phase_LUT": None},
            "gate_states": {"acquisition": "UNKNOWN", "optical_modulation": "UNKNOWN", "simulation_agreement": "UNKNOWN"},
            "scope": "offline_asset_identity_only_NOT_E002_measurement_NOT_hardware_or_control_release",
        }
        manifest_bytes = _json_bytes(manifest)
        _write_new(root / MANIFEST, manifest_bytes)
        verify_pilot_assets(root, expected_manifest_sha256=_digest(manifest_bytes))
        return manifest
    except Exception as exc:
        # No cleanup/deletion, no overwrite, no successful-manifest repair.
        try:
            _write_new(root / "FAILED.json", _json_bytes({"schema": "optical-lab-offline-pilot-failure-1",
                "error": type(exc).__name__ + ": " + str(exc), "captured_frames": 0,
                "hardware_release_approved": False}))
        except OSError:
            pass
        raise


def verify_pilot_assets(output, *, expected_manifest_sha256=None):
    """Verify exact fixed probe arrays and hashes, not any hardware identity.

    Supplying an externally recorded manifest SHA adds that identity check.
    Without one, ``verified`` only describes internal archive consistency.
    The original mutable protocol path is informational; the snapshot is used.
    """
    candidate = Path(output)
    if candidate.is_symlink():
        raise ValueError("pilot output cannot be a symlink")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("pilot output directory required")
    if {p.name for p in root.iterdir()} != _FILES | {MANIFEST}:
        raise ValueError("pilot file list differs, is incomplete, or contains failed evidence")
    for name in _FILES | {MANIFEST}:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular non-symlink pilot file required: " + name)
    raw_manifest = (root / MANIFEST).read_bytes()
    manifest_sha = _digest(raw_manifest)
    if expected_manifest_sha256 is not None:
        if not isinstance(expected_manifest_sha256, str) or not _SHA.fullmatch(expected_manifest_sha256):
            raise ValueError("explicit lowercase manifest SHA256 required")
        if manifest_sha != expected_manifest_sha256:
            raise ValueError("external pilot manifest SHA mismatch")
    manifest = _read_json(raw_manifest)
    if (manifest.get("schema") != SCHEMA or manifest.get("id") != PILOT_ID
            or manifest.get("asset_generation_complete") is not True
            or manifest.get("status") != "PREPARED_NOT_PROJECTED_NOT_CAPTURED"
            or manifest.get("manifest_excludes_itself") is not True):
        raise ValueError("completed offline-only fixed pilot manifest required")
    _exact(manifest["unreleased"], _UNRELEASED, "pilot is not explicitly unmeasured and unreleased")
    _exact(manifest["not_selected"], {"DMD_input": None, "physical_panel_mapping": None,
        "field_hardware_contract": None, "phase_LUT": None}, "pilot cannot claim selected hardware assets")
    _exact(manifest["gate_states"], {"acquisition": "UNKNOWN", "optical_modulation": "UNKNOWN",
        "simulation_agreement": "UNKNOWN"}, "pilot assets do not release evidence gates")
    if manifest["scope"] != "offline_asset_identity_only_NOT_E002_measurement_NOT_hardware_or_control_release":
        raise ValueError("pilot scope changed")
    implementation = manifest["implementation"]
    source_sha = _file_digest(Path(__file__).resolve())
    _exact(implementation, {"module": "optical_lab.probes", "source_sha256": source_sha}, "generator source identity differs")
    hashes = manifest["file_sha256"]
    if set(hashes) != _FILES or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in hashes.values()):
        raise ValueError("fixed pilot file hash table required")
    captured = {name: (root / name).read_bytes() for name in _FILES}
    for name, data in captured.items():
        if _digest(data) != hashes[name]:
            raise ValueError("pilot file SHA mismatch: " + name)
    protocol = _read_json(captured[SNAPSHOT])
    spec = _pilot_spec(protocol)
    _exact(manifest["pilot_spec"], spec, "manifest spec differs from protocol")
    pinfo = manifest["protocol"]
    if (pinfo["id"] != protocol["id"] or pinfo["schema"] != protocol["schema"]
            or pinfo["snapshot"] != SNAPSHOT or pinfo["sha256"] != _digest(captured[SNAPSHOT])):
        raise ValueError("protocol snapshot binding differs")
    arrays = {}
    expected_assets = {}
    for name in ASSET_NAMES:
        npy = f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT.npy"
        png = f"SLM_{name}_CANONICAL600_RAW_GRAY_NO_LUT.png"
        value = np.load(io.BytesIO(captured[npy]), allow_pickle=False)
        if not isinstance(value, np.ndarray) or value.shape != (600, 600) or value.dtype != np.uint8:
            raise ValueError("canonical600 uint8 NPY required: " + name)
        arrays[name] = value
        if captured[png] != _png_bytes(value):
            raise ValueError("fixed-scale grayscale PNG differs from NPY: " + name)
        expected_assets[name] = {"npy": npy, "png": png, "shape_yx": [600, 600], "dtype": "uint8",
            "pixel_sha256": _digest(value.tobytes(order="C")),
            "meaning": "canonical600_zero_NOT_physical_panel_frame" if name == "Z" else "canonical600_finite_support_raw_gray_probe"}
    _exact(manifest["assets"], expected_assets, "asset table/pixel binding differs")
    _validate_arrays(arrays, spec)
    # Recheck only files actually opened here; no original protocol/hardware access.
    for name, data in captured.items():
        if _file_digest(root / name) != _digest(data):
            raise ValueError("pilot file changed during verification: " + name)
    if _file_digest(root / MANIFEST) != manifest_sha or _file_digest(Path(__file__).resolve()) != source_sha:
        raise ValueError("manifest or generator changed during verification")
    return {"schema": "optical-lab-offline-pilot-verification-1", "verified": True,
            "manifest_sha256": manifest_sha, "external_manifest_identity_checked": expected_manifest_sha256 is not None,
            "checked_files": len(_FILES) + 1, "pilot_id": PILOT_ID, "captured_frames": 0,
            "scope": "offline_probe_file_identity_only_NOT_measurement_or_deployment",
            "hardware_accessed": False, "projection_executed": False,
            "hardware_release_approved": False, "control_release_approved": False}
