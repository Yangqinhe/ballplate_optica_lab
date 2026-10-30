"""Typed project configuration loaded from a local TOML file."""

import ast
import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    tomllib = None


@dataclass(frozen=True)
class DisplayConfig:
    dmd_screen: int = 0
    slm_screen: int = 1
    logical_size: Tuple[int, int] = (32, 32)
    modulated_size: Tuple[int, int] = (512, 512)
    # Optional per-device canvases. The legacy modulated_size remains the
    # DMD/CCD registration grid and is used when these keys are omitted.
    dmd_modulated_size: Tuple[int, int] = (512, 512)
    slm_modulated_size: Tuple[int, int] = (512, 512)
    dmd_offset: Tuple[int, int] = (0, 0)
    slm_offset: Tuple[int, int] = (130, 233)


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    roi_offset_x: int = 0
    roi_offset_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    use_trigger: bool = True
    software_trigger: bool = False
    trigger_source: str = "Line0"
    # Empty preserves legacy behavior. BFLY-PGE-50A2 ONN diagnostics explicitly
    # use Rolling after trigger setup; GlobalReset is not a global shutter.
    sensor_shutter_mode: str = ""
    exposure_ms: float = 4.045
    gain_db: float = 0.0
    black_level: float = 0.0
    rotation_deg: float = -89.8
    crop_top: int = 5
    crop_bottom: int = 0
    crop_left: int = 3
    crop_right: int = 3
    show_preview: bool = True


@dataclass(frozen=True)
class TimingConfig:
    dmd_min_interval_s: float = 0.00833
    slm_min_interval_s: float = 0.01667
    intensity_settle_ms: int = 50
    alignment_settle_ms: int = 100
    samples: int = 3


@dataclass(frozen=True)
class OpticsConfig:
    wavelength_m: float = 532e-9
    pixel_pitch_m: float = 8e-6
    focal_length_1_m: float = 0.15
    focal_length_2_m: float = 0.15
    alignment_lens_focal_length_m: float = 0.4
    padding_factor: int = 3


@dataclass(frozen=True)
class PathsConfig:
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class AppConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _tuple2(value: Any, name: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("%s must contain exactly two integers" % name)
    return int(value[0]), int(value[1])


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("[%s] must be a TOML table" % name)
    return value


def _parse_toml_value(value: str) -> Any:
    """Parse the scalar/array subset used by config.example.toml on Python 3.8."""
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


def _load_toml(path: Path) -> Dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    # Python 3.8 compatibility for this project's deliberately simple TOML
    # schema: named tables containing strings, numbers, booleans, and arrays.
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return {
        section: {key: _parse_toml_value(value) for key, value in parser.items(section)}
        for section in parser.sections()
    }


def load_config(path: Path) -> AppConfig:
    """Load configuration, using documented defaults when the file is absent."""
    path = Path(path)
    data: Dict[str, Any] = {}
    if path.exists():
        data = _load_toml(path)

    display = _section(data, "display")
    camera = _section(data, "camera")
    timing = _section(data, "timing")
    optics = _section(data, "optics")
    paths = _section(data, "paths")

    display_defaults = DisplayConfig()
    legacy_modulated_size = _tuple2(
        display.get("modulated_size", display_defaults.modulated_size), "modulated_size"
    )
    display_config = DisplayConfig(
        dmd_screen=int(display.get("dmd_screen", display_defaults.dmd_screen)),
        slm_screen=int(display.get("slm_screen", display_defaults.slm_screen)),
        logical_size=_tuple2(
            display.get("logical_size", display_defaults.logical_size), "logical_size"
        ),
        modulated_size=legacy_modulated_size,
        dmd_modulated_size=_tuple2(
            display.get("dmd_modulated_size", legacy_modulated_size),
            "dmd_modulated_size",
        ),
        slm_modulated_size=_tuple2(
            display.get("slm_modulated_size", legacy_modulated_size),
            "slm_modulated_size",
        ),
        dmd_offset=_tuple2(display.get("dmd_offset", display_defaults.dmd_offset), "dmd_offset"),
        slm_offset=_tuple2(display.get("slm_offset", display_defaults.slm_offset), "slm_offset"),
    )

    camera_defaults = CameraConfig()
    camera_config = CameraConfig(
        **{
            key: camera.get(key, getattr(camera_defaults, key))
            for key in CameraConfig.__dataclass_fields__
        }
    )
    timing_defaults = TimingConfig()
    timing_config = TimingConfig(
        **{
            key: timing.get(key, getattr(timing_defaults, key))
            for key in TimingConfig.__dataclass_fields__
        }
    )
    optics_defaults = OpticsConfig()
    optics_config = OpticsConfig(
        **{
            key: optics.get(key, getattr(optics_defaults, key))
            for key in OpticsConfig.__dataclass_fields__
        }
    )
    output_dir = Path(paths.get("output_dir", PathsConfig().output_dir))
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()

    if display_config.dmd_screen == display_config.slm_screen:
        raise ValueError("DMD and SLM must use different screen indexes")
    if display_config.modulated_size[0] != display_config.modulated_size[1]:
        raise ValueError("display.modulated_size must be square for the alignment model")
    for name, size in (
        ("dmd_modulated_size", display_config.dmd_modulated_size),
        ("slm_modulated_size", display_config.slm_modulated_size),
    ):
        if size[0] != size[1] or size[0] < 1:
            raise ValueError("display.%s must be a positive square" % name)
    if timing_config.samples < 1:
        raise ValueError("timing.samples must be at least 1")
    if camera_config.sensor_shutter_mode not in {"", "Global", "Rolling", "GlobalReset"}:
        raise ValueError("camera.sensor_shutter_mode must be empty, Global, Rolling or GlobalReset")
    if (
        min(
            camera_config.roi_offset_x,
            camera_config.roi_offset_y,
            camera_config.roi_width,
            camera_config.roi_height,
        )
        < 0
    ):
        raise ValueError("camera ROI values must be non-negative")
    if optics_config.padding_factor < 1:
        raise ValueError("optics.padding_factor must be at least 1")

    return AppConfig(
        display=display_config,
        camera=camera_config,
        timing=timing_config,
        optics=optics_config,
        paths=PathsConfig(output_dir=output_dir),
    )


def write_camera_roi(path: Path, roi: Dict[str, Any]) -> None:
    """Persist the camera ROI while preserving the rest of the TOML file."""
    path = Path(path)
    names = {
        "roi_offset_x": int(roi["OffsetX"]),
        "roi_offset_y": int(roi["OffsetY"]),
        "roi_width": int(roi["Width"]),
        "roi_height": int(roi["Height"]),
    }
    if min(names.values()) < 0:
        raise ValueError("camera ROI values must be non-negative")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = re.search(r"(?m)^\[camera\]\s*(?:\r?\n|$)", text)
    if section is None:
        if text and not text.endswith(("\n", "\r")):
            text += "\n"
        text += "\n[camera]\n"
        section = re.search(r"(?m)^\[camera\]\s*(?:\r?\n|$)", text)
    assert section is not None
    next_section = re.search(r"(?m)^\[[^\]]+\]\s*(?:\r?\n|$)", text[section.end() :])
    end = section.end() + next_section.start() if next_section else len(text)
    body = text[section.end() : end]
    newline = "\r\n" if "\r\n" in text else "\n"
    for name, value in names.items():
        pattern = re.compile(
            r"(?m)^(\s*" + re.escape(name) + r"\s*=\s*)[^#\r\n]*(\s*(?:#.*)?)(\r?\n|$)"
        )
        if pattern.search(body):
            body = pattern.sub(
                lambda match, v=value: "%s%d%s%s"
                % (match.group(1), v, match.group(2), match.group(3)),
                body,
                count=1,
            )
        else:
            body += "%s = %d%s" % (name, value, newline)
    updated = text[: section.end()] + body + text[end:]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(path)
