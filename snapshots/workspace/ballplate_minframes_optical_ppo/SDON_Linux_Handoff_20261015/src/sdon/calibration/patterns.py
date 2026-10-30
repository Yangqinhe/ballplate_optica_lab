"""Deterministic DMD targets and SLM phase patterns."""

from typing import Tuple

import cv2
import numpy as np

PATTERN_MODES = (
    "multiscale_fiducial",
    "asymmetric_l",
    "fixed_speckle",
    "checkerboard",
    "center_rectangle",
)


def build_alignment_pattern(
    mode: str = "multiscale_fiducial", size: int = 512, rectangle_size: int = 32
) -> np.ndarray:
    """Build an asymmetric, repeatable DMD alignment target."""
    if mode not in PATTERN_MODES:
        raise ValueError("unknown alignment pattern: %s" % mode)
    if size < 32:
        raise ValueError("alignment pattern size must be at least 32")

    pattern = np.zeros((512, 512), dtype=np.uint8)
    if mode == "multiscale_fiducial":
        white = 255
        cv2.line(pattern, (86, 92), (214, 92), white, 14, cv2.LINE_8)
        cv2.line(pattern, (86, 92), (86, 202), white, 14, cv2.LINE_8)
        cv2.line(pattern, (326, 108), (420, 108), white, 10, cv2.LINE_8)
        cv2.line(pattern, (420, 108), (420, 176), white, 10, cv2.LINE_8)
        cv2.line(pattern, (104, 414), (178, 414), white, 18, cv2.LINE_8)
        cv2.line(pattern, (104, 344), (104, 414), white, 18, cv2.LINE_8)
        cv2.circle(pattern, (174, 180), 24, white, -1, cv2.LINE_8)
        cv2.circle(pattern, (354, 174), 14, white, -1, cv2.LINE_8)
        cv2.circle(pattern, (364, 354), 30, white, 12, cv2.LINE_8)
        cv2.line(pattern, (238, 216), (238, 306), white, 14, cv2.LINE_8)
        cv2.line(pattern, (194, 260), (290, 260), white, 14, cv2.LINE_8)
        cv2.rectangle(pattern, (144, 282), (190, 328), white, 10, cv2.LINE_8)
        cv2.line(pattern, (226, 382), (310, 322), white, 12, cv2.LINE_8)
        cv2.circle(pattern, (302, 194), 7, white, -1, cv2.LINE_8)
    elif mode == "asymmetric_l":
        pattern[150:374, 154:182] = 255
        pattern[346:374, 154:342] = 255
        pattern[190:226, 300:336] = 255
    elif mode == "fixed_speckle":
        rng = np.random.default_rng(20260816)
        logical = (rng.random((32, 32)) < 0.35).astype(np.uint8) * 255
        pattern = np.repeat(np.repeat(logical, 16, axis=0), 16, axis=1)
    elif mode == "center_rectangle":
        side = max(1, min(512, int(rectangle_size)))
        x0 = (512 - side) // 2
        y0 = (512 - side) // 2
        pattern[y0 : y0 + side, x0 : x0 + side] = 255
    else:
        y, x = np.indices((512, 512))
        pattern = (((x * 32 // 512) + (y * 32 // 512)) % 2 * 255).astype(np.uint8)

    if size != 512:
        pattern = cv2.resize(pattern, (size, size), interpolation=cv2.INTER_NEAREST)
    return pattern


def build_offset_rectangle_pattern(
    size: int = 512,
    rectangle_size: int = 128,
    center_offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Build a binary square at a logical offset from the DMD centre.

    The requested square must fit completely on the logical DMD plane.  This
    is used by the full-aperture phase check so clipping cannot silently turn
    a corner measurement into a different-sized illumination aperture.
    """
    size = int(size)
    side = int(rectangle_size)
    offset_x, offset_y = int(center_offset[0]), int(center_offset[1])
    if size < 1 or side < 1 or side > size:
        raise ValueError("rectangle and DMD sizes must be positive, with rectangle <= DMD")
    x0 = (size - side) // 2 + offset_x
    y0 = (size - side) // 2 + offset_y
    if x0 < 0 or y0 < 0 or x0 + side > size or y0 + side > size:
        raise ValueError("offset rectangle must fit completely on the DMD")
    pattern = np.zeros((size, size), dtype=np.uint8)
    pattern[y0 : y0 + side, x0 : x0 + side] = 255
    return pattern


def generate_lens_phase(
    shape: Tuple[int, int] = (512, 512),
    focal_length: float = 0.4,
    center_offset: Tuple[float, float] = (0.0, 0.0),
    wavelength: float = 532e-9,
    pixel_pitch: float = 8e-6,
) -> np.ndarray:
    """Generate an 8-bit thin-lens phase map over [0, 2π)."""
    x = (np.arange(shape[1]) - shape[1] // 2 - center_offset[0]) * pixel_pitch
    y = (np.arange(shape[0]) - shape[0] // 2 - center_offset[1]) * pixel_pitch
    xx, yy = np.meshgrid(x, y)
    phase = -np.pi / (wavelength * focal_length) * (xx ** 2 + yy ** 2)
    return (np.mod(phase, 2 * np.pi) / (2 * np.pi) * 256).astype(np.uint8)


def generate_blazed_grating(
    shape: Tuple[int, int] = (512, 512),
    period_px: int = 32,
    axis: str = "x",
    direction: int = 1,
) -> np.ndarray:
    """Generate one wrapped 0–2π phase ramp for an SLM direction check.

    ``direction`` describes the logical phase-gradient sign before the
    HardwareManager applies the installation-specific horizontal SLM flip.
    The measured CCD result is therefore used to record the real coordinate
    mapping instead of assuming that logical +X must appear at CCD right.
    """
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("grating shape must contain two positive dimensions")
    if int(period_px) < 2:
        raise ValueError("grating period must be at least 2 pixels")
    if axis not in {"x", "y"}:
        raise ValueError("grating axis must be x or y")
    if int(direction) not in {-1, 1}:
        raise ValueError("grating direction must be +1 or -1")

    coordinate = np.arange(shape[1] if axis == "x" else shape[0], dtype=np.int64)
    phase = np.mod(int(direction) * coordinate, int(period_px))
    phase = np.floor(phase.astype(np.float64) / int(period_px) * 256.0).astype(np.uint8)
    if axis == "x":
        return np.repeat(phase[None, :], shape[0], axis=0)
    return np.repeat(phase[:, None], shape[1], axis=1)


def generate_binary_phase_grating(
    shape: Tuple[int, int] = (512, 512),
    test_gray: int = 128,
    period_px: int = 32,
    axis: str = "x",
    reference_gray: int = 0,
) -> np.ndarray:
    """Alternate reference/test gray levels with a 50% duty cycle."""
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("grating shape must contain two positive dimensions")
    if int(period_px) < 4 or int(period_px) % 2:
        raise ValueError("binary grating period must be an even integer >= 4")
    if axis not in {"x", "y"}:
        raise ValueError("grating axis must be x or y")
    if not 0 <= int(test_gray) <= 255 or not 0 <= int(reference_gray) <= 255:
        raise ValueError("SLM gray values must be in [0,255]")
    length = shape[1] if axis == "x" else shape[0]
    coordinate = np.arange(length)
    selector = coordinate % int(period_px) >= int(period_px) // 2
    line = np.where(selector, int(test_gray), int(reference_gray)).astype(np.uint8)
    if axis == "x":
        return np.repeat(line[None, :], shape[0], axis=0)
    return np.repeat(line[:, None], shape[1], axis=1)
