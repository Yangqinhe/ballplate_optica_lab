"""Frame contracts for a future separately authorized hardware adapter."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import torch


@dataclass(frozen=True)
class CCDFrame:
    frame_id: int
    episode: int
    simulated_time_s: float
    captured_unix_s: float
    noise_seed: int
    raw_counts: np.ndarray
    features: np.ndarray
    cache_max_error: float
    acquisition_seconds: float
    source: str = "simulated_ccd"


@dataclass(frozen=True)
class HistorySlot:
    frame: CCDFrame
    repeated_fill: bool = False


class OpticalAcquisition(Protocol):
    """An acquisition must identify the input and return its matching feature."""
    def acquire(self, binary_image: np.ndarray, frame_id: int, episode: int,
                simulated_time_s: float, noise_seed: int) -> tuple[torch.Tensor, CCDFrame]: ...

    def close(self) -> None: ...


class DmdDevice(Protocol):
    def submit_binary_pattern(self, pattern: np.ndarray, pattern_id: int) -> None: ...
    def close(self) -> None: ...


class SlmDevice(Protocol):
    def load_calibrated_mask(self, gray: np.ndarray, mask_version: str,
                             calibration_id: str) -> None: ...
    def close(self) -> None: ...


class CcdDevice(Protocol):
    def read_matching_pattern(self, pattern_id: int, mask_version: str,
                              timeout_s: float) -> CCDFrame: ...
    def close(self) -> None: ...


class HardwareAcquisition:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "真实光路接口尚未启用：需要独立确认器件映射、SLM标定、触发与安全条件。"
            "此类不会导入SDK、打开相机或显示器件图案。")
