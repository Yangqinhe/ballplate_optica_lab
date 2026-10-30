"""Isolated experiment; reuse pinned physical code without editing its project."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "ballplate_optical_diffraction_ppo"
sys.path.insert(0, str(LEGACY))
os.environ.setdefault("MUJOCO_GL", "egl")


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def physical_manifest():
    paths = list((LEGACY / "optical").glob("*.py"))
    paths += [LEGACY / "dashboard/acquisition.py", LEGACY / "env/ballplate_env.py",
              LEGACY / "env/reward.py", LEGACY / "env/ball_camera.py", LEGACY / "config.py"]
    return {str(p.relative_to(LEGACY)): digest(p) for p in sorted(paths)}


def verify_sources(manifest):
    for relative, expected in manifest.items():
        if digest(LEGACY / relative) != expected:
            raise RuntimeError(f"reused physical/task source changed: {relative}")


def json_save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def setup_device(device="cuda", fraction=.4):
    import torch
    torch.set_num_threads(2)
    if device != "cuda":
        return {"device": "cpu"}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not .1 <= fraction <= .45:
        raise ValueError("allocator fraction must be in [.1,.45]")
    torch.cuda.set_per_process_memory_fraction(fraction)
    properties = torch.cuda.get_device_properties(0)
    uuid = "GPU-" + str(properties.uuid).removeprefix("GPU-")
    expected = os.environ.get("BALLPLATE_EXPECTED_GPU_UUID")
    if expected and expected != uuid:
        raise RuntimeError("selected GPU UUID mismatch")
    return {"device": device, "name": properties.name, "uuid": uuid,
            "visible_devices": torch.cuda.device_count(), "allocator_fraction": fraction}


def snapshot(output):
    import shutil
    destination = Path(output) / "source"
    destination.mkdir(exist_ok=False)
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, destination / path.name)
    return {"branch": {p.name: digest(p) for p in ROOT.glob("*.py")},
            "reused_sources": physical_manifest()}


class StopBudget:
    def __init__(self, output, seconds):
        import signal
        self.output, self.seconds = Path(output), seconds
        self.started, self.requested = time.monotonic(), False
        self.previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous[sig] = signal.signal(sig, self.request)

    def request(self, *_):
        self.requested = True

    def __call__(self):
        return self.requested or (self.output / "stop.request").exists() or self.elapsed >= self.seconds

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def close(self):
        import signal
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
