"""Isolated, hardware-disabled innovations; no changes to ongoing E0 sources."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPACT = ROOT.parent / "ballplate_optical_actor_ppo"
sys.path.insert(0, str(COMPACT))
import branch_runtime as base

LEGACY = base.LEGACY
digest, json_save, verify_sources = base.digest, base.json_save, base.verify_sources
physical_manifest, StopBudget = base.physical_manifest, base.StopBudget
DEADLINE_UNIX = 1792024481
GPU_UUID = "GPU-acf43839-e475-deee-1606-de0fa5b8c26e"


def setup(device="cuda"):
    result = base.setup_device(device, fraction=.4)
    if device == "cuda":
        import torch
        # New branch only; E0 and other projects retain their own allocation limits.
        torch.cuda.set_per_process_memory_fraction(.55)
        result["allocator_fraction"] = .55
    return result


def snapshot(output):
    import shutil
    dest = Path(output) / "source"
    dest.mkdir(exist_ok=False)
    for p in ROOT.glob("*.py"):
        shutil.copy2(p, dest / p.name)
    return {"innovation": {p.name: digest(p) for p in ROOT.glob("*.py")},
            "reused_physics_task": physical_manifest()}
