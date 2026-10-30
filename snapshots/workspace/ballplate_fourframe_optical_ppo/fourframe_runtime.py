"""Four raw DMD frames, one optical acquisition; no hardware SDK imports."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
INNOV = ROOT.parent / "ballplate_photonic_innovations"
sys.path.insert(0, str(INNOV))
import innov_runtime as reused

digest = reused.digest
json_save = reused.json_save
DATA = ROOT.parent / "ballplate_optical_actor_ppo/outputs/teacher_c0_seed73001_v1/dataset.pt"
os.environ.setdefault("MUJOCO_GL", "egl")


def setup(device="cuda"):
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-acf43839-e475-deee-1606-de0fa5b8c26e"
        os.environ["BALLPLATE_EXPECTED_GPU_UUID"] = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return reused.setup(device)


def sources():
    paths = [ROOT / name for name in ("fourframe_runtime.py", "fourframe_model.py")]
    paths += [INNOV / name for name in ("photonic_model.py", "canonical_region.py")]
    return {str(path): digest(path) for path in paths}
