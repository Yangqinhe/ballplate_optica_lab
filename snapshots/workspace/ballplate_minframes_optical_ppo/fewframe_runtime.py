"""Independent one/two-frame trials; never alter the live four-frame dashboard."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
FOUR=ROOT.parent/"ballplate_fourframe_optical_ppo"
sys.path.insert(0,str(FOUR))
import fourframe_runtime as shared

digest,json_save,setup=shared.digest,shared.json_save,shared.setup
INNOV=shared.INNOV
FIXTURES=FOUR/"outputs/fourframe_teacher_mixture_seed97211_v1/fixtures.pt"
os.environ.setdefault("MUJOCO_GL","egl")


def sources():
    result=shared.sources()
    for name in ("fewframe_runtime.py","fewframe_model.py"):
        path=ROOT/name;result[str(path)]=digest(path)
    return result
