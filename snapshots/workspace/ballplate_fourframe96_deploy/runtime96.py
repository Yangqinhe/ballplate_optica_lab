"""Read-only archive imports; no SDK, Qt windows, USB, or display enumeration."""
from pathlib import Path
import sys
import os

ROOT=Path(__file__).resolve().parent
MIN=ROOT.parent/"ballplate_minframes_optical_ppo"
HANDOFF=MIN/"SDON_Linux_Handoff_20261015"
sys.path.insert(0,str(MIN))
import fewframe_runtime as shared
sys.path.insert(0,str(HANDOFF/"src"))
setup,digest,json_save=shared.setup,shared.digest,shared.json_save
FIXTURES=shared.FIXTURES
REFERENCE=HANDOFF/"outputs/mnist_sdon/video_bound_requalification_20260910"
PHASE=REFERENCE/"slm_phase_calibration_20260910_100213"
os.environ.setdefault("MUJOCO_GL","egl")


def sources():
    paths=[ROOT/"runtime96.py",ROOT/"model96.py",HANDOFF/"src/mnist_sdon/optical.py",
           HANDOFF/"src/mnist_sdon/config.py",HANDOFF/"src/sdon/config.py",
           PHASE/"slm_phase_lut.npy",PHASE/"slm_phase_response_rad.npy",
           PHASE/"slm_phase_calibration_summary.json",REFERENCE/"alignment_config_latest.json"]
    return {**shared.sources(),**{str(path):digest(path) for path in paths}}
