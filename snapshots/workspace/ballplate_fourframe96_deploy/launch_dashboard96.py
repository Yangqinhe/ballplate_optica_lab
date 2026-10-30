"""Simulation-only desktop launcher; no other processes/windows are closed."""
import os
from pathlib import Path
import subprocess
import sys


if __name__=='__main__':
    root=Path(__file__).resolve().parent;env=os.environ.copy()
    env.setdefault('MUJOCO_GL','egl');env['CUDA_VISIBLE_DEVICES']='GPU-acf43839-e475-deee-1606-de0fa5b8c26e'
    env['BALLPLATE_EXPECTED_GPU_UUID']=env['CUDA_VISIBLE_DEVICES'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    print('RAW96x4 / SLM C150 / CURRENT registered CCD8 / linear18 / SIMULATION ONLY',flush=True)
    raise SystemExit(subprocess.call([sys.executable,str(root/'dashboard96.py'),*sys.argv[1:]],cwd=root,env=env))
