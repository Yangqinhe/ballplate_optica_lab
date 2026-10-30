"""Persistent v21 desktop service; no automatic camera start or projection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from live_manual_v18.launch_realtime import desktop_command
from launch_calibration import environment

LIMITS = dict(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
              NUMEXPR_NUM_THREADS='2', OMP_WAIT_POLICY='PASSIVE')


def main():
    args = sys.argv[1:]
    persistent = '--persistent' in args
    if persistent:
        args.remove('--persistent')
    for flag in ('--output', '--restart-state', '--recovery-record', '--deployment-catalog', '--result-folder'):
        if flag in args:
            i = args.index(flag)+1
            args[i] = str(Path(args[i]).resolve())
    env = environment(args); env.update(LIMITS)
    python = ROOT.parent/'.calibration-venv/bin/python'
    if not persistent:
        return subprocess.call([str(python), str(ROOT/'direct_alignment_gui.py'), *args], cwd=ROOT, env=env)
    output = Path(args[args.index('--output')+1])
    if output.exists():
        raise ValueError('refuse to overwrite GUI session')
    unit = 'ballplate-optical-display-v21-'+uuid.uuid4().hex[:12]+'.service'
    command = desktop_command(ROOT, python, args, env, unit)
    i = command.index(str(python))
    command = command[:i]+['--setenv='+k+'='+v for k,v in LIMITS.items()]+command[i:]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    deadline = time.monotonic()+12
    while time.monotonic() < deadline:
        if (output/'heartbeat.json').exists():
            state = json.loads((output/'heartbeat.json').read_text())
            service = dict(unit=unit, pid=state['pid'], output=str(output), ui_version=19,
                           trigger_revision=6, deployment_ui_revision=21, automatic_capture=False,
                           automatic_projection=False, restart_policy='no', numeric_thread_limits=LIMITS)
            (output/'desktop_service.json').write_text(json.dumps(service, ensure_ascii=False, indent=2))
            print(json.dumps(service, ensure_ascii=False)); return 0
        time.sleep(.1)
    raise RuntimeError('keep physical shielding and inspect '+unit)


if __name__ == '__main__':
    raise SystemExit(main())
