"""Launch r3 with numeric thread limits and explicit preserved fault state."""
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


def command(arguments, env, unit):
    python = ROOT.parent / '.calibration-venv/bin/python'
    cmd = desktop_command(ROOT, python, arguments, env, unit)
    index = cmd.index(str(python))
    return cmd[:index] + ['--setenv=' + k + '=' + v for k, v in LIMITS.items()] + cmd[index:]


def main():
    args = sys.argv[1:]; persistent = '--persistent' in args
    if persistent: args.remove('--persistent')
    for flag in ('--output', '--restart-state'):
        if flag in args:
            index = args.index(flag) + 1; args[index] = str(Path(args[index]).resolve())
    env = environment(args); env.update(LIMITS)
    python = ROOT.parent / '.calibration-venv/bin/python'
    if not persistent: return subprocess.call([str(python), str(ROOT/'direct_alignment_gui.py'), *args], cwd=ROOT, env=env)
    if '--output' not in args: raise ValueError('explicit new output session required')
    output = Path(args[args.index('--output') + 1])
    if output.exists(): raise ValueError('refuse to overwrite existing session')
    unit = 'ballplate-optical-line0-r3-' + uuid.uuid4().hex[:12] + '.service'
    subprocess.run(command(args, env, unit), check=True, cwd=ROOT, env=env)
    end = time.monotonic() + 12
    while time.monotonic() < end:
        if (output/'heartbeat.json').exists():
            h = json.loads((output/'heartbeat.json').read_text())
            row = dict(unit=unit, pid=h['pid'], output=str(output), ui_version=19, trigger_revision=3,
                numeric_thread_limits=LIMITS, automatic_capture=False, automatic_projection=False,
                restart_policy='no', startup_light_blocked_confirmed_by_user=True)
            (output/'desktop_service.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
            print(json.dumps(row, ensure_ascii=False)); return 0
        time.sleep(.1)
    raise RuntimeError('keep light blocked and inspect service ' + unit)


if __name__ == '__main__': raise SystemExit(main())
