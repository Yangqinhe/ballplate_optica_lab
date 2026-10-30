"""Explicit single-owner launcher; no automatic projection or restart."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;sys.path.insert(0,str(PROJECT))
from live_manual_v18.launch_realtime import desktop_command
from launch_calibration import environment
from live_manual_v19_trigger_r5.launch_realtime import LIMITS


def main():
    args=sys.argv[1:];persistent='--persistent' in args
    if persistent:args.remove('--persistent')
    for flag in ('--output','--restart-state','--recovery-record','--deployment-catalog','--result-folder','--adaptation-catalog','--loop-head','--preview-screenshot'):
        if flag in args:
            i=args.index(flag)+1;args[i]=str(Path(args[i]).resolve())
    env=environment(args);env.update(LIMITS);python=PROJECT/'.calibration-venv/bin/python'
    if not persistent:return subprocess.call([str(python),str(ROOT/'direct_alignment_gui.py'),*args],cwd=ROOT,env=env)
    output=Path(args[args.index('--output')+1])
    if output.exists():raise ValueError('new session required')
    unit='ballplate-optical-loop-v23-'+uuid.uuid4().hex[:12]+'.service'
    command=desktop_command(ROOT,python,args,env,unit)
    i=command.index(str(python));command=command[:i]+['--setenv='+k+'='+v for k,v in LIMITS.items()]+command[i:]
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    until=time.monotonic()+15
    while time.monotonic()<until:
        if (output/'heartbeat.json').exists():
            h=json.loads((output/'heartbeat.json').read_text())
            service=dict(unit=unit,pid=h['pid'],output=str(output),ui_version=19,trigger_revision=6,
                deployment_ui_revision=23,automatic_capture=False,automatic_projection=False,restart_policy='no')
            (output/'desktop_service.json').write_text(json.dumps(service,ensure_ascii=False,indent=2))
            print(json.dumps(service,ensure_ascii=False));return 0
        time.sleep(.1)
    raise RuntimeError('keep physical shielding; inspect '+unit)


if __name__=='__main__':raise SystemExit(main())
