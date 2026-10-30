"""Explicit one-shot shielded launch of the sealed two-page configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'E053_raw_workbench_20261026_1'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.execute:print('{"default_IO":0}');return
    if not a.light_blocked_confirmed or (ROOT/'LAUNCH_PLAN.json').exists() or Path('/proc/1918336').exists():
        raise RuntimeError('fresh shielded one-shot launch after old exit required')
    config=json.loads((ROOT/'CONFIG.json').read_text())
    for filename,digest in config['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:raise RuntimeError('source changed: '+filename)
    digest=hashlib.sha256((ROOT/'CONFIG.json').read_bytes()).hexdigest()
    command=json.loads((OLD/'LAUNCH_PLAN.json').read_text())['command']
    replacements={str(OLD/'run.py'):str(ROOT/'run.py'),
        '--unit=ballplate-optical-raw-workbench-20261026-e053.service':'--unit=ballplate-optical-state-validation-20261026-e054.service',
        '28561451285c5629ee3c39d7a8c8ef17496edc531631c39615a42566d6fe54a0':digest}
    if any(command.count(key)!=1 for key in replacements):raise RuntimeError('original launcher layout differs')
    command=[replacements.get(x,x) for x in command]
    with (ROOT/'LAUNCH_PLAN.json').open('x') as f:json.dump(dict(command=command,physical_shielding_confirmed=True,
        no_auto_capture=True,no_auto_projection=True,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,ensure_ascii=False,indent=2)
    result=subprocess.run(command,text=True,capture_output=True,timeout=20)
    with (ROOT/'LAUNCH_RESULT.json').open('x') as f:json.dump(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),f,ensure_ascii=False,indent=2)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
