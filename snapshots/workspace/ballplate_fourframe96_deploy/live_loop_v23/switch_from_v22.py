"""One confirmed, normal-close switch from v22; no forced kill or projection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tools'),str(PROJECT/'live_manual_v17')]
from live_loop_v23.core import DEFAULT_HEAD,DEFAULT_SHA,load_head
from live_adapt_v22.core import digest,write_new
from live_deploy_v20.contract import matching
from inspect_recovery_ui_v20 import inspect
import restart_alignment_gui as restart


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--light-blocked-confirmed',action='store_true')
    p.add_argument('--old-session',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.light_blocked_confirmed:raise RuntimeError('fresh physical shielding confirmation required; no changes')
    if a.output.exists():raise ValueError('new switch folder required')
    proof=inspect(a.old_session,parked=True)
    owner=restart.process(proof['pid'])
    if owner is None or Path(owner['session'])!=a.old_session.resolve() or len(restart.owners())!=1:
        raise RuntimeError('unique unchanged old owner required')
    if proof['heartbeat']['hardware_fault']:raise RuntimeError('old fault needs separate recovery')
    c=json.loads(a.contract.read_text());h,b=load_head(DEFAULT_HEAD,DEFAULT_SHA,audit=True)
    matching(b.contract,c['settings'],c['optical_config'],c['expected_active'],c['registration'])
    if c['original_pid']!=proof['pid'] or c['settings']!=proof['heartbeat']['manual_settings']:
        raise RuntimeError('current stopped controls differ from saved switch contract')
    a.output.mkdir(parents=True,exist_ok=False)
    write_new(a.output/'OLD_OWNER_ZERO_PROOF.json',proof)
    marker=a.output/'RECOVERY_RECORD.json'
    write_new(marker,dict(schema='manual-alignment-recovery-v1',camera_baseline=c['original_fault_before'],
        baseline_source=str(a.contract.resolve()),baseline_source_sha256=digest(a.contract),
        original_fault=None,old_pid=proof['pid'],manual_stop_requires_recheck=True))
    env=dict(os.environ,DISPLAY=':1',XAUTHORITY='/run/user/1000/gdm/Xauthority')
    launch=[str(PROJECT/'.calibration-venv/bin/python'),str(PROJECT/'live_loop_v23/launch_realtime.py'),
        '--persistent','--hardware','--light-blocked-confirmed','--output',str((a.output/'GUI').resolve()),
        '--restart-state',str(a.contract.resolve()),'--recovery-record',str(marker.resolve()),
        '--adaptation-catalog',str(b.path),'--loop-head',str(DEFAULT_HEAD)]
    write_new(a.output/'SWITCH_PLAN.json',dict(old_owner=owner,command=launch,
        light_blocked_confirmed_for_this_switch=True,no_auto_capture=True,no_auto_projection=True,
        no_forced_termination=True,real_actuators_enabled=False))
    subprocess.run([str(PROJECT/'.calibration-venv/bin/python'),str(PROJECT/'restart_alignment_gui.py'),
        '--_native-exit-pid',str(proof['pid']),'--light-blocked-confirmed'],
        cwd=PROJECT,env=env,check=True,timeout=18)
    if not restart.wait_gone(owner,5) or restart.owners():
        raise RuntimeError('old owner still present; no new windows created; no forced stop')
    subprocess.run(launch,cwd=PROJECT,env=env,check=True,timeout=22)
    # Inspector compares actual display pixels and original source hashes.
    current=inspect(a.output/'GUI',parked=True)
    write_new(a.output/'NEW_OWNER_ZERO_PROOF.json',current)
    print(json.dumps(dict(session=str((a.output/'GUI').resolve()),pid=current['pid'],
        actual_DMD_SLM_zero=True,CCD_stopped=True,recheck_required=True,
        auto_capture=False,auto_projection=False),ensure_ascii=False))


if __name__=='__main__':main()
