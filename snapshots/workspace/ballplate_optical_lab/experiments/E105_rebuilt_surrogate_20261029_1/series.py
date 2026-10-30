"""Explicit authorized serial batches through existing owner. Stops on any failure."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from workflow import ROOT,LAB,SESSION,read,write,ref,require,make_client,validate_current

# E105's earlier declared 08:51 HKT finish takes precedence over the later
# baseline timestamp. Reserve six minutes for existing-window control restore.
LAST_BATCH_END_NS=1793321100*10**9  # 2026-10-30 08:45 HKT


def capture_end_ns():
    return min(read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'],LAST_BATCH_END_NS)


def counter_margin(frame_id):
    require(isinstance(frame_id,int) and 0<=frame_id<=65535,'observed 16-bit frame counter required')
    # This does not accept/relabel a wrapped RAW. The old sampler remains intact.
    return 65535-frame_id >= 300*90+512


def wait_counter_window(output):
    """Read-only all-zero scheduling around a confirmed old sampler wrap bug."""
    output.mkdir();baseline=read(ROOT/'BASELINE_01/BASELINE.json')
    client=make_client(output,baseline['expected']);client.ui.deadline=time.monotonic()+410
    start=time.monotonic();history=[]
    while time.monotonic()-start<400:
        require(time.time_ns()+300*10**9<capture_end_ns(),'no time for another full batch and session restore')
        h=client.state();validate_current(h,baseline['expected'])
        require(not h['projection_armed'],'wrap wait requires existing outputs disarmed')
        client.independent_zero();frame=h['latest_frame'];fid=frame['frame_id']
        require(0<frame['fps']<=90,'trigger-rate upper bound changed; no automatic scheduling')
        history.append(dict(unix_ns=h['unix_ns'],frame_id=fid,fps=frame['fps']))
        if counter_margin(fid):
            write(output/'RESULT.json',dict(history=history,elapsed_s=time.monotonic()-start,
                actual_zero_verified=True,no_clicks=True,no_camera_change=True,no_RAW_validation_change=True))
            return
        if len(history)%15==1:print('ALL-ZERO counter-window wait',fid,flush=True)
        time.sleep(2)
    raise TimeoutError('no safe counter window within bounded all-zero wait')


def run(names,prefix,authorization,catalog='PILOT',frames=20,repeats=3,assets='ASSETS_02',roles=None):
    require(authorization.strip(),'explicit authorization required')
    out=ROOT/prefix;out.mkdir()
    require(time.time_ns()<capture_end_ns(),'capture time window ended')
    plan=read(ROOT/assets/'PLAN.json')
    jobs=([j for j in plan['jobs'] if j['role'] in roles] if roles else
          [dict(mask=name,catalog=catalog,frames=frames,repeats=repeats) for name in names])
    require(len(jobs)>0,'empty explicit schedule')
    if roles:
        require(set(roles)<=set(j['role'] for j in jobs),'unknown role')
        gate=read(ROOT/'STRONG_FULL_RESPONSE_01/RESULT.json')
        require(gate['identifiable_fraction']>=.8,'strong pilot observability not passed')
    source=ref(ROOT/'workflow.py')
    write(out/'SCHEDULE.json',dict(masks=names,jobs=jobs,roles=roles,source=ref(__file__),capture_source=source,
        catalog=catalog,assets_plan=ref(ROOT/assets/'PLAN.json'),frames=frames,repeats=repeats,each_batch_maximum_seconds=300,
        capture_end_unix_ns=capture_end_ns(),six_minutes_reserved_for_session_restore=True,
        authorization=authorization,stop_on_first_failure=True,no_closed_loop=True,no_restart=True))
    results=[]
    for i,job in enumerate(jobs):
        require(ref(ROOT/'workflow.py')==source,'capture source changed during schedule')
        # Sampling failure at 65533->1 was preserved. Existing GUI recognizes
        # the camera wrap but its finite sampler requires precisely65535->0.
        # Schedule away from it without patching the running GUI or its gates.
        # The pinned legacy display adapter is import-once. Give each read-only
        # check an isolated process, like each normal capture; never a new GUI.
        wait_path=out/('COUNTER_WINDOW_%02d'%i)
        wait_code=subprocess.call([sys.executable,'-B',__file__,'--counter-wait-output',str(wait_path),
            '--execute-authorized','--authorization',authorization],cwd=LAB)
        if wait_code:
            write(out/'STOPPED.json',dict(reason='read-only all-zero preflight failed; no next projection',batches=results))
            return wait_code
        require(time.time_ns()+300*10**9<capture_end_ns(),
                'insufficient remaining authorization for full bounded batch')
        name=job['mask'];dest=prefix+'_'+(job['catalog'] if roles else name)
        cmd=[str(LAB.parent/'ballplate_fourframe96_deploy/.calibration-venv/bin/python'),'-B',str(ROOT/'workflow.py'),
            '--capture','--execute-authorized','--catalog',job['catalog'],'--mask',name,'--frames',str(job['frames']),'--repeats',str(job['repeats']),
            '--output',dest,'--assets',assets,'--authorization',authorization]
        # Child owns explicit <=300s task and original GUI owns its own bound.
        # Do not abruptly kill a still-restoring operator or recreate windows.
        start=time.monotonic();code=subprocess.call(cmd,cwd=LAB)
        item=dict(index=i,mask=name,returncode=code,seconds=time.monotonic()-start,folder=dest)
        write(out/('BATCH_%02d.json'%i),item);results.append(item)
        if code!=0:
            write(out/'STOPPED.json',dict(reason='failed batch; no next projection',batches=results))
            return 1
        require(read(ROOT/dest/'RESULT.json')['completed'],'missing completion')
        print('SERIES',i+1,'/',len(jobs),'completed',job['catalog'],name,flush=True)
    write(out/'RESULT.json',dict(completed=True,batches=results))
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--masks',nargs='+');p.add_argument('--roles',nargs='+');p.add_argument('--prefix')
    p.add_argument('--counter-wait-output')
    p.add_argument('--catalog',default='PILOT')
    p.add_argument('--assets',default='ASSETS_02')
    p.add_argument('--frames',type=int,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--execute-authorized',action='store_true');p.add_argument('--authorization',default='');a=p.parse_args()
    if not a.execute_authorized:
        print('OFFLINE: no projection; explicit authorization required.');sys.exit(0)
    if a.counter_wait_output:
        target=Path(a.counter_wait_output).resolve()
        require(target.is_relative_to(ROOT) and target!=ROOT and bool(a.authorization.strip()),'explicit authorized local wait output required')
        require(time.time_ns()<read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'],'authorization expired')
        try:wait_counter_window(target)
        except Exception as error:
            if target.is_dir():write(target/'FAILED.json',dict(error=repr(error),no_clicks=True,no_projection=True))
            raise
        sys.exit(0)
    require(a.prefix is not None,'explicit series prefix required')
    require(bool(a.masks)!=bool(a.roles),'choose explicit mask list OR design roles')
    require(all(s.replace('_','').isalnum() for s in [a.prefix,a.assets,*(a.masks or []),*(a.roles or [])]),'simple experiment identifiers required')
    sys.exit(run(a.masks,a.prefix,a.authorization,a.catalog,a.frames,a.repeats,a.assets,a.roles))
