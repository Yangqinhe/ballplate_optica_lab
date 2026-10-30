"""Bounded D/Z/L/Z/D via the ORIGINAL live GUI only. No SDK or restart.

Requires current user low-power/protection consent. Reuses already-cached
binary DMD and learned SLM files, verifies actual output pixels, and retains
an independent expiry watchdog. Never runs an actor or real actuators.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(PROJECT/'tools'),str(PROJECT)]
from static_learned_ccd_v28 import StaticUI,inspect,read_complete,digest,write_new,validate_assets,Input
from select_verified_mode_v17 import owned_popups
from workbench.registry import load_registry


def cached_assets(session,heartbeat):
    events=[json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()]
    result={}
    for role in ('dmd','slm'):
        e=[e for e in events if e.get('state')=='manual_file_loaded' and bool(e.get('SLM'))==(role=='slm')][-1]
        if digest(e['path'])!=e['sha256'] or heartbeat['unix_ns']<=e['unix_ns']:
            raise RuntimeError('cached source hash/causal publication invalid')
        result[role]=e
    return result


def dark(ui,name):
    ui.arm(False);ui.wait(lambda h:not h['projection_armed']);ui.guard(black=True)
    begin=time.time_ns();end=time.monotonic()+1.8
    while time.monotonic()<end:ui.guard(black=True);time.sleep(.1)
    rows=[]
    for _ in range(8):
        ui.guard(black=True);old=set(ui.session.glob('snapshot_*'));ui.button('snapshot')
        end=time.monotonic()+4
        while time.monotonic()<end:
            ui.guard(black=True);fresh=set(ui.session.glob('snapshot_*'))-old
            if len(fresh)>1:raise RuntimeError('ambiguous black snapshot')
            if fresh and (next(iter(fresh))/'metadata.json').exists():folder=next(iter(fresh));break
            time.sleep(.04)
        else:raise TimeoutError('fresh black snapshot missing')
        meta=read_complete(folder/'metadata.json');raw=np.load(folder/'CCD_raw.npy',allow_pickle=False)
        frame=meta['frame']
        if (meta['camera_active']!=ui.active or raw.shape!=(554,552) or raw.dtype!=np.uint8
                or raw.max()>=240 or frame['host_get_started_unix_ns']<begin+1_500_000_000
                or (rows and frame['TimeStamp']<=rows[-1]['frame']['TimeStamp'])):
            raise RuntimeError('black raw freshness/saturation/configuration gate failed')
        for role in ('dmd','slm'):
            if np.asarray(Image.open(folder/(role+'_canonical_requested.png'))).any():
                raise RuntimeError('black snapshot request was not all0')
        rows.append(dict(folder=str(folder),raw_sha256=digest(folder/'CCD_raw.npy'),frame=frame,
                         max_ADU=int(raw.max()),mean_ADU=float(raw.mean())))
        time.sleep(.1)
    result=dict(name=name,verified_unix_ns=begin,snapshots=rows)
    write_new(ui.root/(name+'.json'),result)
    print(json.dumps(dict(stage=name,frames=len(rows),max_ADU=max(r['max_ADU'] for r in rows))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--low-power-confirmed',action='store_true');a=p.parse_args()
    if not a.low_power_confirmed:raise RuntimeError('fresh current user confirmation required')
    cfg,paths=load_registry();session=PROJECT/cfg['runtime']['session'];before=inspect(session,parked=False)
    h=before['heartbeat']
    if (h['hardware_fault'] or not h['live_camera'] or h['latest_frame']['age_seconds']>=1
            or h.get('loop_active') or h.get('adaptation_active') or h.get('deployment_busy')
            or owned_popups(h['pid'],before['windows']['control']['wid'])
            or any(h['manual_settings'][role]!='file' for role in ('dmd','slm'))):
        raise RuntimeError('healthy original idle stream and cached file/file modes required')
    cached=cached_assets(session,h)
    dmd=np.load(cached['dmd']['path'],allow_pickle=False);phase=np.load(cached['slm']['path'],allow_pickle=False)
    validate_assets(dmd,phase)
    if digest(cached['slm']['path'])!=digest(paths['slm_raw_gray']):raise RuntimeError('not current learned phase')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    write_new(a.output/'PREFLIGHT.json',before)
    write_new(a.output/'PLAN.json',dict(role='FRESH_CURRENT_ALIGNMENT_DARK_ZERO_LEARNED_ZERO_DARK',
        authority='current user confirmed unchanged ND/protection/positions and bounded static consent',
        stages=['DARK_BEFORE','FLAT_BEFORE','LEARNED','FLAT_AFTER','DARK_AFTER'],saved_raw_frames=[8,3,3,3,8],
        cached_assets=cached,phase_sha256=digest(paths['slm_raw_gray']),dmd_sha256=digest(cached['dmd']['path']),
        maximum_seconds=120,no_new_asset_load=True,no_new_device_windows=True,no_SDK=True,
        no_camera_or_controller_parameter_writes=True,no_actor_or_closed_loop=True,
        software_settle_s=1.8,hardware_frame_pattern_sync_NOT_verified=True,
        source_sha256=digest(__file__),release_approved=False))
    lock=open(f'/run/user/{os.getuid()}/ballplate_current_public_15328133.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ui=StaticUI(session,before,a.output,time.monotonic()+120);ui.guard(black=True)
    identity=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    watcher=subprocess.Popen([sys.executable,str(PROJECT/'tools/static_learned_ccd_v28.py'),
        '--mode','watchdog','--session',str(session),'--output',str(a.output),'--parent',str(os.getpid()),
        '--parent-start',identity,'--hard-deadline',str(time.time()+140)])
    result=dict(completed=False,error=None,stages=[],new_camera_connection=False,closed_loop=False)
    started=time.monotonic()
    try:
        focus=Input(int(ui.main['wid'],16));focus.close()
        result['stages'].append(dark(ui,'DARK_BEFORE'))
        for name,mode in (('FLAT_BEFORE','zero'),('LEARNED','file'),('FLAT_AFTER','zero')):
            if watcher.poll() is not None:raise RuntimeError('independent watchdog exited early')
            result['stages'].append(ui.stage(name,dmd,phase,mode,0.))
        result['stages'].append(dark(ui,'DARK_AFTER'));result['completed']=True
    except BaseException as error:result['error']=repr(error)
    finally:
        try:
            ui.button('allzero');ui.wait(lambda s:not s['projection_armed']);ui.guard(black=True)
            result.update(actual_outputs_zero=True,camera_left_running=ui.state()['live_camera'])
        except BaseException as error:result.update(cleanup_error=repr(error),completed=False)
        result['wall_seconds']=time.monotonic()-started
        write_new(a.output/'RESULT.json',result);write_new(a.output/'FINISHED.json',dict(unix_ns=time.time_ns()))
        try:watcher.wait(timeout=6)
        except subprocess.TimeoutExpired:result['watchdog_completion_pending']=True
        ui.pixels.close();lock.close()
        print(json.dumps({k:v for k,v in result.items() if k!='stages'},ensure_ascii=False),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
