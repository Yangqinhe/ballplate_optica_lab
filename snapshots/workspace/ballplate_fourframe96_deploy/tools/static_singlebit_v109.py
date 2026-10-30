"""NEW bounded 1-bit optical comparison in ORIGINAL visible GUI only.

Uses exact currently cached DMD/SLM files; no file dialog, new output window,
SDK concurrency, mode/LUT changes or MuJoCo. Public UI snapshots are single
CCD frames, not averaged control inputs. Old overnight authority is not used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tools'),str(PROJECT/'live_manual_v17')]
from optical_ui_v19 import OpticalUI
from inspect_recovery_ui_v20 import inspect
from diagnose_black_restart_v83 import TimedBlackHold
from qualify_manual_black_v108 import Controller,validate_target
from camera_recheck import verify_baseline
from live_adapt_v22.core import digest,write_new
from black_pixels import RootPixels
from display_linux import read_roles,same_signals
from realtime_model import LivePattern,patterns
from select_verified_mode_v17 import window


def latest_assets(session):
    events=[json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()]
    result={}
    for role,is_slm in (('dmd',False),('slm',True)):
        event=[e for e in events if e.get('state')=='manual_file_loaded' and e.get('SLM')==is_slm][-1]
        path=Path(event['path']);array=np.load(path,allow_pickle=False)
        if digest(path)!=event['sha256'] or array.shape!=(600,600) or array.dtype!=np.uint8:
            raise ValueError('cached source hash/shape/type invalid')
        if role=='dmd' and not np.isin(array,(0,255)).all():raise ValueError('nonbinary DMD')
        if role=='slm' and not np.array_equal(array,array[::4,::4].repeat(4,0).repeat(4,1)):
            raise ValueError('SLM must retain exact 4x4 macropixels')
        result[role]=dict(path=str(path.resolve()),sha256=digest(path),event=event)
    return result


def expected_pixels(settings,dmd,phase,illuminated,mode):
    config=LivePattern(**dict(settings,slm=mode))
    a,b=patterns(config,dmd,phase)
    if not illuminated:a=np.zeros_like(a);b=np.zeros_like(b)
    return dict(dmd=np.pad(a,((0,0),(100,100))),slm=np.ascontiguousarray(np.fliplr(b))),a,b


def read_complete(path):
    end=time.monotonic()+2
    while True:
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>end:raise
            time.sleep(.03)


def watchdog_expired(alive,proof,now_ns,hard_deadline):
    return (not alive or proof is None or not 0<=now_ns-proof['unix_ns']<=20_000_000_000
            or now_ns/1e9>hard_deadline)


def find_tab(ui):
    import gi
    gi.require_version('Atspi','2.0');from gi.repository import Atspi
    from trigger_ui_v19 import nodes,inside,extents
    desktop=Atspi.get_desktop(0)
    tabs=[]
    for i in range(desktop.get_child_count()):
        app=desktop.get_child_at_index(i)
        if app.get_process_id()!=ui.pid or app.get_name()!='ballplate-realtime-manual':continue
        tabs.extend(n for n in nodes(app) if n.get_role_name()=='page tab' and n.get_name()=='手动对齐')
    if len(tabs)!=1 or not inside(extents(tabs[0],Atspi),ui.main['rect']):
        raise RuntimeError('unique original ordinary-monitor manual tab required')
    action=tabs[0].get_action_iface()
    ids=[i for i in range(action.get_n_actions()) if action.get_action_name(i).lower() in ('press','click','activate')]
    if len(ids)!=1 or not action.do_action(ids[0]):raise RuntimeError('manual tab action rejected')


def watchdog(session,folder,parent,process_start,hard_deadline):
    before=read_complete(folder/'ORIGINAL_PREFLIGHT.json');ui=OpticalUI(session,before)
    result=dict(intervened=False,error=None,no_SDK=True,no_new_windows=True)
    try:
        while True:
            if (folder/'FINISHED.json').exists():
                result['final_black']=inspect(session,parked=True)
                result['normal_finish_verified']=True;break
            try:
                identity=Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()[19]
                alive=identity==process_start
            except FileNotFoundError:alive=False
            proofs=sorted(folder.glob('PROGRESS_*.json'))
            proof=read_complete(proofs[-1]) if proofs else None
            expired=watchdog_expired(alive,proof,time.time_ns(),hard_deadline)
            if expired:
                result['intervened']=True
                write_new(folder/'WATCHDOG_INTERVENED.json',dict(parent_alive=alive,expired=expired,unix_ns=time.time_ns()))
                ui.button('allzero')
                time.sleep(.2)
                h=ui.state(allow_fault=True)
                if h['busy'] or h['live_camera']:ui.button('release')
                end=time.monotonic()+12
                while time.monotonic()<end:
                    h=ui.state(allow_fault=True)
                    if not h['busy'] and not h['live_camera']:break
                    time.sleep(.1)
                result['final_black']=inspect(session,parked=True)
                break
            time.sleep(.2)
    except BaseException as error:result['error']=repr(error)
    write_new(folder/'WATCHDOG_RESULT.json',result)
    if result['error']:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--manual-target',type=Path)
    p.add_argument('--profile-result',type=Path)
    p.add_argument('--camera-qualification',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--watchdog',action='store_true')
    p.add_argument('--parent',type=int);p.add_argument('--parent-start')
    p.add_argument('--hard-deadline',type=float)
    a=p.parse_args()
    if a.watchdog:return watchdog(a.session,a.output,a.parent,a.parent_start,a.hard_deadline)
    target=validate_target(json.loads(a.manual_target.read_text()))
    qualification=json.loads(a.camera_qualification.read_text())
    if not qualification['passed']:raise ValueError('successful new manual qualification required')
    active=qualification['rows'][0]['active']
    weights=np.load(a.weights,allow_pickle=False)
    if weights.shape!=(8,554,552) or not np.isfinite(weights).all() or weights.min()<0 or weights.max()>1:
        raise ValueError('eight fixed native readout masks required')
    assets=latest_assets(a.session)
    dmd=np.load(assets['dmd']['path'],allow_pickle=False);phase=np.load(assets['slm']['path'],allow_pickle=False)
    before=inspect(a.session,parked=True);settings=before['heartbeat']['manual_settings']
    if (settings['dmd']!='file' or settings['slm']!='file' or settings['full_panel']
            or not before['heartbeat']['geometry_frozen']):
        raise RuntimeError('unchanged cached files and locked geometry required')
    stages=[dict(name=n,illuminated=on,mode=mode) for n,on,mode in
        (('zero_before',False,'zero'),('flat_before',True,'zero'),
         ('structured',True,'file'),('flat_after',True,'zero'),('zero_after',False,'zero'))]
    plan=dict(role='NEW_SINGLEBIT_CACHED_INPUT_STRUCTURED_VS_FLAT_STATIC_ONLY',
        authority='current user 按你的方案继续 after manual-baseline qualification',
        active_seconds=150,maximum_seconds=210,frames_per_stage=8,settle_seconds=1.5,
        watchdog_progress_timeout_seconds=20,source_sha256=digest(__file__),
        manual_target_sha256=digest(a.manual_target),qualification_sha256=digest(a.camera_qualification),
        assets=assets,weights=str(a.weights.resolve()),weights_sha256=digest(a.weights),
        settings=settings,expected_active=active,stages=stages,
        single_CCD_frames_not_control_averaging=True,SLM_mask_hand_designed_NOT_PPO_trained=True,
        no_20ms_no_optical_advantage_claim=True,no_MuJoCo=True,no_new_windows=True,
        old_overnight_authorization_NOT_used=True,release_approved=False)
    if not a.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    a.output.mkdir(parents=True,exist_ok=False);write_new(a.output/'PROSPECTIVE_PROTOCOL.json',plan)
    write_new(a.output/'ORIGINAL_PREFLIGHT.json',before)
    started=time.monotonic();deadline=started+150
    parked_guard=TimedBlackHold(a.session,before,started+210)
    controller=Controller(parked_guard,a.output,a.profile_result)
    ui=OpticalUI(a.session,before);pixels=RootPixels();expected=None;last_id=last_ts=-1
    result=dict(completed=False,error=None,stages=[],release_approved=False)
    watcher=None;camera_requested=False;old_reports=set(a.session.glob('live_*/report.json'))
    source_sha=digest(__file__)

    def check(require_live=True,allow_expiry=False):
        if not allow_expiry and time.monotonic()>deadline:raise TimeoutError('bounded positive-projection deadline')
        if digest(__file__)!=source_sha:raise RuntimeError('running new protocol source changed')
        if (a.output/'WATCHDOG_INTERVENED.json').exists():raise RuntimeError('watchdog intervened; no resume')
        if watcher is not None and watcher.poll() is not None:raise RuntimeError('independent watchdog exited')
        h=ui.state();same_signals(before['roles'],read_roles())
        if Path(f'/proc/{ui.pid}/stat').read_text().rsplit(')',1)[1].split()[19]!=process_start:
            raise RuntimeError('original process lifetime changed')
        if h['actual_roi']!=before['heartbeat']['actual_roi'] or not h['geometry_frozen']:
            raise RuntimeError('ROI or geometry lock changed')
        for key in ('full_panel','x','y','focal_mm','period','square','dmd'):
            if h['manual_settings'][key]!=settings[key]:raise RuntimeError('fixed settings changed: '+key)
        if require_live and (not ui.fresh() or h['camera_active_readback']!=active):
            raise RuntimeError('stale or wrong active camera contract')
        if h.get('adaptation_active') or h.get('deployment_busy'):raise RuntimeError('competing GUI task')
        for role in ('dmd','slm'):
            rectangle=tuple(before['windows'][role]['rect'])
            if expected is None:
                if h['projection_armed'] or not pixels.inspect(rectangle)['all_RGB_zero']:
                    raise RuntimeError('expected actual disarmed zero windows')
            elif not pixels.compare(rectangle,expected[role])['matches']:
                raise RuntimeError('actual output differs from prescribed stage: '+role)
        for asset in assets.values():
            if digest(asset['path'])!=asset['sha256']:raise RuntimeError('cached asset changed')
        write_new(a.output/f'PROGRESS_{time.time_ns()}.json',dict(unix_ns=time.time_ns(),
            phase='live' if require_live else 'parked',display_revision=h['display_revision']))
        return h

    def verify_stage(illuminated,mode):
        nonlocal expected
        desired,canonical_dmd,canonical_slm=expected_pixels(settings,dmd,phase,illuminated,mode)
        expected=desired
        until=time.monotonic()+4
        while time.monotonic()<until:
            h=ui.state()
            if (h['projection_armed']==illuminated and h['display_revision']==h['last_verified_revision']
                    and all(pixels.compare(tuple(before['windows'][r]['rect']),desired[r])['matches'] for r in desired)):
                check();return h,canonical_dmd,canonical_slm
            time.sleep(.05)
        raise RuntimeError('stage actual framebuffer not committed')

    process_start=Path(f'/proc/{ui.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    try:
        check(require_live=False)
        parent_start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
        watcher=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watchdog',
            '--session',str(a.session.resolve()),'--output',str(a.output.resolve()),
            '--parent',str(os.getpid()),'--parent-start',parent_start,
            '--hard-deadline',str(time.time()+180)])
        find_tab(ui);controller.open();controller.start();check(require_live=False)
        camera_requested=True;ui.button('ccd');ui.wait(lambda h:ui.fresh(),20)
        check()
        for spec in stages:
            ui.arm(False);expected=None;check()
            ui.mode('slm',spec['mode']);ui.arm(spec['illuminated'])
            h,canonical_dmd,canonical_slm=verify_stage(spec['illuminated'],spec['mode'])
            verified_ns=time.time_ns();revision=h['display_revision']
            row=dict(**spec,revision=revision,verified_ns=verified_ns,snapshots=[])
            result['stages'].append(row)
            until=time.monotonic()+1.65
            while time.monotonic()<until:check();time.sleep(.15)
            for count in range(8):
                check();old=set(a.session.glob('snapshot_*'));ui.button('snapshot')
                end=time.monotonic()+3;folder=None
                while time.monotonic()<end:
                    found=set(a.session.glob('snapshot_*'))-old
                    if len(found)>1:raise RuntimeError('ambiguous new snapshot')
                    if found and (next(iter(found))/'metadata.json').exists():folder=next(iter(found));break
                    time.sleep(.03)
                if folder is None:raise TimeoutError('new public raw snapshot not saved')
                meta=read_complete(folder/'metadata.json');frame=meta['frame']
                raw=np.load(folder/'CCD_raw.npy',allow_pickle=False)
                sample=dict(folder=str(folder.resolve()),raw_sha256=digest(folder/'CCD_raw.npy'),
                    metadata_sha256=digest(folder/'metadata.json'),frame=frame,
                    max_ADU=int(raw.max()),mean_ADU=float(raw.mean()),
                    raw_integrals8=np.einsum('khw,hw->k',weights,raw.astype(np.float64)).tolist())
                row['snapshots'].append(sample)
                write_new(a.output/f'{spec["name"]}_{count:02d}.json',sample)
                if (raw.dtype!=np.uint8 or raw.shape!=(554,552) or raw.max()>=240
                        or meta['camera_active']!=active or meta['display_revision']!=revision
                        or frame['FrameID']<=last_id or frame['TimeStamp']<=last_ts
                        or frame['host_get_started_unix_ns']<verified_ns+1_500_000_000):
                    raise RuntimeError('raw headroom/contract/revision/freshness gate failed; no retry')
                from PIL import Image
                if (not np.array_equal(np.asarray(Image.open(folder/'dmd_canonical_requested.png')),canonical_dmd)
                        or not np.array_equal(np.asarray(Image.open(folder/'slm_canonical_requested.png')),canonical_slm)):
                    raise RuntimeError('snapshot canonical request differs from prescribed pixels')
                last_id,last_ts=frame['FrameID'],frame['TimeStamp'];time.sleep(.1)
            check();write_new(a.output/f'{spec["name"]}_COMPLETE.json',row)
            print(json.dumps(dict(stage=spec['name'],saved=len(row['snapshots']),
                max_ADU=max(s['max_ADU'] for s in row['snapshots']))),flush=True)
        result['all_stages_completed']=True
    except BaseException as error:result['error']=repr(error)
    finally:
        try:
            ui.button('allzero');expected=None
            if camera_requested:
                h=ui.state(allow_fault=True)
                if h['busy'] or h['live_camera']:ui.button('release')
            end=time.monotonic()+15
            while time.monotonic()<end:
                h=ui.state(allow_fault=True)
                if not h['busy'] and not h['live_camera']:break
                time.sleep(.1)
            result['after']=inspect(a.session,parked=True)
            check(require_live=False,allow_expiry=True)
        except BaseException as error:result['GUI_cleanup_error']=repr(error)
        result['controller_cleanup_errors']=controller.close()
        result['controller_stopped']=controller.stopped_verified
        try:
            reports=set(a.session.glob('live_*/report.json'))-old_reports
            if len(reports)!=1:raise RuntimeError('unique new original-GUI camera report required')
            path=next(iter(reports));camera=read_complete(path)
            result['camera_report']=str(path.resolve());result['camera_report_sha256']=digest(path)
            result['camera_exact_manual_restore']=(camera.get('camera_settings_restored') is True
                and camera.get('camera_before')==target and camera.get('camera_after')==target)
            result['camera_error']=camera.get('error')
            result['incomplete_frames']=camera.get('rejected_incomplete_frames')
            result['final_manual_verification']=verify_baseline(target,lambda:parked_guard.check(force=True))
            result['final_black']=inspect(a.session,parked=True)
        except BaseException as error:result['final_verification_error']=repr(error)
        write_new(a.output/'FINISHED.json',dict(unix_ns=time.time_ns(),all_new_projection_ended=True))
        if watcher is not None:
            try:result['watchdog_exit_code']=watcher.wait(timeout=8)
            except subprocess.TimeoutExpired:result['watchdog_timeout']=True
        result.update(completed=bool(result.get('all_stages_completed') and result['error'] is None
            and not result.get('GUI_cleanup_error') and not result['controller_cleanup_errors']
            and result.get('camera_exact_manual_restore') and not result.get('camera_error')
            and not result.get('incomplete_frames') and result.get('final_manual_verification',{}).get('passed')
            and 'final_black' in result and result.get('watchdog_exit_code')==0
            and not (a.output/'WATCHDOG_INTERVENED.json').exists()),wall_seconds=time.monotonic()-started)
        pixels.close();parked_guard.close()
        write_new(a.output/'STATIC_RESULT.json',result)
        print(json.dumps({k:result.get(k) for k in ('completed','error','GUI_cleanup_error',
            'camera_exact_manual_restore','controller_stopped','wall_seconds')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
