"""Fresh-authorized <=300s static pilot through the original GUI, not an SDK."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from workbench.registry import PROJECT,read,digest,load_registry
sys.path[:0]=[str(PROJECT/'tools'),str(PROJECT)]
from static_learned_ccd_v28 import StaticUI,inspect,expected_pixels,read_complete,write_new,validate_assets,Input
from select_verified_mode_v17 import owned_popups
from workbench.unified.pointer_control import install
from workbench.unified.capture_static import cached_assets


def checked_asset(record):
    path=Path(record['path'])
    if digest(path)!=record['sha256']:raise ValueError('frozen pilot asset changed')
    return np.load(path,allow_pickle=False)


class PilotUI(StaticUI):
    def sample(self,name,count,expected,canonical=None):
        self.guard(black=expected is None,expected=expected)
        start=time.time_ns();revision=self.state()['display_revision'];rows=[]
        end=time.monotonic()+1.8
        while time.monotonic()<end:self.guard(black=expected is None,expected=expected);time.sleep(.08)
        for _ in range(count):
            self.guard(black=expected is None,expected=expected)
            old=set(self.session.glob('snapshot_*'));self.button('snapshot');end=time.monotonic()+4
            while time.monotonic()<end:
                self.guard(black=expected is None,expected=expected)
                new=set(self.session.glob('snapshot_*'))-old
                if len(new)>1:raise RuntimeError('ambiguous raw snapshot')
                if new and (next(iter(new))/'metadata.json').exists():folder=next(iter(new));break
                time.sleep(.04)
            else:raise TimeoutError('fresh snapshot missing')
            meta=read_complete(folder/'metadata.json');raw=np.load(folder/'CCD_raw.npy',allow_pickle=False);frame=meta['frame']
            if (meta['camera_active']!=self.active or meta['display_revision']!=revision or
                    raw.dtype!=np.uint8 or raw.shape!=(554,552) or raw.max()>=240 or
                    frame['host_get_started_unix_ns']<start+1_500_000_000 or
                    frame['TimeStamp']<=self.last_tick or frame['FrameID']<=self.last_frame):
                raise RuntimeError('raw layout/saturation/freshness/revision/contract gate failed')
            self.last_tick=frame['TimeStamp'];self.last_frame=frame['FrameID']
            for role in ('dmd','slm'):
                image=np.asarray(Image.open(folder/(role+'_canonical_requested.png')))
                if canonical is None:
                    if image.any():raise RuntimeError('dark request not actually0')
                elif not np.array_equal(image,canonical[role]):raise RuntimeError('saved request differs from prescribed mask/input')
            rows.append(dict(folder=str(folder),raw_sha256=digest(folder/'CCD_raw.npy'),
                metadata_sha256=digest(folder/'metadata.json'),frame=frame,
                max_ADU=int(raw.max()),mean_ADU=float(raw.mean())))
        row=dict(name=name,revision=revision,verified_unix_ns=start,snapshots=rows,
                 actual_framebuffer_checked=True)
        write_new(self.root/(name+'.json'),row)
        print(dict(stage=name,frames=len(rows),max_ADU=max(r['max_ADU'] for r in rows)),flush=True)
        return row

    def cell(self,name,dmd,phase,count):
        self.guard(black=True);self.mode('dmd','file');self.mode('slm','file')
        settings=dict(self.fixed['manual_settings'],dmd='file',slm='file')
        expected,cd,cs=expected_pixels(settings,dmd,phase,True,'file')
        self.arm(True);end=time.monotonic()+5
        while time.monotonic()<end:
            h=self.guard(expected=expected,transition=True)
            if h['last_verified_revision']==h['display_revision'] and all(
                    self.pixels.compare(tuple(self.before['windows'][r]['rect']),expected[r])['matches'] for r in expected):break
            time.sleep(.08)
        else:raise TimeoutError('actual framebuffer verification absent')
        row=self.sample(name,count,expected,dict(dmd=cd,slm=cs))
        self.arm(False);self.wait(lambda h:not h['projection_armed']);self.guard(black=True)
        return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--low-power-confirmed',action='store_true')
    a=p.parse_args()
    if not a.low_power_confirmed:raise RuntimeError('fresh current user authorization required')
    planpath=a.assets.resolve()/'PLAN.json';plan=read(planpath)
    if plan['maximum_seconds']!=300 or len(plan['cells'])!=12:raise ValueError('bounded pilot plan required')
    for path,sha in plan['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('prepared source changed')
    if digest(plan['hardware_contract'])!=plan['hardware_contract_sha256']:raise ValueError('contract changed')
    inputs={r['id']:r for r in plan['inputs']};masks={r['id']:r for r in plan['masks']}
    for inp in inputs.values():
        for mask in masks.values():validate_assets(checked_asset(inp),checked_asset(mask))
    cfg,_=load_registry();session=PROJECT/cfg['runtime']['session'];before=inspect(session,parked=False);h=before['heartbeat']
    if (h['hardware_fault'] or not h['live_camera'] or h['projection_armed'] or
            h.get('loop_active') or h.get('adaptation_active') or h.get('deployment_busy') or
            owned_popups(h['pid'],before['windows']['control']['wid'])):
        raise RuntimeError('healthy original disarmed idle stream required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    lock=open(f'/run/user/{os.getuid()}/ballplate_current_public_15328133.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_new(a.output/'PREFLIGHT.json',before)
    write_new(a.output/'PROSPECTIVE_PROTOCOL.json',dict(plan_path=str(planpath),plan_sha256=digest(planpath),
        authority='current user confirmed unchanged ND/protection/positions and bounded static acquisition',
        no_SDK_no_window_rebuild=True,original_camera_stream_retained=True,maximum_seconds=300,
        source_sha256={str(Path(__file__).resolve()):digest(__file__)}))
    write_new(a.output/'EXTERNAL_POINTER_COMPATIBILITY.json',install())
    ui=PilotUI(session,before,a.output,time.monotonic()+300);ui.last_tick=-1;ui.last_frame=-1
    ui.guard(black=True);cached=cached_assets(session,h)
    write_new(a.output/'ORIGINAL_CACHED_ASSETS.json',cached)
    identity=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    watcher=subprocess.Popen([sys.executable,str(PROJECT/'tools/static_learned_ccd_v28.py'),
        '--mode','watchdog','--session',str(session),'--output',str(a.output),'--parent',str(os.getpid()),
        '--parent-start',identity,'--hard-deadline',str(time.time()+310)])
    result=dict(completed=False,error=None,stages=[],no_camera_or_controller_parameter_writes=True,
                actual_actuators=False,closed_loop=False,release_approved=False)
    started=time.monotonic()
    try:
        focus=Input(int(ui.main['wid'],16));focus.close()
        result['stages'].append(ui.sample('DARK_BEFORE',16,None))
        current_input=current_mask=None
        for row in plan['cells']:
            if watcher.poll() is not None:raise RuntimeError('independent watchdog exited')
            inp=inputs[row['input_id']];mask=masks[row['mask_id']]
            if inp['id']!=current_input:
                ui.load_live('dmd',Path(inp['path']));current_input=inp['id']
            if mask['id']!=current_mask:
                ui.load_live('slm',Path(mask['path']));current_mask=mask['id']
            result['stages'].append(ui.cell(row['id'],checked_asset(inp),checked_asset(mask),8))
        result['stages'].append(ui.sample('DARK_AFTER',16,None))
        # Restore both exact file caches while disarmed, with independent
        # watchdog/lease still active. Not required for output safety.
        for role in ('dmd','slm'):ui.load_live(role,Path(cached[role]['path']))
        ui.mode('slm',h['manual_settings']['slm']);ui.mode('dmd',h['manual_settings']['dmd'])
        result['original_file_caches_restored']=True;result['completed']=True
    except BaseException as error:result['error']=repr(error)
    finally:
        try:
            ui.button('allzero');ui.wait(lambda s:not s['projection_armed']);ui.guard(black=True)
            result.update(actual_outputs_zero=True,camera_left_running=ui.state()['live_camera'])
        except BaseException as error:result.update(cleanup_error=repr(error),completed=False)
        result['wall_seconds']=time.monotonic()-started
        write_new(a.output/'RESULT.json',result);write_new(a.output/'FINISHED.json',dict(unix_ns=time.time_ns()))
        try:watcher.wait(timeout=6)
        except subprocess.TimeoutExpired:pass
        ui.pixels.close();lock.close()
        print({k:v for k,v in result.items() if k!='stages'},flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
