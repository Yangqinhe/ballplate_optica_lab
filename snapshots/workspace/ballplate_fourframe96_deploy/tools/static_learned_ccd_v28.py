"""Bounded static NEW-mask CCD viewing via existing visible v23 controls.

No SDK, controller writes, injected code, output-window rebuild, camera
restart, ROI changes, LUT application or control actions. Original live stream
is retained. Separate watchdog disarms on expiry or loss of the driver.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tools'),str(PROJECT/'live_manual_v17')]
from live_adapt_v22.core import digest,write_new
from inspect_recovery_ui_v20 import inspect
from optical_ui_v19 import OpticalUI
from select_verified_mode_v17 import window,inside
from load_current_adaptation_bundle_v134 import one_control,press
from night_opt_20261020.manual_io_v17 import accessible,real_dialog,focused_fields
from calibration_desktop import Input
from black_pixels import RootPixels
from display_linux import read_roles,same_signals
from static_singlebit_v109 import expected_pixels,read_complete


def validate_assets(dmd,phase):
    if dmd.shape!=(600,600) or dmd.dtype!=np.uint8 or not np.isin(dmd,(0,255)).all():
        raise ValueError('exact binary600 DMD required')
    count=int(np.count_nonzero(dmd))
    if not 1<=count<=1500:raise ValueError('bounded sparse four-frame input only')
    if phase.shape!=(600,600) or phase.dtype!=np.uint8:
        raise ValueError('canonical raw-gray600 SLM required')
    if not np.array_equal(phase,phase[::4,::4].repeat(4,0).repeat(4,1)):
        raise ValueError('exact4x4 macro-pixels required')
    logical=dmd[12:588,12:588][::3,::3]
    if not np.array_equal(dmd,np.pad(logical.repeat(3,0).repeat(3,1),12)):
        raise ValueError('not the declared four RAW96 packing')
    return count


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    source=PROJECT/'outputs/pipeline_demo_20261022_v27/VIEWER/simulation_1792601219058062435/display_0252.npz'
    research=PROJECT/'outputs/optical_phase_v25_20261021/RESEARCH_ONLY_PAIRED8_BUNDLE_V206'
    manifest=read_complete(research/'RESEARCH_MANIFEST.json')
    relative='learned/SLM_CANONICAL_RAW_GRAY_NO_LUT_NOT_PANEL.npy'
    phase_path=research/relative
    if digest(phase_path)!=manifest['files'][relative]:raise ValueError('learned phase changed')
    with np.load(source,allow_pickle=False) as sample:
        dmd=sample['dmd'].copy();phase=sample['phase'].copy()
    if not np.array_equal(phase,np.load(phase_path,allow_pickle=False)):
        raise ValueError('DMD fixture did not use this learned phase')
    count=validate_assets(dmd,phase)
    np.save(output/'DMD_FOUR_RAW96.npy',dmd,allow_pickle=False)
    np.save(output/'SLM_LEARNED_CANONICAL_RAW_GRAY.npy',phase,allow_pickle=False)
    info=dict(role='NEW_LEARNED_MASK_BOUNDED_STATIC_ASSETS',source_fixture=str(source),
        source_fixture_sha256=digest(source),source_phase=str(phase_path),source_phase_sha256=digest(phase_path),
        dmd_on_pixels=count,canonical=True,preflipped=False,LUT_applied=False,
        hardware_coordinate_origin_NOT_recalibrated=True,closed_loop_enabled=False,
        files={n:digest(output/n) for n in ('DMD_FOUR_RAW96.npy','SLM_LEARNED_CANONICAL_RAW_GRAY.npy')})
    write_new(output/'ASSETS.json',info)
    print(json.dumps(info,ensure_ascii=False))


class StaticUI(OpticalUI):
    def __init__(self,session,before,root,deadline,renew=True):
        self.root=root;self.deadline=deadline;self.renew=renew;self.pixels=RootPixels()
        self.active=before['heartbeat']['camera_active_readback'];self.fixed=before['heartbeat']
        super().__init__(session,before)

    def state(self,allow_fault=False):
        until=time.monotonic()+2.5
        while True:
            h=read_complete(self.session/'heartbeat.json')
            if h.get('deployment_ui_revision')==23 and 'camera_active_readback' in h:break
            if time.monotonic()>until:raise RuntimeError('fully published v23 heartbeat absent')
            time.sleep(.005)
        if h['pid']!=self.pid or not 0<=time.time_ns()-h['unix_ns']<=6e9:
            raise RuntimeError('owner/heartbeat changed')
        if h['hardware_fault'] and not allow_fault:raise RuntimeError(h['hardware_fault'])
        if time.monotonic()-self.last_check>1:
            for group in ('sources','source_configs'):
                for file,sha in self.before['runtime'][group].items():
                    if digest(file)!=sha:raise RuntimeError('loaded source changed: '+file)
            same_signals(self.before['roles'],read_roles())
            for role,row in self.before['windows'].items():
                current=window(row['wid'])
                if current['pid']!=self.pid or not current['viewable']:raise RuntimeError('owned window disappeared')
                if role=='control':
                    if not inside(current['rect'],[1080,600,2560,1440]):raise RuntimeError('main left ordinary monitor')
                    self.main=current
                elif current['rect']!=row['rect']:raise RuntimeError('device window moved')
            self.last_check=time.monotonic()
        return h

    def guard(self,black=False,expected=None,transition=False):
        if time.monotonic()>self.deadline:raise TimeoutError('bounded static test deadline')
        h=self.state()
        if not self.fresh() or h.get('deployment_busy') or h.get('adaptation_active') or h.get('loop_active'):
            raise RuntimeError('single fresh idle camera owner required')
        if (h['camera_active_readback']!=self.active or h['actual_roi']!=self.fixed['actual_roi']
                or h['optical_reference']!=self.fixed['optical_reference'] or not h['geometry_frozen']):
            raise RuntimeError('camera/ROI/reference geometry changed')
        for key in ('x','y','full_panel','square','focal_mm','period'):
            if h['manual_settings'][key]!=self.fixed['manual_settings'][key]:raise RuntimeError('manual geometry changed')
        rectangles={r:tuple(self.before['windows'][r]['rect']) for r in ('dmd','slm')}
        if black:
            if h['projection_armed'] or not all(self.pixels.inspect(r)['all_RGB_zero'] for r in rectangles.values()):
                raise RuntimeError('actual disarmed zeros required')
        elif expected is not None:
            correct=all(self.pixels.compare(rectangles[r],expected[r])['matches'] for r in rectangles)
            if not correct and transition:correct=all(self.pixels.inspect(r)['all_RGB_zero'] for r in rectangles.values())
            if not correct:raise RuntimeError('actual sent pixels differ from declared static pattern')
        if self.renew:
            p=self.root/'LEASE.tmp';p.write_text(json.dumps(dict(unix_ns=time.time_ns())))
            p.replace(self.root/'LEASE.json')
        return h

    def load_live(self,role,path):
        self.guard(black=True);path=path.resolve();sha=digest(path)
        count=len((self.session/'events.jsonl').read_text().splitlines())
        title='载入DMD图' if role=='dmd' else '载入规范SLM灰度图'
        client=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode','open',
            '--session',str(self.session),'--output',str(self.root),'--role',role],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        _,A=accessible(self.pid)
        try:
            until=time.monotonic()+8
            while time.monotonic()<until:
                self.guard(black=True);dialogs=real_dialog(self.pid,title,A)
                if dialogs:break
                if client.poll() not in (None,0):raise RuntimeError(client.communicate()[1])
                time.sleep(.08)
            else:raise TimeoutError('native file chooser absent')
            tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
            rows=[window(re.search(r'0x[0-9a-fA-F]+',s)[0]) for s in tree.splitlines() if '"'+title+'"' in s]
            rows=[r for r in rows if r['pid']==self.pid and r['viewable']]
            if len(dialogs)!=1 or len(rows)!=1:raise RuntimeError('unique PID-owned native chooser required')
            row=rows[0]
            # Transparent GTK shadows may extend; require opaque content to remain ordinary.
            props=subprocess.check_output(['xprop','-id',row['wid'],'_GTK_FRAME_EXTENTS'],text=True)
            match=re.search(r'= (\d+), (\d+), (\d+), (\d+)',props)
            l,r,t,b=map(int,match.groups()) if match else (0,0,0,0)
            x,y,w,hh=row['rect']
            if not inside([x+l,y+t,w-l-r,hh-t-b],[1080,600,2560,1440]):raise RuntimeError('native chooser outside ordinary screen')
            keyboard=Input(int(row['wid'],16))
            try:
                keyboard.chord(ord('l'));time.sleep(.2)
                dialogs=real_dialog(self.pid,title,A)
                if len(dialogs)!=1:raise RuntimeError('chooser vanished')
                fields=focused_fields(dialogs[0],A)
                if len(fields)!=1:raise RuntimeError('one focused native filename field required')
                field=fields[0]
                if not field.get_editable_text_iface().set_text_contents(str(path)):raise RuntimeError('filename edit failed')
                time.sleep(.15)
                active=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
                if int(re.search(r'0x[0-9a-fA-F]+',active)[0],16)!=int(row['wid'],16):raise RuntimeError('chooser focus changed')
                if field.get_text_iface().get_text(0,-1)!=str(path) or digest(path)!=sha:raise RuntimeError('filename/hash differs')
                self.guard(black=True);keyboard.key(0xff0d,True);keyboard.key(0xff0d,False)
            finally:keyboard.close()
            until=time.monotonic()+6
            while time.monotonic()<until:
                self.guard(black=True)
                events=[]
                for line in (self.session/'events.jsonl').read_text().splitlines()[count:]:
                    try:events.append(json.loads(line))
                    except json.JSONDecodeError:pass
                matches=[e for e in events if e.get('state')=='manual_file_loaded' and e.get('SLM')==(role=='slm')]
                if matches:
                    e=matches[-1]
                    if Path(e['path']).resolve()!=path or e['sha256']!=sha:raise RuntimeError('wrong asset loaded')
                    if not real_dialog(self.pid,title,A):
                        client.communicate(timeout=2)
                        if client.returncode:raise RuntimeError('native action client failed')
                        return e
                time.sleep(.07)
            raise TimeoutError('exact new file-load acknowledgement absent')
        except BaseException:
            # File loading is disarmed. Dismiss only this PID/title-owned
            # chooser so the parent's all-zero button is not left modal.
            try:
                dialogs=real_dialog(self.pid,title,A)
                if dialogs:
                    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
                    rows=[window(re.search(r'0x[0-9a-fA-F]+',s)[0]) for s in tree.splitlines() if '"'+title+'"' in s]
                    rows=[r for r in rows if r['pid']==self.pid and r['viewable']]
                    if len(dialogs)==1 and len(rows)==1:
                        k=Input(int(rows[0]['wid'],16))
                        try:k.key(0xff1b,True);k.key(0xff1b,False)
                        finally:k.close()
            finally:raise
        finally:
            if client.poll() is None:client.terminate();client.communicate(timeout=2)

    def stage(self,name,dmd,phase,mode,hold):
        self.arm(False);self.wait(lambda h:not h['projection_armed']);self.guard(black=True)
        self.mode('slm',mode)
        settings=dict(self.fixed['manual_settings'],dmd='file',slm=mode)
        expected,cd,cs=expected_pixels(settings,dmd,phase,True,mode)
        self.arm(True)
        end=time.monotonic()+5
        while time.monotonic()<end:
            h=self.guard(expected=expected,transition=True)
            if h['last_verified_revision']==h['display_revision'] and all(
                self.pixels.compare(tuple(self.before['windows'][r]['rect']),expected[r])['matches'] for r in expected):break
            time.sleep(.08)
        else:raise TimeoutError('sent pixels not acknowledged')
        row=dict(name=name,revision=h['display_revision'],verified_unix_ns=time.time_ns(),snapshots=[])
        end=time.monotonic()+1.8
        while time.monotonic()<end:self.guard(expected=expected);time.sleep(.1)
        previous=None
        for _ in range(3):
            self.guard(expected=expected);old=set(self.session.glob('snapshot_*'));self.button('snapshot')
            end=time.monotonic()+4
            while time.monotonic()<end:
                self.guard(expected=expected);new=set(self.session.glob('snapshot_*'))-old
                if len(new)>1:raise RuntimeError('ambiguous snapshot')
                if new and (next(iter(new))/'metadata.json').exists():folder=next(iter(new));break
                time.sleep(.04)
            else:raise TimeoutError('fresh raw snapshot absent')
            meta=read_complete(folder/'metadata.json');raw=np.load(folder/'CCD_raw.npy',allow_pickle=False);frame=meta['frame']
            from PIL import Image
            if (raw.dtype!=np.uint8 or raw.shape!=(554,552) or int(raw.max())>=240 or meta['camera_active']!=self.active
                    or meta['display_revision']!=row['revision']
                    or frame['host_get_started_unix_ns']<row['verified_unix_ns']+1_500_000_000):
                raise RuntimeError('freshness, saturation, configuration or revision failed')
            if previous and frame['TimeStamp']<=previous['TimeStamp']:raise RuntimeError('stale raw frame')
            for role,array in (('dmd',cd),('slm',cs)):
                if not np.array_equal(np.asarray(Image.open(folder/(role+'_canonical_requested.png'))),array):
                    raise RuntimeError('saved canonical request differs')
            previous=frame
            row['snapshots'].append(dict(folder=str(folder),raw_sha256=digest(folder/'CCD_raw.npy'),
                frame=frame,max_ADU=int(raw.max()),mean_ADU=float(raw.mean())))
            time.sleep(.1)
        write_new(self.root/(name+'.json'),row)
        print(json.dumps(dict(stage=name,revision=row['revision'],actual_raw_max_ADU=max(r['max_ADU'] for r in row['snapshots']),
            hold_seconds=hold,source='ACTUAL_CCD_NOT_SIMULATION')),flush=True)
        if name=='LEARNED':
            from PIL import ImageGrab
            r=self.main['rect'];ImageGrab.grab(bbox=(r[0],r[1],r[0]+r[2],r[1]+r[3]),xdisplay=':1').save(self.root/'LEARNED_LIVE_SCREEN.png')
        end=time.monotonic()+hold
        while time.monotonic()<end:self.guard(expected=expected);time.sleep(.15)
        self.arm(False);self.wait(lambda h:not h['projection_armed']);self.guard(black=True)
        return row


def watchdog(a):
    before=read_complete(a.output/'PREFLIGHT.json')
    ui=StaticUI(a.session,before,a.output,time.monotonic()+240,renew=False)
    result=dict(intervened=False,error=None)
    try:
        while True:
            h=ui.state(allow_fault=True)
            if (a.output/'FINISHED.json').exists():
                if h['projection_armed']:ui.button('allzero');result['intervened']=True
                break
            try:identity=Path(f'/proc/{a.parent}/stat').read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:identity=None
            lease=read_complete(a.output/'LEASE.json')
            if identity!=a.parent_start or not 0<=time.time_ns()-lease['unix_ns']<=12e9 or time.time()>a.hard_deadline:
                ui.button('allzero');result['intervened']=True;break
            time.sleep(.25)
        until=time.monotonic()+4
        while time.monotonic()<until:
            h=ui.state(allow_fault=True)
            if not h['projection_armed'] and all(ui.pixels.inspect(tuple(before['windows'][r]['rect']))['all_RGB_zero'] for r in ('dmd','slm')):
                result['actual_all_zero']=True;break
            time.sleep(.1)
        else:raise RuntimeError('final actual zeros not verified')
    except BaseException as e:result['error']=repr(e)
    finally:ui.pixels.close();write_new(a.output/'WATCHDOG_RESULT.json',result)


def execute(a):
    if not a.low_power_confirmed:raise RuntimeError('fresh user confirmation required')
    assets=read_complete(a.output/'ASSETS.json')
    for file,sha in assets['files'].items():
        if digest(a.output/file)!=sha:raise RuntimeError('prepared asset changed')
    dmd=np.load(a.output/'DMD_FOUR_RAW96.npy',allow_pickle=False)
    phase=np.load(a.output/'SLM_LEARNED_CANONICAL_RAW_GRAY.npy',allow_pickle=False);validate_assets(dmd,phase)
    before=inspect(a.session,parked=False)
    if before['heartbeat']['hardware_fault'] or not before['heartbeat']['live_camera']:raise RuntimeError('live original camera required')
    write_new(a.output/'PREFLIGHT.json',before)
    ui=StaticUI(a.session,before,a.output,time.monotonic()+180)
    ui.guard(black=True)
    plan=dict(authority='current user 确认 after requesting current actual CCD of four inputs and learned mask',
        hold_learned_seconds=a.hold_seconds,maximum_seconds=180,source_sha256=digest(__file__),
        loaded_GUI_sources_unchanged=True,camera_restart=False,camera_parameter_writes=False,
        controller_writes=False,no_new_device_windows=True,same_current_ROI_and_registration=True,
        current_radiometry_gray_phase_response_NOT_qualified=True,comparison_is_reference_only=True,
        no_claim_of_new_actor_or_actual_closed_loop_convergence=True,release_approved=False)
    write_new(a.output/'PROSPECTIVE_PLAN.json',plan)
    parent_start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    watch=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode','watchdog','--session',str(a.session),
        '--output',str(a.output),'--parent',str(os.getpid()),'--parent-start',parent_start,'--hard-deadline',str(time.time()+200)])
    result=dict(completed=False,error=None,stages=[],real_actuators_enabled=False,closed_loop=False)
    try:
        ui.focus=Input(int(ui.main['wid'],16));ui.focus.close()
        for role,name in (('dmd','DMD_FOUR_RAW96.npy'),('slm','SLM_LEARNED_CANONICAL_RAW_GRAY.npy')):
            event=ui.load_live(role,a.output/name);write_new(a.output/('LOADED_'+role+'.json'),event)
        result['stages'].append(ui.stage('FLAT_BEFORE',dmd,phase,'zero',1.))
        result['stages'].append(ui.stage('LEARNED',dmd,phase,'file',a.hold_seconds))
        result['stages'].append(ui.stage('FLAT_AFTER',dmd,phase,'zero',1.))
        result['completed']=True
    except BaseException as e:result['error']=repr(e)
    finally:
        try:
            ui.button('allzero');ui.wait(lambda h:not h['projection_armed']);ui.guard(black=True)
            result['actual_outputs_zero']=True;result['camera_left_running']=ui.state()['live_camera']
        except BaseException as e:result['cleanup_error']=repr(e);result['completed']=False
        write_new(a.output/'RESULT.json',result);write_new(a.output/'FINISHED.json',dict(unix_ns=time.time_ns()))
        try:watch.wait(timeout=6)
        except subprocess.TimeoutExpired:result['watchdog_completion_pending']=True
        ui.pixels.close()
        print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['completed']:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run','open','watchdog'),required=True)
    p.add_argument('--session',type=Path,default=PROJECT/'outputs/live_calibration/INTERACTIVE_LOOP_V23_20261021_1930/GUI')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--role',choices=('dmd','slm'))
    p.add_argument('--low-power-confirmed',action='store_true');p.add_argument('--hold-seconds',type=float,default=60.)
    p.add_argument('--parent',type=int);p.add_argument('--parent-start');p.add_argument('--hard-deadline',type=float)
    a=p.parse_args();a.output=a.output.resolve();a.session=a.session.resolve()
    if not 1<=a.hold_seconds<=60:p.error('bounded viewing only')
    if a.mode=='prepare':return prepare(a.output)
    if a.mode=='watchdog':return watchdog(a)
    if a.mode=='run':return execute(a)
    # Only asynchronous invocation of this exact visible load button. No
    # direct setter or file object is injected into the running GUI.
    h=read_complete(a.session/'heartbeat.json');before=read_complete(a.output/'PREFLIGHT.json')
    if h['pid']!=before['pid'] or h['projection_armed'] or h['hardware_fault'] or not h['live_camera']:
        raise RuntimeError('disarmed current live owner required')
    main=window(before['windows']['control']['wid'])
    button,_=one_control(h['pid'],'push button','载入图像' if a.role=='dmd' else '载入mask',main['rect'])
    press(button)


if __name__=='__main__':main()
