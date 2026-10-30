"""One authorized public-button step; no SDK connection or GUI injection."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import core as c
import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--restored-low-power-confirmed',action='store_true')
    a=p.parse_args()
    if not a.restored_low_power_confirmed:raise ValueError('restored illumination confirmation required')
    out=c.ROOT/'REAL_SINGLE_STEP_01';out.mkdir()
    c.write(out/'AUTHORIZATION.json',dict(user='已恢复，可以单步验证',unix_ns=time.time_ns(),
        scope='center reset, one measured CCD step, pause and zero; no continuous control',source=c.ref(__file__)))
    proof=c.read(c.ROOT/'POST_STARTUP_01/RESULT.json');cfg=c.read(c.ROOT/'CONFIG.json');pid=proof['pid']
    for path,sha in cfg['source_sha256'].items():
        if c.digest(path)!=sha:raise ValueError('running source changed: '+path)
    base=run.base_module()
    base.load('e107_single_display',base.DISPLAY/'run.py').install_display(c.read(c.checked(cfg['display_profile'])))
    from optical_lab import legacy_gui
    clients=legacy_gui._load_clients(base.LEGACY);legacy_gui._install_accessibility_compatibility();clients.install()
    base.load('e107_single_compat',c.EXP/'E067_three_slm_20261026_1/client_compat.py').install()
    n=base.load('e107_single_native',c.EXP/'E029_native_crash_recovery_20261024_1/native_buttons.py')
    n.ORDINARY=[1080,1080,2560,1440]
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    from black_pixels import RootPixels
    def state(allow_fault=False):
        h=c.read(c.ROOT/'GUI/heartbeat.json')
        if h['pid']!=pid or not 0<=time.time_ns()-h['unix_ns']<3e9:raise ValueError('owner heartbeat not fresh')
        if not allow_fault and (h['hardware_fault'] or not h['live_camera']):raise ValueError('camera/display fault')
        return h
    ui=SimpleNamespace(main=proof['windows']['control'],pid=pid,state=state)
    def nodes():
        app,A=accessible(pid);return list(safe_nodes(app)),A
    def click(name):
        tree,A=nodes();matches=[v for v in tree if v.get_role_name()=='push button' and v.get_name()==name
            and v.get_state_set().contains(A.StateType.SHOWING)]
        if len(matches)!=1:raise ValueError('ambiguous public button: '+name)
        return n.click_visible_node(ui,matches[0],A,execute=True)
    def labels():
        tree,A=nodes()
        return [v.get_name() for v in tree if v.get_role_name()=='label' and v.get_state_set().contains(A.StateType.SHOWING)]
    before=state()
    if before['projection_armed'] or before['current_page']!=2 or before['camera_active_readback']!=cfg['expected_active']:
        raise ValueError('not the prepared idle page/configuration')
    runs=c.ROOT/'PAIRED_RUNS'
    previous=set(runs.glob('RUN_*')) if runs.exists() else set()
    if previous:raise ValueError('first single-step validation requires no earlier real runs')
    c.write(out/'BEFORE.json',before)
    clicks=[];error=None;step_path=None;run_path=None
    try:
        clicks.append(click('中心复位'))
        time.sleep(.2)
        clicks.append(click('单步'))
        deadline=time.monotonic()+90;last_inspect=0
        while time.monotonic()<deadline:
            h=state()
            candidates=set(runs.glob('RUN_*'))-previous if runs.exists() else set()
            if len(candidates)>1:raise ValueError('multiple unexpected control runs')
            if candidates:
                run_path=candidates.pop();steps=list(run_path.glob('E*/STEP.json'))
                if len(steps)>1:raise ValueError('single step advanced more than once')
                if steps:
                    step_path=steps[0]
                    try:c.read(step_path)
                    except ValueError:time.sleep(.1);continue
                    if not h['projection_armed']:break
            if time.monotonic()-last_inspect>3:
                messages=labels();last_inspect=time.monotonic()
                failure=[x for x in messages if x.startswith('未继续：')]
                if failure:raise ValueError(failure[0])
            time.sleep(.25)
        else:raise TimeoutError('one-step completion exceeded 90 seconds')
    except Exception as e:error=str(e)
    finally:
        try:clicks.append(click('暂停／全0'))
        except Exception as e:error=(error or '')+'; pause button: '+str(e)
    time.sleep(.5)
    after=state(allow_fault=True);pixels=RootPixels()
    try:zeros={k:pixels.inspect(tuple(proof['windows'][k]['rect'])) for k in ('dmd','slm')}
    finally:pixels.close()
    audit={}
    if step_path is not None and error is None:
        try:
            record=c.read(step_path);raw_path=c.checked(record['real_capture']);meta=c.read(raw_path.with_suffix('.json'))
            package=c.read(c.checked(cfg['paired_control']))
            with np.load(raw_path,allow_pickle=False) as z:
                raw=z['raw'];image=z['image']
                if len(raw)!=10 or meta['filter']['mode']!='mean':raise ValueError('expected current mean10 filter')
                mean_error=float(np.max(np.abs(raw.mean(axis=0,dtype=np.float64)-image)))
                if mean_error>1e-4:raise ValueError('effective mean differs')
                sums=c.integrate(image,package['layout'])
                if not np.allclose(sums,record['real_sums'],rtol=0,atol=1e-4):raise ValueError('RAW integral differs')
                head=c.read(c.checked(package['heads']['real']));control=c.action(sums,head)
                if not np.allclose(control,record['raw_actions'][1],rtol=0,atol=1e-10):raise ValueError('real linear head differs')
                audit=dict(shape=list(raw.shape),raw_mean_ADU=float(raw.mean()),raw_max=int(raw.max()),
                    saturated_pixels=int(np.count_nonzero(raw==255)),effective_mean_error=mean_error,
                    real_sums_exact=True,real_head_exact=True,raw_actions=record['raw_actions'])
            if not record['hardware_CCD_used'] or record['step']!=1:raise ValueError('no actual single step evidence')
            for pre,post in zip(record['states_before'],record['states_after']):
                if pre['time']!=0 or abs(post['time']-.02)>1e-10:raise ValueError('unexpected MuJoCo clock')
            audit.update(hardware_CCD_used=True,states_before=record['states_before'],states_after=record['states_after'])
        except Exception as e:error='post-step audit: '+str(e)
    if after['projection_armed'] or not all(v['all_RGB_zero'] for v in zeros.values()):error=(error or '')+'; outputs not confirmed zero'
    runtime=n._runtime();focus=runtime.Input(n._wid(ui.main['wid']))
    try:
        from PIL import ImageGrab
        x,y,w,h=ui.main['rect'];time.sleep(.2)
        ImageGrab.grab(bbox=(x,y,x+w,y+h),xdisplay=':1').save(out/'live_result.png')
    finally:focus.close()
    c.write(out/'RESULT.json',dict(passed=error is None and step_path is not None,error=error,
        clicks=clicks,run=str(run_path) if run_path else None,step=c.ref(step_path) if step_path else None,
        after=after,zeros=zeros,labels=labels(),audit=audit,convergence_test=False,actual_actuator=False))
    print(dict(passed=error is None and step_path is not None,error=error,run=str(run_path),audit=audit),flush=True)


if __name__=='__main__':main()
