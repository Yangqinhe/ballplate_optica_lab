"""Explicit original native preview controls, no SDK or action on import."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
SESSION=ROOT/'GUI'
NATIVE=LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py'
PID=1390768

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def setup():
    from optical_lab import legacy_gui as old
    from optical_lab.storage import sha256
    if sha256(NATIVE)!='a5c3c7acf9618422f364a602653e02536bb12a0c34497f18455eb18e33d4330d':raise RuntimeError('frozen native helper changed')
    c=old._load_clients(LAB.parent/'ballplate_fourframe96_deploy');old._install_accessibility_compatibility();c.install()
    before=c.inspect(SESSION,parked=False)
    if before['pid']!=PID:raise RuntimeError('exact new E040 PID required')
    def state(allow_fault=False):
        h=c.inspect(SESSION,parked=False)['heartbeat']
        if h['pid']!=PID:raise RuntimeError('owner changed')
        if h['hardware_fault']and not allow_fault:raise RuntimeError(h['hardware_fault'])
        return h
    ui=SimpleNamespace(pid=PID,main=before['windows']['control'],state=state,
                       fresh=lambda:bool(state()['live_camera']and state()['latest_frame']['age_seconds']<1))
    spec=importlib.util.spec_from_file_location('E040_native',NATIVE);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
    return ui,c,n

def wait(predicate,seconds=20):
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        h=read(SESSION/'heartbeat.json')
        if h['pid']!=PID:raise RuntimeError('owner changed')
        if predicate(h):return h
        time.sleep(.1)
    raise TimeoutError('bounded original control transition did not complete')

def restore_gain(ui,c,n,out):
    import gi
    gi.require_version('Atspi','2.0')
    from gi.repository import Atspi as A
    import optical_ui_v19 as optical
    from optical_lab.storage import sha256
    for path,digest in read(SESSION/'spatial_preview_extension.json')['new_source_sha256'].items():
        if sha256(path)!=digest:raise RuntimeError('new GUI extension source changed')
    baseline=ui.state()['camera_active_readback']
    def find(role,name):
        h=ui.state()
        if (not ui.fresh()or h['projection_armed']or h['hardware_fault']
                or any(h.get(k)for k in('loop_active','deployment_busy','adaptation_active'))):raise RuntimeError('fresh idle black camera required')
        desktop=A.get_desktop(0);apps=[desktop.get_child_at_index(i)for i in range(desktop.get_child_count())]
        apps=[a for a in apps if a.get_process_id()==PID and a.get_name()=='ballplate-realtime-manual']
        frames=[node for a in apps for node in optical.nodes(a)if node.get_role_name()=='frame'and node.get_name().startswith('实时光路对齐 v19 ')]
        hits=[node for frame in frames for node in optical.nodes(frame)if node.get_role_name()==role and node.get_name()==name]
        if len(frames)!=1 or len(hits)!=1:raise RuntimeError('unique named original Gain control required')
        node=hits[0];node.clear_cache()
        if not all(node.get_state_set().contains(flag)for flag in(A.StateType.ENABLED,A.StateType.SENSITIVE,A.StateType.SHOWING)):raise RuntimeError('Gain control hidden/disabled')
        if not c.inside(optical.extents(node,A),ui.main['rect']):raise RuntimeError('Gain control outside owner')
        return node
    prior=set(SESSION.glob('live_*/manual_gain_*/transaction.json'))
    spin=find('spin button','相机Gain请求值dB');value=spin.get_value_iface()
    if not value.get_minimum_value()<=11<=value.get_maximum_value():raise RuntimeError('Gain request outside SDK range')
    if not value.set_current_value(11.):raise RuntimeError('staged public Gain request rejected')
    if find('spin button','相机Gain请求值dB').get_value_iface().get_current_value()!=11.:raise RuntimeError('staged Gain request not11')
    if ui.state()['camera_active_readback']!=baseline or set(SESSION.glob('live_*/manual_gain_*/transaction.json'))!=prior:raise RuntimeError('Gain was applied before explicit click')
    write(out/'GAIN_STAGED.json',dict(requested_db=11.,camera_before=baseline,only_request_widget_edited=True))
    write(out/'GAIN_APPLY_CLICK.json',n.click_visible_node(ui,find('push button','应用相机Gain'),A,execute=True))
    until=time.monotonic()+25
    while True:
        h=read(SESSION/'heartbeat.json');fresh=set(SESSION.glob('live_*/manual_gain_*/transaction.json'))-prior
        if len(fresh)>1:raise RuntimeError('competing Gain transactions')
        if fresh:
            transaction=next(iter(fresh))
            try:result=read(transaction)
            except (ValueError,OSError):pass
            else:break
        if h['hardware_fault']or time.monotonic()>until:raise RuntimeError('Gain transaction failed/timed out; no retry')
        time.sleep(.1)
    if not result.get('completed'):raise RuntimeError('Gain transaction incomplete')
    h=wait(lambda h:h['hardware_fault']is not None or h.get('camera_active_readback',{}).get('Gain')=='10.881464004516602'and h['live_camera']and h.get('latest_frame')and h['latest_frame']['age_seconds']<1)
    expected=dict(baseline,Gain='10.881464004516602')
    if h['hardware_fault']or h['camera_active_readback']!=expected:raise RuntimeError('restored Gain/current other parameters not exact')
    write(out/'GAIN_RESTORED.json',dict(transaction_path=str(transaction),transaction=result,heartbeat=h,
        actual_all_zero=c.inspect(SESSION,parked=False),request_db=11.,actual_db=float(expected['Gain']),
        original_Gain_transaction_restarted_acquisition=True,external_SDK_connections=0))
    return h

def action(key,out=ROOT):
    names={'recheck':('RECHECK_CLICK.json','RECHECK_DONE.json'),'ccd':('CCD_START_CLICK.json','CCD_START_DONE.json'),
           'gain':('GAIN_STAGED.json','GAIN_APPLY_CLICK.json','GAIN_RESTORED.json')}[key]
    if any((out/name).exists()for name in names):raise RuntimeError('new explicit records required before any click')
    ui,c,n=setup()
    if key=='recheck':
        write(out/'RECHECK_CLICK.json',n.invoke(ui,'recheck',execute=True))
        h=wait(lambda h:not h['busy']and h['hardware_fault']is None)
        write(out/'RECHECK_DONE.json',dict(heartbeat=h,pixel_check=c.inspect(SESSION,parked=True)))
    elif key=='ccd':
        write(out/'CCD_START_CLICK.json',n.invoke(ui,'ccd',execute=True))
        h=wait(lambda h:h['hardware_fault']is not None or h['live_camera']and h.get('latest_frame')and h['latest_frame']['age_seconds']<1)
        write(out/'CCD_START_DONE.json',dict(heartbeat=h,pixel_check=c.inspect(SESSION,parked=False)))
        if h['hardware_fault']:raise RuntimeError(h['hardware_fault'])
    else:h=restore_gain(ui,c,n,out)
    print(json.dumps(dict(action=key,pid=PID,fault=h['hardware_fault'],live_camera=h['live_camera'])),flush=True)

if __name__=='__main__':
    sys.path.insert(0,str(LAB/'src'));p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('recheck','ccd','gain'));p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--records',type=Path);a=p.parse_args()
    if not a.light_blocked_confirmed:p.error('continued physical shielding required')
    if a.records is not None:
        if not a.records.is_absolute()or a.records.parent!=ROOT:raise ValueError('new direct E040 records directory required')
        a.records.mkdir(exist_ok=False)
    action(a.action,ROOT if a.records is None else a.records)
