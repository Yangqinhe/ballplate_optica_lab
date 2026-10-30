"""Shielded public-UI switch. Retain before stopping; never inject live code."""
import argparse
from pathlib import Path
import subprocess
import re
import sys
from types import SimpleNamespace
import ctypes as C
import time
from run import ROOT, LAB, PRIOR, DISPLAY, REPAIRS, read, write, sha, ref, load


def setup():
    op=load('e078_public_lifecycle',REPAIRS/'operate.py')
    op.ROOT=ROOT;op.PRIOR=PRIOR;op.RET=ROOT/'RETENTION_01';op.SESSION=PRIOR/'GUI'
    identity=read(PRIOR/'POST_STARTUP_01/RESULT.json')
    op.PID=identity['pid'];op.START=identity['start_ticks']
    op.OLD_UNIT='ballplate-optical-shared-main-20261027-e078.service'
    op.UNIT='ballplate-optical-shared-filter-20261027-e079.service'
    original=op.load
    def routed(name,path):
        if name in ('e075_display_profile','e075_no_owner','e075_launch_guard','e075_verify_display'):path=DISPLAY/'run.py'
        return original(name,path)
    op.load=routed
    def state(allow_fault=False):
        h=read(op.SESSION/'heartbeat.json');proc=Path('/proc',str(op.PID))
        op.require(h['pid']==op.PID and 0<=time.time_ns()-h['unix_ns']<3e9,'fresh old owner required')
        op.require((proc/'stat').read_text().rsplit(')',1)[1].split()[19]==op.START,'owner lifetime differs')
        command=read(PRIOR/'LAUNCH_PLAN.json')['command']
        op.require([x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==command[command.index('--')+1:],'owner command differs')
        op.require(not any(h.get(k) for k in ('loop_active','adaptation_active','deployment_busy')) and not h.get('spatial_calibration',{}).get('busy'),'competing task active')
        return h
    op.owner_state=state
    def clients():
        cfg=read(PRIOR/'CONFIG.json')
        adapter=load('e079_old_display',DISPLAY/'run.py')
        if 'display_linux' not in sys.modules:adapter.install_display(read(cfg['display_profile']['path']))
        from optical_lab import legacy_gui
        c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy')
        legacy_gui._install_accessibility_compatibility();c.install()
        load('e079_clients_compat',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
        n=load('e079_native',LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py')
        n.ORDINARY=[1080,1080,2560,1440]
        rows=[]
        for line in subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines():
            if '"光路工作台 · 主界面／八区稳定性' in line:
                row=c.window(re.search(r'0x[0-9a-fA-F]+',line)[0])
                if row['pid']==op.PID and row['viewable']:rows.append(row)
        op.require(len(rows)==1 and c.inside(rows[0]['rect'],n.ORDINARY),'unique old owner ordinary window required')
        return SimpleNamespace(pid=op.PID,main=rows[0],state=state),c,n
    op.clients=clients
    original_click=op.click
    def click(ui,n,name,dialog=None):
        h=state(True)
        if name=='复核并恢复（不投图）' and h['hardware_fault'] is None and not any(h.get(k) for k in ('live_camera','busy','camera_reserved')):
            return dict(clicked=False,reason='normal stop restored baseline; cold check still required')
        return original_click(ui,n,name,dialog)
    op.click=click
    return op


def place(op):
    """Move only the identified ordinary control window; never output windows."""
    h=op.owner_state(True)
    op.require(not h['projection_armed'] and (not h['busy'] or h['live_camera']),'disarmed UI, not a camera transition, required')
    from optical_lab import legacy_gui
    c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy')
    rows=[]
    for line in subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines():
        if '"光路工作台 · 主界面／八区稳定性' in line:
            row=c.window(re.search(r'0x[0-9a-fA-F]+',line)[0])
            if row['pid']==op.PID and row['viewable']:rows.append(row)
    op.require(len(rows)==1,'unique old control client required');main=rows[0]
    class Data(C.Union):_fields_=[('longs',C.c_long*5),('bytes',C.c_char*20)]
    class Message(C.Structure):
        _fields_=[('type',C.c_int),('serial',C.c_ulong),('send_event',C.c_int),('display',C.c_void_p),
            ('window',C.c_ulong),('message_type',C.c_ulong),('format',C.c_int),('data',Data)]
    class Event(C.Union):_fields_=[('message',Message),('pad',C.c_long*24)]
    x=C.CDLL('libX11.so.6');x.XOpenDisplay.argtypes=[C.c_char_p];x.XOpenDisplay.restype=C.c_void_p
    d=x.XOpenDisplay(None);op.require(d,'display unavailable')
    x.XDefaultRootWindow.argtypes=[C.c_void_p];x.XDefaultRootWindow.restype=C.c_ulong
    x.XInternAtom.argtypes=[C.c_void_p,C.c_char_p,C.c_int];x.XInternAtom.restype=C.c_ulong
    x.XSendEvent.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_long,C.POINTER(Event)]
    x.XSync.argtypes=[C.c_void_p,C.c_int];x.XCloseDisplay.argtypes=[C.c_void_p]
    def send(name,values):
        e=Event();e.message.type=33;e.message.send_event=1;e.message.display=d
        e.message.window=int(main['wid'],16);e.message.message_type=x.XInternAtom(d,name,False);e.message.format=32
        for i,v in enumerate(values):e.message.data.longs[i]=v
        op.require(x.XSendEvent(d,x.XDefaultRootWindow(d),False,(1<<19)|(1<<20),C.byref(e)),'WM request rejected')
        x.XSync(d,False)
    try:
        send(b'_NET_WM_STATE',[0,x.XInternAtom(d,b'_NET_WM_STATE_MAXIMIZED_HORZ',False),x.XInternAtom(d,b'_NET_WM_STATE_MAXIMIZED_VERT',False),2])
        time.sleep(.2)
        send(b'_NET_MOVERESIZE_WINDOW',[10|(15<<8)|(2<<12),1120,1140,1840,1220]);time.sleep(.6)
        after=c.window(main['wid']);op.require(after['pid']==op.PID and c.inside(after['rect'],[1080,1080,2560,1440]),'main not inside ordinary screen')
        write(ROOT/'CONTROL_WINDOW_PLACED.json',dict(before=main,after=after,output_windows_not_moved=True,shielding_confirmed=True))
    finally:x.XCloseDisplay(d)


def retain(op):
    # The saved current RAW exists, but disarmed UI has no live comparison.
    # Bind the unique frozen reference matching the complete current camera
    # profile. Restore it as pending, requiring original signature validation.
    h=op.owner_state(True)
    matches=[]
    for path in op.SESSION.glob('live_*/zero_reference_*/frozen_geometry.json'):
        folder=path.parent
        if read(folder/'capture.json')['active']==h['camera_active_readback'] and read(folder/'model_reference.json')==h['optical_reference']:
            matches.append(path)
    op.require(h['geometry_frozen'] and len(matches)==1,'unique matching frozen reference required')
    geometry=read(matches[0])
    write(ROOT/'GEOMETRY_RETENTION.json',dict(registration=geometry,source=ref(matches[0]),
        camera=ref(matches[0].parent/'capture.json'),model=ref(matches[0].parent/'model_reference.json'),
        current_locked_heartbeat=h,restoration_pending_signature_validation=True,
        no_new_comparison_claimed=True,raw_snapshot=ref(op.SESSION/'snapshot_20261027_190614_761675/metadata.json')))
    helper=load('e078_retain_controls',LAB/'experiments/E076_dmd_scale_20261027_1/lifecycle.py')
    helper.ROOT=ROOT;helper.PRIOR=PRIOR
    read_state=op.owner_state;helper.state=lambda _:read_state()
    helper.retain(op)


def configure(op):
    proof=ROOT/'LIFECYCLE_RESULT.json';op.require(read(proof)['completed'],'normal close and baseline required')
    ret=read(ROOT/'RETENTION_01/CURRENT_UI.json');h=ret['heartbeat'];controls=ret['controls']
    def values(name):return [v['value'] for v in controls if v['role']=='spin button' and v['name']==name]
    def one(name):
        rows=values(name);op.require(len(rows)==1,'ambiguous spin '+name);return rows[0]
    def checked(prefix):
        rows=[v for v in controls if v['role']=='check box' and v['name'].startswith(prefix)]
        op.require(len(rows)==1,'ambiguous checkbox '+prefix);return rows[0]['checked']
    cfg=read(ROOT/'PREPARED_04/CONFIG.json')
    display=values('CCD显示设置 · 不进入八区积分／电域网络');stable=values('输入激励与采样')
    op.require(len(display)==4 and len(stable)==4,'control grid differs')
    combos=[v['name'] for v in controls if v['role']=='combo box']
    op.require(all(v in combos for v in ('手动显示增益','关闭时间滤波','CENTER','X_POS12','A/B交替回访')),'retained selections differ')
    op.require(any(v['name'].startswith('已应用R=2；CENTER；') for v in controls),'current applied ball catalog differs')
    exposure=one('ROI与曝光');gain=one('相机Gain请求值dB')
    op.require(gain==0.,'nonzero Gain requires restoration validation')
    snap=read(ROOT/'GEOMETRY_RETENTION.json')
    op.require(sha(snap['source']['path'])==snap['source']['sha256'],'geometry changed')
    registration=snap['registration'];op.require(h['geometry_frozen'] and registration['frozen'],'current locked geometry required')
    folder=Path(registration['reference_folder']);capture=read(folder/'capture.json')
    op.require(registration==read(folder/'frozen_geometry.json') and capture['active']==h['camera_active_readback']
        and read(folder/'model_reference.json')==h['optical_reference'],'current geometry/camera mismatch')
    scales=[v['value'] for v in controls if v['role']=='spin button' and v['name'].startswith('DMD · 1920') and 1.<=v['value']<=1.8]
    op.require(len(scales)==1 and round(scales[0]*600)==registration.get('dmd_side',600),'DMD scale mismatch')
    state=read(ROOT/'RETENTION_01/DISPLAY_STATE.json')
    state.update(manual=dict(mode='fixed',gain=display[0],black=display[1],gamma=display[2]),
        preview_filter=dict(mode='off',alpha=display[3],original=checked('原单帧对照')),
        current_visible_controls=ref(ROOT/'RETENTION_01/CURRENT_UI.json'))
    write(ROOT/'DISPLAY_CURRENT.json',state)
    restart=read(cfg['restart']['path'])
    restart.update(original_pid=op.PID,original_owner_start_ticks=op.START,original_session=str(op.SESSION),
        runtime_path=str(op.SESSION/'runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],
        optical_config=h['optical_reference'],requested_exposure_us=exposure,fault=None,
        registration=registration,expected_active=capture['active'],no_auto_capture=True,no_auto_projection=True)
    for item in (ROOT/'RETENTION_01/CURRENT_UI.json',ROOT/'GEOMETRY_RETENTION.json',
                 folder/'frozen_geometry.json',folder/'capture.json',folder/'model_reference.json'):
        restart['evidence_sha256'][str(item)]=sha(item)
    write(ROOT/'restart_contract.json',restart)
    recovery=read(cfg['recovery']['path']);recovery.update(old_pid=op.PID,
        baseline_source=str(ROOT/'restart_contract.json'),baseline_source_sha256=sha(ROOT/'restart_contract.json'),
        cold_baseline_readonly_verification_passed=True,planned_normal_restart=True,old_process_clean_rollback_claimed=True)
    write(ROOT/'recovery_record.json',recovery)
    cfg.update(restart=ref(ROOT/'restart_contract.json'),recovery=ref(ROOT/'recovery_record.json'),geometry=None,
        display=ref(ROOT/'DISPLAY_CURRENT.json'),caches=ref(ROOT/'RETENTION_01/PREVIEW_CACHES.json'),
        expected_active=h['camera_active_readback'],lifecycle_release=ref(proof),offline_preparation_only=False,
        shared_filter_frames=int(stable[0]),
        replaced_owner_pid=op.PID,camera_requested=dict(exposure_us=exposure,gain_db=gain),retained_dmd_scale=scales[0],
        retained_region_overlay=checked('显示八区划分'),retained_pending_ball_radius=int(one('球扩张半径R（96图像素）')),
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],filter=state['preview_filter']))
    cfg['settings'].update(software_gain=display[0],black_ADU=display[1],gamma=display[2],frames=int(stable[0]),
        repeats=int(stable[1]),settle_seconds=stable[2],interval_seconds=stable[3],switch_inputs=True)
    cfg['retained_test_controls']=dict(selected_ids=dict(input_a='CENTER',input_b='X_POS12'),
        head_enabled=checked('计算候选18参数头'))
    for path in (ROOT/'run.py',PRIOR/'catalog.py',PRIOR/'workbench.py',Path(__file__).resolve(),ROOT/'verify_startup.py',ROOT/'check_visible.py',
                 proof,ROOT/'RETENTION_01/CURRENT_UI.json',*sorted((ROOT/'staged/optical_lab').glob('*.py'))):
        cfg['source_sha256'][str(path)]=sha(path)
    contract=read(cfg['contract']['path']);contract.update(source_sha256=cfg['source_sha256'],expected_active=cfg['expected_active'],
        dmd_scale_feature=dict(default=scales[0],maximum=1.8,manual_only=False,control_qualified=False),control_release_approved=False)
    write(ROOT/'CONTRACT.json',contract);cfg['contract']=ref(ROOT/'CONTRACT.json');write(ROOT/'CONFIG.json',cfg)
    print(ref(ROOT/'CONFIG.json'))


def launch(op):
    cfg=read(ROOT/'CONFIG.json');op.require(read(ROOT/'QA_RETAINED/SMOKE.json')['passed'],'retained smoke required')
    adapter=load('e078_launch_adapter',DISPLAY/'run.py');adapter.no_owner()
    adapter.install_display(read(cfg['display_profile']['path'])).read_roles()
    for path,digest in cfg['source_sha256'].items():op.require(sha(path)==digest,'source changed '+path)
    command=read(PRIOR/'LAUNCH_PLAN.json')['command']
    replacements={str(PRIOR/'run.py'):str(ROOT/'run.py'),str(PRIOR/'CONFIG.json'):str(ROOT/'CONFIG.json'),
        sha(PRIOR/'CONFIG.json'):sha(ROOT/'CONFIG.json'),'--unit='+op.OLD_UNIT:'--unit='+op.UNIT}
    op.require(all(command.count(k)==1 for k in replacements),'launch command differs')
    command=[replacements.get(v,v) for v in command]
    write(ROOT/'LAUNCH_PLAN.json',dict(command=command,shielding_confirmed=True,no_auto_projection=True,lifecycle=cfg['lifecycle_release']))
    result=subprocess.run(command,text=True,capture_output=True,timeout=20)
    write(ROOT/'LAUNCH_RESULT.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    op.require(result.returncode==0,'launch failed; no automatic retry');print(result.stdout+result.stderr)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('place','retain','close','cold','configure','launch'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('current shielding required')
    op=setup()
    if a.action in ('place','retain','configure','launch'):globals()[a.action](op)
    else:getattr(op,a.action)()
