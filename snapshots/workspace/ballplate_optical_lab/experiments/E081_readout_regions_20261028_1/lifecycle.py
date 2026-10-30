"""Shielded public-UI switch. Retain before stopping; never inject live code."""
import argparse
import json
from pathlib import Path
import subprocess
import re
import sys
from types import SimpleNamespace
import ctypes as C
import time
from run import ROOT, LAB, PREVIOUS as PRIOR, DISPLAY, REPAIRS, read, write, sha, ref, load


def setup():
    op=load('e078_public_lifecycle',REPAIRS/'operate.py')
    op.ROOT=ROOT;op.PRIOR=PRIOR;op.RET=ROOT/'RETENTION_01';op.SESSION=PRIOR/'GUI'
    identity=read(PRIOR/'POST_STARTUP_01/RESULT.json')
    op.PID=identity['pid'];op.START=identity['start_ticks']
    op.OLD_UNIT='ballplate-optical-ball-input-20261028-e080.service'
    op.UNIT='ballplate-optical-readout-regions-20261028-e081.service'
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
            if '"光路工作台 · 四球尺寸/填孔 · 共用滤波' in line:
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


def retain(op):
    h=op.owner_state(True);ui,c,n=op.clients()
    # Snapshot only the existing current registration; a shielded image is not
    # used to refit it. Select the unique NEW file from this explicit click.
    before=set(op.SESSION.glob('snapshot_*'))
    evidence=ROOT/'SNAPSHOT_RETENTION';evidence.mkdir()
    write(evidence/'CLICK.json',op.click(ui,n,'保存当前对照'))
    deadline=time.monotonic()+10
    while True:
        new=set(op.SESSION.glob('snapshot_*'))-before
        if len(new)==1 and (next(iter(new))/'comparison.json').is_file():break
        op.require(len(new)<=1 and time.monotonic()<deadline,'unique current snapshot with geometry required')
        time.sleep(.2)
    snapshot=next(iter(new));comparison=read(snapshot/'comparison.json')
    registration=comparison['geometry'];source=Path(registration['reference_folder'])/'frozen_geometry.json'
    op.require(h['geometry_frozen'] and registration==read(source),'snapshot geometry differs from sealed reference')
    op.require(read(snapshot/'metadata.json')['camera_active']==h['camera_active_readback'],'camera changed during retention')
    write(ROOT/'GEOMETRY_RETENTION.json',dict(source=ref(source),registration=registration,
        snapshot_comparison=ref(snapshot/'comparison.json'),shielded_snapshot_NOT_calibration=True))
    folder=op.RET;folder.mkdir();ui,c,n=op.clients();tree,A=op.nodes();controls=[]
    for node in tree:
        if node.get_role_name() in ('spin button','check box','combo box','label','push button'):
            value=node.get_value_iface()
            controls.append(dict(role=node.get_role_name(),name=node.get_name(),value=value.get_current_value() if value else None,
                checked=node.get_state_set().contains(A.StateType.CHECKED),showing=node.get_state_set().contains(A.StateType.SHOWING)))
    write(folder/'CURRENT_UI.json',dict(heartbeat=op.owner_state(True),controls=controls,main=ui.main))
    cfg=read(PRIOR/'CONFIG.json')
    for path,digest in cfg['source_sha256'].items():op.require(sha(path)==digest,'source changed '+path)
    events=(op.SESSION/'events.jsonl').read_bytes();op.require(events.endswith(b'\n'),'complete event snapshot required')
    files={};created=0
    for line in events.splitlines():
        row=json.loads(line)
        if row.get('state')=='created':created+=1
        if row.get('state')=='manual_file_loaded':
            op.require(type(row.get('SLM')) is bool,'explicit cache role required')
            files['slm' if row['SLM'] else 'dmd']={k:row[k] for k in ('path','sha256')}
    op.require(created==1 and set(files)=={'dmd','slm'},'complete current cache events required')
    for item in files.values():op.require(sha(item['path'])==item['sha256'],'cache changed')
    with (folder/'owner_events.jsonl').open('xb') as f:f.write(events)
    write(folder/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    # Shared finite-window filtering no longer emits the old preview-only event.
    # Use captured visible controls in configure(), never invent an old event.
    write(folder/'DISPLAY_STATE.json',read(cfg['display']['path']))
    write(folder/'PARK_CLICK.json',op.click(ui,n,'两路全0，CCD保持（Esc）'))
    end=time.monotonic()+15
    while True:
        now=op.owner_state(True)
        if not now['projection_armed'] and now['last_verified_revision']==now['display_revision']:break
        op.require(time.monotonic()<end,'zero ACK timeout');time.sleep(.15)
    write(folder/'ACTUAL_ZERO.json',op.zeros(c))
    if op.owner_state(True)['live_camera']:write(folder/'STOP_CLICK.json',op.click(ui,n,'停采并恢复原参数'))
    end=time.monotonic()+35
    while any(op.owner_state(True).get(k) for k in ('live_camera','busy','camera_reserved')):
        op.require(time.monotonic()<end,'release timeout');time.sleep(.2)
    write(folder/'STOPPED.json',op.owner_state(True));print('Current controls/caches retained; actual zero; normal CCD release.')


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
    display=values('CCD有效帧滤波与显示 · 滤波共用，显示映射另计');stable=values('输入激励与采样')
    op.require(len(display)==4 and len(stable)==3,'control grid differs')
    combos=[v['name'] for v in controls if v['role']=='combo box']
    op.require(all(v in combos for v in ('手动显示增益','原始单帧（不滤波）','输入 CENTER','输入 X_POS12','位置/速度清单轮测','当前载入四球（保留当前位置/历史）')),'retained selections differ')
    op.require(any('当前载入图\n已应用R=3\n填孔：开' in v['name'] for v in controls),'current applied ball cache differs')
    events=[json.loads(line) for line in (ROOT/'RETENTION_01/owner_events.jsonl').read_text().splitlines()]
    catalog_events=[row for row in events if row.get('state')=='RAW_validation_catalog_loaded']
    op.require(bool(catalog_events),'applied input catalog event required')
    cfg['catalog_preset']=catalog_events[-1]['reference']
    catalog=load('e081_retained_catalog',PRIOR/'catalog.py');catalog.validate(cfg['catalog_preset'])
    op.require(sha(cfg['catalog_preset']['path'])==cfg['catalog_preset']['sha256'],'catalog changed')
    op.require(any(v['name']==cfg['head_preset']['path'] for v in controls),'selected head differs')
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
        shared_filter_frames=int(one('有效帧RAW数N')),
        replaced_owner_pid=op.PID,camera_requested=dict(exposure_us=exposure,gain_db=gain),retained_dmd_scale=scales[0],
        retained_region_overlay=checked('显示八区划分'),retained_pending_ball_radius=int(one('球扩张半径R（96图像素）')),
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],filter=state['preview_filter']))
    cfg['settings'].update(software_gain=display[0],black_ADU=display[1],gamma=display[2],frames=int(one('有效帧RAW数N')),
        repeats=int(stable[0]),settle_seconds=stable[1],interval_seconds=stable[2],switch_inputs=True)
    cfg['retained_test_controls']=dict(mode='sweep',selected_ids=dict(input_a='CENTER',input_b='X_POS12'),
        head_enabled=checked('计算候选18参数头'))
    for path in (ROOT/'run.py',PRIOR/'catalog.py',Path(__file__).resolve(),ROOT/'verify_startup.py',
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('retain','close','cold','configure','launch'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('current shielding required')
    op=setup()
    if a.action in ('retain','configure','launch'):globals()[a.action](op)
    else:getattr(op,a.action)()
