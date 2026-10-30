"""Explicit shielded restart; external public UI only, no live code injection."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
PRIOR=LAB/'experiments/E075_spatial_repair_20261027_1'
ADAPTER=LAB/'experiments/E074_dmd1920_rollback_20261026_1/run.py'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def setup():
    prior=load('run',PRIOR/'run.py')
    op=load('e076_preserved_lifecycle',PRIOR/'operate.py')
    op.ROOT=ROOT;op.PRIOR=PRIOR;op.RET=ROOT/'RETENTION_01';op.SESSION=PRIOR/'GUI'
    identity=read(PRIOR/'POST_STARTUP_01/RESULT.json')
    op.PID=identity['pid'];op.START=identity['start_ticks']
    op.OLD_UNIT='ballplate-optical-spatial-repair-20261027-e075.service'
    op.UNIT='ballplate-optical-dmd-scale-20261027-e076.service'
    original_load=op.load
    def routed(name,path):
        if name in ('e075_display_profile','e075_no_owner','e075_launch_guard','e075_verify_display'):
            path=ADAPTER
        return original_load(name,path)
    op.load=routed
    return op

def state(op):
    h=read(op.SESSION/'heartbeat.json');proc=Path('/proc',str(op.PID))
    op.require(h['pid']==op.PID and 0<=time.time_ns()-h['unix_ns']<3e9,'fresh original owner required')
    op.require((proc/'stat').read_text().rsplit(')',1)[1].split()[19]==op.START,'owner lifetime differs')
    cmd=read(PRIOR/'LAUNCH_PLAN.json')['command']
    op.require([x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==cmd[cmd.index('--')+1:],'owner command differs')
    op.require(not any(h.get(k) for k in ('loop_active','adaptation_active','deployment_busy'))
        and not h.get('spatial_calibration',{}).get('busy'),'competing workflow active')
    return h

def retain(op):
    folder=ROOT/'RETENTION_01';folder.mkdir()
    old_state=op.owner_state
    op.owner_state=lambda allow_fault=False:state(op)
    ui,c,n=op.clients()
    tree,A=op.nodes();controls=[]
    for node in tree:
        if node.get_role_name() in ('spin button','check box','combo box','label','push button'):
            value=node.get_value_iface()
            controls.append(dict(role=node.get_role_name(),name=node.get_name(),
                value=value.get_current_value() if value else None,
                checked=node.get_state_set().contains(A.StateType.CHECKED),
                showing=node.get_state_set().contains(A.StateType.SHOWING)))
    h=state(op)
    write(folder/'CURRENT_UI.json',dict(heartbeat=h,controls=controls,main=ui.main))
    cfg=read(PRIOR/'CONFIG.json')
    for path,digest in cfg['source_sha256'].items():op.require(op.sha(path)==digest,'source changed '+path)
    events=(op.SESSION/'events.jsonl').read_bytes()
    helper=load('e076_retention_math',LAB/'experiments/E045_dense_state_sampling_20261025_1/retention.py')
    fold=helper.fold_events(events)
    files={role:{k:fold[role]['event'][k] for k in ('path','sha256')} for role in ('dmd','slm')}
    for record in files.values():op.require(op.sha(record['path'])==record['sha256'],'cache changed')
    with (folder/'owner_events.jsonl').open('xb') as f:f.write(events)
    write(folder/'DISPLAY_STATE.json',helper.display_state(fold,h['camera_active_readback']))
    write(folder/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    write(folder/'PARK_CLICK.json',op.click(ui,n,'两路全0，CCD保持（Esc）'))
    end=time.monotonic()+15
    while state(op)['projection_armed'] or state(op)['last_verified_revision']!=state(op)['display_revision']:
        op.require(time.monotonic()<end,'black ACK timeout');time.sleep(.15)
    write(folder/'ACTUAL_ZERO.json',op.zeros(c))
    if state(op)['live_camera']:
        write(folder/'STOP_CLICK.json',op.click(ui,n,'停采并恢复原参数'))
    end=time.monotonic()+35
    while any(state(op).get(k) for k in ('live_camera','busy','camera_reserved')):
        op.require(time.monotonic()<end,'camera release timeout');time.sleep(.2)
    write(folder/'STOPPED.json',state(op))
    op.owner_state=old_state
    print('Saved visible controls/caches, actual zero verified, original CCD released normally.')

def configure(op):
    proof=ROOT/'LIFECYCLE_RESULT.json'
    op.require(read(proof)['completed'],'strict lifecycle must pass')
    op.require(not Path('/proc',str(op.PID)).exists(),'old owner must be absent')
    ret=read(ROOT/'RETENTION_01/CURRENT_UI.json');h=ret['heartbeat'];controls=ret['controls']
    def values(name):return [v['value'] for v in controls if v['role']=='spin button' and v['name']==name]
    def checked(prefix):
        rows=[v for v in controls if v['role']=='check box' and v['name'].startswith(prefix)]
        op.require(len(rows)==1,'unique checkbox required '+prefix);return rows[0]['checked']
    def single(name):
        rows=values(name);op.require(len(rows)==1,'unique value required '+name);return rows[0]
    cfg=read(PRIOR/'CONFIG.json')
    display=values('CCD显示设置 · 不进入八区积分／电域网络')
    stable=values('固定参数与测试设计')
    op.require(len(display)==4 and len(stable)==11,'current controls ambiguous')
    combos=[v['name'] for v in controls if v['role']=='combo box']
    op.require('手动显示增益' in combos and '指数平滑（预览采样帧）' in combos,'display modes differ')
    op.require(all(v in combos for v in ('CENTER','VY_POS','REF_SIN_SPARSE')),'selected test inputs changed')
    op.require(not h['geometry_frozen'],'frozen new geometry requires explicit retention')
    op.require([single(k) for k in ('手动显示缩放','手动显示X','手动显示Y')]==[1.,0.,0.],'manual mapping changed')
    exposure=single('ROI与曝光');gain=single('相机Gain请求值dB')
    op.require(gain==0.,'nonzero camera gain needs additional restore verification')
    display_state=read(ROOT/'RETENTION_01/DISPLAY_STATE.json')
    display_state.update(manual=dict(mode='fixed',gain=display[0],black=display[1],gamma=display[2]),
        preview_filter=dict(mode='ema',alpha=display[3],original=checked('原单帧对照')),
        current_visible_controls=op.ref(ROOT/'RETENTION_01/CURRENT_UI.json'),prior_events_are_stale=True)
    write(ROOT/'DISPLAY_CURRENT.json',display_state)
    restart=read(cfg['restart']['path'])
    restart.update(original_pid=op.PID,original_owner_start_ticks=op.START,original_session=str(op.SESSION),
        runtime_path=str(op.SESSION/'runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],
        optical_config=h['optical_reference'],requested_exposure_us=exposure,
        fault=None,registration=None,expected_active=None,no_auto_capture=True,no_auto_projection=True,
        old_process_clean_rollback_claimed=True)
    restart['evidence_sha256'][str(ROOT/'RETENTION_01/CURRENT_UI.json')]=op.sha(ROOT/'RETENTION_01/CURRENT_UI.json')
    write(ROOT/'restart_contract.json',restart)
    recovery=read(cfg['recovery']['path']);recovery.update(old_pid=op.PID,
        baseline_source=str(ROOT/'restart_contract.json'),baseline_source_sha256=op.sha(ROOT/'restart_contract.json'),
        cold_baseline_readonly_verification_passed=True,planned_normal_restart=True,old_process_clean_rollback_claimed=True)
    write(ROOT/'recovery_record.json',recovery)
    cfg.update(restart=op.ref(ROOT/'restart_contract.json'),recovery=op.ref(ROOT/'recovery_record.json'),
        display=op.ref(ROOT/'DISPLAY_CURRENT.json'),caches=op.ref(ROOT/'RETENTION_01/PREVIEW_CACHES.json'),
        expected_active=h['camera_active_readback'],lifecycle_release=op.ref(proof),
        offline_preparation_only=False,replaced_owner_pid=op.PID,
        camera_requested=dict(exposure_us=exposure,gain_db=gain),
        retained_pending_ball_radius=int(single('球扩张半径R（96图像素）')),
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],
                                      filter=display_state['preview_filter']))
    cfg['settings'].update(software_gain=stable[2],black_ADU=stable[3],gamma=stable[4],
        frames=int(stable[5]),repeats=int(stable[6]),settle_seconds=stable[7],interval_seconds=stable[8],
        switch_inputs=checked('测试中换图'))
    cfg['retained_test_controls'].update(reference_mode=checked('参考补偿对照'),
        reference_frames=int(stable[9]),baseline_frames=int(stable[10]),
        head_enabled=checked('计算固定18参数RAW头'))
    for path in (ROOT/'run.py',ROOT/'scaling.py',Path(__file__),proof,ROOT/'RETENTION_01/CURRENT_UI.json'):
        cfg['source_sha256'][str(path)]=op.sha(path)
    contract=read(cfg['contract']['path']);contract.update(source_sha256=cfg['source_sha256'],
        expected_active=cfg['expected_active'],control_release_approved=False,
        dmd_scale_feature=dict(default=1.,maximum=1.8,manual_only=True,physical_verified=False,control_qualified=False))
    write(ROOT/'CONTRACT.json',contract);cfg['contract']=op.ref(ROOT/'CONTRACT.json')
    write(ROOT/'CONFIG.json',cfg);print(op.ref(ROOT/'CONFIG.json'))

def launch(op):
    import subprocess
    cfg=read(ROOT/'CONFIG.json')
    op.require(read(ROOT/'QA_FINAL/DMD_SCALE_SMOKE.json')['passed'],'final configured smoke required')
    adapter=load('e076_launch_adapter',ADAPTER);adapter.no_owner()
    adapter.install_display(read(cfg['display_profile']['path'])).read_roles()
    for path,digest in cfg['source_sha256'].items():op.require(op.sha(path)==digest,'source changed '+path)
    command=read(PRIOR/'LAUNCH_PLAN.json')['command']
    replacements={str(PRIOR/'run.py'):str(ROOT/'run.py'),str(PRIOR/'CONFIG.json'):str(ROOT/'CONFIG.json'),
        op.sha(PRIOR/'CONFIG.json'):op.sha(ROOT/'CONFIG.json'),'--unit='+op.OLD_UNIT:'--unit='+op.UNIT}
    op.require(all(command.count(k)==1 for k in replacements),'old launch command differs')
    command=[replacements.get(v,v) for v in command]
    write(ROOT/'LAUNCH_PLAN.json',dict(command=command,shielding_confirmed=True,no_auto_projection=True,lifecycle=cfg['lifecycle_release']))
    result=subprocess.run(command,text=True,capture_output=True,timeout=20)
    write(ROOT/'LAUNCH_RESULT.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    op.require(result.returncode==0,'launch failed; no automatic retry');print(result.stdout+result.stderr)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('retain','close','cold','configure','launch','verify'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('current physical shielding required')
    op=setup()
    if a.action in ('retain','configure','launch'):globals()[a.action](op)
    else:getattr(op,a.action)()
