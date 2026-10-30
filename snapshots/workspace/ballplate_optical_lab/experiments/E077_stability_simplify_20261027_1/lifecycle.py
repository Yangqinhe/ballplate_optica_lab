"""Explicit shielded switch using existing public UI and exact read-only checks."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from run import ROOT, LAB, PRIOR, DISPLAY, REPAIRS, read, write, sha, ref, load


def setup():
    op=load('e077_public_lifecycle',REPAIRS/'operate.py')
    op.ROOT=ROOT;op.PRIOR=PRIOR;op.RET=ROOT/'RETENTION_01';op.SESSION=PRIOR/'GUI'
    identity=read(PRIOR/'POST_STARTUP_01/RESULT.json')
    op.PID=identity['pid'];op.START=identity['start_ticks']
    op.OLD_UNIT='ballplate-optical-dmd-scale-20261027-e076.service'
    op.UNIT='ballplate-optical-stability-simple-20261027-e077.service'
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
    original_click=op.click
    def click(ui,n,name,dialog=None):
        h=state(True)
        if name=='复核并恢复（不投图）' and h['hardware_fault'] is None and not any(h.get(k) for k in ('live_camera','busy','camera_reserved')):
            return dict(clicked=False,reason='normal stop already restored baseline; final cold check still required')
        return original_click(ui,n,name,dialog)
    op.click=click
    return op


def retain(op):
    folder=ROOT/'SNAPSHOT_RETENTION'
    if not folder.exists():
        folder.mkdir();ui,c,n=op.clients()
        before=set(op.SESSION.glob('snapshot_*/comparison.json'))
        write(folder/'CLICK.json',op.click(ui,n,'保存当前对照'))
        deadline=time.monotonic()+10
        while True:
            new=set(op.SESSION.glob('snapshot_*/comparison.json'))-before
            if new:break
            op.require(time.monotonic()<deadline,'current geometry snapshot missing');time.sleep(.2)
        op.require(len(new)==1,'ambiguous new snapshot')
        comparison=next(iter(new));write(folder/'CURRENT_COMPARISON.json',dict(reference=ref(comparison),data=read(comparison)))
    else:
        saved=read(folder/'CURRENT_COMPARISON.json')
        op.require(sha(saved['reference']['path'])==saved['reference']['sha256'],'retained snapshot changed')
        op.require(saved['data']['model']==op.owner_state()['optical_reference'],'old snapshot model changed')
    helper=load('e077_retain_controls',PRIOR/'lifecycle.py')
    helper.ROOT=ROOT;helper.PRIOR=PRIOR
    read_state=op.owner_state
    helper.state=lambda _:read_state()
    helper.retain(op)


def configure(op):
    proof=ROOT/'LIFECYCLE_RESULT.json';op.require(read(proof)['completed'],'normal close and baseline required')
    ret=read(ROOT/'RETENTION_01/CURRENT_UI.json');h=ret['heartbeat'];controls=ret['controls']
    def values(name):return [v['value'] for v in controls if v['role']=='spin button' and v['name']==name]
    def one(name):
        items=values(name);op.require(len(items)==1,'ambiguous spin '+name);return items[0]
    def checked(prefix):
        items=[v for v in controls if v['role']=='check box' and v['name'].startswith(prefix)]
        op.require(len(items)==1,'ambiguous check '+prefix);return items[0]['checked']
    cfg=read(PRIOR/'CONFIG.json')
    display=values('CCD显示设置 · 不进入八区积分／电域网络');stable=values('固定参数与测试设计')
    op.require(len(display)==4 and len(stable)==11,'control grid differs')
    combos=[v['name'] for v in controls if v['role']=='combo box']
    op.require('手动显示增益' in combos and '指数平滑（预览采样帧）' in combos,'display mode changed')
    op.require(all(v in combos for v in ('CENTER','VY_POS','REF_SIN_SPARSE')),'selected inputs changed')
    exposure=one('ROI与曝光');gain=one('相机Gain请求值dB')
    op.require(gain==0.,'nonzero gain requires explicit restoration verification')
    # Exact matrix came from the public current-comparison snapshot, not mtime.
    snap=read(ROOT/'SNAPSHOT_RETENTION/CURRENT_COMPARISON.json')
    op.require(sha(snap['reference']['path'])==snap['reference']['sha256'],'saved snapshot changed')
    registration=snap['data']['geometry'];op.require(h['geometry_frozen'] and registration['frozen'],'current frozen geometry required')
    folder=Path(registration['reference_folder']);capture=read(folder/'capture.json')
    op.require(registration==read(folder/'frozen_geometry.json') and capture['active']==h['camera_active_readback']
        and read(folder/'model_reference.json')==h['optical_reference'],'current geometry/camera mismatch')
    scale_values=[v['value'] for v in controls if v['role']=='spin button' and v['name'].startswith('DMD · 1920') and 1.<=v['value']<=1.8]
    op.require(len(scale_values)==1,'DMD scale control ambiguous')
    scale=scale_values[0]
    op.require(round(scale*600)==registration.get('dmd_side',600),'geometry scale differs')
    state=read(ROOT/'RETENTION_01/DISPLAY_STATE.json')
    state.update(manual=dict(mode='fixed',gain=display[0],black=display[1],gamma=display[2]),
        preview_filter=dict(mode='ema',alpha=display[3],original=checked('原单帧对照')),
        current_visible_controls=ref(ROOT/'RETENTION_01/CURRENT_UI.json'))
    write(ROOT/'DISPLAY_CURRENT.json',state)
    restart=read(cfg['restart']['path'])
    restart.update(original_pid=op.PID,original_session=str(op.SESSION),runtime_path=str(op.SESSION/'runtime_sources.json'),
        saved_unix_ns=time.time_ns(),settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],
        optical_config=h['optical_reference'],requested_exposure_us=exposure,fault=None,
        registration=registration,expected_active=capture['active'],no_auto_capture=True,no_auto_projection=True)
    for item in (ROOT/'RETENTION_01/CURRENT_UI.json',ROOT/'SNAPSHOT_RETENTION/CURRENT_COMPARISON.json',
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
        replaced_owner_pid=op.PID,camera_requested=dict(exposure_us=exposure,gain_db=gain),retained_dmd_scale=scale,
        retained_region_overlay=checked('显示八区划分'),
        retained_pending_ball_radius=int(one('球扩张半径R（96图像素）')),
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],filter=state['preview_filter']))
    cfg['settings'].update(software_gain=stable[2],black_ADU=stable[3],gamma=stable[4],frames=int(stable[5]),
        repeats=int(stable[6]),settle_seconds=stable[7],interval_seconds=stable[8],switch_inputs=checked('测试中换图'))
    cfg['retained_test_controls'].update(reference_mode=checked('参考补偿对照'),reference_frames=int(stable[9]),
        baseline_frames=int(stable[10]),head_enabled=checked('计算固定18参数RAW头'))
    for path in [ROOT/'run.py',ROOT/'scaling.py',Path(__file__).resolve(),proof,ROOT/'RETENTION_01/CURRENT_UI.json',
                 *sorted((ROOT/'staged/optical_lab').glob('*.py'))]:cfg['source_sha256'][str(path)]=sha(path)
    contract=read(cfg['contract']['path']);contract.update(source_sha256=cfg['source_sha256'],expected_active=cfg['expected_active'],
        dmd_scale_feature=dict(default=scale,maximum=1.8,manual_only=False,control_qualified=False),control_release_approved=False)
    write(ROOT/'CONTRACT.json',contract);cfg['contract']=ref(ROOT/'CONTRACT.json');write(ROOT/'CONFIG.json',cfg)
    print(ref(ROOT/'CONFIG.json'))


def launch(op):
    cfg=read(ROOT/'CONFIG.json');op.require(read(ROOT/'QA_RETAINED/SMOKE.json')['passed'],'retained configuration smoke required')
    adapter=load('e077_launch_adapter',DISPLAY/'run.py');adapter.no_owner()
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('retain','close','cold','configure','launch','verify'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('current shielding required')
    op=setup()
    if a.action in ('retain','configure','launch'):globals()[a.action](op)
    else:getattr(op,a.action)()
