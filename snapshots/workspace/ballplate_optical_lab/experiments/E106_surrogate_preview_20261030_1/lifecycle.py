"""Explicit shielded GUI replacement; public controls, no live code injection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import model
import run

ROOT=run.ROOT;LAB=run.LAB;PRIOR=run.PRIOR
read=model.read;write=model.write;ref=model.ref;sha=model.digest
OLD_UNIT='ballplate-optical-readout-regions-20261028-e081.service'
UNIT='ballplate-optical-surrogate-preview-20261030-e106.service'
TITLE='光路工作台 · 主界面统一参数 · 8/16/32/64读出'

def require(value,message):
    if not value:raise RuntimeError(message)

def setup():
    base=run.base_module();old=sys.modules.get('run');sys.modules['run']=base
    try:op=base.load('e106_public_lifecycle',base.REPAIRS/'operate.py')
    finally:sys.modules['run']=old
    op.ROOT=ROOT;op.PRIOR=PRIOR;op.RET=ROOT/'RETENTION_01';op.SESSION=PRIOR/'GUI'
    identity=read(PRIOR/'POST_STARTUP_01/RESULT.json');op.PID=identity['pid'];op.START=identity['start_ticks']
    op.OLD_UNIT=OLD_UNIT;op.UNIT=UNIT
    original=op.load
    def routed(name,path):
        if name in ('e075_display_profile','e075_no_owner','e075_launch_guard','e075_verify_display'):
            path=base.DISPLAY/'run.py'
        return original(name,path)
    op.load=routed
    def clients():
        cfg=read(PRIOR/'CONFIG.json');adapter=op.load('e075_display_profile',base.DISPLAY/'run.py')
        if 'display_linux' not in sys.modules:adapter.install_display(read(cfg['display_profile']['path']))
        from optical_lab import legacy_gui
        c=legacy_gui._load_clients(base.LEGACY);legacy_gui._install_accessibility_compatibility();c.install()
        base.load('e106_accessible',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
        n=base.load('e106_native',LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py')
        n.ORDINARY=[1080,1080,2560,1440];rows=[]
        for line in subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines():
            if '"'+TITLE in line:
                row=c.window(re.search(r'0x[0-9a-fA-F]+',line)[0])
                if row['pid']==op.PID and row['viewable']:rows.append(row)
        require(len(rows)==1 and c.inside(rows[0]['rect'],n.ORDINARY),'unique original ordinary-screen owner required')
        return SimpleNamespace(pid=op.PID,main=rows[0],state=op.owner_state),c,n
    op.clients=clients
    original_click=op.click
    def click(ui,n,name,dialog=None):
        h=op.owner_state(True)
        if name=='复核并恢复（不投图）' and h['hardware_fault'] is None and not any(h.get(k) for k in ('live_camera','busy','camera_reserved')):
            return dict(clicked=False,reason='normal stop already restored; separate cold verification required')
        return original_click(ui,n,name,dialog)
    op.click=click
    return op

def retain():
    op=setup();base=run.base_module();wf=base.load('e106_retention_client',model.EXPERIMENT/'workflow.py')
    out=op.RET;out.mkdir();h=read(op.SESSION/'heartbeat.json');expected=wf.camera_contract(h)
    write(out/'AUTHORIZATION.json',dict(physical_shielding_confirmed=True,unix_ns=time.time_ns(),
        scope='retain current GUI, normal close, cold readback, new zero windows; no new projection',
        user_confirmation='已遮光，可以切换模型预览版'))
    client=wf.make_client(out,expected)
    require(client.state()['current_page']==0,'main page required before retention')
    snapshot=wf.save_snapshot(client);write(out/'SNAPSHOT.json',snapshot)
    geometry=Path(snapshot['geometry']['reference_folder'])/'frozen_geometry.json'
    require(read(geometry)==snapshot['geometry'],'current frozen geometry changed')
    zero=wf.park(client);main=client.dump_controls();events,files=wf.event_state()
    client.tab('读出稳定性测试');test=client.dump_controls();client.tab('主界面 · 光路与参数')
    write(out/'CURRENT_UI.json',dict(heartbeat=client.state(),main_controls=main,test_controls=test,
        snapshot=snapshot,geometry=ref(geometry),files=files,events=events,windows=client.windows,
        before=expected,output_zero=zero,final_zero=client.zeros()))
    write(out/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    # Stop through the existing public owner; never open a competing camera.
    client.click('push button','停采并恢复原参数')
    stopped(op)

def stopped(op):
    require((op.RET/'CURRENT_UI.json').is_file(),'current parameters must be retained')
    end=time.monotonic()+40
    while True:
        h=read(op.SESSION/'heartbeat.json')
        require(h['pid']==op.PID and 0<=time.time_ns()-h['unix_ns']<3e9,'fresh stop publication required')
        if not any(h.get(k) for k in ('live_camera','busy','camera_reserved')):
            h=op.owner_state(True);break
        require(time.monotonic()<end,'normal camera release timed out');time.sleep(.2)
    write(op.RET/'STOPPED.json',h);require(h['hardware_fault'] is None,'normal stop restoration failed')
    print('Current main/test parameters and caches retained; zero verified; CCD released.',flush=True)

def audit_close():
    """Read-only completion audit after the old bounded waiter timed out."""
    op=setup();out=ROOT/'CLOSE_01'
    require(not Path('/proc',str(op.PID)).exists(),'old process still present')
    require(read(out/'YES_CLICK.json')['executed'],'confirmed normal close action required')
    result=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',OLD_UNIT,
        '-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True).splitlines())
    require(result==dict(MainPID='0',Result='success',ExecMainStatus='0'),'normal service exit not proven')
    exits=[r for line in (op.SESSION/'events.jsonl').read_text().splitlines()
           if (r:=json.loads(line)).get('state')=='operator_confirmed_light_blocked_exit' and r.get('closed_safely')]
    require(len(exits)==1,'unique safe exit event required')
    write(out/'CLOSED.json',dict(normal_exit=True,service=result,event=exits[0],forced_kill=False,
        physical_shielding_confirmed=True,original_wait_timed_out=True,read_only_completion_audit=True))
    print('Read-only normal exit audit PASS.',flush=True)

def configure():
    op=setup();require(read(ROOT/'LIFECYCLE_RESULT.json')['completed'],'normal close / cold check required')
    r=read(op.RET/'CURRENT_UI.json');h=r['heartbeat'];controls=r['main_controls'];tests=r['test_controls']
    def values(name,rows=controls):return [v['value'] for v in rows if v['role']=='spin button' and v['name']==name]
    def one(name):
        rows=values(name);require(len(rows)==1,'unique retained value '+name);return rows[0]
    def checked(prefix):
        rows=[v for v in controls if v['role']=='check box' and v['name'].startswith(prefix)]
        require(len(rows)==1,'unique retained checkbox '+prefix);return rows[0]['checked']
    combos=[v['name'] for v in controls if v['role']=='combo box']
    tc=[v['name'] for v in tests if v['role']=='combo box']
    exposure=one('ROI与曝光');gain=one('相机Gain请求值dB')
    require(gain==0.,'nonzero Gain needs separate startup restoration')
    display=values('CCD有效帧滤波与显示 · 滤波共用，显示映射另计')
    stable=values('输入激励与采样',tests);require(len(display)==4 and len(stable)==3,'control layout differs')
    filt=r['events']['shared_effective_filter_controls']['controls']
    modes={'原始单帧（不滤波）':'off','N帧均值（独立有效帧）':'mean','指数滤波（独立N帧窗）':'ema'}
    mode=[code for title,code in modes.items() if title in combos];require(mode==[filt['mode']],'filter event/current UI differ')
    require(filt['frames']==one('有效帧RAW数N') and filt['alpha']==display[3] and filt['original']==checked('原单帧对照'),'filter values differ')
    mode=[code for title,code in {'手动显示增益':'fixed','原始ADU':'raw','自动对比（原手动预览）':'auto'}.items() if title in combos]
    require(len(mode)==1,'display mode ambiguous')
    state=dict(schema='optical-lab-manual-display-state-1',manual=dict(mode=mode[0],gain=display[0],black=display[1],gamma=display[2]),
        preview_filter={k:filt[k] for k in ('mode','alpha','frames','original')},current_visible_controls=ref(op.RET/'CURRENT_UI.json'),
        camera_gain_separate_actual_dB=float(h['camera_active_readback']['Gain']),display_only=True,hardware_authorization=False)
    write(ROOT/'DISPLAY_CURRENT.json',state)
    cfg=read(ROOT/'PREPARED_03/CONFIG.json');registration=r['snapshot']['geometry']
    require(model.checked(r['geometry']) and read(r['geometry']['path'])==registration and h['geometry_frozen'],'current geometry missing')
    scale=[v for v in values('DMD · 1920×1080输出 / 方图等比缩放') if 1<=v<=1.8]
    require(len(scale)==1 and round(scale[0]*600)==registration.get('dmd_side',600),'DMD scale differs from current geometry')
    cfg['catalog_preset']=r['events']['RAW_validation_catalog_loaded']['reference'];model.checked(cfg['catalog_preset'])
    require(any(v['name']==cfg['head_preset']['path'] for v in controls),'retained head changed')
    restart=read(model.checked(cfg['restart']));restart.update(original_pid=op.PID,original_owner_start_ticks=op.START,
        original_session=str(op.SESSION),runtime_path=str(op.SESSION/'runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],optical_config=h['optical_reference'],
        requested_exposure_us=exposure,fault=None,registration=registration,expected_active=h['camera_active_readback'],
        no_auto_capture=True,no_auto_projection=True,old_process_clean_rollback_claimed=True)
    reference_folder=Path(registration['reference_folder'])
    for item in (op.RET/'CURRENT_UI.json',op.RET/'SNAPSHOT.json',Path(r['geometry']['path']),
                 reference_folder/'capture.json',reference_folder/'model_reference.json'):
        restart['evidence_sha256'][str(item)]=sha(item)
    write(ROOT/'restart_contract.json',restart)
    recovery=read(model.checked(cfg['recovery']));recovery.update(old_pid=op.PID,baseline_source=str(ROOT/'restart_contract.json'),
        baseline_source_sha256=sha(ROOT/'restart_contract.json'),cold_baseline_readonly_verification_passed=True,
        planned_normal_restart=True,old_process_clean_rollback_claimed=True)
    write(ROOT/'recovery_record.json',recovery)
    regions=[int(v.split('区')[0]) for v in combos if '区 → 2输出' in v];require(len(regions)==1,'readout count ambiguous')
    require('位置/速度清单轮测' in tc and '输入 CENTER' in tc and '输入 X_POS12' in tc,'test selections changed')
    cfg.update(restart=ref(ROOT/'restart_contract.json'),recovery=ref(ROOT/'recovery_record.json'),geometry=None,
        display=ref(ROOT/'DISPLAY_CURRENT.json'),caches=ref(op.RET/'PREVIEW_CACHES.json'),expected_active=h['camera_active_readback'],
        lifecycle_release=ref(ROOT/'LIFECYCLE_RESULT.json'),offline_preparation_only=False,shared_filter_frames=int(filt['frames']),
        replaced_owner_pid=op.PID,camera_requested=dict(exposure_us=exposure,gain_db=gain),retained_dmd_scale=scale[0],
        retained_region_count=regions[0],retained_region_overlay=checked('显示读出区域'),
        retained_pending_ball_radius=int(one('球扩张半径R（96图像素）')),
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],filter=state['preview_filter']),
        retained_view_choices=dict(enhance=checked('增强实测预览'),fixed_target=any(v.startswith('对齐：') for v in combos),
            overlay=any(v.startswith('叠加：') for v in combos),result_domain='processed' if any(v.startswith('主界面固定显示映射') for v in tc) else 'raw'),
        retained_test_controls=dict(mode='sweep',selected_ids=dict(input_a='CENTER',input_b='X_POS12'),head_enabled=checked('计算对应分区线性头')))
    cfg['settings'].update(software_gain=display[0],black_ADU=display[1],gamma=display[2],frames=int(filt['frames']),
        repeats=int(stable[0]),settle_seconds=stable[1],interval_seconds=stable[2],switch_inputs=True)
    for item in (ROOT/'run.py',ROOT/'panel.py',ROOT/'model.py',Path(__file__),ROOT/'LIFECYCLE_RESULT.json',op.RET/'CURRENT_UI.json'):
        cfg['source_sha256'][str(item)]=sha(item)
    contract=read(model.checked(cfg['contract']));contract.update(source_sha256=cfg['source_sha256'],expected_active=cfg['expected_active'],
        control_release_approved=False,dmd_scale_feature=dict(default=scale[0],maximum=1.8,control_qualified=False))
    write(ROOT/'CONTRACT.json',contract);cfg['contract']=ref(ROOT/'CONTRACT.json');write(ROOT/'CONFIG.json',cfg)
    print(ref(ROOT/'CONFIG.json'),flush=True)

def launch():
    op=setup();cfg=read(ROOT/'CONFIG.json');require(read(ROOT/'QA_RETAINED_01/SMOKE.json')['passed'],'retained whole-window smoke required')
    adapter=op.load('e075_launch_guard',run.base_module().DISPLAY/'run.py');adapter.no_owner()
    adapter.install_display(read(model.checked(cfg['display_profile']))).read_roles()
    for p,digest in cfg['source_sha256'].items():require(sha(p)==digest,'source changed '+p)
    command=read(PRIOR/'LAUNCH_PLAN.json')['command'];changes={str(PRIOR/'run.py'):str(ROOT/'run.py'),
        str(PRIOR/'CONFIG.json'):str(ROOT/'CONFIG.json'),sha(PRIOR/'CONFIG.json'):sha(ROOT/'CONFIG.json'),
        '--unit='+OLD_UNIT:'--unit='+UNIT}
    require(all(command.count(k)==1 for k in changes),'explicit launch replacement differs')
    command=[changes.get(v,v) for v in command]
    write(ROOT/'LAUNCH_PLAN.json',dict(command=command,shielding_confirmed=True,no_auto_projection=True,lifecycle=cfg['lifecycle_release']))
    result=subprocess.run(command,text=True,capture_output=True,timeout=20)
    write(ROOT/'LAUNCH_RESULT.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    require(result.returncode==0,'launch failed; no automatic retry');print(result.stdout+result.stderr)

def verify():
    base=run.base_module();old=sys.modules.get('run');sys.modules['run']=base
    try:module=base.load('e106_startup_verify',PRIOR/'verify_startup.py')
    finally:sys.modules['run']=old
    module.ROOT=ROOT
    sys.argv=[str(module.__file__),'--light-blocked-confirmed'];module.main()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('retain','stopped','close','audit_close','cold','configure','launch','verify'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    require(a.light_blocked_confirmed,'fresh user shielding confirmation required')
    if a.action=='retain':retain()
    elif a.action=='stopped':stopped(setup())
    elif a.action=='audit_close':audit_close()
    elif a.action in ('configure','launch','verify'):globals()[a.action]()
    else:getattr(setup(),a.action)()
