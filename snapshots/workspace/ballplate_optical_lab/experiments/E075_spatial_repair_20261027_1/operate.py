"""Explicit shielded lifecycle. Public UI only; no live code injection.

Import does not access devices. Retains current controls before normal exit;
strict original camera baseline is checked before a fresh single-owner launch.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from run import ROOT,LAB,PRIOR,read,write,sha,ref,load

SESSION=PRIOR/'GUI';PID=2638427;START='19254854'
OLD_UNIT='ballplate-optical-dmd1920-20261026-e074.service'
UNIT='ballplate-optical-spatial-repair-20261027-e075.service'
RET=ROOT/'RETENTION_01'

def require(value,message):
    if not value:raise RuntimeError(message)

def owner_state(allow_fault=False):
    proc=Path('/proc',str(PID))
    require((proc/'stat').read_text().rsplit(')',1)[1].split()[19]==START,'old owner changed')
    cmd=read(PRIOR/'LAUNCH_PLAN.json')['command']
    require([x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==cmd[cmd.index('--')+1:],'owner command changed')
    h=read(SESSION/'heartbeat.json')
    require(h['pid']==PID and 0<=time.time_ns()-h['unix_ns']<3e9,'old heartbeat stale')
    require(all(h.get(k) is False for k in ('live_camera','projection_armed','loop_active','adaptation_active','deployment_busy')),'stopped disarmed owner required')
    require(allow_fault or not h['busy'] and not h['camera_reserved'],'released owner required')
    require(allow_fault or not h['hardware_fault'],'original fault not cleared by recheck')
    return h

def clients():
    cfg=read(PRIOR/'CONFIG.json')
    adapter=load('e075_display_profile',PRIOR/'run.py')
    if 'display_linux' not in sys.modules:adapter.install_display(read(cfg['display_profile']['path']))
    from optical_lab import legacy_gui
    c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy')
    legacy_gui._install_accessibility_compatibility();c.install()
    load('e075_client_compat',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
    n=load('e075_native',LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py')
    n.ORDINARY=[1080,1080,2560,1440]
    candidates=[]
    for line in subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines():
        if '"光路工作台 · 八区稳定性' in line:
            row=c.window(re.search(r'0x[0-9a-fA-F]+',line)[0])
            if row['pid']==PID and row['viewable']:candidates.append(row)
    require(len(candidates)==1 and c.inside(candidates[0]['rect'],n.ORDINARY),'unique ordinary-screen main required')
    ui=SimpleNamespace(pid=PID,main=candidates[0],state=owner_state)
    return ui,c,n

def nodes():
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    app,A=accessible(PID);return list(safe_nodes(app)),A

def click(ui,n,name,dialog=None):
    tree,A=nodes()
    hits=[v for v in tree if v.get_role_name()=='push button' and v.get_name()==name and v.get_state_set().contains(A.StateType.SHOWING)]
    require(len(hits)==1,'unique visible button required: '+name)
    kw={} if dialog is None else dict(owner_client_wid=dialog['wid'],owner_rect=dialog['rect'])
    return n.click_visible_node(ui,hits[0],A,execute=True,**kw)

def zeros(c):
    from black_pixels import RootPixels
    profile=read(read(PRIOR/'CONFIG.json')['display_profile']['path']);c.same_signals(profile,c.read_roles())
    pixels=RootPixels()
    try:
        result={k:pixels.inspect(tuple(v['desktop_position']+v['desktop_size'])) for k,v in profile.items()}
        require(all(v['all_RGB_zero'] for v in result.values()),'actual output not zero')
        return result
    finally:pixels.close()

def retain():
    require(not RET.exists(),'fresh retention required')
    h=owner_state(True);ui,c,n=clients();z=zeros(c)
    cfg=read(PRIOR/'CONFIG.json')
    for p,d in cfg['source_sha256'].items():require(sha(p)==d,'source changed: '+p)
    tree,A=nodes();controls=[]
    for node in tree:
        if node.get_role_name() in ('spin button','check box','combo box','label','push button'):
            value=node.get_value_iface()
            controls.append(dict(role=node.get_role_name(),name=node.get_name(),
                value=value.get_current_value() if value else None,
                checked=node.get_state_set().contains(A.StateType.CHECKED),showing=node.get_state_set().contains(A.StateType.SHOWING)))
    events=(SESSION/'events.jsonl').read_bytes()
    helper=load('e075_retention_math',LAB/'experiments/E045_dense_state_sampling_20261025_1/retention.py')
    fold=helper.fold_events(events)
    files={role:{k:fold[role]['event'][k] for k in ('path','sha256')} for role in ('dmd','slm')}
    for record in files.values():require(sha(record['path'])==record['sha256'],'cache changed')
    RET.mkdir();write(RET/'CURRENT_UI.json',dict(heartbeat=h,controls=controls,zeros=z,main=ui.main))
    with (RET/'owner_events.jsonl').open('xb') as stream:stream.write(events)
    write(RET/'DISPLAY_STATE.json',helper.display_state(fold,h['camera_active_readback']))
    write(RET/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    print('Retained current controls, full sensor ROI, file caches and actual black outputs; old owner unchanged.')

def close():
    require((RET/'CURRENT_UI.json').exists(),'retain first')
    out=ROOT/'CLOSE_01';out.mkdir()
    ui,c,n=clients();write(out/'BEFORE.json',dict(heartbeat=owner_state(True),zeros=zeros(c)))
    write(out/'RECHECK_CLICK.json',click(ui,n,'复核并恢复（不投图）'))
    end=time.monotonic()+45
    while owner_state(True)['hardware_fault'] is not None:
        require(time.monotonic()<end,'original read-only recheck failed');time.sleep(.25)
    write(out/'RECHECK_DONE.json',dict(heartbeat=owner_state(),zeros=zeros(c)))
    tree,A=nodes()
    names=[v.get_name() for v in tree if v.get_role_name()=='push button' and '退出' in v.get_name() and v.get_state_set().contains(A.StateType.SHOWING)]
    require(len(names)==1,'unique normal exit button required')
    write(out/'CLOSE_CLICK.json',click(ui,n,names[0]))
    yes=False;end=time.monotonic()+30
    while Path('/proc',str(PID)).exists():
        require(time.monotonic()<end,'normal exit timeout; no force kill')
        if not yes:
            tree,A=nodes();dialogs=[v for v in tree if v.get_role_name() in ('dialog','alert') and v.get_state_set().contains(A.StateType.SHOWING)]
            if dialogs:
                require(len(dialogs)==1,'ambiguous dialog')
                title=dialogs[0].get_name();require('光' in title or '退出' in title,'unexpected confirmation')
                rows=[]
                for line in subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines():
                    if '"'+title+'"' in line:
                        row=c.window(re.search(r'0x[0-9a-fA-F]+',line)[0])
                        if row['pid']==PID and row['viewable']:rows.append(row)
                require(len(rows)==1 and c.inside(rows[0]['rect'],ui.main['rect']),'dialog geometry differs')
                props=subprocess.check_output(['xprop','-id',rows[0]['wid'],'WM_TRANSIENT_FOR'],text=True)
                match=re.search(r'window id # (0x[0-9a-fA-F]+)',props)
                require(match and int(match[1],16)==int(ui.main['wid'],16),'dialog parent differs')
                names=[v.get_name() for v in tree if v.get_role_name()=='push button' and v.get_name() in ('Yes','&Yes','是','是(&Y)') and v.get_state_set().contains(A.StateType.SHOWING)]
                require(len(names)==1,'unique affirmative button required')
                write(out/'YES_CLICK.json',click(ui,n,names[0],rows[0]));yes=True
        time.sleep(.2)
    values=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',OLD_UNIT,'-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True).splitlines())
    require(values==dict(MainPID='0',Result='success',ExecMainStatus='0'),'normal service exit not proven')
    exits=[jsonline for line in (SESSION/'events.jsonl').read_text().splitlines() if (jsonline:=__import__('json').loads(line)).get('state')=='operator_confirmed_light_blocked_exit' and jsonline.get('closed_safely') is True]
    require(len(exits)==1,'safe normal exit not proven')
    write(out/'CLOSED.json',dict(normal_exit=True,service=values,event=exits[0],forced_kill=False,physical_shielding_confirmed=True))
    print('Old owner exited normally; physical shielding must remain.')

def cold():
    require(read(ROOT/'CLOSE_01/CLOSED.json')['normal_exit'],'normal exit required')
    adapter=load('e075_no_owner',PRIOR/'run.py');adapter.no_owner()
    sys.path[:0]=[str(LAB.parent/'ballplate_fourframe96_deploy'),str(LAB.parent/'ballplate_fourframe96_deploy/live_manual_v17')]
    from camera_recheck import ReadOnlyCamera,verify_baseline
    baseline=read(read(PRIOR/'CONFIG.json')['recovery']['path'])['camera_baseline']
    result=verify_baseline(baseline,adapter.no_owner,camera_factory=ReadOnlyCamera)
    write(ROOT/'COLD_READONLY.json',result)
    passed=result.get('passed') is True and result.get('connections')==2 and len(result.get('snapshots',[]))==10 and result.get('parameter_writes')==0 and result.get('acquisition_started') is False and result.get('errors')==[] and all(r.get('values')==baseline and r.get('differences')=={} for r in result['snapshots'])
    write(ROOT/'LIFECYCLE_RESULT.json',dict(completed=passed,old_normal_exit_verified=True,cold_baseline_verified=passed,
        physical_shielding_confirmed=True,normal_exit=ref(ROOT/'CLOSE_01/CLOSED.json'),cold_readonly=ref(ROOT/'COLD_READONLY.json')))
    require(passed,'cold baseline mismatch; do not launch')
    print('Original standby baseline: 2 connections / 10 snapshots PASS.')

def configure():
    proof=ROOT/'LIFECYCLE_RESULT.json';require(read(proof)['completed'],'lifecycle must pass')
    require(not Path('/proc',str(PID)).exists(),'old owner still present')
    cfg=read(PRIOR/'CONFIG.json');ret=read(RET/'CURRENT_UI.json');h=ret['heartbeat'];controls=ret['controls']
    def values(name):return [v['value'] for v in controls if v['role']=='spin button' and v['name']==name]
    def checked(prefix):
        rows=[v for v in controls if v['role']=='check box' and v['name'].startswith(prefix)]
        require(len(rows)==1,'unique retained checkbox '+prefix);return rows[0]['checked']
    stable=values('固定参数与测试设计')
    require(len(stable)==11 and stable[0]==250. and stable[1]==0.,'unexpected retained test grid')
    require(values('ROI与曝光')==[250.] and values('相机Gain请求值dB')==[0.],'current photometry changed')
    display=values('CCD显示设置 · 不进入八区积分／电域网络');require(len(display)==4,'display controls ambiguous')
    state=read(RET/'DISPLAY_STATE.json')
    state.update(manual=dict(mode='fixed',gain=display[0],black=display[1],gamma=display[2]),
        preview_filter=dict(mode='ema',alpha=display[3],original=checked('原单帧对照')),
        current_visible_controls=ref(RET/'CURRENT_UI.json'),prior_events_are_stale=True)
    combos=[v['name'] for v in controls if v['role']=='combo box']
    require('手动显示增益' in combos and '指数平滑（预览采样帧）' in combos,'display mode differs')
    require(all(x in combos for x in ('CENTER','VY_POS','REF_SIN_SPARSE')) and values('球扩张半径R（96图像素）')==[2.],'input selections changed')
    write(ROOT/'DISPLAY_CURRENT.json',state)
    restart=read(cfg['restart']['path'])
    restart.update(original_pid=PID,original_owner_start_ticks=START,original_session=str(SESSION),runtime_path=str(SESSION/'runtime_sources.json'),
        saved_unix_ns=time.time_ns(),settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],
        optical_config=h['optical_reference'],requested_exposure_us=250.,fault=None,registration=None,expected_active=None,
        no_auto_capture=True,no_auto_projection=True,old_process_clean_rollback_claimed=True)
    restart['evidence_sha256'][str(RET/'CURRENT_UI.json')]=sha(RET/'CURRENT_UI.json')
    write(ROOT/'restart_contract.json',restart)
    recovery=read(cfg['recovery']['path']);recovery.update(old_pid=PID,baseline_source=str(ROOT/'restart_contract.json'),
        baseline_source_sha256=sha(ROOT/'restart_contract.json'),cold_baseline_readonly_verification_passed=True,
        planned_normal_restart=True,old_process_clean_rollback_claimed=True)
    write(ROOT/'recovery_record.json',recovery)
    cfg.update(restart=ref(ROOT/'restart_contract.json'),recovery=ref(ROOT/'recovery_record.json'),
        display=ref(ROOT/'DISPLAY_CURRENT.json'),caches=ref(RET/'PREVIEW_CACHES.json'),
        expected_active=h['camera_active_readback'],lifecycle_release=ref(proof),offline_preparation_only=False,replaced_owner_pid=PID,
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],filter=state['preview_filter']))
    # Current test widgets match the previous sealed selections; preserve, never run.
    cfg['retained_test_controls']['head_enabled']=False
    sources=dict(cfg['source_sha256'])
    for path in (ROOT/'run.py',Path(__file__),ROOT/'alignment.py',ROOT/'staged/optical_lab/display.py',proof,RET/'CURRENT_UI.json'):
        sources[str(path)]=sha(path)
    cfg['source_sha256']=sources
    contract=read(cfg['contract']['path']);contract.update(source_sha256=sources,expected_active=cfg['expected_active'],
        spatial_repair=dict(display_only_resampling=True,raw_processing_changed=False,actual_display_ack=True,
            calibration_clipping_rejected=True,hardware_verified=False),control_release_approved=False)
    write(ROOT/'CONTRACT.json',contract);cfg['contract']=ref(ROOT/'CONTRACT.json');write(ROOT/'CONFIG.json',cfg)
    print(ref(ROOT/'CONFIG.json'))

def launch():
    cfg=read(ROOT/'CONFIG.json');require(read(cfg['lifecycle_release']['path'])['completed'],'fresh lifecycle required')
    require(read(ROOT/'QA_02/RESULT.json')['size_smoke'],'fresh full-window smoke required')
    adapter=load('e075_launch_guard',PRIOR/'run.py');adapter.no_owner()
    adapter.install_display(read(cfg['display_profile']['path'])).read_roles()
    for path,digest in cfg['source_sha256'].items():require(sha(path)==digest,'source changed: '+path)
    command=read(PRIOR/'LAUNCH_PLAN.json')['command']
    changes={str(PRIOR/'run.py'):str(ROOT/'run.py'),str(PRIOR/'CONFIG.json'):str(ROOT/'CONFIG.json'),
        sha(PRIOR/'CONFIG.json'):sha(ROOT/'CONFIG.json'),'--unit='+OLD_UNIT:'--unit='+UNIT}
    require(all(command.count(k)==1 for k in changes),'old launch command differs')
    command=[changes.get(v,v) for v in command]
    write(ROOT/'LAUNCH_PLAN.json',dict(command=command,shielding_confirmed=True,no_auto_projection=True,lifecycle=cfg['lifecycle_release']))
    r=subprocess.run(command,text=True,capture_output=True,timeout=20)
    write(ROOT/'LAUNCH_RESULT.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    require(r.returncode==0,'launch failed; no automatic retry');print(r.stdout+r.stderr)

def verify():
    cfg=read(ROOT/'CONFIG.json');load('e075_verify_display',PRIOR/'run.py').install_display(read(cfg['display_profile']['path']))
    m=load('e075_verify_public',LAB/'experiments/E073_slm_selector_20261026_1/verify_startup.py')
    m.ROOT=ROOT;original=m.load
    def adapted(name,path):
        value=original(name,path)
        if name=='e071_native_click':value.ORDINARY=[1080,1080,2560,1440]
        return value
    m.load=adapted;sys.argv=[str(m.__file__),'--light-blocked-confirmed'];m.main()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('retain','close','cold','configure','launch','verify'))
    p.add_argument('--light-blocked-confirmed',action='store_true');args=p.parse_args()
    require(args.light_blocked_confirmed,'current shielding required')
    globals()[args.action]()
