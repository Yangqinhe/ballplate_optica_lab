"""One explicitly shielded, normal lifecycle; never force-kills the lab GUI."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import core as c
import run

OLD_UNIT='ballplate-optical-surrogate-preview-20261030-e106.service'
UNIT='ballplate-optical-paired-control-20261030-e107.service'


def helpers():
    # Compose frozen public-control lifecycle utilities in THIS process only.
    # No code injection or monkey-patching inside the running GUI.
    model=c.load(c.PREVIEW/'model.py','e107_switch_model')
    old_run=sys.modules.get('run');old_model=sys.modules.get('model')
    sys.modules['model']=model
    proxy=NS(ROOT=c.ROOT,LAB=c.LAB,PRIOR=c.PREVIEW,base_module=run.base_module)
    sys.modules['run']=proxy
    try:lc=c.load(c.PREVIEW/'lifecycle.py','e107_public_lifecycle')
    finally:
        sys.modules['run']=old_run
        if old_model is None:sys.modules.pop('model',None)
        else:sys.modules['model']=old_model
    lc.ROOT=c.ROOT;lc.PRIOR=c.PREVIEW;lc.LAB=c.LAB;lc.run=proxy
    lc.OLD_UNIT=OLD_UNIT;lc.UNIT=UNIT
    return lc,lc.setup()


def inspect():
    lc,op=helpers();out=c.ROOT/'RETENTION_01';out.mkdir()
    h=op.owner_state(True);ui,client,native=op.clients();tree,A=op.nodes()
    controls=[]
    for node in tree:
        if node.get_role_name() not in ('spin button','check box','combo box','label','push button','page tab'):continue
        v=node.get_value_iface()
        controls.append(dict(role=node.get_role_name(),name=node.get_name(),value=v.get_current_value() if v else None,
            checked=node.get_state_set().contains(A.StateType.CHECKED),showing=node.get_state_set().contains(A.StateType.SHOWING)))
    events={};files={};raw=(op.SESSION/'events.jsonl').read_bytes()
    for line in raw.splitlines():
        row=json.loads(line);events[row.get('state')]=row
        if row.get('state')=='manual_file_loaded':files['slm' if row['SLM'] else 'dmd']={k:row[k] for k in ('path','sha256')}
    for v in files.values():c.checked(v)
    try:zeros=op.zeros(client)
    except Exception as e:zeros=dict(error=str(e),verified=False)
    c.write(out/'CURRENT_UI.json',dict(heartbeat=h,controls=controls,events=events,files=files,main=ui.main,zeros=zeros))
    c.write(out/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    c.write(out/'AUTHORIZATION.json',dict(physical_shielding_confirmed=True,user='已遮光，可以切换控制对比页',
        unix_ns=time.time_ns(),scope='normal retain/close, cold baseline check, new zero windows, shielded CCD-only check'))
    print(json.dumps(dict(pid=h['pid'],fault=h['hardware_fault'],zeros=zeros,controls=[v for v in controls if v['role']!='label']),ensure_ascii=False),flush=True)


def configure():
    lc,op=helpers();proof=c.ROOT/'LIFECYCLE_RESULT.json'
    if not c.read(proof)['completed']:raise ValueError('normal lifecycle required')
    r=c.read(c.ROOT/'RETENTION_01/CURRENT_UI.json');h=r['heartbeat'];rows=r['controls'];events=r['events']
    def values(name):return [v['value'] for v in rows if v['role']=='spin button' and v['name']==name]
    def one(name):
        a=values(name)
        if len(a)!=1:raise ValueError('ambiguous spin '+name+': '+str(a))
        return a[0]
    def check(prefix):
        a=[v for v in rows if v['role']=='check box' and v['name'].startswith(prefix)]
        if len(a)!=1:raise ValueError('ambiguous checkbox '+prefix)
        return a[0]['checked']
    combos=[v['name'] for v in rows if v['role']=='combo box']
    display=values('CCD有效帧滤波与显示 · 滤波共用，显示映射另计')
    if len(display)!=4:raise ValueError('display control layout changed')
    filt=events['shared_effective_filter_controls']['controls']
    if filt['frames']!=one('有效帧RAW数N') or filt['alpha']!=display[3]:raise ValueError('filter controls/events differ')
    mode=[code for title,code in {'手动显示增益':'fixed','原始ADU':'raw','自动对比（原手动预览）':'auto'}.items() if title in combos]
    if len(mode)!=1:raise ValueError('display mode ambiguous')
    state=dict(schema='optical-lab-manual-display-state-1',manual=dict(mode=mode[0],gain=display[0],black=display[1],gamma=display[2]),
        preview_filter={k:filt[k] for k in ('mode','frames','alpha','original')},display_only=True,hardware_authorization=False)
    c.write(c.ROOT/'DISPLAY_CURRENT.json',state)
    cfg=c.read(c.ROOT/'PREPARED_04/CONFIG.json');prior=c.read(c.PREVIEW/'CONFIG.json')
    oldrestart=c.read(c.checked(prior['restart']))
    # A stopped/faulted GUI invalidates its live geometry. Preserve the last
    # explicit locked reference only as PENDING; new CCD signature must restore it.
    geometry=deepcopy(oldrestart['registration'])
    changes=[v for k,v in events.items() if k in ('reference_locked','manual_geometry_locked','manual_reference_locked')]
    if changes:raise ValueError('new reference events require explicit inspection')
    if geometry!=c.read(c.PREVIEW/'MODEL_01/MODEL.json')['geometry']:raise ValueError('retained/model geometry differs')
    scales=[v for v in values('DMD · 1920×1080输出 / 方图等比缩放') if 1<=v<=1.8]
    if len(scales)!=1:raise ValueError('DMD scale ambiguous')
    scale=scales[0]
    if round(scale*600)!=geometry.get('dmd_side',600):raise ValueError('reference scale changed')
    exposure=one('ROI与曝光');gain=one('相机Gain请求值dB')
    if gain!=0:raise ValueError('nonzero Gain startup requires separate restoration')
    restart=deepcopy(oldrestart)
    restart.update(original_pid=op.PID,original_owner_start_ticks=op.START,original_session=str(op.SESSION),
        runtime_path=str(op.SESSION/'runtime_sources.json'),saved_unix_ns=time.time_ns(),settings=dict(h['manual_settings'],dmd='black',slm='zero'),
        roi=h['actual_roi'],optical_config=h['optical_reference'],requested_exposure_us=exposure,fault=None,
        registration=geometry,expected_active=h['camera_active_readback'],no_auto_capture=True,no_auto_projection=True,
        old_process_clean_rollback_claimed=True)
    restart['evidence_sha256'][str(c.ROOT/'RETENTION_01/CURRENT_UI.json')]=c.digest(c.ROOT/'RETENTION_01/CURRENT_UI.json')
    c.write(c.ROOT/'restart_contract.json',restart)
    recovery=c.read(c.checked(prior['recovery']));recovery.update(old_pid=op.PID,baseline_source=str(c.ROOT/'restart_contract.json'),
        baseline_source_sha256=c.digest(c.ROOT/'restart_contract.json'),cold_baseline_readonly_verification_passed=True,
        planned_normal_restart=True,old_process_clean_rollback_claimed=True)
    c.write(c.ROOT/'recovery_record.json',recovery)
    regions=[int(x.split('区')[0]) for x in combos if '区 → 2输出' in x]
    if len(regions)!=1:raise ValueError('readout selection ambiguous')
    stable=values('输入激励与采样')
    if len(stable)!=3:raise ValueError('test controls differ')
    test_mode=next((key for label,key in [('位置/速度清单轮测','sweep'),('固定输入重复','hold'),('A/B交替换图','pair')] if label in combos),None)
    if test_mode is None:raise ValueError('test mode changed: '+str(combos))
    selected=[v.removeprefix('输入 ') for v in combos if v.startswith('输入 ')]
    if len(selected)!=2:raise ValueError('test input selection differs')
    cfg.update(restart=c.ref(c.ROOT/'restart_contract.json'),recovery=c.ref(c.ROOT/'recovery_record.json'),geometry=None,
        display=c.ref(c.ROOT/'DISPLAY_CURRENT.json'),caches=c.ref(c.ROOT/'RETENTION_01/PREVIEW_CACHES.json'),
        expected_active=h['camera_active_readback'],lifecycle_release=c.ref(proof),offline_preparation_only=False,
        shared_filter_frames=int(filt['frames']),replaced_owner_pid=op.PID,camera_requested=dict(exposure_us=exposure,gain_db=gain),
        retained_dmd_scale=scale,retained_region_count=regions[0],retained_region_overlay=check('显示读出区域'),
        retained_pending_ball_radius=int(one('球扩张半径R（96图像素）')),
        retained_manual_controls=dict(settings=h['manual_settings'],orientation=h['simulation_display_orientation']['code'],filter=state['preview_filter']),
        retained_view_choices=dict(enhance=check('增强实测预览'),fixed_target=any(v.startswith('对齐：') for v in combos),
            overlay=any(v.startswith('叠加：') for v in combos),result_domain='processed' if any(v.startswith('主界面固定显示映射') for v in combos) else 'raw'),
        retained_test_controls=dict(mode=test_mode,selected_ids=dict(input_a=selected[0],input_b=selected[1]),head_enabled=check('计算对应分区线性头')),
        catalog_preset=events['RAW_validation_catalog_loaded']['reference'])
    c.checked(cfg['catalog_preset'])
    cfg['settings'].update(software_gain=display[0],black_ADU=display[1],gamma=display[2],frames=int(filt['frames']),
        repeats=int(stable[0]),settle_seconds=stable[1],interval_seconds=stable[2],switch_inputs=test_mode!='hold')
    for path in (Path(__file__),c.ROOT/'RETENTION_01/CURRENT_UI.json',proof):cfg['source_sha256'][str(path)]=c.digest(path)
    contract=c.read(c.checked(cfg['contract']));contract.update(source_sha256=cfg['source_sha256'],expected_active=cfg['expected_active'],control_release_approved=False)
    c.write(c.ROOT/'CONTRACT.json',contract);cfg['contract']=c.ref(c.ROOT/'CONTRACT.json');c.write(c.ROOT/'CONFIG.json',cfg)
    print('Current settings retained; pending geometry requires new camera signature.',flush=True)


def launch():
    lc,op=helpers();cfg=c.read(c.ROOT/'CONFIG.json')
    if not c.read(c.ROOT/'RETAINED_QA_01/QA.json')['passed']:raise ValueError('retained GUI test required')
    adapter=op.load('e075_launch_guard',run.base_module().DISPLAY/'run.py');adapter.no_owner()
    adapter.install_display(c.read(c.checked(cfg['display_profile']))).read_roles()
    for p,sha in cfg['source_sha256'].items():
        if c.digest(p)!=sha:raise ValueError('source changed '+p)
    command=c.read(c.PREVIEW/'LAUNCH_PLAN.json')['command']
    replacements={str(c.PREVIEW/'run.py'):str(c.ROOT/'run.py'),str(c.PREVIEW/'CONFIG.json'):str(c.ROOT/'CONFIG.json'),
        c.digest(c.PREVIEW/'CONFIG.json'):c.digest(c.ROOT/'CONFIG.json'),'--unit='+OLD_UNIT:'--unit='+UNIT}
    if any(command.count(k)!=1 for k in replacements):raise ValueError('launch command differs')
    command=[replacements.get(x,x) for x in command]
    c.write(c.ROOT/'LAUNCH_PLAN.json',dict(command=command,shielding_confirmed=True,no_auto_projection=True))
    r=subprocess.run(command,capture_output=True,text=True,timeout=20)
    c.write(c.ROOT/'LAUNCH_RESULT.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise ValueError(r.stderr)
    print(r.stdout+r.stderr,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inspect','close','cold','configure','launch','verify'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('fresh physical shielding confirmation required')
    if a.action in ('inspect','configure','launch'):globals()[a.action]()
    elif a.action=='verify':
        base=run.base_module();old=sys.modules['run'];sys.modules['run']=base
        try:v=c.load(c.EXP/'E081_readout_regions_20261028_1/verify_startup.py','e107_startup_verify')
        finally:sys.modules['run']=old
        v.ROOT=c.ROOT;sys.argv=[str(v.__file__),'--light-blocked-confirmed'];v.main()
    else:getattr(helpers()[1],a.action)()
