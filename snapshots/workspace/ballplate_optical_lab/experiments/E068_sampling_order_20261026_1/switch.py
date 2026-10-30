"""Explicit shielded normal lifecycle, preserving original camera rollback.

No action on import. Frozen public UI helpers only, no live Python injection.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent; LAB=ROOT.parents[1]
OLD=LAB/'experiments/E054_state_validation_20261026_1'; SESSION=OLD/'GUI'
PID=1926754; START='11345103'; UNIT='ballplate-optical-state-validation-20261026-e054.service'
RET=ROOT/'RETENTION_01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def require(v,message):
    if not v:raise RuntimeError(message)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def owner():
    proc=Path('/proc')/str(PID)
    require((proc/'stat').read_text().rsplit(')',1)[1].split()[19]==START,'owner lifetime changed')
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    plan=read(OLD/'LAUNCH_PLAN.json')['command']
    require(argv==plan[plan.index('--')+1:],'owner argv differs')
    return dict(pid=PID,start_ticks=START,session=str(SESSION),argv=argv,cwd=str((proc/'cwd').resolve()))
def heartbeat(allow_fault=False):
    owner();h=read(SESSION/'heartbeat.json')
    require(h['pid']==PID and 0<=time.time_ns()-h['unix_ns']<3_000_000_000,'stale heartbeat')
    require(allow_fault or not h['hardware_fault'],'hardware fault')
    require(not any(h.get(k) for k in ('loop_active','deployment_busy','adaptation_active')),'busy workflow')
    status=SESSION/'stability_status.json'
    if status.exists():require(not read(status).get('active'),'stability test running')
    return h
def clients():
    from optical_lab import legacy_gui
    c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy');legacy_gui._install_accessibility_compatibility();c.install()
    load('E068_client_compat',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
    p=LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py'
    require(sha(p)=='a5c3c7acf9618422f364a602653e02536bb12a0c34497f18455eb18e33d4330d','native source changed')
    n=load('E053_native',p)
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    rows=[c.window(re.search(r'0x[0-9a-fA-F]+',line)[0]) for line in tree.splitlines() if '"实时光路对齐 v19 ' in line]
    rows=[r for r in rows if r['pid']==PID and r['viewable']]
    require(len(rows)==1 and c.inside(rows[0]['rect'],n.ORDINARY),'exact original control window required')
    ui=SimpleNamespace(pid=PID,main=rows[0],state=heartbeat)
    ui.guard=lambda black=False:heartbeat();ui.fresh=lambda:heartbeat()['latest_frame']['age_seconds']<1
    return ui,c,n
def park():
    out=ROOT/'PARK_01';require(not out.exists(),'fresh park output required');out.mkdir()
    ui,c,n=clients();write(out/'BEFORE.json',heartbeat())
    if heartbeat()['projection_armed']:write(out/'CLICK.json',n.invoke(ui,'allzero',execute=True))
    end=time.monotonic()+10
    while True:
        h=heartbeat()
        if not h['projection_armed'] and h['last_verified_revision']==h['display_revision']:break
        require(time.monotonic()<end,'allzero timeout');time.sleep(.1)
    result=c.inspect(SESSION,parked=False)
    require(all(result['windows'][r]['actual_pixels']['all_RGB_zero'] for r in ('dmd','slm')),'actual outputs not zero')
    write(out/'AFTER.json',result)
    return dict(allzero=True,owner=PID)
def retain():
    require(not RET.exists(),'fresh retention only')
    cfg=read(OLD/'CONFIG.json')
    require(sha(OLD/'CONFIG.json')=='e11c35bbecf8f7a17be0a29b0582e3d49beaba3da0f910ea8b72ca7cb41d5a66','old config changed')
    for p,d in cfg['source_sha256'].items():require(sha(p)==d,'frozen source changed: '+p)
    ui,c,n=clients();h=heartbeat();inspection=c.inspect(SESSION,parked=False)
    require(not h['projection_armed'] and all(inspection['windows'][r]['actual_pixels']['all_RGB_zero'] for r in ('dmd','slm')),'park before retention')
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    app,A=accessible(PID);controls=[]
    for node in safe_nodes(app):
        if node.get_state_set().contains(A.StateType.SHOWING) and node.get_role_name() in ('spin button','check box','combo box'):
            v=node.get_value_iface()
            import optical_ui_v19 as optical
            controls.append(dict(role=node.get_role_name(),name=node.get_name(),value=v.get_current_value() if v else None,
                                 rect=list(optical.extents(node,A)),
                                 checked=node.get_state_set().contains(A.StateType.CHECKED)))
    helper=load('E053_retention_math',LAB/'experiments/E045_dense_state_sampling_20261025_1/retention.py')
    events=(SESSION/'events.jsonl').read_bytes();fold=helper.fold_events(events)
    files={role:{k:fold[role]['event'][k] for k in ('path','sha256')} for role in ('dmd','slm')}
    for r in files.values():require(sha(r['path'])==r['sha256'],'cache changed')
    geometry=read(cfg['geometry']['path'])
    require(geometry['roi']==h['actual_roi'] and geometry['optical_config']==h['optical_reference'] and
            geometry['registration']['orientation']==h['simulation_display_orientation']['code'],'manual geometry differs; do not silently reuse')
    RET.mkdir();write(RET/'CURRENT_UI.json',dict(heartbeat=h,controls=controls,inspection=inspection,owner=owner()))
    with (RET/'owner_events.jsonl').open('xb') as f:f.write(events)
    write(RET/'DISPLAY_STATE.json',helper.display_state(fold,h['camera_active_readback']))
    write(RET/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    # Exposure/Gain requests are read from the visible stability page, not guessed from quantized readback.
    spins=sorted([r for r in controls if r['role']=='spin button' and r['name']=='固定参数与测试设计'],key=lambda r:(r['rect'][1],r['rect'][0]))
    names=('exposure','gain','software_gain','black','gamma','frames','repeats','settle','interval')
    require(h['current_page']==1 and len(spins)==9,'exact visible original stability grid required')
    require(spins[0]['rect'][1]==spins[1]['rect'][1] and
            len({r['rect'][1] for r in spins[2:5]})==1 and len({r['rect'][1] for r in spins[5:8]})==1,
            'original stability grid geometry differs')
    values=dict(zip(('stability_'+name for name in names),(r['value'] for r in spins)))
    def spin(name):return values[name]
    # Store controls first: if accessibility labels differ, no guessed retention is written.
    settings=dict(frames=int(spin('stability_frames')),repeats=int(spin('stability_repeats')),
        settle_seconds=spin('stability_settle'),interval_seconds=spin('stability_interval'),
        software_gain=spin('stability_software_gain'),black_ADU=spin('stability_black'),gamma=spin('stability_gamma'),maximum_seconds=300.)
    switch=[r for r in controls if r['role']=='check box' and r['name'].startswith('测试中换图')]
    require(len(switch)==1,'switch control missing');settings['switch_inputs']=switch[0]['checked']
    camera=dict(exposure_us=spin('stability_exposure'),gain_db=spin('stability_gain'))
    restart=read(cfg['restart']['path']);restart.update(original_pid=PID,original_owner_start_ticks=START,
        original_session=str(SESSION),runtime_path=str(SESSION/'runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],optical_config=h['optical_reference'],
        requested_exposure_us=camera['exposure_us'],fault=None,registration=None,expected_active=None,
        planned_restart_NOT_completed=True,old_process_clean_rollback_claimed=False)
    restart['evidence_sha256'].update({str(RET/'CURRENT_UI.json'):sha(RET/'CURRENT_UI.json')})
    write(RET/'restart_contract.json',restart)
    recovery=read(cfg['recovery']['path']);recovery.update(old_pid=PID,baseline_source=str(RET/'restart_contract.json'),
        baseline_source_sha256=sha(RET/'restart_contract.json'),original_fault=None,planned_normal_restart=True,
        cold_baseline_readonly_verification_passed=False,old_process_clean_rollback_claimed=False)
    write(RET/'recovery_record.json',recovery)
    combos=sorted([r for r in controls if r['role']=='combo box' and r['name'].startswith('输入 ')],key=lambda r:r['rect'][0])
    require(len(combos)==2,'exact input selectors required')
    catalog=LAB/'experiments/E067_three_slm_20261026_1/HARDWARE_BANK_01/CATALOG.json'
    require(sha(catalog)=='1b8a3de49a479ddc2b473422077b99dfbb30feb98e269def37c905ddef8c4605','catalog changed')
    data,validated=load('E068_catalog_retention',OLD/'catalog.py').validate(ref(catalog))
    assets=[dict(input_id=r['input_id'],**r['dmd']) for r in data['inputs']];input_sources={}
    for label,combo in zip(('A','B'),combos):
        hits=[r for r in assets if str(r['input_id'])==combo['name'][3:]]
        require(len(hits)==1,'selected input binding missing')
        input_sources[label]={k:hits[0][k] for k in ('path','sha256')}
    result=dict(original_owner=owner(),retained_current_active=h['camera_active_readback'],
        original_baseline_unchanged=True,settings=settings,camera_requested=camera,catalog_preset=ref(catalog),
        geometry=cfg['geometry'],input_sources=input_sources,source_sha256=cfg['source_sha256'])
    for key,name in [('restart','restart_contract.json'),('recovery','recovery_record.json'),('display','DISPLAY_STATE.json'),('caches','PREVIEW_CACHES.json')]:result[key]=ref(RET/name)
    write(RET/'RESULT.json',result)
    return dict(retained=True,settings=settings,camera=camera)
def lifecycle():
    p=LAB/'experiments/E045_dense_state_sampling_20261025_1/lifecycle.py'
    require(sha(p)=='db6431f676fb6b2bf4b05be6586c1d614d6b25fac666ca92cd5e4b36894de19b','lifecycle changed')
    m=load('E053_frozen_lifecycle',p);m.SELF=Path(__file__).resolve();m.ROOT=ROOT;m.PID=PID;m.START=START;m.SESSION=SESSION
    prior=m.owner_argv
    m.owner_argv=lambda argv,cwd:prior(argv,cwd) or ('--hardware' in argv and any(str(LAB).encode() in x.encode() for x in argv))
    def checked(retention_path,retention_sha256,source_sha256,unit):
        require(Path(retention_path)==RET/'RESULT.json' and sha(retention_path)==retention_sha256 and sha(__file__)==source_sha256 and unit==UNIT,'lifecycle identity mismatch')
        r=read(retention_path);sources=dict(r['source_sha256'])
        from optical_lab import legacy_gui
        sources.update(legacy_gui.dependency_sources(LAB.parent/'ballplate_fourframe96_deploy'))
        for f in [Path(__file__).resolve(),p,RET/'RESULT.json',OLD/'LAUNCH_PLAN.json',m.NATIVE,m.RECHECK]:sources[str(f)]=sha(f)
        for key in ('restart','recovery','display','geometry','caches'):sources[r[key]['path']]=r[key]['sha256']
        for f,d in sources.items():require(sha(f)==d,'lifecycle pinned source changed: '+f)
        baseline=read(r['recovery']['path'])['camera_baseline']
        require(baseline==read(r['restart']['path'])['original_fault_before'] and baseline['TriggerMode']=='Off','baseline changed')
        return dict(retention_ref=ref(retention_path),retention=r,standby_baseline=baseline,
            old_owner=dict(pid=PID,start_ticks=START,session=str(SESSION),unit=UNIT),source_sha256=sources,
            physical_shielding_confirmed=True,forced_kill=False)
    m.checked=checked
    prior_clients=m._clients
    def frozen_clients(plan):
        result=prior_clients(plan)
        load('E068_lifecycle_compat',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
        return result
    m._clients=frozen_clients
    def dialog(c,main):
        from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
        app,A=accessible(PID)
        rows=[v for v in safe_nodes(app) if v.get_role_name() in ('dialog','alert') and v.get_name()==m.TITLE and v.get_state_set().contains(A.StateType.SHOWING)]
        require(len(rows)<=1,'ambiguous exit dialog')
        if not rows:return None
        nodes=[v for v in safe_nodes(rows[0]) if v.get_role_name()=='push button' and v.get_name() in ('Yes','&Yes','是','是(&Y)')]
        tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
        windows=[c.window(re.search(r'0x[0-9a-fA-F]+',line)[0]) for line in tree.splitlines() if '"'+m.TITLE+'"' in line]
        windows=[w for w in windows if w['pid']==PID and w['viewable']]
        require(len(nodes)==len(windows)==1 and c.inside(windows[0]['rect'],main['rect']),'exit dialog owner/geometry mismatch')
        props=subprocess.check_output(['xprop','-id',windows[0]['wid'],'WM_TRANSIENT_FOR'],text=True)
        parent=re.search(r'window id # (0x[0-9a-fA-F]+)',props)
        require(parent and int(parent[1],16)==int(main['wid'],16),'exit dialog parent mismatch')
        return nodes[0],A,windows[0]
    m._dialog=dialog
    return m
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('park','retain','close','verify','configure'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    require(a.light_blocked_confirmed,'current physical shielding required')
    if a.action=='park':r=park()
    elif a.action=='retain':r=retain()
    elif a.action=='close':r=lifecycle().close(RET/'RESULT.json',sha(RET/'RESULT.json'),sha(__file__),UNIT,ROOT/'CLOSE_01',execute=True,light_blocked_confirmed=True)
    elif a.action=='verify':r=lifecycle().verify(ROOT/'CLOSE_01/CLOSED.json',sha(ROOT/'CLOSE_01/CLOSED.json'),sha(__file__),ROOT/'VERIFY_01',execute=True,light_blocked_confirmed=True)
    else:
        release=ROOT/'VERIFY_01/LIFECYCLE_RESULT.json';proof=read(release)
        require(proof['completed'] and proof['cold_baseline_verified'] and not Path('/proc/1926754').exists(),'verified normal exit required')
        cfg=read(OLD/'CONFIG.json');retained=read(RET/'RESULT.json')
        for key in ('restart','recovery','display','geometry','caches','settings','camera_requested','input_sources','catalog_preset'):cfg[key]=retained[key]
        cfg['expected_active']=retained['retained_current_active'];cfg['lifecycle_release']=ref(release)
        sources=dict(cfg['source_sha256'])
        for p in (ROOT/'run.py',ROOT/'launch.py',ROOT/'adapter.py',ROOT/'test_order.py',OLD/'catalog.py',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py',Path(__file__).resolve(),RET/'RESULT.json',release):sources[str(p)]=sha(p)
        cfg['source_sha256']=sources
        contract=read(cfg['contract']['path']);contract['source_sha256']=sources
        catalog=read(cfg['catalog_preset']['path'])
        contract['allowed_dmd']=[dict(input_id=r['input_id'],**r['dmd']) for r in catalog['inputs']]
        contract['sampling_order_update']=dict(adapter=ref(ROOT/'adapter.py'),lease_limit_ms=300,root_cause_proven=False)
        contract['research_test']['camera_requested']=cfg['camera_requested']
        contract['expected_active']=cfg['expected_active']
        contract['raw_workbench']=dict(readout='RAW_mean_native8',control_qualified=False,no_old_head=True)
        write(ROOT/'CONTRACT.json',contract);cfg['contract']=ref(ROOT/'CONTRACT.json')
        write(ROOT/'CONFIG.json',cfg);r=dict(config=ref(ROOT/'CONFIG.json'),no_auto_projection=True,no_auto_capture=True)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
