"""Normal shielded replacement using existing public lifecycle operations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import run
from run import c,ROOT,PRIOR,LAB

OLD_UNIT='ballplate-optical-paired-control-20261030-e107.service'
UNIT='ballplate-optical-paired-control-20261030-e108.service'


def helpers():
    model=c.load(c.PREVIEW/'model.py','e108_lifecycle_math')
    old_run=sys.modules.get('run');old_model=sys.modules.get('model')
    sys.modules['run']=run;sys.modules['model']=model
    try:lc=c.load(c.PREVIEW/'lifecycle.py','e108_lifecycle')
    finally:
        sys.modules['run']=old_run
        if old_model is None:sys.modules.pop('model',None)
        else:sys.modules['model']=old_model
    lc.OLD_UNIT=OLD_UNIT;lc.UNIT=UNIT
    return lc,lc.setup()


def retain():
    lc,op=helpers();out=ROOT/'RETENTION_01';out.mkdir()
    cfg=c.read(PRIOR/'CONFIG.json')
    for p,sha in cfg['source_sha256'].items():
        if c.digest(p)!=sha:raise ValueError('old source changed: '+p)
    def fresh(allow_fault=False):
        h=c.read(op.SESSION/'heartbeat.json')
        if h['pid']!=op.PID or not 0<=time.time_ns()-h['unix_ns']<3e9 or h['projection_armed'] or h['hardware_fault']:
            raise ValueError('fresh zero owner required')
        if Path('/proc',str(op.PID),'stat').read_text().rsplit(')',1)[1].split()[19]!=op.START:raise ValueError('owner changed')
        return h
    h=fresh();ui,client,n=op.clients();ui.state=fresh;tree,A=op.nodes();rows=[]
    for v in tree:
        if v.get_role_name() not in ('spin button','check box','combo box','label','push button','page tab'):continue
        q=v.get_value_iface()
        rows.append(dict(role=v.get_role_name(),name=v.get_name(),value=q.get_current_value() if q else None,
            checked=v.get_state_set().contains(A.StateType.CHECKED),showing=v.get_state_set().contains(A.StateType.SHOWING)))
    events={};files={}
    for line in (op.SESSION/'events.jsonl').read_text().splitlines():
        row=json.loads(line);events[row.get('state')]=row
        if row.get('state')=='manual_file_loaded':files['slm' if row['SLM'] else 'dmd']={k:row[k] for k in ('path','sha256')}
    for v in files.values():c.checked(v)
    c.write(out/'AUTHORIZATION.json',dict(user='已遮光，可以更新显示修正版',unix_ns=time.time_ns(),
        physical_shielding_confirmed=True,scope='normal close, retain settings, zero-window restart and CCD-only verification'))
    c.write(out/'CURRENT_UI.json',dict(heartbeat=h,controls=rows,events=events,files=files,zeros=op.zeros(client),main=ui.main))
    c.write(out/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
    matches=[v for v in tree if v.get_role_name()=='push button' and v.get_name()=='停采并恢复原参数' and v.get_state_set().contains(A.StateType.SHOWING)]
    if len(matches)!=1:raise ValueError('unique normal camera stop button required')
    c.write(out/'STOP_CLICK.json',n.click_visible_node(ui,matches[0],A,execute=True))
    lc.stopped(op)


def configure():
    lc,op=helpers();proof=ROOT/'LIFECYCLE_RESULT.json'
    if not c.read(proof)['completed']:raise ValueError('normal lifecycle incomplete')
    cfg=c.read(PRIOR/'CONFIG.json');r=c.read(ROOT/'RETENTION_01/CURRENT_UI.json');h=r['heartbeat']
    # This is a display-only replacement. Inherit only if the retained controls
    # still equal the already verified config, otherwise stop for explicit review.
    previous=c.read(PRIOR/'RETENTION_01/CURRENT_UI.json')
    def control_values(rows):
        ids=set(cfg['retained_test_controls']['selected_ids'].values())
        def name(v):
            label=v['name']
            return label.removeprefix('输入 ') if v['role']=='combo box' and label.removeprefix('输入 ') in ids else label
        return [(v['role'],name(v),v['value'],v['checked']) for v in rows if v['role'] in ('spin button','combo box','check box')]
    before=control_values(previous['controls']);now=control_values(r['controls'])
    # The new page contributes only radius, impulse and direction spinboxes.
    remaining=list(now)
    for item in before:
        hits=[v for v in remaining if v[:2]==item[:2] and v[3]==item[3] and
            (v[2]==item[2] or isinstance(v[2],(int,float)) and isinstance(item[2],(int,float)) and abs(v[2]-item[2])<1e-9)]
        if not hits:raise ValueError('main configuration changed: '+str(item))
        remaining.remove(hits[0])
    if any(v[0]!='spin button' for v in remaining) or len(remaining)!=3:raise ValueError('unexpected new controls')
    if not h['geometry_frozen'] or h['camera_active_readback']!=cfg['expected_active'] or h['manual_settings']!=cfg['retained_manual_controls']['settings']:
        raise ValueError('retained physical context differs')
    if r['files']!=c.read(c.checked(cfg['caches']))['files']:raise ValueError('cached file identity changed')
    if h['actual_roi']!=c.read(c.checked(cfg['restart']))['roi']:raise ValueError('ROI changed')
    restart=c.read(c.checked(cfg['restart']));restart.update(original_pid=op.PID,original_owner_start_ticks=op.START,
        original_session=str(op.SESSION),runtime_path=str(op.SESSION/'runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),fault=None,no_auto_capture=True,no_auto_projection=True)
    restart['evidence_sha256'][str(ROOT/'RETENTION_01/CURRENT_UI.json')]=c.digest(ROOT/'RETENTION_01/CURRENT_UI.json')
    c.write(ROOT/'restart_contract.json',restart)
    recovery=c.read(c.checked(cfg['recovery']));recovery.update(old_pid=op.PID,baseline_source=str(ROOT/'restart_contract.json'),
        baseline_source_sha256=c.digest(ROOT/'restart_contract.json'),cold_baseline_readonly_verification_passed=True,planned_normal_restart=True)
    c.write(ROOT/'recovery_record.json',recovery)
    cfg.update(restart=c.ref(ROOT/'restart_contract.json'),recovery=c.ref(ROOT/'recovery_record.json'),
        caches=c.ref(ROOT/'RETENTION_01/PREVIEW_CACHES.json'),lifecycle_release=c.ref(proof),replaced_owner_pid=op.PID)
    for path in (ROOT/'run.py',ROOT/'page.py',Path(__file__),proof,ROOT/'RETENTION_01/CURRENT_UI.json'):
        cfg['source_sha256'][str(path)]=c.digest(path)
    contract=c.read(c.checked(cfg['contract']));contract['source_sha256']=cfg['source_sha256']
    c.write(ROOT/'CONTRACT.json',contract);cfg['contract']=c.ref(ROOT/'CONTRACT.json');c.write(ROOT/'CONFIG.json',cfg)
    c.write(ROOT/'RETAINED_CHECK.json',dict(passed=True,original_controls=len(before),new_control_count=len(remaining),
        same_camera=True,same_geometry_pending=True,same_patterns=True,only_display_method_changed=True))
    print('All main controls, patterns, camera and pending geometry retained',flush=True)


def launch():
    cfg=c.read(ROOT/'CONFIG.json')
    if not c.read(ROOT/'WINDOW_QA_01/QA.json')['passed']:raise ValueError('whole window QA required')
    if c.digest(ROOT/'page.py')!=c.digest(PRIOR/'DISPLAY_FIX_01/page.py'):raise ValueError('tested candidate differs')
    base=run.base_module();base.load('e108_launch_guard',base.DISPLAY/'run.py').no_owner()
    for path,sha in cfg['source_sha256'].items():
        if c.digest(path)!=sha:raise ValueError('source changed '+path)
    command=c.read(PRIOR/'LAUNCH_PLAN.json')['command']
    changes={str(PRIOR/'run.py'):str(ROOT/'run.py'),str(PRIOR/'CONFIG.json'):str(ROOT/'CONFIG.json'),
        c.digest(PRIOR/'CONFIG.json'):c.digest(ROOT/'CONFIG.json'),'--unit='+OLD_UNIT:'--unit='+UNIT}
    if any(command.count(k)!=1 for k in changes):raise ValueError('launch command differs')
    command=[changes.get(x,x) for x in command];c.write(ROOT/'LAUNCH_PLAN.json',dict(command=command,shielding_confirmed=True,no_auto_projection=True))
    r=subprocess.run(command,capture_output=True,text=True,timeout=20)
    c.write(ROOT/'LAUNCH_RESULT.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise ValueError(r.stderr)
    print(r.stdout+r.stderr,flush=True)


def verify():
    base=run.base_module();old=sys.modules['run'];sys.modules['run']=base
    try:v=c.load(c.EXP/'E081_readout_regions_20261028_1/verify_startup.py','e108_verify')
    finally:sys.modules['run']=old
    v.ROOT=ROOT;sys.argv=[str(v.__file__),'--light-blocked-confirmed'];v.main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('retain','close','cold','configure','launch','verify'))
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('current shielding required')
    if a.action in ('retain','configure','launch','verify'):globals()[a.action]()
    else:getattr(helpers()[1],a.action)()
