"""Explicit shielded E048 -> E050 lifecycle. External helper, never GUI injection.

Reuse frozen retention/normal-exit/read-only validation, adapting exact owner
identity only. Default is descriptive. Each action requires shielding flags.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from types import SimpleNamespace

SELF=Path(__file__).resolve();ROOT=SELF.parent;LAB=ROOT.parents[1]
OLD=LAB/'experiments/E050_all8_gui_20261026_1';SESSION=OLD/'GUI'
HELPERS=LAB/'experiments/E047_region_overlay_20261025_1'
PID=1884846;START='10570254';UNIT='ballplate-optical-all8-20261026-e050.service'
PREP_SHA='bc3f6ebbe9d8ae1aab0fa207c582414ea947238ec4c4b904d1aaa99192701e11'
LIFE_SHA='65582546ac80c2aa858bbafa17eef4732dad8a847e59daf44b628f214ea77694'
RUN_SHA='a517a8499a1b6937e3483af91100c495b44d3c5412ce4d86723391681b733c4e'
CONTRACT_SHA='f9d98c0d4f4e0f3d285b54cb3144052fd9989e6c4d0031e78223413768640bc7'
RETAIN_SHA='066810172ae9db8f8a51a498dff0152b569d98bdd826e82f0170c6935109e9d6'
LAUNCH_SHA='ffe3288a58da620b23e491742194aa9036dd98dba21b452d28179a46ce2adc6f'
PROFILE=ROOT/'CURRENT_UI.json'
PROFILE_SHA='3eb717bcaacb00d6724390205d7fc4a9e37011cefeef3988937790c860d719c4'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def module(p,digest,name):
    require(p.is_absolute() and not p.is_symlink() and sha(p)==digest,'source changed: '+str(p))
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def prepare_engine():
    p=module(HELPERS/'prepare_restart.py',PREP_SHA,'E050_frozen_retention')
    p.SELF=SELF;p.ROOT=ROOT;p.OLD=OLD;p.SESSION=SESSION;p.PID=PID;p.START=START
    p.RUN=OLD/'run.py';p.RUN_SHA=RUN_SHA;p.CONTRACT=OLD/'CONTRACT_01/CONTRACT.json';p.CONTRACT_SHA=CONTRACT_SHA
    p.RETAIN=OLD/'RETENTION_01/RESULT.json';p.RETAIN_SHA=RETAIN_SHA
    p.RUNTIME_SHA='d7d71a54b2839740709e02b5751d2bf865885a883b01b32fdb2cc96b4b7afb9e'
    p.EXTENSION_SHA='ca32870d59cfa0575ceb6b671fa6c75d6ab961436505439b79373577a8bd4672'
    require(sha(PROFILE)==PROFILE_SHA,'current operator settings changed')
    profile=json.loads(PROFILE.read_text())
    original_write=p.write
    def write_current(path,value):
        # Preserve the operator's NEW capture request; standby restoration
        # target is intentionally unchanged. Only new retention files change.
        if Path(path).name=='restart_contract.json':
            value=dict(value,requested_exposure_us=profile['controls']['exposure'])
        if Path(path).name=='EXPECTED_CAPTURE_CONTEXT.json':
            value=dict(value,requested_exposure_us=profile['controls']['exposure'],
                camera_gain_request_dB=profile['controls']['gain'],camera_gain_actual_dB=float(profile['active']['Gain']))
        return original_write(path,value)
    p.write=write_current
    def original_owner():
        proc=Path('/proc')/str(PID);stat=(proc/'stat').read_bytes();cmd=(proc/'cmdline').read_bytes()
        require(stat.decode().rsplit(')',1)[1].split()[19]==START,'original owner start differs')
        argv=[x.decode() for x in cmd.split(b'\0') if x]
        require(sha(OLD/'LAUNCH_PLAN.json')==LAUNCH_SHA,'launch plan changed')
        planned=json.loads((OLD/'LAUNCH_PLAN.json').read_text())['command']
        require(argv==planned[planned.index('--')+1:],'exact original E048 argv differs')
        return dict(pid=PID,start_ticks=START,session=str(SESSION),argv=argv,cwd=str((proc/'cwd').resolve())),cmd,stat
    p.original_owner=original_owner
    original=p.module
    def load(path,digest,name):
        m=original(path,digest,name)
        if path==p.RUN:
            base=m.base_module();dense=base.dense_module()
            def validate(c,r,*,self_sha256):
                require(self_sha256==RUN_SHA,'retained runner identity mismatch')
                result=dense.validate(c,r,self_sha256=base.DENSE_SHA)
                if r==dict(path=str(p.RETAIN),sha256=RETAIN_SHA):
                    prior=result['contract']['expected_active'];current=profile['active']
                    require({k for k in prior.keys()|current.keys() if prior.get(k)!=current.get(k)}<={'ExposureTime','Gain'},
                        'unexpected operator camera change outside exposure/Gain')
                    from copy import deepcopy
                    result=deepcopy(result);result['contract']['expected_active']=current
                    result['contract']['current_operator_profile']=dict(path=str(PROFILE),sha256=PROFILE_SHA)
                return result
            m.validate=validate
        elif path==p.READER:
            m.OLD=OLD;m.SESSION=SESSION;m.SPOOL=OLD/'GUI_SPOOL';m.PID=PID;m.START=START
            m.CONTRACT=p.CONTRACT;m.CONTRACT_SHA=CONTRACT_SHA
        return m
    p.module=load
    return p

def lifecycle_engine():
    a=module(HELPERS/'lifecycle.py',LIFE_SHA,'E050_frozen_lifecycle_adapter')
    a.SELF=SELF;a.ROOT=ROOT;a.SESSION=SESSION;a.PID=PID;a.START=START
    a.LAUNCH=OLD/'LAUNCH_PLAN.json';a.LAUNCH_SHA=LAUNCH_SHA
    a.RETENTION=ROOT/'RETENTION_01/RESULT.json';a.RETENTION_SHA=sha(a.RETENTION)
    a.RETENTION_MANIFEST_SHA=sha(a.RETENTION.parent/'MANIFEST.json')
    h=a.helper();previous=h.owner_argv
    h.owner_argv=lambda argv,cwd:previous(argv,cwd) or ('--hardware' in argv and any(
        v.endswith('.py') and (Path(cwd)/v).resolve()==OLD/'run.py' for v in argv))
    checked=h.checked
    def checked_sources(*args):
        plan=checked(*args)
        for p,d in ((HELPERS/'prepare_restart.py',PREP_SHA),(HELPERS/'lifecycle.py',LIFE_SHA)):
            h.pin(p,d,plan['source_sha256'])
        return plan
    h.checked=checked_sources
    # Qt's native QMessageBox is an AT-SPI alert on this desktop, not dialog.
    # Still require exact title, PID, visible Yes, X window, and transient parent.
    def dialog(client,main):
        from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
        app,A=accessible(PID)
        rows=[n for n in safe_nodes(app) if n.get_role_name() in ('alert','dialog')
              and n.get_name()==h.TITLE and n.get_process_id()==PID
              and n.get_state_set().contains(A.StateType.SHOWING)]
        require(len(rows)<=1,'ambiguous exit confirmation')
        if not rows:return None
        nodes=[n for n in safe_nodes(rows[0]) if n.get_role_name()=='push button'
               and n.get_name() in ('Yes','&Yes','是','是(&Y)') and n.get_process_id()==PID]
        tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
        windows=[client.window(re.search(r'0x[0-9a-fA-F]+',line)[0]) for line in tree.splitlines() if '"'+h.TITLE+'"' in line]
        windows=[w for w in windows if w['pid']==PID and w['viewable']]
        require(len(nodes)==len(windows)==1 and client.inside(windows[0]['rect'],main['rect']),'exit dialog ownership/geometry mismatch')
        props=subprocess.check_output(['xprop','-id',windows[0]['wid'],'WM_TRANSIENT_FOR'],text=True)
        parent=re.search(r'window id # (0x[0-9a-fA-F]+)',props)
        require(parent is not None and int(parent[1],16)==int(main['wid'],16),'exit dialog transient owner mismatch')
        return nodes[0],A,windows[0]
    h._dialog=dialog
    return h

def park(output):
    p=prepare_engine();p.original_owner();out=Path(output)
    require(out.parent==ROOT and not out.exists(),'new local output required');out.mkdir()
    import sys
    sys.path.insert(0,str(LAB/'src'))
    from optical_lab import legacy_gui
    c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy');legacy_gui._install_accessibility_compatibility();c.install()
    native=module(LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py',
        'a5c3c7acf9618422f364a602653e02536bb12a0c34497f18455eb18e33d4330d','E048_native')
    def state(allow_fault=False):
        p.original_owner();h=p.read(SESSION/'heartbeat.json')
        require(h['pid']==PID and 0<=time.time_ns()-h['unix_ns']<3_000_000_000,'stale original owner')
        return h
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    candidates=[c.window(re.search(r'0x[0-9a-fA-F]+',line)[0]) for line in tree.splitlines()
                if '"实时光路对齐 v19 ' in line]
    candidates=[r for r in candidates if r['pid']==PID and r['viewable']]
    require(len(candidates)==1 and c.inside(candidates[0]['rect'],native.ORDINARY),'one original control monitor window required')
    created=next(json.loads(line) for line in (SESSION/'events.jsonl').read_text().splitlines()
                 if json.loads(line).get('state')=='created')
    c.same_signals(created['video'],c.read_roles())
    write(out/'BEFORE.json',dict(heartbeat=state(),control=candidates[0],physical_shielding_confirmed=True))
    ui=SimpleNamespace(pid=PID,main=candidates[0],state=state)
    ui.guard=lambda black=False:state()
    ui.fresh=lambda:state().get('latest_frame',{}).get('age_seconds',100)<1
    write(out/'ZERO_CLICK.json',native.invoke(ui,'allzero',execute=True) if state()['projection_armed'] else dict(already_disarmed=True,no_click=True))
    end=time.monotonic()+10
    while state()['projection_armed'] or state()['last_verified_revision']!=state()['display_revision']:
        require(time.monotonic()<end,'zero ACK timeout');time.sleep(.1)
    pub=module(LAB/'experiments/E035_mapping_night_20261024_1/public_client.py',
        'da07e1b95f981ac82f544617b62933421c5c5e5eba27e1b6ee4b228112a9149c','E048_public')
    if state().get('current_page') == 0:
        write(out/'MANUAL_TAB.json',dict(already_manual_page=True,no_click=True,heartbeat=state()))
    else:
        # AT-SPI SELECTED is stale on this Qt build. One guarded native click,
        # then confirm the real page via fresh heartbeat; never retry a click.
        deps=pub._deps(c);app,A=deps.accessible(PID)
        tabs=[node for node in deps.native._runtime().nodes(app) if node.get_role_name()=='page tab' and node.get_name()=='手动对齐']
        require(len(tabs)==1,'unique manual tab required')
        result=pub._main_click(ui,tabs[0],A,'page tab',deps.native)
        deadline=time.monotonic()+5
        while state().get('current_page')!=0:
            require(time.monotonic()<deadline,'manual page heartbeat did not update');time.sleep(.1)
        write(out/'MANUAL_TAB.json',dict(action=result,heartbeat_page_verified=0))
    after=c.inspect(SESSION,parked=False)
    require(all(after['windows'][r]['actual_pixels']['all_RGB_zero'] for r in ('dmd','slm')),'physical pixels not zero')
    write(out/'AFTER.json',after)
    return dict(completed=True,zero=True,owner=PID)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',nargs='?',default='describe',choices=['describe','park','retain','close','verify'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--light-blocked-confirmed',action='store_true')
    a=parser.parse_args(argv)
    if a.action=='describe':print(json.dumps(dict(default_IO=0,owner=PID)));return 0
    require(a.execute and a.light_blocked_confirmed,'explicit current shielding and execute required')
    if a.action=='park':result=park(ROOT/'PARK_04')
    elif a.action=='retain':result=prepare_engine().prepare(ROOT/'RETENTION_01',sha(SELF))
    else:
        h=lifecycle_engine()
        if a.action=='close':result=h.close(ROOT/'RETENTION_01/RESULT.json',sha(ROOT/'RETENTION_01/RESULT.json'),sha(SELF),UNIT,ROOT/'CLOSE_01',execute=True,light_blocked_confirmed=True)
        else:result=h.verify(ROOT/'CLOSE_01/CLOSED.json',sha(ROOT/'CLOSE_01/CLOSED.json'),sha(SELF),ROOT/'VERIFY_01',execute=True,light_blocked_confirmed=True)
    print(json.dumps(result,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
