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
OLD=LAB/'experiments/E048_stability_page_20261025_1';SESSION=OLD/'GUI'
HELPERS=LAB/'experiments/E047_region_overlay_20261025_1'
PID=1874284;START='10397061';UNIT='ballplate-optical-stability-20261025-e048.service'
PREP_SHA='bc3f6ebbe9d8ae1aab0fa207c582414ea947238ec4c4b904d1aaa99192701e11'
LIFE_SHA='65582546ac80c2aa858bbafa17eef4732dad8a847e59daf44b628f214ea77694'
RUN_SHA='cdc0fcefda83142d3eb2560d209ede55ca2a8ae19a8395d1eed60a86b7612d4e'
CONTRACT_SHA='ae6b08469e9ce0998917ec719d0858d4c58279fdf05a3f86f4b62081f5bf652e'
RETAIN_SHA='000c10327d84c25e3c58ee07e8e17719d73b807992197dd6421fb9587e4c1ada'
LAUNCH_SHA='3766b0d33cae4dd48b87302094e12f89b68b190b5551fd511f34fca1c72b4987'

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
    p.RUN=OLD/'run.py';p.RUN_SHA=RUN_SHA;p.CONTRACT=OLD/'CONTRACT_03/CONTRACT.json';p.CONTRACT_SHA=CONTRACT_SHA
    p.RETAIN=OLD/'RETENTION_01/RESULT.json';p.RETAIN_SHA=RETAIN_SHA
    p.RUNTIME_SHA='bdcab45b667516664372ac5b85197dba2387ae412afbb6a7c565efdc5e43830f'
    p.EXTENSION_SHA='9f89fff321ef4722e02a01c3c6fb2f804fb4d55a6d42065fc7d3e7c35a8efb53'
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
                return dense.validate(c,r,self_sha256=base.DENSE_SHA)
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
        write(out/'MANUAL_TAB.json',pub.manual_tab(ui,c))
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
