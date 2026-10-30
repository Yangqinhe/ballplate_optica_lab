"""Explicit shielded E047 -> E048 lifecycle. External helper, never GUI injection.

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
OLD=LAB/'experiments/E047_region_overlay_20261025_1';SESSION=OLD/'GUI'
PID=1856825;START='10067144';UNIT='ballplate-optical-native8-20261025-e047.service'
PREP_SHA='bc3f6ebbe9d8ae1aab0fa207c582414ea947238ec4c4b904d1aaa99192701e11'
LIFE_SHA='65582546ac80c2aa858bbafa17eef4732dad8a847e59daf44b628f214ea77694'
RUN_SHA='1331475d0ca4f2121ff0e5c10f7b76ecbcec7b96242f3d0fbdf44da0434d8ddf'
CONTRACT_SHA='a92ff5f8637106518f21aaab3cd21238aa54ff91666d7f8f75560aa94ecd211b'
RETAIN_SHA='c4b34d3c20ce2f3ac4c949166e5657eb6c50a2bf7be78ad3aeb6ff8a95449884'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def module(p,digest,name):
    require(p.is_absolute() and not p.is_symlink() and sha(p)==digest,'source changed: '+str(p))
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def prepare_engine():
    p=module(OLD/'prepare_restart.py',PREP_SHA,'E048_frozen_retention')
    p.SELF=SELF;p.ROOT=ROOT;p.OLD=OLD;p.SESSION=SESSION;p.PID=PID;p.START=START
    p.RUN=OLD/'run.py';p.RUN_SHA=RUN_SHA;p.CONTRACT=OLD/'CONTRACT_05/CONTRACT.json';p.CONTRACT_SHA=CONTRACT_SHA
    p.RETAIN=OLD/'RETENTION_01/RESULT.json';p.RETAIN_SHA=RETAIN_SHA
    p.RUNTIME_SHA='10243c97d4cf0f2fe6b538fb280e0cd84dfb472eb26f2c22d4d6f0b4bca992a0'
    p.EXTENSION_SHA='b2bdb26afd4da85d40d4a80982d56d98e0232de45720ec015f02b208ea115aaf'
    original=p.module
    def load(path,digest,name):
        m=original(path,digest,name)
        if path==p.RUN:
            dense=m.dense_module()
            def validate(c,r,*,self_sha256):
                require(self_sha256==RUN_SHA,'retained runner identity mismatch')
                return dense.validate(c,r,self_sha256=m.DENSE_SHA)
            m.validate=validate
        elif path==p.READER:
            m.OLD=OLD;m.SESSION=SESSION;m.SPOOL=OLD/'GUI_SPOOL';m.PID=PID;m.START=START
            m.CONTRACT=p.CONTRACT;m.CONTRACT_SHA=CONTRACT_SHA
        return m
    p.module=load
    return p

def lifecycle_engine():
    a=module(OLD/'lifecycle.py',LIFE_SHA,'E048_frozen_lifecycle_adapter')
    a.SELF=SELF;a.ROOT=ROOT;a.SESSION=SESSION;a.PID=PID;a.START=START
    a.LAUNCH=OLD/'LAUNCH_PLAN.json';a.LAUNCH_SHA='85d4262d2f27cf8d8f6814fa8ba2e2e34d8390978bc9ec5d12e21165bf07eed8'
    a.RETENTION=ROOT/'RETENTION_01/RESULT.json';a.RETENTION_SHA=sha(a.RETENTION)
    a.RETENTION_MANIFEST_SHA=sha(a.RETENTION.parent/'MANIFEST.json')
    h=a.helper();previous=h.owner_argv
    h.owner_argv=lambda argv,cwd:previous(argv,cwd) or ('--hardware' in argv and any(
        v.endswith('.py') and (Path(cwd)/v).resolve()==OLD/'run.py' for v in argv))
    checked=h.checked
    def checked_sources(*args):
        plan=checked(*args)
        for p,d in ((OLD/'prepare_restart.py',PREP_SHA),(OLD/'lifecycle.py',LIFE_SHA)):
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
    if a.action=='park':result=park(ROOT/'PARK_03')
    elif a.action=='retain':result=prepare_engine().prepare(ROOT/'RETENTION_01',sha(SELF))
    else:
        h=lifecycle_engine()
        if a.action=='close':result=h.close(ROOT/'RETENTION_01/RESULT.json',sha(ROOT/'RETENTION_01/RESULT.json'),sha(SELF),UNIT,ROOT/'CLOSE_01',execute=True,light_blocked_confirmed=True)
        else:result=h.verify(ROOT/'CLOSE_01/CLOSED.json',sha(ROOT/'CLOSE_01/CLOSED.json'),sha(SELF),ROOT/'VERIFY_01',execute=True,light_blocked_confirmed=True)
    print(json.dumps(result,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
