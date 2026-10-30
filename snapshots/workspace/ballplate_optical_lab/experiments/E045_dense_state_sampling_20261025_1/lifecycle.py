"""Explicit shielded normal old-owner release/exit, then cold readonly check.

Default import/describe is stdlib-only and performs no reads or actions.
No forced termination, SDK acquisition, parameter writes, retry clicks or
output-window creation. Closing and cold checking are separate commands.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace


SELF=Path(__file__).resolve();ROOT=SELF.parent;LAB=ROOT.parents[1]
LEGACY=LAB.parent/'ballplate_fourframe96_deploy'
SESSION=LAB/'experiments/E040_temporal_preview_20261025_1/GUI'
PID=1390768;START='6954576';PROC=Path('/proc')
LAUNCH=SESSION.parent/'LAUNCH_PLAN.json'
LAUNCH_SHA='a129d48661188dabe8db9373bf7a94efb1e231ebf43fd76f774eedfffc928df2'
NATIVE=LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py'
NATIVE_SHA='a5c3c7acf9618422f364a602653e02536bb12a0c34497f18455eb18e33d4330d'
LEGACY_GUI=LAB/'src/optical_lab/legacy_gui.py'
LEGACY_GUI_SHA='cf018342e9907248320299c5ef73d3538b0adbde738da0e84f7c9290a4803748'
RECHECK=LEGACY/'live_manual_v17/camera_recheck.py'
RECHECK_SHA='61ea131a134b4e17f2a5deb1a3b3d51f7fcb578df45f8e23b63df3ea7d0a8a3f'
RETENTION_MANIFEST_SHA='dbf8c909e672d6cbf615d2cb213d39ab08d3ae2bc9f3309e252fecc75be49dce'
PURE_CLIENT_SOURCES={
    str(LAB/'src/optical_lab/protocol.py'):'2f651c945db41628181a93d7d77c61433c0ca8dce4ab0a79a41a13772618ab16',
    str(LAB/'src/optical_lab/__init__.py'):'522962c33fa454fe177a5a76aacd3e09e6cc5e96fe31850808e1bf5161988a7b'}
TITLE='退出前确认光源安全'


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def ref(path):return dict(path=str(path),sha256=sha(path))
def pin(path,digest,sources):
    path=Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve()==path
        and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None and sha(path)==digest,
        'explicit regular source/reference SHA differs: '+str(path))
    require(sources.setdefault(str(path),digest)==digest,'contradictory source binding')
    return path
def events(path):
    blob=Path(path).read_bytes();require(blob.endswith(b'\n'),'complete newline-terminated events required')
    return [json.loads(line) for line in blob.splitlines()]
def start_ticks(pid):return (PROC/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
def authorize(execute,blocked):require(execute is True and blocked is True,'explicit execute and current physical shielding required')


def owner_argv(argv,cwd):
    """Recognize old and future E045 entries, including relative -B scripts."""
    if '--hardware' not in argv:return False
    cwd=Path(cwd)
    if '--legacy-root' in argv:
        index=argv.index('--legacy-root')
        if index+1<len(argv) and (cwd/argv[index+1]).resolve()==LEGACY:return True
    if '-m' in argv:
        index=argv.index('-m')
        if index+1<len(argv) and (argv[index+1]=='optical_lab.preview'
            or argv[index+1].startswith('optical_lab.') and 'dense' in argv[index+1]):return True
    for argument in argv[1:]:
        if argument.startswith('-') or not argument.endswith('.py'):continue
        path=(cwd/argument).resolve()
        if path==ROOT/'run.py' or path.is_relative_to(LEGACY) and path.name=='direct_alignment_gui.py':return True
    return False


def owner_inventory():
    owners=[]
    for entry in PROC.iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            argv=[s.decode() for s in (entry/'cmdline').read_bytes().split(b'\0') if s]
            is_owner=owner_argv(argv,(entry/'cwd').resolve())
            if is_owner:owners.append(dict(pid=int(entry.name),start_ticks=start_ticks(int(entry.name)),argv=argv))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return sorted(owners,key=lambda r:r['pid'])


def no_owner():
    require(not (PROC/str(PID)).exists(),'old PID still exists or was reused; no cold SDK access')
    require(owner_inventory()==[],'another optical hardware owner is present; no second SDK owner')


def service_status(unit):
    text=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True)
    return dict(line.split('=',1) for line in text.splitlines() if '=' in line)


def checked(retention_path,retention_sha256,source_sha256,unit):
    sources={};pin(SELF,source_sha256,sources)
    test=ROOT/'test_lifecycle.py';pin(test,sha(test),sources)
    retention=read(pin(retention_path,retention_sha256,sources))
    require(retention.get('schema')=='E045-planned-retention-1'
        and retention.get('status')=='PLANNED_FILES_ONLY_NOT_LIFECYCLE_RELEASE'
        and retention.get('original_baseline_unchanged') is True
        and retention.get('original_owner',{}).get('pid')==PID
        and retention['original_owner'].get('start_ticks')==START
        and retention['original_owner'].get('session')==str(SESSION),'exact E040 planned retention required')
    contract=read(pin(retention['restart_contract']['path'],retention['restart_contract']['sha256'],sources))
    recovery=read(pin(retention['recovery_record']['path'],retention['recovery_record']['sha256'],sources))
    baseline=contract['original_fault_before']
    require(baseline==recovery['camera_baseline'] and baseline['DeviceSerialNumber']=='15328133'
        and baseline['TriggerMode']=='Off' and contract.get('original_pid')==PID
        and contract.get('original_owner_start_ticks')==START,'immutable original standby baseline required, not live capture photometry')
    archive=Path(retention_path).parent/'SOURCE_ARCHIVE_MANIFEST.json'
    artifact_manifest=read(pin(Path(retention_path).parent/'MANIFEST.json',RETENTION_MANIFEST_SHA,sources))
    for relative,digest in artifact_manifest['file_sha256'].items():pin(Path(retention_path).parent/relative,digest,sources)
    for path,digest in read(archive)['original_source_sha256'].items():pin(path,digest,sources)
    for path,digest in ((LAUNCH,LAUNCH_SHA),(NATIVE,NATIVE_SHA),(LEGACY_GUI,LEGACY_GUI_SHA),(RECHECK,RECHECK_SHA)):
        pin(path,digest,sources)
    for path,digest in PURE_CLIENT_SOURCES.items():pin(path,digest,sources)
    command=read(LAUNCH)['command'];units=[s[len('--unit='):] for s in command if s.startswith('--unit=')]
    require(len(units)==1 and unit==units[0] and re.fullmatch(r'[A-Za-z0-9_.@-]+\.service',unit) is not None,
        'CLI old-unit must exactly match the SHA-bound E040 launch command')
    return dict(retention_ref=dict(path=str(retention_path),sha256=retention_sha256),standby_baseline=baseline,
        old_owner=dict(pid=PID,start_ticks=START,session=str(SESSION),unit=unit),source_sha256=sources,
        retention=retention,physical_shielding_confirmed=True,forced_kill=False)


def heartbeat(plan,*,released=False):
    require(start_ticks(PID)==START,'original owner lifetime changed')
    h=read(SESSION/'heartbeat.json')
    require(h.get('pid')==PID and h.get('deployment_ui_revision')==23 and type(h.get('unix_ns')) is int
        and 0<=time.time_ns()-h['unix_ns']<3_000_000_000,'fresh complete original heartbeat required')
    require(h.get('projection_armed') is False and h.get('hardware_fault') is None
        and all(h.get(k) is False for k in ('loop_active','adaptation_active','deployment_busy')),
        'fault-free disarmed idle workflows required')
    if released:
        require(all(h.get(k) is False for k in ('busy','live_camera','camera_reserved')),'original camera not fully released')
    else:
        require(h.get('live_camera') is True and h.get('busy') is True and h.get('camera_reserved') is True,
            'this switch must start from the retained live CCD-only owner')
        require(h.get('camera_active_readback')==plan['retention']['retained_current_active'],'retained active camera profile changed')
    return h


def exact_report(report,baseline):
    require(report.get('error') is None and report.get('camera_settings_restored') is True
        and report.get('camera_before')==baseline and report.get('camera_after')==baseline
        and report.get('restore_differences')=={},'original normal release did not restore the complete immutable standby baseline')


def _clients(plan):
    sys.path.insert(0,str(LAB/'src'))
    from optical_lab import legacy_gui as old
    for path,digest in old.dependency_sources(LEGACY).items():
        require(plan['source_sha256'].get(path)==digest,'public helper dependency missing from retained source closure: '+path)
    c=old._load_clients(LEGACY);old._install_accessibility_compatibility();c.install()
    spec=importlib.util.spec_from_file_location('E045_lifecycle_native',NATIVE)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native.NAMES['release']='停采并恢复原参数'
    return old,c,native


def _dialog(c,main):
    from night_opt_20261020.manual_io_v17 import accessible,real_dialog,safe_nodes
    _,A=accessible(PID);dialogs=real_dialog(PID,TITLE,A)
    require(len(dialogs)<=1,'ambiguous original exit confirmation')
    if not dialogs:return None
    nodes=[n for n in safe_nodes(dialogs[0]) if n.get_role_name()=='push button'
        and n.get_name() in ('Yes','&Yes','是','是(&Y)')]
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    windows=[c.window(re.search(r'0x[0-9a-fA-F]+',line)[0]) for line in tree.splitlines() if '"'+TITLE+'"' in line]
    windows=[w for w in windows if w['pid']==PID and w['viewable']]
    require(len(nodes)==len(windows)==1 and windows[0]['parent']==int(main['wid'],16)
        and c.inside(windows[0]['rect'],main['rect']) and nodes[0].get_process_id()==PID,
        'unique PID-owned original Yes dialog inside main required')
    return nodes[0],A,windows[0]


def prove_closed(plan,out):
    no_owner();service=service_status(plan['old_owner']['unit'])
    require(service==dict(MainPID='0',Result='success',ExecMainStatus='0'),'normal systemd exit status not proven')
    rows=events(SESSION/'events.jsonl');exits=[r for r in rows[plan['exit_event_offset']:] if r.get('state')=='operator_confirmed_light_blocked_exit'
        and r.get('closed_safely') is True]
    require(len(exits)==1,'one new original closed_safely event required')
    reports=set(SESSION.glob('live_*/report.json'))-set(map(Path,plan['reports_before']))
    require(len(reports)==1,'one new original worker cleanup report required; no mtime selection')
    report_path=next(iter(reports));report=read(report_path);exact_report(report,plan['standby_baseline'])
    with (out/'events_after.jsonl').open('xb') as f:f.write((SESSION/'events.jsonl').read_bytes())
    result=dict(schema='E045-normal-old-owner-closed-1',completed=True,old_normal_exit_verified=True,
        physical_shielding_confirmed=True,forced_kill=False,old_owner=plan['old_owner'],retention_ref=plan['retention_ref'],
        standby_baseline=plan['standby_baseline'],cleanup_report=ref(report_path),exit_event=exits[0],service_status=service,
        events_snapshot=ref(out/'events_after.jsonl'),prospective_plan=ref(out/'CLOSE_PLAN.json'),source_sha256=plan['source_sha256'])
    write(out/'CLOSED.json',result);return result


def close(retention_path,retention_sha256,source_sha256,unit,output,*,execute=False,light_blocked_confirmed=False):
    authorize(execute,light_blocked_confirmed)
    plan=checked(retention_path,retention_sha256,source_sha256,unit);out=Path(output)
    require(out.is_absolute() and out.parent==ROOT and not out.exists() and not out.is_symlink(),'one NEW direct E045 close output required')
    require([(r['pid'],r['start_ticks']) for r in owner_inventory()]==[(PID,START)],'exact one original optical owner required')
    require(service_status(unit).get('MainPID')==str(PID),'old systemd unit does not own the exact retained PID')
    require((PROC/str(PID)/'cmdline').read_bytes().split(b'\0')[:-1]==[s.encode() for s in plan['retention']['original_owner']['argv']],
        'original owner command line changed')
    heartbeat(plan);plan.update(exit_event_offset=len(events(SESSION/'events.jsonl')),
        reports_before=[str(p) for p in sorted(SESSION.glob('live_*/report.json'))])
    out.mkdir();write(out/'CLOSE_PLAN.json',plan);actions=[]
    try:
        old,c,native=_clients(plan);before=c.inspect(SESSION,parked=False)
        require(before['pid']==PID,'public discovered wrong owner')
        write(out/'BEFORE_RELEASE.json',before)
        ui=SimpleNamespace(pid=PID,main=before['windows']['control'],state=lambda allow_fault=False:heartbeat(plan,released=False))
        actions.append(dict(action='release',result=native.invoke(ui,'release',execute=True)))
        end=time.monotonic()+25
        while True:
            try:released=heartbeat(plan,released=True)
            except ValueError as error:
                if str(error)!='original camera not fully released':raise
                if time.monotonic()>end:raise TimeoutError('normal camera release timeout; no repeated click')
                time.sleep(.1);continue
            break
        after=c.inspect(SESSION,parked=True);write(out/'RELEASED.json',after)
        reports=set(SESSION.glob('live_*/report.json'))-set(map(Path,plan['reports_before']))
        require(len(reports)==1,'one new worker cleanup report required before closing window')
        exact_report(read(next(iter(reports))),plan['standby_baseline'])
        ui.main=after['windows']['control'];ui.state=lambda allow_fault=False:heartbeat(plan,released=True)
        key=c.Input(int(ui.main['wid'],16))
        try:
            key.key(0xffe9,True);key.key(0xffc1,True);key.key(0xffc1,False);key.key(0xffe9,False)
        finally:key.close()
        actions.append(dict(action='Alt_F4',count=1));yes=False;end=time.monotonic()+20
        while (PROC/str(PID)).exists():
            require(start_ticks(PID)==START,'PID reused during normal exit')
            if time.monotonic()>end:raise TimeoutError('normal exit not finished; request manual normal exit, never force kill')
            if not yes:
                try:candidate=_dialog(c,ui.main)
                except Exception as error:
                    if not (PROC/str(PID)).exists() and (old._known_dead_accessible(error)
                        or type(error) is RuntimeError and str(error)=='one original accessible control owner required'):break
                    raise
                if candidate:
                    node,A,row=candidate
                    action=native.click_visible_node(ui,node,A,owner_client_wid=row['wid'],owner_rect=row['rect'],execute=True)
                    actions.append(dict(action='Yes',result=action));yes=True
            time.sleep(.05)
        write(out/'ACTIONS.json',actions)
        return prove_closed(plan,out)
    except BaseException as error:
        write(out/'CLOSE_FAILED.json',dict(completed=False,error=repr(error),actions=actions,forced_kill=False,
            physical_shielding_confirmed=True,manual_normal_exit_may_be_required=True))
        raise


def confirm_closed(close_plan_path,close_plan_sha256,source_sha256,*,execute=False,light_blocked_confirmed=False):
    """Adopt a user-assisted NORMAL exit; no input/SDK actions, no reclassification."""
    authorize(execute,light_blocked_confirmed);sources={};pin(SELF,source_sha256,sources)
    path=pin(close_plan_path,close_plan_sha256,sources);plan=read(path)
    require(plan['source_sha256'].get(str(SELF))==source_sha256,'same frozen close helper required')
    for filename,digest in plan['source_sha256'].items():pin(filename,digest,sources)
    return prove_closed(plan,path.parent)


def verify(closed_path,closed_sha256,source_sha256,output,*,execute=False,light_blocked_confirmed=False):
    authorize(execute,light_blocked_confirmed);sources={};pin(SELF,source_sha256,sources)
    closed=read(pin(closed_path,closed_sha256,sources));require(closed.get('schema')=='E045-normal-old-owner-closed-1'
        and closed.get('completed') is True and closed.get('old_normal_exit_verified') is True
        and closed.get('physical_shielding_confirmed') is True and closed.get('forced_kill') is False,
        'exact normal old exit evidence required before cold SDK access')
    require(closed['source_sha256'].get(str(SELF))==source_sha256,'same source-pinned lifecycle required')
    for filename,digest in closed['source_sha256'].items():pin(filename,digest,sources)
    plan=read(pin(closed['prospective_plan']['path'],closed['prospective_plan']['sha256'],sources))
    require(closed.get('old_owner')==plan.get('old_owner') and plan['old_owner'].get('pid')==PID
        and plan['old_owner'].get('start_ticks')==START and plan['old_owner'].get('session')==str(SESSION)
        and closed['retention_ref']==plan['retention_ref'] and closed['standby_baseline']==plan['standby_baseline'],
        'closed evidence must identify the exact retained lifetime, session and original baseline')
    baseline=closed['standby_baseline'];exact_report(read(pin(closed['cleanup_report']['path'],closed['cleanup_report']['sha256'],sources)),baseline)
    pin(closed['events_snapshot']['path'],closed['events_snapshot']['sha256'],sources)
    no_owner();require(service_status(plan['old_owner']['unit'])==closed['service_status'],'old service exit evidence changed')
    out=Path(output);require(out.is_absolute() and out.parent==ROOT and not out.exists() and not out.is_symlink(),'NEW direct E045 verification output required')
    out.mkdir();write(out/'VERIFY_PLAN.json',dict(closed_ref=dict(path=str(closed_path),sha256=closed_sha256),
        physical_shielding_confirmed=True,target=baseline,source_sha256=sources))
    sys.path.insert(0,str(LAB/'src'));from optical_lab import legacy_gui as old
    old._load_clients(LEGACY)
    from camera_recheck import ReadOnlyCamera,verify_baseline
    def guard():
        no_owner()
        for filename,digest in sources.items():pin(filename,digest,{})
    result=verify_baseline(baseline,guard,camera_factory=ReadOnlyCamera)
    write(out/'COLD_READONLY.json',result)
    passed=(result.get('passed') is True and result.get('connections')==2 and len(result.get('snapshots',[]))==10
        and result.get('parameter_writes')==0 and result.get('acquisition_started') is False and result.get('errors')==[]
        and all(r.get('values')==baseline and r.get('differences')=={} for r in result['snapshots']))
    final=dict(schema='E045-normal-shielded-release-1',completed=passed,physical_shielding_confirmed=True,
        old_normal_exit_verified=True,cold_baseline_verified=passed,old_owner={'pid':PID,'start_ticks':START},
        retention_ref=closed['retention_ref'],standby_baseline=baseline,forced_kill=False,
        camera_readonly_connections=result.get('connections'),camera_readonly_snapshots=len(result.get('snapshots',[])),
        source_sha256=sources,evidence_sha256={str(p):sha(p) for p in
            (Path(closed_path),Path(closed['prospective_plan']['path']),Path(closed['cleanup_report']['path']),
             Path(closed['events_snapshot']['path']),out/'VERIFY_PLAN.json',out/'COLD_READONLY.json')},
        errors=result.get('errors'),new_camera_acquisition_started=result.get('acquisition_started'),
        acquisition_started=result.get('acquisition_started'),parameter_writes=result.get('parameter_writes'),
        no_framebuffer_black_claim_during_owner_gap=True,no_new_output_windows=True)
    write(out/'LIFECYCLE_RESULT.json',final)
    return final


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=('describe','close','confirm-closed','verify'),default='describe')
    p.add_argument('--execute',action='store_true');p.add_argument('--light-blocked-confirmed',action='store_true')
    p.add_argument('--source-sha256');p.add_argument('--retention',type=Path);p.add_argument('--retention-sha256')
    p.add_argument('--old-unit');p.add_argument('--output',type=Path);p.add_argument('--closed',type=Path);p.add_argument('--closed-sha256')
    p.add_argument('--close-plan',type=Path);p.add_argument('--close-plan-sha256');a=p.parse_args(argv)
    auth=dict(execute=a.execute,light_blocked_confirmed=a.light_blocked_confirmed)
    if a.action=='close':value=close(a.retention,a.retention_sha256,a.source_sha256,a.old_unit,a.output,**auth)
    elif a.action=='confirm-closed':value=confirm_closed(a.close_plan,a.close_plan_sha256,a.source_sha256,**auth)
    elif a.action=='verify':value=verify(a.closed,a.closed_sha256,a.source_sha256,a.output,**auth)
    else:value=dict(execution=False,UI_actions=0,SDK_connections=0,forced_kill=False,
        requirement='explicit source-pinned close then verify, each requires current physical shielding')
    print(json.dumps(value,ensure_ascii=False));return int(a.action=='verify' and not value['completed'])


if __name__=='__main__':raise SystemExit(main())
