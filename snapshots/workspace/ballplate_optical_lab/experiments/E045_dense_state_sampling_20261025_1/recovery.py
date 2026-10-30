"""Separate later-readonly recovery evidence; never reclassify failed cleanup.

Default is side-effect-free. Only an explicitly shielded command can normally
exit the already-released old GUI; cold SDK verification is a separate command.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

SELF=Path(__file__).resolve();ROOT=SELF.parent
LIFECYCLE=ROOT/'lifecycle.py'
LIFECYCLE_SHA='db6431f676fb6b2bf4b05be6586c1d614d6b25fac666ca92cd5e4b36894de19b'
EVIDENCE={
    'old_plan':(ROOT/'CLOSE_01/CLOSE_PLAN.json','07792566d984200d96a619f236cdea1085bca206ea48b7741c33f2b70c0d5999'),
    'old_failure':(ROOT/'CLOSE_01/CLOSE_FAILED.json','fbb08feb36b5375b26cdc7c64f00eed6156852685fdca499062451ad0b03f447'),
    'old_cleanup':(ROOT.parent/'E040_temporal_preview_20261025_1/GUI/live_20261025_140855_964900/report.json','6803fd7e21f81e523aae8b8ff03d601f5c3ef1733091c14a2c76c3de953ea7d8'),
    'later_recheck':(ROOT/'ORIGINAL_RECHECK_01/RESULT.json','968af534a7498d69d9f5abadf6bc61f397a4ff313b59d0e169c2106c30303fd7')}


def engine():
    if hashlib.sha256(LIFECYCLE.read_bytes()).hexdigest()!=LIFECYCLE_SHA:raise ValueError('frozen lifecycle SHA differs')
    spec=importlib.util.spec_from_file_location('E045_recovery_frozen_lifecycle',LIFECYCLE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def readonly_pass(b,report):
    return (report.get('passed') is True and report.get('target')==b and report.get('connections')==2
        and report.get('parameter_writes')==0 and report.get('acquisition_started') is False
        and report.get('errors')==[] and len(report.get('snapshots',[]))==10
        and [r.get('connection') for r in report['snapshots']]==[0]*5+[1]*5
        and all(r.get('values')==b and r.get('differences')=={} for r in report['snapshots']))


def preserved(m,plan):
    values={key:m.read(m.pin(path,digest,plan['source_sha256'])) for key,(path,digest) in EVIDENCE.items()}
    old,fail,cleanup,later=(values[k] for k in ('old_plan','old_failure','old_cleanup','later_recheck'))
    b=plan['standby_baseline'];report=later['report']
    m.require(all(old[k]==plan[k] for k in ('old_owner','retention_ref','standby_baseline')),
        'old failed attempt must identify the same original owner and immutable baseline')
    m.require(fail.get('completed') is False and fail.get('forced_kill') is False
        and len(fail.get('actions',[]))==1 and fail['actions'][0].get('action')=='release'
        and fail['actions'][0]['result'].get('click_count')==1 and fail['actions'][0]['result'].get('executed') is True,
        'preserved single-release failure required; no repeated release')
    m.require(cleanup.get('error') is None and cleanup.get('camera_settings_restored') is False
        and cleanup.get('camera_before')==b and cleanup.get('camera_after')==dict(b,AcquisitionFrameRate='90')
        and b.get('AcquisitionFrameRate')=='95'
        and cleanup.get('restore_differences')=={'AcquisitionFrameRate':{'before':'95','after':'90'}},
        'only the preserved original 95-to-90 cleanup failure is in scope')
    m.require(later.get('old_failure_reclassified') is False and readonly_pass(b,report)
        and report.get('original_failed_stage_accepted') is False and report.get('release_approved') is False,
        'later complete original-baseline readonly PASS must not accept the failed stage')
    m.require(m.read(m.pin(later['report_ref']['path'],later['report_ref']['sha256'],plan['source_sha256']))==report,
        'original GUI recheck report differs from its preserved copy')
    after=later['after'];h=after['heartbeat']
    m.require(after.get('pid')==m.PID and after.get('parked_verification_passed') is True
        and after.get('no_SDK_connection') is True and after.get('no_projection') is True
        and h.get('hardware_fault') is None and h.get('projection_armed') is False,
        'preserved post-recheck original owner must be fault-free and parked')
    checks=report.get('black_framebuffer_observations',[])
    m.require(checks and all(r.get(k,{}).get('all_RGB_zero') is True for r in checks for k in ('dmd','slm')),
        'original recheck must retain real two-panel allzero observations')
    return values


def checked(m,retention,retention_sha,source_sha,unit):
    plan=m.checked(retention,retention_sha,LIFECYCLE_SHA,unit)
    m.pin(SELF,source_sha,plan['source_sha256']);m.pin(ROOT/'test_recovery.py',m.sha(ROOT/'test_recovery.py'),plan['source_sha256'])
    preserved(m,plan);return plan


def close(retention,retention_sha,source_sha,unit,output,*,execute=False,light_blocked_confirmed=False):
    if not execute or not light_blocked_confirmed:raise ValueError('explicit execute and current physical shielding required')
    m=engine();plan=checked(m,retention,retention_sha,source_sha,unit);out=Path(output)
    m.require(out.is_absolute() and out.parent==ROOT and not out.exists(),'NEW direct E045 recovery close output required')
    m.require([(r['pid'],r['start_ticks']) for r in m.owner_inventory()]==[(m.PID,m.START)],'exact sole original owner required')
    m.require(m.service_status(unit).get('MainPID')==str(m.PID),'original service lifetime required')
    m.require((m.PROC/str(m.PID)/'cmdline').read_bytes().split(b'\0')[:-1]
        ==[s.encode() for s in plan['retention']['original_owner']['argv']],'original owner argv differs')
    m.heartbeat(plan,released=True);plan.update(exit_event_offset=len(m.events(m.SESSION/'events.jsonl')),
        reports_before=[str(p) for p in sorted(m.SESSION.glob('live_*/report.json'))])
    out.mkdir();m.write(out/'RECOVERY_CLOSE_PLAN.json',plan);actions=[]
    try:
        old,c,native=m._clients(plan);before=c.inspect(m.SESSION,parked=True);m.write(out/'BEFORE_EXIT.json',before)
        ui=SimpleNamespace(pid=m.PID,main=before['windows']['control'],state=lambda allow_fault=False:m.heartbeat(plan,released=True))
        key=c.Input(int(ui.main['wid'],16))
        try:
            for symbol,pressed in ((0xffe9,True),(0xffc1,True),(0xffc1,False),(0xffe9,False)):key.key(symbol,pressed)
        finally:key.close()
        actions.append(dict(action='Alt_F4',count=1));yes=False;end=time.monotonic()+20
        while (m.PROC/str(m.PID)).exists():
            m.require(m.start_ticks(m.PID)==m.START,'original PID reused during normal exit')
            if time.monotonic()>end:raise TimeoutError('normal exit timeout; no forced termination')
            if not yes:
                try:candidate=m._dialog(c,ui.main)
                except Exception as error:
                    if not (m.PROC/str(m.PID)).exists() and (old._known_dead_accessible(error)
                        or type(error) is RuntimeError and str(error)=='one original accessible control owner required'):break
                    raise
                if candidate:
                    node,A,row=candidate
                    actions.append(dict(action='Yes',result=native.click_visible_node(ui,node,A,owner_client_wid=row['wid'],owner_rect=row['rect'],execute=True)))
                    yes=True
            time.sleep(.05)
        m.write(out/'ACTIONS.json',actions);m.no_owner();service=m.service_status(unit)
        m.require(service==dict(MainPID='0',Result='success',ExecMainStatus='0'),'normal service exit proof required')
        rows=m.events(m.SESSION/'events.jsonl');exits=[r for r in rows[plan['exit_event_offset']:]
            if r.get('state')=='operator_confirmed_light_blocked_exit' and r.get('closed_safely') is True]
        m.require(len(exits)==1 and set(m.SESSION.glob('live_*/report.json'))==set(map(Path,plan['reports_before'])),
            'one new safe exit and no new acquisition cleanup report required')
        with (out/'events_after.jsonl').open('xb') as stream:stream.write((m.SESSION/'events.jsonl').read_bytes())
        result=dict(schema='E045-later-readonly-recovery-closed-1',completed=True,old_normal_exit_verified=True,
            old_cleanup_exact_passed=False,old_failure_reclassified=False,later_baseline_reverified=True,
            physical_shielding_confirmed=True,forced_kill=False,old_owner=plan['old_owner'],retention_ref=plan['retention_ref'],
            standby_baseline=plan['standby_baseline'],prospective_plan=m.ref(out/'RECOVERY_CLOSE_PLAN.json'),
            events_snapshot=m.ref(out/'events_after.jsonl'),exit_event=exits[0],service_status=service,source_sha256=plan['source_sha256'])
        m.write(out/'CLOSED.json',result);return result
    except BaseException as error:
        m.write(out/'RECOVERY_CLOSE_FAILED.json',dict(completed=False,error=repr(error),actions=actions,forced_kill=False));raise


def verify(closed_path,closed_sha,source_sha,output,*,execute=False,light_blocked_confirmed=False):
    if not execute or not light_blocked_confirmed:raise ValueError('explicit execute and current physical shielding required')
    m=engine();sources={};m.pin(SELF,source_sha,sources);closed=m.read(m.pin(closed_path,closed_sha,sources))
    m.require(closed.get('schema')=='E045-later-readonly-recovery-closed-1' and closed.get('completed') is True
        and closed.get('old_normal_exit_verified') is True and closed.get('physical_shielding_confirmed') is True
        and closed.get('forced_kill') is False and closed.get('old_cleanup_exact_passed') is False
        and closed.get('old_failure_reclassified') is False and closed.get('later_baseline_reverified') is True,
        'separate recovery normal-exit evidence required')
    for path,digest in closed['source_sha256'].items():m.pin(path,digest,sources)
    m.require(sources.get(str(SELF))==source_sha and sources.get(str(LIFECYCLE))==LIFECYCLE_SHA,'same frozen helper closure required')
    plan=m.read(m.pin(closed['prospective_plan']['path'],closed['prospective_plan']['sha256'],sources));preserved(m,plan)
    m.require(all(closed[k]==plan[k] for k in ('old_owner','retention_ref','standby_baseline')),'closed original identity differs')
    m.require(all(closed['source_sha256'].get(path)==digest for path,digest in plan['source_sha256'].items()),
        'closed source closure must retain the complete prospective recovery plan')
    m.pin(closed['events_snapshot']['path'],closed['events_snapshot']['sha256'],sources)
    m.no_owner();m.require(m.service_status(plan['old_owner']['unit'])==closed['service_status'],'normal service exit changed')
    out=Path(output);m.require(out.is_absolute() and out.parent==ROOT and not out.exists(),'NEW direct E045 cold verification output required');out.mkdir()
    m.write(out/'VERIFY_PLAN.json',dict(closed_ref=m.ref(closed_path),physical_shielding_confirmed=True,target=plan['standby_baseline'],source_sha256=sources))
    sys.path.insert(0,str(m.LAB/'src'));from optical_lab import legacy_gui as old
    old._load_clients(m.LEGACY);from camera_recheck import ReadOnlyCamera,verify_baseline
    def guard():
        m.no_owner()
        for path,digest in sources.items():m.pin(path,digest,{})
    result=verify_baseline(plan['standby_baseline'],guard,camera_factory=ReadOnlyCamera);m.write(out/'COLD_READONLY.json',result)
    passed=readonly_pass(plan['standby_baseline'],result)
    final=dict(schema='E045-normal-shielded-release-1',completed=passed,physical_shielding_confirmed=True,
        old_normal_exit_verified=True,cold_baseline_verified=passed,old_owner=dict(pid=m.PID,start_ticks=m.START),
        retention_ref=closed['retention_ref'],standby_baseline=plan['standby_baseline'],forced_kill=False,
        old_cleanup_exact_passed=False,old_failure_reclassified=False,later_baseline_reverified=True,
        camera_readonly_connections=result.get('connections'),camera_readonly_snapshots=len(result.get('snapshots',[])),
        parameter_writes=result.get('parameter_writes'),acquisition_started=result.get('acquisition_started'),errors=result.get('errors'),
        source_sha256=sources,evidence_sha256={str(p):m.sha(p) for p in (Path(closed_path),Path(closed['prospective_plan']['path']),
            Path(closed['events_snapshot']['path']),out/'VERIFY_PLAN.json',out/'COLD_READONLY.json')},
        no_framebuffer_black_claim_during_owner_gap=True,no_new_output_windows=True)
    m.write(out/'LIFECYCLE_RESULT.json',final);return final


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=('describe','close','verify'),default='describe')
    p.add_argument('--execute',action='store_true');p.add_argument('--light-blocked-confirmed',action='store_true')
    for name in ('source-sha256','retention','retention-sha256','old-unit','output','closed','closed-sha256'):p.add_argument('--'+name)
    a=p.parse_args(argv);auth=dict(execute=a.execute,light_blocked_confirmed=a.light_blocked_confirmed)
    if a.action=='close':result=close(a.retention,a.retention_sha256,a.source_sha256,a.old_unit,a.output,**auth)
    elif a.action=='verify':result=verify(a.closed,a.closed_sha256,a.source_sha256,a.output,**auth)
    else:result=dict(execution=False,UI_actions=0,SDK_connections=0,forced_kill=False,old_failure_reclassified=False)
    print(json.dumps(result,ensure_ascii=False));return int(a.action=='verify' and not result['completed'])


if __name__=='__main__':raise SystemExit(main())
