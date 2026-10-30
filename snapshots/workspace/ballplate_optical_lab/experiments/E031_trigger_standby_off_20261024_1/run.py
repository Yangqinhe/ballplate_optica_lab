"""Explicit one-node stopped-camera standby adoption; import is stdlib only.

No action without --execute --light-blocked-confirmed. Acquisition and all
optical outputs remain off. Formal capture still uses the original Line0 On
driver; this utility does not modify that driver or old records.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
LEGACY=LAB.parent/'ballplate_fourframe96_deploy'
SESSION=LAB/'experiments/E030_adopted_recovery_20261024_1/GUI'
PID=80408
ADOPTION=LAB/'experiments/E029_native_crash_recovery_20261024_1/BASELINE_ADOPTION.json'
ADOPTION_SHA='38ecda60473d5bae74c87f436139e388ea8dd3a5df96a01ba2d03b02ee238da0'
READABILITY={'AcquisitionFrameRate','AcquisitionFrameRateEnable'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    from optical_lab.storage import sha256
    return sha256(path)
def write(path,value):
    from optical_lab.storage import write_json_new
    write_json_new(path,value)
def differences(before,after):
    return {k:dict(before=before.get(k),after=after.get(k)) for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}


def off_target(before,after):
    """Exact formal fields; record only readability and inactive-delay effects."""
    if set(before)!=set(after) or before.get('TriggerMode')!='On' or after.get('TriggerMode')!='Off':
        raise ValueError('identical full parameter keys and explicit On-to-Off required')
    allowed={'TriggerMode'}|READABILITY
    if before.get('TriggerDelayEnabled')=='0' and after.get('TriggerDelayEnabled')=='0':allowed.add('TriggerDelay')
    changed=differences(before,after)
    if set(changed)-allowed:raise ValueError('formal camera parameters changed beyond TriggerMode')
    for key in READABILITY:
        if key in changed and before[key] is not None and after[key] is not None:
            raise ValueError('only derived readability changes accepted: '+key)
    return copy.deepcopy(after),{k:v for k,v in changed.items() if k!='TriggerMode'}


def commit_close(camera,declared_target):
    """Release SDK resources against an EXPLICIT new Off target, never rebase before.

    Precondition/comparison failure leaves the camera open for the sole On
    rollback. Once cleanup starts all release steps are attempted and every
    error is retained; an imperfect cleanup never counts as success.
    """
    if camera.closed or camera.acquiring or camera.undo or not camera.initialized:
        raise RuntimeError('commit close needs open initialized stopped camera with empty undo')
    actual=camera.parameters();camera.after=copy.deepcopy(actual)
    if actual!=declared_target:raise RuntimeError('commit close actual differs from declared Off target')
    off_target(camera.before,declared_target)
    report=dict(declared_target=copy.deepcopy(declared_target),observed_actual=copy.deepcopy(actual),
        original_before=copy.deepcopy(camera.before),before_unchanged=True,cleanup_started=True,
        acquisition_started=False,undo_entries=0,cleanup_steps=[],cleanup_errors=[],passed=False)
    camera.explicit_commit_close=report
    def cleanup(name,*args):
        try:camera.call(name,*args);report['cleanup_steps'].append(dict(name=name,passed=True))
        except Exception as exc:
            report['cleanup_errors'].append(dict(name=name,error=repr(exc)))
            report['cleanup_steps'].append(dict(name=name,passed=False))
    cleanup('spinCameraDeInit',camera.camera);camera.initialized=False
    if camera.camera:cleanup('spinCameraRelease',camera.camera);camera.camera=type(camera.camera)()
    if camera.cameras:
        cleanup('spinCameraListClear',camera.cameras);cleanup('spinCameraListDestroy',camera.cameras);camera.cameras=type(camera.cameras)()
    if camera.system:cleanup('spinSystemReleaseInstance',camera.system);camera.system=type(camera.system)()
    if camera.lock:
        try:camera.lock.close();report['cleanup_steps'].append(dict(name='camera_lock.close',passed=True))
        except Exception as exc:report['cleanup_errors'].append(dict(name='camera_lock.close',error=repr(exc)))
        camera.lock=None
    camera.closed=True;camera.close_errors=copy.deepcopy(report['cleanup_errors'])
    report['passed']=not report['cleanup_errors']
    if not report['passed']:raise RuntimeError('explicit Off commit cleanup failed: '+repr(report['cleanup_errors']))
    return copy.deepcopy(report)


def change(camera,expected,guard,result):
    """Only one intended Off write; sole failure rollback is original On."""
    attempted=False
    try:
        result['camera_before']=copy.deepcopy(camera.before)
        if camera.before!=expected:raise ValueError('actual before differs from approved original On baseline')
        if camera.acquiring or camera.undo:raise RuntimeError('stopped empty-undo camera required')
        guard();result['write_trace'].append(dict(name='TriggerMode',requested='Off',attempted=True,completed=False))
        attempted=True;camera._write('TriggerMode','Off',camera.map);result['write_trace'][-1]['completed']=True
        actual=camera.parameters();target,derived=off_target(camera.before,actual)
        result.update(camera_after=copy.deepcopy(actual),derived_changes=derived,formal_other_fields_exact_unchanged=True)
        guard();result['commit_close']=commit_close(camera,target)
        return target
    except BaseException:
        result['commit_close']=copy.deepcopy(getattr(camera,'explicit_commit_close',None))
        if not camera.closed:
            if attempted:
                result['rollback']=dict(name='TriggerMode',requested='On',attempted=True,passed=False)
                try:
                    camera._write('TriggerMode','On',camera.map)
                    result['rollback']['actual']=camera.parameters()
                    result['rollback']['passed']=result['rollback']['actual']==camera.before
                except Exception as exc:result['rollback']['error']=repr(exc)
            try:camera.close()
            except Exception as exc:result['original_close_error']=repr(exc)
            result['after_original_close']=copy.deepcopy(getattr(camera,'after',None))
        raise


def require_adoption(value):
    target=value.get('camera_before');snapshots=value.get('snapshots',[])
    required=dict(DeviceSerialNumber='15328133',PixelFormat='Mono8',ExposureAuto='Off',ExposureTime='74.20778274536133',
        GainAuto='Off',Gain='0.0',OffsetX='900',OffsetY='416',Width='552',Height='554',TriggerMode='On',
        TriggerSource='Line0',TriggerActivation='FallingEdge',SensorShutterMode='Rolling',TriggerDelayEnabled='0',TriggerDelay='1397.37')
    if (value.get('schema')!='optical-lab-approved-postcrash-baseline-1' or
            value.get('passed')is not True or value.get('approved_by_user')is not True or value.get('parameter_writes')!=0 or
            value.get('acquisition_started')is not False or value.get('connections')!=2 or len(snapshots)!=10 or
            [sum(s.get('connection')==i for s in snapshots) for i in (0,1)]!=[5,5] or
            not isinstance(target,dict) or any(target.get(k)!=v for k,v in required.items()) or target!=value.get('camera_after') or
            any(s.get('values')!=target or s.get('differences')!={} for s in snapshots)):
        raise ValueError('explicit approved unchanged original manual On adoption required')
    return copy.deepcopy(target)


def execute(blocked):
    if not blocked:raise ValueError('continued physical shielding must be confirmed')
    if any((ROOT/n).exists() for n in ('RESULT.json','OFF_BASELINE.json','REQUEST.json')):
        raise ValueError('new execution only; do not overwrite or automatically repeat a failed attempt')
    from optical_lab import legacy_gui as old
    if sha(ADOPTION)!=ADOPTION_SHA:raise ValueError('explicit approved On baseline changed')
    expected=require_adoption(read(ADOPTION));c=old._load_clients(LEGACY);old._install_accessibility_compatibility()
    observations=[]
    def guard():
        value=c.inspect(SESSION,parked=True)
        if value.get('pid')!=PID or any(value['heartbeat'].get(k)is not False for k in ('camera_reserved','live_camera','projection_armed')):
            raise RuntimeError('exact E030 released all-RGB-zero owner required')
        observations.append(value);return value
    guard()
    from spin_c import Camera
    from camera_recheck import ReadOnlyCamera,verify_baseline
    import spin_c,camera_recheck
    sources=old.dependency_sources(LEGACY)
    for path in (Path(__file__).resolve(),ADOPTION,Path(spin_c.__file__).resolve(),Path(camera_recheck.__file__).resolve()):sources[str(path)]=sha(path)
    if Path(spin_c.__file__).resolve()!=LEGACY/'live_manual_v17/spin_c.py' or Path(camera_recheck.__file__).resolve()!=LEGACY/'live_manual_v17/camera_recheck.py':
        raise ValueError('original explicit C adapter/recheck source required')
    write(ROOT/'REQUEST.json',dict(source_sha256=sources,baseline_file=str(ADOPTION),baseline_sha256=ADOPTION_SHA,
        user_approved=True,physical_shielding_confirmed=True,only_intended_write=dict(TriggerMode='Off'),
        formal_acquisition_driver_unchanged=True,acquisition_started=False,outputs_changed=False))
    result=dict(schema='optical-lab-explicit-trigger-off-standby-commit-1',passed=False,error=None,
        write_trace=[],source_sha256=sources,original_baseline_file=str(ADOPTION),original_baseline_sha256=ADOPTION_SHA,
        original_baseline_expected=expected,user_approved=True,physical_shielding_confirmed=True,
        acquisition_started=False,old_failures_reclassified=False,formal_Line0_On_capture_driver_unchanged=True,
        outputs_changed=False,automatic_retry=False)
    camera=None;verification=None
    try:
        guard();camera=Camera();target=change(camera,expected,guard,result);camera=None
        verification=verify_baseline(target,guard,camera_factory=ReadOnlyCamera)
        result['postcommit_readonly_verification']=verification
        snapshots=verification['snapshots']
        if (not verification['passed'] or verification['connections']!=2 or len(snapshots)!=10 or
                verification.get('parameter_writes')!=0 or verification.get('acquisition_started')is not False or
                [sum(s.get('connection')==i for s in snapshots) for i in (0,1)]!=[5,5] or
                any(s.get('values')!=target or s.get('differences')!={} for s in snapshots)):
            raise RuntimeError('new Off target did not pass two connections/ten exact readonly snapshots')
        if any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('bound source changed')
        guard();result['passed']=True
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if camera is not None and not camera.closed:
            try:camera.close()
            except Exception as exc:result['final_original_close_error']=repr(exc);result['passed']=False
        try:guard();result['final_original_outputs_zero_verified']=True
        except Exception as exc:result['final_guard_error']=repr(exc);result['passed']=False
        result['black_owner_observations']=observations
        write(ROOT/'RESULT.json',result)
    if result['passed']:
        snapshots=verification['snapshots']
        write(ROOT/'OFF_BASELINE.json',dict(schema='optical-lab-postcommit-trigger-off-standby-baseline-1',
            passed=True,approved_by_user=True,camera_before=copy.deepcopy(snapshots[0]['values']),
            camera_after=copy.deepcopy(snapshots[-1]['values']),target=copy.deepcopy(snapshots[0]['values']),
            snapshots=copy.deepcopy(snapshots),connections=2,parameter_writes=0,acquisition_started=False,
            source_result=str(ROOT/'RESULT.json'),source_result_sha256=sha(ROOT/'RESULT.json'),
            original_On_snapshot_retained_in_RESULT=True,formal_Line0_On_capture_driver_unchanged=True,
            old_failures_reclassified=False,source_sha256=sources))
    print(json.dumps(dict(passed=result['passed'],error=result['error'],OFF_BASELINE_created=result['passed'])),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    sys.path.insert(0,str(LAB/'src'))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--light-blocked-confirmed',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('explicit --execute --light-blocked-confirmed required')
    execute(args.light_blocked_confirmed)
