"""Explicit read-only post-reconnection Off baseline verification.

Default import is stdlib only. Never runs the On-based transaction, writes a
node, starts acquisition or changes an old failure to PASS.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
OUTPUT=BASE/'POSTCONNECT04'
PRIOR=BASE/'APPROVED03/RESULT.json'
PRIOR_SHA='c44704f2f1dca69ee932e8eac7642ccf5b99720158ce2624a7526c93b0be6fc3'
LAB=BASE.parents[1]
LEGACY=LAB.parent/'ballplate_fourframe96_deploy'
SESSION=LAB/'experiments/E030_adopted_recovery_20261024_1/GUI'
PID=80408
DERIVED=dict(TriggerDelay=dict(before='1397.37',after='0'),AcquisitionFrameRate=dict(before='90',after='95'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def fixed_target(prior):
    commit=prior.get('commit_close',{});verification=prior.get('postcommit_readonly_verification',{})
    snapshots=verification.get('snapshots',[]);declared=commit.get('declared_target',{})
    if (prior.get('passed')is not False or commit.get('passed')is not True or commit.get('cleanup_errors')!=[] or
            commit.get('before_unchanged')is not True or commit.get('observed_actual')!=declared or
            prior.get('camera_after')!=declared or verification.get('passed')is not False or
            verification.get('parameter_writes')!=0 or verification.get('acquisition_started')is not False or
            verification.get('connections')!=1 or len(snapshots)!=5 or any(s.get('connection')!=0 for s in snapshots)):
        raise ValueError('fixed failed-overall, successful-Off-commit, readonly-reconnection evidence required')
    target=copy.deepcopy(snapshots[0]['values'])
    if set(target)!=set(declared) or any(s.get('values')!=target for s in snapshots):
        raise ValueError('all five original postconnect observations must be identical complete dictionaries')
    delta={k:dict(before=declared[k],after=target[k]) for k in target if declared[k]!=target[k]}
    expected=dict(TriggerMode='Off',ExposureTime='67.17443466186523',ExposureAuto='Off',Gain='0.0',GainAuto='Off',
        OffsetX='900',OffsetY='416',Width='552',Height='554',TriggerSource='Line0',TriggerActivation='FallingEdge',
        SensorShutterMode='Rolling',TriggerDelayEnabled='0',DeviceSerialNumber='15328133',PixelFormat='Mono8')
    if delta!=DERIVED or any(target.get(k)!=v for k,v in expected.items()):
        raise ValueError('only exact observed inactive-delay1397.37-to0 and rate90-to95 may differ; formal fields unchanged')
    return target


def complete(verification,target):
    snapshots=verification.get('snapshots',[])
    return bool(verification.get('passed')is True and verification.get('connections')==2 and len(snapshots)==10 and
        verification.get('parameter_writes')==0 and verification.get('acquisition_started')is False and not verification.get('errors') and
        [sum(s.get('connection')==i for s in snapshots) for i in (0,1)]==[5,5] and
        all(s.get('values')==target and s.get('differences')=={} for s in snapshots))


def execute(blocked):
    if not blocked:raise ValueError('continued explicit physical shielding required')
    if sha(PRIOR)!=PRIOR_SHA:raise ValueError('exact failed APPROVED03 result changed')
    prior=read(PRIOR);target=fixed_target(prior)
    sources=copy.deepcopy(prior['source_sha256'])
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('prior source closure changed: '+path)
    sources[str(PRIOR)]=PRIOR_SHA;sources[str(Path(__file__).resolve())]=sha(__file__)
    OUTPUT.mkdir(exist_ok=False);binding=OUTPUT/'BINDING.json'
    write(binding,dict(schema='optical-lab-postconnect-off-readonly-binding-1',prior_result_file=str(PRIOR),
        prior_result_sha256=PRIOR_SHA,verifier_file=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),
        fixed_postconnect_target=target,allowed_reconnection_differences=DERIVED,
        prior_overall_passed=False,commit_stage_verified=True,approved_by_user=True,physical_shielding_confirmed=True,
        parameter_writes=0,acquisition_started=False,no_On_based_transaction_reexecuted=True,
        formal_capture_required=dict(TriggerMode='On',ExposureTime='74.20778274536133'),source_sha256=sources))
    sources[str(binding)]=sha(binding)
    from optical_lab import legacy_gui as old
    c=old._load_clients(LEGACY);old._install_accessibility_compatibility();observations=[]
    def guard():
        value=c.inspect(SESSION,parked=True)
        if value.get('pid')!=PID or any(value['heartbeat'].get(k)is not False for k in ('camera_reserved','live_camera','projection_armed')):
            raise RuntimeError('exact E030 released and actual all-RGB-zero owner required')
        observations.append(value)
    result=dict(schema='optical-lab-postconnect-off-readonly-verification-1',passed=False,error=None,
        target=target,snapshots=[],connections=0,parameter_writes=0,acquisition_started=False,source_sha256=sources,
        prior_result_file=str(PRIOR),prior_result_sha256=PRIOR_SHA,prior_overall_passed=False,
        old_failures_reclassified=False,approved_by_user=True,physical_shielding_confirmed=True,
        formal_capture_required=dict(TriggerMode='On',ExposureTime='74.20778274536133'))
    try:
        guard()
        from camera_recheck import ReadOnlyCamera,verify_baseline
        import camera_recheck
        if Path(camera_recheck.__file__).resolve()!=LEGACY/'live_manual_v17/camera_recheck.py':
            raise ValueError('original readonly camera source required')
        verification=verify_baseline(target,guard,camera_factory=ReadOnlyCamera)
        result.update(verification)
        if not complete(verification,target):raise RuntimeError('new exact postconnect Off target failed2connections/10reads')
        if any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('bound source changed')
        guard();result['passed']=True
    except BaseException as exc:result.update(passed=False,error=repr(exc))
    finally:
        try:guard();result['final_original_outputs_zero_verified']=True
        except Exception as exc:result.update(passed=False,final_guard_error=repr(exc))
        result['black_owner_observations']=observations;write(OUTPUT/'RESULT.json',result)
    if result['passed']:
        snapshots=result['snapshots']
        write(OUTPUT/'OFF_BASELINE.json',dict(schema='optical-lab-postconnect-trigger-off-standby-baseline-1',
            passed=True,approved_by_user=True,camera_before=copy.deepcopy(snapshots[0]['values']),
            camera_after=copy.deepcopy(snapshots[-1]['values']),target=target,snapshots=copy.deepcopy(snapshots),connections=2,
            parameter_writes=0,acquisition_started=False,source_result=str(OUTPUT/'RESULT.json'),source_result_sha256=sha(OUTPUT/'RESULT.json'),
            prior_result_file=str(PRIOR),prior_result_sha256=PRIOR_SHA,prior_overall_passed=False,
            old_failures_reclassified=False,source_sha256=sources))
    print(json.dumps(dict(passed=result['passed'],error=result['error'],OFF_BASELINE_created=result['passed'])),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    sys.path.insert(0,str(LAB/'src'))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--light-blocked-confirmed',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('explicit --execute --light-blocked-confirmed required')
    execute(args.light_blocked_confirmed)
