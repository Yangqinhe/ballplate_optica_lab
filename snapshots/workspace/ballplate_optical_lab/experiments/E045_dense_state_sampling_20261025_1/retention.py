"""File-only, provenance-preserving preparation for a later shielded restart.

Default import/describe performs no source or owner reads. Explicit prepare
reads files and /proc only; it never stops/launches a process, opens the SDK,
changes camera controls, creates a device window, or authorizes hardware.
The planned contract retains the OLD standby baseline, not live photometry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

SELF=Path(__file__).resolve();ROOT=SELF.parent;LAB=ROOT.parents[1]
LEGACY=LAB.parent/'ballplate_fourframe96_deploy'
OLD=LAB/'experiments/E040_temporal_preview_20261025_1'
SESSION=OLD/'GUI'
PROC=Path('/proc')
EXPECTED_PID=1390768;EXPECTED_START='6954576'
CONTRACT=OLD/'restart_contract.json'
CONTRACT_SHA='5fed291624ce559e4aa9277055fd0dfe505a5dda811205f77faeb4ac9d147896'
RECOVERY=OLD/'recovery_record.json'
RECOVERY_SHA='17ddafb1bcd65c6e3ffd34d5fcf46d897504594e0d84b9f526fc1ea85aa721cb'
PROFILE=LAB/'experiments/E044_repeatable_sampling_20261025_1/CAPTURE_LONG_DYN_F2_D12_02/PROFILE.json'
PROFILE_SHA='ef65bcdf7999a68e8212582bade096938fe66e6bf58c5cc1a07e999e6dbbbe6b'
PREVIEW_RESTART=LAB/'src/optical_lab/preview_restart.py'
PREVIEW_RESTART_SHA='6d3c0bc5c847d83ba209efbeaad8e2327535527892947701a6761f0597c6b703'
RUNTIME_SHA='ba0a8d3c1d530e7b353e4492e37184c71fcba8671cdeb1b107efaff7ae4a3ee9'
EXTENSION_SHA='ddfbc021c9b34d411158aecfba9d9768cf05d171cf43d175d39d4ea4b6cd17c3'
DMD=LAB/'experiments/E044_repeatable_sampling_20261025_1/DYNAMIC_BANK_01/D12/dmd.npy'
DMD_SHA='5e22a0f8773156ebd5934780da5a8eb930883277e568bfd2cdb44472df71000a'
SLM=LAB/'experiments/E035_mapping_night_20261024_1/SLM_TRAIN_02/offline_candidate_assets/L00_canonical600.npy'
SLM_SHA='deb9e7ae6e8b91b855ae3b863a835499a50f137bb6fa522bf46f11dda83b6c6e'
MANUAL=LAB/'experiments/E029_native_crash_recovery_20261024_1/manual_geometry_prior.json'
MANUAL_SHA='44062076dfcaced7266342cc2a280cbea8a04c6c5ed311b6cdbe6a1da92a57fa'


def sha(data):return hashlib.sha256(data).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def file_ref(path):return dict(path=str(path),sha256=sha(Path(path).read_bytes()))
def pin(path,digest):
    path=Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve()==path
        and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,
        'explicit regular input path/SHA required: '+str(path))
    blob=path.read_bytes();require(sha(blob)==digest,'input SHA differs: '+str(path));return blob


def description():
    return dict(schema='E045-planned-retention-1',default_reads_no_sources_or_owner=True,file_only=True,
        expected_old_PID=EXPECTED_PID,expected_old_start=EXPECTED_START,SDK_or_UI_actions=0,
        stop_or_launch=False,hardware_authorization=False,requested_exposure_us=80.,
        cold_baseline_recheck='PENDING_2_CONNECTIONS_10_READONLY_SNAPSHOTS',
        shielding_and_normal_exit='PENDING_EXTERNAL_CONFIRMATION',lifecycle_release_approved=False)


def fold_events(blob):
    require(blob and blob.endswith(b'\n'),'complete newline-terminated owner event snapshot required')
    values={};created=[]
    for number,line in enumerate(blob.splitlines(),1):
        e=json.loads(line);require(isinstance(e,dict),'owner event must be a JSON object')
        reference=dict(line_number=number,line_sha256=sha(line),event=e)
        state=e.get('state')
        if state=='manual_file_loaded':
            require(type(e.get('SLM')) is bool,'file cache role must be an explicit bool')
            values['slm' if e['SLM'] else 'dmd']=reference
        elif state=='manual_display_controls':values['display']=reference
        elif state=='manual_preview_filter_controls':values['filter']=reference
        elif state=='manual_display_requested':values['requested']=reference
        elif state=='created':created.append(reference)
    require(all(k in values for k in ('dmd','slm','display','filter','requested')) and len(created)==1,
        'explicit cache/display/filter/zero/unique-created event evidence required')
    values['created']=created[0]
    return values


def display_state(fold,active):
    event=fold['display']['event'];controls=event['controls']
    require(set(controls)=={'mode','gain','black','gamma'} and controls['mode'] in ('fixed','auto','raw')
        and event.get('display_only') is True and event.get('raw_or_actor_modified') is False,
        'explicit display-only controls required')
    for key,lo,hi in (('gain',1,128),('black',0,255),('gamma',.3,3)):
        v=controls[key];require(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi,'invalid display control: '+key)
    filt=fold['filter']['event'];fc=filt['controls']
    require(filt.get('display_only') is True and filt.get('raw_or_actor_modified') is False
        and fc.get('mode') in ('off','mean2','ema') and type(fc.get('alpha')) in (int,float)
        and math.isfinite(fc['alpha']) and .1<=fc['alpha']<=1 and type(fc.get('original')) is bool,
        'explicit display-only temporal filter controls required')
    return dict(schema='optical-lab-manual-display-state-1',manual=deepcopy(controls),
        preview_filter=deepcopy(fc),preview_filter_is_retention_only_not_dense_measurement_filter=True,
        source_event=fold['display'],preview_filter_source_event=fold['filter'],
        camera_gain_separate_actual_dB=float(active['Gain']),display_only=True,hardware_authorization=False)


def heartbeat_guard(h,runtime,profile,start,now_ns):
    require(h.get('pid')==runtime.get('pid')==EXPECTED_PID and start==EXPECTED_START
        and h.get('deployment_ui_revision')==23 and type(h.get('unix_ns')) is int
        and 0<=now_ns-h['unix_ns']<2_000_000_000,'fresh original PID/start/v23 owner required')
    require(h.get('hardware_fault') is None and h.get('projection_armed') is False
        and all(h.get(k) is False for k in ('loop_active','adaptation_active','deployment_busy','closed_loop_enabled'))
        and type(h.get('live_camera')) is bool and type(h.get('busy')) is bool
        and h['busy']==h['live_camera'] and h.get('last_verified_revision')==h.get('display_revision'),
        'owner must be idle or CCD-only, disarmed/fault-free with current verified zero revision')
    require(h.get('camera_active_readback')==profile['active'] and h.get('manual_settings')==profile['settings']
        and h.get('actual_roi')==profile['actual_roi'] and h.get('optical_reference')==profile['optical_reference']
        and h.get('geometry_frozen')==profile['geometry_frozen'] is False,
        'complete current camera/settings/ROI/model must match the fixed sealed E044 profile')


def owner_arguments(blob,cwd):
    argv=[p.decode('utf-8') for p in blob.split(b'\0') if p]
    require(argv[1:3]==['-m','optical_lab.preview'] and argv.count('-m')==1
        and argv.count('--hardware')==1,'the original optical_lab.preview hardware owner is required')
    def argument(flag):
        require(argv.count(flag)==1 and argv.index(flag)+1<len(argv),'unique owner argument required: '+flag)
        value=Path(argv[argv.index(flag)+1]);return (cwd/value).resolve()
    require(argument('--output')==SESSION and argument('--legacy-root')==LEGACY
        and argument('--restart-state')==CONTRACT and argument('--recovery-record')==RECOVERY,
        'original session/startup contract/recovery/legacy identity differs')
    manual=argument('--manual-geometry-state')
    require(manual==MANUAL and argv.count('--manual-geometry-sha256')==1
        and argv[argv.index('--manual-geometry-sha256')+1]==MANUAL_SHA,'explicit original E029 manual prior required')
    return argv,argument


def _legacy_helper():
    spec=importlib.util.spec_from_file_location('E045_frozen_preview_restart',PREVIEW_RESTART)
    module=importlib.util.module_from_spec(spec);prior=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=prior
    return module


def prepare(output,*,source_sha256,execute_file_only=False):
    """Only make a NEW planned bundle; never stops owner or permits launch."""
    require(execute_file_only is True,'explicit execute_file_only required before reading any owner')
    pin(SELF,source_sha256)
    output=Path(output)
    require(output.is_absolute() and output.parent==ROOT and output.resolve()==output
        and not output.exists() and not output.is_symlink(),'one NEW direct E045 retention child required')
    sources={};blobs={}
    def fixed(path,digest):
        blob=pin(path,digest);sources[str(path)]=digest;blobs[str(path)]=blob;return blob
    fixed(SELF,source_sha256)
    test=ROOT/'test_retention.py';fixed(test,sha(test.read_bytes()))
    old= json.loads(fixed(CONTRACT,CONTRACT_SHA));recovery=json.loads(fixed(RECOVERY,RECOVERY_SHA))
    profile=json.loads(fixed(PROFILE,PROFILE_SHA))
    fixed(PREVIEW_RESTART,PREVIEW_RESTART_SHA)
    runtime=json.loads(fixed(SESSION/'runtime_sources.json',RUNTIME_SHA))
    extension=json.loads(fixed(SESSION/'spatial_preview_extension.json',EXTENSION_SHA))
    for group in (runtime['sources'],runtime['source_configs'],extension['new_source_sha256'],
                  profile.get('sources',{}),old['evidence_sha256']):
        for path,digest in group.items():fixed(path,digest)
    fixed(RECOVERY,RECOVERY_SHA);basepath=Path(recovery['baseline_source'])
    baseline_source=json.loads(fixed(basepath,recovery['baseline_source_sha256']))
    baseline=old['original_fault_before']
    require(recovery['camera_baseline']==baseline and (baseline_source.get('camera_before') or baseline_source.get('target'))==baseline
        and baseline['DeviceSerialNumber']=='15328133' and baseline['TriggerMode']=='Off'
        and old.get('registration') is None and old.get('expected_active') is None,'the immutable old standby baseline must remain unchanged')
    proc=PROC/str(EXPECTED_PID)
    statblob=proc.joinpath('stat').read_bytes()
    start=statblob.decode().split(') ',1)[1].split()[19]
    cmd=proc.joinpath('cmdline').read_bytes();cwd=proc.joinpath('cwd').resolve()
    argv,argument=owner_arguments(cmd,cwd)
    for flag,digest_flag in (('--display-state','--display-state-sha256'),('--preview-caches','--preview-caches-sha256')):
        require(argv.count(digest_flag)==1,'startup state SHA missing')
        startup=json.loads(fixed(argument(flag),argv[argv.index(digest_flag)+1]))
        if flag=='--preview-caches':
            require(startup.get('schema')=='optical-lab-preview-cache-1' and set(startup.get('files',{}))=={'dmd','slm'},
                'old startup cache sidecar differs')
            for reference in startup['files'].values():fixed(reference['path'],reference['sha256'])
    manual=json.loads(fixed(MANUAL,MANUAL_SHA))
    for name,digest in manual['source_reference_sha256'].items():fixed(Path(manual['source_reference_folder'])/name,digest)
    hblob=SESSION.joinpath('heartbeat.json').read_bytes();eblob=SESSION.joinpath('events.jsonl').read_bytes()
    h=json.loads(hblob);fold=fold_events(eblob);heartbeat_guard(h,runtime,profile,start,time.time_ns())
    requested=fold['requested']['event']
    require(requested.get('revision')==h['display_revision'] and requested.get('dmd_nonzero') is False
        and requested.get('slm_nonzero') is False and requested.get('settings')==h['manual_settings'],
        'last owner request must prove matching disarmed all-zero requested panels')
    require(fold['created']['event']['video']==old['display_roles'],'old display role identity changed')
    files={}
    for role,path,digest in (('dmd',DMD,DMD_SHA),('slm',SLM,SLM_SHA)):
        event=fold[role]['event'];require(event.get('path')==str(path) and event.get('sha256')==digest,
            'current folded file cache differs: '+role)
        fixed(path,digest);files[role]=dict(path=str(path),sha256=digest)
    helper=_legacy_helper();helper.validate_manual_geometry_sidecar(MANUAL,MANUAL_SHA)
    helper._validate(old,LEGACY)
    caches=dict(schema='optical-lab-preview-cache-1',files=files)
    display=display_state(fold,h['camera_active_readback'])
    output.mkdir();archive=output/'SOURCE_ARCHIVE';archive.mkdir()
    mapping={}
    try:
        for i,(path,blob) in enumerate(blobs.items()):
            # .blob avoids passing arbitrary old config JSON to legacy baseline
            # discovery; original names and every original SHA remain explicit.
            target=archive/('%03d_'%i+Path(path).name+'.blob')
            with target.open('xb') as stream:stream.write(blob)
            mapping[path]=dict(original_path=path,sha256=sha(blob),archived_path=str(target),archived_sha256=sha(blob))
        for name,blob in (('owner_heartbeat.json',hblob),('owner_events.jsonl',eblob),('owner_cmdline.bin',cmd),
                          ('owner_proc_stat.txt',statblob)):
            with (output/name).open('xb') as stream:stream.write(blob)
        write(output/'baseline_evidence.json',dict(target=baseline,derived_evidence_NOT_new_measurement=True,
            original_contract=file_ref(CONTRACT),original_recovery=file_ref(RECOVERY),original_source=file_ref(basepath),
            standby_cold_readback_performed=False,new_camera_measurement=False))
        write(output/'DISPLAY_STATE.json',display);write(output/'PREVIEW_CACHES.json',caches)
        write(output/'MANUAL_GEOMETRY_REFERENCE.json',dict(schema='E045-original-manual-geometry-reference-1',
            source=dict(path=str(MANUAL),sha256=MANUAL_SHA),unchanged=True,manual_alignment_only=True,
            registration_installed=False,photometry_or_control_qualification=False))
        evidence={str(output/name):sha((output/name).read_bytes()) for name in
            ('owner_heartbeat.json','owner_events.jsonl','owner_cmdline.bin','owner_proc_stat.txt','baseline_evidence.json','DISPLAY_STATE.json',
             'PREVIEW_CACHES.json','MANUAL_GEOMETRY_REFERENCE.json')}
        evidence.update({row['archived_path']:row['sha256'] for row in mapping.values()})
        settings=dict(h['manual_settings'],dmd='black',slm='zero')
        contract=dict(schema='v20-retained-line0-v1',saved_unix_ns=time.time_ns(),original_session=str(SESSION),
            original_pid=EXPECTED_PID,runtime_path=str(SESSION/'runtime_sources.json'),settings=settings,
            optical_config=deepcopy(h['optical_reference']),roi=deepcopy(h['actual_roi']),requested_exposure_us=80.,
            fault=h['hardware_fault'],original_fault_before=deepcopy(baseline),original_last_camera_report=None,
            registration=None,expected_active=None,display_roles=deepcopy(old['display_roles']),evidence_sha256=evidence,
            no_auto_capture=True,no_auto_projection=True,release_approved=False,planned_restart_NOT_completed=True,
            original_owner_start_ticks=EXPECTED_START,cold_baseline_verification='PENDING',
            current_active_photometry_is_NOT_standby_baseline=True)
        helper._validate(contract,LEGACY)
        write(output/'restart_contract.json',contract)
        new_recovery=dict(schema='manual-alignment-recovery-v1',baseline_source=str(output/'restart_contract.json'),
            baseline_source_sha256=sha((output/'restart_contract.json').read_bytes()),camera_baseline=deepcopy(baseline),
            original_fault=h['hardware_fault'],old_pid=EXPECTED_PID,manual_stop_requires_recheck=True,
            planned_normal_restart=True,no_forced_termination=True,old_failures_reclassified=False,
            cold_baseline_readonly_verification_passed=False,original_recovery=file_ref(RECOVERY))
        write(output/'recovery_record.json',new_recovery)
        write(output/'EXPECTED_CAPTURE_CONTEXT.json',dict(schema='E045-retained-capture-context-NOT-calibration-1',
            profile=file_ref(PROFILE),active=profile['active'],settings=h['manual_settings'],roi=h['actual_roi'],
            optical_reference=h['optical_reference'],geometry_frozen=False,registration=None,expected_active=None,
            requested_exposure_us=80.,camera_gain_request_dB=11.,camera_gain_actual_dB=float(profile['active']['Gain']),
            readback_required_before_use=True,hardware_authorization=False))
        finalblob=SESSION.joinpath('heartbeat.json').read_bytes();final=json.loads(finalblob)
        finalevents=SESSION.joinpath('events.jsonl').read_bytes();finalfold=fold_events(finalevents)
        heartbeat_guard(final,runtime,profile,proc.joinpath('stat').read_text().split(') ',1)[1].split()[19],time.time_ns())
        require(finalfold==fold and proc.joinpath('cmdline').read_bytes()==cmd,'owner controls/cache/events changed while preparing')
        for path,digest in sources.items():pin(path,digest)
        with (output/'owner_heartbeat_final.json').open('xb') as stream:stream.write(finalblob)
        result=dict(description(),status='PLANNED_FILES_ONLY_NOT_LIFECYCLE_RELEASE',old_owner_still_running=True,
            output=str(output),restart_contract=file_ref(output/'restart_contract.json'),
            recovery_record=file_ref(output/'recovery_record.json'),display_state=file_ref(output/'DISPLAY_STATE.json'),
            preview_caches=file_ref(output/'PREVIEW_CACHES.json'),manual_geometry=dict(path=str(MANUAL),sha256=MANUAL_SHA),
            original_owner=dict(pid=EXPECTED_PID,start_ticks=EXPECTED_START,session=str(SESSION),argv=argv,cwd=str(cwd)),
            retained_current_active=profile['active'],folded_event_evidence=fold,
            original_baseline_unchanged=True,new_baseline_measurement_claimed=False,
            physical_shielding_confirmed_by_this_helper=False,old_normal_exit_verified=False,
            new_camera_started=False,new_camera_recovery_passed=False,physical_panel_state_independently_checked=False,
            old_failures_reclassified=False,pending=['external physical shielding','normal old owner stop and exit',
                'two cold readonly connections / ten complete baseline snapshots','new startup all-zero checks',
                'new actual capture profile readback / preview cache confirmation'])
        write(output/'RESULT.json',result)
        write(output/'SOURCE_ARCHIVE_MANIFEST.json',dict(schema='E045-retention-source-archives-1',original_source_sha256=sources,
            archived_sources=mapping,mutable_owner_snapshots=dict(heartbeat=file_ref(output/'owner_heartbeat.json'),
                events=file_ref(output/'owner_events.jsonl'),cmdline=file_ref(output/'owner_cmdline.bin'),
                proc_stat=file_ref(output/'owner_proc_stat.txt'),
                final_heartbeat=file_ref(output/'owner_heartbeat_final.json'))))
        manifest=dict(schema='E045-planned-retention-artifacts-1',file_sha256={str(p.relative_to(output)):sha(p.read_bytes())
            for p in output.rglob('*') if p.is_file()},original_source_sha256=sources)
        write(output/'MANIFEST.json',manifest)
        require(all(sha((output/p).read_bytes())==d for p,d in manifest['file_sha256'].items()),'retention artifact SHA self-check failed')
        return dict(status=result['status'],output=str(output),contract_sha256=sha((output/'restart_contract.json').read_bytes()),
            recovery_sha256=sha((output/'recovery_record.json').read_bytes()),manifest_sha256=sha((output/'MANIFEST.json').read_bytes()),
            file_only=True,hardware_actions=0,lifecycle_release_approved=False)
    except BaseException as error:
        write(output/'FAILED.json',dict(status='FAILED_FILES_RETAINED',error=repr(error),hardware_actions=0,lifecycle_release_approved=False))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-file-only',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--source-sha256');args=parser.parse_args(argv)
    if not args.prepare_file_only:
        require(args.output is None and args.source_sha256 is None,'default describe only; no execution fields')
        print(json.dumps(description(),ensure_ascii=False));return 0
    require(args.output is not None and args.source_sha256 is not None,'explicit NEW output/source SHA required')
    print(json.dumps(prepare(args.output,source_sha256=args.source_sha256,execute_file_only=True),ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
