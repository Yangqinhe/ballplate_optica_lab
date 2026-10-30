"""File-only retention of the existing E045 owner; default is descriptive.

No UI/SDK/process lifecycle operations. A prepared bundle is not permission to
restart and never claims a cold camera check or physical panel verification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

SELF=Path(__file__).resolve(); ROOT=SELF.parent; LAB=ROOT.parents[1]
OLD=LAB/'experiments/E045_dense_state_sampling_20261025_1'
SESSION=OLD/'GUI'; PID=1793876; START='9362457'
CONTRACT=OLD/'CONTRACT_02/CONTRACT.json'
CONTRACT_SHA='25efd77a83ea81bc63134c774e910f26ca1992cf439f4088cf3d711daf05aff8'
RETAIN=OLD/'RETENTION_01/RESULT.json'
RETAIN_SHA='926415c75c557ff8f61eac3e1994caed247e070ae1dee3add898537b8b121057'
RUN=OLD/'run.py'; RUN_SHA='45d1bac343890db2bff543e3db00c3941cb9421c410bc0e19c0a76a4becf5028'
HELPER=OLD/'retention.py'; HELPER_SHA='841a0fe6e2be34e66a1a0551acebfcce1a355cf9a17020b5c2a7ad4eeafcf09a'
READER=LAB/'experiments/E046_reference_sampling_20261025_1/load_input.py'
READER_SHA='c47c933e65157663ecd0787319f9362560cde1e9c032c6b9bb9111e8d404e9db'
RUNTIME_SHA='7bb3551690850d9db70726fd55c36da6a181f9a54f4f8d1e78a3a948bf690c4e'
EXTENSION_SHA='bcdd429db8c2588cfdf1da45f120908a240bf143629dd8bd0096fdb10c0a6596'

def require(ok,message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(path),sha256=sha(path))
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def write_blob(path,blob):
    with Path(path).open('xb') as stream:stream.write(blob)
def pin(reference):
    p=Path(reference['path']);require(p.is_absolute() and p.resolve()==p and p.is_file()
        and not p.is_symlink() and sha(p)==reference['sha256'],'fixed source differs: '+str(p));return p
def module(path,digest,name):
    pin(dict(path=str(path),sha256=digest));spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
def original_owner():
    proc=Path('/proc')/str(PID);stat=proc.joinpath('stat').read_bytes()
    require(stat.decode().rsplit(')',1)[1].split()[19]==START,'original owner start differs')
    cmd=proc.joinpath('cmdline').read_bytes();argv=[s.decode() for s in cmd.split(b'\0') if s]
    def arg(name):
        require(argv.count(name)==1,'unique original owner flag missing: '+name)
        return argv[argv.index(name)+1]
    require(str(RUN) in argv and argv.count('--hardware')==1
        and arg('--source-sha256')==RUN_SHA and arg('--contract')==str(CONTRACT)
        and arg('--contract-sha256')==CONTRACT_SHA and arg('--retention')==str(RETAIN)
        and arg('--retention-sha256')==RETAIN_SHA and arg('--output')==str(SESSION)
        and arg('--spool')==str(OLD/'GUI_SPOOL'),'original launcher/contract/session differs')
    return dict(pid=PID,start_ticks=START,session=str(SESSION),argv=argv,cwd=str(proc.joinpath('cwd').resolve())),cmd,stat

def prepare(output,source_sha256):
    pin(dict(path=str(SELF),sha256=source_sha256));output=Path(output)
    require(output.is_absolute() and output.parent==ROOT and not output.exists() and not output.is_symlink(),
        'one NEW direct E047 retention folder required')
    runner=module(RUN,RUN_SHA,'E047_retention_original_run')
    old=runner.validate(dict(path=str(CONTRACT),sha256=CONTRACT_SHA),
        dict(path=str(RETAIN),sha256=RETAIN_SHA),self_sha256=RUN_SHA)
    helper=module(HELPER,HELPER_SHA,'E047_retention_original_helper')
    reader=module(READER,READER_SHA,'E047_retention_readonly_ready')
    contract=old['contract'];oldret=old['retained'];baseline=old['recovery']['camera_baseline']
    require(baseline==old['restart']['original_fault_before'] and baseline['TriggerMode']=='Off',
        'unchanged Off standby baseline required')
    sources=dict(contract['source_sha256'])
    for p,d in ((SELF,source_sha256),(RUN,RUN_SHA),(HELPER,HELPER_SHA),(READER,READER_SHA),
                (CONTRACT,CONTRACT_SHA),(RETAIN,RETAIN_SHA),(SESSION/'runtime_sources.json',RUNTIME_SHA),
                (SESSION/'dense_sampling_extension.json',EXTENSION_SHA)):
        require(sources.setdefault(str(p),d)==d,'contradictory source SHA')
    for p,d in sources.items():pin(dict(path=p,sha256=d))
    extension=read(SESSION/'dense_sampling_extension.json')
    require(extension['owner_pid']==PID and extension['owner_start_ticks']==START
        and extension['source_sha256']==contract['source_sha256'],'loaded source closure/owner differs')
    owner,cmd,stat=original_owner()
    output.mkdir()
    try:
        before=reader.read_ready(contract,output/'HEARTBEAT_READS.jsonl');h=before['heartbeat']
        events=SESSION.joinpath('events.jsonl').read_bytes();fold=helper.fold_events(events)
        requested=fold['requested']['event']
        require(requested.get('revision')==h['display_revision'] and requested.get('dmd_nonzero')is False
            and requested.get('slm_nonzero')is False and requested.get('settings')==h['manual_settings'],
            'matching disarmed all-zero request evidence required')
        require(fold['created']['event']['video']==old['restart']['display_roles'],'original display roles differ')
        files={role:{k:fold[role]['event'][k] for k in ('path','sha256')} for role in ('dmd','slm')}
        allowed=[{k:a[k] for k in ('path','sha256')} for a in contract['allowed_dmd'] if str(a['input_id'])=='0']
        require(len(allowed)==1 and files['dmd']==allowed[0] and files['slm']==contract['fixed_slm'],
            'exact current DMD0 and fixed L00 caches required')
        for r in files.values():pin(r)
        prior=oldret['manual_geometry'];geometry=helper._legacy_helper().validate_manual_geometry_sidecar(prior['path'],prior['sha256'])
        require(geometry['roi']==h['actual_roi'] and geometry['optical_config']==h['optical_reference']
            and geometry['registration']['orientation']==h['simulation_display_orientation']['code']
            and h['geometry_frozen']is False,'retained alignment-only geometry differs')
        for name,value in (('owner_heartbeat.json',h),('owner_status.json',before['status'])):write(output/name,value)
        for name,blob in (('owner_events.jsonl',events),('owner_cmdline.bin',cmd),('owner_proc_stat.txt',stat)):
            write_blob(output/name,blob)
        write_blob(output/'SOURCE.py',SELF.read_bytes())
        write(output/'baseline_evidence.json',dict(target=baseline,derived_evidence_NOT_new_measurement=True,
            original_contract=oldret['restart_contract'],original_recovery=oldret['recovery_record'],
            standby_cold_readback_performed=False,new_camera_measurement=False))
        write(output/'DISPLAY_STATE.json',helper.display_state(fold,h['camera_active_readback']))
        write(output/'PREVIEW_CACHES.json',dict(schema='optical-lab-preview-cache-1',files=files))
        write(output/'MANUAL_GEOMETRY_REFERENCE.json',dict(schema='E047-original-manual-geometry-reference-1',
            source=prior,unchanged=True,manual_alignment_only=True,registration_installed=False,
            photometry_or_control_qualification=False))
        evidence={str(p):sha(p) for p in output.iterdir() if p.is_file() and p.name!='HEARTBEAT_READS.jsonl'}
        evidence.update({str(pin(r)):r['sha256'] for r in (oldret['restart_contract'],oldret['recovery_record'],prior)})
        restart=deepcopy(old['restart'])
        restart.update(saved_unix_ns=time.time_ns(),original_session=str(SESSION),original_pid=PID,
            runtime_path=str(SESSION/'runtime_sources.json'),settings=dict(h['manual_settings'],dmd='black',slm='zero'),
            optical_config=h['optical_reference'],roi=h['actual_roi'],fault=h['hardware_fault'],
            original_owner_start_ticks=START,evidence_sha256=evidence,original_last_camera_report=None,
            planned_restart_NOT_completed=True,cold_baseline_verification='PENDING',
            source_original_restart=oldret['restart_contract'])
        helper._legacy_helper()._validate(restart,LAB.parent/'ballplate_fourframe96_deploy')
        write(output/'restart_contract.json',restart)
        recovery=deepcopy(old['recovery'])
        recovery.update(baseline_source=str(output/'restart_contract.json'),baseline_source_sha256=sha(output/'restart_contract.json'),
            original_fault=h['hardware_fault'],old_pid=PID,original_recovery=oldret['recovery_record'],
            cold_baseline_readonly_verification_passed=False)
        write(output/'recovery_record.json',recovery)
        context=deepcopy(read(RETAIN.parent/'EXPECTED_CAPTURE_CONTEXT.json'))
        context.update(profile=dict(path=str(CONTRACT),sha256=CONTRACT_SHA),active=h['camera_active_readback'],
            settings=h['manual_settings'],roi=h['actual_roi'],optical_reference=h['optical_reference'])
        write(output/'EXPECTED_CAPTURE_CONTEXT.json',context)
        final=reader.read_ready(contract,output/'HEARTBEAT_READS.jsonl')
        final_owner,final_cmd,_=original_owner()
        require(helper.fold_events(SESSION.joinpath('events.jsonl').read_bytes())==fold
            and final_owner==owner and final_cmd==cmd,'current owner controls changed during retention')
        write(output/'owner_heartbeat_final.json',final['heartbeat'])
        result=deepcopy(oldret)
        result.update(schema='E047-planned-retention-1',expected_old_PID=PID,expected_old_start=START,
            output=str(output),original_owner=owner,retained_current_active=h['camera_active_readback'],
            folded_event_evidence=fold,source_retention=dict(path=str(RETAIN),sha256=RETAIN_SHA),
            manual_geometry=prior,status='PLANNED_FILES_ONLY_NOT_LIFECYCLE_RELEASE',lifecycle_release_approved=False,
            physical_shielding_confirmed_by_this_helper=False,old_normal_exit_verified=False,
            new_camera_started=False,new_camera_recovery_passed=False,physical_panel_state_independently_checked=False)
        for key,name in (('restart_contract','restart_contract.json'),('recovery_record','recovery_record.json'),
                         ('display_state','DISPLAY_STATE.json'),('preview_caches','PREVIEW_CACHES.json')):
            result[key]=ref(output/name)
        write(output/'RESULT.json',result)
        # Compatibility proof ONLY: original run.validate checks every original
        # gate. This contract is not a startup or lifecycle release artifact.
        proof=deepcopy(contract);proof['source_sha256']=dict(contract['source_sha256'],**{str(SELF):source_sha256,
            str(output/'RESULT.json'):sha(output/'RESULT.json')})
        proof['compatibility_validation_only_NOT_launch_contract']=True
        write(output/'VALIDATE_ONLY_CONTRACT.json',proof)
        for p,d in sources.items():pin(dict(path=p,sha256=d))
        manifest=dict(schema='E047-planned-retention-artifacts-1',file_sha256={str(p.relative_to(output)):sha(p)
            for p in output.rglob('*') if p.is_file()},original_source_sha256=sources)
        write(output/'MANIFEST.json',manifest)
        runner.validate(ref(output/'VALIDATE_ONLY_CONTRACT.json'),ref(output/'RESULT.json'),self_sha256=RUN_SHA)
        return dict(status=result['status'],retention=ref(output/'RESULT.json'),manifest=ref(output/'MANIFEST.json'),
            original_run_validate_passed=True,hardware_actions=0,old_sources_unchanged=True,
            lifecycle_release_approved=False)
    except BaseException as error:
        if not (output/'FAILED.json').exists():write(output/'FAILED.json',dict(error=repr(error),hardware_actions=0,
            lifecycle_release_approved=False,failed_files_retained=True))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-file-only',action='store_true');parser.add_argument('--source-sha256')
    parser.add_argument('--output',type=Path);a=parser.parse_args(argv)
    if not a.prepare_file_only:
        require(a.output is None and a.source_sha256 is None,'default description only')
        print(json.dumps(dict(file_only=True,hardware_actions=0,expected_owner=PID,
            no_source_or_owner_reads=True,lifecycle_release_approved=False)));return 0
    require(a.output is not None and a.source_sha256 is not None,'explicit NEW output/source SHA required')
    print(json.dumps(prepare(a.output,a.source_sha256),ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
