"""Explicit crash evidence / stopped-camera readback / fresh configuration.

Default no IO. prepare is file-only; camera-readonly requires confirmed shielding
and absent old/new camera owners. No camera writes, acquisition or USB commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
OLD = LAB/'experiments/E068_sampling_order_20261026_1'
BASE = LAB/'experiments/E052_n60_crash_recovery_20261026_1'
TEST = OLD/'TESTS/TEST_20261026_170630_326808'
PID = 2395229

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(p), sha256=sha(p))
def write(p, v):
    with Path(p).open('x') as f: json.dump(v, f, ensure_ascii=False, indent=2)

def no_owner():
    if Path(f'/proc/{PID}').exists(): raise RuntimeError('old owner remains')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == __import__('os').getpid(): continue
        try: argv=(p/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if b'--hardware' in argv and any(b'ballplate_' in x for x in argv):
            raise RuntimeError('hardware owner still exists: '+p.name)

def prepare():
    no_owner(); out=ROOT/'CRASH_01'; out.mkdir()
    crash=Path('/var/crash/_usr_bin_python3.10.1000.crash')
    shutil.copy2(crash, out/'APPORT.crash')
    journal=subprocess.check_output(['journalctl','--user','-u',
        'ballplate-optical-sampling-order-20261026-e068.service','--since','2026-10-26 17:06:20',
        '--until','2026-10-26 17:07:30','--no-pager'],text=True)
    (out/'JOURNAL.txt').write_text(journal)
    files={str(p):sha(p) for p in TEST.rglob('*') if p.is_file()}
    status=read(TEST/'RAW/STATUS.json')
    write(out/'CHECKPOINT.json', dict(crashed_pid=PID, heartbeat=read(OLD/'GUI/heartbeat.json'),
        batch_status=status['core'], completed_batches=len(list(TEST.glob('RAW/results/*/BATCH_RESULT.json'))),
        source_and_partial_file_sha256=files, apport=ref(out/'APPORT.crash'),
        allzero_after_crash_proven=False, old_normal_exit=False,
        partial_batch_accepted=False, actual_closed_loop_run=False))
    print(json.dumps(dict(evidence_preserved=True, raw_prefix_files=len(files), hardware_IO=0)))

def camera_readonly(blocked):
    if not blocked: raise RuntimeError('shielding required')
    no_owner(); out=ROOT/'CAMERA_READONLY_01'; out.mkdir()
    sys.path.insert(0,str(LAB.parent/'ballplate_fourframe96_deploy/live_manual_v17'))
    from camera_recheck import ReadOnlyCamera, differences
    target=read(OLD/'RETENTION_01/recovery_record.json')['camera_baseline']
    result=dict(parameter_writes=0, acquisition_started=False, target=target,
                physical_shielding_confirmed=True, snapshots=[], errors=[])
    for i in range(2):
        camera=None
        try:
            no_owner(); camera=ReadOnlyCamera()
            for j in range(4):
                v=camera.before if j==0 else camera.parameters()
                result['snapshots'].append(dict(connection=i, values=v, differences=differences(target,v)))
                time.sleep(.25)
        except Exception as e: result['errors'].append(repr(e))
        finally:
            if camera:
                try: camera.close()
                except Exception as e: result['errors'].append(repr(e))
                result['snapshots'].append(dict(connection=i,values=camera.after,differences=differences(target,camera.after)))
        if result['errors']: break
    result['passed']=len(result['snapshots'])==10 and not result['errors'] and not any(s['differences'] for s in result['snapshots'])
    result['source_sha256']=sha(__file__);write(out/'RESULT.json',result)
    print(json.dumps(dict(passed=result['passed'], errors=result['errors'],
        differences=[s['differences'] for s in result['snapshots']]),ensure_ascii=False))

def configure():
    no_owner()
    cfg=read(OLD/'CONFIG.json'); checkpoint=ROOT/'CRASH_01/CHECKPOINT.json'
    h=read(OLD/'GUI/heartbeat.json'); plan=read(TEST/'PLAN.json')
    restart=read(cfg['restart']['path'])
    restart.update(original_pid=PID,original_session=str(OLD/'GUI'),
        runtime_path=str(OLD/'GUI/runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],
        optical_config=h['optical_reference'],registration=None,expected_active=None,fault=None,
        requested_exposure_us=250.,no_auto_capture=True,no_auto_projection=True,
        old_process_clean_rollback_claimed=False)
    restart['evidence_sha256'][str(checkpoint)]=sha(checkpoint)
    write(ROOT/'restart_contract.json',restart)
    recovery=read(cfg['recovery']['path'])
    recovery.update(old_pid=PID,baseline_source=str(ROOT/'restart_contract.json'),
        baseline_source_sha256=sha(ROOT/'restart_contract.json'),planned_normal_restart=False,
        original_fault='E068 native ABRT; display factory binding incorrect; no clean rollback',
        manual_stop_requires_recheck=True,old_process_clean_rollback_claimed=False)
    write(ROOT/'recovery_record.json',recovery)
    caches=read(cfg['caches']['path'])
    caches['files']=dict(dmd=plan['input_sources']['A'],slm=ref(TEST/'SLM.npy'))
    write(ROOT/'PREVIEW_CACHES.json',caches)
    sources=dict(cfg['source_sha256'])
    for p in (ROOT/'run.py',ROOT/'display_binding.py',ROOT/'test_binding.py',ROOT/'recover.py',
              checkpoint,ROOT/'restart_contract.json',ROOT/'recovery_record.json',ROOT/'PREVIEW_CACHES.json'):
        sources[str(p)]=sha(p)
    contract=read(TEST/'CONTRACT.json')
    # Exact current fixed profile and all input-source pins remain unchanged.
    contract['source_sha256']=sources
    contract['research_test']['control_qualified']=False
    contract['crash_recovery']=dict(old_pid=PID,old_normal_exit=False,
        actual_factory_binding_corrected=True,native_root_cause_fully_proven=False,no_auto_resume=True)
    write(ROOT/'CONTRACT.json',contract)
    cfg.update(contract=ref(ROOT/'CONTRACT.json'),restart=ref(ROOT/'restart_contract.json'),
        recovery=ref(ROOT/'recovery_record.json'),caches=ref(ROOT/'PREVIEW_CACHES.json'),
        settings=plan['settings'],input_sources=plan['input_sources'],source_sha256=sources,
        crash_recovery=ref(checkpoint))
    cfg.pop('lifecycle_release',None)
    write(ROOT/'CONFIG.json',cfg)
    print(json.dumps(dict(config=ref(ROOT/'CONFIG.json'),hardware_IO=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['describe','prepare','camera-readonly','configure'],nargs='?',default='describe')
    p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='camera-readonly':camera_readonly(a.light_blocked_confirmed)
    elif a.action=='configure':configure()
    else:print('{"default_IO":0}')
