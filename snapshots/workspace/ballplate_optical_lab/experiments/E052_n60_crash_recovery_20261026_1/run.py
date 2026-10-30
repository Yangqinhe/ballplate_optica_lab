"""Fresh crash recovery composition; no old source edits or automatic test.

Default is description only. prepare is file-only forensic/retention work.
Hardware requires explicit current shielding, exact source hashes, absent old
PID and single-owner checks. Original camera baseline remains pending recheck.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
OLD = LAB/'experiments/E051_brightness_warning_20261026_1'
TEST = OLD/'TESTS/TEST_20261026_005308_939161'
LEGACY = LAB.parent/'ballplate_fourframe96_deploy'
MODULATOR = LAB.parent/'ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015/src/sdon/hardware/modulator.py'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(p), sha256=sha(p))
def write(p, value):
    with Path(p).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    if '.' in name:
        parent, child = name.rsplit('.', 1)
        setattr(sys.modules[parent], child, m)
    spec.loader.exec_module(m)
    return m


def prepare():
    if (ROOT/'CONFIG.json').exists(): raise ValueError('prepared once only')
    h = read(OLD/'GUI/heartbeat.json'); plan = read(TEST/'PLAN.json')
    if h['pid'] != 1893915 or Path('/proc/1893915').exists():
        raise RuntimeError('exact crashed owner must be absent')
    contract = read(TEST/'CONTRACT.json')
    for p, d in contract['source_sha256'].items():
        if sha(p) != d: raise RuntimeError('old source changed: '+p)
    write(ROOT/'CRASH_CHECKPOINT.json', dict(heartbeat=h, plan=plan,
        raw_status=read(TEST/'RAW/STATUS.json'), completed_mean_count=len(list(TEST.glob('MEAN_*/READOUT.json'))),
        completed_batch_count=len(list(TEST.glob('RAW/results/*/BATCH_RESULT.json'))),
        old_normal_exit=False, old_allzero_NOT_proven=True, partial_run_not_accepted=True,
        references=[ref(OLD/'GUI/heartbeat.json'), ref(TEST/'PLAN.json'),ref(TEST/'RAW/STATUS.json')]))
    journal = subprocess.check_output(['journalctl','--user','-u',
        'ballplate-optical-brightness-20261026-e051.service','-n','120','--no-pager'],text=True)
    with (ROOT/'CRASH_JOURNAL.txt').open('x') as f: f.write(journal)
    retained = read(OLD/'RETENTION_01/RESULT.json')
    restart = read(retained['restart_contract']['path'])
    restart.update(original_pid=1893915,original_session=str(OLD/'GUI'),
        runtime_path=str(OLD/'GUI/runtime_sources.json'),saved_unix_ns=time.time_ns(),
        settings=dict(h['manual_settings'],dmd='black',slm='zero'),roi=h['actual_roi'],
        optical_config=h['optical_reference'],registration=None,expected_active=None,fault=None,
        requested_exposure_us=contract['research_test']['camera_requested']['exposure_us'],
        no_auto_capture=True,no_auto_projection=True,old_process_clean_rollback_claimed=False)
    restart['evidence_sha256'][str(ROOT/'CRASH_CHECKPOINT.json')]=sha(ROOT/'CRASH_CHECKPOINT.json')
    write(ROOT/'restart_contract.json',restart)
    recovery = read(retained['recovery_record']['path'])
    recovery.update(old_pid=1893915,baseline_source=str(ROOT/'restart_contract.json'),
        baseline_source_sha256=sha(ROOT/'restart_contract.json'),planned_normal_restart=False,
        original_fault='E051 native SIGABRT during N60 A/B test; no clean camera rollback',
        manual_stop_requires_recheck=True,old_process_clean_rollback_claimed=False)
    write(ROOT/'recovery_record.json',recovery)
    sources=dict(contract['source_sha256'])
    for p in (Path(__file__).resolve(),LAB/'src/optical_lab/queued_display.py',ROOT/'CRASH_CHECKPOINT.json',
              ROOT/'restart_contract.json',ROOT/'recovery_record.json'):
        sources[str(p)]=sha(p)
    contract['source_sha256']=sources
    contract['research_test']['control_qualified']=False
    contract['crash_recovery']=dict(old_pid=1893915,old_normal_exit=False,
        synchronous_repaint_removed=True,root_cause_proven=False,no_automatic_resume=True)
    write(ROOT/'CONTRACT.json',contract)
    write(ROOT/'CONFIG.json',dict(contract=ref(ROOT/'CONTRACT.json'),
        restart=ref(ROOT/'restart_contract.json'),recovery=ref(ROOT/'recovery_record.json'),
        display=retained['display_state'],geometry=retained['manual_geometry'],caches=retained['preview_caches'],
        settings=plan['settings'],camera_requested=contract['research_test']['camera_requested'],
        expected_active=h['camera_active_readback'],input_sources=plan['input_sources'],
        source_sha256=sources,no_auto_capture=True,no_auto_projection=True))
    print(json.dumps(dict(prepared=True,config=ref(ROOT/'CONFIG.json'),hardware_actions=0)))


def execute(a):
    if sha(ROOT/'CONFIG.json') != a.config_sha256: raise ValueError('explicit config SHA required')
    cfg=read(ROOT/'CONFIG.json')
    for p,d in cfg['source_sha256'].items():
        if sha(p)!=d: raise RuntimeError('frozen source changed: '+p)
    for key in ('contract','restart','recovery','display','geometry','caches'):
        if sha(cfg[key]['path'])!=cfg[key]['sha256']: raise RuntimeError('configuration changed: '+key)
    if a.hardware:
        if not a.light_blocked_confirmed or Path('/proc/1893915').exists():
            raise RuntimeError('fresh shielding and absent exact crashed owner required')
        # Include all composed lab entrypoints, not only the old GUI module.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid(): continue
            try: argv=(p/'cmdline').read_bytes().split(b'\0')
            except OSError: continue
            if b'--hardware' in argv and any(str(LAB).encode() in x or str(LEGACY).encode() in x for x in argv):
                raise RuntimeError('another hardware GUI exists: '+p.name)
    elif os.environ.get('QT_QPA_PLATFORM')!='offscreen' or a.light_blocked_confirmed:
        raise RuntimeError('offline check requires offscreen and cannot claim shielding')
    output=ROOT/('GUI' if a.hardware else 'QA_GUI')
    spool=ROOT/('GUI_SPOOL' if a.hardware else 'QA_SPOOL')
    if output.exists() or spool.exists(): raise RuntimeError('fresh outputs only')
    from optical_lab import preview,exposure,display
    # All imports below remain before window construction; no live injection.
    for name in ('state_sampling','stability','stability_gui'):
        load('optical_lab.'+name,OLD/'staged/optical_lab'/f'{name}.py')
    from optical_lab.state_sampling_gui import attach
    from optical_lab.region_overlay import attach as overlay_attach,load_layout
    from optical_lab.queued_display import queued_modulator
    import optical_lab.stability_gui as sg
    dense=load('E052_frozen_dense',LAB/'experiments/E045_dense_state_sampling_20261025_1/run.py')
    mask=LAB/'experiments/E035_mapping_night_20261024_1/REFIT_L00_BALANCED_01/BALANCED_CCD_AXES_NATIVE8.npy'
    layout=load_layout(mask,sha(mask),read(cfg['contract']['path'])['fixed_context']['roi'])
    original_load=preview._load_legacy
    stack=ExitStack()
    def load_original(path):
        Window=original_load(path)
        import sdon.hardware.modulator as mod
        if Path(mod.__file__).resolve()!=MODULATOR or sha(MODULATOR)!=cfg['source_sha256'][str(MODULATOR)]:
            raise RuntimeError('unexpected output implementation')
        stack.enter_context(queued_modulator(mod))
        return Window
    preview._load_legacy=load_original
    original_display=display.attach_manual_display
    def display_attach(w,initial=None):
        panel=original_display(w,initial);overlay_attach(w,layout);return panel
    display.attach_manual_display=display_attach
    def attached(w,sampler):
        flow=sg.attach(w,cfg['contract'],ROOT/'TESTS',ref(mask),read(cfg['contract']['path'])['fixed_context']['roi'])
        mapping=dict(frames='frames',repeats='repeats',switch='switch_inputs',settle='settle_seconds',
                     interval='interval_seconds',software_gain='software_gain',black='black_ADU',gamma='gamma')
        for key,field in mapping.items():
            widget=flow.widgets[key];value=cfg['settings'][field]
            widget.setChecked(value) if key=='switch' else widget.setValue(value)
        flow.widgets['exposure'].setValue(cfg['camera_requested']['exposure_us'])
        flow.widgets['gain'].setValue(cfg['camera_requested']['gain_db'])
        for key,label in (('input_a','A'),('input_b','B')):
            hits=[i for i,asset in enumerate(flow.assets) if asset['ref']==cfg['input_sources'][label]]
            if len(hits)!=1: raise RuntimeError('retained test input not unique')
            flow.widgets[key].setCurrentIndex(hits[0])
        # Do not rerun or relabel the incomplete crash-prefix.
        flow.widgets['status'].setText('N60换图测试崩溃后恢复；上一轮不完整，原始数据保留。未自动测试。')
        if a.hardware:
            write(output/'dense_sampling_extension.json',dict(owner_pid=os.getpid(),contract_ref=cfg['contract'],
                source_sha256=cfg['source_sha256'],automatic_capture=False,automatic_projection=False))
    args=['--legacy-root',str(LEGACY),'--output',str(output),
          '--restart-state',cfg['restart']['path'],'--recovery-record',cfg['recovery']['path'],
          '--display-state',cfg['display']['path'],'--display-state-sha256',cfg['display']['sha256'],
          '--manual-geometry-state',cfg['geometry']['path'],'--manual-geometry-sha256',cfg['geometry']['sha256'],
          '--preview-caches',cfg['caches']['path'],'--preview-caches-sha256',cfg['caches']['sha256']]
    args+=['--hardware','--light-blocked-confirmed'] if a.hardware else ['--smoke-test']
    try:
        with stack,dense.startup_composition(exposure,display,attach,cfg['contract'],spool,read(cfg['display']['path']),attached):
            return preview.main(args)
    finally:
        preview._load_legacy=original_load;display.attach_manual_display=original_display


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--hardware',action='store_true');g.add_argument('--offline-smoke',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--config-sha256')
    a=p.parse_args()
    if a.prepare:return prepare()
    if not(a.hardware or a.offline_smoke):print('{"default_IO":0}');return
    # stderr is retained by systemd; fatal native signals now include Python stacks.
    faulthandler.enable(all_threads=True)
    return execute(a)

if __name__=='__main__':main()
