"""E072 rollback with a strictly bound 1920x1080 display profile.

No hardware action on import. No controller writes. Old sources stay unchanged.
Canonical DMD600 remains centered, unscaled; SLM is mirrored exactly once.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
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
SOURCE = LAB/'experiments/E072_ball_size_20261026_1'
OLD = LAB/'experiments/E073_slm_selector_20261026_1'
LEGACY = LAB.parent/'ballplate_fourframe96_deploy'
UNIT = 'ballplate-optical-dmd1920-20261026-e074.service'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def no_owner():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid(): continue
        try: args = (p/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if b'--hardware' in args and any(str(LAB).encode() in a or str(LEGACY).encode() in a for a in args):
            raise RuntimeError('another hardware owner exists: '+p.name)

def install_display(profile=None):
    """Fresh-process backend only; keep identity/layout checks, change exact size."""
    if 'display_linux' in sys.modules:
        raise RuntimeError('display adapter must be installed before legacy imports')
    module = load('display_linux', LEGACY/'live_manual_v17/display_linux.py')
    module.ROLES = dict(dmd=('DP-1-2','DLP6500 DP',(1920,1080)),
                        slm=('DP-1-1','XILINX HDMI',(1920,1080)))
    original = module.read_roles
    if profile is not None:
        def checked():
            current = original()
            module.same_signals(profile, current)
            return current
        module.read_roles = checked
    return module

def dmd_panel(image):
    import numpy as np
    image = np.asarray(image)
    if image.shape != (600,600) or image.dtype != np.uint8 or not np.isin(image,(0,255)).all():
        raise ValueError('binary uint8 canonical600 required; no implicit resize')
    panel = np.zeros((1080,1920), np.uint8)
    panel[240:840,660:1260] = image
    return panel

def show_outputs(self, dmd, slm):
    if not self.hardware: return
    import numpy as np
    panel = dmd_panel(dmd)
    if slm.shape != (1080,1920) or slm.dtype != np.uint8:
        raise ValueError('native SLM uint8 required')
    physical = np.ascontiguousarray(np.fliplr(slm))
    # The frozen modulator has a centered 600-square child for the DMD.
    # Do not feed the full panel through that child: that would shrink it again.
    self.dmd_output.update_image(dmd)
    self.slm_output.update_image(physical)
    self.expected = dict(dmd=panel, slm=physical)
    self.last_apply=time.monotonic(); self.revision+=1; self.mismatches=0
    self.last_verified_revision=None
    self.last_arrays=dict(dmd=dmd.copy(),slm=slm.copy())
    self.record(dict(state='manual_display_requested',revision=self.revision,settings=asdict(self.setting()),
        dmd_sha256=hashlib.sha256(dmd.tobytes()).hexdigest(),
        slm_canonical_sha256=hashlib.sha256(slm.tobytes()).hexdigest(),
        dmd_nonzero=bool(dmd.any()),slm_nonzero=bool(slm.any()),
        blocked_reason=self.projection_block_reason(),SLM_horizontal_flip_count=1,
        frame_pattern_association_verified=False,dmd_native_size=[1920,1080],
        dmd_content_rect=[660,240,600,600],dmd_resampling=False))

def prepare():
    no_owner()
    if (ROOT/'CONFIG.json').exists(): raise RuntimeError('prepare once only')
    cfg = read(SOURCE/'CONFIG.json')
    for p,d in cfg['source_sha256'].items():
        if sha(p)!=d: raise RuntimeError('preserved source changed: '+p)
    status = subprocess.check_output(['systemctl','--user','show',
        'ballplate-optical-slm-select-20261026-e073.service',
        '-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True)
    values=dict(line.split('=',1) for line in status.splitlines())
    if values != dict(MainPID='0',Result='success',ExecMainStatus='0'):
        raise RuntimeError('old service did not exit normally')
    events=[json.loads(line) for line in (OLD/'GUI/events.jsonl').read_text().splitlines()]
    exits=[e for e in events if e.get('state')=='operator_confirmed_light_blocked_exit' and e.get('closed_safely') is True]
    if len(exits)!=1: raise RuntimeError('unique safe-exit evidence required')
    display=install_display()
    profile=display.read_roles()
    expected_edid={k:v['EDID_sha256'] for k,v in read(cfg['restart']['path'])['display_roles'].items()}
    if any(profile[k]['EDID_sha256']!=expected_edid[k] for k in profile):
        raise RuntimeError('physical output EDID changed')
    write(ROOT/'DISPLAY_PROFILE.json',profile)
    with (ROOT/'XRANDR.txt').open('x') as f:
        f.write(subprocess.check_output(['xrandr','--current','--verbose'],text=True,timeout=3))
    sys.path[:0]=[str(LEGACY),str(LEGACY/'live_manual_v17')]
    from camera_recheck import ReadOnlyCamera, verify_baseline
    baseline=read(cfg['recovery']['path'])['camera_baseline']
    result=verify_baseline(baseline,no_owner,camera_factory=ReadOnlyCamera)
    write(ROOT/'COLD_READONLY.json',result)
    passed=(result.get('passed') is True and result.get('connections')==2
        and len(result.get('snapshots',[]))==10 and result.get('parameter_writes')==0
        and result.get('acquisition_started') is False and result.get('errors')==[]
        and all(r.get('values')==baseline and r.get('differences')=={} for r in result['snapshots']))
    write(ROOT/'LIFECYCLE_RESULT.json',dict(completed=passed,old_normal_exit_verified=True,
        cold_baseline_verified=passed,physical_shielding_confirmed=True,
        normal_exit_event=exits[0],service=values,cold_readonly=ref(ROOT/'COLD_READONLY.json'),
        old_heartbeat=ref(OLD/'GUI/heartbeat.json'),old_events=ref(OLD/'GUI/events.jsonl')))
    if not passed: raise RuntimeError('camera standby mismatch; no window created')
    restart=read(cfg['restart']['path'])
    restart.update(display_roles=profile,registration=None,expected_active=None,
        original_pid=2614059,original_session=str(OLD/'GUI'),saved_unix_ns=time.time_ns(),
        no_auto_capture=True,no_auto_projection=True)
    write(ROOT/'restart_contract.json',restart)
    recovery=read(cfg['recovery']['path'])
    recovery.update(old_pid=2614059,baseline_source=str(ROOT/'restart_contract.json'),
        baseline_source_sha256=sha(ROOT/'restart_contract.json'),cold_baseline_readonly_verification_passed=True)
    write(ROOT/'recovery_record.json',recovery)
    cfg.update(restart=ref(ROOT/'restart_contract.json'),recovery=ref(ROOT/'recovery_record.json'),
        lifecycle_release=ref(ROOT/'LIFECYCLE_RESULT.json'),replaced_owner_pid=2614059,
        display_profile=ref(ROOT/'DISPLAY_PROFILE.json'))
    cfg['retained_test_controls']['head_enabled']=False
    sources=dict(cfg['source_sha256'])
    for p in (Path(__file__),SOURCE/'run.py',ROOT/'DISPLAY_PROFILE.json'):
        sources[str(p.resolve())]=sha(p)
    cfg['source_sha256']=sources
    contract=read(cfg['contract']['path'])
    contract.update(source_sha256=sources,original_display_roles=profile,
        dmd_presentation=dict(native=[1920,1080],content_rect=[660,240,600,600],resampling=False),
        old_geometry_invalidated=True,control_release_approved=False)
    write(ROOT/'CONTRACT.json',contract);cfg['contract']=ref(ROOT/'CONTRACT.json')
    write(ROOT/'CONFIG.json',cfg)
    print(json.dumps(dict(prepared=True,profile=profile,config=ref(ROOT/'CONFIG.json')),ensure_ascii=False))

def run_gui():
    cfg=read(ROOT/'CONFIG.json')
    install_display(read(cfg['display_profile']['path']))
    from optical_lab import preview
    original_legacy=preview._load_legacy
    def legacy(path):
        window=original_legacy(path)
        import realtime_gui
        realtime_gui.RealtimeWindow.show_outputs=show_outputs
        return window
    preview._load_legacy=legacy
    original_geometry=preview.attach_manual_geometry_prior
    def geometry(window, prior):
        result=original_geometry(window,prior)
        window.manual_geometry_prior_disable('DMD changed to native1920; old geometry not transferable')
        window.setWindowTitle('光路工作台 · 八区稳定性／球尺寸 · DMD1920回退版')
        window.raw_workbench['note'].setText('DMD1920×1080；原600方图居中、不拉伸。旧倍率/部署资格已失效；先重新对齐。无SLM选择扩展。')
        return result
    preview.attach_manual_geometry_prior=geometry
    runner=load('e074_stable_runner',SOURCE/'run.py')
    runner.ROOT=ROOT;runner.CATALOG=SOURCE/'catalog.py'
    original_load=runner.load
    def routed_load(name,path):
        module=original_load(name,path)
        if name=='e072_frozen_composition':
            frozen_load=module.load
            def source_load(n,p):
                p=Path(p)
                if p.is_relative_to(ROOT/'staged'): p=SOURCE/'staged'/p.relative_to(ROOT/'staged')
                return frozen_load(n,p)
            module.load=source_load
        return module
    runner.load=routed_load
    runner.main()

if __name__=='__main__':
    if sys.argv[1:]==['--prepare','--light-blocked-confirmed']: prepare()
    elif '--hardware' in sys.argv or '--offline-smoke' in sys.argv: run_gui()
    else: print('{"default_IO":0}')
