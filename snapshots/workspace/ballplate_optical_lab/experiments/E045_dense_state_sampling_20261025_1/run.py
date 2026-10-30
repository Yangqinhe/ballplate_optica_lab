"""Startup-only dense sampler composition; default describe has no file/UI/SDK IO.

This reuses the preserved preview entry and adds no GUI inheritance chain. The
original worker is decorated before it starts. Hardware execution requires a
separately verified normal exit and cold standby readback, plus fresh physical
shielding. No camera start, node change, arm or asset load is automated here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

SELF=Path(__file__).resolve();ROOT=SELF.parent;LAB=ROOT.parents[1]
LEGACY=LAB.parent/'ballplate_fourframe96_deploy'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'explicit path/SHA reference required')
    p=Path(ref['path']);d=ref['sha256']
    require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink()
        and isinstance(d,str) and re.fullmatch('[0-9a-f]{64}',d) and sha(p)==d,'fixed regular source SHA differs: '+str(p))
    return p


def read_ref(ref):return json.loads(pin(ref).read_text())


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def description():
    return dict(schema='E045-startup-composition-1',default_file_UI_SDK_IO=0,
        existing_preview_reused=True,new_GUI_inheritance=False,new_SDK_owner=False,
        camera_start=False,camera_node_writes=0,automatic_projection=False,
        default_frames=120,maximum_frames=240,actual_capture_rate='must be measured',
        hardware_requires=['fresh physical shielding','normal old owner exit',
            'two cold readonly connections / ten exact standby snapshots'])


def validate(contract_ref,retention_ref,*,self_sha256):
    pin(dict(path=str(SELF),sha256=self_sha256))
    contract=read_ref(contract_ref);retained=read_ref(retention_ref)
    require(contract.get('schema')=='optical-lab-dense-research-contract-1'
        and contract.get('legacy_quantitative_gate_unchanged') is True,'explicit independent RAW research contract required')
    sources=contract['source_sha256']
    require(sources.get(str(SELF))==self_sha256,'running launcher must be source-bound')
    for path,digest in sources.items():pin(dict(path=path,sha256=digest))
    require(retained.get('status')=='PLANNED_FILES_ONLY_NOT_LIFECYCLE_RELEASE'
        and retained.get('lifecycle_release_approved') is False,'explicit unchanged planned-retention evidence required')
    base=pin(retention_ref).parent
    manifest=json.loads((base/'MANIFEST.json').read_text())
    for name,digest in manifest['file_sha256'].items():
        target=base/name
        require(target.resolve().is_relative_to(base) and sha(target)==digest,'retention artifact differs: '+name)
    for path,digest in manifest['original_source_sha256'].items():pin(dict(path=path,sha256=digest))
    restart=read_ref(retained['restart_contract']);recovery=read_ref(retained['recovery_record'])
    caches=read_ref(retained['preview_caches']);display=read_ref(retained['display_state'])
    pin(retained['manual_geometry'])
    context=json.loads((base/'EXPECTED_CAPTURE_CONTEXT.json').read_text())
    require(contract['expected_active']==context['active']==retained['retained_current_active']
        and contract['fixed_context']=={k:context[k] for k in ('settings','roi','geometry_frozen','optical_reference')},
        'retained current optical context and dense research profile differ')
    require(restart['original_fault_before']==recovery['camera_baseline']
        and restart['original_fault_before']['TriggerMode']=='Off'
        and restart['registration'] is None and restart['expected_active'] is None
        and restart['settings']['dmd']=='black' and restart['settings']['slm']=='zero',
        'unchanged standby, no manufactured calibration and black initialization required')
    require(caches['files']['slm']==contract['fixed_slm']
        and caches['files']['dmd'] in [{k:a[k] for k in ('path','sha256')} for a in contract['allowed_dmd']],
        'retained current file caches must be exact permitted assets')
    for reference in caches['files'].values():pin(reference)
    # Explicitly bind retained evidence as well as runtime modules. The contract
    # generator receives these files through its extra_sources map.
    require(sources.get(str(pin(retention_ref)))==retention_ref['sha256'], 'retention result must be source-bound')
    for module in ('preview','exposure','capture_timeout','display','state_sampling','state_sampling_gui'):
        path=LAB/'src/optical_lab'/('%s.py'%module)
        require(sources.get(str(path))==sha(path),'startup module missing from exact source closure: '+module)
    return dict(contract=contract,retained=retained,display=display,restart=restart,recovery=recovery)


def validate_lifecycle(ref,retention_ref,retained):
    result=read_ref(ref)
    require(result.get('schema')=='E045-normal-shielded-release-1'
        and result.get('physical_shielding_confirmed') is True
        and result.get('old_normal_exit_verified') is True and result.get('cold_baseline_verified') is True
        and result.get('forced_kill') is False and result.get('camera_readonly_connections')==2
        and result.get('camera_readonly_snapshots')==10 and result.get('retention_ref')==retention_ref
        and result.get('parameter_writes')==0 and result.get('acquisition_started') is False
        and result.get('errors')==[],
        'normal shielded exit and exact 2/10 cold readback proof required')
    owner=retained['original_owner']
    require(result.get('old_owner')==dict(pid=owner['pid'],start_ticks=owner['start_ticks'])
        and result.get('standby_baseline')==read_ref(retained['recovery_record'])['camera_baseline'],
        'lifecycle identity or original standby baseline differs')
    for group in ('source_sha256','evidence_sha256'):
        require(isinstance(result.get(group),dict) and result[group], 'nonempty lifecycle '+group+' required')
        for path,digest in result[group].items():pin(dict(path=path,sha256=digest))
    return result


def owner_argv(argv,cwd):
    if '--hardware' not in argv:return False
    if 'optical_lab.preview' in argv:return True
    if any(re.fullmatch(r'live_[A-Za-z0-9_]+\.direct_alignment_gui',a) for a in argv):return True
    for word in argv:
        if not word.endswith('.py'):continue
        path=(Path(cwd)/word).resolve()
        if path==SELF or (path.is_relative_to(LEGACY) and path.name=='direct_alignment_gui.py'):return True
    return False


def no_other_owner():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            argv=[s.decode() for s in (entry/'cmdline').read_bytes().split(b'\0') if s]
            cwd=(entry/'cwd').resolve()
        except (OSError,UnicodeError):continue
        if owner_argv(argv,cwd):
            raise RuntimeError('another optical owner still exists: '+entry.name)


@contextmanager
def startup_composition(exposure,display,attach,contract_ref,spool,retained_display,on_attached):
    """Only fresh-process construction hooks, undone when preview main exits."""
    original=exposure.attach_exposure_warnings
    original_display=display.attach_manual_display
    calls=[]
    def after_exposure(window):
        require(not calls,'single fresh window composition required')
        result=original(window)
        controller=attach(window,contract_ref,spool)
        calls.append(controller);on_attached(window,controller)
        return result
    def restore_display(window,initial=None):
        panel=original_display(window,initial)
        fc=retained_display.get('preview_filter')
        if fc is not None:
            controls=panel.controls
            index=controls.filter_mode.findData(fc['mode'])
            require(index>=0,'retained preview filter mode unavailable')
            controls.filter_mode.setCurrentIndex(index)
            controls.filter_alpha.setValue(fc['alpha']);controls.filter_original.setChecked(fc['original'])
            panel.temporal.reset()  # Never carry previous CCD history into the new owner.
        return panel
    exposure.attach_exposure_warnings=after_exposure;display.attach_manual_display=restore_display
    try:yield
    finally:
        exposure.attach_exposure_warnings=original;display.attach_manual_display=original_display


def execute(args):
    contract_ref=dict(path=str(args.contract),sha256=args.contract_sha256)
    retention_ref=dict(path=str(args.retention),sha256=args.retention_sha256)
    data=validate(contract_ref,retention_ref,self_sha256=args.source_sha256)
    output=args.output;spool=args.spool
    require(output.is_absolute() and spool.is_absolute() and output.parent==ROOT and spool.parent==ROOT
        and output!=spool and not output.exists() and not spool.exists(),'two NEW explicit direct E045 GUI/spool paths required')
    if args.hardware:
        require(args.light_blocked_confirmed and args.lifecycle and args.lifecycle_sha256,
            'fresh physical shielding plus explicit lifecycle result required')
        validate_lifecycle(dict(path=str(args.lifecycle),sha256=args.lifecycle_sha256),retention_ref,data['retained'])
        no_other_owner()
    else:
        require(args.offline_smoke and not args.light_blocked_confirmed and args.lifecycle is None,
            'offline smoke cannot claim physical shielding or lifecycle release')
    # Lazy imports: default description and fixture tests never touch Qt/SDK.
    sys.path.insert(0,str(LAB/'src'))
    from optical_lab import preview,exposure,display
    from optical_lab.state_sampling_gui import attach
    r=data['retained']
    command=['--legacy-root',str(LEGACY),'--output',str(output),
        '--restart-state',r['restart_contract']['path'],'--recovery-record',r['recovery_record']['path'],
        '--display-state',r['display_state']['path'],'--display-state-sha256',r['display_state']['sha256'],
        '--manual-geometry-state',r['manual_geometry']['path'],'--manual-geometry-sha256',r['manual_geometry']['sha256'],
        '--preview-caches',r['preview_caches']['path'],'--preview-caches-sha256',r['preview_caches']['sha256']]
    command+=['--hardware','--light-blocked-confirmed'] if args.hardware else ['--smoke-test']
    def attached(window,controller):
        if not args.hardware:return
        write(output/'dense_sampling_extension.json',dict(description(),unix_ns=time.time_ns(),
            owner_pid=os.getpid(),owner_start_ticks=controller.owner_start,contract_ref=contract_ref,
            retention_ref=retention_ref,lifecycle_ref=dict(path=str(args.lifecycle),sha256=args.lifecycle_sha256),
            source_sha256=data['contract']['source_sha256'],sampler_spool=str(spool),
            startup_only_attached_before_timeout_adapter=True,preview_filter_retained_display_only=True,
            live_camera_gain_NOT_automatically_restored=True,old_preview_metadata_flags_unchanged=True,
            physical_shielding_confirmed=True,control_release_approved=False))
    with startup_composition(exposure,display,attach,contract_ref,spool,data['display'],attached):
        return preview.main(command)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--hardware',action='store_true');mode.add_argument('--offline-smoke',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--source-sha256')
    for key in ('contract','retention','lifecycle'):
        p.add_argument('--'+key,type=Path);p.add_argument('--'+key+'-sha256')
    p.add_argument('--output',type=Path);p.add_argument('--spool',type=Path);a=p.parse_args(argv)
    if not a.hardware and not a.offline_smoke:
        require(all(v is None or v is False for v in vars(a).values()),'default describe takes no execution arguments')
        print(json.dumps(description(),ensure_ascii=False));return 0
    require(all(getattr(a,key) is not None for key in ('source_sha256','contract','contract_sha256','retention',
        'retention_sha256','output','spool')),'explicit source/contract/retention/NEW output/spool required')
    return execute(a)


if __name__=='__main__':raise SystemExit(main())
