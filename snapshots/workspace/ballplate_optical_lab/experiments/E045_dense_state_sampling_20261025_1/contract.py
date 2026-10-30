"""Explicit file-only E045 dense-sampling contract and source archive builder.

Default import/CLI describes and reads no files. --prepare only binds existing
files, writes a NEW bundle and makes byte-for-byte source/data snapshots. It
never imports the GUI/core, runs a launcher, opens an SDK or controls devices.
No self-referential contract hash: requests add the finished contract reference
to exactly the source map written inside that contract.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SELF=Path(__file__).resolve();ROOT=SELF.parent;LAB=ROOT.parents[1]
TEST=SELF.with_name('test_contract.py')
PLAN_REF=dict(path=str(LAB/'experiments/E044_repeatable_sampling_20261025_1/PLAN_LONG_HOLD_01.json'),
    sha256='ee24867863abaab0a64cef21190be24c1ed2982d4995a3655ad66b541c6034f1')
PROFILE_REF=dict(path=str(LAB/'experiments/E043_alpha010_stability_20261025_1/PROFILE_RESTORED_EXPOSURE80_02.json'),
    sha256='716662b137ac3fcd0356488dde43bfeee4e059ce4e78f90ed0f5a59aa201eaf2')
FROZEN={
    str(LAB/'src/optical_lab/state_sampling.py'):'7c8b00ed1365c07f8372ea43b41cd82f6119015f70c2c44cf7009d84be1789cf',
    str(LAB/'tests/test_state_sampling.py'):'d47c9c74a5b92f93c3bd0425703abaa9da81af27dd079981600eec301285377b',
    str(LAB/'src/optical_lab/state_sampling_gui.py'):'c5683e8fe54503cef92b757adb4ef39794cbd7462d88a43f7298d745f8c5d717',
    str(LAB/'tests/test_state_sampling_gui.py'):'2725aa521d607e1d086c361f4295ace9d8abc549905b1e3078f653c87b67960f'}
INPUT_IDS=(0,1,2,'D09','D10','D11','D12')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path,digest):return dict(path=str(path),sha256=digest)


def pin(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'one exact absolute path/SHA reference required')
    path=Path(ref['path']);digest=ref['sha256']
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file()
        and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None and sha(path)==digest,
        'regular explicit file SHA differs: '+str(path))
    return path


def read(ref):return json.loads(pin(ref).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False,sort_keys=True);stream.write('\n')


def description():
    return dict(schema='E045-dense-contract-builder-description-1',offline_only=True,default_file_reads=0,
        GUI_or_SDK_actions=0,launches_started=0,original_plan=PLAN_REF,actual_profile=PROFILE_REF,
        allowed_input_ids=list(INPUT_IDS),maximum_frames=240,default_GUI_frames=120,
        RAW_peak_exclusive_ADU=240,minimum_post_ACK_ns=1_500_000_000,maximum_burst_seconds=60,
        new_output_direct_child_required=True,explicit_self_and_test_and_launcher_SHA_required=True,
        exact_request_source_closure='contract.source_sha256 plus finished contract path/SHA; no recursive own hash',
        legacy_quantitative_gate_unchanged=True,control_release_approved=False)


def build(*,launcher_ref,extra_sources):
    """Pure file reads; callers must explicitly pin this builder and its test."""
    import numpy as np
    require(isinstance(extra_sources,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in extra_sources.items()),
        'explicit additional absolute source path/SHA map required')
    require(str(SELF) in extra_sources and str(TEST) in extra_sources,
        'caller must explicitly pin the builder and its matching tests')
    launcher_path=pin(launcher_ref)
    require(launcher_path.parent==ROOT and launcher_path.suffix=='.py' and launcher_path not in (SELF,TEST),
        'one explicit E045 launcher source is required, never executed by this builder')
    plan=read(PLAN_REF);profile=read(PROFILE_REF)
    require(plan.get('schema')=='E044-long-hold-independent-sampling-1' and plan.get('profile')==PROFILE_REF,
        'fixed E044 plan and actual exposure80 profile provenance differ')
    require(profile.get('schema')=='optical-lab-current-public-profile-1'
        and profile.get('geometry_frozen') is False and profile.get('control_qualified') is False,
        'do not fabricate geometry/control qualification')
    active=deepcopy(profile['active']);roi=deepcopy(profile['actual_roi']);settings=deepcopy(profile['settings'])
    required=dict(DeviceSerialNumber='15328133',PixelFormat='Mono8',ExposureMode='Timed',ExposureAuto='Off',
        ExposureTime='74.20778274536133',GainAuto='Off',Gain='10.881464004516602',TriggerMode='On',
        TriggerSelector='FrameStart',TriggerSource='Line0',TriggerActivation='FallingEdge',AcquisitionMode='Continuous')
    require(all(active.get(k)==v for k,v in required.items()) and roi==dict(OffsetX=900,OffsetY=416,Width=552,Height=554)
        and all(active.get(k)==str(v) for k,v in roi.items()) and settings==dict(dmd='file',square=128,slm='file',
            full_panel=False,x=-410,y=4,focal_mm=400.,period=35),'full frozen actual camera/settings/ROI differ')
    sources={};allowed=[]
    def add(path,digest):
        path=str(path)
        require(path not in sources or sources[path]==digest,'contradictory source SHA binding: '+path)
        pin(reference(path,digest));sources[path]=digest
    for ref in (PLAN_REF,PROFILE_REF,launcher_ref):add(ref['path'],ref['sha256'])
    for path,digest in FROZEN.items():add(path,digest)
    for path,digest in extra_sources.items():add(path,digest)
    for input_id in INPUT_IDS:
        entry=plan['inputs'][str(input_id)];ref={k:entry['dmd'][k] for k in ('path','sha256')}
        require(entry.get('id')==input_id,'explicit input id/order differs')
        add(ref['path'],ref['sha256']);array=np.load(ref['path'],allow_pickle=False)
        require(array.shape==(600,600) and array.dtype==np.uint8 and np.isin(array,(0,255)).all()
            and 0<np.count_nonzero(array)<=3600,'exact sparse binary canonical600 DMD required')
        allowed.append(dict(input_id=input_id,**ref))
    fixed_slm={k:profile['caches']['slm'][k] for k in ('path','sha256')}
    require(plan['source_sha256'].get(fixed_slm['path'])==fixed_slm['sha256']
        and Path(fixed_slm['path']).name=='L00_canonical600.npy','fixed L00 source differs')
    add(fixed_slm['path'],fixed_slm['sha256']);phase=np.load(fixed_slm['path'],allow_pickle=False)
    require(phase.shape==(600,600) and phase.dtype==np.uint8,'fixed canonical600 native SLM required')
    return dict(schema='optical-lab-dense-research-contract-1',expected_active=active,
        maximum_frames=240,RAW_peak_exclusive_ADU=240,minimum_post_ACK_ns=1_500_000_000,
        maximum_burst_seconds=60,readback_policy='each_frame_before_and_after',
        legacy_quantitative_gate_unchanged=True,source_sha256=sources,
        fixed_context=dict(settings=settings,roi=roi,geometry_frozen=False,
            optical_reference=deepcopy(profile['optical_reference'])),fixed_slm=fixed_slm,allowed_dmd=allowed,
        profile_provenance=PROFILE_REF,original_plan_provenance=PLAN_REF,launcher_source=launcher_ref,
        original_display_roles=deepcopy(profile.get('roles',{})),new_owner_identity_is_runtime_only=True,
        no_automatic_projection_or_asset_load=True,research_only=True,control_release_approved=False,
        historical_source_maps_are_provenance_NOT_implicitly_imported=True,
        source_closure_rule='request exact source_sha256 plus this finished contract path/SHA')


def prepare(output,*,launcher_ref,extra_sources):
    output=Path(output)
    require(output.is_absolute() and output.parent==ROOT and output.resolve()==output and not output.exists()
        and not output.is_symlink(),'one NEW direct E045 output directory required')
    contract=build(launcher_ref=launcher_ref,extra_sources=extra_sources)
    output.mkdir();path=output/'CONTRACT.json';write(path,contract);contract_ref=reference(path,sha(path))
    request_sources=dict(contract['source_sha256'],**{str(path):contract_ref['sha256']})
    archive=output/'SOURCE_ARCHIVE';archive.mkdir();index=[]
    for filename,digest in sorted(request_sources.items()):
        original=pin(reference(filename,digest));blob=original.read_bytes()
        require(hashlib.sha256(blob).hexdigest()==digest,'source changed during byte snapshot')
        target=archive/(digest+'__'+original.name)
        if not target.exists():
            with target.open('xb') as stream:stream.write(blob)
        require(sha(target)==digest,'source archive byte copy differs')
        index.append(dict(original=reference(original,digest),archived=reference(target,digest)))
    write(output/'SOURCE_INDEX.json',dict(schema='E045-complete-byte-source-index-1',files=index,
        includes_finished_contract=True,recursive_self_contract_SHA=False,original_files_modified=False))
    bundle=dict(schema='E045-dense-contract-bundle-1',contract_ref=contract_ref,request_source_sha256=request_sources,
        source_index=reference(output/'SOURCE_INDEX.json',sha(output/'SOURCE_INDEX.json')),
        GUI_or_SDK_actions=0,launcher_executed=False,control_release_approved=False)
    write(output/'BUNDLE.json',bundle)
    products={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'MANIFEST.json',dict(schema='E045-dense-contract-manifest-1',source_sha256=request_sources,
        file_sha256=products,manifest_excludes_itself=True))
    return dict(contract_ref=contract_ref,bundle=reference(output/'BUNDLE.json',sha(output/'BUNDLE.json')),
        manifest=reference(output/'MANIFEST.json',sha(output/'MANIFEST.json')),archived_files=len(index),
        GUI_or_SDK_actions=0,launcher_executed=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--launcher',type=Path);parser.add_argument('--launcher-sha256')
    parser.add_argument('--extra-sources-json',type=Path);parser.add_argument('--extra-sources-sha256')
    args=parser.parse_args(argv)
    if not args.prepare:print(json.dumps(description(),ensure_ascii=False));return 0
    if any(v is None for v in (args.output,args.launcher,args.launcher_sha256,args.extra_sources_json,args.extra_sources_sha256)):
        parser.error('explicit NEW output, launcher SHA and additional source map file SHA required')
    ref=reference(args.extra_sources_json,args.extra_sources_sha256);extra=read(ref)
    require(isinstance(extra,dict),'additional source file must be one absolute path/SHA map')
    require(str(args.extra_sources_json) not in extra,'source-map file must not recursively bind its own SHA')
    extra[str(args.extra_sources_json)]=args.extra_sources_sha256
    result=prepare(args.output,launcher_ref=reference(args.launcher,args.launcher_sha256),extra_sources=extra)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
