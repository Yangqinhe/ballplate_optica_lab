"""Explicitly approved EXACT observed TriggerMode-coupled standby transition.

Import is stdlib only. No generic exposure allowance, no ExposureTime SDK
write, no active On-exposure qualification and no frozen-source edits.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
OUTPUT=BASE/'APPROVED03'
ORIGINAL=BASE/'run.py'
ORIGINAL_SHA='15ee5e16b69c8b23cde4dbe3f1e844840afb250c9f2f9edc58117cda9fce4526'
OBSERVED=BASE/'DIAGNOSTIC02/OBSERVED_01.json'
OBSERVED_SHA='d4d70b161004dabeda54194ca1cc30bef08b8815a7ebf3328b69f4e18c6e89b5'
DIAGNOSTIC=BASE/'DIAGNOSTIC02/RESULT.json'
DIAGNOSTIC_SHA='787f449065151515118413824cfed1f02459e677015c33c54868224fdf5c8e45'
ON_EXPOSURE='74.20778274536133'
OFF_EXPOSURE='67.17443466186523'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
def differences(before,after):
    return {k:dict(before=before.get(k),after=after.get(k)) for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}


def validate_evidence(observed,diagnostic,approved_on):
    on=observed['camera_before'];off=observed['camera_after']
    expected=dict(TriggerMode=dict(before='On',after='Off'),
        ExposureTime=dict(before=ON_EXPOSURE,after=OFF_EXPOSURE),
        AcquisitionFrameRate=dict(before=None,after='90'))
    if (on!=approved_on or set(on)!=set(off) or differences(on,off)!=expected or observed.get('differences')!=expected or
            observed.get('complete_parameter_dicts')is not True or observed.get('validation_not_relaxed')is not True or
            diagnostic.get('passed')is not False or diagnostic.get('camera_before')!=on or
            diagnostic.get('rollback',{}).get('passed')is not True or diagnostic['rollback'].get('actual')!=on or
            diagnostic.get('after_original_close')!=on or diagnostic.get('final_original_outputs_zero_verified')is not True or
            diagnostic.get('original_close_error') or diagnostic.get('final_original_close_error') or diagnostic.get('commit_close')is not None):
        raise ValueError('exact failed diagnostic, approved observation and complete original On rollback required')
    return copy.deepcopy(on),copy.deepcopy(off),expected


def exact_validator(approved_on,approved_off):
    def validate(before,after):
        if before!=approved_on or after!=approved_off:
            raise ValueError('only exact explicitly approved full On-to-Off dictionaries accepted')
        return copy.deepcopy(after),{k:v for k,v in differences(before,after).items() if k!='TriggerMode'}
    return validate


def annotated_change(original):
    """Preserve original return/exception; make the shared report truthful."""
    def change(camera,expected,guard,result):
        succeeded=False
        try:
            target=original(camera,expected,guard,result);succeeded=True;return target
        finally:
            if 'camera_after' in result and 'camera_before' in result:
                diff=differences(result['camera_before'],result['camera_after'])
                result['formal_other_fields_exact_unchanged']=not(set(diff)-{'TriggerMode'})
            result['only_explicitly_approved_transition']=succeeded
            result['exposure_node_written']=False
            result['active_On_exposure_recheck_required']=ON_EXPOSURE
    return change


def execute(blocked):
    if not blocked:raise ValueError('continued explicit physical shielding required')
    bindings={str(ORIGINAL):ORIGINAL_SHA,str(OBSERVED):OBSERVED_SHA,str(DIAGNOSTIC):DIAGNOSTIC_SHA}
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('explicit diagnostic/original source SHA differs: '+path)
    spec=importlib.util.spec_from_file_location('frozen_E031_explicit_approved_off',ORIGINAL)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if sha(m.ADOPTION)!=m.ADOPTION_SHA:raise ValueError('original approved On baseline changed')
    on,off,diff=validate_evidence(read(OBSERVED),read(DIAGNOSTIC),read(m.ADOPTION)['camera_before'])
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    OUTPUT.mkdir(exist_ok=False);binding=OUTPUT/'BINDING.json'
    write(binding,dict(schema='optical-lab-approved-exact-off-binding-1',
        original_file=str(ORIGINAL),original_sha256=ORIGINAL_SHA,wrapper_file=str(Path(__file__).resolve()),wrapper_sha256=sha(__file__),
        observation_file=str(OBSERVED),observation_sha256=OBSERVED_SHA,
        diagnostic_result_file=str(DIAGNOSTIC),diagnostic_result_sha256=DIAGNOSTIC_SHA,
        approved_on_before=on,approved_off_target=off,approved_differences=diff,
        approved_by_user=True,physical_shielding_confirmed=True,
        authority='User explicitly approves observed Off67.17443466186523 standby; formal On must recheck74.20778274536133',
        formal_capture_required=dict(TriggerMode='On',ExposureTime=ON_EXPOSURE),
        intended_SDK_write=dict(TriggerMode='Off'),exposure_node_written=False,
        target_GUI_or_frozen_source_modified=False,external_namespace_only=True,
        result_annotation_namespace_only=True,original_module_file_preserved=True,
        acceptance='entire observed On and Off dictionaries exact; no generalized tolerance',
        source_sha256=bindings))
    bindings[str(binding)]=sha(binding)
    from optical_lab import legacy_gui as old
    dependency_sources=old.dependency_sources
    def bound_sources(legacy):
        result=dependency_sources(legacy)
        for path,digest in bindings.items():
            if sha(path)!=digest or path in result and result[path]!=digest:
                raise ValueError('approved external source closure changed: '+path)
            result[path]=digest
        return result
    m.ROOT=OUTPUT;m.off_target=exact_validator(on,off);m.change=annotated_change(m.change)
    old.dependency_sources=bound_sources
    try:m.execute(True)
    finally:old.dependency_sources=dependency_sources


if __name__=='__main__':
    sys.path.insert(0,str(BASE.parents[1]/'src'))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--light-blocked-confirmed',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('explicit --execute --light-blocked-confirmed required')
    execute(args.light_blocked_confirmed)
