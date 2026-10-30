"""Offline, SHA-bound historical forty-input R2 bank; no GUI/SDK ownership.

Labels and trajectory splits are retained from an explicitly selected old
catalog. New camera measurements do NOT make these inputs new blind holdouts.
No SLM or old RAW8 controller is inherited by this input-only artifact.
"""
from copy import deepcopy
import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from .study import encode_history, _input
from .storage import sha256, write_json_new


SCHEMA = 'optical-lab-processed8-historical-r2-input-bank-1'
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _bound_bytes(path, expected):
    p = Path(path)
    if not p.is_absolute() or p.is_symlink() or not _SHA.fullmatch(expected or ''):
        raise ValueError('explicit absolute regular path and lowercase SHA required')
    blob = p.read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError('source SHA mismatch: ' + str(p))
    return blob


def pack_dmd(raw):
    """Exact chronological 2x2 RAW96 -> repeat3 -> pad12 binary canvas600."""
    a = np.asarray(raw)
    if a.dtype != np.uint8 or a.shape != (4, 1, 96, 96) or not np.isin(a, (0,255)).all():
        raise ValueError('four binary uint8 RAW96 frames required')
    mosaic = np.block([[a[0,0],a[1,0]],[a[2,0],a[3,0]]])
    result = np.pad(mosaic.repeat(3,0).repeat(3,1),12)
    _input(a, result)
    return result


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def prepare(head_path, head_sha256, catalog_path, catalog_sha256, output):
    """Validate ALL inputs before creating a NEW directory. Never fit a head."""
    head_path, catalog_path = Path(head_path), Path(catalog_path)
    head_blob = _bound_bytes(head_path, head_sha256)
    catalog_blob = _bound_bytes(catalog_path, catalog_sha256)
    head, catalog = json.loads(head_blob), json.loads(catalog_blob)
    if (Path(head.get('bundle', '')) != catalog_path or head.get('bundle_sha256') != catalog_sha256
            or catalog.get('schema') != 'real-single-frame-adaptation-inputs-v22'):
        raise ValueError('explicit historical HEAD/catalog binding required')
    rows = catalog.get('cases', [])
    if (len(rows) != 40 or len({r['id'] for r in rows}) != 40
            or any(type(r['id']) is not int or not 0 <= r['id'] <= 999 for r in rows)):
        raise ValueError('exactly forty unique integer historical input IDs required')
    expected_counts = {'train':20,'select':8,'test':12}
    actual_counts = {s:sum(r['split']==s for r in rows) for s in expected_counts}
    if actual_counts != expected_counts or sum(actual_counts.values()) != 40:
        raise ValueError('unchanged20/8/12 input splits required')
    groups = {s:{r['trajectory'] for r in rows if r['split']==s} for s in expected_counts}
    if ({s:len(g) for s,g in groups.items()} != {'train':10,'select':4,'test':6}
            or any(groups[a] & groups[b] for a,b in [('train','select'),('train','test'),('select','test')])):
        raise ValueError('unchanged disjoint10/4/6 trajectory groups required')
    bindings = catalog['bound_sha256']
    checked = {}
    for path, expected in bindings.items():
        _bound_bytes(path, expected)
        checked[path] = expected
    staged = []
    for row in rows:
        if row.get('RAW_input_ages_s') != [.18,.12,.06,0.]:
            raise ValueError('unchanged180/120/60/0ms history ages required')
        target = np.asarray(row['teacher_action_rad'], float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('finite two-action historical teacher label required')
        original = {}
        for key in ('raw96','dmd','raw128_teacher_only'):
            path = row[key]
            if path not in bindings:
                raise ValueError('every original image must be catalog SHA-bound: '+key)
            original[key] = _bound_bytes(path, bindings[path])
        raw = np.load(io.BytesIO(original['raw96']), allow_pickle=False)
        old_dmd = np.load(io.BytesIO(original['dmd']), allow_pickle=False)
        _input(raw, old_dmd)
        encoded, geometry = encode_history(raw, 2)
        staged.append((row, original, encoded, geometry, pack_dmd(encoded)))
    # Preserve exact original bytes and recheck identity before any new output.
    _bound_bytes(head_path, head_sha256)
    _bound_bytes(catalog_path, catalog_sha256)
    for path, expected in checked.items():
        _bound_bytes(path, expected)
    root = Path(output).resolve()
    root.mkdir(exist_ok=False)
    provenance = root/'provenance'; provenance.mkdir()
    for filename, blob in [('historical_HEAD.json',head_blob),('historical_catalog.json',catalog_blob)]:
        with (provenance/filename).open('xb') as stream: stream.write(blob)
    source_map = {}
    for path in (Path(__file__),Path(__file__).with_name('study.py'),
                 Path(__file__).with_name('probes.py'),Path(__file__).with_name('storage.py')):
        blob = path.read_bytes()
        with (provenance/path.name).open('xb') as stream: stream.write(blob)
        source_map[str(path.resolve())] = hashlib.sha256(blob).hexdigest()
    plan = dict(schema=SCHEMA, historical_input_label_set=True, new_blind_holdout=False,
        camera_measurements_already_collected=False, input_count=40,
        historical_head=dict(path=str(head_path),sha256=head_sha256,
                             role='LABEL_SET_REFERENCE_ONLY_NOT_CONTROLLER'),
        historical_catalog=dict(path=str(catalog_path),sha256=catalog_sha256),
        historical_catalog_bound_sources=checked, generator_sources=source_map,
        original_input_order=[r['id'] for r in rows], split_input_counts=actual_counts,
        split_trajectory_counts={s:len(v) for s,v in groups.items()},
        dmd_encoding=dict(name='R2',radius_logical_px=2,
            algorithm='encode_history independent Euclidean disk dilation, zero border',
            frame_shape=[96,96],history_ages_s=[.18,.12,.06,0.],
            chronological_order='TL/TR/BL/BR oldest_to_newest',scale_repeat=3,
            black_border_pixels=12,canvas_shape=[600,600]),
        SLM_is_NOT_inherited=True,old_HEAD_is_NOT_controller=True,
        expected_feature_domain='NEW processed8 contract required at camera collection',
        hardware_actions_enabled=False,real_actuators_enabled=False,
        production_qualified=False,release_approved=False)
    write_json_new(root/'PLAN.json',plan)
    inputs = []
    for source, original, encoded, geometry, dmd in staged:
        prefix = f"input_{source['id']:03d}"
        generated = {}
        for key, name, blob in (
            ('raw96',prefix+'_SOURCE_RAW96.npy',original['raw96']),
            ('encoded_raw96',prefix+'_R2_RAW96.npy',_npy(encoded)),
            ('dmd',prefix+'_DMD.npy',_npy(dmd))):
            path = root/name
            with path.open('xb') as stream: stream.write(blob)
            generated[key] = dict(path=str(path),sha256=sha256(path))
        inputs.append(dict(id=source['id'],trajectory=source['trajectory'],split=source['split'],
            teacher_action_rad=deepcopy(source['teacher_action_rad']),
            RAW_input_ages_s=[.18,.12,.06,0.],historical_input_label_set=True,
            new_blind_holdout=False,dmd_encoding='R2',source_case=deepcopy(source),
            source_images={k:dict(path=source[k],sha256=bindings[source[k]])
                           for k in ('raw96','dmd','raw128_teacher_only')},
            encoding_geometry=geometry,**generated))
    write_json_new(root/'INPUTS.json',dict(schema=SCHEMA,rows=inputs,
        historical_input_label_set=True,new_blind_holdout=False,
        original_labels_groups_splits_preserved=True,SLM_is_NOT_inherited=True,
        old_HEAD_is_NOT_controller=True,release_approved=False))
    sources = {r['source_case']['raw96']:r['source_images']['raw96']['sha256'] for r in inputs}
    write_json_new(root/'RESULT.json',dict(schema=SCHEMA,input_count=len(inputs),
        split_input_counts=actual_counts,historical_input_label_set=True,new_blind_holdout=False,
        original_image_files_SHA_checked=sum(3 for _ in inputs),
        original_distinct_RAW96_paths=len(sources),all_catalog_bindings_verified=len(bindings),
        R2_encoding_and_packing_verified=True,SLM_is_NOT_inherited=True,
        old_HEAD_is_NOT_controller=True,new_camera_data_required=True,
        new_processed8_head_required=True,release_approved=False))
    manifest = dict(schema=SCHEMA,file_sha256={str(p.relative_to(root)):sha256(p)
        for p in sorted(root.rglob('*')) if p.is_file()},manifest_excludes_itself=True,
        historical_input_label_set=True,new_blind_holdout=False,release_approved=False)
    write_json_new(root/'MANIFEST.json',manifest)
    return dict(output=str(root),inputs_path=str(root/'INPUTS.json'),input_count=len(inputs),
                manifest_sha256=sha256(root/'MANIFEST.json'))


def verify(folder, expected_manifest_sha256):
    root = Path(folder).resolve()
    manifest = json.loads(_bound_bytes(root/'MANIFEST.json',expected_manifest_sha256))
    if manifest.get('schema') != SCHEMA or manifest.get('new_blind_holdout') is not False:
        raise ValueError('explicit historical input bank required')
    expected = manifest['file_sha256']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    if actual != set(expected): raise ValueError('bank file set changed')
    for name, digest in expected.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('bank path escape')
        _bound_bytes(path,digest)
    inputs = json.loads((root/'INPUTS.json').read_text())
    for row in inputs['rows']:
        raw = np.load(io.BytesIO(_bound_bytes(row['raw96']['path'],row['raw96']['sha256'])),allow_pickle=False)
        encoded, geometry = encode_history(raw,2)
        actual_encoded = np.load(io.BytesIO(_bound_bytes(row['encoded_raw96']['path'],row['encoded_raw96']['sha256'])),allow_pickle=False)
        dmd = np.load(io.BytesIO(_bound_bytes(row['dmd']['path'],row['dmd']['sha256'])),allow_pickle=False)
        if geometry != row['encoding_geometry'] or not np.array_equal(encoded,actual_encoded) or not np.array_equal(pack_dmd(encoded),dmd):
            raise ValueError('R2 geometry or chronological packing changed')
        for ref in row['source_images'].values(): _bound_bytes(ref['path'],ref['sha256'])
    return dict(verified=True,input_count=len(inputs['rows']),checked_files=len(expected),new_blind_holdout=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head',type=Path,required=True)
    parser.add_argument('--head-sha256',required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--catalog-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = prepare(args.head,args.head_sha256,args.catalog,args.catalog_sha256,args.output)
    result['verified'] = verify(args.output,result['manifest_sha256'])
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__': main()
