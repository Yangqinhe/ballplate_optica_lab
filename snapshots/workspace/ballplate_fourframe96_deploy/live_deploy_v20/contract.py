"""Immutable current-Line0 alignment evidence. No SDK or output writes."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'live_manual_v17'))
from diffraction_model import OpticalReference,reference_signature


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(contract):
    if contract['schema']!='v20-retained-line0-v1':raise ValueError('unknown Line0 resume schema')
    for path,sha in contract['evidence_sha256'].items():
        if digest(path)!=sha:raise ValueError('original evidence changed: '+path)
    runtime=json.loads(Path(contract['runtime_path']).read_text())
    for group in ('sources','source_configs'):
        for path,sha in runtime[group].items():
            if digest(path)!=sha:raise ValueError('old loaded source changed: '+path)
    if contract['original_fault_before']['DeviceSerialNumber']!='15328133':raise ValueError('wrong rollback target')
    baseline_bound=False
    for path in contract['evidence_sha256']:
        if Path(path).suffix!='.json':continue
        data=json.loads(Path(path).read_text())
        if (data.get('camera_before') or data.get('target'))==contract['original_fault_before']:
            baseline_bound=True
    if not baseline_bound:raise ValueError('original rollback target not bound to its camera evidence')
    if contract['settings']['dmd']=='file' or contract['settings']['slm']=='file':
        raise ValueError('file caches are not silently restored')
    if not 50<=contract['requested_exposure_us']<=4000:raise ValueError('exposure outside diagnostic range')
    config=OpticalReference(**contract['optical_config']);config.validate()
    if contract['registration'] is not None:
        folder=Path(contract['registration']['reference_folder'])
        reg=json.loads((folder/'frozen_geometry.json').read_text())
        capture=json.loads((folder/'capture.json').read_text())
        model=json.loads((folder/'model_reference.json').read_text())
        active=capture['active']
        expected=dict(DeviceSerialNumber='15328133',TriggerMode='On',TriggerSource='Line0',
                      TriggerActivation='FallingEdge',SensorShutterMode='Rolling',PixelFormat='Mono8')
        if any(active.get(k)!=v for k,v in expected.items()):raise ValueError('not a Line0 reference')
        if (reg!=contract['registration'] or not reg['frozen'] or model!=contract['optical_config']
                or reg['signature']!=reference_signature(config,active)
                or active!=contract['expected_active']):raise ValueError('reference/model contract mismatch')
        if contract['roi']!={k:int(active[k]) for k in ('OffsetX','OffsetY','Width','Height')}:
            raise ValueError('reference ROI mismatch')
        if abs(contract['requested_exposure_us']-float(active['ExposureTime']))>50:
            raise ValueError('reference exposure differs from requested exposure')
        for name in ('frozen_geometry.json','capture.json','model_reference.json'):
            if str(folder/name) not in contract['evidence_sha256']:raise ValueError('unbound reference')
    return contract


def matching(contract,settings,config,active,registration):
    validate(contract)
    fixed=('full_panel','x','y','focal_mm','period','square')
    if (any(settings[k]!=contract['settings'][k] for k in fixed)
            or config!=contract['optical_config'] or active!=contract['expected_active']
            or registration!=contract['registration']):
        raise ValueError('当前配置已变化；重新冻结并准备测试包，不能静默套用旧数据')


def initial_native_matrix(registration,roi):
    import numpy as np
    if registration['orientation']!=5:raise ValueError('only validated orientation5 transfer supported')
    m=np.asarray(registration['matrix_sensor'],float).copy()
    # Registered600 is the central crop of the archived ASM 1800x1920 grid.
    # This transfer is a surrogate prior, not a new optical calibration.
    m[:,2]+=m[:,:2]@[600.,660.]
    m[:,2]-=[roi['OffsetX'],roi['OffsetY']]
    return m
