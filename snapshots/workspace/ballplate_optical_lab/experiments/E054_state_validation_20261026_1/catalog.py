"""Explicit file-only validation and fresh-window catalog controls; no SDK.

Load new binary input assets while parked, not new code or optical settings.
No runtime injection. The resulting per-test contract pins catalog provenance.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np

SCHEMA='optical-lab-four-history-validation-inputs-1'


def bound(ref, maximum):
    if not isinstance(ref,dict) or set(ref)!= {'path','sha256'}:
        raise ValueError('explicit path/SHA required')
    p=Path(ref['path'])
    if (not p.is_absolute() or p.is_symlink() or p.resolve()!=p or not p.is_file()
            or p.stat().st_size>maximum or not re.fullmatch('[0-9a-f]{64}',ref['sha256'])):
        raise ValueError('bounded regular absolute path/lowercase SHA required')
    data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('file SHA mismatch')
    return data


def validate(ref):
    data=json.loads(bound(ref,512_000))
    expected=dict(shape=[600,600],frame_shape=[4,1,96,96],history_ages_s=[.18,.12,.06,0.],
                  mosaic='oldest_top_left_row_major',repeat=3,pad=12)
    if data.get('schema')!=SCHEMA or data.get('encoding')!=expected:
        raise ValueError('canonical four-history encoding required')
    rows=data.get('inputs')
    if not isinstance(rows,list) or not 2<=len(rows)<=64:raise ValueError('2..64 inputs required')
    ids=set();hashes=set();assets=[]
    for row in rows:
        name=row.get('input_id')
        if not isinstance(name,str) or not re.fullmatch('[A-Z0-9_]{1,48}',name) or name in ids:
            raise ValueError('unique bounded input ID required')
        pos=np.asarray(row.get('position_xy_m'),float);vel=np.asarray(row.get('velocity_xy_m_s'),float)
        history=np.asarray(row.get('history_xy_m'),float)
        if (pos.shape!=(2,) or vel.shape!=(2,) or history.shape!=(4,2)
                or not np.isfinite(pos).all() or not np.isfinite(vel).all() or not np.isfinite(history).all()
                or np.abs(pos).max()>.15 or np.abs(history).max()>.18 or np.abs(vel).max()>.2
                or not np.allclose(history[-1],pos,rtol=0,atol=1e-10)):
            raise ValueError('bounded finite state and chronological history required')
        image=np.load(io.BytesIO(bound(row['dmd'],1_000_000)),allow_pickle=False)
        frames=np.load(io.BytesIO(bound(row['encoded_raw96'],100_000)),allow_pickle=False)
        if (image.dtype!=np.uint8 or image.shape!=(600,600) or not np.isin(image,(0,255)).all()
                or frames.dtype!=np.uint8 or frames.shape!=(4,1,96,96) or not np.isin(frames,(0,255)).all()
                or not 0<np.count_nonzero(image)<=20_000):
            raise ValueError('bounded sparse binary600 and encoded4x96 required')
        mosaic=np.block([[frames[0,0],frames[1,0]],[frames[2,0],frames[3,0]]])
        if not np.array_equal(image,np.pad(mosaic.repeat(3,0).repeat(3,1),12)):
            raise ValueError('DMD differs from exact encoded historical frames')
        pixel_hash=hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_hash in hashes:raise ValueError('duplicate physical input images')
        ids.add(name);hashes.add(pixel_hash);image.setflags(write=False)
        assets.append(dict(input_id=name,ref=deepcopy(row['dmd']),image=image))
    return data,assets


def install(window,ref):
    """Explicit visible Load action: validate everything before replacing lists."""
    flow=window.stability_workflow
    if (flow.active or flow.future is not None or flow.current_core() is not None
            or window.state_sampling_controller.current_core() is not None
            or window.armed.isChecked() or window.fault
            or any(getattr(window,k,False) for k in ('deployment_busy','loop_active','adaptation_active','override','roi_busy','reference_busy'))):
        raise RuntimeError('load only when parked, healthy and no active sampling/task')
    if window.hardware and not flow.zero_verified():raise RuntimeError('actual zero ACK required before catalog load')
    data,assets=validate(ref)
    code=Path(__file__).resolve();digest=hashlib.sha256(code.read_bytes()).hexdigest()
    if flow.template['source_sha256'].get(str(code))!=digest:
        raise RuntimeError('catalog loader must be pinned by fresh-window startup')
    # Bind exact manifest AND its image sources to each subsequently created test.
    template=deepcopy(flow.template)
    template['source_sha256'][ref['path']]=ref['sha256']
    template['input_catalog']=dict(reference=deepcopy(ref),metadata=data,
                                   loaded_by_explicit_operator_action=True,control_qualified=False)
    for row in data['inputs']:
        for key in ('dmd','encoded_raw96'):template['source_sha256'][row[key]['path']]=row[key]['sha256']
    flow.template=template;flow.assets=assets
    for key,index in (('input_a',0),('input_b',1)):
        combo=flow.widgets[key];old=combo.blockSignals(True)
        try:
            combo.clear()
            for asset in assets:combo.addItem('输入 '+asset['input_id'],asset['input_id'])
            combo.setCurrentIndex(index)
        finally:combo.blockSignals(old)
    flow.widgets['permission'].setChecked(False)
    flow.widgets['status'].setText('已加载%d种输入，仅更新选择清单，未换图/未采集。A/B统计只表示所选两种状态，不是全域控制验证。'%len(assets))
    if window.hardware:
        window.record(dict(state='RAW_validation_catalog_loaded',reference=ref,
                           input_ids=[a['input_id'] for a in assets],projection=False,camera_writes=0))
    return dict(inputs=len(assets),projection=False,camera_writes=0)


def attach(window,preset):
    """Only fresh startup; add controls INSIDE existing config, no third tab."""
    if window.worker is not None or hasattr(window,'validation_catalog_controls'):
        raise RuntimeError('attach once at fresh startup')
    from PyQt5 import QtWidgets
    config=window.stability_workflow.widgets['config'];grid=config.layout()
    group=QtWidgets.QGroupBox('验证输入清单 · 只加载，不投图');layout=QtWidgets.QGridLayout(group)
    path=QtWidgets.QLineEdit(preset['path']);digest=QtWidgets.QLineEdit(preset['sha256'])
    path.setAccessibleName('验证输入清单绝对路径');digest.setAccessibleName('验证输入清单SHA256')
    layout.addWidget(QtWidgets.QLabel('清单路径'),0,0);layout.addWidget(path,0,1,1,3)
    layout.addWidget(QtWidgets.QLabel('SHA256'),1,0);layout.addWidget(digest,1,1,1,2)
    button=QtWidgets.QPushButton('加载验证输入清单（不投图）');layout.addWidget(button,1,3)
    note=QtWidgets.QLabel('可在当前窗口加载新清单；固定四历史编码。只有点击“一键测试”并授权后才投所选A/B，不自动扫描全部输入。')
    note.setWordWrap(True);layout.addWidget(note,2,0,1,4)
    def clicked():
        try:
            result=install(window,dict(path=path.text().strip(),sha256=digest.text().strip()))
            note.setText('已加载%d种输入；默认未授权测试。曝光/Gain/SLM/ROI均未改变。'%result['inputs'])
        except Exception as error:note.setText('未加载：'+str(error))
    button.clicked.connect(clicked);grid.addWidget(group,grid.rowCount(),0,1,6)
    window.validation_catalog_controls=dict(path=path,digest=digest,button=button,note=note)
    return window.validation_catalog_controls
