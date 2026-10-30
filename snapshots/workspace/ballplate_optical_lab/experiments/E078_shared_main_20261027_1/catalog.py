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
import time
import numpy as np

SCHEMA='optical-lab-four-history-validation-inputs-1'
SIZE_POLICY='disk-dilation-from-original-raw96-v1'


def file_ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def encode_ball_size(raw,radius):
    """File-only disk dilation, never scale positions, tiles or the whole DMD.

    R0 is identity; R2 is byte-identical to the deployed dynamic R2 algorithm.
    Always start from SOURCE_RAW96, never cumulatively dilate encoded inputs.
    """
    from optical_lab.study import _component_count
    if type(radius) is not int or not 0<=radius<=8:
        raise ValueError('球扩张半径必须为0..8整数（96图像素，不是物理球半径）')
    if (not isinstance(raw,np.ndarray) or raw.dtype!=np.uint8 or raw.shape!=(4,1,96,96)
            or not np.isin(raw,(0,255)).all()):
        raise ValueError('需要原始四幅uint8二值96图')
    result=np.zeros_like(raw);geometry=[]
    offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)
             if dx*dx+dy*dy<=radius*radius]
    for i,image in enumerate(raw[:,0]):
        mask=image>0
        if not mask.any() or _component_count(mask)!=1:
            raise ValueError('原始每幅必须只有一个非空连通球，不能把标定图当球扩张')
        padded=np.pad(mask,radius);grown=np.zeros_like(mask)
        for dy,dx in offsets:
            grown|=padded[radius+dy:radius+dy+96,radius+dx:radius+dx+96]
        old_yx=np.argwhere(mask);new_yx=np.argwhere(grown)
        lo,hi=new_yx.min(0),new_yx.max(0);oldlo,oldhi=old_yx.min(0),old_yx.max(0)
        delta=new_yx.mean(0)-old_yx.mean(0);margin=int(min(*lo,*(95-hi)))
        if margin<2 or np.linalg.norm(delta)>.25:
            raise ValueError(f'第{i}幅扩大后靠近边界或栅格质心偏移>0.25像素；请减小R')
        result[i,0]=grown.astype(np.uint8)*255
        geometry.append(dict(frame_index=i,source_on=int(mask.sum()),encoded_on=int(grown.sum()),
            source_bbox_wh_raw_px=(oldhi-oldlo+1)[::-1].tolist(),
            encoded_bbox_wh_raw_px=(hi-lo+1)[::-1].tolist(),
            encoded_bbox_wh_DMD_px=((hi-lo+1)*3)[::-1].tolist(),
            encoded_bbox_xyxy=[int(lo[1]),int(lo[0]),int(hi[1]+1),int(hi[0]+1)],
            source_centroid_xy=old_yx.mean(0)[::-1].tolist(),
            encoded_centroid_xy=new_yx.mean(0)[::-1].tolist(),centroid_delta_xy=delta[::-1].tolist(),
            centroid_delta_L2_raw_px=float(np.linalg.norm(delta)),minimum_edge_margin_raw_px=margin,
            DMD_on_pixels=int(grown.sum())*9,area_ratio=float(grown.sum()/mask.sum())))
    return result,geometry


def pack_frames(frames):
    mosaic=np.block([[frames[0,0],frames[1,0]],[frames[2,0],frames[3,0]]])
    return np.pad(mosaic.repeat(3,0).repeat(3,1),12)


def size_info(row,frames):
    """Verify supplied metadata, or identify known original R0..R8 bytes."""
    if row.get('input_kind')=='calibration_probe_not_MuJoCo_state':
        return dict(kind='fixed_calibration_probe',resized=False)
    if 'SOURCE_RAW96' not in row:
        if 'ball_size' in row:raise ValueError('球尺寸声明缺少原始SOURCE_RAW96证据')
        return dict(kind='unknown',reason='未提供原始SOURCE_RAW96；不可自动改尺寸')
    raw=np.load(io.BytesIO(bound(row['SOURCE_RAW96'],100_000)),allow_pickle=False)
    declared=row.get('ball_size');candidates=[declared.get('radius_raw_px')] if declared else [2,0,1,3,4,5,6,7,8]
    for radius in candidates:
        try:encoded,geometry=encode_ball_size(raw,radius)
        except ValueError:
            if declared:raise
            continue
        if np.array_equal(encoded,frames):
            info=dict(kind='four_ball_history',policy=SIZE_POLICY,radius_raw_px=radius,
                radius_DMD_px=3*radius,source_raw96=deepcopy(row['SOURCE_RAW96']),geometry=geometry,
                DMD_on_pixels=sum(v['DMD_on_pixels'] for v in geometry),
                source_history_and_tile_positions_unchanged=True,repeat=3,pad=12,physical_ball_model_changed=False,
                matches_original_R2_bytes=radius==2,control_qualified=False)
            if declared and declared!=info:raise ValueError('球尺寸元数据与实际图像不一致')
            return info
    raise ValueError('输入图不匹配原始球及声明的尺寸算法')


def resize_catalog(reference,radius,destination,preserve_ids=()):
    """Create fresh immutable assets/manifest only; no SDK or display calls."""
    data,_=validate(reference);destination=Path(destination).resolve()
    if destination.exists():raise ValueError('尺寸实验目录必须全新，不能覆盖旧数据')
    prepared=[]
    for row in data['inputs']:
        if row.get('input_kind')=='calibration_probe_not_MuJoCo_state' or row['input_id'] in preserve_ids:
            prepared.append((deepcopy(row),None,None));continue
        if 'SOURCE_RAW96' not in row:raise ValueError(row['input_id']+'缺少原始球图，不能猜测尺寸')
        raw=np.load(io.BytesIO(bound(row['SOURCE_RAW96'],100_000)),allow_pickle=False)
        encoded,_=encode_ball_size(raw,radius);image=pack_frames(encoded)
        if np.count_nonzero(image)>20_000:raise ValueError('超过现有稀疏图像面积上限')
        updated=deepcopy(row);updated.pop('ball_size',None);updated.pop('encoding_quality',None)
        updated['ball_size']=size_info(updated,encoded)
        prepared.append((updated,encoded,image))
    destination.mkdir(parents=True)
    for row,encoded,image in prepared:
        if encoded is None:continue
        child=destination/row['input_id'];child.mkdir()
        for name,array in [('encoded_raw96',encoded),('dmd',image)]:
            path=child/(name+'.npy')
            with path.open('xb') as f:np.save(f,array,allow_pickle=False)
            row[name]=file_ref(path)
    data['inputs']=[v[0] for v in prepared]
    data['ball_size_experiment']=dict(policy=SIZE_POLICY,radius_raw_px=radius,
        parent_catalog=deepcopy(reference),preserved_reference_ids=list(preserve_ids),
        fixed_calibration_probes_unchanged=True,hardware_actions=0,control_qualified=False,
        source=file_ref(__file__))
    path=destination/'CATALOG.json'
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    result=file_ref(path);validate(result)
    return result


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
        row['ball_size']=size_info(row,frames)
        ids.add(name);hashes.add(pixel_hash);image.setflags(write=False)
        assets.append(dict(input_id=name,ref=deepcopy(row['dmd']),image=image))
    return data,assets


def require_parked(window):
    flow=window.stability_workflow
    if (flow.active or flow.future is not None or flow.current_core() is not None
            or window.state_sampling_controller.current_core() is not None
            or window.armed.isChecked() or window.fault
            or any(getattr(window,k,False) for k in ('deployment_busy','loop_active','adaptation_active','override','roi_busy','reference_busy'))):
        raise RuntimeError('load only when parked, healthy and no active sampling/task')
    if window.hardware and not flow.zero_verified():raise RuntimeError('actual zero ACK required before catalog load')


def install(window,ref):
    """Explicit visible Load action: validate everything before replacing lists."""
    require_parked(window)
    flow=window.stability_workflow
    data,assets=validate(ref)
    probes={r['input_id'] for r in data['inputs'] if r.get('input_kind')=='calibration_probe_not_MuJoCo_state' or r.get('state_fields_are_loader_placeholders')}
    assets=[a for a in assets if a['input_id'] not in probes]
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
        if 'SOURCE_RAW96' in row:
            source=row['SOURCE_RAW96'];template['source_sha256'][source['path']]=source['sha256']
    prior={key:flow.widgets[key].currentData() for key in ('input_a','input_b','input_ref') if key in flow.widgets}
    flow.template=template;flow.assets=assets
    for key,index in (('input_a',0),('input_b',1)):
        combo=flow.widgets[key];old=combo.blockSignals(True)
        try:
            combo.clear()
            for asset in assets:combo.addItem('输入 '+asset['input_id'],asset['input_id'])
            selected=combo.findData(prior.get(key));combo.setCurrentIndex(selected if selected>=0 else index)
        finally:combo.blockSignals(old)
    flow.widgets['permission'].setChecked(False)
    flow.update_budget()
    refresh_size_view(window)
    flow.widgets['status'].setText('已加载%d种输入，仅更新选择清单，未换图/未采集。A/B统计只表示所选两种状态，不是全域控制验证。'%len(assets))
    if window.hardware:
        window.record(dict(state='RAW_validation_catalog_loaded',reference=ref,
                           input_ids=[a['input_id'] for a in assets],projection=False,camera_writes=0))
    return dict(inputs=len(assets),projection=False,camera_writes=0)


def refresh_size_view(window):
    """Main-page source of truth: applied catalogue radius, not pending text."""
    if not hasattr(window,'ball_size_controls'):return
    from PyQt5 import QtGui,QtCore
    flow=window.stability_workflow;q=window.ball_size_controls
    name=q['input'].currentData()
    asset=next((v for v in flow.assets if v['input_id']==name),None)
    row=next((v for v in flow.template.get('input_catalog',{}).get('metadata',{}).get('inputs',[]) if v['input_id']==name),{})
    meta=row.get('ball_size',{});q['preview'].clear()
    if asset is not None:
        a=np.ascontiguousarray(asset['image']);image=QtGui.QImage(a.data,600,600,a.strides[0],QtGui.QImage.Format_Grayscale8).copy()
        q['preview'].setPixmap(QtGui.QPixmap.fromImage(image).scaled(130,130,QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation))
    q['actual'].setText(f'已应用R={meta.get("radius_raw_px","未知")}；{name}；'
        f'源600²亮像素{meta.get("DMD_on_pixels","未知")}。\nR单位为96图像素；位置/历史间隔不变，DMD等比缩放另行生效。')


def apply_size(window):
    require_parked(window)
    flow=window.stability_workflow;q=window.ball_size_controls
    reference=deepcopy(flow.template['input_catalog']['reference'])
    folder=flow.output_root.parent/'BALL_SIZE_ASSETS'/('R%d_%d'%(q['radius'].value(),time.time_ns()))
    output=resize_catalog(reference,q['radius'].value(),folder)
    result=install(window,output)
    refresh_main_inputs(window)
    load_main_ball(window)
    q['note'].setText('球尺寸已用于主界面四球和后续测试输入；仍未投图。SLM/曝光/Gain/ROI不变。')
    return result


def refresh_main_inputs(window):
    if not hasattr(window,'ball_size_controls'):return
    q=window.ball_size_controls;combo=q['input'];prior=combo.currentData();blocked=combo.blockSignals(True)
    combo.clear()
    for a in window.stability_workflow.assets:combo.addItem(a['input_id'],a['input_id'])
    index=combo.findData(prior if prior else 'CENTER');combo.setCurrentIndex(max(0,index));combo.blockSignals(blocked)
    refresh_size_view(window)


def load_main_ball(window):
    require_parked(window)
    q=window.ball_size_controls;name=q['input'].currentData()
    asset=next(v for v in window.stability_workflow.assets if v['input_id']==name)
    window.dmd_file=asset['image'].copy()
    window.dmd_mode.setCurrentIndex(window.dmd_mode.findData('file'))
    window.changed()
    q['note'].setText('已将 '+name+' 载入DMD缓存；投图开关保持关闭，测试与主界面共用已应用球尺寸。')


def attach(window,preset):
    """Only test inputs live on page2. Device, ball and head choices live on main."""
    if window.worker is not None or hasattr(window,'ball_size_controls'):
        raise RuntimeError('fresh startup required')
    from PyQt5 import QtWidgets
    flow=window.stability_workflow;q=flow.widgets
    # One visible file picker; SHA is recorded automatically, never typed twice.
    load=QtWidgets.QPushButton('载入位置/速度输入清单（不投图）')
    q['config'].layout().addWidget(load,5,0,1,6)
    def choose():
        path,_=QtWidgets.QFileDialog.getOpenFileName(window,'选择位置/速度清单','','JSON (*.json)')
        if not path:return
        try:install(window,file_ref(path));refresh_main_inputs(window)
        except Exception as error:q['status'].setText('未加载：'+str(error))
    load.clicked.connect(choose)
    window.validation_catalog_controls=dict(button=load)
    dmd=window.dmd_mode.parentWidget();parent=dmd.parentWidget().layout()
    group=QtWidgets.QGroupBox('四球大小与主界面输入');grid=QtWidgets.QGridLayout(group)
    radius=QtWidgets.QSpinBox();radius.setRange(0,8);radius.setValue(2);radius.setKeyboardTracking(False)
    radius.setObjectName('main_ball_radius');radius.setAccessibleName('球扩张半径R（96图像素）')
    select=QtWidgets.QComboBox();select.setObjectName('main_ball_input')
    apply=QtWidgets.QPushButton('应用球大小并载入四球（不投图）');preview=QtWidgets.QLabel();preview.setFixedSize(130,130)
    load_ball=QtWidgets.QPushButton('仅载入所选四球（不投图）')
    actual=QtWidgets.QLabel();actual.setWordWrap(True);note=QtWidgets.QLabel('先置全0，再应用；只改待输入数值不会改变测试图。');note.setWordWrap(True)
    grid.addWidget(QtWidgets.QLabel('球扩张R'),0,0);grid.addWidget(radius,0,1);grid.addWidget(select,1,0,1,2)
    grid.addWidget(apply,2,0,1,2);grid.addWidget(load_ball,3,0,1,2);grid.addWidget(preview,4,0);grid.addWidget(actual,4,1);grid.addWidget(note,5,0,1,2)
    parent.insertWidget(parent.indexOf(dmd)+1,group)
    window.ball_size_controls=dict(group=group,radius=radius,input=select,apply=apply,preview=preview,actual=actual,note=note)
    def perform(action):
        try:action(window)
        except Exception as error:note.setText('未采用：'+str(error))
    apply.clicked.connect(lambda:perform(apply_size));load_ball.clicked.connect(lambda:perform(load_main_ball))
    radius.valueChanged.connect(lambda:note.setText('球大小待应用；测试仍使用“已应用”的输入图。'))
    select.currentIndexChanged.connect(lambda:refresh_size_view(window))
    head=QtWidgets.QGroupBox('候选RAW线性头 · 仅离线评分，可选');head_layout=QtWidgets.QVBoxLayout(head)
    q['head_enabled']=QtWidgets.QCheckBox('计算候选18参数头的控制指标')
    q['head_load']=QtWidgets.QPushButton('选择头JSON');q['head_path']=QtWidgets.QLabel('未选择：部署映射指标UNKNOWN');q['head_path'].setWordWrap(True)
    for key in ('head_enabled','head_load','head_path'):head_layout.addWidget(q[key])
    q['head_load'].clicked.connect(flow.choose_head)
    parent.insertWidget(max(0,parent.count()-1),head)
    refresh_main_inputs(window)
    return window.validation_catalog_controls
