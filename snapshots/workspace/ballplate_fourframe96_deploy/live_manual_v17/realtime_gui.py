"""Single owner, live preview plus explicit bounded point calibration; no RL."""
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import os
import sys
import time

import numpy as np
from PIL import Image
from PyQt5 import QtCore,QtGui,QtWidgets

from bench import Bench
from black_pixels import RootPixels
from camera_recheck import RecheckWorker
from realtime_model import LivePattern,patterns,load_image,drag_roi,display_previews
from realtime_worker import LiveCameraWorker,write_json
from diffraction_model import OpticalReference,reference_signature
from diffraction_worker import DiffractionWorker
from display_orientation import load_hint,orientation_name,edited_orientation
from stable_views import StatusLabel,SensorView,SharedSensorScale
from realtime_recovery import recovery_target
from display_linux import read_roles,same_signals
from slm_coordinates import load_slm_hint
from spatial_calibration import SpatialPlan,geometry_overlay


class CCDView(SensorView):
    selected=QtCore.pyqtSignal(object)
    def __init__(self):
        super().__init__('等待连续CCD采集')
        self.active_roi=None;self.image_rect=None;self.drag_start=None
        self.band=QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle,self)

    def setText(self,text):
        super().setText(text);self.drag_start=None;self.active_roi=None
        if hasattr(self,'band'):self.band.hide()

    def set_frame(self,frame,active,enhance,prepared=False):
        roi={k:int(active[k]) for k in ('OffsetX','OffsetY','Width','Height')}
        if roi!=self.active_roi:self.drag_start=None;self.band.hide()
        self.active_roi=roi
        if prepared:values=frame
        else:
            raw,enhanced=display_previews(frame);values=enhanced if enhance else raw
        self.set_image(values,(roi['Width'],roi['Height']))

    def mousePressEvent(self,event):
        if event.button()==QtCore.Qt.LeftButton and self.image_rect and self.active_roi:
            x,y,w,h=self.image_rect
            if x<=event.x()<x+w and y<=event.y()<y+h:
                self.drag_start=event.pos();self.band.setGeometry(QtCore.QRect(event.pos(),QtCore.QSize()));self.band.show()

    def mouseMoveEvent(self,event):
        if self.drag_start is not None:self.band.setGeometry(QtCore.QRect(self.drag_start,event.pos()).normalized())

    def mouseReleaseEvent(self,event):
        if self.drag_start is None:return
        start=self.drag_start;self.drag_start=None
        if not self.image_rect or not self.active_roi:self.band.hide();return
        try:
            roi=drag_roi((start.x(),start.y()),(event.x(),event.y()),self.image_rect,self.active_roi)
            self.selected.emit(roi)
        except ValueError:self.band.hide()


class RealtimeWindow(Bench):
    def __init__(self,output,hardware=False):
        self.hardware=hardware;self.output=Path(output)
        self.worker=None;self.history=[];self.running=False;self.live_ready=False
        self.fault=None;self.fault_before=None;self.last_packet=None;self.last_sequence=0
        self.actual_roi=dict(OffsetX=0,OffsetY=0,Width=0,Height=0);self.candidate=None
        self.dmd_file=None;self.slm_file=None;self.override=False;self.roi_busy=False
        self.dirty=True;self.revision=0;self.expected={};self.pixels=None
        self.last_apply=0.;self.last_verify=0.;self.last_heartbeat=0.;self.mismatches=0
        self.last_verified_revision=None;self.last_gate=None
        self.previous_gui_tick=None;self.gui_gaps_ms=deque(maxlen=240)
        self.last_camera_report=None;self.recheck_context=None;self.building=True
        self.diffraction=None;self.comparison_epoch=0;self.last_comp_submit=0.
        self.last_comparison=None;self.registration=None;self.reference_busy=False
        self.comparison_error=None;self.reference_request=None
        self.spatial_busy=False;self.spatial_display_pending=None;self.last_spatial_report=None
        self.orientation_hint=load_hint()
        self.slm_coordinate_hint=load_slm_hint()
        if hardware:
            super().__init__(self.output,low_power_confirmed=False,diagnostic_scope='manual_realtime_unqualified')
            self.request_timer.stop();self.pixels=RootPixels()
            self.dmd_output=self.windows['dmd'];self.slm_output=self.windows['slm']
        else:
            QtWidgets.QMainWindow.__init__(self)
            self.windows={};self.baseline={};self.busy=False;self.output_lock=None
        self.build_ui();self.building=False
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)
        self.render_patterns();self.black_outputs();self.refresh();self.show()
        if hardware:
            files={}
            for module in list(sys.modules.values()):
                name=getattr(module,'__file__',None)
                if name and str(Path(__file__).parent) in str(Path(name).resolve()) and str(name).endswith('.py'):
                    path=Path(name).resolve();files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(self.output/'runtime_sources.json',dict(pid=os.getpid(),sources=files,
                source_configs={self.orientation_hint['path']:self.orientation_hint['sha256'],
                                self.slm_coordinate_hint['path']:self.slm_coordinate_hint['sha256']},
                no_auto_capture=True,no_auto_projection=True,single_output_and_sdk_owner=True))
            write_json(self.output/'display_orientation_hint.json',self.orientation_hint)
            write_json(self.output/'slm_phase_coordinate_hint.json',self.slm_coordinate_hint)
            self.record(dict(state='manual_realtime_initialized',no_auto_capture=True,no_auto_projection=True,
                             raw_phase_no_LUT=True,SLM_horizontal_flip_count=1))

    def build_ui(self):
        old=self.takeCentralWidget()
        root=QtWidgets.QWidget();self.setCentralWidget(root);layout=QtWidgets.QVBoxLayout(root)
        title=QtWidgets.QLabel('实时对齐 · 实测CCD / 仿真CCD / 差值')
        title.setStyleSheet('font-size:23px;font-weight:bold');layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel('拖动SLM中心直接更新；仿真默认固定理想居中目标。形状比较辅助粗对齐，不自动扫描、不运行策略。'
            if self.hardware else '离线界面预览：不连接相机，不创建DMD/SLM输出窗。'))
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.armed=QtWidgets.QCheckBox('允许投图（已确认当前ND／低功率条件）');row.addWidget(self.armed)
        self.start_button=QtWidgets.QPushButton('开始实时（DMD＋SLM＋CCD）')
        self.start_button.clicked.connect(lambda:self.start_live(project=True));row.addWidget(self.start_button)
        self.camera_only_button=QtWidgets.QPushButton('仅启动CCD（不投图）')
        self.camera_only_button.clicked.connect(lambda:self.start_live(project=False));row.addWidget(self.camera_only_button)
        self.stop_button=QtWidgets.QPushButton('停止／两路全0（Esc）');self.stop_button.setStyleSheet('color:#b02020;font-weight:bold')
        self.stop_button.clicked.connect(self.stop_live);row.addWidget(self.stop_button)
        self.black_button=QtWidgets.QPushButton('仅DMD置0，CCD继续');self.black_button.clicked.connect(self.dmd_black);row.addWidget(self.black_button)
        self.snap_button=QtWidgets.QPushButton('保存当前快照');self.snap_button.clicked.connect(self.snapshot);row.addWidget(self.snap_button)
        self.armed.toggled.connect(self.arm_changed)
        self.output_state=StatusLabel('DMD/SLM未投图：尚未开始',2)
        self.output_state.setWordWrap(True);self.output_state.setStyleSheet('font-size:16px;color:#9a5100')
        layout.addWidget(self.output_state)
        middle=QtWidgets.QHBoxLayout();layout.addLayout(middle,1)
        left_widget=QtWidgets.QWidget()
        left=QtWidgets.QVBoxLayout(left_widget)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(left_widget)
        scroll.setFixedWidth(465);middle.addWidget(scroll)
        self.build_spatial_controls(left)
        box=QtWidgets.QGroupBox('DMD · 600×600二值画布');body=QtWidgets.QVBoxLayout(box)
        self.dmd_mode=QtWidgets.QComboBox()
        for text,key in [('非对称标定图','fiducial'),('中心方块','square'),('标准白方块（ROI）','white'),('全黑','black'),('载入的二值图','file')]:self.dmd_mode.addItem(text,key)
        body.addWidget(self.dmd_mode)
        row=QtWidgets.QHBoxLayout();body.addLayout(row)
        self.square=QtWidgets.QSpinBox();self.square.setRange(16,600);self.square.setValue(128)
        row.addWidget(QtWidgets.QLabel('中心方块边长'));row.addWidget(self.square)
        button=QtWidgets.QPushButton('载入图像');button.clicked.connect(lambda:self.load_file(False));row.addWidget(button)
        left.addWidget(box)
        box=QtWidgets.QGroupBox('SLM · 实时加载原始灰度');body=QtWidgets.QVBoxLayout(box)
        row=QtWidgets.QHBoxLayout();body.addLayout(row)
        self.slm_mode=QtWidgets.QComboBox()
        for text,key in [('透镜','lens'),('零相位','zero'),('光栅 +X','+x'),('光栅 −X','-x'),('光栅 +Y','+y'),('光栅 −Y','-y'),('载入灰度图','file')]:self.slm_mode.addItem(text,key)
        row.addWidget(self.slm_mode)
        self.full=QtWidgets.QCheckBox('整屏');self.full.setChecked(True);row.addWidget(self.full)
        button=QtWidgets.QPushButton('载入');button.clicked.connect(lambda:self.load_file(True));row.addWidget(button)
        self.spins={};self.sliders={}
        form=QtWidgets.QGridLayout();body.addLayout(form)
        for i,(key,label,low,high) in enumerate((('x','中心X',-960,960),('y','中心Y',-540,540))):
            spin=QtWidgets.QSpinBox();spin.setRange(low,high);spin.setKeyboardTracking(False)
            slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);slider.setRange(low,high);slider.setPageStep(16)
            spin.valueChanged.connect(slider.setValue);slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(self.changed)
            self.spins[key]=spin;self.sliders[key]=slider
            form.addWidget(QtWidgets.QLabel(label),i,0);form.addWidget(slider,i,1);form.addWidget(spin,i,2)
        self.focal=QtWidgets.QDoubleSpinBox();self.focal.setRange(50,3000);self.focal.setValue(400);self.focal.setSuffix(' mm');self.focal.setKeyboardTracking(False)
        self.period=QtWidgets.QSpinBox();self.period.setRange(4,512);self.period.setValue(32);self.period.setKeyboardTracking(False)
        row=QtWidgets.QHBoxLayout();body.addLayout(row)
        row.addWidget(QtWidgets.QLabel('透镜f'));row.addWidget(self.focal);row.addWidget(QtWidgets.QLabel('光栅周期'));row.addWidget(self.period)
        self.phase_caption=StatusLabel('',2);body.addWidget(self.phase_caption)
        left.addWidget(box)
        box=QtWidgets.QGroupBox('ROI对齐 · 可框选或自动定位');body=QtWidgets.QVBoxLayout(box)
        instruction=QtWidgets.QLabel('在右侧CCD画面拖框，再确认ROI。自动定位使用标准白方块。')
        instruction.setWordWrap(True);body.addWidget(instruction)
        row=QtWidgets.QHBoxLayout();body.addLayout(row)
        self.auto_button=QtWidgets.QPushButton('标准方块定位');self.auto_button.clicked.connect(self.auto_roi);row.addWidget(self.auto_button)
        self.apply_roi_button=QtWidgets.QPushButton('确认ROI');self.apply_roi_button.clicked.connect(self.apply_roi);row.addWidget(self.apply_roi_button)
        self.full_button=QtWidgets.QPushButton('恢复全幅');self.full_button.clicked.connect(self.full_sensor);row.addWidget(self.full_button)
        self.roi_text=StatusLabel('当前：等待相机实际读回；尚未选ROI',3);body.addWidget(self.roi_text)
        row=QtWidgets.QHBoxLayout();body.addLayout(row)
        self.exposure=QtWidgets.QDoubleSpinBox();self.exposure.setRange(50,4000);self.exposure.setValue(1000);self.exposure.setSuffix(' µs');self.exposure.setKeyboardTracking(False)
        row.addWidget(QtWidgets.QLabel('曝光'));row.addWidget(self.exposure)
        button=QtWidgets.QPushButton('应用曝光');button.clicked.connect(self.apply_exposure);row.addWidget(button)
        left.addWidget(box)
        row=QtWidgets.QHBoxLayout();left.addLayout(row)
        self.dmd_preview=QtWidgets.QLabel();self.slm_preview=QtWidgets.QLabel()
        for widget in (self.dmd_preview,self.slm_preview):
            widget.setFixedSize(210,150);widget.setAlignment(QtCore.Qt.AlignCenter);widget.setStyleSheet('background:#111');row.addWidget(widget)
        caption=QtWidgets.QLabel('左：DMD请求预览；右：SLM物理方向请求预览。\n参考λ=532nm、pitch=8µm；未加相位LUT。预览不是实测。');caption.setWordWrap(True);left.addWidget(caption)
        self.build_comparison_controls(left)
        left.addStretch()
        right=QtWidgets.QVBoxLayout();middle.addLayout(right,1)
        instruction=StatusLabel('新流程：仅启动CCD → 中心点检查 → 25点空间配准＋16点验证；SLM频谱定位另测。原ROI/手动功能保留。',2)
        right.addWidget(instruction)
        self.view=CCDView();self.view.selected.connect(self.select_roi)
        self.enhance=QtWidgets.QCheckBox('仅显示增强（不改变原始CCD）');self.enhance.setChecked(True);right.addWidget(self.enhance)
        self.telemetry=StatusLabel('CCD未启动',3);right.addWidget(self.telemetry)
        bar=QtWidgets.QHBoxLayout();bar.addWidget(self.target_mode,1);bar.addWidget(self.diff_mode,1);right.addLayout(bar)
        bar=QtWidgets.QHBoxLayout();bar.addWidget(QtWidgets.QLabel('仅仿真方向：'))
        self.direction_buttons=[]
        for text,action in (('左右镜像','horizontal'),('上下镜像','vertical'),('逆时针90°','rotate_ccw')):
            button=QtWidgets.QPushButton(text);button.clicked.connect(lambda checked=False,a=action:self.edit_preview_orientation(a))
            self.direction_buttons.append(button);bar.addWidget(button)
        self.reset_direction=QtWidgets.QPushButton('恢复平台方向');self.reset_direction.clicked.connect(self.restore_platform_orientation)
        self.direction_buttons.append(self.reset_direction);bar.addWidget(self.reset_direction)
        self.direction_status=StatusLabel('',2);bar.addWidget(self.direction_status,1);right.addLayout(bar)
        bar=QtWidgets.QHBoxLayout();bar.addWidget(self.zero_button);bar.addWidget(self.checkpoint_button)
        bar.addWidget(self.mapping_status,1);right.addLayout(bar)
        compare_row=QtWidgets.QHBoxLayout();right.addLayout(compare_row,1)
        self.sim_view=SensorView('等待图案与CCD');self.diff_view=SensorView('等待比较')
        for title,view in (('实测CCD · 拖框选择ROI',self.view),('仿真CCD · 同ROI同倍率',self.sim_view),('差值／叠加 · 同一ROI',self.diff_view)):
            box=QtWidgets.QGroupBox(title);body=QtWidgets.QVBoxLayout(box)
            body.addWidget(view);compare_row.addWidget(box,1)
        self.shared_sensor_scale=SharedSensorScale((self.view,self.sim_view,self.diff_view),self)
        self.comparison_status=StatusLabel('尚无比较。默认参数来自论文：λ532nm，f₁150mm / f₂100mm；SLM采样8µm。',3)
        right.addWidget(self.comparison_status)
        self.state=StatusLabel('默认全0待机。点“三路开始”加载图案并采集，或选择“仅CCD”。',2);layout.addWidget(self.state)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.recheck=QtWidgets.QPushButton('故障只读复核（不投图）');self.recheck.clicked.connect(self.start_recheck);row.addWidget(self.recheck)
        self.log=QtWidgets.QPlainTextEdit();self.log.setReadOnly(True);self.log.setFixedHeight(70);row.addWidget(self.log,1)
        button=QtWidgets.QPushButton('遮光后退出' if self.hardware else '关闭预览');button.clicked.connect(self.close);row.addWidget(button)
        self.shortcut=QtWidgets.QShortcut(QtGui.QKeySequence('Escape'),self);self.shortcut.activated.connect(self.stop_live)
        for control in (self.dmd_mode,self.slm_mode):control.currentIndexChanged.connect(self.changed)
        for control in (self.square,self.focal,self.period):control.valueChanged.connect(self.changed)
        self.full.toggled.connect(self.scope_changed)
        self.setWindowTitle('实时光路对齐 v17 · 分层标定 / DMD点阵'+(' · 真实器件' if self.hardware else ' · 离线预览'))
        primary=QtWidgets.QApplication.primaryScreen().availableGeometry();self.setGeometry(primary.x()+40,primary.y()+40,1800,1100)
        if old:old.deleteLater()

    def note(self,text):
        self.state.setText(text)
        if hasattr(self,'log'):self.log.appendPlainText(text)
        print(text,flush=True)

    def setting(self):
        return LivePattern(self.dmd_mode.currentData(),self.square.value(),self.slm_mode.currentData(),
            self.full.isChecked(),self.spins['x'].value(),self.spins['y'].value(),self.focal.value(),self.period.value())

    def build_spatial_controls(self,left):
        box=QtWidgets.QGroupBox('分层标定① · DMD → CCD（不是SLM配准）')
        body=QtWidgets.QVBoxLayout(box)
        row=QtWidgets.QHBoxLayout();body.addLayout(row)
        self.point_radius=QtWidgets.QSpinBox();self.point_radius.setRange(4,24);self.point_radius.setValue(8)
        row.addWidget(QtWidgets.QLabel('圆点半径 / DMD视频像素'));row.addWidget(self.point_radius)
        self.pilot_button=QtWidgets.QPushButton('1. 测中心点（Alt+1）');self.pilot_button.setShortcut('Alt+1')
        self.pilot_button.clicked.connect(lambda:self.start_spatial('pilot'));body.addWidget(self.pilot_button)
        self.grid_button=QtWidgets.QPushButton('2. 测25点 + 16验证点（Alt+2）');self.grid_button.setShortcut('Alt+2')
        self.grid_button.clicked.connect(lambda:self.start_spatial('grid'));body.addWidget(self.grid_button)
        self.spatial_status=StatusLabel('未测。先仅启动CCD，再测中心点；使用全幅采集。\nSLM固定零灰度；测后全0，不恢复手动投图。',4)
        body.addWidget(self.spatial_status)
        label=QtWidgets.QLabel('只辨识空间几何；不自动移动器件、不写入LUT/ASM。后续另测SLM频谱落点。')
        label.setWordWrap(True);body.addWidget(label);left.addWidget(box)

    def pilot_matches(self):
        r=self.last_spatial_report or {}
        return bool(r.get('completed') and r.get('analysis',{}).get('pilot_validated')
                    and r.get('plan',{}).get('radius')==self.point_radius.value()
                    and r.get('requested_exposure_us')==self.exposure.value())

    def start_spatial(self,mode):
        if (not self.hardware or not self.running or not self.live_ready or self.fault
                or self.override or self.roi_busy or self.reference_busy):return
        if mode=='grid' and not self.pilot_matches():
            self.note('请先在相同圆点半径与曝光下通过中心点检查。');return
        plan=SpatialPlan(mode=mode,radius=self.point_radius.value()).validate()
        answer=QtWidgets.QMessageBox.question(self,'有界空间标定',
            f'确认仍为原ND／低功率照明，工作光束及OFF光均有防护？\n'
            f'本轮{len(plan.groups())}组，每组3帧，最多240秒；使用全幅CCD。\n'
            'SLM固定零灰度，DMD仅投单个圆点。测后两路全0，CCD继续；Esc停止。\n'
            '这一步不证明SLM已对齐，也不自动采用结果。',
            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
        if answer!=QtWidgets.QMessageBox.Yes:return
        self.spatial_busy=True;self.reference_busy=True;self.override=True
        self.registration=None;self.comparison_changed()
        self.armed.setChecked(True)
        try:self.worker.request_spatial(plan)
        except Exception as error:
            self.spatial_busy=False;self.reference_busy=False;self.override=False
            self.armed.setChecked(False);self.black_outputs();self.note(str(error));self.refresh();return
        self.spatial_status.setText('正在采集：'+mode+'；固定计划、独立验证，尚未完成。')
        self.note('开始有界空间标定；等待屏幕实际读回后采样，不运行ASM或SLM细扫。');self.refresh()

    def spatial_received(self,report):
        self.last_spatial_report=report
        if report.get('error'):
            self.spatial_status.setText('未采用：'+report['error']+'；原始帧保留。')
        else:
            r=report['analysis']
            if r['mode']=='pilot':
                text=('中心点检查通过' if r['pilot_validated'] else '中心点重复性未通过')+f'；返回漂移{r["return_drift_px"]:.3f} CCD像素。尚无全场几何。'
            else:
                text=('空间几何独立验证通过' if r['spatial_geometry_validated'] else '空间几何验证未通过')+f'；验证P95={r["validation_p95_px"]:.3f}，最大={r["validation_max_px"]:.3f} CCD像素。'
            self.spatial_status.setText(text+'\nSLM频率映射/LUT未标定；结果未自动写入模型。')
            self.sim_view.set_image(geometry_overlay(r,report['active']),
                (int(report['active']['Width']),int(report['active']['Height'])))
            self.diff_view.setText('点阵误差箭头×20（左侧图示）\n绿色=拟合点；橙色=独立验证点\n不逐图配准，不是ASM仿真CCD')
        self.note('空间标定记录：'+report['folder']);self.refresh()

    def finish_spatial_display(self,error=None):
        request=self.spatial_display_pending
        if request is None:return
        if error is None:
            if self.last_verified_revision!=self.revision:return
            request['video_verification']=dict(matches=True,revision=self.revision,
                DMD_sha256=hashlib.sha256(request['dmd'].tobytes()).hexdigest(),
                SLM_canonical_sha256=hashlib.sha256(request['slm'].tobytes()).hexdigest(),
                actual_framebuffer_checked=True,hardware_synchronized=False)
        else:request['error']=str(error)
        self.spatial_display_pending=None;request['done'].set()

    def changed(self,*args):
        if not self.building:
            self.dirty=True;self.comparison_changed()

    def build_comparison_controls(self,left):
        box=QtWidgets.QGroupBox('仿真参数 / 显示映射（未标定）');body=QtWidgets.QVBoxLayout(box)
        self.target_mode=QtWidgets.QComboBox()
        self.target_mode.addItem('固定理想居中目标（用于对齐）',True)
        self.target_mode.addItem('跟随实际SLM的XY（仅观察模型）',False)
        self.target_mode.currentIndexChanged.connect(self.comparison_changed)
        self.diff_mode=QtWidgets.QComboBox()
        self.diff_mode.addItem('差值：红=实测偏强，蓝=仿真偏强',False)
        self.diff_mode.addItem('叠加：红=实测，绿=仿真，黄=重合',True)
        self.diff_mode.currentIndexChanged.connect(self.repaint_comparison)
        self.zero_button=QtWidgets.QPushButton('显示参考配准（旧零相位法）')
        self.zero_button.clicked.connect(self.capture_zero_reference)
        self.mapping_status=StatusLabel('未标定倍率/位置：当前仅粗略初值；SLM的X/Y不能修正倍率误差。',3)
        button=QtWidgets.QPushButton('保存粗对齐检查点（不自动扫描）')
        button.clicked.connect(self.save_coarse_checkpoint)
        self.checkpoint_button=button
        toggle=QtWidgets.QToolButton();toggle.setText('展开仿真参数 / 视场映射');toggle.setCheckable(True);body.addWidget(toggle)
        advanced=QtWidgets.QWidget();form=QtWidgets.QFormLayout(advanced);advanced.hide()
        toggle.toggled.connect(advanced.setVisible);body.addWidget(advanced)
        self.optics={}
        for key,label,low,high,value,unit,decimals in (
            ('wavelength','波长',350,1100,532,' nm',1),('pitch','共同网格采样',2,30,8,' µm',2),
            ('f1','Len2焦距（DMD侧）',50,1000,150,' mm',1),('f2','Len3焦距（CCD侧）',50,1000,100,' mm',1),
            ('phase_depth','假定灰度相位深度',.1,12.566,2*np.pi,' rad',4)):
            spin=QtWidgets.QDoubleSpinBox();spin.setRange(low,high);spin.setDecimals(decimals);spin.setValue(value)
            spin.setSuffix(unit);spin.setKeyboardTracking(False);spin.valueChanged.connect(self.optics_changed)
            self.optics[key]=spin;form.addRow(label,spin)
        self.slm_flip_y=QtWidgets.QCheckBox('Y上下反射（仅仿真，不改投屏）')
        self.slm_flip_y.setChecked(self.slm_coordinate_hint['slm_to_model_flip_y'])
        self.slm_flip_y.setToolTip('SLM相位平面到模型的独立坐标变换，由正负光栅测得；不是CCD镜像或硬件相位翻转。')
        self.slm_flip_y.toggled.connect(self.optics_changed);form.addRow('SLM相位坐标',self.slm_flip_y)
        self.orientation=QtWidgets.QComboBox()
        for code in range(8):self.orientation.addItem(orientation_name(code),code)
        self.orientation.setCurrentIndex(self.orientation_hint['simulation_to_native_CCD_orientation'])
        form.addRow('仿真方向（CCD不旋转）',self.orientation)
        self.map_scale=QtWidgets.QDoubleSpinBox();self.map_scale.setRange(.1,5);self.map_scale.setDecimals(3);self.map_scale.setValue(1);self.map_scale.setSingleStep(.02)
        self.map_x=QtWidgets.QDoubleSpinBox();self.map_y=QtWidgets.QDoubleSpinBox()
        for control in (self.map_x,self.map_y):control.setRange(-3000,3000);control.setDecimals(1);control.setSuffix(' CCD px');control.setSingleStep(2)
        for label,control in (('显示缩放',self.map_scale),('显示平移X',self.map_x),('显示平移Y',self.map_y)):
            control.setKeyboardTracking(False);form.addRow(label,control);control.valueChanged.connect(self.comparison_changed)
        self.orientation.currentIndexChanged.connect(self.comparison_changed)
        reset=QtWidgets.QPushButton('清除冻结几何 / 复位显示映射');reset.clicked.connect(self.clear_mapping);form.addRow(reset)
        note=QtWidgets.QLabel('这些控件只改仿真/显示，不改实物透镜。共同8µm网格为参考近似；灰度/255×2π不是实测LUT。差值独立归一化，仅比较形状。')
        note.setWordWrap(True);body.addWidget(note);left.addWidget(box)

    def optical_config(self):
        o={key:spin.value() for key,spin in self.optics.items()}
        return OpticalReference(wavelength_m=o['wavelength']*1e-9,pitch_m=o['pitch']*1e-6,
            f1_m=o['f1']*.001,f2_m=o['f2']*.001,phase_depth_rad=o['phase_depth'],slm_flip_y=self.slm_flip_y.isChecked())

    def comparison_changed(self,*args):
        if self.building:return
        self.comparison_epoch+=1;self.last_comp_submit=0.;self.comparison_error=None
        self.last_comparison=None
        if hasattr(self,'comparison_status'):self.comparison_status.setText('参考或控件已改变，等待后台更新；旧比较不再评分。')
        self.update_direction_status()

    def update_direction_status(self):
        if not hasattr(self,'direction_status'):return
        code=self.registration['orientation'] if self.registration else self.orientation.currentData()
        self.direction_status.setText(orientation_name(code)+(' · 零参考已冻结' if self.registration else ' · 仅显示方向，几何未标定')+
                                     '；SLM相位Y'+('反射' if self.slm_flip_y.isChecked() else '原向')+'（仅仿真）')
        enabled=not self.registration and not self.reference_busy
        self.orientation.setEnabled(enabled)
        for button in self.direction_buttons:button.setEnabled(enabled)

    def edit_preview_orientation(self,action):
        if self.registration or self.reference_busy:return
        self.orientation.setCurrentIndex(edited_orientation(self.orientation.currentData(),action))

    def restore_platform_orientation(self):
        if not self.registration and not self.reference_busy:
            self.orientation.setCurrentIndex(self.orientation_hint['simulation_to_native_CCD_orientation'])

    def optics_changed(self,*args):
        if self.building:return
        self.clear_mapping();self.comparison_changed()

    def reset_mapping_controls(self):
        # ROI/exposure invalidates scale/translation, not optical mirror parity.
        # Keep the latest camera-direction choice rather than reverting to raw ASM.
        for control,value in ((self.map_scale,1.),(self.map_x,0.),(self.map_y,0.)):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)

    def clear_mapping(self):
        self.registration=None;self.reset_mapping_controls()
        self.mapping_status.setText('未标定倍率/位置：当前仅粗略初值；请采零相位参考后再精调SLM。')
        self.orientation.setEnabled(True);self.comparison_changed()

    def ensure_diffraction_worker(self):
        if self.diffraction is None:
            self.diffraction=DiffractionWorker()
            self.diffraction.result.connect(self.comparison_received)
            self.diffraction.failed.connect(self.comparison_failed);self.diffraction.start()

    def reference_request_for(self,packet):
        return dict(epoch=self.comparison_epoch,settings=self.setting(),config=self.optical_config(),
            centered=self.target_mode.currentData(),dmd_file=self.dmd_file,slm_file=self.slm_file,
            active=dict(packet['active']),measured=packet['preview_raw'],frame_id=packet['meta']['FrameID'],
            frame_received=packet['received_monotonic'],mapping=dict(orientation=self.orientation.currentData(),
                scale=self.map_scale.value(),dx=self.map_x.value(),dy=self.map_y.value(),registration=self.registration))

    def update_comparison(self,now):
        if not self.running or not self.live_ready:
            if self.last_comparison:self.describe_comparison(now)
            return
        if not self.last_packet:return
        if self.hardware and (not self.can_project() or self.override):
            self.comparison_status.setText('未比较：当前未投手动图案／临时接管。下方若有图，仅为旧参考。');return
        if now-self.last_packet['received_monotonic']>1.5:
            self.comparison_status.setText('CCD已陈旧，停止更新比较；下方仅为留图。');return
        if self.registration and self.registration['signature']!=reference_signature(self.optical_config(),self.last_packet['active']):
            self.clear_mapping();self.note('ROI/曝光/模型条件已变，冻结几何已失效；可重新采零相位参考。')
        if now-self.last_comp_submit>=.4:
            self.last_comp_submit=now;self.ensure_diffraction_worker()
            self.diffraction.submit(self.reference_request_for(self.last_packet))
        if self.last_comparison:
            self.describe_comparison(now)

    def comparison_received(self,result):
        if result['kind']=='zero':
            if not self.last_packet or result['signature']!=reference_signature(self.optical_config(),self.last_packet['active']):
                self.note('零相位参考条件已改变，未采用映射。');return
            self.registration=result['registration'];self.reset_mapping_controls();self.orientation.setEnabled(False)
            self.orientation.blockSignals(True);self.orientation.setCurrentIndex(self.registration['orientation']);self.orientation.blockSignals(False)
            write_json(Path(self.registration['reference_folder'])/'frozen_geometry.json',self.registration)
            linear=np.asarray(self.registration['matrix_sensor'],float)[:,:2]
            sx,sy=np.linalg.norm(linear,axis=0)
            self.mapping_status.setText('倍率/位置已锁定：sx %.4f、sy %.4f CCD像素/模型格；零参考NCC %.3f。不是相位/LUT标定。'%
                (sx,sy,self.registration['correlation_zero']))
            self.comparison_changed();self.note('零相位几何已冻结；继续手动调SLM，目标不会逐候选重拟合。');return
        if result['epoch']!=self.comparison_epoch:return
        self.last_comparison=result;self.comparison_error=None;self.repaint_comparison()
        self.describe_comparison(time.monotonic());self.refresh()

    def comparison_failed(self,result):
        if result['epoch']!=self.comparison_epoch and result['kind']!='zero':return
        self.comparison_error=result['error'];self.comparison_status.setText('仿真／配准未采用：'+result['error'])
        if result['kind']=='zero':self.note('零参考配准失败：'+result['error']+'；可继续人工观察，不伪造几何。')

    def repaint_comparison(self,*args):
        if self.last_comparison:
            r=self.last_comparison
            active=r.get('active') or (self.last_packet or {}).get('active') or self.view.active_roi
            extent=None if not active else (int(active['Width']),int(active['Height']))
            self.sim_view.set_image(r['sim_gray'],extent)
            self.diff_view.set_image(r['overlay_rgb'] if self.diff_mode.currentData() else r['difference_rgb'],extent)

    def describe_comparison(self,now):
        r=self.last_comparison;m=r['metrics'];age=now-r['frame_received']
        fresh=self.running and self.live_ready and (not self.hardware or (self.can_project() and not self.override)) and age<2.
        score='NCC %.3f / 归一化RMSE %.3f'%(m['correlation'],m['normalized_rmse']) if m['valid'] else '对比度不足，不显示有效相关分数'
        config=self.optical_config()
        self.comparison_status.setText('%s · %s · f₁/f₂ %.0f/%.0fmm · FrameID %s，帧龄%.2fs，后台%.0fms\n%s；独立归一化形状差，不是ADU/相位误差；自由运行未硬同步。'%
            ('实时辅助' if fresh else '旧比较／更新中','冻结零相位几何' if self.registration else '未配准的显示映射',
             config.f1_m*1000,config.f2_m*1000,r['frame_id'],age,r['compute_ms'],score))

    def capture_zero_reference(self):
        if not self.can_project() or self.roi_busy or self.reference_busy or self.override:return
        self.reference_busy=True;self.override=True;self.reference_request=dict(config=self.optical_config(),settings=self.setting())
        self.worker.request_zero_reference();self.note('采零相位暗/非对称参考；同一ROI、曝光和SDK，不扫描SLM。');self.refresh()

    def zero_reference_captured(self,payload):
        # The camera worker sends one bounded explicit dataset, not a frame stream.
        request=dict(kind='zero',epoch=self.comparison_epoch,centered=True,
            config=self.reference_request['config'],settings=self.reference_request['settings'],
            measured=payload['measured'],active=payload['active'],folder=payload['folder'])
        self.ensure_diffraction_worker();self.diffraction.submit(request)
        write_json(Path(payload['folder'])/'model_reference.json',asdict(request['config']))
        self.note('参考已采集，后台拟合一次几何；CCD已恢复手动实时。')

    def save_coarse_checkpoint(self):
        if (not self.hardware or not self.last_comparison or not self.can_project() or self.override
                or time.monotonic()-self.last_comparison['frame_received']>2.):
            self.note('请先保持手动投图与新鲜比较，再保存粗对齐检查点。');return
        self.snapshot(checkpoint=True)

    def scope_changed(self,full):
        self.building=True
        for key,limit in zip(('x','y'),((960,540) if full else (660,240))):
            self.spins[key].setRange(-limit,limit);self.sliders[key].setRange(-limit,limit)
        self.building=False;self.changed()

    def load_file(self,phase):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'载入规范SLM灰度图' if phase else '载入DMD图','', '图像 (*.png *.bmp *.tif *.tiff *.jpg *.npy)')
        if not path:return
        try:
            image=load_image(path,phase)
            if phase:
                self.slm_file=image;self.full.setChecked(image.shape==(1080,1920));self.slm_mode.setCurrentIndex(6)
            else:self.dmd_file=image;self.dmd_mode.setCurrentIndex(4)
            self.changed()
            if self.hardware:self.record(dict(state='manual_file_loaded',path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),SLM=phase))
        except Exception as error:self.note(str(error))

    def paint_preview(self,label,values):
        values=np.ascontiguousarray(values);h,w=values.shape[:2]
        rgb=values.ndim==3
        image=QtGui.QImage(values.data,w,h,w*(3 if rgb else 1),
            QtGui.QImage.Format_RGB888 if rgb else QtGui.QImage.Format_Grayscale8).copy()
        label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(label.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation))

    def render_patterns(self):
        self.dirty=False
        try:
            self.desired_dmd,self.desired_slm=patterns(self.setting(),self.dmd_file,self.slm_file)
            self.paint_preview(self.dmd_preview,self.desired_dmd);self.paint_preview(self.slm_preview,np.fliplr(self.desired_slm))
            s=self.setting();self.phase_caption.setText(('整屏透镜中心=(%d,%d)'%(960+s.x,540+s.y) if s.full_panel else '600²窗口偏移=(%d,%d)'%(s.x,s.y))+
                '；X向右/Y向下。'+('实时加载中' if self.can_project() else '尚未投图'))
            if self.can_project() and not self.override:self.show_outputs(self.desired_dmd,self.desired_slm)
        except Exception as error:self.note(str(error));self.black_outputs()

    def can_project(self):return self.hardware and self.running and self.live_ready and self.armed.isChecked() and not self.fault

    def projection_block_reason(self):
        if not self.hardware:return '离线预览，不连接硬件'
        if self.fault:return self.fault
        if not self.running:return '尚未开始／已停止'
        if not self.live_ready:return '等待CCD首张完整安全帧'
        if not self.armed.isChecked():return '当前仅CCD，未允许DMD/SLM投图'
        if self.override:return 'ROI定位／重配临时接管，不加载手动图案'
        return None

    def update_output_state(self):
        reason=self.projection_block_reason()
        if reason:text='DMD/SLM未投图：'+reason
        else:
            verified=self.last_verified_revision==self.revision
            def level(role):return '非零图案' if self.expected.get(role,np.zeros(1)).any() else '全0图案'
            text=('实际屏幕读回一致' if verified else '已请求投图，等待屏幕读回')+'：DMD '+level('dmd')+'；SLM '+level('slm')
        if self.output_state.text()!=text:self.output_state.setText(text)
        gate=(reason,self.armed.isChecked(),self.running,self.live_ready,self.override)
        if self.hardware and gate!=self.last_gate:
            self.last_gate=gate
            self.record(dict(state='projection_gate',blocked_reason=reason,armed=gate[1],running=gate[2],live_ready=gate[3],override=gate[4]))

    def show_outputs(self,dmd,slm):
        if not self.hardware:return
        self.dmd_output.update_image(dmd)
        physical=np.ascontiguousarray(np.fliplr(slm));self.slm_output.update_image(physical)
        panel=np.zeros((600,800),np.uint8);panel[:,100:700]=dmd
        self.expected=dict(dmd=panel,slm=physical);self.last_apply=time.monotonic();self.revision+=1;self.mismatches=0
        self.last_verified_revision=None
        self.last_arrays=dict(dmd=dmd.copy(),slm=slm.copy())
        self.record(dict(state='manual_display_requested',revision=self.revision,settings=asdict(self.setting()),
                         dmd_sha256=hashlib.sha256(dmd.tobytes()).hexdigest(),slm_canonical_sha256=hashlib.sha256(slm.tobytes()).hexdigest(),
                         dmd_nonzero=bool(dmd.any()),slm_nonzero=bool(slm.any()),blocked_reason=self.projection_block_reason(),
                         SLM_horizontal_flip_count=1,frame_pattern_association_verified=False))

    def black_outputs(self):
        if not self.hardware:return
        self.show_outputs(np.zeros((600,600),np.uint8),np.zeros((1080,1920),np.uint8))

    def arm_changed(self,enabled):
        if not enabled:
            self.black_outputs();self.finish_spatial_display('操作者取消投图')
            if self.spatial_busy and self.worker:self.worker.stop()
        if self.hardware:self.record(dict(state='projection_confirmation_changed',enabled=bool(enabled)))
        self.dirty=True
        self.refresh()

    def dmd_black(self):
        self.dmd_mode.setCurrentIndex(3)
        if self.can_project() and not self.override:self.show_outputs(np.zeros((600,600),np.uint8),self.desired_slm)
        else:self.black_outputs()

    def start_live(self,project=True):
        if not self.hardware or self.fault or (self.worker and self.worker.isRunning()):return
        if project and not self.armed.isChecked():
            answer=QtWidgets.QMessageBox.question(self,'启动三路实时',
                '此操作会将所选图案加载到DMD、SLM，并启动CCD。\n确认仍为已衰减的ND／低功率照明，且工作光束和OFF光均有防护？',
                QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
            if answer!=QtWidgets.QMessageBox.Yes:return
            self.armed.setChecked(True)
        if not project:self.armed.setChecked(False)
        self.record(dict(state='manual_live_start',projection_requested=bool(project),low_power_confirmed=self.armed.isChecked()))
        self.black_outputs();self.running=True;self.live_ready=False;self.override=False;self.roi_busy=False
        self.dirty=True
        self.last_sequence=0;self.last_packet=None;self.view.setText('正在打开唯一相机15328133…')
        folder=self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker=LiveCameraWorker(self.baseline,folder,self.exposure.value())
        self.history.append(self.worker);self.worker.ready.connect(self.camera_ready)
        self.worker.configured.connect(self.camera_configured);self.worker.roi_found.connect(self.roi_found)
        self.worker.zero_reference.connect(self.zero_reference_captured)
        self.worker.spatial_result.connect(self.spatial_received)
        self.worker.display.connect(self.display_request);self.worker.message.connect(self.note)
        self.worker.black.connect(self.black_outputs);self.worker.result.connect(self.camera_finished)
        self.worker.finished.connect(self.refresh);self.worker.start();self.refresh()

    def camera_ready(self,payload):
        self.live_ready=True;self.dirty=True
        self.note('首张CCD完整帧已收到；'+('DMD/SLM开始随控件实时加载。' if self.armed.isChecked() else '当前仅CCD：DMD/SLM保持全0，可勾选允许投图。'))
        self.refresh()

    def camera_configured(self,payload):
        self.last_packet=None;self.comparison_changed()
        if self.registration and self.registration['signature']!=reference_signature(self.optical_config(),payload['active']):self.clear_mapping()
        if self.actual_roi!=payload['actual_roi']:
            self.view.setText('ROI已改变，等待新CCD')
            self.sim_view.setText('ROI已改变，等待新比较');self.diff_view.setText('ROI已改变，等待新比较')
        self.actual_roi=payload['actual_roi'];self.candidate=None
        self.roi_text.setText('实际ROI：'+str(self.actual_roi)+'；曝光实际'+payload['active']['ExposureTime']+'µs')
        self.refresh()

    def display_request(self,request):
        deferred=False
        try:
            kind=request['kind']
            if kind=='hold_black':self.override=True;self.black_outputs()
            elif kind=='spatial_park':
                self.finish_spatial_display('测量结束')
                self.spatial_busy=False;self.reference_busy=False;self.override=False
                self.armed.setChecked(False);self.black_outputs();self.dirty=True
            elif kind=='spatial_override':
                if not self.can_project() or not self.spatial_busy:
                    raise RuntimeError('当前不允许点阵投图')
                if self.spatial_display_pending:raise RuntimeError('已有点阵投图请求未确认')
                self.override=True;self.show_outputs(request['dmd'],request['slm'])
                self.paint_preview(self.dmd_preview,request['dmd'])
                self.paint_preview(self.slm_preview,np.fliplr(request['slm']))
                self.spatial_display_pending=request;deferred=True
            elif kind=='resume':
                self.override=False;self.roi_busy=False;self.reference_busy=False;self.dirty=True
                if not self.can_project():self.black_outputs()
            elif kind in ('roi_override','reference_override'):
                if not self.can_project():raise RuntimeError('当前不允许ROI投图')
                self.override=True;self.show_outputs(request['dmd'],request['slm'])
            else:raise RuntimeError('unknown display command')
        except Exception as error:request['error']=str(error);self.black_outputs()
        finally:
            if not deferred:request['done'].set()
            self.refresh()

    def select_roi(self,roi):
        self.candidate=roi;self.roi_text.setText('待确认ROI：'+str(roi));self.refresh()

    def roi_found(self,result):self.roi_busy=False;self.select_roi(result['requested_roi'])

    def auto_roi(self):
        if not self.can_project() or self.roi_busy or self.reference_busy or self.override:return
        self.roi_busy=True;self.worker.auto_roi();self.note('标准方块定位中；CCD继续显示，结果需点确认ROI。');self.refresh()

    def configure_camera(self,roi):
        if not self.running or not self.live_ready or self.roi_busy or self.reference_busy:return
        self.override=True;self.black_outputs();self.worker.configure(roi,self.exposure.value())

    def apply_roi(self):
        if self.candidate:self.configure_camera(self.candidate)
    def full_sensor(self):self.configure_camera(dict(OffsetX=0,OffsetY=0,Width=0,Height=0))
    def apply_exposure(self):self.configure_camera(self.actual_roi)

    def stop_live(self):
        self.finish_spatial_display('停止点阵测量')
        self.spatial_busy=False
        self.running=False;self.live_ready=False;self.override=False;self.roi_busy=False;self.reference_busy=False
        self.armed.setChecked(False);self.black_outputs()
        self.dirty=True
        if self.worker and self.worker.isRunning():self.worker.stop()
        self.comparison_status.setText('已停止；下方为旧参考，不是当前CCD比较。')
        self.note('已请求停止并切两路全0；保留器件窗口，CCD最后一帧不再视为实时。');self.refresh()

    def cancel(self):self.stop_live()

    def camera_finished(self,report):
        self.finish_spatial_display('相机采集已结束');self.spatial_busy=False
        self.last_camera_report=report;self.running=False;self.live_ready=False;self.roi_busy=False;self.override=False;self.reference_busy=False
        self.armed.setChecked(False);self.black_outputs()
        if not report.get('camera_settings_restored'):
            self.fault='相机恢复未核验：'+str(report.get('restoration_error') or report.get('error'))
            self.fault_before=report.get('camera_before')
        text=report.get('error') or ('实时采集已停止，相机已恢复原参数。' if report.get('camera_settings_restored')
                                   else '实时采集已停止，原参数恢复尚未通过。')
        self.note((self.fault+'；' if self.fault else '')+text);self.refresh()

    def verify_parked_outputs(self):
        if self.running or self.live_ready or self.armed.isChecked():
            raise RuntimeError('先停止采集和投图，再复核；不自动改变运行状态')
        if set(self.windows)!=set(('dmd','slm')) or set(self.expected)!=set(self.windows):
            raise RuntimeError('缺少原器件输出窗或全0请求')
        same_signals(self.baseline,read_roles())
        checks={}
        for role,window in self.windows.items():
            row=self.baseline[role];g=window.geometry()
            rectangle=row['desktop_position']+row['desktop_size']
            if [g.x(),g.y(),g.width(),g.height()]!=rectangle or not window.isVisible():
                raise RuntimeError(role+'原输出窗不可见或位置改变')
            if not window.wait_until_applied(0):raise RuntimeError(role+'全0请求尚未完成')
            expected=self.expected[role]
            if expected.shape!=(row['desktop_size'][1],row['desktop_size'][0]) or expected.any():
                raise RuntimeError(role+'请求不是原生尺寸全0')
            checks[role]=self.pixels.compare(tuple(rectangle),expected)
            if not checks[role]['matches']:raise RuntimeError(role+'实际输出不是全0，保留故障锁')
        return checks

    def start_recheck(self):
        if not self.hardware or not self.fault or self.recheck_context or (self.worker and self.worker.isRunning()):return
        target=recovery_target(self.fault,self.fault_before,self.last_camera_report)
        if not target:self.note('没有可追溯的原始相机before，不能猜测复核目标。');return
        try:checks=self.verify_parked_outputs()
        except Exception as error:self.note('原位复核未启动：'+str(error));return
        self.recheck_context=dict(fault=self.fault,target=target,precheck=checks)
        self.record(dict(state='original_baseline_recheck_started',context=self.recheck_context,
                         no_automatic_capture=True,no_automatic_projection=True))
        self.worker=RecheckWorker(target,self.baseline,self.output/('recheck_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        self.history.append(self.worker);self.worker.result.connect(self.recheck_finished);self.worker.finished.connect(self.refresh)
        self.worker.start();self.refresh()

    def recheck_finished(self,report):
        context=self.recheck_context;self.recheck_context=None
        result=dict(camera_report=report,original_context=context,unlocked=False)
        try:
            if not context or self.fault!=context['fault']:raise RuntimeError('故障已改变，旧复核不能解锁')
            if (report.get('passed') is not True or report.get('target')!=context['target']
                    or report.get('parameter_writes')!=0 or report.get('acquisition_started') is not False
                    or report.get('connections')!=2 or len(report.get('snapshots',[]))!=10
                    or report.get('errors') or report.get('error')
                    or any(s.get('differences') or s.get('values')!=context['target'] for s in report.get('snapshots',[]))):
                raise RuntimeError('原始相机参数的只读重复复核未通过')
            result['postcheck']=self.verify_parked_outputs()
            self.fault=None;self.fault_before=None;self.last_verified_revision=self.revision
            result['unlocked']=True
            self.note('原窗口全0、显示身份和原始相机参数复核通过；可重新开始，不自动投图/采集。')
        except Exception as error:
            result['error']=str(error);self.note('复核未通过，保留故障锁：'+str(error))
        self.record(dict(state='original_baseline_recheck_finished',result=result))
        self.refresh()

    def refresh(self):
        active=bool(self.worker and self.worker.isRunning())
        if not hasattr(self,'start_button'):return
        self.start_button.setEnabled(self.hardware and not active and not self.fault)
        self.camera_only_button.setEnabled(self.hardware and not active and not self.fault)
        self.snap_button.setEnabled(bool(self.running and self.last_packet and not self.spatial_busy))
        self.auto_button.setEnabled(bool(self.can_project() and not self.override and not self.roi_busy and not self.reference_busy))
        self.apply_roi_button.setEnabled(bool(self.running and self.live_ready and self.candidate and not self.roi_busy and not self.reference_busy))
        self.full_button.setEnabled(bool(self.running and self.live_ready and not self.roi_busy and not self.reference_busy))
        target=recovery_target(self.fault,self.fault_before,self.last_camera_report)
        self.recheck.setEnabled(bool(self.hardware and self.fault and target and not active and not self.recheck_context))
        self.armed.setEnabled(self.hardware and not self.fault)
        available=bool(self.hardware and self.running and self.live_ready and not self.fault
                       and not self.override and not self.reference_busy and not self.roi_busy)
        self.pilot_button.setEnabled(available)
        self.grid_button.setEnabled(available and self.pilot_matches())
        self.point_radius.setEnabled(not self.reference_busy)
        self.update_direction_status()
        if hasattr(self,'zero_button'):
            self.zero_button.setEnabled(bool(self.can_project() and not self.override and not self.roi_busy and not self.reference_busy))
            self.checkpoint_button.setEnabled(bool(self.can_project() and not self.override and self.last_comparison))
            for control in (self.dmd_mode,self.slm_mode,self.full,self.square,self.focal,self.period,
                            *self.spins.values(),*self.sliders.values(),*self.optics.values(),self.slm_flip_y,self.exposure):
                control.setEnabled(not self.reference_busy)

    def verify_outputs(self,now):
        if not self.hardware or not self.expected:return
        if self.fault and self.fault.startswith('显示核验失败'):return
        if isinstance(self.worker,RecheckWorker) and self.worker.isRunning():return
        if now-self.last_apply<.15 or now-self.last_verify<1.:return
        self.last_verify=now
        try:
            for role,window in self.windows.items():
                row=self.baseline[role];g=window.geometry()
                if [g.x(),g.y(),g.width(),g.height()]!=row['desktop_position']+row['desktop_size']:
                    raise RuntimeError(role+'输出窗口位置改变')
                if not window.wait_until_applied(0):return
                checked=self.pixels.compare((*row['desktop_position'],*row['desktop_size']),self.expected[role])
                if not checked['matches']:
                    self.record(dict(state='actual_video_mismatch',role=role,revision=self.revision,
                                     pixels=checked,consecutive_observations=self.mismatches+1,
                                     expected_nonzero=bool(self.expected[role].any()),cause_not_yet_identified=True))
                    raise RuntimeError(role+'实际显示被覆盖或与请求不同')
            if self.last_verified_revision!=self.revision:
                self.last_verified_revision=self.revision
                self.record(dict(state='actual_video_verified',revision=self.revision,
                    dmd_nonzero=bool(self.expected['dmd'].any()),slm_nonzero=bool(self.expected['slm'].any()),
                    physical_optical_response_verified=False))
            self.mismatches=0
        except Exception as error:
            self.mismatches+=1
            if self.mismatches>=2:
                self.fault='显示核验失败：'+str(error);self.stop_live()

    def tick(self):
        now=time.monotonic()
        if self.previous_gui_tick is not None:self.gui_gaps_ms.append((now-self.previous_gui_tick)*1000)
        self.previous_gui_tick=now
        if self.dirty and not self.override:self.render_patterns()
        if isinstance(self.worker,LiveCameraWorker):
            sequence,packet=self.worker.mailbox.get()
            if packet is not None and sequence!=self.last_sequence:
                self.last_sequence=sequence;self.last_packet=packet
                key='preview_enhanced' if self.enhance.isChecked() else 'preview_raw'
                self.view.set_frame(packet[key],packet['active'],False,prepared=True)
        if self.last_packet:
            p=self.last_packet;age=now-p['received_monotonic']
            live=self.running and age<1.5
            self.telemetry.setText(('%s · CCD %.1f fps · FrameID %s · 最后帧 %.2fs · 最大值 %d/255 · 完整%d / 坏%d\n'
                '实际%s×%s，曝光%sµs，Gain%s；自由运行，非硬同步。GUI预览最多20fps，采集与显示分别计时。') %
                ('实时' if live else '已停止／旧帧',p['fps'],p['meta']['FrameID'],age,p['max_ADU'],p['frames'],p['bad'],
                 p['active']['Width'],p['active']['Height'],p['active']['ExposureTime'],p['active']['Gain']))
        # camera_ready can arrive before tick consumes the first mailbox
        # packet. CCD-only never toggles the projection checkbox to refresh
        # controls, so readiness must follow actual fresh frames here.
        self.snap_button.setEnabled(bool(self.running and self.last_packet and not self.spatial_busy
            and now-self.last_packet['received_monotonic']<=1.5))
        self.verify_outputs(now)
        self.finish_spatial_display()
        self.update_output_state()
        self.update_comparison(now)
        if self.hardware and now-self.last_heartbeat>=2:
            self.last_heartbeat=now
            row=dict(pid=os.getpid(),unix_ns=time.time_ns(),busy=bool(self.worker and self.worker.isRunning()),
                     live_camera=self.running,projection_armed=self.armed.isChecked(),hardware_fault=self.fault,
                     display_revision=self.revision,state_text=self.state.text(),mode='continuous_manual',
                     projection_blocked_reason=self.projection_block_reason(),last_verified_revision=self.last_verified_revision,
                     gui_timer_gap_ms=None if not self.gui_gaps_ms else dict(latest=self.gui_gaps_ms[-1],
                         recent_max=max(self.gui_gaps_ms),recent_over_250ms=sum(gap>250 for gap in self.gui_gaps_ms),
                         sample_count=len(self.gui_gaps_ms)),
                     latest_frame=None if not self.last_packet else dict(frame_id=self.last_packet['meta']['FrameID'],
                         fps=self.last_packet['fps'],age_seconds=now-self.last_packet['received_monotonic'],
                         timing=self.last_packet.get('timing')),
                     optical_reference=asdict(self.optical_config()),
                     manual_settings=asdict(self.setting()),actual_roi=self.actual_roi,
                     comparison=None if not self.last_comparison else dict(epoch=self.comparison_epoch,
                         frame_id=self.last_comparison['frame_id'],compute_ms=self.last_comparison['compute_ms'],
                         frame_age_seconds=now-self.last_comparison['frame_received'],metrics=self.last_comparison['metrics']),
                     geometry_frozen=bool(self.registration),automatic_fine_scan=False,
                     spatial_calibration=dict(busy=self.spatial_busy,status=self.spatial_status.text(),
                         folder=None if not self.last_spatial_report else self.last_spatial_report['folder'],
                         geometry_installed_in_model=False),
                     matched_sensor_display=True,
                     slm_phase_to_model=dict(flip_y=self.slm_flip_y.isChecked(),simulation_only=True,
                                            hardware_output_transform_changed=False,CCD_orientation_changed=False),
                     view_geometry=dict(measured=[self.view.width(),self.view.height()],
                         simulated=[self.sim_view.width(),self.sim_view.height()],difference=[self.diff_view.width(),self.diff_view.height()],
                         measured_image_rect=self.view.image_rect,simulated_image_rect=self.sim_view.image_rect),
                     simulation_display_orientation=dict(code=self.orientation.currentData(),
                         name=orientation_name(self.orientation.currentData()),display_only=True,
                         hint_code=self.orientation_hint['simulation_to_native_CCD_orientation']),
                     heartbeat_is_NOT_physical_safety_interlock=True)
            tmp=self.output/'heartbeat.tmp';write_json(tmp,row);tmp.replace(self.output/'heartbeat.json')

    def snapshot(self,checked=False,*,checkpoint=False):
        if self.spatial_busy:
            self.note('点阵原始帧由采集线程自动保存；不另存易混淆的手动快照。');return
        if not self.hardware or not self.last_packet or not self.running:return
        p=self.last_packet
        if time.monotonic()-p['received_monotonic']>1.5:self.note('CCD帧已陈旧，未保存为实时快照');return
        folder=self.output/('snapshot_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir()
        np.save(folder/'CCD_raw.npy',p['image'],allow_pickle=False);Image.fromarray(p['image']).save(folder/'CCD_raw.png')
        for name,array in self.last_arrays.items():Image.fromarray(array).save(folder/(name+'_canonical_requested.png'))
        write_json(folder/'metadata.json',dict(frame=p['meta'],camera_active=p['active'],settings=asdict(self.setting()),
            display_revision=self.revision,frame_pattern_association_verified=False,raw_phase_no_LUT=True,release_approved=False))
        if self.last_comparison:
            r=self.last_comparison
            for name in ('sim_gray','difference_rgb','overlay_rgb','measured_preview'):
                Image.fromarray(r[name]).save(folder/(name+'.png'))
            for name in ('target_dmd','target_phase'):Image.fromarray(r[name]).save(folder/(name+'_canonical.png'))
            write_json(folder/'comparison.json',dict(model=asdict(self.optical_config()),
                comparison_frame_id=r['frame_id'],comparison_frame_age_seconds=time.monotonic()-r['frame_received'],
                snapshot_frame_id=p['meta']['FrameID'],same_camera_frame=r['frame_id']==p['meta']['FrameID'],
                metrics=r['metrics'],target_settings=asdict(r['target_settings']),
                target_centered=self.target_mode.currentData(),geometry=self.registration,matrix_sensor=r['matrix_sensor'],
                manual_display_mapping=dict(scale=self.map_scale.value(),dx=self.map_x.value(),dy=self.map_y.value()),
                simulation_to_native_CCD_orientation=self.orientation.currentData(),
                display_orientation_hint=self.orientation_hint,raw_CCD_direction_unchanged=True,
                slm_phase_coordinate_hint=self.slm_coordinate_hint,
                matched_sensor_display=True,raw_CCD_not_resized_for_control=True,
                fine_alignment_prerequisites=bool(self.registration and self.target_mode.currentData()
                    and self.setting().slm=='lens' and r['metrics']['valid']
                    and self.map_scale.value()==1 and self.map_x.value()==0 and self.map_y.value()==0
                    and time.monotonic()-r['frame_received']<2.),
                frame_pattern_association_verified=False,release_approved=False,
                coarse_checkpoint=checkpoint,fine_scan_started=False,
                requires_operator_coarse_done_confirmation=True))
        if checkpoint:self.note('粗对齐检查点已保存；尚未自动细扫，等待你确认粗对齐完成。')
        self.note('快照已保存：'+str(folder))

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():self.stop_live();event.ignore();return
        if self.diffraction and self.diffraction.isRunning():
            self.diffraction.stop()
            if not self.diffraction.wait(200):
                self.note('后台仿真正在结束，请稍后再关闭；器件窗口保持原位。');event.ignore();return
        if not self.hardware:self.timer.stop();event.accept();return
        self.busy=False
        Bench.closeEvent(self,event)
        if event.isAccepted():
            self.timer.stop()
            if self.pixels:self.pixels.close();self.pixels=None
            self.record(dict(state='operator_confirmed_light_blocked_exit',closed_safely=self.closed_safely))


def main():
    import cv2
    cv2.setNumThreads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware',action='store_true');parser.add_argument('--light-blocked-confirmed',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.hardware and (not args.light_blocked_confirmed or args.smoke_test):parser.error('真实器件启动须用户确认物理遮光；smoke不能连硬件')
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling,True)
    app=QtWidgets.QApplication(['ballplate-realtime-manual'])
    if args.hardware and app.platformName()!='xcb':raise RuntimeError('real local X11 required')
    output=args.output or Path(__file__).resolve().parents[1]/'outputs/live_calibration'/('MANUAL_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    window=RealtimeWindow(output,args.hardware)
    if args.smoke_test:
        assert not window.windows and not window.output.exists()
        window.spins['x'].setValue(100);window.render_patterns();window.start_live()
        assert window.worker is None
        QtCore.QTimer.singleShot(250,app.quit)
    app.exec_()


if __name__=='__main__':main()
