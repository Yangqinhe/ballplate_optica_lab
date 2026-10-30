"""Minimal single-frame adaptation page; no hardware or simulator ownership."""
import json
from pathlib import Path
import time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from live_deploy_v20.page import detector_overlay, SensorView
from live_deploy_v21.page import display_signal
from live_adapt_v22.core import Bundle


class AdaptationPage(QtWidgets.QWidget):
    collect_requested = QtCore.pyqtSignal()
    pilot_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.bundle = None; self.head_path = None; self.busy = False; self.last_view = 0.
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('真实单帧CCD → 固定8区积分 → Linear(8,2)，共18参数 → 仅MuJoCo'))
        row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit(); self.path.setReadOnly(True)
        self.load = QtWidgets.QPushButton('载入单帧适配包'); self.load.clicked.connect(self.choose)
        row.addWidget(self.path, 1); row.addWidget(self.load); layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox(); self.mode.addItem('固定学习mask', 'learned'); self.mode.addItem('平面mask消融（独立重拟合）', 'flat')
        self.confirm = QtWidgets.QCheckBox('原ND／低功率、防护及器件位置未变；允许有界采集')
        row.addWidget(self.mode); row.addWidget(self.confirm, 1); layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.collect = QtWidgets.QPushButton('1. 采集单帧并适配线性头')
        self.pilot = QtWidgets.QPushButton('2. 运行3秒MuJoCo慢速短闭环')
        self.cancel = QtWidgets.QPushButton('取消／两路全0')
        for b in (self.collect, self.pilot, self.cancel): row.addWidget(b)
        layout.addLayout(row)
        self.collect.clicked.connect(self.collect_requested); self.pilot.clicked.connect(self.pilot_requested)
        self.cancel.clicked.connect(self.cancel_requested)
        self.confirm.toggled.connect(self.update_buttons); self.mode.currentIndexChanged.connect(self.mode_changed)
        self.status = QtWidgets.QLabel('未开始。不会自动投图、加载旧actor或启动闭环。')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 80); layout.addWidget(self.progress)
        self.ccd = SensorView('CCD单帧（仅显示增强）')
        self.ccd.setMinimumSize(420, 360)
        layout.addWidget(self.ccd, 1)
        self.values = QtWidgets.QLabel('8维原始积分：—'); self.values.setWordWrap(True); layout.addWidget(self.values)
        self.summary = QtWidgets.QPlainTextEdit(); self.summary.setReadOnly(True); self.summary.setMaximumHeight(190)
        layout.addWidget(self.summary)
        layout.addWidget(QtWidgets.QLabel('短闭环暂停仿真等待真实光路，非20ms实时验收；没有实物执行器。Esc随时全0。'))
        self.update_buttons()

    def choose(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '单帧适配包', '', 'JSON (*.json)')
        if path:
            try: self.load_bundle(path)
            except Exception as error: self.status.setText('未采用：'+str(error))

    def load_bundle(self, path):
        if self.busy: raise RuntimeError('cannot change bundle during acquisition')
        bundle = Bundle(path)
        self.bundle = bundle; self.path.setText(str(bundle.path)); self.head_path = None
        self.progress.setRange(0, len(bundle.cases)); self.progress.setValue(0)
        self.summary.clear(); self.confirm.setChecked(False)
        self.status.setText('已载入80组预分轨迹输入；先启动唯一CCD并复核原ROI/倍率，再确认条件。')
        self.update_buttons()

    def mode_changed(self):
        self.head_path = None; self.update_buttons()

    def update_buttons(self):
        ready = self.bundle is not None and self.confirm.isChecked() and not self.busy
        self.collect.setEnabled(ready); self.pilot.setEnabled(ready and self.head_path is not None)
        self.cancel.setEnabled(self.busy)
        for w in (self.load, self.mode, self.confirm): w.setEnabled(not self.busy)

    def set_busy(self, value):
        self.busy = value; self.update_buttons()

    def fresh_view(self, packet, gain, black, gamma):
        if not self.bundle or time.monotonic()-self.last_view < .2: return
        self.last_view = time.monotonic(); raw = packet['image']
        if raw.shape != self.bundle.weights.shape[1:]: return
        self.ccd.set_image(detector_overlay(display_signal(raw, gain, black, gamma), self.bundle.weights), raw.shape[::-1])

    def on_progress(self, data):
        if data['kind'] == 'input':
            self.progress.setRange(0, data['total']); self.progress.setValue(data['count'])
            self.status.setText(f"真实采集 {data['count']}/{data['total']} · {data['split']} · 每张原图独立作为样本")
            self.values.setText('8维原始积分：'+', '.join(f'{v:.1f}' for v in data['raw_integrals8']))
        elif data['kind'] == 'mujoco_step':
            self.status.setText(f"MuJoCo {data['simulation_time_s']:.2f}/3.00 s · 光学查询{data['query_latency_s']:.2f}s（仿真已暂停等待）")
            self.values.setText('位置(m)：'+str(data['position_m'])+'；动作(rad)：'+str(data['action_limited_rad']))

    def finished(self, result):
        self.set_busy(False); report = result['report']; head = result.get('head')
        self.head_path = None
        if head:
            self.summary.setPlainText(json.dumps(dict(test=head['metrics']['test'],
                improvement=head['test_vs_train_constant_MSE_improvement'],
                criteria=head['test_criteria'], short_pilot_eligible=head['short_MuJoCo_pilot_eligible']),
                ensure_ascii=False, indent=2))
            if head['short_MuJoCo_pilot_eligible']:
                self.head_path = Path(result['folder'])/'HEAD.json'
                self.status.setText('单帧动作门槛通过，可运行3秒MuJoCo慢速短闭环；不是实时或稳定性认证。')
            else:
                self.status.setText('已保存真实数据与18参数头，但未见轨迹门槛未过：不运行短闭环。')
        elif report['task'] == 'MuJoCo_slow_pilot':
            self.summary.setPlainText(json.dumps({k: v for k, v in report.get('pilot', {}).items() if k != 'trace'}, ensure_ascii=False, indent=2))
            self.status.setText('短闭环已结束，输出全0。结果：'+str(report.get('error') or '见下方记录'))
        else:
            self.status.setText('未采用结果：'+str(report.get('error') or result.get('fit_error') or report.get('park_error')))
        self.status.setToolTip(result['folder']); self.update_buttons()
