"""Display gain/black level; never changes sensor values, readouts or outputs."""
import json
from pathlib import Path
import time

import numpy as np
from PyQt5 import QtWidgets
from live_deploy_v20.page import DeploymentPage, detector_overlay
from live_deploy_v20.contract import digest


def display_signal(values, gain=16., black=0., gamma=1.):
    """A new uint8 display buffer, not an in-place operation on raw/mean data."""
    if not 1 <= gain <= 128 or not 0 <= black <= 255 or not .3 <= gamma <= 3:
        raise ValueError('bounded display controls required')
    a = np.asarray(values, np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('finite sensor plane required')
    level = np.clip((a-black)*gain/255., 0., 1.)
    return np.rint(np.power(level, gamma)*255.).astype(np.uint8)


class DisplayGainPage(DeploymentPage):
    def __init__(self, parent=None):
        self.raw_frame = None
        self.prediction_signal = None
        self.last_result = None
        self.last_stage = None
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout()
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItems(['实时CCD · 原始单帧', '历史学习图 · 背景扣除均值（非实时）'])
        row.addWidget(self.view_mode)
        row.addWidget(QtWidgets.QLabel('仅显示增益'))
        self.display_gain = QtWidgets.QDoubleSpinBox()
        self.display_gain.setRange(1, 128); self.display_gain.setValue(16); self.display_gain.setSuffix(' ×')
        row.addWidget(self.display_gain)
        row.addWidget(QtWidgets.QLabel('显示黑电平'))
        self.display_black = QtWidgets.QDoubleSpinBox()
        self.display_black.setRange(0, 255); self.display_black.setValue(10); self.display_black.setSuffix(' ADU')
        row.addWidget(self.display_black)
        row.addWidget(QtWidgets.QLabel('γ'))
        self.display_gamma = QtWidgets.QDoubleSpinBox()
        self.display_gamma.setRange(.3, 3); self.display_gamma.setSingleStep(.1); self.display_gamma.setValue(.5)
        row.addWidget(self.display_gamma)
        reset = QtWidgets.QPushButton('重置显示')
        reset.clicked.connect(self.reset_display); row.addWidget(reset)
        row.addWidget(QtWidgets.QLabel('不改CCD增益／曝光／原图／积分／评分'), 1)
        self.layout().insertLayout(5, row)
        for widget in (self.display_gain, self.display_black, self.display_gamma):
            widget.valueChanged.connect(self.repaint_display)
        self.view_mode.currentIndexChanged.connect(self.repaint_display)
        self.ccd.setToolTip('显示增强只改变屏幕像素；数字读出始终来自未修改原图。')
        self.prediction.setToolTip('预测为背景扣除信号，使用与实测相同的显示增益／γ，不再各自归一化。')

    def reset_display(self):
        self.display_gain.setValue(16); self.display_black.setValue(10); self.display_gamma.setValue(.5)

    def load_bundle(self, path):
        super().load_bundle(path)
        self.last_result = None; self.last_folder = None
        self.readouts.clearContents()
        self.result_text.setText('新测试包未运行；显示增强不修改原始采集和评分。')
        if hasattr(self, 'view_mode'):
            self.view_mode.setCurrentIndex(0); self.repaint_display()
        self.update_buttons()

    def show_case(self, index):
        super().show_case(index)
        self.prediction_signal = self.bundle.cases[index]['predictions']['learned'].copy()
        if hasattr(self, 'view_mode'):
            if self.last_result and self.last_result['case_id'] != index:
                self.last_result = None; self.view_mode.setCurrentIndex(0)
                self.readouts.clearContents()
            self.repaint_display()

    def fresh_view(self, packet):
        if not self.bundle or time.monotonic()-self.last_view < .2:
            return
        self.last_view = time.monotonic()
        if packet['image'].shape != self.bundle.weights.shape[1:]:
            return
        self.raw_frame = dict(image=packet['image'].copy(), meta=dict(packet['meta']))
        self.repaint_display()

    def repaint_display(self, *unused):
        if not hasattr(self, 'display_gain') or not self.bundle:
            return
        gain, gamma = self.display_gain.value(), self.display_gamma.value()
        history = self.view_mode.currentIndex() == 1
        if history:
            if not self.last_result:
                self.ccd.setText('暂无历史学习图；可切换实时CCD。')
            else:
                values = self.last_result['observed_signal']
                shown = display_signal(values, gain, 0, gamma)
                self.ccd.set_image(detector_overlay(shown, self.bundle.weights), values.shape[::-1])
                self.ccd.parentWidget().setTitle('历史学习图 · 背景扣除均值（非实时）')
        elif self.raw_frame is not None:
            values = self.raw_frame['image']
            shown = display_signal(values, gain, self.display_black.value(), gamma)
            self.ccd.set_image(detector_overlay(shown, self.bundle.weights), values.shape[::-1])
            self.ccd.parentWidget().setTitle(f'实时CCD · RAW峰值{int(values.max())} ADU · 显示×{gain:g}')
        else:
            self.ccd.setText('等待实时CCD；显示增益不会自动启动相机或投图。')
        prediction = self.last_result['predicted_signal'] if history and self.last_result else self.prediction_signal
        if prediction is not None:
            self.prediction.set_image(display_signal(prediction, gain, 0, gamma), prediction.shape[::-1])
            self.prediction.parentWidget().setTitle('研究预测 · 同增益／γ（背景已扣除）')

    def set_busy(self, value, total_cases=1):
        super().set_busy(value, total_cases)
        if value:
            self.last_result = None
            self.view_mode.setCurrentIndex(0)
            self.repaint_display()

    def on_progress(self, data):
        super().on_progress(data)
        if data['kind'] == 'stage_start':
            self.last_stage = data['stage']
        if data['kind'] == 'case_complete':
            self.last_result = dict(case_id=data['case_id'], observed_signal=data['observed_signal'].copy(),
                                   predicted_signal=data['predicted_signal'].copy())
            self.prediction_signal = self.last_result['predicted_signal']
            self.repaint_display()

    def finished(self, report):
        super().finished(report)
        if self.last_result:
            self.view_mode.setCurrentIndex(1)
            self.repaint_display()
            self.status.setText(self.status.text()+' 当前显示历史学习图，非实时；可切换单帧实时CCD。')

    def restore_result(self, folder):
        """Read-only history restore; never treat a recorded image as live CCD."""
        if self.busy or not self.bundle:
            raise RuntimeError('load bundle while idle before restoring history')
        folder = Path(folder)
        plan = json.loads((folder/'PROSPECTIVE_PROTOCOL.json').read_text())
        report = json.loads((folder/'capture.json').read_text())
        if plan['catalog_sha256'] != self.bundle.catalog_sha or not report['completed'] or report['error']:
            raise ValueError('history is not a completed capture of this frozen bundle')
        case = report['cases'][-1]
        if not case['analysis']:
            raise ValueError('missing historical analysis')
        images = {}
        for name in ('background_start', 'learned_1'):
            stage = next(s for s in case['stages'] if s['name'] == name)
            if len(stage['frames']) != plan['frames_per_stage']:
                raise ValueError('incomplete historical stack')
            raw = []
            for i, record in enumerate(stage['frames'], 1):
                path = folder/f"input_{case['id']:02d}"/name/f'CCD_raw_{i:03d}.npy'
                if digest(path) != record['raw_sha256']:
                    raise ValueError('historical raw file changed')
                frame = np.load(path, allow_pickle=False)
                if frame.dtype != np.uint8 or frame.shape != self.bundle.weights.shape[1:] or frame.max() >= 240:
                    raise ValueError('historical raw layout/peak changed')
                raw.append(frame)
            images[name] = np.stack(raw).mean(0)
        gain = report['frozen_one_global_gain']
        self.on_progress(dict(kind='case_complete', case_id=case['id'], analysis=case['analysis'], gain=gain,
            observed_signal=images['learned_1']-images['background_start'],
            predicted_signal=self.bundle.cases[case['id']]['predictions']['learned']*gain))
        self.finished(report)
        self.confirm.setChecked(False); self.frozen = False; self.update_buttons()
        self.status.setText('已恢复历史测试图／评分（非实时）。未启动相机或投图；仅显示增强，不改原始数值。')
        self.status.setToolTip(str(folder.resolve()))
