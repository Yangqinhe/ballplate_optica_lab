"""Direct alignment UI: DMD/SLM, ROI and fixed-model CCD comparison only.

The proven v17 device/camera/display backend is reused read-only. No point-grid
or batch-probe workflow is exposed or executable from this window. Nothing
connects to real devices unless --hardware and a new light-blocked confirmation
are both supplied. Offline preview does not create output windows or files.
"""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent / 'live_manual_v17'
sys.path.insert(0, str(BACKEND))

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from realtime_gui import RealtimeWindow as BackendWindow, CCDView
from realtime_worker import write_json
from realtime_recovery import recovery_target
from stable_views import StatusLabel, SensorView, SharedSensorScale


class DisabledWorkflowStatus:
    """Compatibility for the backend heartbeat, not a hidden calibration UI."""
    def text(self):
        return '本版本不启用点阵、分层标定或自动扫描；只做手动实时对图。'


class DirectAlignmentWindow(BackendWindow):
    def __init__(self, output, hardware=False):
        super().__init__(output, hardware)
        if hardware:
            sources = {}
            for module in list(sys.modules.values()):
                name = getattr(module, '__file__', None)
                if not name:
                    continue
                path = Path(name).resolve()
                if path.suffix == '.py' and path.parent in (ROOT, BACKEND):
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(self.output / 'runtime_sources.json', dict(
                pid=os.getpid(), sources=sources,
                source_configs={self.orientation_hint['path']: self.orientation_hint['sha256'],
                                self.slm_coordinate_hint['path']: self.slm_coordinate_hint['sha256']},
                ui_version=18, workflow='direct_CCD_comparison', backend_version=17,
                no_auto_capture=True, no_auto_projection=True,
                single_output_and_sdk_owner=True, point_calibration_enabled=False,
                automatic_fine_scan=False, batch_probes_enabled=False))
            self.record(dict(state='direct_alignment_ui_initialized', ui_version=18,
                             point_calibration_enabled=False, automatic_fine_scan=False))

    def build_ui(self):
        old = self.takeCentralWidget()
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(12, 10, 12, 10)
        title = QtWidgets.QLabel('光路对齐 · 实测CCD / 仿真CCD / 差值')
        title.setStyleSheet('font-size:22px;font-weight:bold')
        layout.addWidget(title)

        bar = QtWidgets.QHBoxLayout()
        layout.addLayout(bar)
        self.armed = QtWidgets.QCheckBox('允许投图（ND／低功率条件已确认）')
        bar.addWidget(self.armed)
        self.start_button = QtWidgets.QPushButton('开始实时')
        self.start_button.clicked.connect(lambda: self.start_live(project=True))
        bar.addWidget(self.start_button)
        self.camera_only_button = QtWidgets.QPushButton('仅CCD')
        self.camera_only_button.clicked.connect(lambda: self.start_live(project=False))
        bar.addWidget(self.camera_only_button)
        self.stop_button = QtWidgets.QPushButton('停止／两路全0（Esc）')
        self.stop_button.setStyleSheet('color:#b02020;font-weight:bold')
        self.stop_button.clicked.connect(self.stop_live)
        bar.addWidget(self.stop_button)
        self.snap_button = QtWidgets.QPushButton('保存当前对照')
        self.snap_button.clicked.connect(self.snapshot)
        bar.addWidget(self.snap_button)
        bar.addStretch()
        self.armed.toggled.connect(self.arm_changed)
        self.output_state = StatusLabel('默认全0待机；未自动投图或采集。', 2)
        layout.addWidget(self.output_state)

        middle = QtWidgets.QHBoxLayout()
        layout.addLayout(middle, 1)
        left_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        left.setContentsMargins(4, 4, 4, 4)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_widget)
        scroll.setFixedWidth(410)
        middle.addWidget(scroll)

        box = QtWidgets.QGroupBox('DMD · 标定输入')
        body = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        body.addLayout(row)
        self.dmd_mode = QtWidgets.QComboBox()
        # Retain the backend's explicit black index and existing file contract.
        for text, key in [('非对称标定图', 'fiducial'), ('中心方块', 'square'),
                          ('标准白方块（ROI）', 'white'), ('全黑', 'black'), ('载入的二值图', 'file')]:
            self.dmd_mode.addItem(text, key)
        row.addWidget(self.dmd_mode, 1)
        self.dmd_load_button = QtWidgets.QPushButton('载入图像')
        self.dmd_load_button.clicked.connect(lambda: self.load_file(False))
        row.addWidget(self.dmd_load_button)
        row = QtWidgets.QHBoxLayout()
        body.addLayout(row)
        self.square = QtWidgets.QSpinBox()
        self.square.setRange(16, 600)
        self.square.setValue(128)
        row.addWidget(QtWidgets.QLabel('方块边长'))
        row.addWidget(self.square)
        self.black_button = QtWidgets.QPushButton('DMD全0，CCD继续')
        self.black_button.clicked.connect(self.dmd_black)
        row.addWidget(self.black_button)
        left.addWidget(box)

        box = QtWidgets.QGroupBox('SLM · mask与位置')
        body = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        body.addLayout(row)
        self.slm_mode = QtWidgets.QComboBox()
        for text, key in [('透镜', 'lens'), ('固定灰度0', 'zero'), ('光栅 +X', '+x'),
                          ('光栅 −X', '-x'), ('光栅 +Y', '+y'), ('光栅 −Y', '-y'), ('载入灰度图', 'file')]:
            self.slm_mode.addItem(text, key)
        row.addWidget(self.slm_mode, 1)
        self.slm_load_button = QtWidgets.QPushButton('载入mask')
        self.slm_load_button.clicked.connect(lambda: self.load_file(True))
        row.addWidget(self.slm_load_button)
        self.full = QtWidgets.QCheckBox('整屏SLM')
        self.full.setChecked(True)
        body.addWidget(self.full)
        self.spins, self.sliders = {}, {}
        form = QtWidgets.QGridLayout()
        body.addLayout(form)
        for i, (key, label, low, high) in enumerate((('x', '中心X', -960, 960), ('y', '中心Y', -540, 540))):
            spin = QtWidgets.QSpinBox()
            spin.setRange(low, high)
            spin.setKeyboardTracking(False)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(low, high)
            slider.setPageStep(16)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(self.changed)
            self.spins[key], self.sliders[key] = spin, slider
            form.addWidget(QtWidgets.QLabel(label), i, 0)
            form.addWidget(slider, i, 1)
            form.addWidget(spin, i, 2)
        row = QtWidgets.QHBoxLayout()
        body.addLayout(row)
        self.focal = QtWidgets.QDoubleSpinBox()
        self.focal.setRange(50, 3000)
        self.focal.setValue(400)
        self.focal.setSuffix(' mm')
        self.focal.setKeyboardTracking(False)
        self.period = QtWidgets.QSpinBox()
        self.period.setRange(4, 512)
        self.period.setValue(32)
        self.period.setKeyboardTracking(False)
        row.addWidget(QtWidgets.QLabel('透镜f'))
        row.addWidget(self.focal)
        row.addWidget(QtWidgets.QLabel('光栅周期'))
        row.addWidget(self.period)
        self.phase_caption = StatusLabel('', 2)
        body.addWidget(self.phase_caption)
        left.addWidget(box)

        box = QtWidgets.QGroupBox('ROI与曝光')
        body = QtWidgets.QVBoxLayout(box)
        instruction = QtWidgets.QLabel('在实测CCD上拖框，再点“应用框选”；也可用标准白方块定位。')
        instruction.setWordWrap(True)
        body.addWidget(instruction)
        row = QtWidgets.QHBoxLayout()
        body.addLayout(row)
        self.auto_button = QtWidgets.QPushButton('白方块定位')
        self.auto_button.clicked.connect(self.auto_roi)
        row.addWidget(self.auto_button)
        self.apply_roi_button = QtWidgets.QPushButton('应用框选')
        self.apply_roi_button.clicked.connect(self.apply_roi)
        row.addWidget(self.apply_roi_button)
        self.full_button = QtWidgets.QPushButton('恢复全幅')
        self.full_button.clicked.connect(self.full_sensor)
        row.addWidget(self.full_button)
        self.roi_text = StatusLabel('当前ROI：等待CCD实际读回。', 2)
        body.addWidget(self.roi_text)
        row = QtWidgets.QHBoxLayout()
        body.addLayout(row)
        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setRange(50, 4000)
        self.exposure.setValue(1000)
        self.exposure.setSuffix(' µs')
        self.exposure.setKeyboardTracking(False)
        row.addWidget(QtWidgets.QLabel('曝光'))
        row.addWidget(self.exposure)
        self.exposure_button = QtWidgets.QPushButton('应用曝光')
        self.exposure_button.clicked.connect(self.apply_exposure)
        row.addWidget(self.exposure_button)
        left.addWidget(box)

        box = QtWidgets.QGroupBox('请求预览 · DMD / SLM')
        body = QtWidgets.QHBoxLayout(box)
        self.dmd_preview, self.slm_preview = QtWidgets.QLabel(), QtWidgets.QLabel()
        for widget in (self.dmd_preview, self.slm_preview):
            widget.setFixedSize(176, 112)
            widget.setAlignment(QtCore.Qt.AlignCenter)
            widget.setStyleSheet('background:#111')
            body.addWidget(widget)
        left.addWidget(box)

        self.build_comparison_controls(left)
        left.addStretch()
        right = QtWidgets.QVBoxLayout()
        middle.addLayout(right, 1)
        bar = QtWidgets.QHBoxLayout()
        self.target_mode = QtWidgets.QComboBox()
        self.target_mode.addItem('对齐：仿真中心固定，手动调SLM的X/Y', True)
        self.target_mode.addItem('预测：仿真跟随当前SLM的X/Y', False)
        self.target_mode.currentIndexChanged.connect(self.comparison_changed)
        bar.addWidget(self.target_mode, 2)
        self.diff_mode = QtWidgets.QComboBox()
        self.diff_mode.addItem('叠加：红=实测，绿=仿真，黄=重合', True)
        self.diff_mode.addItem('差值：红=实测偏强，蓝=仿真偏强', False)
        self.diff_mode.currentIndexChanged.connect(self.repaint_comparison)
        bar.addWidget(self.diff_mode, 2)
        self.enhance = QtWidgets.QCheckBox('增强实测预览')
        self.enhance.setChecked(True)
        bar.addWidget(self.enhance)
        right.addLayout(bar)
        bar = QtWidgets.QHBoxLayout()
        self.zero_button = QtWidgets.QPushButton('采参考并锁定倍率／位置')
        self.zero_button.clicked.connect(self.capture_zero_reference)
        self.zero_button.setToolTip('一次有界暗场/固定灰度参考配准，之后不随XY自动重新拟合；不是相位LUT标定。')
        bar.addWidget(self.zero_button)
        self.mapping_reset_button = QtWidgets.QPushButton('解除锁定')
        self.mapping_reset_button.clicked.connect(self.clear_mapping)
        bar.addWidget(self.mapping_reset_button)
        self.mapping_status = StatusLabel('未锁定：先选ROI，再采一次参考；随后只调SLM。', 2)
        bar.addWidget(self.mapping_status, 1)
        right.addLayout(bar)
        self.view = CCDView()
        self.view.selected.connect(self.select_roi)
        self.sim_view, self.diff_view = SensorView('等待仿真'), SensorView('等待比较')
        compare = QtWidgets.QHBoxLayout()
        right.addLayout(compare, 1)
        for title, view in (('实测CCD · 可拖框选择ROI', self.view),
                            ('仿真CCD · 同ROI同倍率', self.sim_view),
                            ('叠加／差值 · 同一ROI', self.diff_view)):
            box = QtWidgets.QGroupBox(title)
            body = QtWidgets.QVBoxLayout(box)
            body.addWidget(view)
            compare.addWidget(box, 1)
        self.shared_sensor_scale = SharedSensorScale((self.view, self.sim_view, self.diff_view), self)
        self.telemetry = StatusLabel('CCD未启动。', 2)
        right.addWidget(self.telemetry)
        self.comparison_status = StatusLabel('λ532nm · Len2/Len3 150/100mm · 未标定LUT；差值仅比较归一化形状。', 2)
        right.addWidget(self.comparison_status)
        self.state = StatusLabel('流程：选输入和mask → 开始实时 → ROI → 采参考锁定 → 手动调XY。', 2)
        layout.addWidget(self.state)
        self.log_toggle = QtWidgets.QToolButton()
        self.log_toggle.setText('运行日志')
        self.log_toggle.setCheckable(True)
        bottom = QtWidgets.QHBoxLayout()
        layout.addLayout(bottom)
        bottom.addWidget(self.log_toggle)
        self.recheck = QtWidgets.QPushButton('故障只读复核（不投图）')
        self.recheck.clicked.connect(self.start_recheck)
        bottom.addWidget(self.recheck)
        bottom.addStretch()
        button = QtWidgets.QPushButton('遮光后退出' if self.hardware else '关闭预览')
        button.clicked.connect(self.close)
        bottom.addWidget(button)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(65)
        self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible)
        layout.addWidget(self.log)

        self.spatial_status = DisabledWorkflowStatus()
        self.shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Escape'), self)
        self.shortcut.activated.connect(self.stop_live)
        for control in (self.dmd_mode, self.slm_mode):
            control.currentIndexChanged.connect(self.changed)
        for control in (self.square, self.focal, self.period):
            control.valueChanged.connect(self.changed)
        self.full.toggled.connect(self.scope_changed)
        self.setWindowTitle('实时光路对齐 v18 · 直接对图' + (' · 真实器件' if self.hardware else ' · 离线预览'))
        self.setMinimumSize(1400, 850)
        primary = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.setGeometry(primary.x() + 40, primary.y() + 40, 1840, 1020)
        if old:
            old.deleteLater()

    def build_comparison_controls(self, left):
        """Secondary model/direction knobs live in a collapsed section."""
        toggle = QtWidgets.QToolButton()
        toggle.setText('仿真参数与方向 ▸')
        toggle.setCheckable(True)
        left.addWidget(toggle)
        advanced = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(advanced)
        advanced.hide()
        toggle.toggled.connect(advanced.setVisible)
        left.addWidget(advanced)
        self.advanced_toggle, self.advanced_panel = toggle, advanced
        self.optics = {}
        for key, label, low, high, value, unit, decimals in (
            ('wavelength', '波长', 350, 1100, 532, ' nm', 1),
            ('pitch', '参考共同网格', 2, 30, 8, ' µm', 2),
            ('f1', 'Len2焦距', 50, 1000, 150, ' mm', 1),
            ('f2', 'Len3焦距', 50, 1000, 100, ' mm', 1),
            ('phase_depth', '假定灰度相位深度', .1, 12.566, 2 * np.pi, ' rad', 4)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.setSuffix(unit)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.optics_changed)
            self.optics[key] = spin
            form.addRow(label, spin)
        self.slm_flip_y = QtWidgets.QCheckBox('SLM相位Y反射（仅模型）')
        self.slm_flip_y.setChecked(self.slm_coordinate_hint['slm_to_model_flip_y'])
        self.slm_flip_y.toggled.connect(self.optics_changed)
        form.addRow(self.slm_flip_y)
        self.orientation = QtWidgets.QComboBox()
        from display_orientation import orientation_name
        for code in range(8):
            self.orientation.addItem(orientation_name(code), code)
        self.orientation.setCurrentIndex(self.orientation_hint['simulation_to_native_CCD_orientation'])
        self.orientation.currentIndexChanged.connect(self.comparison_changed)
        form.addRow('仿真方向', self.orientation)
        direction = QtWidgets.QHBoxLayout()
        self.direction_buttons = []
        for text, action in (('左右镜像', 'horizontal'), ('上下镜像', 'vertical'), ('旋转90°', 'rotate_ccw')):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, a=action: self.edit_preview_orientation(a))
            self.direction_buttons.append(button)
            direction.addWidget(button)
        form.addRow(direction)
        self.reset_direction = QtWidgets.QPushButton('恢复平台方向')
        self.reset_direction.clicked.connect(self.restore_platform_orientation)
        self.direction_buttons.append(self.reset_direction)
        form.addRow(self.reset_direction)
        self.direction_status = StatusLabel('', 2)
        form.addRow(self.direction_status)
        self.map_scale = QtWidgets.QDoubleSpinBox()
        self.map_scale.setRange(.1, 5)
        self.map_scale.setDecimals(3)
        self.map_scale.setValue(1)
        self.map_scale.setSingleStep(.02)
        self.map_x, self.map_y = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for control in (self.map_x, self.map_y):
            control.setRange(-3000, 3000)
            control.setDecimals(1)
            control.setSuffix(' CCD px')
            control.setSingleStep(2)
        for label, control in (('手动显示缩放', self.map_scale), ('手动显示X', self.map_x), ('手动显示Y', self.map_y)):
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self.comparison_changed)
            form.addRow(label, control)
        note = QtWidgets.QLabel('只改仿真，不动实物。8µm共同网格和灰度相位深度均为近似，未加载实测LUT。手动显示偏移须保留记录，不能当成SLM已对齐。')
        note.setWordWrap(True)
        form.addRow(note)

    def refresh(self):
        if not hasattr(self, 'start_button'):
            return
        active = bool(self.worker and self.worker.isRunning())
        self.start_button.setEnabled(self.hardware and not active and not self.fault)
        self.camera_only_button.setEnabled(self.hardware and not active and not self.fault)
        fresh = bool(self.running and self.last_packet and time.monotonic() - self.last_packet['received_monotonic'] <= 1.5)
        self.snap_button.setEnabled(fresh)
        available = bool(self.can_project() and not self.override and not self.roi_busy and not self.reference_busy)
        self.auto_button.setEnabled(available)
        self.zero_button.setEnabled(available)
        configurable = bool(self.running and self.live_ready and not self.roi_busy and not self.reference_busy)
        self.apply_roi_button.setEnabled(bool(configurable and self.candidate))
        self.full_button.setEnabled(configurable)
        self.exposure_button.setEnabled(configurable)
        self.mapping_reset_button.setEnabled(not self.reference_busy)
        target = recovery_target(self.fault, self.fault_before, self.last_camera_report)
        self.recheck.setVisible(bool(self.fault))
        self.recheck.setEnabled(bool(self.hardware and self.fault and target and not active and not self.recheck_context))
        self.armed.setEnabled(self.hardware and not self.fault)
        self.update_direction_status()
        for control in (self.dmd_mode, self.slm_mode, self.full, self.square, self.focal, self.period,
                        *self.spins.values(), *self.sliders.values(), *self.optics.values(),
                        self.slm_flip_y, self.exposure, self.dmd_load_button, self.slm_load_button,
                        self.target_mode, self.map_scale, self.map_x, self.map_y):
            control.setEnabled(not self.reference_busy)

    def start_spatial(self, mode):
        self.note('直接对图版不运行点阵或自动扫描；请使用ROI和一次参考配准。')

    def spatial_received(self, report):
        # There is no UI route to request these from the shared worker.
        self.stop_live()
        self.note('收到本版本未请求的点阵结果，已停止；不采用结果。')


def main():
    import cv2
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--light-blocked-confirmed', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--preview-screenshot', type=Path,
                        help='Save an OFFLINE-only UI preview; never use with --hardware')
    args = parser.parse_args()
    if args.hardware and (not args.light_blocked_confirmed or args.smoke_test or args.preview_screenshot):
        parser.error('真实器件启动须新确认物理遮光；测试/截图参数不能连接硬件')
    if args.preview_screenshot and args.preview_screenshot.exists():
        parser.error('不覆盖已有预览图')
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if args.hardware and app.platformName() != 'xcb':
        raise RuntimeError('real local X11 required')
    output = args.output or ROOT.parent / 'outputs/live_calibration' / ('DIRECT_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    window = DirectAlignmentWindow(output, args.hardware)
    if args.smoke_test or args.preview_screenshot:
        assert not window.windows and not output.exists()
        window.start_live()
        assert window.worker is None
        if args.preview_screenshot:
            def capture():
                args.preview_screenshot.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(args.preview_screenshot)):
                    raise RuntimeError('offline preview save failed')
                app.quit()
            QtCore.QTimer.singleShot(250, capture)
        else:
            QtCore.QTimer.singleShot(250, app.quit)
    app.exec_()


if __name__ == '__main__':
    main()
