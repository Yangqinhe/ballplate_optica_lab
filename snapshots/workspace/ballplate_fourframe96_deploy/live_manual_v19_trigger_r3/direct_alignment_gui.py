"""Same workbench, bounded numeric threads and explicit fault recovery.

No automatic camera/projection. A carried fault must pass the unchanged
read-only two-connection original-baseline recheck, never merely reboot away.
"""
import os
# Must precede numpy/scipy/Qt imports and apply to all worker-created pools.
THREAD_ENV = {'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
              'NUMEXPR_NUM_THREADS': '2', 'OMP_WAIT_POLICY': 'PASSIVE'}
os.environ.update(THREAD_ENV)
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from live_manual_v19_trigger_r2.direct_alignment_gui import OrderedTriggerWindow
from live_manual_v19_trigger_r3.state import validate
from realtime_worker import write_json
from stable_views import StatusLabel
from display_linux import same_signals
from PyQt5 import QtCore, QtWidgets


class RecoverableTriggerWindow(OrderedTriggerWindow):
    def __init__(self, output, hardware=False, restart_state=None):
        super().__init__(output, hardware, legacy_reference=None)
        self.setWindowTitle('实时光路对齐 v19 · Line0硬触发 r3' + (' · 真实器件' if hardware else ' · 离线预览'))
        if restart_state is not None: self.restore_controls(restart_state)
        if hardware:
            path = self.output / 'runtime_sources.json'; runtime = json.loads(path.read_text())
            runtime['sources'].update({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
            runtime.update(trigger_revision=3, numeric_thread_env=THREAD_ENV, recovery_visible=True,
                           allzero_does_not_cancel_readonly_recheck=True, shortcut_auto_repeat=False)
            write_json(path, runtime)
        self.refresh()

    def build_ui(self):
        super().build_ui()
        self.shortcut.setAutoRepeat(False)
        self.stop_button.setAutoRepeat(False)
        row = QtWidgets.QHBoxLayout()
        self.recovery_status = StatusLabel('', 2)
        row.addWidget(self.recovery_status, 1)
        # Move the existing strict recheck to the top, not a weaker reset button.
        self.centralWidget().layout().itemAt(self.centralWidget().layout().count()-2).layout().removeWidget(self.recheck)
        self.recheck.setText('复核并恢复（不投图）')
        self.recheck.setToolTip('只读比较两次相机连接与原故障前参数；不采图、不写参数，合格才解锁。')
        row.addWidget(self.recheck)
        self.cancel_recheck = QtWidgets.QPushButton('取消只读复核')
        self.cancel_recheck.clicked.connect(self.cancel_recovery)
        row.addWidget(self.cancel_recheck)
        self.centralWidget().layout().insertLayout(3, row)

    def restore_controls(self, snapshot):
        state = validate(snapshot)
        if self.running or self.worker is not None or self.armed.isChecked(): raise RuntimeError('startup restore only')
        if self.hardware: same_signals(state['display_roles'], self.baseline)
        s = state['settings']; cfg = state['optical_config']
        widgets = [self.dmd_mode, self.slm_mode, self.full, self.square, self.focal, self.period, self.exposure,
                   self.orientation, self.slm_flip_y, *self.spins.values(), *self.sliders.values(), *self.optics.values()]
        for widget in widgets: widget.blockSignals(True)
        try:
            for combo, key in ((self.dmd_mode, 'dmd'), (self.slm_mode, 'slm')):
                index = combo.findData(s[key])
                if index < 0: raise ValueError('unsupported restored mode')
                combo.setCurrentIndex(index)
            self.full.setChecked(s['full_panel']); self.square.setValue(s['square'])
            self.focal.setValue(s['focal_mm']); self.period.setValue(s['period'])
            for key in ('x', 'y'):
                self.spins[key].setValue(s[key]); self.sliders[key].setValue(s[key])
            self.exposure.setValue(state['requested_exposure_us'])
            for key, value in dict(wavelength=cfg['wavelength_m']*1e9, pitch=cfg['pitch_m']*1e6,
                f1=cfg['f1_m']*1000, f2=cfg['f2_m']*1000, phase_depth=cfg['phase_depth_rad']).items():
                self.optics[key].setValue(value)
            self.slm_flip_y.setChecked(cfg['slm_flip_y'])
            self.orientation.setCurrentIndex(state['original_heartbeat']['simulation_display_orientation']['code'])
        finally:
            for widget in widgets: widget.blockSignals(False)
        self.actual_roi = dict(state['roi'])
        self.registration = self.pending_registration = self.starting_registration = None
        self.fault = state['fault']; self.fault_before = dict(state['original_fault_before'])
        self.last_camera_report = dict(state['original_last_camera_report'])
        self.roi_text.setText('保留ROI：' + str(self.actual_roi) + '；尚未采集')
        self.mapping_status.setText('保留手调参数；旧倍率未自动复用。模型参数改变后需重新采参考。')
        self.render_patterns(); self.black_outputs(); self.refresh()
        if self.hardware: self.record(dict(state='fault_restart_controls_retained', snapshot=state,
            fault_cleared=False, geometry_restored=False, no_auto_capture=True, no_auto_projection=True))

    def stop_live(self):
        if self.recheck_context and self.worker and self.worker.isRunning():
            # This worker is READ ONLY: there is no acquisition to stop.
            self.armed.setChecked(False); self.black_outputs()
            self.note('两路保持全0；只读复核继续。如需中断，请点“取消只读复核”。')
            self.refresh(); return
        super().stop_live()

    def cancel_recovery(self):
        if self.recheck_context and self.worker and self.worker.isRunning():
            self.worker.stop(); self.black_outputs()
            self.note('已请求取消只读复核；保持故障锁和两路全0。')

    def refresh(self):
        super().refresh()
        if not hasattr(self, 'recovery_status'): return
        checking = bool(self.recheck_context)
        self.recheck.setVisible(True)
        self.cancel_recheck.setEnabled(checking)
        self.cancel_recheck.setVisible(checking)
        if checking:
            text = '正在只读复核原相机参数（不采图、不投图）；全0键不会取消复核。'
            color = '#9a6500'
        elif self.fault:
            text = '保护停采，不是实时画面：' + self.fault + '。检查光强/曝光后，点右侧只读复核恢复。'
            color = '#b02020'
        elif self.running:
            text = 'CCD运行中；饱和、坏帧或缺触发仍会保护停采。'; color = '#17603b'
        else:
            text = '待机：CCD未启动，两路全0。先点“仅CCD（硬触发）”；不自动投图。'; color = '#465366'
        self.recovery_status.setText(text)
        self.recovery_status.setStyleSheet('color:' + color + ';font-weight:bold')
        if self.fault:
            self.telemetry.setText('CCD已停采；最后图像仅供排错，不是实时。故障和复核记录已保存。')

    def closeEvent(self, event):
        if self.recheck_context and self.worker and self.worker.isRunning():
            self.cancel_recovery(); event.ignore(); return
        super().closeEvent(event)


def main():
    import cv2
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware', action='store_true'); p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--output', type=Path); p.add_argument('--restart-state', type=Path); p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test): p.error('真实启动须当次物理遮光确认')
    snapshot = None if a.restart_state is None else validate(json.loads(a.restart_state.read_text()))
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName() != 'xcb': raise RuntimeError('真实输出须本地X11')
    output = a.output or PROJECT / 'outputs/live_calibration' / ('LINE0_R3_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    w = RecoverableTriggerWindow(output, a.hardware, snapshot)
    if a.smoke_test:
        assert not w.windows and w.worker is None and not output.exists()
        QtCore.QTimer.singleShot(250, app.quit)
    app.exec_()


if __name__ == '__main__': main()
