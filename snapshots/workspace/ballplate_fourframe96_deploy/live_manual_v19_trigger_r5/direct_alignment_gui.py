"""Recoverable one-window workbench with direct Line0 startup.

No automatic capture/projection. Existing strict recheck and all fatal gates
remain. Does not qualify SLM synchronization or policy deployment.
"""
import os
THREAD_ENV = {'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
              'NUMEXPR_NUM_THREADS': '2', 'OMP_WAIT_POLICY': 'PASSIVE'}
os.environ.update(THREAD_ENV)
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from live_manual_v19_trigger_r3.direct_alignment_gui import RecoverableTriggerWindow
from live_manual_v19_trigger_r3.state import validate
from live_manual_v19_trigger_r5.worker import ManualPreviewWorker
from realtime_worker import write_json
from PyQt5 import QtCore, QtWidgets


class ManualPreviewWindow(RecoverableTriggerWindow):
    def __init__(self, output, hardware=False, restart_state=None, recovery_record=None):
        self.preview_background_warning=None
        self._fault_view_shown=None
        self._extra_heartbeat=None
        if recovery_record is not None:
            from live_manual_v19_trigger_r5.recovery_record import validate_recovery
            recovery_record=validate_recovery(recovery_record)
        super().__init__(output, hardware, restart_state)
        self.setWindowTitle('实时光路对齐 v19 · 手动预览 r5' + (' · 真实器件' if hardware else ' · 离线预览'))
        if hardware:
            path = self.output/'runtime_sources.json'; runtime = json.loads(path.read_text())
            runtime['sources'].update({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
            runtime.update(trigger_revision=5, startup_sequence='direct_Line0_no_Software_source',
                no_startup_Software_source=True, no_free_run_fallback=True, no_auto_retry=True,
                cold_start_reconnect_qualified=False)
            write_json(path, runtime)
        if recovery_record is not None:
            marker=recovery_record
            self.fault='手动终止旧进程后必须复核原参数：'+(marker.get('original_fault') or '未确认旧相机已回退')
            self.fault_before=dict(marker['camera_baseline'])
        self.refresh()

    def build_ui(self):
        super().build_ui()
        self.ui_clock=QtWidgets.QLabel('界面心跳 --:--:--')
        self.centralWidget().layout().itemAt(1).layout().addWidget(self.ui_clock)

    def camera_ready(self,info):
        rows=info.get('zero_input_baselines',[])
        self.preview_background_warning=max((r['mean_ADU'] for r in rows if r['high_background_warning']),default=None)
        super().camera_ready(info)
        self.refresh()

    def camera_finished(self,report):
        super().camera_finished(report)
        self.refresh()

    def refresh(self):
        super().refresh()
        if not hasattr(self,'view'):return
        if self.fault and self._fault_view_shown!=self.fault:
            self.view.setText('CCD已停止（不是实时画面）\\n'+self.fault+'\\n检查曝光/光强后，点顶部“复核并恢复（不投图）”。')
            self._fault_view_shown=self.fault
        elif not self.fault:
            self._fault_view_shown=None
        if self.running and self.preview_background_warning is not None and not self.fault:
            self.recovery_status.setText('CCD实时预览中；零输入背景偏亮（均值 %.1f）。未通过暗场标定，先减小环境光/曝光；饱和保护仍启用。'%self.preview_background_warning)
            self.recovery_status.setStyleSheet('color:#9a6500;font-weight:bold')
            self.zero_button.setEnabled(False)
            self.zero_button.setToolTip('启动零输入背景偏亮：只允许手动观察，不把它当作暗场/定量标定通过。')

    def tick(self):
        super().tick()
        if self.last_packet and 'zero_input_background_warning_mean_ADU' in self.last_packet:
            warning=self.last_packet['zero_input_background_warning_mean_ADU']
            if warning!=self.preview_background_warning:
                self.preview_background_warning=warning;self.refresh()
        if hasattr(self,'ui_clock'):
            self.ui_clock.setText('界面心跳 '+time.strftime('%H:%M:%S'))
        if self.hardware and self._extra_heartbeat!=self.last_heartbeat:
            self._extra_heartbeat=self.last_heartbeat
            path=self.output/'heartbeat.json'
            if path.exists():
                row=json.loads(path.read_text())
                row.update(trigger_revision=5,startup_profile='manual_preview_zero_input_not_physical_dark',
                    preview_background_warning_mean_ADU=self.preview_background_warning,
                    quantitative_dark_qualified=False)
                tmp=self.output/'heartbeat_preview.tmp';write_json(tmp,row);tmp.replace(path)

    def start_live(self, project=True):
        if not self.hardware or self.fault or (self.worker and self.worker.isRunning()):
            return
        if project and not self.armed.isChecked():
            answer = QtWidgets.QMessageBox.question(self, '启动三路实时',
                '确认仍为ND／低功率、工作光束和OFF光均有防护？\n先检验硬触发，再允许所选图案。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
            self.armed.setChecked(True)
        if not project:
            self.armed.setChecked(False)
        self.preview_background_warning=None
        self.starting_registration = self.registration
        self.record(dict(state='manual_line0_r5_start', projection_requested=bool(project),
            requested_roi=dict(self.actual_roi), requested_exposure_us=self.exposure.value()))
        self.black_outputs(); self.running = True; self.live_ready = False
        self.override = False; self.roi_busy = False; self.dirty = True
        self.last_sequence = 0; self.last_packet = None
        self.view.setText('正在检查Line0与3张零输入帧（不是物理遮光暗场），未投图…')
        folder = self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker = ManualPreviewWorker(self.baseline, folder, self.exposure.value(), roi=dict(self.actual_roi))
        self.history.append(self.worker)
        for signal, callback in (('ready', self.camera_ready), ('configured', self.camera_configured),
                ('roi_found', self.roi_found), ('zero_reference', self.zero_reference_captured),
                ('spatial_result', self.spatial_received), ('display', self.display_request),
                ('message', self.note), ('black', self.black_outputs), ('result', self.camera_finished),
                ('finished', self.refresh)):
            getattr(self.worker, signal).connect(callback)
        self.worker.start(); self.refresh()


def main():
    import cv2
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware', action='store_true'); p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--recovery-record', type=Path)
    p.add_argument('--output', type=Path); p.add_argument('--restart-state', type=Path); p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test): p.error('真实启动须当次物理遮光确认')
    snapshot = None if a.restart_state is None else validate(json.loads(a.restart_state.read_text()))
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName() != 'xcb': raise RuntimeError('真实输出须本地X11')
    output = a.output or PROJECT / 'outputs/live_calibration' / ('LINE0_R5_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    w = ManualPreviewWindow(output, a.hardware, snapshot,
        None if a.recovery_record is None else json.loads(a.recovery_record.read_text()))
    if a.smoke_test:
        assert not w.windows and w.worker is None and not output.exists()
        QtCore.QTimer.singleShot(250, app.quit)
    app.exec_()


if __name__ == '__main__': main()
