"""Same direct-comparison workbench with explicit Line0 camera operation.

No automatic capture/projection, no existing process injection, no old
free-run geometry re-signing. Creating real output windows requires NEW
physical light-blocked confirmation. Default invocation is offline preview.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from live_manual_v18_restart.direct_alignment_gui import RestartSafeWindow
from live_manual_v18_restart.resume_state import validate as validate_legacy_reference
from live_manual_v19_trigger.worker import HardwareCameraWorker
from realtime_worker import write_json
from diffraction_model import reference_signature
from PyQt5 import QtCore, QtWidgets


class HardwareTriggerWindow(RestartSafeWindow):
    def __init__(self, output, hardware=False, legacy_reference=None):
        self.last_trigger_heartbeat = None
        self.legacy_reference_evidence = None
        super().__init__(output, hardware, resume=None)
        if legacy_reference is not None:
            # Validated historical evidence supplies only controls/ROI. Never
            # copy its matrix under a manufactured new TriggerMode signature.
            contract = validate_legacy_reference(legacy_reference)
            self.restore_pending(contract)
            self.legacy_reference_evidence = contract
            self.pending_registration = None
            self.starting_registration = None
            self.registration = None
            self.mapping_status.setText('旧参考已保留；触发条件已改，需重新采零灰度参考，未复用旧倍率。')
            self.slm_mode.setCurrentIndex(self.slm_mode.findData('zero'))
            self.render_patterns(); self.black_outputs()
        if hardware:
            path = self.output/'runtime_sources.json'
            runtime = json.loads(path.read_text())
            for module in list(sys.modules.values()):
                filename = getattr(module, '__file__', None)
                if filename and Path(filename).suffix == '.py':
                    path_source = Path(filename).resolve()
                    if path_source.parent in (ROOT, PROJECT/'tools'):
                        runtime['sources'][str(path_source)] = hashlib.sha256(path_source.read_bytes()).hexdigest()
            runtime.update(ui_version=19, workflow='direct_CCD_comparison_Line0',
                trigger_revision=1, camera_backend='explicit Line0/FallingEdge/Rolling',
                stop_output_keeps_camera_connected=True, no_auto_reconnect=True,
                cold_start_reconnect_qualified=False, SLM_hardware_synchronized=False)
            write_json(path, runtime)
            self.record(dict(state='line0_ui_initialized', no_auto_capture=True, no_auto_projection=True,
                legacy_geometry_reused=False, legacy_reference=legacy_reference,
                no_auto_reconnect=True, no_free_run_fallback=True))
        self.refresh()

    def build_ui(self):
        super().build_ui()
        self.setWindowTitle('实时光路对齐 v19 · Line0硬触发' + (' · 真实器件' if self.hardware else ' · 离线预览'))
        self.camera_only_button.setText('仅CCD（硬触发）')
        self.stop_button.setText('两路全0，CCD保持（Esc）')
        self.stop_button.setToolTip('日常停投图不关闭CCD，不重新初始化SDK；缺脉冲／异常仍会停止采集并恢复参数。')
        self.release_button = QtWidgets.QPushButton('停采并恢复原参数')
        self.release_button.setToolTip('释放相机；以后再开始属于一次新的连接，重连可靠性尚未验收。')
        self.release_button.clicked.connect(self.release_camera)
        bar = self.centralWidget().layout().itemAt(1).layout()
        bar.insertWidget(bar.count()-1, self.release_button)
        self.state.setText('Line0下降沿硬触发；先仅CCD验收。日常全0保持连接，停采按钮才释放相机。')

    def start_live(self, project=True):
        if not self.hardware or self.fault or (self.worker and self.worker.isRunning()):
            return
        if project and not self.armed.isChecked():
            answer = QtWidgets.QMessageBox.question(self, '启动三路实时',
                '确认仍为ND／低功率，工作光束和OFF光均有防护？\n'
                '将先检验3张硬触发黑场，再允许所选DMD/SLM图案。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
            self.armed.setChecked(True)
        if not project:
            self.armed.setChecked(False)
        self.starting_registration = self.registration
        self.record(dict(state='manual_line0_start', projection_requested=bool(project),
            requested_roi=dict(self.actual_roi), requested_exposure_us=self.exposure.value(),
            single_connection_until_explicit_release=True))
        self.black_outputs(); self.running = True; self.live_ready = False
        self.override = False; self.roi_busy = False; self.dirty = True
        self.last_sequence = 0; self.last_packet = None
        self.view.setText('正在打开唯一相机，验证Line0触发及3张黑场…')
        folder = self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker = HardwareCameraWorker(self.baseline, folder, self.exposure.value(), roi=dict(self.actual_roi))
        self.history.append(self.worker)
        for signal, callback in (('ready', self.camera_ready), ('configured', self.camera_configured),
                ('roi_found', self.roi_found), ('zero_reference', self.zero_reference_captured),
                ('spatial_result', self.spatial_received), ('display', self.display_request),
                ('message', self.note), ('black', self.black_outputs), ('result', self.camera_finished),
                ('finished', self.refresh)):
            getattr(self.worker, signal).connect(callback)
        self.worker.start(); self.refresh()

    def stop_live(self):
        if (self.worker and self.worker.isRunning() and self.running and self.live_ready
                and not self.fault and not self.roi_busy and not self.reference_busy):
            self.armed.setChecked(False)
            self.black_outputs(); self.dirty = True
            self.note('DMD/SLM已全0；CCD仍硬触发采集、连接保持。需释放时点“停采并恢复原参数”。')
            self.refresh()
        else:
            # Startup cancellation, calibration cancellation and faults must
            # still terminate acquisition; this is NOT a blanket keep-alive.
            self.release_camera()

    def release_camera(self):
        super().stop_live()

    def camera_finished(self, report):
        super().camera_finished(report)
        if report.get('error'):
            self.fault = self.fault or '硬触发故障：'+report['error']
            self.fault_before = report.get('camera_before')
            self.note(self.fault+'；已停止，不自动重连／切自由采集。')
            self.refresh()

    def refresh(self):
        super().refresh()
        if hasattr(self, 'release_button'):
            self.release_button.setEnabled(bool(self.worker and self.worker.isRunning()))

    def tick(self):
        super().tick()
        if self.last_packet:
            text = self.telemetry.text().replace('自由运行，非硬同步。',
                'Line0下降沿硬触发；SLM未硬同步。')
            self.telemetry.setText(text)
            self.comparison_status.setText(self.comparison_status.text().replace('自由运行未硬同步。',
                '相机Line0硬触发；SLM未硬同步，图案逐帧关联未验收。'))
        if self.hardware and self.last_heartbeat != self.last_trigger_heartbeat:
            self.last_trigger_heartbeat = self.last_heartbeat
            path = self.output/'heartbeat.json'
            if path.exists():
                row = json.loads(path.read_text())
                row.update(ui_version=19, mode='Line0_manual_continuous',
                    camera_reserved=bool(self.worker and self.worker.isRunning()),
                    trigger_contract=dict(mode='On', source='Line0', selector='FrameStart',
                        activation='FallingEdge', shutter='Rolling', desired_contract_NOT_live_readback=True),
                    camera_active_readback=None if not self.last_packet else self.last_packet['active'],
                    old_free_run_reference_reused=False, SLM_hardware_synchronized=False,
                    hardware_20ms_accepted=False)
                tmp = self.output/'heartbeat_trigger.tmp'
                write_json(tmp, row); tmp.replace(path)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.release_camera(); event.ignore(); return
        super().closeEvent(event)


def main():
    import cv2
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--legacy-reference', type=Path)
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    if args.hardware and (not args.light_blocked_confirmed or args.smoke_test):
        p.error('真实器件窗口需要当次物理遮光确认；离线测试不能连接硬件')
    legacy = None if args.legacy_reference is None else validate_legacy_reference(json.loads(args.legacy_reference.read_text()))
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if args.hardware and app.platformName() != 'xcb':
        raise RuntimeError('真实输出必须在本地X11，不能用虚拟显示')
    output = args.output or PROJECT/'outputs/live_calibration'/('LINE0_GUI_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    window = HardwareTriggerWindow(output, args.hardware, legacy)
    if args.smoke_test:
        assert not window.windows and window.worker is None and not output.exists()
        window.start_live(project=False)
        assert window.worker is None
        QtCore.QTimer.singleShot(250, app.quit)
    app.exec_()


if __name__ == '__main__':
    main()
