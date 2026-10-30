"""Same v19 window, additive ordered-ROI/idempotent configuration repair."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from live_manual_v19_trigger.direct_alignment_gui import HardwareTriggerWindow, validate_legacy_reference
from live_manual_v19_trigger_r2.worker import OrderedTriggerWorker
from realtime_worker import write_json
from PyQt5 import QtCore, QtWidgets


class OrderedTriggerWindow(HardwareTriggerWindow):
    def __init__(self, output, hardware=False, legacy_reference=None):
        super().__init__(output, hardware, legacy_reference)
        self.setWindowTitle('实时光路对齐 v19 · Line0硬触发 r2' + (' · 真实器件' if hardware else ' · 离线预览'))
        if hardware:
            path = self.output/'runtime_sources.json'
            runtime = json.loads(path.read_text())
            for module in list(sys.modules.values()):
                filename = getattr(module, '__file__', None)
                if filename and Path(filename).resolve().parent == ROOT and filename.endswith('.py'):
                    source = str(Path(filename).resolve())
                    runtime['sources'][source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            runtime.update(trigger_revision=2, trigger_Off_before_ROI=True,
                           unchanged_configuration_does_not_restart=True)
            write_json(path, runtime)

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
        self.starting_registration = self.registration
        self.record(dict(state='manual_line0_r2_start', projection_requested=bool(project),
            requested_roi=dict(self.actual_roi), requested_exposure_us=self.exposure.value()))
        self.black_outputs(); self.running = True; self.live_ready = False
        self.override = False; self.roi_busy = False; self.dirty = True
        self.last_sequence = 0; self.last_packet = None
        self.view.setText('正在检查Line0硬触发与黑场，未投图…')
        folder = self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker = OrderedTriggerWorker(self.baseline, folder, self.exposure.value(), roi=dict(self.actual_roi))
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
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--legacy-reference', type=Path)
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    if args.hardware and (not args.light_blocked_confirmed or args.smoke_test):
        p.error('新建输出窗口必须有当次物理遮光确认；离线测试不接硬件')
    legacy = None if args.legacy_reference is None else validate_legacy_reference(json.loads(args.legacy_reference.read_text()))
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if args.hardware and app.platformName() != 'xcb':
        raise RuntimeError('真实窗口必须使用本地X11')
    output = args.output or PROJECT/'outputs/live_calibration'/('LINE0_R2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    window = OrderedTriggerWindow(output, args.hardware, legacy)
    if args.smoke_test:
        assert not window.windows and window.worker is None and not output.exists()
        window.start_live(project=False)
        assert window.worker is None
        QtCore.QTimer.singleShot(250, app.quit)
    app.exec_()


if __name__ == '__main__':
    main()
