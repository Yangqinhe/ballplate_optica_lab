"""Additive restart repair: same v18 controls and unmodified v17 drivers.

Keeps the requested ROI on camera restart. An explicitly loaded old geometric
reference is PENDING until the new active camera/model signature matches.
Never auto starts capture, projection, fitting, or hardware closed-loop.
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
sys.path.insert(0, str(ROOT.parent))
from live_manual_v18.direct_alignment_gui import DirectAlignmentWindow
from realtime_worker import LiveCameraWorker, write_json
from diffraction_model import reference_signature
from display_linux import same_signals
from PyQt5 import QtCore, QtWidgets
try:
    from .resume_state import validate
except ImportError:
    from resume_state import validate


class RestartSafeWindow(DirectAlignmentWindow):
    def __init__(self, output, hardware=False, resume=None):
        self.pending_registration = None
        self.starting_registration = None
        super().__init__(output, hardware)
        if resume is not None:
            self.restore_pending(resume)
        if hardware:
            path = self.output/'runtime_sources.json'
            runtime = json.loads(path.read_text())
            for module in list(sys.modules.values()):
                filename = getattr(module, '__file__', None)
                if filename and Path(filename).resolve().parent == ROOT and filename.endswith('.py'):
                    filename = str(Path(filename).resolve())
                    runtime['sources'][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            runtime.update(restart_revision=1, retained_ROI_on_restart=True,
                           old_reference_requires_new_signature=True)
            write_json(path, runtime)

    def restore_pending(self, contract):
        if self.running or self.armed.isChecked() or self.worker is not None:
            raise RuntimeError('resume only before any camera start')
        contract = validate(contract)
        if self.hardware:
            same_signals(contract['display_roles'], self.baseline)
        if asdict(self.optical_config()) != contract['optical_config']:
            raise ValueError('new model defaults do not match saved reference; do not guess')
        s = contract['settings']
        widgets = [self.dmd_mode, self.slm_mode, self.square, self.full, self.focal,
                   self.period, self.exposure, *self.spins.values(), *self.sliders.values()]
        for w in widgets: w.blockSignals(True)
        try:
            for combo, key in ((self.dmd_mode, 'dmd'), (self.slm_mode, 'slm')):
                index = combo.findData(s[key])
                if index < 0: raise ValueError('unknown retained mode')
                combo.setCurrentIndex(index)
            self.square.setValue(s['square']); self.full.setChecked(s['full_panel'])
            self.focal.setValue(s['focal_mm']); self.period.setValue(s['period'])
            for key in ('x', 'y'):
                self.spins[key].setValue(s[key]); self.sliders[key].setValue(s[key])
            self.exposure.setValue(contract['requested_exposure_us'])
        finally:
            for w in widgets: w.blockSignals(False)
        self.actual_roi = dict(contract['roi'])
        self.pending_registration = dict(contract['registration'])
        self.registration = None
        self.roi_text.setText('待复核ROI：'+str(self.actual_roi)+'；尚未开启CCD')
        self.mapping_status.setText('旧倍率待复核：新CCD参数完全匹配后才恢复，不重新拟合。')
        self.render_patterns(); self.black_outputs(); self.refresh()
        if self.hardware:
            self.record(dict(state='explicit_alignment_resume_pending', contract=contract,
                             no_automatic_capture=True, no_automatic_projection=True))

    def clear_mapping(self):
        self.pending_registration = None
        super().clear_mapping()

    def start_live(self, project=True):
        if not self.hardware or self.fault or (self.worker and self.worker.isRunning()):
            return
        # Retain the original confirmation UI, with exactly the original default No.
        if project and not self.armed.isChecked():
            answer = QtWidgets.QMessageBox.question(self, '启动三路实时',
                '此操作会将所选图案加载到DMD、SLM，并启动CCD。\n'
                '确认仍为已衰减的ND／低功率照明，且工作光束和OFF光均有防护？',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
            self.armed.setChecked(True)
        if not project:
            self.armed.setChecked(False)
        self.starting_registration = self.registration or self.pending_registration
        self.record(dict(state='manual_live_start', projection_requested=bool(project),
            low_power_confirmed=self.armed.isChecked(), requested_roi=dict(self.actual_roi),
            retained_geometry_signature=(self.starting_registration or {}).get('signature')))
        self.black_outputs(); self.running = True; self.live_ready = False
        self.override = False; self.roi_busy = False; self.dirty = True
        self.last_sequence = 0; self.last_packet = None
        self.view.setText('正在按保留ROI打开唯一相机15328133…')
        folder = self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker = LiveCameraWorker(self.baseline, folder, self.exposure.value(), roi=dict(self.actual_roi))
        self.history.append(self.worker)
        for signal, callback in (('ready', self.camera_ready), ('configured', self.camera_configured),
                ('roi_found', self.roi_found), ('zero_reference', self.zero_reference_captured),
                ('spatial_result', self.spatial_received), ('display', self.display_request),
                ('message', self.note), ('black', self.black_outputs), ('result', self.camera_finished),
                ('finished', self.refresh)):
            getattr(self.worker, signal).connect(callback)
        self.worker.start(); self.refresh()

    def camera_configured(self, payload):
        registration = self.starting_registration
        self.starting_registration = None
        if registration is not None and (
                registration['signature'] != reference_signature(self.optical_config(), payload['active'])
                or payload['actual_roi'] != self.actual_roi):
            # Fail before camera-ready/projection; never silently re-fit/rebaseline.
            self.fault = '恢复条件不匹配：CCD/模型与保留几何不一致；未投图，需人工复核'
            self.stop_live()
            return
        super().camera_configured(payload)
        if registration is not None:
            self.registration = dict(registration); self.pending_registration = None
            self.orientation.setCurrentIndex(registration['orientation'])
            self.mapping_status.setText('原倍率已恢复：新CCD/模型签名一致；未重新拟合。')
            self.comparison_changed(); self.update_direction_status()
            if self.hardware:
                self.record(dict(state='retained_geometry_revalidated', signature=registration['signature'],
                                 active=payload['active'], newly_fitted=False))

    def camera_ready(self, payload):
        if self.fault or not self.running:
            self.black_outputs()
            return
        super().camera_ready(payload)


def main():
    import cv2
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume-contract', type=Path)
    p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test):
        p.error('真实窗口需要当次遮光确认；离线测试不能连接硬件')
    resume = None if a.resume_contract is None else validate(json.loads(a.resume_contract.read_text()))
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName() != 'xcb':
        raise RuntimeError('real local X11 required')
    output = a.output or ROOT.parent/'outputs/live_calibration'/('DIRECT_RESTART_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    w = RestartSafeWindow(output, a.hardware, resume)
    if a.smoke_test:
        assert not w.windows and not output.exists()
        w.start_live(); assert w.worker is None
        QtCore.QTimer.singleShot(250, app.quit)
    app.exec_()


if __name__ == '__main__':
    main()
