"""Additive v22: same outputs/SDK, real single-frame data and MuJoCo only."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from PyQt5 import QtCore, QtWidgets
from live_deploy_v21.direct_alignment_gui import DisplayGainWindow
from live_deploy_v20.contract import matching
from live_adapt_v22.page import AdaptationPage
from live_adapt_v22.worker import AdaptationWorker
from realtime_worker import write_json


class AdaptationWindow(DisplayGainWindow):
    def __init__(self, *args, adaptation_catalog=None, **kwargs):
        self.adaptation_active = False; self.adaptation_kind = None; self._adapt_heartbeat = None
        super().__init__(*args, **kwargs)
        self.setWindowTitle('实时光路对齐 v19 · 单帧适配 v22 · 仅MuJoCo'+(' · 真实器件' if self.hardware else ' · 离线预览'))
        if adaptation_catalog:
            self.adaptation_page.load_bundle(adaptation_catalog); self.tabs.setCurrentIndex(2)
        if self.hardware:
            path = self.output/'runtime_sources.json'; runtime = json.loads(path.read_text())
            runtime['sources'].update({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
            runtime.update(deployment_ui_revision=22, real_single_frame_adaptation=True,
                           real_actuators_enabled=False, slow_MuJoCo_pilot_enabled=True,
                           realtime_20ms_qualified=False)
            write_json(path, runtime)

    def build_ui(self):
        super().build_ui()
        self.adaptation_page = AdaptationPage(); self.tabs.addTab(self.adaptation_page, '单帧适配 / MuJoCo')
        self.adaptation_page.collect_requested.connect(lambda: self.run_adaptation(False))
        self.adaptation_page.pilot_requested.connect(lambda: self.run_adaptation(True))
        self.adaptation_page.cancel_requested.connect(self.cancel_deployment)

    def run_adaptation(self, pilot=False):
        page = self.adaptation_page; queued = False
        try:
            if (not self.hardware or self.fault or self.deployment_busy or self.roi_busy
                    or self.reference_busy or self.override or not self.running or not self.live_ready
                    or not isinstance(self.worker, AdaptationWorker) or not self.worker.isRunning()
                    or not self.last_packet or time.monotonic()-self.last_packet['received_monotonic'] > 1):
                raise RuntimeError('先启动唯一CCD并确认新鲜完整帧，且没有其他任务')
            if not page.bundle or not page.confirm.isChecked():
                raise RuntimeError('载入适配包并确认本次原低功率／防护条件')
            if pilot and not page.head_path:
                raise RuntimeError('先通过未见轨迹单帧动作门槛')
            matching(page.bundle.contract, asdict(self.setting()), asdict(self.optical_config()),
                     self.last_packet['active'], self.registration)
            page.bundle.verify()
            folder = self.output/(('mujoco_pilot_' if pilot else 'single_frame_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.deployment_busy = self.adaptation_active = True
            self.adaptation_kind = 'pilot' if pilot else 'dataset'
            self.override = True; self.dirty = False; self._deployment_fingerprint = self.fingerprint()
            self._internal_park = True; self.armed.setChecked(True); self._internal_park = False
            for i in (0, 1): self.tabs.setTabEnabled(i, False)
            self.tabs.setCurrentIndex(2); page.set_busy(True)
            self.worker.request_adaptation(page.bundle, folder, page.mode.currentData(), page.head_path if pilot else None)
            queued = True
            self.record(dict(state='bounded_single_frame_requested', task=self.adaptation_kind,
                             output=str(folder), real_actuators_enabled=False,
                             software_settle_seconds=1.5, realtime_20ms_qualified=False))
        except Exception as error:
            if self.adaptation_active:
                self.cancel_deployment()
                if not queued: self.adaptation_finished(dict(report=dict(task='none', error=str(error)), folder='', head=None))
            page.status.setText('未开始：'+str(error))
        self.refresh()

    def adaptation_finished(self, result):
        self.deployment_busy = self.adaptation_active = False
        self.adaptation_kind = None; self.override = False
        for i in (0, 1): self.tabs.setTabEnabled(i, True)
        self._internal_park = True; self.armed.setChecked(False); self._internal_park = False
        self.black_outputs(); self.dirty = True
        self.adaptation_page.finished(result); self.refresh()
        self.record(dict(state='bounded_single_frame_finished', folder=result['folder'],
                         error=result['report'].get('error'),
                         eligible=bool(result.get('head', {}).get('short_MuJoCo_pilot_eligible')) if result.get('head') else False,
                         real_actuators_enabled=False))

    def update_comparison(self, now):
        if hasattr(self, 'tabs') and self.tabs.currentIndex() == 2: return
        super().update_comparison(now)

    def tick(self):
        super().tick()
        if hasattr(self, 'adaptation_page') and self.last_packet and self.running:
            p = self.deployment_page
            self.adaptation_page.fresh_view(self.last_packet, p.display_gain.value(), p.display_black.value(), p.display_gamma.value())
        if self.hardware and self._adapt_heartbeat != self.last_heartbeat:
            self._adapt_heartbeat = self.last_heartbeat; path = self.output/'heartbeat.json'
            if path.exists():
                h = json.loads(path.read_text()); h.update(deployment_ui_revision=22,
                    adaptation_active=self.adaptation_active, adaptation_kind=self.adaptation_kind,
                    real_actuators_enabled=False, realtime_20ms_qualified=False)
                tmp = self.output/'heartbeat_adaptation.tmp'; write_json(tmp, h); tmp.replace(path)

    def refresh(self):
        super().refresh()
        if self.adaptation_active:
            self.recovery_status.setText('单帧适配/仅MuJoCo任务中：配置锁定；Esc全0取消；不控制实物。')

    def start_live(self, project=True):
        # Same proven r5/v20 startup, replacing ONLY the worker class.
        if not self.hardware or self.fault or (self.worker and self.worker.isRunning()): return
        if project and not self.armed.isChecked():
            answer = QtWidgets.QMessageBox.question(self, '启动三路实时',
                '确认仍为原ND／低功率和光束防护？先检验Line0，再允许所选图案。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes: return
            self.armed.setChecked(True)
        if not project: self.armed.setChecked(False)
        self.preview_background_warning = None
        self.starting_registration = self.registration or self.pending_registration
        self.record(dict(state='manual_line0_deployment_start', projection_requested=bool(project),
            requested_roi=dict(self.actual_roi), requested_exposure_us=self.exposure.value()))
        self.black_outputs(); self.running = True; self.live_ready = False
        self.override = False; self.roi_busy = False; self.dirty = True
        self.last_sequence = 0; self.last_packet = None
        self.view.setText('验证Line0与零输入帧；不自动开始适配或闭环…')
        folder = self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker = AdaptationWorker(self.baseline, folder, self.exposure.value(), roi=dict(self.actual_roi))
        self.history.append(self.worker)
        for signal, callback in (('ready', self.camera_ready), ('configured', self.camera_configured),
                ('roi_found', self.roi_found), ('zero_reference', self.zero_reference_captured),
                ('spatial_result', self.spatial_received), ('display', self.display_request),
                ('message', self.note), ('black', self.black_outputs), ('result', self.camera_finished),
                ('finished', self.refresh)):
            getattr(self.worker, signal).connect(callback)
        self.worker.static_progress.connect(self.deployment_page.on_progress)
        self.worker.static_result.connect(self.deployment_finished)
        self.worker.adaptation_progress.connect(self.adaptation_page.on_progress)
        self.worker.adaptation_result.connect(self.adaptation_finished)
        self.worker.start(); self.refresh()


def main():
    import cv2
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware', action='store_true'); p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--restart-state', type=Path)
    p.add_argument('--recovery-record', type=Path); p.add_argument('--deployment-catalog', type=Path)
    p.add_argument('--result-folder', type=Path); p.add_argument('--adaptation-catalog', type=Path)
    p.add_argument('--smoke-test', action='store_true'); p.add_argument('--preview-screenshot', type=Path)
    a = p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test or a.preview_screenshot):
        p.error('new device windows require fresh physical shielding confirmation')
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName() != 'xcb': raise RuntimeError('local X11 required')
    w = AdaptationWindow(a.output, a.hardware,
        None if not a.restart_state else json.loads(a.restart_state.read_text()),
        None if not a.recovery_record else json.loads(a.recovery_record.read_text()),
        a.deployment_catalog, result_folder=a.result_folder, adaptation_catalog=a.adaptation_catalog)
    if a.smoke_test or a.preview_screenshot:
        assert not w.windows and w.worker is None and not a.output.exists()
        w.show(); app.processEvents()
        if a.preview_screenshot:
            if a.preview_screenshot.exists(): raise ValueError('do not overwrite QA screenshot')
            w.grab().save(str(a.preview_screenshot))
        QtCore.QTimer.singleShot(300, app.quit)
    app.exec_()


if __name__ == '__main__': main()
