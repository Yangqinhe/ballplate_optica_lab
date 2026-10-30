"""Additive v21: display gain only, same v20 worker and device ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from PyQt5 import QtCore, QtWidgets
from live_deploy_v20.direct_alignment_gui import DeploymentWindow
from live_deploy_v21.page import DisplayGainPage
from realtime_worker import write_json


class DisplayGainWindow(DeploymentWindow):
    def __init__(self, *args, result_folder=None, **kwargs):
        self._gain_heartbeat = None
        super().__init__(*args, **kwargs)
        self.setWindowTitle('实时光路对齐 v19 · 对齐/部署 v21 · 仅显示增益'+
                            (' · 真实器件' if self.hardware else ' · 离线预览'))
        if result_folder:
            try:
                self.deployment_page.restore_result(result_folder)
            except Exception as error:
                self.deployment_page.status.setText('历史图未采用：'+str(error)+'；未自动采集或投图。')
            self.tabs.setCurrentIndex(1)
        if self.hardware:
            path = self.output/'runtime_sources.json'
            runtime = json.loads(path.read_text())
            runtime['sources'].update({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
            runtime.update(deployment_ui_revision=21, display_gain_only=True,
                           camera_gain_or_exposure_changed_by_display_controls=False)
            write_json(path, runtime)

    def build_ui(self):
        super().build_ui()
        old = self.deployment_page
        self.tabs.removeTab(1)
        self.deployment_page = DisplayGainPage()
        self.tabs.addTab(self.deployment_page, '离线部署测试')
        self.deployment_page.freeze_requested.connect(self.freeze_deployment)
        self.deployment_page.run_requested.connect(self.run_deployment)
        self.deployment_page.cancel_requested.connect(self.cancel_deployment)
        old.deleteLater()
        self.setMinimumHeight(1220)

    def tick(self):
        super().tick()
        if self.hardware and self._gain_heartbeat != self.last_heartbeat:
            self._gain_heartbeat = self.last_heartbeat
            path = self.output/'heartbeat.json'
            if path.exists():
                state = json.loads(path.read_text())
                p = self.deployment_page
                state.update(deployment_ui_revision=21, display_only=dict(gain=p.display_gain.value(),
                    black_ADU=p.display_black.value(), gamma=p.display_gamma.value(),
                    history_NOT_live=p.view_mode.currentIndex() == 1), closed_loop_enabled=False)
                tmp = self.output/'heartbeat_gain.tmp'
                write_json(tmp, state); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--restart-state', type=Path)
    p.add_argument('--recovery-record', type=Path)
    p.add_argument('--deployment-catalog', type=Path)
    p.add_argument('--result-folder', type=Path)
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--preview-screenshot', type=Path)
    a = p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test or a.preview_screenshot):
        p.error('new real output windows require this-switch physical shielding confirmation')
    if a.preview_screenshot and a.preview_screenshot.exists():
        p.error('do not overwrite layout QA')
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName() != 'xcb':
        raise RuntimeError('local X11 required')
    w = DisplayGainWindow(a.output, a.hardware,
        None if not a.restart_state else json.loads(a.restart_state.read_text()),
        None if not a.recovery_record else json.loads(a.recovery_record.read_text()),
        a.deployment_catalog, result_folder=a.result_folder)
    if a.smoke_test or a.preview_screenshot:
        assert not w.windows and w.worker is None and not a.output.exists()
        w.show(); app.processEvents()
        if a.preview_screenshot:
            w.grab().save(str(a.preview_screenshot))
        QtCore.QTimer.singleShot(300, app.quit)
    app.exec_()


if __name__ == '__main__':
    main()
