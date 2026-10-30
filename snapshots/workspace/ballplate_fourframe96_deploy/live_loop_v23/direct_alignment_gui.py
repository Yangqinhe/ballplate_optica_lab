"""Unified live pipeline on the existing workbench. Default: black, stopped."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; sys.path.insert(0,str(PROJECT))
from PyQt5 import QtCore, QtWidgets
from live_adapt_v22.direct_alignment_gui import AdaptationWindow
from live_deploy_v20.contract import matching
from live_adapt_v22.core import digest
from realtime_worker import write_json
from live_loop_v23.core import DEFAULT_HEAD, DEFAULT_SHA
from live_loop_v23.page import LoopPage
from live_loop_v23.worker import LoopWorker


class LoopWindow(AdaptationWindow):
    def __init__(self,*args,loop_head=DEFAULT_HEAD,**kwargs):
        self.loop_active=False; self.loop_sequence=0; self._loop_heartbeat=None
        super().__init__(*args,**kwargs)
        self.setWindowTitle('实时光路对齐 v19 · 全链路互动 v23 · 真实光路→仅MuJoCo'+(' · 真实器件' if self.hardware else ' · 离线界面预览'))
        if loop_head:
            try:self.loop_page.load_head(loop_head,DEFAULT_SHA if Path(loop_head).resolve()==DEFAULT_HEAD else None)
            except Exception as error:self.loop_page.status.setText('默认头未载入：'+str(error))
        self.tabs.setCurrentIndex(3)
        # Disable old stride1 pilot: current actor is trained on stride3 inputs.
        self.adaptation_page.pilot_requested.disconnect()
        self.adaptation_page.pilot.hide()
        self.adaptation_page.mode.setItemText(0,'固定先验mask（不是新学习权重）')
        if self.hardware:
            path=self.output/'runtime_sources.json';r=json.loads(path.read_text())
            r['sources'].update({str(p):digest(p) for p in ROOT.glob('*.py')})
            r.update(deployment_ui_revision=23,interactive_real_optical_MuJoCo=True,
                live_frame_counter_raw_preserved=True,real_actuators_enabled=False,realtime_20ms_qualified=False)
            write_json(path,r)
        self.refresh()

    def build_ui(self):
        super().build_ui();self.loop_page=LoopPage();self.tabs.addTab(self.loop_page,'实时全链路 / 手动扰动')
        self.loop_page.start_requested.connect(self.run_loop)
        self.loop_page.stop_requested.connect(self.cancel_deployment)
        self.loop_page.push_requested.connect(self.queue_push)

    def run_loop(self):
        p=self.loop_page;queued=False
        try:
            if (not self.hardware or self.fault or self.deployment_busy or self.roi_busy or self.reference_busy
                    or self.override or not self.running or not self.live_ready or not isinstance(self.worker,LoopWorker)
                    or not self.worker.isRunning() or not self.last_packet
                    or time.monotonic()-self.last_packet['received_monotonic']>1):
                raise RuntimeError('请先启动顶部“仅CCD”，等待新鲜帧、无故障且配置锁定')
            if not p.bundle or not p.head_path or not p.confirm.isChecked():
                raise RuntimeError('先载入头并确认本轮低功率／防护条件')
            matching(p.bundle.contract,asdict(self.setting()),asdict(self.optical_config()),self.last_packet['active'],self.registration)
            p.bundle.verify()
            folder=self.output/('interactive_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.deployment_busy=self.loop_active=True;self.override=True;self.dirty=False
            self._deployment_fingerprint=self.fingerprint()
            self._internal_park=True;self.armed.setChecked(True);self._internal_park=False
            for i in range(3):self.tabs.setTabEnabled(i,False)
            self.tabs.setCurrentIndex(3);p.set_busy(True);self.loop_sequence=0
            self.worker.request_loop(p.head_path,digest(p.head_path),folder,p.seconds.value(),[p.x.value()/1000,p.y.value()/1000])
            queued=True
            self.record(dict(state='interactive_real_optical_requested',folder=str(folder),real_actuators_enabled=False))
        except Exception as error:
            if self.loop_active:
                self.cancel_deployment()
                if not queued:self.loop_finished(dict(folder='',report=dict(completed=False,actions=0,error=str(error))))
            p.status.setText('未开始：'+str(error))
        self.refresh()

    def queue_push(self,axis,sign,speed):
        try:
            if not self.loop_active or not isinstance(self.worker,LoopWorker):raise RuntimeError('当前没有MuJoCo闭环')
            event=self.worker.pushes.add(axis,sign,speed);self.loop_page.queued(event)
            self.record(dict(state='MuJoCo_only_push_queued',event=event))
        except Exception as error:self.loop_page.push_status.setText('未施加：'+str(error))

    def loop_finished(self,result):
        # Flush latest mailbox before the final result clears the running flag.
        self.consume_loop_view()
        self.deployment_busy=self.loop_active=False;self.override=False
        for i in range(3):self.tabs.setTabEnabled(i,True)
        self._internal_park=True;self.armed.setChecked(False);self._internal_park=False
        self.black_outputs();self.dirty=True;self.loop_page.finished(result);self.refresh()
        self.record(dict(state='interactive_real_optical_finished',folder=result['folder'],report=result['report']))

    def cancel_deployment(self):
        if self.loop_active and isinstance(self.worker,LoopWorker):
            self.worker.pushes.close();self.loop_page.status.setText('正在停止：不再施加扰动／不复用旧帧，等待保存原始记录…')
        super().cancel_deployment()

    def consume_loop_view(self):
        if not isinstance(self.worker,LoopWorker):return
        n,data=self.worker.loop_view.get()
        if data and n!=self.loop_sequence:
            self.loop_sequence=n;self.loop_page.on_view(data)

    def tick(self):
        super().tick()
        if hasattr(self,'loop_page'):self.consume_loop_view()
        if self.loop_active:self.snap_button.setEnabled(False)
        if self.hardware and self._loop_heartbeat!=self.last_heartbeat:
            self._loop_heartbeat=self.last_heartbeat;path=self.output/'heartbeat.json'
            if path.exists():
                h=json.loads(path.read_text());h.update(deployment_ui_revision=23,loop_active=self.loop_active,
                    real_actuators_enabled=False,realtime_20ms_qualified=False,
                    closed_loop_enabled=self.loop_active,loop_last_step=self.loop_page.latest_step,
                    current_page=self.tabs.currentIndex())
                tmp=self.output/'heartbeat_loop.tmp';write_json(tmp,h);tmp.replace(path)

    def update_comparison(self,now):
        if hasattr(self,'tabs') and self.tabs.currentIndex()==3:return
        super().update_comparison(now)

    def refresh(self):
        super().refresh()
        if self.loop_active:self.recovery_status.setText('真实光路→仅MuJoCo；配置锁定，方向按钮可加扰动；Esc立即全0。')

    def start_live(self,project=True):
        # Proven v22 startup, same signals and transaction; only worker differs.
        if not self.hardware or self.fault or (self.worker and self.worker.isRunning()):return
        if project and not self.armed.isChecked():
            answer=QtWidgets.QMessageBox.question(self,'启动三路实时','确认原ND／低功率和光束防护未变？',
                QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
            if answer!=QtWidgets.QMessageBox.Yes:return
            self.armed.setChecked(True)
        if not project:self.armed.setChecked(False)
        self.preview_background_warning=None;self.starting_registration=self.registration or self.pending_registration
        self.record(dict(state='interactive_Line0_start',projection_requested=bool(project),requested_roi=dict(self.actual_roi)))
        self.black_outputs();self.running=True;self.live_ready=False;self.override=False;self.roi_busy=False;self.dirty=True
        self.last_sequence=0;self.last_packet=None;self.loop_sequence=0
        self.view.setText('检查Line0与3张零输入帧；不自动开始闭环…')
        folder=self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker=LoopWorker(self.baseline,folder,self.exposure.value(),roi=dict(self.actual_roi));self.history.append(self.worker)
        for name,callback in (('ready',self.camera_ready),('configured',self.camera_configured),('roi_found',self.roi_found),
                ('zero_reference',self.zero_reference_captured),('spatial_result',self.spatial_received),
                ('display',self.display_request),('message',self.note),('black',self.black_outputs),('result',self.camera_finished),('finished',self.refresh)):
            getattr(self.worker,name).connect(callback)
        self.worker.static_progress.connect(self.deployment_page.on_progress);self.worker.static_result.connect(self.deployment_finished)
        self.worker.adaptation_progress.connect(self.adaptation_page.on_progress);self.worker.adaptation_result.connect(self.adaptation_finished)
        self.worker.loop_result.connect(self.loop_finished);self.worker.start();self.refresh()


def main():
    import cv2
    cv2.setNumThreads(2)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware',action='store_true');p.add_argument('--light-blocked-confirmed',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--restart-state',type=Path)
    p.add_argument('--recovery-record',type=Path);p.add_argument('--deployment-catalog',type=Path)
    p.add_argument('--result-folder',type=Path);p.add_argument('--adaptation-catalog',type=Path)
    p.add_argument('--loop-head',type=Path,default=DEFAULT_HEAD)
    p.add_argument('--smoke-test',action='store_true');p.add_argument('--preview-screenshot',type=Path)
    a=p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test or a.preview_screenshot):
        p.error('new device windows require fresh physical shielding confirmation')
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling,True)
    app=QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName()!='xcb':raise RuntimeError('local X11 required')
    w=LoopWindow(a.output,a.hardware,None if not a.restart_state else json.loads(a.restart_state.read_text()),
        None if not a.recovery_record else json.loads(a.recovery_record.read_text()),a.deployment_catalog,
        result_folder=a.result_folder,adaptation_catalog=a.adaptation_catalog,loop_head=a.loop_head)
    w.show()
    if a.smoke_test or a.preview_screenshot:
        assert not w.windows and w.worker is None and not a.output.exists()
        app.processEvents()
        if a.preview_screenshot:
            if a.preview_screenshot.exists():raise ValueError('do not overwrite QA')
            w.grab().save(str(a.preview_screenshot))
        QtCore.QTimer.singleShot(300,app.quit)
    app.exec_()


if __name__=='__main__':main()
