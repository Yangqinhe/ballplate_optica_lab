"""One desktop owner, common hardware controls, alignment + deployment tabs."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE')
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;sys.path.insert(0,str(PROJECT))
import numpy as np
from PyQt5 import QtCore,QtWidgets
from live_manual_v19_trigger_r5.direct_alignment_gui import ManualPreviewWindow
from live_manual_v19_trigger_r3.state import validate as validate_legacy_controls
from live_deploy_v20.contract import validate as validate_retained,matching
from live_deploy_v20.worker import DeploymentWorker
from live_deploy_v20.page import DeploymentPage
from display_linux import same_signals
from realtime_worker import write_json


class DeploymentWindow(ManualPreviewWindow):
    def __init__(self,output,hardware=False,restart_state=None,recovery_record=None,catalog=None):
        self.deployment_busy=False;self.deployment_pending=None;self.deployment_frozen=False
        self._internal_park=False;self._deployment_heartbeat=None
        modern=restart_state is not None and restart_state.get('schema')=='v20-retained-line0-v1'
        if modern:validate_retained(restart_state)
        super().__init__(output,hardware,None if modern else restart_state,recovery_record)
        if modern:self.restore_modern(restart_state,keep_fault=recovery_record is not None)
        self.setWindowTitle('实时光路对齐 v19 · 对齐/离线部署 v20'+(' · 真实器件' if hardware else ' · 离线预览'))
        if catalog is not None:
            self.deployment_page.load_bundle(catalog);self.tabs.setCurrentIndex(1)
        if hardware:
            path=self.output/'runtime_sources.json';runtime=json.loads(path.read_text())
            runtime['sources'].update({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
            runtime.update(trigger_revision=6,deployment_ui_revision=20,tabs_share_one_SDK_and_output_owner=True,
                bounded_static_replay_enabled=True,closed_loop_enabled=False)
            write_json(path,runtime)
        self.refresh()

    def build_ui(self):
        super().build_ui()
        layout=self.centralWidget().layout();item=layout.takeAt(4)
        middle=item.layout()
        if not isinstance(middle,QtWidgets.QHBoxLayout):raise RuntimeError('unexpected original alignment layout')
        middle.setParent(None);alignment=QtWidgets.QWidget();alignment.setLayout(middle)
        self.tabs=QtWidgets.QTabWidget();self.tabs.addTab(alignment,'手动对齐')
        self.deployment_page=DeploymentPage();self.tabs.addTab(self.deployment_page,'离线部署测试')
        layout.insertWidget(4,self.tabs,1)
        self.setMinimumHeight(1160)
        self.deployment_page.freeze_requested.connect(self.freeze_deployment)
        self.deployment_page.run_requested.connect(self.run_deployment)
        self.deployment_page.cancel_requested.connect(self.cancel_deployment)

    def restore_modern(self,c,keep_fault=False):
        if self.hardware:same_signals(c['display_roles'],self.baseline)
        widgets=[self.dmd_mode,self.slm_mode,self.full,self.square,self.focal,self.period,self.exposure,
            self.orientation,self.slm_flip_y,*self.spins.values(),*self.sliders.values(),*self.optics.values()]
        for w in widgets:w.blockSignals(True)
        try:
            s=c['settings'];opt=c['optical_config']
            self.dmd_mode.setCurrentIndex(self.dmd_mode.findData(s['dmd']))
            self.slm_mode.setCurrentIndex(self.slm_mode.findData(s['slm']))
            self.full.setChecked(s['full_panel']);self.square.setValue(s['square'])
            self.focal.setValue(s['focal_mm']);self.period.setValue(s['period']);self.exposure.setValue(c['requested_exposure_us'])
            for key in ('x','y'):self.spins[key].setValue(s[key]);self.sliders[key].setValue(s[key])
            for key,val in dict(wavelength=opt['wavelength_m']*1e9,pitch=opt['pitch_m']*1e6,
                    f1=opt['f1_m']*1000,f2=opt['f2_m']*1000,phase_depth=opt['phase_depth_rad']).items():
                self.optics[key].setValue(val)
            self.slm_flip_y.setChecked(opt['slm_flip_y'])
            if c['registration']:self.orientation.setCurrentIndex(c['registration']['orientation'])
        finally:
            for w in widgets:w.blockSignals(False)
        self.actual_roi=dict(c['roi']);self.registration=None;self.pending_registration=c['registration']
        self.fault_before=dict(c['original_fault_before']);self.last_camera_report=c['original_last_camera_report']
        if not keep_fault:self.fault=c['fault']
        self.roi_text.setText('保留ROI：'+str(self.actual_roi))
        self.mapping_status.setText('原倍率待新CCD签名复核：保留原矩阵，不重新拟合。')
        self.render_patterns();self.black_outputs()
        if self.hardware:self.record(dict(state='line0_reference_retained_pending',contract=c,no_auto_capture=True,no_auto_projection=True))

    def freeze_deployment(self):
        page=self.deployment_page
        try:
            if (not self.hardware or self.fault or not self.running or not self.live_ready or not self.last_packet
                    or time.monotonic()-self.last_packet['received_monotonic']>1):
                raise RuntimeError('先启动唯一CCD并确认当前完整帧')
            if self.deployment_busy or self.roi_busy or self.reference_busy or self.override:raise RuntimeError('其他任务尚未结束')
            if not page.bundle:raise RuntimeError('先加载准备好的测试包')
            matching(page.bundle.contract,asdict(self.setting()),asdict(self.optical_config()),
                self.last_packet['active'],self.registration)
            page.bundle.verify();self.deployment_frozen=True;page.frozen=True
            page.status.setText('已冻结当前ROI/倍率/曝光/SLM位置与资产。勾选本次低功率确认，再运行；不加载旧actor。')
        except Exception as error:
            self.deployment_frozen=False;page.frozen=False;page.status.setText('未冻结：'+str(error))
        page.update_buttons()

    def run_deployment(self,case_ids):
        page=self.deployment_page
        queued=False
        try:
            if not self.deployment_frozen or not page.confirm.isChecked():raise RuntimeError('先冻结配置并确认本次低功率条件')
            if self.deployment_busy or not isinstance(self.worker,DeploymentWorker):raise RuntimeError('需要本窗口的唯一CCD线程')
            if not self.running or not self.live_ready or not self.worker.isRunning() or not self.last_packet:
                raise RuntimeError('CCD未运行，不能开始测试')
            matching(page.bundle.contract,asdict(self.setting()),asdict(self.optical_config()),
                self.last_packet['active'],self.registration)
            if time.monotonic()-self.last_packet['received_monotonic']>1:raise RuntimeError('CCD不是新鲜帧')
            folder=self.output/('deployment_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.deployment_busy=True;self.override=True;self.dirty=False
            self._deployment_fingerprint=self.fingerprint()
            self._internal_park=True;self.armed.setChecked(True);self._internal_park=False
            self.tabs.setTabEnabled(0,False);self.tabs.setCurrentIndex(1)
            page.set_busy(True,len(set([8,*case_ids])))
            self.worker.request_static(page.bundle,case_ids,folder)
            queued=True
            self.record(dict(state='bounded_static_deployment_requested',case_ids=case_ids,folder=str(folder),no_closed_loop=True))
        except Exception as error:
            if self.deployment_busy:
                self.cancel_deployment()
                if not queued:
                    self.deployment_busy=False;self.override=False;self.tabs.setTabEnabled(0,True);page.set_busy(False)
            page.status.setText('未开始：'+str(error))
        self.refresh()

    def fingerprint(self):
        return json.dumps(dict(settings=asdict(self.setting()),config=asdict(self.optical_config()),
            registration=self.registration,roi=self.actual_roi,exposure=self.exposure.value()),sort_keys=True)

    def cancel_deployment(self):
        if self.worker and isinstance(self.worker,DeploymentWorker):self.worker.cancel_static()
        self.finish_deployment_display('操作者取消测试')
        self._internal_park=True;self.armed.setChecked(False);self._internal_park=False
        self.black_outputs()
        if self.deployment_busy:self.deployment_page.status.setText('正在取消并全0，等待采集线程保存已有原图；不重试。')

    def arm_changed(self,enabled):
        if not enabled and self.deployment_busy and not self._internal_park:self.cancel_deployment()
        super().arm_changed(enabled)

    def stop_live(self):
        if self.deployment_busy:self.cancel_deployment();return
        super().stop_live()

    def release_camera(self):
        if self.deployment_busy:self.cancel_deployment()
        super().release_camera()

    def display_request(self,request):
        if request['kind']=='deployment_frame':
            try:
                if not self.deployment_busy or not self.can_project():raise RuntimeError('当前不允许静态投图')
                if self.deployment_pending:raise RuntimeError('已有静态投图等待确认')
                self.override=True;self.show_outputs(request['dmd'],request['slm'])
                self.deployment_pending=request
            except Exception as error:
                request['error']=str(error);request['done'].set();self.black_outputs()
            return
        if request['kind']=='deployment_park':
            self.finish_deployment_display('测试结束')
            self._internal_park=True;self.armed.setChecked(False);self._internal_park=False
            self.override=False;self.black_outputs();self.dirty=True
            request['done'].set();return
        super().display_request(request)

    def finish_deployment_display(self,error=None):
        request=self.deployment_pending
        if request is None:return
        if error is None:
            if self.last_verified_revision!=self.revision:return
            request['video_verification']=dict(actual_framebuffer_checked=True,revision=self.revision,
                verified_unix_ns=time.time_ns(),hardware_pattern_association_verified=False)
        else:request['error']=str(error)
        self.deployment_pending=None;request['done'].set()

    def deployment_finished(self,report):
        self.deployment_busy=False;self.override=False;self.tabs.setTabEnabled(0,True)
        self._internal_park=True;self.armed.setChecked(False);self._internal_park=False
        self.black_outputs();self.dirty=True;self.deployment_page.finished(report);self.refresh()

    def update_comparison(self,now):
        if self.deployment_busy or (hasattr(self,'tabs') and self.tabs.currentIndex()==1):return
        super().update_comparison(now)

    def tick(self):
        super().tick();self.finish_deployment_display()
        if self.deployment_busy and self.fingerprint()!=self._deployment_fingerprint:
            self.cancel_deployment()
        if hasattr(self,'deployment_page') and self.last_packet and self.running:
            self.deployment_page.fresh_view(self.last_packet)
        if self.hardware and self._deployment_heartbeat!=self.last_heartbeat:
            self._deployment_heartbeat=self.last_heartbeat;path=self.output/'heartbeat.json'
            if path.exists():
                h=json.loads(path.read_text());h.update(trigger_revision=6,deployment_ui_revision=20,
                    deployment_busy=self.deployment_busy,deployment_page_frozen=self.deployment_frozen,
                    closed_loop_enabled=False)
                tmp=self.output/'heartbeat_deployment.tmp';write_json(tmp,h);tmp.replace(path)

    def update_output_state(self):
        if self.deployment_busy:
            verified=self.last_verified_revision==self.revision
            levels=[]
            for role in ('dmd','slm'):
                levels.append(role.upper()+('非零' if self.expected.get(role,np.zeros(1)).any() else '全0'))
            self.output_state.setText('静态部署页接管 · '+('/'.join(levels))+
                (' · 实际读回一致' if verified else ' · 等待实际读回'))
            return
        super().update_output_state()

    def refresh(self):
        super().refresh()
        if hasattr(self,'deployment_page'):
            self.deployment_page.update_buttons()
            if self.deployment_busy:
                self.recovery_status.setText('离线部署静态测试运行中：对齐控件锁定；Esc/取消随时全0，不运行控制器。')

    def closeEvent(self,event):
        if self.deployment_busy:self.cancel_deployment();event.ignore();return
        super().closeEvent(event)

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
        self.starting_registration = self.registration or self.pending_registration
        self.record(dict(state='manual_line0_deployment_start', projection_requested=bool(project),
            requested_roi=dict(self.actual_roi), requested_exposure_us=self.exposure.value()))
        self.black_outputs(); self.running = True; self.live_ready = False
        self.override = False; self.roi_busy = False; self.dirty = True
        self.last_sequence = 0; self.last_packet = None
        self.view.setText('正在检查Line0与3张零输入帧（不是物理遮光暗场），未投图…')
        folder = self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.worker = DeploymentWorker(self.baseline, folder, self.exposure.value(), roi=dict(self.actual_roi))
        self.history.append(self.worker)
        for signal, callback in (('ready', self.camera_ready), ('configured', self.camera_configured),
                ('roi_found', self.roi_found), ('zero_reference', self.zero_reference_captured),
                ('spatial_result', self.spatial_received), ('display', self.display_request),
                ('message', self.note), ('black', self.black_outputs), ('result', self.camera_finished),
                ('finished', self.refresh)):
            getattr(self.worker, signal).connect(callback)
        self.worker.static_progress.connect(self.deployment_page.on_progress)
        self.worker.static_result.connect(self.deployment_finished)
        self.worker.start(); self.refresh()


def main():
    import cv2
    cv2.setNumThreads(2)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware',action='store_true');p.add_argument('--light-blocked-confirmed',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--restart-state',type=Path)
    p.add_argument('--recovery-record',type=Path);p.add_argument('--deployment-catalog',type=Path)
    p.add_argument('--smoke-test',action='store_true');a=p.parse_args()
    if a.hardware and (not a.light_blocked_confirmed or a.smoke_test):p.error('真实新建输出窗须当次物理遮光确认')
    snapshot=None if a.restart_state is None else json.loads(a.restart_state.read_text())
    if snapshot is not None:
        (validate_retained if snapshot.get('schema')=='v20-retained-line0-v1' else validate_legacy_controls)(snapshot)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling,True)
    app=QtWidgets.QApplication(['ballplate-realtime-manual'])
    if a.hardware and app.platformName()!='xcb':raise RuntimeError('真实器件需要本地X11')
    output=a.output or PROJECT/'outputs/live_calibration'/('DEPLOYMENT_V20_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    w=DeploymentWindow(output,a.hardware,snapshot,
        None if a.recovery_record is None else json.loads(a.recovery_record.read_text()),a.deployment_catalog)
    if a.smoke_test:
        assert not w.windows and w.worker is None and not output.exists()
        w.tabs.setCurrentIndex(1);QtCore.QTimer.singleShot(250,app.quit)
    app.exec_()


if __name__=='__main__':main()
