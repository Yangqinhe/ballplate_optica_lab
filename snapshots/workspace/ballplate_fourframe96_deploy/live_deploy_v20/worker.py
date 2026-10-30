"""Static stages run INSIDE the existing Line0 camera owner thread."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from PyQt5 import QtCore
from .contract import digest
from live_manual_v19_trigger_r5.worker import ManualPreviewWorker
from realtime_model import LivePattern,patterns
from realtime_worker import write_json
from .protocol import STAGES,features,fit_zero_gain,analyze_case


class StaticCancelled(InterruptedError):pass


class DeploymentWorker(ManualPreviewWorker):
    static_progress=QtCore.pyqtSignal(object)
    static_result=QtCore.pyqtSignal(object)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.static_request=None;self.static_busy=False;self.static_cancel=threading.Event()

    def request_static(self,bundle,case_ids,output):
        plan=bundle.plan(case_ids)
        with self.command_lock:
            if (self.static_request or self.static_busy or self.configuration or self.reference_requested
                    or self.spatial_requested or self.auto_requested.is_set()):
                raise RuntimeError('已有相机任务，不能同时开始静态测试')
            if not self.camera or not self.camera.acquiring or self.startup_dark_required:
                raise RuntimeError('先启动唯一CCD并等待完整帧')
            self.static_cancel.clear();self.static_request=(bundle,plan,Path(output))

    def cancel_static(self):self.static_cancel.set()

    def next_frame(self):
        pending=None
        if not self.static_busy:
            with self.command_lock:
                if self.static_request:
                    pending=self.static_request;self.static_request=None
        if pending:
            self.static_busy=True
            try:self.collect_static(*pending)
            finally:self.static_busy=False
        return super().next_frame()

    def collect_static(self,bundle,plan,folder):
        folder.mkdir(parents=True,exist_ok=False)
        plan.update(camera_active=dict(self.camera.active),camera_owner_session=str(self.directory.resolve()),
            bound_analysis_sources={str(Path(__file__).resolve()):digest(__file__)})
        write_json(folder/'PROSPECTIVE_PROTOCOL.json',plan)
        np.save(folder/'FIXED_NATIVE8_WEIGHTS.npy',bundle.weights,allow_pickle=False)
        report=dict(completed=False,cancelled=False,error=None,cases=[],folder=str(folder.resolve()),
            no_closed_loop=True,old_actor_NOT_loaded=True,release_approved=False)
        start=time.monotonic();deadline=start+plan['maximum_seconds'];fatal=None
        initial_count=self.count;initial_bad=len(self.bad);initial_writes=len(self.camera.write_trace)
        gain=None

        def check():
            self.check_stop()
            if self.static_cancel.is_set():raise StaticCancelled('操作者取消；已请求全0')
            if time.monotonic()>deadline:raise RuntimeError('本次静态测试达到事前时间上限')

        def active_check():
            self.camera.revalidate(self.exposure,self.requested_roi)
            if self.camera.active!=bundle.contract['expected_active']:
                raise RuntimeError('相机参数不同于冻结参考，停止测试')

        def save():
            write_json(folder/'capture.json',report)

        try:
            check();bundle.verify();active_check()
            for case_id in plan['case_ids']:
                case=bundle.cases[case_id];case_folder=folder/f'input_{case_id:02d}';case_folder.mkdir()
                entry=dict(id=case_id,stages=[],analysis=None)
                report['cases'].append(entry);stacks={}
                for name in STAGES:
                    check();bundle.verify();active_check()
                    stage=case_folder/name;stage.mkdir()
                    cfg=replace(LivePattern(**bundle.contract['settings']),dmd='file',
                        slm='file' if name.startswith('learned') else 'zero',full_panel=False)
                    dmd,slm=patterns(cfg,case['dmd_array'],bundle.phase)
                    if name.startswith('background'):
                        dmd=np.zeros_like(dmd);slm=np.zeros_like(slm)
                    self.static_progress.emit(dict(kind='stage_start',case_id=case_id,stage=name,
                        completed_frames=0,gain=gain))
                    try:request=self.show('deployment_frame',dmd,slm)
                    except RuntimeError:
                        check();raise
                    checked=request.get('video_verification')
                    if not checked or not checked.get('actual_framebuffer_checked'):
                        raise RuntimeError('没有原输出窗口的实际读回核验')
                    verified_ns=checked['verified_unix_ns']
                    row=dict(name=name,display_verification=checked,frames=[],settle_frames=0)
                    entry['stages'].append(row);save()
                    end=time.monotonic()+plan['settle_seconds']
                    while time.monotonic()<end:
                        check();self.capture_one();row['settle_frames']+=1
                    images=[]
                    for index in range(plan['frames_per_stage']):
                        check();active_check()
                        image,meta=self.capture_one()
                        path=stage/f'CCD_raw_{index+1:03d}.npy'
                        np.save(path,image,allow_pickle=False)
                        record=dict(frame=meta,camera_active=dict(self.camera.active),
                            raw_sha256=digest(path),max_ADU=int(image.max()),mean_ADU=float(image.mean()),
                            actual_display_verified=True,frame_pattern_association_verified=False,
                            software_settle_NOT_hardware_pattern_synchronization=True)
                        write_json(stage/f'CCD_raw_{index+1:03d}.json',record)
                        row['frames'].append(record);save()
                        if (image.max()>=240 or meta['host_get_started_unix_ns']<
                                verified_ns+int(plan['settle_seconds']*1e9)):
                            raise RuntimeError('静态样本峰值/新鲜度未通过；已保留原图，不补帧')
                        images.append(image)
                        self.static_progress.emit(dict(kind='sample_saved',case_id=case_id,stage=name,
                            completed_frames=len(images),total_frames=plan['frames_per_stage']))
                    stacks[name]=np.stack(images)
                    if case_id==8 and name=='zero_start':
                        gain=fit_zero_gain(stacks[name].mean(0)-stacks['background_start'].mean(0),
                            case['predictions']['zero'])
                        report['frozen_one_global_gain']=gain
                        write_json(folder/'FROZEN_ZERO_ONLY_GAIN.json',dict(gain=gain,
                            frozen_before_any_learned_projection=True,calibration_input=8,
                            source_states=['background_start','zero_start'],no_region_or_learned_fit=True,
                            wall_unix_ns=time.time_ns()))
                    if not name.startswith('background'):
                        stat=features(stacks[name],stacks['background_start'],bundle.weights)
                        self.static_progress.emit(dict(kind='stage_complete',case_id=case_id,stage=name,
                            stats=stat,gain=gain))
                result=analyze_case(stacks,case['predictions'],bundle.weights,gain,plan['rules'])
                entry['analysis']=result;write_json(case_folder/'ANALYSIS.json',result)
                self.static_progress.emit(dict(kind='case_complete',case_id=case_id,analysis=result,
                    gain=gain,
                    observed_signal=stacks['learned_1'].mean(0)-stacks['background_start'].mean(0),
                    predicted_signal=case['predictions']['learned']*gain))
                save()
                if not result['static_criteria_passed'] and case_id!=plan['case_ids'][-1]:
                    report['stopped_on_static_criterion_failure']=True
                    report['not_executed_case_ids']=plan['case_ids'][len(report['cases']):]
                    break
            report['completed']=len(report['cases'])==len(plan['case_ids'])
            report['static_criteria_passed']=bool(report['completed'] and all(
                c['analysis'] and c['analysis']['static_criteria_passed'] for c in report['cases']))
        except InterruptedError as error:
            report.update(cancelled=True,error=str(error))
        except BaseException as error:
            report.update(error=str(error),traceback=traceback.format_exc());fatal=error
        finally:
            try:
                if self.stopped.is_set():self.black.emit()
                else:self.show('deployment_park')
            except Exception as error:
                self.black.emit();report['park_error']=str(error);fatal=fatal or error
            report.update(valid_stream_frames=self.count-initial_count,
                rejected_incomplete_frames=self.bad[initial_bad:],wall_seconds=time.monotonic()-start,
                same_SDK_owner=True,camera_still_acquiring=bool(self.camera.acquiring),
                camera_node_writes_during_test=len(self.camera.write_trace)-initial_writes,
                camera_active_at_end=dict(self.camera.active),camera_final_rollback_not_yet_requested=True)
            save();self.static_result.emit(report)
        if fatal is not None:raise RuntimeError('静态测试失败：'+str(fatal)) from fatal
