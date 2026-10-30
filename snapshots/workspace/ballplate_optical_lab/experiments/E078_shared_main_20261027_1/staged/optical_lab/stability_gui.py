"""Fresh-owner stability page; original camera-thread sampler, no new SDK.

Imported by an explicit new launcher only. Existing loaded GUI/source files
are never edited or injected. Test results are research observations, not a
new deployment qualification. RAW archive/analysis stays off the Qt thread.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from . import stability
from .state_sampling_gui import (SamplingController, _capture_adapter, _pin,
                                 _value, _write_new, BUSY)
from .state_sampling import DenseBurstCore
from .region_overlay import layout_from_weights, layout_for_roi, weights_for_layout


def require(ok, text):
    if not ok:
        raise ValueError(text)


def complete_hold_without_zero(status, *, active, hold_mode, more_visits):
    """Only a healthy within-run hold may defer zero between independent means."""
    return bool(active and hold_mode and more_visits and status is not None
                and status.get('state') in ('WRITING', 'COMPLETE')
                and status.get('error') is None and status.get('writer_error') is None
                and status.get('captured_frames') == status.get('valid_frames') == status.get('requested_frames'))


class Workflow:
    """GUI-thread coordinator. The capture wrapper sees only current_core/leases."""
    def __init__(self, window, template_ref, output_root, mask_ref, mask_roi):
        from .state_sampling_gui import LeaseBox
        self.window = window; self.template_ref = deepcopy(template_ref)
        self.template = json.loads(_pin(template_ref).read_text())
        self.output_root = Path(output_root); self.mask_ref = deepcopy(mask_ref)
        self.mask_roi = dict(mask_roi); self.weights = np.load(_pin(mask_ref), allow_pickle=False)
        stability.validate_weights(self.weights, (mask_roi['Height'], mask_roi['Width']))
        self.base_layout = layout_from_weights(self.weights, mask_roi)
        self.assets = []
        for item in self.template['allowed_dmd']:
            ref = {k: item[k] for k in ('path', 'sha256')}
            image = np.load(_pin(ref), allow_pickle=False)
            self.assets.append(dict(input_id=item['input_id'], ref=ref, image=image))
        self.active = False; self.stage = 'IDLE'; self.sampler = None
        self.leases = LeaseBox(); self._lock = threading.Lock()
        self.widgets = {}; self._locked = []; self.future = None; self.result = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stability-file-only')
        self.error = None; self.folder = None; self.parameters_pending = False
        self.expected_active = None; self.visits = []; self.zero_deadline = None
        self.head_record = None

    def current_core(self):
        with self._lock:
            sampler = self.sampler
        return sampler.current_core() if sampler is not None else None

    def profile(self):
        w = self.window; packet = getattr(w, 'last_packet', None)
        require(w.hardware and w.running and w.live_ready and w.worker is not None and not w.fault,
                '请先在原窗口启动唯一CCD，等待正常新帧')
        require(packet is not None and 0 <= time.monotonic()-packet['received_monotonic'] < 1., 'CCD没有新鲜图像')
        require(all(hasattr(w, k) and getattr(w, k) is False for k in BUSY), '其他流程／曝光事务正在运行')
        require(not getattr(w.worker, 'manual_gain_state', {}).get('busy'), '相机Gain事务未完成')
        return deepcopy(packet['active'])

    def lock_controls(self):
        w = self.window; page = self.widgets['page']
        controls = [w.tabs.widget(i) for i in range(w.tabs.count()) if w.tabs.widget(i) is not page]
        controls += [self.widgets['config'], self.widgets['start'], self.widgets['permission']]
        self._locked = [(c, c.isEnabled()) for c in controls]
        for c, _ in self._locked: c.setEnabled(False)
        self.widgets['cancel'].setEnabled(True)

    def unlock_controls(self):
        for control, enabled in self._locked: control.setEnabled(enabled)
        self._locked = []; self.widgets['cancel'].setEnabled(False)

    def main_parameters(self):
        """Read the ONE main-page configuration; never write camera or preview."""
        w = self.window
        controls = w.manual_display.controls
        display = controls.parameters(w.enhance.isChecked())
        fixed = display['mode']=='fixed'
        return dict(camera_actual=deepcopy((getattr(w,'last_packet',None) or {}).get('active',{})),
            roi=deepcopy(w.actual_roi), slm=_value(w.setting()),
            dmd_side=getattr(w,'_dmd_side',600), display=display,
            comparison_mapping=dict(software_gain=display['gain'] if fixed else 1.,
                black_ADU=display['black'] if fixed else 0.,gamma=display['gamma'] if fixed else 1.),
            preview_filter_NOT_used=True, primary_readout='RAW mean -> 8 fixed regions')

    def settings(self):
        q = self.widgets
        params = self.main_parameters()['comparison_mapping']
        mode = q['mode'].currentData()
        return stability.Settings(frames=q['frames'].value(), repeats=q['repeats'].value(),
            switch_inputs=mode!='hold', settle_seconds=q['settle'].value(),
            interval_seconds=q['interval'].value(), **params).validate()

    def selection(self):
        q = self.widgets; mode=q['mode'].currentData()
        if mode=='sweep':
            metadata=self.template.get('input_catalog',{}).get('metadata',{}).get('inputs',[])
            names={r['input_id'] for r in metadata
                   if r.get('input_kind')!='calibration_probe_not_MuJoCo_state'
                   and not r.get('state_fields_are_loader_placeholders')}
            selected={a['input_id']:a for a in self.assets if a['input_id'] in names}
            require(2<=len(selected)<=16,'清单轮测需要2–16种真实位置/速度输入；不包含参考标定图')
        else:
            selected={'A':self.assets[q['input_a'].currentIndex()]}
            if mode=='pair':selected['B']=self.assets[q['input_b'].currentIndex()]
        hashes=[hashlib.sha256(a['image'].tobytes()).hexdigest() for a in selected.values()]
        require(len(set(hashes))==len(hashes),'所选激励图像相同，不能作为不同状态比较')
        return selected

    def choose_head(self):
        from PyQt5 import QtWidgets
        if self.active:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self.window,'选择18参数RAW线性头（仅诊断）','','JSON (*.json)')
        if not path:return
        try:
            self.head_record=stability.load_head(path)
            self.widgets['head_path'].setText(path+' · SHA '+self.head_record['reference']['sha256'][:12])
            self.widgets['head_enabled'].setChecked(True)
        except Exception as error:
            self.head_record=None;self.widgets['head_enabled'].setChecked(False)
            self.widgets['head_path'].setText('未加载：'+str(error))

    def zero(self):
        self.window.armed.setChecked(False)
        self.window.black_outputs()

    def zero_verified(self):
        w = self.window; arrays = getattr(w, 'last_arrays', {})
        return bool(not w.armed.isChecked() and arrays and set(arrays) >= {'dmd', 'slm'}
                    and all(not arrays[k].any() for k in ('dmd', 'slm'))
                    and w.last_verified_revision == w.revision)

    def start(self):
        created_folder = None
        try:
            require(not self.active and self.current_core() is None and self.future is None,
                    '上一轮未结束／尚有原始数据待保存')
            require(self.widgets['permission'].isChecked(), '请先确认原ND／低功率与光束防护条件')
            original = self.window.state_sampling_controller
            require(original.current_core() is None, '原固定输入RAW页正在采样')
            selected = self.selection()
            settings = replace(self.settings(), input_ids=tuple(selected))
            active = self.profile(); main = self.main_parameters()
            gain = float(active['Gain']); notices = []
            if main['display']['mode']!='fixed':
                notices.append('主界面非固定显示映射；处理后对照保持原始ADU，不做逐图自动归一化')
            # Freeze the actually generated SLM panel, independent of its GUI mode.
            from realtime_model import patterns
            cache = self.window.dmd_file
            _, phase = patterns(self.window.setting(), np.zeros((600,600),np.uint8) if cache is None else cache, self.window.slm_file)
            require(phase.dtype == np.uint8 and phase.shape == (1080, 1920), '当前SLM图案无效')
            self.original_input=dict(mode=_value(self.window.setting())['dmd'],
                                     cache=None if cache is None else cache.copy())
            self.zero()
            self.window.dmd_mode.setCurrentIndex(self.window.dmd_mode.findData('file'))
            s = _value(self.window.setting())
            layout = layout_for_roi(self.base_layout, self.window.actual_roi)
            self.weights = weights_for_layout(layout)
            side = getattr(self.window, '_dmd_side', 600)
            if layout['geometry_changed'] or side != 600:
                notices.append('使用当前ROI和DMD缩放；旧控制头资格不继承')
            first = next(iter(selected))
            self.window.dmd_file = selected[first]['image'].copy()
            self.run_head=None
            if 'head_enabled' in self.widgets and self.widgets['head_enabled'].isChecked():
                try:
                    require(self.head_record is not None, '未加载控制头')
                    require(stability.file_ref(self.head_record['reference']['path'])==self.head_record['reference'], '控制头文件已变化')
                    self.run_head=deepcopy(self.head_record)
                except (ValueError, OSError) as error:
                    notices.append(str(error)+'，跳过控制头诊断，仍测八区稳定性')
            catalog=self.template.get('input_catalog',{}).get('metadata',{})
            catalog_rows={v['input_id']:v for v in catalog.get('inputs',[])}
            self.run_labels={k:deepcopy(catalog_rows.get(v['input_id'],{})) for k,v in selected.items()}
            self.run_input_geometry={k:deepcopy(catalog_rows.get(v['input_id'],{}).get('ball_size',{'kind':'unknown'}))
                                     for k,v in selected.items()}
            controls=getattr(self.window,'ball_size_controls',None)
            if controls is not None:
                for key in selected:
                    geometry=self.run_input_geometry[key]
                    if geometry.get('kind')=='four_ball_history' and geometry['radius_raw_px']!=controls['radius'].value():
                        notices.append(f'输入{key}使用已应用R={geometry["radius_raw_px"]}，未应用的球尺寸不参与本次测试')
            self.zero()  # Explicit Start authorizes this bounded test, not projection on startup.
            self.output_root.mkdir(parents=True, exist_ok=True)
            folder = self.output_root/('TEST_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1_000_000))
            folder.mkdir(); self.folder = folder; created_folder = folder
            with (folder/'SLM.npy').open('xb') as f: np.save(f, phase, allow_pickle=False)
            with (folder/'REGIONS.npy').open('xb') as f: np.save(f, self.weights, allow_pickle=False)
            _write_new(folder/'REGIONS.json', layout)
            run_mask = stability.file_ref(folder/'REGIONS.npy')
            contract = deepcopy(self.template)
            contract['dmd_presentation'] = dict(native=[1920,1080], side=side, scale=side/600.)
            contract['brightness_policy'] = 'warning_only'
            contract['expected_active'] = active
            contract['fixed_context'] = dict(settings=s, roi=deepcopy(self.window.actual_roi),
                geometry_frozen=bool(self.window.registration), optical_reference=_value(self.window.optical_config()))
            contract['fixed_slm'] = stability.file_ref(folder/'SLM.npy')
            contract['allowed_dmd'] = [dict(**asset['ref'], input_id=k) for k,asset in selected.items()]
            # Immutable per-test contract, not a change to the deployed policy contract.
            contract['research_test'] = dict(settings=asdict(settings), mask=run_mask, layout=layout,
                shared_main=main, camera_requested=dict(exposure_us=float(self.window.worker.exposure), gain_db=gain),
                camera_actual=active, notices=notices, control_qualified=False, parent_contract=self.template_ref)
            contract['source_sha256'][self.template_ref['path']] = self.template_ref['sha256']
            if self.run_head:
                contract['source_sha256'][self.run_head['reference']['path']]=self.run_head['reference']['sha256']
            contract['research_test']['fixed_head_diagnostic']=self.run_head
            contract['research_test']['target_labels_score_only']=self.run_labels
            contract['research_test']['input_geometry']=self.run_input_geometry
            for module in (Path(__file__).resolve(), Path(stability.__file__).resolve()):
                require(contract['source_sha256'].get(str(module)) == stability.file_ref(module)['sha256'], '新页源码未绑定到启动配置')
            _write_new(folder/'CONTRACT.json', contract)
            self.run_settings = settings; self.expected_active = active; self.selected = selected
            self.frozen_context = contract['fixed_context']
            self.run_context=dict(mask_sha256=run_mask['sha256'],readout_mask_ref=run_mask,
                                  slm_sha256=contract['fixed_slm']['sha256'],camera_actual=active,
                                  input_geometry=self.run_input_geometry, dmd_scale=side/600., region_layout=layout, shared_main=main)
            self.notices = notices
            self.queue = stability.schedule(settings); self.index = 0; self.visits = []; self.error = None
            self.result = None; self.future = None; self.deadline = time.monotonic()+settings.maximum_seconds
            if 'table' in self.widgets:
                for key in ('table','gates','noise_table'): self.widgets[key].clearContents()
                self.widgets['plot'].clear(); self.widgets['analysis_plot'].clear()
                self.widgets['summary'].setText('新一轮尚未完成；不沿用上一轮判定。')
                self.widgets['folder'].setText(str(folder)); self.widgets['open'].setEnabled(False)
            with self._lock: self.sampler = None
            self.not_before = 0.; self.stage = 'INITIAL_ZERO'; self.active = True
            self.lock_controls()
            sampler = SamplingController(self.window, stability.file_ref(folder/'CONTRACT.json'), folder/'RAW', DenseBurstCore)
            # The workflow owns all UI locking and input transitions. The original
            # sampler still owns its exact per-frame validation/lease/archive logic.
            sampler._lock_controls = lambda: None
            sampler._unlock_controls = lambda: None
            sampler._requests = lambda: None  # no second external request channel for this test
            original_zero = sampler._zero
            def between_queries_zero():
                core = sampler.current_core(); status = None if core is None else core.snapshot()
                if not complete_hold_without_zero(status, active=self.active,
                        hold_mode=not (self.run_settings.switch_inputs), more_visits=self.index+1 < len(self.queue)):
                    original_zero()
            sampler._zero = between_queries_zero
            with self._lock: self.sampler = sampler
            self.leases = sampler.leases
            _write_new(folder/'PLAN.json', dict(settings=asdict(settings), sequence=self.queue,
                input_geometry=self.run_input_geometry,
                contract=stability.file_ref(folder/'CONTRACT.json'), input_sources={k:v['ref'] for k,v in selected.items()},
                same_original_camera_owner=True, reset_each_query=True, pipeline='RAW mean -> black/gamma -> native8',
                actual_framebuffer_ACK_required=True, physically_trigger_synchronized=False,
                hold_mode_does_not_blank_between_queries=not (settings.switch_inputs), control_qualified=False,
                accepted_RAW_budget=sum(v.get('frames',settings.frames) for v in self.queue),
                reference_frames_used=0))
            self.widgets['status'].setText('测试已冻结当前参数；等待全0核验后加载输入。'+('；'.join(notices)))
            self.widgets['summary'].setText('本轮提示：'+('；'.join(notices) or '无参数差异'))
        except Exception as error:
            if self.active: self.fail(str(error))
            else:
                if getattr(self,'original_input',None) is not None and not self.window.armed.isChecked():
                    self.restore_input()
                if created_folder is not None:
                    _write_new(created_folder/'START_FAILED.json', dict(completed=False,
                        error=str(error), no_test_projection_started=True, control_qualified=False))
                self.widgets['status'].setText('未开始：'+str(error))

    def restore_input(self):
        saved=getattr(self,'original_input',None)
        if saved is None:return
        require(not self.window.armed.isChecked(),'恢复主界面输入时必须保持全0')
        self.window.dmd_file=saved['cache']
        self.window.dmd_mode.setCurrentIndex(self.window.dmd_mode.findData(saved['mode']))
        self.window.changed();self.original_input=None

    def load_next(self):
        item = self.queue[self.index]
        self.window.dmd_file = self.selected[item['input_id']]['image'].copy()
        self.window.changed()
        self.window.armed.setChecked(True)
        self.window.render_patterns()
        self.stage = 'SETTLING'; self.not_before = time.monotonic()
        self.window.record(dict(state='stability_input_requested', test=str(self.folder),
            index=self.index, item=item, dmd=self.selected[item['input_id']]['ref'],
            complete_settings=_value(self.window.setting()), no_control=True))

    def fail(self, reason):
        if self.error is None: self.error = reason
        core = self.current_core()
        if core is not None: core.fail(reason)
        self.zero(); self.stage = 'STOPPING'; self.zero_deadline = time.monotonic()+10
        self.widgets['status'].setText('已请求停止／全0，保留失败数据：'+reason)

    def cancel(self):
        if self.active:
            if self.stage == 'EXPORT':
                self.zero(); self.widgets['status'].setText('采样已结束且两路全0，等待结果文件封存。')
            else: self.fail('操作者取消')

    def sample_complete(self, ref, item):
        local=replace(self.run_settings,frames=item.get('frames',self.run_settings.frames))
        row, mean, processed = stability.analyze_archive(ref, self.weights, local, item, self.expected_active)
        out = self.folder/('MEAN_%02d_%s'%(self.index, item['input_id'])); out.mkdir()
        with (out/'RAW_mean.npy').open('xb') as f: np.save(f, mean, allow_pickle=False)
        with (out/'processed.npy').open('xb') as f: np.save(f, processed, allow_pickle=False)
        row['mean_file']=stability.file_ref(out/'RAW_mean.npy')
        row['brightness_maps'] = stability.export_brightness_maps(ref, out)
        _write_new(out/'READOUT.json', row)
        return row

    def analyze_and_export(self):
        result=stability.assess(self.visits,self.run_settings,self.run_head,self.run_labels,self.run_context)
        result.update(contract=stability.file_ref(self.folder/'CONTRACT.json'),final_actual_zero_verified=True,
            same_camera_owner=True,all_visits_completed=True, notices=self.notices,
            dmd_scale=self.run_context['dmd_scale'], region_layout=self.run_context['region_layout'])
        stability.export(result,self.folder/'RESULTS')
        return result

    def poll(self):
        try:
            self.refresh_actual()
            if not self.active: return
            if self.stage not in ('STOPPING', 'FINAL_ZERO', 'EXPORT'):
                require(time.monotonic() < self.deadline, '本轮达到300秒有界时限')
                require(self.profile() == self.expected_active, '相机实际参数改变，本轮终止且不混算')
                require(_value(self.window.setting()) == self.frozen_context['settings']
                        and self.window.actual_roi == self.frozen_context['roi']
                        and getattr(self.window, '_dmd_side', 600)/600. == self.run_context['dmd_scale'],
                        '测试中的SLM设置、ROI或DMD缩放改变；停止本轮，保留已有数据')
            if self.sampler is not None:
                status = self.sampler.poll()
                require(status is not None, '原采样器状态异常：'+str(self.sampler.last_error))
            else: status = None
            if self.stage == 'STOPPING':
                if self.future is not None and not self.future.done(): return
                if self.current_core() is not None: return  # retain unsealed RAW on writer failure
                if self.future is not None:
                    try: self.future.result()
                    except Exception: pass
                    self.future = None
                if self.zero_verified(): self.finish_failed()
                elif time.monotonic() > self.zero_deadline:
                    self.widgets['status'].setText('已停止测试，但全0未核验；请检查原窗口故障，不能宣称安全完成。')
                return
            if self.stage == 'INITIAL_ZERO':
                if self.zero_verified(): self.load_next()
            elif self.stage == 'SETTLING':
                require(self.window.armed.isChecked(), '投图已被取消')
                self.widgets['status'].setText('等待第%d/%d组图案稳定：%s'%(
                    self.index+1, len(self.queue), self.sampler.last_error or '输出核验／等待采样'))
                if (self.sampler.ack_ns is not None and time.time_ns() >= self.sampler.ack_ns+int(self.run_settings.settle_seconds*1e9)
                        and time.monotonic() >= self.not_before):
                    self.sampler.start(self.queue[self.index].get('frames',self.run_settings.frames), 'VISIT_%02d'%self.index, duration_seconds=60)
                    self.stage = 'SAMPLING'
            elif self.stage == 'SAMPLING':
                s = status['core']
                require(s is not None and s['error'] is None and s['writer_error'] is None,
                        '采样失败：'+str(None if s is None else s['error']))
                message = '第%d/%d组 · %s 第%d次 · %s · %d/%d RAW'%(
                    self.index+1, len(self.queue), self.queue[self.index]['input_id'], self.queue[self.index]['visit'],
                    s['state'], s['captured_frames'], self.queue[self.index].get('frames',self.run_settings.frames))
                if s.get('brightness_warning_frames',0):
                    message += ' · 亮度告警，继续采集：%d帧≥240，255像素次数%d'%(
                        s['brightness_warning_frames'],s.get('clipped_pixel_observations',0))
                self.widgets['status'].setText(message)
                if s.get('archive_result') is not None:
                    require(s['state'] == 'COMPLETE', '封存未通过完整性检查')
                    self.future = self.executor.submit(self.sample_complete, deepcopy(s['archive_result']), deepcopy(self.queue[self.index]))
                    self.stage = 'ANALYZING'
            elif self.stage == 'ANALYZING' and self.future.done():
                row = self.future.result(); self.future = None; self.visits.append(row); self.index += 1
                if self.index == len(self.queue):
                    self.zero(); self.stage = 'FINAL_ZERO'; self.zero_deadline = time.monotonic()+10
                elif self.run_settings.switch_inputs:
                    self.stage = 'BETWEEN'; self.not_before = time.monotonic()+self.run_settings.interval_seconds
                else:
                    require(self.window.armed.isChecked(), '保持图案期间投图被取消')
                    self.stage = 'SETTLING'; self.not_before = time.monotonic()+self.run_settings.interval_seconds
            elif self.stage == 'BETWEEN':
                if self.zero_verified() and time.monotonic() >= self.not_before: self.load_next()
            elif self.stage == 'FINAL_ZERO':
                if self.zero_verified():
                    self.future = self.executor.submit(self.analyze_and_export)
                    self.stage = 'EXPORT'
                else: require(time.monotonic() < self.zero_deadline, '结束全0核验超时')
            elif self.stage == 'EXPORT' and self.future.done():
                self.result = self.future.result(); self.future = None
                self.widgets['status'].setText('完成。已核验两路全0；各输入分别统计，RAW与CSV/PNG/PDF已保存。')
                self.restore_input()
                self.populate_results(); self.active = False; self.stage = 'COMPLETE'; self.unlock_controls()
        except Exception as error:
            if self.active:
                if self.stage == 'STOPPING': self.widgets['status'].setText('停止处理中：'+str(error))
                else: self.fail(str(error))
            else: self.widgets['actual'].setText('等待正常CCD：'+str(error))

    def finish_failed(self):
        _write_new(self.folder/'FAILED.json', dict(completed=False, error=self.error,
            completed_visits=self.visits, remaining_sequence=self.queue[self.index:],
            final_actual_zero_verified=True, failed_prefix_not_pooled=True, control_qualified=False))
        self.restore_input()
        self.active = False; self.stage = 'FAILED'; self.unlock_controls()
        self.widgets['status'].setText('测试未完成，不生成正式波动判定；失败前缀已保留，两路全0。原因：'+self.error)


    def refresh_actual(self):
        try:
            main=self.main_parameters(); a=main['camera_actual']; d=main['display']
            actual='曝光%s µs / Gain%s dB；ROI%s；DMD %d²；SLM %s (%s,%s)；显示×%g / 黑%g / γ%g。'%(
                a.get('ExposureTime','—'),a.get('Gain','—'),main['roi'],main['dmd_side'],
                main['slm'].get('slm'),main['slm'].get('x'),main['slm'].get('y'),d['gain'],d['black'],d['gamma'])
            if self.widgets['actual'].text()!=actual:self.widgets['actual'].setText(actual)
        except (AttributeError,KeyError):pass

    def populate_results(self):
        q=self.widgets; combo=q['result_input']; previous=combo.blockSignals(True)
        combo.clear();combo.addItems(list(self.result['inputs']));combo.blockSignals(previous)
        self.show_result()

    def show_result(self):
        from PyQt5 import QtGui, QtWidgets, QtCore
        if not self.result:return
        q=self.widgets; label=q['result_input'].currentText()
        if label not in self.result['inputs']:return
        domain=q['result_domain'].currentData();info=self.result['inputs'][label][domain]
        fmt=lambda v:'—' if v is None else f'{v:.4f}' if isinstance(v,(float,int)) else str(v)
        for r in info['regions']:
            bright=self.result['photometry'][label]['regions']
            values=[r['region']]+[r[k] for k in ('mean','sample_SD','CV_percent','peak_to_peak_percent','first_to_last_percent')]
            values+=([bright[r['region']]['max_ADU'],bright[r['region']]['clipped_percent']] if bright else [None,None])
            for col,value in enumerate(values):q['table'].setItem(r['region'],col,QtWidgets.QTableWidgetItem(fmt(value)))
        diag=self.result['diagnostics'][label]
        for r in diag['regions']:
            values=[r['region']]+[r[k] for k in ('within_RAW_CV_percent','across_mean_CV_percent',
                'empirical_SD_reduction','max_within_half_change_percent','fitted_change_percent')]
            for col,value in enumerate(values):q['noise_table'].setItem(r['region'],col,QtWidgets.QTableWidgetItem(fmt(value)))
        gates=self.result['readiness']['gates'];q['gates'].setRowCount(len(gates))
        for i,r in enumerate(gates):
            for col,key in enumerate(('item','status','value','criterion')):
                item=QtWidgets.QTableWidgetItem(fmt(r[key]))
                if key=='status':item.setForeground(QtGui.QColor({'PASS':'#16803a','FAIL':'#bd3326'}.get(r[key],'#8b6900')))
                q['gates'].setItem(i,col,item)
        q['gates'].resizeRowsToContents()
        q['summary'].setText(self.result['readiness']['recommendation']+
            f'\n输入 {label} / {domain}：RMS {fmt(info["relative_vector_RMS_percent"])}%；'
            f'最大两次差 {fmt(info["maximum_pair_L2_percent"])}%；最大区域CV {fmt(info["maximum_region_CV_percent"])}%。'
            f'\n实际回访跨度 {fmt(diag["observed_span_s"])} s；共封存 {self.result["accepted_raw_total"]} RAW。'
            +'\n提示：'+'；'.join(self.result.get('notices',[])))
        def picture(widget,name,height):
            widget.clear();path=self.folder/'RESULTS'/name
            if path.exists():widget.setPixmap(QtGui.QPixmap(str(path)).scaled(1100,height,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            else:widget.setText('无对应图表，不沿用旧图')
        picture(q['plot'],f'stability_{label}_{domain}.png',460)
        name=f'diagnostics_{label}.png' if q['analysis_chart'].currentData()=='noise' else 'separation.png'
        picture(q['analysis_plot'],name,500)
        q['folder'].setText(str(self.folder));q['open'].setEnabled(True)


def attach(window, template_ref, output_root, mask_ref, mask_roi):
    """Fresh startup only; one original camera owner, no reference workflow."""
    from PyQt5 import QtCore, QtWidgets, QtGui
    import live_loop_v23.direct_alignment_gui as gui
    require(window.worker is None and hasattr(window,'state_sampling_controller')
            and not hasattr(window,'stability_workflow'),'只能在新窗口启动时装配')
    flow=Workflow(window,template_ref,output_root,mask_ref,mask_roi)
    require(not getattr(gui.LoopWorker.capture_one,'_optical_stability_capture',False),'不可重复安装采集适配器')
    wrapped=_capture_adapter(gui.LoopWorker.capture_one,flow);wrapped._optical_stability_capture=True
    gui.LoopWorker.capture_one=wrapped
    previous=window.state_sampling_controller.start
    def guarded(*args,**kwargs):
        require(not flow.active,'稳定性测试正在运行');return previous(*args,**kwargs)
    window.state_sampling_controller.start=guarded
    page=QtWidgets.QWidget();outer=QtWidgets.QVBoxLayout(page);outer.setContentsMargins(0,0,0,0)
    scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);outer.addWidget(scroll)
    body=QtWidgets.QWidget();scroll.setWidget(body);layout=QtWidgets.QVBoxLayout(body)
    q=flow.widgets;q['page']=page
    header=QtWidgets.QLabel('继承主界面已应用的CCD / DMD / SLM / 球大小。每个有效帧独立平均N张RAW；不使用参考帧或预览EMA。')
    header.setWordWrap(True);layout.addWidget(header)
    q['actual']=QtWidgets.QLabel('主界面参数将在采样开始时冻结；本页不写相机配置。');q['actual'].setWordWrap(True);layout.addWidget(q['actual'])
    config=QtWidgets.QGroupBox('输入激励与采样');grid=QtWidgets.QGridLayout(config);q['config']=config;layout.addWidget(config)
    q['mode']=QtWidgets.QComboBox();q['mode'].setObjectName('stability_mode')
    for title,key in [('固定单图重复','hold'),('A/B交替回访','pair'),('位置/速度清单轮测','sweep')]:q['mode'].addItem(title,key)
    grid.addWidget(QtWidgets.QLabel('测试方式'),0,0);grid.addWidget(q['mode'],0,1,1,3)
    for name,col in [('input_a',0),('input_b',3)]:
        q[name]=QtWidgets.QComboBox();q[name].setObjectName('stability_'+name)
        for asset in flow.assets:q[name].addItem(asset['input_id'],asset['input_id'])
        grid.addWidget(QtWidgets.QLabel('输入A' if name=='input_a' else '输入B'),1,col);grid.addWidget(q[name],1,col+1,1,2)
    q['input_b'].setCurrentIndex(min(1,len(flow.assets)-1))
    def spin(name,row,col,title,low,high,value,integer=False):
        c=QtWidgets.QSpinBox() if integer else QtWidgets.QDoubleSpinBox()
        c.setRange(low,high);c.setValue(value);c.setKeyboardTracking(False);c.setObjectName('stability_'+name)
        q[name]=c;grid.addWidget(QtWidgets.QLabel(title),row,col);grid.addWidget(c,row,col+1)
    spin('frames',2,0,'每次平均N',2,120,50,True)
    spin('repeats',2,2,'每种输入重复',3,20,3,True)
    spin('settle',2,4,'换图稳定等待 s',1.5,15,1.5)
    spin('interval',3,0,'回访间隔 s',0,15,0)
    q['budget']=QtWidgets.QLabel();q['budget'].setWordWrap(True);grid.addWidget(q['budget'],4,0,1,6)
    def update_budget():
        mode=q['mode'].currentData();q['input_a'].setEnabled(mode!='sweep');q['input_b'].setEnabled(mode=='pair')
        try:
            count=len(flow.selection());n=q['frames'].value();repeats=q['repeats'].value()
            q['budget'].setText(f'{count}种输入 × {repeats}次 × {n}张RAW = {count*repeats*n}张。'
                +'每次重新清空累加器；轮测各输入分别统计；每批最多300秒，未完成不算通过。')
        except (ValueError,IndexError) as error:q['budget'].setText(str(error))
    for key in ('frames','repeats'):q[key].valueChanged.connect(update_budget)
    for key in ('mode','input_a','input_b'):q[key].currentIndexChanged.connect(update_budget)
    flow.update_budget=update_budget;update_budget()
    line=QtWidgets.QHBoxLayout();layout.addLayout(line)
    q['permission']=QtWidgets.QCheckBox('现场条件已确认，允许本轮有界投图');line.addWidget(q['permission'])
    q['start']=QtWidgets.QPushButton('一键测试');q['start'].setObjectName('stability_start');line.addWidget(q['start'])
    q['cancel']=QtWidgets.QPushButton('停止测试／全0');q['cancel'].setEnabled(False);line.addWidget(q['cancel'])
    q['start'].clicked.connect(flow.start);q['cancel'].clicked.connect(flow.cancel)
    q['status']=QtWidgets.QLabel('未开始。过曝仅提示，不中断；无新帧、显示异常或采集中参数变化仍停止。')
    q['status'].setWordWrap(True);layout.addWidget(q['status'])
    q['summary']=QtWidgets.QLabel('部署结论：未验证。稳定性、输入区分、映射准确性和闭环恢复必须分开判断。')
    q['summary'].setWordWrap(True);layout.addWidget(q['summary'])
    def table(key,rows,headers,height):
        t=QtWidgets.QTableWidget(rows,len(headers));q[key]=t;t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().hide();t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch);t.setMinimumHeight(height);layout.addWidget(t)
    table('gates',0,['部署前检查','判定','本轮数值 / 缺项','门槛 / 解释'],310)
    selection=QtWidgets.QHBoxLayout();layout.addLayout(selection)
    q['result_input']=QtWidgets.QComboBox();selection.addWidget(q['result_input'])
    q['result_domain']=QtWidgets.QComboBox();q['result_domain'].addItem('RAW均值主读出','raw')
    q['result_domain'].addItem('主界面固定显示映射对照（不用于旧RAW头）','processed');selection.addWidget(q['result_domain'])
    for key in ('result_input','result_domain'):q[key].currentIndexChanged.connect(flow.show_result)
    table('table',8,['区域','均值','SD','CV %','峰峰差 %','末次−首次 %','RAW峰值','255占比 %'],275)
    q['plot']=QtWidgets.QLabel('完成后显示R0–R7独立均值波动');q['plot'].setAlignment(QtCore.Qt.AlignCenter);q['plot'].setFixedHeight(460);layout.addWidget(q['plot'])
    note=QtWidgets.QLabel('以下均为RAW诊断：单帧噪声与跨次有效帧波动分开；窗口前后变化用于观察是否仍在稳定，不能仅靠增加N消除慢漂移。')
    note.setWordWrap(True);layout.addWidget(note)
    table('noise_table',8,['区域','窗内单帧CV %','跨次均值CV %','SD降低倍数','最大窗内前后差 %','跨次拟合变化 %'],275)
    q['analysis_chart']=QtWidgets.QComboBox();q['analysis_chart'].addItem('短时 / 跨次波动与八区相关性','noise')
    q['analysis_chart'].addItem('不同输入可区分度矩阵','separation');q['analysis_chart'].currentIndexChanged.connect(flow.show_result);layout.addWidget(q['analysis_chart'])
    q['analysis_plot']=QtWidgets.QLabel();q['analysis_plot'].setAlignment(QtCore.Qt.AlignCenter);q['analysis_plot'].setFixedHeight(500);layout.addWidget(q['analysis_plot'])
    q['folder']=QtWidgets.QLabel('本轮数据目录：未生成');q['folder'].setWordWrap(True);layout.addWidget(q['folder'])
    q['open']=QtWidgets.QPushButton('打开本轮结果目录');q['open'].setEnabled(False);layout.addWidget(q['open'])
    q['open'].clicked.connect(lambda:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(flow.folder))))
    window.tabs.addTab(page,'八区稳定性测试')
    timer=QtCore.QTimer(window);timer.setInterval(100);timer.timeout.connect(flow.poll);timer.start()
    flow.timer=timer;window.stability_workflow=flow
    return flow
