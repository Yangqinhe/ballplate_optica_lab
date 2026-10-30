"""Fresh-owner stability page; original camera-thread sampler, no new SDK.

Imported by an explicit new launcher only. Existing loaded GUI/source files
are never edited or injected. Test results are research observations, not a
new deployment qualification. RAW archive/analysis stays off the Qt thread.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
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
        require(w.actual_roi == self.mask_roi, 'ROI已改变，不可沿用原八区坐标')
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

    def settings(self):
        q = self.widgets
        return stability.Settings(frames=q['frames'].value(), repeats=q['repeats'].value(),
            switch_inputs=q['switch'].isChecked(), settle_seconds=q['settle'].value(),
            interval_seconds=q['interval'].value(), software_gain=q['software_gain'].value(),
            black_ADU=q['black'].value(), gamma=q['gamma'].value()).validate()

    def zero(self):
        self.window.armed.setChecked(False)
        self.window.black_outputs()

    def zero_verified(self):
        w = self.window; arrays = getattr(w, 'last_arrays', {})
        return bool(not w.armed.isChecked() and arrays and set(arrays) >= {'dmd', 'slm'}
                    and all(not arrays[k].any() for k in ('dmd', 'slm'))
                    and w.last_verified_revision == w.revision)

    def apply_camera(self, which):
        """An explicit Apply button uses only the existing owner's transactions."""
        try:
            require(not self.active, '测试中不能修改相机参数')
            self.profile()
            require(self.window.state_sampling_controller.current_core() is None, '原RAW采样正在进行')
            self.zero()
            if which == 'exposure':
                self.window.exposure.setValue(self.widgets['exposure'].value())
                self.window.apply_exposure()
            else:
                self.window.camera_gain.setValue(self.widgets['gain'].value())
                self.window.camera_gain_apply.click()
            self.widgets['status'].setText('已向原CCD线程提交参数；以实际回读为准，完成后再开始测试。')
        except Exception as error:
            self.widgets['status'].setText(str(error))

    def start(self):
        created_folder = None
        try:
            require(not self.active and self.current_core() is None and self.future is None,
                    '上一轮未结束／尚有原始数据待保存')
            require(self.widgets['permission'].isChecked(), '请先确认原ND／低功率与光束防护条件')
            original = self.window.state_sampling_controller
            require(original.current_core() is None, '原固定输入RAW页正在采样')
            settings = self.settings(); active = self.profile()
            require(abs(float(self.window.worker.exposure)-self.widgets['exposure'].value()) < .001,
                    '曝光请求尚未应用；请点击“应用曝光”或同步已应用参数')
            requested_gain = getattr(self.window.worker, 'manual_gain_state', {}).get('requested_db')
            require(requested_gain is not None and abs(float(requested_gain)-self.widgets['gain'].value()) < .001,
                    'Gain请求尚未应用；请点击“应用Gain”或同步已应用参数')
            s = _value(self.window.setting())
            require(s['dmd'] == s['slm'] == 'file', '先在手动页将DMD／SLM设为“载入图像”模式')
            phase = np.asarray(self.window.slm_file)
            require(phase.dtype == np.uint8 and phase.shape == (600, 600), '需要当前规范600×600 SLM缓存')
            indices = {'A': self.widgets['input_a'].currentIndex(), 'B': self.widgets['input_b'].currentIndex()}
            selected = {k: self.assets[i] for k, i in indices.items()}
            require(not settings.switch_inputs or not np.array_equal(selected['A']['image'], selected['B']['image']),
                    '换图测试必须选择不同的A和B')
            self.zero()  # Explicit Start authorizes this bounded test, not projection on startup.
            self.output_root.mkdir(parents=True, exist_ok=True)
            folder = self.output_root/('TEST_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1_000_000))
            folder.mkdir(); self.folder = folder; created_folder = folder
            with (folder/'SLM.npy').open('xb') as f: np.save(f, phase, allow_pickle=False)
            contract = deepcopy(self.template)
            contract['expected_active'] = active
            contract['fixed_context'] = dict(settings=s, roi=deepcopy(self.window.actual_roi),
                geometry_frozen=bool(self.window.registration), optical_reference=_value(self.window.optical_config()))
            contract['fixed_slm'] = stability.file_ref(folder/'SLM.npy')
            contract['allowed_dmd'] = [dict(**selected[k]['ref'], input_id=k) for k in ('A', 'B') if k == 'A' or settings.switch_inputs]
            # Immutable per-test contract, not a change to the deployed policy contract.
            contract['research_test'] = dict(settings=asdict(settings), mask=self.mask_ref,
                camera_requested=dict(exposure_us=self.window.exposure.value(), gain_db=self.window.camera_gain.value()),
                camera_actual=active, control_qualified=False, parent_contract=self.template_ref)
            contract['source_sha256'][self.template_ref['path']] = self.template_ref['sha256']
            for module in (Path(__file__).resolve(), Path(stability.__file__).resolve()):
                require(contract['source_sha256'].get(str(module)) == stability.file_ref(module)['sha256'], '新页源码未绑定到启动配置')
            _write_new(folder/'CONTRACT.json', contract)
            self.run_settings = settings; self.expected_active = active; self.selected = selected
            self.queue = stability.schedule(settings); self.index = 0; self.visits = []; self.error = None
            self.result = None; self.future = None; self.deadline = time.monotonic()+settings.maximum_seconds
            if 'table' in self.widgets:
                self.widgets['table'].clearContents(); self.widgets['plot'].clear()
                self.widgets['summary'].setText('新一轮尚未完成；不沿用上一轮表格或判定。')
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
                        hold_mode=not self.run_settings.switch_inputs, more_visits=self.index+1 < len(self.queue)):
                    original_zero()
            sampler._zero = between_queries_zero
            with self._lock: self.sampler = sampler
            self.leases = sampler.leases
            _write_new(folder/'PLAN.json', dict(settings=asdict(settings), sequence=self.queue,
                contract=stability.file_ref(folder/'CONTRACT.json'), input_sources={k:v['ref'] for k,v in selected.items()},
                same_original_camera_owner=True, reset_each_query=True, pipeline='RAW mean -> black/gamma -> native8',
                actual_framebuffer_ACK_required=True, physically_trigger_synchronized=False,
                hold_mode_does_not_blank_between_queries=not settings.switch_inputs, control_qualified=False))
            self.widgets['status'].setText('测试已冻结参数；等待两路全0核验，然后加载第一个输入。')
        except Exception as error:
            if self.active: self.fail(str(error))
            else:
                if created_folder is not None:
                    _write_new(created_folder/'START_FAILED.json', dict(completed=False,
                        error=str(error), no_test_projection_started=True, control_qualified=False))
                self.widgets['status'].setText('未开始：'+str(error))

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
        row, mean, processed = stability.analyze_archive(ref, self.weights, self.run_settings, item, self.expected_active)
        out = self.folder/('MEAN_%02d_%s'%(self.index, item['input_id'])); out.mkdir()
        with (out/'RAW_mean.npy').open('xb') as f: np.save(f, mean, allow_pickle=False)
        with (out/'processed.npy').open('xb') as f: np.save(f, processed, allow_pickle=False)
        _write_new(out/'READOUT.json', row)
        return row

    def poll(self):
        try:
            self.refresh_actual()
            if not self.active: return
            if self.stage not in ('STOPPING', 'FINAL_ZERO', 'EXPORT'):
                require(time.monotonic() < self.deadline, '本轮达到300秒有界时限')
                require(self.profile() == self.expected_active, '相机实际参数改变，本轮终止且不混算')
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
                if (self.sampler.ack_ns is not None and time.time_ns() >= self.sampler.ack_ns+int(self.run_settings.settle_seconds*1e9)
                        and time.monotonic() >= self.not_before):
                    self.sampler.start(self.run_settings.frames, 'VISIT_%02d'%self.index, duration_seconds=60)
                    self.stage = 'SAMPLING'
            elif self.stage == 'SAMPLING':
                s = status['core']
                require(s is not None and s['error'] is None and s['writer_error'] is None,
                        '采样失败：'+str(None if s is None else s['error']))
                self.widgets['status'].setText('第%d/%d组 · %s 第%d次 · %s · %d/%d RAW'%(
                    self.index+1, len(self.queue), self.queue[self.index]['input_id'], self.queue[self.index]['visit'],
                    s['state'], s['captured_frames'], self.run_settings.frames))
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
                    self.result = stability.analyze(self.visits, self.run_settings)
                    self.result.update(contract=stability.file_ref(self.folder/'CONTRACT.json'), final_actual_zero_verified=True,
                        same_camera_owner=True, all_visits_completed=True)
                    self.future = self.executor.submit(stability.export, deepcopy(self.result), self.folder/'RESULTS')
                    self.stage = 'EXPORT'
                else: require(time.monotonic() < self.zero_deadline, '结束全0核验超时')
            elif self.stage == 'EXPORT' and self.future.done():
                self.future.result(); self.future = None
                self.widgets['status'].setText('完成。已核验两路全0；A/B分别统计，RAW与CSV/PNG/PDF已保存。')
                self.show_result(); self.active = False; self.stage = 'COMPLETE'; self.unlock_controls()
        except Exception as error:
            if self.active:
                if self.stage == 'STOPPING': self.widgets['status'].setText('停止处理中：'+str(error))
                else: self.fail(str(error))
            else: self.widgets['actual'].setText('等待正常CCD：'+str(error))

    def finish_failed(self):
        _write_new(self.folder/'FAILED.json', dict(completed=False, error=self.error,
            completed_visits=self.visits, remaining_sequence=self.queue[self.index:],
            final_actual_zero_verified=True, failed_prefix_not_pooled=True, control_qualified=False))
        self.active = False; self.stage = 'FAILED'; self.unlock_controls()
        self.widgets['status'].setText('测试未完成，不生成正式波动判定；失败前缀已保留，两路全0。原因：'+self.error)

    def refresh_actual(self):
        packet = getattr(self.window, 'last_packet', None)
        if packet and packet.get('active'):
            a = packet['active']; gain_state = getattr(self.window.worker, 'manual_gain_state', {})
            if gain_state.get('available'):
                self.widgets['gain'].setRange(gain_state['minimum_db'], gain_state['maximum_db'])
            self.widgets['actual'].setText('实际回读：曝光%s µs；相机Gain %s dB；ROI %s。测试开始时冻结实际值，软件γ/ADU只用于均值后读出。'%(
                a.get('ExposureTime'), a.get('Gain'), self.window.actual_roi))

    def show_result(self):
        from PyQt5 import QtGui, QtWidgets, QtCore
        if not self.result: return
        label = self.widgets['result_input'].currentText()
        if label not in self.result['inputs']:
            self.widgets['result_input'].setCurrentIndex(0)
            return
        domain = self.widgets['result_domain'].currentData(); info = self.result['inputs'][label][domain]
        table = self.widgets['table']
        fmt = lambda value: '—' if value is None else '%.4f'%value
        for r in info['regions']:
            values = [str(r['region'])]+[fmt(r[k]) for k in ('mean', 'sample_SD', 'CV_percent', 'peak_to_peak_percent', 'first_to_last_percent')]
            for col, value in enumerate(values): table.setItem(r['region'], col, QtWidgets.QTableWidgetItem(value))
        p = info['pair45']
        self.widgets['summary'].setText('输入%s · %d次独立回访 · 八区RMS %s%% · 最大两次差 %s%% · 最大区域CV %s%%\n%s（仅研究参考，不是闭环放行）\nR4+5：CV %s%%；R4−5：SD %s，范围 %s；R4/R5相关系数 %s（少量回访不作因果结论）'%(
            label, info['independent_visits'], fmt(info['relative_vector_RMS_percent']), fmt(info['maximum_pair_L2_percent']),
            fmt(info['maximum_region_CV_percent']), info['reference_gate'], fmt(p['sum']['CV_percent']),
            fmt(p['difference_SD']), fmt(p['difference_range']), fmt(p['correlation'])))
        picture = self.folder/('RESULTS/stability_%s_%s.png'%(label, domain))
        # Never leave a previous input/domain's chart beside a new table.
        self.widgets['plot'].clear()
        if picture.exists():
            self.widgets['plot'].setPixmap(QtGui.QPixmap(str(picture)).scaled(1100, 510, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.widgets['plot'].setText('本轮缺少对应八区曲线文件；请查看八区表格，不显示其他输入/读出域的旧图。')
        self.widgets['folder'].setText(str(self.folder)); self.widgets['open'].setEnabled(True)


def attach(window, template_ref, output_root, mask_ref, mask_roi):
    """Only before camera startup; compose once with the original camera owner."""
    from PyQt5 import QtCore, QtWidgets, QtGui
    import live_loop_v23.direct_alignment_gui as gui
    require(window.worker is None and hasattr(window, 'state_sampling_controller')
            and not hasattr(window, 'stability_workflow'), '新页只允许在新进程启动阶段装配一次')
    flow = Workflow(window, template_ref, output_root, mask_ref, mask_roi)
    require(not getattr(gui.LoopWorker.capture_one, '_optical_stability_capture', False), '不可重复安装采集适配器')
    wrapped = _capture_adapter(gui.LoopWorker.capture_one, flow)
    wrapped._optical_stability_capture = True
    gui.LoopWorker.capture_one = wrapped
    previous = window.state_sampling_controller.start
    def guarded_original_start(*args, **kwargs):
        require(not flow.active, '八区测试正在运行，不允许第二个采样任务')
        return previous(*args, **kwargs)
    window.state_sampling_controller.start = guarded_original_start
    page = QtWidgets.QWidget(); outer = QtWidgets.QVBoxLayout(page); outer.setContentsMargins(0, 0, 0, 0)
    scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); outer.addWidget(scroll)
    body = QtWidgets.QWidget(); scroll.setWidget(body); layout = QtWidgets.QVBoxLayout(body)
    header = QtWidgets.QLabel('独立N帧均值 → 固定黑电平/γ → 八区；不使用预览EMA，不推进MuJoCo。测试结束／取消后全0。')
    header.setWordWrap(True); layout.addWidget(header)
    config = QtWidgets.QGroupBox('固定参数与测试设计'); grid = QtWidgets.QGridLayout(config); layout.addWidget(config)
    q = flow.widgets; q.update(page=page, config=config)
    def spin(name, row, col, title, low, high, value, decimals=2, integer=False):
        control = QtWidgets.QSpinBox() if integer else QtWidgets.QDoubleSpinBox()
        if not integer: control.setDecimals(decimals)
        control.setRange(low, high); control.setValue(value); control.setKeyboardTracking(False)
        control.setObjectName('stability_'+name); q[name] = control
        grid.addWidget(QtWidgets.QLabel(title), row, col); grid.addWidget(control, row, col+1)
        return control
    spin('exposure', 0, 0, '曝光请求 µs', 50, 4000, window.exposure.value())
    apply_exp = QtWidgets.QPushButton('应用曝光（先全0）'); grid.addWidget(apply_exp, 0, 2)
    spin('gain', 0, 3, '相机Gain请求 dB', 0, 18.062, 11)
    apply_gain = QtWidgets.QPushButton('应用Gain（先全0）'); grid.addWidget(apply_gain, 0, 5)
    apply_exp.clicked.connect(lambda: flow.apply_camera('exposure')); apply_gain.clicked.connect(lambda: flow.apply_camera('gain'))
    spin('software_gain', 1, 0, '读出软件增益', 1, 32, 1)
    spin('black', 1, 2, '读出黑电平 ADU', 0, 254, 12.5)
    spin('gamma', 1, 4, '读出 γ', .1, 3, .5)
    spin('frames', 2, 0, '每次平均RAW数 N', 2, 120, 30, integer=True)
    spin('repeats', 2, 2, '每种输入重复次数', 3, 20, 6, integer=True)
    spin('settle', 2, 4, '投图核验后等待 s', 1.5, 15, 1.5)
    spin('interval', 3, 0, '组间额外等待 s', 0, 15, 2)
    q['switch'] = QtWidgets.QCheckBox('测试中换图：A↔B（分别统计）'); grid.addWidget(q['switch'], 3, 2, 1, 4)
    for name, col in [('input_a', 0), ('input_b', 3)]:
        combo = QtWidgets.QComboBox(); combo.setObjectName('stability_'+name)
        for asset in flow.assets: combo.addItem('输入 '+str(asset['input_id']), asset['input_id'])
        q[name] = combo; grid.addWidget(QtWidgets.QLabel('目标A' if name == 'input_a' else '对照B'), 4, col); grid.addWidget(combo, 4, col+1, 1, 2)
    q['input_b'].setCurrentIndex(min(1, len(flow.assets)-1)); q['input_b'].setEnabled(False)
    q['switch'].toggled.connect(q['input_b'].setEnabled)
    q['actual'] = QtWidgets.QLabel('等待CCD实际参数；调整曝光/Gain需点击对应应用按钮，再开始测试。')
    q['actual'].setWordWrap(True); layout.addWidget(q['actual'])
    sync = QtWidgets.QPushButton('同步CCD已应用的曝光/Gain请求值（不写相机）'); grid.addWidget(sync, 5, 0, 1, 6)
    def sync_camera():
        worker = window.worker
        if worker is not None:
            q['exposure'].setValue(float(worker.exposure))
            state = getattr(worker, 'manual_gain_state', {})
            if state.get('requested_db') is not None: q['gain'].setValue(float(state['requested_db']))
    sync.clicked.connect(sync_camera)
    line = QtWidgets.QHBoxLayout(); layout.addLayout(line)
    q['permission'] = QtWidgets.QCheckBox('已确认原ND／低功率及光束防护，允许本轮有界投图'); line.addWidget(q['permission'])
    q['start'] = QtWidgets.QPushButton('一键测试'); q['start'].setObjectName('stability_start'); line.addWidget(q['start'])
    q['cancel'] = QtWidgets.QPushButton('停止测试／全0'); q['cancel'].setEnabled(False); line.addWidget(q['cancel'])
    q['start'].clicked.connect(flow.start); q['cancel'].clicked.connect(flow.cancel)
    q['status'] = QtWidgets.QLabel('未开始。每轮最多300秒；无结果时不显示PASS。达到240ADU的批次保留但不计有效结果。')
    q['status'].setWordWrap(True); layout.addWidget(q['status'])
    selection = QtWidgets.QHBoxLayout(); layout.addLayout(selection)
    q['result_input'] = QtWidgets.QComboBox(); q['result_input'].addItems(['A', 'B']); selection.addWidget(q['result_input'])
    q['result_domain'] = QtWidgets.QComboBox(); q['result_domain'].addItem('均值后黑电平/γ读出', 'processed'); q['result_domain'].addItem('RAW均值读出', 'raw'); selection.addWidget(q['result_domain'])
    q['result_input'].currentIndexChanged.connect(flow.show_result); q['result_domain'].currentIndexChanged.connect(flow.show_result)
    q['table'] = QtWidgets.QTableWidget(8, 6); q['table'].setHorizontalHeaderLabels(['Region', '均值', '样本SD', 'CV %', '峰峰差/均值 %', '末次−首次 %'])
    q['table'].verticalHeader().hide()
    q['table'].setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    q['table'].horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); q['table'].setMinimumHeight(285); layout.addWidget(q['table'])
    q['summary'] = QtWidgets.QLabel('尚无完整独立重复测试；差分接近零时不计算差分CV。'); q['summary'].setWordWrap(True); layout.addWidget(q['summary'])
    plot_note = QtWidgets.QLabel('八区同步波动曲线（R0–R7，2×4）：纵轴为相对各区均值的变化%，八区同尺度；随上方A/B及RAW/处理后选择同步切换。每点是独立N帧均值，不是单张RAW。')
    plot_note.setWordWrap(True); layout.addWidget(plot_note)
    q['plot'] = QtWidgets.QLabel('测试完成后同时显示全部8个区域'); q['plot'].setAlignment(QtCore.Qt.AlignCenter); layout.addWidget(q['plot'])
    q['folder'] = QtWidgets.QLabel('原始数据与结果目录：尚未生成'); q['folder'].setWordWrap(True); layout.addWidget(q['folder'])
    q['open'] = QtWidgets.QPushButton('打开本轮结果目录'); q['open'].setEnabled(False); layout.addWidget(q['open'])
    q['open'].clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(flow.folder))))
    window.tabs.addTab(page, '八区稳定性测试')
    timer = QtCore.QTimer(window); timer.setInterval(100); timer.timeout.connect(flow.poll); timer.start()
    flow.timer = timer; window.stability_workflow = flow
    return flow
