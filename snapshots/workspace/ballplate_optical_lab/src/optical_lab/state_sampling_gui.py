"""Startup-only same-owner dense RAW sampling UI; import never opens Qt/SDK.

Install after exposure warnings and before timeout diagnostics/camera start.
Only the existing worker's capture_one is decorated. The GUI grants a short
immutable fixed-input lease; SDK reads/capture stay on that same worker thread.
No input load, arm, laser command, SDK owner or replacement frame is added.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict,is_dataclass
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

SELF=Path(__file__).resolve()
SAFE=re.compile(r'[A-Za-z0-9_-]{1,64}\Z')
BUSY=('roi_busy','reference_busy','spatial_busy','deployment_busy','adaptation_active','loop_active','override')
TERMINAL=('COMPLETED','COMPLETE','FAILED','FAIL','CANCELLED','CANCELED')


def _require(ok,message):
    if not ok:raise ValueError(message)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin(reference):
    _require(isinstance(reference,dict) and set(reference)=={'path','sha256'},'explicit path/SHA reference required')
    path=Path(reference['path'])
    _require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink()
        and re.fullmatch('[0-9a-f]{64}',reference['sha256']) is not None and _sha(path)==reference['sha256'],
        'regular fixed source/asset SHA differs: '+str(path))
    return path


def _value(value):return asdict(value) if is_dataclass(value) else deepcopy(value)


def _start_ticks():
    # Linux proc field22, robust to spaces in the comm field.
    return Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]


def _write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


class LeaseBox:
    """Worker reads copies only; no worker access to Qt widgets."""
    def __init__(self):self._lock=threading.Lock();self._lease=None
    def set(self,value):
        with self._lock:self._lease=deepcopy(value)
    def get(self):
        with self._lock:return deepcopy(self._lease)


def _capture_adapter(original,controller):
    @wraps(original)
    def capture(worker,*args,**kwargs):
        dark=kwargs.get('dark',args[0] if args else False)
        core=controller.current_core()
        if dark or core is None or core.snapshot()['state'] not in ('PENDING','CAPTURING'):
            return original(worker,*args,**kwargs)
        try:
            lease=controller.leases.get()
            before=worker.camera.parameters()  # genuine same-owner SDK read, not camera.active
            before_ns=time.time_ns()
            accepted=core.before_capture(before,lease,before_ns)
            result=original(worker,*args,**kwargs)  # exactly one original capture, no retry
            if accepted:
                after=worker.camera.parameters();after_ns=time.time_ns()
                raw,metadata=result
                core.feed(raw,metadata,after,controller.leases.get(),after_ns)
            return result
        except BaseException as error:
            try:core.fail(repr(error))
            except BaseException:pass  # never replace original SDK/guard error or cleanup
            raise
    capture._optical_state_sampling=True
    return capture


class SamplingController:
    """Small GUI-side fixed-context gate; capture/writing belong to core."""
    def __init__(self,window,contract_ref,spool_root,core_factory):
        import numpy as np
        self.window=window;self.contract_ref=dict(contract_ref)
        self.contract=json.loads(_pin(contract_ref).read_text())
        c=self.contract
        _require(c.get('schema')=='optical-lab-dense-research-contract-1'
            and c.get('maximum_frames')==240 and c.get('RAW_peak_exclusive_ADU')==240
            and c.get('readback_policy')=='each_frame_before_and_after'
            and c.get('legacy_quantitative_gate_unchanged') is True,'explicit dense research contract required')
        context=c['fixed_context'];self.fixed_context=deepcopy(context)
        _require(set(context)=={'settings','roi','geometry_frozen','optical_reference'}
            and type(context['geometry_frozen']) is bool,'complete exact existing optical context required')
        self.expected_active=deepcopy(c['expected_active']);self.source_sha256=dict(c['source_sha256'])
        for path,digest in self.source_sha256.items():_pin(dict(path=path,sha256=digest))
        _require(self.source_sha256.get(str(SELF))==_sha(SELF),'running GUI sampler source must be bound')
        self.source_sha256[str(_pin(contract_ref))]=contract_ref['sha256']
        self.slm_ref={k:c['fixed_slm'][k] for k in ('path','sha256')}
        self.slm=np.load(_pin(self.slm_ref),allow_pickle=False)
        _require(self.slm.dtype==np.uint8 and self.slm.shape==(600,600),'fixed canonical600 SLM required')
        self.allowed=[]
        for asset in c['allowed_dmd']:
            reference={k:asset[k] for k in ('path','sha256')};image=np.load(_pin(reference),allow_pickle=False)
            _require(image.shape==(600,600) and image.dtype==np.uint8 and np.isin(image,(0,255)).all()
                and 0<np.count_nonzero(image)<=3600,'permitted original sparse canonical600 binary input required')
            self.allowed.append(dict(reference=reference,input_id=asset['input_id'],image=image,
                bytes_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        _require(self.allowed,'one explicit allowed input required')
        self.spool=Path(spool_root)
        _require(self.spool.is_absolute() and self.spool.resolve()==self.spool and not self.spool.exists()
            and self.spool.parent.is_dir(),'one NEW absolute GUI sampler spool required')
        self.spool.mkdir();(self.spool/'requests').mkdir();(self.spool/'results').mkdir()
        self.core_factory=core_factory;self.leases=LeaseBox();self._lock=threading.Lock();self.core=None
        self.owner_pid=os.getpid();self.owner_start=_start_ticks();self.request=None;self.seen=set()
        self.stable_signature=None;self.ack_ns=None;self.last_lease=None;self.last_error=None
        self.zero_requested=False;self._locked=[];self._last_status=None;self.widgets={}
        self.last_core_status=None;self.context_unavailable_reason=None;self.zero_recorded=False
        self.maximum_seconds=c.get('maximum_burst_seconds',60)
        _require(type(self.maximum_seconds) in (int,float) and 5<=self.maximum_seconds<=295,
            'bounded 5..295s capture duration required')

    def current_core(self):
        with self._lock:return self.core

    def _asset(self,requested=None):
        import numpy as np
        image=self.window.dmd_file
        _require(isinstance(image,np.ndarray),'current DMD cache absent')
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        matches=[a for a in self.allowed if a['bytes_sha256']==digest and np.array_equal(a['image'],image)]
        if requested is not None:matches=[a for a in matches if a['reference']==requested]
        _require(len(matches)==1,'current DMD bytes not uniquely bound to requested allowlisted source')
        _require(isinstance(self.window.slm_file,np.ndarray) and np.array_equal(self.window.slm_file,self.slm),
            'current canonical SLM cache differs')
        return matches[0]

    def observe(self,requested=None):
        w=self.window
        _require(w.hardware is True and w.running is True and w.live_ready is True and w.worker is not None,
            'original live camera owner is not ready')
        _require(w.fault is None and w.armed.isChecked() is True and type(w.revision) is int and w.revision>=1
            and w.last_verified_revision==w.revision,'current projection lacks verified healthy revision')
        _require(all(hasattr(w,key) and type(getattr(w,key)) is bool for key in BUSY),
            'complete original busy-state booleans required; no missing defaults')
        flags={key:getattr(w,key) for key in BUSY}
        _require(not any(flags.values()),'another original workflow is busy')
        settings=_value(w.setting());roi=deepcopy(w.actual_roi)
        _require(settings==self.fixed_context['settings'] and roi==self.fixed_context['roi']
            and bool(w.registration)==self.fixed_context['geometry_frozen']
            and _value(w.optical_config())==self.fixed_context['optical_reference'],
            'fixed settings/ROI/existing geometry or optical reference changed')
        _require(settings.get('dmd')==settings.get('slm')=='file','both exact file modes required')
        asset=self._asset(requested)
        return dict(schema='optical-lab-dense-input-lease-1',owner_pid=self.owner_pid,owner_start=self.owner_start,
            revision=w.revision,verified_revision=w.last_verified_revision,dmd=asset['reference'],slm=self.slm_ref,
            input_id=asset['input_id'],settings=settings,roi=roi,armed=True,fault=None,busy_flags=flags,
            geometry_frozen=self.fixed_context['geometry_frozen'],optical_reference=self.fixed_context['optical_reference'])

    def _lock_controls(self):
        w=self.window;items=[w.tabs.widget(i) for i in range(w.tabs.count()) if w.tabs.widget(i) is not self.widgets.get('page')]
        # Global armed-off / stop / Esc stay available as independent escapes.
        items.extend(getattr(w,key,None) for key in ('start_button','camera_only_button','snap_button'))
        self._locked=[(control,control.isEnabled()) for control in items if control is not None]
        for control,_ in self._locked:control.setEnabled(False)

    def _unlock_controls(self):
        for control,enabled in self._locked:control.setEnabled(enabled)
        self._locked=[]

    def start(self,frames,request_id=None,*,requested_dmd=None,revision=None,duration_seconds=None):
        _require(self.current_core() is None,'previous capture has not finished/archive is not sealed')
        _require(type(frames) is int and 1<=frames<=self.contract['maximum_frames'],'N must be integer1..240')
        request_id=request_id or ('GUI_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1_000_000))
        _require(isinstance(request_id,str) and SAFE.fullmatch(request_id) and request_id not in self.seen,
            'one new bounded request identifier required')
        current=self.observe(requested_dmd);now_ns=time.time_ns()
        _require(revision is None or revision==current['revision'],'requested revision changed')
        signature=json.dumps(current,sort_keys=True,separators=(',',':'))
        _require(signature==self.stable_signature and self.ack_ns is not None
            and now_ns>=self.ack_ns+1_500_000_000,'wait for newly observed same verified input settle >=1.5s')
        current.update(ack_verified_ns=self.ack_ns,min_frame_start_ns=self.ack_ns+1_500_000_000,
            updated_monotonic_ns=time.monotonic_ns())
        duration=self.maximum_seconds if duration_seconds is None else duration_seconds
        _require(type(duration) in (int,float) and 5<=duration<=self.maximum_seconds,'request duration exceeds fixed bound')
        output=self.spool/'results'/request_id
        spec=dict(schema='optical-lab-dense-burst-request-1',request_id=request_id,frames=frames,output=str(output),
            deadline_unix_ns=now_ns+int(duration*1e9),contract_ref=self.contract_ref,source_sha256=self.source_sha256)
        core=self.core_factory(spec,self.expected_active,current)
        # Constructor pins the complete source graph and may take >300ms.
        # It has not been published to the worker: reacquire the same GUI
        # context, preserve ACK/cutoff, and refresh ONLY lease monotonic time.
        self.seen.add(request_id)
        self.request=deepcopy(spec);self.last_error=None;self.zero_requested=False
        self.last_core_status=None;self.zero_recorded=False
        try:
            refreshed=self.observe(current['dmd'])
            refreshed.update(ack_verified_ns=current['ack_verified_ns'],
                min_frame_start_ns=current['min_frame_start_ns'],updated_monotonic_ns=time.monotonic_ns())
            _require({k:v for k,v in refreshed.items() if k!='updated_monotonic_ns'}==
                {k:v for k,v in current.items() if k!='updated_monotonic_ns'},
                'fixed input context changed during source validation; no RAW started')
            self._lock_controls()
            refreshed['updated_monotonic_ns']=time.monotonic_ns()
        except BaseException as error:
            # Keep this allocated zero-RAW core until its failed archive is
            # sealed; never discard RAM or expose a stale PENDING core.
            self.last_error=repr(error);self.leases.set(None);self.last_lease=deepcopy(current)
            core.fail(repr(error))
            with self._lock:self.core=core
            if not self._locked:
                try:self._lock_controls()
                except BaseException:pass
            try:self._zero()
            except BaseException as failure:self.last_error+='; zero request failed: '+repr(failure)
            raise
        self.leases.set(refreshed);self.last_lease=deepcopy(refreshed)
        with self._lock:self.core=core
        return deepcopy(spec)

    def cancel(self):
        core=self.current_core()
        if core is not None:core.cancel('operator cancelled dense sampling')

    def _zero(self):
        if not self.zero_requested:
            self.zero_requested=True
            self.window.armed.setChecked(False)  # original slot/finally remain authoritative
            self.window.black_outputs()

    def public_request(self,value):
        required={'schema','action','request_id','owner_pid','owner_start','contract_ref','revision','dmd','slm','frames','duration_seconds'}
        _require(isinstance(value,dict) and set(value)==required
            and value['schema']=='optical-lab-dense-public-request-1' and value['action']=='capture_current'
            and value['owner_pid']==self.owner_pid and value['owner_start']==self.owner_start
            and value['contract_ref']==self.contract_ref and value['slm']==self.slm_ref,
            'only same-owner explicit current-input capture request is allowed')
        return self.start(value['frames'],value['request_id'],requested_dmd=value['dmd'],revision=value['revision'],
            duration_seconds=value['duration_seconds'])

    def _requests(self):
        for path in sorted((self.spool/'requests').glob('REQUEST_*.json')):
            if path.name in self.seen:continue
            self.seen.add(path.name)
            response=dict(request_path=str(path),request_sha256=None,accepted=False,error=None)
            try:
                _require(path.is_file() and not path.is_symlink() and path.parent==self.spool/'requests'
                    and path.stat().st_size<=32768,'regular bounded request file required')
                response['request_sha256']=_sha(path);value=json.loads(path.read_text())
                response['spec']=self.public_request(value);response['accepted']=True
            except BaseException as error:response['error']=repr(error)
            _write_new(self.spool/(path.stem+'_ACK.json'),response)

    def tick(self):
        core=self.current_core();status=self.last_core_status
        if core is not None:status=core.snapshot()
        sampling=core is not None and status['state'] in ('PENDING','CAPTURING')
        if core is None or sampling:
            try:
                requested=None if self.last_lease is None or core is None else self.last_lease['dmd']
                observed=self.observe(requested);signature=json.dumps(observed,sort_keys=True,separators=(',',':'))
                if core is None:
                    if signature!=self.stable_signature:self.stable_signature=signature;self.ack_ns=time.time_ns()
                    observed.update(ack_verified_ns=self.ack_ns,min_frame_start_ns=self.ack_ns+1_500_000_000,
                        updated_monotonic_ns=time.monotonic_ns())
                else:
                    observed.update(ack_verified_ns=self.last_lease['ack_verified_ns'],
                        min_frame_start_ns=self.last_lease['min_frame_start_ns'],updated_monotonic_ns=time.monotonic_ns())
                    _require({k:v for k,v in observed.items() if k!='updated_monotonic_ns'}==
                        {k:v for k,v in self.last_lease.items() if k!='updated_monotonic_ns'},'active immutable input lease changed')
                self.leases.set(observed);self.context_unavailable_reason=None
            except BaseException as error:
                self.context_unavailable_reason=repr(error);self.leases.set(None)
                if core is None:self.stable_signature=None;self.ack_ns=None
                else:core.fail(repr(error))
        if core is not None:
            status=core.snapshot()
            if status['state'] not in ('PENDING','CAPTURING'):
                self._zero()
                self.leases.set(None)
                # A failed writer retains RAM prefix and stays locked. It is
                # not a sealed result and cannot permit a replacement burst.
                if status.get('archive_result') is not None:
                    with self._lock:self.core=None
                    self._unlock_controls();self.last_lease=None;self.stable_signature=None;self.ack_ns=None
            self.last_core_status=status
        self._requests()
        ready=self.current_core() is None and self.ack_ns is not None and time.time_ns()>=self.ack_ns+1_500_000_000
        arrays=getattr(self.window,'last_arrays',{})
        expected_zero=bool(arrays and set(arrays)>={'dmd','slm'} and all(not arrays[k].any() for k in ('dmd','slm')))
        verified_zero=bool(self.zero_requested and expected_zero and not self.window.armed.isChecked()
            and self.window.last_verified_revision==self.window.revision)
        zero=dict(requested=self.zero_requested,expected_arrays_zero=expected_zero,
            original_GUI_RootPixels_verified=verified_zero,revision=self.window.revision,
            last_verified_revision=self.window.last_verified_revision,
            independent_physical_interlock_or_new_readback=False)
        if verified_zero and not self.zero_recorded and self.request is not None:
            _write_new(self.spool/(self.request['request_id']+'_ZERO.json'),dict(unix_ns=time.time_ns(),**zero));self.zero_recorded=True
        public=dict(schema='optical-lab-dense-public-status-1',owner_pid=self.owner_pid,owner_start=self.owner_start,
            contract_ref=self.contract_ref,capture_ready=ready,request=self.request,core=status,error=self.last_error,
            available_lease=self.leases.get(),context_unavailable_reason=self.context_unavailable_reason,
            zero=zero,GUI_lease_max_age_ms=300,
            no_automatic_arm_or_asset_load=True,legacy_quantitative_gate_unchanged=True)
        encoded=json.dumps(public,ensure_ascii=False,sort_keys=True,allow_nan=False)
        if encoded!=self._last_status:
            temporary=self.spool/'STATUS.tmp';_write_new(temporary,public);temporary.replace(self.spool/'STATUS.json');self._last_status=encoded
        if self.widgets:
            self.widgets['start'].setEnabled(ready);self.widgets['frames'].setEnabled(self.current_core() is None)
            self.widgets['cancel'].setEnabled(self.current_core() is not None)
            label='等待当前固定输入投图、像素核验及1.5s稳定' if status is None else '%s · %s/%s RAW'%(status['state'],status['captured_frames'],status['requested_frames'])
            self.widgets['status'].setText(label+(' · '+self.last_error if self.last_error else ''))
        return public

    def poll(self):
        """Qt callback boundary: preserve prefix and zero on reporting faults."""
        try:return self.tick()
        except BaseException as error:
            self.last_error=repr(error);self.leases.set(None)
            core=self.current_core()
            if core is not None:
                try:core.fail(repr(error))
                except BaseException:pass
            try:self._zero()
            except BaseException as failure:self.last_error+='; zero request failed: '+repr(failure)
            if self.widgets:self.widgets['status'].setText('采样故障：'+self.last_error)
            return None


def attach(window,contractref,GUIspoolroot):
    """Explicit fresh-launch composition; never call on an existing live owner."""
    import live_loop_v23.direct_alignment_gui as gui
    from .state_sampling import DenseBurstCore
    from PyQt5 import QtCore,QtWidgets
    current=gui.LoopWorker
    _require(window.worker is None and getattr(current,'__module__',None)=='optical_lab.exposure'
        and getattr(current,'__name__',None)=='WarningPreviewWorker'
        and not getattr(current.capture_one,'_optical_state_sampling',False)
        and not getattr(current.capture_one,'_optical_timeout_diagnostics',False)
        and not hasattr(window,'state_sampling_controller'),
        'attach exactly once after exposure and before timeout/camera startup')
    controller=SamplingController(window,contractref,GUIspoolroot,DenseBurstCore)
    current.capture_one=_capture_adapter(current.capture_one,controller)
    page=QtWidgets.QWidget();layout=QtWidgets.QVBoxLayout(page)
    layout.addWidget(QtWidgets.QLabel('原CCD owner密集RAW；固定当前输入、不自动投图。MuJoCo暂停，采完独立封存后全0。'))
    row=QtWidgets.QHBoxLayout();layout.addLayout(row)
    frames=QtWidgets.QSpinBox();frames.setRange(1,240);frames.setValue(120);row.addWidget(QtWidgets.QLabel('RAW数N'));row.addWidget(frames)
    start=QtWidgets.QPushButton('采当前固定输入RAW');start.setObjectName('dense_burst_capture_current');row.addWidget(start)
    cancel=QtWidgets.QPushButton('取消本次采样');cancel.setObjectName('dense_burst_cancel');row.addWidget(cancel)
    status=QtWidgets.QLabel('待机；默认不采集、不投图。');status.setWordWrap(True);layout.addWidget(status);layout.addStretch()
    controller.widgets=dict(page=page,frames=frames,start=start,cancel=cancel,status=status)
    def request():
        try:controller.start(frames.value())
        except Exception as error:controller.last_error=repr(error)
    start.clicked.connect(request);cancel.clicked.connect(controller.cancel)
    window.tabs.addTab(page,'固定输入RAW采样')
    timer=QtCore.QTimer(window);timer.setInterval(100);timer.timeout.connect(controller.poll);timer.start()
    controller.timer=timer;window.state_sampling_controller=controller
    controller.poll()
    return controller
