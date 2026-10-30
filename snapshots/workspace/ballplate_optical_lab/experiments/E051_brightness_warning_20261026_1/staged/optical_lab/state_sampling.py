"""Bounded same-owner RAW burst core: no Qt, SDK, projection or control code.

The camera owner supplies a fresh complete SDK readback before and after each
capture. This module only copies RAM in those calls. One separate file-only
writer archives the finite buffer. Legacy preview/quantitative flags remain
unchanged; the new research contract is explicit and cannot release control.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np


REQUEST_SCHEMA='optical-lab-dense-burst-request-1'
CONTRACT_SCHEMA='optical-lab-dense-research-contract-1'
LEASE_SCHEMA='optical-lab-dense-input-lease-1'
MAX_FRAMES=240
MAX_PAYLOAD_BYTES=73_400_000
MAX_LEASE_AGE_NS=300_000_000
MIN_POST_ACK_NS=1_500_000_000
SELF=Path(__file__).resolve()
_LEASE_KEYS={'schema','owner_pid','owner_start','revision','verified_revision','dmd','slm','settings',
    'roi','ack_verified_ns','min_frame_start_ns','updated_monotonic_ns','armed','fault','busy_flags'}


def _require(ok,message):
    if not ok:raise ValueError(message)


def _integer(value,minimum=0):return type(value) is int and value>=minimum


def _clone(value):
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True))


def _same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(value):
    _require(isinstance(value,dict) and set(value)=={'path','sha256'},'exact path/SHA reference required')
    path=Path(value['path']);digest=value['sha256']
    _require(path.is_absolute() and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,
        'absolute reference/lowercase SHA required')
    return path,digest


def _pin(path,digest):
    path=Path(path)
    _require(path.is_file() and not path.is_symlink() and path.resolve()==path and _sha(path)==digest,
        'regular source/reference SHA differs: '+str(path))


def _profile(value):
    value=_clone(value)
    expected=dict(DeviceSerialNumber='15328133',PixelFormat='Mono8',ExposureMode='Timed',ExposureAuto='Off',
        GainAuto='Off',AcquisitionMode='Continuous',TriggerMode='On',TriggerSelector='FrameStart',
        TriggerSource='Line0',TriggerActivation='FallingEdge')
    _require(all(value.get(key)==actual for key,actual in expected.items()),'declared Mono8/Line0/manual camera profile required')
    for name in ('ExposureTime','Gain','BlackLevel'):
        number=float(value[name]);_require(math.isfinite(number) and number>=0,'finite actual '+name+' required')
    roi={key:int(value[key]) for key in ('OffsetX','OffsetY','Width','Height')}
    _require(roi['OffsetX']>=0 and roi['OffsetY']>=0 and roi['Width']>0 and roi['Height']>0,'native actual ROI required')
    return value,roi


def _lease(value,roi,now_monotonic_ns):
    value=_clone(value)
    _require(isinstance(value,dict) and _LEASE_KEYS<=set(value) and value['schema']==LEASE_SCHEMA,'explicit full lease required')
    _require(_integer(value['owner_pid'],1) and isinstance(value['owner_start'],str) and value['owner_start'].isdigit(),
        'exact owner PID/start identity required')
    _require(_integer(value['revision'],1) and type(value['verified_revision']) is int
        and value['verified_revision']==value['revision'],'actual verified display revision required')
    _reference(value['dmd']);_reference(value['slm'])
    _require(isinstance(value['settings'],dict) and _same(value['roi'],roi),'settings and complete native ROI required')
    _require(_integer(value['ack_verified_ns'],1) and _integer(value['min_frame_start_ns'],value['ack_verified_ns']+MIN_POST_ACK_NS),
        'ACK and settled frame-start cutoff required')
    updated=value['updated_monotonic_ns']
    _require(_integer(updated) and 0<=now_monotonic_ns-updated<=MAX_LEASE_AGE_NS,'GUI lease stale or from future')
    _require(value['armed'] is True and value['fault'] is None and isinstance(value['busy_flags'],dict)
        and bool(value['busy_flags']) and all(type(v) is bool and not v for v in value['busy_flags'].values()),
        'armed fault-free idle input lease required')
    return value


class DenseBurstCore:
    """One finite burst; SDK-owner calls before_capture/feed, never GUI tick.

    Constructors may read explicit source/contract files. before_capture/feed
    do not touch files. Feed preserves even rejected native-sized RAW before
    failing. An arbitrarily oversized malformed buffer cannot violate the RAM
    bound: that case explicitly reports rejected_RAW_not_copied and never PASS.
    The original owner must retain its own malformed/incomplete rejection.
    """
    def __init__(self,spec,expected_active,initial_lease):
        self._lock=threading.RLock();self._thread=None;self._state='PENDING';self._error=None
        self._writer_error=None;self._archive_result=None;self._records=[];self._valid=0;self._used=0
        self._inflight=None;self._last=None;self._pre_cutoff_frames=0
        self._owner_thread_id=None
        self.spec=_clone(spec)
        _require(self.spec.get('schema')==REQUEST_SCHEMA,'explicit dense request schema required')
        frames=self.spec.get('frames');_require(type(frames) is int and 1<=frames<=MAX_FRAMES,'frames must be integer1..240')
        request_id=self.spec.get('request_id')
        _require(isinstance(request_id,str) and re.fullmatch('[A-Za-z0-9_.-]{1,64}',request_id) is not None,'safe unique request_id required')
        self.output=Path(self.spec['output'])
        _require(self.output.is_absolute() and self.output.parent.is_dir() and self.output.parent.resolve()==self.output.parent
            and not self.output.exists() and not self.output.is_symlink(),'fresh absolute output directory required')
        deadline=self.spec.get('deadline_unix_ns');now=time.time_ns()
        _require(_integer(deadline,now+1) and deadline-now<=295_000_000_000,'explicit current deadline at most295s required')
        self.expected_active,self.roi=_profile(expected_active)
        self.shape=(self.roi['Height'],self.roi['Width'])
        self.payload_capacity=frames*math.prod(self.shape)
        _require(self.payload_capacity<=MAX_PAYLOAD_BYTES,'native burst payload exceeds73.4MB hard bound')
        self.initial_lease=_lease(initial_lease,self.roi,time.monotonic_ns())
        self._frozen_lease={k:v for k,v in self.initial_lease.items() if k!='updated_monotonic_ns'}
        contract_path,contract_sha=_reference(self.spec['contract_ref']);_pin(contract_path,contract_sha)
        self.contract=json.loads(contract_path.read_text(encoding='utf-8'))
        self.brightness_policy=self.contract.get('brightness_policy','strict_headroom')
        _require(self.brightness_policy in ('strict_headroom','warning_only'),'unknown brightness policy')
        _require(self.contract.get('schema')==CONTRACT_SCHEMA and _same(self.contract.get('expected_active'),self.expected_active)
            and self.contract.get('maximum_frames')==MAX_FRAMES and self.contract.get('RAW_peak_exclusive_ADU')==240
            and self.contract.get('readback_policy')=='each_frame_before_and_after'
            and self.contract.get('minimum_post_ACK_ns',MIN_POST_ACK_NS)==MIN_POST_ACK_NS
            and self.contract.get('legacy_quantitative_gate_unchanged') is True,'explicit independent dense research profile contract required')
        sources=self.spec.get('source_sha256');_require(isinstance(sources,dict) and str(SELF) in sources,'core source SHA must be bound')
        contract_sources=self.contract.get('source_sha256')
        _require(isinstance(contract_sources,dict) and str(SELF) in contract_sources,'complete contract source closure required')
        expected_sources=dict(contract_sources)
        _require(str(contract_path) not in expected_sources or expected_sources[str(contract_path)]==contract_sha,
            'contract self-reference contradicts its explicit reference')
        expected_sources[str(contract_path)]=contract_sha
        _require(_same(sources,expected_sources),'request must bind exact contract sources plus contract reference; no lost/extra bindings')
        for path,digest in sources.items():
            _reference({'path':path,'sha256':digest});_pin(path,digest)
        for key in ('dmd','slm'):
            path,digest=_reference(self.initial_lease[key]);_pin(path,digest)
        self._buffer=np.empty(self.payload_capacity,dtype=np.uint8)

    def _check(self,active,lease,readback_ns):
        now=time.time_ns();_require(now<self.spec['deadline_unix_ns'],'burst deadline expired')
        _require(_integer(readback_ns,1) and 0<=now-readback_ns<=MAX_LEASE_AGE_NS,'fresh actual SDK readback time required')
        active=_clone(active);_require(_same(active,self.expected_active),'complete actual camera profile differs')
        lease=_lease(lease,self.roi,time.monotonic_ns())
        _require(_same({k:v for k,v in lease.items() if k!='updated_monotonic_ns'},self._frozen_lease),
            'owner/input/revision/settings/ROI/lease changed during burst')
        return active,lease

    def before_capture(self,active_readback,lease,readback_ns):
        """Owner-thread only; True means feed exactly the next camera result."""
        with self._lock:
            if self._state not in ('PENDING','CAPTURING'):return False
            try:
                if self._owner_thread_id is None:self._owner_thread_id=threading.get_ident()
                _require(threading.get_ident()==self._owner_thread_id,'burst capture callbacks changed owner thread')
                active,lease=self._check(active_readback,lease,readback_ns)
                _require(self._inflight is None,'duplicate before_capture without its frame')
                if time.time_ns()<lease['min_frame_start_ns']:
                    self._pre_cutoff_frames+=1;return False
                self._state='CAPTURING'
                self._inflight=dict(active=active,lease=lease,readback_ns=readback_ns)
                return True
            except Exception as error:self._fail_locked(str(error));return False

    def feed(self,raw,metadata,active_readback,lease,readback_ns):
        """Owner-thread only: bounded copy + checks, no SDK, Qt or disk IO."""
        with self._lock:
            if self._state!='CAPTURING':return self.snapshot()
            record=dict(index=len(self._records),valid=False,rejection=None,raw_retained=False)
            try:
                _require(threading.get_ident()==self._owner_thread_id,'feed is not in the original capture owner thread')
                a=np.asarray(raw)
                _require(not a.dtype.hasobject and a.ndim==2 and a.size>0,'nonempty nonobject2D RAW required')
                nbytes=int(a.nbytes)
                _require(self._used+nbytes<=self.payload_capacity,'rejected_RAW_not_copied: malformed frame exceeds finite payload budget')
                offset=self._used
                target=self._buffer[offset:offset+nbytes].view(a.dtype).reshape(a.shape)
                np.copyto(target,a,casting='no');self._used+=nbytes
                record.update(raw_retained=True,buffer_offset=offset,nbytes=nbytes,dtype=a.dtype.str,shape=list(a.shape))
                record['original_metadata']=_clone(metadata)
                active,lease=self._check(active_readback,lease,readback_ns)
                record.update(active_readback_after=active,lease_after=lease,readback_after_ns=readback_ns,
                    active_readback_before=deepcopy(self._inflight['active']) if self._inflight else None,
                    lease_before=deepcopy(self._inflight['lease']) if self._inflight else None,
                    readback_before_ns=self._inflight['readback_ns'] if self._inflight else None)
                _require(self._inflight is not None,'feed requires matching owner-thread before_capture')
                _require(a.dtype==np.uint8 and a.shape==self.shape,'native Mono8 RAW dtype/ROI differs')
                meta=record['original_metadata']
                _require(meta.get('IsIncomplete') is not True,'original camera metadata declares incomplete image')
                for key in ('FrameID','TimeStamp','logical_sequence','host_get_started_unix_ns','host_received_unix_ns','Width','Height','Stride','BufferSize'):
                    _require(_integer(meta.get(key)), 'integer original camera metadata required: '+key)
                _require(meta['pixel_format']=='Mono8' and (meta['Height'],meta['Width'])==self.shape
                    and meta['Stride']>=meta['Width']
                    and (meta['Height']-1)*meta['Stride']+meta['Width']<=meta['BufferSize']<=100_000_000,
                    'original native pixel/buffer header differs')
                _require(self._inflight['readback_ns']<=meta['host_get_started_unix_ns']<=meta['host_received_unix_ns']<=readback_ns,
                    'fresh SDK before/frame/after chronology differs')
                _require(meta['host_get_started_unix_ns']>=lease['min_frame_start_ns'],'frame began before settled input ACK')
                if self._last is not None:
                    previous=self._last
                    _require(meta['TimeStamp']>previous['TimeStamp'] and meta['logical_sequence']==previous['logical_sequence']+1
                        and meta['host_get_started_unix_ns']>previous['host_get_started_unix_ns'],'duplicate/reordered frame or logical capture gap')
                    step=meta['FrameID']-previous['FrameID']
                    if step<=0:
                        _require(previous['FrameID']==65535 and meta['FrameID']==0 and meta.get('frame_id_wrap') is True,
                            'duplicate/reset raw FrameID')
                        step=1
                    record['native_FrameID_step']=step
                    record['native_TimeStamp_delta']=meta['TimeStamp']-previous['TimeStamp']
                maximum=int(a.max());record['max_ADU']=maximum
                record.update(near_saturation_pixels=int(np.count_nonzero(a>=240)),
                    clipped_pixels=int(np.count_nonzero(a==255)),brightness_policy=self.brightness_policy,
                    brightness_warning=maximum>=240,photometric_headroom_passed=maximum<240,
                    raw_structurally_valid_NOT_photometric_qualification=True)
                if self.brightness_policy=='strict_headroom':
                    _require(maximum<240,'RAW headroom failed: max_ADU>=240')
                record['valid']=True;self._valid+=1;self._last=deepcopy(meta);self._inflight=None
                self._records.append(record)
                if self._valid==self.spec['frames']:self._begin_write_locked(True)
            except Exception as error:
                record['rejection']=str(error);self._records.append(record);self._inflight=None
                self._fail_locked(str(error))
            return self.snapshot()

    def _fail_locked(self,reason):
        # WRITING already owns an immutable finished or failed acquisition.
        # Stop/black remains the GUI's independent responsibility; never cancel
        # the only archive of its retained RAW or hold its lock during disk IO.
        if self._state in ('WRITING','COMPLETE','FAIL'):return
        self._error=self._error or str(reason);self._inflight=None
        self._begin_write_locked(False)

    def fail(self,reason):
        with self._lock:self._fail_locked(reason);return self.snapshot()

    def cancel(self,reason='operator cancelled'):return self.fail(reason)

    def snapshot(self):
        with self._lock:
            return dict(state=self._state,requested_frames=self.spec['frames'],captured_frames=sum(r['raw_retained'] for r in self._records),
                valid_frames=self._valid,error=self._error,writer_error=self._writer_error,output=str(self.output),
                brightness_warning_frames=sum(bool(r.get('brightness_warning')) for r in self._records),
                clipped_pixel_observations=sum(r.get('clipped_pixels',0) for r in self._records),
                payload_bytes_copied=self._used,payload_capacity_bytes=self.payload_capacity,
                buffer_released=self._buffer is None,archive_result=deepcopy(self._archive_result))

    status=snapshot

    def join(self,timeout=None):
        with self._lock:thread=self._thread
        if thread is not None:thread.join(timeout)
        return self.snapshot()

    def _begin_write_locked(self,success):
        self._state='WRITING';self._buffer.setflags(write=False)
        self._thread=threading.Thread(target=self._write,args=(success,),name='dense-RAW-file-writer',daemon=False)
        self._thread.start()

    def _write(self,success):
        """File-only background task. Never accesses a camera/window callback."""
        try:
            self.output.mkdir(exist_ok=False);frames=[]
            def write_json(path,value):
                with path.open('x',encoding='utf-8') as f:
                    json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
            for record in self._records:
                row=deepcopy(record)
                if row['raw_retained']:
                    a=self._buffer[row['buffer_offset']:row['buffer_offset']+row['nbytes']].view(np.dtype(row['dtype'])).reshape(row['shape'])
                    path=self.output/('RAW_%03d.npy'%row['index'])
                    with path.open('xb') as f:np.save(f,a,allow_pickle=False)
                    row['raw']=dict(path=str(path),sha256=_sha(path))
                    row['raw_pixel_sha256']=hashlib.sha256(a.tobytes()).hexdigest()
                else:row['raw']=None
                row.update(research_contract=deepcopy(self.spec['contract_ref']),
                    source_sha256=deepcopy(self.spec['source_sha256']),
                    new_research_frame_valid=row['valid'],legacy_metadata_flags_unchanged=True,
                    frame_pattern_association_is_software_only=True,control_release_approved=False)
                path=self.output/('FRAME_%03d.json'%row['index']);write_json(path,row)
                frames.append(dict(index=row['index'],valid=row['valid'],raw=row['raw'],
                    metadata=dict(path=str(path),sha256=_sha(path))))
            source_error=None
            try:
                for path,digest in self.spec['source_sha256'].items():_pin(path,digest)
                _pin(*_reference(self.spec['contract_ref']))
                for key in ('dmd','slm'):_pin(*_reference(self.initial_lease[key]))
            except Exception as error:source_error=str(error)
            with self._lock:
                if source_error:self._error=self._error or source_error
                passed=success and self._error is None and self._valid==self.spec['frames'] and len(frames)==self._valid
                result=dict(schema='optical-lab-dense-burst-result-1',completed=passed,error=self._error,
                    request=self.spec,expected_active=self.expected_active,initial_lease=self.initial_lease,
                    requested_frames=self.spec['frames'],captured_frames=sum(f['raw'] is not None for f in frames),
                    valid_frames=self._valid,frames=frames,payload_bytes_copied=self._used,
                    brightness_policy=self.brightness_policy,
                    brightness_warning_frames=sum(bool(r.get('brightness_warning')) for r in self._records),
                    clipped_pixel_observations=sum(r.get('clipped_pixels',0) for r in self._records),
                    completion_is_acquisition_validity_NOT_photometric_qualification=True,
                    source_recheck_error=source_error,before_cutoff_preview_frames_not_sampled=self._pre_cutoff_frames,
                    minimum_post_ACK_ns=MIN_POST_ACK_NS,
                    dense_burst_NOT_guaranteed_contiguous_native_frames=True,
                    actual_capture_rate_not_predeclared=True,SDK_connections_opened=0,camera_node_writes=0,
                    capture_thread_disk_writes=0,legacy_quantitative_gate_unchanged=True,
                    requires_external_allzero_verification=True,control_release_approved=False)
            write_json(self.output/'BATCH_RESULT.json',result)
            products={p.name:_sha(p) for p in self.output.iterdir() if p.is_file()}
            write_json(self.output/'MANIFEST.json',dict(schema='optical-lab-dense-burst-manifest-1',
                file_sha256=products,source_sha256=self.spec['source_sha256'],contract_ref=self.spec['contract_ref'],
                manifest_excludes_itself=True))
            archive_sha=_sha(self.output/'BATCH_RESULT.json')
            with self._lock:
                self._archive_result=dict(path=str(self.output/'BATCH_RESULT.json'),sha256=archive_sha)
                self._buffer=None;self._state='COMPLETE' if passed else 'FAIL'
        except Exception as error:
            with self._lock:
                self._writer_error=repr(error);self._error=self._error or 'archive failure: '+repr(error);self._state='FAIL'


class DenseBurstCoordinator:
    """One active core only. No camera/window owner is constructed here."""
    def __init__(self):self._lock=threading.Lock();self.current=None

    def request(self,spec,expected_active,initial_lease):
        with self._lock:
            if self.current is not None:
                state=self.current.snapshot()
                _require(state['state'] in ('COMPLETE','FAIL') and state['buffer_released'],
                    'previous burst active or unarchived RAW still retained')
            self.current=DenseBurstCore(spec,expected_active,initial_lease)
            return self.current
