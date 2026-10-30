"""Offline fakes/offscreen Qt only: no SDK, camera, display or real GUI owner."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
from optical_lab import state_sampling_gui as gui
from optical_lab.state_sampling import DenseBurstCore


class Widget:
    def __init__(self,enabled=True):self.enabled=enabled
    def isEnabled(self):return self.enabled
    def setEnabled(self,value):self.enabled=value


class Check(Widget):
    def __init__(self,value=True):super().__init__();self.value=value
    def isChecked(self):return self.value
    def setChecked(self,value):self.value=value


class Tabs:
    def __init__(self):self.pages=[Widget(),Widget()]
    def count(self):return len(self.pages)
    def widget(self,index):return self.pages[index]


class Window:
    def __init__(self):
        self.hardware=True;self.running=True;self.live_ready=True;self.worker=object();self.fault=None
        self.armed=Check();self.revision=9;self.last_verified_revision=9;self.registration=None
        self.actual_roi=dict(OffsetX=0,OffsetY=0,Width=4,Height=2)
        self.settings=dict(dmd='file',slm='file',x=-410,y=4)
        self.optics=dict(f1_m=.15,f2_m=.1,wavelength_m=532e-9)
        self.tabs=Tabs();self.start_button=Widget();self.camera_only_button=Widget();self.snap_button=Widget();self.stop_button=Widget()
        for name in gui.BUSY:setattr(self,name,False)
        self.dmd_file=np.zeros((600,600),np.uint8);self.dmd_file[300,300]=255
        self.slm_file=np.zeros((600,600),np.uint8);self.black_calls=0
        self.last_arrays=dict(dmd=self.dmd_file.copy(),slm=np.zeros((1080,1920),np.uint8))
    def setting(self):return deepcopy(self.settings)
    def optical_config(self):return deepcopy(self.optics)
    def black_outputs(self):
        self.black_calls+=1;self.revision+=1;self.last_verified_revision=None
        self.last_arrays=dict(dmd=np.zeros((600,600),np.uint8),slm=np.zeros((1080,1920),np.uint8))


class FakeCore:
    def __init__(self,spec,active,lease):
        self.spec=deepcopy(spec);self.state='PENDING';self.events=[];self.result=None;self.writer_error=None
        self.accept=True;self.error=None;self.count=0
    def snapshot(self):return dict(state=self.state,requested_frames=self.spec.get('frames',3),captured_frames=self.count,
        valid_frames=self.count,error=self.error,archive_result=self.result,writer_error=self.writer_error)
    def before_capture(self,active,lease,stamp):self.events.append(('before',deepcopy(active),lease,stamp,threading.get_ident()));return self.accept
    def feed(self,raw,meta,active,lease,stamp):self.count+=1;self.events.append(('feed',raw,deepcopy(meta),deepcopy(active),lease,stamp,threading.get_ident()))
    def fail(self,reason):self.error=reason;self.state='WRITING';self.events.append(('fail',reason));return self.snapshot()
    def cancel(self,reason):return self.fail(reason)


class Camera:
    def __init__(self,active):self.active=dict(active);self.calls=[]
    def parameters(self):self.calls.append(threading.get_ident());return dict(self.active)


class Worker:
    def __init__(self,active):self.camera=Camera(active);self.count=0;self.captures=[];self.error=None
    def original(self,dark=False):
        self.captures.append((dark,threading.get_ident()))
        if self.error is not None:raise self.error
        started=time.time_ns();self.count+=1;raw=np.full((2,4),20,np.uint8)
        return raw,dict(FrameID=self.count,FrameID_raw=self.count,TimeStamp=self.count*12_500_000,
            logical_sequence=self.count,host_get_started_unix_ns=started,host_received_unix_ns=time.time_ns(),
            Width=4,Height=2,Stride=4,BufferSize=8,pixel_format='Mono8',manual_preview_only=True,quantitative_valid=False)


class StateSamplingGUI(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.window=Window()
        self.active=dict(DeviceSerialNumber='15328133',PixelFormat='Mono8',ExposureMode='Timed',ExposureAuto='Off',
            GainAuto='Off',AcquisitionMode='Continuous',TriggerMode='On',TriggerSelector='FrameStart',
            TriggerSource='Line0',TriggerActivation='FallingEdge',ExposureTime='74.20778',Gain='10.881464',BlackLevel='0',
            OffsetX='0',OffsetY='0',Width='4',Height='2')
        def save(name,array):
            path=self.root/name;np.save(path,array,allow_pickle=False);return dict(path=str(path),sha256=gui._sha(path))
        self.dmd=save('dmd.npy',self.window.dmd_file);self.slm=save('slm.npy',self.window.slm_file)
        corepath=Path(sys.modules['optical_lab.state_sampling'].__file__).resolve()
        c=dict(schema='optical-lab-dense-research-contract-1',maximum_frames=240,RAW_peak_exclusive_ADU=240,
            readback_policy='each_frame_before_and_after',legacy_quantitative_gate_unchanged=True,
            expected_active=self.active,source_sha256={str(gui.SELF):gui._sha(gui.SELF),str(corepath):gui._sha(corepath)},
            fixed_context=dict(settings=self.window.settings,roi=self.window.actual_roi,geometry_frozen=False,optical_reference=self.window.optics),
            fixed_slm=self.slm,allowed_dmd=[dict(input_id='D09',**self.dmd)])
        path=self.root/'contract.json';path.write_text(json.dumps(c));self.contract=dict(path=str(path),sha256=gui._sha(path))
        self.controller=gui.SamplingController(self.window,self.contract,self.root/'spool',FakeCore)

    def ready(self):
        self.controller.tick();self.controller.ack_ns=time.time_ns()-2_000_000_000
        return self.controller.tick()

    def test_import_has_no_Qt_SDK_or_IO_and_attach_requires_fresh_start(self):
        code="import sys;from optical_lab import state_sampling_gui;assert not any(x.startswith(('PyQt','PySpin','spin_c','mujoco','numpy')) for x in sys.modules)"
        subprocess.run([sys.executable,'-B','-c',code],check=True)

    def test_wrapper_dark_idle_and_presettle_keep_one_original_frame_no_retry(self):
        worker=Worker(self.active);controller=self.controller
        method=gui._capture_adapter(Worker.original,controller)
        method(worker);self.assertEqual(len(worker.camera.calls),0)
        self.ready();controller.start(3,'PRESETTLE');core=controller.current_core();core.accept=False
        method(worker,dark=True);self.assertEqual(len(worker.camera.calls),0)
        result=method(worker)
        self.assertEqual(len(worker.captures),3);self.assertEqual(len(worker.camera.calls),1)
        self.assertEqual([x[0] for x in core.events],['before']);self.assertEqual(result[1]['FrameID'],3)

    def test_same_owner_before_original_after_feed_and_metadata_unmodified(self):
        self.ready();self.controller.start(3,'ORDER');core=self.controller.current_core();worker=Worker(self.active)
        value=gui._capture_adapter(Worker.original,self.controller)(worker)
        self.assertEqual(len(worker.captures),1);self.assertEqual(len(worker.camera.calls),2)
        before,after=core.events;self.assertEqual((before[0],after[0]),('before','feed'))
        self.assertLessEqual(before[3],value[1]['host_get_started_unix_ns'])
        self.assertGreaterEqual(after[5],value[1]['host_received_unix_ns'])
        self.assertTrue(after[2]['manual_preview_only']);self.assertFalse(after[2]['quantitative_valid'])
        self.assertEqual(before[-1],after[-1]);self.assertEqual(before[-1],threading.get_ident())

    def test_original_timeout_preserved_fail_before_propagation_and_no_retry(self):
        self.ready();self.controller.start(3,'TIMEOUT');worker=Worker(self.active)
        failure=RuntimeError('spinCameraGetNextImageEx: -1011');worker.error=failure
        with self.assertRaises(RuntimeError) as got:gui._capture_adapter(Worker.original,self.controller)(worker)
        self.assertIs(got.exception,failure);self.assertEqual(len(worker.captures),1)
        self.assertEqual(self.controller.current_core().events[-1][0],'fail')

    def test_real_core_fake_camera_seals_three_RAW_no_old_Gain0_gate(self):
        self.controller.core_factory=DenseBurstCore;self.ready();self.controller.start(3,'REAL_CORE')
        worker=Worker(self.active);method=gui._capture_adapter(Worker.original,self.controller)
        for _ in range(3):method(worker)
        core=self.controller.current_core();state=core.join(3)
        self.assertEqual(state['state'],'COMPLETE');self.assertEqual(state['valid_frames'],3)
        data=json.loads(Path(state['archive_result']['path']).read_text())
        self.assertTrue(data['completed']);self.assertEqual(data['expected_active']['Gain'],'10.881464')
        for row in data['frames']:
            meta=json.loads(Path(row['metadata']['path']).read_text())
            self.assertTrue(meta['original_metadata']['manual_preview_only'])
            self.assertFalse(meta['original_metadata']['quantitative_valid'])
        self.assertEqual(len(worker.camera.calls),6);self.assertEqual(len(worker.captures),3)

    def test_source_constructor_over_300ms_refreshes_only_mono_before_publish(self):
        self.ready();clock=[time.monotonic_ns()];created=[]
        def slow_factory(spec,active,lease):
            core=DenseBurstCore(spec,active,lease);created.append(core)
            clock[0]+=350_000_000  # source hashing latency, no wall-clock sleep
            self.assertIsNone(self.controller.current_core())
            return core
        self.controller.core_factory=slow_factory
        with mock.patch.object(gui.time,'monotonic_ns',side_effect=lambda:clock[0]):
            self.controller.start(1,'SLOW_SOURCE')
            initial=created[0].initial_lease;published=self.controller.leases.get()
            self.assertEqual(published['updated_monotonic_ns']-initial['updated_monotonic_ns'],350_000_000)
            self.assertEqual({k:v for k,v in initial.items() if k!='updated_monotonic_ns'},
                {k:v for k,v in published.items() if k!='updated_monotonic_ns'})
            gui._capture_adapter(Worker.original,self.controller)(Worker(self.active))
        state=created[0].join(3)
        self.assertEqual(state['state'],'COMPLETE');self.assertEqual(state['valid_frames'],1)

    def test_source_constructor_context_change_archives_zero_RAW_and_zeros(self):
        self.ready();created=[]
        def changed_factory(spec,active,lease):
            core=DenseBurstCore(spec,active,lease);created.append(core)
            self.window.revision+=1;self.window.last_verified_revision=self.window.revision
            return core
        self.controller.core_factory=changed_factory
        with self.assertRaisesRegex(ValueError,'changed during source validation'):
            self.controller.start(1,'SOURCE_CONTEXT_CHANGED')
        self.assertIs(self.controller.current_core(),created[0]);self.assertIsNone(self.controller.leases.get())
        self.assertFalse(self.window.armed.isChecked());self.assertEqual(self.window.black_calls,1)
        self.assertTrue(all(not w.isEnabled() for w in self.window.tabs.pages))
        state=created[0].join(3);self.assertEqual(state['state'],'FAIL');self.assertEqual(state['captured_frames'],0)
        data=json.loads(Path(state['archive_result']['path']).read_text())
        self.assertFalse(data['completed']);self.assertEqual(data['frames'],[])
        self.assertIn('changed during source validation',data['error'])

    def test_lease_ack_constant_no_missing_busy_defaults_and_exact_false_geometry(self):
        first=self.ready();ack=first['available_lease']['ack_verified_ns'];self.assertFalse(first['available_lease']['geometry_frozen'])
        self.controller.start(3,'LEASE');self.controller.tick();lease=self.controller.leases.get()
        self.assertEqual(lease['ack_verified_ns'],ack);self.assertEqual(lease['min_frame_start_ns'],ack+1_500_000_000)
        self.assertEqual(lease['dmd'],self.dmd);self.assertEqual(lease['slm'],self.slm)
        del self.window.roi_busy
        self.controller.tick();self.assertEqual(self.controller.current_core().snapshot()['state'],'WRITING')
        self.assertIsNone(self.controller.leases.get());self.assertFalse(self.window.armed.isChecked())

    def test_lock_excludes_global_esc_stop_armed_and_zero_before_writer_seal(self):
        self.ready();self.controller.start(3,'LOCK');core=self.controller.current_core()
        self.assertTrue(all(not w.isEnabled() for w in self.window.tabs.pages))
        self.assertTrue(self.window.armed.isEnabled());self.assertTrue(self.window.stop_button.isEnabled())
        core.state='WRITING';self.controller.tick()
        self.assertFalse(self.window.armed.isChecked());self.assertEqual(self.window.black_calls,1)
        self.assertIs(self.controller.current_core(),core);self.assertTrue(all(not w.isEnabled() for w in self.window.tabs.pages))
        core.state='COMPLETE';core.result=dict(path='/fake/result',sha256='f'*64)
        last=self.controller.tick();self.assertIsNone(self.controller.current_core())
        self.assertTrue(all(w.isEnabled() for w in self.window.tabs.pages));self.assertFalse(last['capture_ready'])
        self.window.last_verified_revision=self.window.revision
        final=self.controller.tick();self.assertEqual(final['core']['state'],'COMPLETE')
        self.assertTrue(final['zero']['original_GUI_RootPixels_verified']);self.assertTrue((self.controller.spool/'LOCK_ZERO.json').is_file())
        self.assertEqual(self.window.black_calls,1)

    def test_writer_failure_preserves_lock_and_prevents_replacement(self):
        self.ready();self.controller.start(3,'WRITEFAIL');core=self.controller.current_core()
        core.state='FAIL';core.writer_error='disk full'
        self.controller.tick();self.assertIs(self.controller.current_core(),core)
        with self.assertRaises(ValueError):self.controller.start(3,'REPLACEMENT')
        self.assertFalse(self.window.armed.isChecked())

    def test_public_request_only_same_owner_current_asset_no_arbitrary_action(self):
        self.ready();value=dict(schema='optical-lab-dense-public-request-1',action='capture_current',request_id='PUBLIC',
            owner_pid=self.controller.owner_pid,owner_start=self.controller.owner_start,contract_ref=self.contract,
            revision=self.window.revision,dmd=self.dmd,slm=self.slm,frames=3,duration_seconds=10)
        for key,wrong in (('action','arm'),('owner_pid',-1),('owner_start','0'),('revision',999),('frames',3.0),
                ('dmd',dict(path='/tmp/no',sha256='a'*64))):
            with self.subTest(key=key),self.assertRaises(ValueError):self.controller.public_request(dict(value,**{key:wrong}))
        spec=self.controller.public_request(value)
        self.assertEqual(spec['frames'],3);self.assertEqual(Path(spec['output']).parent,self.controller.spool/'results')
        self.assertEqual(self.window.black_calls,0);self.assertTrue(self.window.armed.isChecked())

    def test_offscreen_attach_keeps_exposure_class_and_accepts_timeout_composition(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        try:from PyQt5 import QtWidgets
        except ImportError:self.skipTest('offline PyQt5 not installed in this Python')
        from optical_lab.capture_timeout import attach_timeout_diagnostics
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(['offline-sampler-test'])
        base=type('LoopWorker',(),{})
        worker=type('WarningPreviewWorker',(base,),dict(__module__='optical_lab.exposure',capture_one=Worker.original))
        package=types.ModuleType('live_loop_v23');package.__path__=[]
        module=types.ModuleType('live_loop_v23.direct_alignment_gui');module.LoopWorker=worker
        worker_module=types.ModuleType('live_loop_v23.worker');worker_module.LoopWorker=base
        package.direct_alignment_gui=module;package.worker=worker_module
        window=QtWidgets.QMainWindow();window.worker=None;window.hardware=False;window.tabs=QtWidgets.QTabWidget(window)
        window.tabs.addTab(QtWidgets.QWidget(),'old');window.setCentralWidget(window.tabs)
        window.armed=QtWidgets.QCheckBox();window.revision=1;window.last_verified_revision=None;window.last_arrays={}
        window.black_outputs=mock.Mock()
        with mock.patch.dict(sys.modules,{'live_loop_v23':package,'live_loop_v23.direct_alignment_gui':module,'live_loop_v23.worker':worker_module}):
            controller=gui.attach(window,self.contract,self.root/'qt_spool')
            self.assertEqual(controller.widgets['frames'].value(),120);self.assertEqual(window.tabs.count(),2)
            self.assertTrue(worker.capture_one._optical_state_sampling);self.assertFalse(controller.widgets['start'].isEnabled())
            self.assertEqual(worker.__module__,'optical_lab.exposure');attach_timeout_diagnostics(window)
            self.assertTrue(worker.capture_one._optical_timeout_diagnostics)
            with self.assertRaises(ValueError):gui.attach(window,self.contract,self.root/'twice')
            controller.timer.stop();window.close();window.deleteLater();app.processEvents()
        window.black_outputs.assert_not_called()


if __name__=='__main__':unittest.main()
