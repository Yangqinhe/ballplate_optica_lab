"""Pure RAW fixtures and file writers; never import Qt, SDK or legacy GUI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from optical_lab import state_sampling as s


class DenseSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.active=dict(DeviceSerialNumber='15328133',PixelFormat='Mono8',ExposureMode='Timed',ExposureAuto='Off',
            GainAuto='Off',AcquisitionMode='Continuous',TriggerMode='On',TriggerSelector='FrameStart',
            TriggerSource='Line0',TriggerActivation='FallingEdge',ExposureTime='74.20778274536133',
            Gain='10.881464004516602',BlackLevel='4.0771484375',OffsetX='900',OffsetY='416',Width='4',Height='4',
            SensorShutterMode='Rolling',AcquisitionFrameRate=None)
        self.dmd=self.root/'dmd.npy';self.slm=self.root/'slm.npy'
        np.save(self.dmd,np.zeros((2,2),np.uint8),allow_pickle=False)
        np.save(self.slm,np.zeros((2,2),np.uint8),allow_pickle=False)
        now=time.time_ns()
        self.lease=dict(schema=s.LEASE_SCHEMA,owner_pid=1234,owner_start='5678',revision=20,verified_revision=20,
            dmd=self.ref(self.dmd),slm=self.ref(self.slm),settings={'dmd':'file','slm':'file','x':-410},
            roi={'OffsetX':900,'OffsetY':416,'Width':4,'Height':4},ack_verified_ns=now-3_000_000_000,
            min_frame_start_ns=now-1_000_000_000,updated_monotonic_ns=time.monotonic_ns(),armed=True,fault=None,
            busy_flags={'configuration':False,'gain':False,'exposure':False,'loop':False},geometry_frozen=False)
        self.contract_path=self.root/'CONTRACT.json';self.write_contract()
        self.counter=0

    @staticmethod
    def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def write_contract(self,**changes):
        value=dict(schema=s.CONTRACT_SCHEMA,expected_active=self.active,maximum_frames=240,
            RAW_peak_exclusive_ADU=240,readback_policy='each_frame_before_and_after',legacy_quantitative_gate_unchanged=True,
            source_sha256={str(s.SELF):self.ref(s.SELF)['sha256']},minimum_post_ACK_ns=1_500_000_000)
        value.update(changes);self.contract_path.write_text(json.dumps(value),encoding='utf-8')

    def fresh_lease(self,**changes):
        value=deepcopy(self.lease);value['updated_monotonic_ns']=time.monotonic_ns();value.update(changes);return value

    def spec(self,frames=2,name=None):
        self.counter+=1
        return dict(schema=s.REQUEST_SCHEMA,request_id='FAKE_%d'%self.counter,frames=frames,
            output=str(self.root/(name or ('burst_%d'%self.counter))),deadline_unix_ns=time.time_ns()+10_000_000_000,
            contract_ref=self.ref(self.contract_path),source_sha256={str(s.SELF):self.ref(s.SELF)['sha256'],
                str(self.contract_path):self.ref(self.contract_path)['sha256']})

    def core(self,frames=2):return s.DenseBurstCore(self.spec(frames),self.active,self.fresh_lease())

    def feed(self,core,index,*,raw=None,meta_changes=None,after=None,lease=None,do_before=True):
        active=self.active
        if do_before:self.assertTrue(core.before_capture(active,self.fresh_lease(),time.time_ns()))
        started=time.time_ns();received=time.time_ns()
        meta=dict(FrameID=100+index,FrameID_raw=100+index,TimeStamp=1000+index*100,
            logical_sequence=1+index,host_get_started_unix_ns=started,host_received_unix_ns=received,
            Width=4,Height=4,Stride=4,BufferSize=16,pixel_format='Mono8',
            manual_preview_only=True,quantitative_valid=False)
        meta.update(meta_changes or {})
        image=np.full((4,4),50+index,np.uint8) if raw is None else raw
        return core.feed(image,meta,active if after is None else after,
            self.fresh_lease() if lease is None else lease,time.time_ns())

    def result(self,core):
        status=core.join(5);self.assertIn(status['state'],('COMPLETE','FAIL'))
        return json.loads((core.output/'BATCH_RESULT.json').read_text())

    def test_success_exact_original_RAW_metadata_and_new_contract_separate(self):
        core=self.core();self.feed(core,0);self.feed(core,1);result=self.result(core)
        self.assertTrue(result['completed']);self.assertEqual(result['valid_frames'],2)
        self.assertEqual(result['captured_frames'],2);self.assertTrue(core.snapshot()['buffer_released'])
        frame=json.loads(Path(result['frames'][0]['metadata']['path']).read_text())
        self.assertTrue(frame['original_metadata']['manual_preview_only'])
        self.assertFalse(frame['original_metadata']['quantitative_valid'])
        self.assertTrue(frame['new_research_frame_valid']);self.assertFalse(frame['control_release_approved'])
        self.assertEqual(frame['source_sha256'],core.spec['source_sha256'])
        self.assertFalse(result['initial_lease']['geometry_frozen'])
        np.testing.assert_array_equal(np.load(result['frames'][0]['raw']['path'],allow_pickle=False),np.full((4,4),50,np.uint8))
        manifest=json.loads((core.output/'MANIFEST.json').read_text())
        for name,digest in manifest['file_sha256'].items():self.assertEqual(self.ref(core.output/name)['sha256'],digest)

    def test_bounds_N1_240_and_734MB_payload(self):
        for invalid in (0,241,True,1.5):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                s.DenseBurstCore(self.spec(invalid),self.active,self.fresh_lease())
        self.active.update(Width='552',Height='554');self.lease['roi'].update(Width=552,Height=554);self.write_contract()
        core=self.core(240);self.assertEqual(core.payload_capacity,73_393_920)
        self.assertLessEqual(core.payload_capacity,s.MAX_PAYLOAD_BYTES)
        core.cancel();self.assertFalse(self.result(core)['completed'])

    def test_payload_limit_prevents_larger_ROI_allocation(self):
        self.active.update(Width='553',Height='554');self.lease['roi'].update(Width=553,Height=554);self.write_contract()
        with self.assertRaises(ValueError):self.core(240)

    def test_missing_or_legacy_Gain_contract_not_implicitly_accepted(self):
        self.write_contract(legacy_quantitative_gate_unchanged=False)
        with self.assertRaises(ValueError):self.core()
        self.write_contract(expected_active=dict(self.active,Gain='0'))
        with self.assertRaises(ValueError):self.core()

    def test_contract_source_closure_is_exact_no_missing_extra_or_changed_bindings(self):
        for action in ('missing','extra','changed'):
            spec=self.spec(1)
            if action=='missing':del spec['source_sha256'][str(self.contract_path)]
            elif action=='extra':spec['source_sha256'][str(self.dmd)]=self.ref(self.dmd)['sha256']
            else:spec['source_sha256'][str(s.SELF)]='a'*64
            with self.subTest(action=action),self.assertRaises(ValueError):s.DenseBurstCore(spec,self.active,self.fresh_lease())

    def test_post_ACK_settle_cannot_be_shortened_by_request_or_contract(self):
        lease=self.fresh_lease();lease['min_frame_start_ns']=lease['ack_verified_ns']+1_499_999_999
        with self.assertRaises(ValueError):s.DenseBurstCore(self.spec(1),self.active,lease)
        self.write_contract(minimum_post_ACK_ns=1)
        with self.assertRaises(ValueError):self.core(1)

    def test_no_future_stale_or_incomplete_lease(self):
        for delta in ({'updated_monotonic_ns':0},{'armed':False},{'fault':'bad'},
                {'verified_revision':19},{'busy_flags':{'gain':True}},{'owner_start':'abc'},
                {'roi':dict(self.lease['roi'],OffsetX=901)}):
            with self.subTest(delta=delta),self.assertRaises(ValueError):
                s.DenseBurstCore(self.spec(),self.active,self.fresh_lease(**delta))

    def test_before_profile_mismatch_stops_with_zero_RAW(self):
        core=self.core();self.assertFalse(core.before_capture(dict(self.active,Gain='0'),self.fresh_lease(),time.time_ns()))
        result=self.result(core);self.assertFalse(result['completed']);self.assertEqual(result['captured_frames'],0)

    def test_after_profile_failure_retains_rejected_RAW(self):
        core=self.core();self.feed(core,0)
        self.feed(core,1,after=dict(self.active,ExposureTime='75'))
        result=self.result(core);self.assertFalse(result['completed'])
        self.assertEqual((result['captured_frames'],result['valid_frames']),(2,1))
        self.assertIsNotNone(result['frames'][1]['raw']);self.assertFalse(result['frames'][1]['valid'])

    def test_changed_input_revision_settings_or_geometry_never_mix(self):
        for change in ({'revision':21,'verified_revision':21},{'settings':{'dmd':'other'}},{'geometry_frozen':True}):
            with self.subTest(change=change):
                core=self.core(1);self.feed(core,0,lease=self.fresh_lease(**change))
                result=self.result(core);self.assertFalse(result['completed']);self.assertEqual(result['captured_frames'],1)

    def test_frame_prior_to_ACK_cutoff_never_accepted(self):
        core=self.core(1)
        self.feed(core,0,meta_changes={'host_get_started_unix_ns':self.lease['min_frame_start_ns']-1})
        result=self.result(core);self.assertFalse(result['completed']);self.assertEqual(result['captured_frames'],1)

    def test_before_waits_for_future_settle_without_sample_or_failure(self):
        self.lease['min_frame_start_ns']=time.time_ns()+100_000_000
        core=self.core(1);self.assertFalse(core.before_capture(self.active,self.fresh_lease(),time.time_ns()))
        self.assertEqual(core.snapshot()['state'],'PENDING');self.assertEqual(core.snapshot()['captured_frames'],0)
        core.cancel();self.result(core)

    def test_missing_pair_and_duplicate_metadata_rejected_not_filled(self):
        core=self.core(2)
        self.assertTrue(core.before_capture(self.active,self.fresh_lease(),time.time_ns()))
        self.assertFalse(core.before_capture(self.active,self.fresh_lease(),time.time_ns()))
        self.result(core)
        core=self.core(3);self.feed(core,0);self.feed(core,1,meta_changes={'TimeStamp':1000,'FrameID':100})
        result=self.result(core);self.assertFalse(result['completed']);self.assertEqual(result['valid_frames'],1)
        self.feed(core,2,do_before=False);self.assertEqual(core.snapshot()['captured_frames'],2)

    def test_raw_headroom_240_strict_and_rejected_frame_archived(self):
        core=self.core(1);a=np.zeros((4,4),np.uint8);a[1,2]=240;self.feed(core,0,raw=a)
        result=self.result(core);self.assertFalse(result['completed']);self.assertEqual(result['captured_frames'],1)
        saved=np.load(result['frames'][0]['raw']['path'],allow_pickle=False);self.assertEqual(saved[1,2],240)

    def test_explicit_incomplete_metadata_is_FAIL_with_original_RAW_kept(self):
        core=self.core(1);self.feed(core,0,meta_changes={'IsIncomplete':True})
        result=self.result(core);self.assertFalse(result['completed']);self.assertEqual(result['captured_frames'],1)
        metadata=json.loads(Path(result['frames'][0]['metadata']['path']).read_text())
        self.assertTrue(metadata['original_metadata']['IsIncomplete']);self.assertFalse(metadata['new_research_frame_valid'])

    def test_native_shape_stride_buffer_and_counter_contract(self):
        changes=({'Stride':3},{'BufferSize':15},{'pixel_format':'RGB8'},{'logical_sequence':True},{'Height':5})
        for change in changes:
            with self.subTest(change=change):
                core=self.core(1);self.feed(core,0,meta_changes=change);self.assertFalse(self.result(core)['completed'])

    def test_raw_source_is_copied_not_later_mutable_alias(self):
        core=self.core(2);a=np.full((4,4),7,np.uint8);self.feed(core,0,raw=a);a.fill(99)
        self.feed(core,1);result=self.result(core)
        np.testing.assert_array_equal(np.load(result['frames'][0]['raw']['path'],allow_pickle=False),np.full((4,4),7,np.uint8))

    def test_fresh_readback_before_and_after_frame_required(self):
        core=self.core(1);self.assertTrue(core.before_capture(self.active,self.fresh_lease(),time.time_ns()))
        self.feed(core,0,meta_changes={'host_received_unix_ns':time.time_ns()+10_000_000},do_before=False)
        self.assertFalse(self.result(core)['completed'])
        core=self.core(1);self.assertFalse(core.before_capture(self.active,self.fresh_lease(),time.time_ns()-1_000_000_000))
        self.assertFalse(self.result(core)['completed'])

    def test_owner_thread_cannot_change(self):
        core=self.core(2);self.feed(core,0);result=[]
        t=threading.Thread(target=lambda:result.append(core.before_capture(self.active,self.fresh_lease(),time.time_ns())))
        t.start();t.join();self.assertEqual(result,[False]);self.assertFalse(self.result(core)['completed'])

    def test_native_frame_ID_gaps_reported_honestly_not_called_contiguous(self):
        core=self.core(2);self.feed(core,0);self.feed(core,1,meta_changes={'FrameID':103})
        result=self.result(core);self.assertTrue(result['completed'])
        meta=json.loads(Path(result['frames'][1]['metadata']['path']).read_text())
        self.assertEqual(meta['native_FrameID_step'],3);self.assertTrue(result['dense_burst_NOT_guaranteed_contiguous_native_frames'])

    def test_cancel_keeps_all_original_prefix(self):
        core=self.core(3);self.feed(core,0);core.cancel('operator stop');result=self.result(core)
        self.assertFalse(result['completed']);self.assertEqual(result['captured_frames'],1)
        self.assertEqual(result['error'],'operator stop')

    def test_no_capture_thread_file_IO_and_writer_does_not_block_status(self):
        core=self.core(1);owner=threading.get_ident();entered=threading.Event();release=threading.Event()
        oldsave=np.save;threads=[]
        def slow_save(*args,**kwargs):
            threads.append(threading.get_ident());entered.set();release.wait(2);return oldsave(*args,**kwargs)
        with patch.object(np,'save',side_effect=slow_save):
            self.feed(core,0);self.assertTrue(entered.wait(1))
            before=time.monotonic();status=core.snapshot();core.cancel();elapsed=time.monotonic()-before
            self.assertEqual(status['state'],'WRITING');self.assertLess(elapsed,.2)
            release.set();result=self.result(core)
        self.assertTrue(result['completed']);self.assertTrue(all(t!=owner for t in threads))

    def test_all_Path_file_IO_is_in_writer_not_capture_callbacks(self):
        core=self.core(1);owner=threading.get_ident();original=Path.open
        def checked_open(path,*args,**kwargs):
            self.assertNotEqual(threading.get_ident(),owner,'capture thread performed file IO')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',new=checked_open):
            self.feed(core,0);status=core.join(5)
        self.assertEqual(status['state'],'COMPLETE')

    def test_source_change_cannot_pass_but_RAW_already_retained(self):
        source=self.root/'bound.py';source.write_text('original')
        self.write_contract(source_sha256={str(s.SELF):self.ref(s.SELF)['sha256'],str(source):self.ref(source)['sha256']})
        spec=self.spec(1);spec['source_sha256'][str(source)]=self.ref(source)['sha256']
        core=s.DenseBurstCore(spec,self.active,self.fresh_lease());source.write_text('changed')
        self.feed(core,0);result=self.result(core);self.assertFalse(result['completed'])
        self.assertEqual(result['captured_frames'],1);self.assertIsNotNone(result['source_recheck_error'])

    def test_writer_failure_retains_buffer_and_blocks_new_coordinator_request(self):
        coordinator=s.DenseBurstCoordinator();core=coordinator.request(self.spec(1),self.active,self.fresh_lease())
        with patch.object(np,'save',side_effect=OSError('disk failed')):
            self.feed(core,0);status=core.join(5)
        self.assertEqual(status['state'],'FAIL');self.assertFalse(status['buffer_released'])
        self.assertIn('disk failed',status['writer_error'])
        with self.assertRaises(ValueError):coordinator.request(self.spec(1),self.active,self.fresh_lease())

    def test_coordinator_one_active_and_fresh_output_only(self):
        c=s.DenseBurstCoordinator();core=c.request(self.spec(1),self.active,self.fresh_lease())
        with self.assertRaises(ValueError):c.request(self.spec(1),self.active,self.fresh_lease())
        self.feed(core,0);self.result(core)
        other=c.request(self.spec(1),self.active,self.fresh_lease());other.cancel();self.result(other)
        with self.assertRaises(ValueError):s.DenseBurstCore(core.spec,self.active,self.fresh_lease())


if __name__=='__main__':unittest.main()
