import ctypes as C
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'live_manual_v17')]
from live_loop_v23.core import FrameOrder,PushQueue,validate_push,limited_action,load_head,DEFAULT_HEAD,DEFAULT_SHA


class OrderTests(unittest.TestCase):
    def test_normal(self):
        f=FrameOrder();self.assertEqual(f.accept(1,100)['logical_sequence'],1)
        self.assertEqual(f.accept(3,200)['logical_sequence'],2)

    def test_wrap_raw_retained(self):
        f=FrameOrder();f.accept(65535,125000000)
        r=f.accept(1,127100000)
        self.assertEqual(r['FrameID_raw'],1);self.assertEqual(r['logical_sequence'],2)
        self.assertTrue(r['frame_id_wrap']);self.assertEqual(r['observed_16bit_wraps'],1)

    def test_duplicates_rejected(self):
        f=FrameOrder();f.accept(5,100)
        with self.assertRaises(RuntimeError):f.accept(5,200)

    def test_timestamp_rejected(self):
        for fid in (6,1):
            f=FrameOrder();f.accept(65535,125000000)
            with self.assertRaises(RuntimeError):f.accept(fid,125000000)

    def test_random_reset_not_wrap(self):
        for previous,current in ((3000,1),(65000,255),(65535,1000)):
            f=FrameOrder();f.accept(previous,125000000)
            with self.assertRaises(RuntimeError):f.accept(current,127100000)

    def test_large_gap_wrap_rejected(self):
        f=FrameOrder();f.accept(65535,125000000)
        with self.assertRaises(RuntimeError):f.accept(1,625000001)

    def test_native64_counter(self):
        f=FrameOrder();f.accept(65535,100);r=f.accept(65536,200)
        self.assertFalse(r['frame_id_wrap'])


class PushTests(unittest.TestCase):
    def test_no_push_while_stopped(self):
        with self.assertRaises(RuntimeError):PushQueue().add(0,1,.06)
    def test_take_once(self):
        q=PushQueue();q.begin();e=q.add(1,-1,.06)
        self.assertEqual(q.take(),[e]);self.assertEqual(q.take(),[])
    def test_end_abandons(self):
        q=PushQueue();q.begin();q.add(0,1,.06)
        self.assertEqual(len(q.end()),1);self.assertEqual(q.take(),[])
    def test_close_preserves_unapplied_evidence(self):
        q=PushQueue();q.begin();q.add(0,1,.06);q.close()
        with self.assertRaises(RuntimeError):q.add(0,1,.06)
        self.assertEqual(len(q.end()),1)
    def test_bound_queue(self):
        q=PushQueue();q.begin()
        for i in range(8):q.add(0,1,.06)
        with self.assertRaises(RuntimeError):q.add(0,1,.06)
    def test_bad_magnitude(self):
        for speed in (0,.16,float('nan'),float('inf')):
            q=PushQueue();q.begin()
            with self.assertRaises(ValueError):q.add(0,1,speed)
    def test_no_duplicate_ids(self):
        q=PushQueue();q.begin();a=q.add(0,1,.06);b=q.add(0,1,.06)
        self.assertNotEqual(a['id'],b['id'])
    def test_illegal_axis(self):
        q=PushQueue();q.begin()
        with self.assertRaises(ValueError):q.add(2,1,.06)


class ActorTests(unittest.TestCase):
    def test_original_head(self):
        h,b=load_head(DEFAULT_HEAD,DEFAULT_SHA)
        self.assertEqual(h['parameter_count'],18);self.assertEqual(b.data['input_frame_stride'],3)
    def test_slew_and_limit(self):
        h=dict(weight_raw_ADU_to_rad=np.ones((2,8)).tolist(),bias_rad=[0,0])
        raw,a=limited_action(h,np.ones(8),[0,0]);np.testing.assert_equal(a,[.08,.08])
        raw,a=limited_action(h,np.ones(8),[.79,.79]);np.testing.assert_equal(a,[.8,.8])
    def test_nan_no_fallback(self):
        h=dict(weight_raw_ADU_to_rad=np.ones((2,8)).tolist(),bias_rad=[0,0])
        with self.assertRaises(ValueError):limited_action(h,np.full(8,np.nan),[0,0])
    def test_batch_not_one_frame(self):
        h=dict(weight_raw_ADU_to_rad=np.ones((2,8)).tolist(),bias_rad=[0,0])
        with self.assertRaises(ValueError):limited_action(h,np.ones((2,8)),[0,0])
    def test_invalid_prior_command(self):
        h=dict(weight_raw_ADU_to_rad=np.ones((2,8)).tolist(),bias_rad=[0,0])
        with self.assertRaises(ValueError):limited_action(h,np.ones(8),[2,2])


class CaptureTests(unittest.TestCase):
    def fake(self,ids=(65535,1),incomplete=False,mono=True):
        from live_loop_v23.camera import StreamingCamera
        camera=object.__new__(StreamingCamera);camera.acquiring=True;camera.camera=C.c_void_p(3)
        camera.frame_order=FrameOrder();camera.released=0;camera.i=0
        backing=(C.c_uint8*12)(*range(12));camera.backing=backing
        def call(name,*args):
            if name=='spinCameraGetNextImageEx':args[-1]._obj.value=7
            elif name=='spinImageIsIncomplete':args[-1]._obj.value=int(incomplete)
            elif name=='spinImageGetPixelFormatName':args[1].value=b'Mono8' if mono else b'Mono16'
            elif name=='spinImageGetData':args[-1]._obj.value=C.addressof(backing)
            elif name=='spinImageGetFrameID':args[-1]._obj.value=ids[camera.i]
            elif name=='spinImageGetTimeStamp':args[-1]._obj.value=125000000+camera.i*2100000
            elif name=='spinImageGetStatus':args[-1]._obj.value=9
            elif name=='spinImageGetStatusDescription':args[1].value=b'test incomplete'
            elif name=='spinImageRelease':camera.released+=1;camera.i+=1
            else:
                key=name.removeprefix('spinImageGet')
                args[-1]._obj.value={'Width':3,'Height':3,'Stride':4,'BufferSize':12}[key]
        camera.call=call
        return camera
    def test_copy_geometry_and_wrap(self):
        c=self.fake();a,m=c.capture();b,n=c.capture()
        np.testing.assert_equal(a,[[0,1,2],[4,5,6],[8,9,10]])
        self.assertEqual(n['FrameID'],1);self.assertTrue(n['frame_id_wrap']);self.assertEqual(c.released,2)
    def test_duplicate_releases(self):
        c=self.fake((2,2));c.capture()
        with self.assertRaises(RuntimeError):c.capture()
        self.assertEqual(c.released,2)
    def test_incomplete_releases(self):
        from spin_c import IncompleteFrame
        c=self.fake(incomplete=True)
        with self.assertRaises(IncompleteFrame):c.capture()
        self.assertEqual(c.released,1)
    def test_wrong_format_releases(self):
        c=self.fake(mono=False)
        with self.assertRaises(RuntimeError):c.capture()
        self.assertEqual(c.released,1)


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtWidgets
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(['loop-tests','-platform','offscreen'])
    def test_default_no_hardware(self):
        from live_loop_v23.direct_alignment_gui import LoopWindow
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'unused'
            w=LoopWindow(target,False);self.assertIsNone(w.worker);self.assertFalse(w.windows)
            self.assertFalse(w.loop_page.start.isEnabled());self.assertFalse(target.exists());w.close()
    def test_push_buttons_gated(self):
        from live_loop_v23.page import LoopPage
        w=LoopPage();self.assertTrue(all(not b.isEnabled() for b in w.push_buttons))
        w.set_busy(True);self.assertTrue(all(b.isEnabled() for b in w.push_buttons));w.close()
    def test_latest_view_retains_all_steps(self):
        from live_loop_v23.page import LoopPage
        w=LoopPage();w.set_busy(True)
        rows=[dict(step=i,simulation_time_s=.02*(i+1),position_m=[.01,0],velocity_world_m_s=[0,0],
            push_events_applied=[],raw_integrals8=[1]*8,action_limited_rad=[0,0],reward=1.) for i in range(3)]
        w.on_view(dict(kind='input',trace_rows=rows,sequence=3));self.assertEqual(len(w.trace.rows),3)
        w.on_view(dict(kind='input',trace_rows=rows,sequence=3));self.assertEqual(len(w.trace.rows),3);w.close()
    def test_actual_and_ideal_view_no_raw_mutation(self):
        from live_loop_v23.page import LoopPage
        w=LoopPage();w.load_head();w.set_busy(True)
        raw=np.full((554,552),20,np.uint8);saved=raw.copy()
        row=dict(kind='optical',actual=raw,ideal=raw.astype(float),actual_sequence=0,sequence=0,
            frame_id=123,simulation_time_s=0,raw_integrals8=[1]*8)
        w.on_view(row);w.gain.setValue(32);w.black.setValue(5)
        np.testing.assert_equal(raw,saved);self.assertIsNotNone(w.views['actual'].image)
        self.assertIn('步0',w.labels['actual'].text());w.close()


if __name__=='__main__':unittest.main()
