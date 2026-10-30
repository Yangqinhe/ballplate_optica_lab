"""No hardware tests: math, finite filter windows, provenance and state semantics."""
from copy import deepcopy
import sys
import unittest
import numpy as np
import core as c

sys.path.insert(0,str(c.LAB/'src'))
c.load(c.EXP/'E081_readout_regions_20261028_1/staged/optical_lab/effective_frame.py','optical_lab.effective_frame')
from optical_lab.effective_frame import FilterSpec


class MathTests(unittest.TestCase):
    def setUp(self):self.package=c.read(c.ROOT/'HEADS_02/PACKAGE.json');self.layout=self.package['layout']
    def test_import_no_hardware(self):
        for module in ('PySpin','torch','mujoco','PyQt5'):self.assertNotIn(module,sys.modules)
    def test_full_ROI_partition(self):
        x=np.ones(self.layout['shape']);z=c.integrate(x,self.layout)
        self.assertEqual(z.sum(),x.size);self.assertEqual(len(z),64)
        np.testing.assert_equal(z,[r['pixels'] for r in self.layout['regions']])
    def test_integration_matches_frozen_dataset(self):
        sys.path.insert(0,str(c.MODEL))
        old=c.load(c.MODEL/'dataset.py','e107_test_dataset')
        x=np.random.default_rng(0).random(self.layout['shape'])*255
        np.testing.assert_allclose(c.integrate(x,self.layout),old.integrate(x,self.layout),rtol=1e-11,atol=1e-7)
    def test_negative_or_nan_invalid_input(self):
        for x in (np.zeros((2,2)),np.full(self.layout['shape'],np.nan)):
            with self.assertRaises(ValueError):c.integrate(x,self.layout)
    def test_heads_130_and_affine(self):
        for source in self.package['heads'].values():
            h=c.read(c.checked(source));self.assertEqual(h['parameters'],130)
            a=np.arange(64,dtype=float);b=np.ones(64)
            np.testing.assert_allclose(c.action(a+b,h)-c.action(a,h),b@np.array(h['weights_sum']),atol=1e-12)
    def test_history_ages(self):
        h=c.History()
        for i in range(10):
            a=np.zeros((1,96,96),np.uint8);a[0,0,i]=255;r=h.push(a)
        self.assertEqual([np.argmax(x) for x in r],[0,3,6,9])
    def test_history_initial_repeat_and_clear(self):
        a=np.zeros((1,96,96),np.uint8);a[0,0,4]=255
        x=c.History().push(a);self.assertTrue(all(np.array_equal(y,a) for y in x))
        a[:]=255;self.assertFalse(np.array_equal(x[0],a))
    def test_impulse_units(self):
        np.testing.assert_allclose(c.impulse_velocity(.005,0,.05),[.1,0],atol=1e-12)
        np.testing.assert_allclose(c.impulse_velocity(.005,90,.05),[0,.1],atol=1e-12)
    def test_random_disk(self):
        rng=np.random.default_rng(2);x=np.array([c.random_xy(rng,12) for _ in range(1000)])
        self.assertLessEqual(np.linalg.norm(x,axis=1).max(),.12)
        self.assertAlmostEqual(float((x*x).sum(1).mean()),.12**2/2,delta=.001)


class BurstTests(unittest.TestCase):
    def setUp(self):self.b=c.Burst();self.a={'Gain':'0'}
    def arm(self,mode='mean',n=2,alpha=.5):return self.b.arm(12,3,self.a,FilterSpec(mode,n,alpha),(2,2))
    def feed(self,t,value,stamp,active=None):
        self.b.feed(t,np.full((2,2),value,np.uint8),dict(TimeStamp=stamp,FrameID=stamp),12,3,active or self.a)
    def test_mean_keeps_float(self):
        t=self.arm();self.feed(t,10,1);self.feed(t,11,2)
        result,n,error=self.b.snapshot();np.testing.assert_equal(result['image'],10.5)
        self.assertEqual(n,2);self.assertIsNone(error);self.assertEqual(result['raw'].dtype,np.uint8)
    def test_ema_from_current_window_only(self):
        t=self.arm('ema',2,.5);self.feed(t,10,1);self.feed(t,20,2)
        np.testing.assert_equal(self.b.snapshot()[0]['image'],15)
        t=self.arm('ema',2,.5);self.feed(t,40,3);self.feed(t,40,4)
        np.testing.assert_equal(self.b.snapshot()[0]['image'],40)
    def test_mean_N120(self):
        t=self.arm('mean',120)
        for i in range(120):self.feed(t,i,i)
        np.testing.assert_equal(self.b.snapshot()[0]['image'],59.5)
    def test_stale_inflight_discarded(self):
        old=self.arm();new=self.arm();self.feed(old,255,1)
        self.assertEqual(self.b.snapshot()[1],0);self.feed(new,4,2);self.feed(new,4,3)
        np.testing.assert_equal(self.b.snapshot()[0]['image'],4)
    def test_cancel_clears_history(self):
        t=self.arm();self.feed(t,200,1);self.b.cancel();self.feed(t,255,2)
        self.assertIsNone(self.b.snapshot()[0]);self.assertEqual(self.b.snapshot()[1],0)
    def test_duplicate_timestamp_stops_not_old_frame(self):
        t=self.arm();self.feed(t,10,1);self.feed(t,11,1)
        self.assertIsNotNone(self.b.snapshot()[2]);self.assertIsNone(self.b.snapshot()[0])
    def test_camera_change_rejected(self):
        t=self.arm();self.feed(t,10,1,{'Gain':'1'})
        self.assertIsNotNone(self.b.snapshot()[2]);self.assertIsNone(self.b.snapshot()[0])
    def test_saturation_warning_only(self):
        t=self.arm();self.feed(t,255,1);self.feed(t,255,2)
        self.assertIsNone(self.b.snapshot()[2]);np.testing.assert_equal(self.b.snapshot()[0]['image'],255)
    def test_only_one_result_no_extra_frames(self):
        t=self.arm();self.feed(t,10,1);self.feed(t,12,2);self.feed(t,200,3)
        self.assertEqual(self.b.snapshot()[1],2);np.testing.assert_equal(self.b.snapshot()[0]['image'],11)
    def test_spec_invalid(self):
        for mode,n,alpha in [('mean',0,.5),('mean',121,.5),('ema',2,0),('off',2,.5)]:
            with self.assertRaises(ValueError):self.arm(mode,n,alpha)

if __name__=='__main__':unittest.main()
