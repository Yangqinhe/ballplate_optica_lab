"""Offline timing tests with fake clocks/CCD; no device or running-GUI access."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np
from PyQt5 import QtCore
import test_hold as hold
from test_hold import page,c,ROOT,COMPUTE,app


class SettleTests(unittest.TestCase):
    setUp=hold.HoldTests.setUp
    tearDown=hold.HoldTests.tearDown

    def begin(self,ms=200):
        p=self.p;p.settle_ms.setValue(ms);p.go=True;p.started=100.
        p.context=Mock(return_value={'camera':{'Gain':'0'}})
        p.spec=NS(mode='mean',frames=10,alpha=1.)
        p.burst=Mock();self.w.worker=NS(epoch=2);self.w.last_verified_revision=7
        with patch.object(page.time,'monotonic',return_value=100.):p.begin_settle(7)

    def poll_at(self,t):
        with patch.object(page.time,'monotonic',return_value=t):self.p.poll()

    def test_limits_default_and_1ms_manual_precision(self):
        q=self.p.settle_ms
        self.assertEqual((q.minimum(),q.maximum(),q.value()),(10,1000,1000))
        for requested,expected in ((0,10),(10,10),(11,11),(200,200),(1000,1000),(1500,1000)):
            q.setValue(requested);self.assertEqual(q.value(),expected)
        self.assertFalse(q.keyboardTracking())

    def test_setting_persists_without_changing_physics_or_projection(self):
        self.p.settle_ms.setValue(230)
        q=QtCore.QSettings(str(page.PREFERENCES_PATH),QtCore.QSettings.IniFormat)
        self.assertEqual(q.value('settle_ms',type=int),230)
        self.p.send.assert_not_called();self.assertEqual(self.w.revision,7)

    def test_start_clock_only_after_matching_display_verification(self):
        self.begin();self.w.last_verified_revision=6
        self.poll_at(100.5);self.assertIsNone(self.p.settle_at);self.p.burst.arm.assert_not_called()
        self.w.last_verified_revision=7;self.poll_at(100.6)
        self.assertAlmostEqual(self.p.settle_at,100.8)
        self.assertEqual(self.p.settle_timer.interval(),200)
        self.assertEqual(self.p.settle_timer.timerType(),QtCore.Qt.PreciseTimer)

    def test_sampling_never_starts_before_requested_deadline(self):
        self.begin();self.poll_at(100.1);self.poll_at(100.299)
        self.p.burst.arm.assert_not_called();self.poll_at(100.301)
        self.p.burst.arm.assert_called_once();self.assertEqual(self.p.stage,'CAPTURE')
        self.assertEqual(self.p.spec.frames,10)
        self.assertAlmostEqual(self.p.timing['verified_to_arm_ms'],201.)

    def test_edit_during_wait_does_not_shorten_current_input(self):
        self.begin(1000);self.poll_at(100.1);self.p.settle_ms.setValue(10)
        self.poll_at(100.12);self.p.burst.arm.assert_not_called()
        self.assertEqual(self.p.timing['settle_requested_ms'],1000)
        self.poll_at(101.101);self.p.burst.arm.assert_called_once()
        with patch.object(page.time,'monotonic',return_value=102.):self.p.begin_settle(8)
        self.assertEqual(self.p.timing['settle_requested_ms'],10)

    def test_ten_ms_is_not_forced_through_old_40ms_poll(self):
        self.begin(10);self.poll_at(100.1)
        self.assertEqual(self.p.settle_timer.interval(),10)
        self.poll_at(100.111);self.p.burst.arm.assert_called_once()
        self.assertFalse(self.p.settle_timer.isActive())

    def test_early_timer_is_rearmed_for_remaining_time(self):
        self.begin(100);self.poll_at(100.1);self.p.settle_timer.stop()
        self.poll_at(100.199);self.p.burst.arm.assert_not_called()
        self.assertTrue(self.p.settle_timer.isActive());self.assertLessEqual(self.p.settle_timer.interval(),2)

    def test_pause_cancels_timer_and_late_timeout_cannot_capture(self):
        self.begin();self.poll_at(100.1);self.p.pause()
        self.assertFalse(self.p.settle_timer.isActive())
        self.poll_at(101.);self.p.burst.arm.assert_not_called()

    def test_new_projection_or_display_fault_still_stops_capture(self):
        self.begin();self.poll_at(100.1);self.w.revision=8;self.poll_at(100.4)
        self.assertFalse(self.p.go);self.p.burst.arm.assert_not_called()
        self.assertIn('投图发生改变',self.p.info.text())

    def test_same_input_armed_only_once(self):
        self.begin();self.p.burst.snapshot.return_value=(None,0,None)
        for t in (100.1,100.301,100.302,100.34):self.poll_at(t)
        self.p.burst.arm.assert_called_once()

    def test_per_capture_metadata_preserves_requested_and_actual_wait(self):
        @dataclass
        class Spec:
            mode:str='mean'
            frames:int=2
            alpha:float=1.
        raw=np.arange(8,dtype=np.uint8).reshape(2,2,2)
        result=dict(raw=raw,image=raw.mean(axis=0),meta=[{'FrameID':1},{'FrameID':2}])
        timing=dict(settle_requested_ms=200,verified_to_arm_ms=204.5)
        path=Path(self.temp.name)/'capture.npz'
        page.save_capture(path,result,Spec(),{'camera':{}},9,timing)
        timing['settle_requested_ms']=10
        meta=c.read(path.with_suffix('.json'))
        self.assertEqual(meta['sampling_timing']['settle_requested_ms'],200)
        self.assertEqual(meta['sampling_timing']['verified_to_arm_ms'],204.5)
        self.assertTrue(meta['independent_filter_reset'])
        with np.load(path) as a:np.testing.assert_array_equal(a['raw'],raw)


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(hold.HoldTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(SettleTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    c.write(ROOT/'TEST_RESULT.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        page=c.ref(ROOT/'page.py'),test=c.ref(__file__),hardware_actions=0))
    raise SystemExit(not result.wasSuccessful())
