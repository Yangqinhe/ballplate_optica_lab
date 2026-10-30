"""Offline adapter regression: no SDK, hardware GUI or real output creation."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
sys.path.insert(0,str(LAB/'src'))
import optical_lab
spec=importlib.util.spec_from_file_location('optical_lab.state_sampling_gui',ROOT/'adapter.py')
gui=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=gui
optical_lab.state_sampling_gui=gui
spec.loader.exec_module(gui)
spec=importlib.util.spec_from_file_location('E068_existing_tests',LAB/'tests/test_state_sampling_gui.py')
fake=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake)
StateSamplingGUI=fake.StateSamplingGUI
from optical_lab.state_sampling import DenseBurstCore

class ReadbackOrdering(unittest.TestCase):
    def check_case(self,case):
        test=StateSamplingGUI('test_same_owner_before_original_after_feed_and_metadata_unmodified')
        test.setUp()
        try:
            test.controller.core_factory=DenseBurstCore;test.ready();test.controller.start(1,'ORDER_REGRESSION')
            core=test.controller.current_core();worker=fake.Worker(test.active)
            clock=[gui.time.monotonic_ns()];original=worker.camera.parameters;calls=[0]
            def parameters():
                calls[0]+=1
                if calls[0]==1:
                    clock[0]+=350_000_000
                    if case!='stale':
                        if case=='revision':
                            test.window.revision+=1;test.window.last_verified_revision=test.window.revision
                        test.controller.tick()
                values=original()
                if case=='camera':values['Gain']='3.0'
                return values
            worker.camera.parameters=parameters
            with mock.patch.object(gui.time,'monotonic_ns',side_effect=lambda:clock[0]):
                gui._capture_adapter(fake.Worker.original,test.controller)(worker)
            status=core.join(3)
            self.assertEqual(status['valid_frames'],1 if case=='fresh' else 0)
            self.assertEqual(len(worker.captures),1)
        finally:test.doCleanups()
    def test_refresh_during_slow_read(self):self.check_case('fresh')
    def test_truly_stale_still_rejected(self):self.check_case('stale')
    def test_revision_change_still_rejected(self):self.check_case('revision')
    def test_camera_change_still_rejected(self):self.check_case('camera')

if __name__=='__main__':unittest.main(verbosity=2)
