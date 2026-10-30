"""Offscreen observation latching; no SDK, device windows or live controls."""
from copy import deepcopy
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import numpy as np
import run
from run import c,ROOT,PRIOR,COMPUTE
from PyQt5 import QtWidgets
import page

app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
FOLDER=ROOT.parent/'E108_paired_display_fix_20261030_1/PAIRED_RUNS/RUN_1793353034436016112/E001_T000000_Q000003'
RECORD=c.read(FOLDER/'STEP.json')
with np.load(FOLDER/'observation.npz') as a:IMAGES={k:a[k].copy() for k in a.files}
with np.load(c.checked(RECORD['real_capture'])) as a:MEASURED=a['image'].copy()


def pixels(widget):
    im=widget.image
    return None if im is None else im.bits().asstring(im.byteCount())


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.w=QtWidgets.QWidget();w=self.w;w.tabs=QtWidgets.QTabWidget(w)
        for title in ('main','stability'):w.tabs.addTab(QtWidgets.QWidget(),title)
        w.stop_button=QtWidgets.QPushButton(w);w.hardware=False;w.fault=None;w.running=True
        w.armed=QtWidgets.QCheckBox(w);w.armed.setChecked(True);w.revision=7
        params=dict(mode='fixed',gain=128.,black=10.,gamma=1.)
        controls=NS(mode=NS(currentData=lambda:'fixed'),parameters=lambda *_:params)
        w.manual_display=NS(controls=controls);w.enhance=NS(isChecked=lambda:True)
        self.temp=__import__('tempfile').TemporaryDirectory()
        self.preferences_patch=__import__('unittest.mock',fromlist=['patch']).patch.object(page,'PREFERENCES_PATH',Path(self.temp.name)/'ui.ini')
        self.preferences_patch.start()
        self.p=page.Controller(w,c.ref(COMPUTE/'HEADS_02/PACKAGE.json'),ROOT/'NOT_USED_BY_TESTS')
        self.p.timer.stop();self.p.send=Mock();self.p.images={k:v.copy() for k,v in IMAGES.items()}
    def tearDown(self):
        self.p.close();self.w.deleteLater();app.processEvents();self.preferences_patch.stop();self.temp.cleanup()
    def committed(self):
        self.p.pending_ccd=MEASURED.copy();self.p.commit_view(RECORD)
    def signature(self):
        return [(pixels(v['dmd']),pixels(v['slm']),pixels(v['ccd']),None if v['bars'].values is None else v['bars'].values.copy()) for v in self.p.views]
    def same(self,before):
        for actual,old in zip(self.signature(),before):
            self.assertEqual(actual[:3],old[:3]);np.testing.assert_array_equal(actual[3],old[3])
    def test_first_window_has_explicit_wait_not_fake_image(self):
        self.p.hold_view('采集中')
        self.assertIn('等待首个',self.p.views[1]['ccd'].title);self.assertIsNone(self.p.views[1]['ccd'].image)
    def test_all_wait_phases_preserve_complete_observation(self):
        self.committed();old=self.signature()
        for phase in ('等待下一帧','换图稳定中','采集中','归档与积分中','已暂停'):
            self.p.hold_view(phase);self.same(old)
            self.assertIn('上一有效帧',self.p.views[1]['ccd'].title);self.assertIn('t=0.00s',self.p.views[1]['ccd'].title)
            self.assertIn('不是正在采集',self.p.note.text())
    def test_prepare_does_not_blank_or_replace_previous_model_and_raw(self):
        self.committed();old=self.signature();self.p.busy=dict(id=42,op='prepare')
        self.p.accept(dict(id=42,ok=True,result=dict(archive=str(FOLDER/'observation.npz'),states=RECORD['states_after'])))
        self.same(old);self.p.send.assert_called_once_with('cancel')
    def test_complete_pair_commits_together_and_marks_time(self):
        self.committed();old=pixels(self.p.views[1]['ccd'])
        self.p.images['simulation_CCD']=self.p.images['simulation_CCD']+1.25
        self.p.pending_ccd=MEASURED+1.25
        record=deepcopy(RECORD);record['step']=2
        for state in record['states_before']:state['time']=.02
        record['real_sums']=c.integrate(self.p.pending_ccd,self.p.package['layout']).tolist()
        record['simulation_sums']=c.integrate(self.p.images['simulation_CCD'],self.p.package['layout']).tolist()
        self.p.commit_view(record)
        self.assertNotEqual(pixels(self.p.views[1]['ccd']),old);self.assertIsNone(self.p.pending_ccd)
        for v,key in zip(self.p.views,('simulation_sums','real_sums')):
            self.assertIn('t=0.02s',v['ccd'].title);np.testing.assert_array_equal(v['bars'].values,record[key])
    def test_pause_keeps_previous_result_and_drops_unconsumed_candidate(self):
        self.committed();old=self.signature();self.p.pending_ccd=MEASURED+2
        self.p.pause();self.same(old);self.assertIsNone(self.p.pending_ccd)
        self.assertIn('已暂停',self.p.views[1]['ccd'].title)
    def test_pause_inflight_atomic_advance_keeps_its_candidate(self):
        self.committed();self.p.pending_ccd=MEASURED+2;self.p.busy=dict(id=4,op='advance')
        self.p.pause();self.assertIsNotNone(self.p.pending_ccd)
    def test_error_retains_labeled_result_without_reusing_it(self):
        self.committed();old=self.signature();self.p.fail('fixture timeout')
        self.same(old);self.assertFalse(self.p.go);self.assertIn('fixture timeout',self.p.info.text())
    def test_confirmed_reset_clears_prior_episode_images_and_tooltips(self):
        self.committed();self.p.busy=dict(id=3,op='reset')
        self.p.accept(dict(id=3,ok=True,result=dict(states=RECORD['states_before'])))
        self.assertIsNone(self.p.presented)
        for v in self.p.views:
            for key in ('ccd','dmd','slm'):self.assertIsNone(v[key].image)
            self.assertIsNone(v['bars'].values);self.assertEqual(v['bars'].toolTip(),'')
    def test_no_candidate_cannot_relabel_old_frame_as_new(self):
        self.committed();old=self.signature()
        with self.assertRaisesRegex(ValueError,'新有效帧'):self.p.commit_view(RECORD)
        self.same(old)
    def test_capture_completion_not_shown_before_step_is_complete(self):
        self.committed();old=self.signature();self.p.go=True;self.p.started=time.monotonic()
        self.p.stage='CAPTURE';self.p.revision=7;self.p.capture_at=time.monotonic();self.p.spec=NS(mode='mean',frames=10,alpha=1.)
        self.p.ctx={};self.p.observation=dict(archive=str(FOLDER/'observation.npz'))
        self.p.burst=Mock();self.p.burst.snapshot.return_value=(dict(image=MEASURED+2),10,None)
        self.p.writer.shutdown();self.p.writer=Mock();self.p.poll()
        self.assertEqual(self.p.stage,'SAVING');self.same(old)
        np.testing.assert_array_equal(self.p.pending_ccd,MEASURED+2)
    def test_display_buffer_and_control_arrays_are_unchanged(self):
        self.p.pending_ccd=MEASURED.copy();before=self.p.pending_ccd.copy();image_before=self.p.images['simulation_CCD'].copy()
        self.p.commit_view(RECORD);np.testing.assert_array_equal(MEASURED,before)
        np.testing.assert_array_equal(self.p.images['simulation_CCD'],image_before)
        np.testing.assert_array_equal(c.integrate(MEASURED,self.p.package['layout']),RECORD['real_sums'])
    def test_screenshot_waiting_retains_visible_effective_frame(self):
        self.committed();self.p.hold_view('采集中');self.p.page.resize(1820,1040);self.p.page.show();app.processEvents()
        self.assertFalse(self.p.views[1]['ccd'].image.isNull())
        self.p.page.grab().save(str(ROOT/'waiting_preview_QA.png'))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HoldTests))
    c.write(ROOT/'HOLD_TEST_RESULT.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,page=c.ref(ROOT/'page.py'),
        fixture=c.ref(FOLDER/'STEP.json'),no_new_hardware=True,real_data_replay_only=True))
    raise SystemExit(not result.wasSuccessful())
