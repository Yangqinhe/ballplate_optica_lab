"""Offscreen UI and fake-camera tests. No real SDK or output windows."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from PyQt5 import QtCore,QtWidgets
from .contract import PROJECT,validate,matching
from .protocol import PreparedBundle,pack_dmd,fit_zero_gain,sums
from .worker import DeploymentWorker
from .direct_alignment_gui import DeploymentWindow
from .page import detector_overlay
from live_manual_v19_trigger import test_trigger as fixtures

CATALOG=PROJECT/'outputs/deployment_page_v20_20261020_v1/PREPARED_v3/catalog.json'
CONTRACT=PROJECT/'outputs/deployment_page_v20_20261020_v1/RETAINED_CONTRACT.json'
GATES=json.loads((PROJECT/'outputs/live_calibration/LINE0_LEARNED_STATIC_A_20261020_v1/SETUP_AND_ANALYSIS_PLAN.json').read_text())


class PureTests(unittest.TestCase):
    def test_exact_chronological_four_frame_packing(self):
        raw=np.zeros((4,1,96,96),np.uint8);raw[0]=255;raw[3]=255
        dmd=pack_dmd(raw)
        self.assertEqual(dmd.shape,(600,600));self.assertEqual(dmd.dtype,np.uint8)
        self.assertEqual(dmd[:12].sum(),0);self.assertEqual(dmd[12:300,12:300].min(),255)
        self.assertEqual(dmd[12:300,300:588].max(),0);self.assertEqual(dmd[300:588,300:588].min(),255)
        with self.assertRaises(ValueError):pack_dmd(raw[:3])
    def test_one_gain_is_bounded_and_zero_only(self):
        p=np.arange(20,dtype=float).reshape(4,5)
        self.assertAlmostEqual(fit_zero_gain(p*.7,p),.7)
        with self.assertRaises(ValueError):fit_zero_gain(p*10,p)
        with self.assertRaises(ValueError):fit_zero_gain(p,np.zeros_like(p))
    def test_fixed_detector_overlay_never_modifies_raw(self):
        a=np.arange(48,dtype=np.uint8).reshape(6,8);old=a.copy()
        w=np.zeros((8,6,8),np.float32)
        for k in range(8):w[k,:,k]=1
        shown=detector_overlay(a,w)
        self.assertEqual(shown.shape,(6,8,3));np.testing.assert_array_equal(a,old)
    def test_prepared_sources_and_exact_macro_pixels(self):
        if not CATALOG.exists():self.skipTest('local prepared bundle unavailable')
        b=PreparedBundle(CATALOG);b.verify();plan=b.plan([0,16])
        self.assertEqual(plan['case_ids'],[8,0,16]);self.assertEqual(plan['frames_per_stage'],8)
        self.assertEqual(plan['settle_seconds'],1.5);self.assertFalse(plan['closed_loop_approved'])
        self.assertTrue(plan['old_actor_NOT_loaded']);self.assertEqual(b.weights.shape[0],8)
    def test_catalog_mutation_is_not_resigned(self):
        if not CATALOG.exists():self.skipTest('local bundle unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog.json';path.write_bytes(CATALOG.read_bytes());b=PreparedBundle(path)
            value=json.loads(path.read_text());value['some_new_field']=1;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):b.verify()
            with self.assertRaises(ValueError):b.plan([8])
    def test_current_contract_never_refits_or_ignores_settings(self):
        if not CONTRACT.exists():self.skipTest('local contract unavailable')
        c=validate(json.loads(CONTRACT.read_text()))
        matching(c,c['settings'],c['optical_config'],c['expected_active'],c['registration'])
        bad=dict(c['settings'],y=c['settings']['y']+1)
        with self.assertRaises(ValueError):matching(c,bad,c['optical_config'],c['expected_active'],c['registration'])


class FakeBundle:
    def __init__(self,zero,learned,weights,active):
        self.contract=dict(settings=dict(dmd='fiducial',square=128,slm='lens',full_panel=False,x=0,y=0,focal_mm=400.,period=36),expected_active=active)
        self.weights=weights;self.phase=np.zeros((600,600),np.uint8)
        self.cases={8:dict(dmd_array=np.zeros((600,600),np.uint8),predictions=dict(zero=zero,learned=learned))}
    def verify(self):pass
    def plan(self,ids):return dict(maximum_seconds=20,case_ids=[8],frames_per_stage=8,settle_seconds=0,rules=GATES)


class StaticWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)/'batch'
        self.camera=fixtures.FakeCamera();self.camera.write_trace=[];self.camera.acquiring=True
        self.factory=Mock(return_value=self.camera)
        self.worker=DeploymentWorker({},Path(self.tmp.name)/'camera',4000,fixtures.ROI,camera_factory=self.factory)
        self.worker.camera=self.camera;self.worker.startup_dark_required=False
        self.stage=None;self.n=0;self.calls=[];self.fail=None;self.ack=True
        self.zero=np.indices((6,8)).sum(0).astype(float)+12
        self.learned=np.flip(self.zero,1).copy()+np.arange(8)[None]*.4
        w=np.zeros((8,6,8),np.float32)
        for k in range(8):w[k,:,k]=1
        self.bundle=FakeBundle(self.zero,self.learned,w,self.camera.active.copy())
        self.result=[];self.events=[];self.rng=np.random.default_rng(382)
        self.worker.static_result.connect(self.result.append,QtCore.Qt.DirectConnection)
        self.worker.static_progress.connect(self.progress,QtCore.Qt.DirectConnection)
        self.worker.show=self.show
        self.camera.capture=self.capture
    def tearDown(self):self.tmp.cleanup()
    def progress(self,e):
        self.events.append(e)
        if e['kind']=='stage_start':self.stage=e['stage'];self.n=0
    def show(self,kind,dmd=None,slm=None):
        self.calls.append(kind)
        if kind=='deployment_frame' and self.stage=='learned_1':
            self.assertTrue((self.folder/'FROZEN_ZERO_ONLY_GAIN.json').exists())
        return dict(video_verification=dict(actual_framebuffer_checked=self.ack,verified_unix_ns=time.time_ns()))
    def capture(self,timeout_ms):
        self.n+=1;self.camera.captures+=1
        if self.fail and self.stage=='learned_1':raise self.fail
        signal=self.learned if self.stage.startswith('learned') else self.zero
        if self.stage.startswith('background'):signal=np.zeros_like(self.zero)
        a=np.clip(np.round(10+signal+self.rng.normal(0,.5,self.zero.shape)),0,255).astype(np.uint8)
        return a,dict(FrameID=self.camera.captures,TimeStamp=self.camera.captures*16_670_000,
            host_get_started_unix_ns=time.time_ns(),host_received_unix_ns=time.time_ns())
    def run_capture(self):self.worker.collect_static(self.bundle,self.bundle.plan([8]),self.folder)
    def test_same_camera_transaction_gain_frozen_before_learned_and_all_raws_retained(self):
        self.run_capture();self.factory.assert_not_called()
        self.assertTrue(self.result[0]['completed']);self.assertIsNone(self.result[0]['error'])
        self.assertEqual(len(list(self.folder.glob('input_08/*/CCD_raw_*.npy'))),56)
        self.assertEqual(self.result[0]['camera_node_writes_during_test'],0)
        self.assertEqual(self.calls[-1],'deployment_park');self.assertTrue(self.camera.acquiring)
        gain=json.loads((self.folder/'FROZEN_ZERO_ONLY_GAIN.json').read_text())
        self.assertTrue(gain['frozen_before_any_learned_projection'])
        self.assertFalse(self.result[0]['release_approved'])
    def test_cancel_parks_without_restarting_camera_or_failing_away_saved_raws(self):
        def cancel(e):
            if e['kind']=='sample_saved' and e['stage']=='zero_start' and e['completed_frames']==2:self.worker.cancel_static()
        self.worker.static_progress.connect(cancel,QtCore.Qt.DirectConnection)
        self.run_capture();self.assertTrue(self.result[0]['cancelled'])
        self.assertEqual(len(list(self.folder.glob('input_08/zero_start/*.npy'))),2)
        self.assertEqual(self.calls[-1],'deployment_park');self.assertTrue(self.camera.acquiring)
    def test_first_incomplete_stops_without_retry(self):
        self.fail=fixtures.IncompleteFrame(dict(frame_id=17,status=9))
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertEqual(len(self.result[0]['rejected_incomplete_frames']),1)
        self.assertEqual(self.n,1);self.assertFalse((self.folder/'input_08/zero_return_1').exists())
    def test_first_timeout_stops_without_retry(self):
        self.fail=RuntimeError('spinCameraGetNextImageEx: -1011: timeout')
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertIn('-1011',self.result[0]['error']);self.assertEqual(self.n,1)
    def test_bright_saved_static_sample_is_rejected_without_replacement(self):
        old=self.camera.capture
        def bright(timeout_ms):
            a,m=old(timeout_ms)
            if self.stage=='learned_1':a[0,0]=240
            return a,m
        self.camera.capture=bright
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertEqual(self.n,1)
        raw=self.folder/'input_08/learned_1/CCD_raw_001.npy'
        self.assertTrue(raw.exists());self.assertEqual(int(np.load(raw).max()),240)
    def test_stale_frame_is_saved_but_never_accepted(self):
        old=self.camera.capture
        def stale(timeout_ms):
            a,m=old(timeout_ms);m['host_get_started_unix_ns']=0
            return a,m
        self.camera.capture=stale
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertEqual(self.camera.captures,1);self.assertTrue((self.folder/'input_08/background_start/CCD_raw_001.npy').exists())
    def test_no_actual_display_ack_means_no_sample_capture(self):
        self.ack=False
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertEqual(self.camera.captures,0);self.assertIn('实际读回',self.result[0]['error'])
    def test_active_parameter_mismatch_stops_before_projection(self):
        self.bundle.contract['expected_active']=dict(self.camera.active,ExposureTime='2000')
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertNotIn('deployment_frame',self.calls)
    def test_stop_command_is_cancel_not_camera_fault(self):
        self.worker.stop();self.run_capture()
        self.assertTrue(self.result[0]['cancelled']);self.assertEqual(self.camera.captures,0)
    def test_second_pending_job_is_rejected(self):
        self.worker.request_static(self.bundle,[8],self.folder)
        with self.assertRaises(RuntimeError):self.worker.request_static(self.bundle,[8],self.folder)


class TabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(['v20-test','-platform','offscreen'])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.w=DeploymentWindow(Path(self.tmp.name)/'unused',False)
        self.w.timer.stop()
    def tearDown(self):
        self.w.hardware=False;self.w.deployment_busy=False;self.w.worker=None;self.w.running=False
        self.w.close();self.w.deleteLater();self.app.processEvents();self.tmp.cleanup()
    def test_two_tabs_share_common_controls_without_any_camera_or_output_creation(self):
        self.assertEqual(self.w.tabs.count(),2)
        self.assertEqual(self.w.tabs.tabText(1),'离线部署测试')
        self.assertIsNotNone(self.w.centralWidget().layout().itemAt(1).layout())
        self.assertIsNone(self.w.worker);self.assertFalse(self.w.windows);self.assertFalse(self.w.output.exists())
    def test_switching_tabs_does_not_project_or_open_devices(self):
        with patch.object(self.w,'black_outputs') as black,patch.object(self.w,'show_outputs') as show,patch.object(self.w,'start_live') as start:
            self.w.tabs.setCurrentIndex(1);self.w.tabs.setCurrentIndex(0)
        black.assert_not_called();show.assert_not_called();start.assert_not_called()
    def test_freeze_requires_actual_fresh_camera_not_offscreen_ui(self):
        self.w.freeze_deployment();self.assertFalse(self.w.deployment_page.frozen)
        self.assertIn('完整帧',self.w.deployment_page.status.text())
    def test_load_bundle_populates_six_fixtures_but_does_not_arm(self):
        if not CATALOG.exists():self.skipTest('local bundle unavailable')
        self.w.deployment_page.load_bundle(CATALOG)
        self.assertEqual(self.w.deployment_page.cases.count(),6)
        self.assertFalse(self.w.deployment_page.run_one.isEnabled());self.assertFalse(self.w.armed.isChecked())
    def test_retained_on_reference_is_pending_until_new_active_matches(self):
        if not CONTRACT.exists():self.skipTest('local contract unavailable')
        c=validate(json.loads(CONTRACT.read_text()));self.w.restore_modern(c)
        self.assertIsNone(self.w.registration);self.assertEqual(self.w.pending_registration,c['registration'])
        self.w.starting_registration=self.w.pending_registration
        self.w.camera_configured(dict(active=c['expected_active'],actual_roi=c['roi']))
        self.assertEqual(self.w.registration,c['registration']);self.assertIsNone(self.w.pending_registration)
    def test_failed_queue_request_does_not_leave_tabs_locked(self):
        self.w.hardware=True;self.w.running=True;self.w.live_ready=True;self.w.deployment_frozen=True
        self.w.worker=Mock(spec=DeploymentWorker);self.w.worker.isRunning.return_value=True
        self.w.worker.request_static.side_effect=RuntimeError('pending test failure')
        self.w.last_packet=dict(active={},received_monotonic=time.monotonic())
        self.w.deployment_page.bundle=SimpleNamespace(contract={});self.w.deployment_page.confirm.setChecked(True)
        with patch('live_deploy_v20.direct_alignment_gui.matching'),patch.object(self.w,'record'),patch.object(self.w,'black_outputs'):
            self.w.run_deployment([8])
        self.assertFalse(self.w.deployment_busy);self.assertTrue(self.w.tabs.isTabEnabled(0))
        self.assertFalse(self.w.deployment_page.busy);self.assertFalse(self.w.armed.isChecked())
    def test_parameter_change_changes_run_fingerprint(self):
        old=self.w.fingerprint();self.w.spins['y'].setValue(-51)
        self.assertNotEqual(old,self.w.fingerprint())
    def test_global_stop_during_batch_cancels_and_blanks_without_sdk_reopen(self):
        self.w.worker=Mock(spec=DeploymentWorker);self.w.deployment_busy=True
        with patch.object(self.w,'black_outputs') as black:
            self.w.stop_live();black.assert_called()
        self.w.worker.cancel_static.assert_called_once();self.w.worker.stop.assert_not_called()


if __name__=='__main__':unittest.main()
