import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
from PyQt5 import QtCore,QtWidgets
from live_manual_v19_trigger import test_trigger as fixtures
from .worker import ManualPreviewWorker
from .direct_alignment_gui import ManualPreviewWindow
from .recovery_record import validate_recovery
import restart_alignment_gui as restart


class PreviewWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'run'
        self.camera=fixtures.FakeCamera();self.camera.worker=None
        self.worker=ManualPreviewWorker({},self.path,4000,fixtures.ROI,camera_factory=lambda:self.camera)
        self.camera.worker=self.worker
        self.results=[];self.ready=[]
        self.worker.result.connect(self.results.append,QtCore.Qt.DirectConnection)
        self.worker.ready.connect(self.ready.append,QtCore.Qt.DirectConnection)
    def tearDown(self):self.tmp.cleanup()

    def test_ambient_mean_warns_without_claiming_dark_qualification(self):
        self.camera.images={i:np.full((6,8),56,np.uint8) for i in range(1,6)}
        self.worker.run();r=self.results[0]
        self.assertIsNone(r['error']);self.assertEqual(len(self.ready),1)
        self.assertEqual(self.camera.negative_controls,0)
        self.assertFalse(r['quantitative_dark_qualified'])
        self.assertEqual(len(r['zero_input_baselines']),3)
        self.assertTrue(all(row['high_background_warning'] for row in r['zero_input_baselines']))
        self.assertEqual(len(list((self.path/'startup_zero_input_0001').glob('*.npy'))),3)
        self.assertFalse((self.path/'startup_dark_0001').exists())

    def test_replay_actual_rejected_first_raw_does_not_fail_on_mean(self):
        raw=Path(__file__).resolve().parents[1]/'outputs/live_calibration/LINE0_REALTIME_R4_20261020_v1/live_20261020_210017_935781/rejected_0001/CCD_raw.npy'
        if not raw.exists():self.skipTest('preserved local failure unavailable')
        sha=hashlib.sha256(raw.read_bytes()).hexdigest();image=np.load(raw,allow_pickle=False)
        roi=dict(OffsetX=888,OffsetY=416,Width=image.shape[1],Height=image.shape[0])
        self.worker.requested_roi=roi
        self.camera.active.update({k:str(v) for k,v in roi.items()})
        self.camera.roi=lambda:dict(roi)
        self.camera.images={i:image.copy() for i in range(1,6)}
        self.worker.run()
        self.assertIsNone(self.results[0]['error']);self.assertEqual(len(self.ready),1)
        self.assertTrue(self.results[0]['zero_input_baselines'][0]['high_background_warning'])
        np.testing.assert_array_equal(np.load(next((self.path/'startup_zero_input_0001').glob('*.npy'))),image)
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),sha)

    def test_startup_peak_240_still_rejects_first_raw_without_retry(self):
        self.camera.images[1]=np.full((6,8),240,np.uint8)
        self.worker.run();self.assertEqual(self.camera.captures,1)
        self.assertEqual(self.ready,[]);self.assertIn('240',self.results[0]['error'])
        self.assertTrue((self.path/'rejected_0001/CCD_raw.npy').exists())

    def test_live_coherent_saturation_still_rejects(self):
        self.camera.images[4]=np.full((6,8),255,np.uint8)
        self.worker.run();self.assertEqual(self.camera.captures,4)
        self.assertIn('饱和',self.results[0]['error'])
        self.assertTrue(self.results[0]['camera_settings_restored'])

    def test_first_incomplete_is_fatal(self):
        self.camera.errors[1]=fixtures.IncompleteFrame(dict(frame_id=1,status=9))
        self.worker.run();self.assertEqual(self.camera.captures,1)
        self.assertEqual(len(self.results[0]['rejected_incomplete_frames']),1)
        self.assertEqual(self.ready,[])

    def test_first_timeout_is_fatal(self):
        self.camera.errors[1]=RuntimeError('spinCameraGetNextImageEx: -1011: timeout')
        self.worker.run();self.assertEqual(self.camera.captures,1)
        self.assertIn('-1011',self.results[0]['error']);self.assertEqual(self.ready,[])


class PreviewWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(['r5-test','-platform','offscreen'])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.w=ManualPreviewWindow(Path(self.tmp.name)/'unused',False)
        self.w.timer.stop()
    def tearDown(self):
        self.w.running=False;self.w.close();self.w.deleteLater();self.app.processEvents();self.tmp.cleanup()
    def test_offline_no_camera_output_or_session(self):
        self.w.start_live(project=False)
        self.assertIsNone(self.w.worker);self.assertFalse(self.w.windows);self.assertFalse(self.w.output.exists())
    def test_fault_replaces_stale_view_with_explicit_stopped_message(self):
        self.w.fault='测试故障';self.w.refresh()
        self.assertIn('不是实时',self.w.view.text());self.assertIn('测试故障',self.w.view.text())
        self.assertFalse(self.w.start_button.isEnabled())
    def test_bright_background_warns_and_does_not_enable_zero_calibration(self):
        self.w.running=True;self.w.preview_background_warning=56.2;self.w.refresh()
        self.assertIn('背景偏亮',self.w.recovery_status.text());self.assertFalse(self.w.zero_button.isEnabled())
    def test_visible_clock_updates_while_camera_stopped(self):
        self.w.tick();self.assertRegex(self.w.ui_clock.text(),r'界面心跳 \d\d:\d\d:\d\d')


class ManualRestartTests(unittest.TestCase):
    def test_unconfirmed_shielding_does_not_close_or_launch(self):
        with patch.object(restart,'owners',return_value=[]),patch('builtins.input',return_value='no'),\
                patch.object(restart.sys,'argv',['restart_alignment_gui.py']),patch.object(restart.subprocess,'run') as run:
            with self.assertRaises(RuntimeError):restart.main()
        run.assert_not_called()
    def test_never_treats_this_test_or_non_gui_process_as_target(self):
        self.assertIsNone(restart.process(os.getpid()))
    def test_recovery_binds_baseline_and_never_waives_recheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'before.json';baseline=dict(DeviceSerialNumber='15328133',TriggerMode='Off')
            path.write_text(json.dumps(dict(camera_before=baseline)))
            record=dict(schema='manual-alignment-recovery-v1',camera_baseline=baseline,
                baseline_source=str(path),baseline_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                manual_stop_requires_recheck=True)
            self.assertEqual(validate_recovery(record),record)
            with self.assertRaises(ValueError):validate_recovery(dict(record,manual_stop_requires_recheck=False))
            path.write_text('{}')
            with self.assertRaises(ValueError):validate_recovery(record)


if __name__=='__main__':unittest.main()
