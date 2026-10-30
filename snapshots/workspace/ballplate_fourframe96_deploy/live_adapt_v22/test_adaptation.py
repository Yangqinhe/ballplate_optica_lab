"""No devices. Single-frame math, split isolation, IPC and GUI regressions."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from live_adapt_v22.core import RULES, fit_head, predict, validate_samples, audit_capture, digest, write_new


def samples():
    rng = np.random.default_rng(220)
    groups = np.repeat(np.arange(20), 4)
    split = np.repeat(['train']*12+['select']*4+['test']*4, 4)
    latent = rng.normal(size=(80, 8))
    x = 450000+5000*latent[:, None, :]+rng.normal(0, 20, (80, 8, 8))
    y = np.column_stack((latent[:, 0]+latent[:, 3], latent[:, 2]-latent[:, 5]))*.07
    return x, y, groups, split


class MathTests(unittest.TestCase):
    def test_exactly_linear18_raw_readout(self):
        x, y, g, s = samples(); head = fit_head(x, y, g, s)
        self.assertEqual(np.asarray(head['weight_raw_ADU_to_rad']).size+len(head['bias_rad']), 18)
        self.assertTrue(head['short_MuJoCo_pilot_eligible'])
        self.assertLess(head['metrics']['test']['MAE_rad'], .005)
        self.assertEqual(predict(head, x[0, 0]).shape, (2,))

    def test_trajectory_split_leakage_rejected(self):
        x, y, g, s = samples(); g[-1] = 0
        with self.assertRaisesRegex(ValueError, 'leakage'): fit_head(x, y, g, s)

    def test_selection_does_not_use_test_labels(self):
        x, y, g, s = samples(); h1 = fit_head(x, y, g, s)
        y[s == 'test'] *= -1
        h2 = fit_head(x, y, g, s)
        self.assertEqual(h1['weight_raw_ADU_to_rad'], h2['weight_raw_ADU_to_rad'])
        self.assertEqual(h1['bias_rad'], h2['bias_rad'])
        self.assertEqual(h1['selected_ridge'], h2['selected_ridge'])
        self.assertFalse(h2['short_MuJoCo_pilot_eligible'])

    def test_constant_features_rejected(self):
        x, y, g, s = samples(); x[:, :, 0] = 1
        with self.assertRaisesRegex(ValueError, 'constant detector'): fit_head(x, y, g, s)

    def test_no_mean_sneaked_into_single_frame_score(self):
        x, y, g, s = samples(); x += np.random.default_rng(22).normal(0, 10000, x.shape)
        head = fit_head(x, y, g, s)
        self.assertGreater(head['metrics']['test']['MAE_rad'], .02)
        self.assertFalse(head['short_MuJoCo_pilot_eligible'])
        self.assertEqual(head['CCD_history'], 0)

    def test_nonfinite_rejected(self):
        x, y, g, s = samples(); x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): fit_head(x, y, g, s)

    def test_teacher_action_limits_checked(self):
        x, y, g, s = samples(); y[0, 0] = 2
        with self.assertRaises(ValueError): fit_head(x, y, g, s)

    def test_batch_and_individual_predictions_same(self):
        x, y, g, s = samples(); head = fit_head(x, y, g, s)
        np.testing.assert_allclose(predict(head, x[0]), np.stack([predict(head, a) for a in x[0]]))

    def test_independent_test_group_gate(self):
        x, y, g, s = samples(); g[s == 'test'] = 19
        h = fit_head(x, y, g, s)
        self.assertFalse(h['test_criteria']['independent_groups'])

    def test_write_new_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'record.json'; write_new(p, {'x': 1})
            with self.assertRaises(FileExistsError): write_new(p, {'x': 2})
            self.assertEqual(json.loads(p.read_text())['x'], 1)


class CaptureTests(unittest.TestCase):
    def fixture(self, folder):
        weights = np.ones((8, 2, 2), np.float32)/8
        active = {'TriggerSource': 'Line0'}
        bundle = SimpleNamespace(sha='bundle', weights=weights, contract={'expected_active': active},
                                 cases=[{'id': 0}], verify=lambda: None)
        frames = []
        for i in range(4):
            p = folder/f'{i}.npy'; np.save(p, np.full((2, 2), 12+i, np.uint8))
            frames.append(dict(raw_relative=p.name, raw_sha256=digest(p), camera_active=active,
                frame=dict(FrameID=i+1, TimeStamp=(i+1)*100, host_get_started_unix_ns=2_000_000_001+i)))
        plan = dict(bundle_sha256='bundle', repeats=4, settle_seconds=1.5)
        report = dict(completed=True, error=None, rejected_incomplete_frames=[], actual_park_verified=True,
            camera_node_writes_during_test=0, camera_active_at_end=active,
            inputs=[dict(id=0, display_verification=dict(actual_framebuffer_checked=True, verified_unix_ns=1), frames=frames)])
        write_new(folder/'PROSPECTIVE_PROTOCOL.json', plan); write_new(folder/'capture.json', report)
        return bundle, report

    def test_features_recomputed_from_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); b, _ = self.fixture(folder)
            x, _, _ = audit_capture(b, folder)
            np.testing.assert_array_equal(x[0, :, 0], [6., 6.5, 7., 7.5])

    def test_corrupted_raw_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); b, _ = self.fixture(folder)
            np.save(folder/'0.npy', np.zeros((2, 2), np.uint8))
            with self.assertRaisesRegex(ValueError, 'hash'): audit_capture(b, folder)

    def test_incomplete_stream_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); b, r = self.fixture(folder)
            r['rejected_incomplete_frames'] = [{'id': 3}]
            (folder/'capture.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError): audit_capture(b, folder)

    def test_stale_frame_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); b, r = self.fixture(folder)
            r['inputs'][0]['frames'][0]['frame']['host_get_started_unix_ns'] = 1
            (folder/'capture.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'freshness'): audit_capture(b, folder)

    def test_unverified_park_blocks_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); b, r = self.fixture(folder)
            r['actual_park_verified'] = False
            (folder/'capture.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError): audit_capture(b, folder)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from live_adapt_v22.direct_alignment_gui import AdaptationWindow

    def test_three_tabs_no_devices_at_start(self):
        from live_adapt_v22.direct_alignment_gui import AdaptationWindow
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'NOT_CREATED'
            w = AdaptationWindow(output, hardware=False)
            self.assertEqual(w.tabs.count(), 3); self.assertIsNone(w.worker)
            self.assertFalse(w.windows); self.assertFalse(output.exists())
            self.assertFalse(w.adaptation_page.collect.isEnabled())
            self.assertFalse(w.adaptation_page.pilot.isEnabled())
            w.deleteLater(); self.app.processEvents()

    def test_display_only_no_raw_mutation(self):
        from live_deploy_v21.page import display_signal
        a = np.arange(100, dtype=np.uint8).reshape(10, 10); old = a.copy()
        display_signal(a, 16, 10, .5)
        np.testing.assert_array_equal(a, old)

    def test_hardware_off_never_starts_collection(self):
        from live_adapt_v22.direct_alignment_gui import AdaptationWindow
        with tempfile.TemporaryDirectory() as directory:
            w = AdaptationWindow(Path(directory)/'NONE', hardware=False)
            w.run_adaptation(); self.assertFalse(w.adaptation_active)
            self.assertIsNone(w.worker); w.deleteLater(); self.app.processEvents()


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from live_adapt_v22.direct_alignment_gui import AdaptationWindow

    def worker(self):
        from live_adapt_v22.worker import AdaptationWorker
        w = AdaptationWorker(None, Path('/tmp/not-created-by-single-frame-tests'),
            exposure=1000, roi=dict(OffsetX=888, OffsetY=416, Width=552, Height=554),
            camera_factory=lambda: None)
        w.camera = SimpleNamespace(acquiring=True, active={'source': 'Line0'})
        w.startup_dark_required = False
        return w

    def test_same_owner_queue_does_not_connect_camera(self):
        w = self.worker(); w.request_adaptation(None, '/tmp/NEVER_CREATED')
        self.assertIsNotNone(w.adaptation_request)
        with self.assertRaises(RuntimeError): w.request_adaptation(None, '/tmp/NEVER_CREATED')

    def test_no_free_run_or_missing_camera_fallback(self):
        w = self.worker(); w.camera.acquiring = False
        with self.assertRaises(RuntimeError): w.request_adaptation(None, '/tmp/NEVER_CREATED')

    def test_pending_static_job_blocks_adaptation(self):
        w = self.worker(); w.static_busy = True
        with self.assertRaises(RuntimeError): w.request_adaptation(None, '/tmp/NEVER_CREATED')

    def test_cancel_check_precedes_capture(self):
        w = self.worker(); w.static_cancel.set()
        b = SimpleNamespace(contract={'expected_active': dict(w.camera.active)})
        with self.assertRaises(InterruptedError): w.check_job(b, time.monotonic()+10)

    def test_deadline_enforced(self):
        w = self.worker(); b = SimpleNamespace(contract={'expected_active': dict(w.camera.active)})
        with self.assertRaises(TimeoutError): w.check_job(b, time.monotonic()-1)

    def test_stale_first_frame_rejected_not_retried(self):
        w = self.worker(); b = SimpleNamespace(contract={'expected_active': dict(w.camera.active)})
        with patch.object(w, 'capture_one', return_value=(np.zeros((2, 2), np.uint8), {'host_get_started_unix_ns': 1})) as capture:
            with self.assertRaises(RuntimeError): w.checked_capture(b, time.monotonic()+10, 1, 1.5)
            self.assertEqual(capture.call_count, 1)

    def test_single_frame_peak_limit_not_display_scaled(self):
        w = self.worker(); b = SimpleNamespace(contract={'expected_active': dict(w.camera.active)})
        with patch.object(w, 'capture_one', return_value=(np.full((2, 2), 240, np.uint8), {'host_get_started_unix_ns': 2_000_000_000})):
            with self.assertRaises(RuntimeError): w.checked_capture(b, time.monotonic()+10, 1, 1.5)

    def test_active_camera_drift_blocks_job(self):
        w = self.worker(); b = SimpleNamespace(contract={'expected_active': {'source': 'Software'}})
        with self.assertRaises(RuntimeError): w.check_job(b, time.monotonic()+10)


if __name__ == '__main__': unittest.main()
