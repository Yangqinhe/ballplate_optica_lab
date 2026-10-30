"""Offscreen display-only tests; no SDK connections or device windows."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
from PyQt5 import QtWidgets
from live_deploy_v21.direct_alignment_gui import DisplayGainWindow
from live_deploy_v21.page import display_signal
from live_deploy_v20.contract import PROJECT

CATALOG = PROJECT/'outputs/refined_live_demo_20261020_v1/catalog.json'
OLD_CATALOG = PROJECT/'outputs/deployment_page_v20_20261020_v1/PREPARED_v3/catalog.json'
RUN = PROJECT/'outputs/live_calibration/MANUAL_RESTART_20261020_214924_220784/GUI/deployment_20261020_222504_386648'


class SignalTests(unittest.TestCase):
    def test_identity(self):
        a = np.arange(256, dtype=np.uint8).reshape(16, 16)
        np.testing.assert_array_equal(display_signal(a, 1, 0, 1), a)

    def test_gain_has_no_uint8_underflow_and_raw_is_unchanged(self):
        a = np.array([[0, 9, 10, 11, 23, 255]], np.uint8)
        old = a.copy()
        np.testing.assert_array_equal(display_signal(a, 16, 10), [[0, 0, 0, 16, 208, 255]])
        np.testing.assert_array_equal(a, old)

    def test_gamma_and_negative_signal_do_not_change_float_means(self):
        a = np.array([[-.5, 0., .1, .5, 1.]])
        old = a.copy()
        self.assertEqual(display_signal(a, 16, 0, .5)[0, 0], 0)
        self.assertGreater(display_signal(a, 16, 0, .5)[0, -1], display_signal(a, 16, 0, 1)[0, -1])
        np.testing.assert_array_equal(a, old)

    def test_invalid_controls_and_nonfinite_data_are_rejected(self):
        for gain, black, gamma in ((0, 0, 1), (129, 0, 1), (16, -1, 1), (16, 0, 0)):
            with self.assertRaises(ValueError):
                display_signal(np.zeros((2, 2)), gain, black, gamma)
        with self.assertRaises(ValueError):
            display_signal(np.array([[np.nan]]))


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(['display-only-tests'])

    def setUp(self):
        if not CATALOG.exists():
            self.skipTest('frozen local research catalog absent')
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)/'NO_HARDWARE_SESSION'
        self.window = DisplayGainWindow(self.output, False, catalog=CATALOG)
        self.page = self.window.deployment_page

    def tearDown(self):
        if hasattr(self, 'window'):
            self.window.close(); self.window.deleteLater(); self.app.processEvents()
            self.temp.cleanup()

    def test_two_tabs_and_display_defaults_without_any_hardware(self):
        self.assertEqual(self.window.tabs.count(), 2)
        self.assertEqual(self.page.display_gain.value(), 16)
        self.assertEqual(self.page.display_black.value(), 10)
        self.assertIsNone(self.window.worker)
        self.assertFalse(self.window.windows)
        self.assertFalse(self.output.exists())

    def test_display_controls_never_change_output_fingerprint_or_arm(self):
        before = self.window.fingerprint()
        self.page.display_gain.setValue(64)
        self.page.display_black.setValue(9)
        self.page.display_gamma.setValue(.6)
        self.page.view_mode.setCurrentIndex(1)
        self.assertEqual(self.window.fingerprint(), before)
        self.assertIsNone(self.window.worker)
        self.assertFalse(self.window.armed.isChecked())
        self.assertFalse(self.output.exists())

    def test_live_raw_and_prediction_buffers_are_not_mutated(self):
        a = np.full(self.page.bundle.weights.shape[1:], 12, np.uint8)
        old = a.copy()
        predicted = self.page.prediction_signal.copy()
        self.page.fresh_view(dict(image=a, meta=dict(FrameID=1)))
        self.page.display_gain.setValue(32)
        self.page.display_black.setValue(11)
        np.testing.assert_array_equal(a, old)
        np.testing.assert_array_equal(self.page.raw_frame['image'], old)
        np.testing.assert_array_equal(self.page.prediction_signal, predicted)

    def test_wrong_roi_is_not_rendered_as_current_sensor(self):
        self.page.fresh_view(dict(image=np.zeros((2, 2), np.uint8), meta={}))
        self.assertIsNone(self.page.raw_frame)

    def test_history_is_explicit_and_uses_original_saved_analysis(self):
        if not RUN.exists():
            self.skipTest('local completed capture absent')
        self.page.restore_result(RUN)
        self.assertEqual(self.page.view_mode.currentIndex(), 1)
        self.assertIn('非实时', self.page.status.text())
        self.assertFalse(self.page.run_one.isEnabled())
        self.assertFalse(self.page.confirm.isChecked())
        self.assertEqual(self.page.readouts.item(0, 1).text(), '924.23')
        self.assertIn('NCC=0.735', self.page.result_text.text())
        self.assertIsNone(self.window.worker)

    def test_history_gain_does_not_change_any_readout_or_score(self):
        self.page.restore_result(RUN)
        scores = self.page.result_text.text()
        table = [[self.page.readouts.item(i, j).text() for j in range(6)] for i in range(8)]
        signal = self.page.last_result['observed_signal'].copy()
        self.page.display_gain.setValue(128)
        self.page.display_gamma.setValue(.5)
        self.page.display_black.setValue(100)
        self.assertEqual(self.page.result_text.text(), scores)
        self.assertEqual([[self.page.readouts.item(i, j).text() for j in range(6)] for i in range(8)], table)
        np.testing.assert_array_equal(self.page.last_result['observed_signal'], signal)

    def test_new_bundle_clears_old_results_and_disables_run(self):
        self.page.restore_result(RUN)
        self.page.load_bundle(OLD_CATALOG)
        self.assertIsNone(self.page.last_result)
        self.assertIsNone(self.page.last_folder)
        self.assertIsNone(self.page.readouts.item(0, 1))
        self.assertIn('未运行', self.page.result_text.text())
        self.assertFalse(self.page.run_one.isEnabled())

    def test_modified_history_is_rejected(self):
        copy = Path(self.temp.name)/'changed-history'
        shutil.copytree(RUN, copy)
        (copy/'input_08/learned_1/CCD_raw_001.npy').write_bytes(b'altered test fixture')
        with self.assertRaises(ValueError):
            self.page.restore_result(copy)
        self.assertFalse(self.window.armed.isChecked())

    def test_bad_history_does_not_abort_window_or_create_hardware(self):
        extra = DisplayGainWindow(Path(self.temp.name)/'NO_EXTRA_SESSION', False,
                                  catalog=CATALOG, result_folder=Path(self.temp.name)/'missing')
        self.assertIn('历史图未采用', extra.deployment_page.status.text())
        self.assertIsNone(extra.worker)
        self.assertFalse(extra.windows)
        extra.close(); extra.deleteLater()


if __name__ == '__main__':
    unittest.main()
