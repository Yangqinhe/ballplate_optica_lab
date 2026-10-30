"""Offline actual-factory regression: no SDK, USB or real X11 output."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from display_binding import bound_modulators, verify_instances
from optical_lab.queued_display import queued_modulator


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get('QT_QPA_PLATFORM') != 'offscreen':
            raise RuntimeError('offscreen only')
        from optical_lab.preview import _load_legacy
        _load_legacy(LAB.parent/'ballplate_fourframe96_deploy')
        import sdon.hardware.modulator as canonical
        import bench
        from PyQt5 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.canonical, cls.actual = canonical, bench.modulator
        cls.source = Path(bench.MODULATOR_SOURCE).resolve()
        import hashlib
        cls.digest = hashlib.sha256(cls.source.read_bytes()).hexdigest()

    def test_previous_context_does_not_cover_factory(self):
        before = self.actual.OpticalModulator.update_image
        self.assertIsNot(self.actual.OpticalModulator, self.canonical.OpticalModulator)
        with queued_modulator(self.canonical):
            self.assertIs(self.actual.OpticalModulator.update_image, before)
            self.assertIsNot(self.canonical.OpticalModulator.update_image, before)

    def test_actual_factory_1000_changes_no_repaint(self):
        from PyQt5 import QtCore, QtWidgets
        evidence = {}
        original = self.actual.OpticalModulator.update_image
        with bound_modulators(self.canonical, queued_modulator, self.source, self.digest, evidence):
            w = self.actual.OpticalModulator(self.app.primaryScreen(), (64, 64), 'OFFLINE', (0, 0))
            w.resize(80, 80); w.show()
            owner = SimpleNamespace(hardware=False, windows={'dmd': w})
            self.assertTrue(verify_instances(owner, evidence)['instances_verified'])
            # Direct method calls to synchronous QWidget.repaint must be gone.
            with patch.object(QtWidgets.QWidget, 'repaint', side_effect=AssertionError('sync repaint')):
                for i in range(1000):
                    w.update_image(np.full((64, 64), i % 256, dtype=np.uint8))
                    w.update_offset(i % 3, i % 5)
                    if i % 10 == 0: self.app.processEvents()
                self.app.processEvents()
            self.assertEqual(w._applied_sequence, 1000)
            self.assertEqual(w._applied_hash, w._requested_hash)
            w.close(); w.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.assertIs(self.actual.OpticalModulator.update_image, original)

    def test_wrong_source_rejected_before_patch(self):
        with self.assertRaises(RuntimeError):
            with bound_modulators(self.canonical, queued_modulator, self.source, '0'*64, {}):
                self.fail('entered wrong source context')

    def test_unpatched_instance_rejected(self):
        evidence = {}
        with bound_modulators(self.canonical, queued_modulator, self.source, self.digest, evidence):
            fake = SimpleNamespace(hardware=True, windows={'dmd': object()})
            with self.assertRaises(RuntimeError): verify_instances(fake, evidence)


if __name__ == '__main__': unittest.main()
