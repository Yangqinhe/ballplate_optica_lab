from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from PyQt5 import QtWidgets
from .direct_alignment_gui import RecoverableTriggerWindow, THREAD_ENV
from .launch_realtime import command, LIMITS


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(['r3-tests', '-platform', 'offscreen'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = RecoverableTriggerWindow(Path(self.tmp.name)/'unused', False)
        self.w.timer.stop()

    def tearDown(self):
        self.w.worker = None; self.w.recheck_context = None; self.w.hardware = False
        self.w.close(); self.w.deleteLater(); self.app.processEvents(); self.tmp.cleanup()

    def test_offline_does_not_connect_create_windows_or_write_session(self):
        self.w.start_live(project=False)
        self.assertFalse(self.w.windows); self.assertIsNone(self.w.worker)
        self.assertFalse(self.w.output.exists()); self.assertFalse(self.w.armed.isChecked())
        self.assertFalse(self.w.shortcut.autoRepeat())

    def test_fault_banner_distinguishes_protection_from_frozen_display(self):
        self.w.fault = 'CCD连片接近饱和'; self.w.refresh()
        self.assertIn('保护停采', self.w.recovery_status.text())
        self.assertIn('不是实时', self.w.telemetry.text())
        self.assertFalse(self.w.start_button.isEnabled()); self.assertFalse(self.w.armed.isEnabled())
        self.assertEqual(self.w.recheck.text(), '复核并恢复（不投图）')

    def test_allzero_does_not_cancel_readonly_recheck(self):
        worker = Mock(); worker.isRunning.return_value = True
        self.w.worker = worker; self.w.recheck_context = {'fault': 'test'}; self.w.fault = 'test'
        with patch.object(self.w, 'black_outputs') as black:
            self.w.stop_live(); black.assert_called()
        worker.stop.assert_not_called()
        self.assertFalse(self.w.armed.isChecked())

    def test_explicit_cancel_still_cancels_and_never_unlocks(self):
        worker = Mock(); worker.isRunning.return_value = True
        self.w.worker = worker; self.w.recheck_context = {'fault': 'test'}; self.w.fault = 'test'
        self.w.cancel_recovery(); worker.stop.assert_called_once()
        self.assertEqual(self.w.fault, 'test')

    def test_acquisition_fault_stop_still_stops_real_worker(self):
        worker = Mock(); worker.isRunning.return_value = True
        self.w.worker = worker; self.w.running = self.w.live_ready = True; self.w.fault = 'fault'
        self.w.stop_live(); worker.stop.assert_called_once()
        self.assertFalse(self.w.running)

    def test_restore_preserves_user_knobs_but_never_reuses_invalid_geometry(self):
        h = dict(simulation_display_orientation=dict(code=5))
        state = dict(settings=dict(dmd='fiducial', slm='lens', full_panel=False, square=128,
            focal_mm=400., period=36, x=-411, y=-53),
            optical_config=dict(wavelength_m=532e-9, pitch_m=7e-6, f1_m=.149, f2_m=.1,
                phase_depth_rad=6.2832, slm_flip_y=True), requested_exposure_us=1000,
            roi=dict(OffsetX=888, OffsetY=416, Width=552, Height=554), fault='original saturation',
            original_fault_before=dict(DeviceSerialNumber='15328133'), original_last_camera_report={}, original_heartbeat=h)
        with patch('live_manual_v19_trigger_r3.direct_alignment_gui.validate', return_value=state):
            self.w.restore_controls(state)
        self.assertEqual(self.w.spins['x'].value(), -411); self.assertEqual(self.w.spins['y'].value(), -53)
        self.assertEqual(self.w.optics['pitch'].value(), 7); self.assertEqual(self.w.optics['f1'].value(), 149)
        self.assertEqual(self.w.exposure.value(), 1000); self.assertEqual(self.w.fault, 'original saturation')
        self.assertIsNone(self.w.registration); self.assertIsNone(self.w.pending_registration)
        self.assertIsNone(self.w.worker); self.assertFalse(self.w.armed.isChecked())

    def test_launcher_limits_reach_child_systemd_unit(self):
        cmd = command(['--hardware', '--light-blocked-confirmed', '--output', '/tmp/new-session'], {}, 'test.service')
        self.assertEqual(LIMITS, THREAD_ENV)
        for key, value in LIMITS.items(): self.assertIn('--setenv='+key+'='+value, cmd)
        self.assertIn('--property=Restart=no', cmd)


if __name__ == '__main__': unittest.main()
