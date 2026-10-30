"""Bounded CAMERA-ONLY Line0 test; existing v18 output windows stay black.

The stopped GUI has released its camera lock. This process temporarily owns
that lock, never creates/changes an output window, never sends TriggerSoftware,
and restores the complete original camera baseline. No persistent settings,
firmware, GPIO voltage/direction, DMD controller or host network writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'live_manual_v17'))
from black_pixels import RootPixels
from calibration_camera import CalibrationCamera
from display_linux import read_roles, same_signals
from spin_c import IncompleteFrame
from capture_ccd_transport_observation_v18 import host_transport


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def require_parked(state, pid, now_ns):
    age = now_ns-state['unix_ns']
    if state['pid'] != pid or not 0 <= age <= 5_000_000_000:
        raise RuntimeError('original owner heartbeat stale or changed')
    if any(state.get(k) for k in ('busy', 'live_camera', 'projection_armed', 'hardware_fault')):
        raise RuntimeError('original GUI must remain stopped, disarmed and fault-free')


class BlackHold:
    """Read-only monitoring, NOT an independent physical safety interlock."""
    def __init__(self, session, preflight):
        if not preflight.get('parked_verification_passed'):
            raise RuntimeError('successful parked preflight required')
        self.session, self.preflight = Path(session), preflight
        self.pid = preflight['pid']
        self.identity = self.process_identity()
        self.pixels = RootPixels()
        self.samples = []
        self.last_pixels = self.last_identity = -float('inf')
        self.check(force=True)

    def process_identity(self):
        return (Path('/proc')/str(self.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]

    def check(self, force=False):
        state = json.loads((self.session/'heartbeat.json').read_text())
        require_parked(state, self.pid, time.time_ns())
        if state['actual_roi'] != self.preflight['heartbeat']['actual_roi']:
            raise RuntimeError('GUI retained ROI changed')
        if state['geometry_frozen'] != self.preflight['heartbeat']['geometry_frozen']:
            raise RuntimeError('GUI geometry lock changed')
        now = time.monotonic()
        if force or now-self.last_identity >= 1:
            if self.process_identity() != self.identity:
                raise RuntimeError('original display process lifetime changed')
            for group in ('sources', 'source_configs'):
                for name, sha in self.preflight['runtime'][group].items():
                    if digest(name) != sha:
                        raise RuntimeError('loaded runtime source/config changed: '+name)
            same_signals(self.preflight['roles'], read_roles())
            for role in ('control', 'dmd', 'slm'):
                row = self.preflight['windows'][role]
                info = subprocess.check_output(['xwininfo', '-id', row['wid']], text=True, timeout=2)
                import re
                rect = [int(re.search(r'(?m)^\s*'+key+r':\s*(-?\d+)', info)[1])
                        for key in ('Absolute upper-left X', 'Absolute upper-left Y', 'Width', 'Height')]
                props = subprocess.check_output(['xprop', '-id', row['wid'], '_NET_WM_PID'], text=True, timeout=2)
                match = re.search(r'_NET_WM_PID\(CARDINAL\) = (\d+)', props)
                if rect != row['rect'] or 'IsViewable' not in info or not match or int(match[1]) != self.pid:
                    raise RuntimeError('owned window moved/disappeared: '+role)
            self.last_identity = now
        if force or now-self.last_pixels >= .25:
            rows = {role: self.pixels.inspect(tuple(self.preflight['windows'][role]['rect']))
                    for role in ('dmd', 'slm')}
            if not all(row['all_RGB_zero'] for row in rows.values()):
                raise RuntimeError('existing DMD/SLM framebuffer is no longer all-zero')
            self.samples.append(dict(unix_ns=time.time_ns(), pixels=rows))
            self.last_pixels = now

    def close(self):
        self.pixels.close()


TRIGGER_KEYS = ('TriggerMode', 'TriggerSelector', 'TriggerSource', 'TriggerActivation', 'SensorShutterMode')
TRIGGER_EXPECTED = dict(TriggerMode='On', TriggerSelector='FrameStart', TriggerSource='Line0',
                        TriggerActivation='FallingEdge', SensorShutterMode='Rolling')


def validate_active(active, roi):
    expected = dict(TRIGGER_EXPECTED, PixelFormat='Mono8', ExposureMode='Timed',
                    ExposureAuto='Off', GainAuto='Off', AcquisitionMode='Continuous')
    for key, value in expected.items():
        if active.get(key) != value:
            raise RuntimeError('hardware-trigger contract mismatch: '+key)
    if active.get('TriggerDelayEnabled') not in ('0', 'false', 'False'):
        raise RuntimeError('unexpected enabled trigger delay; no implicit delay changes')
    if abs(float(active['ExposureTime'])-4000) > 50 or float(active['Gain']) != 0:
        raise RuntimeError('unexpected exposure/gain in hardware-trigger test')
    if any(int(active[key]) != value for key, value in roi.items()):
        raise RuntimeError('hardware-trigger ROI mismatch')


class Line0Camera(CalibrationCamera):
    """Independent additive transaction; frozen free-run adapter stays intact."""
    def __init__(self, guard):
        self.guard, self.cleaning, self.trigger_before = guard, False, None
        self.write_trace = []
        guard.check()
        super().__init__()

    def _write(self, name, value, node_map):
        if not self.cleaning:
            self.guard.check()
        row = dict(name=name, requested=str(value), cleanup=self.cleaning, unix_ns=time.time_ns())
        self.write_trace.append(row)
        result = super()._write(name, value, node_map)
        row['readback'] = self.read(name, node_map)
        return result

    def integer(self, name, value):
        if not self.cleaning:
            self.guard.check()
        return super().integer(name, value)

    def configure_trigger(self):
        if self.acquiring or self.before.get('TriggerMode') != 'Off':
            raise RuntimeError('probe starts stopped from an explicit TriggerMode=Off baseline')
        # Register complete rollback BEFORE the first trigger write. Enabling
        # TriggerMode itself may change SensorShutterMode on this model.
        snapshot = {key: self.before[key] for key in TRIGGER_KEYS}
        if any(value is None for value in snapshot.values()):
            raise RuntimeError('missing original trigger/shutter readback')
        if self.before.get('TriggerSelector') != 'FrameStart':
            raise RuntimeError('non-FrameStart baseline requires a separate selector-aware contract')
        self.trigger_before = snapshot
        self._write('TriggerMode', 'Off', self.map)
        for key in ('TriggerSelector', 'TriggerSource', 'TriggerActivation'):
            self._write(key, TRIGGER_EXPECTED[key], self.map)
        self._write('TriggerMode', 'On', self.map)
        self.after_trigger_enable = self.parameters()
        # Reapply AFTER On, not just before it; this is platform-specific.
        self._write('SensorShutterMode', 'Rolling', self.map)

    def start_trigger_test(self, roi):
        self.set_roi(roi)
        self.set_temporary('GainAuto', 'Off')
        self.set_temporary('Gain', '0')
        self.set_temporary('AcquisitionMode', 'Continuous')
        self.set_temporary('StreamBufferHandlingMode', 'NewestOnly', self.stream)
        if self.before.get('GevTimestampTickFrequency') != '125000000':
            raise RuntimeError('unknown GigE packet-delay clock')
        self.set_temporary('GevSCPD', str(max(1000, int(self.before['GevSCPD']))))
        self.configure_trigger()
        self._write('ExposureTime', '4000', self.map)
        self.active = self.parameters()
        validate_active(self.active, roi)
        self.guard.check(force=True)
        self.call('spinCameraBeginAcquisition', self.camera)
        self.acquiring = True

    def close(self):
        if self.closed:
            return
        self.cleaning = True
        errors = []
        if self.acquiring:
            try:
                self.call('spinCameraEndAcquisition', self.camera)
                self.acquiring = False
                time.sleep(.35)
            except Exception as error:
                errors.append('EndAcquisition: '+str(error))
        if self.initialized and self.trigger_before:
            # Restore trigger/shutter first, then full-sensor ROI, exposure,
            # gain and stream settings via the unchanged parent transaction.
            sequence = [('TriggerMode', 'Off')]
            sequence += [(key, self.trigger_before[key]) for key in
                         ('TriggerSelector', 'TriggerSource', 'TriggerActivation', 'TriggerMode', 'SensorShutterMode')]
            for key, value in sequence:
                try:
                    self._write(key, value, self.map)
                    if self.read(key) != value:
                        raise RuntimeError('exact trigger/shutter rollback mismatch: '+key)
                except Exception as error:
                    errors.append(str(error))
        try:
            super().close()
        except Exception as error:
            errors.append(str(error))
        self.trigger_restore_errors = errors
        if errors:
            raise RuntimeError('; '.join(errors))


def sequence_summary(records):
    if len(records) < 3:
        raise RuntimeError('insufficient actual hardware-trigger frames')
    ids = np.asarray([row['FrameID'] for row in records], dtype=np.int64)
    stamps = np.asarray([row['TimeStamp'] for row in records], dtype=np.int64)
    host = np.asarray([row['host_monotonic_received_ns'] for row in records], dtype=np.int64)
    if any(np.any(np.diff(values) <= 0) for values in (ids, stamps, host)):
        raise RuntimeError('duplicate/stale/non-monotonic trigger frame sequence')
    interval_ms = np.diff(stamps)/1e6  # Spinnaker image timestamps are nanoseconds.
    return dict(frames=len(records), frame_id_first=int(ids[0]), frame_id_last=int(ids[-1]),
                delivered_frame_id_gaps=int(np.maximum(np.diff(ids)-1, 0).sum()),
                camera_timestamp_interval_ms={str(q): float(np.percentile(interval_ms, q))
                                              for q in (0, 50, 95, 99, 100)},
                receive_fps=float((len(records)-1)*1e9/(host[-1]-host[0])),
                camera_timestamp_fps=float((len(records)-1)*1e9/(stamps[-1]-stamps[0])),
                NOT_trigger_input_pulse_count=True, NOT_pattern_to_control_latency=True)


def collect(camera, guard, frames=120, maximum_seconds=12):
    records, stack = [], []
    deadline = time.monotonic()+maximum_seconds
    for _ in range(frames):
        guard.check()
        if time.monotonic() >= deadline:
            raise RuntimeError('bounded trigger acquisition expired; no retry/fallback')
        # One missing pulse sequence times out; NEVER switch to free-run.
        image, row = camera.capture(timeout_ms=1000)
        row['host_monotonic_received_ns'] = time.monotonic_ns()
        if int(image.max()) >= 240 or float(image.mean()) > 30:
            raise RuntimeError('unexpected bright/saturated frame during held-black test')
        if records and any(row[key] <= records[-1][key] for key in ('FrameID', 'TimeStamp')):
            raise RuntimeError('stale/out-of-order hardware-trigger frame')
        records.append(row)
        # Save early/late raw diagnostics, not 120 near-identical dark arrays.
        if len(records) in (1, 2, frames-1, frames):
            stack.append(image)
        camera.test_records = records
        camera.test_saved_raw = stack
    expected_roi = {key: int(camera.active[key]) for key in ('OffsetX', 'OffsetY', 'Width', 'Height')}
    validate_active(camera.parameters(), expected_roi)
    guard.check(force=True)
    return sequence_summary(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpio-wiring-confirmed', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    baseline_report = json.loads(args.baseline_report.read_text())
    expected = baseline_report['camera_after']
    if (expected != baseline_report['camera_before'] or not baseline_report.get('camera_settings_restored')
            or expected.get('DeviceSerialNumber') != '15328133' or expected.get('TriggerMode') != 'Off'):
        raise RuntimeError('an exactly restored known camera baseline is required')
    plan = dict(kind='HELD_BLACK_LINE0_HARDWARE_TRIGGER_CAMERA_ONLY',
                frames=120, maximum_acquisition_seconds=12, one_get_timeout_ms=1000,
                trigger=TRIGGER_EXPECTED, exposure_requested_us=4000,
                roi=preflight['heartbeat']['actual_roi'],
                gpio_user_reply='GPIO触发线已按原平台接好' if args.gpio_wiring_confirmed else None,
                no_projection=True, no_new_windows=True, no_GUI_restart=True,
                ownership='idle original GUI retains ONLY display windows; this process exclusively locks released CCD',
                no_TriggerSoftware=True, no_free_run_fallback=True, no_DMD_controller_writes=True,
                no_GPIO_voltage_or_direction_writes=True, restores_original_baseline=True,
                no_persistent_UserSetSave=True, SLM_hardware_sync_NOT_established=True,
                electrical_waveform_and_edge_phase_NOT_measured=True,
                raw_max_ceiling_exclusive=240, maximum_dark_mean_ADU=30,
                incomplete_frame_limit=0, hardware_20ms_accepted=False, release_approved=False,
                preflight_sha256=digest(args.preflight), baseline_report_sha256=digest(args.baseline_report),
                source_sha256={str(path): digest(path) for path in
                    (Path(__file__), PROJECT/'tools/capture_ccd_transport_observation_v18.py',
                     PROJECT/'live_manual_v17/spin_c.py', PROJECT/'live_manual_v17/calibration_camera.py')})
    if not args.execute:
        print(json.dumps(dict(plan_only=True, SDK_initialized=False, **plan), ensure_ascii=False, indent=2))
        return
    if not args.gpio_wiring_confirmed:
        parser.error('current physical GPIO connection confirmation required')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output/'PROSPECTIVE_PROTOCOL.json', plan)
    camera, guard = None, None
    report = dict(passed=False, errors=[], plan=plan, camera_owner_pid=os.getpid())
    started = time.monotonic()
    try:
        guard = BlackHold(args.session.resolve(), preflight)
        report['transport_before'] = host_transport()
        camera = Line0Camera(guard)
        report['camera_before'] = camera.before
        if camera.before != expected:
            raise RuntimeError('actual camera baseline differs from last exact restoration')
        camera.start_trigger_test(plan['roi'])
        report['camera_after_TriggerMode_On_before_Rolling_reapply'] = camera.after_trigger_enable
        report['camera_active'] = camera.active
        print('Line0/FallingEdge/On/Rolling readback confirmed; waiting for external-trigger frames.', flush=True)
        report['timing'] = collect(camera, guard, plan['frames'], plan['maximum_acquisition_seconds'])
        report['stream_statistics'] = camera.stream_statistics()
        report['camera_active_final'] = camera.parameters()
        report['passed'] = True
    except IncompleteFrame as error:
        report['errors'].append(str(error))
        report['incomplete_frame'] = error.record
    except BaseException as error:
        report['errors'].append(str(error))
    finally:
        if camera:
            report['frame_records'] = getattr(camera, 'test_records', [])
            report['last_active_snapshot'] = getattr(camera, 'active', None)
            report['after_TriggerMode_On_snapshot'] = getattr(camera, 'after_trigger_enable', None)
            saved = getattr(camera, 'test_saved_raw', [])
            if saved:
                np.savez_compressed(args.output/'CCD_trigger_dark_raw.npz', frames=np.stack(saved))
            try:
                camera.close()
            except Exception as error:
                report['errors'].append('cleanup: '+str(error))
            report.update(camera_after=getattr(camera, 'after', None), write_trace=camera.write_trace,
                          exact_baseline_restored=getattr(camera, 'after', None) == expected and camera.closed
                              and not getattr(camera, 'close_errors', [])
                              and not getattr(camera, 'trigger_restore_errors', []))
            if not report['exact_baseline_restored']:
                report['errors'].append('complete original camera baseline not restored')
        try:
            if guard:
                guard.check(force=True)
                report['black_monitor_samples'] = guard.samples
                report['unchanged_GUI_parked_and_black'] = True
            report['transport_after'] = host_transport()
        except Exception as error:
            report['errors'].append('final guard: '+str(error))
        finally:
            if guard:
                guard.close()
        report['passed'] = bool(report['passed'] and not report['errors'])
        report['wall_seconds'] = time.monotonic()-started
        write_new(args.output/'report.json', report)
    print(json.dumps({key: report.get(key) for key in
        ('passed', 'errors', 'timing', 'exact_baseline_restored', 'unchanged_GUI_parked_and_black', 'wall_seconds')},
        ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
