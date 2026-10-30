"""Bounded all-zero A/B startup diagnosis, ORIGINAL display owner unchanged.

Only runs after the GUI releases its SDK lock. Exactly one camera owner at
a time. No projection, GPIO/network/firmware changes or Software triggers.
Each timeout remains a failed trial; later telemetry is NOT a frame retry.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from live_adapt_v22.core import digest, write_new
from inspect_recovery_ui_v20 import inspect
from probe_line0_trigger_v18 import BlackHold
from live_manual_v19_trigger_r2.camera import OrderedTriggerCamera
from live_manual_v19_trigger.camera import validate_contract
from camera_recheck import verify_baseline


def nic_snapshot():
    base = Path('/sys/class/net/eno1/statistics')
    return {key: int((base/key).read_text()) for key in
        ('rx_bytes', 'rx_packets', 'rx_errors', 'rx_dropped', 'tx_bytes', 'tx_packets')}


def telemetry(camera):
    nodes = {}
    for key in ('AcquisitionStatus', 'AcquisitionStatusSelector', 'LineStatusAll',
                'LineSelector', 'LineStatus', 'GevSCDA', 'GevSCPHostPort',
                'GevSCCFGExtendedChunkData', 'GevHeartbeatTimeout'):
        try:
            nodes[key] = camera.read(key)
        except RuntimeError as error:
            nodes[key] = dict(unavailable=str(error))
    return dict(unix_ns=time.time_ns(), nodes=nodes, stream=camera.stream_statistics(), nic=nic_snapshot())


class TimedBlackHold(BlackHold):
    def __init__(self, session, preflight, deadline):
        self.deadline = deadline
        super().__init__(session, preflight)

    def check(self, force=False):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('bounded camera-only diagnosis expired; no further acquisition')
        return super().check(force=force)


def wait_black(guard, seconds):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        guard.check()
        time.sleep(min(.05, max(0, end-time.monotonic())))


def classify(error):
    value = str(error)
    if value.startswith('spinCameraGetNextImageEx: -1011:'):
        return 'startup_or_capture_timeout_NO_frame_retry'
    return 'fatal_non_timeout_error'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--original-camera-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not 1 <= a.blocks <= 5:
        p.error('one to five ABBA blocks only')
    before = json.loads(a.original_camera_report.read_text())
    expected = before['camera_before']
    if (expected['DeviceSerialNumber'] != '15328133' or expected['TriggerMode'] != 'Off'
            or before.get('error') or not before.get('camera_settings_restored')):
        raise ValueError('clean original BEFORE standby target required, never a guessed baseline')
    sequence = [0., 1.25, 1.25, 0.]*a.blocks
    plan = dict(role='NEW_BOUNDED_ALL_ZERO_CAMERA_STARTUP_DIAGNOSIS_NOT_OPTICAL_CONTROL',
        original_camera_report=str(a.original_camera_report.resolve()),
        original_camera_report_sha256=digest(a.original_camera_report), source_sha256=digest(__file__),
        post_BeginAcquisition_wait_seconds=sequence, frames_per_trial=31,
        retrieval_timeout_ms_UNCHANGED=1000, maximum_wall_seconds=600,
        failed_frame_NO_retry=True, failed_trials_retained_before_next_predeclared_independent_trial=True,
        delayed_statistics_after_timeout_are_NOT_frame_acceptance=True,
        ROI=dict(OffsetX=888, OffsetY=416, Width=552, Height=554),
        requested_exposure_us=1000., gain=0., trigger='FrameStart/Line0/FallingEdge/Rolling',
        requires_original_GUI_camera_released=True, sole_SDK_owner_during_each_trial=True,
        no_new_output_windows_no_projection_no_software_triggers=True,
        no_network_GPIO_firmware_or_persistent_parameter_changes=True,
        external_camera_only_diagnostic_OWNED_by_this_process_NOT_a_second_concurrent_camera=True,
        reference='https://softwareservices.flir.com/Spinnaker/4.4/examples/python/BufferHandling.html',
        note='Official old-camera startup pause is a hypothesis, NOT an established cause', release_approved=False)
    if not a.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    a.output.mkdir(parents=True, exist_ok=False)
    write_new(a.output/'PROSPECTIVE_BLACK_STARTUP_DIAGNOSIS.json', plan)
    preflight = inspect(a.session, parked=True)
    write_new(a.output/'ORIGINAL_WINDOW_BLACK_PREFLIGHT.json', preflight)
    started = time.monotonic(); guard = TimedBlackHold(a.session, preflight, started+600)
    rows = []; result = dict(completed=False, error=None, no_projection=True, no_new_windows=True,
        original_failed_trials_unchanged=True, real_actuators_enabled=False, release_approved=False)
    camera = None
    try:
        for index, delay in enumerate(sequence):
            guard.check(force=True)
            folder = a.output/f'trial_{index:02d}_wait{delay:.2f}s'
            folder.mkdir(exist_ok=False)
            row = dict(index=index, wait_s=delay, accepted_frames=0, error=None, capture_passed=False,
                restoration_passed=False, capture_retry_count=0)
            records = []; before_nic = nic_snapshot(); frame_error = None
            try:
                camera = OrderedTriggerCamera(guard)
                if camera.before != expected:
                    raise RuntimeError('camera BEFORE differs from original GUI standby target')
                row['before'] = camera.before
                camera.set_roi(plan['ROI'])
                row['pre_start_telemetry'] = telemetry(camera)
                t0 = time.monotonic(); camera.start_source(1000., 'Line0')
                row['source_start_call_s'] = time.monotonic()-t0
                row['immediate_started_telemetry'] = telemetry(camera)
                wait_black(guard, delay)
                row['after_wait_telemetry'] = telemetry(camera)
                for count in range(31):
                    guard.check()
                    at = time.monotonic()
                    image, meta = camera.capture(timeout_ms=1000)
                    elapsed = time.monotonic()-at
                    if image.dtype != np.uint8 or image.shape != (554, 552) or int(image.max()) >= 240:
                        np.save(folder/'REJECTED_raw.npy', image, allow_pickle=False)
                        raise RuntimeError('zero-input raw dtype/ROI/headroom gate failed')
                    if records and (meta['FrameID'] <= records[-1]['FrameID'] or meta['TimeStamp'] <= records[-1]['TimeStamp']):
                        raise RuntimeError('fresh diagnostic frame ordering failed')
                    records.append(dict(**meta, get_wall_s=elapsed, max_ADU=int(image.max()),
                        mean_ADU=float(image.mean())))
                    if count == 0:
                        np.save(folder/'FIRST_raw.npy', image, allow_pickle=False)
                    row['accepted_frames'] += 1
                row['final_telemetry'] = telemetry(camera)
                validate_contract(camera.parameters(), 1000., plan['ROI'], 'Line0')
                row['capture_passed'] = True
            except Exception as error:
                row['error'] = repr(error); row['error_class'] = classify(error)
                frame_error = error
                if camera is not None and camera.acquiring:
                    row['at_failure_telemetry'] = telemetry(camera)
                    if row['error_class'] == 'startup_or_capture_timeout_NO_frame_retry':
                        # Observation only: NEVER call GetNextImage a second
                        # time after the failed retrieval in this trial.
                        wait_black(guard, 2.5)
                        row['late_telemetry_NO_frame_retry'] = telemetry(camera)
            finally:
                if camera is not None:
                    try:
                        camera.close(); row['restoration_passed'] = camera.after == expected
                    except Exception as error:
                        row['restoration_error'] = repr(error)
                    row['after'] = getattr(camera, 'after', None)
                    row['trigger_write_trace'] = getattr(camera, 'write_trace', [])
                    camera = None
                row['nic_delta'] = {k: nic_snapshot()[k]-v for k, v in before_nic.items()}
                row['frame_records'] = records
                write_new(folder/'TRIAL_RESULT.json', row)
                rows.append(row)
                print(json.dumps({k:row.get(k) for k in ('index','wait_s','accepted_frames','error',
                    'restoration_passed','source_start_call_s')}, ensure_ascii=False), flush=True)
            if not row['restoration_passed']:
                raise RuntimeError('exact camera restoration failed; no further trials')
            if frame_error is not None and row['error_class'] != 'startup_or_capture_timeout_NO_frame_retry':
                raise RuntimeError('non-timeout diagnostic fault; no further trials')
            wait_black(guard, .5)
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        if camera is not None:
            try: camera.close()
            except Exception as error: result['last_close_error'] = repr(error)
        try:
            result['original_strict_READONLY_final'] = verify_baseline(expected, lambda:guard.check(force=True))
            result['standby_exactly_restored'] = result['original_strict_READONLY_final']['passed']
        except Exception as error:
            result['final_readonly_error'] = repr(error)
        result['final_original_black'] = inspect(a.session, parked=True)
        guard.close()
        groups = []
        for delay in (0., 1.25):
            cohort = [r for r in rows if r['wait_s'] == delay]
            groups.append(dict(wait_s=delay, trials=len(cohort), passed=sum(r['capture_passed'] for r in cohort),
                timeout_count=sum(r.get('error_class') == 'startup_or_capture_timeout_NO_frame_retry' for r in cohort)))
        result.update(rows=rows, groups=groups, wall_seconds=time.monotonic()-started,
            actual_outputs_remained_zero=True, camera_released=True, source_sha256=digest(__file__))
        write_new(a.output/'BLACK_STARTUP_DIAGNOSIS_RESULT.json', result)
        print(json.dumps(dict(completed=result['completed'], error=result['error'], groups=groups,
            standby_exactly_restored=result.get('standby_exactly_restored')), ensure_ascii=False), flush=True)
    if result['error'] or not result.get('standby_exactly_restored'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
