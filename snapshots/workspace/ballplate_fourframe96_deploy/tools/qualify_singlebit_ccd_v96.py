"""New user-reported DMD1bit cohort; bounded all-zero CCD requalification.

Keeps the original display owner and original Line0 arming sequence. Uses
the additive original-Gain snapshot transaction; never edits the running
GUI or a prior trial. No controller configuration is inferred from a model
number, and no camera frame rate is called a measured trigger pulse rate.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from diagnose_black_restart_v83 import TimedBlackHold, wait_black, telemetry, classify
from diagnose_settled_trigger_writes_v88 import uncached_contract
from live_manual_v19_trigger_r7.camera import TransactionalGainCamera
from live_manual_v19_trigger.camera import validate_contract
from inspect_recovery_ui_v20 import inspect
from camera_recheck import verify_baseline
from live_adapt_v22.core import digest, write_new


ROI = dict(OffsetX=888, OffsetY=416, Width=552, Height=554)


def timing_summary(frames):
    if len(frames) < 3:
        raise ValueError('at least3 complete frames required')
    for key in ('FrameID', 'TimeStamp', 'host_received_unix_ns'):
        if any(b[key] <= a[key] for a, b in zip(frames, frames[1:])):
            raise ValueError('non-increasing '+key)
    intervals = np.diff([r['TimeStamp'] for r in frames])/1e6
    return dict(frames=len(frames), first_FrameID=frames[0]['FrameID'], last_FrameID=frames[-1]['FrameID'],
        sensor_timestamp_delivered_fps=(len(frames)-1)*1e9/(frames[-1]['TimeStamp']-frames[0]['TimeStamp']),
        host_delivered_fps=(len(frames)-1)*1e9/(frames[-1]['host_received_unix_ns']-frames[0]['host_received_unix_ns']),
        sensor_interval_ms={str(q): float(np.percentile(intervals, q)) for q in (0, 50, 95, 99, 100)},
        delivered_frame_id_gaps=sum(max(0, b['FrameID']-a['FrameID']-1) for a, b in zip(frames, frames[1:])),
        NOT_trigger_pulse_count_or_frequency=True, NOT_pattern_to_action_latency=True,
        NOT_correct_integration_window_qualification=True)


def capture_trial(guard, target, folder, index, count, maximum_seconds):
    camera = None
    row = dict(index=index, requested_frames=count, capture_passed=False, frames=[], error=None,
               restoration_passed=False, capture_retries=0)
    folder.mkdir(exist_ok=False)
    try:
        guard.check(force=True)
        camera = TransactionalGainCamera(guard)
        if camera.before != target:
            raise RuntimeError('original standby baseline mismatch; not silently replaced')
        camera.set_roi(ROI)
        row['gain_transaction'] = camera.gain_startup_transaction
        camera.start_source(1000., 'Line0')
        row['active'] = uncached_contract(camera)
        validate_contract(row['active'], 1000., ROI, 'Line0')
        row['initial_telemetry'] = telemetry(camera)
        until = time.monotonic()+maximum_seconds
        for j in range(count):
            guard.check()
            if time.monotonic() >= until:
                raise TimeoutError('declared independent-trial acquisition deadline expired')
            raw, meta = camera.capture(timeout_ms=1000)
            if raw.dtype != np.uint8 or raw.shape != (554, 552) or int(raw.max()) >= 240:
                np.save(folder/'REJECTED_raw.npy', raw, allow_pickle=False)
                raise RuntimeError('all-zero raw format/ROI/headroom gate failed')
            if row['frames'] and any(meta[k] <= row['frames'][-1][k] for k in ('FrameID', 'TimeStamp', 'host_received_unix_ns')):
                raise RuntimeError('non-monotonic complete camera frame; no substitute image')
            row['frames'].append(dict(**meta, max_ADU=int(raw.max()), mean_ADU=float(raw.mean())))
            if j in (0, count-1):
                np.save(folder/f'raw_{j:04d}.npy', raw, allow_pickle=False)
        validate_contract(uncached_contract(camera), 1000., ROI, 'Line0')
        row['timing'] = timing_summary(row['frames'])
        row['final_telemetry'] = telemetry(camera)
        row['capture_passed'] = True
    except Exception as error:
        row.update(error=repr(error), error_class=classify(error))
        if camera is not None:
            row['failure_telemetry'] = telemetry(camera)
    finally:
        if camera is not None:
            try:
                camera.close()
                row['restoration_passed'] = camera.after == target
            except Exception as error:
                row['close_error'] = repr(error)
            row['after'] = getattr(camera, 'after', None)
            row['write_trace'] = camera.write_trace
        write_new(folder/'TRIAL_RESULT.json', row)
    print(json.dumps(dict(trial=index, requested_frames=count, frames=len(row['frames']),
        capture_passed=row['capture_passed'], error=row['error'], restoration_passed=row['restoration_passed']), ensure_ascii=False), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--original-camera-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        raise ValueError('explicit new bounded all-zero qualification required')
    previous = json.loads(args.original_camera_report.read_text())
    target = previous['camera_before']
    if not previous.get('camera_settings_restored') or previous.get('camera_after') != target:
        raise ValueError('exactly restored original target required, no guessed baseline')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='PROSPECTIVE_NEW_SINGLEBIT_CCD_COHORT_V96',
        source_sha256=digest(__file__), adapter_sha256=digest(PROJECT/'live_manual_v19_trigger_r7/camera.py'),
        original_report_sha256=digest(args.original_camera_report), bit_depth=1,
        bit_depth_source='USER_REPORTED_ONLY_NOT_CONTROLLER_READBACK',
        DMD_exposure_dark_trigger_configuration_UNMEASURED=True,
        independent_start_trials=12, frames_per_start=31, first_image_timeout_ms_UNCHANGED=1000,
        long_fresh_connection_frames=3000, long_trial_ONLY_if_ALL_start_trials_pass=True,
        maximum_wall_seconds=450, maximum_short_capture_seconds=4, maximum_long_capture_seconds=90,
        no_software_triggers_or_free_run=True, no_bad_image_retry=True,
        no_projection=True, no_new_windows=True, no_existing_GUI_source_edits=True,
        no_controller_GPIO_network_firmware_persistent_writes=True,
        requested_exposure_us=1000., requested_gain=0., ROI=ROI,
        inference_or_closed_loop_qualification=False, release_approved=False)
    write_new(args.output/'PROSPECTIVE_SINGLEBIT_QUALIFICATION.json', plan)
    preflight = inspect(args.session, parked=True)
    guard = TimedBlackHold(args.session, preflight, time.monotonic()+450)
    rows, long_trial = [], None
    result = dict(completed=False, error=None)
    try:
        for index in range(12):
            row = capture_trial(guard, target, args.output/f'start_{index:02d}', index, 31, 4)
            rows.append(row)
            if not row['restoration_passed'] or (row['error'] and row['error_class'] != 'startup_or_capture_timeout_NO_frame_retry'):
                raise RuntimeError('non-timeout or restoration fault; no further starts')
            wait_black(guard, .2)
        if all(r['capture_passed'] for r in rows):
            long_trial = capture_trial(guard, target, args.output/'long_fresh_connection', 'long', 3000, 90)
            if not long_trial['capture_passed'] or not long_trial['restoration_passed']:
                raise RuntimeError('long connection capture/restoration failed')
        else:
            result['long_trial_skipped_reason'] = 'At least one predeclared startup failed; do not select a lucky run.'
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        try:
            result['final_readonly'] = verify_baseline(target, lambda: guard.check(force=True))
            result['final_black'] = inspect(args.session, parked=True)
        except Exception as error:
            result['final_verification_error'] = repr(error)
        guard.close()
        result.update(rows=rows, long_trial=long_trial, new_DMD_bit_depth_user_reported=1,
            no_projection=True, no_new_windows=True, release_approved=False,
            NOT_a_matched_AB_causal_proof=True, hardware_20ms_qualified=False,
            static_optical_integration_window_qualified=False)
        result['short_starts'] = dict(trials=len(rows), passed=sum(r['capture_passed'] for r in rows),
            exact_restores=sum(r['restoration_passed'] for r in rows))
        result['camera_only_sequence_qualified_this_batch'] = bool(result['completed']
            and result.get('final_readonly', {}).get('passed') and len(rows) == 12
            and all(r['capture_passed'] and r['restoration_passed'] for r in rows)
            and long_trial and long_trial['capture_passed'] and long_trial['restoration_passed'])
        write_new(args.output/'SINGLEBIT_CCD_QUALIFICATION_RESULT.json', result)
        print(json.dumps({k: result[k] for k in ('completed', 'error', 'short_starts', 'camera_only_sequence_qualified_this_batch')}, ensure_ascii=False), flush=True)
    if result['error'] or not result.get('final_readonly', {}).get('passed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
