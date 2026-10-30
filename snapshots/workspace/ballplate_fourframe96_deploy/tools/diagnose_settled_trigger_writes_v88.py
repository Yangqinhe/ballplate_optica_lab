"""Black-only trigger/shutter transition settling and uncached readback test.

No new projection, windows, trigger commands or GPIO changes. Comparisons
use the same existing Line0/FallingEdge/Rolling settings. Invalidating a
GenApi read cache is not changing the device configuration.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from diagnose_black_restart_v83 import TimedBlackHold, wait_black, telemetry, classify
from live_manual_v19_trigger_r2.camera import OrderedTriggerCamera
from live_manual_v19_trigger.camera import validate_contract
from inspect_recovery_ui_v20 import inspect
from camera_recheck import verify_baseline
from live_adapt_v22.core import digest, write_new


REFRESH_KEYS = ('TriggerMode', 'TriggerSource', 'TriggerSelector',
                'TriggerActivation', 'SensorShutterMode', 'ExposureAuto', 'ExposureTime')


def uncached_contract(camera):
    fn = camera.lib.spinNodeInvalidateNode
    fn.argtypes = [C.c_void_p]
    fn.restype = C.c_int
    for key in REFRESH_KEYS:
        with camera.node(key) as node:
            camera.call('spinNodeInvalidateNode', node)
    return camera.parameters()


class SettledWriteCamera(OrderedTriggerCamera):
    def _write(self, name, value, node_map):
        result = super()._write(name, value, node_map)
        if not self.cleaning and name in ('TriggerMode', 'SensorShutterMode', 'ExposureTime'):
            wait_black(self.guard, .15)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--original-camera-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise ValueError('explicit bounded black-only diagnostic required')
    target = json.loads(args.original_camera_report.read_text())['camera_before']
    modes = ['unchanged_order', 'settled_each_write', 'settled_each_write', 'unchanged_order'] * 4
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output/'PROSPECTIVE_SETTLED_WRITES.json', dict(sequence=modes,
        source_sha256=digest(__file__), frames_per_trial=31, maximum_wall_seconds=240,
        requested_ROI=dict(OffsetX=888, OffsetY=416, Width=552, Height=554),
        both_modes_uncached_post_begin_readback=True, wait_after_relevant_write_s=.15,
        all_frames_REQUIRE_Line0=True, no_projection=True, no_new_windows=True,
        no_software_commands=True, no_image_retry=True, failed_trials_retained=True,
        no_device_reset_or_network_changes=True, release_approved=False))
    preflight = inspect(args.session, parked=True)
    guard = TimedBlackHold(args.session, preflight, time.monotonic()+240)
    rows, camera = [], None
    result = dict(completed=False, error=None)
    try:
        for index, mode in enumerate(modes):
            row = dict(index=index, mode=mode, capture_passed=False, frames=[], error=None, restoration_passed=False)
            guard.check(force=True)
            try:
                cls = SettledWriteCamera if mode == 'settled_each_write' else OrderedTriggerCamera
                camera = cls(guard)
                if camera.before != target:
                    raise RuntimeError('original standby baseline mismatch')
                roi = dict(OffsetX=888, OffsetY=416, Width=552, Height=554)
                camera.set_roi(roi)
                camera.start_source(1000., 'Line0')
                row['cached_active'] = camera.active
                row['uncached_active'] = uncached_contract(camera)
                row['cache_differences'] = {k: dict(cached=camera.active[k], fresh=row['uncached_active'][k])
                    for k in camera.active if camera.active[k] != row['uncached_active'][k]}
                validate_contract(row['uncached_active'], 1000., roi, 'Line0')
                row['at_start'] = telemetry(camera)
                for j in range(31):
                    guard.check()
                    raw, meta = camera.capture(timeout_ms=1000)
                    if raw.shape != (554, 552) or raw.max() >= 240:
                        raise RuntimeError('ROI/headroom gate failed')
                    if row['frames'] and any(meta[k] <= row['frames'][-1][k] for k in ('FrameID', 'TimeStamp')):
                        raise RuntimeError('frame ID/sensor timestamp did not advance')
                    row['frames'].append(meta)
                validate_contract(uncached_contract(camera), 1000., roi, 'Line0')
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
                    row['restore_differences'] = {k: dict(expected=target.get(k), actual=(row['after'] or {}).get(k))
                        for k in target if target[k] != (row['after'] or {}).get(k)}
                    row['write_trace'] = camera.write_trace
                    camera = None
                write_new(args.output/f'TRIAL_{index:02d}.json', row)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('index', 'mode', 'capture_passed', 'error', 'restoration_passed')}), flush=True)
            if not row['restoration_passed'] or (row['error'] and row['error_class'] != 'startup_or_capture_timeout_NO_frame_retry'):
                raise RuntimeError('non-timeout or restoration fault; no further trials')
            wait_black(guard, .2)
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception as error:
                result['last_close_error'] = repr(error)
        try:
            result['final_readonly'] = verify_baseline(target, lambda: guard.check(force=True))
            result['final_black'] = inspect(args.session, parked=True)
        except Exception as error:
            result['final_verification_error'] = repr(error)
        guard.close()
        result.update(rows=rows, release_approved=False, no_projection=True, no_new_windows=True)
        result['groups'] = {mode: dict(trials=sum(r['mode'] == mode for r in rows),
            passed=sum(r['mode'] == mode and r['capture_passed'] for r in rows)) for mode in set(modes)}
        write_new(args.output/'SETTLED_WRITES_RESULT.json', result)
        print(json.dumps(dict(completed=result['completed'], error=result['error'], groups=result['groups'])), flush=True)
    if result['error'] or not result.get('final_readonly', {}).get('passed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
