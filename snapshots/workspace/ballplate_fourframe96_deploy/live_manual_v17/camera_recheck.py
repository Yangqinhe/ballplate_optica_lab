"""Read-only fault recheck, while the existing owner keeps both outputs black.

This does not repair nodes, rebaseline a failure, acknowledge a failed stage,
start acquisition or project anything. Only an exact, repeated match to the
failed stage's BEFORE snapshot can release the projection fault latch.
"""
from pathlib import Path
import json
import threading
import time
import traceback

from PyQt5 import QtCore

from black_pixels import RootPixels
from display_linux import read_roles, same_signals
from spin_c import Camera


class ReadOnlyCamera(Camera):
    def _write(self, *args, **kwargs):
        raise RuntimeError('read-only recheck forbids camera node writes')


def differences(expected, actual):
    return {k: dict(expected=expected.get(k), actual=actual.get(k))
            for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}


def verify_baseline(expected, guard, stopped=lambda: False, camera_factory=ReadOnlyCamera,
                    pause=time.sleep):
    required = ('PixelFormat', 'ExposureAuto', 'ExposureTime', 'GainAuto', 'Gain',
                'OffsetX', 'OffsetY', 'Width', 'Height', 'TriggerMode', 'BlackLevel')
    if (expected.get('DeviceSerialNumber') != '15328133'
            or any(expected.get(k) is None for k in required)):
        raise ValueError('missing original camera BEFORE baseline; no guessed target accepted')
    result = dict(passed=False, acquisition_started=False, parameter_writes=0,
                  target=expected, snapshots=[], connections=0, errors=[],
                  original_failed_stage_accepted=False, release_approved=False)
    for connection in range(2):
        camera = None
        try:
            if stopped(): raise RuntimeError('recheck cancelled')
            guard()
            camera = camera_factory()
            result['connections'] += 1
            snapshots = [camera.before]
            for _ in range(3):
                if stopped(): raise RuntimeError('recheck cancelled')
                guard(); pause(.25)
                snapshots.append(camera.parameters())
            for snapshot in snapshots:
                result['snapshots'].append(dict(connection=connection, values=snapshot,
                    differences=differences(expected, snapshot)))
            if any(differences(expected, s) for s in snapshots):
                raise RuntimeError('camera differs from the original BEFORE baseline; fault retained')
        except Exception as error:
            result['errors'].append(str(error))
        finally:
            if camera is not None:
                try: camera.close()
                except Exception as error: result['errors'].append('close: '+str(error))
                after = getattr(camera, 'after', {})
                result['snapshots'].append(dict(connection=connection, values=after,
                    differences=differences(expected, after)))
                if differences(expected, after): result['errors'].append('close snapshot mismatch')
        if result['errors']: break
    guard()
    if stopped(): result['errors'].append('recheck cancelled')
    result['passed'] = bool(not result['errors'] and result['connections'] == 2
                            and len(result['snapshots']) == 10)
    return result


class RecheckWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(object)

    def __init__(self, expected, video, directory):
        super().__init__()
        self.expected, self.video, self.directory = dict(expected), video, Path(directory)
        self.stop_requested = threading.Event()

    def stop(self): self.stop_requested.set()

    def run(self):
        report = dict(passed=False, acquisition_started=False, parameter_writes=0,
                      release_approved=False)
        pixels = None
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            pixels = RootPixels()
            checks = []
            def guard():
                same_signals(self.video, read_roles())
                observation = {}
                for role, row in self.video.items():
                    observation[role] = pixels.inspect((*row['desktop_position'], *row['desktop_size']))
                    if not observation[role]['all_RGB_zero']:
                        raise RuntimeError(role+': existing framebuffer is not all black; fault retained')
                checks.append(observation)
            report = verify_baseline(self.expected, guard, self.stop_requested.is_set)
            report['black_framebuffer_observations'] = checks
        except Exception as error:
            report.update(passed=False, error=str(error), traceback=traceback.format_exc())
        finally:
            if pixels: pixels.close()
            if self.directory.exists():
                (self.directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            self.result.emit(report)
